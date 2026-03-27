#!/usr/bin/env python3
"""Benchmark JAX chunk-size sensitivity for linear and logistic execution."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state
from g.compute.logistic import LogisticMethod, compute_logistic_association_chunk
from g.engine import compute_logistic_association_with_missing_exclusion
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data

if TYPE_CHECKING:
    from g.models import GenotypeChunk


@dataclass(frozen=True)
class ChunkSizeMeasurement:
    """Warmed runtime measurement for one chunk size and path."""

    chunk_size: int
    warmed_durations_seconds: list[float]
    warmed_mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class ChunkSweepReport:
    """Structured chunk-size sweep report."""

    backend: str
    repeat_count: int
    chunk_sizes: list[int]
    linear_compute: list[ChunkSizeMeasurement]
    logistic_standard_compute: list[ChunkSizeMeasurement]
    logistic_fallback_compute: list[ChunkSizeMeasurement]


def block_tree_until_ready(value: Any) -> Any:
    """Synchronize a JAX pytree and return it unchanged."""
    return jax.block_until_ready(value)


def time_operation(operation: Any, checksum_operation: Any, repeat_count: int) -> ChunkSizeMeasurement:
    """Measure warmed runtime for one operation."""
    operation()
    warmed_durations_seconds: list[float] = []
    checksum = 0.0
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        value = operation()
        block_tree_until_ready(value)
        warmed_durations_seconds.append(time.perf_counter() - start_time)
        checksum = float(checksum_operation(value))
    return ChunkSizeMeasurement(
        chunk_size=0,
        warmed_durations_seconds=warmed_durations_seconds,
        warmed_mean_seconds=sum(warmed_durations_seconds) / len(warmed_durations_seconds),
        checksum=checksum,
    )


def attach_chunk_size(measurement: ChunkSizeMeasurement, chunk_size: int) -> ChunkSizeMeasurement:
    """Return a copy of a measurement with the chunk size attached."""
    return ChunkSizeMeasurement(
        chunk_size=chunk_size,
        warmed_durations_seconds=measurement.warmed_durations_seconds,
        warmed_mean_seconds=measurement.warmed_mean_seconds,
        checksum=measurement.checksum,
    )


def checksum_linear_result(linear_result: Any) -> float:
    """Build a stable checksum from a linear result tree."""
    return float(np.asarray(jnp.sum(linear_result.beta) + jnp.sum(linear_result.p_value)))


def checksum_logistic_result(logistic_result: Any) -> float:
    """Build a stable checksum from a logistic result tree."""
    return float(np.asarray(jnp.sum(logistic_result.beta) + jnp.sum(logistic_result.p_value)))


def load_first_chunk(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
) -> GenotypeChunk:
    """Load the first chunk at the requested chunk size."""
    return next(
        iter_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=chunk_size,
        )
    )


def load_first_fallback_chunk(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    search_variant_limit: int,
) -> GenotypeChunk:
    """Load the first chunk that exercises hybrid logistic fallback."""
    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=sample_indices,
        expected_individual_identifiers=expected_individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=search_variant_limit,
    ):
        logistic_evaluation = compute_logistic_association_with_missing_exclusion(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_chunk=genotype_chunk,
            max_iterations=100,
            tolerance=1.0e-8,
        )
        method_code = np.asarray(logistic_evaluation.logistic_result.method_code)
        if np.any(method_code == LogisticMethod.FIRTH):
            return genotype_chunk
    raise RuntimeError("Could not find a fallback chunk in the configured search range.")


def main() -> None:
    """Run a warmed chunk-size sweep on the current JAX backend."""
    bed_prefix = Path("data/1kg_chr22_full")
    chunk_sizes = [256, 512, 1024, 2048]
    repeat_count = 5
    search_variant_limit = 8192

    continuous_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_cont.txt"),
        phenotype_name="phenotype_continuous",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )
    binary_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_bin.txt"),
        phenotype_name="phenotype_binary",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=True,
    )

    linear_measurements: list[ChunkSizeMeasurement] = []
    logistic_standard_measurements: list[ChunkSizeMeasurement] = []
    logistic_fallback_measurements: list[ChunkSizeMeasurement] = []

    for chunk_size in chunk_sizes:
        continuous_chunk = load_first_chunk(
            bed_prefix=bed_prefix,
            sample_indices=continuous_sample_data.sample_indices,
            expected_individual_identifiers=continuous_sample_data.individual_identifiers,
            chunk_size=chunk_size,
        )
        binary_standard_chunk = load_first_chunk(
            bed_prefix=bed_prefix,
            sample_indices=binary_sample_data.sample_indices,
            expected_individual_identifiers=binary_sample_data.individual_identifiers,
            chunk_size=chunk_size,
        )
        binary_fallback_chunk = load_first_fallback_chunk(
            bed_prefix=bed_prefix,
            sample_indices=binary_sample_data.sample_indices,
            expected_individual_identifiers=binary_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            covariate_matrix=binary_sample_data.covariate_matrix,
            phenotype_vector=binary_sample_data.phenotype_vector,
            search_variant_limit=search_variant_limit,
        )

        linear_association_state = prepare_linear_association_state(
            covariate_matrix=continuous_sample_data.covariate_matrix,
            phenotype_vector=continuous_sample_data.phenotype_vector,
        )
        linear_measurement = time_operation(
            operation=lambda: compute_linear_association_chunk(
                linear_association_state=linear_association_state,
                genotype_matrix=continuous_chunk.genotypes,
            ),
            checksum_operation=checksum_linear_result,
            repeat_count=repeat_count,
        )
        logistic_standard_measurement = time_operation(
            operation=lambda: compute_logistic_association_chunk(
                covariate_matrix=binary_sample_data.covariate_matrix,
                phenotype_vector=binary_sample_data.phenotype_vector,
                genotype_matrix=binary_standard_chunk.genotypes,
                max_iterations=100,
                tolerance=1.0e-8,
            ),
            checksum_operation=checksum_logistic_result,
            repeat_count=repeat_count,
        )
        logistic_fallback_measurement = time_operation(
            operation=lambda: (
                compute_logistic_association_with_missing_exclusion(
                    covariate_matrix=binary_sample_data.covariate_matrix,
                    phenotype_vector=binary_sample_data.phenotype_vector,
                    genotype_chunk=binary_fallback_chunk,
                    max_iterations=100,
                    tolerance=1.0e-8,
                ).logistic_result
            ),
            checksum_operation=checksum_logistic_result,
            repeat_count=repeat_count,
        )

        linear_measurements.append(attach_chunk_size(linear_measurement, chunk_size))
        logistic_standard_measurements.append(attach_chunk_size(logistic_standard_measurement, chunk_size))
        logistic_fallback_measurements.append(attach_chunk_size(logistic_fallback_measurement, chunk_size))

    report = ChunkSweepReport(
        backend=jax.default_backend(),
        repeat_count=repeat_count,
        chunk_sizes=chunk_sizes,
        linear_compute=linear_measurements,
        logistic_standard_compute=logistic_standard_measurements,
        logistic_fallback_compute=logistic_fallback_measurements,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
