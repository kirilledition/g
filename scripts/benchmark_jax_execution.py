#!/usr/bin/env python3
"""Benchmark JAX execution surfaces relevant to the Phase 2 GPU arm."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from bed_reader import open_bed
from bed_reader._open_bed import get_num_threads

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state
from g.engine import (
    build_linear_output_frame,
    build_logistic_output_frame,
    compute_logistic_association_with_missing_exclusion,
)
from g.io.plink import iter_genotype_chunks, read_bed_chunk_host
from g.io.tabular import load_aligned_sample_data

if TYPE_CHECKING:
    from g.models import GenotypeChunk


@dataclass(frozen=True)
class DeviceSummary:
    """Summary of one JAX-visible device."""

    platform: str
    device_kind: str
    id: int


@dataclass(frozen=True)
class RuntimeSummary:
    """High-level JAX runtime state relevant to GPU bring-up."""

    jax_version: str
    default_backend: str
    local_device_count: int
    devices: list[DeviceSummary]
    x64_enabled: bool


@dataclass(frozen=True)
class TimedMeasurement:
    """One first-run and warmed benchmark result."""

    first_run_seconds: float
    warmed_durations_seconds: list[float]
    warmed_mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class JaxExecutionBenchmarkReport:
    """Structured benchmark report for the JAX execution path."""

    runtime: RuntimeSummary
    variant_limit: int
    chunk_size: int
    repeat_count: int
    host_chunk_read_seconds: float
    device_put: TimedMeasurement
    linear_compute: TimedMeasurement
    linear_format: TimedMeasurement
    logistic_compute: TimedMeasurement
    logistic_format: TimedMeasurement


def collect_runtime_summary() -> RuntimeSummary:
    """Collect JAX backend and visible device information."""
    return RuntimeSummary(
        jax_version=jax.__version__,
        default_backend=jax.default_backend(),
        local_device_count=jax.local_device_count(),
        devices=[
            DeviceSummary(
                platform=device.platform,
                device_kind=device.device_kind,
                id=device.id,
            )
            for device in jax.devices()
        ],
        x64_enabled=bool(jax.config.read("jax_enable_x64")),
    )


def block_tree_until_ready(value: Any) -> Any:
    """Synchronize a JAX pytree and return it unchanged."""
    return jax.block_until_ready(value)


def time_operation(operation: Any, checksum_operation: Any, repeat_count: int) -> TimedMeasurement:
    """Measure first-run and warmed timings for one operation."""
    start_time = time.perf_counter()
    first_value = operation()
    block_tree_until_ready(first_value)
    first_run_seconds = time.perf_counter() - start_time
    checksum = float(checksum_operation(first_value))

    warmed_durations_seconds: list[float] = []
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        warmed_value = operation()
        block_tree_until_ready(warmed_value)
        warmed_durations_seconds.append(time.perf_counter() - start_time)
        checksum = float(checksum_operation(warmed_value))

    return TimedMeasurement(
        first_run_seconds=first_run_seconds,
        warmed_durations_seconds=warmed_durations_seconds,
        warmed_mean_seconds=sum(warmed_durations_seconds) / len(warmed_durations_seconds),
        checksum=checksum,
    )


def checksum_linear_result(linear_result: Any) -> float:
    """Build a stable checksum from a linear result tree."""
    return float(np.asarray(jnp.sum(linear_result.beta) + jnp.sum(linear_result.p_value)))


def checksum_logistic_result(logistic_evaluation: Any) -> float:
    """Build a stable checksum from a logistic evaluation tree."""
    logistic_result = logistic_evaluation.logistic_result
    return float(np.asarray(jnp.sum(logistic_result.beta) + jnp.sum(logistic_result.p_value)))


def checksum_frame(output_frame: Any) -> float:
    """Build a stable checksum from a formatted Polars frame."""
    return float(output_frame.select(pl.col("p_value").sum()).item())


def main() -> None:
    """Benchmark representative JAX transfer, compute, and formatting surfaces."""
    bed_prefix = Path("data/1kg_chr22_full")
    chunk_size = 256
    variant_limit = chunk_size
    repeat_count = 5

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

    bed_path = bed_prefix.with_suffix(".bed")
    sample_index_array = np.ascontiguousarray(continuous_sample_data.sample_indices, dtype=np.intp)
    with open_bed(str(bed_path)) as bed_handle:
        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))
        start_time = time.perf_counter()
        genotype_matrix_host = read_bed_chunk_host(
            bed_handle=bed_handle,
            bed_path=bed_path,
            sample_index_array=sample_index_array,
            variant_start=0,
            variant_stop=variant_limit,
            num_threads=num_threads,
        )
        host_chunk_read_seconds = time.perf_counter() - start_time

    continuous_chunk: GenotypeChunk = next(
        iter_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=continuous_sample_data.sample_indices,
            expected_individual_identifiers=continuous_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    )
    binary_chunk: GenotypeChunk = next(
        iter_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=binary_sample_data.sample_indices,
            expected_individual_identifiers=binary_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    )

    linear_association_state = prepare_linear_association_state(
        covariate_matrix=continuous_sample_data.covariate_matrix,
        phenotype_vector=continuous_sample_data.phenotype_vector,
    )

    device_put_measurement = time_operation(
        operation=lambda: jax.device_put(genotype_matrix_host),
        checksum_operation=lambda genotype_matrix: float(np.asarray(jnp.sum(genotype_matrix))),
        repeat_count=repeat_count,
    )
    linear_compute_measurement = time_operation(
        operation=lambda: compute_linear_association_chunk(
            linear_association_state=linear_association_state,
            genotype_matrix=continuous_chunk.genotypes,
        ),
        checksum_operation=checksum_linear_result,
        repeat_count=repeat_count,
    )
    linear_result = compute_linear_association_chunk(
        linear_association_state=linear_association_state,
        genotype_matrix=continuous_chunk.genotypes,
    )
    block_tree_until_ready(linear_result)
    linear_format_measurement = time_operation(
        operation=lambda: build_linear_output_frame(
            metadata=continuous_chunk.metadata,
            allele_one_frequency=continuous_chunk.allele_one_frequency,
            observation_count=continuous_chunk.observation_count,
            linear_result=linear_result,
        ),
        checksum_operation=checksum_frame,
        repeat_count=repeat_count,
    )
    logistic_compute_measurement = time_operation(
        operation=lambda: compute_logistic_association_with_missing_exclusion(
            covariate_matrix=binary_sample_data.covariate_matrix,
            phenotype_vector=binary_sample_data.phenotype_vector,
            genotype_chunk=binary_chunk,
            max_iterations=100,
            tolerance=1.0e-8,
        ),
        checksum_operation=checksum_logistic_result,
        repeat_count=repeat_count,
    )
    logistic_evaluation = compute_logistic_association_with_missing_exclusion(
        covariate_matrix=binary_sample_data.covariate_matrix,
        phenotype_vector=binary_sample_data.phenotype_vector,
        genotype_chunk=binary_chunk,
        max_iterations=100,
        tolerance=1.0e-8,
    )
    block_tree_until_ready(logistic_evaluation)
    logistic_format_measurement = time_operation(
        operation=lambda: build_logistic_output_frame(
            metadata=binary_chunk.metadata,
            allele_one_frequency=logistic_evaluation.allele_one_frequency,
            observation_count=logistic_evaluation.observation_count,
            logistic_result=logistic_evaluation.logistic_result,
        ),
        checksum_operation=checksum_frame,
        repeat_count=repeat_count,
    )

    report = JaxExecutionBenchmarkReport(
        runtime=collect_runtime_summary(),
        variant_limit=variant_limit,
        chunk_size=chunk_size,
        repeat_count=repeat_count,
        host_chunk_read_seconds=host_chunk_read_seconds,
        device_put=device_put_measurement,
        linear_compute=linear_compute_measurement,
        linear_format=linear_format_measurement,
        logistic_compute=logistic_compute_measurement,
        logistic_format=logistic_format_measurement,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
