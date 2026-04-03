#!/usr/bin/env python3
"""Benchmark JAX execution surfaces relevant to the Phase 2 GPU arm."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
from g.compute.logistic import LogisticMethod, compute_logistic_association_chunk
from g.engine import (
    LinearChunkAccumulator,
    LogisticChunkAccumulator,
    build_linear_output_frame,
    build_logistic_output_frame,
    compute_logistic_association_with_missing_exclusion,
    iter_linear_output_frames,
    iter_logistic_output_frames,
)
from g.io.plink import (
    iter_genotype_chunks,
    iter_linear_genotype_chunks,
    load_aligned_sample_data,
    preprocess_genotype_matrix_arrays,
    read_bed_chunk_host,
)

if TYPE_CHECKING:
    from g.models import GenotypeChunk, LinearGenotypeChunk


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
    nvidia_smi_available: bool
    nvidia_smi_gpus: list[str]


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
    linear_preprocess: TimedMeasurement
    linear_compute: TimedMeasurement
    linear_format: TimedMeasurement
    linear_full_chunk_loop: TimedMeasurement
    logistic_standard_chunk_compute: TimedMeasurement | None
    logistic_standard_chunk_format: TimedMeasurement | None
    logistic_standard_chunk_compute_and_format: TimedMeasurement | None
    logistic_fallback_chunk_compute: TimedMeasurement | None
    logistic_fallback_chunk_format: TimedMeasurement | None
    logistic_fallback_chunk_compute_and_format: TimedMeasurement | None
    logistic_full_chunk_loop: TimedMeasurement | None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the JAX execution benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark JAX execution surfaces for GWAS kernels.")
    parser.add_argument("--bed-prefix", type=Path, default=Path("data/1kg_chr22_full"), help="PLINK dataset prefix.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Variants per chunk.")
    parser.add_argument("--variant-limit", type=int, help="Variant count used for standard-chunk measurements.")
    parser.add_argument(
        "--glm",
        choices=("linear", "logistic", "all"),
        default="all",
        help="Benchmark scope. Use 'linear' to skip logistic measurements.",
    )
    parser.add_argument(
        "--search-variant-limit",
        type=int,
        default=4096,
        help="Variant search window used to locate a fallback-heavy chunk.",
    )
    parser.add_argument("--repeat-count", type=int, default=5, help="Number of warmed timing repetitions.")
    parser.add_argument("--output-path", type=Path, help="Optional JSON output path.")
    return parser


def collect_runtime_summary() -> RuntimeSummary:
    """Collect JAX backend and visible device information."""
    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi_gpus: list[str] = []
    if nvidia_smi_path is not None:
        completed_process = subprocess.run(
            [nvidia_smi_path, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed_process.returncode == 0:
            nvidia_smi_gpus = [line for line in completed_process.stdout.splitlines() if line]

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
        nvidia_smi_available=nvidia_smi_path is not None,
        nvidia_smi_gpus=nvidia_smi_gpus,
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


def count_firth_variants(logistic_evaluation: Any) -> int:
    """Count Firth-fallback variants in one logistic evaluation."""
    method_code = np.asarray(logistic_evaluation.logistic_result.method_code)
    return int(np.count_nonzero(method_code == LogisticMethod.FIRTH))


def checksum_frame(output_frame: Any) -> float:
    """Build a stable checksum from a formatted Polars frame."""
    return float(output_frame.select(pl.col("p_value").sum()).item())


def checksum_frame_list(output_frames: list[LinearChunkAccumulator] | list[LogisticChunkAccumulator]) -> float:
    """Build a stable checksum from a list of chunk accumulators."""
    checksum = 0.0
    for output_frame in output_frames:
        if isinstance(output_frame, LinearChunkAccumulator):
            checksum += float(np.asarray(output_frame.linear_result.p_value).sum())
        else:
            checksum += float(np.asarray(output_frame.logistic_result.p_value).sum())
    return checksum


def select_fallback_logistic_chunk(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    search_variant_limit: int,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> GenotypeChunk:
    """Find one chunk that exercises hybrid logistic fallback."""
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
        firth_variant_count = count_firth_variants(logistic_evaluation)
        if firth_variant_count > 0:
            return genotype_chunk

    raise RuntimeError("Could not find a fallback logistic chunk in the configured search range.")


def main() -> None:
    """Benchmark representative JAX transfer, compute, and formatting surfaces."""
    arguments = build_argument_parser().parse_args()
    bed_prefix = arguments.bed_prefix
    chunk_size = arguments.chunk_size
    variant_limit = arguments.variant_limit or chunk_size
    search_variant_limit = arguments.search_variant_limit
    repeat_count = arguments.repeat_count
    benchmark_scope = arguments.glm

    continuous_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_cont.txt"),
        phenotype_name="phenotype_continuous",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )
    binary_sample_data = None
    if benchmark_scope in {"logistic", "all"}:
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

    continuous_chunk: LinearGenotypeChunk = next(
        iter_linear_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=continuous_sample_data.sample_indices,
            expected_individual_identifiers=continuous_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    )
    logistic_standard_chunk_compute_measurement = None
    logistic_standard_chunk_format_measurement = None
    logistic_standard_chunk_compute_and_format_measurement = None
    logistic_fallback_chunk_compute_measurement = None
    logistic_fallback_chunk_format_measurement = None
    logistic_fallback_chunk_compute_and_format_measurement = None
    logistic_full_chunk_loop_measurement = None
    if benchmark_scope in {"logistic", "all"}:
        assert binary_sample_data is not None
        binary_chunk = next(
            iter_genotype_chunks(
                bed_prefix=bed_prefix,
                sample_indices=binary_sample_data.sample_indices,
                expected_individual_identifiers=binary_sample_data.individual_identifiers,
                chunk_size=chunk_size,
                variant_limit=variant_limit,
            )
        )
        fallback_chunk = select_fallback_logistic_chunk(
            bed_prefix=bed_prefix,
            sample_indices=binary_sample_data.sample_indices,
            expected_individual_identifiers=binary_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            search_variant_limit=search_variant_limit,
            covariate_matrix=binary_sample_data.covariate_matrix,
            phenotype_vector=binary_sample_data.phenotype_vector,
        )

        logistic_standard_chunk_compute_measurement = time_operation(
            operation=lambda: compute_logistic_association_chunk(
                covariate_matrix=binary_sample_data.covariate_matrix,
                phenotype_vector=binary_sample_data.phenotype_vector,
                genotype_matrix=binary_chunk.genotypes,
                max_iterations=100,
                tolerance=1.0e-8,
            ),
            checksum_operation=lambda logistic_result: float(
                np.asarray(jnp.sum(logistic_result.beta) + jnp.sum(logistic_result.p_value))
            ),
            repeat_count=repeat_count,
        )
        standard_logistic_result = compute_logistic_association_chunk(
            covariate_matrix=binary_sample_data.covariate_matrix,
            phenotype_vector=binary_sample_data.phenotype_vector,
            genotype_matrix=binary_chunk.genotypes,
            max_iterations=100,
            tolerance=1.0e-8,
        )
        block_tree_until_ready(standard_logistic_result)
        logistic_standard_chunk_format_measurement = time_operation(
            operation=lambda: build_logistic_output_frame(
                metadata=binary_chunk.metadata,
                allele_one_frequency=binary_chunk.allele_one_frequency,
                observation_count=binary_chunk.observation_count,
                logistic_result=standard_logistic_result,
            ),
            checksum_operation=checksum_frame,
            repeat_count=repeat_count,
        )
        logistic_standard_chunk_compute_and_format_measurement = time_operation(
            operation=lambda: build_logistic_output_frame(
                metadata=binary_chunk.metadata,
                allele_one_frequency=binary_chunk.allele_one_frequency,
                observation_count=binary_chunk.observation_count,
                logistic_result=compute_logistic_association_chunk(
                    covariate_matrix=binary_sample_data.covariate_matrix,
                    phenotype_vector=binary_sample_data.phenotype_vector,
                    genotype_matrix=binary_chunk.genotypes,
                    max_iterations=100,
                    tolerance=1.0e-8,
                ),
            ),
            checksum_operation=checksum_frame,
            repeat_count=repeat_count,
        )
        logistic_fallback_chunk_compute_measurement = time_operation(
            operation=lambda: compute_logistic_association_with_missing_exclusion(
                covariate_matrix=binary_sample_data.covariate_matrix,
                phenotype_vector=binary_sample_data.phenotype_vector,
                genotype_chunk=fallback_chunk,
                max_iterations=100,
                tolerance=1.0e-8,
            ),
            checksum_operation=checksum_logistic_result,
            repeat_count=repeat_count,
        )
        fallback_logistic_evaluation = compute_logistic_association_with_missing_exclusion(
            covariate_matrix=binary_sample_data.covariate_matrix,
            phenotype_vector=binary_sample_data.phenotype_vector,
            genotype_chunk=fallback_chunk,
            max_iterations=100,
            tolerance=1.0e-8,
        )
        block_tree_until_ready(fallback_logistic_evaluation)
        logistic_fallback_chunk_format_measurement = time_operation(
            operation=lambda: build_logistic_output_frame(
                metadata=fallback_chunk.metadata,
                allele_one_frequency=fallback_logistic_evaluation.allele_one_frequency,
                observation_count=fallback_logistic_evaluation.observation_count,
                logistic_result=fallback_logistic_evaluation.logistic_result,
            ),
            checksum_operation=checksum_frame,
            repeat_count=repeat_count,
        )
        logistic_fallback_chunk_compute_and_format_measurement = time_operation(
            operation=lambda: build_logistic_output_frame(
                metadata=fallback_chunk.metadata,
                allele_one_frequency=(
                    logistic_evaluation := compute_logistic_association_with_missing_exclusion(
                        covariate_matrix=binary_sample_data.covariate_matrix,
                        phenotype_vector=binary_sample_data.phenotype_vector,
                        genotype_chunk=fallback_chunk,
                        max_iterations=100,
                        tolerance=1.0e-8,
                    )
                ).allele_one_frequency,
                observation_count=logistic_evaluation.observation_count,
                logistic_result=logistic_evaluation.logistic_result,
            ),
            checksum_operation=checksum_frame,
            repeat_count=repeat_count,
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
    linear_preprocess_measurement = time_operation(
        operation=lambda: preprocess_genotype_matrix_arrays(jax.device_put(genotype_matrix_host)),
        checksum_operation=lambda preprocessed_arrays: float(
            np.asarray(jnp.sum(preprocessed_arrays.genotypes) + jnp.sum(preprocessed_arrays.allele_one_frequency))
        ),
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
    linear_full_chunk_loop_measurement = time_operation(
        operation=lambda: list(
            iter_linear_output_frames(
                bed_prefix=bed_prefix,
                phenotype_path=Path("data/pheno_cont.txt"),
                phenotype_name="phenotype_continuous",
                covariate_path=Path("data/covariates.txt"),
                covariate_names=("age", "sex"),
                chunk_size=chunk_size,
                variant_limit=2048,
            )
        ),
        checksum_operation=checksum_frame_list,
        repeat_count=repeat_count,
    )
    if benchmark_scope in {"logistic", "all"}:
        logistic_full_chunk_loop_measurement = time_operation(
            operation=lambda: list(
                iter_logistic_output_frames(
                    bed_prefix=bed_prefix,
                    phenotype_path=Path("data/pheno_bin.txt"),
                    phenotype_name="phenotype_binary",
                    covariate_path=Path("data/covariates.txt"),
                    covariate_names=("age", "sex"),
                    chunk_size=chunk_size,
                    variant_limit=2048,
                    max_iterations=100,
                    tolerance=1.0e-8,
                )
            ),
            checksum_operation=checksum_frame_list,
            repeat_count=repeat_count,
        )
    report = JaxExecutionBenchmarkReport(
        runtime=collect_runtime_summary(),
        variant_limit=variant_limit,
        chunk_size=chunk_size,
        repeat_count=repeat_count,
        host_chunk_read_seconds=host_chunk_read_seconds,
        device_put=device_put_measurement,
        linear_preprocess=linear_preprocess_measurement,
        linear_compute=linear_compute_measurement,
        linear_format=linear_format_measurement,
        linear_full_chunk_loop=linear_full_chunk_loop_measurement,
        logistic_standard_chunk_compute=logistic_standard_chunk_compute_measurement,
        logistic_standard_chunk_format=logistic_standard_chunk_format_measurement,
        logistic_standard_chunk_compute_and_format=logistic_standard_chunk_compute_and_format_measurement,
        logistic_fallback_chunk_compute=logistic_fallback_chunk_compute_measurement,
        logistic_fallback_chunk_format=logistic_fallback_chunk_format_measurement,
        logistic_fallback_chunk_compute_and_format=logistic_fallback_chunk_compute_and_format_measurement,
        logistic_full_chunk_loop=logistic_full_chunk_loop_measurement,
    )
    report_json = json.dumps(asdict(report), indent=2)
    if arguments.output_path is not None:
        arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_path.write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    main()
