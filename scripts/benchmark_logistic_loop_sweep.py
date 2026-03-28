#!/usr/bin/env python3
"""Benchmark full logistic loop behavior across chunk sizes and boundaries."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import polars as pl

from g.engine import build_logistic_output_frame, compute_logistic_association_with_missing_exclusion
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data


@dataclass(frozen=True)
class LoopMeasurement:
    """Warmed runtime measurement for one logistic loop mode."""

    chunk_size: int
    warmed_durations_seconds: list[float]
    warmed_mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class LogisticLoopSweepReport:
    """Structured logistic loop benchmark report."""

    backend: str
    variant_limit: int
    repeat_count: int
    chunk_sizes: list[int]
    compute_only: list[LoopMeasurement]
    compute_and_format: list[LoopMeasurement]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the logistic loop sweep."""
    parser = argparse.ArgumentParser(description="Benchmark full logistic loop behavior across chunk sizes.")
    parser.add_argument("--bed-prefix", type=Path, default=Path("data/1kg_chr22_full"), help="PLINK dataset prefix.")
    parser.add_argument("--variant-limit", type=int, default=4096, help="Maximum number of variants to process.")
    parser.add_argument("--repeat-count", type=int, default=5, help="Number of warmed timing repetitions.")
    parser.add_argument(
        "--chunk-sizes",
        default="128,256,512,1024",
        help="Comma-separated chunk sizes to benchmark.",
    )
    parser.add_argument("--output-path", type=Path, help="Optional JSON output path.")
    return parser


def parse_chunk_sizes(raw_chunk_sizes: str) -> list[int]:
    """Parse a comma-separated chunk size list."""
    return [int(chunk_size.strip()) for chunk_size in raw_chunk_sizes.split(",") if chunk_size.strip()]


def block_tree_until_ready(value: Any) -> Any:
    """Synchronize a JAX pytree and return it unchanged."""
    return jax.block_until_ready(value)


def checksum_frame(output_frame: pl.DataFrame) -> float:
    """Build a stable checksum from one output frame."""
    return float(output_frame.select(pl.col("p_value").sum()).item())


def time_operation(operation: Any, repeat_count: int) -> tuple[list[float], float]:
    """Warm and measure one benchmark operation."""
    operation()
    warmed_durations_seconds: list[float] = []
    checksum = 0.0
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = float(operation())
        warmed_durations_seconds.append(time.perf_counter() - start_time)
    return warmed_durations_seconds, checksum


def benchmark_compute_only(
    bed_prefix: Path,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Benchmark chunk iteration plus logistic compute without output formatting."""
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_bin.txt"),
        phenotype_name="phenotype_binary",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=True,
    )
    checksum = 0.0
    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        logistic_evaluation = compute_logistic_association_with_missing_exclusion(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
            genotype_chunk=genotype_chunk,
            max_iterations=100,
            tolerance=1.0e-8,
        )
        block_tree_until_ready(logistic_evaluation)
        checksum += float(np.asarray(logistic_evaluation.logistic_result.p_value).sum())
    return checksum


def benchmark_compute_and_format(
    bed_prefix: Path,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Benchmark chunk iteration, logistic compute, and frame formatting."""
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_bin.txt"),
        phenotype_name="phenotype_binary",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=True,
    )
    checksum = 0.0
    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        logistic_evaluation = compute_logistic_association_with_missing_exclusion(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
            genotype_chunk=genotype_chunk,
            max_iterations=100,
            tolerance=1.0e-8,
        )
        output_frame = build_logistic_output_frame(
            metadata=genotype_chunk.metadata,
            allele_one_frequency=logistic_evaluation.allele_one_frequency,
            observation_count=logistic_evaluation.observation_count,
            logistic_result=logistic_evaluation.logistic_result,
        )
        checksum += checksum_frame(output_frame)
    return checksum


def main() -> None:
    """Run a logistic loop sweep on the active JAX backend."""
    arguments = build_argument_parser().parse_args()
    bed_prefix = arguments.bed_prefix
    variant_limit = arguments.variant_limit
    repeat_count = arguments.repeat_count
    chunk_sizes = parse_chunk_sizes(arguments.chunk_sizes)

    compute_only_measurements: list[LoopMeasurement] = []
    compute_and_format_measurements: list[LoopMeasurement] = []

    for chunk_size in chunk_sizes:
        compute_only_durations_seconds, compute_only_checksum = time_operation(
            operation=lambda chunk_size=chunk_size: benchmark_compute_only(
                bed_prefix=bed_prefix,
                chunk_size=chunk_size,
                variant_limit=variant_limit,
            ),
            repeat_count=repeat_count,
        )
        compute_only_measurements.append(
            LoopMeasurement(
                chunk_size=chunk_size,
                warmed_durations_seconds=compute_only_durations_seconds,
                warmed_mean_seconds=sum(compute_only_durations_seconds) / len(compute_only_durations_seconds),
                checksum=compute_only_checksum,
            )
        )

        compute_and_format_durations_seconds, compute_and_format_checksum = time_operation(
            operation=lambda chunk_size=chunk_size: benchmark_compute_and_format(
                bed_prefix=bed_prefix,
                chunk_size=chunk_size,
                variant_limit=variant_limit,
            ),
            repeat_count=repeat_count,
        )
        compute_and_format_measurements.append(
            LoopMeasurement(
                chunk_size=chunk_size,
                warmed_durations_seconds=compute_and_format_durations_seconds,
                warmed_mean_seconds=sum(compute_and_format_durations_seconds)
                / len(compute_and_format_durations_seconds),
                checksum=compute_and_format_checksum,
            )
        )

    report = LogisticLoopSweepReport(
        backend=jax.default_backend(),
        variant_limit=variant_limit,
        repeat_count=repeat_count,
        chunk_sizes=chunk_sizes,
        compute_only=compute_only_measurements,
        compute_and_format=compute_and_format_measurements,
    )
    report_json = json.dumps(asdict(report), indent=2)
    if arguments.output_path is not None:
        arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_path.write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    main()
