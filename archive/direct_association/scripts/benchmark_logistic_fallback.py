#!/usr/bin/env python3
"""Benchmark logistic fallback orchestration on representative chunks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from g.compute.logistic import LogisticMethod

from g.engine import compute_logistic_association_with_missing_exclusion
from g.io.plink import iter_genotype_chunks, load_aligned_sample_data


@dataclass(frozen=True)
class LogisticChunkBenchmarkResult:
    """Structured benchmark output for logistic fallback orchestration."""

    variant_limit: int
    chunk_size: int
    repeat_count: int
    chunk_durations_seconds: list[float]
    mean_seconds: float
    firth_variant_count: int
    standard_variant_count: int
    checksum: float


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the fallback benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark logistic fallback orchestration.")
    parser.add_argument("--bed-prefix", type=Path, default=Path("data/1kg_chr22_full"), help="PLINK dataset prefix.")
    parser.add_argument("--phenotype-path", type=Path, default=Path("data/pheno_bin.txt"), help="Phenotype file.")
    parser.add_argument("--covariate-path", type=Path, default=Path("data/covariates.txt"), help="Covariate file.")
    parser.add_argument("--variant-limit", type=int, default=2048, help="Maximum number of variants to process.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Variants per chunk.")
    parser.add_argument("--repeat-count", type=int, default=5, help="Number of warmed timing repetitions.")
    parser.add_argument("--output-path", type=Path, help="Optional JSON output path.")
    return parser


def benchmark_logistic_chunks(
    bed_prefix: Path,
    phenotype_path: Path,
    covariate_path: Path,
    variant_limit: int,
    chunk_size: int,
) -> tuple[float, int, int]:
    """Run logistic association chunking and return timing checksum counters."""
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=phenotype_path,
        phenotype_name="phenotype_binary",
        covariate_path=covariate_path,
        covariate_names=("age", "sex"),
        is_binary_trait=True,
    )

    checksum = 0.0
    firth_variant_count = 0
    standard_variant_count = 0
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
        method_code = np.asarray(logistic_evaluation.logistic_result.method_code)
        firth_variant_count += int(np.count_nonzero(method_code == LogisticMethod.FIRTH))
        standard_variant_count += int(np.count_nonzero(method_code == LogisticMethod.STANDARD))
        checksum += float(np.asarray(logistic_evaluation.logistic_result.beta).sum())
        checksum += float(np.asarray(logistic_evaluation.logistic_result.p_value).sum())
    return checksum, firth_variant_count, standard_variant_count


def main() -> None:
    """Run the logistic fallback benchmark on the standard chr22 binary dataset."""
    arguments = build_argument_parser().parse_args()
    bed_prefix = arguments.bed_prefix
    phenotype_path = arguments.phenotype_path
    covariate_path = arguments.covariate_path
    variant_limit = arguments.variant_limit
    chunk_size = arguments.chunk_size
    repeat_count = arguments.repeat_count

    benchmark_logistic_chunks(
        bed_prefix=bed_prefix,
        phenotype_path=phenotype_path,
        covariate_path=covariate_path,
        variant_limit=variant_limit,
        chunk_size=chunk_size,
    )

    chunk_durations_seconds: list[float] = []
    checksum = 0.0
    firth_variant_count = 0
    standard_variant_count = 0
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum, firth_variant_count, standard_variant_count = benchmark_logistic_chunks(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            covariate_path=covariate_path,
            variant_limit=variant_limit,
            chunk_size=chunk_size,
        )
        chunk_durations_seconds.append(time.perf_counter() - start_time)

    result = LogisticChunkBenchmarkResult(
        variant_limit=variant_limit,
        chunk_size=chunk_size,
        repeat_count=repeat_count,
        chunk_durations_seconds=chunk_durations_seconds,
        mean_seconds=sum(chunk_durations_seconds) / len(chunk_durations_seconds),
        firth_variant_count=firth_variant_count,
        standard_variant_count=standard_variant_count,
        checksum=checksum,
    )
    result_json = json.dumps(asdict(result), indent=2)
    if arguments.output_path is not None:
        arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_path.write_text(result_json)
    print(result_json)


if __name__ == "__main__":
    main()
