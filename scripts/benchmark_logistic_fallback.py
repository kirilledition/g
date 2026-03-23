#!/usr/bin/env python3
"""Benchmark logistic fallback orchestration on representative chunks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from g.compute.logistic import LOGISTIC_METHOD_FIRTH, LOGISTIC_METHOD_STANDARD
from g.engine import compute_logistic_association_with_missing_exclusion
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data


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
        firth_variant_count += int(np.count_nonzero(method_code == LOGISTIC_METHOD_FIRTH))
        standard_variant_count += int(np.count_nonzero(method_code == LOGISTIC_METHOD_STANDARD))
        checksum += float(np.asarray(logistic_evaluation.logistic_result.beta).sum())
        checksum += float(np.asarray(logistic_evaluation.logistic_result.p_value).sum())
    return checksum, firth_variant_count, standard_variant_count


def main() -> None:
    """Run the logistic fallback benchmark on the standard chr22 binary dataset."""
    bed_prefix = Path("data/1kg_chr22_full")
    phenotype_path = Path("data/pheno_bin.txt")
    covariate_path = Path("data/covariates.txt")
    variant_limit = 2048
    chunk_size = 256
    repeat_count = 5

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
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
