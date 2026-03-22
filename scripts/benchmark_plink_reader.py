#!/usr/bin/env python3
"""Benchmark the optimized PLINK BED chunk reader against the prior path."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed
from bed_reader._open_bed import get_num_threads

from g.io.plink import read_bed_chunk
from g.io.tabular import load_aligned_sample_data

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ReaderBenchmarkResult:
    """Structured timing results for the reader comparison."""

    variant_limit: int
    chunk_size: int
    repeat_count: int
    old_seconds: list[float]
    new_seconds: list[float]
    old_mean_seconds: float
    new_mean_seconds: float
    speedup_ratio: float
    checksum: float


def read_old_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through the original `bed_handle.read` path and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        observed_individual_identifiers = np.asarray(bed_handle.iid)[sample_index_array]
        if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
            message = "BED sample order does not match the aligned phenotype/covariate order."
            raise ValueError(message)

        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix = jnp.asarray(
                bed_handle.read(
                    index=np.s_[sample_index_array, variant_start:variant_stop],
                    dtype=np.float64,
                    order="C",
                )
            )
            checksum += float(jnp.nansum(genotype_matrix))

    return checksum


def read_new_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through the optimized direct-reader path and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        observed_individual_identifiers = np.asarray(bed_handle.iid)[sample_index_array]
        if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
            message = "BED sample order does not match the aligned phenotype/covariate order."
            raise ValueError(message)

        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))
        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix = read_bed_chunk(
                bed_handle=bed_handle,
                bed_path=bed_path,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
                num_threads=num_threads,
            )
            checksum += float(jnp.nansum(genotype_matrix))

    return checksum


def time_reader(read_operation: Callable[[], float], repeat_count: int) -> tuple[list[float], float]:
    """Measure repeated reader timings and return durations with final checksum."""
    duration_seconds: list[float] = []
    checksum = 0.0
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = read_operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return duration_seconds, checksum


def main() -> None:
    """Run the reader benchmark on the standard chr22 dataset and print JSON."""
    bed_prefix = Path("data/1kg_chr22_full")
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_cont.txt"),
        phenotype_name="phenotype_continuous",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )
    sample_index_array = np.ascontiguousarray(aligned_sample_data.sample_indices, dtype=np.intp)
    variant_limit = 2048
    chunk_size = 256
    repeat_count = 5

    old_seconds, old_checksum = time_reader(
        lambda: read_old_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    new_seconds, new_checksum = time_reader(
        lambda: read_new_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )

    if not np.isclose(old_checksum, new_checksum, atol=1.0e-6):
        message = f"Benchmark checksum mismatch: old={old_checksum}, new={new_checksum}."
        raise ValueError(message)

    benchmark_result = ReaderBenchmarkResult(
        variant_limit=variant_limit,
        chunk_size=chunk_size,
        repeat_count=repeat_count,
        old_seconds=old_seconds,
        new_seconds=new_seconds,
        old_mean_seconds=sum(old_seconds) / len(old_seconds),
        new_mean_seconds=sum(new_seconds) / len(new_seconds),
        speedup_ratio=(sum(old_seconds) / len(old_seconds)) / (sum(new_seconds) / len(new_seconds)),
        checksum=old_checksum,
    )
    print(json.dumps(asdict(benchmark_result), indent=2))


if __name__ == "__main__":
    main()
