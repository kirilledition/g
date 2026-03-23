#!/usr/bin/env python3
"""Benchmark PLINK chunk-ingestion paths and host-boundary overhead."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed
from bed_reader._open_bed import get_num_threads

from g.io.plink import (
    iter_genotype_chunks,
    preprocess_genotype_matrix_native,
    read_bed_chunk,
    read_bed_chunk_host,
    read_bed_chunk_native,
)
from g.io.tabular import load_aligned_sample_data

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BenchmarkPathResult:
    """Repeated timing and checksum output for one benchmark path."""

    durations_seconds: list[float]
    mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class ReaderBenchmarkReport:
    """Structured benchmark output for raw and chunk-level reader paths."""

    variant_limit: int
    chunk_size: int
    repeat_count: int
    bed_handle_read: BenchmarkPathResult
    read_f64: BenchmarkPathResult
    rust_native: BenchmarkPathResult
    read_f64_plus_rust_preprocess: BenchmarkPathResult
    iter_python: BenchmarkPathResult
    iter_rust_preprocess: BenchmarkPathResult
    iter_rust: BenchmarkPathResult
    speedup_vs_bed_handle_read: dict[str, float]
    speedup_vs_iter_python: dict[str, float]


def validate_sample_order(
    observed_individual_identifiers: np.ndarray,
    expected_individual_identifiers: np.ndarray,
) -> None:
    """Ensure benchmark reads use the aligned sample order."""
    if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
        message = "BED sample order does not match the aligned phenotype/covariate order."
        raise ValueError(message)


def force_ready_sum(genotype_matrix: jax.Array) -> float:
    """Synchronize a genotype matrix and return a deterministic checksum."""
    return float(np.asarray(jnp.nansum(genotype_matrix)))


def read_bed_handle_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through `bed_handle.read` and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        validate_sample_order(
            observed_individual_identifiers=np.asarray(bed_handle.iid)[sample_index_array],
            expected_individual_identifiers=expected_individual_identifiers,
        )

        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix = jnp.asarray(
                bed_handle.read(
                    index=np.s_[sample_index_array, variant_start:variant_stop],
                    dtype=np.float64,
                    order="C",
                )
            )
            checksum += force_ready_sum(genotype_matrix)

    return checksum


def read_direct_f64_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through the optimized `read_f64` helper and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        validate_sample_order(
            observed_individual_identifiers=np.asarray(bed_handle.iid)[sample_index_array],
            expected_individual_identifiers=expected_individual_identifiers,
        )

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
            checksum += force_ready_sum(genotype_matrix)

    return checksum


def read_native_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through the current Rust native path and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    for variant_start in range(0, variant_limit, chunk_size):
        variant_stop = min(variant_limit, variant_start + chunk_size)
        genotype_matrix = read_bed_chunk_native(
            bed_path=bed_path,
            sample_index_array=sample_index_array,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )
        checksum += force_ready_sum(genotype_matrix)
    return checksum


def read_direct_f64_plus_native_preprocess_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read through `read_f64`, then preprocess through Rust, and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        validate_sample_order(
            observed_individual_identifiers=np.asarray(bed_handle.iid)[sample_index_array],
            expected_individual_identifiers=expected_individual_identifiers,
        )

        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))
        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix_host = read_bed_chunk_host(
                bed_handle=bed_handle,
                bed_path=bed_path,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
                num_threads=num_threads,
            )
            preprocessed_chunk = preprocess_genotype_matrix_native(genotype_matrix_host)
            checksum += float(np.asarray(preprocessed_chunk.genotypes).sum())
            checksum += float(np.asarray(preprocessed_chunk.observation_count).sum())
    return checksum


def iterate_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
    *,
    use_native_reader: bool,
    use_native_preprocessing: bool,
) -> float:
    """Run the full genotype chunk iterator and return a checksum."""
    checksum = 0.0
    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=sample_indices,
        expected_individual_identifiers=expected_individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        use_native_reader=use_native_reader,
        use_native_preprocessing=use_native_preprocessing,
    ):
        checksum += float(np.asarray(genotype_chunk.genotypes).sum())
        checksum += float(np.asarray(genotype_chunk.observation_count).sum())
    return checksum


def time_operation(read_operation: Callable[[], float], repeat_count: int) -> BenchmarkPathResult:
    """Warm and repeatedly time one benchmark operation."""
    warmup_checksum = read_operation()
    duration_seconds: list[float] = []
    checksum = warmup_checksum
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = read_operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return BenchmarkPathResult(
        durations_seconds=duration_seconds,
        mean_seconds=sum(duration_seconds) / len(duration_seconds),
        checksum=checksum,
    )


def validate_checksums(
    bed_handle_read: BenchmarkPathResult,
    read_f64: BenchmarkPathResult,
    rust_native: BenchmarkPathResult,
    read_f64_plus_rust_preprocess: BenchmarkPathResult,
    iter_python: BenchmarkPathResult,
    iter_rust_preprocess: BenchmarkPathResult,
    iter_rust: BenchmarkPathResult,
) -> None:
    """Ensure equivalent benchmark paths produce matching checksums."""
    raw_reader_checksum = bed_handle_read.checksum
    for name, checksum in {
        "read_f64": read_f64.checksum,
        "rust_native": rust_native.checksum,
    }.items():
        if not np.isclose(raw_reader_checksum, checksum, atol=1.0e-6):
            message = f"Raw reader checksum mismatch for {name}: {checksum} vs {raw_reader_checksum}."
            raise ValueError(message)

    iterator_checksum = iter_python.checksum
    for name, checksum in {
        "read_f64_plus_rust_preprocess": read_f64_plus_rust_preprocess.checksum,
        "iter_rust_preprocess": iter_rust_preprocess.checksum,
        "iter_rust": iter_rust.checksum,
    }.items():
        if not np.isclose(iterator_checksum, checksum, atol=1.0e-6):
            message = f"Chunk iterator checksum mismatch for {name}: {checksum} vs {iterator_checksum}."
            raise ValueError(message)


def main() -> None:
    """Run raw-reader and chunk-iterator benchmarks on the standard chr22 dataset."""
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
    variant_limit = 4096
    chunk_size = 256
    repeat_count = 5

    bed_handle_read = time_operation(
        lambda: read_bed_handle_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    read_f64 = time_operation(
        lambda: read_direct_f64_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    rust_native = time_operation(
        lambda: read_native_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    read_f64_plus_rust_preprocess = time_operation(
        lambda: read_direct_f64_plus_native_preprocess_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    iter_python = time_operation(
        lambda: iterate_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            use_native_reader=False,
            use_native_preprocessing=False,
        ),
        repeat_count=repeat_count,
    )
    iter_rust_preprocess = time_operation(
        lambda: iterate_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            use_native_reader=False,
            use_native_preprocessing=True,
        ),
        repeat_count=repeat_count,
    )
    iter_rust = time_operation(
        lambda: iterate_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            use_native_reader=True,
            use_native_preprocessing=False,
        ),
        repeat_count=repeat_count,
    )

    validate_checksums(
        bed_handle_read=bed_handle_read,
        read_f64=read_f64,
        rust_native=rust_native,
        read_f64_plus_rust_preprocess=read_f64_plus_rust_preprocess,
        iter_python=iter_python,
        iter_rust_preprocess=iter_rust_preprocess,
        iter_rust=iter_rust,
    )

    benchmark_report = ReaderBenchmarkReport(
        variant_limit=variant_limit,
        chunk_size=chunk_size,
        repeat_count=repeat_count,
        bed_handle_read=bed_handle_read,
        read_f64=read_f64,
        rust_native=rust_native,
        read_f64_plus_rust_preprocess=read_f64_plus_rust_preprocess,
        iter_python=iter_python,
        iter_rust_preprocess=iter_rust_preprocess,
        iter_rust=iter_rust,
        speedup_vs_bed_handle_read={
            "bed_handle_read": 1.0,
            "read_f64": bed_handle_read.mean_seconds / read_f64.mean_seconds,
            "rust_native": bed_handle_read.mean_seconds / rust_native.mean_seconds,
        },
        speedup_vs_iter_python={
            "iter_python": 1.0,
            "iter_rust_preprocess": iter_python.mean_seconds / iter_rust_preprocess.mean_seconds,
            "iter_rust": iter_python.mean_seconds / iter_rust.mean_seconds,
        },
    )
    print(json.dumps(asdict(benchmark_report), indent=2))


if __name__ == "__main__":
    main()
