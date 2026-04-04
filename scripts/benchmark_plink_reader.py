#!/usr/bin/env python3
"""Benchmark genotype-ingestion paths, including prefetch overlap."""

from __future__ import annotations

import argparse
import json
import time
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed
from bed_reader._open_bed import get_num_threads

from g.io.plink import read_bed_chunk
from g.io.source import (
    iter_genotype_chunks_from_source,
    load_aligned_sample_data_from_source,
    resolve_genotype_source_config,
)

if typing.TYPE_CHECKING:
    import collections.abc

    from g.io.source import GenotypeSourceConfig


@dataclass(frozen=True)
class BenchmarkPathResult:
    """Repeated timing and checksum output for one benchmark path."""

    durations_seconds: list[float]
    mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class ReaderBenchmarkReport:
    """Structured benchmark output for the supported reader paths."""

    source_format: str
    source_path: str
    variant_limit: int
    chunk_size: int
    repeat_count: int
    prefetch_chunks: int
    bed_handle_read: BenchmarkPathResult | None
    direct_float32_read: BenchmarkPathResult | None
    sequential_iterator: BenchmarkPathResult
    prefetched_iterator: BenchmarkPathResult
    speedup_vs_sequential_iterator: dict[str, float]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for the reader benchmark."""
    argument_parser = argparse.ArgumentParser(description="Benchmark genotype reader paths.")
    argument_parser.add_argument("--bfile", type=Path, default=Path("data/1kg_chr22_full"))
    argument_parser.add_argument("--bgen", type=Path, default=None)
    argument_parser.add_argument("--sample", type=Path, default=None)
    argument_parser.add_argument("--pheno", type=Path, default=Path("data/pheno_cont.txt"))
    argument_parser.add_argument("--pheno-name", default="phenotype_continuous")
    argument_parser.add_argument("--covar", type=Path, default=Path("data/covariates.txt"))
    argument_parser.add_argument("--covar-names", default="age,sex")
    argument_parser.add_argument("--chunk-size", type=int, default=256)
    argument_parser.add_argument("--variant-limit", type=int, default=4096)
    argument_parser.add_argument("--repeat-count", type=int, default=5)
    argument_parser.add_argument("--prefetch-chunks", type=int, default=1)
    return argument_parser


def parse_covariate_names(raw_covariate_names: str) -> tuple[str, ...]:
    """Parse a comma-separated covariate string into a tuple."""
    return tuple(name.strip() for name in raw_covariate_names.split(",") if name.strip())


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
                    dtype=np.float32,
                    order="C",
                )
            )
            checksum += force_ready_sum(genotype_matrix)

    return checksum


def read_direct_float32_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through the optimized `read_f32` helper and return a checksum."""
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


def iterate_supported_genotype_chunks(
    genotype_source_config: GenotypeSourceConfig,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
    *,
    prefetch_chunks: int,
) -> float:
    """Run the maintained genotype chunk iterator and return a checksum."""
    checksum = 0.0
    for genotype_chunk in iter_genotype_chunks_from_source(
        genotype_source_config=genotype_source_config,
        sample_indices=sample_indices,
        expected_individual_identifiers=expected_individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        prefetch_chunks=prefetch_chunks,
    ):
        checksum += float(np.asarray(genotype_chunk.genotypes).sum())
        checksum += float(np.asarray(genotype_chunk.observation_count).sum())
    return checksum


def time_operation(
    read_operation: collections.abc.Callable[[], float],
    repeat_count: int,
) -> BenchmarkPathResult:
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
    sequential_iterator: BenchmarkPathResult,
    prefetched_iterator: BenchmarkPathResult,
    bed_handle_read: BenchmarkPathResult | None,
    direct_float32_read: BenchmarkPathResult | None,
) -> None:
    """Ensure equivalent benchmark paths produce matching checksums."""
    if not np.isfinite(sequential_iterator.checksum):
        message = f"Sequential iterator checksum is not finite: {sequential_iterator.checksum}."
        raise ValueError(message)
    if not np.isclose(sequential_iterator.checksum, prefetched_iterator.checksum, atol=1.0e-6):
        message = (
            f"Sequential and prefetched iterator checksums differ: {sequential_iterator.checksum} "
            f"vs {prefetched_iterator.checksum}."
        )
        raise ValueError(message)
    if bed_handle_read is None or direct_float32_read is None:
        return
    if not np.isclose(bed_handle_read.checksum, direct_float32_read.checksum, atol=1.0e-6):
        message = (
            f"Raw reader checksum mismatch for direct_float32_read: {direct_float32_read.checksum} "
            f"vs {bed_handle_read.checksum}."
        )
        raise ValueError(message)


def main() -> None:
    """Run raw-reader and chunk-iterator benchmarks on the requested dataset."""
    arguments = build_argument_parser().parse_args()
    genotype_source_config = resolve_genotype_source_config(arguments.bfile, arguments.bgen, arguments.sample)
    covariate_names = parse_covariate_names(arguments.covar_names)
    aligned_sample_data = load_aligned_sample_data_from_source(
        genotype_source_config=genotype_source_config,
        phenotype_path=arguments.pheno,
        phenotype_name=arguments.pheno_name,
        covariate_path=arguments.covar,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    sample_index_array = np.ascontiguousarray(aligned_sample_data.sample_indices, dtype=np.intp)

    bed_handle_read: BenchmarkPathResult | None = None
    direct_float32_read: BenchmarkPathResult | None = None
    if genotype_source_config.source_format == "plink":
        bed_prefix = genotype_source_config.source_path
        bed_handle_read = time_operation(
            lambda: read_bed_handle_chunk_path(
                bed_prefix=bed_prefix,
                sample_index_array=sample_index_array,
                expected_individual_identifiers=aligned_sample_data.individual_identifiers,
                chunk_size=arguments.chunk_size,
                variant_limit=arguments.variant_limit,
            ),
            repeat_count=arguments.repeat_count,
        )
        direct_float32_read = time_operation(
            lambda: read_direct_float32_chunk_path(
                bed_prefix=bed_prefix,
                sample_index_array=sample_index_array,
                expected_individual_identifiers=aligned_sample_data.individual_identifiers,
                chunk_size=arguments.chunk_size,
                variant_limit=arguments.variant_limit,
            ),
            repeat_count=arguments.repeat_count,
        )

    sequential_iterator = time_operation(
        lambda: iterate_supported_genotype_chunks(
            genotype_source_config=genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            prefetch_chunks=0,
        ),
        repeat_count=arguments.repeat_count,
    )
    prefetched_iterator = time_operation(
        lambda: iterate_supported_genotype_chunks(
            genotype_source_config=genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            prefetch_chunks=arguments.prefetch_chunks,
        ),
        repeat_count=arguments.repeat_count,
    )

    validate_checksums(
        sequential_iterator=sequential_iterator,
        prefetched_iterator=prefetched_iterator,
        bed_handle_read=bed_handle_read,
        direct_float32_read=direct_float32_read,
    )

    benchmark_report = ReaderBenchmarkReport(
        source_format=genotype_source_config.source_format,
        source_path=str(genotype_source_config.source_path),
        variant_limit=arguments.variant_limit,
        chunk_size=arguments.chunk_size,
        repeat_count=arguments.repeat_count,
        prefetch_chunks=arguments.prefetch_chunks,
        bed_handle_read=bed_handle_read,
        direct_float32_read=direct_float32_read,
        sequential_iterator=sequential_iterator,
        prefetched_iterator=prefetched_iterator,
        speedup_vs_sequential_iterator={
            "sequential_iterator": 1.0,
            "prefetched_iterator": sequential_iterator.mean_seconds / prefetched_iterator.mean_seconds,
            **(
                {}
                if bed_handle_read is None or direct_float32_read is None
                else {
                    "bed_handle_read": sequential_iterator.mean_seconds / bed_handle_read.mean_seconds,
                    "direct_float32_read": sequential_iterator.mean_seconds / direct_float32_read.mean_seconds,
                }
            ),
        },
    )
    print(json.dumps(asdict(benchmark_report), indent=2))


if __name__ == "__main__":
    main()
