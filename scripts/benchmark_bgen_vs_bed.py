#!/usr/bin/env python3
"""Benchmark BED and BGEN paths on the same dataset."""

from __future__ import annotations

import argparse
import json
import time
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from g.engine import iter_linear_output_frames
from g.io.source import (
    GenotypeSourceConfig,
    build_bgen_source_config,
    build_plink_source_config,
    iter_genotype_chunks_from_source,
    load_aligned_sample_data_from_source,
)

if typing.TYPE_CHECKING:
    import collections.abc


@dataclass(frozen=True)
class BenchmarkPathResult:
    """Repeated timing and checksum output for one benchmark path."""

    durations_seconds: list[float]
    mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class SourceBenchmarkReport:
    """Benchmark output for one genotype source format."""

    source_format: str
    source_path: str
    sample_path: str | None
    chunk_iterator: BenchmarkPathResult
    linear_association: BenchmarkPathResult


@dataclass(frozen=True)
class BgenVsBedBenchmarkReport:
    """Structured benchmark output for matched BED and BGEN runs."""

    variant_limit: int
    chunk_size: int
    repeat_count: int
    prefetch_chunks: int
    bed: SourceBenchmarkReport
    bgen: SourceBenchmarkReport
    relative_runtime: dict[str, float]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for the matched BED-vs-BGEN benchmark."""
    argument_parser = argparse.ArgumentParser(description="Benchmark BED and BGEN ingestion on the same dataset.")
    argument_parser.add_argument("--bfile", type=Path, default=Path("data/1kg_chr22_full"))
    argument_parser.add_argument("--bgen", type=Path, default=Path("data/1kg_chr22_full.bgen"))
    argument_parser.add_argument("--sample", type=Path, default=Path("data/1kg_chr22_full.sample"))
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


def force_ready_sum(array_value: jax.Array) -> float:
    """Synchronize a JAX array and return a deterministic checksum."""
    return float(np.asarray(jax.device_get(jax.numpy.nansum(array_value))))


def benchmark_chunk_iterator(
    genotype_source_config: GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...],
    chunk_size: int,
    variant_limit: int,
    prefetch_chunks: int,
) -> float:
    """Benchmark the maintained genotype chunk iterator for one source."""
    aligned_sample_data = load_aligned_sample_data_from_source(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    checksum = 0.0
    for genotype_chunk in iter_genotype_chunks_from_source(
        genotype_source_config=genotype_source_config,
        sample_indices=aligned_sample_data.sample_indices,
        expected_individual_identifiers=aligned_sample_data.individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        prefetch_chunks=prefetch_chunks,
    ):
        checksum += force_ready_sum(genotype_chunk.genotypes)
        checksum += force_ready_sum(genotype_chunk.allele_one_frequency)
        checksum += force_ready_sum(genotype_chunk.observation_count)
    return checksum


def benchmark_linear_association(
    genotype_source_config: GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...],
    chunk_size: int,
    variant_limit: int,
    prefetch_chunks: int,
) -> float:
    """Benchmark the maintained linear association pipeline for one source."""
    checksum = 0.0
    for linear_chunk_accumulator in iter_linear_output_frames(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        prefetch_chunks=prefetch_chunks,
    ):
        checksum += force_ready_sum(linear_chunk_accumulator.linear_result.beta)
        checksum += force_ready_sum(linear_chunk_accumulator.linear_result.standard_error)
        checksum += force_ready_sum(linear_chunk_accumulator.allele_one_frequency)
    return checksum


def time_operation(operation: collections.abc.Callable[[], float], repeat_count: int) -> BenchmarkPathResult:
    """Warm and repeatedly time one benchmark operation."""
    warmup_checksum = operation()
    duration_seconds: list[float] = []
    checksum = warmup_checksum
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return BenchmarkPathResult(
        durations_seconds=duration_seconds,
        mean_seconds=sum(duration_seconds) / len(duration_seconds),
        checksum=checksum,
    )


def validate_alignment(
    bed_source_config: GenotypeSourceConfig,
    bgen_source_config: GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path,
    covariate_names: tuple[str, ...],
) -> None:
    """Ensure the benchmark compares the same aligned samples for both formats."""
    bed_aligned_sample_data = load_aligned_sample_data_from_source(
        genotype_source_config=bed_source_config,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    bgen_aligned_sample_data = load_aligned_sample_data_from_source(
        genotype_source_config=bgen_source_config,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    if not np.array_equal(
        bed_aligned_sample_data.individual_identifiers,
        bgen_aligned_sample_data.individual_identifiers,
    ):
        message = "BED and BGEN aligned sample identifiers differ. Benchmark inputs must describe the same cohort."
        raise ValueError(message)


def main() -> None:
    """Run a matched BED-vs-BGEN benchmark and print structured JSON."""
    arguments = build_argument_parser().parse_args()
    covariate_names = parse_covariate_names(arguments.covar_names)
    bed_source_config = build_plink_source_config(arguments.bfile)
    bgen_source_config = build_bgen_source_config(arguments.bgen, sample_path=arguments.sample)

    validate_alignment(
        bed_source_config=bed_source_config,
        bgen_source_config=bgen_source_config,
        phenotype_path=arguments.pheno,
        phenotype_name=arguments.pheno_name,
        covariate_path=arguments.covar,
        covariate_names=covariate_names,
    )

    bed_chunk_iterator = time_operation(
        lambda: benchmark_chunk_iterator(
            genotype_source_config=bed_source_config,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            prefetch_chunks=arguments.prefetch_chunks,
        ),
        repeat_count=arguments.repeat_count,
    )
    bgen_chunk_iterator = time_operation(
        lambda: benchmark_chunk_iterator(
            genotype_source_config=bgen_source_config,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            prefetch_chunks=arguments.prefetch_chunks,
        ),
        repeat_count=arguments.repeat_count,
    )
    bed_linear_association = time_operation(
        lambda: benchmark_linear_association(
            genotype_source_config=bed_source_config,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            prefetch_chunks=arguments.prefetch_chunks,
        ),
        repeat_count=arguments.repeat_count,
    )
    bgen_linear_association = time_operation(
        lambda: benchmark_linear_association(
            genotype_source_config=bgen_source_config,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            prefetch_chunks=arguments.prefetch_chunks,
        ),
        repeat_count=arguments.repeat_count,
    )

    if not np.isclose(bed_chunk_iterator.checksum, bgen_chunk_iterator.checksum, atol=1.0e-5):
        message = (
            f"BED and BGEN chunk iterator checksums differ: {bed_chunk_iterator.checksum} "
            f"vs {bgen_chunk_iterator.checksum}."
        )
        raise ValueError(message)
    if not np.isclose(bed_linear_association.checksum, bgen_linear_association.checksum, atol=1.0e-5):
        message = (
            f"BED and BGEN linear association checksums differ: {bed_linear_association.checksum} "
            f"vs {bgen_linear_association.checksum}."
        )
        raise ValueError(message)

    benchmark_report = BgenVsBedBenchmarkReport(
        variant_limit=arguments.variant_limit,
        chunk_size=arguments.chunk_size,
        repeat_count=arguments.repeat_count,
        prefetch_chunks=arguments.prefetch_chunks,
        bed=SourceBenchmarkReport(
            source_format=bed_source_config.source_format,
            source_path=str(bed_source_config.source_path),
            sample_path=None,
            chunk_iterator=bed_chunk_iterator,
            linear_association=bed_linear_association,
        ),
        bgen=SourceBenchmarkReport(
            source_format=bgen_source_config.source_format,
            source_path=str(bgen_source_config.source_path),
            sample_path=None if bgen_source_config.sample_path is None else str(bgen_source_config.sample_path),
            chunk_iterator=bgen_chunk_iterator,
            linear_association=bgen_linear_association,
        ),
        relative_runtime={
            "chunk_iterator_bgen_vs_bed": bgen_chunk_iterator.mean_seconds / bed_chunk_iterator.mean_seconds,
            "linear_association_bgen_vs_bed": (
                bgen_linear_association.mean_seconds / bed_linear_association.mean_seconds
            ),
        },
    )
    print(json.dumps(asdict(benchmark_report), indent=2))


if __name__ == "__main__":
    main()
