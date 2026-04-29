#!/usr/bin/env python3
"""Benchmark TSV vs Arrow chunks output modes for REGENIE step 2 linear regression.

This script compares three output configurations:
1. TSV mode - Single TSV file output
2. Arrow chunks mode (no finalization) - Chunked Arrow IPC files only
3. Arrow chunks mode (with finalization) - Chunked Arrow IPC files + final Parquet

The benchmark measures both isolated I/O performance and end-to-end runtime,
with output validation to ensure numerical equivalence across all modes.
"""

from __future__ import annotations

import contextlib
import json
import resource
import shutil
import tempfile
import time
import typing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
import polars as pl

from g import api, engine, jax_setup, types
from g.io import output as output_module

if typing.TYPE_CHECKING:
    import collections.abc

DEFAULT_BGEN_PATH = Path("data/1kg_chr22_full.bgen")
DEFAULT_SAMPLE_PATH = Path("data/1kg_chr22_full.sample")
DEFAULT_PHENOTYPE_PATH = Path("data/pheno_cont.txt")
DEFAULT_COVARIATE_PATH = Path("data/covariates.txt")
DEFAULT_PREDICTION_LIST_PATH = Path("data/baselines/regenie_step1_qt_pred.list")
DEFAULT_OUTPUT_BASE_DIRECTORY = Path("data/benchmarks/output_modes")
DEFAULT_CHUNK_SIZE = 2048
WARMUP_VARIANT_LIMIT = 1024
PHENOTYPE_NAME = "phenotype_continuous"
COVARIATE_NAMES = ("age", "sex")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark execution configuration.

    Attributes:
        bgen_path: Path to BGEN genotype file.
        sample_path: Path to BGEN sample file.
        phenotype_path: Path to phenotype file.
        phenotype_name: Name of phenotype column.
        covariate_path: Path to covariate file.
        covariate_names: Tuple of covariate column names.
        prediction_list_path: Path to REGENIE step 1 prediction list.
        chunk_size: Number of variants per chunk.
        device: JAX execution device.
        output_base_directory: Base directory for benchmark outputs.

    """

    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    phenotype_name: str
    covariate_path: Path
    covariate_names: tuple[str, ...]
    prediction_list_path: Path
    chunk_size: int
    device: types.Device
    output_base_directory: Path


@dataclass(frozen=True)
class BenchmarkResult:
    """Results from one benchmark run.

    Attributes:
        mode_name: Human-readable name for this mode.
        output_mode: Output mode enum value.
        finalize_parquet: Whether parquet finalization was enabled.
        total_wall_time_seconds: Total wall-clock time from start to finish.
        output_writing_time_seconds: Isolated I/O time for output writing.
        compute_time_seconds: Estimated compute time (total - I/O).
        total_variants: Number of variants processed.
        chunk_count: Number of chunks processed.
        variants_per_second: Throughput metric.
        output_size_bytes: Total disk space used by output.
        peak_memory_mb: Peak RSS memory usage in MB.
        output_path: Path to TSV output file if applicable.
        output_run_directory: Path to run directory for chunked output.
        final_parquet_path: Path to final parquet file if applicable.
        error: Error message if benchmark failed.

    """

    mode_name: str
    output_mode: types.OutputMode
    finalize_parquet: bool
    total_wall_time_seconds: float
    output_writing_time_seconds: float
    compute_time_seconds: float
    total_variants: int
    chunk_count: int
    variants_per_second: float
    output_size_bytes: int
    peak_memory_mb: float
    output_path: Path | None = None
    output_run_directory: Path | None = None
    final_parquet_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Output validation results.

    Attributes:
        passed: Whether validation passed.
        row_count_match: Whether row counts match.
        statistics_match: Whether statistics are numerically equivalent.
        message: Validation message or error details.

    """

    passed: bool
    row_count_match: bool
    statistics_match: bool
    message: str


@contextlib.contextmanager
def timer() -> collections.abc.Iterator[dict[str, float]]:
    """Context manager that records elapsed time.

    Yields:
        Dictionary with timing information that gets populated on exit.

    """
    timing: dict[str, float] = {"start": 0.0, "end": 0.0, "elapsed": 0.0}
    timing["start"] = time.perf_counter()
    try:
        yield timing
    finally:
        timing["end"] = time.perf_counter()
        timing["elapsed"] = timing["end"] - timing["start"]


def get_peak_memory_mb() -> float:
    """Get peak RSS memory usage in MB.

    Returns:
        Peak memory in megabytes.

    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux, ru_maxrss is in KB; on macOS it's in bytes
    return usage.ru_maxrss / 1024


def count_arrow_chunks(chunks_directory: Path) -> int:
    """Count committed Arrow chunk files.

    Args:
        chunks_directory: Directory containing chunk files.

    Returns:
        Number of chunk_*.arrow files found.

    """
    return len(list(chunks_directory.glob("chunk_*.arrow")))


def measure_output_size(result: BenchmarkResult) -> int:
    """Measure total disk space used by output.

    Args:
        result: Benchmark result containing output paths.

    Returns:
        Total size in bytes.

    """
    total_size = 0

    if result.output_path is not None and result.output_path.exists():
        # TSV mode: single file
        total_size = result.output_path.stat().st_size
    elif result.final_parquet_path is not None and result.final_parquet_path.exists():
        # Arrow mode with finalization: final parquet file
        total_size = result.final_parquet_path.stat().st_size
    elif result.output_run_directory is not None:
        # Arrow mode without finalization: sum all chunks
        chunks_directory = result.output_run_directory / "chunks"
        if chunks_directory.exists():
            for chunk_file in chunks_directory.glob("chunk_*.arrow"):
                total_size += chunk_file.stat().st_size

    return total_size


def run_warmup(config: BenchmarkConfig) -> float:
    """Run small warmup to compile GPU kernels.

    Executes a small subset of variants to trigger JAX kernel compilation,
    ensuring subsequent benchmarks measure steady-state performance.

    Args:
        config: Benchmark configuration.

    Returns:
        Warmup duration in seconds.

    """
    print("Running warmup to compile GPU kernels...")
    with tempfile.TemporaryDirectory() as temp_dir:
        warmup_output = Path(temp_dir) / "warmup"
        start_time = time.perf_counter()

        api.regenie2_linear(
            bgen=config.bgen_path,
            sample=config.sample_path,
            pheno=config.phenotype_path,
            pheno_name=config.phenotype_name,
            covar=config.covariate_path,
            covar_names=config.covariate_names,
            pred=config.prediction_list_path,
            out=warmup_output,
            compute=api.ComputeConfig(
                chunk_size=config.chunk_size,
                device=config.device,
                variant_limit=WARMUP_VARIANT_LIMIT,
                output_mode=types.OutputMode.TSV,
            ),
        )

        warmup_duration = time.perf_counter() - start_time

    print(f"Warmup completed in {warmup_duration:.2f}s")
    return warmup_duration


def run_tsv_benchmark(
    config: BenchmarkConfig,
    output_directory: Path,
) -> BenchmarkResult:
    """Benchmark TSV output mode.

    Args:
        config: Benchmark configuration.
        output_directory: Directory for output files.

    Returns:
        Benchmark results.

    """
    print("\n" + "=" * 80)
    print("Running TSV mode benchmark...")
    print("=" * 80)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "results"

    # Instrument write_frame_iterator_to_tsv to measure I/O time
    original_write_tsv = engine.write_frame_iterator_to_tsv
    io_timing: dict[str, float] = {}

    def instrumented_write_tsv(*args: typing.Any, **kwargs: typing.Any) -> None:
        nonlocal io_timing
        with timer() as io_timing:
            original_write_tsv(*args, **kwargs)

    try:
        # Temporarily replace the function
        setattr(engine, "write_frame_iterator_to_tsv", instrumented_write_tsv)

        # Run benchmark
        with timer() as total_timing:
            artifacts = api.regenie2_linear(
                bgen=config.bgen_path,
                sample=config.sample_path,
                pheno=config.phenotype_path,
                pheno_name=config.phenotype_name,
                covar=config.covariate_path,
                covar_names=config.covariate_names,
                pred=config.prediction_list_path,
                out=output_path,
                compute=api.ComputeConfig(
                    chunk_size=config.chunk_size,
                    device=config.device,
                    output_mode=types.OutputMode.TSV,
                ),
            )

        # Restore original function
        setattr(engine, "write_frame_iterator_to_tsv", original_write_tsv)

        # Load result to count variants
        assert artifacts.sumstats_tsv is not None
        result_frame = pl.read_csv(artifacts.sumstats_tsv, separator="\t")
        total_variants = len(result_frame)
        chunk_count = (total_variants + config.chunk_size - 1) // config.chunk_size

        # Measure output size
        output_size = artifacts.sumstats_tsv.stat().st_size

        total_time = total_timing["elapsed"]
        io_time = io_timing.get("elapsed", 0.0)
        compute_time = total_time - io_time
        throughput = total_variants / total_time if total_time > 0 else 0.0

        print(f"  Total time: {total_time:.2f}s")
        print(f"  I/O time: {io_time:.2f}s")
        print(f"  Compute time: {compute_time:.2f}s")
        print(f"  Variants: {total_variants:,}")
        print(f"  Throughput: {throughput:,.0f} variants/sec")
        print(f"  Output size: {output_size / 1024 / 1024:.1f} MB")

        return BenchmarkResult(
            mode_name="TSV",
            output_mode=types.OutputMode.TSV,
            finalize_parquet=False,
            total_wall_time_seconds=total_time,
            output_writing_time_seconds=io_time,
            compute_time_seconds=compute_time,
            total_variants=total_variants,
            chunk_count=chunk_count,
            variants_per_second=throughput,
            output_size_bytes=output_size,
            peak_memory_mb=get_peak_memory_mb(),
            output_path=artifacts.sumstats_tsv,
        )

    except Exception as error:
        print(f"  ERROR: {error}")
        return BenchmarkResult(
            mode_name="TSV",
            output_mode=types.OutputMode.TSV,
            finalize_parquet=False,
            total_wall_time_seconds=0.0,
            output_writing_time_seconds=0.0,
            compute_time_seconds=0.0,
            total_variants=0,
            chunk_count=0,
            variants_per_second=0.0,
            output_size_bytes=0,
            peak_memory_mb=get_peak_memory_mb(),
            error=str(error),
        )


def run_arrow_benchmark(
    config: BenchmarkConfig,
    output_directory: Path,
    finalize: bool,
) -> BenchmarkResult:
    """Benchmark Arrow chunks output mode.

    Args:
        config: Benchmark configuration.
        output_directory: Directory for output files.
        finalize: Whether to finalize chunks to Parquet.

    Returns:
        Benchmark results.

    """
    mode_label = "Arrow (with finalization)" if finalize else "Arrow (no finalization)"
    print("\n" + "=" * 80)
    print(f"Running {mode_label} benchmark...")
    print("=" * 80)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "results"

    # Instrument persist_chunked_results to measure I/O time
    original_persist = output_module.persist_chunked_results
    original_finalize = output_module.finalize_chunks_to_parquet
    persist_timing: dict[str, float] = {}
    finalize_timing: dict[str, float] = {}

    def instrumented_persist(*args: typing.Any, **kwargs: typing.Any) -> None:
        nonlocal persist_timing
        with timer() as persist_timing:
            original_persist(*args, **kwargs)

    def instrumented_finalize(*args: typing.Any, **kwargs: typing.Any) -> Path:
        nonlocal finalize_timing
        with timer() as finalize_timing:
            return original_finalize(*args, **kwargs)

    try:
        # Temporarily replace the functions
        setattr(output_module, "persist_chunked_results", instrumented_persist)
        setattr(output_module, "finalize_chunks_to_parquet", instrumented_finalize)

        # Run benchmark
        with timer() as total_timing:
            artifacts = api.regenie2_linear(
                bgen=config.bgen_path,
                sample=config.sample_path,
                pheno=config.phenotype_path,
                pheno_name=config.phenotype_name,
                covar=config.covariate_path,
                covar_names=config.covariate_names,
                pred=config.prediction_list_path,
                out=output_path,
                compute=api.ComputeConfig(
                    chunk_size=config.chunk_size,
                    device=config.device,
                    output_mode=types.OutputMode.ARROW_CHUNKS,
                    output_run_directory=output_directory,
                    finalize_parquet=finalize,
                ),
            )

        # Restore original functions
        setattr(output_module, "persist_chunked_results", original_persist)
        setattr(output_module, "finalize_chunks_to_parquet", original_finalize)

        # Count chunks and variants
        assert artifacts.output_run_directory is not None
        chunks_directory = artifacts.output_run_directory / "chunks"
        chunk_count = count_arrow_chunks(chunks_directory)

        # Load one output to count total variants
        if finalize and artifacts.final_parquet is not None:
            result_frame = pl.read_parquet(artifacts.final_parquet)
        else:
            # Load first chunk to get schema, then scan all
            chunk_files = sorted(chunks_directory.glob("chunk_*.arrow"))
            result_frame = pl.concat([pl.read_ipc(f) for f in chunk_files])

        total_variants = len(result_frame)

        # Calculate timing
        total_time = total_timing["elapsed"]
        persist_time = persist_timing.get("elapsed", 0.0)
        finalize_time = finalize_timing.get("elapsed", 0.0)
        io_time = persist_time + finalize_time
        compute_time = total_time - io_time
        throughput = total_variants / total_time if total_time > 0 else 0.0

        # Measure output size
        if finalize and artifacts.final_parquet is not None:
            output_size = artifacts.final_parquet.stat().st_size
        else:
            output_size = sum(f.stat().st_size for f in chunks_directory.glob("chunk_*.arrow"))

        print(f"  Total time: {total_time:.2f}s")
        print(f"  I/O time: {io_time:.2f}s (persist: {persist_time:.2f}s, finalize: {finalize_time:.2f}s)")
        print(f"  Compute time: {compute_time:.2f}s")
        print(f"  Variants: {total_variants:,}")
        print(f"  Chunks: {chunk_count}")
        print(f"  Throughput: {throughput:,.0f} variants/sec")
        print(f"  Output size: {output_size / 1024 / 1024:.1f} MB")

        return BenchmarkResult(
            mode_name=mode_label,
            output_mode=types.OutputMode.ARROW_CHUNKS,
            finalize_parquet=finalize,
            total_wall_time_seconds=total_time,
            output_writing_time_seconds=io_time,
            compute_time_seconds=compute_time,
            total_variants=total_variants,
            chunk_count=chunk_count,
            variants_per_second=throughput,
            output_size_bytes=output_size,
            peak_memory_mb=get_peak_memory_mb(),
            output_run_directory=artifacts.output_run_directory,
            final_parquet_path=artifacts.final_parquet,
        )

    except Exception as error:
        print(f"  ERROR: {error}")
        return BenchmarkResult(
            mode_name=mode_label,
            output_mode=types.OutputMode.ARROW_CHUNKS,
            finalize_parquet=finalize,
            total_wall_time_seconds=0.0,
            output_writing_time_seconds=0.0,
            compute_time_seconds=0.0,
            total_variants=0,
            chunk_count=0,
            variants_per_second=0.0,
            output_size_bytes=0,
            peak_memory_mb=get_peak_memory_mb(),
            error=str(error),
        )


def validate_outputs(
    tsv_result: BenchmarkResult,
    arrow_result: BenchmarkResult,
) -> ValidationResult:
    """Validate that TSV and Arrow outputs are numerically equivalent.

    Args:
        tsv_result: Results from TSV benchmark.
        arrow_result: Results from Arrow benchmark.

    Returns:
        Validation results.

    """
    print("\n" + "=" * 80)
    print("Validating output equivalence...")
    print("=" * 80)

    try:
        # Load TSV output
        if tsv_result.output_path is None or not tsv_result.output_path.exists():
            return ValidationResult(
                passed=False,
                row_count_match=False,
                statistics_match=False,
                message="TSV output file not found",
            )

        tsv_frame = pl.read_csv(tsv_result.output_path, separator="\t")

        # Load Arrow output
        if arrow_result.final_parquet_path is not None and arrow_result.final_parquet_path.exists():
            arrow_frame = pl.read_parquet(arrow_result.final_parquet_path)
        elif arrow_result.output_run_directory is not None:
            chunks_directory = arrow_result.output_run_directory / "chunks"
            chunk_files = sorted(chunks_directory.glob("chunk_*.arrow"))
            if not chunk_files:
                return ValidationResult(
                    passed=False,
                    row_count_match=False,
                    statistics_match=False,
                    message="No Arrow chunks found",
                )
            arrow_frame = pl.concat([pl.read_ipc(f) for f in chunk_files])
        else:
            return ValidationResult(
                passed=False,
                row_count_match=False,
                statistics_match=False,
                message="Arrow output not found",
            )

        # Sort both by key columns for comparison
        sort_columns = ["chromosome", "position", "variant_identifier"]
        tsv_sorted = tsv_frame.sort(sort_columns)
        arrow_sorted = arrow_frame.sort(sort_columns)

        # Check row counts
        tsv_rows = len(tsv_sorted)
        arrow_rows = len(arrow_sorted)
        row_count_match = tsv_rows == arrow_rows

        if not row_count_match:
            return ValidationResult(
                passed=False,
                row_count_match=False,
                statistics_match=False,
                message=f"Row count mismatch: TSV has {tsv_rows}, Arrow has {arrow_rows}",
            )

        print(f"  Row counts match: {tsv_rows:,} variants")

        # Compare statistics columns with tolerance
        statistics_columns = ["beta", "standard_error", "chi_squared", "log10_p_value"]
        tolerance = 1e-6

        statistics_match = True
        for column in statistics_columns:
            tsv_values = tsv_sorted[column].to_numpy()
            arrow_values = arrow_sorted[column].to_numpy()

            # Handle NaN/invalid values
            tsv_valid = ~np.isnan(tsv_values)
            arrow_valid = ~np.isnan(arrow_values)

            if not np.array_equal(tsv_valid, arrow_valid):
                print(f"  ✗ {column}: validity masks differ")
                statistics_match = False
                continue

            # Compare valid values
            valid_mask = tsv_valid & arrow_valid
            tsv_valid_values = tsv_values[valid_mask]
            arrow_valid_values = arrow_values[valid_mask]

            if not np.allclose(tsv_valid_values, arrow_valid_values, atol=tolerance, rtol=0):
                max_diff = np.max(np.abs(tsv_valid_values - arrow_valid_values))
                print(f"  ✗ {column}: numerical difference detected (max diff: {max_diff:.2e})")
                statistics_match = False
            else:
                print(f"  ✓ {column}: numerically equivalent")

        # Check is_valid flags
        tsv_valid_flags = tsv_sorted["is_valid"].to_numpy()
        arrow_valid_flags = arrow_sorted["is_valid"].to_numpy()

        if not np.array_equal(tsv_valid_flags, arrow_valid_flags):
            print("  ✗ is_valid flags differ")
            statistics_match = False
        else:
            print("  ✓ is_valid flags match")

        passed = row_count_match and statistics_match

        if passed:
            message = "All outputs are numerically equivalent"
        else:
            message = "Outputs differ in statistics or validity flags"

        return ValidationResult(
            passed=passed,
            row_count_match=row_count_match,
            statistics_match=statistics_match,
            message=message,
        )

    except Exception as error:
        return ValidationResult(
            passed=False,
            row_count_match=False,
            statistics_match=False,
            message=f"Validation error: {error}",
        )


def generate_report(
    config: BenchmarkConfig,
    warmup_duration: float,
    tsv_result: BenchmarkResult,
    arrow_no_finalize_result: BenchmarkResult,
    arrow_finalize_result: BenchmarkResult,
    validation_no_finalize: ValidationResult,
    validation_finalize: ValidationResult,
    run_directory: Path,
) -> str:
    """Generate formatted benchmark report.

    Args:
        config: Benchmark configuration.
        warmup_duration: Warmup duration in seconds.
        tsv_result: TSV benchmark results.
        arrow_no_finalize_result: Arrow (no finalization) results.
        arrow_finalize_result: Arrow (with finalization) results.
        validation_no_finalize: Validation for Arrow (no finalization).
        validation_finalize: Validation for Arrow (with finalization).
        run_directory: Benchmark run directory.

    Returns:
        Formatted report string.

    """
    lines = []
    lines.append("=" * 80)
    lines.append("Output Mode Benchmark: TSV vs Arrow Chunks")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("Configuration:")
    lines.append(f"  Dataset: {config.bgen_path}")
    lines.append(f"  Sample file: {config.sample_path}")
    lines.append(f"  Phenotype: {config.phenotype_name}")
    lines.append(f"  Covariates: {', '.join(config.covariate_names)}")
    lines.append(f"  Prediction list: {config.prediction_list_path}")
    lines.append(f"  Device: {config.device.value.upper()}")
    lines.append(f"  Chunk size: {config.chunk_size}")
    lines.append("")

    # Warmup
    lines.append(f"Warmup: {warmup_duration:.2f}s")
    lines.append("")

    # Results table
    lines.append("Results:")
    lines.append("┌─────────────────────────┬────────────┬──────────┬──────────┬──────────┬───────────┐")
    lines.append("│ Mode                    │ Total (s)  │ I/O (s)  │ Comp (s) │ Var/sec  │ Size (MB) │")
    lines.append("├─────────────────────────┼────────────┼──────────┼──────────┼──────────┼───────────┤")

    results = [tsv_result, arrow_no_finalize_result, arrow_finalize_result]
    for result in results:
        if result.error is None:
            lines.append(
                f"│ {result.mode_name:<23} │ {result.total_wall_time_seconds:>10.2f} │ "
                f"{result.output_writing_time_seconds:>8.2f} │ {result.compute_time_seconds:>8.2f} │ "
                f"{result.variants_per_second:>8,.0f} │ {result.output_size_bytes / 1024 / 1024:>9.1f} │"
            )
        else:
            lines.append(f"│ {result.mode_name:<23} │ ERROR: {result.error:<60} │")

    lines.append("└─────────────────────────┴────────────┴──────────┴──────────┴──────────┴───────────┘")
    lines.append("")

    # Speedup analysis
    if tsv_result.error is None:
        lines.append("Speedup vs TSV:")

        if arrow_no_finalize_result.error is None:
            total_speedup = tsv_result.total_wall_time_seconds / arrow_no_finalize_result.total_wall_time_seconds
            io_speedup = tsv_result.output_writing_time_seconds / arrow_no_finalize_result.output_writing_time_seconds
            size_ratio = tsv_result.output_size_bytes / arrow_no_finalize_result.output_size_bytes
            lines.append(f"  Arrow (no finalize):   {total_speedup:.2f}x faster (total), {io_speedup:.2f}x faster (I/O only)")
            lines.append(f"                         {size_ratio:.1f}x smaller on disk")

        if arrow_finalize_result.error is None:
            total_speedup = tsv_result.total_wall_time_seconds / arrow_finalize_result.total_wall_time_seconds
            io_speedup = tsv_result.output_writing_time_seconds / arrow_finalize_result.output_writing_time_seconds
            size_ratio = tsv_result.output_size_bytes / arrow_finalize_result.output_size_bytes
            lines.append(f"  Arrow (with finalize): {total_speedup:.2f}x faster (total), {io_speedup:.2f}x faster (I/O only)")
            lines.append(f"                         {size_ratio:.1f}x smaller on disk")

        lines.append("")

    # Validation
    lines.append("Validation:")
    if validation_no_finalize.passed:
        lines.append(f"  ✓ Arrow (no finalize): {validation_no_finalize.message}")
    else:
        lines.append(f"  ✗ Arrow (no finalize): {validation_no_finalize.message}")

    if validation_finalize.passed:
        lines.append(f"  ✓ Arrow (finalize):    {validation_finalize.message}")
    else:
        lines.append(f"  ✗ Arrow (finalize):    {validation_finalize.message}")

    lines.append("")

    # Recommendation
    lines.append("Recommendation:")
    if arrow_no_finalize_result.error is None and tsv_result.error is None:
        total_speedup = tsv_result.total_wall_time_seconds / arrow_no_finalize_result.total_wall_time_seconds
        size_ratio = tsv_result.output_size_bytes / arrow_no_finalize_result.output_size_bytes

        if total_speedup > 1.05:  # At least 5% faster
            lines.append(
                f"  Use Arrow chunks mode for {(total_speedup - 1) * 100:.0f}% faster total time "
                f"and {size_ratio:.1f}x smaller disk footprint."
            )
            lines.append(
                "  Enable finalization only if downstream tools require single-file Parquet format."
            )
        else:
            lines.append("  Both modes show similar performance. Choose based on workflow requirements:")
            lines.append("    - TSV: simpler, single-file output, human-readable")
            lines.append("    - Arrow: resumable, compressed, better for large datasets")
    else:
        lines.append("  Unable to provide recommendation due to benchmark errors.")

    lines.append("")
    lines.append(f"Full results saved to: {run_directory}")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_json_report(
    config: BenchmarkConfig,
    warmup_duration: float,
    tsv_result: BenchmarkResult,
    arrow_no_finalize_result: BenchmarkResult,
    arrow_finalize_result: BenchmarkResult,
    validation_no_finalize: ValidationResult,
    validation_finalize: ValidationResult,
    output_path: Path,
) -> None:
    """Save machine-readable JSON report.

    Args:
        config: Benchmark configuration.
        warmup_duration: Warmup duration in seconds.
        tsv_result: TSV benchmark results.
        arrow_no_finalize_result: Arrow (no finalization) results.
        arrow_finalize_result: Arrow (with finalization) results.
        validation_no_finalize: Validation for Arrow (no finalization).
        validation_finalize: Validation for Arrow (with finalization).
        output_path: Path to save JSON report.

    """
    report_data = {
        "benchmark_timestamp": datetime.now(UTC).isoformat(),
        "configuration": {
            "bgen_path": str(config.bgen_path),
            "sample_path": str(config.sample_path),
            "phenotype_path": str(config.phenotype_path),
            "phenotype_name": config.phenotype_name,
            "covariate_path": str(config.covariate_path),
            "covariate_names": list(config.covariate_names),
            "prediction_list_path": str(config.prediction_list_path),
            "chunk_size": config.chunk_size,
            "device": config.device.value,
        },
        "warmup_duration_seconds": warmup_duration,
        "results": {
            "tsv": asdict(tsv_result),
            "arrow_no_finalize": asdict(arrow_no_finalize_result),
            "arrow_with_finalize": asdict(arrow_finalize_result),
        },
        "validation": {
            "arrow_no_finalize": asdict(validation_no_finalize),
            "arrow_with_finalize": asdict(validation_finalize),
        },
    }

    # Convert Path objects to strings in results
    for mode_key in ["tsv", "arrow_no_finalize", "arrow_with_finalize"]:
        result_dict = report_data["results"][mode_key]
        for path_key in ["output_path", "output_run_directory", "final_parquet_path"]:
            if result_dict.get(path_key) is not None:
                result_dict[path_key] = str(result_dict[path_key])

    with output_path.open("w") as file_handle:
        json.dump(report_data, file_handle, indent=2)


def main() -> None:
    """Execute complete benchmark suite.

    Runs warmup, then benchmarks three output modes:
    1. TSV
    2. Arrow chunks (no finalization)
    3. Arrow chunks (with finalization)

    Validates outputs and generates comprehensive report.
    """
    # Create timestamped run directory
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_directory = DEFAULT_OUTPUT_BASE_DIRECTORY / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Output Mode Benchmark for REGENIE Step 2 Linear Regression")
    print("=" * 80)
    print(f"Run directory: {run_directory}")
    print("")

    # Build configuration
    config = BenchmarkConfig(
        bgen_path=DEFAULT_BGEN_PATH,
        sample_path=DEFAULT_SAMPLE_PATH,
        phenotype_path=DEFAULT_PHENOTYPE_PATH,
        phenotype_name=PHENOTYPE_NAME,
        covariate_path=DEFAULT_COVARIATE_PATH,
        covariate_names=COVARIATE_NAMES,
        prediction_list_path=DEFAULT_PREDICTION_LIST_PATH,
        chunk_size=DEFAULT_CHUNK_SIZE,
        device=types.Device.GPU,
        output_base_directory=run_directory,
    )

    # Configure JAX for GPU
    jax_setup.configure_jax_device(config.device)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print("")

    # Run warmup
    warmup_duration = run_warmup(config)

    # Run benchmarks
    tsv_result = run_tsv_benchmark(config, run_directory / "tsv")
    arrow_no_finalize_result = run_arrow_benchmark(config, run_directory / "arrow_no_finalize", finalize=False)
    arrow_finalize_result = run_arrow_benchmark(config, run_directory / "arrow_with_finalize", finalize=True)

    # Validate outputs
    validation_no_finalize = validate_outputs(tsv_result, arrow_no_finalize_result)
    validation_finalize = validate_outputs(tsv_result, arrow_finalize_result)

    # Generate reports
    print("\n")
    report_text = generate_report(
        config=config,
        warmup_duration=warmup_duration,
        tsv_result=tsv_result,
        arrow_no_finalize_result=arrow_no_finalize_result,
        arrow_finalize_result=arrow_finalize_result,
        validation_no_finalize=validation_no_finalize,
        validation_finalize=validation_finalize,
        run_directory=run_directory,
    )
    print(report_text)

    # Save reports
    report_file = run_directory / "benchmark_report.txt"
    report_file.write_text(report_text)

    json_file = run_directory / "benchmark_results.json"
    save_json_report(
        config=config,
        warmup_duration=warmup_duration,
        tsv_result=tsv_result,
        arrow_no_finalize_result=arrow_no_finalize_result,
        arrow_finalize_result=arrow_finalize_result,
        validation_no_finalize=validation_no_finalize,
        validation_finalize=validation_finalize,
        output_path=json_file,
    )

    print(f"\nReports saved:")
    print(f"  Text: {report_file}")
    print(f"  JSON: {json_file}")


if __name__ == "__main__":
    main()
