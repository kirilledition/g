#!/usr/bin/env python3
"""Capture detailed cProfile + JAX profiler reports for full-chromosome runs.

This script generates comprehensive plain-text profiling reports combining:
- cProfiler detailed function call statistics
- JAX profiler traces for GPU execution
- Memory profiling
- Per-phase timing breakdowns
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.profiler
import numpy as np

from g.engine import iter_linear_output_frames, iter_logistic_output_frames

if TYPE_CHECKING:
    from collections.abc import Iterator
    import polars as pl


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for detailed profiling."""
    parser = argparse.ArgumentParser(
        description="Profile full chr22 GWAS run with cProfile + JAX tracing."
    )
    parser.add_argument(
        "--bfile", required=True, type=Path, help="PLINK dataset prefix."
    )
    parser.add_argument(
        "--pheno", required=True, type=Path, help="Phenotype table path."
    )
    parser.add_argument(
        "--pheno-name", required=True, help="Phenotype column name to analyze."
    )
    parser.add_argument(
        "--covar", required=True, type=Path, help="Covariate table path."
    )
    parser.add_argument(
        "--covar-names",
        default="age,sex",
        help="Comma-separated covariate names.",
    )
    parser.add_argument(
        "--glm",
        required=True,
        choices=("linear", "logistic"),
        help="Association model.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Variants per chunk."
    )
    parser.add_argument(
        "--variant-limit", type=int, help="Optional variant cap."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum logistic IRLS iterations.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-8,
        help="Logistic convergence tolerance.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for all profiling outputs.",
    )
    parser.add_argument(
        "--report-name",
        default="profile_report",
        help="Base name for output files.",
    )
    parser.add_argument(
        "--enable-jax-trace",
        action="store_true",
        help="Enable JAX profiler trace capture.",
    )
    parser.add_argument(
        "--enable-memory-profile",
        action="store_true",
        help="Enable JAX device memory profiling.",
    )
    parser.add_argument(
        "--cprofile-sort",
        default="cumulative",
        choices=("cumulative", "time", "calls", "name"),
        help="Sort key for cProfile statistics.",
    )
    return parser


def parse_covariate_names(raw_covariate_names: str) -> tuple[str, ...] | None:
    """Parse a comma-separated covariate name list."""
    covariate_names = tuple(
        name.strip() for name in raw_covariate_names.split(",") if name.strip()
    )
    return covariate_names or None


def run_and_materialize_frames(
    frame_iterator: Iterator[pl.DataFrame],
) -> dict[str, int]:
    """Force a full iterator run and return statistics."""
    total_variants = 0
    chunk_count = 0
    for output_frame in frame_iterator:
        total_variants += output_frame.height
        chunk_count += 1
    return {"total_variants": total_variants, "chunk_count": chunk_count}


def format_profiling_report(
    stats_stream: io.StringIO,
    execution_stats: dict[str, float | int],
    glm_mode: str,
) -> str:
    """Format comprehensive profiling report as plain text."""
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append(" " * 20 + "GWAS ENGINE DETAILED PROFILING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Execution Summary
    report_lines.append("EXECUTION SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"  Model Type:          {glm_mode.upper()}")
    report_lines.append(f"  Total Variants:      {execution_stats.get('total_variants', 'N/A')}")
    report_lines.append(f"  Total Chunks:        {execution_stats.get('chunk_count', 'N/A')}")
    report_lines.append(f"  Wall Clock Time:     {execution_stats.get('wall_time', 'N/A'):.2f} seconds")
    report_lines.append(f"  Variants/Second:     {execution_stats.get('variants_per_second', 'N/A'):.0f}")
    report_lines.append("")
    
    # JAX Device Info
    report_lines.append("JAX DEVICE CONFIGURATION")
    report_lines.append("-" * 80)
    devices = jax.devices()
    report_lines.append(f"  Number of Devices:   {len(devices)}")
    for idx, device in enumerate(devices):
        report_lines.append(f"  Device {idx}:           {device}")
    report_lines.append(f"  Default Backend:     {jax.default_backend()}")
    report_lines.append("")
    
    # cProfile Statistics
    report_lines.append("DETAILED FUNCTION CALL STATISTICS (cProfile)")
    report_lines.append("-" * 80)
    report_lines.append(stats_stream.getvalue())
    report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def generate_cprofile_report(
    profiler: cProfile.Profile,
    output_path: Path,
    sort_key: str,
    limit: int = 200,
) -> None:
    """Generate detailed plain-text cProfile report."""
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(sort_key)
    
    # Print all function statistics (limited by limit parameter)
    stats.print_stats(limit)
    
    # Also print call statistics for top functions
    stream.write("\n")
    stream.write("=" * 80 + "\n")
    stream.write("CALLER STATISTICS (Who called the top functions)\n")
    stream.write("=" * 80 + "\n")
    stats.print_callers(50)
    
    stream.write("\n")
    stream.write("=" * 80 + "\n")
    stream.write("CALLEE STATISTICS (What the top functions called)\n")
    stream.write("=" * 80 + "\n")
    stats.print_callees(50)
    
    # Write to file
    with open(output_path, "w") as file_handle:
        file_handle.write(stream.getvalue())
    
    stream.close()


def main() -> None:
    """Capture detailed profiling for full chr22 association run."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    covariate_names = parse_covariate_names(arguments.covar_names)
    
    # Create output directories
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    jax_trace_dir = arguments.output_dir / f"{arguments.report_name}_jax_trace"
    
    print(f"Profiling {arguments.glm.upper()} regression on full chromosome 22")
    print(f"Output directory: {arguments.output_dir}")
    print(f"Chunk size: {arguments.chunk_size}")
    print("-" * 80)
    
    # Initialize cProfiler
    profiler = cProfile.Profile()
    
    # Prepare execution context
    if arguments.glm == "linear":
        frame_iterator = iter_linear_output_frames(
            bed_prefix=arguments.bfile,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
        )
    else:
        frame_iterator = iter_logistic_output_frames(
            bed_prefix=arguments.bfile,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            max_iterations=arguments.max_iterations,
            tolerance=arguments.tolerance,
        )
    
    # Execute with profiling
    start_time = time.perf_counter()
    
    profiler.enable()
    
    if arguments.enable_jax_trace:
        # Run with JAX tracing enabled
        with jax.profiler.trace(jax_trace_dir, create_perfetto_trace=True):
            execution_stats = run_and_materialize_frames(frame_iterator)
    else:
        # Run without JAX tracing
        execution_stats = run_and_materialize_frames(frame_iterator)
    
    profiler.disable()
    
    end_time = time.perf_counter()
    wall_time = end_time - start_time
    
    # Calculate derived statistics
    total_variants = execution_stats["total_variants"]
    variants_per_second = total_variants / wall_time if wall_time > 0 else 0
    
    execution_stats.update({
        "wall_time": wall_time,
        "variants_per_second": variants_per_second,
    })
    
    print(f"\nExecution complete:")
    print(f"  Variants processed: {total_variants}")
    print(f"  Chunks processed:   {execution_stats['chunk_count']}")
    print(f"  Wall clock time:    {wall_time:.2f} seconds")
    print(f"  Throughput:         {variants_per_second:.0f} variants/second")
    print("-" * 80)
    
    # Generate cProfile report (detailed)
    cprofile_path = arguments.output_dir / f"{arguments.report_name}_cprofile.txt"
    print(f"Generating cProfile report: {cprofile_path}")
    generate_cprofile_report(
        profiler,
        cprofile_path,
        arguments.cprofile_sort,
        limit=200,  # Full detailed breakdown - top 200 functions
    )
    
    # Dump raw cProfile stats for later analysis
    cprofile_raw_path = arguments.output_dir / f"{arguments.report_name}_cprofile.prof"
    profiler.dump_stats(str(cprofile_raw_path))
    print(f"Raw cProfile stats saved: {cprofile_raw_path}")
    
    # Save memory profile if enabled
    if arguments.enable_memory_profile:
        memory_profile_path = arguments.output_dir / f"{arguments.report_name}_memory.prof"
        jax.profiler.save_device_memory_profile(memory_profile_path)
        print(f"Memory profile saved: {memory_profile_path}")
    
    # Generate comprehensive summary report
    summary_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=summary_stream)
    stats.strip_dirs()
    stats.sort_stats(arguments.cprofile_sort)
    stats.print_stats(100)  # Top 100 for summary
    
    summary_report = format_profiling_report(
        summary_stream,
        execution_stats,
        arguments.glm,
    )
    
    summary_path = arguments.output_dir / f"{arguments.report_name}_summary.txt"
    with open(summary_path, "w") as file_handle:
        file_handle.write(summary_report)
    print(f"Summary report saved: {summary_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print(summary_report)
    print("=" * 80)
    
    print(f"\nAll profiling outputs saved to: {arguments.output_dir}")
    print("Files generated:")
    print(f"  - {cprofile_path.name} (detailed function statistics)")
    print(f"  - {cprofile_raw_path.name} (raw cProfile data)")
    if arguments.enable_jax_trace:
        print(f"  - {jax_trace_dir}/ (JAX trace directory with Perfetto trace)")
    if arguments.enable_memory_profile:
        print(f"  - {memory_profile_path.name} (device memory profile)")
    print(f"  - {summary_path.name} (execution summary)")


if __name__ == "__main__":
    main()
