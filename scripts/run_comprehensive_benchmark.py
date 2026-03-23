#!/usr/bin/env python3
"""Comprehensive benchmark suite for GWAS Engine performance testing.

This script:
1. Runs PLINK 2 baselines
2. Runs g benchmarks on CPU and GPU with different chunk sizes
3. Runs each configuration twice (warmup + actual timing)
4. Verifies correctness by comparing outputs with PLINK
5. Generates a summary report
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    tool: str
    mode: str  # linear or logistic
    device: str  # cpu, gpu, or plink
    chunk_size: int | None
    run_number: int  # 1 for warmup, 2 for actual
    elapsed_seconds: float
    output_file: Path | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    tool: str
    mode: str
    device: str
    chunk_size: int | None
    command: list[str]
    output_file: Path


def run_command_with_timing(command: list[str]) -> tuple[float, bool]:
    """Run a command and return elapsed time and success status."""
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start_time
        return elapsed, True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"Command failed: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return elapsed, False


def run_plink_benchmark(
    mode: str,
    data_dir: Path,
    output_dir: Path,
) -> list[BenchmarkResult]:
    """Run PLINK 2 baseline benchmark."""
    results = []

    prefix = "linear" if mode == "linear" else "logistic"
    output_file = output_dir / f"plink_{prefix}.tsv"

    if mode == "linear":
        command = [
            "plink2",
            "--bfile", str(data_dir / "1kg_chr22_full"),
            "--pheno", str(data_dir / "pheno_cont.txt"),
            "--pheno-name", "phenotype_continuous",
            "--covar", str(data_dir / "covariates.txt"),
            "--glm", "linear",
            "--out", str(output_dir / f"plink_{prefix}"),
            "--threads", "16",
        ]
    else:
        command = [
            "plink2",
            "--bfile", str(data_dir / "1kg_chr22_full"),
            "--pheno", str(data_dir / "pheno_bin.txt"),
            "--pheno-name", "phenotype_binary",
            "--covar", str(data_dir / "covariates.txt"),
            "--glm", "logistic", "firth",
            "--out", str(output_dir / f"plink_{prefix}"),
            "--threads", "16",
        ]

    # PLINK only needs one run (no warmup needed)
    print(f"Running PLINK {mode}...")
    elapsed, success = run_command_with_timing(command)

    if success:
        results.append(BenchmarkResult(
            tool="PLINK 2",
            mode=mode,
            device="plink",
            chunk_size=None,
            run_number=1,
            elapsed_seconds=elapsed,
            output_file=output_file,
        ))
        print(f"  Completed in {elapsed:.2f}s")
    else:
        print("  FAILED")

    return results


def run_g_benchmark(
    mode: str,
    device: str,
    chunk_size: int,
    data_dir: Path,
    output_dir: Path,
    run_number: int,
) -> BenchmarkResult | None:
    """Run g benchmark with specified configuration."""
    prefix = f"g_{mode}_{device}_chunk{chunk_size}_run{run_number}"
    output_file = output_dir / f"{prefix}.tsv"

    command = [
        "uv", "run", "g",
        "--bfile", str(data_dir / "1kg_chr22_full"),
        "--pheno", str(data_dir / f"pheno_{'cont' if mode == 'linear' else 'bin'}.txt"),
        "--pheno-name", f"phenotype_{'continuous' if mode == 'linear' else 'binary'}",
        "--covar", str(data_dir / "covariates.txt"),
        "--covar-names", "age,sex",
        "--glm", mode,
        "--out", str(output_dir / prefix),
        "--device", device,
        "--chunk-size", str(chunk_size),
    ]

    print(f"Running g {mode} on {device} (chunk={chunk_size}, run={run_number})...")
    elapsed, success = run_command_with_timing(command)

    if success:
        print(f"  Completed in {elapsed:.2f}s")
        return BenchmarkResult(
            tool="g",
            mode=mode,
            device=device,
            chunk_size=chunk_size,
            run_number=run_number,
            elapsed_seconds=elapsed,
            output_file=output_file,
        )
    print("  FAILED")
    return None


def verify_correctness(
    plink_file: Path,
    g_file: Path,
    mode: str,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """Verify g output matches PLINK within tolerance."""
    import numpy as np
    import polars as pl

    try:
        plink_df = pl.read_csv(plink_file, separator="\t")
        g_df = pl.read_csv(g_file, separator="\t")

        # Align by variant position
        plink_sorted = plink_df.sort("POS")
        g_sorted = g_df.sort("position")

        # Compare key columns
        comparisons = {}

        if mode == "linear":
            beta_col_plink = "BETA"
            beta_col_g = "beta"
            p_col_plink = "P"
            p_col_g = "p_value"
        else:
            beta_col_plink = "BETA"
            beta_col_g = "beta"
            p_col_plink = "P"
            p_col_g = "p_value"

        # Calculate max absolute differences
        beta_diff = np.max(np.abs(
            plink_sorted[beta_col_plink].to_numpy() - g_sorted[beta_col_g].to_numpy()
        ))
        p_diff = np.max(np.abs(
            plink_sorted[p_col_plink].to_numpy() - g_sorted[p_col_g].to_numpy()
        ))

        comparisons["max_beta_diff"] = float(beta_diff)
        comparisons["max_p_diff"] = float(p_diff)
        comparisons["passed"] = beta_diff < tolerance and p_diff < tolerance

        return comparisons

    except Exception as e:
        print(f"Verification error: {e}")
        return {"error": str(e), "passed": False}


def main():
    """Run complete benchmark suite."""
    data_dir = Path("data")
    output_dir = Path("data/benchmark_results")
    output_dir.mkdir(exist_ok=True)

    all_results: list[BenchmarkResult] = []

    print("=" * 80)
    print("GWAS ENGINE PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    print()

    # 1. Run PLINK baselines
    print("STEP 1: Running PLINK 2 baselines...")
    print("-" * 80)
    all_results.extend(run_plink_benchmark("linear", data_dir, output_dir))
    all_results.extend(run_plink_benchmark("logistic", data_dir, output_dir))
    print()

    # 2. Run g benchmarks (CPU and GPU, both chunk sizes, 2 runs each)
    print("STEP 2: Running g benchmarks...")
    print("-" * 80)

    chunk_sizes = [512, 2048]
    devices = ["cpu", "gpu"]
    modes = ["linear", "logistic"]

    for mode in modes:
        for device in devices:
            for chunk_size in chunk_sizes:
                print(f"\nConfiguration: {mode} | {device} | chunk={chunk_size}")
                print("-" * 40)

                # Run twice (run 1 = warmup, run 2 = actual timing)
                for run_number in [1, 2]:
                    result = run_g_benchmark(
                        mode=mode,
                        device=device,
                        chunk_size=chunk_size,
                        data_dir=data_dir,
                        output_dir=output_dir,
                        run_number=run_number,
                    )
                    if result:
                        all_results.append(result)

    print()
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Print formatted results table
    print(f"{'Tool':<15} {'Mode':<10} {'Device':<8} {'Chunk':<8} {'Run':<6} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 80)

    # Get PLINK baselines for speedup calculation
    plink_linear_time = next(
        (r.elapsed_seconds for r in all_results if r.tool == "PLINK 2" and r.mode == "linear"),
        None
    )
    plink_logistic_time = next(
        (r.elapsed_seconds for r in all_results if r.tool == "PLINK 2" and r.mode == "logistic"),
        None
    )

    for result in all_results:
        if result.tool == "PLINK 2":
            speedup = "1.0x (baseline)"
        else:
            baseline = plink_linear_time if result.mode == "linear" else plink_logistic_time
            if baseline:
                slowdown = result.elapsed_seconds / baseline
                speedup = f"{slowdown:.1f}x"
            else:
                speedup = "N/A"

        chunk_str = str(result.chunk_size) if result.chunk_size else "N/A"
        run_str = "warmup" if result.run_number == 1 else "actual"

        print(f"{result.tool:<15} {result.mode:<10} {result.device:<8} {chunk_str:<8} {run_str:<6} {result.elapsed_seconds:<12.2f} {speedup:<10}")

    print()

    # 3. Verify correctness
    print("=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    print()

    if plink_linear_time:
        plink_linear_file = output_dir / "plink_linear.tsv"
        # Check the second run (actual timing) for each configuration
        for result in all_results:
            if result.tool == "g" and result.run_number == 2 and result.output_file:
                print(f"Verifying {result.mode} {result.device} chunk={result.chunk_size}...")
                comparison = verify_correctness(
                    plink_linear_file if result.mode == "linear" else output_dir / "plink_logistic.tsv",
                    result.output_file,
                    result.mode,
                )
                if "error" in comparison:
                    print(f"  ERROR: {comparison['error']}")
                else:
                    status = "✓ PASS" if comparison["passed"] else "✗ FAIL"
                    print(f"  {status} - Beta diff: {comparison['max_beta_diff']:.2e}, P diff: {comparison['max_p_diff']:.2e}")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
