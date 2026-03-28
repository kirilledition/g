#!/usr/bin/env python3
"""Run full Phase 1 evaluation against PLINK baselines."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from benchmark import BaselinePaths


@dataclass(frozen=True)
class RuntimeComparison:
    """Runtime comparison between PLINK and the Phase 1 engine."""

    plink_seconds: float
    phase1_seconds: float
    speed_ratio_plink_over_phase1: float


@dataclass(frozen=True)
class LinearParitySummary:
    """Full linear parity summary."""

    variant_count: int
    max_abs_beta_difference: float
    max_abs_standard_error_difference: float
    max_abs_t_statistic_difference: float
    max_abs_p_value_difference: float
    allele_reversal_count: int
    max_abs_abs_beta_difference: float
    max_abs_abs_t_statistic_difference: float
    p_value_difference_over_1e_minus_5_count: int


@dataclass(frozen=True)
class LogisticParitySummary:
    """Full logistic parity summary for hybrid logistic regression."""

    compared_variant_count: int
    non_firth_variant_count: int
    firth_variant_count: int
    method_mismatch_count: int
    error_code_mismatch_count: int
    max_abs_beta_difference: float
    max_abs_standard_error_difference: float
    max_abs_z_statistic_difference: float
    max_abs_p_value_difference: float
    firth_max_abs_beta_difference: float
    firth_max_abs_standard_error_difference: float
    firth_max_abs_z_statistic_difference: float
    firth_max_abs_p_value_difference: float
    allele_reversal_count: int
    max_abs_abs_beta_difference: float
    max_abs_abs_z_statistic_difference: float
    standard_error_difference_over_1e_minus_4_count: int
    p_value_difference_over_1e_minus_4_count: int


@dataclass(frozen=True)
class Phase1EvaluationReport:
    """Combined Phase 1 parity and runtime report."""

    linear_runtime: RuntimeComparison
    logistic_runtime: RuntimeComparison
    linear_parity: LinearParitySummary
    logistic_parity: LogisticParitySummary


def time_command(command_arguments: list[str]) -> float:
    """Run a command and return elapsed seconds.

    Args:
        command_arguments: Command and arguments.

    Returns:
        Runtime in seconds.

    Raises:
        RuntimeError: The command fails.

    """
    start_time = time.perf_counter()
    completed_process = subprocess.run(command_arguments, check=False, capture_output=True, text=True)
    duration_seconds = time.perf_counter() - start_time
    if completed_process.returncode != 0:
        message = completed_process.stderr.strip() or f"Command failed: {' '.join(command_arguments)}"
        raise RuntimeError(message)
    return duration_seconds


def run_phase1_linear(baseline_paths: BaselinePaths) -> tuple[pl.DataFrame, float]:
    """Run the full Phase 1 linear engine and return results with runtime."""
    from g.engine import run_linear_association

    start_time = time.perf_counter()
    result_frame = run_linear_association(
        bed_prefix=baseline_paths.bed_prefix,
        phenotype_path=baseline_paths.continuous_phenotype_path,
        phenotype_name="phenotype_continuous",
        covariate_path=baseline_paths.covariate_path,
        covariate_names=("age", "sex"),
        chunk_size=2048,
        variant_limit=None,
    )
    duration_seconds = time.perf_counter() - start_time
    return result_frame, duration_seconds


def run_phase1_logistic(baseline_paths: BaselinePaths) -> tuple[pl.DataFrame, float]:
    """Run the full Phase 1 logistic engine and return results with runtime."""
    from g.engine import run_logistic_association

    start_time = time.perf_counter()
    result_frame = run_logistic_association(
        bed_prefix=baseline_paths.bed_prefix,
        phenotype_path=baseline_paths.binary_phenotype_path,
        phenotype_name="phenotype_binary",
        covariate_path=baseline_paths.covariate_path,
        covariate_names=("age", "sex"),
        chunk_size=512,
        variant_limit=None,
        max_iterations=50,
        tolerance=1.0e-8,
    )
    duration_seconds = time.perf_counter() - start_time
    return result_frame, duration_seconds


def expression_value(data_frame: pl.DataFrame, expression: pl.Expr) -> float:
    """Evaluate a Polars expression and return its scalar value."""
    return float(data_frame.select(expression.alias("value")).item())


def expression_value_or_zero(data_frame: pl.DataFrame, expression: pl.Expr) -> float:
    """Evaluate a Polars expression, returning zero for empty frames."""
    if data_frame.height == 0:
        return 0.0
    return expression_value(data_frame, expression)


def align_to_plink_a1(
    joined_frame: pl.DataFrame,
    baseline_allele_column: str,
    statistic_column: str,
) -> pl.DataFrame:
    """Align engine effect directions to PLINK's reported A1 allele.

    Args:
        joined_frame: Joined engine and PLINK result frame.
        baseline_allele_column: Baseline column containing PLINK's A1 allele.
        statistic_column: Engine statistic column to direction-align.

    Returns:
        Frame with direction-adjusted effect/statistic columns.

    Raises:
        ValueError: If allele labels do not line up between both outputs.

    """
    aligned_frame = joined_frame.with_columns(
        pl.when(pl.col(baseline_allele_column) == pl.col("allele_one"))
        .then(pl.lit(1.0))
        .when(pl.col(baseline_allele_column) == pl.col("allele_two"))
        .then(pl.lit(-1.0))
        .otherwise(None)
        .alias("allele_alignment_sign"),
    )
    if aligned_frame.get_column("allele_alignment_sign").null_count() > 0:
        message = "At least one variant could not be aligned to PLINK's reported A1 allele."
        raise ValueError(message)
    return aligned_frame.with_columns(
        (pl.col("beta") * pl.col("allele_alignment_sign")).alias("aligned_beta"),
        (pl.col(statistic_column) * pl.col("allele_alignment_sign")).alias("aligned_statistic"),
    )


def summarize_linear_parity(baseline_paths: BaselinePaths, phase1_frame: pl.DataFrame) -> LinearParitySummary:
    """Summarize full linear parity against PLINK output."""
    baseline_frame = (
        pl.read_csv(baseline_paths.baseline_directory / "plink_cont.phenotype_continuous.glm.linear", separator="\t")
        .filter(pl.col("TEST") == "ADD")
        .with_row_index("row_index")
        .select("row_index", "ID", "A1", "BETA", "SE", "T_STAT", "P")
        .rename({"ID": "variant_identifier", "SE": "baseline_standard_error", "P": "baseline_p_value"})
    )
    phase1_indexed_frame = phase1_frame.with_row_index("row_index")
    joined_frame = align_to_plink_a1(
        phase1_indexed_frame.join(baseline_frame, on="row_index", how="inner"),
        baseline_allele_column="A1",
        statistic_column="t_statistic",
    )
    return LinearParitySummary(
        variant_count=joined_frame.height,
        max_abs_beta_difference=expression_value(joined_frame, (pl.col("aligned_beta") - pl.col("BETA")).abs().max()),
        max_abs_standard_error_difference=expression_value(
            joined_frame,
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs().max(),
        ),
        max_abs_t_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic") - pl.col("T_STAT")).abs().max(),
        ),
        max_abs_p_value_difference=expression_value(
            joined_frame,
            (pl.col("p_value") - pl.col("baseline_p_value")).abs().max(),
        ),
        allele_reversal_count=joined_frame.filter(pl.col("allele_alignment_sign") < 0).height,
        max_abs_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta").abs() - pl.col("BETA").abs()).abs().max(),
        ),
        max_abs_abs_t_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic").abs() - pl.col("T_STAT").abs()).abs().max(),
        ),
        p_value_difference_over_1e_minus_5_count=joined_frame.filter(
            (pl.col("p_value") - pl.col("baseline_p_value")).abs() > 1.0e-5,
        ).height,
    )


def summarize_logistic_parity(baseline_paths: BaselinePaths, phase1_frame: pl.DataFrame) -> LogisticParitySummary:
    """Summarize full logistic parity against PLINK hybrid output."""
    baseline_frame = (
        pl.read_csv(
            baseline_paths.baseline_directory / "plink_bin.phenotype_binary.glm.logistic.hybrid",
            separator="\t",
        )
        .filter(pl.col("TEST") == "ADD")
        .with_row_index("row_index")
        .select("row_index", "ID", "A1", "OR", "LOG(OR)_SE", "Z_STAT", "P", "FIRTH?", "ERRCODE")
        .with_columns(pl.col("OR").log().alias("baseline_beta"))
        .rename({"ID": "variant_identifier", "LOG(OR)_SE": "baseline_standard_error", "P": "baseline_p_value"})
    )
    phase1_indexed_frame = phase1_frame.with_row_index("row_index")
    joined_frame = align_to_plink_a1(
        phase1_indexed_frame.join(baseline_frame, on="row_index", how="inner"),
        baseline_allele_column="A1",
        statistic_column="z_statistic",
    )
    firth_joined_frame = joined_frame.filter(pl.col("FIRTH?") == "Y")
    return LogisticParitySummary(
        compared_variant_count=joined_frame.height,
        non_firth_variant_count=joined_frame.filter(pl.col("FIRTH?") == "N").height,
        firth_variant_count=firth_joined_frame.height,
        method_mismatch_count=joined_frame.filter(pl.col("firth_flag") != pl.col("FIRTH?")).height,
        error_code_mismatch_count=joined_frame.filter(pl.col("error_code") != pl.col("ERRCODE")).height,
        max_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta") - pl.col("baseline_beta")).abs().max(),
        ),
        max_abs_standard_error_difference=expression_value(
            joined_frame,
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs().max(),
        ),
        max_abs_z_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic") - pl.col("Z_STAT")).abs().max(),
        ),
        max_abs_p_value_difference=expression_value(
            joined_frame,
            (pl.col("p_value") - pl.col("baseline_p_value")).abs().max(),
        ),
        firth_max_abs_beta_difference=expression_value_or_zero(
            firth_joined_frame,
            (pl.col("aligned_beta") - pl.col("baseline_beta")).abs().max(),
        ),
        firth_max_abs_standard_error_difference=expression_value_or_zero(
            firth_joined_frame,
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs().max(),
        ),
        firth_max_abs_z_statistic_difference=expression_value_or_zero(
            firth_joined_frame,
            (pl.col("aligned_statistic") - pl.col("Z_STAT")).abs().max(),
        ),
        firth_max_abs_p_value_difference=expression_value_or_zero(
            firth_joined_frame,
            (pl.col("p_value") - pl.col("baseline_p_value")).abs().max(),
        ),
        allele_reversal_count=joined_frame.filter(pl.col("allele_alignment_sign") < 0).height,
        max_abs_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta").abs() - pl.col("baseline_beta").abs()).abs().max(),
        ),
        max_abs_abs_z_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic").abs() - pl.col("Z_STAT").abs()).abs().max(),
        ),
        standard_error_difference_over_1e_minus_4_count=joined_frame.filter(
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs() > 1.0e-4,
        ).height,
        p_value_difference_over_1e_minus_4_count=joined_frame.filter(
            (pl.col("p_value") - pl.col("baseline_p_value")).abs() > 1.0e-4,
        ).height,
    )


def build_runtime_comparison(plink_seconds: float, phase1_seconds: float) -> RuntimeComparison:
    """Build a runtime comparison structure."""
    return RuntimeComparison(
        plink_seconds=plink_seconds,
        phase1_seconds=phase1_seconds,
        speed_ratio_plink_over_phase1=phase1_seconds / plink_seconds,
    )


def main() -> None:
    """Run the full Phase 1 evaluation and save a report."""
    from benchmark import (
        build_baseline_paths,
        build_plink_binary_command,
        build_plink_continuous_command,
        resolve_required_executable,
    )

    baseline_paths = build_baseline_paths()
    plink_executable = resolve_required_executable("PLINK2_BIN", "plink2")

    plink_linear_seconds = time_command(build_plink_continuous_command(plink_executable, baseline_paths))
    plink_logistic_seconds = time_command(build_plink_binary_command(plink_executable, baseline_paths))
    phase1_linear_frame, phase1_linear_seconds = run_phase1_linear(baseline_paths)
    phase1_logistic_frame, phase1_logistic_seconds = run_phase1_logistic(baseline_paths)

    report = Phase1EvaluationReport(
        linear_runtime=build_runtime_comparison(plink_linear_seconds, phase1_linear_seconds),
        logistic_runtime=build_runtime_comparison(plink_logistic_seconds, phase1_logistic_seconds),
        linear_parity=summarize_linear_parity(baseline_paths, phase1_linear_frame),
        logistic_parity=summarize_logistic_parity(baseline_paths, phase1_logistic_frame),
    )
    report_path = baseline_paths.data_directory / "phase1_evaluation_report.json"
    report_path.write_text(f"{json.dumps(asdict(report), indent=2)}\n")
    print(report_path)


if __name__ == "__main__":
    main()
