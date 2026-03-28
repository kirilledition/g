#!/usr/bin/env python3
"""Run full Phase 1 evaluation against PLINK and Hail baselines."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

if TYPE_CHECKING:
    from benchmark import BaselinePaths


@dataclass(frozen=True)
class RuntimeComparison:
    """Runtime comparison between one baseline and the Phase 1 engine."""

    baseline_name: str
    baseline_seconds: float
    phase1_seconds: float
    speed_ratio_baseline_over_phase1: float


@dataclass(frozen=True)
class LinearParitySummary:
    """Full linear parity summary."""

    variant_count: int
    max_abs_beta_difference: float
    max_abs_standard_error_difference: float
    max_abs_t_statistic_difference: float
    max_abs_log10_p_value_difference: float
    allele_reversal_count: int
    max_abs_abs_beta_difference: float
    max_abs_abs_t_statistic_difference: float
    log10_p_value_difference_over_1e_minus_5_count: int


@dataclass(frozen=True)
class LogisticParitySummary:
    """Full logistic parity summary for PLINK hybrid logistic regression."""

    compared_variant_count: int
    non_firth_variant_count: int
    firth_variant_count: int
    method_mismatch_count: int
    error_code_mismatch_count: int
    max_abs_beta_difference: float
    max_abs_standard_error_difference: float
    max_abs_z_statistic_difference: float
    max_abs_log10_p_value_difference: float
    firth_max_abs_beta_difference: float
    firth_max_abs_standard_error_difference: float
    firth_max_abs_z_statistic_difference: float
    firth_max_abs_log10_p_value_difference: float
    allele_reversal_count: int
    max_abs_abs_beta_difference: float
    max_abs_abs_z_statistic_difference: float
    standard_error_difference_over_1e_minus_4_count: int
    log10_p_value_difference_over_1e_minus_4_count: int


@dataclass(frozen=True)
class LogisticHailHybridParitySummary:
    """Parity summary against a Hail hybrid approximation."""

    compared_variant_count: int
    non_firth_variant_count: int
    firth_variant_count: int
    wald_missing_variant_count: int
    firth_missing_variant_count: int
    max_abs_beta_difference: float
    max_abs_log10_p_value_difference: float
    non_firth_max_abs_standard_error_difference: float
    non_firth_max_abs_z_statistic_difference: float
    non_firth_max_abs_log10_p_value_difference: float
    firth_max_abs_beta_difference: float
    firth_max_abs_log10_p_value_difference: float
    allele_reversal_count: int
    max_abs_abs_beta_difference: float


@dataclass(frozen=True)
class Phase1EvaluationReport:
    """Combined Phase 1 parity and runtime report."""

    hail_cache_prepare_runtime: RuntimeComparison
    linear_plink_runtime: RuntimeComparison
    logistic_plink_runtime: RuntimeComparison
    linear_hail_runtime: RuntimeComparison
    logistic_hail_wald_runtime: RuntimeComparison
    logistic_hail_firth_runtime: RuntimeComparison
    logistic_hail_hybrid_upper_bound_runtime: RuntimeComparison
    linear_plink_parity: LinearParitySummary
    logistic_plink_parity: LogisticParitySummary
    linear_hail_parity: LinearParitySummary
    logistic_hail_hybrid_parity: LogisticHailHybridParitySummary


class Phase1RunResult(NamedTuple):
    """Result frame and runtime for one Phase 1 execution."""

    result_frame: pl.DataFrame
    duration_seconds: float


MINIMUM_POSITIVE_P_VALUE = 1.0e-300


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


def run_phase1_linear(baseline_paths: BaselinePaths) -> Phase1RunResult:
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
    return Phase1RunResult(result_frame=result_frame, duration_seconds=duration_seconds)


def run_phase1_logistic(baseline_paths: BaselinePaths) -> Phase1RunResult:
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
    return Phase1RunResult(result_frame=result_frame, duration_seconds=duration_seconds)


def expression_value(data_frame: pl.DataFrame, expression: pl.Expr) -> float:
    """Evaluate a Polars expression and return its scalar value."""
    return float(data_frame.select(expression.alias("value")).item())


def expression_value_or_zero(data_frame: pl.DataFrame, expression: pl.Expr) -> float:
    """Evaluate a Polars expression, returning zero for empty frames."""
    if data_frame.height == 0:
        return 0.0
    return expression_value(data_frame, expression)


def log10_p_value_expression(column_name: str) -> pl.Expr:
    """Build a clipped base-10 log p-value expression."""
    return pl.col(column_name).clip(lower_bound=MINIMUM_POSITIVE_P_VALUE).log10()


def absolute_log10_p_value_difference(left_column_name: str, right_column_name: str) -> pl.Expr:
    """Build an absolute log10 p-value difference expression."""
    return (log10_p_value_expression(left_column_name) - log10_p_value_expression(right_column_name)).abs()


def with_allele_pair_key(data_frame: pl.DataFrame, first_column: str, second_column: str) -> pl.DataFrame:
    """Attach an order-insensitive allele-pair key for joining result tables."""
    return data_frame.with_columns(
        pl.when(pl.col(first_column) <= pl.col(second_column))
        .then(pl.concat_str([pl.col(first_column), pl.lit("|"), pl.col(second_column)]))
        .otherwise(pl.concat_str([pl.col(second_column), pl.lit("|"), pl.col(first_column)]))
        .alias("allele_pair_key")
    )


def align_to_effect_allele(
    joined_frame: pl.DataFrame,
    baseline_effect_allele_column: str,
    statistic_column: str,
) -> pl.DataFrame:
    """Align engine effect directions to a baseline effect allele."""
    aligned_frame = joined_frame.with_columns(
        pl.when(pl.col(baseline_effect_allele_column) == pl.col("allele_one"))
        .then(pl.lit(1.0))
        .when(pl.col(baseline_effect_allele_column) == pl.col("allele_two"))
        .then(pl.lit(-1.0))
        .otherwise(None)
        .alias("allele_alignment_sign"),
    )
    if aligned_frame.get_column("allele_alignment_sign").null_count() > 0:
        message = "At least one variant could not be aligned to the baseline effect allele."
        raise ValueError(message)
    return aligned_frame.with_columns(
        (pl.col("beta") * pl.col("allele_alignment_sign")).alias("aligned_beta"),
        (pl.col(statistic_column) * pl.col("allele_alignment_sign")).alias("aligned_statistic"),
    )


def summarize_linear_plink_parity(baseline_paths: BaselinePaths, phase1_frame: pl.DataFrame) -> LinearParitySummary:
    """Summarize full linear parity against PLINK output."""
    baseline_frame = (
        pl.read_csv(baseline_paths.baseline_directory / "plink_cont.phenotype_continuous.glm.linear", separator="\t")
        .filter(pl.col("TEST") == "ADD")
        .with_row_index("row_index")
        .select("row_index", "ID", "A1", "BETA", "SE", "T_STAT", "P")
        .rename({"ID": "variant_identifier", "SE": "baseline_standard_error", "P": "baseline_p_value"})
    )
    phase1_indexed_frame = phase1_frame.with_row_index("row_index")
    joined_frame = align_to_effect_allele(
        phase1_indexed_frame.join(baseline_frame, on="row_index", how="inner"),
        baseline_effect_allele_column="A1",
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
        max_abs_log10_p_value_difference=expression_value(
            joined_frame,
            absolute_log10_p_value_difference("p_value", "baseline_p_value").max(),
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
        log10_p_value_difference_over_1e_minus_5_count=joined_frame.filter(
            absolute_log10_p_value_difference("p_value", "baseline_p_value") > 1.0e-5,
        ).height,
    )


def summarize_linear_hail_parity(baseline_paths: BaselinePaths, phase1_frame: pl.DataFrame) -> LinearParitySummary:
    """Summarize full linear parity against Hail output."""
    from benchmark import hail_output_path

    baseline_frame = with_allele_pair_key(
        pl.read_csv(hail_output_path(baseline_paths, "hail_cont"), separator="\t")
        .with_columns(
            pl.col("chromosome").cast(pl.String),
            pl.col("position").cast(pl.Int64),
            pl.col("variant_identifier").cast(pl.String),
            pl.col("allele_one").cast(pl.String),
            pl.col("allele_two").cast(pl.String),
            pl.col("beta").cast(pl.Float64, strict=False),
            pl.col("standard_error").cast(pl.Float64, strict=False),
            pl.col("t_statistic").cast(pl.Float64, strict=False),
            pl.col("p_value").cast(pl.Float64, strict=False),
        )
        .select(
            "chromosome",
            "position",
            "variant_identifier",
            "allele_one",
            "allele_two",
            "beta",
            "standard_error",
            "t_statistic",
            "p_value",
        )
        .rename(
            {
                "allele_one": "baseline_allele_one",
                "beta": "baseline_beta",
                "standard_error": "baseline_standard_error",
                "t_statistic": "baseline_t_statistic",
                "p_value": "baseline_p_value",
            }
        ),
        first_column="baseline_allele_one",
        second_column="allele_two",
    )
    phase1_frame = with_allele_pair_key(phase1_frame, first_column="allele_one", second_column="allele_two")
    joined_frame = align_to_effect_allele(
        phase1_frame.join(
            baseline_frame,
            on=["chromosome", "position", "variant_identifier", "allele_pair_key"],
            how="inner",
        ),
        baseline_effect_allele_column="baseline_allele_one",
        statistic_column="t_statistic",
    )
    return LinearParitySummary(
        variant_count=joined_frame.height,
        max_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta") - pl.col("baseline_beta")).abs().max(),
        ),
        max_abs_standard_error_difference=expression_value(
            joined_frame,
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs().max(),
        ),
        max_abs_t_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic") - pl.col("baseline_t_statistic")).abs().max(),
        ),
        max_abs_log10_p_value_difference=expression_value(
            joined_frame,
            absolute_log10_p_value_difference("p_value", "baseline_p_value").max(),
        ),
        allele_reversal_count=joined_frame.filter(pl.col("allele_alignment_sign") < 0).height,
        max_abs_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta").abs() - pl.col("baseline_beta").abs()).abs().max(),
        ),
        max_abs_abs_t_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic").abs() - pl.col("baseline_t_statistic").abs()).abs().max(),
        ),
        log10_p_value_difference_over_1e_minus_5_count=joined_frame.filter(
            absolute_log10_p_value_difference("p_value", "baseline_p_value") > 1.0e-5,
        ).height,
    )


def summarize_logistic_plink_parity(baseline_paths: BaselinePaths, phase1_frame: pl.DataFrame) -> LogisticParitySummary:
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
    joined_frame = align_to_effect_allele(
        phase1_indexed_frame.join(baseline_frame, on="row_index", how="inner"),
        baseline_effect_allele_column="A1",
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
        max_abs_log10_p_value_difference=expression_value(
            joined_frame,
            absolute_log10_p_value_difference("p_value", "baseline_p_value").max(),
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
        firth_max_abs_log10_p_value_difference=expression_value_or_zero(
            firth_joined_frame,
            absolute_log10_p_value_difference("p_value", "baseline_p_value").max(),
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
        log10_p_value_difference_over_1e_minus_4_count=joined_frame.filter(
            absolute_log10_p_value_difference("p_value", "baseline_p_value") > 1.0e-4,
        ).height,
    )


def summarize_logistic_hail_hybrid_parity(
    baseline_paths: BaselinePaths,
    phase1_frame: pl.DataFrame,
) -> LogisticHailHybridParitySummary:
    """Summarize logistic parity against a Hail Wald/Firth hybrid approximation."""
    from benchmark import hail_output_path

    hail_wald_frame = with_allele_pair_key(
        pl.read_csv(hail_output_path(baseline_paths, "hail_bin_wald"), separator="\t")
        .with_columns(
            pl.col("chromosome").cast(pl.String),
            pl.col("position").cast(pl.Int64),
            pl.col("variant_identifier").cast(pl.String),
            pl.col("allele_one").cast(pl.String),
            pl.col("allele_two").cast(pl.String),
            pl.col("beta").cast(pl.Float64, strict=False),
            pl.col("standard_error").cast(pl.Float64, strict=False),
            pl.col("z_statistic").cast(pl.Float64, strict=False),
            pl.col("p_value").cast(pl.Float64, strict=False),
        )
        .select(
            "chromosome",
            "position",
            "variant_identifier",
            "allele_one",
            "allele_two",
            "beta",
            "standard_error",
            "z_statistic",
            "p_value",
        )
        .rename(
            {
                "allele_one": "hail_wald_allele_one",
                "allele_two": "hail_wald_allele_two",
                "beta": "hail_wald_beta",
                "standard_error": "hail_wald_standard_error",
                "z_statistic": "hail_wald_z_statistic",
                "p_value": "hail_wald_p_value",
            }
        ),
        first_column="hail_wald_allele_one",
        second_column="hail_wald_allele_two",
    )
    hail_firth_frame = with_allele_pair_key(
        pl.read_csv(hail_output_path(baseline_paths, "hail_bin_firth"), separator="\t")
        .with_columns(
            pl.col("chromosome").cast(pl.String),
            pl.col("position").cast(pl.Int64),
            pl.col("variant_identifier").cast(pl.String),
            pl.col("allele_one").cast(pl.String),
            pl.col("allele_two").cast(pl.String),
            pl.col("beta").cast(pl.Float64, strict=False),
            pl.col("p_value").cast(pl.Float64, strict=False),
        )
        .select(
            "chromosome",
            "position",
            "variant_identifier",
            "allele_one",
            "allele_two",
            "beta",
            "p_value",
        )
        .rename(
            {
                "allele_one": "hail_firth_allele_one",
                "allele_two": "hail_firth_allele_two",
                "beta": "hail_firth_beta",
                "p_value": "hail_firth_p_value",
            }
        ),
        first_column="hail_firth_allele_one",
        second_column="hail_firth_allele_two",
    )
    phase1_frame = with_allele_pair_key(phase1_frame, first_column="allele_one", second_column="allele_two")
    joined_frame = (
        phase1_frame.join(
            hail_wald_frame,
            on=["chromosome", "position", "variant_identifier", "allele_pair_key"],
            how="left",
        )
        .join(
            hail_firth_frame,
            on=["chromosome", "position", "variant_identifier", "allele_pair_key"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("firth_flag") == "Y")
            .then(pl.col("hail_firth_allele_one"))
            .otherwise(pl.col("hail_wald_allele_one"))
            .alias("selected_hail_allele_one"),
            pl.when(pl.col("firth_flag") == "Y")
            .then(pl.col("hail_firth_beta"))
            .otherwise(pl.col("hail_wald_beta"))
            .alias("selected_hail_beta"),
            pl.when(pl.col("firth_flag") == "Y")
            .then(pl.col("hail_firth_p_value"))
            .otherwise(pl.col("hail_wald_p_value"))
            .alias("selected_hail_p_value"),
        )
        .with_columns(
            pl.when(pl.col("selected_hail_allele_one") == pl.col("allele_one"))
            .then(pl.lit(1.0))
            .when(pl.col("selected_hail_allele_one") == pl.col("allele_two"))
            .then(pl.lit(-1.0))
            .otherwise(None)
            .alias("allele_alignment_sign"),
        )
        .with_columns((pl.col("beta") * pl.col("allele_alignment_sign")).alias("aligned_beta"))
    )

    compared_frame = joined_frame.filter(
        pl.col("selected_hail_beta").is_not_null()
        & pl.col("selected_hail_p_value").is_not_null()
        & pl.col("allele_alignment_sign").is_not_null(),
    )
    non_firth_frame = compared_frame.filter(pl.col("firth_flag") == "N")
    firth_frame = compared_frame.filter(pl.col("firth_flag") == "Y")
    return LogisticHailHybridParitySummary(
        compared_variant_count=compared_frame.height,
        non_firth_variant_count=non_firth_frame.height,
        firth_variant_count=firth_frame.height,
        wald_missing_variant_count=joined_frame.filter(
            (pl.col("firth_flag") == "N") & pl.col("hail_wald_beta").is_null(),
        ).height,
        firth_missing_variant_count=joined_frame.filter(
            (pl.col("firth_flag") == "Y") & pl.col("hail_firth_beta").is_null(),
        ).height,
        max_abs_beta_difference=expression_value_or_zero(
            compared_frame,
            (pl.col("aligned_beta") - pl.col("selected_hail_beta")).abs().max(),
        ),
        max_abs_log10_p_value_difference=expression_value_or_zero(
            compared_frame,
            absolute_log10_p_value_difference("p_value", "selected_hail_p_value").max(),
        ),
        non_firth_max_abs_standard_error_difference=expression_value_or_zero(
            non_firth_frame,
            (pl.col("standard_error") - pl.col("hail_wald_standard_error")).abs().max(),
        ),
        non_firth_max_abs_z_statistic_difference=expression_value_or_zero(
            non_firth_frame,
            (pl.col("z_statistic") - pl.col("hail_wald_z_statistic")).abs().max(),
        ),
        non_firth_max_abs_log10_p_value_difference=expression_value_or_zero(
            non_firth_frame,
            absolute_log10_p_value_difference("p_value", "hail_wald_p_value").max(),
        ),
        firth_max_abs_beta_difference=expression_value_or_zero(
            firth_frame,
            (pl.col("aligned_beta") - pl.col("hail_firth_beta")).abs().max(),
        ),
        firth_max_abs_log10_p_value_difference=expression_value_or_zero(
            firth_frame,
            absolute_log10_p_value_difference("p_value", "hail_firth_p_value").max(),
        ),
        allele_reversal_count=compared_frame.filter(pl.col("allele_alignment_sign") < 0).height,
        max_abs_abs_beta_difference=expression_value_or_zero(
            compared_frame,
            (pl.col("aligned_beta").abs() - pl.col("selected_hail_beta").abs()).abs().max(),
        ),
    )


def build_runtime_comparison(baseline_name: str, baseline_seconds: float, phase1_seconds: float) -> RuntimeComparison:
    """Build a runtime comparison structure."""
    return RuntimeComparison(
        baseline_name=baseline_name,
        baseline_seconds=baseline_seconds,
        phase1_seconds=phase1_seconds,
        speed_ratio_baseline_over_phase1=baseline_seconds / phase1_seconds,
    )


def main() -> None:
    """Run the full Phase 1 evaluation and save a report."""
    from benchmark import (
        build_baseline_paths,
        build_hail_cache_prepare_command,
        build_hail_suite_command,
        build_plink_binary_command,
        build_plink_continuous_command,
        ensure_hail_environment,
        hail_output_path,
        load_hail_suite_report,
        resolve_required_executable,
    )

    baseline_paths = build_baseline_paths()
    plink_executable = resolve_required_executable("PLINK2_BIN", "plink2")

    # Check if Hail baselines exist or should be run
    hail_cont_path = hail_output_path(baseline_paths, "hail_cont")
    hail_bin_wald_path = hail_output_path(baseline_paths, "hail_bin_wald")
    hail_bin_firth_path = hail_output_path(baseline_paths, "hail_bin_firth")
    hail_baselines_exist = all(p.exists() for p in [hail_cont_path, hail_bin_wald_path, hail_bin_firth_path])

    if os.environ.get("HAIL_INCLUDE") or hail_baselines_exist:
        hail_python_executable = ensure_hail_environment()

        if os.environ.get("HAIL_INCLUDE"):
            # Run Hail benchmarks if explicitly enabled
            hail_cache_prepare_seconds = time_command(
                build_hail_cache_prepare_command(hail_python_executable, baseline_paths, cache_mode="reuse")
            )
            hail_suite_command_arguments = build_hail_suite_command(
                hail_python_executable,
                baseline_paths,
                cache_mode="require",
            )
            time_command(hail_suite_command_arguments)
        else:
            # Use cached values from existing report if not running
            hail_cache_prepare_seconds = 0.0

        hail_suite_report = load_hail_suite_report(baseline_paths.hail_suite_report_path)
        hail_step_reports = {
            step_report["output_name"]: step_report
            for step_report in hail_suite_report["step_reports"]
        }
        hail_linear_seconds = float(hail_step_reports["hail_cont"]["duration_seconds"])
        hail_logistic_wald_seconds = float(hail_step_reports["hail_bin_wald"]["duration_seconds"])
        hail_logistic_firth_seconds = float(hail_step_reports["hail_bin_firth"]["duration_seconds"])
    else:
        # Hail baselines not available - will be skipped in report
        hail_cache_prepare_seconds = 0.0
        hail_linear_seconds = 0.0
        hail_logistic_wald_seconds = 0.0
        hail_logistic_firth_seconds = 0.0

    plink_linear_seconds = time_command(build_plink_continuous_command(plink_executable, baseline_paths))
    plink_logistic_seconds = time_command(build_plink_binary_command(plink_executable, baseline_paths))
    phase1_linear_run = run_phase1_linear(baseline_paths)
    phase1_logistic_run = run_phase1_logistic(baseline_paths)

    report = Phase1EvaluationReport(
        hail_cache_prepare_runtime=build_runtime_comparison(
            "hail_matrix_table_prepare",
            hail_cache_prepare_seconds,
            phase1_linear_run.duration_seconds,
        ),
        linear_plink_runtime=build_runtime_comparison(
            "plink_linear",
            plink_linear_seconds,
            phase1_linear_run.duration_seconds,
        ),
        logistic_plink_runtime=build_runtime_comparison(
            "plink_logistic_hybrid",
            plink_logistic_seconds,
            phase1_logistic_run.duration_seconds,
        ),
        linear_hail_runtime=build_runtime_comparison(
            "hail_linear",
            hail_linear_seconds,
            phase1_linear_run.duration_seconds,
        ),
        logistic_hail_wald_runtime=build_runtime_comparison(
            "hail_logistic_wald",
            hail_logistic_wald_seconds,
            phase1_logistic_run.duration_seconds,
        ),
        logistic_hail_firth_runtime=build_runtime_comparison(
            "hail_logistic_firth",
            hail_logistic_firth_seconds,
            phase1_logistic_run.duration_seconds,
        ),
        logistic_hail_hybrid_upper_bound_runtime=build_runtime_comparison(
            "hail_logistic_hybrid_upper_bound",
            hail_logistic_wald_seconds + hail_logistic_firth_seconds,
            phase1_logistic_run.duration_seconds,
        ),
        linear_plink_parity=summarize_linear_plink_parity(baseline_paths, phase1_linear_run.result_frame),
        logistic_plink_parity=summarize_logistic_plink_parity(baseline_paths, phase1_logistic_run.result_frame),
        linear_hail_parity=summarize_linear_hail_parity(baseline_paths, phase1_linear_run.result_frame),
        logistic_hail_hybrid_parity=summarize_logistic_hail_hybrid_parity(
            baseline_paths,
            phase1_logistic_run.result_frame,
        ),
    )
    report_path = baseline_paths.data_directory / "phase1_evaluation_report.json"
    report_path.write_text(f"{json.dumps(asdict(report), indent=2)}\n")
    print(report_path)


if __name__ == "__main__":
    main()
