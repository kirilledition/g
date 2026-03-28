#!/usr/bin/env python3
"""Run full Phase 1 evaluation against PLINK and Hail baselines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark import BaselinePaths


@dataclass(frozen=True)
class RuntimeComparison:
    """Runtime comparison between one baseline and the Phase 1 engine."""

    baseline_name: str
    phase1_backend: str
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
class Plink1LogisticParitySummary:
    """Best-effort logistic parity summary against PLINK 1.9."""

    compared_variant_count: int
    max_abs_beta_difference: float
    max_abs_standard_error_difference: float
    max_abs_z_statistic_difference: float
    max_abs_log10_p_value_difference: float
    allele_reversal_count: int
    max_abs_abs_beta_difference: float
    max_abs_abs_z_statistic_difference: float


class Phase1CommandResult(NamedTuple):
    """Output table and runtime for one subprocess Phase 1 execution."""

    success: bool
    result_frame: pl.DataFrame | None
    duration_seconds: float | None
    output_path: Path
    error_message: str | None


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


def read_whitespace_table(table_path: Path) -> pl.DataFrame:
    """Read a whitespace-delimited results table into a Polars frame."""
    table_lines = [line.strip() for line in table_path.read_text().splitlines() if line.strip()]
    if not table_lines:
        raise ValueError(f"Results table is empty: {table_path}")
    header_columns = table_lines[0].split()
    rows: list[dict[str, str]] = []
    for line in table_lines[1:]:
        values = line.split()
        if len(values) != len(header_columns):
            raise ValueError(
                f"Unexpected column count in {table_path}: expected {len(header_columns)}, got {len(values)}"
            )
        rows.append(dict(zip(header_columns, values, strict=True)))
    return pl.DataFrame(rows)


def build_phase1_command(
    baseline_paths: BaselinePaths,
    association_mode: str,
    phase1_backend: str,
    output_prefix: Path,
) -> list[str]:
    """Build one subprocess command for the Phase 1 CLI."""
    if association_mode == "linear":
        return [
            sys.executable,
            "-c",
            "from g.cli import main; main()",
            "linear",
            "--bfile",
            str(baseline_paths.bed_prefix),
            "--pheno",
            str(baseline_paths.continuous_phenotype_path),
            "--pheno-name",
            "phenotype_continuous",
            "--covar",
            str(baseline_paths.covariate_path),
            "--covar-names",
            "age,sex",
            "--chunk-size",
            "2048",
            "--device",
            phase1_backend,
            "--out",
            str(output_prefix),
        ]
    return [
        sys.executable,
        "-c",
        "from g.cli import main; main()",
        "logistic",
        "--bfile",
        str(baseline_paths.bed_prefix),
        "--pheno",
        str(baseline_paths.binary_phenotype_path),
        "--pheno-name",
        "phenotype_binary",
        "--covar",
        str(baseline_paths.covariate_path),
        "--covar-names",
        "age,sex",
        "--chunk-size",
        "512",
        "--device",
        phase1_backend,
        "--max-iterations",
        "50",
        "--tolerance",
        "1.0e-8",
        "--out",
        str(output_prefix),
    ]


def output_path_for_phase1_run(output_prefix: Path, association_mode: str) -> Path:
    """Return the expected result path for one Phase 1 CLI run."""
    output_suffix = ".linear.tsv" if association_mode == "linear" else ".logistic.tsv"
    return output_prefix.with_suffix(output_suffix)


def run_phase1_command(
    baseline_paths: BaselinePaths,
    association_mode: str,
    phase1_backend: str,
    output_prefix: Path,
) -> Phase1CommandResult:
    """Run one Phase 1 CLI command in a subprocess and load its results."""
    command_arguments = build_phase1_command(
        baseline_paths=baseline_paths,
        association_mode=association_mode,
        phase1_backend=phase1_backend,
        output_prefix=output_prefix,
    )
    command_environment = os.environ.copy()
    command_environment["JAX_PLATFORMS"] = phase1_backend
    expected_output_path = output_path_for_phase1_run(output_prefix, association_mode)

    start_time = time.perf_counter()
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
        env=command_environment,
    )
    duration_seconds = time.perf_counter() - start_time

    if completed_process.returncode != 0:
        error_message = completed_process.stderr.strip() or completed_process.stdout.strip() or "Phase 1 run failed."
        return Phase1CommandResult(
            success=False,
            result_frame=None,
            duration_seconds=None,
            output_path=expected_output_path,
            error_message=error_message,
        )

    if not expected_output_path.exists():
        return Phase1CommandResult(
            success=False,
            result_frame=None,
            duration_seconds=None,
            output_path=expected_output_path,
            error_message=f"Expected output file was not written: {expected_output_path}",
        )

    return Phase1CommandResult(
        success=True,
        result_frame=pl.read_csv(expected_output_path, separator="\t"),
        duration_seconds=duration_seconds,
        output_path=expected_output_path,
        error_message=None,
    )


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


def summarize_linear_plink1_parity(baseline_paths: BaselinePaths, phase1_frame: pl.DataFrame) -> LinearParitySummary:
    """Summarize best-effort linear parity against PLINK 1.9 output."""
    baseline_path = baseline_paths.baseline_directory / "plink1_cont.assoc.linear"
    baseline_table = read_whitespace_table(baseline_path)
    variant_column_name = "SNP" if "SNP" in baseline_table.columns else "ID"
    statistic_column_name = "STAT" if "STAT" in baseline_table.columns else "T_STAT"
    has_standard_error = "SE" in baseline_table.columns
    baseline_frame = (
        (baseline_table.filter(pl.col("TEST") == "ADD") if "TEST" in baseline_table.columns else baseline_table)
        .with_columns(
            pl.col(variant_column_name).cast(pl.String),
            pl.col("A1").cast(pl.String),
            pl.col("BETA").cast(pl.Float64, strict=False),
            pl.col(statistic_column_name).cast(pl.Float64, strict=False),
            pl.col("P").cast(pl.Float64, strict=False),
        )
        .with_columns(
            pl.col("SE").cast(pl.Float64, strict=False)
            if has_standard_error
            else pl.lit(None).cast(pl.Float64).alias("SE")
        )
        .select(variant_column_name, "A1", "BETA", "SE", statistic_column_name, "P")
        .rename(
            {
                variant_column_name: "variant_identifier",
                "SE": "baseline_standard_error",
                statistic_column_name: "baseline_t_statistic",
                "P": "baseline_p_value",
            }
        )
    )
    joined_frame = align_to_effect_allele(
        phase1_frame.join(baseline_frame, on="variant_identifier", how="inner"),
        baseline_effect_allele_column="A1",
        statistic_column="t_statistic",
    )
    return LinearParitySummary(
        variant_count=joined_frame.height,
        max_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta") - pl.col("BETA")).abs().max(),
        ),
        max_abs_standard_error_difference=expression_value(
            joined_frame,
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs().max()
            if has_standard_error
            else pl.lit(float("nan")),
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
            (pl.col("aligned_beta").abs() - pl.col("BETA").abs()).abs().max(),
        ),
        max_abs_abs_t_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic").abs() - pl.col("baseline_t_statistic").abs()).abs().max(),
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


def summarize_logistic_plink1_parity(
    baseline_paths: BaselinePaths,
    phase1_frame: pl.DataFrame,
) -> Plink1LogisticParitySummary:
    """Summarize best-effort logistic parity against PLINK 1.9 output."""
    baseline_path = baseline_paths.baseline_directory / "plink1_bin.assoc.logistic"
    baseline_table = read_whitespace_table(baseline_path)
    variant_column_name = "SNP" if "SNP" in baseline_table.columns else "ID"
    statistic_column_name = "STAT" if "STAT" in baseline_table.columns else "Z_STAT"
    standard_error_column_name = "SE" if "SE" in baseline_table.columns else "LOG(OR)_SE"
    has_standard_error = standard_error_column_name in baseline_table.columns
    beta_column_expression = (
        pl.col("BETA").cast(pl.Float64, strict=False)
        if "BETA" in baseline_table.columns
        else pl.col("OR").cast(pl.Float64, strict=False).log()
    )
    baseline_frame = (
        (baseline_table.filter(pl.col("TEST") == "ADD") if "TEST" in baseline_table.columns else baseline_table)
        .with_columns(
            pl.col(variant_column_name).cast(pl.String),
            pl.col("A1").cast(pl.String),
            beta_column_expression.alias("baseline_beta"),
            pl.col(statistic_column_name).cast(pl.Float64, strict=False).alias("baseline_z_statistic"),
            pl.col("P").cast(pl.Float64, strict=False).alias("baseline_p_value"),
        )
        .with_columns(
            pl.col(standard_error_column_name).cast(pl.Float64, strict=False).alias("baseline_standard_error")
            if has_standard_error
            else pl.lit(None).cast(pl.Float64).alias("baseline_standard_error")
        )
        .select(
            variant_column_name,
            "A1",
            "baseline_beta",
            "baseline_standard_error",
            "baseline_z_statistic",
            "baseline_p_value",
        )
        .rename({variant_column_name: "variant_identifier"})
    )
    joined_frame = align_to_effect_allele(
        phase1_frame.join(baseline_frame, on="variant_identifier", how="inner"),
        baseline_effect_allele_column="A1",
        statistic_column="z_statistic",
    )
    return Plink1LogisticParitySummary(
        compared_variant_count=joined_frame.height,
        max_abs_beta_difference=expression_value(
            joined_frame,
            (pl.col("aligned_beta") - pl.col("baseline_beta")).abs().max(),
        ),
        max_abs_standard_error_difference=expression_value(
            joined_frame,
            (pl.col("standard_error") - pl.col("baseline_standard_error")).abs().max()
            if has_standard_error
            else pl.lit(float("nan")),
        ),
        max_abs_z_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic") - pl.col("baseline_z_statistic")).abs().max(),
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
        max_abs_abs_z_statistic_difference=expression_value(
            joined_frame,
            (pl.col("aligned_statistic").abs() - pl.col("baseline_z_statistic").abs()).abs().max(),
        ),
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


def build_runtime_comparison(
    baseline_name: str,
    baseline_seconds: float,
    phase1_backend: str,
    phase1_seconds: float,
) -> RuntimeComparison:
    """Build a runtime comparison structure."""
    return RuntimeComparison(
        baseline_name=baseline_name,
        phase1_backend=phase1_backend,
        baseline_seconds=baseline_seconds,
        phase1_seconds=phase1_seconds,
        speed_ratio_baseline_over_phase1=baseline_seconds / phase1_seconds,
    )


def build_runtime_comparison_or_none(
    baseline_name: str,
    baseline_seconds: float | None,
    phase1_backend: str,
    phase1_seconds: float | None,
) -> RuntimeComparison | None:
    """Build a runtime comparison only when both timings are available."""
    if baseline_seconds is None or phase1_seconds is None:
        return None
    return build_runtime_comparison(
        baseline_name=baseline_name,
        baseline_seconds=baseline_seconds,
        phase1_backend=phase1_backend,
        phase1_seconds=phase1_seconds,
    )


def serialize_dataclass_or_none(value: object | None) -> dict[str, object] | None:
    """Convert an optional dataclass value to a JSON-serializable dictionary."""
    if value is None:
        return None
    return asdict(value)


def main() -> None:
    """Run the full Phase 1 evaluation and save a report."""
    from benchmark import (
        build_baseline_paths,
        build_hail_cache_prepare_command,
        build_hail_suite_command,
        build_plink1_binary_command,
        build_plink1_continuous_command,
        build_plink2_binary_command,
        build_plink2_continuous_command,
        ensure_hail_environment,
        hail_output_path,
        load_hail_suite_report,
        resolve_required_executable,
    )

    baseline_paths = build_baseline_paths()
    plink1_executable = resolve_required_executable("PLINK1_BIN", "plink")
    plink2_executable = resolve_required_executable("PLINK2_BIN", "plink2")

    hail_continuous_path = hail_output_path(baseline_paths, "hail_cont")
    hail_binary_wald_path = hail_output_path(baseline_paths, "hail_bin_wald")
    hail_binary_firth_path = hail_output_path(baseline_paths, "hail_bin_firth")
    hail_baselines_exist = all(
        path.exists() for path in [hail_continuous_path, hail_binary_wald_path, hail_binary_firth_path]
    )

    if os.environ.get("HAIL_INCLUDE") or hail_baselines_exist:
        hail_python_executable = ensure_hail_environment()

        if os.environ.get("HAIL_INCLUDE"):
            hail_cache_prepare_seconds: float | None = time_command(
                build_hail_cache_prepare_command(hail_python_executable, baseline_paths, cache_mode="reuse")
            )
            hail_suite_command_arguments = build_hail_suite_command(
                hail_python_executable,
                baseline_paths,
                cache_mode="require",
            )
            time_command(hail_suite_command_arguments)
        else:
            hail_cache_prepare_seconds = 0.0

        hail_suite_report = load_hail_suite_report(baseline_paths.hail_suite_report_path)
        hail_step_reports = {
            step_report["output_name"]: step_report for step_report in hail_suite_report["step_reports"]
        }
        hail_linear_seconds: float | None = float(hail_step_reports["hail_cont"]["duration_seconds"])
        hail_logistic_wald_seconds: float | None = float(hail_step_reports["hail_bin_wald"]["duration_seconds"])
        hail_logistic_firth_seconds: float | None = float(hail_step_reports["hail_bin_firth"]["duration_seconds"])
    else:
        hail_cache_prepare_seconds = None
        hail_linear_seconds = None
        hail_logistic_wald_seconds = None
        hail_logistic_firth_seconds = None

    plink1_linear_seconds = time_command(build_plink1_continuous_command(plink1_executable, baseline_paths))
    plink1_logistic_seconds = time_command(build_plink1_binary_command(plink1_executable, baseline_paths))
    plink2_linear_seconds = time_command(build_plink2_continuous_command(plink2_executable, baseline_paths))
    plink2_logistic_seconds = time_command(build_plink2_binary_command(plink2_executable, baseline_paths))

    phase1_linear_cpu_run = run_phase1_command(
        baseline_paths=baseline_paths,
        association_mode="linear",
        phase1_backend="cpu",
        output_prefix=baseline_paths.data_directory / "phase1_linear_cpu",
    )
    phase1_logistic_cpu_run = run_phase1_command(
        baseline_paths=baseline_paths,
        association_mode="logistic",
        phase1_backend="cpu",
        output_prefix=baseline_paths.data_directory / "phase1_logistic_cpu",
    )
    phase1_linear_gpu_run = run_phase1_command(
        baseline_paths=baseline_paths,
        association_mode="linear",
        phase1_backend="gpu",
        output_prefix=baseline_paths.data_directory / "phase1_linear_gpu",
    )
    phase1_logistic_gpu_run = run_phase1_command(
        baseline_paths=baseline_paths,
        association_mode="logistic",
        phase1_backend="gpu",
        output_prefix=baseline_paths.data_directory / "phase1_logistic_gpu",
    )

    if not phase1_linear_cpu_run.success or phase1_linear_cpu_run.result_frame is None:
        raise RuntimeError(phase1_linear_cpu_run.error_message or "CPU linear Phase 1 run failed.")
    if not phase1_logistic_cpu_run.success or phase1_logistic_cpu_run.result_frame is None:
        raise RuntimeError(phase1_logistic_cpu_run.error_message or "CPU logistic Phase 1 run failed.")

    linear_plink1_parity_error: str | None = None
    linear_plink1_parity: LinearParitySummary | None = None
    try:
        linear_plink1_parity = summarize_linear_plink1_parity(baseline_paths, phase1_linear_cpu_run.result_frame)
    except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
        linear_plink1_parity_error = str(error)

    logistic_plink1_parity_error: str | None = None
    logistic_plink1_parity: Plink1LogisticParitySummary | None = None
    try:
        logistic_plink1_parity = summarize_logistic_plink1_parity(baseline_paths, phase1_logistic_cpu_run.result_frame)
    except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
        logistic_plink1_parity_error = str(error)

    report = {
        "linear_plink1_cpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink1_linear",
                baseline_seconds=plink1_linear_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_linear_cpu_run.duration_seconds,
            )
        ),
        "linear_plink1_gpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink1_linear",
                baseline_seconds=plink1_linear_seconds,
                phase1_backend="gpu",
                phase1_seconds=phase1_linear_gpu_run.duration_seconds,
            )
        ),
        "logistic_plink1_cpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink1_logistic",
                baseline_seconds=plink1_logistic_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_logistic_cpu_run.duration_seconds,
            )
        ),
        "logistic_plink1_gpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink1_logistic",
                baseline_seconds=plink1_logistic_seconds,
                phase1_backend="gpu",
                phase1_seconds=phase1_logistic_gpu_run.duration_seconds,
            )
        ),
        "linear_plink2_cpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink2_linear",
                baseline_seconds=plink2_linear_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_linear_cpu_run.duration_seconds,
            )
        ),
        "linear_plink2_gpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink2_linear",
                baseline_seconds=plink2_linear_seconds,
                phase1_backend="gpu",
                phase1_seconds=phase1_linear_gpu_run.duration_seconds,
            )
        ),
        "logistic_plink2_cpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink2_logistic_hybrid",
                baseline_seconds=plink2_logistic_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_logistic_cpu_run.duration_seconds,
            )
        ),
        "logistic_plink2_gpu_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="plink2_logistic_hybrid",
                baseline_seconds=plink2_logistic_seconds,
                phase1_backend="gpu",
                phase1_seconds=phase1_logistic_gpu_run.duration_seconds,
            )
        ),
        "hail_cache_prepare_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="hail_matrix_table_prepare",
                baseline_seconds=hail_cache_prepare_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_linear_cpu_run.duration_seconds,
            )
        ),
        "linear_hail_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="hail_linear",
                baseline_seconds=hail_linear_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_linear_cpu_run.duration_seconds,
            )
        ),
        "logistic_hail_wald_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="hail_logistic_wald",
                baseline_seconds=hail_logistic_wald_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_logistic_cpu_run.duration_seconds,
            )
        ),
        "logistic_hail_firth_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="hail_logistic_firth",
                baseline_seconds=hail_logistic_firth_seconds,
                phase1_backend="cpu",
                phase1_seconds=phase1_logistic_cpu_run.duration_seconds,
            )
        ),
        "logistic_hail_hybrid_upper_bound_runtime": serialize_dataclass_or_none(
            build_runtime_comparison_or_none(
                baseline_name="hail_logistic_hybrid_upper_bound",
                baseline_seconds=(hail_logistic_wald_seconds + hail_logistic_firth_seconds)
                if hail_logistic_wald_seconds is not None and hail_logistic_firth_seconds is not None
                else None,
                phase1_backend="cpu",
                phase1_seconds=phase1_logistic_cpu_run.duration_seconds,
            )
        ),
        "phase1_run_status": {
            "linear_cpu": {
                "success": phase1_linear_cpu_run.success,
                "duration_seconds": phase1_linear_cpu_run.duration_seconds,
                "output_path": str(phase1_linear_cpu_run.output_path),
                "error_message": phase1_linear_cpu_run.error_message,
            },
            "linear_gpu": {
                "success": phase1_linear_gpu_run.success,
                "duration_seconds": phase1_linear_gpu_run.duration_seconds,
                "output_path": str(phase1_linear_gpu_run.output_path),
                "error_message": phase1_linear_gpu_run.error_message,
            },
            "logistic_cpu": {
                "success": phase1_logistic_cpu_run.success,
                "duration_seconds": phase1_logistic_cpu_run.duration_seconds,
                "output_path": str(phase1_logistic_cpu_run.output_path),
                "error_message": phase1_logistic_cpu_run.error_message,
            },
            "logistic_gpu": {
                "success": phase1_logistic_gpu_run.success,
                "duration_seconds": phase1_logistic_gpu_run.duration_seconds,
                "output_path": str(phase1_logistic_gpu_run.output_path),
                "error_message": phase1_logistic_gpu_run.error_message,
            },
        },
        "linear_plink2_parity": asdict(
            summarize_linear_plink_parity(baseline_paths, phase1_linear_cpu_run.result_frame)
        ),
        "logistic_plink2_parity": asdict(
            summarize_logistic_plink_parity(baseline_paths, phase1_logistic_cpu_run.result_frame)
        ),
        "linear_hail_parity": serialize_dataclass_or_none(
            summarize_linear_hail_parity(baseline_paths, phase1_linear_cpu_run.result_frame)
            if hail_linear_seconds is not None
            else None
        ),
        "logistic_hail_hybrid_parity": serialize_dataclass_or_none(
            summarize_logistic_hail_hybrid_parity(baseline_paths, phase1_logistic_cpu_run.result_frame)
            if hail_logistic_wald_seconds is not None and hail_logistic_firth_seconds is not None
            else None
        ),
        "linear_plink1_parity": serialize_dataclass_or_none(linear_plink1_parity),
        "linear_plink1_parity_error": linear_plink1_parity_error,
        "logistic_plink1_parity": serialize_dataclass_or_none(logistic_plink1_parity),
        "logistic_plink1_parity_error": logistic_plink1_parity_error,
    }
    report_path = baseline_paths.data_directory / "phase1_evaluation_report.json"
    report_path.write_text(f"{json.dumps(report, indent=2)}\n")
    print(report_path)


if __name__ == "__main__":
    main()
