#!/usr/bin/env python3
"""Run a comparison-oriented benchmark between original regenie and g."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import typing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

if typing.TYPE_CHECKING:
    from scripts import benchmark as baseline_benchmark
else:
    try:
        from scripts import benchmark as baseline_benchmark
    except ModuleNotFoundError:
        import benchmark as baseline_benchmark


PARITY_BETA_ATOL = 1.0e-3
PARITY_LOG10P_ATOL = 1.5e-2
G_FINALIZE_PARQUET = True


@dataclass(frozen=True)
class ComparisonProgramResult:
    """Structured benchmark result for one program run."""

    program_name: str
    implementation: str
    trait_type: str
    step: int
    device: str
    status: str
    wall_time_seconds: float | None
    variants_per_second: float | None
    peak_memory_megabytes: float | None
    stdout_log_path: str | None
    stderr_log_path: str | None
    output_paths: list[str]
    output_row_count: int | None
    prediction_list_present: bool | None
    notes: str | None = None


@dataclass(frozen=True)
class QuantitativeStep2Agreement:
    """Agreement summary between original regenie and g for step 2 quantitative."""

    comparable: bool
    merged_variant_count: int
    beta_max_abs_error: float | None
    beta_mean_abs_error: float | None
    beta_allclose_within_tolerance: bool | None
    log10p_max_abs_error: float | None
    log10p_mean_abs_error: float | None
    log10p_allclose_within_tolerance: bool | None
    notes: str | None = None


@dataclass(frozen=True)
class BinaryStep2Agreement:
    """Agreement summary between original regenie and g for step 2 binary."""

    comparable: bool
    merged_variant_count: int
    beta_max_abs_error: float | None
    beta_mean_abs_error: float | None
    log10p_max_abs_error: float | None
    log10p_mean_abs_error: float | None
    extra_match_rate: float | None
    notes: str | None = None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark original regenie against g REGENIE step 2.")
    parser.add_argument("--include-gpu", action="store_true", help="Run g quantitative step 2 on GPU.")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU benchmark even if available.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for g runs.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Chunk size for g runs.")
    parser.add_argument(
        "--only-quantitative-step2",
        action="store_true",
        help="Benchmark only original regenie quantitative step 2 against g step 2.",
    )
    parser.add_argument(
        "--only-binary-step2",
        action="store_true",
        help="Benchmark only original regenie binary step 2 against g step 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks/regenie_comparison"),
        help="Directory where report and logs are written.",
    )
    return parser


def run_command_with_logs(
    *,
    command_arguments: list[str],
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> tuple[bool, float, str | None]:
    """Run one command and persist logs."""
    start_time = time.perf_counter()
    completed_process = subprocess.run(command_arguments, check=False, capture_output=True, text=True)
    duration_seconds = time.perf_counter() - start_time
    stdout_log_path.write_text(completed_process.stdout)
    stderr_log_path.write_text(completed_process.stderr)
    if completed_process.returncode != 0:
        error_message = completed_process.stderr.strip() or f"Command exited with code {completed_process.returncode}."
        return False, duration_seconds, error_message
    return True, duration_seconds, None


def count_regenie_step2_rows(output_prefix: Path) -> int | None:
    """Count data rows in a regenie step 2 output file."""
    result_path = output_prefix.with_name(f"{output_prefix.name}_phenotype_continuous.regenie")
    if not result_path.exists():
        result_path = output_prefix.with_name(f"{output_prefix.name}_phenotype_binary.regenie")
    if not result_path.exists():
        return None
    with result_path.open() as result_file:
        line_count = sum(1 for line in result_file if line.strip())
    return max(0, line_count - 1)


def count_table_rows(table_path: Path) -> int | None:
    """Count rows in a tabular output file."""
    if not table_path.exists():
        return None
    if table_path.suffix == ".parquet":
        return int(pd.read_parquet(table_path).shape[0])
    return int(pd.read_csv(table_path, sep="\t").shape[0])


def build_not_implemented_result(
    *,
    program_name: str,
    trait_type: str,
    step: int,
    device: str,
) -> ComparisonProgramResult:
    """Build a not-implemented placeholder result."""
    return ComparisonProgramResult(
        program_name=program_name,
        implementation="g",
        trait_type=trait_type,
        step=step,
        device=device,
        status="not_implemented",
        wall_time_seconds=None,
        variants_per_second=None,
        peak_memory_megabytes=None,
        stdout_log_path=None,
        stderr_log_path=None,
        output_paths=[],
        output_row_count=None,
        prediction_list_present=None,
        notes="This g workflow is not implemented in the active public surface.",
    )


def build_regenie_program_specs(
    regenie_executable: str,
    baseline_paths: baseline_benchmark.BaselinePaths,
    *,
    only_quantitative_step2: bool = False,
    only_binary_step2: bool = False,
) -> list[tuple[str, str, int, list[str], Path]]:
    """Build original regenie comparison specs."""
    if only_quantitative_step2:
        return [
            (
                "regenie_step2_quantitative",
                "quantitative",
                2,
                baseline_benchmark.build_regenie_step2_continuous_command(regenie_executable, baseline_paths),
                baseline_paths.baseline_directory / "regenie_step2_qt",
            )
        ]
    if only_binary_step2:
        return [
            (
                "regenie_step2_binary",
                "binary",
                2,
                baseline_benchmark.build_regenie_step2_command(regenie_executable, baseline_paths),
                baseline_paths.baseline_directory / "regenie_step2",
            )
        ]

    return [
        (
            "regenie_step1_binary",
            "binary",
            1,
            baseline_benchmark.build_regenie_step1_command(regenie_executable, baseline_paths),
            baseline_paths.baseline_directory / "regenie_step1",
        ),
        (
            "regenie_step2_binary",
            "binary",
            2,
            baseline_benchmark.build_regenie_step2_command(regenie_executable, baseline_paths),
            baseline_paths.baseline_directory / "regenie_step2",
        ),
        (
            "regenie_step1_quantitative",
            "quantitative",
            1,
            baseline_benchmark.build_regenie_step1_continuous_command(regenie_executable, baseline_paths),
            baseline_paths.baseline_directory / "regenie_step1_qt",
        ),
        (
            "regenie_step2_quantitative",
            "quantitative",
            2,
            baseline_benchmark.build_regenie_step2_continuous_command(regenie_executable, baseline_paths),
            baseline_paths.baseline_directory / "regenie_step2_qt",
        ),
    ]


def build_g_step2_command(
    *,
    uv_executable: str,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_prefix: Path,
    device: str,
    chunk_size: int,
    variant_limit: int | None,
    trait_type: str = "quantitative",
) -> list[str]:
    """Build g step2 CLI command."""
    phenotype_path = baseline_paths.continuous_phenotype_path
    phenotype_name = "phenotype_continuous"
    prediction_list_path = baseline_paths.regenie_qt_prediction_list_path
    if trait_type == "binary":
        phenotype_path = baseline_paths.binary_phenotype_path
        phenotype_name = "phenotype_binary"
        prediction_list_path = baseline_paths.regenie_prediction_list_path
    command_arguments = [
        uv_executable,
        "run",
        "g",
        "regenie2",
        "--bgen",
        str(baseline_paths.bgen_path),
        "--sample",
        str(baseline_paths.sample_path),
        "--pheno",
        str(phenotype_path),
        "--pheno-name",
        phenotype_name,
        "--covar",
        str(baseline_paths.covariate_path),
        "--covar-names",
        "age,sex",
        "--pred",
        str(prediction_list_path),
        "--out",
        str(output_prefix),
        "--trait-type",
        trait_type,
        "--chunk-size",
        str(chunk_size),
        "--device",
        device,
    ]
    if G_FINALIZE_PARQUET:
        command_arguments.append("--finalize-parquet")
    if variant_limit is not None:
        command_arguments.extend(["--variant-limit", str(variant_limit)])
    return command_arguments


def run_regenie_program(
    *,
    program_name: str,
    trait_type: str,
    step: int,
    command_arguments: list[str],
    output_prefix: Path,
    log_directory: Path,
) -> ComparisonProgramResult:
    """Run one original regenie program and collect metadata."""
    stdout_log_path = log_directory / f"{program_name}.stdout.log"
    stderr_log_path = log_directory / f"{program_name}.stderr.log"
    success, duration_seconds, error_message = run_command_with_logs(
        command_arguments=command_arguments,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )
    output_paths: list[str] = []
    output_row_count: int | None = None
    prediction_list_present: bool | None = None
    variants_per_second: float | None = None
    if step == 1:
        prediction_path = output_prefix.parent / f"{output_prefix.name}_pred.list"
        prediction_list_present = prediction_path.exists()
        output_paths = [str(prediction_path)] if prediction_list_present else []
    else:
        if trait_type == "quantitative":
            result_path = output_prefix.parent / f"{output_prefix.name}_phenotype_continuous.regenie"
        else:
            result_path = output_prefix.parent / f"{output_prefix.name}_phenotype_binary.regenie"
        if result_path.exists():
            output_paths = [str(result_path)]
        output_row_count = count_regenie_step2_rows(output_prefix)
        if output_row_count is not None and duration_seconds > 0:
            variants_per_second = output_row_count / duration_seconds
    status = "success" if success else "failed"
    return ComparisonProgramResult(
        program_name=program_name,
        implementation="regenie",
        trait_type=trait_type,
        step=step,
        device="external_cpu",
        status=status,
        wall_time_seconds=duration_seconds,
        variants_per_second=variants_per_second,
        peak_memory_megabytes=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        output_paths=output_paths,
        output_row_count=output_row_count,
        prediction_list_present=prediction_list_present,
        notes=error_message,
    )


def run_g_step2(
    *,
    program_name: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    output_prefix: Path,
    log_directory: Path,
) -> ComparisonProgramResult:
    """Run one g step2 program and collect metadata."""
    stdout_log_path = log_directory / f"{program_name}.stdout.log"
    stderr_log_path = log_directory / f"{program_name}.stderr.log"
    association_suffix = ".regenie2_binary.run" if trait_type == "binary" else ".regenie2_linear.run"
    output_run_directory = output_prefix.with_suffix(association_suffix)
    if output_run_directory.exists():
        shutil.rmtree(output_run_directory)
    success, duration_seconds, error_message = run_command_with_logs(
        command_arguments=command_arguments,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )
    output_path = output_run_directory / "final.parquet"
    output_row_count = count_table_rows(output_path)
    variants_per_second = None
    if output_row_count is not None and duration_seconds > 0:
        variants_per_second = output_row_count / duration_seconds
    status = "success" if success else "failed"
    return ComparisonProgramResult(
        program_name=program_name,
        implementation="g",
        trait_type=trait_type,
        step=2,
        device=device,
        status=status,
        wall_time_seconds=duration_seconds,
        variants_per_second=variants_per_second,
        peak_memory_megabytes=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        output_paths=[str(output_path)] if output_path.exists() else [],
        output_row_count=output_row_count,
        prediction_list_present=None,
        notes=error_message,
    )


def load_g_output_frame(g_output_path: Path) -> pd.DataFrame:
    """Load one g quantitative step 2 Parquet output table."""
    return pd.read_parquet(g_output_path)


def summarize_quantitative_step2_agreement(
    *,
    regenie_output_path: Path | None,
    g_output_path: Path | None,
) -> QuantitativeStep2Agreement:
    """Compare beta/log10p agreement for quantitative step2 outputs."""
    if regenie_output_path is None or g_output_path is None:
        return QuantitativeStep2Agreement(
            comparable=False,
            merged_variant_count=0,
            beta_max_abs_error=None,
            beta_mean_abs_error=None,
            beta_allclose_within_tolerance=None,
            log10p_max_abs_error=None,
            log10p_mean_abs_error=None,
            log10p_allclose_within_tolerance=None,
            notes="One or both outputs are missing.",
        )
    baseline_frame = pd.read_csv(regenie_output_path, sep=r"\s+")
    observed_frame = load_g_output_frame(g_output_path)

    required_observed_columns = {"chromosome", "position", "variant_identifier", "allele_one", "allele_two"}
    required_baseline_columns = {"CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1"}
    if required_observed_columns.issubset(observed_frame.columns) and required_baseline_columns.issubset(
        baseline_frame.columns
    ):
        observed_frame = observed_frame.assign(
            chromosome=observed_frame["chromosome"].astype(str),
            position=pd.to_numeric(observed_frame["position"], downcast="integer"),
            variant_identifier=observed_frame["variant_identifier"].astype(str),
            allele_one=observed_frame["allele_one"].astype(str),
            allele_two=observed_frame["allele_two"].astype(str),
        )
        baseline_frame = baseline_frame.assign(
            CHROM=baseline_frame["CHROM"].astype(str),
            GENPOS=pd.to_numeric(baseline_frame["GENPOS"], downcast="integer"),
            ID=baseline_frame["ID"].astype(str),
            ALLELE0=baseline_frame["ALLELE0"].astype(str),
            ALLELE1=baseline_frame["ALLELE1"].astype(str),
        )
        merged_frame = observed_frame.merge(
            baseline_frame[["CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "BETA", "LOG10P"]],
            left_on=["chromosome", "position", "variant_identifier", "allele_two", "allele_one"],
            right_on=["CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1"],
            how="inner",
        )
    else:
        merged_frame = observed_frame.merge(
            baseline_frame.rename(columns={"ID": "variant_identifier"})[["variant_identifier", "BETA", "LOG10P"]],
            on="variant_identifier",
            how="inner",
        )
    if merged_frame.empty:
        return QuantitativeStep2Agreement(
            comparable=False,
            merged_variant_count=0,
            beta_max_abs_error=None,
            beta_mean_abs_error=None,
            beta_allclose_within_tolerance=None,
            log10p_max_abs_error=None,
            log10p_mean_abs_error=None,
            log10p_allclose_within_tolerance=None,
            notes="No overlapping variants between outputs.",
        )
    beta_error_series = (merged_frame["beta"] - merged_frame["BETA"]).abs()
    log10p_error_series = (merged_frame["log10_p_value"] - merged_frame["LOG10P"]).abs()
    return QuantitativeStep2Agreement(
        comparable=True,
        merged_variant_count=int(merged_frame.shape[0]),
        beta_max_abs_error=float(beta_error_series.max()),
        beta_mean_abs_error=float(beta_error_series.mean()),
        beta_allclose_within_tolerance=bool((beta_error_series <= PARITY_BETA_ATOL).all()),
        log10p_max_abs_error=float(log10p_error_series.max()),
        log10p_mean_abs_error=float(log10p_error_series.mean()),
        log10p_allclose_within_tolerance=bool((log10p_error_series <= PARITY_LOG10P_ATOL).all()),
    )


def normalize_binary_extra(extra_series: pd.Series) -> pd.Series:
    """Normalize binary correction labels for comparison."""
    return extra_series.fillna("NA").astype(str)


def summarize_binary_step2_agreement(
    *,
    regenie_output_path: Path | None,
    g_output_path: Path | None,
) -> BinaryStep2Agreement:
    """Compare beta/log10p agreement for binary step2 outputs."""
    if regenie_output_path is None or g_output_path is None:
        return BinaryStep2Agreement(
            comparable=False,
            merged_variant_count=0,
            beta_max_abs_error=None,
            beta_mean_abs_error=None,
            log10p_max_abs_error=None,
            log10p_mean_abs_error=None,
            extra_match_rate=None,
            notes="One or both outputs are missing.",
        )
    baseline_frame = pd.read_csv(regenie_output_path, sep=r"\s+")
    observed_frame = load_g_output_frame(g_output_path)
    key_columns = ["CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1"]
    for column_name in key_columns:
        baseline_frame[column_name] = baseline_frame[column_name].astype(str)
        observed_frame[column_name] = observed_frame[column_name].astype(str)
    baseline_frame["GENPOS"] = pd.to_numeric(baseline_frame["GENPOS"], downcast="integer")
    observed_frame["GENPOS"] = pd.to_numeric(observed_frame["GENPOS"], downcast="integer")
    merged_frame = observed_frame.merge(
        baseline_frame[[*key_columns, "BETA", "LOG10P", "EXTRA"]],
        on=key_columns,
        how="inner",
        suffixes=("_g", "_regenie"),
    )
    if merged_frame.empty:
        return BinaryStep2Agreement(
            comparable=False,
            merged_variant_count=0,
            beta_max_abs_error=None,
            beta_mean_abs_error=None,
            log10p_max_abs_error=None,
            log10p_mean_abs_error=None,
            extra_match_rate=None,
            notes="No overlapping variants between outputs.",
        )
    beta_error_series = (merged_frame["BETA_g"] - merged_frame["BETA_regenie"]).abs()
    log10p_error_series = (merged_frame["LOG10P_g"] - merged_frame["LOG10P_regenie"]).abs()
    extra_match_series = normalize_binary_extra(merged_frame["EXTRA_g"]) == normalize_binary_extra(
        merged_frame["EXTRA_regenie"]
    )
    return BinaryStep2Agreement(
        comparable=True,
        merged_variant_count=int(merged_frame.shape[0]),
        beta_max_abs_error=float(beta_error_series.max()),
        beta_mean_abs_error=float(beta_error_series.mean()),
        log10p_max_abs_error=float(log10p_error_series.max()),
        log10p_mean_abs_error=float(log10p_error_series.mean()),
        extra_match_rate=float(extra_match_series.mean()),
    )


def extract_program(result_list: list[ComparisonProgramResult], program_name: str) -> ComparisonProgramResult | None:
    """Find one result by program name."""
    for result in result_list:
        if result.program_name == program_name:
            return result
    return None


def result_output_path(result: ComparisonProgramResult) -> Path | None:
    """Resolve first output path."""
    if not result.output_paths:
        return None
    return Path(result.output_paths[0])


def write_text_summary(
    *,
    report_path: Path,
    results: list[ComparisonProgramResult],
    agreement_cpu: QuantitativeStep2Agreement,
    agreement_gpu: QuantitativeStep2Agreement | None,
    binary_agreement_cpu: BinaryStep2Agreement | None = None,
    binary_agreement_gpu: BinaryStep2Agreement | None = None,
) -> None:
    """Write a short text summary."""
    lines: list[str] = []
    lines.append("REGENIE Comparison Benchmark Summary")
    lines.append("")
    for result in results:
        lines.append(
            f"{result.program_name}: status={result.status}, wall_time_seconds={result.wall_time_seconds}, "
            f"rows={result.output_row_count}, device={result.device}",
        )
    regenie_qt = extract_program(results, "regenie_step2_quantitative")
    g_cpu = extract_program(results, "g_regenie2_quantitative_step2_cpu")
    g_gpu = extract_program(results, "g_regenie2_quantitative_step2_gpu")
    lines.append("")
    lines.append("Direct Runtime Comparisons (Quantitative Step 2)")
    if regenie_qt is not None and regenie_qt.status == "success" and regenie_qt.wall_time_seconds is not None:
        if g_cpu is not None and g_cpu.status == "success" and g_cpu.wall_time_seconds is not None:
            cpu_speedup = regenie_qt.wall_time_seconds / g_cpu.wall_time_seconds
            cpu_delta = g_cpu.wall_time_seconds - regenie_qt.wall_time_seconds
            lines.append(
                f"regenie vs g CPU: speedup={cpu_speedup:.4f}x, delta_seconds={cpu_delta:.4f}",
            )
        if g_gpu is not None and g_gpu.status == "success" and g_gpu.wall_time_seconds is not None:
            gpu_speedup = regenie_qt.wall_time_seconds / g_gpu.wall_time_seconds
            gpu_delta = g_gpu.wall_time_seconds - regenie_qt.wall_time_seconds
            lines.append(
                f"regenie vs g GPU: speedup={gpu_speedup:.4f}x, delta_seconds={gpu_delta:.4f}",
            )
    lines.append("")
    lines.append("Numeric Agreement (Quantitative Step 2)")
    lines.append(
        f"g CPU comparable={agreement_cpu.comparable}, merged_variants={agreement_cpu.merged_variant_count}, "
        f"beta_allclose={agreement_cpu.beta_allclose_within_tolerance}, "
        f"log10p_allclose={agreement_cpu.log10p_allclose_within_tolerance}",
    )
    if agreement_gpu is not None:
        lines.append(
            f"g GPU comparable={agreement_gpu.comparable}, merged_variants={agreement_gpu.merged_variant_count}, "
            f"beta_allclose={agreement_gpu.beta_allclose_within_tolerance}, "
            f"log10p_allclose={agreement_gpu.log10p_allclose_within_tolerance}",
        )
    if binary_agreement_cpu is not None:
        lines.append("")
        lines.append("Numeric Agreement (Binary Step 2)")
        lines.append(
            f"g CPU comparable={binary_agreement_cpu.comparable}, "
            f"merged_variants={binary_agreement_cpu.merged_variant_count}, "
            f"beta_max_abs_error={binary_agreement_cpu.beta_max_abs_error}, "
            f"log10p_max_abs_error={binary_agreement_cpu.log10p_max_abs_error}, "
            f"extra_match_rate={binary_agreement_cpu.extra_match_rate}",
        )
    if binary_agreement_gpu is not None:
        lines.append(
            f"g GPU comparable={binary_agreement_gpu.comparable}, "
            f"merged_variants={binary_agreement_gpu.merged_variant_count}, "
            f"beta_max_abs_error={binary_agreement_gpu.beta_max_abs_error}, "
            f"log10p_max_abs_error={binary_agreement_gpu.log10p_max_abs_error}, "
            f"extra_match_rate={binary_agreement_gpu.extra_match_rate}",
        )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the comparison benchmark."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    baseline_paths = baseline_benchmark.build_baseline_paths()
    baseline_benchmark.validate_input_files(baseline_paths)
    regenie_executable = baseline_benchmark.resolve_required_executable("REGENIE_BIN", "regenie")
    uv_executable = baseline_benchmark.resolve_required_executable("UV_BIN", "uv")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    log_directory = arguments.output_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    results: list[ComparisonProgramResult] = []
    for program_name, trait_type, step, command_arguments, output_prefix in build_regenie_program_specs(
        regenie_executable,
        baseline_paths,
        only_quantitative_step2=arguments.only_quantitative_step2,
        only_binary_step2=arguments.only_binary_step2,
    ):
        results.append(
            run_regenie_program(
                program_name=program_name,
                trait_type=trait_type,
                step=step,
                command_arguments=command_arguments,
                output_prefix=output_prefix,
                log_directory=log_directory,
            )
        )

    if not arguments.only_quantitative_step2 and not arguments.only_binary_step2:
        results.append(
            build_not_implemented_result(
                program_name="g_regenie2_binary_step1",
                trait_type="binary",
                step=1,
                device="cpu",
            )
        )
        results.append(
            build_not_implemented_result(
                program_name="g_regenie2_quantitative_step1",
                trait_type="quantitative",
                step=1,
                device="cpu",
            )
        )

    active_trait_type = "binary" if arguments.only_binary_step2 else "quantitative"
    active_trait_label = "binary" if active_trait_type == "binary" else "quantitative"
    g_output_cpu_prefix = arguments.output_dir / (
        "g_regenie2_binary_step2_cpu" if active_trait_type == "binary" else "g_regenie2_qt_step2_cpu"
    )
    g_cpu_command_arguments = build_g_step2_command(
        uv_executable=uv_executable,
        baseline_paths=baseline_paths,
        output_prefix=g_output_cpu_prefix,
        device="cpu",
        chunk_size=arguments.chunk_size,
        variant_limit=arguments.variant_limit,
        trait_type=active_trait_type,
    )
    results.append(
        run_g_step2(
            program_name=f"g_regenie2_{active_trait_label}_step2_cpu",
            trait_type=active_trait_type,
            device="cpu",
            command_arguments=g_cpu_command_arguments,
            output_prefix=g_output_cpu_prefix,
            log_directory=log_directory,
        )
    )

    run_gpu = arguments.include_gpu and not arguments.cpu_only
    if run_gpu:
        g_output_gpu_prefix = arguments.output_dir / (
            "g_regenie2_binary_step2_gpu" if active_trait_type == "binary" else "g_regenie2_qt_step2_gpu"
        )
        g_gpu_command_arguments = build_g_step2_command(
            uv_executable=uv_executable,
            baseline_paths=baseline_paths,
            output_prefix=g_output_gpu_prefix,
            device="gpu",
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            trait_type=active_trait_type,
        )
        results.append(
            run_g_step2(
                program_name=f"g_regenie2_{active_trait_label}_step2_gpu",
                trait_type=active_trait_type,
                device="gpu",
                command_arguments=g_gpu_command_arguments,
                output_prefix=g_output_gpu_prefix,
                log_directory=log_directory,
            )
        )
    else:
        results.append(
            ComparisonProgramResult(
                program_name=f"g_regenie2_{active_trait_label}_step2_gpu",
                implementation="g",
                trait_type=active_trait_type,
                step=2,
                device="gpu",
                status="not_implemented",
                wall_time_seconds=None,
                variants_per_second=None,
                peak_memory_megabytes=None,
                stdout_log_path=None,
                stderr_log_path=None,
                output_paths=[],
                output_row_count=None,
                prediction_list_present=None,
                notes="GPU run skipped (enable with --include-gpu).",
            )
        )

    if not arguments.only_quantitative_step2 and not arguments.only_binary_step2:
        g_binary_output_cpu_prefix = arguments.output_dir / "g_regenie2_binary_step2_cpu"
        g_binary_cpu_command_arguments = build_g_step2_command(
            uv_executable=uv_executable,
            baseline_paths=baseline_paths,
            output_prefix=g_binary_output_cpu_prefix,
            device="cpu",
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            trait_type="binary",
        )
        results.append(
            run_g_step2(
                program_name="g_regenie2_binary_step2_cpu",
                trait_type="binary",
                device="cpu",
                command_arguments=g_binary_cpu_command_arguments,
                output_prefix=g_binary_output_cpu_prefix,
                log_directory=log_directory,
            )
        )
        if run_gpu:
            g_binary_output_gpu_prefix = arguments.output_dir / "g_regenie2_binary_step2_gpu"
            g_binary_gpu_command_arguments = build_g_step2_command(
                uv_executable=uv_executable,
                baseline_paths=baseline_paths,
                output_prefix=g_binary_output_gpu_prefix,
                device="gpu",
                chunk_size=arguments.chunk_size,
                variant_limit=arguments.variant_limit,
                trait_type="binary",
            )
            results.append(
                run_g_step2(
                    program_name="g_regenie2_binary_step2_gpu",
                    trait_type="binary",
                    device="gpu",
                    command_arguments=g_binary_gpu_command_arguments,
                    output_prefix=g_binary_output_gpu_prefix,
                    log_directory=log_directory,
                )
            )
        else:
            results.append(
                ComparisonProgramResult(
                    program_name="g_regenie2_binary_step2_gpu",
                    implementation="g",
                    trait_type="binary",
                    step=2,
                    device="gpu",
                    status="not_implemented",
                    wall_time_seconds=None,
                    variants_per_second=None,
                    peak_memory_megabytes=None,
                    stdout_log_path=None,
                    stderr_log_path=None,
                    output_paths=[],
                    output_row_count=None,
                    prediction_list_present=None,
                    notes="GPU run skipped (enable with --include-gpu).",
                )
            )

    regenie_quantitative_result = extract_program(results, "regenie_step2_quantitative")
    regenie_binary_result = extract_program(results, "regenie_step2_binary")
    g_cpu_result = extract_program(results, "g_regenie2_quantitative_step2_cpu")
    g_gpu_result = extract_program(results, "g_regenie2_quantitative_step2_gpu")
    g_binary_cpu_result = extract_program(results, "g_regenie2_binary_step2_cpu")
    g_binary_gpu_result = extract_program(results, "g_regenie2_binary_step2_gpu")
    agreement_cpu = summarize_quantitative_step2_agreement(
        regenie_output_path=(
            result_output_path(regenie_quantitative_result)
            if regenie_quantitative_result is not None and regenie_quantitative_result.status == "success"
            else None
        ),
        g_output_path=(
            result_output_path(g_cpu_result) if g_cpu_result is not None and g_cpu_result.status == "success" else None
        ),
    )
    agreement_gpu = summarize_quantitative_step2_agreement(
        regenie_output_path=(
            result_output_path(regenie_quantitative_result)
            if regenie_quantitative_result is not None and regenie_quantitative_result.status == "success"
            else None
        ),
        g_output_path=(
            result_output_path(g_gpu_result) if g_gpu_result is not None and g_gpu_result.status == "success" else None
        ),
    )
    binary_agreement_cpu = summarize_binary_step2_agreement(
        regenie_output_path=(
            result_output_path(regenie_binary_result)
            if regenie_binary_result is not None and regenie_binary_result.status == "success"
            else None
        ),
        g_output_path=(
            result_output_path(g_binary_cpu_result)
            if g_binary_cpu_result is not None and g_binary_cpu_result.status == "success"
            else None
        ),
    )
    binary_agreement_gpu = summarize_binary_step2_agreement(
        regenie_output_path=(
            result_output_path(regenie_binary_result)
            if regenie_binary_result is not None and regenie_binary_result.status == "success"
            else None
        ),
        g_output_path=(
            result_output_path(g_binary_gpu_result)
            if g_binary_gpu_result is not None and g_binary_gpu_result.status == "success"
            else None
        ),
    )

    report_data: dict[str, typing.Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": asdict(baseline_benchmark.collect_hardware_summary()),
        "results": [asdict(result) for result in results],
        "comparisons": {
            "quantitative_step2": {
                "agreement_cpu": asdict(agreement_cpu),
                "agreement_gpu": asdict(agreement_gpu),
            },
            "binary_step2": {
                "agreement_cpu": asdict(binary_agreement_cpu),
                "agreement_gpu": asdict(binary_agreement_gpu),
            },
        },
    }
    if (
        regenie_quantitative_result is not None
        and regenie_quantitative_result.status == "success"
        and regenie_quantitative_result.wall_time_seconds is not None
    ):
        comparison_runtime: dict[str, typing.Any] = {}
        if g_cpu_result is not None and g_cpu_result.status == "success" and g_cpu_result.wall_time_seconds is not None:
            comparison_runtime["regenie_vs_g_cpu"] = {
                "speedup_ratio": regenie_quantitative_result.wall_time_seconds / g_cpu_result.wall_time_seconds,
                "absolute_delta_seconds": (
                    g_cpu_result.wall_time_seconds - regenie_quantitative_result.wall_time_seconds
                ),
            }
        if g_gpu_result is not None and g_gpu_result.status == "success" and g_gpu_result.wall_time_seconds is not None:
            comparison_runtime["regenie_vs_g_gpu"] = {
                "speedup_ratio": regenie_quantitative_result.wall_time_seconds / g_gpu_result.wall_time_seconds,
                "absolute_delta_seconds": (
                    g_gpu_result.wall_time_seconds - regenie_quantitative_result.wall_time_seconds
                ),
            }
        report_data["comparisons"]["quantitative_step2"]["runtime"] = comparison_runtime

    json_report_path = arguments.output_dir / "benchmark_report.json"
    json_report_path.write_text(f"{json.dumps(report_data, indent=2)}\n")
    text_report_path = arguments.output_dir / "benchmark_summary.txt"
    write_text_summary(
        report_path=text_report_path,
        results=results,
        agreement_cpu=agreement_cpu,
        agreement_gpu=agreement_gpu,
        binary_agreement_cpu=binary_agreement_cpu,
        binary_agreement_gpu=binary_agreement_gpu,
    )
    print(f"Wrote benchmark report: {json_report_path}")
    print(f"Wrote benchmark summary: {text_report_path}")


if __name__ == "__main__":
    main()
