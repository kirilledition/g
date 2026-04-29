#!/usr/bin/env python3
"""Run cached Hail linear and logistic baselines in one session."""

from __future__ import annotations

import argparse
import json
import time
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import hail as hail_library  # type: ignore

if typing.TYPE_CHECKING:
    from scripts.run_hail_baseline import (
        DEFAULT_HAIL_DRIVER_MEMORY,
        DEFAULT_HAIL_MASTER,
        load_or_prepare_matrix_table,
        prepare_matrix_table,
        run_linear_baseline,
        run_logistic_baseline,
    )
else:
    try:
        from scripts.run_hail_baseline import (
            DEFAULT_HAIL_DRIVER_MEMORY,
            DEFAULT_HAIL_MASTER,
            load_or_prepare_matrix_table,
            prepare_matrix_table,
            run_linear_baseline,
            run_logistic_baseline,
        )
    except ModuleNotFoundError:
        from run_hail_baseline import (
            DEFAULT_HAIL_DRIVER_MEMORY,
            DEFAULT_HAIL_MASTER,
            load_or_prepare_matrix_table,
            prepare_matrix_table,
            run_linear_baseline,
            run_logistic_baseline,
        )


@dataclass(frozen=True)
class HailSuiteStepReport:
    """Execution metadata for one analysis step inside the Hail suite."""

    output_name: str
    model_name: str
    test_name: str | None
    sample_count: int
    variant_count: int
    duration_seconds: float
    output_path: str


@dataclass(frozen=True)
class HailSuiteReport:
    """Execution metadata for a full cached Hail suite run."""

    cache_path: str | None
    cache_mode: str
    cache_used: bool
    cache_refreshed: bool
    cache_prepare_seconds: float
    total_duration_seconds: float
    log_path: str
    hail_version: str
    step_reports: list[HailSuiteStepReport]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Hail suite."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--bfile", required=True, help="PLINK dataset prefix without suffix.")
    argument_parser.add_argument("--covar", required=True, help="Covariate table path.")
    argument_parser.add_argument("--covar-names", required=True, help="Comma-separated covariate names.")
    argument_parser.add_argument("--continuous-pheno", required=True, help="Continuous phenotype table path.")
    argument_parser.add_argument("--continuous-pheno-name", required=True, help="Continuous phenotype column name.")
    argument_parser.add_argument("--binary-pheno", required=True, help="Binary phenotype table path.")
    argument_parser.add_argument("--binary-pheno-name", required=True, help="Binary phenotype column name.")
    argument_parser.add_argument("--linear-out", required=True, help="Linear output TSV path.")
    argument_parser.add_argument("--wald-out", required=True, help="Logistic Wald output TSV path.")
    argument_parser.add_argument("--firth-out", required=True, help="Logistic Firth output TSV path.")
    argument_parser.add_argument("--log-path", required=True, help="Hail log path.")
    argument_parser.add_argument("--report-path", required=True, help="Suite JSON report path.")
    argument_parser.add_argument("--matrix-table-cache", help="Optional MatrixTable cache path.")
    argument_parser.add_argument(
        "--cache-mode",
        choices=("reuse", "refresh", "require", "disable"),
        default="reuse",
        help="How to use the optional MatrixTable cache.",
    )
    return argument_parser


def finalize_step_report(
    step_start_time: float,
    output_name: str,
    output_path: Path,
    result_table: hail_library.Table,
    model_name: str,
    test_name: str | None,
    sample_count: int,
) -> HailSuiteStepReport:
    """Export one result table and measure its full step duration."""
    variant_count = result_table.count()
    result_table.export(str(output_path))
    return HailSuiteStepReport(
        output_name=output_name,
        model_name=model_name,
        test_name=test_name,
        sample_count=sample_count,
        variant_count=variant_count,
        duration_seconds=time.perf_counter() - step_start_time,
        output_path=str(output_path),
    )


def main() -> None:
    """Run the cached Hail suite and persist a structured JSON report."""
    command_line_arguments = build_argument_parser().parse_args()
    bed_prefix = Path(command_line_arguments.bfile)
    covariate_path = Path(command_line_arguments.covar)
    continuous_phenotype_path = Path(command_line_arguments.continuous_pheno)
    binary_phenotype_path = Path(command_line_arguments.binary_pheno)
    linear_output_path = Path(command_line_arguments.linear_out)
    wald_output_path = Path(command_line_arguments.wald_out)
    firth_output_path = Path(command_line_arguments.firth_out)
    log_path = Path(command_line_arguments.log_path)
    report_path = Path(command_line_arguments.report_path)
    matrix_table_cache_path = (
        Path(command_line_arguments.matrix_table_cache)
        if command_line_arguments.matrix_table_cache is not None
        else None
    )
    covariate_names = tuple(name.strip() for name in command_line_arguments.covar_names.split(",") if name.strip())

    for path in [linear_output_path, wald_output_path, firth_output_path, log_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    hail_library.init(
        log=str(log_path),
        master=DEFAULT_HAIL_MASTER,
        spark_conf={"spark.driver.memory": DEFAULT_HAIL_DRIVER_MEMORY},
    )
    suite_start_time = time.perf_counter()
    matrix_table_load_result = load_or_prepare_matrix_table(
        bed_prefix=bed_prefix,
        matrix_table_cache_path=matrix_table_cache_path,
        cache_mode=command_line_arguments.cache_mode,
    )
    base_matrix_table = matrix_table_load_result.matrix_table.persist()

    linear_matrix_table = prepare_matrix_table(
        matrix_table=base_matrix_table,
        phenotype_path=continuous_phenotype_path,
        covariate_path=covariate_path,
        phenotype_name=command_line_arguments.continuous_pheno_name,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    linear_sample_count = linear_matrix_table.count_cols()
    linear_step_start_time = time.perf_counter()
    linear_result_table = run_linear_baseline(linear_matrix_table, covariate_names)
    linear_step_report = finalize_step_report(
        step_start_time=linear_step_start_time,
        output_name="hail_cont",
        output_path=linear_output_path,
        result_table=linear_result_table,
        model_name="linear",
        test_name=None,
        sample_count=linear_sample_count,
    )

    binary_matrix_table = prepare_matrix_table(
        matrix_table=base_matrix_table,
        phenotype_path=binary_phenotype_path,
        covariate_path=covariate_path,
        phenotype_name=command_line_arguments.binary_pheno_name,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    binary_sample_count = binary_matrix_table.count_cols()
    logistic_wald_step_start_time = time.perf_counter()
    logistic_wald_result_table = run_logistic_baseline(
        matrix_table=binary_matrix_table,
        covariate_names=covariate_names,
        test_name="wald",
        sample_count=binary_sample_count,
    )
    logistic_wald_step_report = finalize_step_report(
        step_start_time=logistic_wald_step_start_time,
        output_name="hail_bin_wald",
        output_path=wald_output_path,
        result_table=logistic_wald_result_table,
        model_name="logistic",
        test_name="wald",
        sample_count=binary_sample_count,
    )
    logistic_firth_step_start_time = time.perf_counter()
    logistic_firth_result_table = run_logistic_baseline(
        matrix_table=binary_matrix_table,
        covariate_names=covariate_names,
        test_name="firth",
        sample_count=binary_sample_count,
    )
    logistic_firth_step_report = finalize_step_report(
        step_start_time=logistic_firth_step_start_time,
        output_name="hail_bin_firth",
        output_path=firth_output_path,
        result_table=logistic_firth_result_table,
        model_name="logistic",
        test_name="firth",
        sample_count=binary_sample_count,
    )

    step_reports = [linear_step_report, logistic_wald_step_report, logistic_firth_step_report]

    suite_report = HailSuiteReport(
        cache_path=str(matrix_table_cache_path) if matrix_table_cache_path is not None else None,
        cache_mode=command_line_arguments.cache_mode,
        cache_used=matrix_table_load_result.cache_used,
        cache_refreshed=matrix_table_load_result.cache_refreshed,
        cache_prepare_seconds=matrix_table_load_result.cache_prepare_seconds,
        total_duration_seconds=time.perf_counter() - suite_start_time,
        log_path=str(log_path),
        hail_version=hail_library.__version__,
        step_reports=step_reports,
    )
    report_path.write_text(f"{json.dumps(asdict(suite_report), indent=2)}\n")
    hail_library.stop()
    print(report_path)


if __name__ == "__main__":
    main()
