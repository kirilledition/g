#!/usr/bin/env python3
"""Run cached Hail linear and logistic baselines in one session."""

from __future__ import annotations

import json
import time
import typing
from dataclasses import asdict, dataclass

import hail as hail_library  # type: ignore
import hydra

from tooling.benchmark import run_hail_baseline
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    from pathlib import Path

    import omegaconf


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


@dataclass(frozen=True)
class HailSuiteArguments:
    """Resolved parameters for the cached Hail benchmark suite.

    Attributes:
        bfile: PLINK dataset prefix without suffix.
        covar: Covariate table path.
        covar_names: Comma-separated covariate names.
        continuous_pheno: Continuous phenotype table path.
        continuous_pheno_name: Continuous phenotype column name.
        binary_pheno: Binary phenotype table path.
        binary_pheno_name: Binary phenotype column name.
        linear_out: Linear output TSV path.
        wald_out: Logistic Wald output TSV path.
        firth_out: Logistic Firth output TSV path.
        log_path: Hail log path.
        report_path: Suite JSON report path.
        matrix_table_cache: Optional MatrixTable cache path.
        cache_mode: MatrixTable cache mode.

    """

    bfile: Path
    covar: Path
    covar_names: str
    continuous_pheno: Path
    continuous_pheno_name: str
    binary_pheno: Path
    binary_pheno_name: str
    linear_out: Path
    wald_out: Path
    firth_out: Path
    log_path: Path
    report_path: Path
    matrix_table_cache: Path | None
    cache_mode: str


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


def run_tool(arguments: HailSuiteArguments) -> None:
    """Run the cached Hail suite and persist a structured JSON report."""
    bed_prefix = arguments.bfile
    covariate_path = arguments.covar
    continuous_phenotype_path = arguments.continuous_pheno
    binary_phenotype_path = arguments.binary_pheno
    linear_output_path = arguments.linear_out
    wald_output_path = arguments.wald_out
    firth_output_path = arguments.firth_out
    log_path = arguments.log_path
    report_path = arguments.report_path
    matrix_table_cache_path = arguments.matrix_table_cache
    covariate_names = tuple(name.strip() for name in arguments.covar_names.split(",") if name.strip())

    for path in [linear_output_path, wald_output_path, firth_output_path, log_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    hail_library.init(
        log=str(log_path),
        master=run_hail_baseline.DEFAULT_HAIL_MASTER,
        spark_conf={"spark.driver.memory": run_hail_baseline.DEFAULT_HAIL_DRIVER_MEMORY},
    )
    suite_start_time = time.perf_counter()
    matrix_table_load_result = run_hail_baseline.load_or_prepare_matrix_table(
        bed_prefix=bed_prefix,
        matrix_table_cache_path=matrix_table_cache_path,
        cache_mode=arguments.cache_mode,
    )
    base_matrix_table = matrix_table_load_result.matrix_table.persist()

    linear_matrix_table = run_hail_baseline.prepare_matrix_table(
        matrix_table=base_matrix_table,
        phenotype_path=continuous_phenotype_path,
        covariate_path=covariate_path,
        phenotype_name=arguments.continuous_pheno_name,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    linear_sample_count = linear_matrix_table.count_cols()
    linear_step_start_time = time.perf_counter()
    linear_result_table = run_hail_baseline.run_linear_baseline(linear_matrix_table, covariate_names)
    linear_step_report = finalize_step_report(
        step_start_time=linear_step_start_time,
        output_name="hail_cont",
        output_path=linear_output_path,
        result_table=linear_result_table,
        model_name="linear",
        test_name=None,
        sample_count=linear_sample_count,
    )

    binary_matrix_table = run_hail_baseline.prepare_matrix_table(
        matrix_table=base_matrix_table,
        phenotype_path=binary_phenotype_path,
        covariate_path=covariate_path,
        phenotype_name=arguments.binary_pheno_name,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    binary_sample_count = binary_matrix_table.count_cols()
    logistic_wald_step_start_time = time.perf_counter()
    logistic_wald_result_table = run_hail_baseline.run_logistic_baseline(
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
    logistic_firth_result_table = run_hail_baseline.run_logistic_baseline(
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
        cache_mode=arguments.cache_mode,
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


def required_path(tool_values: dict[str, typing.Any], key: str) -> Path:
    """Return a required path from a Hydra tool config."""
    path = tooling_hydra_arguments.path_or_none(tool_values[key])
    if path is None:
        message = f"tool.{key} is required."
        raise ValueError(message)
    return path


def build_arguments_from_config(config: omegaconf.DictConfig) -> HailSuiteArguments:
    """Resolve Hail suite parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return HailSuiteArguments(
        bfile=required_path(tool_values, "bfile"),
        covar=required_path(tool_values, "covar"),
        covar_names=str(tool_values["covar_names"]),
        continuous_pheno=required_path(tool_values, "continuous_pheno"),
        continuous_pheno_name=str(tool_values["continuous_pheno_name"]),
        binary_pheno=required_path(tool_values, "binary_pheno"),
        binary_pheno_name=str(tool_values["binary_pheno_name"]),
        linear_out=required_path(tool_values, "linear_out"),
        wald_out=required_path(tool_values, "wald_out"),
        firth_out=required_path(tool_values, "firth_out"),
        log_path=required_path(tool_values, "log_path"),
        report_path=required_path(tool_values, "report_path"),
        matrix_table_cache=tooling_hydra_arguments.path_or_none(tool_values["matrix_table_cache"]),
        cache_mode=str(tool_values["cache_mode"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_hail_suite")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the cached Hail suite from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the cached Hail suite from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
