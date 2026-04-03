#!/usr/bin/env python3
"""Run Hail association baselines and export benchmark-friendly outputs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import hail as hail_library  # type: ignore

DEFAULT_HAIL_MASTER = "local[1]"
DEFAULT_HAIL_DRIVER_MEMORY = "8g"


@dataclass(frozen=True)
class HailRunReport:
    """Structured metadata for one exported Hail baseline run."""

    model_name: str
    test_name: str | None
    sample_count: int
    variant_count: int
    cache_path: str | None
    cache_mode: str
    cache_used: bool
    cache_refreshed: bool
    cache_prepare_seconds: float
    association_seconds: float
    duration_seconds: float
    output_path: str
    log_path: str
    hail_version: str


@dataclass(frozen=True)
class MatrixTableLoadResult:
    """Result of loading or preparing a base MatrixTable cache.

    Attributes:
        matrix_table: The loaded Hail MatrixTable.
        cache_used: Whether an existing cache was used.
        cache_refreshed: Whether the cache was refreshed.
        cache_prepare_seconds: Time spent preparing the cache.

    """

    matrix_table: hail_library.MatrixTable
    cache_used: bool
    cache_refreshed: bool
    cache_prepare_seconds: float


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Hail baseline runs."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--bfile", required=True, help="PLINK dataset prefix without suffix.")
    argument_parser.add_argument("--pheno", required=True, help="Phenotype table path.")
    argument_parser.add_argument("--pheno-name", required=True, help="Phenotype column name.")
    argument_parser.add_argument("--covar", required=True, help="Covariate table path.")
    argument_parser.add_argument("--covar-names", required=True, help="Comma-separated covariate names.")
    argument_parser.add_argument("--glm", required=True, choices=("linear", "logistic"), help="Association model.")
    argument_parser.add_argument(
        "--logistic-test",
        choices=("wald", "firth"),
        default="wald",
        help="Hail logistic test to run when --glm=logistic.",
    )
    argument_parser.add_argument("--out", required=True, help="Output TSV path.")
    argument_parser.add_argument("--log-path", required=True, help="Hail log path.")
    argument_parser.add_argument(
        "--matrix-table-cache",
        help="Optional MatrixTable cache path stored under local data/.",
    )
    argument_parser.add_argument(
        "--cache-mode",
        choices=("reuse", "refresh", "require", "disable"),
        default="reuse",
        help="How to use the optional MatrixTable cache.",
    )
    argument_parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help="Only import and cache the MatrixTable without running a regression.",
    )
    return argument_parser


def import_keyed_table(table_path: Path) -> hail_library.Table:
    """Import a sample-aligned TSV and key it by IID."""
    return hail_library.import_table(str(table_path), impute=True).key_by("IID")


def import_plink_matrix_table(bed_prefix: Path) -> hail_library.MatrixTable:
    """Import a PLINK dataset into Hail."""
    return hail_library.import_plink(
        bed=str(bed_prefix.with_suffix(".bed")),
        bim=str(bed_prefix.with_suffix(".bim")),
        fam=str(bed_prefix.with_suffix(".fam")),
        a2_reference=True,
    )


def load_or_prepare_matrix_table(
    bed_prefix: Path,
    matrix_table_cache_path: Path | None,
    cache_mode: str,
) -> MatrixTableLoadResult:
    """Load the base MatrixTable from cache or import and cache it.

    Returns:
        Matrix table, whether cache was used, whether cache was refreshed, and cache preparation seconds.

    Raises:
        FileNotFoundError: The cache is required but missing.

    """
    if cache_mode == "disable" or matrix_table_cache_path is None:
        preparation_start_time = time.perf_counter()
        matrix_table = import_plink_matrix_table(bed_prefix)
        return MatrixTableLoadResult(
            matrix_table=matrix_table,
            cache_used=False,
            cache_refreshed=False,
            cache_prepare_seconds=time.perf_counter() - preparation_start_time,
        )

    if cache_mode == "require":
        if not matrix_table_cache_path.exists():
            raise FileNotFoundError(f"Required Hail MatrixTable cache is missing: {matrix_table_cache_path}")
        preparation_start_time = time.perf_counter()
        matrix_table = hail_library.read_matrix_table(str(matrix_table_cache_path))
        return MatrixTableLoadResult(
            matrix_table=matrix_table,
            cache_used=True,
            cache_refreshed=False,
            cache_prepare_seconds=time.perf_counter() - preparation_start_time,
        )

    if cache_mode == "reuse" and matrix_table_cache_path.exists():
        preparation_start_time = time.perf_counter()
        matrix_table = hail_library.read_matrix_table(str(matrix_table_cache_path))
        return MatrixTableLoadResult(
            matrix_table=matrix_table,
            cache_used=True,
            cache_refreshed=False,
            cache_prepare_seconds=time.perf_counter() - preparation_start_time,
        )

    preparation_start_time = time.perf_counter()
    matrix_table = import_plink_matrix_table(bed_prefix)
    matrix_table_cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_table.write(str(matrix_table_cache_path), overwrite=True)
    cached_matrix_table = hail_library.read_matrix_table(str(matrix_table_cache_path))
    return MatrixTableLoadResult(
        matrix_table=cached_matrix_table,
        cache_used=True,
        cache_refreshed=True,
        cache_prepare_seconds=time.perf_counter() - preparation_start_time,
    )


def build_covariate_expressions(
    matrix_table: hail_library.MatrixTable,
    covariate_names: tuple[str, ...],
) -> list[hail_library.expr.expressions.Float64Expression]:
    """Build numeric covariate expressions including the intercept."""
    covariate_expressions: list[hail_library.expr.expressions.Float64Expression] = [hail_library.float64(1.0)]
    for covariate_name in covariate_names:
        covariate_expressions.append(hail_library.float64(matrix_table.covariates[covariate_name]))
    return covariate_expressions


def prepare_matrix_table(
    matrix_table: hail_library.MatrixTable,
    phenotype_path: Path,
    covariate_path: Path,
    phenotype_name: str,
    covariate_names: tuple[str, ...],
    *,
    is_binary_trait: bool,
) -> hail_library.MatrixTable:
    """Attach phenotype and covariate annotations to a base MatrixTable."""
    phenotype_table = import_keyed_table(phenotype_path)
    covariate_table = import_keyed_table(covariate_path)
    matrix_table = matrix_table.annotate_cols(
        phenotype=phenotype_table[matrix_table.s],
        covariates=covariate_table[matrix_table.s],
    )

    required_covariates_are_defined = True
    required_covariate_mask = hail_library.literal(required_covariates_are_defined)
    for covariate_name in covariate_names:
        required_covariate_mask = required_covariate_mask & hail_library.is_defined(
            matrix_table.covariates[covariate_name]
        )
    matrix_table = matrix_table.filter_cols(
        hail_library.is_defined(matrix_table.phenotype)
        & hail_library.is_defined(matrix_table.covariates)
        & hail_library.is_defined(matrix_table.phenotype[phenotype_name])
        & required_covariate_mask,
    )
    if is_binary_trait:
        matrix_table = matrix_table.annotate_cols(
            analysis_phenotype=hail_library.if_else(
                hail_library.int32(matrix_table.phenotype[phenotype_name]) == 2,
                hail_library.float64(1.0),
                hail_library.float64(0.0),
            )
        )
    else:
        matrix_table = matrix_table.annotate_cols(
            analysis_phenotype=hail_library.float64(matrix_table.phenotype[phenotype_name]),
        )
    return matrix_table


def build_row_metadata_table(result_table: hail_library.Table) -> hail_library.Table:
    """Attach shared row metadata fields used by downstream comparisons."""
    return result_table.annotate(
        chromosome=result_table.locus.contig,
        position=result_table.locus.position,
        variant_identifier=result_table.rsid,
        allele_one=result_table.alleles[1],
        allele_two=result_table.alleles[0],
    )


def run_linear_baseline(
    matrix_table: hail_library.MatrixTable,
    covariate_names: tuple[str, ...],
) -> hail_library.Table:
    """Run Hail linear regression and standardize the output schema."""
    result_table = hail_library.linear_regression_rows(
        y=matrix_table.analysis_phenotype,
        x=hail_library.float64(matrix_table.GT.n_alt_alleles()),
        covariates=build_covariate_expressions(matrix_table, covariate_names),
        pass_through=["rsid"],
    )
    result_table = build_row_metadata_table(result_table)
    return result_table.select(
        "chromosome",
        "position",
        "variant_identifier",
        "allele_one",
        "allele_two",
        observation_count=result_table.n,
        beta=result_table.beta,
        standard_error=result_table.standard_error,
        t_statistic=result_table.t_stat,
        z_statistic=hail_library.missing(hail_library.tfloat64),
        chi_squared_statistic=hail_library.missing(hail_library.tfloat64),
        p_value=result_table.p_value,
        fit_converged=hail_library.missing(hail_library.tbool),
        fit_exploded=hail_library.missing(hail_library.tbool),
        fit_iteration_count=hail_library.missing(hail_library.tint32),
        hail_test=hail_library.str("linear"),
    )


def run_logistic_baseline(
    matrix_table: hail_library.MatrixTable,
    covariate_names: tuple[str, ...],
    test_name: str,
    sample_count: int,
) -> hail_library.Table:
    """Run Hail logistic regression and standardize the output schema."""
    logistic_kwargs: dict[str, float | int | str | list[hail_library.expr.expressions.Float64Expression]] = {
        "test": test_name,
        "y": matrix_table.analysis_phenotype,
        "x": hail_library.float64(matrix_table.GT.n_alt_alleles()),
        "covariates": build_covariate_expressions(matrix_table, covariate_names),
        "pass_through": ["rsid"],
    }
    if test_name == "wald":
        logistic_kwargs["max_iterations"] = 50
        logistic_kwargs["tolerance"] = 1.0e-8

    result_table = hail_library.logistic_regression_rows(**logistic_kwargs)
    result_table = build_row_metadata_table(result_table)
    if test_name == "wald":
        return result_table.select(
            "chromosome",
            "position",
            "variant_identifier",
            "allele_one",
            "allele_two",
            observation_count=hail_library.int32(sample_count),
            beta=result_table.beta,
            standard_error=result_table.standard_error,
            t_statistic=hail_library.missing(hail_library.tfloat64),
            z_statistic=result_table.z_stat,
            chi_squared_statistic=hail_library.missing(hail_library.tfloat64),
            p_value=result_table.p_value,
            fit_converged=result_table.fit.converged,
            fit_exploded=result_table.fit.exploded,
            fit_iteration_count=result_table.fit.n_iterations,
            hail_test=hail_library.str(test_name),
        )

    return result_table.select(
        "chromosome",
        "position",
        "variant_identifier",
        "allele_one",
        "allele_two",
        observation_count=hail_library.int32(sample_count),
        beta=result_table.beta,
        standard_error=hail_library.missing(hail_library.tfloat64),
        t_statistic=hail_library.missing(hail_library.tfloat64),
        z_statistic=hail_library.missing(hail_library.tfloat64),
        chi_squared_statistic=result_table.chi_sq_stat,
        p_value=result_table.p_value,
        fit_converged=result_table.fit.converged,
        fit_exploded=result_table.fit.exploded,
        fit_iteration_count=result_table.fit.n_iterations,
        hail_test=hail_library.str(test_name),
    )


def main() -> None:
    """Execute one Hail baseline run and export its result table."""
    command_line_arguments = build_argument_parser().parse_args()
    bed_prefix = Path(command_line_arguments.bfile)
    phenotype_path = Path(command_line_arguments.pheno)
    covariate_path = Path(command_line_arguments.covar)
    output_path = Path(command_line_arguments.out)
    log_path = Path(command_line_arguments.log_path)
    matrix_table_cache_path = (
        Path(command_line_arguments.matrix_table_cache)
        if command_line_arguments.matrix_table_cache is not None
        else None
    )
    covariate_names = tuple(name.strip() for name in command_line_arguments.covar_names.split(",") if name.strip())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    hail_library.init(
        log=str(log_path),
        master=DEFAULT_HAIL_MASTER,
        spark_conf={"spark.driver.memory": DEFAULT_HAIL_DRIVER_MEMORY},
    )
    start_time = time.perf_counter()
    matrix_table_load_result = load_or_prepare_matrix_table(
        bed_prefix=bed_prefix,
        matrix_table_cache_path=matrix_table_cache_path,
        cache_mode=command_line_arguments.cache_mode,
    )
    if command_line_arguments.prepare_cache_only:
        hail_library.stop()
        run_report = HailRunReport(
            model_name=command_line_arguments.glm,
            test_name=None,
            sample_count=0,
            variant_count=0,
            cache_path=str(matrix_table_cache_path) if matrix_table_cache_path is not None else None,
            cache_mode=command_line_arguments.cache_mode,
            cache_used=matrix_table_load_result.cache_used,
            cache_refreshed=matrix_table_load_result.cache_refreshed,
            cache_prepare_seconds=matrix_table_load_result.cache_prepare_seconds,
            association_seconds=0.0,
            duration_seconds=time.perf_counter() - start_time,
            output_path=str(output_path),
            log_path=str(log_path),
            hail_version=hail_library.__version__,
        )
        print(json.dumps(asdict(run_report), indent=2))
        return

    matrix_table = prepare_matrix_table(
        matrix_table=matrix_table_load_result.matrix_table,
        phenotype_path=phenotype_path,
        covariate_path=covariate_path,
        phenotype_name=command_line_arguments.pheno_name,
        covariate_names=covariate_names,
        is_binary_trait=command_line_arguments.glm == "logistic",
    )

    association_start_time = time.perf_counter()
    if command_line_arguments.glm == "linear":
        result_table = run_linear_baseline(matrix_table=matrix_table, covariate_names=covariate_names)
        test_name: str | None = None
    else:
        test_name = command_line_arguments.logistic_test
        sample_count = matrix_table.count_cols()
        result_table = run_logistic_baseline(
            matrix_table=matrix_table,
            covariate_names=covariate_names,
            test_name=test_name,
            sample_count=sample_count,
        )
    association_seconds = time.perf_counter() - association_start_time

    sample_count = matrix_table.count_cols()
    variant_count = result_table.count()
    result_table.export(str(output_path))
    duration_seconds = time.perf_counter() - start_time
    hail_library.stop()

    run_report = HailRunReport(
        model_name=command_line_arguments.glm,
        test_name=test_name,
        sample_count=sample_count,
        variant_count=variant_count,
        cache_path=str(matrix_table_cache_path) if matrix_table_cache_path is not None else None,
        cache_mode=command_line_arguments.cache_mode,
        cache_used=matrix_table_load_result.cache_used,
        cache_refreshed=matrix_table_load_result.cache_refreshed,
        cache_prepare_seconds=matrix_table_load_result.cache_prepare_seconds,
        association_seconds=association_seconds,
        duration_seconds=duration_seconds,
        output_path=str(output_path),
        log_path=str(log_path),
        hail_version=hail_library.__version__,
    )
    print(json.dumps(asdict(run_report), indent=2))


if __name__ == "__main__":
    main()
