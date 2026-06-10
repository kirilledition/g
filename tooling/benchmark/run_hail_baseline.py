#!/usr/bin/env python3
"""Run Hail association baselines and export benchmark-friendly outputs."""

from __future__ import annotations

import json
import time
import typing
from dataclasses import asdict, dataclass

import hail as hail_library  # type: ignore
import hydra

from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    from pathlib import Path

    import omegaconf

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


@dataclass(frozen=True)
class HailBaselineArguments:
    """Resolved parameters for one Hail baseline run.

    Attributes:
        bfile: PLINK dataset prefix without suffix.
        pheno: Phenotype table path.
        pheno_name: Phenotype column name.
        covar: Covariate table path.
        covar_names: Comma-separated covariate names.
        glm: Association model.
        logistic_test: Hail logistic test for logistic models.
        out: Output TSV path.
        log_path: Hail log path.
        matrix_table_cache: Optional MatrixTable cache path.
        cache_mode: MatrixTable cache mode.
        prepare_cache_only: Whether to import/cache without regression.

    """

    bfile: Path
    pheno: Path
    pheno_name: str
    covar: Path
    covar_names: str
    glm: str
    logistic_test: str
    out: Path
    log_path: Path
    matrix_table_cache: Path | None
    cache_mode: str
    prepare_cache_only: bool


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


def run_tool(arguments: HailBaselineArguments) -> None:
    """Execute one Hail baseline run and export its result table."""
    bed_prefix = arguments.bfile
    phenotype_path = arguments.pheno
    covariate_path = arguments.covar
    output_path = arguments.out
    log_path = arguments.log_path
    matrix_table_cache_path = arguments.matrix_table_cache
    covariate_names = tuple(name.strip() for name in arguments.covar_names.split(",") if name.strip())

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
        cache_mode=arguments.cache_mode,
    )
    if arguments.prepare_cache_only:
        hail_library.stop()
        run_report = HailRunReport(
            model_name=arguments.glm,
            test_name=None,
            sample_count=0,
            variant_count=0,
            cache_path=str(matrix_table_cache_path) if matrix_table_cache_path is not None else None,
            cache_mode=arguments.cache_mode,
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
        phenotype_name=arguments.pheno_name,
        covariate_names=covariate_names,
        is_binary_trait=arguments.glm == "logistic",
    )

    association_start_time = time.perf_counter()
    if arguments.glm == "linear":
        result_table = run_linear_baseline(matrix_table=matrix_table, covariate_names=covariate_names)
        test_name: str | None = None
    else:
        test_name = arguments.logistic_test
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
        model_name=arguments.glm,
        test_name=test_name,
        sample_count=sample_count,
        variant_count=variant_count,
        cache_path=str(matrix_table_cache_path) if matrix_table_cache_path is not None else None,
        cache_mode=arguments.cache_mode,
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


def required_path(tool_values: dict[str, typing.Any], key: str) -> Path:
    """Return a required path from a Hydra tool config."""
    path = tooling_hydra_arguments.path_or_none(tool_values[key])
    if path is None:
        message = f"tool.{key} is required."
        raise ValueError(message)
    return path


def build_arguments_from_config(config: omegaconf.DictConfig) -> HailBaselineArguments:
    """Resolve Hail baseline parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return HailBaselineArguments(
        bfile=required_path(tool_values, "bfile"),
        pheno=required_path(tool_values, "pheno"),
        pheno_name=str(tool_values["pheno_name"]),
        covar=required_path(tool_values, "covar"),
        covar_names=str(tool_values["covar_names"]),
        glm=str(tool_values["glm"]),
        logistic_test=str(tool_values["logistic_test"]),
        out=required_path(tool_values, "out"),
        log_path=required_path(tool_values, "log_path"),
        matrix_table_cache=tooling_hydra_arguments.path_or_none(tool_values["matrix_table_cache"]),
        cache_mode=str(tool_values["cache_mode"]),
        prepare_cache_only=tooling_hydra_arguments.boolean_value(tool_values["prepare_cache_only"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_hail_baseline")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run one Hail baseline from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run one Hail baseline from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
