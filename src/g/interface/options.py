"""Option metadata for the REGENIE-compatible interface."""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass

from g import types


class SupportLevel(enum.StrEnum):
    """Support status for a user-facing option."""

    SUPPORTED = "supported"
    RECOGNIZED_UNSUPPORTED = "recognized_unsupported"
    G_EXTENSION = "g_extension"
    DEPRECATED_ALIAS = "deprecated_alias"


class OptionValueType(enum.StrEnum):
    """Value domain for a user-facing option."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PATH = "path"


@dataclass(frozen=True)
class OptionSpec:
    """Metadata for one CLI and TOML option.

    Attributes:
        name: User-facing option name without leading dashes.
        destination: Python destination name used by Click and Python callers.
        section: TOML section name.
        support_level: Whether the option is supported by this engine.
        help_text: Concise user-facing description.
        cli_flags: Click-style CLI flags.
        type: Scalar value kind.
        default: Unspecified CLI default value.
        multiple: Whether the option may be repeated.
        is_flag: Whether the option is a boolean flag.
        accepted_values: Accepted string values for choice options.

    """

    name: str
    destination: str
    support_level: SupportLevel
    section: str
    help_text: str
    cli_flags: tuple[str, ...] = ()
    type: OptionValueType = OptionValueType.STRING
    default: object | None = None
    multiple: bool = False
    is_flag: bool = False
    accepted_values: tuple[str, ...] = ()


DEVICE_VALUES = tuple(item.value for item in types.Device)
TRUSTED_BGEN_VALIDATION_MODE_VALUES = tuple(item.value for item in types.TrustedBgenValidationMode)
SAMPLE_KEY_MODE_VALUES = tuple(item.value for item in types.SampleKeyMode)
MULTI_PHENOTYPE_SAMPLE_MODE_VALUES = tuple(item.value for item in types.MultiPhenotypeSampleMode)
NULL_LOGISTIC_NONCONVERGENCE_POLICY_VALUES = tuple(item.value for item in types.NullLogisticNonconvergencePolicy)
OUTPUT_FORMAT_VALUES = tuple(item.value for item in types.OutputFormat)
RESUME_MODE_VALUES = tuple(item.value for item in types.ResumeMode)
JAX_MATMUL_PRECISION_VALUES = tuple(item.value for item in types.JaxMatmulPrecision)
FLOATING_POINT_DTYPE_VALUES = tuple(item.value for item in types.FloatingPointDtype)
ARROW_COMPRESSION_VALUES = tuple(item.value for item in types.ArrowCompression)
TELEMETRY_MODE_VALUES = tuple(item.value for item in types.TelemetryMode)


SUPPORTED_REGENIE_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        "step",
        "step",
        SupportLevel.SUPPORTED,
        "trait",
        "REGENIE analysis step. Only step 2 is supported.",
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "qt",
        "qt",
        SupportLevel.SUPPORTED,
        "trait",
        "Analyze quantitative traits.",
        cli_flags=("--qt/--no-qt", "qt"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "bt",
        "bt",
        SupportLevel.SUPPORTED,
        "trait",
        "Analyze binary traits.",
        cli_flags=("--bt/--no-bt", "bt"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec("bgen", "bgen", SupportLevel.SUPPORTED, "input", "BGEN genotype file.", type=OptionValueType.PATH),
    OptionSpec(
        "sample",
        "sample",
        SupportLevel.SUPPORTED,
        "input",
        "BGEN sample file.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "phenoFile",
        "pheno_file",
        SupportLevel.SUPPORTED,
        "input",
        "Phenotype table.",
        cli_flags=("--phenoFile", "pheno_file"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "phenoCol",
        "pheno_col",
        SupportLevel.SUPPORTED,
        "input",
        "Phenotype column.",
        cli_flags=("--phenoCol", "pheno_col"),
        multiple=True,
    ),
    OptionSpec(
        "phenoColList",
        "pheno_col_list",
        SupportLevel.SUPPORTED,
        "input",
        "Phenotype column list.",
        cli_flags=("--phenoColList", "pheno_col_list"),
    ),
    OptionSpec(
        "covarFile",
        "covar_file",
        SupportLevel.SUPPORTED,
        "input",
        "Covariate table.",
        cli_flags=("--covarFile", "covar_file"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "covarCol",
        "covar_col",
        SupportLevel.SUPPORTED,
        "input",
        "Covariate column.",
        cli_flags=("--covarCol", "covar_col"),
        multiple=True,
    ),
    OptionSpec(
        "covarColList",
        "covar_col_list",
        SupportLevel.SUPPORTED,
        "input",
        "Covariate column list.",
        cli_flags=("--covarColList", "covar_col_list"),
    ),
    OptionSpec(
        "pred",
        "pred",
        SupportLevel.SUPPORTED,
        "input",
        "REGENIE step 1 prediction list.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "bsize",
        "bsize",
        SupportLevel.SUPPORTED,
        "trait",
        "Variants per processing block.",
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "threads",
        "threads",
        SupportLevel.SUPPORTED,
        "trait",
        "Requested CPU thread count.",
        type=OptionValueType.INTEGER,
    ),
    OptionSpec("out", "out", SupportLevel.SUPPORTED, "output", "Output prefix.", type=OptionValueType.PATH),
    OptionSpec(
        "firth",
        "firth",
        SupportLevel.SUPPORTED,
        "binary",
        "Use Firth fallback.",
        cli_flags=("--firth/--no-firth", "firth"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "approx",
        "approx",
        SupportLevel.SUPPORTED,
        "binary",
        "Use approximate Firth fallback.",
        cli_flags=("--approx/--no-approx", "approx"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "pThresh",
        "p_threshold",
        SupportLevel.SUPPORTED,
        "binary",
        "Fallback p-value threshold.",
        cli_flags=("--pThresh", "p_threshold"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "firth-se",
        "firth_se",
        SupportLevel.SUPPORTED,
        "binary",
        "Use Firth-derived standard errors.",
        cli_flags=("--firth-se/--no-firth-se", "firth_se"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
)

UNSUPPORTED_REGENIE_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        "bed",
        "bed",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "input",
        "PLINK BED input is unsupported.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "pgen",
        "pgen",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "input",
        "PLINK2 PGEN input is unsupported.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "keep",
        "keep",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Sample keep lists are unsupported.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "remove",
        "remove",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Sample remove lists are unsupported.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "extract",
        "extract",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Variant extract lists are unsupported.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "exclude",
        "exclude",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Variant exclude lists are unsupported.",
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "catCovarList",
        "cat_covar_list",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "input",
        "Categorical covariates are unsupported.",
        cli_flags=("--catCovarList", "cat_covar_list"),
    ),
    OptionSpec("test", "test", SupportLevel.RECOGNIZED_UNSUPPORTED, "trait", "Alternative tests are unsupported."),
    OptionSpec(
        "t2e",
        "t2e",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "trait",
        "Time-to-event traits are unsupported.",
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "spa",
        "spa",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "binary",
        "SPA fallback is unsupported.",
        cli_flags=("--spa/--no-spa", "spa"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
)

G_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        "g-device",
        "g_device",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "JAX execution device.",
        cli_flags=("--g-device", "g_device"),
        accepted_values=DEVICE_VALUES,
    ),
    OptionSpec(
        "g-staging-depth",
        "g_staging_depth",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Positive native callback staging depth.",
        cli_flags=("--g-staging-depth", "g_staging_depth"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-variant-limit",
        "g_variant_limit",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Debug variant cap.",
        cli_flags=("--g-variant-limit", "g_variant_limit"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-trusted-no-missing-diploid",
        "g_trusted_no_missing_diploid",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Use the trusted BGEN no-missing diploid fast path.",
        cli_flags=("--g-trusted-no-missing-diploid/--no-g-trusted-no-missing-diploid", "g_trusted_no_missing_diploid"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-trusted-bgen-validation-mode",
        "g_trusted_bgen_validation_mode",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Trusted BGEN validation mode.",
        cli_flags=("--g-trusted-bgen-validation-mode", "g_trusted_bgen_validation_mode"),
        accepted_values=TRUSTED_BGEN_VALIDATION_MODE_VALUES,
    ),
    OptionSpec(
        "g-sample-key-mode",
        "g_sample_key_mode",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Sample key mode.",
        cli_flags=("--g-sample-key-mode", "g_sample_key_mode"),
        accepted_values=SAMPLE_KEY_MODE_VALUES,
    ),
    OptionSpec(
        "g-multi-phenotype-sample-mode",
        "g_multi_phenotype_sample_mode",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Multi-phenotype sample mode. complete-case batches phenotypes on the shared complete-case "
        "intersection and is not equivalent to separate single-phenotype REGENIE runs.",
        cli_flags=("--g-multi-phenotype-sample-mode", "g_multi_phenotype_sample_mode"),
        accepted_values=MULTI_PHENOTYPE_SAMPLE_MODE_VALUES,
    ),
    OptionSpec(
        "g-output-format",
        "g_output_format",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Output materialization format.",
        cli_flags=("--g-output-format", "g_output_format"),
        accepted_values=OUTPUT_FORMAT_VALUES,
    ),
    OptionSpec(
        "g-output-run-directory",
        "g_output_run_directory",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Internal run directory.",
        cli_flags=("--g-output-run-directory", "g_output_run_directory"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-writer-threads",
        "g_writer_threads",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Output writer thread count.",
        cli_flags=("--g-writer-threads", "g_writer_threads"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-writer-queue-depth",
        "g_writer_queue_depth",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Output writer queue depth.",
        cli_flags=("--g-writer-queue-depth", "g_writer_queue_depth"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-resume",
        "g_resume",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Resume a previous run.",
        cli_flags=("--g-resume/--no-g-resume", "g_resume"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-resume-mode",
        "g_resume_mode",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Resume validation mode.",
        cli_flags=("--g-resume-mode", "g_resume_mode"),
        accepted_values=RESUME_MODE_VALUES,
    ),
    OptionSpec(
        "g-finalize-parquet",
        "g_finalize_parquet",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Finalize Parquet output.",
        cli_flags=("--g-finalize-parquet/--no-g-finalize-parquet", "g_finalize_parquet"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-firth-batch-size",
        "g_firth_batch_size",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth batch size.",
        cli_flags=("--g-firth-batch-size", "g_firth_batch_size"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-candidate-capacity",
        "g_firth_candidate_capacity",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth candidate capacity.",
        cli_flags=("--g-firth-candidate-capacity", "g_firth_candidate_capacity"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-binary-null-maximum-iterations",
        "g_binary_null_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum null-logistic iterations.",
        cli_flags=("--g-binary-null-maximum-iterations", "g_binary_null_maximum_iterations"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-binary-null-coefficient-tolerance",
        "g_binary_null_coefficient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Null-logistic coefficient tolerance.",
        cli_flags=("--g-binary-null-coefficient-tolerance", "g_binary_null_coefficient_tolerance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-null-logistic-nonconvergence",
        "g_null_logistic_nonconvergence",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Policy when a binary null-logistic chromosome fit does not converge.",
        cli_flags=("--g-null-logistic-nonconvergence", "g_null_logistic_nonconvergence"),
        accepted_values=NULL_LOGISTIC_NONCONVERGENCE_POLICY_VALUES,
    ),
    OptionSpec(
        "g-binary-minimum-probability",
        "g_binary_minimum_probability",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Binary logistic probability clipping floor.",
        cli_flags=("--g-binary-minimum-probability", "g_binary_minimum_probability"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-binary-minimum-variance",
        "g_binary_minimum_variance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Binary variance and information-matrix numerical floor.",
        cli_flags=("--g-binary-minimum-variance", "g_binary_minimum_variance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-binary-relative-variance-tolerance",
        "g_binary_relative_variance_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Relative score-test variance floor multiplier.",
        cli_flags=("--g-binary-relative-variance-tolerance", "g_binary_relative_variance_tolerance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-maximum-iterations",
        "g_firth_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum Firth iterations.",
        cli_flags=("--g-firth-maximum-iterations", "g_firth_maximum_iterations"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-gradient-tolerance",
        "g_firth_gradient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth gradient tolerance.",
        cli_flags=("--g-firth-gradient-tolerance", "g_firth_gradient_tolerance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-coefficient-tolerance",
        "g_firth_coefficient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth coefficient tolerance.",
        cli_flags=("--g-firth-coefficient-tolerance", "g_firth_coefficient_tolerance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-likelihood-tolerance",
        "g_firth_likelihood_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth likelihood tolerance.",
        cli_flags=("--g-firth-likelihood-tolerance", "g_firth_likelihood_tolerance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-maximum-step-size",
        "g_firth_maximum_step_size",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth maximum step size.",
        cli_flags=("--g-firth-maximum-step-size", "g_firth_maximum_step_size"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-pseudo-maximum-iterations",
        "g_firth_pseudo_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum outer pseudo-Firth iterations.",
        cli_flags=("--g-firth-pseudo-maximum-iterations", "g_firth_pseudo_maximum_iterations"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-pseudo-inner-maximum-iterations",
        "g_firth_pseudo_inner_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum inner pseudo-Firth iterations.",
        cli_flags=("--g-firth-pseudo-inner-maximum-iterations", "g_firth_pseudo_inner_maximum_iterations"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-newton-raphson-zero-start-iterations",
        "g_firth_newton_raphson_zero_start_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum zero-start scalar Firth Newton-Raphson iterations.",
        cli_flags=(
            "--g-firth-newton-raphson-zero-start-iterations",
            "g_firth_newton_raphson_zero_start_iterations",
        ),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-line-search-maximum-attempts",
        "g_firth_line_search_maximum_attempts",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum scalar Firth line-search attempts.",
        cli_flags=("--g-firth-line-search-maximum-attempts", "g_firth_line_search_maximum_attempts"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-step-halving-maximum-attempts",
        "g_firth_step_halving_maximum_attempts",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum Firth step-halving attempts.",
        cli_flags=("--g-firth-step-halving-maximum-attempts", "g_firth_step_halving_maximum_attempts"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-firth-initial-response-scale",
        "g_firth_initial_response_scale",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Pseudo-response scale for block full-model Firth initialization.",
        cli_flags=("--g-firth-initial-response-scale", "g_firth_initial_response_scale"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-sparse-carrier-dosage-threshold",
        "g_firth_sparse_carrier_dosage_threshold",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Raw dosage threshold for sparse carrier-only Firth samples.",
        cli_flags=("--g-firth-sparse-carrier-dosage-threshold", "g_firth_sparse_carrier_dosage_threshold"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-firth-step-halving-scale",
        "g_firth_step_halving_scale",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Full-model Firth step multiplier after each rejected backtracking attempt.",
        cli_flags=("--g-firth-step-halving-scale", "g_firth_step_halving_scale"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-null-firth-maximum-iterations",
        "g_null_firth_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum covariate-only null Firth iterations.",
        cli_flags=("--g-null-firth-maximum-iterations", "g_null_firth_maximum_iterations"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-null-firth-gradient-tolerance",
        "g_null_firth_gradient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Covariate-only null Firth gradient tolerance.",
        cli_flags=("--g-null-firth-gradient-tolerance", "g_null_firth_gradient_tolerance"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-null-firth-maximum-step-size",
        "g_null_firth_maximum_step_size",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Covariate-only null Firth maximum step size.",
        cli_flags=("--g-null-firth-maximum-step-size", "g_null_firth_maximum_step_size"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-null-firth-fallback-iteration-multiplier",
        "g_null_firth_fallback_iteration_multiplier",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Covariate-only null Firth fallback iteration multiplier.",
        cli_flags=(
            "--g-null-firth-fallback-iteration-multiplier",
            "g_null_firth_fallback_iteration_multiplier",
        ),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-null-firth-fallback-step-divisor",
        "g_null_firth_fallback_step_divisor",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Covariate-only null Firth fallback step divisor.",
        cli_flags=("--g-null-firth-fallback-step-divisor", "g_null_firth_fallback_step_divisor"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-null-firth-line-search-maximum-attempts",
        "g_null_firth_line_search_maximum_attempts",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum covariate-only null Firth line-search attempts.",
        cli_flags=("--g-null-firth-line-search-maximum-attempts", "g_null_firth_line_search_maximum_attempts"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-null-firth-step-halving-scale",
        "g_null_firth_step_halving_scale",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Null Firth step multiplier after each rejected line-search attempt.",
        cli_flags=("--g-null-firth-step-halving-scale", "g_null_firth_step_halving_scale"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-use-block-firth-math",
        "g_use_block_firth_math",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Use block-oriented Firth math.",
        cli_flags=("--g-use-block-firth-math/--no-g-use-block-firth-math", "g_use_block_firth_math"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-bgen-decode-tile-variant-count",
        "g_bgen_decode_tile_variant_count",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Native BGEN decode tile variant count.",
        cli_flags=("--g-bgen-decode-tile-variant-count", "g_bgen_decode_tile_variant_count"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-score-dtype",
        "g_score_dtype",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Floating-point dtype for score-test kernels.",
        cli_flags=("--g-score-dtype", "g_score_dtype"),
        accepted_values=FLOATING_POINT_DTYPE_VALUES,
    ),
    OptionSpec(
        "g-firth-dtype",
        "g_firth_dtype",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Floating-point dtype for Firth internals. Only float64 is currently supported.",
        cli_flags=("--g-firth-dtype", "g_firth_dtype"),
        accepted_values=FLOATING_POINT_DTYPE_VALUES,
    ),
    OptionSpec(
        "g-jax-cache-dir",
        "g_jax_cache_dir",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "JAX cache directory.",
        cli_flags=("--g-jax-cache-dir", "g_jax_cache_dir"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-jax-matmul-precision",
        "g_jax_matmul_precision",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "JAX matmul precision.",
        cli_flags=("--g-jax-matmul-precision", "g_jax_matmul_precision"),
        accepted_values=JAX_MATMUL_PRECISION_VALUES,
    ),
    OptionSpec(
        "g-jax-enable-x64",
        "g_jax_enable_x64",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable JAX to create x64 arrays; score-test dtype is controlled by g-score-dtype.",
        cli_flags=("--g-jax-enable-x64/--no-g-jax-enable-x64", "g_jax_enable_x64"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-jax-persistent-cache",
        "g_jax_persistent_cache",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable the JAX persistent compilation cache.",
        cli_flags=("--g-jax-persistent-cache/--no-g-jax-persistent-cache", "g_jax_persistent_cache"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-jax-persistent-cache-min-entry-size-bytes",
        "g_jax_persistent_cache_min_entry_size_bytes",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Minimum JAX persistent-cache entry size.",
        cli_flags=("--g-jax-persistent-cache-min-entry-size-bytes", "g_jax_persistent_cache_min_entry_size_bytes"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-jax-persistent-cache-min-compile-time-seconds",
        "g_jax_persistent_cache_min_compile_time_seconds",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Minimum JAX compile time for persistent-cache writes.",
        cli_flags=(
            "--g-jax-persistent-cache-min-compile-time-seconds",
            "g_jax_persistent_cache_min_compile_time_seconds",
        ),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-jax-xla-autotune-cache",
        "g_jax_xla_autotune_cache",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable node-local XLA autotune caches.",
        cli_flags=("--g-jax-xla-autotune-cache/--no-g-jax-xla-autotune-cache", "g_jax_xla_autotune_cache"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-jax-transfer-guard",
        "g_jax_transfer_guard",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable JAX transfer-guard diagnostics.",
        cli_flags=("--g-jax-transfer-guard/--no-g-jax-transfer-guard", "g_jax_transfer_guard"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-output-chunks-per-arrow-file",
        "g_output_chunks_per_arrow_file",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Number of engine chunks grouped into one Arrow file.",
        cli_flags=("--g-output-chunks-per-arrow-file", "g_output_chunks_per_arrow_file"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-output-arrow-compression",
        "g_output_arrow_compression",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Arrow IPC compression for internal chunk files.",
        cli_flags=("--g-output-arrow-compression", "g_output_arrow_compression"),
        accepted_values=ARROW_COMPRESSION_VALUES,
    ),
    OptionSpec(
        "g-telemetry",
        "g_telemetry",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Run telemetry mode.",
        cli_flags=("--g-telemetry", "g_telemetry"),
        accepted_values=TELEMETRY_MODE_VALUES,
    ),
    OptionSpec(
        "g-log-dir",
        "g_log_dir",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Directory for run telemetry logs.",
        cli_flags=("--g-log-dir", "g_log_dir"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-stage-timings-json",
        "g_stage_timings_json",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Write stage timing diagnostics to JSON.",
        cli_flags=("--g-stage-timings-json", "g_stage_timings_json"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-log-filter",
        "g_log_filter",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Rust/Python tracing filter.",
        cli_flags=("--g-log-filter", "g_log_filter"),
    ),
    OptionSpec(
        "g-log-file",
        "g_log_file",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Write Rust JSONL tracing logs to this file.",
        cli_flags=("--g-log-file", "g_log_file"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-log-stderr",
        "g_log_stderr",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Write compact tracing logs to stderr.",
        cli_flags=("--g-log-stderr/--no-g-log-stderr", "g_log_stderr"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-progress-interval-seconds",
        "g_progress_interval_seconds",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Minimum seconds between progress telemetry events.",
        cli_flags=("--g-progress-interval-seconds", "g_progress_interval_seconds"),
        type=OptionValueType.FLOAT,
    ),
    OptionSpec(
        "g-progress-interval-chunks",
        "g_progress_interval_chunks",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Minimum chunks between progress telemetry events.",
        cli_flags=("--g-progress-interval-chunks", "g_progress_interval_chunks"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-profile-summary-json",
        "g_profile_summary_json",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Write aggregate profile summary JSON.",
        cli_flags=("--g-profile-summary-json", "g_profile_summary_json"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-trace-file",
        "g_trace_file",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Write high-volume Rust trace JSONL events to this file.",
        cli_flags=("--g-trace-file", "g_trace_file"),
        type=OptionValueType.PATH,
    ),
    OptionSpec(
        "g-trace-filter",
        "g_trace_filter",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Rust tracing filter for the trace file.",
        cli_flags=("--g-trace-filter", "g_trace_filter"),
    ),
    OptionSpec(
        "g-log-queue-size",
        "g_log_queue_size",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Buffered line count for non-blocking log writers.",
        cli_flags=("--g-log-queue-size", "g_log_queue_size"),
        type=OptionValueType.INTEGER,
    ),
    OptionSpec(
        "g-log-lossy",
        "g_log_lossy",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Drop logs instead of blocking when the log queue is full.",
        cli_flags=("--g-log-lossy/--no-g-log-lossy", "g_log_lossy"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-include-source-location",
        "g_include_source_location",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Include source file and line in JSON tracing events.",
        cli_flags=("--g-include-source-location/--no-g-include-source-location", "g_include_source_location"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
    OptionSpec(
        "g-include-span-events",
        "g_include_span_events",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Include tracing span lifecycle events.",
        cli_flags=("--g-include-span-events/--no-g-include-span-events", "g_include_span_events"),
        type=OptionValueType.BOOLEAN,
        is_flag=True,
    ),
)

OPTION_SPECS: tuple[OptionSpec, ...] = SUPPORTED_REGENIE_OPTIONS + UNSUPPORTED_REGENIE_OPTIONS + G_OPTIONS
OPTION_SPEC_BY_NAME: dict[str, OptionSpec] = {option_spec.name: option_spec for option_spec in OPTION_SPECS}
OPTION_SPEC_BY_DESTINATION: dict[str, OptionSpec] = {
    option_spec.destination: option_spec for option_spec in OPTION_SPECS
}


def supported_option_names() -> frozenset[str]:
    """Return all supported REGENIE-compatible option names."""
    return frozenset(
        option_spec.name
        for option_spec in OPTION_SPECS
        if option_spec.support_level in {SupportLevel.SUPPORTED, SupportLevel.G_EXTENSION}
    )


def unsupported_option_names() -> frozenset[str]:
    """Return recognized but unsupported REGENIE option names."""
    return frozenset(
        option_spec.name
        for option_spec in OPTION_SPECS
        if option_spec.support_level == SupportLevel.RECOGNIZED_UNSUPPORTED
    )


def explain_option(name: str) -> str:
    """Return a concise explanation for an option."""
    option_spec = OPTION_SPEC_BY_NAME[name]
    return f"{option_spec.name}: {option_spec.support_level.value}. {option_spec.help_text}"


def iter_explanations() -> typing.Iterator[str]:
    """Yield explanations for all known options."""
    for option_spec in OPTION_SPECS:
        yield explain_option(option_spec.name)
