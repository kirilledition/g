"""Option metadata for the REGENIE-compatible interface."""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass


class SupportLevel(enum.StrEnum):
    """Support status for a user-facing option."""

    SUPPORTED = "supported"
    RECOGNIZED_UNSUPPORTED = "recognized_unsupported"
    G_EXTENSION = "g_extension"
    DEPRECATED_ALIAS = "deprecated_alias"


@dataclass(frozen=True)
class OptionSpec:
    """Metadata for one CLI and TOML option.

    Attributes:
        name: User-facing option name without leading dashes.
        destination: Normalized internal destination name.
        support_level: Whether the option is supported by this engine.
        section: TOML section name.
        help_text: Concise user-facing description.

    """

    name: str
    destination: str
    support_level: SupportLevel
    section: str
    help_text: str


SUPPORTED_REGENIE_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec("step", "step", SupportLevel.SUPPORTED, "trait", "REGENIE analysis step."),
    OptionSpec("qt", "qt", SupportLevel.SUPPORTED, "trait", "Analyze quantitative traits."),
    OptionSpec("bt", "bt", SupportLevel.SUPPORTED, "trait", "Analyze binary traits."),
    OptionSpec("bgen", "bgen", SupportLevel.SUPPORTED, "input", "BGEN genotype file."),
    OptionSpec("sample", "sample", SupportLevel.SUPPORTED, "input", "BGEN sample file."),
    OptionSpec("phenoFile", "pheno_file", SupportLevel.SUPPORTED, "input", "Phenotype table."),
    OptionSpec("phenoCol", "pheno_col", SupportLevel.SUPPORTED, "input", "Phenotype column."),
    OptionSpec("phenoColList", "pheno_col_list", SupportLevel.SUPPORTED, "input", "Phenotype column list."),
    OptionSpec("covarFile", "covar_file", SupportLevel.SUPPORTED, "input", "Covariate table."),
    OptionSpec("covarCol", "covar_col", SupportLevel.SUPPORTED, "input", "Covariate column."),
    OptionSpec("covarColList", "covar_col_list", SupportLevel.SUPPORTED, "input", "Covariate column list."),
    OptionSpec("pred", "pred", SupportLevel.SUPPORTED, "input", "REGENIE step 1 prediction list."),
    OptionSpec("bsize", "bsize", SupportLevel.SUPPORTED, "trait", "Variants per processing block."),
    OptionSpec("threads", "threads", SupportLevel.SUPPORTED, "trait", "Requested CPU thread count."),
    OptionSpec("out", "out", SupportLevel.SUPPORTED, "output", "Output prefix."),
    OptionSpec("firth", "firth", SupportLevel.SUPPORTED, "binary", "Use Firth fallback."),
    OptionSpec("approx", "approx", SupportLevel.SUPPORTED, "binary", "Use approximate Firth fallback."),
    OptionSpec("pThresh", "p_threshold", SupportLevel.SUPPORTED, "binary", "Fallback p-value threshold."),
    OptionSpec("firth-se", "firth_se", SupportLevel.SUPPORTED, "binary", "Use Firth-derived standard errors."),
)

UNSUPPORTED_REGENIE_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec("bed", "bed", SupportLevel.RECOGNIZED_UNSUPPORTED, "input", "PLINK BED input is unsupported."),
    OptionSpec("pgen", "pgen", SupportLevel.RECOGNIZED_UNSUPPORTED, "input", "PLINK2 PGEN input is unsupported."),
    OptionSpec("keep", "keep", SupportLevel.RECOGNIZED_UNSUPPORTED, "filters", "Sample keep lists are unsupported."),
    OptionSpec(
        "remove",
        "remove",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Sample remove lists are unsupported.",
    ),
    OptionSpec(
        "extract",
        "extract",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Variant extract lists are unsupported.",
    ),
    OptionSpec(
        "exclude",
        "exclude",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "filters",
        "Variant exclude lists are unsupported.",
    ),
    OptionSpec(
        "catCovarList",
        "cat_covar_list",
        SupportLevel.RECOGNIZED_UNSUPPORTED,
        "input",
        "Categorical covariates are unsupported.",
    ),
    OptionSpec("test", "test", SupportLevel.RECOGNIZED_UNSUPPORTED, "trait", "Alternative tests are unsupported."),
    OptionSpec("t2e", "t2e", SupportLevel.RECOGNIZED_UNSUPPORTED, "trait", "Time-to-event traits are unsupported."),
    OptionSpec("spa", "spa", SupportLevel.RECOGNIZED_UNSUPPORTED, "binary", "SPA fallback is unsupported."),
)

G_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec("g-device", "device", SupportLevel.G_EXTENSION, "g.compute", "JAX execution device."),
    OptionSpec(
        "g-staging-depth",
        "staging_depth",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Native callback staging depth.",
    ),
    OptionSpec("g-variant-limit", "variant_limit", SupportLevel.G_EXTENSION, "g.compute", "Debug variant cap."),
    OptionSpec(
        "g-trusted-no-missing-diploid",
        "trusted_no_missing_diploid",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Use the trusted BGEN no-missing diploid fast path.",
    ),
    OptionSpec(
        "g-trusted-bgen-validation-mode",
        "trusted_bgen_validation_mode",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Trusted BGEN validation mode.",
    ),
    OptionSpec("g-sample-key-mode", "sample_key_mode", SupportLevel.G_EXTENSION, "g.compute", "Sample key mode."),
    OptionSpec(
        "g-allow-duplicate-iid-alignment",
        "allow_duplicate_iid_alignment",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Allow duplicate-IID alignment in IID sample-key mode.",
    ),
    OptionSpec("g-output-format", "format", SupportLevel.G_EXTENSION, "g.output", "Output materialization format."),
    OptionSpec(
        "g-output-run-directory",
        "output_run_directory",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Internal run directory.",
    ),
    OptionSpec(
        "g-writer-threads",
        "writer_threads",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Output writer thread count.",
    ),
    OptionSpec(
        "g-writer-queue-depth",
        "writer_queue_depth",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Output writer queue depth.",
    ),
    OptionSpec("g-resume", "resume", SupportLevel.G_EXTENSION, "g.output", "Resume a previous run."),
    OptionSpec("g-resume-mode", "resume_mode", SupportLevel.G_EXTENSION, "g.output", "Resume validation mode."),
    OptionSpec(
        "g-finalize-parquet",
        "finalize_parquet",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Finalize Parquet output.",
    ),
    OptionSpec("g-firth-batch-size", "firth_batch_size", SupportLevel.G_EXTENSION, "g.compute", "Firth batch size."),
    OptionSpec(
        "g-firth-candidate-capacity",
        "firth_candidate_capacity",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth candidate capacity.",
    ),
    OptionSpec(
        "g-binary-null-maximum-iterations",
        "binary_null_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum null-logistic iterations.",
    ),
    OptionSpec(
        "g-binary-null-coefficient-tolerance",
        "binary_null_coefficient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Null-logistic coefficient tolerance.",
    ),
    OptionSpec(
        "g-firth-maximum-iterations",
        "firth_maximum_iterations",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Maximum Firth iterations.",
    ),
    OptionSpec(
        "g-firth-gradient-tolerance",
        "firth_gradient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth gradient tolerance.",
    ),
    OptionSpec(
        "g-firth-coefficient-tolerance",
        "firth_coefficient_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth coefficient tolerance.",
    ),
    OptionSpec(
        "g-firth-likelihood-tolerance",
        "firth_likelihood_tolerance",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth likelihood tolerance.",
    ),
    OptionSpec(
        "g-firth-maximum-step-size",
        "firth_maximum_step_size",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Firth maximum step size.",
    ),
    OptionSpec(
        "g-use-block-firth-math",
        "use_block_firth_math",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Use block-oriented Firth math.",
    ),
    OptionSpec(
        "g-bgen-decode-tile-variant-count",
        "bgen_decode_tile_variant_count",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Native BGEN decode tile variant count.",
    ),
    OptionSpec("g-jax-cache-dir", "jax_cache_dir", SupportLevel.G_EXTENSION, "g.compute", "JAX cache directory."),
    OptionSpec(
        "g-jax-matmul-precision",
        "jax_matmul_precision",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "JAX matmul precision.",
    ),
    OptionSpec(
        "g-jax-persistent-cache",
        "jax_persistent_cache",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable the JAX persistent compilation cache.",
    ),
    OptionSpec(
        "g-jax-persistent-cache-min-entry-size-bytes",
        "jax_persistent_cache_min_entry_size_bytes",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Minimum JAX persistent-cache entry size.",
    ),
    OptionSpec(
        "g-jax-persistent-cache-min-compile-time-seconds",
        "jax_persistent_cache_min_compile_time_seconds",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Minimum JAX compile time for persistent-cache writes.",
    ),
    OptionSpec(
        "g-jax-xla-autotune-cache",
        "jax_xla_autotune_cache",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable node-local XLA autotune caches.",
    ),
    OptionSpec(
        "g-jax-transfer-guard",
        "jax_transfer_guard",
        SupportLevel.G_EXTENSION,
        "g.compute",
        "Enable JAX transfer-guard diagnostics.",
    ),
    OptionSpec(
        "g-output-chunks-per-arrow-file",
        "chunks_per_arrow_file",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Number of engine chunks grouped into one Arrow file.",
    ),
    OptionSpec(
        "g-output-arrow-compression",
        "arrow_compression",
        SupportLevel.G_EXTENSION,
        "g.output",
        "Arrow IPC compression for internal chunk files.",
    ),
    OptionSpec(
        "g-stage-timings-json",
        "stage_timings_json",
        SupportLevel.G_EXTENSION,
        "g.diagnostics",
        "Write stage timing diagnostics to JSON.",
    ),
)

OPTION_SPECS: tuple[OptionSpec, ...] = SUPPORTED_REGENIE_OPTIONS + UNSUPPORTED_REGENIE_OPTIONS + G_OPTIONS
OPTION_SPEC_BY_NAME: dict[str, OptionSpec] = {option_spec.name: option_spec for option_spec in OPTION_SPECS}


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
