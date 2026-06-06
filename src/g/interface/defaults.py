"""Packaged default configuration catalog."""

from __future__ import annotations

import functools
import hashlib
import importlib.resources
import json
import typing
from dataclasses import dataclass

import msgspec

from g import types
from g.interface import config_layers, options, toml_schema

DEFAULT_CONFIG_RESOURCE = "config.default.toml"
OPTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DefaultOptionCatalog:
    """Validated packaged defaults.

    Attributes:
        toml_config: Typed packaged TOML config.
        raw_toml: Packaged TOML dictionary.
        normalized_options: Defaults keyed by canonical option name.
        default_config_hash: Stable hash of the packaged defaults.

    """

    toml_config: toml_schema.TomlConfig
    raw_toml: typing.Mapping[str, typing.Any]
    normalized_options: typing.Mapping[str, typing.Any]
    default_config_hash: str


@dataclass(frozen=True)
class TraitRuntimeDefaults:
    """Packaged trait defaults converted to runtime types.

    Attributes:
        step: REGENIE step.
        trait_type: Default trait family.
        bsize: Variant chunk size.

    """

    step: int
    trait_type: types.RegenieTraitType
    bsize: int


@dataclass(frozen=True)
class BinaryRuntimeDefaults:
    """Packaged binary-trait defaults converted to runtime types.

    Attributes:
        firth: Whether Firth fallback is enabled.
        approx: Whether approximate Firth fallback is enabled.
        p_threshold: Score-test p-value threshold for fallback.
        firth_se: Whether successful Firth rows use LRT-derived standard errors.

    """

    firth: bool
    approx: bool
    p_threshold: float
    firth_se: bool


@dataclass(frozen=True)
class GComputeRuntimeDefaults:
    """Packaged engine compute defaults converted to runtime types.

    Attributes:
        device: Default JAX execution device.
        staging_depth: Native callback staging depth.
        trusted_no_missing_diploid: Trusted BGEN no-missing diploid flag.
        trusted_bgen_validation_mode: Trusted BGEN validation policy.
        sample_key_mode: Phenotype/sample key matching mode.
        multi_phenotype_sample_mode: Multi-phenotype sample handling mode.
        firth_batch_size: Fixed Firth fallback batch size.
        firth_candidate_capacity: Preferred Firth candidate capacity.
        binary_null_maximum_iterations: Maximum null-logistic IRLS iterations.
        binary_null_coefficient_tolerance: Null-logistic coefficient tolerance.
        null_logistic_nonconvergence_policy: Null-logistic non-convergence policy.
        binary_minimum_probability: Binary probability clipping floor.
        binary_minimum_variance: Binary variance floor.
        binary_relative_variance_tolerance: Relative score-test variance floor.
        firth_maximum_iterations: Maximum Firth solver iterations.
        firth_gradient_tolerance: Firth gradient tolerance.
        firth_coefficient_tolerance: Firth coefficient tolerance.
        firth_likelihood_tolerance: Firth likelihood tolerance.
        firth_maximum_step_size: Firth maximum step size.
        firth_pseudo_maximum_iterations: Pseudo-Firth maximum iterations.
        firth_pseudo_inner_maximum_iterations: Pseudo-Firth inner maximum iterations.
        firth_newton_raphson_zero_start_iterations: Zero-start Newton-Raphson iterations.
        firth_line_search_maximum_attempts: Firth line-search attempts.
        firth_step_halving_maximum_attempts: Firth step-halving attempts.
        firth_initial_response_scale: Firth initial response scale.
        firth_sparse_carrier_dosage_threshold: Sparse-carrier dosage threshold.
        firth_step_halving_scale: Firth step-halving scale.
        null_firth_maximum_iterations: Null-Firth maximum iterations.
        null_firth_gradient_tolerance: Null-Firth gradient tolerance.
        null_firth_maximum_step_size: Null-Firth maximum step size.
        null_firth_fallback_iteration_multiplier: Null-Firth fallback iteration multiplier.
        null_firth_fallback_step_divisor: Null-Firth fallback step divisor.
        null_firth_line_search_maximum_attempts: Null-Firth line-search attempts.
        null_firth_step_halving_scale: Null-Firth step-halving scale.
        use_block_firth_math: Whether block Firth math is enabled.
        bgen_decode_tile_variant_count: Native BGEN decode tile size.
        gpu_genotype_format: GPU genotype transfer format.
        score_dtype: Score-test floating-point dtype.
        firth_dtype: Firth floating-point dtype.
        jax_persistent_cache: Whether JAX persistent compilation cache is enabled.
        jax_persistent_cache_min_entry_size_bytes: Minimum persistent-cache entry size.
        jax_persistent_cache_min_compile_time_seconds: Minimum persistent-cache compile time.
        jax_xla_autotune_cache: Whether XLA autotune cache is enabled.
        jax_transfer_guard: Whether JAX transfer guard diagnostics are enabled.

    """

    device: types.Device
    staging_depth: int
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    sample_key_mode: types.SampleKeyMode
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode
    firth_batch_size: int
    firth_candidate_capacity: int
    binary_null_maximum_iterations: int
    binary_null_coefficient_tolerance: float
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy
    binary_minimum_probability: float
    binary_minimum_variance: float
    binary_relative_variance_tolerance: float
    firth_maximum_iterations: int
    firth_gradient_tolerance: float
    firth_coefficient_tolerance: float
    firth_likelihood_tolerance: float
    firth_maximum_step_size: float
    firth_pseudo_maximum_iterations: int
    firth_pseudo_inner_maximum_iterations: int
    firth_newton_raphson_zero_start_iterations: int
    firth_line_search_maximum_attempts: int
    firth_step_halving_maximum_attempts: int
    firth_initial_response_scale: float
    firth_sparse_carrier_dosage_threshold: float
    firth_step_halving_scale: float
    null_firth_maximum_iterations: int
    null_firth_gradient_tolerance: float
    null_firth_maximum_step_size: float
    null_firth_fallback_iteration_multiplier: int
    null_firth_fallback_step_divisor: float
    null_firth_line_search_maximum_attempts: int
    null_firth_step_halving_scale: float
    use_block_firth_math: bool
    bgen_decode_tile_variant_count: int
    gpu_genotype_format: types.GpuGenotypeFormat
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    jax_persistent_cache: bool
    jax_persistent_cache_min_entry_size_bytes: int
    jax_persistent_cache_min_compile_time_seconds: int
    jax_xla_autotune_cache: bool
    jax_transfer_guard: bool


@dataclass(frozen=True)
class GOutputRuntimeDefaults:
    """Packaged engine output defaults converted to runtime types.

    Attributes:
        format: Output materialization format.
        writer_threads: Background writer thread count.
        writer_queue_depth: Background writer queue depth.
        chunks_per_arrow_file: Arrow chunks per finalized Parquet part.
        arrow_compression: Arrow IPC compression codec.
        parquet_compression: Parquet compression codec.
        resume: Whether resume mode is enabled by default.
        resume_mode: Resume validation mode.
        finalize_parquet: Whether final Parquet materialization is enabled.

    """

    format: types.OutputFormat
    writer_threads: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression
    parquet_compression: types.ParquetCompression
    resume: bool
    resume_mode: types.ResumeMode
    finalize_parquet: bool


@dataclass(frozen=True)
class GDiagnosticsRuntimeDefaults:
    """Packaged engine diagnostics defaults converted to runtime types.

    Attributes:
        telemetry: Telemetry mode.
        log_filter: Native logging filter.
        log_stderr: Whether logs are emitted to stderr.
        progress_interval_seconds: Progress heartbeat interval in seconds.
        progress_interval_chunks: Progress heartbeat interval in chunks.
        trace_filter: Native tracing filter.
        log_queue_size: Native logging queue size.
        log_lossy: Whether native logging may drop records under pressure.
        include_source_location: Whether traces include source location.
        include_span_events: Whether traces include span lifecycle events.

    """

    telemetry: types.TelemetryMode
    log_filter: str
    log_stderr: bool
    progress_interval_seconds: float
    progress_interval_chunks: int
    trace_filter: str
    log_queue_size: int
    log_lossy: bool
    include_source_location: bool
    include_span_events: bool


@dataclass(frozen=True)
class PackagedRuntimeDefaults:
    """Packaged defaults grouped by runtime config section.

    Attributes:
        trait: Trait-family defaults.
        binary: Binary-trait defaults.
        g_compute: Engine compute defaults.
        g_output: Engine output defaults.
        g_diagnostics: Engine diagnostics defaults.

    """

    trait: TraitRuntimeDefaults
    binary: BinaryRuntimeDefaults
    g_compute: GComputeRuntimeDefaults
    g_output: GOutputRuntimeDefaults
    g_diagnostics: GDiagnosticsRuntimeDefaults


@functools.cache
def load_default_option_catalog() -> DefaultOptionCatalog:
    """Load, normalize, validate, and hash packaged default options."""
    default_toml_bytes = load_default_toml_bytes()
    toml_config = config_layers.decode_toml_bytes(default_toml_bytes, source=DEFAULT_CONFIG_RESOURCE)
    raw_toml = config_layers.decode_toml_builtin_mapping(default_toml_bytes, source=DEFAULT_CONFIG_RESOURCE)
    normalized_options = normalize_default_toml(raw_toml)
    validate_default_catalog(normalized_options)
    return DefaultOptionCatalog(
        toml_config=toml_config,
        raw_toml=raw_toml,
        normalized_options=normalized_options,
        default_config_hash=build_default_config_hash(raw_toml),
    )


@functools.cache
def load_packaged_runtime_defaults() -> PackagedRuntimeDefaults:
    """Load packaged defaults converted to runtime-facing types."""
    toml_config = load_default_option_catalog().toml_config
    trait_section = required_default_section(toml_config.trait, "trait")
    binary_section = required_default_section(toml_config.binary, "binary")
    g_section = required_default_section(toml_config.g, "g")
    g_compute_section = required_default_section(g_section.compute, "g.compute")
    g_output_section = required_default_section(g_section.output, "g.output")
    g_diagnostics_section = required_default_section(g_section.diagnostics, "g.diagnostics")
    return PackagedRuntimeDefaults(
        trait=build_trait_runtime_defaults(trait_section),
        binary=build_binary_runtime_defaults(binary_section),
        g_compute=build_g_compute_runtime_defaults(g_compute_section),
        g_output=build_g_output_runtime_defaults(g_output_section),
        g_diagnostics=build_g_diagnostics_runtime_defaults(g_diagnostics_section),
    )


def load_default_toml_config() -> toml_schema.TomlConfig:
    """Load the packaged default TOML file into the typed schema."""
    return config_layers.decode_toml_bytes(
        load_default_toml_bytes(),
        source=DEFAULT_CONFIG_RESOURCE,
    )


def load_default_toml_bytes() -> bytes:
    """Load packaged default TOML file bytes."""
    default_config_resource = importlib.resources.files("g").joinpath(DEFAULT_CONFIG_RESOURCE)
    return default_config_resource.read_bytes()


def load_raw_default_toml() -> dict[str, typing.Any]:
    """Load the packaged default TOML file."""
    return dict(load_default_option_catalog().raw_toml)


def normalize_default_toml(raw_toml: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize default TOML paths to canonical option names."""
    normalized_options: dict[str, typing.Any] = {}
    for canonical_name, option_value in config_layers.flatten_toml_mapping(raw_toml).items():
        if canonical_name in normalized_options:
            message = f"Default config contains duplicate default for {canonical_name!r}."
            raise ValueError(message)
        normalized_options[canonical_name] = option_value
    return normalized_options


def flatten_toml_options(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML sections into canonical option names where possible."""
    return config_layers.flatten_toml_mapping(raw_options)


def flatten_g_toml_section(raw_g_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML tables below the reserved [g.*] namespace."""
    return config_layers.flatten_g_toml_section(raw_g_options)


def flatten_toml_section(section_name: str, section_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten one TOML section through the option registry."""
    return config_layers.flatten_toml_section(section_name, section_options)


def required_default_section[TomlStructT: msgspec.Struct](
    section: TomlStructT | msgspec.UnsetType,
    section_name: str,
) -> TomlStructT:
    """Return a required packaged default TOML section."""
    if section is msgspec.UNSET:
        message = f"Default config is missing required section [{section_name}]."
        raise ValueError(message)
    return section


def required_default_value[DefaultValueT](
    raw_value: DefaultValueT | msgspec.UnsetType,
    option_name: str,
) -> DefaultValueT:
    """Return a required packaged default TOML value."""
    if raw_value is msgspec.UNSET:
        message = f"Default config is missing required default option {option_name!r}."
        raise ValueError(message)
    return raw_value


def build_trait_runtime_defaults(trait_section: toml_schema.TraitToml) -> TraitRuntimeDefaults:
    """Build runtime trait defaults from the packaged TOML section."""
    return TraitRuntimeDefaults(
        step=required_default_value(trait_section.step, "step"),
        trait_type=resolve_packaged_trait_type(trait_section),
        bsize=required_default_value(trait_section.bsize, "bsize"),
    )


def resolve_packaged_trait_type(trait_section: toml_schema.TraitToml) -> types.RegenieTraitType:
    """Resolve the packaged trait default flags into one trait family."""
    quantitative_trait_default = required_default_value(trait_section.qt, "qt")
    binary_trait_default = required_default_value(trait_section.bt, "bt")
    if quantitative_trait_default and binary_trait_default:
        message = "Default config has mutually exclusive qt/bt defaults."
        raise ValueError(message)
    if binary_trait_default:
        return types.RegenieTraitType.BINARY
    return types.RegenieTraitType.QUANTITATIVE


def build_binary_runtime_defaults(binary_section: toml_schema.BinaryToml) -> BinaryRuntimeDefaults:
    """Build runtime binary-trait defaults from the packaged TOML section."""
    return BinaryRuntimeDefaults(
        firth=required_default_value(binary_section.firth, "firth"),
        approx=required_default_value(binary_section.approx, "approx"),
        p_threshold=required_default_value(binary_section.p_threshold, "pThresh"),
        firth_se=required_default_value(binary_section.firth_se, "firth-se"),
    )


def build_g_compute_runtime_defaults(g_compute_section: toml_schema.GComputeToml) -> GComputeRuntimeDefaults:
    """Build runtime engine compute defaults from the packaged TOML section."""
    return GComputeRuntimeDefaults(
        device=types.Device(required_default_value(g_compute_section.device, "g-device")),
        staging_depth=required_default_value(g_compute_section.staging_depth, "g-staging-depth"),
        trusted_no_missing_diploid=required_default_value(
            g_compute_section.trusted_no_missing_diploid,
            "g-trusted-no-missing-diploid",
        ),
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode(
            required_default_value(
                g_compute_section.trusted_bgen_validation_mode,
                "g-trusted-bgen-validation-mode",
            )
        ),
        sample_key_mode=types.SampleKeyMode(
            required_default_value(g_compute_section.sample_key_mode, "g-sample-key-mode")
        ),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode(
            required_default_value(
                g_compute_section.multi_phenotype_sample_mode,
                "g-multi-phenotype-sample-mode",
            )
        ),
        firth_batch_size=required_default_value(g_compute_section.firth_batch_size, "g-firth-batch-size"),
        firth_candidate_capacity=required_default_value(
            g_compute_section.firth_candidate_capacity,
            "g-firth-candidate-capacity",
        ),
        binary_null_maximum_iterations=required_default_value(
            g_compute_section.binary_null_maximum_iterations,
            "g-binary-null-maximum-iterations",
        ),
        binary_null_coefficient_tolerance=required_default_value(
            g_compute_section.binary_null_coefficient_tolerance,
            "g-binary-null-coefficient-tolerance",
        ),
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy(
            required_default_value(
                g_compute_section.null_logistic_nonconvergence,
                "g-null-logistic-nonconvergence",
            )
        ),
        binary_minimum_probability=required_default_value(
            g_compute_section.binary_minimum_probability,
            "g-binary-minimum-probability",
        ),
        binary_minimum_variance=required_default_value(
            g_compute_section.binary_minimum_variance,
            "g-binary-minimum-variance",
        ),
        binary_relative_variance_tolerance=required_default_value(
            g_compute_section.binary_relative_variance_tolerance,
            "g-binary-relative-variance-tolerance",
        ),
        firth_maximum_iterations=required_default_value(
            g_compute_section.firth_maximum_iterations,
            "g-firth-maximum-iterations",
        ),
        firth_gradient_tolerance=required_default_value(
            g_compute_section.firth_gradient_tolerance,
            "g-firth-gradient-tolerance",
        ),
        firth_coefficient_tolerance=required_default_value(
            g_compute_section.firth_coefficient_tolerance,
            "g-firth-coefficient-tolerance",
        ),
        firth_likelihood_tolerance=required_default_value(
            g_compute_section.firth_likelihood_tolerance,
            "g-firth-likelihood-tolerance",
        ),
        firth_maximum_step_size=required_default_value(
            g_compute_section.firth_maximum_step_size,
            "g-firth-maximum-step-size",
        ),
        firth_pseudo_maximum_iterations=required_default_value(
            g_compute_section.firth_pseudo_maximum_iterations,
            "g-firth-pseudo-maximum-iterations",
        ),
        firth_pseudo_inner_maximum_iterations=required_default_value(
            g_compute_section.firth_pseudo_inner_maximum_iterations,
            "g-firth-pseudo-inner-maximum-iterations",
        ),
        firth_newton_raphson_zero_start_iterations=required_default_value(
            g_compute_section.firth_newton_raphson_zero_start_iterations,
            "g-firth-newton-raphson-zero-start-iterations",
        ),
        firth_line_search_maximum_attempts=required_default_value(
            g_compute_section.firth_line_search_maximum_attempts,
            "g-firth-line-search-maximum-attempts",
        ),
        firth_step_halving_maximum_attempts=required_default_value(
            g_compute_section.firth_step_halving_maximum_attempts,
            "g-firth-step-halving-maximum-attempts",
        ),
        firth_initial_response_scale=required_default_value(
            g_compute_section.firth_initial_response_scale,
            "g-firth-initial-response-scale",
        ),
        firth_sparse_carrier_dosage_threshold=required_default_value(
            g_compute_section.firth_sparse_carrier_dosage_threshold,
            "g-firth-sparse-carrier-dosage-threshold",
        ),
        firth_step_halving_scale=required_default_value(
            g_compute_section.firth_step_halving_scale,
            "g-firth-step-halving-scale",
        ),
        null_firth_maximum_iterations=required_default_value(
            g_compute_section.null_firth_maximum_iterations,
            "g-null-firth-maximum-iterations",
        ),
        null_firth_gradient_tolerance=required_default_value(
            g_compute_section.null_firth_gradient_tolerance,
            "g-null-firth-gradient-tolerance",
        ),
        null_firth_maximum_step_size=required_default_value(
            g_compute_section.null_firth_maximum_step_size,
            "g-null-firth-maximum-step-size",
        ),
        null_firth_fallback_iteration_multiplier=required_default_value(
            g_compute_section.null_firth_fallback_iteration_multiplier,
            "g-null-firth-fallback-iteration-multiplier",
        ),
        null_firth_fallback_step_divisor=required_default_value(
            g_compute_section.null_firth_fallback_step_divisor,
            "g-null-firth-fallback-step-divisor",
        ),
        null_firth_line_search_maximum_attempts=required_default_value(
            g_compute_section.null_firth_line_search_maximum_attempts,
            "g-null-firth-line-search-maximum-attempts",
        ),
        null_firth_step_halving_scale=required_default_value(
            g_compute_section.null_firth_step_halving_scale,
            "g-null-firth-step-halving-scale",
        ),
        use_block_firth_math=required_default_value(g_compute_section.use_block_firth_math, "g-use-block-firth-math"),
        bgen_decode_tile_variant_count=required_default_value(
            g_compute_section.bgen_decode_tile_variant_count,
            "g-bgen-decode-tile-variant-count",
        ),
        gpu_genotype_format=types.GpuGenotypeFormat(
            required_default_value(g_compute_section.gpu_genotype_format, "g-gpu-genotype-format")
        ),
        score_dtype=types.FloatingPointDtype(required_default_value(g_compute_section.score_dtype, "g-score-dtype")),
        firth_dtype=types.FloatingPointDtype(required_default_value(g_compute_section.firth_dtype, "g-firth-dtype")),
        jax_persistent_cache=required_default_value(
            g_compute_section.jax_persistent_cache,
            "g-jax-persistent-cache",
        ),
        jax_persistent_cache_min_entry_size_bytes=required_default_value(
            g_compute_section.jax_persistent_cache_min_entry_size_bytes,
            "g-jax-persistent-cache-min-entry-size-bytes",
        ),
        jax_persistent_cache_min_compile_time_seconds=required_default_value(
            g_compute_section.jax_persistent_cache_min_compile_time_seconds,
            "g-jax-persistent-cache-min-compile-time-seconds",
        ),
        jax_xla_autotune_cache=required_default_value(
            g_compute_section.jax_xla_autotune_cache,
            "g-jax-xla-autotune-cache",
        ),
        jax_transfer_guard=required_default_value(g_compute_section.jax_transfer_guard, "g-jax-transfer-guard"),
    )


def build_g_output_runtime_defaults(g_output_section: toml_schema.GOutputToml) -> GOutputRuntimeDefaults:
    """Build runtime engine output defaults from the packaged TOML section."""
    return GOutputRuntimeDefaults(
        format=types.OutputFormat(required_default_value(g_output_section.format, "g-output-format")),
        writer_threads=required_default_value(g_output_section.writer_threads, "g-writer-threads"),
        writer_queue_depth=required_default_value(g_output_section.writer_queue_depth, "g-writer-queue-depth"),
        chunks_per_arrow_file=required_default_value(
            g_output_section.chunks_per_arrow_file,
            "g-output-chunks-per-arrow-file",
        ),
        arrow_compression=types.ArrowCompression(
            required_default_value(g_output_section.arrow_compression, "g-output-arrow-compression")
        ),
        parquet_compression=types.ParquetCompression(
            required_default_value(g_output_section.parquet_compression, "g-output-parquet-compression")
        ),
        resume=required_default_value(g_output_section.resume, "g-resume"),
        resume_mode=types.ResumeMode(required_default_value(g_output_section.resume_mode, "g-resume-mode")),
        finalize_parquet=required_default_value(g_output_section.finalize_parquet, "g-finalize-parquet"),
    )


def build_g_diagnostics_runtime_defaults(
    g_diagnostics_section: toml_schema.GDiagnosticsToml,
) -> GDiagnosticsRuntimeDefaults:
    """Build runtime engine diagnostics defaults from the packaged TOML section."""
    return GDiagnosticsRuntimeDefaults(
        telemetry=types.TelemetryMode(required_default_value(g_diagnostics_section.telemetry, "g-telemetry")),
        log_filter=required_default_value(g_diagnostics_section.log_filter, "g-log-filter"),
        log_stderr=required_default_value(g_diagnostics_section.log_stderr, "g-log-stderr"),
        progress_interval_seconds=required_default_value(
            g_diagnostics_section.progress_interval_seconds,
            "g-progress-interval-seconds",
        ),
        progress_interval_chunks=required_default_value(
            g_diagnostics_section.progress_interval_chunks,
            "g-progress-interval-chunks",
        ),
        trace_filter=required_default_value(g_diagnostics_section.trace_filter, "g-trace-filter"),
        log_queue_size=required_default_value(g_diagnostics_section.log_queue_size, "g-log-queue-size"),
        log_lossy=required_default_value(g_diagnostics_section.log_lossy, "g-log-lossy"),
        include_source_location=required_default_value(
            g_diagnostics_section.include_source_location,
            "g-include-source-location",
        ),
        include_span_events=required_default_value(
            g_diagnostics_section.include_span_events,
            "g-include-span-events",
        ),
    )


def validate_default_catalog(normalized_options: typing.Mapping[str, typing.Any]) -> None:
    """Validate packaged default coverage and policy compliance."""
    unknown_option_names = sorted(
        option_name for option_name in normalized_options if option_name not in options.OPTION_SPEC_BY_NAME
    )
    if unknown_option_names:
        formatted_names = ", ".join(unknown_option_names)
        message = f"Default config contains unknown option(s): {formatted_names}."
        raise ValueError(message)

    missing_default_names = sorted(
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy == options.DefaultPolicy.VALUE and option_spec.name not in normalized_options
    )
    if missing_default_names:
        formatted_names = ", ".join(missing_default_names)
        message = f"Default config is missing required default option(s): {formatted_names}."
        raise ValueError(message)

    invalid_default_names = sorted(
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy
        in {
            options.DefaultPolicy.REQUIRED_AT_RUNTIME,
            options.DefaultPolicy.UNSUPPORTED,
            options.DefaultPolicy.DERIVED,
        }
        and option_spec.name in normalized_options
    )
    if invalid_default_names:
        formatted_names = ", ".join(invalid_default_names)
        message = f"Default config contains non-defaultable option(s): {formatted_names}."
        raise ValueError(message)


def build_default_config_hash(raw_toml: typing.Mapping[str, typing.Any]) -> str:
    """Build a stable SHA-256 hash for the packaged default config."""
    normalized_payload = normalize_hash_value(raw_toml)
    encoded_payload = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def normalize_hash_value(value: typing.Any) -> typing.Any:
    """Normalize TOML values into a stable JSON-compatible shape."""
    if isinstance(value, dict):
        return {
            str(item_key): normalize_hash_value(item_value)
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [normalize_hash_value(item_value) for item_value in value]
    return value
