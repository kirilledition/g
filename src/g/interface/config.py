"""Typed REGENIE-compatible configuration and TOML helpers."""

from __future__ import annotations

import dataclasses
import importlib.resources
import os
import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path

from g import types
from g.interface import options

DEFAULT_BSIZE = 8192
DEFAULT_P_THRESHOLD = 0.05
DEFAULT_FIRTH_BATCH_SIZE = 64
DEFAULT_FIRTH_CANDIDATE_CAPACITY = 1024
DEFAULT_BINARY_NULL_MAXIMUM_ITERATIONS = 50
DEFAULT_BINARY_NULL_COEFFICIENT_TOLERANCE = 1.0e-6
DEFAULT_FIRTH_MAXIMUM_ITERATIONS = 250
DEFAULT_FIRTH_GRADIENT_TOLERANCE = 2.5e-4
DEFAULT_FIRTH_COEFFICIENT_TOLERANCE = 2.5e-4
DEFAULT_FIRTH_LIKELIHOOD_TOLERANCE = 2.5e-4
DEFAULT_FIRTH_MAXIMUM_STEP_SIZE = 5.0
DEFAULT_FIRTH_PSEUDO_MAXIMUM_ITERATIONS = 50
DEFAULT_FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS = 25
DEFAULT_FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS = 100
DEFAULT_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
DEFAULT_FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS = 12
DEFAULT_FIRTH_INITIAL_RESPONSE_SCALE = 4.863891244002886
DEFAULT_FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD = 1.0e-4
DEFAULT_FIRTH_STEP_HALVING_SCALE = 0.5
DEFAULT_NULL_FIRTH_MAXIMUM_ITERATIONS = 1000
DEFAULT_NULL_FIRTH_GRADIENT_TOLERANCE = 50.0e-6
DEFAULT_NULL_FIRTH_MAXIMUM_STEP_SIZE = 25.0
DEFAULT_NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER = 5
DEFAULT_NULL_FIRTH_FALLBACK_STEP_DIVISOR = 5.0
DEFAULT_NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
DEFAULT_NULL_FIRTH_STEP_HALVING_SCALE = 0.5
DEFAULT_BGEN_DECODE_TILE_VARIANT_COUNT = 64
DEFAULT_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = -1
DEFAULT_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS = 0
DEFAULT_LOG_FILTER = "info"
DEFAULT_TRACE_FILTER = "g.native.bgen=trace,g.output=debug"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0
DEFAULT_PROGRESS_INTERVAL_CHUNKS = 10
DEFAULT_LOG_QUEUE_SIZE = 65536
DEFAULT_CONFIG_RESOURCE = "config.default.toml"
QUANTITATIVE_BINARY_ONLY_OPTION_NAMES = ("firth", "approx", "firth-se", "spa", "pThresh")


def load_default_g_output_option(option_name: str) -> typing.Any:
    """Load one packaged default output option."""
    default_config_resource = importlib.resources.files("g").joinpath(DEFAULT_CONFIG_RESOURCE)
    with default_config_resource.open("rb") as config_file:
        default_options = tomllib.load(config_file)
    return default_options["g"]["output"][option_name]


DEFAULT_OUTPUT_WRITER_THREADS = int(load_default_g_output_option("writer-threads"))
DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH = int(load_default_g_output_option("writer-queue-depth"))
DEFAULT_OUTPUT_CHUNKS_PER_ARROW_FILE = int(load_default_g_output_option("chunks-per-arrow-file"))


@dataclass(frozen=True)
class InputConfig:
    """Input files and column selections for one REGENIE step 2 run."""

    bgen: Path | None = None
    sample: Path | None = None
    pheno_file: Path | None = None
    pheno_columns: tuple[str, ...] = ()
    covar_file: Path | None = None
    covar_columns: tuple[str, ...] = ()
    pred: Path | None = None


@dataclass(frozen=True)
class TraitConfig:
    """Trait-family and block-size settings."""

    step: int = 2
    trait_type: types.RegenieTraitType = types.RegenieTraitType.QUANTITATIVE
    bsize: int = DEFAULT_BSIZE
    threads: int | None = None


@dataclass(frozen=True)
class BinaryConfig:
    """Binary-trait fallback settings."""

    firth: bool = False
    approx: bool = False
    spa: bool = False
    p_threshold: float = DEFAULT_P_THRESHOLD
    firth_se: bool = False


@dataclass(frozen=True)
class GComputeConfig:
    """Engine-specific runtime and batching settings."""

    device: types.Device = types.Device.CPU
    staging_depth: int = 1
    variant_limit: int | None = None
    trusted_no_missing_diploid: bool = False
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS
    sample_key_mode: types.SampleKeyMode = types.SampleKeyMode.IID
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode = types.MultiPhenotypeSampleMode.PER_PHENOTYPE
    firth_batch_size: int = DEFAULT_FIRTH_BATCH_SIZE
    firth_candidate_capacity: int = DEFAULT_FIRTH_CANDIDATE_CAPACITY
    binary_null_maximum_iterations: int = DEFAULT_BINARY_NULL_MAXIMUM_ITERATIONS
    binary_null_coefficient_tolerance: float = DEFAULT_BINARY_NULL_COEFFICIENT_TOLERANCE
    firth_maximum_iterations: int = DEFAULT_FIRTH_MAXIMUM_ITERATIONS
    firth_gradient_tolerance: float = DEFAULT_FIRTH_GRADIENT_TOLERANCE
    firth_coefficient_tolerance: float = DEFAULT_FIRTH_COEFFICIENT_TOLERANCE
    firth_likelihood_tolerance: float = DEFAULT_FIRTH_LIKELIHOOD_TOLERANCE
    firth_maximum_step_size: float = DEFAULT_FIRTH_MAXIMUM_STEP_SIZE
    firth_pseudo_maximum_iterations: int = DEFAULT_FIRTH_PSEUDO_MAXIMUM_ITERATIONS
    firth_pseudo_inner_maximum_iterations: int = DEFAULT_FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS
    firth_newton_raphson_zero_start_iterations: int = DEFAULT_FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS
    firth_line_search_maximum_attempts: int = DEFAULT_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS
    firth_step_halving_maximum_attempts: int = DEFAULT_FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS
    firth_initial_response_scale: float = DEFAULT_FIRTH_INITIAL_RESPONSE_SCALE
    firth_sparse_carrier_dosage_threshold: float = DEFAULT_FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD
    firth_step_halving_scale: float = DEFAULT_FIRTH_STEP_HALVING_SCALE
    null_firth_maximum_iterations: int = DEFAULT_NULL_FIRTH_MAXIMUM_ITERATIONS
    null_firth_gradient_tolerance: float = DEFAULT_NULL_FIRTH_GRADIENT_TOLERANCE
    null_firth_maximum_step_size: float = DEFAULT_NULL_FIRTH_MAXIMUM_STEP_SIZE
    null_firth_fallback_iteration_multiplier: int = DEFAULT_NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER
    null_firth_fallback_step_divisor: float = DEFAULT_NULL_FIRTH_FALLBACK_STEP_DIVISOR
    null_firth_line_search_maximum_attempts: int = DEFAULT_NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS
    null_firth_step_halving_scale: float = DEFAULT_NULL_FIRTH_STEP_HALVING_SCALE
    use_block_firth_math: bool = False
    bgen_decode_tile_variant_count: int = DEFAULT_BGEN_DECODE_TILE_VARIANT_COUNT
    jax_cache_dir: Path | None = None
    jax_matmul_precision: types.JaxMatmulPrecision | None = None
    jax_persistent_cache: bool = True
    jax_persistent_cache_min_entry_size_bytes: int = DEFAULT_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES
    jax_persistent_cache_min_compile_time_seconds: int = DEFAULT_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS
    jax_xla_autotune_cache: bool = False
    jax_transfer_guard: bool = False


@dataclass(frozen=True)
class GOutputConfig:
    """Engine-specific output settings."""

    out: Path | None = None
    format: types.OutputFormat = types.OutputFormat.PARQUET
    output_run_directory: Path | None = None
    writer_threads: int = DEFAULT_OUTPUT_WRITER_THREADS
    writer_queue_depth: int = DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH
    chunks_per_arrow_file: int = DEFAULT_OUTPUT_CHUNKS_PER_ARROW_FILE
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD
    resume: bool = False
    resume_mode: types.ResumeMode = types.ResumeMode.FAST
    finalize_parquet: bool = False


@dataclass(frozen=True)
class GDiagnosticsConfig:
    """Engine diagnostics settings."""

    telemetry: types.TelemetryMode = types.TelemetryMode.PROGRESS
    log_dir: Path | None = None
    stage_timings_json: Path | None = None
    log_filter: str = DEFAULT_LOG_FILTER
    log_file: Path | None = None
    log_stderr: bool = True
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS
    progress_interval_chunks: int = DEFAULT_PROGRESS_INTERVAL_CHUNKS
    profile_summary_json: Path | None = None
    trace_file: Path | None = None
    trace_filter: str = DEFAULT_TRACE_FILTER
    log_queue_size: int = DEFAULT_LOG_QUEUE_SIZE
    log_lossy: bool = True
    include_source_location: bool = False
    include_span_events: bool = False


@dataclass(frozen=True)
class RegenieConfig:
    """Complete normalized configuration for the shared REGENIE runner."""

    input: InputConfig = dataclasses.field(default_factory=InputConfig)
    trait: TraitConfig = dataclasses.field(default_factory=TraitConfig)
    binary: BinaryConfig = dataclasses.field(default_factory=BinaryConfig)
    g_compute: GComputeConfig = dataclasses.field(default_factory=GComputeConfig)
    g_output: GOutputConfig = dataclasses.field(default_factory=GOutputConfig)
    g_diagnostics: GDiagnosticsConfig = dataclasses.field(default_factory=GDiagnosticsConfig)
    explicit_options: frozenset[str] = dataclasses.field(default_factory=frozenset, compare=False, repr=False)

    @classmethod
    def from_toml(cls, path: Path | str) -> RegenieConfig:
        """Load a normalized configuration from TOML."""
        return load_toml(Path(path))

    @classmethod
    def from_options(cls, raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
        """Build a normalized configuration from Python option names."""
        return from_options(raw_options)

    def to_toml(self) -> str:
        """Serialize the configuration as deterministic TOML."""
        return dumps_toml(self)


def split_name_list(raw_names: str | typing.Iterable[str] | None) -> tuple[str, ...]:
    """Normalize comma-delimited or iterable column names."""
    if raw_names is None:
        return ()
    if isinstance(raw_names, str):
        return tuple(stripped_name for name in raw_names.split(",") if (stripped_name := name.strip()))
    return tuple(stripped_name for name in raw_names if (stripped_name := str(name).strip()))


def path_or_none(raw_value: typing.Any) -> Path | None:
    """Convert an optional path-like value."""
    if raw_value is None:
        return None
    return Path(str(raw_value))


def optional_string(raw_value: typing.Any) -> str | None:
    """Convert an optional string value."""
    if raw_value is not None:
        return str(raw_value)
    return None


def bool_or_false(raw_value: typing.Any) -> bool:
    """Convert TOML or Python option booleans."""
    return bool(raw_value) if raw_value is not None else False


def bool_or_default(raw_value: typing.Any, *, default: bool) -> bool:
    """Convert an optional boolean using a caller-provided default."""
    return bool(raw_value) if raw_value is not None else default


def normalize_trait_type(*, qt: bool | None, bt: bool | None) -> types.RegenieTraitType:
    """Resolve REGENIE trait flags into one trait type."""
    if qt and bt:
        message = "--qt and --bt are mutually exclusive."
        raise ValueError(message)
    if bt:
        return types.RegenieTraitType.BINARY
    return types.RegenieTraitType.QUANTITATIVE


def merge_option_dictionaries(
    base_options: typing.Mapping[str, typing.Any],
    override_options: typing.Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    """Merge options while ignoring unspecified overrides."""
    merged_options = dict(base_options)
    for option_name, option_value in override_options.items():
        if option_value is not None:
            merged_options[option_name] = option_value
    raw_trait_type = override_options.get("trait_type")
    if raw_trait_type is not None:
        trait_type = types.RegenieTraitType(str(raw_trait_type))
        merged_options["qt"] = trait_type == types.RegenieTraitType.QUANTITATIVE
        merged_options["bt"] = trait_type == types.RegenieTraitType.BINARY
    if override_options.get("qt") is True:
        merged_options["bt"] = False
    if override_options.get("bt") is True:
        merged_options["qt"] = False
    return merged_options


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from CLI/TOML/Python option dictionaries."""
    return from_option_layers(
        base_options=load_default_option_dictionary(),
        explicit_option_layers=(raw_options,),
    )


def from_option_layers(
    *,
    base_options: typing.Mapping[str, typing.Any],
    explicit_option_layers: typing.Iterable[typing.Mapping[str, typing.Any]],
) -> RegenieConfig:
    """Build a normalized config from base options and explicit override layers."""
    normalized_options = normalize_option_dictionary(base_options)
    explicit_option_names: set[str] = set()
    for raw_option_layer in explicit_option_layers:
        normalized_option_layer = normalize_option_dictionary(raw_option_layer)
        normalized_options = merge_option_dictionaries(normalized_options, normalized_option_layer)
        explicit_option_names.update(normalized_option_layer)
    return from_normalized_options(
        normalized_options=normalized_options,
        explicit_options=frozenset(explicit_option_names),
    )


def from_normalized_options(
    *,
    normalized_options: typing.Mapping[str, typing.Any],
    explicit_options: frozenset[str],
) -> RegenieConfig:
    """Build a normalized config from already-merged normalized options."""
    trait_type = resolve_configured_trait_type(normalized_options)
    reject_quantitative_binary_only_options(
        explicit_options=explicit_options,
        trait_type=trait_type,
    )
    reject_unsupported_options(normalized_options)
    validate_unknown_options(normalized_options)
    pheno_columns = resolve_exclusive_columns(
        normalized_options,
        repeated_key="phenoCol",
        list_key="phenoColList",
        repeated_snake_key="pheno_col",
        list_snake_key="pheno_col_list",
    )
    covar_columns = resolve_exclusive_columns(
        normalized_options,
        repeated_key="covarCol",
        list_key="covarColList",
        repeated_snake_key="covar_col",
        list_snake_key="covar_col_list",
    )
    config = RegenieConfig(
        input=InputConfig(
            bgen=path_or_none(normalized_options.get("bgen")),
            sample=path_or_none(normalized_options.get("sample")),
            pheno_file=path_or_none(normalized_options.get("phenoFile")),
            pheno_columns=pheno_columns,
            covar_file=path_or_none(normalized_options.get("covarFile")),
            covar_columns=covar_columns,
            pred=path_or_none(normalized_options.get("pred")),
        ),
        trait=TraitConfig(
            step=int(normalized_options.get("step", 2)),
            trait_type=trait_type,
            bsize=int(normalized_options.get("bsize", DEFAULT_BSIZE)),
            threads=optional_int(normalized_options.get("threads")),
        ),
        binary=BinaryConfig(
            firth=bool_or_false(normalized_options.get("firth")),
            approx=bool_or_false(normalized_options.get("approx")),
            spa=bool_or_false(normalized_options.get("spa")),
            p_threshold=float(normalized_options.get("pThresh", DEFAULT_P_THRESHOLD)),
            firth_se=bool_or_false(normalized_options.get("firth-se")),
        ),
        g_compute=GComputeConfig(
            device=types.Device(str(normalized_options.get("g-device", types.Device.CPU.value))),
            staging_depth=int(normalized_options.get("g-staging-depth", 1)),
            variant_limit=optional_int(normalized_options.get("g-variant-limit")),
            trusted_no_missing_diploid=bool_or_false(normalized_options.get("g-trusted-no-missing-diploid")),
            trusted_bgen_validation_mode=types.TrustedBgenValidationMode(
                str(
                    normalized_options.get(
                        "g-trusted-bgen-validation-mode",
                        types.TrustedBgenValidationMode.CACHE_ON_MISS.value,
                    )
                )
            ),
            sample_key_mode=types.SampleKeyMode(
                str(normalized_options.get("g-sample-key-mode", types.SampleKeyMode.IID.value))
            ),
            multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode(
                str(
                    normalized_options.get(
                        "g-multi-phenotype-sample-mode",
                        types.MultiPhenotypeSampleMode.PER_PHENOTYPE.value,
                    )
                )
            ),
            firth_batch_size=int(normalized_options.get("g-firth-batch-size", DEFAULT_FIRTH_BATCH_SIZE)),
            firth_candidate_capacity=int(
                normalized_options.get("g-firth-candidate-capacity", DEFAULT_FIRTH_CANDIDATE_CAPACITY)
            ),
            binary_null_maximum_iterations=int(
                normalized_options.get(
                    "g-binary-null-maximum-iterations",
                    DEFAULT_BINARY_NULL_MAXIMUM_ITERATIONS,
                )
            ),
            binary_null_coefficient_tolerance=float(
                normalized_options.get(
                    "g-binary-null-coefficient-tolerance",
                    DEFAULT_BINARY_NULL_COEFFICIENT_TOLERANCE,
                )
            ),
            firth_maximum_iterations=int(
                normalized_options.get("g-firth-maximum-iterations", DEFAULT_FIRTH_MAXIMUM_ITERATIONS)
            ),
            firth_gradient_tolerance=float(
                normalized_options.get("g-firth-gradient-tolerance", DEFAULT_FIRTH_GRADIENT_TOLERANCE)
            ),
            firth_coefficient_tolerance=float(
                normalized_options.get("g-firth-coefficient-tolerance", DEFAULT_FIRTH_COEFFICIENT_TOLERANCE)
            ),
            firth_likelihood_tolerance=float(
                normalized_options.get("g-firth-likelihood-tolerance", DEFAULT_FIRTH_LIKELIHOOD_TOLERANCE)
            ),
            firth_maximum_step_size=float(
                normalized_options.get("g-firth-maximum-step-size", DEFAULT_FIRTH_MAXIMUM_STEP_SIZE)
            ),
            firth_pseudo_maximum_iterations=int(
                normalized_options.get(
                    "g-firth-pseudo-maximum-iterations",
                    DEFAULT_FIRTH_PSEUDO_MAXIMUM_ITERATIONS,
                )
            ),
            firth_pseudo_inner_maximum_iterations=int(
                normalized_options.get(
                    "g-firth-pseudo-inner-maximum-iterations",
                    DEFAULT_FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS,
                )
            ),
            firth_newton_raphson_zero_start_iterations=int(
                normalized_options.get(
                    "g-firth-newton-raphson-zero-start-iterations",
                    DEFAULT_FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS,
                )
            ),
            firth_line_search_maximum_attempts=int(
                normalized_options.get(
                    "g-firth-line-search-maximum-attempts",
                    DEFAULT_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
                )
            ),
            firth_step_halving_maximum_attempts=int(
                normalized_options.get(
                    "g-firth-step-halving-maximum-attempts",
                    DEFAULT_FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS,
                )
            ),
            firth_initial_response_scale=float(
                normalized_options.get(
                    "g-firth-initial-response-scale",
                    DEFAULT_FIRTH_INITIAL_RESPONSE_SCALE,
                )
            ),
            firth_sparse_carrier_dosage_threshold=float(
                normalized_options.get(
                    "g-firth-sparse-carrier-dosage-threshold",
                    DEFAULT_FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD,
                )
            ),
            firth_step_halving_scale=float(
                normalized_options.get("g-firth-step-halving-scale", DEFAULT_FIRTH_STEP_HALVING_SCALE)
            ),
            null_firth_maximum_iterations=int(
                normalized_options.get("g-null-firth-maximum-iterations", DEFAULT_NULL_FIRTH_MAXIMUM_ITERATIONS)
            ),
            null_firth_gradient_tolerance=float(
                normalized_options.get("g-null-firth-gradient-tolerance", DEFAULT_NULL_FIRTH_GRADIENT_TOLERANCE)
            ),
            null_firth_maximum_step_size=float(
                normalized_options.get("g-null-firth-maximum-step-size", DEFAULT_NULL_FIRTH_MAXIMUM_STEP_SIZE)
            ),
            null_firth_fallback_iteration_multiplier=int(
                normalized_options.get(
                    "g-null-firth-fallback-iteration-multiplier",
                    DEFAULT_NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER,
                )
            ),
            null_firth_fallback_step_divisor=float(
                normalized_options.get(
                    "g-null-firth-fallback-step-divisor",
                    DEFAULT_NULL_FIRTH_FALLBACK_STEP_DIVISOR,
                )
            ),
            null_firth_line_search_maximum_attempts=int(
                normalized_options.get(
                    "g-null-firth-line-search-maximum-attempts",
                    DEFAULT_NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
                )
            ),
            null_firth_step_halving_scale=float(
                normalized_options.get(
                    "g-null-firth-step-halving-scale",
                    DEFAULT_NULL_FIRTH_STEP_HALVING_SCALE,
                )
            ),
            use_block_firth_math=bool_or_false(normalized_options.get("g-use-block-firth-math")),
            bgen_decode_tile_variant_count=int(
                normalized_options.get(
                    "g-bgen-decode-tile-variant-count",
                    DEFAULT_BGEN_DECODE_TILE_VARIANT_COUNT,
                )
            ),
            jax_cache_dir=path_or_none(normalized_options.get("g-jax-cache-dir")),
            jax_matmul_precision=optional_jax_matmul_precision(normalized_options.get("g-jax-matmul-precision")),
            jax_persistent_cache=bool_or_default(normalized_options.get("g-jax-persistent-cache"), default=True),
            jax_persistent_cache_min_entry_size_bytes=int(
                normalized_options.get(
                    "g-jax-persistent-cache-min-entry-size-bytes",
                    DEFAULT_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
                )
            ),
            jax_persistent_cache_min_compile_time_seconds=int(
                normalized_options.get(
                    "g-jax-persistent-cache-min-compile-time-seconds",
                    DEFAULT_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
                )
            ),
            jax_xla_autotune_cache=bool_or_false(normalized_options.get("g-jax-xla-autotune-cache")),
            jax_transfer_guard=bool_or_false(normalized_options.get("g-jax-transfer-guard")),
        ),
        g_output=GOutputConfig(
            out=path_or_none(normalized_options.get("out")),
            format=types.OutputFormat(str(normalized_options.get("g-output-format", types.OutputFormat.PARQUET.value))),
            output_run_directory=path_or_none(normalized_options.get("g-output-run-directory")),
            writer_threads=int(normalized_options.get("g-writer-threads", DEFAULT_OUTPUT_WRITER_THREADS)),
            writer_queue_depth=int(normalized_options.get("g-writer-queue-depth", DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH)),
            chunks_per_arrow_file=int(
                normalized_options.get("g-output-chunks-per-arrow-file", DEFAULT_OUTPUT_CHUNKS_PER_ARROW_FILE)
            ),
            arrow_compression=types.ArrowCompression(
                str(normalized_options.get("g-output-arrow-compression", types.ArrowCompression.ZSTD.value))
            ),
            resume=bool_or_false(normalized_options.get("g-resume")),
            resume_mode=types.ResumeMode(str(normalized_options.get("g-resume-mode", types.ResumeMode.FAST.value))),
            finalize_parquet=bool_or_false(normalized_options.get("g-finalize-parquet")),
        ),
        g_diagnostics=GDiagnosticsConfig(
            telemetry=types.TelemetryMode(
                str(normalized_options.get("g-telemetry", types.TelemetryMode.PROGRESS.value))
            ),
            log_dir=path_or_none(normalized_options.get("g-log-dir")),
            stage_timings_json=path_or_none(normalized_options.get("g-stage-timings-json")),
            log_filter=optional_string(normalized_options.get("g-log-filter")) or DEFAULT_LOG_FILTER,
            log_file=path_or_none(normalized_options.get("g-log-file")),
            log_stderr=bool_or_default(normalized_options.get("g-log-stderr"), default=True),
            progress_interval_seconds=float(
                normalized_options.get("g-progress-interval-seconds", DEFAULT_PROGRESS_INTERVAL_SECONDS)
            ),
            progress_interval_chunks=int(
                normalized_options.get("g-progress-interval-chunks", DEFAULT_PROGRESS_INTERVAL_CHUNKS)
            ),
            profile_summary_json=path_or_none(normalized_options.get("g-profile-summary-json")),
            trace_file=path_or_none(normalized_options.get("g-trace-file")),
            trace_filter=optional_string(normalized_options.get("g-trace-filter")) or DEFAULT_TRACE_FILTER,
            log_queue_size=int(normalized_options.get("g-log-queue-size", DEFAULT_LOG_QUEUE_SIZE)),
            log_lossy=bool_or_default(normalized_options.get("g-log-lossy"), default=True),
            include_source_location=bool_or_false(normalized_options.get("g-include-source-location")),
            include_span_events=bool_or_false(normalized_options.get("g-include-span-events")),
        ),
        explicit_options=explicit_options,
    )
    validate_config(config)
    return config


def optional_int(raw_value: typing.Any) -> int | None:
    """Convert an optional integer value."""
    if raw_value is None:
        return None
    return int(raw_value)


def optional_jax_matmul_precision(raw_value: typing.Any) -> types.JaxMatmulPrecision | None:
    """Convert optional JAX matmul precision."""
    if raw_value is None:
        return None
    return types.JaxMatmulPrecision(str(raw_value))


def resolve_configured_trait_type(normalized_options: typing.Mapping[str, typing.Any]) -> types.RegenieTraitType:
    """Resolve trait type from normalized option names."""
    raw_trait_type = normalized_options.get("trait_type")
    if raw_trait_type is not None and normalized_options.get("qt") is None and normalized_options.get("bt") is None:
        return types.RegenieTraitType(str(raw_trait_type))
    return normalize_trait_type(
        qt=typing.cast("bool | None", normalized_options.get("qt")),
        bt=typing.cast("bool | None", normalized_options.get("bt")),
    )


def normalize_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize snake-case aliases and nested dictionaries into option names."""
    normalized_options: dict[str, typing.Any] = {}
    for option_name, option_value in flatten_option_dictionary(raw_options).items():
        normalized_name = normalize_option_name(option_name)
        normalized_options[normalized_name] = option_value
    return normalized_options


def flatten_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-style nested dictionaries into CLI-style names."""
    flattened_options: dict[str, typing.Any] = {}
    section_prefixes = {
        "input": "",
        "trait": "",
        "binary": "",
        "output": "",
        "g.compute": "g-",
        "g.output": "g-",
        "g.diagnostics": "g-",
    }
    for option_name, option_value in raw_options.items():
        if isinstance(option_value, dict):
            nested_prefix = section_prefixes.get(option_name)
            if nested_prefix is None and option_name == "g":
                flattened_options.update(flatten_g_section(option_value))
                continue
            if nested_prefix is None:
                flattened_options[option_name] = option_value
                continue
            for nested_name, nested_value in option_value.items():
                flattened_options[f"{nested_prefix}{nested_name}"] = nested_value
        else:
            flattened_options[option_name] = option_value
    return flattened_options


def flatten_g_section(raw_g_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten the reserved TOML [g.*] namespace."""
    flattened_options: dict[str, typing.Any] = {}
    g_output_aliases = {
        "format": "g-output-format",
        "output-run-directory": "g-output-run-directory",
        "writer-threads": "g-writer-threads",
        "writer-queue-depth": "g-writer-queue-depth",
        "chunks-per-arrow-file": "g-output-chunks-per-arrow-file",
        "arrow-compression": "g-output-arrow-compression",
        "resume": "g-resume",
        "resume-mode": "g-resume-mode",
        "finalize-parquet": "g-finalize-parquet",
    }
    g_diagnostics_aliases = {
        "telemetry": "g-telemetry",
        "log-dir": "g-log-dir",
        "stage-timings-json": "g-stage-timings-json",
        "log-filter": "g-log-filter",
        "log-file": "g-log-file",
        "log-stderr": "g-log-stderr",
        "progress-interval-seconds": "g-progress-interval-seconds",
        "progress-interval-chunks": "g-progress-interval-chunks",
        "profile-summary-json": "g-profile-summary-json",
        "trace-file": "g-trace-file",
        "trace-filter": "g-trace-filter",
        "log-queue-size": "g-log-queue-size",
        "log-lossy": "g-log-lossy",
        "include-source-location": "g-include-source-location",
        "include-span-events": "g-include-span-events",
    }
    for section_name, section_options in raw_g_options.items():
        if not isinstance(section_options, dict):
            flattened_options[f"g-{section_name}"] = section_options
            continue
        for option_name, option_value in section_options.items():
            if section_name == "output":
                flattened_options[g_output_aliases.get(option_name, f"g-{option_name}")] = option_value
            elif section_name == "diagnostics":
                flattened_options[g_diagnostics_aliases.get(option_name, f"g-{option_name}")] = option_value
            else:
                flattened_options[f"g-{option_name}"] = option_value
    return flattened_options


def normalize_option_name(option_name: str) -> str:
    """Map Pythonic names to REGENIE-compatible names."""
    if option_name == "trait_type":
        return option_name
    if option_name in options.OPTION_SPEC_BY_NAME:
        return option_name
    destination_option_spec = options.OPTION_SPEC_BY_DESTINATION.get(option_name)
    if destination_option_spec is not None:
        return destination_option_spec.name
    aliases = {
        "pheno_file": "phenoFile",
        "pheno": "phenoFile",
        "pheno_col": "phenoCol",
        "pheno_name": "phenoCol",
        "pheno_col_list": "phenoColList",
        "covar_file": "covarFile",
        "covar": "covarFile",
        "covar_col": "covarCol",
        "covar_col_list": "covarColList",
        "covar_names": "covarColList",
        "p_threshold": "pThresh",
        "p_thresh": "pThresh",
        "firth_se": "firth-se",
        "chunk_size": "bsize",
        "trait_type": "trait_type",
        "g_device": "g-device",
        "device": "g-device",
        "g_staging_depth": "g-staging-depth",
        "staging_depth": "g-staging-depth",
        "g_variant_limit": "g-variant-limit",
        "variant_limit": "g-variant-limit",
        "g_trusted_bgen_validation_mode": "g-trusted-bgen-validation-mode",
        "trusted_bgen_validation_mode": "g-trusted-bgen-validation-mode",
        "g_sample_key_mode": "g-sample-key-mode",
        "sample_key_mode": "g-sample-key-mode",
        "g_multi_phenotype_sample_mode": "g-multi-phenotype-sample-mode",
        "multi_phenotype_sample_mode": "g-multi-phenotype-sample-mode",
        "g_output_format": "g-output-format",
        "output_format": "g-output-format",
        "g_output_run_directory": "g-output-run-directory",
        "output_run_directory": "g-output-run-directory",
        "g_writer_threads": "g-writer-threads",
        "output_writer_thread_count": "g-writer-threads",
        "g_writer_queue_depth": "g-writer-queue-depth",
        "output_writer_queue_depth": "g-writer-queue-depth",
        "g_resume": "g-resume",
        "resume": "g-resume",
        "g_resume_mode": "g-resume-mode",
        "resume_mode": "g-resume-mode",
        "g_finalize_parquet": "g-finalize-parquet",
        "finalize_parquet": "g-finalize-parquet",
        "g_firth_batch_size": "g-firth-batch-size",
        "g_firth_candidate_capacity": "g-firth-candidate-capacity",
        "g_binary_null_maximum_iterations": "g-binary-null-maximum-iterations",
        "g_binary_null_coefficient_tolerance": "g-binary-null-coefficient-tolerance",
        "g_firth_maximum_iterations": "g-firth-maximum-iterations",
        "g_firth_gradient_tolerance": "g-firth-gradient-tolerance",
        "g_firth_coefficient_tolerance": "g-firth-coefficient-tolerance",
        "g_firth_likelihood_tolerance": "g-firth-likelihood-tolerance",
        "g_firth_maximum_step_size": "g-firth-maximum-step-size",
        "g_firth_pseudo_maximum_iterations": "g-firth-pseudo-maximum-iterations",
        "g_firth_pseudo_inner_maximum_iterations": "g-firth-pseudo-inner-maximum-iterations",
        "g_firth_newton_raphson_zero_start_iterations": "g-firth-newton-raphson-zero-start-iterations",
        "g_firth_line_search_maximum_attempts": "g-firth-line-search-maximum-attempts",
        "g_firth_step_halving_maximum_attempts": "g-firth-step-halving-maximum-attempts",
        "g_firth_initial_response_scale": "g-firth-initial-response-scale",
        "g_firth_sparse_carrier_dosage_threshold": "g-firth-sparse-carrier-dosage-threshold",
        "g_firth_step_halving_scale": "g-firth-step-halving-scale",
        "g_null_firth_maximum_iterations": "g-null-firth-maximum-iterations",
        "g_null_firth_gradient_tolerance": "g-null-firth-gradient-tolerance",
        "g_null_firth_maximum_step_size": "g-null-firth-maximum-step-size",
        "g_null_firth_fallback_iteration_multiplier": "g-null-firth-fallback-iteration-multiplier",
        "g_null_firth_fallback_step_divisor": "g-null-firth-fallback-step-divisor",
        "g_null_firth_line_search_maximum_attempts": "g-null-firth-line-search-maximum-attempts",
        "g_null_firth_step_halving_scale": "g-null-firth-step-halving-scale",
        "g_use_block_firth_math": "g-use-block-firth-math",
        "g_bgen_decode_tile_variant_count": "g-bgen-decode-tile-variant-count",
        "g_jax_cache_dir": "g-jax-cache-dir",
        "g_jax_matmul_precision": "g-jax-matmul-precision",
        "g_jax_persistent_cache": "g-jax-persistent-cache",
        "g_jax_persistent_cache_min_entry_size_bytes": "g-jax-persistent-cache-min-entry-size-bytes",
        "g_jax_persistent_cache_min_compile_time_seconds": "g-jax-persistent-cache-min-compile-time-seconds",
        "g_jax_xla_autotune_cache": "g-jax-xla-autotune-cache",
        "g_jax_transfer_guard": "g-jax-transfer-guard",
        "g_output_chunks_per_arrow_file": "g-output-chunks-per-arrow-file",
        "g_output_arrow_compression": "g-output-arrow-compression",
        "g_telemetry": "g-telemetry",
        "telemetry": "g-telemetry",
        "g_log_dir": "g-log-dir",
        "log_dir": "g-log-dir",
        "g_stage_timings_json": "g-stage-timings-json",
        "g_log_filter": "g-log-filter",
        "log_filter": "g-log-filter",
        "g_log_file": "g-log-file",
        "log_file": "g-log-file",
        "g_log_stderr": "g-log-stderr",
        "log_stderr": "g-log-stderr",
        "g_progress_interval_seconds": "g-progress-interval-seconds",
        "progress_interval_seconds": "g-progress-interval-seconds",
        "g_progress_interval_chunks": "g-progress-interval-chunks",
        "progress_interval_chunks": "g-progress-interval-chunks",
        "g_profile_summary_json": "g-profile-summary-json",
        "profile_summary_json": "g-profile-summary-json",
        "g_trace_file": "g-trace-file",
        "trace_file": "g-trace-file",
        "g_trace_filter": "g-trace-filter",
        "trace_filter": "g-trace-filter",
        "g_log_queue_size": "g-log-queue-size",
        "log_queue_size": "g-log-queue-size",
        "g_log_lossy": "g-log-lossy",
        "log_lossy": "g-log-lossy",
        "g_include_source_location": "g-include-source-location",
        "include_source_location": "g-include-source-location",
        "g_include_span_events": "g-include-span-events",
        "include_span_events": "g-include-span-events",
        "trusted_no_missing_diploid": "g-trusted-no-missing-diploid",
        "g_trusted_no_missing_diploid": "g-trusted-no-missing-diploid",
    }
    return aliases.get(option_name, option_name.replace("_", "-") if option_name.startswith("g_") else option_name)


def reject_unsupported_options(normalized_options: typing.Mapping[str, typing.Any]) -> None:
    """Reject recognized REGENIE flags that are intentionally unsupported."""
    for option_name in options.unsupported_option_names():
        option_value = normalized_options.get(option_name)
        if option_value is not None and option_value is not False:
            if option_name == "pgen":
                message = "--pgen is a valid REGENIE option, but g currently supports BGEN Step 2 only. Use --bgen."
            elif option_name == "bed":
                message = "--bed is a valid REGENIE option, but g currently supports BGEN Step 2 only. Use --bgen."
            elif option_name == "spa":
                message = "--spa is a valid REGENIE option, but g does not yet implement SPA fallback."
            else:
                message = f"--{option_name} is a valid REGENIE option, but g does not currently support it."
            raise ValueError(message)


def validate_unknown_options(normalized_options: typing.Mapping[str, typing.Any]) -> None:
    """Reject unknown Python, CLI, or TOML options."""
    known_options = (
        options.supported_option_names()
        | options.unsupported_option_names()
        | {
            "trait_type",
            "g-trusted-no-missing-diploid",
        }
    )
    for option_name in normalized_options:
        if option_name not in known_options:
            message = f"Unknown g regenie option: {option_name}"
            raise ValueError(message)


def reject_quantitative_binary_only_options(
    *,
    explicit_options: frozenset[str],
    trait_type: types.RegenieTraitType,
) -> None:
    """Reject binary-only options when the configured trait type is quantitative."""
    if trait_type != types.RegenieTraitType.QUANTITATIVE:
        return
    binary_only_option_names = tuple(
        option_name for option_name in QUANTITATIVE_BINARY_ONLY_OPTION_NAMES if option_name in explicit_options
    )
    raise_for_quantitative_binary_only_options(binary_only_option_names)


def resolve_exclusive_columns(
    normalized_options: typing.Mapping[str, typing.Any],
    *,
    repeated_key: str,
    list_key: str,
    repeated_snake_key: str,
    list_snake_key: str,
) -> tuple[str, ...]:
    """Resolve repeated column options and comma-delimited column-list options."""
    del repeated_snake_key, list_snake_key
    repeated_columns = split_name_list(normalized_options.get(repeated_key))
    list_columns = split_name_list(normalized_options.get(list_key))
    if repeated_columns and list_columns:
        message = f"Use either --{repeated_key} or --{list_key}, not both."
        raise ValueError(message)
    return repeated_columns or list_columns


def validate_config(config: RegenieConfig) -> None:
    """Validate a complete normalized config."""
    if config.trait.step == 1:
        message = "--step 1 is recognized, but g currently supports REGENIE Step 2 only."
        raise ValueError(message)
    if config.trait.step != 2:
        message = "g regenie requires --step 2."
        raise ValueError(message)
    if config.input.bgen is None:
        message = "Exactly one genotype source is required; currently only --bgen is supported."
        raise ValueError(message)
    if config.input.pheno_file is None:
        message = "--phenoFile is required."
        raise ValueError(message)
    if not config.input.pheno_columns:
        message = "At least one --phenoCol or --phenoColList entry is required."
        raise ValueError(message)
    validate_unique_phenotype_names(config.input.pheno_columns)
    if config.input.pred is None:
        message = "--pred is required for REGENIE Step 2."
        raise ValueError(message)
    if config.g_output.out is None:
        message = "--out is required."
        raise ValueError(message)
    if config.trait.bsize <= 0:
        message = "--bsize must be positive."
        raise ValueError(message)
    if config.trait.threads is not None and config.trait.threads <= 0:
        message = "--threads must be positive when provided."
        raise ValueError(message)
    if config.g_compute.staging_depth <= 0:
        message = "--g-staging-depth must be positive."
        raise ValueError(message)
    if config.g_compute.variant_limit is not None and config.g_compute.variant_limit <= 0:
        message = "--g-variant-limit must be positive when provided."
        raise ValueError(message)
    validate_positive_integer("--g-firth-batch-size", config.g_compute.firth_batch_size)
    validate_positive_integer("--g-firth-candidate-capacity", config.g_compute.firth_candidate_capacity)
    validate_positive_integer(
        "--g-binary-null-maximum-iterations",
        config.g_compute.binary_null_maximum_iterations,
    )
    validate_positive_float(
        "--g-binary-null-coefficient-tolerance",
        config.g_compute.binary_null_coefficient_tolerance,
    )
    validate_positive_integer("--g-firth-maximum-iterations", config.g_compute.firth_maximum_iterations)
    validate_positive_float("--g-firth-gradient-tolerance", config.g_compute.firth_gradient_tolerance)
    validate_positive_float("--g-firth-coefficient-tolerance", config.g_compute.firth_coefficient_tolerance)
    validate_positive_float("--g-firth-likelihood-tolerance", config.g_compute.firth_likelihood_tolerance)
    validate_positive_float("--g-firth-maximum-step-size", config.g_compute.firth_maximum_step_size)
    validate_positive_integer(
        "--g-firth-pseudo-maximum-iterations",
        config.g_compute.firth_pseudo_maximum_iterations,
    )
    validate_positive_integer(
        "--g-firth-pseudo-inner-maximum-iterations",
        config.g_compute.firth_pseudo_inner_maximum_iterations,
    )
    validate_positive_integer(
        "--g-firth-newton-raphson-zero-start-iterations",
        config.g_compute.firth_newton_raphson_zero_start_iterations,
    )
    validate_positive_integer(
        "--g-firth-line-search-maximum-attempts",
        config.g_compute.firth_line_search_maximum_attempts,
    )
    validate_positive_integer(
        "--g-firth-step-halving-maximum-attempts",
        config.g_compute.firth_step_halving_maximum_attempts,
    )
    validate_positive_float("--g-firth-initial-response-scale", config.g_compute.firth_initial_response_scale)
    validate_positive_float(
        "--g-firth-sparse-carrier-dosage-threshold",
        config.g_compute.firth_sparse_carrier_dosage_threshold,
    )
    validate_positive_float("--g-firth-step-halving-scale", config.g_compute.firth_step_halving_scale)
    validate_positive_integer("--g-null-firth-maximum-iterations", config.g_compute.null_firth_maximum_iterations)
    validate_positive_float("--g-null-firth-gradient-tolerance", config.g_compute.null_firth_gradient_tolerance)
    validate_positive_float("--g-null-firth-maximum-step-size", config.g_compute.null_firth_maximum_step_size)
    validate_positive_integer(
        "--g-null-firth-fallback-iteration-multiplier",
        config.g_compute.null_firth_fallback_iteration_multiplier,
    )
    validate_positive_float(
        "--g-null-firth-fallback-step-divisor",
        config.g_compute.null_firth_fallback_step_divisor,
    )
    validate_positive_integer(
        "--g-null-firth-line-search-maximum-attempts",
        config.g_compute.null_firth_line_search_maximum_attempts,
    )
    validate_positive_float("--g-null-firth-step-halving-scale", config.g_compute.null_firth_step_halving_scale)
    validate_positive_integer(
        "--g-bgen-decode-tile-variant-count",
        config.g_compute.bgen_decode_tile_variant_count,
    )
    validate_quantitative_binary_config(config)
    if config.g_output.writer_threads <= 0:
        message = "--g-writer-threads must be positive."
        raise ValueError(message)
    if config.g_output.writer_queue_depth <= 0:
        message = "--g-writer-queue-depth must be positive."
        raise ValueError(message)
    if config.g_output.chunks_per_arrow_file <= 0:
        message = "--g-output-chunks-per-arrow-file must be positive."
        raise ValueError(message)
    validate_positive_float(
        "--g-progress-interval-seconds",
        config.g_diagnostics.progress_interval_seconds,
    )
    validate_positive_integer("--g-progress-interval-chunks", config.g_diagnostics.progress_interval_chunks)
    validate_positive_integer("--g-log-queue-size", config.g_diagnostics.log_queue_size)
    if not (0.0 < config.binary.p_threshold < 1.0):
        message = "--pThresh must be in (0, 1)."
        raise ValueError(message)
    if config.binary.firth and not config.binary.approx:
        message = "Exact --firth is not implemented yet. Use --firth --approx."
        raise ValueError(message)
    if config.binary.approx and not config.binary.firth:
        message = "--approx requires --firth."
        raise ValueError(message)


def validate_unique_phenotype_names(phenotype_names: tuple[str, ...]) -> None:
    """Validate that phenotype names identify unique output metadata entries."""
    seen_phenotype_names = set[str]()
    duplicate_phenotype_names = list[str]()
    for phenotype_name in phenotype_names:
        if phenotype_name in seen_phenotype_names:
            duplicate_phenotype_names.append(phenotype_name)
        seen_phenotype_names.add(phenotype_name)
    if duplicate_phenotype_names:
        duplicate_summary = ", ".join(sorted(set(duplicate_phenotype_names)))
        message = f"Duplicate phenotype names are not allowed: {duplicate_summary}."
        raise ValueError(message)


def validate_quantitative_binary_config(config: RegenieConfig) -> None:
    """Reject binary-only configuration for quantitative traits."""
    if config.trait.trait_type != types.RegenieTraitType.QUANTITATIVE:
        return
    binary_only_option_names: list[str] = []
    if config.binary.firth or "firth" in config.explicit_options:
        binary_only_option_names.append("firth")
    if config.binary.approx or "approx" in config.explicit_options:
        binary_only_option_names.append("approx")
    if config.binary.firth_se or "firth-se" in config.explicit_options:
        binary_only_option_names.append("firth-se")
    if config.binary.spa or "spa" in config.explicit_options:
        binary_only_option_names.append("spa")
    if config.binary.p_threshold != DEFAULT_P_THRESHOLD or "pThresh" in config.explicit_options:
        binary_only_option_names.append("pThresh")
    raise_for_quantitative_binary_only_options(tuple(binary_only_option_names))


def raise_for_quantitative_binary_only_options(option_names: tuple[str, ...]) -> None:
    """Raise a clear error for binary-only options used with quantitative traits."""
    if not option_names:
        return
    formatted_option_names = ", ".join(f"--{option_name}" for option_name in option_names)
    message = f"{formatted_option_names} can only be used with --bt; omit binary-only options when using --qt."
    raise ValueError(message)


def validate_positive_integer(option_name: str, value: int) -> None:
    """Validate that an integer config value is positive."""
    if value <= 0:
        message = f"{option_name} must be positive."
        raise ValueError(message)


def validate_positive_float(option_name: str, value: float) -> None:
    """Validate that a floating-point config value is positive."""
    if value <= 0.0:
        message = f"{option_name} must be positive."
        raise ValueError(message)


def load_toml(path: Path) -> RegenieConfig:
    """Load a configuration from a TOML file."""
    raw_config = read_toml_option_dictionary(path)
    return from_options(raw_config)


def load_default_option_dictionary() -> dict[str, typing.Any]:
    """Load packaged default runtime options."""
    default_config_resource = importlib.resources.files("g").joinpath(DEFAULT_CONFIG_RESOURCE)
    with default_config_resource.open("rb") as config_file:
        return tomllib.load(config_file)


def read_toml_option_dictionary(path: Path) -> dict[str, typing.Any]:
    """Read a TOML option dictionary from disk."""
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def write_toml(config: RegenieConfig, path: Path | str) -> None:
    """Write a deterministic TOML file."""
    Path(path).write_text(dumps_toml(config), encoding="utf-8")


def dumps_toml(config: RegenieConfig) -> str:
    """Serialize a configuration to TOML."""
    sections = build_toml_sections(config)
    lines: list[str] = []
    for section_name, section_values in sections.items():
        if not section_values:
            continue
        lines.append(f"[{section_name}]")
        for key, value in section_values.items():
            lines.append(f"{format_toml_key(key)} = {format_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_toml_sections(config: RegenieConfig) -> dict[str, dict[str, typing.Any]]:
    """Build TOML sections for a config."""
    input_section: dict[str, typing.Any] = {}
    if config.input.bgen is not None:
        input_section["bgen"] = config.input.bgen
    if config.input.sample is not None:
        input_section["sample"] = config.input.sample
    if config.input.pheno_file is not None:
        input_section["phenoFile"] = config.input.pheno_file
    if len(config.input.pheno_columns) == 1:
        input_section["phenoCol"] = config.input.pheno_columns[0]
    elif config.input.pheno_columns:
        input_section["phenoColList"] = ",".join(config.input.pheno_columns)
    if config.input.covar_file is not None:
        input_section["covarFile"] = config.input.covar_file
    if len(config.input.covar_columns) == 1:
        input_section["covarCol"] = config.input.covar_columns[0]
    elif config.input.covar_columns:
        input_section["covarColList"] = ",".join(config.input.covar_columns)
    if config.input.pred is not None:
        input_section["pred"] = config.input.pred
    binary_section: dict[str, typing.Any] = {}
    if config.trait.trait_type == types.RegenieTraitType.BINARY:
        binary_section = {
            "firth": config.binary.firth,
            "approx": config.binary.approx,
            "spa": config.binary.spa,
            "pThresh": config.binary.p_threshold,
            "firth-se": config.binary.firth_se,
        }
    return {
        "input": input_section,
        "trait": {
            "step": config.trait.step,
            "qt": config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE,
            "bt": config.trait.trait_type == types.RegenieTraitType.BINARY,
            "bsize": config.trait.bsize,
            **optional_mapping("threads", config.trait.threads),
        },
        "binary": binary_section,
        "output": {
            **optional_mapping("out", config.g_output.out),
        },
        "g.compute": {
            "device": config.g_compute.device.value,
            "staging-depth": config.g_compute.staging_depth,
            **optional_mapping("variant-limit", config.g_compute.variant_limit),
            "trusted-no-missing-diploid": config.g_compute.trusted_no_missing_diploid,
            "trusted-bgen-validation-mode": config.g_compute.trusted_bgen_validation_mode.value,
            "sample-key-mode": config.g_compute.sample_key_mode.value,
            "multi-phenotype-sample-mode": config.g_compute.multi_phenotype_sample_mode.value,
            "firth-batch-size": config.g_compute.firth_batch_size,
            "firth-candidate-capacity": config.g_compute.firth_candidate_capacity,
            "binary-null-maximum-iterations": config.g_compute.binary_null_maximum_iterations,
            "binary-null-coefficient-tolerance": config.g_compute.binary_null_coefficient_tolerance,
            "firth-maximum-iterations": config.g_compute.firth_maximum_iterations,
            "firth-gradient-tolerance": config.g_compute.firth_gradient_tolerance,
            "firth-coefficient-tolerance": config.g_compute.firth_coefficient_tolerance,
            "firth-likelihood-tolerance": config.g_compute.firth_likelihood_tolerance,
            "firth-maximum-step-size": config.g_compute.firth_maximum_step_size,
            "firth-pseudo-maximum-iterations": config.g_compute.firth_pseudo_maximum_iterations,
            "firth-pseudo-inner-maximum-iterations": config.g_compute.firth_pseudo_inner_maximum_iterations,
            "firth-newton-raphson-zero-start-iterations": (
                config.g_compute.firth_newton_raphson_zero_start_iterations
            ),
            "firth-line-search-maximum-attempts": config.g_compute.firth_line_search_maximum_attempts,
            "firth-step-halving-maximum-attempts": config.g_compute.firth_step_halving_maximum_attempts,
            "firth-initial-response-scale": config.g_compute.firth_initial_response_scale,
            "firth-sparse-carrier-dosage-threshold": config.g_compute.firth_sparse_carrier_dosage_threshold,
            "firth-step-halving-scale": config.g_compute.firth_step_halving_scale,
            "null-firth-maximum-iterations": config.g_compute.null_firth_maximum_iterations,
            "null-firth-gradient-tolerance": config.g_compute.null_firth_gradient_tolerance,
            "null-firth-maximum-step-size": config.g_compute.null_firth_maximum_step_size,
            "null-firth-fallback-iteration-multiplier": (
                config.g_compute.null_firth_fallback_iteration_multiplier
            ),
            "null-firth-fallback-step-divisor": config.g_compute.null_firth_fallback_step_divisor,
            "null-firth-line-search-maximum-attempts": config.g_compute.null_firth_line_search_maximum_attempts,
            "null-firth-step-halving-scale": config.g_compute.null_firth_step_halving_scale,
            "use-block-firth-math": config.g_compute.use_block_firth_math,
            "bgen-decode-tile-variant-count": config.g_compute.bgen_decode_tile_variant_count,
            **optional_mapping("jax-cache-dir", config.g_compute.jax_cache_dir),
            **optional_mapping(
                "jax-matmul-precision",
                None if config.g_compute.jax_matmul_precision is None else config.g_compute.jax_matmul_precision.value,
            ),
            "jax-persistent-cache": config.g_compute.jax_persistent_cache,
            "jax-persistent-cache-min-entry-size-bytes": config.g_compute.jax_persistent_cache_min_entry_size_bytes,
            "jax-persistent-cache-min-compile-time-seconds": (
                config.g_compute.jax_persistent_cache_min_compile_time_seconds
            ),
            "jax-xla-autotune-cache": config.g_compute.jax_xla_autotune_cache,
            "jax-transfer-guard": config.g_compute.jax_transfer_guard,
        },
        "g.output": {
            "format": config.g_output.format.value,
            **optional_mapping("output-run-directory", config.g_output.output_run_directory),
            "writer-threads": config.g_output.writer_threads,
            "writer-queue-depth": config.g_output.writer_queue_depth,
            "chunks-per-arrow-file": config.g_output.chunks_per_arrow_file,
            "arrow-compression": config.g_output.arrow_compression.value,
            "resume": config.g_output.resume,
            "resume-mode": config.g_output.resume_mode.value,
            "finalize-parquet": config.g_output.finalize_parquet,
        },
        "g.diagnostics": {
            "telemetry": config.g_diagnostics.telemetry.value,
            **optional_mapping("log-dir", config.g_diagnostics.log_dir),
            **optional_mapping("stage-timings-json", config.g_diagnostics.stage_timings_json),
            "log-filter": config.g_diagnostics.log_filter,
            **optional_mapping("log-file", config.g_diagnostics.log_file),
            "log-stderr": config.g_diagnostics.log_stderr,
            "progress-interval-seconds": config.g_diagnostics.progress_interval_seconds,
            "progress-interval-chunks": config.g_diagnostics.progress_interval_chunks,
            **optional_mapping("profile-summary-json", config.g_diagnostics.profile_summary_json),
            **optional_mapping("trace-file", config.g_diagnostics.trace_file),
            "trace-filter": config.g_diagnostics.trace_filter,
            "log-queue-size": config.g_diagnostics.log_queue_size,
            "log-lossy": config.g_diagnostics.log_lossy,
            "include-source-location": config.g_diagnostics.include_source_location,
            "include-span-events": config.g_diagnostics.include_span_events,
        },
    }


def optional_mapping(key: str, value: typing.Any) -> dict[str, typing.Any]:
    """Return a single-key mapping only when the value is present."""
    if value is None:
        return {}
    return {key: value}


def format_toml_key(key: str) -> str:
    """Quote TOML keys only when required."""
    if "-" in key:
        return f'"{key}"'
    return key


def format_toml_value(value: typing.Any) -> str:
    """Format one TOML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Path):
        return format_toml_string(os.fspath(value))
    return format_toml_string(str(value))


def format_toml_string(value: str) -> str:
    """Format a TOML basic string."""
    escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_value}"'


def build_template() -> str:
    """Return a documented starter config without comments."""
    template_config = RegenieConfig(
        input=InputConfig(
            bgen=Path("data/chr22.bgen"),
            sample=Path("data/chr22.sample"),
            pheno_file=Path("data/pheno.tsv"),
            pheno_columns=("BMI",),
            covar_file=Path("data/covar.tsv"),
            covar_columns=("age", "sex", "PC1", "PC2"),
            pred=Path("data/step1_pred.list"),
        ),
        g_output=GOutputConfig(out=Path("results/bmi")),
    )
    return dumps_toml(template_config)
