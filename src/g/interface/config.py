"""Typed REGENIE-compatible configuration and TOML helpers."""

from __future__ import annotations

import dataclasses
import functools
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import msgspec
import msgspec.inspect

from g import types
from g.interface import config_layers, defaults, options, toml_schema


def load_default_option_dictionary() -> dict[str, typing.Any]:
    """Load packaged default runtime options."""
    return dict(defaults.load_default_option_catalog().raw_toml)


QUANTITATIVE_BINARY_ONLY_OPTION_NAMES = ("firth", "approx", "firth-se", "spa", "pThresh")
INPUT_RUNTIME_FIELD_NAMES = frozenset({"bgen", "sample", "pheno_file", "covar_file", "pred"})
TRAIT_RUNTIME_FIELD_NAMES = frozenset({"step", "bsize", "threads"})
G_COMPUTE_FIELD_RENAMES = {"null_logistic_nonconvergence": "null_logistic_nonconvergence_policy"}


def packaged_default_option(option_name: str) -> typing.Any:
    """Return one packaged default option by canonical name."""
    return required_option(defaults.load_default_option_catalog().normalized_options, option_name)


def required_option(normalized_options: typing.Mapping[str, typing.Any], option_name: str) -> typing.Any:
    """Return a required resolved option or fail loudly."""
    if option_name not in normalized_options:
        message = f"Default config is missing required default option {option_name!r}."
        raise ValueError(message)
    return normalized_options[option_name]


def optional_option(normalized_options: typing.Mapping[str, typing.Any], option_name: str) -> typing.Any | None:
    """Return an optional resolved option."""
    return normalized_options.get(option_name)


def default_bool_option(option_name: str) -> bool:
    """Return one packaged boolean default."""
    return bool(packaged_default_option(option_name))


def default_int_option(option_name: str) -> int:
    """Return one packaged integer default."""
    return int(packaged_default_option(option_name))


def default_float_option(option_name: str) -> float:
    """Return one packaged floating-point default."""
    return float(packaged_default_option(option_name))


def default_string_option(option_name: str) -> str:
    """Return one packaged string default."""
    return str(packaged_default_option(option_name))


def default_trait_type() -> types.RegenieTraitType:
    """Return the packaged default trait type."""
    return normalize_trait_type(
        qt=default_bool_option("qt"),
        bt=default_bool_option("bt"),
    )


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

    step: int = dataclasses.field(default_factory=lambda: default_int_option("step"))
    trait_type: types.RegenieTraitType = dataclasses.field(default_factory=default_trait_type)
    bsize: int = dataclasses.field(default_factory=lambda: default_int_option("bsize"))
    threads: int | None = None


@dataclass(frozen=True)
class BinaryConfig:
    """Binary-trait fallback settings."""

    firth: bool = dataclasses.field(default_factory=lambda: default_bool_option("firth"))
    approx: bool = dataclasses.field(default_factory=lambda: default_bool_option("approx"))
    spa: bool = False
    p_threshold: float = dataclasses.field(default_factory=lambda: default_float_option("pThresh"))
    firth_se: bool = dataclasses.field(default_factory=lambda: default_bool_option("firth-se"))


@dataclass(frozen=True)
class GComputeConfig:
    """Engine-specific runtime and batching settings."""

    device: types.Device = dataclasses.field(default_factory=lambda: types.Device(default_string_option("g-device")))
    staging_depth: int = dataclasses.field(default_factory=lambda: default_int_option("g-staging-depth"))
    variant_limit: int | None = None
    trusted_no_missing_diploid: bool = dataclasses.field(
        default_factory=lambda: default_bool_option("g-trusted-no-missing-diploid")
    )
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = dataclasses.field(
        default_factory=lambda: types.TrustedBgenValidationMode(default_string_option("g-trusted-bgen-validation-mode"))
    )
    sample_key_mode: types.SampleKeyMode = dataclasses.field(
        default_factory=lambda: types.SampleKeyMode(default_string_option("g-sample-key-mode"))
    )
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode = dataclasses.field(
        default_factory=lambda: types.MultiPhenotypeSampleMode(default_string_option("g-multi-phenotype-sample-mode"))
    )
    firth_batch_size: int = dataclasses.field(default_factory=lambda: default_int_option("g-firth-batch-size"))
    firth_candidate_capacity: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-candidate-capacity")
    )
    binary_null_maximum_iterations: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-binary-null-maximum-iterations")
    )
    binary_null_coefficient_tolerance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-binary-null-coefficient-tolerance")
    )
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = dataclasses.field(
        default_factory=lambda: types.NullLogisticNonconvergencePolicy(
            default_string_option("g-null-logistic-nonconvergence")
        )
    )
    binary_minimum_probability: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-binary-minimum-probability")
    )
    binary_minimum_variance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-binary-minimum-variance")
    )
    binary_relative_variance_tolerance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-binary-relative-variance-tolerance")
    )
    firth_maximum_iterations: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-maximum-iterations")
    )
    firth_gradient_tolerance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-gradient-tolerance")
    )
    firth_coefficient_tolerance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-coefficient-tolerance")
    )
    firth_likelihood_tolerance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-likelihood-tolerance")
    )
    firth_maximum_step_size: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-maximum-step-size")
    )
    firth_pseudo_maximum_iterations: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-pseudo-maximum-iterations")
    )
    firth_pseudo_inner_maximum_iterations: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-pseudo-inner-maximum-iterations")
    )
    firth_newton_raphson_zero_start_iterations: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-newton-raphson-zero-start-iterations")
    )
    firth_line_search_maximum_attempts: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-line-search-maximum-attempts")
    )
    firth_step_halving_maximum_attempts: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-firth-step-halving-maximum-attempts")
    )
    firth_initial_response_scale: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-initial-response-scale")
    )
    firth_sparse_carrier_dosage_threshold: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-sparse-carrier-dosage-threshold")
    )
    firth_step_halving_scale: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-firth-step-halving-scale")
    )
    null_firth_maximum_iterations: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-null-firth-maximum-iterations")
    )
    null_firth_gradient_tolerance: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-null-firth-gradient-tolerance")
    )
    null_firth_maximum_step_size: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-null-firth-maximum-step-size")
    )
    null_firth_fallback_iteration_multiplier: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-null-firth-fallback-iteration-multiplier")
    )
    null_firth_fallback_step_divisor: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-null-firth-fallback-step-divisor")
    )
    null_firth_line_search_maximum_attempts: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-null-firth-line-search-maximum-attempts")
    )
    null_firth_step_halving_scale: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-null-firth-step-halving-scale")
    )
    use_block_firth_math: bool = dataclasses.field(
        default_factory=lambda: default_bool_option("g-use-block-firth-math")
    )
    bgen_decode_tile_variant_count: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-bgen-decode-tile-variant-count")
    )
    gpu_genotype_format: types.GpuGenotypeFormat = dataclasses.field(
        default_factory=lambda: types.GpuGenotypeFormat(default_string_option("g-gpu-genotype-format"))
    )
    score_dtype: types.FloatingPointDtype = dataclasses.field(
        default_factory=lambda: types.FloatingPointDtype(default_string_option("g-score-dtype"))
    )
    firth_dtype: types.FloatingPointDtype = dataclasses.field(
        default_factory=lambda: types.FloatingPointDtype(default_string_option("g-firth-dtype"))
    )
    jax_cache_dir: Path | None = None
    jax_matmul_precision: types.JaxMatmulPrecision | None = None
    jax_persistent_cache: bool = dataclasses.field(
        default_factory=lambda: default_bool_option("g-jax-persistent-cache")
    )
    jax_persistent_cache_min_entry_size_bytes: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-jax-persistent-cache-min-entry-size-bytes")
    )
    jax_persistent_cache_min_compile_time_seconds: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-jax-persistent-cache-min-compile-time-seconds")
    )
    jax_xla_autotune_cache: bool = dataclasses.field(
        default_factory=lambda: default_bool_option("g-jax-xla-autotune-cache")
    )
    jax_transfer_guard: bool = dataclasses.field(default_factory=lambda: default_bool_option("g-jax-transfer-guard"))


@dataclass(frozen=True)
class GOutputConfig:
    """Engine-specific output settings."""

    out: Path | None = None
    format: types.OutputFormat = dataclasses.field(
        default_factory=lambda: types.OutputFormat(default_string_option("g-output-format"))
    )
    output_run_directory: Path | None = None
    writer_threads: int = dataclasses.field(default_factory=lambda: default_int_option("g-writer-threads"))
    writer_queue_depth: int = dataclasses.field(default_factory=lambda: default_int_option("g-writer-queue-depth"))
    chunks_per_arrow_file: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-output-chunks-per-arrow-file")
    )
    arrow_compression: types.ArrowCompression = dataclasses.field(
        default_factory=lambda: types.ArrowCompression(default_string_option("g-output-arrow-compression"))
    )
    resume: bool = dataclasses.field(default_factory=lambda: default_bool_option("g-resume"))
    resume_mode: types.ResumeMode = dataclasses.field(
        default_factory=lambda: types.ResumeMode(default_string_option("g-resume-mode"))
    )
    finalize_parquet: bool = dataclasses.field(default_factory=lambda: default_bool_option("g-finalize-parquet"))


@dataclass(frozen=True)
class GDiagnosticsConfig:
    """Engine diagnostics settings."""

    telemetry: types.TelemetryMode = dataclasses.field(
        default_factory=lambda: types.TelemetryMode(default_string_option("g-telemetry"))
    )
    log_dir: Path | None = None
    stage_timings_json: Path | None = None
    log_filter: str = dataclasses.field(default_factory=lambda: default_string_option("g-log-filter"))
    log_file: Path | None = None
    log_stderr: bool = dataclasses.field(default_factory=lambda: default_bool_option("g-log-stderr"))
    progress_interval_seconds: float = dataclasses.field(
        default_factory=lambda: default_float_option("g-progress-interval-seconds")
    )
    progress_interval_chunks: int = dataclasses.field(
        default_factory=lambda: default_int_option("g-progress-interval-chunks")
    )
    profile_summary_json: Path | None = None
    trace_file: Path | None = None
    trace_filter: str = dataclasses.field(default_factory=lambda: default_string_option("g-trace-filter"))
    log_queue_size: int = dataclasses.field(default_factory=lambda: default_int_option("g-log-queue-size"))
    log_lossy: bool = dataclasses.field(default_factory=lambda: default_bool_option("g-log-lossy"))
    include_source_location: bool = dataclasses.field(
        default_factory=lambda: default_bool_option("g-include-source-location")
    )
    include_span_events: bool = dataclasses.field(default_factory=lambda: default_bool_option("g-include-span-events"))


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


def optional_string(raw_value: typing.Any) -> str | None:
    """Convert an optional string value."""
    if raw_value is not None:
        return str(raw_value)
    return None


def normalize_trait_type(*, qt: bool | None, bt: bool | None) -> types.RegenieTraitType:
    """Resolve REGENIE trait flags into one trait type."""
    if qt and bt:
        message = "--qt and --bt are mutually exclusive."
        raise ValueError(message)
    if bt:
        return types.RegenieTraitType.BINARY
    return types.RegenieTraitType.QUANTITATIVE


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from CLI/TOML/Python option dictionaries."""
    explicit_layer = config_layers.option_dictionary_to_toml_config_layer(raw_options, source="Python options")
    return from_toml_config_layers(
        base_config=defaults.load_default_option_catalog().toml_config,
        explicit_layers=(explicit_layer,),
    )


def from_toml_config_layers(
    *,
    base_config: toml_schema.TomlConfig,
    explicit_layers: typing.Iterable[config_layers.TomlConfigLayer],
) -> RegenieConfig:
    """Build a resolved config by overlaying typed TOML config layers."""
    merged_toml_config = base_config
    explicit_option_names: set[str] = set()
    for explicit_layer in explicit_layers:
        reject_layer_trait_flag_conflict(explicit_layer.toml_config)
        merged_toml_config = config_layers.overlay_toml_configs(merged_toml_config, explicit_layer.toml_config)
        merged_toml_config = apply_trait_flag_layer_precedence(merged_toml_config, explicit_layer.toml_config)
        explicit_option_names.update(explicit_layer.explicit_options)
    return from_toml_config(
        toml_config=merged_toml_config,
        explicit_options=frozenset(explicit_option_names),
    )


def reject_layer_trait_flag_conflict(toml_config: toml_schema.TomlConfig) -> None:
    """Reject one config layer that explicitly enables both trait flags."""
    trait_section = toml_config.trait
    if trait_section is msgspec.UNSET:
        return
    if trait_section.qt is True and trait_section.bt is True:
        message = "--qt and --bt are mutually exclusive."
        raise ValueError(message)


def apply_trait_flag_layer_precedence(
    merged_config: toml_schema.TomlConfig,
    override_config: toml_schema.TomlConfig,
) -> toml_schema.TomlConfig:
    """Translate explicit trait flag selections across config layers."""
    override_trait_section = override_config.trait
    merged_trait_section = merged_config.trait
    if override_trait_section is msgspec.UNSET or merged_trait_section is msgspec.UNSET:
        return merged_config

    trait_updates: dict[str, typing.Any] = {}
    if override_trait_section.qt is True:
        trait_updates["bt"] = False
    if override_trait_section.bt is True:
        trait_updates["qt"] = False
    if not trait_updates:
        return merged_config

    updated_trait_section = config_layers.replace_struct_values(merged_trait_section, trait_updates)
    return typing.cast(
        "toml_schema.TomlConfig",
        config_layers.replace_struct_values(merged_config, {"trait": updated_trait_section}),
    )


def from_toml_config(
    *,
    toml_config: toml_schema.TomlConfig,
    explicit_options: frozenset[str],
) -> RegenieConfig:
    """Build a resolved config from a merged typed TOML config."""
    input_section = section_or_default(toml_config.input, toml_schema.InputToml)
    trait_section = section_or_default(toml_config.trait, toml_schema.TraitToml)
    binary_section = section_or_default(toml_config.binary, toml_schema.BinaryToml)
    output_section = section_or_default(toml_config.output, toml_schema.OutputToml)
    g_section = section_or_default(toml_config.g, toml_schema.GNamespaceToml)
    g_compute_section = section_or_default(g_section.compute, toml_schema.GComputeToml)
    g_output_section = section_or_default(g_section.output, toml_schema.GOutputToml)
    g_diagnostics_section = section_or_default(g_section.diagnostics, toml_schema.GDiagnosticsToml)
    normalized_options = config_layers.toml_config_to_option_dictionary(toml_config)
    trait_type = resolve_configured_toml_trait_type(trait_section)
    reject_quantitative_binary_only_options(
        explicit_options=explicit_options,
        trait_type=trait_type,
    )
    reject_unsupported_options(normalized_options)
    validate_unknown_options(normalized_options)
    reject_missing_resolved_default_options(normalized_options)
    pheno_columns = resolve_exclusive_column_values(
        repeated_value=input_section.pheno_col,
        list_value=input_section.pheno_col_list,
        repeated_key="phenoCol",
        list_key="phenoColList",
    )
    covar_columns = resolve_exclusive_column_values(
        repeated_value=input_section.covar_col,
        list_value=input_section.covar_col_list,
        repeated_key="covarCol",
        list_key="covarColList",
    )
    input_values = toml_struct_runtime_mapping(input_section, include_fields=INPUT_RUNTIME_FIELD_NAMES)
    input_values["pheno_columns"] = pheno_columns
    input_values["covar_columns"] = covar_columns
    trait_values = toml_struct_runtime_mapping(trait_section, include_fields=TRAIT_RUNTIME_FIELD_NAMES)
    trait_values["trait_type"] = trait_type
    g_output_values = toml_struct_runtime_mapping(output_section)
    g_output_values.update(toml_struct_runtime_mapping(g_output_section))
    config = convert_runtime_config(
        {
            "input": input_values,
            "trait": trait_values,
            "binary": toml_struct_runtime_mapping(binary_section),
            "g_compute": toml_struct_runtime_mapping(g_compute_section, field_renames=G_COMPUTE_FIELD_RENAMES),
            "g_output": g_output_values,
            "g_diagnostics": toml_struct_runtime_mapping(g_diagnostics_section),
            "explicit_options": explicit_options,
        },
        RegenieConfig,
        section_name="root",
    )
    validate_config(config)
    return config


def toml_struct_runtime_mapping(
    toml_section: msgspec.Struct,
    *,
    include_fields: frozenset[str] | None = None,
    field_renames: typing.Mapping[str, str] | None = None,
) -> dict[str, typing.Any]:
    """Convert one TOML struct section to runtime field names."""
    section_values: dict[str, typing.Any] = {}
    for field_name in toml_struct_field_names(type(toml_section)):
        if include_fields is not None and field_name not in include_fields:
            continue
        field_value = getattr(toml_section, field_name)
        if field_value is msgspec.UNSET:
            continue
        runtime_field_name = field_name
        if field_renames is not None:
            runtime_field_name = field_renames.get(field_name, field_name)
        section_values[runtime_field_name] = field_value
    return section_values


@functools.cache
def toml_struct_field_names(struct_type: type[msgspec.Struct]) -> tuple[str, ...]:
    """Return field names for one TOML msgspec struct type."""
    struct_information = typing.cast("msgspec.inspect.StructType", msgspec.inspect.type_info(struct_type))
    return tuple(field_information.name for field_information in struct_information.fields)


def convert_runtime_config[ConfigT](
    section_values: typing.Mapping[str, typing.Any],
    config_type: type[ConfigT],
    *,
    section_name: str,
) -> ConfigT:
    """Convert resolved TOML values into a runtime config dataclass."""
    try:
        return msgspec.convert(
            section_values,
            type=config_type,
            strict=True,
            dec_hook=decode_runtime_value,
        )
    except msgspec.ValidationError as error:
        message = f"Invalid resolved {section_name} config: {error}"
        raise ValueError(message) from error


def decode_runtime_value(target_type: type[typing.Any], value: typing.Any) -> typing.Any:
    """Decode custom runtime field types for msgspec conversion."""
    if target_type is Path:
        return Path(str(value))
    raise NotImplementedError


def section_or_default[TomlStructT: msgspec.Struct](
    section: TomlStructT | msgspec.UnsetType,
    section_type: type[TomlStructT],
) -> TomlStructT:
    """Return a TOML section or an empty section instance when absent."""
    if section is msgspec.UNSET:
        return section_type()
    return section


def optional_toml_value[TomlValueT](raw_value: TomlValueT | msgspec.UnsetType) -> TomlValueT | None:
    """Return an optional TOML value, mapping absent fields to None."""
    if raw_value is msgspec.UNSET:
        return None
    return raw_value


def resolve_configured_toml_trait_type(trait_section: toml_schema.TraitToml) -> types.RegenieTraitType:
    """Resolve trait type from the typed TOML trait section."""
    return normalize_trait_type(
        qt=optional_toml_value(trait_section.qt),
        bt=optional_toml_value(trait_section.bt),
    )


def resolve_exclusive_column_values(
    *,
    repeated_value: toml_schema.TomlStringList,
    list_value: toml_schema.TomlStringList,
    repeated_key: str,
    list_key: str,
) -> tuple[str, ...]:
    """Resolve repeated and comma-delimited column-list TOML values."""
    repeated_columns = split_name_list(optional_toml_value(repeated_value))
    list_columns = split_name_list(optional_toml_value(list_value))
    if repeated_columns and list_columns:
        message = f"Use either --{repeated_key} or --{list_key}, not both."
        raise ValueError(message)
    return repeated_columns or list_columns


def normalize_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize snake-case aliases and nested dictionaries into option names."""
    return config_layers.normalize_option_dictionary(raw_options)


def flatten_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-style nested dictionaries into CLI-style names."""
    return config_layers.flatten_toml_mapping(raw_options)


def normalize_option_name(option_name: str) -> str:
    """Map Pythonic names to REGENIE-compatible names."""
    return config_layers.normalize_option_name(option_name)


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
    known_options = options.supported_option_names() | options.unsupported_option_names() | {"trait_type"}
    for option_name in normalized_options:
        if option_name not in known_options:
            message = f"Unknown g regenie option: {option_name}"
            raise ValueError(message)


def reject_missing_resolved_default_options(normalized_options: typing.Mapping[str, typing.Any]) -> None:
    """Reject resolved configs that lost packaged value defaults."""
    missing_option_names = tuple(
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy == options.DefaultPolicy.VALUE and option_spec.name not in normalized_options
    )
    if not missing_option_names:
        return
    message = f"Default config is missing required default option {missing_option_names[0]!r}."
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
    validate_probability_floor("--g-binary-minimum-probability", config.g_compute.binary_minimum_probability)
    validate_positive_float("--g-binary-minimum-variance", config.g_compute.binary_minimum_variance)
    validate_positive_float(
        "--g-binary-relative-variance-tolerance",
        config.g_compute.binary_relative_variance_tolerance,
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
    if config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.PACKED8:
        if config.g_compute.device != types.Device.GPU:
            message = "--g-gpu-genotype-format=packed8 requires --g-device=gpu."
            raise ValueError(message)
        if len(config.input.pheno_columns) != 1:
            message = "--g-gpu-genotype-format=packed8 currently supports one phenotype at a time."
            raise ValueError(message)
    if config.g_compute.firth_dtype != types.FloatingPointDtype.FLOAT64:
        message = "--g-firth-dtype currently supports float64 only."
        raise ValueError(message)
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
    if config.binary.p_threshold != default_float_option("pThresh") or "pThresh" in config.explicit_options:
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


def validate_probability_floor(option_name: str, value: float) -> None:
    """Validate that a probability floor remains below a symmetric midpoint."""
    validate_positive_float(option_name, value)
    if value >= 0.5:
        message = f"{option_name} must be less than 0.5."
        raise ValueError(message)


def load_toml(path: Path) -> RegenieConfig:
    """Load a configuration from a TOML file."""
    toml_layer = config_layers.decode_toml_file_layer(path)
    return from_toml_config_layers(
        base_config=defaults.load_default_option_catalog().toml_config,
        explicit_layers=(toml_layer,),
    )


def read_toml_option_dictionary(path: Path) -> dict[str, typing.Any]:
    """Read a TOML option dictionary from disk."""
    return config_layers.toml_config_to_builtin_mapping(config_layers.decode_toml_file(path))


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
            "null-logistic-nonconvergence": config.g_compute.null_logistic_nonconvergence_policy.value,
            "binary-minimum-probability": config.g_compute.binary_minimum_probability,
            "binary-minimum-variance": config.g_compute.binary_minimum_variance,
            "binary-relative-variance-tolerance": config.g_compute.binary_relative_variance_tolerance,
            "firth-maximum-iterations": config.g_compute.firth_maximum_iterations,
            "firth-gradient-tolerance": config.g_compute.firth_gradient_tolerance,
            "firth-coefficient-tolerance": config.g_compute.firth_coefficient_tolerance,
            "firth-likelihood-tolerance": config.g_compute.firth_likelihood_tolerance,
            "firth-maximum-step-size": config.g_compute.firth_maximum_step_size,
            "firth-pseudo-maximum-iterations": config.g_compute.firth_pseudo_maximum_iterations,
            "firth-pseudo-inner-maximum-iterations": config.g_compute.firth_pseudo_inner_maximum_iterations,
            "firth-newton-raphson-zero-start-iterations": (config.g_compute.firth_newton_raphson_zero_start_iterations),
            "firth-line-search-maximum-attempts": config.g_compute.firth_line_search_maximum_attempts,
            "firth-step-halving-maximum-attempts": config.g_compute.firth_step_halving_maximum_attempts,
            "firth-initial-response-scale": config.g_compute.firth_initial_response_scale,
            "firth-sparse-carrier-dosage-threshold": config.g_compute.firth_sparse_carrier_dosage_threshold,
            "firth-step-halving-scale": config.g_compute.firth_step_halving_scale,
            "null-firth-maximum-iterations": config.g_compute.null_firth_maximum_iterations,
            "null-firth-gradient-tolerance": config.g_compute.null_firth_gradient_tolerance,
            "null-firth-maximum-step-size": config.g_compute.null_firth_maximum_step_size,
            "null-firth-fallback-iteration-multiplier": (config.g_compute.null_firth_fallback_iteration_multiplier),
            "null-firth-fallback-step-divisor": config.g_compute.null_firth_fallback_step_divisor,
            "null-firth-line-search-maximum-attempts": config.g_compute.null_firth_line_search_maximum_attempts,
            "null-firth-step-halving-scale": config.g_compute.null_firth_step_halving_scale,
            "use-block-firth-math": config.g_compute.use_block_firth_math,
            "bgen-decode-tile-variant-count": config.g_compute.bgen_decode_tile_variant_count,
            "gpu-genotype-format": config.g_compute.gpu_genotype_format.value,
            "score-dtype": config.g_compute.score_dtype.value,
            "firth-dtype": config.g_compute.firth_dtype.value,
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
        "metadata": {
            "default-config-hash": defaults.load_default_option_catalog().default_config_hash,
            "option-schema-version": defaults.OPTION_SCHEMA_VERSION,
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
    """Format one TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Path):
        return format_toml_string(os.fspath(value))
    if isinstance(value, list | tuple):
        return f"[{', '.join(format_toml_value(item_value) for item_value in value)}]"
    return format_toml_string(str(value))


def format_toml_string(value: str) -> str:
    """Format a TOML basic string."""
    escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_value}"'


def build_template() -> str:
    """Return a starter config with placeholders and packaged defaults."""
    raw_default_toml = defaults.load_default_option_catalog().raw_toml
    lines: list[str] = []
    lines.extend(build_commented_section("input"))
    lines.append("")
    lines.extend(build_commented_section("output"))
    lines.append("")
    append_raw_template_sections(lines, raw_default_toml)
    return "\n".join(lines).rstrip() + "\n"


def build_commented_section(section_name: str) -> list[str]:
    """Build commented placeholder lines for one no-default TOML section."""
    placeholder_lines = [f"[{section_name}]"]
    for option_spec in options.OPTION_SPECS:
        if option_spec.section != section_name:
            continue
        if option_spec.default_policy not in {
            options.DefaultPolicy.REQUIRED_AT_RUNTIME,
            options.DefaultPolicy.ABSENT_IS_NONE,
        }:
            continue
        placeholder_value = template_placeholder_for_option(option_spec)
        if placeholder_value is None:
            continue
        placeholder_lines.append(f"# {format_toml_key(typing.cast('str', option_spec.toml_key))} = {placeholder_value}")
    return placeholder_lines


def template_placeholder_for_option(option_spec: options.OptionSpec) -> str | None:
    """Return a TOML placeholder value for a no-default option."""
    placeholders = {
        "bgen": format_toml_string("data/chr22.bgen"),
        "sample": format_toml_string("data/chr22.sample"),
        "phenoFile": format_toml_string("data/pheno.tsv"),
        "phenoCol": format_toml_string("BMI"),
        "covarFile": format_toml_string("data/covar.tsv"),
        "covarColList": format_toml_string("age,sex,PC1,PC2"),
        "pred": format_toml_string("data/step1_pred.list"),
        "out": format_toml_string("results/bmi"),
    }
    return placeholders.get(option_spec.name)


def append_raw_template_sections(lines: list[str], raw_default_toml: typing.Mapping[str, typing.Any]) -> None:
    """Append packaged default TOML sections to template lines."""
    for section_name, section_value in raw_default_toml.items():
        if not isinstance(section_value, dict):
            continue
        if section_name == "g":
            for g_section_name, g_section_value in section_value.items():
                if isinstance(g_section_value, dict):
                    append_raw_template_section(lines, f"g.{g_section_name}", g_section_value)
            continue
        if section_name == "binary":
            append_commented_raw_template_section(lines, section_name, section_value)
            continue
        append_raw_template_section(lines, section_name, section_value)


def append_raw_template_section(
    lines: list[str],
    section_name: str,
    section_values: typing.Mapping[str, typing.Any],
) -> None:
    """Append one raw TOML section to template lines."""
    lines.append(f"[{section_name}]")
    for key, value in section_values.items():
        lines.append(f"{format_toml_key(key)} = {format_toml_value(value)}")
    lines.append("")


def append_commented_raw_template_section(
    lines: list[str],
    section_name: str,
    section_values: typing.Mapping[str, typing.Any],
) -> None:
    """Append one raw TOML section as comments."""
    lines.append(f"# [{section_name}]")
    for key, value in section_values.items():
        lines.append(f"# {format_toml_key(key)} = {format_toml_value(value)}")
    lines.append("")
