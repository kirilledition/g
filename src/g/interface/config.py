"""Thin Python boundary for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import collections.abc
import functools
import typing

import g._core
from g import types

if typing.TYPE_CHECKING:
    from pathlib import Path

InputConfig = g._core.InputConfig
TraitConfig = g._core.TraitConfig
BinaryConfig = g._core.BinaryConfig
GComputeConfig = g._core.GComputeConfig
GOutputConfig = g._core.GOutputConfig
GDiagnosticsConfig = g._core.GDiagnosticsConfig
RegenieConfig = g._core.RegenieConfig

FLAT_OPTION_SECTIONS: dict[str, tuple[str, str]] = {
    "bgen": ("input", "bgen"),
    "sample": ("input", "sample"),
    "phenoFile": ("input", "phenoFile"),
    "phenoCol": ("input", "phenoCol"),
    "phenoColList": ("input", "phenoColList"),
    "covarFile": ("input", "covarFile"),
    "covarCol": ("input", "covarCol"),
    "covarColList": ("input", "covarColList"),
    "pred": ("input", "pred"),
    "step": ("trait", "step"),
    "trait_type": ("trait", "trait_type"),
    "qt": ("trait", "qt"),
    "bt": ("trait", "bt"),
    "bsize": ("trait", "bsize"),
    "threads": ("trait", "threads"),
    "firth": ("binary", "firth"),
    "approx": ("binary", "approx"),
    "pThresh": ("binary", "pThresh"),
    "firth-se": ("binary", "firth-se"),
    "firth_se": ("binary", "firth-se"),
    "out": ("output", "out"),
    "format": ("output", "format"),
    "output_run_directory": ("output", "output_run_directory"),
    "g-output-format": ("output", "format"),
    "g-output-run-directory": ("output", "output_run_directory"),
    "g-writer-threads": ("output", "writer_threads"),
    "g-writer-queue-depth": ("output", "writer_queue_depth"),
    "g-output-chunks-per-arrow-file": ("output", "chunks_per_arrow_file"),
    "g-output-arrow-compression": ("output", "arrow_compression"),
    "g-output-parquet-compression": ("output", "parquet_compression"),
    "g-resume": ("output", "resume"),
    "g-resume-mode": ("output", "resume_mode"),
    "g-finalize-parquet": ("output", "finalize_parquet"),
    "g-device": ("compute", "device"),
    "g-staging-depth": ("compute", "staging_depth"),
    "g-result-in-flight-limit": ("compute", "result_in_flight_limit"),
    "g-dosage-buffer-limit": ("compute", "dosage_buffer_limit"),
    "g-variant-limit": ("compute", "variant_limit"),
    "g-trusted-no-missing-diploid": ("compute", "trusted_no_missing_diploid"),
    "g-trusted-bgen-validation-mode": ("compute", "trusted_bgen_validation_mode"),
    "g-sample-key-mode": ("compute", "sample_key_mode"),
    "g-multi-phenotype-sample-mode": ("compute", "multi_phenotype_sample_mode"),
    "g-firth-batch-size": ("compute", "firth_batch_size"),
    "g-firth-candidate-capacity": ("compute", "firth_candidate_capacity"),
    "g-binary-null-maximum-iterations": ("compute", "binary_null_maximum_iterations"),
    "g-binary-null-coefficient-tolerance": ("compute", "binary_null_coefficient_tolerance"),
    "g-null-logistic-nonconvergence": ("compute", "null_logistic_nonconvergence_policy"),
    "g-binary-minimum-probability": ("compute", "binary_minimum_probability"),
    "g-binary-minimum-variance": ("compute", "binary_minimum_variance"),
    "g-binary-relative-variance-tolerance": ("compute", "binary_relative_variance_tolerance"),
    "g-linear-minimum-variance": ("compute", "linear_minimum_variance"),
    "g-linear-relative-variance-tolerance": ("compute", "linear_relative_variance_tolerance"),
    "g-firth-maximum-iterations": ("compute", "firth_maximum_iterations"),
    "g-firth-gradient-tolerance": ("compute", "firth_gradient_tolerance"),
    "g-firth-coefficient-tolerance": ("compute", "firth_coefficient_tolerance"),
    "g-firth-likelihood-tolerance": ("compute", "firth_likelihood_tolerance"),
    "g-firth-maximum-step-size": ("compute", "firth_maximum_step_size"),
    "g-firth-pseudo-maximum-iterations": ("compute", "firth_pseudo_maximum_iterations"),
    "g-firth-pseudo-inner-maximum-iterations": ("compute", "firth_pseudo_inner_maximum_iterations"),
    "g-firth-newton-raphson-zero-start-iterations": ("compute", "firth_newton_raphson_zero_start_iterations"),
    "g-firth-line-search-maximum-attempts": ("compute", "firth_line_search_maximum_attempts"),
    "g-firth-step-halving-maximum-attempts": ("compute", "firth_step_halving_maximum_attempts"),
    "g-firth-initial-response-scale": ("compute", "firth_initial_response_scale"),
    "g-firth-sparse-carrier-dosage-threshold": ("compute", "firth_sparse_carrier_dosage_threshold"),
    "g-firth-step-halving-scale": ("compute", "firth_step_halving_scale"),
    "g-null-firth-maximum-iterations": ("compute", "null_firth_maximum_iterations"),
    "g-null-firth-gradient-tolerance": ("compute", "null_firth_gradient_tolerance"),
    "g-null-firth-maximum-step-size": ("compute", "null_firth_maximum_step_size"),
    "g-null-firth-fallback-iteration-multiplier": ("compute", "null_firth_fallback_iteration_multiplier"),
    "g-null-firth-fallback-step-divisor": ("compute", "null_firth_fallback_step_divisor"),
    "g-null-firth-line-search-maximum-attempts": ("compute", "null_firth_line_search_maximum_attempts"),
    "g-null-firth-step-halving-scale": ("compute", "null_firth_step_halving_scale"),
    "g-use-block-firth-math": ("compute", "use_block_firth_math"),
    "g-bgen-decode-tile-variant-count": ("compute", "bgen_decode_tile_variant_count"),
    "g-gpu-genotype-format": ("compute", "gpu_genotype_format"),
    "g-score-dtype": ("compute", "score_dtype"),
    "g-firth-dtype": ("compute", "firth_dtype"),
    "g-jax-cache-dir": ("compute", "jax_cache_dir"),
    "g-jax-matmul-precision": ("compute", "jax_matmul_precision"),
    "g-jax-persistent-cache": ("compute", "jax_persistent_cache"),
    "g-jax-persistent-cache-min-entry-size-bytes": ("compute", "jax_persistent_cache_min_entry_size_bytes"),
    "g-jax-persistent-cache-min-compile-time-seconds": ("compute", "jax_persistent_cache_min_compile_time_seconds"),
    "g-jax-xla-autotune-cache": ("compute", "jax_xla_autotune_cache"),
    "g-jax-transfer-guard": ("compute", "jax_transfer_guard"),
    "g-telemetry": ("diagnostics", "telemetry"),
    "g-log-dir": ("diagnostics", "log_dir"),
    "g-stage-timings-json": ("diagnostics", "stage_timings_json"),
    "g-log-filter": ("diagnostics", "log_filter"),
    "g-log-file": ("diagnostics", "log_file"),
    "g-log-stderr": ("diagnostics", "log_stderr"),
    "g-progress-interval-seconds": ("diagnostics", "progress_interval_seconds"),
    "g-progress-interval-chunks": ("diagnostics", "progress_interval_chunks"),
    "g-profile-summary-json": ("diagnostics", "profile_summary_json"),
    "g-trace-file": ("diagnostics", "trace_file"),
    "g-trace-filter": ("diagnostics", "trace_filter"),
    "g-trace-event-cap": ("diagnostics", "trace_event_cap"),
    "g-log-queue-size": ("diagnostics", "log_queue_size"),
    "g-log-lossy": ("diagnostics", "log_lossy"),
    "g-include-source-location": ("diagnostics", "include_source_location"),
    "g-include-span-events": ("diagnostics", "include_span_events"),
}

BOOLEAN_PYTHON_OPTIONS = {
    "qt",
    "bt",
    "firth",
    "approx",
    "firth-se",
    "firth_se",
    "g-trusted-no-missing-diploid",
    "g-use-block-firth-math",
    "g-jax-persistent-cache",
    "g-jax-xla-autotune-cache",
    "g-jax-transfer-guard",
    "g-resume",
    "g-finalize-parquet",
    "g-log-stderr",
    "g-log-lossy",
    "g-include-source-location",
    "g-include-span-events",
}

BOOLEAN_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
BOOLEAN_FALSE_VALUES = frozenset(("0", "false", "no", "off"))

NATIVE_CONFIG_SECTION_NAMES = frozenset(("input", "trait", "binary", "compute", "output", "diagnostics", "metadata"))


def normalize_python_options(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize flat Python option dictionaries into native TOML sections."""
    normalized_options: dict[str, typing.Any] = {}
    for option_name, option_value in raw_options.items():
        option_target = FLAT_OPTION_SECTIONS.get(option_name)
        if option_target is None:
            if option_name in NATIVE_CONFIG_SECTION_NAMES:
                if isinstance(option_value, collections.abc.Mapping):
                    section_options = normalized_options.setdefault(option_name, {})
                    if isinstance(section_options, dict):
                        section_options.update(option_value)
                    else:
                        normalized_options[option_name] = dict(option_value)
                else:
                    normalized_options[option_name] = option_value
                continue
            if isinstance(option_value, collections.abc.Mapping):
                message = f"Unknown g regenie option: {flatten_unknown_option_name(option_name, option_value)}"
            else:
                message = f"Unknown g regenie option: {option_name}"
            raise ValueError(message)
        if option_value is None:
            continue
        section_name, section_option_name = option_target
        section_options = normalized_options.setdefault(section_name, {})
        if not isinstance(section_options, dict):
            normalized_options[option_name] = option_value
            continue
        section_options[section_option_name] = normalize_python_option_value(option_name, option_value)
    return normalized_options


def normalize_python_option_value(option_name: str, option_value: typing.Any) -> typing.Any:
    """Normalize Python option values before native TOML conversion."""
    if option_name not in BOOLEAN_PYTHON_OPTIONS:
        return option_value
    if isinstance(option_value, bool):
        return option_value
    if isinstance(option_value, str):
        normalized_value = option_value.strip().lower()
        if normalized_value in BOOLEAN_TRUE_VALUES:
            return True
        if normalized_value in BOOLEAN_FALSE_VALUES:
            return False
    message = "Boolean option value must be a bool or one of true/false/on/off/yes/no/1/0."
    raise ValueError(message)


def flatten_unknown_option_name(option_name: str, option_value: collections.abc.Mapping[str, typing.Any]) -> str:
    """Build a dotted option name for unknown nested Python options."""
    if not option_value:
        return option_name
    nested_key = next(iter(option_value))
    nested_value = option_value[nested_key]
    if isinstance(nested_value, collections.abc.Mapping):
        return f"{option_name}.{flatten_unknown_option_name(str(nested_key), nested_value)}"
    return f"{option_name}.{nested_key}"


def split_name_list(value: str | None) -> tuple[str, ...]:
    """Split a comma-delimited REGENIE name list."""
    if value is None:
        return ()
    return tuple(name.strip() for name in value.split(",") if name.strip())


def optional_string(value: object | None) -> str | None:
    """Normalize optional string-like config values."""
    if value is None:
        return None
    return str(value)


def normalize_option_name(option_name: str) -> str:
    """Normalize legacy Python option aliases to CLI-style names."""
    legacy_option_names = {
        "g_null_logistic_nonconvergence_policy": "g-null-logistic-nonconvergence",
    }
    return legacy_option_names.get(option_name, option_name)


def normalize_trait_type(
    *,
    qt: bool | None = None,
    bt: bool | None = None,
    trait_type: types.RegenieTraitType | str | None = None,
) -> types.RegenieTraitType:
    """Normalize quantitative/binary trait selectors."""
    if qt is True and bt is True:
        message = "--qt and --bt are mutually exclusive."
        raise ValueError(message)
    if bt is True:
        return types.RegenieTraitType.BINARY
    if qt is True:
        return types.RegenieTraitType.QUANTITATIVE
    if trait_type is None:
        return types.RegenieTraitType.QUANTITATIVE
    return types.RegenieTraitType(trait_type)


def flatten_toml_mapping(raw_mapping: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-shaped config mappings into Python option names."""
    flattened_options: dict[str, typing.Any] = {}
    for key, value in raw_mapping.items():
        if key == "g" and isinstance(value, collections.abc.Mapping):
            flatten_g_section(value, flattened_options)
        elif isinstance(value, collections.abc.Mapping):
            flatten_mapping_section(prefix=key, raw_mapping=value, flattened_options=flattened_options)
        else:
            flattened_options[key] = value
    return flattened_options


def flatten_g_section(
    raw_mapping: typing.Mapping[str, typing.Any],
    flattened_options: dict[str, typing.Any],
) -> None:
    """Flatten a `[g.*]` TOML section into `g-*` Python options."""
    for key, value in raw_mapping.items():
        if key == "compute" and isinstance(value, collections.abc.Mapping):
            for option_name, option_value in value.items():
                flattened_options[f"g-{option_name.replace('_', '-')}"] = option_value
        elif key == "output" and isinstance(value, collections.abc.Mapping):
            for option_name, option_value in value.items():
                flattened_options[f"g-output-{option_name.replace('_', '-')}"] = option_value
        elif key == "diagnostics" and isinstance(value, collections.abc.Mapping):
            for option_name, option_value in value.items():
                flattened_options[f"g-{option_name.replace('_', '-')}"] = option_value
        else:
            flattened_options[f"g.{key}"] = value


def flatten_mapping_section(
    *,
    prefix: str,
    raw_mapping: typing.Mapping[str, typing.Any],
    flattened_options: dict[str, typing.Any],
) -> None:
    """Flatten an unknown TOML section using dotted keys."""
    for key, value in raw_mapping.items():
        flattened_key = f"{prefix}.{key}"
        if isinstance(value, collections.abc.Mapping):
            flatten_mapping_section(prefix=flattened_key, raw_mapping=value, flattened_options=flattened_options)
        else:
            flattened_options[flattened_key] = value


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from Python option dictionaries."""
    return g._core.config_from_options(normalize_python_options(raw_options))


def explicit_options(config: RegenieConfig) -> frozenset[str]:
    """Return explicit legacy option names when available."""
    del config
    return frozenset()


typing.cast("typing.Any", RegenieConfig).from_options = staticmethod(from_options)
typing.cast("typing.Any", RegenieConfig).explicit_options = property(explicit_options)


@functools.cache
def load_packaged_config() -> RegenieConfig:
    """Load packaged TOML defaults as a complete unvalidated runtime config."""
    return g._core.load_packaged_config()


def load_toml(path: Path) -> RegenieConfig:
    """Load a configuration from a TOML file."""
    return g._core.config_from_toml(path)


def validate_config(config: RegenieConfig) -> None:
    """Validate a complete normalized config."""
    if config.is_validated:
        return
    g._core.validate_regenie_config(config)


def validate_config_for_run(config: RegenieConfig) -> None:
    """Validate a complete normalized config at the execution boundary."""
    g._core.validate_regenie_config_for_run(config)


def write_toml(config: RegenieConfig, path: Path | str) -> None:
    """Write deterministic TOML."""
    g._core.write_config_toml(config, path)


def dumps_toml(config: RegenieConfig) -> str:
    """Serialize a configuration to TOML."""
    return g._core.dumps_config_toml(config)
