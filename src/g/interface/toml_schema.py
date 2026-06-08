"""Typed TOML schema for the REGENIE-compatible interface."""

from __future__ import annotations

import typing

import msgspec
import msgspec.inspect

type TomlStringList = str | list[str] | msgspec.UnsetType
type TomlValue = str | int | float | bool | list[str] | msgspec.UnsetType


class InputToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Input-file TOML section."""

    bgen: str | msgspec.UnsetType = msgspec.UNSET
    sample: str | msgspec.UnsetType = msgspec.UNSET
    pheno_file: str | msgspec.UnsetType = msgspec.field(default=msgspec.UNSET, name="phenoFile")
    pheno_col: TomlStringList = msgspec.field(default=msgspec.UNSET, name="phenoCol")
    pheno_col_list: TomlStringList = msgspec.field(default=msgspec.UNSET, name="phenoColList")
    covar_file: str | msgspec.UnsetType = msgspec.field(default=msgspec.UNSET, name="covarFile")
    covar_col: TomlStringList = msgspec.field(default=msgspec.UNSET, name="covarCol")
    covar_col_list: TomlStringList = msgspec.field(default=msgspec.UNSET, name="covarColList")
    pred: str | msgspec.UnsetType = msgspec.UNSET
    bed: str | msgspec.UnsetType = msgspec.UNSET
    pgen: str | msgspec.UnsetType = msgspec.UNSET
    cat_covar_list: TomlStringList = msgspec.field(default=msgspec.UNSET, name="catCovarList")


class FiltersToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Sample and variant filter TOML section."""

    keep: str | msgspec.UnsetType = msgspec.UNSET
    remove: str | msgspec.UnsetType = msgspec.UNSET
    extract: str | msgspec.UnsetType = msgspec.UNSET
    exclude: str | msgspec.UnsetType = msgspec.UNSET


class TraitToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Trait-family TOML section."""

    step: int | msgspec.UnsetType = msgspec.UNSET
    qt: bool | msgspec.UnsetType = msgspec.UNSET
    bt: bool | msgspec.UnsetType = msgspec.UNSET
    bsize: int | msgspec.UnsetType = msgspec.UNSET
    threads: int | msgspec.UnsetType = msgspec.UNSET
    test: str | msgspec.UnsetType = msgspec.UNSET
    t2e: bool | msgspec.UnsetType = msgspec.UNSET


class BinaryToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Binary-trait TOML section."""

    firth: bool | msgspec.UnsetType = msgspec.UNSET
    approx: bool | msgspec.UnsetType = msgspec.UNSET
    p_threshold: float | msgspec.UnsetType = msgspec.field(default=msgspec.UNSET, name="pThresh")
    firth_se: bool | msgspec.UnsetType = msgspec.field(default=msgspec.UNSET, name="firth-se")
    spa: bool | msgspec.UnsetType = msgspec.UNSET


class OutputToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Output TOML section."""

    out: str | msgspec.UnsetType = msgspec.UNSET


class GComputeToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True, rename="kebab"):
    """Engine compute TOML section."""

    device: str | msgspec.UnsetType = msgspec.UNSET
    staging_depth: int | msgspec.UnsetType = msgspec.UNSET
    result_in_flight_limit: int | msgspec.UnsetType = msgspec.UNSET
    dosage_buffer_limit: int | msgspec.UnsetType = msgspec.UNSET
    variant_limit: int | msgspec.UnsetType = msgspec.UNSET
    trusted_no_missing_diploid: bool | msgspec.UnsetType = msgspec.UNSET
    trusted_bgen_validation_mode: str | msgspec.UnsetType = msgspec.UNSET
    sample_key_mode: str | msgspec.UnsetType = msgspec.UNSET
    multi_phenotype_sample_mode: str | msgspec.UnsetType = msgspec.UNSET
    firth_batch_size: int | msgspec.UnsetType = msgspec.UNSET
    firth_candidate_capacity: int | msgspec.UnsetType = msgspec.UNSET
    binary_null_maximum_iterations: int | msgspec.UnsetType = msgspec.UNSET
    binary_null_coefficient_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    null_logistic_nonconvergence: str | msgspec.UnsetType = msgspec.UNSET
    binary_minimum_probability: float | msgspec.UnsetType = msgspec.UNSET
    binary_minimum_variance: float | msgspec.UnsetType = msgspec.UNSET
    binary_relative_variance_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    linear_minimum_variance: float | msgspec.UnsetType = msgspec.UNSET
    linear_relative_variance_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    firth_maximum_iterations: int | msgspec.UnsetType = msgspec.UNSET
    firth_gradient_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    firth_coefficient_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    firth_likelihood_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    firth_maximum_step_size: float | msgspec.UnsetType = msgspec.UNSET
    firth_pseudo_maximum_iterations: int | msgspec.UnsetType = msgspec.UNSET
    firth_pseudo_inner_maximum_iterations: int | msgspec.UnsetType = msgspec.UNSET
    firth_newton_raphson_zero_start_iterations: int | msgspec.UnsetType = msgspec.UNSET
    firth_line_search_maximum_attempts: int | msgspec.UnsetType = msgspec.UNSET
    firth_step_halving_maximum_attempts: int | msgspec.UnsetType = msgspec.UNSET
    firth_initial_response_scale: float | msgspec.UnsetType = msgspec.UNSET
    firth_sparse_carrier_dosage_threshold: float | msgspec.UnsetType = msgspec.UNSET
    firth_step_halving_scale: float | msgspec.UnsetType = msgspec.UNSET
    null_firth_maximum_iterations: int | msgspec.UnsetType = msgspec.UNSET
    null_firth_gradient_tolerance: float | msgspec.UnsetType = msgspec.UNSET
    null_firth_maximum_step_size: float | msgspec.UnsetType = msgspec.UNSET
    null_firth_fallback_iteration_multiplier: int | msgspec.UnsetType = msgspec.UNSET
    null_firth_fallback_step_divisor: float | msgspec.UnsetType = msgspec.UNSET
    null_firth_line_search_maximum_attempts: int | msgspec.UnsetType = msgspec.UNSET
    null_firth_step_halving_scale: float | msgspec.UnsetType = msgspec.UNSET
    use_block_firth_math: bool | msgspec.UnsetType = msgspec.UNSET
    bgen_decode_tile_variant_count: int | msgspec.UnsetType = msgspec.UNSET
    gpu_genotype_format: str | msgspec.UnsetType = msgspec.UNSET
    score_dtype: str | msgspec.UnsetType = msgspec.UNSET
    firth_dtype: str | msgspec.UnsetType = msgspec.UNSET
    jax_cache_dir: str | msgspec.UnsetType = msgspec.UNSET
    jax_matmul_precision: str | msgspec.UnsetType = msgspec.UNSET
    jax_persistent_cache: bool | msgspec.UnsetType = msgspec.UNSET
    jax_persistent_cache_min_entry_size_bytes: int | msgspec.UnsetType = msgspec.UNSET
    jax_persistent_cache_min_compile_time_seconds: int | msgspec.UnsetType = msgspec.UNSET
    jax_xla_autotune_cache: bool | msgspec.UnsetType = msgspec.UNSET
    jax_transfer_guard: bool | msgspec.UnsetType = msgspec.UNSET


class GOutputToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True, rename="kebab"):
    """Engine output TOML section."""

    format: str | msgspec.UnsetType = msgspec.UNSET
    output_run_directory: str | msgspec.UnsetType = msgspec.UNSET
    writer_threads: int | msgspec.UnsetType = msgspec.UNSET
    writer_queue_depth: int | msgspec.UnsetType = msgspec.UNSET
    chunks_per_arrow_file: int | msgspec.UnsetType = msgspec.UNSET
    arrow_compression: str | msgspec.UnsetType = msgspec.UNSET
    parquet_compression: str | msgspec.UnsetType = msgspec.UNSET
    resume: bool | msgspec.UnsetType = msgspec.UNSET
    resume_mode: str | msgspec.UnsetType = msgspec.UNSET
    finalize_parquet: bool | msgspec.UnsetType = msgspec.UNSET


class GDiagnosticsToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True, rename="kebab"):
    """Engine diagnostics TOML section."""

    telemetry: str | msgspec.UnsetType = msgspec.UNSET
    log_dir: str | msgspec.UnsetType = msgspec.UNSET
    stage_timings_json: str | msgspec.UnsetType = msgspec.UNSET
    log_filter: str | msgspec.UnsetType = msgspec.UNSET
    log_file: str | msgspec.UnsetType = msgspec.UNSET
    log_stderr: bool | msgspec.UnsetType = msgspec.UNSET
    progress_interval_seconds: float | msgspec.UnsetType = msgspec.UNSET
    progress_interval_chunks: int | msgspec.UnsetType = msgspec.UNSET
    profile_summary_json: str | msgspec.UnsetType = msgspec.UNSET
    trace_file: str | msgspec.UnsetType = msgspec.UNSET
    trace_filter: str | msgspec.UnsetType = msgspec.UNSET
    trace_event_cap: int | msgspec.UnsetType = msgspec.UNSET
    log_queue_size: int | msgspec.UnsetType = msgspec.UNSET
    log_lossy: bool | msgspec.UnsetType = msgspec.UNSET
    include_source_location: bool | msgspec.UnsetType = msgspec.UNSET
    include_span_events: bool | msgspec.UnsetType = msgspec.UNSET


class GNamespaceToml(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Engine-specific TOML namespace."""

    compute: GComputeToml | msgspec.UnsetType = msgspec.UNSET
    output: GOutputToml | msgspec.UnsetType = msgspec.UNSET
    diagnostics: GDiagnosticsToml | msgspec.UnsetType = msgspec.UNSET


class TomlConfig(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Complete public TOML config schema."""

    input: InputToml | msgspec.UnsetType = msgspec.UNSET
    filters: FiltersToml | msgspec.UnsetType = msgspec.UNSET
    trait: TraitToml | msgspec.UnsetType = msgspec.UNSET
    binary: BinaryToml | msgspec.UnsetType = msgspec.UNSET
    output: OutputToml | msgspec.UnsetType = msgspec.UNSET
    g: GNamespaceToml | msgspec.UnsetType = msgspec.UNSET
    metadata: dict[str, typing.Any] | msgspec.UnsetType = msgspec.UNSET


SECTION_STRUCT_TYPES: dict[str, type[msgspec.Struct]] = {
    "input": InputToml,
    "filters": FiltersToml,
    "trait": TraitToml,
    "binary": BinaryToml,
    "output": OutputToml,
    "g.compute": GComputeToml,
    "g.output": GOutputToml,
    "g.diagnostics": GDiagnosticsToml,
}


def schema_toml_paths() -> frozenset[tuple[str, str]]:
    """Return TOML option paths accepted by the typed schema."""
    schema_paths: set[tuple[str, str]] = set()
    for section_name, struct_type in SECTION_STRUCT_TYPES.items():
        struct_information = typing.cast("msgspec.inspect.StructType", msgspec.inspect.type_info(struct_type))
        for field_information in struct_information.fields:
            schema_paths.add((section_name, field_information.encode_name))
    return frozenset(schema_paths)
