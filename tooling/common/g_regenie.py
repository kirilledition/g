"""Typed command and Python API renderers for ``g regenie`` tooling runs."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path


class RegenieTraitKind(enum.StrEnum):
    """Supported REGENIE step 2 trait kinds."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class RegenieDevice(enum.StrEnum):
    """Supported tooling execution devices."""

    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class RegenieInputSpec:
    """Input and output paths for one REGENIE step 2 run.

    Attributes:
        bgen_path: Input BGEN path.
        sample_path: Optional sample path.
        phenotype_path: Phenotype file path.
        phenotype_columns: Phenotype column names.
        covariate_path: Optional covariate file path.
        covariate_columns: Covariate column names.
        prediction_list_path: REGENIE step 1 prediction-list path.
        output_prefix: Output prefix passed to ``--out``.

    """

    bgen_path: Path
    sample_path: Path | None
    phenotype_path: Path
    phenotype_columns: tuple[str, ...]
    covariate_path: Path | None
    covariate_columns: tuple[str, ...]
    prediction_list_path: Path
    output_prefix: Path


@dataclass(frozen=True)
class RegenieComputeOptions:
    """Compute options shared by CLI and Python API renderers.

    Attributes:
        device: Execution device.
        bsize: Variant block size.
        threads: Optional CPU/Rayon thread count.
        staging_depth: Optional native callback staging depth.
        native_callback_batch_size: Optional native callback batch size.
        result_in_flight_limit: Optional output result in-flight limit.
        dosage_buffer_limit: Optional dosage buffer limit.
        variant_limit: Optional variant cap.
        trusted_no_missing_diploid: Optional trusted BGEN fast-path setting.
        trusted_bgen_validation_mode: Optional trusted BGEN validation mode.
        bgen_decode_tile_variant_count: Optional BGEN decode tile size.
        firth_batch_size: Optional Firth batch size.
        firth_candidate_capacity: Optional Firth candidate capacity.
        gpu_genotype_format: Optional GPU genotype representation.
        jax_cache_dir: Optional persistent-cache directory.
        jax_persistent_cache: Optional persistent-cache toggle.
        jax_persistent_cache_min_entry_size_bytes: Optional persistent-cache entry threshold.
        jax_persistent_cache_min_compile_time_seconds: Optional persistent-cache compile-time threshold.
        jax_xla_autotune_cache: Optional XLA autotune-cache toggle.

    """

    device: RegenieDevice
    bsize: int
    threads: int | None
    staging_depth: int | None
    native_callback_batch_size: int | None
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    variant_limit: int | None
    trusted_no_missing_diploid: bool | None
    trusted_bgen_validation_mode: str | None
    bgen_decode_tile_variant_count: int | None
    firth_batch_size: int | None
    firth_candidate_capacity: int | None
    gpu_genotype_format: str | None
    jax_cache_dir: Path | None
    jax_persistent_cache: bool | None
    jax_persistent_cache_min_entry_size_bytes: int | None
    jax_persistent_cache_min_compile_time_seconds: int | None
    jax_xla_autotune_cache: bool | None


@dataclass(frozen=True)
class RegenieOutputOptions:
    """Output options shared by CLI and Python API renderers.

    Attributes:
        output_format: Optional output format.
        output_run_directory: Optional explicit output run directory.
        writer_threads: Optional output writer thread count.
        writer_queue_depth: Optional output writer queue depth.
        chunks_per_arrow_file: Optional Arrow chunk grouping.
        arrow_compression: Optional Arrow compression codec.
        parquet_compression: Optional Parquet compression codec.
        output_statistic_dtype: Optional output statistic dtype.
        finalize_parquet: Optional finalization toggle.

    """

    output_format: str | None
    output_run_directory: Path | None
    writer_threads: int | None
    writer_queue_depth: int | None
    chunks_per_arrow_file: int | None
    arrow_compression: str | None
    parquet_compression: str | None
    output_statistic_dtype: str | None
    finalize_parquet: bool | None


@dataclass(frozen=True)
class RegenieDiagnosticsOptions:
    """Diagnostics options shared by CLI and Python API renderers.

    Attributes:
        telemetry: Optional telemetry mode.
        log_dir: Optional log directory.
        stage_timings_json: Optional stage timing JSON path.
        profile_summary_json: Optional profile summary JSON path.
        log_file: Optional structured log file path.
        log_filter: Optional log filter.
        log_stderr: Optional stderr logging toggle.
        progress_interval_seconds: Optional progress interval.
        progress_interval_chunks: Optional progress chunk interval.

    """

    telemetry: str | None
    log_dir: Path | None
    stage_timings_json: Path | None
    profile_summary_json: Path | None
    log_file: Path | None
    log_filter: str | None
    log_stderr: bool | None
    progress_interval_seconds: float | None
    progress_interval_chunks: int | None


@dataclass(frozen=True)
class RegenieBinaryOptions:
    """Binary-trait options for one REGENIE step 2 run.

    Attributes:
        firth: Whether to enable Firth correction.
        approx: Whether to enable approximate Firth.
        firth_se: Optional Firth standard-error toggle.
        p_threshold: Optional binary p-value threshold.

    """

    firth: bool
    approx: bool
    firth_se: bool | None
    p_threshold: float | None


@dataclass(frozen=True)
class RegenieRunSpec:
    """Complete REGENIE run specification for tooling.

    Attributes:
        trait_kind: Trait kind.
        command_prefix: Command prefix for CLI rendering, for example ``("g", "regenie")``.
        inputs: Input and output paths.
        compute: Compute options.
        output: Output options.
        diagnostics: Diagnostics options.
        binary: Binary options when ``trait_kind`` is binary.

    """

    trait_kind: RegenieTraitKind
    command_prefix: tuple[str, ...]
    inputs: RegenieInputSpec
    compute: RegenieComputeOptions
    output: RegenieOutputOptions
    diagnostics: RegenieDiagnosticsOptions
    binary: RegenieBinaryOptions | None


def render_g_regenie_cli(spec: RegenieRunSpec) -> list[str]:
    """Render a run spec as shell-free ``g regenie`` CLI arguments.

    Args:
        spec: REGENIE run specification.

    Returns:
        Command argument vector.

    Raises:
        ValueError: If the spec is inconsistent.

    """
    validate_regenie_run_spec(spec)
    command_arguments = list(spec.command_prefix)
    append_option(command_arguments, "--step", 2)
    append_option(command_arguments, "--bgen", spec.inputs.bgen_path)
    append_optional_option(command_arguments, "--sample", spec.inputs.sample_path)
    append_option(command_arguments, "--phenoFile", spec.inputs.phenotype_path)
    append_column_options(command_arguments, "--phenoCol", "--phenoColList", spec.inputs.phenotype_columns)
    append_optional_option(command_arguments, "--covarFile", spec.inputs.covariate_path)
    if spec.inputs.covariate_columns:
        append_option(command_arguments, "--covarColList", ",".join(spec.inputs.covariate_columns))
    append_option(command_arguments, "--pred", spec.inputs.prediction_list_path)
    append_option(command_arguments, "--out", spec.inputs.output_prefix)
    append_option(command_arguments, "--device", spec.compute.device.value)
    append_option(command_arguments, "--bsize", spec.compute.bsize)
    append_trait_cli_arguments(command_arguments, spec)
    append_compute_cli_arguments(command_arguments, spec.compute)
    append_output_cli_arguments(command_arguments, spec.output)
    append_diagnostics_cli_arguments(command_arguments, spec.diagnostics)
    return command_arguments


def render_python_api_options(spec: RegenieRunSpec) -> dict[str, object]:
    """Render a run spec as ``api.regenie.from_options`` payload.

    Args:
        spec: REGENIE run specification.

    Returns:
        Python options dictionary.

    Raises:
        ValueError: If the spec is inconsistent.

    """
    validate_regenie_run_spec(spec)
    options: dict[str, object] = {
        "step": 2,
        "bgen": spec.inputs.bgen_path,
        "phenoFile": spec.inputs.phenotype_path,
        "pred": spec.inputs.prediction_list_path,
        "out": spec.inputs.output_prefix,
        "bsize": spec.compute.bsize,
        "compute": build_compute_options(spec.compute),
        "output": build_output_options(spec.output),
        "diagnostics": build_diagnostics_options(spec.diagnostics),
    }
    if spec.inputs.sample_path is not None:
        options["sample"] = spec.inputs.sample_path
    if spec.inputs.covariate_path is not None:
        options["covarFile"] = spec.inputs.covariate_path
    if spec.inputs.covariate_columns:
        options["covarColList"] = ",".join(spec.inputs.covariate_columns)
    if len(spec.inputs.phenotype_columns) == 1:
        options["phenoCol"] = spec.inputs.phenotype_columns[0]
    else:
        options["phenoColList"] = ",".join(spec.inputs.phenotype_columns)
    if spec.compute.threads is not None:
        options["threads"] = spec.compute.threads
    if spec.trait_kind == RegenieTraitKind.BINARY:
        options["bt"] = True
        if spec.binary is not None:
            options.update(build_binary_options(spec.binary))
    else:
        options["qt"] = True
    return options


def validate_regenie_run_spec(spec: RegenieRunSpec) -> None:
    """Validate basic run-spec invariants.

    Args:
        spec: REGENIE run specification.

    Raises:
        ValueError: If the spec is inconsistent.

    """
    if not spec.command_prefix:
        message = "REGENIE CLI command_prefix must not be empty."
        raise ValueError(message)
    if not spec.inputs.phenotype_columns:
        message = "REGENIE run spec must include at least one phenotype column."
        raise ValueError(message)
    if any(not phenotype_column.strip() for phenotype_column in spec.inputs.phenotype_columns):
        message = "REGENIE phenotype columns must not contain empty values."
        raise ValueError(message)
    if spec.trait_kind == RegenieTraitKind.BINARY and spec.binary is None:
        message = "Binary REGENIE run spec requires binary options."
        raise ValueError(message)
    if spec.trait_kind == RegenieTraitKind.QUANTITATIVE and spec.binary is not None:
        message = "Quantitative REGENIE run spec must not include binary options."
        raise ValueError(message)


def append_option(command_arguments: list[str], option_name: str, value: object) -> None:
    """Append an option and value.

    Args:
        command_arguments: Command argument vector to mutate.
        option_name: Option name.
        value: Option value.

    """
    command_arguments.extend([option_name, str(value)])


def append_optional_option(command_arguments: list[str], option_name: str, value: object | None) -> None:
    """Append an option only when a value is present.

    Args:
        command_arguments: Command argument vector to mutate.
        option_name: Option name.
        value: Optional option value.

    """
    if value is not None:
        append_option(command_arguments, option_name, value)


def append_optional_boolean_flag(
    command_arguments: list[str],
    *,
    value: bool | None,
    enabled_flag: str,
    disabled_flag: str | None,
) -> None:
    """Append an explicit boolean flag when a value is present.

    Args:
        command_arguments: Command argument vector to mutate.
        value: Optional boolean value.
        enabled_flag: Flag for ``True``.
        disabled_flag: Optional flag for ``False``.

    """
    if value is None:
        return
    if value:
        command_arguments.append(enabled_flag)
    elif disabled_flag is not None:
        command_arguments.append(disabled_flag)


def append_column_options(
    command_arguments: list[str],
    single_option_name: str,
    list_option_name: str,
    columns: tuple[str, ...],
) -> None:
    """Append single-column or list-column options.

    Args:
        command_arguments: Command argument vector to mutate.
        single_option_name: Option for one column.
        list_option_name: Option for many columns.
        columns: Column names.

    """
    if len(columns) == 1:
        append_option(command_arguments, single_option_name, columns[0])
        return
    append_option(command_arguments, list_option_name, ",".join(columns))


def append_trait_cli_arguments(command_arguments: list[str], spec: RegenieRunSpec) -> None:
    """Append trait-specific CLI arguments.

    Args:
        command_arguments: Command argument vector to mutate.
        spec: REGENIE run specification.

    """
    if spec.trait_kind == RegenieTraitKind.BINARY:
        command_arguments.append("--bt")
        if spec.binary is not None:
            append_optional_boolean_flag(
                command_arguments,
                value=spec.binary.firth,
                enabled_flag="--firth",
                disabled_flag=None,
            )
            append_optional_boolean_flag(
                command_arguments,
                value=spec.binary.approx,
                enabled_flag="--approx",
                disabled_flag=None,
            )
            append_optional_boolean_flag(
                command_arguments,
                value=spec.binary.firth_se,
                enabled_flag="--firth-se",
                disabled_flag="--no-firth-se",
            )
            append_optional_option(command_arguments, "--pThresh", spec.binary.p_threshold)
        return
    command_arguments.append("--qt")


def append_compute_cli_arguments(command_arguments: list[str], compute_options: RegenieComputeOptions) -> None:
    """Append compute CLI arguments.

    Args:
        command_arguments: Command argument vector to mutate.
        compute_options: Compute options.

    """
    append_optional_option(command_arguments, "--threads", compute_options.threads)
    append_optional_option(command_arguments, "--staging_depth", compute_options.staging_depth)
    append_optional_option(
        command_arguments, "--native_callback_batch_size", compute_options.native_callback_batch_size
    )
    append_optional_option(command_arguments, "--result_in_flight_limit", compute_options.result_in_flight_limit)
    append_optional_option(command_arguments, "--dosage_buffer_limit", compute_options.dosage_buffer_limit)
    append_optional_option(command_arguments, "--variant_limit", compute_options.variant_limit)
    append_optional_boolean_flag(
        command_arguments,
        value=compute_options.trusted_no_missing_diploid,
        enabled_flag="--trusted_no_missing_diploid",
        disabled_flag="--no-trusted_no_missing_diploid",
    )
    append_optional_option(
        command_arguments,
        "--trusted_bgen_validation_mode",
        compute_options.trusted_bgen_validation_mode,
    )
    append_optional_option(
        command_arguments,
        "--bgen_decode_tile_variant_count",
        compute_options.bgen_decode_tile_variant_count,
    )
    append_optional_option(command_arguments, "--firth_batch_size", compute_options.firth_batch_size)
    append_optional_option(command_arguments, "--firth_candidate_capacity", compute_options.firth_candidate_capacity)
    append_optional_option(command_arguments, "--gpu_genotype_format", compute_options.gpu_genotype_format)
    append_optional_boolean_flag(
        command_arguments,
        value=compute_options.jax_persistent_cache,
        enabled_flag="--jax_persistent_cache",
        disabled_flag="--no-jax_persistent_cache",
    )
    append_optional_option(command_arguments, "--jax_cache_dir", compute_options.jax_cache_dir)
    append_optional_option(
        command_arguments,
        "--jax_persistent_cache_min_entry_size_bytes",
        compute_options.jax_persistent_cache_min_entry_size_bytes,
    )
    append_optional_option(
        command_arguments,
        "--jax_persistent_cache_min_compile_time_seconds",
        compute_options.jax_persistent_cache_min_compile_time_seconds,
    )
    append_optional_boolean_flag(
        command_arguments,
        value=compute_options.jax_xla_autotune_cache,
        enabled_flag="--jax_xla_autotune_cache",
        disabled_flag="--no-jax_xla_autotune_cache",
    )


def append_output_cli_arguments(command_arguments: list[str], output_options: RegenieOutputOptions) -> None:
    """Append output CLI arguments.

    Args:
        command_arguments: Command argument vector to mutate.
        output_options: Output options.

    """
    append_optional_option(command_arguments, "--format", output_options.output_format)
    append_optional_option(command_arguments, "--output_run_directory", output_options.output_run_directory)
    append_optional_option(command_arguments, "--writer_threads", output_options.writer_threads)
    append_optional_option(command_arguments, "--writer_queue_depth", output_options.writer_queue_depth)
    append_optional_option(command_arguments, "--chunks_per_arrow_file", output_options.chunks_per_arrow_file)
    append_optional_option(command_arguments, "--arrow_compression", output_options.arrow_compression)
    append_optional_option(command_arguments, "--parquet_compression", output_options.parquet_compression)
    append_optional_option(command_arguments, "--output_statistic_dtype", output_options.output_statistic_dtype)
    append_optional_boolean_flag(
        command_arguments,
        value=output_options.finalize_parquet,
        enabled_flag="--finalize_parquet",
        disabled_flag="--no-finalize_parquet",
    )


def append_diagnostics_cli_arguments(
    command_arguments: list[str],
    diagnostics_options: RegenieDiagnosticsOptions,
) -> None:
    """Append diagnostics CLI arguments.

    Args:
        command_arguments: Command argument vector to mutate.
        diagnostics_options: Diagnostics options.

    """
    append_optional_option(command_arguments, "--telemetry", diagnostics_options.telemetry)
    append_optional_option(command_arguments, "--log_dir", diagnostics_options.log_dir)
    append_optional_option(command_arguments, "--stage_timings_json", diagnostics_options.stage_timings_json)
    append_optional_option(command_arguments, "--profile_summary_json", diagnostics_options.profile_summary_json)
    append_optional_option(command_arguments, "--log_file", diagnostics_options.log_file)
    append_optional_option(command_arguments, "--log_filter", diagnostics_options.log_filter)
    append_optional_boolean_flag(
        command_arguments,
        value=diagnostics_options.log_stderr,
        enabled_flag="--log_stderr",
        disabled_flag="--no-log_stderr",
    )
    append_optional_option(
        command_arguments,
        "--progress_interval_seconds",
        diagnostics_options.progress_interval_seconds,
    )
    append_optional_option(
        command_arguments,
        "--progress_interval_chunks",
        diagnostics_options.progress_interval_chunks,
    )


def build_compute_options(compute_options: RegenieComputeOptions) -> dict[str, object]:
    """Build Python API compute options.

    Args:
        compute_options: Compute options.

    Returns:
        Python options mapping.

    """
    options: dict[str, object] = {"device": compute_options.device.value}
    add_optional_option(options, "staging_depth", compute_options.staging_depth)
    add_optional_option(options, "native_callback_batch_size", compute_options.native_callback_batch_size)
    add_optional_option(options, "result_in_flight_limit", compute_options.result_in_flight_limit)
    add_optional_option(options, "dosage_buffer_limit", compute_options.dosage_buffer_limit)
    add_optional_option(options, "variant_limit", compute_options.variant_limit)
    add_optional_option(options, "trusted_no_missing_diploid", compute_options.trusted_no_missing_diploid)
    add_optional_option(options, "trusted_bgen_validation_mode", compute_options.trusted_bgen_validation_mode)
    add_optional_option(options, "bgen_decode_tile_variant_count", compute_options.bgen_decode_tile_variant_count)
    add_optional_option(options, "firth_batch_size", compute_options.firth_batch_size)
    add_optional_option(options, "firth_candidate_capacity", compute_options.firth_candidate_capacity)
    add_optional_option(options, "gpu_genotype_format", compute_options.gpu_genotype_format)
    add_optional_option(options, "jax_cache_dir", compute_options.jax_cache_dir)
    add_optional_option(options, "jax_persistent_cache", compute_options.jax_persistent_cache)
    add_optional_option(
        options,
        "jax_persistent_cache_min_entry_size_bytes",
        compute_options.jax_persistent_cache_min_entry_size_bytes,
    )
    add_optional_option(
        options,
        "jax_persistent_cache_min_compile_time_seconds",
        compute_options.jax_persistent_cache_min_compile_time_seconds,
    )
    add_optional_option(options, "jax_xla_autotune_cache", compute_options.jax_xla_autotune_cache)
    return options


def build_output_options(output_options: RegenieOutputOptions) -> dict[str, object]:
    """Build Python API output options.

    Args:
        output_options: Output options.

    Returns:
        Python options mapping.

    """
    options: dict[str, object] = {}
    add_optional_option(options, "format", output_options.output_format)
    add_optional_option(options, "output_run_directory", output_options.output_run_directory)
    add_optional_option(options, "writer_threads", output_options.writer_threads)
    add_optional_option(options, "writer_queue_depth", output_options.writer_queue_depth)
    add_optional_option(options, "chunks_per_arrow_file", output_options.chunks_per_arrow_file)
    add_optional_option(options, "arrow_compression", output_options.arrow_compression)
    add_optional_option(options, "parquet_compression", output_options.parquet_compression)
    add_optional_option(options, "output_statistic_dtype", output_options.output_statistic_dtype)
    add_optional_option(options, "finalize_parquet", output_options.finalize_parquet)
    return options


def build_diagnostics_options(diagnostics_options: RegenieDiagnosticsOptions) -> dict[str, object]:
    """Build Python API diagnostics options.

    Args:
        diagnostics_options: Diagnostics options.

    Returns:
        Python options mapping.

    """
    options: dict[str, object] = {}
    add_optional_option(options, "telemetry", diagnostics_options.telemetry)
    add_optional_option(options, "log_dir", diagnostics_options.log_dir)
    add_optional_option(options, "stage_timings_json", diagnostics_options.stage_timings_json)
    add_optional_option(options, "profile_summary_json", diagnostics_options.profile_summary_json)
    add_optional_option(options, "log_file", diagnostics_options.log_file)
    add_optional_option(options, "log_filter", diagnostics_options.log_filter)
    add_optional_option(options, "log_stderr", diagnostics_options.log_stderr)
    add_optional_option(options, "progress_interval_seconds", diagnostics_options.progress_interval_seconds)
    add_optional_option(options, "progress_interval_chunks", diagnostics_options.progress_interval_chunks)
    return options


def build_binary_options(binary_options: RegenieBinaryOptions) -> dict[str, object]:
    """Build Python API binary options.

    Args:
        binary_options: Binary options.

    Returns:
        Python options mapping.

    """
    options: dict[str, object] = {
        "firth": binary_options.firth,
        "approx": binary_options.approx,
    }
    add_optional_option(options, "firth_se", binary_options.firth_se)
    add_optional_option(options, "pThresh", binary_options.p_threshold)
    return options


def add_optional_option(options: dict[str, object], option_name: str, value: object | None) -> None:
    """Add one Python API option when present.

    Args:
        options: Options mapping to mutate.
        option_name: Option name.
        value: Optional option value.

    """
    if value is not None:
        options[option_name] = value


def expected_output_run_directory(spec: RegenieRunSpec) -> Path:
    """Infer the default single-trait output run directory.

    Args:
        spec: REGENIE run specification.

    Returns:
        Expected output run directory.

    Raises:
        ValueError: If the spec has multiple phenotypes.

    """
    if spec.output.output_run_directory is not None:
        return spec.output.output_run_directory
    if len(spec.inputs.phenotype_columns) != 1:
        message = "Default output run directory inference only supports one phenotype column."
        raise ValueError(message)
    association_mode = "regenie2_binary" if spec.trait_kind == RegenieTraitKind.BINARY else "regenie2_linear"
    return Path(f"{spec.inputs.output_prefix}.g") / f"{spec.inputs.phenotype_columns[0]}.{association_mode}.run"
