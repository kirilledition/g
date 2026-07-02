"""Immutable execution plans for REGENIE-compatible runs."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.io import output, source

if typing.TYPE_CHECKING:
    from g.interface import config


@dataclass(frozen=True)
class KernelConfig:
    """Engine kernel and batching settings.

    Attributes:
        chunk_size: Variant block size.
        device: JAX device requested for execution.
        staging_depth: Native callback staging depth.
        native_callback_batch_size: Native-to-Python callback chunk batch size.
        result_in_flight_limit: Optional cap for result chunks awaiting materialization.
        dosage_buffer_limit: Optional cap for reusable native dosage decode buffers.
        variant_limit: Optional debug cap on variants.
        thread_count: Requested CPU thread count.
        bgen_decode_tile_variant_count: Native BGEN decode tile variant count.
        gpu_genotype_format: Host-to-device genotype representation for GPU kernels.
        trusted_no_missing_diploid: Whether BGEN records can use the trusted diploid fast path.
        trusted_bgen_validation_mode: Validation policy for trusted BGEN decoding.
        alignment_config: Sample alignment settings consumed by the native dispatcher.
        multi_phenotype_sample_mode: Sample handling for multi-phenotype requests.
        binary_kernel_config: Static binary JAX kernel settings.
        linear_numerical_config: Static linear JAX numerical settings.

    """

    chunk_size: int
    device: types.Device
    staging_depth: int
    native_callback_batch_size: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    variant_limit: int | None
    thread_count: int | None
    bgen_decode_tile_variant_count: int
    gpu_genotype_format: types.GpuGenotypeFormat
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    alignment_config: config.GComputeConfig
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode
    binary_kernel_config: regenie2_binary_config.BinaryKernelConfig | None
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None


@dataclass(frozen=True)
class OutputPlan:
    """Output materialization and writer settings.

    Attributes:
        output_prefix: User-facing output prefix.
        output_run_root: Root directory for per-phenotype chunked runs.
        resume: Whether to resume a previous run.
        resume_mode: Resume validation mode.
        writer_settings: Output writer and finalization settings.

    """

    output_prefix: Path
    output_run_root: Path
    resume: bool
    resume_mode: types.ResumeMode
    writer_settings: output.OutputWriterSettings


@dataclass(frozen=True)
class PhenotypeRunPlan:
    """Prepared run state for one phenotype.

    Attributes:
        phenotype_name: Phenotype column name.
        output_run_paths: Chunked output paths for the phenotype.
        existing_manifest: Existing manifest loaded for resume, if present.
        effective_config_path: Path where the effective TOML config is written.

    """

    phenotype_name: str
    output_run_paths: output.OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None
    effective_config_path: Path


@dataclass(frozen=True)
class PhenotypeComputeGroup:
    """Planned phenotype group that can share one compute delivery.

    Attributes:
        group_mode: Grouping mode selected by planning or alignment.
        phenotype_indices: Zero-based phenotype indices in run/output order.
        phenotype_names: Phenotype names in compute order.
        sample_mode: Sample handling semantics for this group.
        sample_set_fingerprint: Stable sample-set fingerprint when alignment is available.
        covariate_design_fingerprint: Stable covariate design fingerprint when alignment is available.
        prediction_alignment_fingerprint: Stable prediction alignment fingerprint when available.

    """

    group_mode: types.PhenotypeComputeGroupMode
    phenotype_indices: tuple[int, ...]
    phenotype_names: tuple[str, ...]
    sample_mode: types.MultiPhenotypeSampleMode
    sample_set_fingerprint: str | None
    covariate_design_fingerprint: str | None
    prediction_alignment_fingerprint: str | None


def build_phenotype_compute_group_id(phenotype_compute_group: PhenotypeComputeGroup) -> str:
    """Build a deterministic identifier for a resolved phenotype compute group."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    return native_host_planning_policy.build_phenotype_compute_group_id_value(
        phenotype_compute_group.group_mode.value,
        phenotype_compute_group.phenotype_indices,
        phenotype_compute_group.phenotype_names,
        phenotype_compute_group.sample_mode.value,
        phenotype_compute_group.sample_set_fingerprint,
        phenotype_compute_group.covariate_design_fingerprint,
        phenotype_compute_group.prediction_alignment_fingerprint,
    )


@dataclass(frozen=True)
class RegenieExecutionPlan:
    """Complete immutable execution plan for one REGENIE-compatible request.

    Attributes:
        association_mode: Statistical association engine to run.
        genotype_source_config: BGEN source configuration.
        phenotype_path: Phenotype table path.
        prediction_list_path: REGENIE step 1 prediction list.
        covariate_path: Optional covariate table path.
        covariate_names: Optional covariate column names.
        phenotype_run_plans: Per-phenotype output and manifest plans.
        phenotype_compute_groups: Planned phenotype compute groups.
        binary_correction_plan: Normalized binary fallback settings.
        kernel_config: Engine kernel and batching settings.
        output_plan: Output materialization settings.
        stage_timings_json: Optional stage timing diagnostics path.

    """

    association_mode: types.AssociationMode
    genotype_source_config: source.GenotypeSourceConfig
    phenotype_path: Path
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    phenotype_run_plans: tuple[PhenotypeRunPlan, ...]
    phenotype_compute_groups: tuple[PhenotypeComputeGroup, ...]
    binary_correction_plan: types.BinaryCorrectionPlan
    kernel_config: KernelConfig
    output_plan: OutputPlan
    stage_timings_json: Path | None


def normalize_binary_correction_config(binary_config: config.BinaryConfig) -> types.BinaryCorrectionPlan:
    """Normalize REGENIE-style binary correction flags into an internal plan."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    correction_payload = native_host_planning_policy.normalize_binary_correction_payload(
        binary_config.firth,
        binary_config.approx,
        binary_config.spa,
        binary_config.p_threshold,
        binary_config.firth_se,
    )
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod(typing.cast("str", correction_payload["method"])),
        p_threshold=typing.cast("float", correction_payload["p_threshold"]),
        firth_se=typing.cast("bool", correction_payload["firth_se"]),
    )


def build_binary_kernel_config(compute_config: config.GComputeConfig) -> regenie2_binary_config.BinaryKernelConfig:
    """Build immutable binary JAX kernel settings from public compute config."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=compute_config.binary_minimum_probability,
            minimum_variance=compute_config.binary_minimum_variance,
            relative_variance_tolerance=compute_config.binary_relative_variance_tolerance,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=compute_config.binary_null_maximum_iterations,
            coefficient_tolerance=compute_config.binary_null_coefficient_tolerance,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=compute_config.firth_batch_size,
            candidate_capacity=compute_config.firth_candidate_capacity,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=compute_config.firth_maximum_iterations,
            gradient_tolerance=compute_config.firth_gradient_tolerance,
            coefficient_tolerance=compute_config.firth_coefficient_tolerance,
            likelihood_tolerance=compute_config.firth_likelihood_tolerance,
            maximum_step_size=compute_config.firth_maximum_step_size,
            pseudo_maximum_iterations=compute_config.firth_pseudo_maximum_iterations,
            pseudo_inner_maximum_iterations=compute_config.firth_pseudo_inner_maximum_iterations,
            newton_raphson_zero_start_iterations=compute_config.firth_newton_raphson_zero_start_iterations,
            line_search_maximum_attempts=compute_config.firth_line_search_maximum_attempts,
            step_halving_maximum_attempts=compute_config.firth_step_halving_maximum_attempts,
            initial_response_scale=compute_config.firth_initial_response_scale,
            sparse_carrier_dosage_threshold=compute_config.firth_sparse_carrier_dosage_threshold,
            step_halving_scale=compute_config.firth_step_halving_scale,
            use_block_math=compute_config.use_block_firth_math,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=compute_config.null_firth_maximum_iterations,
            gradient_tolerance=compute_config.null_firth_gradient_tolerance,
            maximum_step_size=compute_config.null_firth_maximum_step_size,
            fallback_iteration_multiplier=compute_config.null_firth_fallback_iteration_multiplier,
            fallback_step_divisor=compute_config.null_firth_fallback_step_divisor,
            line_search_maximum_attempts=compute_config.null_firth_line_search_maximum_attempts,
            step_halving_scale=compute_config.null_firth_step_halving_scale,
        ),
    )


def build_regenie_execution_plan(
    regenie_config: config.RegenieConfig,
    *,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> RegenieExecutionPlan:
    """Build a complete execution plan from a validated public config."""
    run_request = compile_run_request_payload(regenie_config)
    input_request = require_mapping(run_request, "input")
    association_mode = types.AssociationMode(typing.cast("str", run_request["association_mode"]))
    output_plan = build_output_plan_from_run_request(run_request)
    kernel_config = build_kernel_config_from_run_request(regenie_config, run_request)
    phenotype_run_plans = tuple(
        build_phenotype_run_plan_from_request(
            phenotype_run_request=phenotype_run_request,
            association_mode=association_mode,
            output_plan=output_plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        for phenotype_run_request in require_mapping_sequence(run_request, "phenotype_runs")
    )
    return RegenieExecutionPlan(
        association_mode=association_mode,
        genotype_source_config=source.GenotypeSourceConfig(
            source_path=Path(typing.cast("str", input_request["bgen_path"])),
            sample_path=optional_path_from_request(input_request["sample_path"]),
        ),
        phenotype_path=Path(typing.cast("str", input_request["phenotype_path"])),
        prediction_list_path=Path(typing.cast("str", input_request["prediction_list_path"])),
        covariate_path=optional_path_from_request(input_request["covariate_path"]),
        covariate_names=string_tuple_from_request(input_request["covariate_names"]) or None,
        phenotype_run_plans=phenotype_run_plans,
        phenotype_compute_groups=tuple(
            adapt_phenotype_compute_group_payload(group_payload)
            for group_payload in require_mapping_sequence(run_request, "phenotype_compute_groups")
        ),
        binary_correction_plan=adapt_binary_correction_plan(require_mapping(run_request, "correction")),
        kernel_config=kernel_config,
        output_plan=output_plan,
        stage_timings_json=optional_path_from_request(run_request["stage_timings_json"]),
    )


def compile_run_request_payload(regenie_config: config.RegenieConfig) -> dict[str, typing.Any]:
    """Compile a resolved config into the native requested-run payload."""
    payload = _core.compile_run_request_payload(regenie_config)
    if not isinstance(payload, dict):
        message = "Native run request payload must be a JSON object."
        raise TypeError(message)
    return typing.cast("dict[str, typing.Any]", payload)


def build_output_plan_from_run_request(run_request: dict[str, typing.Any]) -> OutputPlan:
    """Adapt the native output writer plan into the existing Python dataclass."""
    output_request = require_mapping(run_request, "output")
    return OutputPlan(
        output_prefix=Path(typing.cast("str", output_request["output_prefix"])),
        output_run_root=Path(typing.cast("str", output_request["output_run_root"])),
        resume=typing.cast("bool", output_request["resume"]),
        resume_mode=types.ResumeMode(typing.cast("str", output_request["resume_mode"])),
        writer_settings=output.OutputWriterSettings(
            finalize_parquet=typing.cast("bool", output_request["finalize_parquet"]),
            writer_thread_count=typing.cast("int", output_request["writer_thread_count"]),
            writer_queue_depth=typing.cast("int", output_request["writer_queue_depth"]),
            chunks_per_arrow_file=typing.cast("int", output_request["chunks_per_arrow_file"]),
            arrow_compression=types.ArrowCompression(typing.cast("str", output_request["arrow_compression"])),
            parquet_compression=types.ParquetCompression(typing.cast("str", output_request["parquet_compression"])),
            output_format=types.OutputFormat(typing.cast("str", output_request["output_format"])),
            output_statistic_dtype=types.FloatingPointDtype(
                typing.cast("str", output_request["output_statistic_dtype"])
            ),
        ),
    )


def build_kernel_config_from_run_request(
    regenie_config: config.RegenieConfig,
    run_request: dict[str, typing.Any],
) -> KernelConfig:
    """Adapt native requested-run compute fields into the existing kernel config."""
    compute_request = require_mapping(run_request, "compute")
    trait_request = require_mapping(run_request, "trait_request")
    trait_type = types.RegenieTraitType(typing.cast("str", trait_request["trait_type"]))
    return KernelConfig(
        chunk_size=typing.cast("int", trait_request["chunk_size"]),
        device=types.Device(typing.cast("str", compute_request["device"])),
        staging_depth=typing.cast("int", compute_request["staging_depth"]),
        native_callback_batch_size=typing.cast("int", compute_request["native_callback_batch_size"]),
        result_in_flight_limit=typing.cast("int | None", compute_request["result_in_flight_limit"]),
        dosage_buffer_limit=typing.cast("int | None", compute_request["dosage_buffer_limit"]),
        variant_limit=typing.cast("int | None", compute_request["variant_limit"]),
        thread_count=typing.cast("int | None", trait_request["thread_count"]),
        bgen_decode_tile_variant_count=typing.cast("int", compute_request["bgen_decode_tile_variant_count"]),
        gpu_genotype_format=types.GpuGenotypeFormat(
            typing.cast("str", compute_request["requested_gpu_genotype_format"])
        ),
        trusted_no_missing_diploid=typing.cast("bool", compute_request["trusted_no_missing_diploid"]),
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode(
            typing.cast("str", compute_request["trusted_bgen_validation_mode"])
        ),
        alignment_config=regenie_config.g_compute,
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode(
            typing.cast("str", compute_request["multi_phenotype_sample_mode"])
        ),
        binary_kernel_config=(
            build_binary_kernel_config(regenie_config.g_compute)
            if trait_type == types.RegenieTraitType.BINARY
            else None
        ),
        linear_numerical_config=(
            regenie2_linear_config.LinearNumericalConfig(
                minimum_variance=regenie_config.g_compute.linear_minimum_variance,
                relative_variance_tolerance=regenie_config.g_compute.linear_relative_variance_tolerance,
            )
            if trait_type == types.RegenieTraitType.QUANTITATIVE
            else None
        ),
    )


def build_phenotype_run_plan_from_request(
    *,
    phenotype_run_request: dict[str, typing.Any],
    association_mode: types.AssociationMode,
    output_plan: OutputPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> PhenotypeRunPlan:
    """Prepare output paths from one native phenotype run request."""
    prepared_output_run = output.prepare_output_run(
        output_root=output_plan.output_run_root / typing.cast("str", phenotype_run_request["output_directory_name"]),
        association_mode=association_mode,
        output_format=output_plan.writer_settings.output_format,
        resume=output_plan.resume,
        resume_mode=output_plan.resume_mode,
        runtime_compatibility_token=runtime_compatibility_token,
    )
    return PhenotypeRunPlan(
        phenotype_name=typing.cast("str", phenotype_run_request["phenotype_name"]),
        output_run_paths=prepared_output_run.output_run_paths,
        existing_manifest=prepared_output_run.existing_manifest,
        effective_config_path=prepared_output_run.output_run_paths.run_directory / "effective_config.toml",
    )


def adapt_binary_correction_plan(correction_payload: dict[str, typing.Any]) -> types.BinaryCorrectionPlan:
    """Adapt native correction payload to the existing Python correction plan."""
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod(typing.cast("str", correction_payload["method"])),
        p_threshold=typing.cast("float", correction_payload["p_threshold"]),
        firth_se=typing.cast("bool", correction_payload["firth_se"]),
    )


def build_phenotype_compute_groups(
    *,
    phenotype_names: tuple[str, ...],
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode,
) -> tuple[PhenotypeComputeGroup, ...]:
    """Build config-time phenotype compute groups."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    return tuple(
        adapt_phenotype_compute_group_payload(group_payload)
        for group_payload in native_host_planning_policy.build_phenotype_compute_groups_payload(
            phenotype_names,
            multi_phenotype_sample_mode.value,
        )
    )


def resolve_association_mode(trait_type: types.RegenieTraitType) -> types.AssociationMode:
    """Resolve a trait family to the native association mode."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    return types.AssociationMode(native_host_planning_policy.resolve_association_mode_value(trait_type.value))


def build_kernel_config(regenie_config: config.RegenieConfig) -> KernelConfig:
    """Build engine kernel settings from a public config."""
    return KernelConfig(
        chunk_size=regenie_config.trait.bsize,
        device=regenie_config.g_compute.device,
        staging_depth=regenie_config.g_compute.staging_depth,
        native_callback_batch_size=regenie_config.g_compute.native_callback_batch_size,
        result_in_flight_limit=regenie_config.g_compute.result_in_flight_limit,
        dosage_buffer_limit=regenie_config.g_compute.dosage_buffer_limit,
        variant_limit=regenie_config.g_compute.variant_limit,
        thread_count=regenie_config.trait.threads,
        bgen_decode_tile_variant_count=regenie_config.g_compute.bgen_decode_tile_variant_count,
        gpu_genotype_format=regenie_config.g_compute.gpu_genotype_format,
        trusted_no_missing_diploid=regenie_config.g_compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=regenie_config.g_compute.trusted_bgen_validation_mode,
        alignment_config=regenie_config.g_compute,
        multi_phenotype_sample_mode=regenie_config.g_compute.multi_phenotype_sample_mode,
        binary_kernel_config=(
            build_binary_kernel_config(regenie_config.g_compute)
            if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
            else None
        ),
        linear_numerical_config=(
            regenie2_linear_config.LinearNumericalConfig(
                minimum_variance=regenie_config.g_compute.linear_minimum_variance,
                relative_variance_tolerance=regenie_config.g_compute.linear_relative_variance_tolerance,
            )
            if regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
            else None
        ),
    )


def build_phenotype_run_plan(
    *,
    phenotype_index: int,
    phenotype_name: str,
    association_mode: types.AssociationMode,
    output_plan: OutputPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> PhenotypeRunPlan:
    """Prepare output paths and resume manifest state for one phenotype."""
    output_directory_name = build_phenotype_output_directory_name(phenotype_index, phenotype_name)
    prepared_output_run = output.prepare_output_run(
        output_root=output_plan.output_run_root / output_directory_name,
        association_mode=association_mode,
        output_format=output_plan.writer_settings.output_format,
        resume=output_plan.resume,
        resume_mode=output_plan.resume_mode,
        runtime_compatibility_token=runtime_compatibility_token,
    )
    return PhenotypeRunPlan(
        phenotype_name=phenotype_name,
        output_run_paths=prepared_output_run.output_run_paths,
        existing_manifest=prepared_output_run.existing_manifest,
        effective_config_path=prepared_output_run.output_run_paths.run_directory / "effective_config.toml",
    )


def build_phenotype_output_directory_name(phenotype_index: int, phenotype_name: str) -> str:
    """Build a deterministic safe directory name for one phenotype output."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    return native_host_planning_policy.build_phenotype_output_directory_name(phenotype_index, phenotype_name)


def adapt_phenotype_compute_group_payload(group_payload: dict[str, object]) -> PhenotypeComputeGroup:
    """Adapt a native group payload to the public Python execution-plan shape."""
    return PhenotypeComputeGroup(
        group_mode=types.PhenotypeComputeGroupMode(typing.cast("str", group_payload["group_mode"])),
        phenotype_indices=tuple(typing.cast("typing.Sequence[int]", group_payload["phenotype_indices"])),
        phenotype_names=tuple(typing.cast("typing.Sequence[str]", group_payload["phenotype_names"])),
        sample_mode=types.MultiPhenotypeSampleMode(typing.cast("str", group_payload["sample_mode"])),
        sample_set_fingerprint=typing.cast("str | None", group_payload["sample_set_fingerprint"]),
        covariate_design_fingerprint=typing.cast("str | None", group_payload["covariate_design_fingerprint"]),
        prediction_alignment_fingerprint=typing.cast(
            "str | None",
            group_payload["prediction_alignment_fingerprint"],
        ),
    )


def require_mapping(payload: dict[str, typing.Any], key: str) -> dict[str, typing.Any]:
    """Return a nested mapping from a native JSON payload."""
    value = payload[key]
    if not isinstance(value, dict):
        message = f"Native run request field {key!r} must be an object."
        raise TypeError(message)
    return typing.cast("dict[str, typing.Any]", value)


def require_mapping_sequence(payload: dict[str, typing.Any], key: str) -> tuple[dict[str, typing.Any], ...]:
    """Return a tuple of nested mappings from a native JSON payload."""
    value = payload[key]
    if not isinstance(value, list | tuple):
        message = f"Native run request field {key!r} must be a sequence."
        raise TypeError(message)
    mappings: list[dict[str, typing.Any]] = []
    for item in value:
        if not isinstance(item, dict):
            message = f"Native run request field {key!r} must contain only objects."
            raise TypeError(message)
        mappings.append(typing.cast("dict[str, typing.Any]", item))
    return tuple(mappings)


def optional_path_from_request(value: object) -> Path | None:
    """Adapt an optional path string from a native JSON payload."""
    if value is None:
        return None
    if not isinstance(value, str):
        message = "Native run request optional path must be a string or null."
        raise TypeError(message)
    return Path(value)


def string_tuple_from_request(value: object) -> tuple[str, ...]:
    """Adapt a string list from a native JSON payload."""
    if not isinstance(value, list | tuple):
        message = "Native run request string sequence must be a sequence."
        raise TypeError(message)
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            message = "Native run request string sequence must contain only strings."
            raise TypeError(message)
        strings.append(item)
    return tuple(strings)
