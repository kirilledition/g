"""Immutable execution plans for REGENIE-compatible runs."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import config as regenie2_linear_config

if typing.TYPE_CHECKING:
    from g.interface import config

type BinaryKernelConfig = regenie2_binary_config.BinaryKernelConfig
type LinearNumericalConfig = regenie2_linear_config.LinearNumericalConfig


@dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one resolved BGEN input source.

    Attributes:
        source_path: BGEN genotype file path.
        sample_path: Explicit Oxford sample file path, or None to use embedded BGEN sample identifiers.

    """

    source_path: Path
    sample_path: Path | None

    def __post_init__(self) -> None:
        """Validate the configured BGEN source path."""
        if self.source_path.suffix != ".bgen":
            message = f"Expected a .bgen source path, found '{self.source_path}'."
            raise ValueError(message)


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
    binary_kernel_config: BinaryKernelConfig | None
    linear_numerical_config: LinearNumericalConfig | None


@dataclass(frozen=True)
class OutputWriterPlan:
    """Output writer and finalization settings for a run.

    Attributes:
        finalize_parquet: Whether chunk output should be finalized to one Parquet file.
        writer_thread_count: Number of writer worker threads.
        writer_queue_depth: Maximum queued chunk writes.
        chunks_per_arrow_file: Number of chunks per Arrow output file.
        arrow_compression: Arrow IPC compression codec.
        parquet_compression: Parquet finalization compression codec.
        output_format: Chunk output format.
        output_statistic_dtype: Persisted dtype for public statistic columns.

    """

    finalize_parquet: bool
    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression
    parquet_compression: types.ParquetCompression
    output_format: types.OutputFormat
    output_statistic_dtype: types.FloatingPointDtype


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
    writer_settings: OutputWriterPlan


@dataclass(frozen=True)
class PhenotypeRunPlan:
    """Requested output plan state for one phenotype.

    Attributes:
        phenotype_name: Phenotype column name.
        output_directory_name: Native planned output directory name under the output run root.

    """

    phenotype_name: str
    output_directory_name: str


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
    genotype_source_config: GenotypeSourceConfig
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


def build_binary_kernel_config(compute_config: config.GComputeConfig) -> BinaryKernelConfig:
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
) -> RegenieExecutionPlan:
    """Build a complete execution plan from a validated public config."""
    run_request = compile_run_request(regenie_config)
    input_request = run_request.input
    association_mode = types.AssociationMode(run_request.association_mode)
    output_plan = build_output_plan_from_run_request(run_request)
    kernel_config = build_kernel_config_from_run_request(regenie_config, run_request)
    phenotype_run_plans = tuple(
        adapt_phenotype_run_plan(phenotype_run_request) for phenotype_run_request in run_request.phenotype_runs
    )
    return RegenieExecutionPlan(
        association_mode=association_mode,
        genotype_source_config=GenotypeSourceConfig(
            source_path=Path(input_request.bgen_path),
            sample_path=optional_path_from_request(input_request.sample_path),
        ),
        phenotype_path=Path(input_request.phenotype_path),
        prediction_list_path=Path(input_request.prediction_list_path),
        covariate_path=optional_path_from_request(input_request.covariate_path),
        covariate_names=tuple(input_request.covariate_names) or None,
        phenotype_run_plans=phenotype_run_plans,
        phenotype_compute_groups=tuple(
            adapt_phenotype_compute_group(group_request) for group_request in run_request.phenotype_compute_groups
        ),
        binary_correction_plan=adapt_binary_correction_plan(run_request.correction),
        kernel_config=kernel_config,
        output_plan=output_plan,
        stage_timings_json=optional_path_from_request(run_request.stage_timings_json),
    )


def compile_run_request(regenie_config: config.RegenieConfig) -> _core.NativeRunRequest:
    """Compile a resolved config into the typed native requested-run handle."""
    return _core.compile_run_request(regenie_config)


def build_output_plan_from_run_request(run_request: _core.NativeRunRequest) -> OutputPlan:
    """Adapt the native output writer plan into the existing Python dataclass."""
    output_request = run_request.output
    return OutputPlan(
        output_prefix=Path(output_request.output_prefix),
        output_run_root=Path(output_request.output_run_root),
        resume=output_request.resume,
        resume_mode=types.ResumeMode(output_request.resume_mode),
        writer_settings=OutputWriterPlan(
            finalize_parquet=output_request.finalize_parquet,
            writer_thread_count=output_request.writer_thread_count,
            writer_queue_depth=output_request.writer_queue_depth,
            chunks_per_arrow_file=output_request.chunks_per_arrow_file,
            arrow_compression=types.ArrowCompression(output_request.arrow_compression),
            parquet_compression=types.ParquetCompression(output_request.parquet_compression),
            output_format=types.OutputFormat(output_request.output_format),
            output_statistic_dtype=types.FloatingPointDtype(output_request.output_statistic_dtype),
        ),
    )


def build_kernel_config_from_run_request(
    regenie_config: config.RegenieConfig,
    run_request: _core.NativeRunRequest,
) -> KernelConfig:
    """Adapt native requested-run compute fields into the existing kernel config."""
    compute_request = run_request.compute
    trait_request = run_request.trait_request
    trait_type = types.RegenieTraitType(trait_request.trait_type)
    return KernelConfig(
        chunk_size=trait_request.chunk_size,
        device=types.Device(compute_request.device),
        staging_depth=compute_request.staging_depth,
        native_callback_batch_size=compute_request.native_callback_batch_size,
        result_in_flight_limit=compute_request.result_in_flight_limit,
        dosage_buffer_limit=compute_request.dosage_buffer_limit,
        variant_limit=compute_request.variant_limit,
        thread_count=trait_request.thread_count,
        bgen_decode_tile_variant_count=compute_request.bgen_decode_tile_variant_count,
        gpu_genotype_format=types.GpuGenotypeFormat(compute_request.requested_gpu_genotype_format),
        trusted_no_missing_diploid=compute_request.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode(compute_request.trusted_bgen_validation_mode),
        alignment_config=regenie_config.g_compute,
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode(compute_request.multi_phenotype_sample_mode),
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


def adapt_binary_correction_plan(correction_request: _core.NativeRunRequestCorrection) -> types.BinaryCorrectionPlan:
    """Adapt native correction settings to the existing Python correction plan."""
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod(correction_request.method),
        p_threshold=correction_request.p_threshold,
        firth_se=correction_request.firth_se,
    )


def build_phenotype_compute_groups(
    *,
    phenotype_names: tuple[str, ...],
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode,
) -> tuple[PhenotypeComputeGroup, ...]:
    """Build config-time phenotype compute groups."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    return tuple(
        adapt_phenotype_compute_group(group_plan)
        for group_plan in native_host_planning_policy.build_phenotype_compute_groups(
            phenotype_names,
            multi_phenotype_sample_mode.value,
        )
    )


def resolve_association_mode(trait_type: types.RegenieTraitType) -> types.AssociationMode:
    """Resolve a trait family to the native association mode."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    return types.AssociationMode(native_host_planning_policy.resolve_association_mode_value(trait_type.value))


def adapt_phenotype_run_plan(phenotype_run_plan: _core.NativePhenotypeRunPlan) -> PhenotypeRunPlan:
    """Adapt a native phenotype-run plan to the Python execution-plan shape."""
    return PhenotypeRunPlan(
        phenotype_name=phenotype_run_plan.phenotype_name,
        output_directory_name=phenotype_run_plan.output_directory_name,
    )


def adapt_phenotype_compute_group(
    group_request: _core.NativePhenotypeComputeGroup | _core.NativeHostPhenotypeComputeGroupPlan,
) -> PhenotypeComputeGroup:
    """Adapt a native compute group to the public Python shape."""
    return PhenotypeComputeGroup(
        group_mode=types.PhenotypeComputeGroupMode(group_request.group_mode),
        phenotype_indices=tuple(group_request.phenotype_indices),
        phenotype_names=tuple(group_request.phenotype_names),
        sample_mode=types.MultiPhenotypeSampleMode(group_request.sample_mode),
        sample_set_fingerprint=group_request.sample_set_fingerprint,
        covariate_design_fingerprint=group_request.covariate_design_fingerprint,
        prediction_alignment_fingerprint=group_request.prediction_alignment_fingerprint,
    )


def optional_path_from_request(value: str | None) -> Path | None:
    """Adapt an optional path string from a native requested-run handle."""
    if value is None:
        return None
    return Path(value)
