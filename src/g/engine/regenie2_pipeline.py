"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import logging
import time
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import g.engine.callbacks.binary as callback_binary
import g.engine.callbacks.grouped as callback_grouped
import g.engine.callbacks.linear as callback_linear
import g.engine.callbacks.shared as callback_shared
from g import _core, execution_plan, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.engine import backend_planner, native_dispatch, preflight, telemetry, timing
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.io import source

REGENIE_COMPUTE_PATCH_TARGETS = (regenie2_binary, regenie2_linear)
logger = logging.getLogger(__name__)


def require_binary_kernel_config(
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None,
) -> regenie2_binary_config.BinaryKernelConfig:
    """Return the binary kernel config or fail at an internal boundary."""
    if kernel_config is None:
        message = "Binary kernel config is required for binary association."
        raise ValueError(message)
    return kernel_config


def require_linear_numerical_config(
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None,
) -> regenie2_linear_config.LinearNumericalConfig:
    """Return linear numerical settings, using package defaults for direct pipeline calls."""
    return linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG


@dataclass(frozen=True)
class OutputWriterSettings:
    """Output writer and finalization settings for a pipeline run.

    Attributes:
        finalize_parquet: Whether chunk output should be finalized to one Parquet file.
        writer_thread_count: Number of writer worker threads.
        writer_queue_depth: Maximum queued chunk writes.
        chunks_per_arrow_file: Number of chunks per Arrow output file.
        arrow_compression: Arrow IPC compression codec.
        parquet_compression: Parquet finalization compression codec.
        output_format: Chunk output format.

    """

    finalize_parquet: bool
    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression
    parquet_compression: types.ParquetCompression
    output_format: types.OutputFormat


@dataclass(frozen=True)
class Regenie2PipelineContext:
    """Resolved lifecycle settings shared by REGENIE step 2 pipelines.

    Attributes:
        association_mode: Association mode for compute and output.
        genotype_source_config: BGEN source configuration.
        phenotype_path: Phenotype file path.
        prediction_list_path: REGENIE step 1 prediction list path.
        covariate_path: Optional covariate file path.
        chunk_size: Requested native BGEN chunk size.
        variant_limit: Optional variant processing limit.
        trusted_no_missing_diploid: User-requested trusted BGEN mode.
        trusted_bgen_validation_mode: Trusted BGEN validation policy.
        bgen_decode_tile_variant_count: Native BGEN decode tile size.
        jax_device: Requested JAX device.
        jax_matmul_precision: Optional JAX matmul precision policy.
        score_dtype: Score-test compute dtype.
        firth_dtype: Firth compute dtype.
        gpu_genotype_format: Native genotype delivery format.
        backend_plan: Concrete backend selected for association execution.
        correction_plan: Binary correction settings.
        binary_kernel_config: Resolved binary kernel config when binary.
        linear_numerical_config: Resolved linear numerical config when quantitative.
        writer_settings: Output writer settings.
        stage_timing_recorder: Optional stage timing recorder for this run.
        telemetry_session: Optional telemetry sink.
        alignment_config: Optional sample alignment settings.
        phenotype_compute_groups: Planned phenotype compute groups.

    """

    association_mode: types.AssociationMode
    genotype_source_config: source.GenotypeSourceConfig
    phenotype_path: Path
    prediction_list_path: Path
    covariate_path: Path | None
    chunk_size: int
    variant_limit: int | None
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    bgen_decode_tile_variant_count: int
    jax_device: types.Device
    jax_matmul_precision: types.JaxMatmulPrecision | None
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    gpu_genotype_format: types.GpuGenotypeFormat
    backend_plan: backend_planner.AssociationBackendPlan
    correction_plan: types.BinaryCorrectionPlan
    binary_kernel_config: regenie2_binary_config.BinaryKernelConfig | None
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None
    writer_settings: OutputWriterSettings
    stage_timing_recorder: timing.StageTimingRecorder | None
    telemetry_session: telemetry.TelemetrySession | None
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...]

    @property
    def uses_packed8_genotypes(self) -> bool:
        """Return whether native delivery should use packed8 probability pairs."""
        return self.backend_plan.uses_variant_major_packed8_delivery

    @property
    def effective_trusted_no_missing_diploid(self) -> bool:
        """Return trusted BGEN mode after packed8 requirements are applied."""
        return self.trusted_no_missing_diploid or self.uses_packed8_genotypes

    @property
    def is_binary_trait(self) -> bool:
        """Return whether this context is for binary trait association."""
        return self.association_mode == types.AssociationMode.REGENIE2_BINARY


@dataclass(frozen=True)
class InitializedPipelineOutputs:
    """Manifest-validated output runs ready for writer creation.

    Attributes:
        committed_chunk_identifier_sets: Committed chunk identifiers accepted per output run.

    """

    committed_chunk_identifier_sets: tuple[set[int], ...]


@dataclass(frozen=True)
class PipelineWriterSessions:
    """Output writer sessions ready to receive native chunks.

    Attributes:
        writer_sessions: Native writer sessions in output-run order.

    """

    writer_sessions: tuple[typing.Any, ...]


@dataclass(frozen=True)
class PreparedMultiPhenotypeGroupDelivery:
    """Prepared compute callback and writers for one compatible phenotype group.

    Attributes:
        compute_group: Phenotype compute group represented by this delivery.
        phenotype_indices: Original phenotype indices represented by this group.
        run_input: Aligned multi-phenotype input for this compatible group.
        callback: Compute callback for this group.
        writer_sessions: Output writer sessions in this group's phenotype order.
        committed_chunk_identifier_sets: Committed chunks in this group's phenotype order.

    """

    compute_group: execution_plan.PhenotypeComputeGroup
    phenotype_indices: tuple[int, ...]
    run_input: native_dispatch.NativeBgenMultiRunInput
    callback: object
    writer_sessions: tuple[typing.Any, ...]
    committed_chunk_identifier_sets: tuple[set[int], ...]


def build_output_writer_settings(
    *,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
    output_format: types.OutputFormat,
) -> OutputWriterSettings:
    """Build output writer settings from public pipeline arguments."""
    return OutputWriterSettings(
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        output_format=output_format,
    )


def build_regenie2_pipeline_context(
    *,
    association_mode: types.AssociationMode,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    prediction_list_path: Path,
    covariate_path: Path | None,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    gpu_genotype_format: types.GpuGenotypeFormat,
    correction_plan: types.BinaryCorrectionPlan,
    binary_kernel_config: regenie2_binary_config.BinaryKernelConfig | None,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None,
    writer_settings: OutputWriterSettings,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] = (),
) -> Regenie2PipelineContext:
    """Build a resolved lifecycle context for a REGENIE step 2 run."""
    resolved_stage_timing_recorder: timing.StageTimingRecorder | None
    if stage_timing_recorder is None:
        resolved_stage_timing_recorder = timing.build_stage_timing_recorder()
    else:
        resolved_stage_timing_recorder = stage_timing_recorder
    backend_plan = backend_planner.plan_association_backend(
        association_mode=association_mode,
        jax_device=jax_device,
        gpu_genotype_format=gpu_genotype_format,
    )
    return Regenie2PipelineContext(
        association_mode=association_mode,
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        gpu_genotype_format=gpu_genotype_format,
        backend_plan=backend_plan,
        correction_plan=correction_plan,
        binary_kernel_config=binary_kernel_config,
        linear_numerical_config=linear_numerical_config,
        writer_settings=writer_settings,
        stage_timing_recorder=resolved_stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=phenotype_compute_groups,
    )


def build_single_phenotype_compute_groups(phenotype_name: str) -> tuple[execution_plan.PhenotypeComputeGroup, ...]:
    """Build the default compute group for a direct single-phenotype pipeline call."""
    return execution_plan.build_phenotype_compute_groups(
        phenotype_names=(phenotype_name,),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
    )


def resolve_multi_phenotype_compute_groups(
    *,
    phenotype_names: tuple[str, ...],
    sample_mode: types.MultiPhenotypeSampleMode | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
) -> tuple[execution_plan.PhenotypeComputeGroup, ...]:
    """Resolve planned multi-phenotype compute groups for direct and planned calls."""
    if sample_mode not in (
        types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        types.MultiPhenotypeSampleMode.COMPLETE_CASE,
    ):
        message = "Multi-phenotype sample mode must be per-phenotype or complete-case."
        raise ValueError(message)
    resolved_compute_groups = phenotype_compute_groups or execution_plan.build_phenotype_compute_groups(
        phenotype_names=phenotype_names,
        multi_phenotype_sample_mode=sample_mode,
    )
    validate_phenotype_compute_groups(
        phenotype_names=phenotype_names,
        sample_mode=sample_mode,
        phenotype_compute_groups=resolved_compute_groups,
    )
    return resolved_compute_groups


def validate_phenotype_compute_groups(
    *,
    phenotype_names: tuple[str, ...],
    sample_mode: types.MultiPhenotypeSampleMode,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...],
) -> None:
    """Validate planned compute groups against the pipeline request."""
    if not phenotype_compute_groups:
        message = "At least one phenotype compute group is required."
        raise ValueError(message)
    observed_phenotype_names: list[str | None] = [None] * len(phenotype_names)
    for phenotype_compute_group in phenotype_compute_groups:
        if phenotype_compute_group.sample_mode != sample_mode:
            message = "Phenotype compute group sample mode does not match the pipeline sample mode."
            raise ValueError(message)
        for phenotype_index, phenotype_name in zip(
            phenotype_compute_group.phenotype_indices,
            phenotype_compute_group.phenotype_names,
            strict=True,
        ):
            if phenotype_index < 0 or phenotype_index >= len(phenotype_names):
                message = f"Phenotype compute group index {phenotype_index} is outside the request."
                raise ValueError(message)
            if phenotype_names[phenotype_index] != phenotype_name:
                message = "Phenotype compute group names do not match the request order."
                raise ValueError(message)
            if observed_phenotype_names[phenotype_index] is not None:
                message = f"Phenotype '{phenotype_name}' appears in multiple compute groups."
                raise ValueError(message)
            observed_phenotype_names[phenotype_index] = phenotype_name
    if tuple(observed_phenotype_names) != phenotype_names:
        message = "Phenotype compute groups must cover every requested phenotype exactly once."
        raise ValueError(message)
    if sample_mode == types.MultiPhenotypeSampleMode.COMPLETE_CASE and len(phenotype_compute_groups) != 1:
        message = "Complete-case execution requires one shared phenotype compute group."
        raise ValueError(message)


def select_by_phenotype_indices(
    values: tuple[typing.Any, ...],
    phenotype_indices: tuple[int, ...],
) -> tuple[typing.Any, ...]:
    """Select values in phenotype compute group order."""
    return tuple(values[phenotype_index] for phenotype_index in phenotype_indices)


def require_complete_case_compute_group(
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...],
) -> execution_plan.PhenotypeComputeGroup:
    """Return the planned complete-case compute group."""
    for phenotype_compute_group in phenotype_compute_groups:
        if phenotype_compute_group.group_mode == types.PhenotypeComputeGroupMode.COMPLETE_CASE:
            return phenotype_compute_group
    message = "A complete-case phenotype compute group is required."
    raise ValueError(message)


def open_pipeline_bgen_engine(
    *,
    context: Regenie2PipelineContext,
    pipeline_label: str,
    phenotype_name: str | None = None,
    phenotype_count: int | None = None,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN engine and emit shared telemetry."""
    engine_start_time = time.perf_counter()
    logger.debug("Opening native BGEN engine for %s pipeline.", pipeline_label)
    if context.telemetry_session is not None:
        telemetry_fields: dict[str, typing.Any] = {
            "association_mode": context.association_mode.value,
            "association_backend_kind": context.backend_plan.backend_kind.value,
            "device": context.backend_plan.jax_device.value,
            "genotype_format": context.backend_plan.genotype_format.value,
        }
        if phenotype_name is not None:
            telemetry_fields["phenotype"] = phenotype_name
        if phenotype_count is not None:
            telemetry_fields["phenotype_count"] = phenotype_count
        context.telemetry_session.log_event("association_backend_selected", **telemetry_fields)
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=context.genotype_source_config,
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    logger.debug(
        "Native BGEN engine opened for %s pipeline: sample_count=%s variant_count=%s.",
        pipeline_label,
        engine.sample_count,
        engine.variant_count,
    )
    if context.telemetry_session is not None:
        telemetry_fields: dict[str, typing.Any] = {
            "association_mode": context.association_mode.value,
            "association_backend_kind": context.backend_plan.backend_kind.value,
            "sample_count": int(engine.sample_count),
            "variant_count": int(engine.variant_count),
        }
        if phenotype_name is not None:
            telemetry_fields["phenotype"] = phenotype_name
        if phenotype_count is not None:
            telemetry_fields["phenotype_count"] = phenotype_count
        context.telemetry_session.log_event("bgen_engine_opened", **telemetry_fields)
    return engine


def build_pipeline_manifest_header(
    *,
    context: Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...],
    sample_count: int,
    variant_count: int,
    multi_phenotype_sample_mode: output.MultiPhenotypeSampleMode = output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
) -> dict[str, typing.Any]:
    """Build the current manifest header for one output run."""
    return output.build_current_run_manifest_header(
        association_mode=context.association_mode,
        association_backend_kind=context.backend_plan.backend_kind,
        bgen_path=context.genotype_source_config.source_path,
        sample_path=context.genotype_source_config.resolved_sample_path,
        phenotype_path=context.phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        prediction_list_path=context.prediction_list_path,
        sample_count=sample_count,
        variant_count=variant_count,
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        binary_correction_plan=context.correction_plan,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        sample_key_mode=native_dispatch.resolve_sample_key_mode(context.alignment_config),
        binary_kernel_config=context.binary_kernel_config if context.is_binary_trait else None,
        bgen_decode_tile_variant_count=context.bgen_decode_tile_variant_count,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
        jax_device=context.jax_device,
        jax_matmul_precision=context.jax_matmul_precision,
        gpu_genotype_format=context.gpu_genotype_format,
        score_dtype=context.score_dtype,
        firth_dtype=context.firth_dtype,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        output_format=context.writer_settings.output_format,
        finalize_parquet=context.writer_settings.finalize_parquet,
        writer_thread_count=context.writer_settings.writer_thread_count,
        writer_queue_depth=context.writer_settings.writer_queue_depth,
        chunks_per_arrow_file=context.writer_settings.chunks_per_arrow_file,
        arrow_compression=context.writer_settings.arrow_compression,
        parquet_compression=context.writer_settings.parquet_compression,
    )


def initialize_pipeline_output_runs(
    *,
    output_run_paths_by_trait: tuple[output.OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[dict[str, typing.Any], ...],
    resume: bool,
    resume_mode: types.ResumeMode,
) -> InitializedPipelineOutputs:
    """Validate/write output manifests and return committed chunk sets."""
    initialized_output_runs = tuple(
        output.initialize_output_run(
            output_run_paths=output_run_paths,
            existing_manifest=existing_manifest,
            current_header=current_header,
            resume=resume,
            resume_mode=resume_mode,
        )
        for output_run_paths, existing_manifest, current_header in zip(
            output_run_paths_by_trait,
            existing_manifests_by_trait,
            current_headers_by_trait,
            strict=True,
        )
    )
    return InitializedPipelineOutputs(
        committed_chunk_identifier_sets=tuple(
            set(initialized_output_run.committed_chunk_identifiers)
            for initialized_output_run in initialized_output_runs
        )
    )


def create_pipeline_writer_sessions(
    *,
    context: Regenie2PipelineContext,
    output_run_paths_by_trait: tuple[output.OutputRunPaths, ...],
) -> PipelineWriterSessions:
    """Create output writer sessions and record preparation timing."""
    writer_start_time = time.perf_counter()
    logger.debug("Creating output writer(s) for %s pipeline.", context.association_mode.value)
    writer_sessions = tuple(
        output.create_output_writer_session(
            output_run_paths,
            context.association_mode,
            writer_thread_count=context.writer_settings.writer_thread_count,
            writer_queue_depth=context.writer_settings.writer_queue_depth,
            finalize_parquet=context.writer_settings.finalize_parquet,
            output_format=context.writer_settings.output_format,
            chunks_per_arrow_file=context.writer_settings.chunks_per_arrow_file,
            arrow_compression=context.writer_settings.arrow_compression,
            parquet_compression=context.writer_settings.parquet_compression,
            collect_stage_timings=timing.should_collect_exact_stage_timings(context.stage_timing_recorder),
        )
        for output_run_paths in output_run_paths_by_trait
    )
    timing.record_stage_duration(context.stage_timing_recorder, "output_writer_preparation", writer_start_time)
    return PipelineWriterSessions(writer_sessions=writer_sessions)


def log_sample_alignment_completed(
    *,
    context: Regenie2PipelineContext,
    sample_count: int | None = None,
    covariate_count: int | None = None,
    phenotype_name: str | None = None,
    phenotype_count: int | None = None,
    phenotype_group_count: int | None = None,
) -> None:
    """Emit sample-alignment telemetry with mode-specific fields."""
    if context.telemetry_session is None:
        return
    telemetry_fields: dict[str, typing.Any] = {
        "association_mode": context.association_mode.value,
    }
    if phenotype_name is not None:
        telemetry_fields["phenotype"] = phenotype_name
    if phenotype_count is not None:
        telemetry_fields["phenotype_count"] = phenotype_count
    if sample_count is not None:
        telemetry_fields["sample_count"] = sample_count
    if covariate_count is not None:
        telemetry_fields["covariate_count"] = covariate_count
    if phenotype_group_count is not None:
        telemetry_fields["phenotype_group_count"] = phenotype_group_count
    context.telemetry_session.log_event("sample_alignment_completed", **telemetry_fields)


def log_prediction_source_loaded(
    *,
    context: Regenie2PipelineContext,
    phenotype_name: str | None = None,
    phenotype_count: int | None = None,
) -> None:
    """Emit prediction-source telemetry with mode-specific fields."""
    if context.telemetry_session is None:
        return
    telemetry_fields: dict[str, typing.Any] = {
        "association_mode": context.association_mode.value,
    }
    if phenotype_name is not None:
        telemetry_fields["phenotype"] = phenotype_name
    if phenotype_count is not None:
        telemetry_fields["phenotype_count"] = phenotype_count
    context.telemetry_session.log_event("prediction_source_loaded", **telemetry_fields)


def load_single_trait_run_input(
    *,
    context: Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    phenotype_name: str,
    covariate_names: tuple[str, ...] | None,
    pipeline_label: str,
) -> native_dispatch.NativeBgenRunInput:
    """Load one phenotype's aligned native inputs and emit telemetry."""
    alignment_start_time = time.perf_counter()
    logger.debug("Loading aligned native sample, phenotype, and covariate inputs for %s pipeline.", pipeline_label)
    run_input = native_dispatch.load_native_bgen_run_input(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    sample_count = int(run_input.sample_indices.shape[0])
    covariate_count = len(run_input.native_aligned_sample_data.covariate_names)
    logger.debug(
        "Aligned %s pipeline inputs: sample_count=%s covariate_count=%s.",
        pipeline_label,
        sample_count,
        covariate_count,
    )
    log_sample_alignment_completed(
        context=context,
        phenotype_name=phenotype_name,
        sample_count=sample_count,
        covariate_count=covariate_count,
    )
    return run_input


def build_single_trait_prediction_source(
    *,
    context: Regenie2PipelineContext,
    run_input: native_dispatch.NativeBgenRunInput,
    phenotype_name: str,
    pipeline_label: str,
) -> typing.Any:
    """Load one phenotype's REGENIE prediction source and emit telemetry."""
    prediction_start_time = time.perf_counter()
    logger.debug("Loading REGENIE prediction source for %s pipeline.", pipeline_label)
    prediction_source = native_dispatch.build_regenie_prediction_source(
        prediction_list_path=context.prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "prediction_source_load", prediction_start_time)
    log_prediction_source_loaded(context=context, phenotype_name=phenotype_name)
    return prediction_source


def run_single_trait_preflight(
    *,
    context: Regenie2PipelineContext,
    run_input: native_dispatch.NativeBgenRunInput,
    prediction_source: typing.Any,
    engine: _core.Regenie2RunEngine,
    phenotype_name: str,
    pipeline_label: str,
) -> None:
    """Run preflight validation for one phenotype and emit telemetry."""
    preflight_start_time = time.perf_counter()
    logger.debug("Running preflight validation for %s pipeline.", pipeline_label)
    preflight_report = preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=context.variant_limit,
        is_binary_trait=context.is_binary_trait,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "preflight_validation", preflight_start_time)
    logger.debug(
        "Preflight validation passed for %s pipeline: sample_count=%s covariate_count=%s chromosome_count=%s.",
        pipeline_label,
        preflight_report.sample_count,
        preflight_report.covariate_count,
        preflight_report.chromosome_count,
    )
    if context.telemetry_session is not None:
        context.telemetry_session.log_event(
            "preflight_completed",
            association_mode=context.association_mode.value,
            phenotype=phenotype_name,
            sample_count=preflight_report.sample_count,
            covariate_count=preflight_report.covariate_count,
            chromosome_count=preflight_report.chromosome_count,
        )


def build_single_trait_callback(
    *,
    context: Regenie2PipelineContext,
    run_input: native_dispatch.NativeBgenRunInput,
    prediction_source: typing.Any,
    writer_session: typing.Any,
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> object:
    """Build the association-specific single-trait callback."""
    if context.is_binary_trait:
        return callback_binary.BinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_session=writer_session,
            correction_plan=context.correction_plan,
            kernel_config=require_binary_kernel_config(context.binary_kernel_config),
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
        )
    return callback_linear.LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=context.score_dtype,
        linear_numerical_config=require_linear_numerical_config(context.linear_numerical_config),
        stage_timing_recorder=context.stage_timing_recorder,
        telemetry_session=context.telemetry_session,
    )


def run_single_trait_bgen_pipeline(
    *,
    context: Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...] | None,
    output_run_paths: output.OutputRunPaths,
    existing_manifest: dict[str, typing.Any] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> Path | None:
    """Run a single-trait REGENIE step 2 BGEN pipeline lifecycle."""
    pipeline_label = "binary" if context.is_binary_trait else "linear"
    logger.info("Starting %s REGENIE step 2 BGEN pipeline.", pipeline_label)
    engine = open_pipeline_bgen_engine(
        context=context,
        pipeline_label=pipeline_label,
        phenotype_name=phenotype_name,
    )
    run_input = load_single_trait_run_input(
        context=context,
        engine=engine,
        phenotype_name=phenotype_name,
        covariate_names=covariate_names,
        pipeline_label=pipeline_label,
    )
    prediction_source = build_single_trait_prediction_source(
        context=context,
        run_input=run_input,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    run_single_trait_preflight(
        context=context,
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    current_header = build_pipeline_manifest_header(
        context=context,
        phenotype_name=phenotype_name,
        covariate_names=tuple(run_input.native_aligned_sample_data.covariate_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        variant_count=int(engine.variant_count),
    )
    initialized_outputs = initialize_pipeline_output_runs(
        output_run_paths_by_trait=(output_run_paths,),
        existing_manifests_by_trait=(existing_manifest,),
        current_headers_by_trait=(current_header,),
        resume=resume,
        resume_mode=resume_mode,
    )
    writer_sessions = create_pipeline_writer_sessions(
        context=context,
        output_run_paths_by_trait=(output_run_paths,),
    )
    writer_session = writer_sessions.writer_sessions[0]
    callback = build_single_trait_callback(
        context=context,
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )
    return native_dispatch.run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=initialized_outputs.committed_chunk_identifier_sets[0],
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=context.stage_timing_recorder,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
    )


def run_regenie2_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    existing_manifest: dict[str, typing.Any] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    parquet_compression: types.ParquetCompression,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    gpu_genotype_format: types.GpuGenotypeFormat = types.GpuGenotypeFormat.DOSAGE,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    writer_settings = build_output_writer_settings(
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        output_format=output_format,
    )
    context = build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=types.BinaryCorrectionPlan(),
        binary_kernel_config=None,
        linear_numerical_config=linear_numerical_config,
        writer_settings=writer_settings,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=build_single_phenotype_compute_groups(phenotype_name),
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=phenotype_name,
        covariate_names=covariate_names,
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        resume=resume,
        resume_mode=resume_mode,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
    )


def run_regenie2_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    existing_manifest: dict[str, typing.Any] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    parquet_compression: types.ParquetCompression,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    gpu_genotype_format: types.GpuGenotypeFormat = types.GpuGenotypeFormat.DOSAGE,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
        types.NullLogisticNonconvergencePolicy.FAIL
    ),
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    resolved_kernel_config = require_binary_kernel_config(kernel_config)
    writer_settings = build_output_writer_settings(
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        output_format=output_format,
    )
    context = build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=None,
        writer_settings=writer_settings,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=build_single_phenotype_compute_groups(phenotype_name),
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=phenotype_name,
        covariate_names=covariate_names,
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        resume=resume,
        resume_mode=resume_mode,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )


def run_regenie2_multi_phenotype_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    parquet_compression: types.ParquetCompression,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    gpu_genotype_format: types.GpuGenotypeFormat = types.GpuGenotypeFormat.DOSAGE,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
    sample_mode: types.MultiPhenotypeSampleMode | None = None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None = None,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple quantitative phenotypes."""
    return run_regenie2_multi_phenotype_bgen_pipeline(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests_by_phenotype=existing_manifests_by_phenotype,
        resume=resume,
        resume_mode=resume_mode,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=None,
        linear_numerical_config=linear_numerical_config,
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        sample_mode=sample_mode,
        phenotype_compute_groups=phenotype_compute_groups,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
    )


def run_regenie2_multi_phenotype_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    parquet_compression: types.ParquetCompression,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    gpu_genotype_format: types.GpuGenotypeFormat = types.GpuGenotypeFormat.DOSAGE,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
        types.NullLogisticNonconvergencePolicy.FAIL
    ),
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
    sample_mode: types.MultiPhenotypeSampleMode | None = None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None = None,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple binary phenotypes."""
    resolved_kernel_config = require_binary_kernel_config(kernel_config)
    return run_regenie2_multi_phenotype_bgen_pipeline(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests_by_phenotype=existing_manifests_by_phenotype,
        resume=resume,
        resume_mode=resume_mode,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=correction_plan,
        kernel_config=resolved_kernel_config,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        sample_mode=sample_mode,
        phenotype_compute_groups=phenotype_compute_groups,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
    )


def run_regenie2_multi_phenotype_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    output_format: types.OutputFormat,
    gpu_genotype_format: types.GpuGenotypeFormat,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None,
    sample_mode: types.MultiPhenotypeSampleMode | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
    association_mode: types.AssociationMode,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
) -> tuple[Path | None, ...]:
    """Shared implementation for multi-phenotype BGEN pipelines."""
    resolved_compute_groups = resolve_multi_phenotype_compute_groups(
        phenotype_names=phenotype_names,
        sample_mode=sample_mode,
        phenotype_compute_groups=phenotype_compute_groups,
    )
    resolved_kernel_config = (
        require_binary_kernel_config(kernel_config)
        if association_mode == types.AssociationMode.REGENIE2_BINARY
        else None
    )
    writer_settings = build_output_writer_settings(
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        output_format=output_format,
    )
    context = build_regenie2_pipeline_context(
        association_mode=association_mode,
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=linear_numerical_config,
        writer_settings=writer_settings,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=resolved_compute_groups,
    )
    if sample_mode == types.MultiPhenotypeSampleMode.PER_PHENOTYPE:
        return run_regenie2_grouped_per_phenotype_bgen_pipeline(
            context=context,
            phenotype_names=phenotype_names,
            covariate_names=covariate_names,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests_by_phenotype=existing_manifests_by_phenotype,
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        )
    if sample_mode != types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        message = "Multi-phenotype sample mode must be per-phenotype or complete-case."
        raise ValueError(message)
    logger.info("Starting multi-phenotype REGENIE step 2 BGEN pipeline.")
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    planned_compute_group = require_complete_case_compute_group(context.phenotype_compute_groups)
    engine = open_pipeline_bgen_engine(
        context=context,
        pipeline_label="multi-phenotype",
        phenotype_count=len(planned_compute_group.phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    logger.debug("Loading aligned native sample, phenotype, and covariate inputs for multi-phenotype pipeline.")
    run_input = native_dispatch.load_native_bgen_multi_run_input(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_names=planned_compute_group.phenotype_names,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
    )
    resolved_compute_group = native_dispatch.build_resolved_complete_case_phenotype_compute_group(
        run_input=run_input,
        prediction_list_path=context.prediction_list_path,
        planned_compute_groups=context.phenotype_compute_groups,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    logger.debug(
        "Aligned multi-phenotype pipeline inputs: sample_count=%s phenotype_count=%s covariate_count=%s.",
        int(run_input.sample_indices.shape[0]),
        len(run_input.phenotype_names),
        len(run_input.native_multi_aligned_sample_data.covariate_names),
    )
    log_sample_alignment_completed(
        context=context,
        phenotype_count=len(run_input.phenotype_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        covariate_count=len(run_input.native_multi_aligned_sample_data.covariate_names),
    )
    prediction_start_time = time.perf_counter()
    logger.debug("Loading REGENIE prediction source for multi-phenotype pipeline.")
    prediction_source = native_dispatch.build_multi_regenie_prediction_source(
        prediction_list_path=context.prediction_list_path,
        run_input=run_input,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "prediction_source_load", prediction_start_time)
    return run_prepared_multi_phenotype_bgen_group(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=resolved_compute_group,
        output_run_paths_by_phenotype=typing.cast(
            "tuple[output.OutputRunPaths, ...]",
            select_by_phenotype_indices(output_run_paths_by_phenotype, resolved_compute_group.phenotype_indices),
        ),
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests=typing.cast(
            "tuple[dict[str, typing.Any] | None, ...]",
            select_by_phenotype_indices(existing_manifests, resolved_compute_group.phenotype_indices),
        ),
        resume=resume,
        resume_mode=resume_mode,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        output_sample_mode=output.MultiPhenotypeSampleMode.COMPLETE_CASE_INTERSECTION,
    )


def run_regenie2_grouped_per_phenotype_bgen_pipeline(
    *,
    context: Regenie2PipelineContext,
    phenotype_names: tuple[str, ...],
    covariate_names: tuple[str, ...] | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Group independently aligned phenotypes and run one BGEN pass per compatible group."""
    logger.info("Starting grouped per-phenotype REGENIE step 2 BGEN pipeline.")
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    engine = open_pipeline_bgen_engine(
        context=context,
        pipeline_label="grouped per-phenotype",
        phenotype_count=len(phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    grouped_run_inputs = native_dispatch.load_native_bgen_grouped_run_inputs(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=context.prediction_list_path,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
        planned_compute_groups=context.phenotype_compute_groups,
    )
    timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    logger.info(
        "Prepared %s compatible per-phenotype group(s) for %s phenotype(s).",
        len(grouped_run_inputs),
        len(phenotype_names),
    )
    log_sample_alignment_completed(
        context=context,
        phenotype_count=len(phenotype_names),
        phenotype_group_count=len(grouped_run_inputs),
    )

    if should_use_union_grouped_bgen_delivery(context=context, grouped_run_inputs=grouped_run_inputs):
        return run_prepared_grouped_per_phenotype_union_bgen_pipeline(
            context=context,
            engine=engine,
            grouped_run_inputs=grouped_run_inputs,
            phenotype_names=phenotype_names,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests=existing_manifests,
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        )

    final_parquet_paths_by_index: list[Path | None] = [None] * len(phenotype_names)
    for grouped_run_input in grouped_run_inputs:
        compute_group = grouped_run_input.compute_group
        group_multi_run_input = grouped_run_input.run_input
        group_final_parquet_paths = run_prepared_multi_phenotype_bgen_group(
            context=context,
            engine=engine,
            run_input=group_multi_run_input,
            prediction_source=grouped_run_input.prediction_source,
            compute_group=compute_group,
            output_run_paths_by_phenotype=typing.cast(
                "tuple[output.OutputRunPaths, ...]",
                select_by_phenotype_indices(output_run_paths_by_phenotype, compute_group.phenotype_indices),
            ),
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests=typing.cast(
                "tuple[dict[str, typing.Any] | None, ...]",
                select_by_phenotype_indices(existing_manifests, compute_group.phenotype_indices),
            ),
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            output_sample_mode=output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
        )
        for phenotype_index, final_parquet_path in zip(
            compute_group.phenotype_indices,
            group_final_parquet_paths,
            strict=True,
        ):
            final_parquet_paths_by_index[phenotype_index] = final_parquet_path
    return tuple(final_parquet_paths_by_index)


def build_union_sample_indices(
    grouped_run_inputs: tuple[native_dispatch.NativeBgenGroupedRunInput, ...],
) -> npt.NDArray[np.int64]:
    """Build an ordered union sample selection for compatible phenotype groups."""
    seen_sample_indices: set[int] = set()
    union_sample_indices: list[int] = []
    for grouped_run_input in grouped_run_inputs:
        for raw_sample_index in grouped_run_input.run_input.sample_indices:
            sample_index = int(raw_sample_index)
            if sample_index in seen_sample_indices:
                continue
            seen_sample_indices.add(sample_index)
            union_sample_indices.append(sample_index)
    return np.asarray(union_sample_indices, dtype=np.int64)


def build_group_sample_position_array(
    *,
    union_sample_indices: npt.NDArray[np.int64],
    group_sample_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.intp]:
    """Map one group's sample order to positions in the union decode buffer."""
    union_position_by_sample_index = {
        int(sample_index): sample_position for sample_position, sample_index in enumerate(union_sample_indices)
    }
    return np.asarray(
        [union_position_by_sample_index[int(sample_index)] for sample_index in group_sample_indices],
        dtype=np.intp,
    )


def should_use_union_grouped_bgen_delivery(
    *,
    context: Regenie2PipelineContext,
    grouped_run_inputs: tuple[native_dispatch.NativeBgenGroupedRunInput, ...],
) -> bool:
    """Return whether grouped per-phenotype delivery should use one union decode pass."""
    if len(grouped_run_inputs) <= 1:
        return False
    if context.uses_packed8_genotypes:
        return False
    if not context.effective_trusted_no_missing_diploid:
        return False
    union_sample_count = int(build_union_sample_indices(grouped_run_inputs).shape[0])
    grouped_sample_count = sum(
        int(grouped_run_input.run_input.sample_indices.shape[0]) for grouped_run_input in grouped_run_inputs
    )
    return union_sample_count < grouped_sample_count


def intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: tuple[set[int], ...],
) -> set[int]:
    """Return chunk identifiers already committed by every output in a delivery."""
    if not committed_chunk_identifier_sets:
        return set()
    return set.intersection(*committed_chunk_identifier_sets)


def run_prepared_grouped_per_phenotype_union_bgen_pipeline(
    *,
    context: Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    grouped_run_inputs: tuple[native_dispatch.NativeBgenGroupedRunInput, ...],
    phenotype_names: tuple[str, ...],
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Run overlapping per-phenotype groups through one union-sample BGEN delivery."""
    union_sample_indices = build_union_sample_indices(grouped_run_inputs)
    logger.info(
        "Using union per-phenotype BGEN delivery: group_count=%s union_sample_count=%s grouped_sample_count=%s.",
        len(grouped_run_inputs),
        int(union_sample_indices.shape[0]),
        sum(int(grouped_run_input.run_input.sample_indices.shape[0]) for grouped_run_input in grouped_run_inputs),
    )
    prepared_deliveries = tuple(
        prepare_multi_phenotype_bgen_group_delivery(
            context=context,
            engine=engine,
            run_input=grouped_run_input.run_input,
            prediction_source=grouped_run_input.prediction_source,
            compute_group=grouped_run_input.compute_group,
            output_run_paths_by_phenotype=typing.cast(
                "tuple[output.OutputRunPaths, ...]",
                select_by_phenotype_indices(
                    output_run_paths_by_phenotype,
                    grouped_run_input.compute_group.phenotype_indices,
                ),
            ),
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests=typing.cast(
                "tuple[dict[str, typing.Any] | None, ...]",
                select_by_phenotype_indices(
                    existing_manifests,
                    grouped_run_input.compute_group.phenotype_indices,
                ),
            ),
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            output_sample_mode=output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
        )
        for grouped_run_input in grouped_run_inputs
    )
    group_fanouts = tuple(
        callback_shared.MultiPhenotypeGroupFanout(
            callback=prepared_delivery.callback,
            sample_position_array=build_group_sample_position_array(
                union_sample_indices=union_sample_indices,
                group_sample_indices=prepared_delivery.run_input.sample_indices,
            ),
        )
        for prepared_delivery in prepared_deliveries
    )
    union_run_input = native_dispatch.NativeBgenUnionRunInput(sample_indices=union_sample_indices)
    writer_sessions = tuple(
        writer_session
        for prepared_delivery in prepared_deliveries
        for writer_session in prepared_delivery.writer_sessions
    )
    committed_chunk_identifier_sets = tuple(
        committed_chunk_identifier_set
        for prepared_delivery in prepared_deliveries
        for committed_chunk_identifier_set in prepared_delivery.committed_chunk_identifier_sets
    )
    final_parquet_paths = native_dispatch.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=union_run_input,
        committed_chunk_identifiers=intersect_committed_chunk_identifier_sets(committed_chunk_identifier_sets),
        writer_sessions=writer_sessions,
        callback=callback_grouped.GroupedMultiPhenotypeFanoutCallback(group_fanouts),
        stage_timing_recorder=context.stage_timing_recorder,
        variant_major_packed8_probability_pairs=False,
        pipeline_label="Grouped per-phenotype union native BGEN",
    )
    final_parquet_paths_by_index: list[Path | None] = [None] * len(phenotype_names)
    final_path_offset = 0
    for prepared_delivery in prepared_deliveries:
        group_final_paths = final_parquet_paths[
            final_path_offset : final_path_offset + len(prepared_delivery.phenotype_indices)
        ]
        final_path_offset += len(prepared_delivery.phenotype_indices)
        for phenotype_index, final_parquet_path in zip(
            prepared_delivery.phenotype_indices,
            group_final_paths,
            strict=True,
        ):
            final_parquet_paths_by_index[phenotype_index] = final_parquet_path
    return tuple(final_parquet_paths_by_index)


def prepare_multi_phenotype_bgen_group_delivery(
    *,
    context: Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    output_sample_mode: output.MultiPhenotypeSampleMode,
) -> PreparedMultiPhenotypeGroupDelivery:
    """Prepare one compatible phenotype group for native BGEN delivery."""
    log_prediction_source_loaded(context=context, phenotype_count=len(run_input.phenotype_names))
    preflight_start_time = time.perf_counter()
    logger.debug("Running preflight validation for multi-phenotype pipeline.")
    run_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "preflight_validation", preflight_start_time)
    logger.debug("Preflight validation passed for multi-phenotype pipeline.")
    if context.telemetry_session is not None:
        context.telemetry_session.log_event(
            "preflight_completed",
            association_mode=context.association_mode.value,
            phenotype_count=len(run_input.phenotype_names),
            sample_count=int(run_input.sample_indices.shape[0]),
        )
    current_headers = tuple(
        build_pipeline_manifest_header(
            context=context,
            phenotype_name=phenotype_name,
            covariate_names=tuple(run_input.native_multi_aligned_sample_data.covariate_names),
            sample_count=int(run_input.sample_indices.shape[0]),
            variant_count=int(engine.variant_count),
            multi_phenotype_sample_mode=output_sample_mode,
        )
        for phenotype_name in compute_group.phenotype_names
    )
    initialized_outputs = initialize_pipeline_output_runs(
        output_run_paths_by_trait=output_run_paths_by_phenotype,
        existing_manifests_by_trait=existing_manifests,
        current_headers_by_trait=current_headers,
        resume=resume,
        resume_mode=resume_mode,
    )
    committed_chunk_identifier_sets = initialized_outputs.committed_chunk_identifier_sets
    writer_sessions = create_pipeline_writer_sessions(
        context=context,
        output_run_paths_by_trait=output_run_paths_by_phenotype,
    )
    writer_session_tuple = writer_sessions.writer_sessions
    if context.is_binary_trait:
        binary_kernel_config = require_binary_kernel_config(context.binary_kernel_config)
        callback = callback_binary.MultiBinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_session_tuple,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            correction_plan=context.correction_plan,
            kernel_config=binary_kernel_config,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
        )
    else:
        callback = callback_linear.MultiLinearRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_session_tuple,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            linear_numerical_config=require_linear_numerical_config(context.linear_numerical_config),
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
        )
    return PreparedMultiPhenotypeGroupDelivery(
        compute_group=compute_group,
        phenotype_indices=compute_group.phenotype_indices,
        run_input=run_input,
        callback=callback,
        writer_sessions=writer_session_tuple,
        committed_chunk_identifier_sets=committed_chunk_identifier_sets,
    )


def run_prepared_multi_phenotype_bgen_group(
    *,
    context: Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    output_sample_mode: output.MultiPhenotypeSampleMode,
) -> tuple[Path | None, ...]:
    """Run one prepared compatible phenotype group through one BGEN pass."""
    prepared_delivery = prepare_multi_phenotype_bgen_group_delivery(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=compute_group,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests=existing_manifests,
        resume=resume,
        resume_mode=resume_mode,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        output_sample_mode=output_sample_mode,
    )
    return run_bgen_engine_with_multi_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=intersect_committed_chunk_identifier_sets(
            prepared_delivery.committed_chunk_identifier_sets
        ),
        writer_sessions=prepared_delivery.writer_sessions,
        callback=prepared_delivery.callback,
        stage_timing_recorder=context.stage_timing_recorder,
        writer_finish_thread_count=context.writer_settings.writer_thread_count,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
    )


def run_multi_preflight(
    *,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    engine: _core.Regenie2RunEngine,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
) -> None:
    """Run shared batched preflight checks for a multi-trait run."""
    preflight.run_regenie2_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        is_binary_trait=run_input.is_binary_trait,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def run_bgen_engine_with_multi_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[typing.Any, ...],
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int = 1,
    variant_major_packed8_probability_pairs: bool = False,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery once and close all per-phenotype writers."""
    return native_dispatch.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_sessions=writer_sessions,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        writer_finish_thread_count=writer_finish_thread_count,
        variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
        pipeline_label="Multi-phenotype native BGEN",
    )
