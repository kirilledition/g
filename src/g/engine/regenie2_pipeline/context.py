"""Shared context and planning helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from g import execution_plan, types
from g.engine import backend_planner
from g.engine.regenie2_pipeline import compute_config, outputs, schedule, telemetry_events, timing

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core
    from g.engine.regenie2_pipeline import callbacks, inputs


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
        requested_gpu_genotype_format: User-requested genotype delivery format.
        gpu_genotype_format: Native genotype delivery format.
        backend_plan: Concrete backend selected for association execution.
        correction_plan: Binary correction settings.
        binary_kernel_config: Resolved binary kernel config when binary.
        linear_numerical_config: Resolved linear numerical config when quantitative.
        writer_settings: Output writer settings.
        stage_timing_recorder: Optional stage timing recorder for this run.
        telemetry_session: Optional telemetry sink.
        input_fingerprint_cache: Run-scoped input fingerprint cache.
        alignment_config: Optional sample alignment settings.
        phenotype_compute_groups: Planned phenotype compute groups.
        runtime_compatibility_token: Native token proving runtime checks passed.
        output_initialized_callback: Callback invoked after output manifests
            validate successfully for one or more phenotypes.

    """

    association_mode: types.AssociationMode
    genotype_source_config: execution_plan.GenotypeSourceConfig
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
    requested_gpu_genotype_format: types.GpuGenotypeFormat
    gpu_genotype_format: types.GpuGenotypeFormat
    backend_plan: backend_planner.AssociationBackendPlan
    correction_plan: types.BinaryCorrectionPlan
    binary_kernel_config: compute_config.BinaryKernelConfig | None
    linear_numerical_config: compute_config.LinearNumericalConfig | None
    writer_settings: outputs.OutputWriterSettings
    stage_timing_recorder: timing.StageTimingRecorder | None
    telemetry_session: telemetry_events.TelemetrySession | None
    input_fingerprint_cache: outputs.ManifestFileFingerprintCache
    alignment_config: inputs.SampleAlignmentConfigProtocol | None
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...]
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None

    @property
    def uses_packed8_genotypes(self) -> bool:
        """Return whether native delivery should use packed8 probability pairs."""
        return self.backend_plan.uses_variant_major_packed8_delivery

    @property
    def effective_trusted_no_missing_diploid(self) -> bool:
        """Return trusted BGEN mode after packed8 requirements are applied."""
        return schedule.resolve_effective_trusted_no_missing_diploid(
            trusted_no_missing_diploid=self.trusted_no_missing_diploid,
            uses_packed8_genotypes=self.uses_packed8_genotypes,
        )

    @property
    def is_binary_trait(self) -> bool:
        """Return whether this context is for binary trait association."""
        return self.association_mode == types.AssociationMode.REGENIE2_BINARY


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
    run_input: inputs.NativeBgenMultiRunInput
    callback: callbacks.MultiPhenotypeGroupCallbackProtocol
    writer_sessions: tuple[typing.Any, ...]
    committed_chunk_identifier_sets: tuple[set[int], ...]


def build_regenie2_pipeline_context(
    *,
    association_mode: types.AssociationMode,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
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
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    gpu_genotype_format: types.GpuGenotypeFormat,
    correction_plan: types.BinaryCorrectionPlan,
    binary_kernel_config: compute_config.BinaryKernelConfig | None,
    linear_numerical_config: compute_config.LinearNumericalConfig | None,
    writer_settings: outputs.OutputWriterSettings,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry_events.TelemetrySession | None,
    alignment_config: inputs.SampleAlignmentConfigProtocol | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...],
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> Regenie2PipelineContext:
    """Build a resolved lifecycle context for a REGENIE step 2 run."""
    resolved_stage_timing_recorder: timing.StageTimingRecorder | None
    if stage_timing_recorder is None:
        resolved_stage_timing_recorder = timing.build_stage_timing_recorder(
            None,
            force=False,
        )
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
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        gpu_genotype_format=gpu_genotype_format,
        backend_plan=backend_plan,
        correction_plan=correction_plan,
        binary_kernel_config=binary_kernel_config,
        linear_numerical_config=linear_numerical_config,
        writer_settings=writer_settings,
        stage_timing_recorder=resolved_stage_timing_recorder,
        telemetry_session=telemetry_session,
        input_fingerprint_cache=outputs.build_manifest_file_fingerprint_cache(),
        alignment_config=alignment_config,
        phenotype_compute_groups=phenotype_compute_groups,
        runtime_compatibility_token=runtime_compatibility_token,
        output_initialized_callback=output_initialized_callback,
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
