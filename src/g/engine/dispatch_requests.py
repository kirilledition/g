"""Typed engine dispatch requests for REGENIE step 2."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core, execution_plan, types
    from g.engine import timing
    from g.engine.native_dispatch import models as native_dispatch_models
    from g.engine.regenie2_pipeline import compute_config, outputs, telemetry_events


@dataclass(frozen=True)
class PipelineCommonRequest:
    """Shared native pipeline dispatch settings.

    Attributes:
        genotype_source_config: BGEN source configuration.
        phenotype_path: Phenotype file path.
        prediction_list_path: REGENIE step 1 prediction list.
        covariate_path: Optional covariate file path.
        covariate_names: Optional covariate column names.
        chunk_size: Native variant chunk size.
        variant_limit: Optional variant cap.
        staging_depth: Native callback staging depth.
        native_callback_batch_size: Native-to-Python callback chunk batch size.
        result_in_flight_limit: Optional materialization backlog cap.
        dosage_buffer_limit: Optional native dosage decode buffer cap.
        resume: Whether output resume is enabled.
        resume_mode: Resume validation policy.
        writer_settings: Output writer settings.
        trusted_no_missing_diploid: Trusted BGEN fast-path policy.
        trusted_bgen_validation_mode: Trusted BGEN validation policy.
        bgen_decode_tile_variant_count: Native BGEN decode tile size.
        jax_device: Requested JAX device.
        jax_matmul_precision: Optional JAX matmul precision.
        score_dtype: Score-test compute dtype.
        firth_dtype: Firth compute dtype.
        stage_timing_recorder: Optional stage timing recorder.
        telemetry_session: Optional telemetry session.
        alignment_config: Sample alignment settings.
        runtime_compatibility_token: Native token proving runtime checks passed.
        output_initialized_callback: Callback after manifest initialization.

    """

    genotype_source_config: execution_plan.GenotypeSourceConfig
    phenotype_path: Path
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    chunk_size: int
    variant_limit: int | None
    staging_depth: int
    native_callback_batch_size: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    resume: bool
    resume_mode: types.ResumeMode
    writer_settings: outputs.OutputWriterSettings
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    bgen_decode_tile_variant_count: int
    jax_device: types.Device
    jax_matmul_precision: types.JaxMatmulPrecision | None
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    stage_timing_recorder: timing.StageTimingRecorder | None
    telemetry_session: telemetry_events.TelemetrySession | None
    alignment_config: native_dispatch_models.SampleAlignmentConfigProtocol | None
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None


@dataclass(frozen=True)
class SingleTraitPipelineRequest:
    """Request for one-phenotype REGENIE step 2 execution.

    Attributes:
        common: Shared dispatch settings.
        phenotype_name: Phenotype column to run.
        output_run_paths: Output paths for this phenotype.
        existing_manifest: Optional existing output manifest.
        association_mode: Statistical association mode.
        correction_plan: Binary correction settings.
        binary_kernel_config: Binary kernel config when binary.
        linear_numerical_config: Linear numerical config when quantitative.
        gpu_genotype_format: Requested native genotype delivery format.
        null_logistic_nonconvergence_policy: Binary null-model nonconvergence policy.

    """

    common: PipelineCommonRequest
    phenotype_name: str
    output_run_paths: outputs.OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None
    association_mode: types.AssociationMode
    correction_plan: types.BinaryCorrectionPlan
    binary_kernel_config: compute_config.BinaryKernelConfig | None
    linear_numerical_config: compute_config.LinearNumericalConfig | None
    gpu_genotype_format: types.GpuGenotypeFormat
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy


@dataclass(frozen=True)
class MultiTraitPipelineRequest:
    """Request for multi-phenotype REGENIE step 2 execution.

    Attributes:
        common: Shared dispatch settings.
        phenotype_names: Phenotype columns to run.
        output_run_paths_by_phenotype: Output paths in phenotype order.
        existing_manifests_by_phenotype: Optional existing manifests in phenotype order.
        association_mode: Statistical association mode.
        correction_plan: Binary correction settings.
        binary_kernel_config: Binary kernel config when binary.
        linear_numerical_config: Linear numerical config when quantitative.
        gpu_genotype_format: Requested native genotype delivery format.
        null_logistic_nonconvergence_policy: Binary null-model nonconvergence policy.
        sample_mode: Multi-phenotype sample mode.
        phenotype_compute_groups: Planned phenotype compute groups.

    """

    common: PipelineCommonRequest
    phenotype_names: tuple[str, ...]
    output_run_paths_by_phenotype: tuple[outputs.OutputRunPaths, ...]
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None
    association_mode: types.AssociationMode
    correction_plan: types.BinaryCorrectionPlan
    binary_kernel_config: compute_config.BinaryKernelConfig | None
    linear_numerical_config: compute_config.LinearNumericalConfig | None
    gpu_genotype_format: types.GpuGenotypeFormat
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy
    sample_mode: types.MultiPhenotypeSampleMode | None
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None
