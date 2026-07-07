//! BGEN delivery and GPU genotype-format scheduling policy.

use crate::schedule::ScheduleError;

const DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE: i64 = 1;
const GPU_GENOTYPE_FORMAT_AUTO: &str = "auto";
const GPU_GENOTYPE_FORMAT_DOSAGE: &str = "dosage";
const GPU_GENOTYPE_FORMAT_PACKED8: &str = "packed8";
const JAX_DEVICE_GPU: &str = "gpu";
const JAX_DEVICE_CPU: &str = "cpu";
const GPU_FORMAT_RESOLUTION_EXPLICIT: &str = "explicit";
const GPU_FORMAT_RESOLUTION_RESUME_MANIFEST: &str = "resume_manifest";
const GPU_FORMAT_RESOLUTION_NON_GPU_DEVICE: &str = "non_gpu_device";
const GPU_FORMAT_RESOLUTION_TRUSTED_VALIDATION_PASSED: &str = "trusted_validation_passed";
const GPU_FORMAT_RESOLUTION_TRUSTED_VALIDATION_FAILED: &str = "trusted_validation_failed";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuGenotypeFormatResolutionPlan {
    pub requested_gpu_genotype_format: String,
    pub resolved_gpu_genotype_format: Option<String>,
    pub resolution_reason: Option<String>,
    pub fallback_error: Option<String>,
    pub requires_trusted_validation: bool,
}

impl GpuGenotypeFormatResolutionPlan {
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.resolved_gpu_genotype_format.is_some()
    }

    #[must_use]
    pub fn should_log_auto_resolution(&self) -> bool {
        self.resolution_reason
            .as_deref()
            .is_some_and(|resolution_reason| resolution_reason != GPU_FORMAT_RESOLUTION_EXPLICIT)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryCleanupOutcome {
    Failure,
    Interrupted,
    InterruptedCleanupFailure,
    Success,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryCleanupAction {
    AbortCallback,
    AbortWriterSessions,
    DrainCallback,
    FinishInterruptedWriterSessions,
    FinishWriterSessions,
    WriteStageTimingSnapshot,
}

impl BgenDeliveryCleanupAction {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::AbortCallback => "abort_callback",
            Self::AbortWriterSessions => "abort_writer_sessions",
            Self::DrainCallback => "drain_callback",
            Self::FinishInterruptedWriterSessions => "finish_interrupted_writer_sessions",
            Self::FinishWriterSessions => "finish_writer_sessions",
            Self::WriteStageTimingSnapshot => "write_stage_timing_snapshot",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BgenDeliveryCleanupPlan {
    cleanup_actions: Vec<BgenDeliveryCleanupAction>,
}

impl BgenDeliveryCleanupPlan {
    #[must_use]
    pub fn cleanup_actions(&self) -> &[BgenDeliveryCleanupAction] {
        &self.cleanup_actions
    }

    #[must_use]
    pub fn drain_callback(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::DrainCallback)
    }

    #[must_use]
    pub fn finish_writer_sessions(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::FinishWriterSessions)
    }

    #[must_use]
    pub fn finish_interrupted_writer_sessions(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::FinishInterruptedWriterSessions)
    }

    #[must_use]
    pub fn abort_callback(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::AbortCallback)
    }

    #[must_use]
    pub fn abort_writer_sessions(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::AbortWriterSessions)
    }

    #[must_use]
    pub fn write_stage_timing_snapshot(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::WriteStageTimingSnapshot)
    }

    fn contains_cleanup_action(&self, cleanup_action: BgenDeliveryCleanupAction) -> bool {
        self.cleanup_actions.contains(&cleanup_action)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryMethod {
    DosageNativeMultiAlignedSamples,
    DosageNativeAlignedSamples,
    DosageSampleIndices,
    Packed8NativeMultiAlignedSamples,
    Packed8NativeAlignedSamples,
    Packed8SampleIndices,
}

impl BgenDeliveryMethod {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::DosageNativeMultiAlignedSamples => "dosage_native_multi_aligned_samples",
            Self::DosageNativeAlignedSamples => "dosage_native_aligned_samples",
            Self::DosageSampleIndices => "dosage_sample_indices",
            Self::Packed8NativeMultiAlignedSamples => "packed8_native_multi_aligned_samples",
            Self::Packed8NativeAlignedSamples => "packed8_native_aligned_samples",
            Self::Packed8SampleIndices => "packed8_sample_indices",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenDeliveryInvocationPlan {
    pub delivery_method: BgenDeliveryMethod,
    pub callback_batch_size: usize,
}

fn gpu_genotype_format_is_supported(gpu_genotype_format: &str) -> bool {
    matches!(gpu_genotype_format, GPU_GENOTYPE_FORMAT_AUTO | GPU_GENOTYPE_FORMAT_DOSAGE | GPU_GENOTYPE_FORMAT_PACKED8,)
}

fn concrete_gpu_genotype_format(gpu_genotype_format: &str) -> Option<&'static str> {
    match gpu_genotype_format {
        GPU_GENOTYPE_FORMAT_DOSAGE => Some(GPU_GENOTYPE_FORMAT_DOSAGE),
        GPU_GENOTYPE_FORMAT_PACKED8 => Some(GPU_GENOTYPE_FORMAT_PACKED8),
        _ => None,
    }
}

fn validate_gpu_genotype_format(gpu_genotype_format: &str) -> Result<(), ScheduleError> {
    if gpu_genotype_format_is_supported(gpu_genotype_format) {
        return Ok(());
    }
    Err(ScheduleError::UnsupportedGpuGenotypeFormat { gpu_genotype_format: gpu_genotype_format.to_string() })
}

fn validate_jax_device(jax_device: &str) -> Result<(), ScheduleError> {
    if matches!(jax_device, JAX_DEVICE_CPU | JAX_DEVICE_GPU) {
        return Ok(());
    }
    Err(ScheduleError::UnsupportedJaxDevice { jax_device: jax_device.to_string() })
}

fn resolved_gpu_genotype_format_plan(
    requested_gpu_genotype_format: &str,
    resolved_gpu_genotype_format: &str,
    resolution_reason: &str,
    fallback_error: Option<String>,
) -> GpuGenotypeFormatResolutionPlan {
    GpuGenotypeFormatResolutionPlan {
        requested_gpu_genotype_format: requested_gpu_genotype_format.to_string(),
        resolved_gpu_genotype_format: Some(resolved_gpu_genotype_format.to_string()),
        resolution_reason: Some(resolution_reason.to_string()),
        fallback_error,
        requires_trusted_validation: false,
    }
}

fn trusted_validation_required_gpu_genotype_format_plan(
    requested_gpu_genotype_format: &str,
) -> GpuGenotypeFormatResolutionPlan {
    GpuGenotypeFormatResolutionPlan {
        requested_gpu_genotype_format: requested_gpu_genotype_format.to_string(),
        resolved_gpu_genotype_format: None,
        resolution_reason: None,
        fallback_error: None,
        requires_trusted_validation: true,
    }
}

#[must_use]
pub fn resolve_manifest_gpu_genotype_format(
    resume: bool,
    manifest_gpu_genotype_format: Option<&str>,
    association_backend_genotype_format: Option<&str>,
) -> Option<&'static str> {
    if !resume {
        return None;
    }
    match manifest_gpu_genotype_format {
        Some(gpu_genotype_format) => concrete_gpu_genotype_format(gpu_genotype_format),
        None => association_backend_genotype_format.and_then(concrete_gpu_genotype_format),
    }
}

#[must_use]
pub fn resolve_effective_trusted_no_missing_diploid(
    requested_trusted_no_missing_diploid: bool,
    variant_major_packed8_probability_pairs: bool,
) -> bool {
    requested_trusted_no_missing_diploid || variant_major_packed8_probability_pairs
}

/// Resolve an `auto` GPU genotype format to dosage for paths that cannot use
/// packed8.
///
/// # Errors
///
/// Returns an error when the requested GPU genotype format is unsupported.
pub fn plan_gpu_genotype_format_auto_to_dosage(
    requested_gpu_genotype_format: &str,
    resolution_reason: &str,
) -> Result<GpuGenotypeFormatResolutionPlan, ScheduleError> {
    validate_gpu_genotype_format(requested_gpu_genotype_format)?;
    if requested_gpu_genotype_format != GPU_GENOTYPE_FORMAT_AUTO {
        return Ok(resolved_gpu_genotype_format_plan(
            requested_gpu_genotype_format,
            requested_gpu_genotype_format,
            GPU_FORMAT_RESOLUTION_EXPLICIT,
            None,
        ));
    }
    Ok(resolved_gpu_genotype_format_plan(
        requested_gpu_genotype_format,
        GPU_GENOTYPE_FORMAT_DOSAGE,
        resolution_reason,
        None,
    ))
}

/// Plan single-trait binary `gpu_genotype_format=auto` resolution before any
/// BGEN trusted validation side effects.
///
/// # Errors
///
/// Returns an error when the requested GPU genotype format or JAX device value
/// is unsupported.
pub fn plan_single_trait_binary_gpu_genotype_format_resolution(
    requested_gpu_genotype_format: &str,
    manifest_gpu_genotype_format: Option<&str>,
    association_backend_genotype_format: Option<&str>,
    resume: bool,
    jax_device: &str,
) -> Result<GpuGenotypeFormatResolutionPlan, ScheduleError> {
    validate_gpu_genotype_format(requested_gpu_genotype_format)?;
    validate_jax_device(jax_device)?;
    if requested_gpu_genotype_format != GPU_GENOTYPE_FORMAT_AUTO {
        return Ok(resolved_gpu_genotype_format_plan(
            requested_gpu_genotype_format,
            requested_gpu_genotype_format,
            GPU_FORMAT_RESOLUTION_EXPLICIT,
            None,
        ));
    }
    if let Some(manifest_gpu_genotype_format) =
        resolve_manifest_gpu_genotype_format(resume, manifest_gpu_genotype_format, association_backend_genotype_format)
    {
        return Ok(resolved_gpu_genotype_format_plan(
            requested_gpu_genotype_format,
            manifest_gpu_genotype_format,
            GPU_FORMAT_RESOLUTION_RESUME_MANIFEST,
            None,
        ));
    }
    if jax_device != JAX_DEVICE_GPU {
        return Ok(resolved_gpu_genotype_format_plan(
            requested_gpu_genotype_format,
            GPU_GENOTYPE_FORMAT_DOSAGE,
            GPU_FORMAT_RESOLUTION_NON_GPU_DEVICE,
            None,
        ));
    }
    Ok(trusted_validation_required_gpu_genotype_format_plan(requested_gpu_genotype_format))
}

#[must_use]
pub fn plan_auto_gpu_genotype_format_after_trusted_validation(
    fallback_error: Option<&str>,
) -> GpuGenotypeFormatResolutionPlan {
    if let Some(fallback_error) = fallback_error {
        return resolved_gpu_genotype_format_plan(
            GPU_GENOTYPE_FORMAT_AUTO,
            GPU_GENOTYPE_FORMAT_DOSAGE,
            GPU_FORMAT_RESOLUTION_TRUSTED_VALIDATION_FAILED,
            Some(fallback_error.to_string()),
        );
    }
    resolved_gpu_genotype_format_plan(
        GPU_GENOTYPE_FORMAT_AUTO,
        GPU_GENOTYPE_FORMAT_PACKED8,
        GPU_FORMAT_RESOLUTION_TRUSTED_VALIDATION_PASSED,
        None,
    )
}

/// Resolve the callback batch size for one native BGEN delivery mode.
///
/// # Errors
///
/// Returns an error when the requested batch size is non-positive, cannot fit
/// in `usize`, or requests packed8 callback batching, which is not supported.
pub fn resolve_delivery_callback_batch_size(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
) -> Result<usize, ScheduleError> {
    let requested_callback_batch_size = callback_batch_size.unwrap_or(DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE);
    if requested_callback_batch_size <= 0 {
        return Err(ScheduleError::NonPositiveCallbackBatchSize);
    }
    let resolved_callback_batch_size = usize::try_from(requested_callback_batch_size)
        .map_err(|_| ScheduleError::CallbackBatchSizeOverflow { callback_batch_size: requested_callback_batch_size })?;
    if variant_major_packed8_probability_pairs && resolved_callback_batch_size > 1 {
        return Err(ScheduleError::Packed8CallbackBatchSize);
    }
    Ok(resolved_callback_batch_size)
}

/// Plan one native BGEN delivery invocation.
///
/// # Errors
///
/// Returns an error when the requested callback batch size is invalid for the
/// selected delivery mode.
pub fn plan_bgen_delivery_invocation(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> Result<BgenDeliveryInvocationPlan, ScheduleError> {
    let callback_batch_size =
        resolve_delivery_callback_batch_size(callback_batch_size, variant_major_packed8_probability_pairs)?;
    let delivery_method = resolve_bgen_delivery_method(
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    );
    Ok(BgenDeliveryInvocationPlan { delivery_method, callback_batch_size })
}

/// Resolve the callback batch size for grouped union BGEN delivery.
///
/// # Errors
///
/// Returns an error when the requested batch size is non-positive, cannot fit
/// in `usize`, or requests grouped union callback batching, which is not
/// supported.
pub fn resolve_grouped_union_callback_batch_size(callback_batch_size: i64) -> Result<usize, ScheduleError> {
    let resolved_callback_batch_size = resolve_delivery_callback_batch_size(Some(callback_batch_size), false)?;
    if resolved_callback_batch_size > 1 {
        return Err(ScheduleError::GroupedUnionCallbackBatchSize);
    }
    Ok(resolved_callback_batch_size)
}

#[must_use]
pub const fn resolve_bgen_delivery_method(
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> BgenDeliveryMethod {
    match (
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    ) {
        (true, true, _) => BgenDeliveryMethod::Packed8NativeMultiAlignedSamples,
        (true, false, true) => BgenDeliveryMethod::Packed8NativeAlignedSamples,
        (true, false, false) => BgenDeliveryMethod::Packed8SampleIndices,
        (false, true, _) => BgenDeliveryMethod::DosageNativeMultiAlignedSamples,
        (false, false, true) => BgenDeliveryMethod::DosageNativeAlignedSamples,
        (false, false, false) => BgenDeliveryMethod::DosageSampleIndices,
    }
}

/// Plan cleanup side effects after native BGEN delivery exits.
#[must_use]
pub fn plan_bgen_delivery_cleanup(
    cleanup_outcome: BgenDeliveryCleanupOutcome,
    callback_finished: bool,
) -> BgenDeliveryCleanupPlan {
    match cleanup_outcome {
        BgenDeliveryCleanupOutcome::Success => build_bgen_delivery_cleanup_plan(&[
            BgenDeliveryCleanupAction::DrainCallback,
            BgenDeliveryCleanupAction::FinishWriterSessions,
            BgenDeliveryCleanupAction::WriteStageTimingSnapshot,
        ]),
        BgenDeliveryCleanupOutcome::Interrupted => {
            let mut cleanup_actions =
                if callback_finished { Vec::new() } else { vec![BgenDeliveryCleanupAction::DrainCallback] };
            cleanup_actions.extend([
                BgenDeliveryCleanupAction::FinishInterruptedWriterSessions,
                BgenDeliveryCleanupAction::WriteStageTimingSnapshot,
            ]);
            build_bgen_delivery_cleanup_plan(&cleanup_actions)
        }
        BgenDeliveryCleanupOutcome::Failure | BgenDeliveryCleanupOutcome::InterruptedCleanupFailure => {
            build_bgen_delivery_cleanup_plan(&[
                BgenDeliveryCleanupAction::AbortCallback,
                BgenDeliveryCleanupAction::AbortWriterSessions,
                BgenDeliveryCleanupAction::WriteStageTimingSnapshot,
            ])
        }
    }
}

pub(crate) fn build_bgen_delivery_cleanup_plan(
    cleanup_actions: &[BgenDeliveryCleanupAction],
) -> BgenDeliveryCleanupPlan {
    BgenDeliveryCleanupPlan { cleanup_actions: cleanup_actions.to_vec() }
}
