use crate::schedule::ScheduleError;

use super::types::GpuGenotypeFormatResolutionPlan;

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
