use super::{
    DEVICE_GPU, JAX_CPU_PLATFORM_NAME, JAX_CUDA_PLATFORM_NAME, JAX_MATMUL_PRECISION_FLOAT32, JaxRuntimeSetupPayload,
    JaxRuntimeSetupSideEffectPlan, XLA_AUXILIARY_CACHE_DISABLED, XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE,
};

#[allow(clippy::fn_params_excessive_bools)]
#[must_use]
pub fn resolve_jax_runtime_setup(
    requested_device: &str,
    cache_directory: &str,
    matmul_precision: Option<&str>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> JaxRuntimeSetupPayload {
    let (gpu_validation_status, gpu_validation_message) = if requested_device == DEVICE_GPU {
        ("pending".to_string(), None)
    } else {
        ("skipped".to_string(), Some("CPU runtime requested; GPU validation skipped.".to_string()))
    };
    let platform_name = if requested_device == DEVICE_GPU { JAX_CUDA_PLATFORM_NAME } else { JAX_CPU_PLATFORM_NAME };
    let matmul_precision = matmul_precision.unwrap_or(JAX_MATMUL_PRECISION_FLOAT32).to_string();
    let (xla_auxiliary_cache_mode, xla_auxiliary_cache_reason) = if persistent_cache && xla_autotune_cache {
        (XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE, "XLA auxiliary cache was requested")
    } else if persistent_cache {
        (XLA_AUXILIARY_CACHE_DISABLED, "XLA auxiliary cache was not requested")
    } else {
        (XLA_AUXILIARY_CACHE_DISABLED, "persistent compilation cache is disabled")
    };
    JaxRuntimeSetupPayload {
        requested_device: requested_device.to_string(),
        platform_name: platform_name.to_string(),
        cache_directory: cache_directory.to_string(),
        matmul_precision,
        persistent_cache_enabled: persistent_cache,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode: xla_auxiliary_cache_mode.to_string(),
        xla_auxiliary_cache_reason: xla_auxiliary_cache_reason.to_string(),
        transfer_guard_enabled: transfer_guard,
        gpu_validation_status,
        gpu_validation_message,
    }
}

#[must_use]
pub fn plan_jax_runtime_setup_side_effects(
    requested_device: &str,
    persistent_cache_enabled: bool,
) -> JaxRuntimeSetupSideEffectPlan {
    JaxRuntimeSetupSideEffectPlan {
        should_create_cache_directory: persistent_cache_enabled,
        should_validate_gpu: requested_device == DEVICE_GPU,
    }
}

#[must_use]
pub fn complete_jax_runtime_setup_validation(
    setup: &JaxRuntimeSetupPayload,
    gpu_validation_status: &str,
    gpu_validation_message: Option<&str>,
) -> JaxRuntimeSetupPayload {
    let mut completed_setup = setup.clone();
    completed_setup.gpu_validation_status = gpu_validation_status.to_string();
    completed_setup.gpu_validation_message = gpu_validation_message.map(str::to_string);
    completed_setup
}
