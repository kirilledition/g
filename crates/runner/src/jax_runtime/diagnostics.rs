use serde::Serialize;

use super::{JaxGpuValidationStatus, JaxRuntimeSetupSession};

const DIAGNOSTIC_LEVEL_ERROR: &str = "error";
const DIAGNOSTIC_LEVEL_INFO: &str = "info";

#[derive(Serialize)]
struct JaxPlatformSelectedFields<'fields> {
    requested_device: &'fields str,
    platform: &'fields str,
}

#[derive(Serialize)]
struct JaxPersistentCacheConfiguredFields<'fields> {
    enabled: bool,
    cache_directory: Option<&'fields str>,
    min_entry_size_bytes: Option<i64>,
    min_compile_time_seconds: Option<i64>,
}

#[derive(Serialize)]
struct JaxAuxiliaryCacheConfiguredFields<'fields> {
    enabled: bool,
    mode: &'fields str,
    reason: &'fields str,
}

#[derive(Serialize)]
struct JaxTransferGuardConfiguredFields {
    enabled: bool,
}

#[derive(Serialize)]
struct JaxGpuValidationFields<'fields> {
    status: &'fields str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<&'fields str>,
}

/// Emit the typed diagnostics derived from one completed JAX setup.
///
/// # Errors
///
/// Returns an error message when diagnostic serialization or telemetry output
/// fails.
pub(crate) fn emit_jax_runtime_setup_diagnostics(
    setup_session: &JaxRuntimeSetupSession<'_>,
    telemetry_session: &g_runtime::TelemetryRunSession,
    thread_name: &str,
) -> Result<(), String> {
    let policy = setup_session.policy;
    let platform_name = policy.platform_name();
    let platform_message = match policy.device {
        g_plan::Device::Cpu => "Selected JAX platform cpu.",
        g_plan::Device::Gpu => "Selected JAX platform cuda.",
    };
    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_platform_selected",
        platform_message,
        &JaxPlatformSelectedFields { requested_device: policy.device.as_str(), platform: platform_name },
    )?;

    let cache_policy = policy.persistent_cache.as_ref();
    let cache_directory = cache_policy.map(|cache_policy| cache_policy.directory.path().to_string_lossy());
    let persistent_cache_message = if cache_policy.is_some() {
        "JAX persistent compilation cache enabled."
    } else {
        "JAX persistent compilation cache disabled."
    };
    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_persistent_cache_configured",
        persistent_cache_message,
        &JaxPersistentCacheConfiguredFields {
            enabled: cache_policy.is_some(),
            cache_directory: cache_directory.as_deref(),
            min_entry_size_bytes: cache_policy.map(|cache_policy| cache_policy.min_entry_size_bytes),
            min_compile_time_seconds: cache_policy.map(|cache_policy| cache_policy.min_compile_time_seconds),
        },
    )?;

    let xla_auxiliary_cache_mode = policy.xla_auxiliary_cache_mode();
    let xla_auxiliary_cache_enabled = cache_policy.is_some_and(|cache_policy| cache_policy.xla_autotune_cache_enabled);
    let xla_auxiliary_cache_message = if xla_auxiliary_cache_enabled {
        "XLA auxiliary persistent cache enabled."
    } else {
        "XLA auxiliary persistent cache disabled."
    };
    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_xla_auxiliary_cache_configured",
        xla_auxiliary_cache_message,
        &JaxAuxiliaryCacheConfiguredFields {
            enabled: xla_auxiliary_cache_enabled,
            mode: xla_auxiliary_cache_mode,
            reason: policy.xla_auxiliary_cache_reason(),
        },
    )?;

    let transfer_guard_message = if policy.transfer_guard_enabled {
        "JAX transfer guard diagnostics enabled."
    } else {
        "JAX transfer guard diagnostics disabled."
    };
    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_transfer_guard_configured",
        transfer_guard_message,
        &JaxTransferGuardConfiguredFields { enabled: policy.transfer_guard_enabled },
    )?;

    let gpu_validation_status = setup_session.gpu_validation_status;
    let gpu_validation_level = if gpu_validation_status == JaxGpuValidationStatus::Failed {
        DIAGNOSTIC_LEVEL_ERROR
    } else {
        DIAGNOSTIC_LEVEL_INFO
    };
    let gpu_validation_status_name = gpu_validation_status.as_str();
    let gpu_validation_message = match gpu_validation_status {
        JaxGpuValidationStatus::Pending => "JAX GPU validation pending.",
        JaxGpuValidationStatus::Skipped => "JAX GPU validation skipped.",
        JaxGpuValidationStatus::Succeeded => "JAX GPU validation succeeded.",
        JaxGpuValidationStatus::Failed => "JAX GPU validation failed.",
    };
    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        gpu_validation_level,
        "jax_gpu_validation",
        gpu_validation_message,
        &JaxGpuValidationFields {
            status: gpu_validation_status_name,
            message: setup_session.gpu_validation_message.as_deref(),
        },
    )
}

fn emit_jax_runtime_diagnostic<Fields>(
    telemetry_session: &g_runtime::TelemetryRunSession,
    thread_name: &str,
    level: &str,
    event_name: &str,
    message: &str,
    fields: &Fields,
) -> Result<(), String>
where
    Fields: Serialize,
{
    if telemetry_session.is_enabled() {
        return telemetry_session
            .emit_current_event(thread_name, event_name, level, fields)
            .map_err(|error| error.to_string());
    }
    g_runtime::emit_diagnostic_event(level, event_name, message, fields)
        .map_err(|error| format!("Failed to serialize JAX runtime diagnostic event fields: {error}"))
}
