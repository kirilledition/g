use serde::Serialize;

use super::{
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS, JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, JaxGpuValidationStatus,
    JaxRuntimeSetupSession, XLA_AUXILIARY_CACHE_DISABLED,
};

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

    let cache_directory = policy.cache_directory.path().to_string_lossy();
    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_persistent_cache_configured",
        "JAX persistent compilation cache enabled.",
        &JaxPersistentCacheConfiguredFields {
            enabled: true,
            cache_directory: Some(cache_directory.as_ref()),
            min_entry_size_bytes: Some(JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES),
            min_compile_time_seconds: Some(JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS),
        },
    )?;

    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_xla_auxiliary_cache_configured",
        "XLA auxiliary persistent cache disabled.",
        &JaxAuxiliaryCacheConfiguredFields {
            enabled: false,
            mode: XLA_AUXILIARY_CACHE_DISABLED,
            reason: "XLA auxiliary caching is fixed off",
        },
    )?;

    emit_jax_runtime_diagnostic(
        telemetry_session,
        thread_name,
        DIAGNOSTIC_LEVEL_INFO,
        "jax_transfer_guard_configured",
        "JAX transfer guard diagnostics disabled.",
        &JaxTransferGuardConfiguredFields { enabled: false },
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::path::PathBuf;

    use super::emit_jax_runtime_setup_diagnostics;
    use crate::jax_runtime::{JaxCacheDirectory, JaxGpuValidationStatus, JaxRuntimePolicy, JaxRuntimeSetupSession};
    use crate::test_support::{TemporaryRunFixture, execute_isolated_test_body};

    #[test]
    fn diagnostics_serialize_cpu_and_failed_gpu_setup_states() {
        if !execute_isolated_test_body(
            "jax_runtime::diagnostics::tests::diagnostics_serialize_cpu_and_failed_gpu_setup_states",
            "G_RUNNER_JAX_DIAGNOSTICS_TEST_CHILD",
        ) {
            return;
        }
        let fixture = TemporaryRunFixture::new();
        let telemetry_path = fixture.root_path().join("telemetry/events.jsonl");
        let policy = g_runtime::NativeRunSessionPolicy {
            log_filter: "info".to_string(),
            log_stderr: false,
            log_file: None,
            telemetry_stream_file: Some(telemetry_path.clone()),
            stage_timing_file: None,
            profile_summary_file: None,
            queue_size: 32,
            lossy: false,
            include_source_location: false,
            include_span_events: false,
        };
        let mut process_runtime_state = g_runtime::ProcessRuntimeState::default();
        let mut native_session = g_runtime::NativeRunSession::new(&mut process_runtime_state, policy)
            .expect("enabled telemetry session should open");
        let telemetry_session = native_session.telemetry_session().clone();

        let cpu_policy = JaxRuntimePolicy {
            device: g_plan::Device::Cpu,
            cache_directory: JaxCacheDirectory::Explicit(PathBuf::from("cpu-cache")),
        };
        let cpu_session = JaxRuntimeSetupSession::new(true, &cpu_policy);
        emit_jax_runtime_setup_diagnostics(&cpu_session, &telemetry_session, "test-thread")
            .expect("CPU diagnostics should serialize");

        let gpu_policy = JaxRuntimePolicy {
            device: g_plan::Device::Gpu,
            cache_directory: JaxCacheDirectory::Explicit(PathBuf::from("gpu-cache")),
        };
        let mut gpu_session = JaxRuntimeSetupSession::new(true, &gpu_policy);
        gpu_session.complete_gpu_validation(JaxGpuValidationStatus::Failed, Cow::Borrowed("no device"));
        emit_jax_runtime_setup_diagnostics(&gpu_session, &telemetry_session, "test-thread")
            .expect("GPU failure diagnostics should serialize");
        telemetry_session.finish("test-thread").expect("telemetry session should finish");
        native_session.finish_logging().expect("logging session should finish");
        drop(native_session);

        let telemetry_text = std::fs::read_to_string(&telemetry_path).expect("telemetry output should be readable");
        let records = telemetry_text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("telemetry line should be valid JSON"))
            .collect::<Vec<_>>();
        let jax_records = records
            .iter()
            .filter(|record| record["event"].as_str().is_some_and(|event| event.starts_with("jax_")))
            .collect::<Vec<_>>();
        assert_eq!(jax_records.len(), 10);
        let cpu_platform = jax_records
            .iter()
            .find(|record| record["event"] == "jax_platform_selected" && record["requested_device"] == "cpu")
            .expect("CPU platform diagnostic should be present");
        assert_eq!(cpu_platform["platform"], "cpu");
        let gpu_platform = jax_records
            .iter()
            .find(|record| record["event"] == "jax_platform_selected" && record["requested_device"] == "gpu")
            .expect("GPU platform diagnostic should be present");
        assert_eq!(gpu_platform["platform"], "cuda");
        let skipped_validation = jax_records
            .iter()
            .find(|record| record["event"] == "jax_gpu_validation" && record["status"] == "skipped")
            .expect("skipped validation diagnostic should be present");
        assert_eq!(skipped_validation["message"], "CPU runtime requested; GPU validation skipped.");
        let failed_validation = jax_records
            .iter()
            .find(|record| record["event"] == "jax_gpu_validation" && record["status"] == "failed")
            .expect("failed validation diagnostic should be present");
        assert_eq!(failed_validation["message"], "no device");
        assert_eq!(records.iter().filter(|record| record["event"] == "telemetry_session_closed").count(), 1);
    }
}
