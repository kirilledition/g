//! Deterministic JAX runtime setup policy and diagnostics.

use std::fs;

mod config_updates;
mod diagnostics;
mod gpu_validation;
mod setup;

pub use config_updates::plan_jax_runtime_config_updates;
pub use diagnostics::{
    build_jax_runtime_setup_diagnostic_events, plan_jax_runtime_diagnostic_record,
    serialize_jax_runtime_diagnostic_fields_json,
};
pub use gpu_validation::{default_nvidia_driver_probe_paths, nvidia_driver_files_are_visible, plan_jax_gpu_validation};
pub use setup::{
    complete_jax_runtime_setup_validation, plan_jax_runtime_setup_side_effects, resolve_jax_runtime_setup,
};

const DEVICE_GPU: &str = "gpu";
const JAX_CONFIG_COMPILATION_CACHE_DIR: &str = "jax_compilation_cache_dir";
const JAX_CONFIG_DEFAULT_MATMUL_PRECISION: &str = "jax_default_matmul_precision";
const JAX_CONFIG_ENABLE_X64: &str = "jax_enable_x64";
const JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES: &str = "jax_persistent_cache_enable_xla_caches";
const JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS: &str = "jax_persistent_cache_min_compile_time_secs";
const JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES: &str = "jax_persistent_cache_min_entry_size_bytes";
const JAX_CONFIG_PLATFORMS: &str = "jax_platforms";
const JAX_CONFIG_TRANSFER_GUARD: &str = "jax_transfer_guard";
const JAX_CUDA_PLATFORM_NAME: &str = "cuda";
const JAX_CPU_PLATFORM_NAME: &str = "cpu";
const JAX_GPU_DEVICE_PLATFORM_NAME: &str = "gpu";
const JAX_MATMUL_PRECISION_FLOAT32: &str = "float32";
const JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR: &str = "error";
const JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO: &str = "info";
const PYTHON_LOGGING_LEVEL_ERROR: &str = "ERROR";
const PYTHON_LOGGING_LEVEL_INFO: &str = "INFO";
const JAX_RUNTIME_GPU_VALIDATION_FAILED: &str = "failed";
const JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED: &str = "succeeded";
const JAX_TRANSFER_GUARD_DISALLOW: &str = "disallow";
const NVIDIA_CONTROL_DEVICE_PATH: &str = "/dev/nvidiactl";
const NVIDIA_DRIVER_DIRECTORY_PATH: &str = "/proc/driver/nvidia";
const NVIDIA_UVM_DEVICE_PATH: &str = "/dev/nvidia-uvm";
const XLA_AUXILIARY_CACHE_DISABLED: &str = "none";
const XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE: &str = "xla_gpu_per_fusion_autotune_cache_dir";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeSetupPayload {
    pub requested_device: String,
    pub platform_name: String,
    pub cache_directory: String,
    pub matmul_precision: String,
    pub persistent_cache_enabled: bool,
    pub persistent_cache_min_entry_size_bytes: i64,
    pub persistent_cache_min_compile_time_seconds: i64,
    pub xla_auxiliary_cache_mode: String,
    pub xla_auxiliary_cache_reason: String,
    pub transfer_guard_enabled: bool,
    pub gpu_validation_status: String,
    pub gpu_validation_message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxDeviceObservation {
    pub platform: String,
    pub description: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxGpuValidationPlan {
    pub status: String,
    pub message: String,
    pub should_raise: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvidiaDriverProbePathsPayload {
    pub control_device_path: String,
    pub uvm_device_path: String,
    pub driver_directory_path: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeSetupSideEffectPlan {
    pub should_create_cache_directory: bool,
    pub should_validate_gpu: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeSetupSession {
    should_configure: bool,
    setup: JaxRuntimeSetupPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JaxRuntimeConfigValue {
    Boolean(bool),
    Integer(i64),
    Text(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeConfigUpdatePayload {
    pub setting_name: String,
    pub value: JaxRuntimeConfigValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JaxRuntimeDiagnosticValue {
    Boolean(bool),
    Integer(i64),
    Text(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeDiagnosticFieldPayload {
    pub name: String,
    pub value: JaxRuntimeDiagnosticValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeDiagnosticEventPayload {
    pub event_name: String,
    pub level: String,
    pub message: String,
    pub fields: Vec<JaxRuntimeDiagnosticFieldPayload>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeDiagnosticRecordPlan {
    pub logging_level_name: String,
    pub should_emit_telemetry: bool,
    pub telemetry_level: String,
}

impl JaxRuntimeSetupSession {
    #[must_use]
    pub fn new(should_configure: bool, setup: JaxRuntimeSetupPayload) -> Self {
        Self { should_configure, setup }
    }

    #[must_use]
    pub const fn should_configure(&self) -> bool {
        self.should_configure
    }

    #[must_use]
    pub const fn setup(&self) -> &JaxRuntimeSetupPayload {
        &self.setup
    }

    #[must_use]
    pub fn side_effect_plan(&self) -> JaxRuntimeSetupSideEffectPlan {
        plan_jax_runtime_setup_side_effects(&self.setup.requested_device, self.setup.persistent_cache_enabled)
    }

    #[must_use]
    pub fn config_updates(&self) -> Vec<JaxRuntimeConfigUpdatePayload> {
        plan_jax_runtime_config_updates(&self.setup)
    }

    #[must_use]
    pub fn diagnostic_events(&self) -> Vec<JaxRuntimeDiagnosticEventPayload> {
        build_jax_runtime_setup_diagnostic_events(&self.setup)
    }

    /// Create the persistent JAX cache directory when requested by setup policy.
    ///
    /// # Errors
    ///
    /// Returns an error when persistent caching is enabled and the cache
    /// directory cannot be created.
    pub fn create_cache_directory_if_configured(&self) -> Result<bool, std::io::Error> {
        if !self.side_effect_plan().should_create_cache_directory {
            return Ok(false);
        }
        fs::create_dir_all(&self.setup.cache_directory)?;
        Ok(true)
    }

    #[must_use]
    pub fn complete_validation(
        &mut self,
        gpu_validation_status: &str,
        gpu_validation_message: Option<&str>,
    ) -> JaxRuntimeSetupPayload {
        self.setup = complete_jax_runtime_setup_validation(&self.setup, gpu_validation_status, gpu_validation_message);
        self.setup.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_test_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("g-runtime-{name}-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn resolves_jax_runtime_setup_payload() {
        let setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 1024, 5, true, true);
        assert_eq!(setup.platform_name, JAX_CUDA_PLATFORM_NAME);
        assert_eq!(setup.matmul_precision, JAX_MATMUL_PRECISION_FLOAT32);
        assert_eq!(setup.xla_auxiliary_cache_mode, XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE);
        assert!(setup.transfer_guard_enabled);

        let cpu_setup = resolve_jax_runtime_setup("cpu", "cache", Some("highest"), false, 0, 0, true, false);
        assert_eq!(cpu_setup.platform_name, JAX_CPU_PLATFORM_NAME);
        assert_eq!(cpu_setup.gpu_validation_status, "skipped");
        assert_eq!(cpu_setup.matmul_precision, "highest");
    }

    #[test]
    fn plans_jax_runtime_diagnostic_recording() {
        assert_eq!(
            plan_jax_runtime_diagnostic_record("info", true),
            JaxRuntimeDiagnosticRecordPlan {
                logging_level_name: "INFO".to_string(),
                should_emit_telemetry: true,
                telemetry_level: "info".to_string(),
            },
        );
        assert_eq!(
            plan_jax_runtime_diagnostic_record("error", false),
            JaxRuntimeDiagnosticRecordPlan {
                logging_level_name: "ERROR".to_string(),
                should_emit_telemetry: false,
                telemetry_level: "error".to_string(),
            },
        );
    }

    #[test]
    fn plans_jax_runtime_setup_side_effects() {
        assert_eq!(
            plan_jax_runtime_setup_side_effects("cpu", true),
            JaxRuntimeSetupSideEffectPlan { should_create_cache_directory: true, should_validate_gpu: false },
        );
        assert_eq!(
            plan_jax_runtime_setup_side_effects("gpu", false),
            JaxRuntimeSetupSideEffectPlan { should_create_cache_directory: false, should_validate_gpu: true },
        );
    }

    #[test]
    fn jax_runtime_setup_session_owns_setup_state() {
        let setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 0, 0, false, false);
        let mut session = JaxRuntimeSetupSession::new(true, setup);

        assert!(session.should_configure());
        assert_eq!(
            session.side_effect_plan(),
            JaxRuntimeSetupSideEffectPlan { should_create_cache_directory: true, should_validate_gpu: true },
        );
        assert_eq!(session.config_updates()[0].setting_name, JAX_CONFIG_PLATFORMS);

        let completed_setup = session.complete_validation(JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED, Some("gpu ready"));

        assert_eq!(completed_setup.gpu_validation_status, JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED);
        assert_eq!(session.setup().gpu_validation_message, Some("gpu ready".to_string()));
        assert_eq!(
            session.diagnostic_events()[4].fields[0].value,
            JaxRuntimeDiagnosticValue::Text("succeeded".to_string())
        );
    }

    #[test]
    fn jax_runtime_setup_session_owns_cache_directory_creation() {
        let cache_directory = temporary_test_path("jax-cache-create");
        let setup = resolve_jax_runtime_setup(
            "cpu",
            cache_directory.to_str().expect("cache path should be valid UTF-8"),
            None,
            true,
            0,
            0,
            false,
            false,
        );
        let session = JaxRuntimeSetupSession::new(true, setup);

        assert!(session.create_cache_directory_if_configured().expect("cache directory should be created"));
        assert!(cache_directory.exists());

        std::fs::remove_dir_all(cache_directory).expect("remove JAX cache test directory");
    }

    #[test]
    fn jax_runtime_setup_session_skips_cache_directory_without_persistent_cache() {
        let cache_directory = temporary_test_path("jax-cache-skip");
        let setup = resolve_jax_runtime_setup(
            "cpu",
            cache_directory.to_str().expect("cache path should be valid UTF-8"),
            None,
            false,
            0,
            0,
            false,
            false,
        );
        let session = JaxRuntimeSetupSession::new(true, setup);

        assert!(!session.create_cache_directory_if_configured().expect("cache directory should be skipped"));
        assert!(!cache_directory.exists());
    }

    #[test]
    fn detects_visible_nvidia_driver_files() {
        let test_root = temporary_test_path("nvidia-driver");
        let control_device_path = test_root.join("nvidiactl");
        let uvm_device_path = test_root.join("nvidia-uvm");
        let driver_directory_path = test_root.join("driver");
        std::fs::create_dir_all(&test_root).expect("create nvidia driver test root");

        assert!(!nvidia_driver_files_are_visible(&control_device_path, &uvm_device_path, &driver_directory_path,));

        std::fs::write(&uvm_device_path, b"").expect("create nvidia uvm test file");

        assert!(nvidia_driver_files_are_visible(&control_device_path, &uvm_device_path, &driver_directory_path,));

        std::fs::remove_dir_all(test_root).expect("remove nvidia driver test root");
    }

    #[test]
    fn exposes_default_nvidia_driver_probe_paths() {
        assert_eq!(
            default_nvidia_driver_probe_paths(),
            NvidiaDriverProbePathsPayload {
                control_device_path: "/dev/nvidiactl".to_string(),
                uvm_device_path: "/dev/nvidia-uvm".to_string(),
                driver_directory_path: "/proc/driver/nvidia".to_string(),
            },
        );
    }

    #[test]
    fn completes_jax_runtime_setup_validation() {
        let setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 0, 0, false, false);

        let completed_setup =
            complete_jax_runtime_setup_validation(&setup, JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED, Some("gpu ready"));

        assert_eq!(completed_setup.requested_device, "gpu");
        assert_eq!(completed_setup.cache_directory, "cache");
        assert_eq!(completed_setup.gpu_validation_status, JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED);
        assert_eq!(completed_setup.gpu_validation_message, Some("gpu ready".to_string()));
    }

    #[test]
    fn builds_jax_runtime_setup_diagnostic_events() {
        let setup = resolve_jax_runtime_setup("cpu", "cache", None, false, 0, 0, true, true);

        let events = build_jax_runtime_setup_diagnostic_events(&setup);

        assert_eq!(events.iter().map(|event| event.event_name.as_str()).collect::<Vec<_>>(), {
            vec![
                "jax_platform_selected",
                "jax_persistent_cache_configured",
                "jax_xla_auxiliary_cache_configured",
                "jax_transfer_guard_configured",
                "jax_gpu_validation",
            ]
        });
        assert_eq!(events[0].fields[1].value, JaxRuntimeDiagnosticValue::Text("cpu".to_string()));
        assert_eq!(events[1].fields[0].value, JaxRuntimeDiagnosticValue::Boolean(false));
        assert_eq!(events[2].fields[0].value, JaxRuntimeDiagnosticValue::Boolean(false));
        assert_eq!(events[3].fields[0].value, JaxRuntimeDiagnosticValue::Boolean(true));
        assert_eq!(events[4].fields[0].value, JaxRuntimeDiagnosticValue::Text("skipped".to_string()));
    }

    #[test]
    fn serializes_jax_runtime_diagnostic_fields_json() {
        let fields = vec![
            JaxRuntimeDiagnosticFieldPayload {
                name: "enabled".to_string(),
                value: JaxRuntimeDiagnosticValue::Boolean(true),
            },
            JaxRuntimeDiagnosticFieldPayload {
                name: "entry_count".to_string(),
                value: JaxRuntimeDiagnosticValue::Integer(7),
            },
            JaxRuntimeDiagnosticFieldPayload {
                name: "platform".to_string(),
                value: JaxRuntimeDiagnosticValue::Text("cuda".to_string()),
            },
        ];
        let fields_text = serialize_jax_runtime_diagnostic_fields_json(&fields).expect("fields should serialize");
        let fields_payload: serde_json::Value =
            serde_json::from_str(&fields_text).expect("fields should be valid JSON");

        assert_eq!(
            fields_payload,
            serde_json::json!({
                "enabled": true,
                "entry_count": 7,
                "platform": "cuda",
            }),
        );
    }

    #[test]
    fn marks_failed_jax_gpu_validation_diagnostic_as_error() {
        let mut setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 0, 0, false, false);
        setup.gpu_validation_status = "failed".to_string();
        setup.gpu_validation_message = Some("no gpu".to_string());

        let events = build_jax_runtime_setup_diagnostic_events(&setup);

        assert_eq!(events[4].level, JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR);
        assert_eq!(events[4].fields[1].value, JaxRuntimeDiagnosticValue::Text("no gpu".to_string()));
    }

    #[test]
    fn plans_ordered_jax_runtime_config_updates() {
        let setup = resolve_jax_runtime_setup("cpu", "cache", Some("highest"), true, 1024, 5, true, true);

        let updates = plan_jax_runtime_config_updates(&setup);

        assert_eq!(
            updates,
            vec![
                text_config_update(JAX_CONFIG_PLATFORMS, JAX_CPU_PLATFORM_NAME.to_string()),
                boolean_config_update(JAX_CONFIG_ENABLE_X64, true),
                text_config_update(JAX_CONFIG_DEFAULT_MATMUL_PRECISION, "highest".to_string()),
                text_config_update(JAX_CONFIG_COMPILATION_CACHE_DIR, "cache".to_string()),
                integer_config_update(JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, 1024),
                integer_config_update(JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS, 5),
                text_config_update(
                    JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES,
                    XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE.to_string(),
                ),
                text_config_update(JAX_CONFIG_TRANSFER_GUARD, JAX_TRANSFER_GUARD_DISALLOW.to_string()),
            ]
        );
    }

    #[test]
    fn plans_jax_gpu_validation_outcomes() {
        let missing_driver_plan = plan_jax_gpu_validation(false, false, &[]);
        assert_eq!(missing_driver_plan.status, JAX_RUNTIME_GPU_VALIDATION_FAILED);
        assert!(missing_driver_plan.should_raise);
        assert!(missing_driver_plan.message.contains("cannot see the NVIDIA driver"));

        let backend_failure_plan = plan_jax_gpu_validation(true, true, &[]);
        assert_eq!(backend_failure_plan.status, JAX_RUNTIME_GPU_VALIDATION_FAILED);
        assert!(backend_failure_plan.should_raise);
        assert!(backend_failure_plan.message.contains("no CUDA-enabled JAX backend"));

        let cpu_only_plan = plan_jax_gpu_validation(
            true,
            false,
            &[JaxDeviceObservation { platform: "cpu".to_string(), description: "CpuDevice(id=0)".to_string() }],
        );
        assert_eq!(cpu_only_plan.status, JAX_RUNTIME_GPU_VALIDATION_FAILED);
        assert!(cpu_only_plan.should_raise);
        assert!(cpu_only_plan.message.contains("Observed devices: CpuDevice(id=0)."));

        let gpu_plan = plan_jax_gpu_validation(
            true,
            false,
            &[JaxDeviceObservation {
                platform: JAX_GPU_DEVICE_PLATFORM_NAME.to_string(),
                description: "GpuDevice(id=0)".to_string(),
            }],
        );
        assert_eq!(gpu_plan.status, JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED);
        assert!(!gpu_plan.should_raise);
        assert_eq!(gpu_plan.message, "JAX reported at least one GPU device.");
    }
}
