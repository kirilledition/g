//! Deterministic JAX runtime setup policy and diagnostics.

use std::fs;

mod config_updates;
mod diagnostics;
mod gpu_validation;
mod setup;

pub use config_updates::plan_jax_runtime_config_updates;
pub use diagnostics::{build_jax_runtime_setup_diagnostic_events, plan_jax_runtime_diagnostic_record};
pub use gpu_validation::{default_nvidia_driver_probe_paths, nvidia_driver_files_are_visible, plan_jax_gpu_validation};
pub use setup::resolve_jax_runtime_setup;

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

pub struct JaxRuntimeDiagnosticFields<'fields> {
    fields: &'fields [JaxRuntimeDiagnosticFieldPayload],
}

impl<'fields> JaxRuntimeDiagnosticFields<'fields> {
    #[must_use]
    pub const fn new(fields: &'fields [JaxRuntimeDiagnosticFieldPayload]) -> Self {
        Self { fields }
    }
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
        if !self.setup.persistent_cache_enabled {
            return Ok(false);
        }
        fs::create_dir_all(&self.setup.cache_directory)?;
        Ok(true)
    }

    pub fn complete_validation(&mut self, gpu_validation_status: &str, gpu_validation_message: Option<&str>) {
        self.setup.gpu_validation_status = gpu_validation_status.to_string();
        self.setup.gpu_validation_message = gpu_validation_message.map(str::to_string);
    }
}
