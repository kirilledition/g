//! Runner-owned JAX runtime setup policy and host contracts.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

mod config_updates;
mod diagnostics;
mod gpu_validation;
mod policy;
mod setup;

pub(crate) use policy::{JaxRuntimeState, build_jax_runtime_policy};

pub(crate) use config_updates::plan_jax_runtime_config_updates;
pub(crate) use diagnostics::emit_jax_runtime_setup_diagnostics;
pub(crate) use gpu_validation::{nvidia_driver_files_are_visible, plan_jax_gpu_validation};

const JAX_CONFIG_COMPILATION_CACHE_DIR: &str = "jax_compilation_cache_dir";
const JAX_CONFIG_DEFAULT_MATMUL_PRECISION: &str = "jax_default_matmul_precision";
const JAX_CONFIG_ENABLE_X64: &str = "jax_enable_x64";
const JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES: &str = "jax_persistent_cache_enable_xla_caches";
const JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS: &str = "jax_persistent_cache_min_compile_time_secs";
const JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES: &str = "jax_persistent_cache_min_entry_size_bytes";
const JAX_CONFIG_PLATFORMS: &str = "jax_platforms";
const JAX_MATMUL_PRECISION: &str = "float32";
const JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS: i64 = 0;
const JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES: i64 = -1;
const JAX_CUDA_PLATFORM_NAME: &str = "cuda";
const JAX_CPU_PLATFORM_NAME: &str = "cpu";
const JAX_GPU_DEVICE_PLATFORM_NAME: &str = "gpu";
const NVIDIA_CONTROL_DEVICE_PATH: &str = "/dev/nvidiactl";
const NVIDIA_DRIVER_DIRECTORY_PATH: &str = "/proc/driver/nvidia";
const NVIDIA_UVM_DEVICE_PATH: &str = "/dev/nvidia-uvm";
const XLA_AUXILIARY_CACHE_DISABLED: &str = "none";

#[derive(Clone)]
enum JaxCacheDirectory {
    Default(PathBuf),
    Explicit(PathBuf),
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct JaxRuntimePolicy {
    device: g_plan::Device,
    cache_directory: JaxCacheDirectory,
}

#[derive(Debug, Eq, PartialEq)]
pub struct JaxDevice {
    pub platform: String,
    pub description: String,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct JaxGpuValidationPlan {
    pub status: JaxGpuValidationStatus,
    pub message: Cow<'static, str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JaxGpuValidationStatus {
    Pending,
    Skipped,
    Succeeded,
    Failed,
}

pub(crate) struct JaxRuntimeSetupSession<'policy> {
    pub(crate) should_configure: bool,
    policy: &'policy JaxRuntimePolicy,
    pub(crate) gpu_validation_status: JaxGpuValidationStatus,
    gpu_validation_message: Option<Cow<'static, str>>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum JaxRuntimeConfigValue<'value> {
    Boolean(bool),
    Integer(i64),
    Text(Cow<'value, str>),
}

#[derive(Debug, Eq, PartialEq)]
pub struct JaxRuntimeConfigUpdate<'value> {
    pub setting_name: &'static str,
    pub value: JaxRuntimeConfigValue<'value>,
}

impl PartialEq for JaxCacheDirectory {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // The first resolved default remains authoritative after JAX is
            // configured; ambient temp-directory changes do not request a new
            // process-global policy.
            (Self::Default(_), Self::Default(_)) => true,
            (Self::Explicit(left), Self::Explicit(right)) => left == right,
            (Self::Default(_), Self::Explicit(_)) | (Self::Explicit(_), Self::Default(_)) => false,
        }
    }
}

impl Eq for JaxCacheDirectory {}

impl JaxCacheDirectory {
    fn path(&self) -> &Path {
        match self {
            Self::Default(path) | Self::Explicit(path) => path,
        }
    }
}

impl JaxRuntimePolicy {
    const fn platform_name(&self) -> &'static str {
        match self.device {
            g_plan::Device::Cpu => JAX_CPU_PLATFORM_NAME,
            g_plan::Device::Gpu => JAX_CUDA_PLATFORM_NAME,
        }
    }
}

impl JaxGpuValidationStatus {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Skipped => "skipped",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{JaxCacheDirectory, JaxGpuValidationStatus, JaxRuntimePolicy};

    #[test]
    fn default_cache_identity_ignores_ambient_path_but_explicit_identity_does_not() {
        assert!(
            JaxCacheDirectory::Default(PathBuf::from("first")) == JaxCacheDirectory::Default(PathBuf::from("second"))
        );
        assert!(
            JaxCacheDirectory::Default(PathBuf::from("same")) != JaxCacheDirectory::Explicit(PathBuf::from("same"))
        );
        assert!(
            JaxCacheDirectory::Explicit(PathBuf::from("first")) != JaxCacheDirectory::Explicit(PathBuf::from("second"))
        );
    }

    #[test]
    fn platform_and_validation_names_are_stable() {
        let cpu_policy = JaxRuntimePolicy {
            device: g_plan::Device::Cpu,
            cache_directory: JaxCacheDirectory::Explicit(PathBuf::from("cache")),
        };
        let gpu_policy =
            JaxRuntimePolicy { device: g_plan::Device::Gpu, cache_directory: cpu_policy.cache_directory.clone() };
        assert_eq!(cpu_policy.platform_name(), "cpu");
        assert_eq!(gpu_policy.platform_name(), "cuda");
        assert_eq!(JaxGpuValidationStatus::Pending.as_str(), "pending");
        assert_eq!(JaxGpuValidationStatus::Skipped.as_str(), "skipped");
        assert_eq!(JaxGpuValidationStatus::Succeeded.as_str(), "succeeded");
        assert_eq!(JaxGpuValidationStatus::Failed.as_str(), "failed");
    }
}
