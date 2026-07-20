use std::borrow::Cow;
use std::fs;

use super::{JaxGpuValidationStatus, JaxRuntimePolicy, JaxRuntimeSetupSession};

impl JaxRuntimePolicy {
    /// Create the persistent JAX cache directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache directory cannot be created.
    pub(crate) fn create_cache_directory_if_configured(&self) -> Result<(), std::io::Error> {
        fs::create_dir_all(self.cache_directory.path())?;
        Ok(())
    }
}

impl<'policy> JaxRuntimeSetupSession<'policy> {
    #[must_use]
    pub(crate) const fn new(should_configure: bool, policy: &'policy JaxRuntimePolicy) -> Self {
        let (gpu_validation_status, gpu_validation_message) = match policy.device {
            g_plan::Device::Gpu => (JaxGpuValidationStatus::Pending, None),
            g_plan::Device::Cpu => {
                (JaxGpuValidationStatus::Skipped, Some(Cow::Borrowed("CPU runtime requested; GPU validation skipped.")))
            }
        };
        Self { should_configure, policy, gpu_validation_status, gpu_validation_message }
    }

    pub(crate) fn complete_gpu_validation(&mut self, status: JaxGpuValidationStatus, message: Cow<'static, str>) {
        self.gpu_validation_status = status;
        self.gpu_validation_message = Some(message);
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::path::PathBuf;

    use crate::jax_runtime::{JaxCacheDirectory, JaxGpuValidationStatus, JaxRuntimePolicy, JaxRuntimeSetupSession};

    #[test]
    fn cache_directory_creation_is_idempotent() {
        let fixture = crate::test_support::TemporaryRunFixture::new();
        let cache_directory = fixture.root_path().join("nested/cache");
        let policy = JaxRuntimePolicy {
            device: g_plan::Device::Cpu,
            cache_directory: JaxCacheDirectory::Explicit(cache_directory.clone()),
        };
        policy.create_cache_directory_if_configured().expect("cache directory should be created");
        policy.create_cache_directory_if_configured().expect("existing cache directory should be accepted");
        assert!(cache_directory.is_dir());
    }

    #[test]
    fn setup_session_initializes_cpu_and_gpu_validation_states() {
        let cpu_policy = JaxRuntimePolicy {
            device: g_plan::Device::Cpu,
            cache_directory: JaxCacheDirectory::Explicit(PathBuf::from("cpu-cache")),
        };
        let cpu_session = JaxRuntimeSetupSession::new(true, &cpu_policy);
        assert!(cpu_session.should_configure);
        assert_eq!(cpu_session.gpu_validation_status, JaxGpuValidationStatus::Skipped);
        assert_eq!(
            cpu_session.gpu_validation_message.as_deref(),
            Some("CPU runtime requested; GPU validation skipped.")
        );

        let gpu_policy = JaxRuntimePolicy {
            device: g_plan::Device::Gpu,
            cache_directory: JaxCacheDirectory::Explicit(PathBuf::from("gpu-cache")),
        };
        let mut gpu_session = JaxRuntimeSetupSession::new(false, &gpu_policy);
        assert!(!gpu_session.should_configure);
        assert_eq!(gpu_session.gpu_validation_status, JaxGpuValidationStatus::Pending);
        assert_eq!(gpu_session.gpu_validation_message, None);
        gpu_session.complete_gpu_validation(JaxGpuValidationStatus::Succeeded, Cow::Borrowed("device visible"));
        assert_eq!(gpu_session.gpu_validation_status, JaxGpuValidationStatus::Succeeded);
        assert_eq!(gpu_session.gpu_validation_message.as_deref(), Some("device visible"));
    }
}
