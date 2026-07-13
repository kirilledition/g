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
