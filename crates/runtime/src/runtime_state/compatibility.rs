use crate::error::RuntimeCompatibilityError;
use crate::runtime_policy::{NativeRunSessionPolicy, describe_logging_subscriber_policy};

use super::ProcessRuntimeState;

impl ProcessRuntimeState {
    /// Require all process-global runtime settings to be compatible.
    ///
    /// # Errors
    ///
    /// Returns an error when any requested process-global runtime setting
    /// conflicts with previously configured state.
    pub fn require_compatible_runtime_policy(
        &self,
        logging_policy: &NativeRunSessionPolicy,
        requested_rayon_thread_count: Option<i64>,
    ) -> Result<(), RuntimeCompatibilityError> {
        self.require_compatible_logging_policy(logging_policy)?;
        self.require_compatible_rayon_thread_count(requested_rayon_thread_count)
    }

    /// Require logging compatibility with previously configured process state.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured different process-global
    /// logging settings.
    pub(crate) fn require_compatible_logging_policy(
        &self,
        requested_policy: &NativeRunSessionPolicy,
    ) -> Result<(), RuntimeCompatibilityError> {
        let Some(configured_policy) = self.logging_subscriber_policy.as_ref() else {
            return Ok(());
        };
        let requested_subscriber_policy = requested_policy.subscriber_policy();
        if configured_policy == &requested_subscriber_policy {
            return Ok(());
        }
        Err(RuntimeCompatibilityError::new(format!(
            "Logging subscriber policy is process-global for this Python process. \
             Configured policy: {}. Requested policy: {}. \
             Start a fresh Python process for incompatible logging settings.",
            describe_logging_subscriber_policy(configured_policy),
            describe_logging_subscriber_policy(&requested_subscriber_policy),
        )))
    }

    pub(crate) fn record_logging_subscriber_policy(&mut self, logging_policy: &NativeRunSessionPolicy) {
        self.logging_subscriber_policy = Some(logging_policy.subscriber_policy().into_owned());
    }
}
