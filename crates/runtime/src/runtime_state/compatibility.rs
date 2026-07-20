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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::test_support::disabled_session_policy;

    #[test]
    fn process_state_accepts_run_owned_changes_and_rejects_global_changes() {
        let mut state = ProcessRuntimeState::default();
        let configured = disabled_session_policy();
        assert_eq!(state.require_compatible_runtime_policy(&configured, Some(8)), Ok(()));

        state.record_logging_subscriber_policy(&configured);
        state.rayon_thread_count = Some(8);

        let mut run_owned_changes = disabled_session_policy();
        run_owned_changes.stage_timing_file = Some(PathBuf::from("stage.json"));
        run_owned_changes.profile_summary_file = Some(PathBuf::from("profile.json"));
        run_owned_changes.queue_size = 1;
        run_owned_changes.lossy = true;
        assert_eq!(state.require_compatible_runtime_policy(&run_owned_changes, None), Ok(()));
        assert_eq!(state.require_compatible_runtime_policy(&run_owned_changes, Some(8)), Ok(()));

        let thread_error = state
            .require_compatible_runtime_policy(&run_owned_changes, Some(9))
            .expect_err("changed Rayon thread count should fail");
        assert!(thread_error.to_string().contains("Configured thread count: 8. Requested thread count: 9"));

        let mut logging_change = disabled_session_policy();
        logging_change.include_span_events = true;
        let logging_error = state
            .require_compatible_runtime_policy(&logging_change, Some(8))
            .expect_err("changed logging topology should fail first");
        assert!(logging_error.to_string().contains("Logging subscriber policy is process-global"));
        assert!(logging_error.to_string().contains("include-span-events=True"));
    }
}
