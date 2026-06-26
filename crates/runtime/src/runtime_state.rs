//! Process runtime compatibility state.

use std::error::Error;
use std::fmt;

use crate::runtime_policy::{LoggingRuntimePolicyPayload, describe_logging_runtime_policy};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessRuntimeState {
    pub logging_policy: Option<LoggingRuntimePolicyPayload>,
    pub rayon_thread_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCompatibilityError {
    message: String,
}

impl RuntimeCompatibilityError {
    #[must_use]
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl fmt::Display for RuntimeCompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for RuntimeCompatibilityError {}

impl ProcessRuntimeState {
    /// Require logging compatibility with previously configured process state.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured different process-global
    /// logging settings.
    pub fn require_compatible_logging_policy(
        &self,
        requested_policy: &LoggingRuntimePolicyPayload,
    ) -> Result<(), RuntimeCompatibilityError> {
        let Some(configured_policy) = self.logging_policy.as_ref() else {
            return Ok(());
        };
        if configured_policy == requested_policy {
            return Ok(());
        }
        Err(RuntimeCompatibilityError::new(format!(
            "Logging runtime policy is process-global for this Python process. \
             Configured policy: {}. Requested policy: {}. \
             Start a fresh Python process for incompatible logging settings.",
            describe_logging_runtime_policy(configured_policy),
            describe_logging_runtime_policy(requested_policy),
        )))
    }

    pub fn record_logging_policy(&mut self, logging_policy: LoggingRuntimePolicyPayload) {
        self.logging_policy = Some(logging_policy);
    }

    /// Require Rayon compatibility with previously configured process state.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured a different Rayon global
    /// thread count.
    pub fn require_compatible_rayon_thread_count(
        &self,
        requested_thread_count: Option<i64>,
    ) -> Result<(), RuntimeCompatibilityError> {
        let Some(requested_thread_count) = requested_thread_count else {
            return Ok(());
        };
        let Some(configured_thread_count) = self.rayon_thread_count else {
            return Ok(());
        };
        if configured_thread_count == requested_thread_count {
            return Ok(());
        }
        Err(RuntimeCompatibilityError::new(format!(
            "Rayon --threads is process-global for this Python process. \
             Configured thread count: {configured_thread_count}. Requested thread count: {requested_thread_count}. \
             Start a fresh Python process for incompatible Rayon settings."
        )))
    }

    pub fn record_rayon_thread_count(&mut self, thread_count: i64) {
        self.rayon_thread_count = Some(thread_count);
    }

    #[must_use]
    pub fn effective_rayon_thread_count(&self, requested_thread_count: Option<i64>) -> Option<i64> {
        self.rayon_thread_count.or(requested_thread_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_policy(log_filter: &str) -> LoggingRuntimePolicyPayload {
        LoggingRuntimePolicyPayload {
            log_filter: log_filter.to_string(),
            log_file: None,
            log_stderr: true,
            log_queue_size: 1024,
            log_lossy: true,
            include_source_location: false,
            include_span_events: false,
            trace_file: None,
            trace_filter: "info".to_string(),
            trace_event_cap: None,
        }
    }

    #[test]
    fn rejects_incompatible_logging_policy() {
        let mut state = ProcessRuntimeState::default();
        state.record_logging_policy(build_policy("info"));

        let error = state.require_compatible_logging_policy(&build_policy("debug")).unwrap_err();

        assert!(error.to_string().contains("Logging runtime policy is process-global"));
        assert!(error.to_string().contains("Configured policy: log-filter=info"));
        assert!(error.to_string().contains("Requested policy: log-filter=debug"));
    }

    #[test]
    fn rejects_incompatible_rayon_thread_count() {
        let mut state = ProcessRuntimeState::default();
        state.record_rayon_thread_count(4);

        let error = state.require_compatible_rayon_thread_count(Some(8)).unwrap_err();

        assert!(error.to_string().contains("Rayon --threads is process-global"));
        assert_eq!(state.effective_rayon_thread_count(Some(8)), Some(4));
    }
}
