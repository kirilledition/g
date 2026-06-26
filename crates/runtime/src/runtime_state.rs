//! Process runtime compatibility state.

use std::error::Error;
use std::fmt;

use crate::runtime_policy::{LoggingRuntimePolicyPayload, describe_logging_runtime_policy};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimePolicyPayload {
    pub device: String,
    pub cache_directory: Option<String>,
    pub matmul_precision: Option<String>,
    pub persistent_cache: bool,
    pub persistent_cache_min_entry_size_bytes: i64,
    pub persistent_cache_min_compile_time_seconds: i64,
    pub xla_autotune_cache: bool,
    pub transfer_guard: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessRuntimeState {
    pub logging_policy: Option<LoggingRuntimePolicyPayload>,
    pub rayon_thread_count: Option<i64>,
    pub jax_policy: Option<JaxRuntimePolicyPayload>,
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

    /// Require JAX compatibility with previously configured process state.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured different process-global
    /// JAX runtime settings.
    pub fn require_compatible_jax_policy(
        &self,
        requested_policy: &JaxRuntimePolicyPayload,
    ) -> Result<(), RuntimeCompatibilityError> {
        let Some(configured_policy) = self.jax_policy.as_ref() else {
            return Ok(());
        };
        if configured_policy == requested_policy {
            return Ok(());
        }
        Err(RuntimeCompatibilityError::new(format!(
            "JAX runtime is already configured for this Python process with {}. \
             A later run requested incompatible settings: {}. \
             JAX backend, platform, and compilation cache settings are process-global; start a fresh Python process \
             for incompatible runtime settings.",
            describe_jax_runtime_policy(configured_policy),
            describe_jax_runtime_policy(requested_policy),
        )))
    }

    pub fn record_jax_policy(&mut self, jax_policy: JaxRuntimePolicyPayload) {
        self.jax_policy = Some(jax_policy);
    }
}

#[must_use]
pub fn describe_jax_runtime_policy(policy: &JaxRuntimePolicyPayload) -> String {
    let cache_directory = policy.cache_directory.as_deref().unwrap_or("<default>");
    let matmul_precision = policy.matmul_precision.as_deref().unwrap_or("<default>");
    format!(
        "device={}, \
         jax-cache-dir={cache_directory}, \
         jax-matmul-precision={matmul_precision}, \
         jax-persistent-cache={}, \
         jax-persistent-cache-min-entry-size-bytes={}, \
         jax-persistent-cache-min-compile-time-seconds={}, \
         jax-xla-autotune-cache={}, \
         jax-transfer-guard={}",
        policy.device,
        policy.persistent_cache,
        policy.persistent_cache_min_entry_size_bytes,
        policy.persistent_cache_min_compile_time_seconds,
        policy.xla_autotune_cache,
        policy.transfer_guard,
    )
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

    fn build_jax_policy(cache_directory: Option<&str>) -> JaxRuntimePolicyPayload {
        JaxRuntimePolicyPayload {
            device: "cpu".to_string(),
            cache_directory: cache_directory.map(str::to_string),
            matmul_precision: None,
            persistent_cache: true,
            persistent_cache_min_entry_size_bytes: 0,
            persistent_cache_min_compile_time_seconds: 0,
            xla_autotune_cache: false,
            transfer_guard: false,
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

    #[test]
    fn rejects_incompatible_jax_policy() {
        let mut state = ProcessRuntimeState::default();
        state.record_jax_policy(build_jax_policy(Some("/tmp/first-cache")));

        let error = state.require_compatible_jax_policy(&build_jax_policy(Some("/tmp/second-cache"))).unwrap_err();

        assert!(error.to_string().contains("JAX runtime is already configured"));
        assert!(error.to_string().contains("jax-cache-dir=/tmp/first-cache"));
        assert!(error.to_string().contains("jax-cache-dir=/tmp/second-cache"));
    }
}
