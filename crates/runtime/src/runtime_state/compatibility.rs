use crate::error::RuntimeCompatibilityError;
use crate::runtime_policy::{LoggingRuntimePolicyPayload, describe_logging_runtime_policy};

use super::{JaxRuntimePolicyPayload, ProcessRuntimeState, RuntimePolicyPayload};

impl ProcessRuntimeState {
    /// Require all process-global runtime settings to be compatible.
    ///
    /// # Errors
    ///
    /// Returns an error when any requested process-global runtime setting
    /// conflicts with previously configured state.
    pub fn require_compatible_runtime_policy(
        &self,
        logging_policy: &LoggingRuntimePolicyPayload,
        requested_rayon_thread_count: Option<i64>,
        jax_policy: &JaxRuntimePolicyPayload,
    ) -> Result<(), RuntimeCompatibilityError> {
        self.require_compatible_logging_policy(logging_policy)?;
        self.require_compatible_rayon_thread_count(requested_rayon_thread_count)?;
        self.require_compatible_jax_policy(jax_policy)
    }

    /// Require all process-global runtime settings from a run policy.
    ///
    /// # Errors
    ///
    /// Returns an error when any requested process-global runtime setting
    /// conflicts with previously configured state.
    pub fn require_compatible_runtime_policy_payload(
        &self,
        runtime_policy: &RuntimePolicyPayload,
    ) -> Result<(), RuntimeCompatibilityError> {
        self.require_compatible_runtime_policy(
            &runtime_policy.logging_policy,
            runtime_policy.rayon_thread_count,
            &runtime_policy.jax_policy,
        )
    }

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
}
