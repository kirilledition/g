//! Process runtime compatibility state.

use std::error::Error;
use std::fmt;

use crate::jax_runtime;
use crate::rayon_runtime;
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeStateSnapshotPayload {
    pub logging_policy: Option<LoggingRuntimePolicyPayload>,
    pub rayon_thread_count: Option<i64>,
    pub jax_policy: Option<JaxRuntimePolicyPayload>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePolicyPayload {
    pub logging_policy: LoggingRuntimePolicyPayload,
    pub rayon_thread_count: Option<i64>,
    pub jax_policy: JaxRuntimePolicyPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunRuntime {
    pub runtime_policy: RuntimePolicyPayload,
    pub compatibility_token: RuntimeCompatibilityToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCompatibilityToken {
    _private: (),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RayonThreadPoolConfigurationPlan {
    pub should_configure: bool,
    pub thread_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeSetupLifecyclePlan {
    pub should_configure: bool,
}

#[allow(clippy::fn_params_excessive_bools)]
#[must_use]
pub fn build_jax_runtime_policy_payload(
    device: &str,
    cache_directory: Option<&str>,
    matmul_precision: Option<&str>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> JaxRuntimePolicyPayload {
    JaxRuntimePolicyPayload {
        device: device.to_string(),
        cache_directory: cache_directory.map(str::to_string),
        matmul_precision: matmul_precision.map(str::to_string),
        persistent_cache,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_autotune_cache,
        transfer_guard,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCompatibilityError {
    message: String,
}

#[derive(Debug)]
pub enum RayonThreadPoolConfigurationError {
    RuntimeCompatibility(RuntimeCompatibilityError),
    RuntimeConfiguration { thread_count: i64, source: rayon_runtime::RayonRuntimeError },
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

impl fmt::Display for RayonThreadPoolConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeCompatibility(error) => error.fmt(formatter),
            Self::RuntimeConfiguration { thread_count, source } => {
                if matches!(source, rayon_runtime::RayonRuntimeError::InvalidThreadCount) {
                    source.fmt(formatter)
                } else {
                    formatter.write_str(&rayon_runtime::format_global_rayon_thread_pool_configuration_error(
                        *thread_count,
                        &source.to_string(),
                    ))
                }
            }
        }
    }
}

impl Error for RayonThreadPoolConfigurationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuntimeCompatibility(error) => Some(error),
            Self::RuntimeConfiguration { source, .. } => Some(source),
        }
    }
}

impl ProcessRuntimeState {
    #[must_use]
    pub fn snapshot(&self) -> RuntimeStateSnapshotPayload {
        RuntimeStateSnapshotPayload {
            logging_policy: self.logging_policy.clone(),
            rayon_thread_count: self.rayon_thread_count,
            jax_policy: self.jax_policy.clone(),
        }
    }

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
    ) -> Result<RuntimeCompatibilityToken, RuntimeCompatibilityError> {
        self.require_compatible_logging_policy(logging_policy)?;
        self.require_compatible_rayon_thread_count(requested_rayon_thread_count)?;
        self.require_compatible_jax_policy(jax_policy)?;
        Ok(RuntimeCompatibilityToken { _private: () })
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
    ) -> Result<RuntimeCompatibilityToken, RuntimeCompatibilityError> {
        self.require_compatible_runtime_policy(
            &runtime_policy.logging_policy,
            runtime_policy.rayon_thread_count,
            &runtime_policy.jax_policy,
        )
    }

    /// Build a run-scoped runtime handle after compatibility checks pass.
    ///
    /// # Errors
    ///
    /// Returns an error when any requested process-global runtime setting
    /// conflicts with previously configured state.
    pub fn build_run_runtime(
        &self,
        runtime_policy: RuntimePolicyPayload,
    ) -> Result<RunRuntime, RuntimeCompatibilityError> {
        let compatibility_token = self.require_compatible_runtime_policy_payload(&runtime_policy)?;
        Ok(RunRuntime { runtime_policy, compatibility_token })
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

    /// Plan process-global Rayon thread-pool configuration for one request.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested thread count conflicts with a
    /// previously configured Rayon global thread count.
    pub fn plan_rayon_thread_pool_configuration(
        &self,
        requested_thread_count: i64,
    ) -> Result<RayonThreadPoolConfigurationPlan, RuntimeCompatibilityError> {
        self.require_compatible_rayon_thread_count(Some(requested_thread_count))?;
        if self.rayon_thread_count == Some(requested_thread_count) {
            return Ok(RayonThreadPoolConfigurationPlan { should_configure: false, thread_count: None });
        }
        Ok(RayonThreadPoolConfigurationPlan { should_configure: true, thread_count: Some(requested_thread_count) })
    }

    /// Configure the process-global Rayon thread pool and record the result.
    ///
    /// # Errors
    ///
    /// Returns an error when the request is incompatible with previous state or
    /// Rayon rejects global thread-pool initialization for this process.
    pub fn configure_rayon_thread_pool(
        &mut self,
        requested_thread_count: i64,
    ) -> Result<RayonThreadPoolConfigurationPlan, RayonThreadPoolConfigurationError> {
        let plan = self
            .plan_rayon_thread_pool_configuration(requested_thread_count)
            .map_err(RayonThreadPoolConfigurationError::RuntimeCompatibility)?;
        let Some(thread_count) = plan.thread_count else {
            return Ok(plan);
        };
        let runtime_thread_count =
            usize::try_from(thread_count).map_err(|_| RayonThreadPoolConfigurationError::RuntimeConfiguration {
                thread_count,
                source: rayon_runtime::RayonRuntimeError::InvalidThreadCount,
            })?;
        rayon_runtime::configure_global_rayon_thread_pool(runtime_thread_count)
            .map_err(|source| RayonThreadPoolConfigurationError::RuntimeConfiguration { thread_count, source })?;
        self.record_rayon_thread_count(thread_count);
        Ok(plan)
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

    /// Record successful process-global JAX runtime setup.
    ///
    /// # Errors
    ///
    /// Returns an error when the completed setup conflicts with previously
    /// configured process-global JAX runtime settings.
    pub fn complete_jax_runtime_setup(
        &mut self,
        requested_policy: JaxRuntimePolicyPayload,
    ) -> Result<(), RuntimeCompatibilityError> {
        self.require_compatible_jax_policy(&requested_policy)?;
        self.record_jax_policy(requested_policy);
        Ok(())
    }

    /// Plan whether JAX runtime setup should run for one request.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured incompatible
    /// process-global JAX runtime settings.
    pub fn plan_jax_runtime_setup_lifecycle(
        &self,
        requested_policy: &JaxRuntimePolicyPayload,
    ) -> Result<JaxRuntimeSetupLifecyclePlan, RuntimeCompatibilityError> {
        self.require_compatible_jax_policy(requested_policy)?;
        Ok(JaxRuntimeSetupLifecyclePlan { should_configure: self.jax_policy.as_ref() != Some(requested_policy) })
    }

    /// Build a run-scoped JAX setup session after lifecycle compatibility checks.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured incompatible
    /// process-global JAX runtime settings.
    pub fn build_jax_runtime_setup_session(
        &self,
        requested_policy: &JaxRuntimePolicyPayload,
        resolved_cache_directory: &str,
    ) -> Result<jax_runtime::JaxRuntimeSetupSession, RuntimeCompatibilityError> {
        let lifecycle_plan = self.plan_jax_runtime_setup_lifecycle(requested_policy)?;
        let setup = jax_runtime::resolve_jax_runtime_setup(
            &requested_policy.device,
            resolved_cache_directory,
            requested_policy.matmul_precision.as_deref(),
            requested_policy.persistent_cache,
            requested_policy.persistent_cache_min_entry_size_bytes,
            requested_policy.persistent_cache_min_compile_time_seconds,
            requested_policy.xla_autotune_cache,
            requested_policy.transfer_guard,
        );
        Ok(jax_runtime::JaxRuntimeSetupSession::new(lifecycle_plan.should_configure, setup))
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
        build_jax_runtime_policy_payload("cpu", cache_directory, None, true, 0, 0, false, false)
    }

    #[test]
    fn builds_jax_runtime_policy_payload() {
        assert_eq!(
            build_jax_runtime_policy_payload("gpu", Some("/tmp/cache"), Some("highest"), false, 1024, 5, true, true,),
            JaxRuntimePolicyPayload {
                device: "gpu".to_string(),
                cache_directory: Some("/tmp/cache".to_string()),
                matmul_precision: Some("highest".to_string()),
                persistent_cache: false,
                persistent_cache_min_entry_size_bytes: 1024,
                persistent_cache_min_compile_time_seconds: 5,
                xla_autotune_cache: true,
                transfer_guard: true,
            },
        );
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
    fn plans_rayon_thread_pool_configuration_from_process_state() {
        let mut state = ProcessRuntimeState::default();

        assert_eq!(
            state.plan_rayon_thread_pool_configuration(4).unwrap(),
            RayonThreadPoolConfigurationPlan { should_configure: true, thread_count: Some(4) },
        );

        state.record_rayon_thread_count(4);
        assert_eq!(
            state.plan_rayon_thread_pool_configuration(4).unwrap(),
            RayonThreadPoolConfigurationPlan { should_configure: false, thread_count: None },
        );
        assert!(state.plan_rayon_thread_pool_configuration(8).unwrap_err().to_string().contains("Rayon --threads"));
    }

    #[test]
    fn configures_rayon_thread_pool_from_process_state() {
        let mut state = ProcessRuntimeState::default();

        let error = state.configure_rayon_thread_pool(0).unwrap_err();

        assert!(matches!(
            error,
            RayonThreadPoolConfigurationError::RuntimeConfiguration {
                source: rayon_runtime::RayonRuntimeError::InvalidThreadCount,
                ..
            },
        ));
        assert_eq!(error.to_string(), "Rayon thread count must be positive.");

        state.record_rayon_thread_count(4);
        let skip_plan = state.configure_rayon_thread_pool(4).expect("matching configured count should skip setup");

        assert_eq!(skip_plan, RayonThreadPoolConfigurationPlan { should_configure: false, thread_count: None });
        assert!(
            state.configure_rayon_thread_pool(8).unwrap_err().to_string().contains("Rayon --threads is process-global")
        );
    }

    #[test]
    fn snapshots_process_runtime_state() {
        let mut state = ProcessRuntimeState::default();
        let logging_policy = build_policy("info");
        let jax_policy = build_jax_policy(Some("/tmp/cache"));

        state.record_logging_policy(logging_policy.clone());
        state.record_rayon_thread_count(4);
        state.record_jax_policy(jax_policy.clone());

        assert_eq!(
            state.snapshot(),
            RuntimeStateSnapshotPayload {
                logging_policy: Some(logging_policy),
                rayon_thread_count: Some(4),
                jax_policy: Some(jax_policy),
            },
        );
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

    #[test]
    fn plans_jax_runtime_setup_lifecycle_from_process_state() {
        let mut state = ProcessRuntimeState::default();
        let requested_policy = build_jax_policy(Some("/tmp/cache"));

        assert_eq!(
            state.plan_jax_runtime_setup_lifecycle(&requested_policy).unwrap(),
            JaxRuntimeSetupLifecyclePlan { should_configure: true },
        );

        state.record_jax_policy(requested_policy.clone());
        assert_eq!(
            state.plan_jax_runtime_setup_lifecycle(&requested_policy).unwrap(),
            JaxRuntimeSetupLifecyclePlan { should_configure: false },
        );
        assert!(
            state
                .plan_jax_runtime_setup_lifecycle(&build_jax_policy(Some("/tmp/second-cache")))
                .unwrap_err()
                .to_string()
                .contains("JAX runtime is already configured")
        );
    }

    #[test]
    fn completes_jax_runtime_setup_from_process_state() {
        let mut state = ProcessRuntimeState::default();
        let requested_policy = build_jax_policy(Some("/tmp/cache"));

        state
            .complete_jax_runtime_setup(requested_policy.clone())
            .expect("first compatible JAX setup should be recorded");

        assert_eq!(state.jax_policy, Some(requested_policy.clone()));
        state
            .complete_jax_runtime_setup(requested_policy)
            .expect("matching repeated JAX setup completion should be accepted");
        assert!(
            state
                .complete_jax_runtime_setup(build_jax_policy(Some("/tmp/second-cache")))
                .unwrap_err()
                .to_string()
                .contains("JAX runtime is already configured")
        );
    }

    #[test]
    fn builds_jax_runtime_setup_session_from_process_state() {
        let mut state = ProcessRuntimeState::default();
        let requested_policy = build_jax_policy(Some("/tmp/cache"));

        let configure_session = state
            .build_jax_runtime_setup_session(&requested_policy, "/tmp/cache")
            .expect("compatible JAX policy should build a setup session");

        assert!(configure_session.should_configure());
        assert_eq!(configure_session.setup().platform_name, "cpu");
        assert_eq!(configure_session.setup().cache_directory, "/tmp/cache");

        state.record_jax_policy(requested_policy.clone());
        let skip_session = state
            .build_jax_runtime_setup_session(&requested_policy, "/tmp/cache")
            .expect("matching configured JAX policy should build a skip session");

        assert!(!skip_session.should_configure());
        assert!(
            state
                .build_jax_runtime_setup_session(&build_jax_policy(Some("/tmp/second-cache")), "/tmp/second-cache")
                .unwrap_err()
                .to_string()
                .contains("JAX runtime is already configured")
        );
    }

    #[test]
    fn issues_runtime_compatibility_token_after_all_checks_pass() {
        let mut state = ProcessRuntimeState::default();
        state.record_logging_policy(build_policy("info"));
        state.record_rayon_thread_count(4);
        state.record_jax_policy(build_jax_policy(Some("/tmp/cache")));

        let token = state
            .require_compatible_runtime_policy(&build_policy("info"), Some(4), &build_jax_policy(Some("/tmp/cache")))
            .expect("matching process-global policy should issue a token");

        assert_eq!(token, RuntimeCompatibilityToken { _private: () });
    }

    #[test]
    fn issues_runtime_compatibility_token_from_policy_payload() {
        let mut state = ProcessRuntimeState::default();
        state.record_logging_policy(build_policy("info"));
        state.record_jax_policy(build_jax_policy(Some("/tmp/cache")));
        let runtime_policy = RuntimePolicyPayload {
            logging_policy: build_policy("info"),
            rayon_thread_count: None,
            jax_policy: build_jax_policy(Some("/tmp/cache")),
        };

        let token = state
            .require_compatible_runtime_policy_payload(&runtime_policy)
            .expect("matching runtime policy payload should issue a token");

        assert_eq!(token, RuntimeCompatibilityToken { _private: () });
    }

    #[test]
    fn builds_run_runtime_after_compatibility_checks() {
        let mut state = ProcessRuntimeState::default();
        state.record_logging_policy(build_policy("info"));
        let runtime_policy = RuntimePolicyPayload {
            logging_policy: build_policy("info"),
            rayon_thread_count: None,
            jax_policy: build_jax_policy(Some("/tmp/cache")),
        };

        let run_runtime = state
            .build_run_runtime(runtime_policy.clone())
            .expect("matching runtime policy should produce run runtime handle");

        assert_eq!(run_runtime.runtime_policy, runtime_policy);
        assert_eq!(run_runtime.compatibility_token, RuntimeCompatibilityToken { _private: () });
    }
}
