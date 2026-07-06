use crate::error::RuntimeCompatibilityError;
use crate::jax_runtime;
use crate::runtime_paths;

use super::ProcessRuntimeState;

const DEFAULT_JAX_CACHE_DIRECTORY_NAME: &str = "g-jax-cache";

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

impl ProcessRuntimeState {
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

    /// Record successful JAX runtime setup after a native setup session finishes.
    ///
    /// # Errors
    ///
    /// Returns an error when the completed setup conflicts with previously
    /// configured process-global JAX runtime settings, or when the setup session
    /// still has pending or failed GPU validation.
    pub fn complete_jax_runtime_setup_session(
        &mut self,
        requested_policy: JaxRuntimePolicyPayload,
        setup_session: &jax_runtime::JaxRuntimeSetupSession,
    ) -> Result<(), RuntimeCompatibilityError> {
        if setup_session.should_configure() {
            let setup = setup_session.setup();
            if setup.gpu_validation_status == "pending" {
                return Err(RuntimeCompatibilityError::new(
                    "Cannot record JAX runtime setup before GPU validation completes.".to_string(),
                ));
            }
            if setup.gpu_validation_status == "failed" {
                return Err(RuntimeCompatibilityError::new(
                    "Cannot record failed JAX runtime setup as process-global runtime state.".to_string(),
                ));
            }
        }
        self.complete_jax_runtime_setup(requested_policy)
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

    /// Build a run-scoped JAX setup session and resolve default cache paths natively.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured incompatible
    /// process-global JAX runtime settings.
    pub fn build_jax_runtime_setup_session_resolving_cache_directory(
        &self,
        requested_policy: &JaxRuntimePolicyPayload,
    ) -> Result<jax_runtime::JaxRuntimeSetupSession, RuntimeCompatibilityError> {
        let resolved_cache_directory = resolve_jax_runtime_cache_directory(requested_policy);
        self.build_jax_runtime_setup_session(requested_policy, &resolved_cache_directory)
    }
}

#[must_use]
pub fn resolve_jax_runtime_cache_directory(requested_policy: &JaxRuntimePolicyPayload) -> String {
    requested_policy.cache_directory.clone().unwrap_or_else(|| {
        runtime_paths::default_local_cache_directory(DEFAULT_JAX_CACHE_DIRECTORY_NAME).to_string_lossy().into_owned()
    })
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
