use std::path::{Path, PathBuf};

use g_runtime::RuntimeCompatibilityError;

use super::{JaxCacheDirectory, JaxGpuValidationStatus, JaxRuntimePolicy, JaxRuntimeSetupSession};

const DEFAULT_JAX_CACHE_DIRECTORY_NAME: &str = "g-jax-cache";
const UNKNOWN_USER_NAME: &str = "unknown";

#[derive(Default)]
pub(crate) struct JaxRuntimeState {
    configuration: JaxRuntimeConfiguration,
}

#[derive(Default)]
enum JaxRuntimeConfiguration {
    #[default]
    Unconfigured,
    Configuring(JaxRuntimePolicy),
    Configured(JaxRuntimePolicy),
}

pub(crate) fn build_jax_runtime_policy(
    run_plan: &g_plan::RunPlan,
) -> Result<JaxRuntimePolicy, RuntimeCompatibilityError> {
    let cache_directory = if let Some(configured_directory) = run_plan.compute.jax_cache_directory.as_deref() {
        JaxCacheDirectory::Explicit(expand_home_directory(configured_directory)?)
    } else {
        JaxCacheDirectory::Default(default_jax_runtime_cache_directory())
    };
    Ok(JaxRuntimePolicy { device: run_plan.compute.device, cache_directory })
}

impl JaxRuntimeState {
    pub(crate) fn require_mutually_compatible<'policy>(
        policies: impl IntoIterator<Item = &'policy JaxRuntimePolicy>,
    ) -> Result<(), RuntimeCompatibilityError> {
        let mut indexed_policies = policies.into_iter().enumerate();
        let Some((_, first_policy)) = indexed_policies.next() else {
            return Ok(());
        };
        for (run_index, requested_policy) in indexed_policies {
            if first_policy != requested_policy {
                return Err(RuntimeCompatibilityError::new(format!(
                    "Run 1 and run {} request incompatible process-global JAX settings. \
                     First policy: {}. Requested policy: {}.",
                    run_index + 1,
                    describe_jax_runtime_policy(first_policy),
                    describe_jax_runtime_policy(requested_policy),
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn require_compatible(
        &self,
        requested_policy: &JaxRuntimePolicy,
    ) -> Result<(), RuntimeCompatibilityError> {
        let configured_policy = match &self.configuration {
            JaxRuntimeConfiguration::Unconfigured => return Ok(()),
            JaxRuntimeConfiguration::Configuring(attempted_policy) => {
                return Err(RuntimeCompatibilityError::new(format!(
                    "A previous JAX runtime setup attempt did not complete. It attempted {}. \
                     JAX may be partially configured or initialized, so this process cannot safely run another \
                     setup (requested {}). Start a fresh Python process.",
                    describe_jax_runtime_policy(attempted_policy),
                    describe_jax_runtime_policy(requested_policy),
                )));
            }
            JaxRuntimeConfiguration::Configured(configured_policy) if configured_policy == requested_policy => {
                return Ok(());
            }
            JaxRuntimeConfiguration::Configured(configured_policy) => configured_policy,
        };
        Err(RuntimeCompatibilityError::new(format!(
            "JAX runtime is already configured for this Python process with {}. \
             A later run requested incompatible settings: {}. \
             JAX backend, platform, and compilation cache settings are process-global; start a fresh Python process \
             for incompatible runtime settings.",
            describe_jax_runtime_policy(configured_policy),
            describe_jax_runtime_policy(requested_policy),
        )))
    }

    pub(crate) fn setup_preparation_required(
        &self,
        requested_policy: &JaxRuntimePolicy,
    ) -> Result<bool, RuntimeCompatibilityError> {
        self.require_compatible(requested_policy)?;
        Ok(matches!(self.configuration, JaxRuntimeConfiguration::Unconfigured))
    }

    pub(crate) fn reserve_setup<'policy>(
        &mut self,
        requested_policy: &'policy JaxRuntimePolicy,
    ) -> Result<JaxRuntimeSetupSession<'policy>, RuntimeCompatibilityError> {
        self.require_compatible(requested_policy)?;
        let should_configure = matches!(self.configuration, JaxRuntimeConfiguration::Unconfigured);
        if should_configure {
            self.configuration = JaxRuntimeConfiguration::Configuring(requested_policy.clone());
        }
        Ok(JaxRuntimeSetupSession::new(should_configure, requested_policy))
    }

    pub(crate) fn complete_setup(
        &mut self,
        requested_policy: JaxRuntimePolicy,
        gpu_validation_status: JaxGpuValidationStatus,
    ) -> Result<(), RuntimeCompatibilityError> {
        if gpu_validation_status == JaxGpuValidationStatus::Pending {
            return Err(RuntimeCompatibilityError::new(
                "Cannot record JAX runtime setup before GPU validation completes.".to_string(),
            ));
        }
        if gpu_validation_status == JaxGpuValidationStatus::Failed {
            return Err(RuntimeCompatibilityError::new(
                "Cannot record failed JAX runtime setup as process-global runtime state.".to_string(),
            ));
        }
        match &self.configuration {
            JaxRuntimeConfiguration::Configuring(attempted_policy) if attempted_policy == &requested_policy => {}
            JaxRuntimeConfiguration::Configuring(attempted_policy) => {
                return Err(RuntimeCompatibilityError::new(format!(
                    "JAX setup completion policy {} does not match the in-progress policy {}. Start a fresh Python process.",
                    describe_jax_runtime_policy(&requested_policy),
                    describe_jax_runtime_policy(attempted_policy),
                )));
            }
            JaxRuntimeConfiguration::Unconfigured => {
                return Err(RuntimeCompatibilityError::new(
                    "Cannot complete JAX runtime setup before recording the configuration attempt.".to_string(),
                ));
            }
            JaxRuntimeConfiguration::Configured(configured_policy) => {
                return Err(RuntimeCompatibilityError::new(format!(
                    "JAX runtime setup was already completed with {}.",
                    describe_jax_runtime_policy(configured_policy),
                )));
            }
        }
        self.configuration = JaxRuntimeConfiguration::Configured(requested_policy);
        Ok(())
    }
}

fn expand_home_directory(path_value: &str) -> Result<PathBuf, RuntimeCompatibilityError> {
    if path_value == "~" {
        return home_directory();
    }
    let Some(relative_path) = path_value.strip_prefix("~/") else {
        return Ok(PathBuf::from(path_value));
    };
    Ok(home_directory()?.join(Path::new(relative_path)))
}

fn home_directory() -> Result<PathBuf, RuntimeCompatibilityError> {
    std::env::var_os("HOME").filter(|directory| !directory.is_empty()).map(PathBuf::from).ok_or_else(|| {
        RuntimeCompatibilityError::new("Cannot expand jax_cache_directory because HOME is not set.".to_string())
    })
}

fn default_jax_runtime_cache_directory() -> PathBuf {
    let user_name = std::env::var("USER")
        .ok()
        .filter(|name| !name.is_empty())
        .or_else(|| std::env::var("LOGNAME").ok().filter(|name| !name.is_empty()))
        .unwrap_or_else(|| UNKNOWN_USER_NAME.to_string());
    std::env::temp_dir().join(user_name).join(DEFAULT_JAX_CACHE_DIRECTORY_NAME)
}

fn describe_jax_runtime_policy(policy: &JaxRuntimePolicy) -> String {
    let cache_directory = match &policy.cache_directory {
        JaxCacheDirectory::Default(_) => "<default>".into(),
        JaxCacheDirectory::Explicit(path) => path.to_string_lossy(),
    };
    format!("device={}, jax-cache-directory={cache_directory}", policy.device.as_str())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{
        JaxRuntimeConfiguration, JaxRuntimeState, build_jax_runtime_policy, describe_jax_runtime_policy,
        expand_home_directory,
    };
    use crate::jax_runtime::{JaxCacheDirectory, JaxGpuValidationStatus, JaxRuntimePolicy};

    fn policy(device: g_plan::Device, cache_directory: &str) -> JaxRuntimePolicy {
        JaxRuntimePolicy { device, cache_directory: JaxCacheDirectory::Explicit(PathBuf::from(cache_directory)) }
    }

    #[test]
    fn run_plan_policy_preserves_device_and_expands_explicit_home() {
        let mut run_plan =
            crate::test_support::run_plan(Path::new("runner-jax-policy"), g_plan::AssociationMode::Regenie2Linear);
        run_plan.compute.device = g_plan::Device::Gpu;
        run_plan.compute.jax_cache_directory = Some("/cache/explicit".to_string());
        let explicit_policy = build_jax_runtime_policy(&run_plan).expect("explicit policy should build");
        assert_eq!(explicit_policy.device, g_plan::Device::Gpu);
        assert_eq!(explicit_policy.cache_directory.path(), Path::new("/cache/explicit"));

        run_plan.compute.jax_cache_directory = None;
        let default_policy = build_jax_runtime_policy(&run_plan).expect("default policy should build");
        assert!(default_policy.cache_directory.path().ends_with("g-jax-cache"));

        if let Some(home_directory) = std::env::var_os("HOME") {
            assert_eq!(
                expand_home_directory("~/jax-cache").expect("home-relative cache should expand"),
                PathBuf::from(home_directory.clone()).join("jax-cache")
            );
            assert_eq!(expand_home_directory("~").expect("home cache should expand"), PathBuf::from(home_directory));
        }
        assert_eq!(
            expand_home_directory("relative/cache").expect("ordinary paths should pass through"),
            PathBuf::from("relative/cache")
        );
    }

    #[test]
    fn mutually_compatible_policy_check_accepts_empty_and_identical_sets() {
        assert!(JaxRuntimeState::require_mutually_compatible(std::iter::empty()).is_ok());
        let first_policy = policy(g_plan::Device::Cpu, "cache");
        let second_policy = policy(g_plan::Device::Cpu, "cache");
        assert!(JaxRuntimeState::require_mutually_compatible([&first_policy, &second_policy]).is_ok());
    }

    #[test]
    fn mutually_compatible_policy_check_names_first_conflicting_run() {
        let first_policy = policy(g_plan::Device::Cpu, "first-cache");
        let second_policy = policy(g_plan::Device::Cpu, "first-cache");
        let third_policy = policy(g_plan::Device::Gpu, "second-cache");
        let error = JaxRuntimeState::require_mutually_compatible([&first_policy, &second_policy, &third_policy])
            .expect_err("incompatible policy should fail");
        let message = error.to_string();
        assert!(message.contains("Run 1 and run 3"));
        assert!(message.contains("device=cpu"));
        assert!(message.contains("device=gpu"));
    }

    #[test]
    fn runtime_state_reserves_completes_and_reuses_one_policy() {
        let requested_policy = policy(g_plan::Device::Cpu, "cache");
        let mut state = JaxRuntimeState::default();
        assert!(state.require_compatible(&requested_policy).is_ok());
        assert!(state.setup_preparation_required(&requested_policy).expect("setup query should succeed"));
        let setup_session = state.reserve_setup(&requested_policy).expect("setup should reserve");
        assert!(setup_session.should_configure);
        assert!(state.require_compatible(&requested_policy).is_err());
        state
            .complete_setup(requested_policy.clone(), JaxGpuValidationStatus::Skipped)
            .expect("completed setup should be recorded");
        assert!(state.require_compatible(&requested_policy).is_ok());
        assert!(!state.setup_preparation_required(&requested_policy).expect("configured setup query should succeed"));
        let reused_session = state.reserve_setup(&requested_policy).expect("configured policy should be reusable");
        assert!(!reused_session.should_configure);
    }

    #[test]
    fn runtime_state_rejects_pending_failed_mismatched_and_duplicate_completion() {
        let requested_policy = policy(g_plan::Device::Cpu, "cache");
        let conflicting_policy = policy(g_plan::Device::Gpu, "other-cache");

        let mut pending_state = JaxRuntimeState::default();
        pending_state.reserve_setup(&requested_policy).expect("setup should reserve");
        assert!(pending_state.complete_setup(requested_policy.clone(), JaxGpuValidationStatus::Pending).is_err());

        let mut failed_state = JaxRuntimeState::default();
        failed_state.reserve_setup(&requested_policy).expect("setup should reserve");
        assert!(failed_state.complete_setup(requested_policy.clone(), JaxGpuValidationStatus::Failed).is_err());

        let mut mismatched_state = JaxRuntimeState::default();
        mismatched_state.reserve_setup(&requested_policy).expect("setup should reserve");
        assert!(mismatched_state.complete_setup(conflicting_policy.clone(), JaxGpuValidationStatus::Skipped).is_err());

        let mut unconfigured_state = JaxRuntimeState::default();
        assert!(unconfigured_state.complete_setup(requested_policy.clone(), JaxGpuValidationStatus::Skipped).is_err());

        let mut configured_state =
            JaxRuntimeState { configuration: JaxRuntimeConfiguration::Configured(requested_policy.clone()) };
        assert!(configured_state.complete_setup(requested_policy.clone(), JaxGpuValidationStatus::Skipped).is_err());
        let incompatible_error = configured_state
            .require_compatible(&conflicting_policy)
            .expect_err("configured state should reject a conflicting policy");
        assert!(incompatible_error.to_string().contains("process-global"));
    }

    #[test]
    fn policy_description_hides_default_path_and_displays_explicit_path() {
        let default_policy = JaxRuntimePolicy {
            device: g_plan::Device::Cpu,
            cache_directory: JaxCacheDirectory::Default(PathBuf::from("/ambient/default")),
        };
        assert_eq!(describe_jax_runtime_policy(&default_policy), "device=cpu, jax-cache-directory=<default>");
        assert_eq!(
            describe_jax_runtime_policy(&policy(g_plan::Device::Gpu, "/explicit/cache")),
            "device=gpu, jax-cache-directory=/explicit/cache"
        );
    }
}
