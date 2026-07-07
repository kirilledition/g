//! Process runtime compatibility state.

#[cfg(test)]
use crate::rayon_runtime;
#[cfg(test)]
use crate::runtime_policy::LoggingRuntimePolicyPayload;

mod compatibility;
mod jax_policy;
mod policy;
mod process;
mod rayon_pool;
mod run;
mod token;

pub use jax_policy::{
    JaxRuntimePolicyPayload, JaxRuntimeSetupLifecyclePlan, build_jax_runtime_policy_payload,
    describe_jax_runtime_policy, resolve_jax_runtime_cache_directory,
};
pub use policy::RuntimePolicyPayload;
pub use process::{ProcessRuntimeState, RuntimeStateSnapshotPayload};
pub use rayon_pool::{RayonThreadPoolConfigurationError, RayonThreadPoolConfigurationPlan};
pub use run::RunRuntime;
pub use token::RuntimeCompatibilityToken;

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
    fn completes_jax_runtime_setup_from_session() {
        let mut state = ProcessRuntimeState::default();
        let requested_policy = build_jax_policy(Some("/tmp/cache"));
        let completed_cpu_session = state
            .build_jax_runtime_setup_session(&requested_policy, "/tmp/cache")
            .expect("compatible JAX policy should build a setup session");

        state
            .complete_jax_runtime_setup_session(requested_policy.clone(), &completed_cpu_session)
            .expect("completed CPU JAX setup should be recorded");

        assert_eq!(state.jax_policy, Some(requested_policy));
    }

    #[test]
    fn rejects_pending_jax_runtime_setup_session_completion() {
        let mut state = ProcessRuntimeState::default();
        let requested_policy =
            build_jax_runtime_policy_payload("gpu", Some("/tmp/cache"), None, true, 0, 0, false, false);
        let pending_gpu_session = state
            .build_jax_runtime_setup_session(&requested_policy, "/tmp/cache")
            .expect("compatible JAX policy should build a setup session");

        let error = state
            .complete_jax_runtime_setup_session(requested_policy, &pending_gpu_session)
            .expect_err("pending GPU validation should not be recorded");

        assert_eq!(error.to_string(), "Cannot record JAX runtime setup before GPU validation completes.");
    }

    #[test]
    fn rejects_failed_jax_runtime_setup_session_completion() {
        let mut state = ProcessRuntimeState::default();
        let requested_policy =
            build_jax_runtime_policy_payload("gpu", Some("/tmp/cache"), None, true, 0, 0, false, false);
        let mut failed_gpu_session = state
            .build_jax_runtime_setup_session(&requested_policy, "/tmp/cache")
            .expect("compatible JAX policy should build a setup session");
        let _failed_setup = failed_gpu_session.complete_validation("failed", Some("no GPU"));

        let error = state
            .complete_jax_runtime_setup_session(requested_policy, &failed_gpu_session)
            .expect_err("failed GPU validation should not be recorded");

        assert_eq!(error.to_string(), "Cannot record failed JAX runtime setup as process-global runtime state.",);
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
    fn builds_jax_runtime_setup_session_with_native_cache_directory_resolution() {
        let state = ProcessRuntimeState::default();
        let default_policy = build_jax_policy(None);
        let explicit_policy = build_jax_policy(Some("/tmp/cache"));

        let default_session = state
            .build_jax_runtime_setup_session_resolving_cache_directory(&default_policy)
            .expect("default-cache JAX policy should build a setup session");
        let explicit_session = state
            .build_jax_runtime_setup_session_resolving_cache_directory(&explicit_policy)
            .expect("explicit-cache JAX policy should build a setup session");

        assert!(default_session.setup().cache_directory.ends_with("/g-jax-cache"));
        assert_eq!(explicit_session.setup().cache_directory, "/tmp/cache");
        assert_eq!(resolve_jax_runtime_cache_directory(&explicit_policy), "/tmp/cache");
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
