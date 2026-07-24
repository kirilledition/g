//! Projection from application planning into generic runtime resources.

use std::path::Path;

use g_runtime::NativeRunSessionPolicy;

const EVENTS_JSONL_FILE_NAME: &str = "events.jsonl";
const LOG_QUEUE_SIZE: usize = 65_536;
const PROFILE_SUMMARY_JSON_FILE_NAME: &str = "profile.summary.json";

#[must_use]
pub(crate) fn project_native_run_session_policy(
    run_plan: &g_plan::RunPlan,
    diagnostics_directory: &Path,
) -> NativeRunSessionPolicy {
    let telemetry_enabled = run_plan.telemetry != g_plan::TelemetryMode::Off;
    let telemetry_stream_file = telemetry_enabled.then(|| diagnostics_directory.join(EVENTS_JSONL_FILE_NAME));
    let profile_summary_file = (run_plan.telemetry == g_plan::TelemetryMode::Profile)
        .then(|| diagnostics_directory.join(PROFILE_SUMMARY_JSON_FILE_NAME));
    NativeRunSessionPolicy {
        log_filter: "info".to_string(),
        log_stderr: true,
        log_file: None,
        telemetry_stream_file,
        stage_timing_file: None,
        profile_summary_file,
        queue_size: LOG_QUEUE_SIZE,
        lossy: true,
        include_source_location: false,
        include_span_events: false,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        EVENTS_JSONL_FILE_NAME, LOG_QUEUE_SIZE, PROFILE_SUMMARY_JSON_FILE_NAME, project_native_run_session_policy,
    };

    #[test]
    fn session_policy_projects_off_progress_and_profile_resources() {
        let mut run_plan =
            crate::test_support::run_plan(Path::new("runner-policy"), g_plan::AssociationMode::Regenie2Linear);

        let diagnostics_directory = Path::new("claimed-attempt/diagnostics/owner-test");
        let off_policy = project_native_run_session_policy(&run_plan, diagnostics_directory);
        assert_eq!(off_policy.telemetry_stream_file, None);
        assert_eq!(off_policy.profile_summary_file, None);
        assert_eq!(off_policy.queue_size, LOG_QUEUE_SIZE);
        assert!(off_policy.log_stderr);
        assert!(off_policy.lossy);
        assert!(!off_policy.include_source_location);
        assert!(!off_policy.include_span_events);

        run_plan.telemetry = g_plan::TelemetryMode::Progress;
        let progress_policy = project_native_run_session_policy(&run_plan, diagnostics_directory);
        assert_eq!(
            progress_policy.telemetry_stream_file.as_deref(),
            Some(diagnostics_directory.join(EVENTS_JSONL_FILE_NAME).as_path())
        );
        assert_eq!(progress_policy.profile_summary_file, None);

        run_plan.telemetry = g_plan::TelemetryMode::Profile;
        let profile_policy = project_native_run_session_policy(&run_plan, diagnostics_directory);
        assert_eq!(
            profile_policy.profile_summary_file.as_deref(),
            Some(diagnostics_directory.join(PROFILE_SUMMARY_JSON_FILE_NAME).as_path())
        );
    }
}
