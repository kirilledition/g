//! Projection from application planning into generic runtime resources.

use std::path::Path;

use g_runtime::NativeRunSessionPolicy;

const EVENTS_JSONL_FILE_NAME: &str = "events.jsonl";
const LOG_QUEUE_SIZE: usize = 65_536;
const PROFILE_SUMMARY_JSON_FILE_NAME: &str = "profile.summary.json";

#[must_use]
pub(crate) fn project_native_run_session_policy(run_plan: &g_plan::RunPlan) -> NativeRunSessionPolicy {
    let telemetry_enabled = run_plan.telemetry != g_plan::TelemetryMode::Off;
    let telemetry_directory = Path::new(&run_plan.output.output_run_root).join("logs");
    let telemetry_stream_file = telemetry_enabled.then(|| telemetry_directory.join(EVENTS_JSONL_FILE_NAME));
    let profile_summary_file = (run_plan.telemetry == g_plan::TelemetryMode::Profile)
        .then(|| telemetry_directory.join(PROFILE_SUMMARY_JSON_FILE_NAME));
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
