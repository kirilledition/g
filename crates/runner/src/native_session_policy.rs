//! Projection from application planning into generic runtime resources.

use std::path::{Path, PathBuf};

use g_runtime::NativeRunSessionPolicy;

const EVENTS_JSONL_FILE_NAME: &str = "events.jsonl";
const PROFILE_SUMMARY_JSON_FILE_NAME: &str = "profile.summary.json";

#[must_use]
pub(crate) fn project_native_run_session_policy(run_plan: &g_plan::RunPlan) -> NativeRunSessionPolicy {
    let diagnostics = &run_plan.diagnostics;
    let telemetry_enabled = diagnostics.telemetry != g_plan::TelemetryMode::Off;
    let telemetry_directory_required = telemetry_enabled
        && (diagnostics.log_file.is_none()
            || (diagnostics.telemetry == g_plan::TelemetryMode::Profile && diagnostics.profile_summary_path.is_none()));
    let telemetry_directory = if telemetry_directory_required {
        Some(
            diagnostics
                .log_directory
                .as_deref()
                .map_or_else(|| Path::new(&run_plan.output.output_run_root).join("logs"), PathBuf::from),
        )
    } else {
        None
    };
    let telemetry_stream_file = if telemetry_enabled {
        diagnostics
            .log_file
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| telemetry_directory.as_ref().map(|directory| directory.join(EVENTS_JSONL_FILE_NAME)))
    } else {
        None
    };
    let profile_summary_file = diagnostics.profile_summary_path.as_deref().map(PathBuf::from).or_else(|| {
        (diagnostics.telemetry == g_plan::TelemetryMode::Profile)
            .then(|| telemetry_directory.as_ref().map(|directory| directory.join(PROFILE_SUMMARY_JSON_FILE_NAME)))
            .flatten()
    });
    let log_file =
        if telemetry_stream_file.is_none() { diagnostics.log_file.as_deref().map(PathBuf::from) } else { None };
    NativeRunSessionPolicy {
        log_filter: diagnostics.log_filter.clone(),
        log_stderr: diagnostics.log_to_stderr,
        log_file,
        telemetry_stream_file,
        stage_timing_file: diagnostics.stage_timings_path.as_deref().map(PathBuf::from),
        profile_summary_file,
        queue_size: usize::try_from(diagnostics.log_queue_size)
            .expect("u32 queue capacity fits usize on supported 64-bit targets"),
        lossy: diagnostics.lossy_logging,
        include_source_location: diagnostics.include_source_location,
        include_span_events: diagnostics.include_span_events,
    }
}
