//! Deterministic process runtime policy helpers.

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct LoggingRuntimePolicyPayload {
    pub log_filter: String,
    pub log_file: Option<String>,
    pub log_stderr: bool,
    pub log_queue_size: i64,
    pub log_lossy: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
    pub trace_file: Option<String>,
    pub trace_filter: String,
    pub trace_event_cap: Option<i64>,
}

#[must_use]
pub(crate) fn build_logging_runtime_policy(
    run_plan: &g_plan::RunPlan,
    telemetry_paths: &crate::telemetry_policy::TelemetryPathsPayload,
) -> LoggingRuntimePolicyPayload {
    let diagnostics = &run_plan.diagnostics;
    let telemetry_stream_file = telemetry_paths.stream_file.clone();
    let telemetry_stream_file_is_some = telemetry_stream_file.is_some();
    let resolved_log_file = if telemetry_stream_file_is_some { None } else { diagnostics.log_file.clone() };
    let resolved_trace_file = telemetry_stream_file.or_else(|| diagnostics.trace_file.clone());
    let resolved_trace_filter = diagnostics.log_filter.clone();
    let resolved_trace_event_cap = None;
    LoggingRuntimePolicyPayload {
        log_filter: diagnostics.log_filter.clone(),
        log_file: resolved_log_file,
        log_stderr: diagnostics.log_to_stderr,
        log_queue_size: i64::from(diagnostics.log_queue_size),
        log_lossy: diagnostics.lossy_logging,
        include_source_location: diagnostics.include_source_location,
        include_span_events: diagnostics.include_span_events,
        trace_file: resolved_trace_file,
        trace_filter: resolved_trace_filter,
        trace_event_cap: resolved_trace_event_cap,
    }
}

#[must_use]
pub fn describe_logging_runtime_policy(policy: &LoggingRuntimePolicyPayload) -> String {
    format!(
        "log-filter={}, log-file={}, log-stderr={}, log-queue-size={}, log-lossy={}, \
         include-source-location={}, include-span-events={}, trace-file={}, trace-filter={}, trace-event-cap={}",
        policy.log_filter,
        optional_text(policy.log_file.as_deref()),
        python_bool(policy.log_stderr),
        policy.log_queue_size,
        python_bool(policy.log_lossy),
        python_bool(policy.include_source_location),
        python_bool(policy.include_span_events),
        optional_text(policy.trace_file.as_deref()),
        policy.trace_filter,
        optional_i64_text(policy.trace_event_cap),
    )
}

fn optional_text(value: Option<&str>) -> &str {
    value.unwrap_or("<none>")
}

fn optional_i64_text(value: Option<i64>) -> String {
    value.map_or_else(|| "<none>".to_string(), |number| number.to_string())
}

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}
