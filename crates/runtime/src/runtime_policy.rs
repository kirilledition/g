//! Deterministic process runtime policy helpers.

use crate::telemetry_policy::TelemetryMode;

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

#[allow(clippy::too_many_arguments)]
#[allow(clippy::fn_params_excessive_bools)]
#[must_use]
pub fn build_logging_runtime_policy(
    log_filter: String,
    log_file: Option<String>,
    log_stderr: bool,
    log_queue_size: i64,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: Option<String>,
    trace_filter: String,
    trace_event_cap: Option<i64>,
    telemetry_mode: TelemetryMode,
    telemetry_stream_file: Option<String>,
) -> LoggingRuntimePolicyPayload {
    let telemetry_stream_file_is_some = telemetry_stream_file.is_some();
    let resolved_log_file = if telemetry_stream_file_is_some { None } else { log_file };
    let resolved_trace_file = telemetry_stream_file.or(trace_file);
    let resolved_trace_filter = if telemetry_stream_file_is_some && !telemetry_mode.trace_enabled() {
        log_filter.clone()
    } else {
        trace_filter
    };
    let resolved_trace_event_cap = if telemetry_mode.trace_enabled() { trace_event_cap } else { None };
    LoggingRuntimePolicyPayload {
        log_filter,
        log_file: resolved_log_file,
        log_stderr,
        log_queue_size,
        log_lossy,
        include_source_location,
        include_span_events,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_stream_owns_trace_file_and_filter_policy() {
        let policy = build_logging_runtime_policy(
            "info".to_string(),
            Some("run.log".to_string()),
            true,
            256,
            false,
            true,
            false,
            Some("trace.jsonl".to_string()),
            "debug".to_string(),
            Some(100),
            "profile",
            Some("events.jsonl".to_string()),
        );

        assert_eq!(policy.log_file, None);
        assert_eq!(policy.trace_file, Some("events.jsonl".to_string()));
        assert_eq!(policy.trace_filter, "info");
        assert_eq!(policy.trace_event_cap, None);
    }
}
