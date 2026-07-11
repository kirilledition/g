//! Generic native-session and process logging policy.

use std::borrow::Cow;
use std::path::PathBuf;

/// Resolved resources and formatting policy for one native run.
///
/// The runner projects this policy from its application plan. Runtime owns
/// only the concrete resources and never interprets application planning
/// enums or output layout.
#[derive(Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct NativeRunSessionPolicy {
    pub log_filter: String,
    pub log_stderr: bool,
    pub log_file: Option<PathBuf>,
    pub telemetry_stream_file: Option<PathBuf>,
    pub stage_timing_file: Option<PathBuf>,
    pub profile_summary_file: Option<PathBuf>,
    pub queue_size: usize,
    pub lossy: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct LoggingSubscriberPolicy<'policy> {
    pub(crate) log_filter: Cow<'policy, str>,
    pub(crate) log_stderr: bool,
    pub(crate) log_file_enabled: bool,
    pub(crate) structured_log_enabled: bool,
    pub(crate) include_source_location: bool,
    pub(crate) include_span_events: bool,
}

impl NativeRunSessionPolicy {
    #[must_use]
    pub(crate) fn subscriber_policy(&self) -> LoggingSubscriberPolicy<'_> {
        LoggingSubscriberPolicy {
            log_filter: Cow::Borrowed(&self.log_filter),
            log_stderr: self.log_stderr,
            log_file_enabled: self.log_file.is_some(),
            structured_log_enabled: self.telemetry_stream_file.is_some(),
            include_source_location: self.include_source_location,
            include_span_events: self.include_span_events,
        }
    }
}

impl LoggingSubscriberPolicy<'_> {
    #[must_use]
    pub(crate) fn into_owned(self) -> LoggingSubscriberPolicy<'static> {
        LoggingSubscriberPolicy {
            log_filter: Cow::Owned(self.log_filter.into_owned()),
            log_stderr: self.log_stderr,
            log_file_enabled: self.log_file_enabled,
            structured_log_enabled: self.structured_log_enabled,
            include_source_location: self.include_source_location,
            include_span_events: self.include_span_events,
        }
    }
}

#[must_use]
pub(crate) fn describe_logging_subscriber_policy(policy: &LoggingSubscriberPolicy<'_>) -> String {
    format!(
        "log-filter={}, log-stderr={}, log-file-enabled={}, structured-log-enabled={}, \
         include-source-location={}, include-span-events={}",
        policy.log_filter,
        python_bool(policy.log_stderr),
        python_bool(policy.log_file_enabled),
        python_bool(policy.structured_log_enabled),
        python_bool(policy.include_source_location),
        python_bool(policy.include_span_events),
    )
}

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}
