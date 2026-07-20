//! Generic native-session and process logging policy.

use std::borrow::Cow;
use std::path::PathBuf;

use crate::error::RuntimeCompatibilityError;

/// Resolved resources and formatting policy for one native run.
///
/// The runner projects this policy from its application plan. Runtime owns
/// only the concrete resources and never interprets application planning
/// enums or output layout.
#[derive(Debug, Eq, PartialEq)]
// These switches describe independent logging and telemetry capabilities, not
// mutually exclusive states that would benefit from an enum state machine.
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
// The projected subscriber retains the same independent capability switches.
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
    /// Require another run to use the same process-global logging policy.
    ///
    /// # Errors
    ///
    /// Returns an error when subscriber configuration differs.
    pub fn require_compatible_process_logging_policy(
        &self,
        requested_policy: &Self,
    ) -> Result<(), RuntimeCompatibilityError> {
        let configured_subscriber_policy = self.subscriber_policy();
        let requested_subscriber_policy = requested_policy.subscriber_policy();
        if configured_subscriber_policy == requested_subscriber_policy {
            return Ok(());
        }
        Err(RuntimeCompatibilityError::new(format!(
            "Process-global logging policies differ. Configured policy: {}. Requested policy: {}.",
            describe_logging_subscriber_policy(&configured_subscriber_policy),
            describe_logging_subscriber_policy(&requested_subscriber_policy),
        )))
    }

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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::path::PathBuf;

    use super::*;
    use crate::test_support::disabled_session_policy;

    #[test]
    fn subscriber_compatibility_uses_only_process_global_topology() {
        let configured = disabled_session_policy();
        let mut requested = disabled_session_policy();
        requested.log_file = Some(PathBuf::from("second.log"));
        assert!(configured.require_compatible_process_logging_policy(&requested).is_err());

        let mut configured_with_files = disabled_session_policy();
        configured_with_files.log_file = Some(PathBuf::from("first.log"));
        configured_with_files.telemetry_stream_file = Some(PathBuf::from("first.jsonl"));
        let mut requested_with_files = disabled_session_policy();
        requested_with_files.log_file = Some(PathBuf::from("second.log"));
        requested_with_files.telemetry_stream_file = Some(PathBuf::from("second.jsonl"));
        requested_with_files.queue_size = 1;
        requested_with_files.lossy = true;
        assert_eq!(configured_with_files.require_compatible_process_logging_policy(&requested_with_files), Ok(()));
    }

    #[test]
    fn every_subscriber_setting_participates_in_compatibility() {
        let configured = disabled_session_policy();
        let mut variants = Vec::new();

        let mut log_filter = disabled_session_policy();
        log_filter.log_filter = "debug".to_owned();
        variants.push(log_filter);

        let mut log_stderr = disabled_session_policy();
        log_stderr.log_stderr = true;
        variants.push(log_stderr);

        let mut log_file = disabled_session_policy();
        log_file.log_file = Some(PathBuf::from("run.log"));
        variants.push(log_file);

        let mut telemetry = disabled_session_policy();
        telemetry.telemetry_stream_file = Some(PathBuf::from("events.jsonl"));
        variants.push(telemetry);

        let mut source_location = disabled_session_policy();
        source_location.include_source_location = true;
        variants.push(source_location);

        let mut span_events = disabled_session_policy();
        span_events.include_span_events = true;
        variants.push(span_events);

        for requested in variants {
            let error = configured
                .require_compatible_process_logging_policy(&requested)
                .expect_err("changed subscriber setting should be incompatible");
            assert!(error.to_string().contains("Process-global logging policies differ"));
        }
    }

    #[test]
    fn subscriber_description_and_owned_projection_are_stable() {
        let policy = LoggingSubscriberPolicy {
            log_filter: Cow::Borrowed("g=trace"),
            log_stderr: true,
            log_file_enabled: false,
            structured_log_enabled: true,
            include_source_location: false,
            include_span_events: true,
        };
        assert_eq!(
            describe_logging_subscriber_policy(&policy),
            "log-filter=g=trace, log-stderr=True, log-file-enabled=False, structured-log-enabled=True, \
             include-source-location=False, include-span-events=True"
        );
        let owned = policy.into_owned();
        assert!(matches!(&owned.log_filter, Cow::Owned(_)));
        assert_eq!(owned.log_filter, "g=trace");
    }
}
