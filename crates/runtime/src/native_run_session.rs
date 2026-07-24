//! Native run-scoped telemetry, timing, and shutdown ownership.

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use crate::diagnostics::emit_diagnostic_event;
use crate::logging_sink::{RunLoggingSession, initialize_logging_sinks};
use crate::runtime_policy::NativeRunSessionPolicy;
use crate::runtime_state::ProcessRuntimeState;
use crate::shutdown::{SigtermShutdownScope, begin_sigterm_shutdown_scope};
use crate::telemetry_session::generate_run_id;
use crate::{
    LoggingSinkError, RuntimeCompatibilityError, ShutdownError, StageTimingRecorder, TelemetryRunError,
    TelemetryRunSession,
};

#[derive(Debug)]
pub enum NativeRunSessionError {
    Compatibility(RuntimeCompatibilityError),
    Shutdown(ShutdownError),
    Logging(LoggingSinkError),
    Telemetry(TelemetryRunError),
}

impl fmt::Display for NativeRunSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compatibility(error) => error.fmt(formatter),
            Self::Shutdown(error) => error.fmt(formatter),
            Self::Logging(error) => error.fmt(formatter),
            Self::Telemetry(error) => error.fmt(formatter),
        }
    }
}

impl Error for NativeRunSessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Compatibility(error) => Some(error),
            Self::Shutdown(error) => Some(error),
            Self::Logging(error) => Some(error),
            Self::Telemetry(error) => Some(error),
        }
    }
}

#[derive(Serialize)]
struct TimingOutputDiagnosticFields<'session> {
    stage_timing_path: Option<&'session Path>,
    profile_summary_path: Option<&'session Path>,
    run_id: &'session str,
}

struct NativeTimingSession {
    recorder: StageTimingRecorder,
    run_id: Arc<str>,
}

pub struct NativeRunSession {
    policy: NativeRunSessionPolicy,
    logging_session: RunLoggingSession,
    telemetry_session: TelemetryRunSession,
    timing_session: Option<NativeTimingSession>,
    run_start_time: Instant,
    _shutdown_scope: SigtermShutdownScope,
}

impl NativeRunSession {
    /// Validate process compatibility and open run-scoped native resources.
    ///
    /// # Errors
    ///
    /// Returns an error before resource creation for incompatible subscriber
    /// policy, or when signal handling and logging/telemetry writers cannot be
    /// initialized.
    pub fn new(
        process_runtime_state: &mut ProcessRuntimeState,
        policy: NativeRunSessionPolicy,
    ) -> Result<Self, NativeRunSessionError> {
        process_runtime_state
            .require_compatible_logging_policy(&policy)
            .map_err(NativeRunSessionError::Compatibility)?;
        let shutdown_scope = begin_sigterm_shutdown_scope().map_err(NativeRunSessionError::Shutdown)?;
        let timing_enabled = policy.stage_timing_file.is_some() || policy.profile_summary_file.is_some();
        let run_id = (policy.telemetry_stream_file.is_some() || timing_enabled).then(generate_run_id);
        let telemetry_session = policy
            .telemetry_stream_file
            .as_deref()
            .zip(run_id.as_ref())
            .map(|(stream_file, run_id)| {
                TelemetryRunSession::new(stream_file, policy.queue_size, policy.lossy, Arc::clone(run_id))
            })
            .transpose()
            .map_err(NativeRunSessionError::Telemetry)?
            .unwrap_or_default();
        let timing_session = if timing_enabled {
            run_id.map(|run_id| NativeTimingSession { recorder: StageTimingRecorder::default(), run_id })
        } else {
            None
        };
        let logging_session = RunLoggingSession::new(&policy).map_err(NativeRunSessionError::Logging)?;
        initialize_logging_sinks(&policy).map_err(NativeRunSessionError::Logging)?;
        process_runtime_state.record_logging_subscriber_policy(&policy);
        Ok(Self {
            policy,
            logging_session,
            telemetry_session,
            timing_session,
            run_start_time: Instant::now(),
            _shutdown_scope: shutdown_scope,
        })
    }

    pub fn record_stage_duration(&mut self, stage_name: &str, start_time: Instant) {
        if let Some(timing_session) = self.timing_session.as_mut() {
            timing_session.recorder.add_stage_duration(stage_name, start_time.elapsed().as_secs_f64());
        }
    }

    #[must_use]
    pub const fn policy(&self) -> &NativeRunSessionPolicy {
        &self.policy
    }

    #[must_use]
    pub const fn telemetry_session(&self) -> &TelemetryRunSession {
        &self.telemetry_session
    }

    pub fn stage_timing_recorder(&mut self) -> Option<&mut StageTimingRecorder> {
        self.timing_session.as_mut().map(|timing_session| &mut timing_session.recorder)
    }

    /// Flush the asynchronous stderr and plain-file writers owned by this run.
    ///
    /// # Errors
    ///
    /// Returns a logging error when a dynamic-writer registry is unavailable.
    pub fn finish_logging(&mut self) -> Result<(), LoggingSinkError> {
        self.logging_session.finish()
    }

    /// Write configured timing outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when a timing payload cannot be written.
    pub fn finish_timing(&mut self) -> Result<(), crate::TimingFileError> {
        let Some(timing_session) = self.timing_session.as_mut() else {
            return Ok(());
        };
        timing_session.recorder.add_stage_duration("runner_total", self.run_start_time.elapsed().as_secs_f64());
        let run_id = timing_session.run_id.as_ref();
        if let Err(error) = emit_diagnostic_event(
            "debug",
            "runner_final_timing_outputs_write_started",
            "Writing final timing outputs.",
            &TimingOutputDiagnosticFields {
                stage_timing_path: self.policy.stage_timing_file.as_deref(),
                profile_summary_path: self.policy.profile_summary_file.as_deref(),
                run_id,
            },
        ) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tracing::warn!(target: "g.runtime", error = %error, "Failed to emit timing diagnostic event.");
            }));
        }
        timing_session.recorder.write_final_timing_outputs(
            self.policy.stage_timing_file.as_deref(),
            self.policy.profile_summary_file.as_deref(),
            Some(run_id),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::test_support::{TemporaryDirectory, disabled_session_policy, execute_isolated_test_body};

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn native_session_validates_before_state_commit_and_owns_complete_lifecycle() {
        if !execute_isolated_test_body(
            "native_run_session::tests::native_session_validates_before_state_commit_and_owns_complete_lifecycle",
            "G_RUNTIME_NATIVE_SESSION_TEST_CHILD",
        ) {
            return;
        }
        let temporary_directory = TemporaryDirectory::new("native-session");
        let mut process_state = ProcessRuntimeState::default();

        let blocking_file = temporary_directory.path().join("blocking-file");
        std::fs::write(&blocking_file, b"file").expect("blocking fixture should be written");
        let mut invalid_telemetry_policy = disabled_session_policy();
        invalid_telemetry_policy.telemetry_stream_file = Some(blocking_file.join("events.jsonl"));
        let telemetry_error = NativeRunSession::new(&mut process_state, invalid_telemetry_policy)
            .err()
            .expect("invalid telemetry path should fail");
        assert!(matches!(&telemetry_error, NativeRunSessionError::Telemetry(_)));
        assert!(telemetry_error.source().is_some());
        assert_eq!(process_state, ProcessRuntimeState::default());
        assert!(!crate::sigterm_shutdown_requested());

        let mut invalid_filter_policy = disabled_session_policy();
        invalid_filter_policy.log_filter = "g.runtime=not-a-level".to_owned();
        let logging_error = NativeRunSession::new(&mut process_state, invalid_filter_policy)
            .err()
            .expect("invalid initial filter should fail");
        assert!(matches!(&logging_error, NativeRunSessionError::Logging(LoggingSinkError::InvalidLogFilter { .. })));
        assert!(logging_error.source().is_some());
        assert_eq!(process_state, ProcessRuntimeState::default());
        assert!(!crate::sigterm_shutdown_requested());

        let disabled_policy = disabled_session_policy();
        let mut disabled_session =
            NativeRunSession::new(&mut process_state, disabled_policy).expect("disabled session should open");
        assert!(!disabled_session.telemetry_session().is_enabled());
        assert!(disabled_session.stage_timing_recorder().is_none());
        assert!(disabled_session.timing_session.is_none());
        disabled_session.record_stage_duration(
            "ignored",
            Instant::now().checked_sub(Duration::from_millis(1)).expect("one millisecond should be representable"),
        );
        disabled_session.finish_timing().expect("disabled timing should be a no-op");

        let shutdown_error = NativeRunSession::new(&mut process_state, disabled_session_policy())
            .err()
            .expect("overlapping native session should fail shutdown ownership");
        assert!(matches!(&shutdown_error, NativeRunSessionError::Shutdown(_)));
        assert!(shutdown_error.source().is_some());
        disabled_session.finish_logging().expect("disabled logging should finish");
        drop(disabled_session);

        let stage_path = temporary_directory.path().join("timing/stages.json");
        let profile_path = temporary_directory.path().join("timing/profile.json");
        let mut timing_policy = disabled_session_policy();
        timing_policy.stage_timing_file = Some(stage_path.clone());
        timing_policy.profile_summary_file = Some(profile_path.clone());
        let mut timing_session =
            NativeRunSession::new(&mut process_state, timing_policy).expect("timing session should open");
        assert!(!timing_session.telemetry_session().is_enabled());
        let run_id = timing_session.timing_session.as_ref().expect("timing state should exist").run_id.clone();
        assert_eq!(run_id.len(), 32);
        timing_session
            .stage_timing_recorder()
            .expect("timing recorder should exist")
            .add_stage_duration("compute", 0.25);
        timing_session.record_stage_duration(
            "prepare",
            Instant::now().checked_sub(Duration::from_millis(1)).expect("one millisecond should be representable"),
        );
        timing_session.finish_timing().expect("timing outputs should be written");
        timing_session.finish_logging().expect("timing session logging should finish");

        let stage_payload: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&stage_path).expect("stage timing output should be readable"))
                .expect("stage timing output should parse");
        let profile_payload: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&profile_path).expect("profile summary should be readable"))
                .expect("profile summary should parse");
        assert_eq!(profile_payload["schema_version"], 0);
        assert_eq!(profile_payload["run_id"], run_id.as_ref());
        assert!(stage_payload.get("schema_version").is_none());
        let compute_total =
            profile_payload["stage_totals_seconds"]["compute"].as_f64().expect("compute total should be numeric");
        assert!((compute_total - 0.25).abs() < TOLERANCE);
        assert_eq!(profile_payload["stage_counts"]["compute"], 1);
        assert_eq!(profile_payload["stage_counts"]["prepare"], 1);
        assert_eq!(profile_payload["stage_counts"]["runner_total"], 1);
        drop(timing_session);

        let incompatible_path = temporary_directory.path().join("must-not-exist.json");
        let mut incompatible_policy = disabled_session_policy();
        incompatible_policy.log_stderr = true;
        incompatible_policy.stage_timing_file = Some(incompatible_path.clone());
        let compatibility_error = NativeRunSession::new(&mut process_state, incompatible_policy)
            .err()
            .expect("changed subscriber topology should fail");
        assert!(matches!(&compatibility_error, NativeRunSessionError::Compatibility(_)));
        assert!(compatibility_error.source().is_some());
        assert!(!incompatible_path.exists());
        assert!(!crate::sigterm_shutdown_requested());
    }
}
