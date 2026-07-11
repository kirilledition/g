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
            tracing::warn!(target: "g.runtime", error = %error, "Failed to emit timing diagnostic event.");
        }
        timing_session.recorder.write_final_timing_outputs(
            self.policy.stage_timing_file.as_deref(),
            self.policy.profile_summary_file.as_deref(),
            Some(run_id),
        )
    }
}
