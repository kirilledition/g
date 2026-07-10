//! Native run-scoped telemetry, timing, and shutdown ownership.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::run_events::emit_diagnostic_event;
use crate::runtime_policy::build_logging_runtime_policy;
use crate::telemetry_policy::resolve_telemetry_paths;
use crate::{
    LoggingRuntimePolicyPayload, ShutdownError, SigtermShutdownScope, StageTimingRecorder, TelemetryPathError,
    TelemetryRunError, TelemetryRunSession, begin_sigterm_shutdown_scope,
    build_final_timing_outputs_write_started_diagnostic_payload, generate_run_id,
};

#[derive(Debug)]
pub enum NativeRunSessionError {
    Shutdown(ShutdownError),
    TelemetryPath(TelemetryPathError),
    Telemetry(TelemetryRunError),
    QueueSizeOutOfRange,
}

impl fmt::Display for NativeRunSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shutdown(error) => error.fmt(formatter),
            Self::TelemetryPath(error) => error.fmt(formatter),
            Self::Telemetry(error) => error.fmt(formatter),
            Self::QueueSizeOutOfRange => formatter.write_str("Telemetry queue size does not fit native usize."),
        }
    }
}

impl Error for NativeRunSessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Shutdown(error) => Some(error),
            Self::TelemetryPath(error) => Some(error),
            Self::Telemetry(error) => Some(error),
            Self::QueueSizeOutOfRange => None,
        }
    }
}

pub struct NativeRunSession {
    pub logging_policy: LoggingRuntimePolicyPayload,
    pub telemetry_session: TelemetryRunSession,
    pub stage_timing_recorder: Option<StageTimingRecorder>,
    stage_timing_path: Option<String>,
    profile_summary_path: Option<String>,
    run_id: String,
    run_start_time: Instant,
    _shutdown_scope: SigtermShutdownScope,
}

impl NativeRunSession {
    /// Open run-scoped native lifecycle resources from the canonical plan.
    ///
    /// # Errors
    ///
    /// Returns an error when signal handling, telemetry paths, or the telemetry
    /// writer cannot be initialized.
    pub fn new(run_plan: &g_plan::RunPlan) -> Result<Self, NativeRunSessionError> {
        let shutdown_scope = begin_sigterm_shutdown_scope().map_err(NativeRunSessionError::Shutdown)?;
        let telemetry_paths = resolve_telemetry_paths(run_plan).map_err(NativeRunSessionError::TelemetryPath)?;
        let logging_policy = build_logging_runtime_policy(run_plan, &telemetry_paths);
        let run_id = generate_run_id();
        let stage_timing_path = telemetry_paths.stage_timings_json.clone();
        let profile_summary_path = telemetry_paths.profile_summary_json.clone();
        let stage_timing_recorder =
            StageTimingRecorder::from_config(stage_timing_path.is_some(), profile_summary_path.is_some());
        let queue_size = usize::try_from(run_plan.diagnostics.log_queue_size)
            .map_err(|_| NativeRunSessionError::QueueSizeOutOfRange)?;
        let telemetry_session = TelemetryRunSession::new(
            run_plan.diagnostics.telemetry,
            telemetry_paths.stream_file.as_ref().map(PathBuf::from),
            queue_size,
            run_plan.diagnostics.lossy_logging,
            i64::from(run_plan.diagnostics.trace_event_cap),
            Some(run_id.clone()),
        )
        .map_err(NativeRunSessionError::Telemetry)?;
        Ok(Self {
            logging_policy,
            telemetry_session,
            stage_timing_recorder,
            stage_timing_path,
            profile_summary_path,
            run_id,
            run_start_time: Instant::now(),
            _shutdown_scope: shutdown_scope,
        })
    }

    pub fn record_stage_duration(&mut self, stage_name: &str, start_time: Instant) {
        if let Some(recorder) = self.stage_timing_recorder.as_mut() {
            recorder.add_stage_duration(stage_name.to_string(), start_time.elapsed().as_secs_f64());
        }
    }

    /// Write configured timing outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when a timing payload cannot be written.
    pub fn finish_timing(&mut self) -> Result<(), crate::TimingFileError> {
        let Some(recorder) = self.stage_timing_recorder.as_mut() else {
            return Ok(());
        };
        recorder.add_stage_duration("runner_total".to_string(), self.run_start_time.elapsed().as_secs_f64());
        let diagnostic = build_final_timing_outputs_write_started_diagnostic_payload(
            self.stage_timing_path.as_deref(),
            self.profile_summary_path.as_deref(),
            Some(&self.run_id),
        );
        if let Err(error) = emit_diagnostic_event(
            diagnostic.level,
            diagnostic.event_name,
            diagnostic.message,
            &serde_json::json!({
                "stage_timing_path": diagnostic.stage_timing_path,
                "profile_summary_path": diagnostic.profile_summary_path,
                "run_id": diagnostic.run_id,
            }),
        ) {
            tracing::warn!(target: "g.runtime", error = %error, "Failed to emit timing diagnostic event.");
        }
        recorder.write_final_timing_outputs(
            self.stage_timing_path.as_deref().map(Path::new),
            self.profile_summary_path.as_deref().map(Path::new),
            Some(self.run_id.clone()),
        )
    }
}

impl Drop for NativeRunSession {
    fn drop(&mut self) {
        let _ = crate::shutdown_logging_sinks();
    }
}
