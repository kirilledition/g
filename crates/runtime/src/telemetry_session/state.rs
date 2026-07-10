use crate::telemetry_policy;

use super::generate_run_id;

const PROGRESS_TICK_EVENT_NAME: &str = "progress_tick";
const PROGRESS_TICK_EVENT_LEVEL: &str = "info";

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryProgressThrottleState {
    progress_interval_seconds: f64,
    progress_interval_chunks: i64,
    last_progress_time_seconds: Option<f64>,
    last_progress_chunk_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryRunSessionState {
    run_id: String,
    policy: telemetry_policy::TelemetrySessionPolicyPayload,
    progress_throttle: TelemetryProgressThrottleState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryRunSessionWriterPlan {
    pub should_open_writer: bool,
    pub event_cap: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryEventEmissionPlan {
    pub should_emit: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryProgressEmissionPlan {
    pub should_emit: bool,
    pub event_name: String,
    pub level: String,
}

impl TelemetryProgressThrottleState {
    #[must_use]
    pub fn new(progress_interval_seconds: f64, progress_interval_chunks: i64) -> Self {
        Self {
            progress_interval_seconds,
            progress_interval_chunks,
            last_progress_time_seconds: None,
            last_progress_chunk_count: 0,
        }
    }

    #[must_use]
    pub fn should_emit_progress_at(&mut self, processed_chunk_count: i64, current_time_seconds: f64) -> bool {
        if let Some(last_progress_time_seconds) = self.last_progress_time_seconds {
            let elapsed_seconds = current_time_seconds - last_progress_time_seconds;
            let elapsed_chunks = processed_chunk_count - self.last_progress_chunk_count;
            if elapsed_seconds < self.progress_interval_seconds && elapsed_chunks < self.progress_interval_chunks {
                return false;
            }
        }

        self.last_progress_time_seconds = Some(current_time_seconds);
        self.last_progress_chunk_count = processed_chunk_count;
        true
    }
}

impl TelemetryRunSessionState {
    #[must_use]
    pub fn new(
        telemetry_mode: telemetry_policy::TelemetryMode,
        trace_event_cap: i64,
        progress_interval_seconds: f64,
        progress_interval_chunks: i64,
        run_id: Option<String>,
    ) -> Self {
        Self {
            run_id: run_id.unwrap_or_else(generate_run_id),
            policy: telemetry_policy::resolve_telemetry_session_policy(telemetry_mode, trace_event_cap),
            progress_throttle: TelemetryProgressThrottleState::new(progress_interval_seconds, progress_interval_chunks),
        }
    }

    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.policy.enabled
    }

    #[must_use]
    pub const fn profile_enabled(&self) -> bool {
        self.policy.profile_enabled
    }

    #[must_use]
    pub const fn event_cap(&self) -> Option<i64> {
        self.policy.event_cap
    }

    #[must_use]
    pub fn writer_plan(&self, stream_file_configured: bool) -> TelemetryRunSessionWriterPlan {
        TelemetryRunSessionWriterPlan {
            should_open_writer: self.policy.enabled && stream_file_configured,
            event_cap: self.policy.event_cap,
        }
    }

    #[must_use]
    pub fn plan_event_emission(&self, has_native_telemetry_session: bool) -> TelemetryEventEmissionPlan {
        TelemetryEventEmissionPlan { should_emit: self.policy.enabled && has_native_telemetry_session }
    }

    #[must_use]
    pub fn plan_progress_emission_at(
        &mut self,
        processed_chunk_count: i64,
        current_time_seconds: f64,
        has_native_telemetry_session: bool,
    ) -> TelemetryProgressEmissionPlan {
        let should_emit_progress = self.policy.enabled
            && self.progress_throttle.should_emit_progress_at(processed_chunk_count, current_time_seconds);
        plan_telemetry_progress_emission(self.policy.enabled, has_native_telemetry_session, should_emit_progress)
    }
}

#[must_use]
pub fn plan_telemetry_progress_emission(
    telemetry_enabled: bool,
    has_native_telemetry_session: bool,
    should_emit_progress: bool,
) -> TelemetryProgressEmissionPlan {
    TelemetryProgressEmissionPlan {
        should_emit: telemetry_enabled && has_native_telemetry_session && should_emit_progress,
        event_name: PROGRESS_TICK_EVENT_NAME.to_string(),
        level: PROGRESS_TICK_EVENT_LEVEL.to_string(),
    }
}
