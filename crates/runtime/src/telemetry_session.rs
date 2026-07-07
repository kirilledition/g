//! Runtime-owned telemetry session state and payload helpers.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::telemetry_policy;
use serde_json::Value as JsonValue;
use uuid::Uuid;

const PROGRESS_TICK_EVENT_NAME: &str = "progress_tick";
const PROGRESS_TICK_EVENT_LEVEL: &str = "info";
const TELEMETRY_SESSION_CLOSED_EVENT_NAME: &str = "telemetry_session_closed";
const TELEMETRY_SESSION_CLOSED_EVENT_LEVEL: &str = "debug";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TelemetryCapAction {
    Write,
    Drop,
}

#[derive(Debug)]
pub struct TelemetryEventCapState {
    path: PathBuf,
    event_cap: Option<usize>,
    lossy: bool,
    written_event_count: AtomicUsize,
    dropped_event_count: AtomicUsize,
    exceeded: AtomicBool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryWriterCounterSnapshot {
    pub accepted_event_count: usize,
    pub written_event_count: usize,
    pub dropped_event_count: usize,
    pub cap_dropped_event_count: usize,
    pub queue_dropped_event_count: usize,
    pub event_cap_exceeded: bool,
    pub lossy: bool,
    pub event_cap: Option<usize>,
    pub finish_flush_duration_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryCloseMetadataPayload {
    pub writer_counters: TelemetryWriterCounterSnapshot,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryCloseEventPayload {
    pub event_name: String,
    pub level: String,
    pub writer_counters: TelemetryWriterCounterSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryEventEnvelope {
    pub schema_version: i64,
    pub run_id: String,
    pub timestamp: String,
    pub level: String,
    pub source: &'static str,
    pub target: &'static str,
    pub event: String,
    pub process_identifier: u32,
    pub thread_name: String,
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryClosePlan {
    pub should_close: bool,
    pub use_native_close_with_event: bool,
    pub should_emit_legacy_close_event: bool,
    pub legacy_close_event_name: String,
    pub legacy_close_event_level: String,
}

impl TelemetryEventCapState {
    #[must_use]
    pub fn new(path: &Path, event_cap: Option<usize>, lossy: bool) -> Self {
        Self {
            path: path.to_path_buf(),
            event_cap,
            lossy,
            written_event_count: AtomicUsize::new(0),
            dropped_event_count: AtomicUsize::new(0),
            exceeded: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub fn has_event_cap(&self) -> bool {
        self.event_cap.is_some()
    }

    pub fn record_uncapped_event_count(&self, event_count: usize) {
        if event_count > 0 {
            self.written_event_count.fetch_add(event_count, Ordering::Relaxed);
        }
    }

    /// Reserve one event under the configured trace cap.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when a lossless capped writer has already reached
    /// its configured event limit.
    pub fn reserve_event(&self) -> io::Result<TelemetryCapAction> {
        let Some(event_cap) = self.event_cap else {
            self.written_event_count.fetch_add(1, Ordering::Relaxed);
            return Ok(TelemetryCapAction::Write);
        };

        loop {
            let written_event_count = self.written_event_count.load(Ordering::Acquire);
            if written_event_count >= event_cap {
                self.mark_exceeded();
                if self.lossy {
                    self.dropped_event_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(TelemetryCapAction::Drop);
                }
                return Err(io::Error::other(self.cap_exceeded_error_message()));
            }
            if self
                .written_event_count
                .compare_exchange_weak(
                    written_event_count,
                    written_event_count + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(TelemetryCapAction::Write);
            }
        }
    }

    #[must_use]
    pub fn should_fail_for_cap_exceeded(&self) -> bool {
        self.exceeded.load(Ordering::Acquire) && !self.lossy
    }

    #[must_use]
    pub fn counter_snapshot(
        &self,
        queue_dropped_event_count: usize,
        finish_flush_duration_seconds: Option<f64>,
    ) -> TelemetryWriterCounterSnapshot {
        let accepted_event_count = self.written_event_count.load(Ordering::Acquire);
        let cap_dropped_event_count = self.dropped_event_count.load(Ordering::Acquire);
        TelemetryWriterCounterSnapshot {
            accepted_event_count,
            written_event_count: accepted_event_count.saturating_sub(queue_dropped_event_count),
            dropped_event_count: cap_dropped_event_count.saturating_add(queue_dropped_event_count),
            cap_dropped_event_count,
            queue_dropped_event_count,
            event_cap_exceeded: self.exceeded.load(Ordering::Acquire),
            lossy: self.lossy,
            event_cap: self.event_cap,
            finish_flush_duration_seconds,
        }
    }

    #[must_use]
    pub fn cap_exceeded_error_message(&self) -> String {
        let event_cap = self.event_cap.unwrap_or(0);
        format!(
            "Trace telemetry event cap exceeded at {event_cap} events for {}. \
             Increase --trace_event_cap or set --trace_event_cap 0 to disable the cap for intentional deep traces. \
             Use --log_lossy to drop events after the cap instead of failing.",
            self.path.display()
        )
    }

    #[must_use]
    pub fn cap_exceeded_drop_message(&self) -> String {
        let event_cap = self.event_cap.unwrap_or(0);
        format!(
            "Trace telemetry event cap reached at {event_cap} events for {}; dropping additional trace events because log_lossy is enabled.",
            self.path.display()
        )
    }

    fn mark_exceeded(&self) {
        if !self.exceeded.swap(true, Ordering::AcqRel) && self.lossy {
            tracing::warn!(
                target: "g.logging",
                g_event = "native_telemetry_event_cap_exceeded",
                event_cap = self.event_cap.unwrap_or(0),
                lossy = self.lossy,
                path = %self.path.display(),
                message = %self.cap_exceeded_drop_message(),
                "Tracing writer reached event cap and started dropping events."
            );
        }
    }
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
        plan_telemetry_event_emission(self.policy.enabled, has_native_telemetry_session)
    }

    #[must_use]
    pub fn should_emit_progress_at(&mut self, processed_chunk_count: i64, current_time_seconds: f64) -> bool {
        self.progress_throttle.should_emit_progress_at(processed_chunk_count, current_time_seconds)
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

impl TelemetryWriterCounterSnapshot {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            accepted_event_count: 0,
            written_event_count: 0,
            dropped_event_count: 0,
            cap_dropped_event_count: 0,
            queue_dropped_event_count: 0,
            event_cap_exceeded: false,
            lossy: true,
            event_cap: None,
            finish_flush_duration_seconds: None,
        }
    }
}

#[must_use]
pub fn build_telemetry_close_metadata(
    writer_counters: TelemetryWriterCounterSnapshot,
) -> TelemetryCloseMetadataPayload {
    TelemetryCloseMetadataPayload { writer_counters }
}

#[must_use]
pub fn build_telemetry_close_event_payload(
    writer_counters: TelemetryWriterCounterSnapshot,
) -> TelemetryCloseEventPayload {
    TelemetryCloseEventPayload {
        event_name: TELEMETRY_SESSION_CLOSED_EVENT_NAME.to_string(),
        level: TELEMETRY_SESSION_CLOSED_EVENT_LEVEL.to_string(),
        writer_counters,
    }
}

#[must_use]
pub fn build_telemetry_event_envelope(
    run_id: &str,
    event: &str,
    level: &str,
    timestamp: &str,
    process_identifier: u32,
    thread_name: &str,
) -> TelemetryEventEnvelope {
    TelemetryEventEnvelope {
        schema_version: 1,
        run_id: run_id.to_string(),
        timestamp: timestamp.to_string(),
        level: level.to_uppercase(),
        source: "python",
        target: "g.engine.telemetry",
        event: event.to_string(),
        process_identifier,
        thread_name: thread_name.to_string(),
    }
}

#[must_use]
pub fn build_current_telemetry_event_envelope(
    run_id: &str,
    event: &str,
    level: &str,
    thread_name: &str,
) -> TelemetryEventEnvelope {
    build_telemetry_event_envelope(
        run_id,
        event,
        level,
        &current_telemetry_timestamp(),
        std::process::id(),
        thread_name,
    )
}

/// Serialize one telemetry payload as stable JSON text.
///
/// # Errors
///
/// Returns a serialization error when the payload cannot be rendered as JSON.
pub fn serialize_telemetry_payload_json_text(payload: &JsonValue) -> Result<String, serde_json::Error> {
    serde_json::to_string(payload)
}

/// Serialize one telemetry payload as a JSONL record.
///
/// # Errors
///
/// Returns a serialization error when the payload cannot be rendered as JSON.
pub fn serialize_telemetry_payload_json_line(payload: &JsonValue) -> Result<String, serde_json::Error> {
    serialize_telemetry_payload_json_text(payload).map(|json_text| format!("{json_text}\n"))
}

#[must_use]
pub const fn plan_telemetry_event_emission(
    telemetry_enabled: bool,
    has_native_telemetry_session: bool,
) -> TelemetryEventEmissionPlan {
    TelemetryEventEmissionPlan { should_emit: telemetry_enabled && has_native_telemetry_session }
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

#[must_use]
pub fn plan_telemetry_close(has_telemetry_session: bool, is_native_telemetry_session: bool) -> TelemetryClosePlan {
    TelemetryClosePlan {
        should_close: has_telemetry_session && is_native_telemetry_session,
        use_native_close_with_event: has_telemetry_session && is_native_telemetry_session,
        should_emit_legacy_close_event: false,
        legacy_close_event_name: TELEMETRY_SESSION_CLOSED_EVENT_NAME.to_string(),
        legacy_close_event_level: TELEMETRY_SESSION_CLOSED_EVENT_LEVEL.to_string(),
    }
}

#[must_use]
pub fn generate_run_id() -> String {
    format!("{:032x}", Uuid::new_v4().as_u128())
}

fn current_telemetry_timestamp() -> String {
    let elapsed_seconds = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |duration| duration.as_secs_f64());
    telemetry_policy::format_timestamp(elapsed_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_lossy_state_drops_after_limit_and_reports_counters() {
        let state = TelemetryEventCapState::new(Path::new("events.jsonl"), Some(2), true);

        assert_eq!(state.reserve_event().unwrap(), TelemetryCapAction::Write);
        assert_eq!(state.reserve_event().unwrap(), TelemetryCapAction::Write);
        assert_eq!(state.reserve_event().unwrap(), TelemetryCapAction::Drop);

        let counters = state.counter_snapshot(0, Some(0.5));
        assert_eq!(counters.accepted_event_count, 2);
        assert_eq!(counters.written_event_count, 2);
        assert_eq!(counters.dropped_event_count, 1);
        assert_eq!(counters.cap_dropped_event_count, 1);
        assert!(counters.event_cap_exceeded);
        assert_eq!(counters.finish_flush_duration_seconds, Some(0.5));
    }

    #[test]
    fn capped_lossless_state_errors_after_limit() {
        let state = TelemetryEventCapState::new(Path::new("events.jsonl"), Some(1), false);

        assert_eq!(state.reserve_event().unwrap(), TelemetryCapAction::Write);
        let error = state.reserve_event().unwrap_err();

        assert!(error.to_string().contains("Trace telemetry event cap exceeded at 1 events"));
        assert!(state.should_fail_for_cap_exceeded());
    }

    #[test]
    fn builds_python_telemetry_event_envelope() {
        let envelope = build_telemetry_event_envelope("run-1", "started", "info", "2026-01-01T00:00:00Z", 42, "main");

        assert_eq!(envelope.schema_version, 1);
        assert_eq!(envelope.run_id, "run-1");
        assert_eq!(envelope.level, "INFO");
        assert_eq!(envelope.source, "python");
        assert_eq!(envelope.target, "g.engine.telemetry");
        assert_eq!(envelope.process_identifier, 42);
    }

    #[test]
    fn builds_current_telemetry_event_envelope() {
        let envelope = build_current_telemetry_event_envelope("run-1", "started", "debug", "main");

        assert_eq!(envelope.run_id, "run-1");
        assert_eq!(envelope.level, "DEBUG");
        assert_eq!(envelope.process_identifier, std::process::id());
        assert_eq!(envelope.thread_name, "main");
        assert!(envelope.timestamp.contains('T'));
        assert!(envelope.timestamp.ends_with('Z'));
    }

    #[test]
    fn serializes_telemetry_payload_as_json_line() {
        let payload = serde_json::json!({
            "event": "started",
            "run_id": "run-1",
            "nested": {"values": [1, 2, true, null]},
        });

        let json_line = serialize_telemetry_payload_json_line(&payload).unwrap();

        assert_eq!(json_line, "{\"event\":\"started\",\"nested\":{\"values\":[1,2,true,null]},\"run_id\":\"run-1\"}\n");
        assert!(json_line.ends_with('\n'));
    }

    #[test]
    fn progress_throttle_emits_first_event() {
        let mut state = TelemetryProgressThrottleState::new(999.0, 10);

        assert!(state.should_emit_progress_at(1, 0.0));
    }

    #[test]
    fn progress_throttle_suppresses_until_time_or_chunk_threshold() {
        let mut state = TelemetryProgressThrottleState::new(5.0, 10);

        assert!(state.should_emit_progress_at(1, 10.0));
        assert!(!state.should_emit_progress_at(2, 11.0));
        assert!(state.should_emit_progress_at(11, 11.5));
        assert!(!state.should_emit_progress_at(12, 12.0));
        assert!(state.should_emit_progress_at(12, 16.5));
    }

    #[test]
    fn plans_telemetry_event_and_progress_emission() {
        assert_eq!(plan_telemetry_event_emission(true, true), TelemetryEventEmissionPlan { should_emit: true });
        assert_eq!(plan_telemetry_event_emission(false, true), TelemetryEventEmissionPlan { should_emit: false });
        assert_eq!(plan_telemetry_event_emission(true, false), TelemetryEventEmissionPlan { should_emit: false });
        assert_eq!(
            plan_telemetry_progress_emission(true, true, true),
            TelemetryProgressEmissionPlan {
                should_emit: true,
                event_name: "progress_tick".to_string(),
                level: "info".to_string(),
            },
        );
        assert_eq!(
            plan_telemetry_progress_emission(true, true, false),
            TelemetryProgressEmissionPlan {
                should_emit: false,
                event_name: "progress_tick".to_string(),
                level: "info".to_string(),
            },
        );
    }

    #[test]
    fn plans_telemetry_close_paths() {
        assert_eq!(
            plan_telemetry_close(false, false),
            TelemetryClosePlan {
                should_close: false,
                use_native_close_with_event: false,
                should_emit_legacy_close_event: false,
                legacy_close_event_name: "telemetry_session_closed".to_string(),
                legacy_close_event_level: "debug".to_string(),
            },
        );
        assert_eq!(
            plan_telemetry_close(true, true),
            TelemetryClosePlan {
                should_close: true,
                use_native_close_with_event: true,
                should_emit_legacy_close_event: false,
                legacy_close_event_name: "telemetry_session_closed".to_string(),
                legacy_close_event_level: "debug".to_string(),
            },
        );
        assert_eq!(
            plan_telemetry_close(true, false),
            TelemetryClosePlan {
                should_close: false,
                use_native_close_with_event: false,
                should_emit_legacy_close_event: false,
                legacy_close_event_name: "telemetry_session_closed".to_string(),
                legacy_close_event_level: "debug".to_string(),
            },
        );
    }

    #[test]
    fn builds_telemetry_close_metadata_payload() {
        let writer_counters = TelemetryWriterCounterSnapshot::empty();

        let metadata = build_telemetry_close_metadata(writer_counters.clone());

        assert_eq!(metadata.writer_counters, writer_counters);
    }

    #[test]
    fn builds_telemetry_close_event_payload() {
        let writer_counters = TelemetryWriterCounterSnapshot::empty();

        let payload = build_telemetry_close_event_payload(writer_counters.clone());

        assert_eq!(payload.event_name, "telemetry_session_closed");
        assert_eq!(payload.level, "debug");
        assert_eq!(payload.writer_counters, writer_counters);
    }

    #[test]
    fn generates_python_compatible_run_identifier() {
        let run_id = generate_run_id();

        assert_eq!(run_id.len(), 32);
        assert!(run_id.chars().all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
    }

    #[test]
    fn telemetry_run_session_state_owns_policy_writer_plan_and_progress() {
        let mut state = TelemetryRunSessionState::new("trace", 2, 5.0, 10, Some("run-1".to_string()));

        assert_eq!(state.run_id(), "run-1");
        assert!(state.enabled());
        assert!(state.profile_enabled());
        assert_eq!(state.event_cap(), Some(2));
        assert_eq!(
            state.writer_plan(true),
            TelemetryRunSessionWriterPlan { should_open_writer: true, event_cap: Some(2) },
        );
        assert_eq!(
            state.writer_plan(false),
            TelemetryRunSessionWriterPlan { should_open_writer: false, event_cap: Some(2) },
        );
        assert_eq!(state.plan_event_emission(true), TelemetryEventEmissionPlan { should_emit: true });
        assert_eq!(
            state.plan_progress_emission_at(1, 0.0, true),
            TelemetryProgressEmissionPlan {
                should_emit: true,
                event_name: "progress_tick".to_string(),
                level: "info".to_string(),
            },
        );
        assert_eq!(
            state.plan_progress_emission_at(2, 1.0, true),
            TelemetryProgressEmissionPlan {
                should_emit: false,
                event_name: "progress_tick".to_string(),
                level: "info".to_string(),
            },
        );
    }
}
