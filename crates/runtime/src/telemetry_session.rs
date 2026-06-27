//! Runtime-owned telemetry session state and payload helpers.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use uuid::Uuid;

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
pub fn generate_run_id() -> String {
    format!("{:032x}", Uuid::new_v4().as_u128())
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
    fn generates_python_compatible_run_identifier() {
        let run_id = generate_run_id();

        assert_eq!(run_id.len(), 32);
        assert!(run_id.chars().all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
    }
}
