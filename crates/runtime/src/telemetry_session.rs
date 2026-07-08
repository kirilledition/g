//! Runtime-owned telemetry session state and payload helpers.

mod cap;
mod close;
mod envelope;
mod serialization;
mod state;

pub use cap::{TelemetryCapAction, TelemetryEventCapState, TelemetryWriterCounterSnapshot};
pub use close::{
    TelemetryCloseEventPayload, TelemetryCloseMetadataPayload, TelemetryClosePlan, build_telemetry_close_event_payload,
    build_telemetry_close_metadata, plan_telemetry_close,
};
pub use envelope::{
    TelemetryEventEnvelope, build_current_telemetry_event_envelope, build_telemetry_event_envelope, generate_run_id,
};
pub use serialization::{serialize_telemetry_payload_json_line, serialize_telemetry_payload_json_text};
pub use state::{
    TelemetryEventEmissionPlan, TelemetryProgressEmissionPlan, TelemetryProgressThrottleState,
    TelemetryRunSessionState, TelemetryRunSessionWriterPlan, plan_telemetry_event_emission,
    plan_telemetry_progress_emission,
};

#[cfg(test)]
use std::path::Path;

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
