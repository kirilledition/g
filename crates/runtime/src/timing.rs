//! Native stage timing recorder state and aggregate bookkeeping.

mod final_outputs;
mod payloads;
mod queue_backpressure;
mod recorder;
mod state;
mod transfer_metadata;

pub use final_outputs::{
    FinalTimingOutputContext, FinalTimingOutputsWriteResultPayload, FinalTimingOutputsWriteStartedDiagnosticPayload,
    StageTimingRecorderPlan, TimingFileError, TimingFileWritePlan,
    build_final_timing_outputs_write_started_diagnostic_payload, plan_stage_timing_recorder, plan_timing_file_write,
    resolve_final_timing_output_context, serialize_final_timing_outputs_write_started_diagnostic_fields_json,
    should_collect_exact_stage_timings, write_profile_summary_payload, write_stage_timing_snapshot_payload,
};
pub use payloads::{
    ChunkStageSummary, ChunkStageTiming, NullLogisticDiagnosticValue, NullLogisticSummary, NumericDiagnosticValue,
    ProfileSummaryPayload, StageTimingSnapshotPayload,
};
pub use queue_backpressure::{QueueBackpressureAccumulator, QueueBackpressureKey, QueueBackpressureSnapshot};
pub use recorder::StageTimingRecorder;
pub use state::StageTimingState;
pub use transfer_metadata::{
    TransferMetadataAccumulator, TransferMetadataError, TransferMetadataKey, TransferMetadataObservation,
    TransferMetadataSnapshot, build_transfer_metadata_observation,
};

#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_stage_queue_and_transfer_timing_state() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("load".to_string(), 0.25);
        state.add_chunk_stage_duration(ChunkStageTiming {
            chunk_identifier: 1,
            chromosome: "22".to_string(),
            variant_start_index: 0,
            variant_stop_index: 4,
            variant_count: 4,
            stage_name: "load".to_string(),
            duration_seconds: 0.75,
        });
        state.add_queue_backpressure_observation(
            QueueBackpressureKey { queue_name: "writer".to_string(), operation_name: "send".to_string() },
            3,
            8,
            0.4,
            0.1,
        );
        state.add_transfer_metadata(
            TransferMetadataKey {
                transfer_name: "host_to_device".to_string(),
                array_role: "genotype".to_string(),
                dtype_name: "float32".to_string(),
                dimension_count: 2,
            },
            128,
            32,
        );

        assert_eq!(state.stage_counts["load"], 2);
        assert!((state.stage_totals_seconds["load"] - 1.0).abs() < f64::EPSILON);
        assert_eq!(state.chunk_stage_timings.len(), 1);
        assert_eq!(state.queue_backpressure.values().next().unwrap().max_depth, 3);
        assert_eq!(state.transfer_metadata.values().next().unwrap().total_bytes, 128);
    }

    #[test]
    fn builds_profile_summary_payload_from_timing_state() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("native_engine_delivery".to_string(), 2.0);
        state.add_stage_duration("output_write".to_string(), 4.0);
        state.set_native_bgen_profile(BTreeMap::from([
            ("variant_decode_count".to_string(), 8),
            ("selected_sample_count".to_string(), 10),
        ]));
        state.add_chunk_stage_duration(ChunkStageTiming {
            chunk_identifier: 0,
            chromosome: "22".to_string(),
            variant_start_index: 0,
            variant_stop_index: 8,
            variant_count: 8,
            stage_name: "python_callback".to_string(),
            duration_seconds: 0.5,
        });
        state.add_binary_chunk_diagnostics(BTreeMap::from([
            ("score_test_candidate_count".to_string(), NumericDiagnosticValue::Integer(2)),
            ("firth_iteration_min".to_string(), NumericDiagnosticValue::Integer(4)),
            ("firth_iteration_max".to_string(), NumericDiagnosticValue::Integer(8)),
        ]));

        let summary = state.build_profile_summary(Some("run-1".to_string()));

        assert_eq!(summary.schema_version, 1);
        assert_eq!(summary.run_id.as_deref(), Some("run-1"));
        assert!((summary.derived_metrics["native_variant_decode_per_second"] - 4.0).abs() < f64::EPSILON);
        assert!((summary.derived_metrics["output_variant_rows_per_second"] - 2.0).abs() < f64::EPSILON);
        assert_eq!(summary.chunk_stage_summary["python_callback"].count, 1);
        assert_eq!(summary.binary_chunk_summary["chunk_count"], NumericDiagnosticValue::Integer(1));
        assert_eq!(
            summary.binary_chunk_summary["score_test_candidate_count_total"],
            NumericDiagnosticValue::Float(2.0)
        );
        assert_eq!(summary.binary_chunk_summary["firth_iteration_max"], NumericDiagnosticValue::Float(8.0));
    }

    #[test]
    fn builds_stage_timing_snapshot_payload_with_derived_metrics() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("host_to_device_transfer".to_string(), 2.0);
        state.add_transfer_metadata(
            TransferMetadataKey {
                transfer_name: "host_to_device_transfer".to_string(),
                array_role: "genotype_matrix".to_string(),
                dtype_name: "float32".to_string(),
                dimension_count: 2,
            },
            96,
            24,
        );

        let payload = state.build_stage_timing_snapshot_payload();

        assert_eq!(payload.stage_counts["host_to_device_transfer"], 1);
        assert_eq!(payload.transfer_metadata.len(), 1);
        assert!((payload.derived_metrics["host_to_device_transfer_bytes_per_second"] - 48.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stage_timing_recorder_owns_policy_and_state() {
        assert!(StageTimingRecorder::from_config(false, false).is_none());
        let mut recorder = StageTimingRecorder::from_config(true, false).expect("recorder should be created");

        assert!(recorder.exact_stage_timings());
        assert!(recorder.should_collect_exact_stage_timings());
        assert!(recorder.should_write_timing_file(true));
        assert!(!recorder.should_write_timing_file(false));

        recorder.add_stage_duration("jax_compute".to_string(), 2.0);
        recorder.set_native_bgen_profile(BTreeMap::from([("variant_decode_count".to_string(), 8)]));

        let payload = recorder.build_stage_timing_snapshot_payload();
        assert_eq!(payload.stage_counts["jax_compute"], 1);
        assert!((payload.derived_metrics["jax_variant_compute_per_second"] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn builds_transfer_metadata_observation_from_shape_dimensions() {
        let observation =
            build_transfer_metadata_observation("host_to_device_transfer", "genotype_matrix", "float32", &[4, 8], 4)
                .unwrap();

        assert_eq!(
            observation,
            TransferMetadataObservation {
                key: TransferMetadataKey {
                    transfer_name: "host_to_device_transfer".to_string(),
                    array_role: "genotype_matrix".to_string(),
                    dtype_name: "float32".to_string(),
                    dimension_count: 2,
                },
                byte_count: 128,
                element_count: 32,
            },
        );

        let scalar_observation =
            build_transfer_metadata_observation("device_to_host_materialization", "beta", "float64", &[], 8).unwrap();
        assert_eq!(scalar_observation.key.dimension_count, 0);
        assert_eq!(scalar_observation.byte_count, 8);
        assert_eq!(scalar_observation.element_count, 1);
    }

    #[test]
    fn rejects_invalid_transfer_metadata_shape_inputs() {
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[1, -1], 4).unwrap_err(),
            TransferMetadataError::NegativeDimension { dimension: -1 },
        );
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[1], 0).unwrap_err(),
            TransferMetadataError::NonPositiveItemSize { item_size: 0 },
        );
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[i64::MAX, 2], 4).unwrap_err(),
            TransferMetadataError::ElementCountOverflow,
        );
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[i64::MAX], 2).unwrap_err(),
            TransferMetadataError::ByteCountOverflow,
        );
    }

    #[test]
    fn records_transfer_metadata_from_shape_dimensions() {
        let mut state = StageTimingState::default();
        state
            .add_transfer_metadata_for_shape("host_to_device_transfer", "genotype_matrix", "float32", &[4, 8], 4)
            .unwrap();

        let payload = state.build_stage_timing_snapshot_payload();

        assert_eq!(
            payload.transfer_metadata,
            vec![TransferMetadataSnapshot {
                transfer_name: "host_to_device_transfer".to_string(),
                array_role: "genotype_matrix".to_string(),
                dtype_name: "float32".to_string(),
                dimension_count: 2,
                observation_count: 1,
                total_bytes: 128,
                max_bytes: 128,
                total_elements: 32,
            }],
        );
    }

    #[test]
    fn writes_stage_timing_and_profile_summary_payloads() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("native_engine_delivery".to_string(), 2.0);
        state.set_native_bgen_profile(BTreeMap::from([("variant_decode_count".to_string(), 8)]));
        let directory_path = create_test_directory("writes_stage_timing_and_profile_summary_payloads");
        let stage_timing_path = directory_path.join("nested").join("stage-timings.json");
        let profile_summary_path = directory_path.join("profile.summary.json");

        write_stage_timing_snapshot_payload(&stage_timing_path, &state.build_stage_timing_snapshot_payload())
            .expect("stage timing payload should be written");
        write_profile_summary_payload(&profile_summary_path, &state.build_profile_summary(Some("run-1".to_string())))
            .expect("profile summary payload should be written");

        let stage_timing_text =
            std::fs::read_to_string(&stage_timing_path).expect("stage timing payload should be readable");
        let profile_summary_text =
            std::fs::read_to_string(&profile_summary_path).expect("profile summary payload should be readable");
        assert!(stage_timing_text.ends_with('\n'));
        assert!(profile_summary_text.ends_with('\n'));
        let stage_timing_payload: serde_json::Value =
            serde_json::from_str(&stage_timing_text).expect("stage timing payload should be valid JSON");
        let profile_summary_payload: serde_json::Value =
            serde_json::from_str(&profile_summary_text).expect("profile summary payload should be valid JSON");
        assert_eq!(stage_timing_payload["derived_metrics"]["native_variant_decode_per_second"], serde_json::json!(4.0));
        assert_eq!(profile_summary_payload["run_id"], serde_json::json!("run-1"));
        assert_eq!(
            profile_summary_payload["derived_metrics"]["native_variant_decode_per_second"],
            serde_json::json!(4.0)
        );

        std::fs::remove_dir_all(directory_path).expect("test timing directory should be removed");
    }

    #[test]
    fn stage_timing_recorder_writes_final_timing_outputs() {
        let mut recorder = StageTimingRecorder::new(false);
        recorder.add_stage_duration("native_engine_delivery".to_string(), 2.0);
        recorder.set_native_bgen_profile(BTreeMap::from([("variant_decode_count".to_string(), 8)]));
        let directory_path = create_test_directory("stage_timing_recorder_writes_final_timing_outputs");
        let stage_timing_path = directory_path.join("nested").join("stage-timings.json");
        let profile_summary_path = directory_path.join("profile.summary.json");

        let result = recorder
            .write_final_timing_outputs(
                Some(stage_timing_path.as_path()),
                Some(profile_summary_path.as_path()),
                Some("run-1".to_string()),
            )
            .expect("final timing outputs should be written");

        assert_eq!(
            result,
            FinalTimingOutputsWriteResultPayload { wrote_stage_timing_snapshot: true, wrote_profile_summary: true },
        );
        assert_eq!(
            recorder.write_final_timing_outputs(None, None, Some("run-1".to_string())).unwrap(),
            FinalTimingOutputsWriteResultPayload { wrote_stage_timing_snapshot: false, wrote_profile_summary: false },
        );
        let profile_summary_text =
            std::fs::read_to_string(&profile_summary_path).expect("profile summary should be readable");
        let profile_summary_payload: serde_json::Value =
            serde_json::from_str(&profile_summary_text).expect("profile summary should be JSON");
        assert_eq!(profile_summary_payload["run_id"], serde_json::json!("run-1"));
        assert!(stage_timing_path.exists());

        std::fs::remove_dir_all(directory_path).expect("test timing directory should be removed");
    }

    #[test]
    fn resolves_exact_stage_timing_collection_policy() {
        assert!(should_collect_exact_stage_timings(true));
        assert!(!should_collect_exact_stage_timings(false));
    }

    #[test]
    fn plans_stage_timing_recorder_creation() {
        assert_eq!(
            plan_stage_timing_recorder(false, false),
            StageTimingRecorderPlan { should_create: false, exact_stage_timings: false },
        );
        assert_eq!(
            plan_stage_timing_recorder(false, true),
            StageTimingRecorderPlan { should_create: true, exact_stage_timings: false },
        );
        assert_eq!(
            plan_stage_timing_recorder(true, false),
            StageTimingRecorderPlan { should_create: true, exact_stage_timings: true },
        );
    }

    #[test]
    fn plans_timing_file_writes() {
        assert_eq!(plan_timing_file_write(true, true), TimingFileWritePlan { should_write: true },);
        assert_eq!(plan_timing_file_write(false, true), TimingFileWritePlan { should_write: false },);
        assert_eq!(plan_timing_file_write(true, false), TimingFileWritePlan { should_write: false },);
    }

    #[test]
    fn builds_final_timing_outputs_write_started_diagnostic_payload() {
        assert_eq!(
            build_final_timing_outputs_write_started_diagnostic_payload(
                Some("timings.json"),
                Some("profile.summary.json"),
                Some("run-1"),
            ),
            FinalTimingOutputsWriteStartedDiagnosticPayload {
                level: "debug",
                event_name: "runner_final_timing_outputs_write_started",
                message: "Writing final timing outputs.",
                stage_timing_path: Some("timings.json".to_string()),
                profile_summary_path: Some("profile.summary.json".to_string()),
                run_id: Some("run-1".to_string()),
            },
        );
    }

    #[test]
    fn serializes_final_timing_outputs_write_started_diagnostic_fields_json() {
        let payload = build_final_timing_outputs_write_started_diagnostic_payload(
            Some("timings.json"),
            Some("profile.summary.json"),
            Some("run-1"),
        );
        let fields_text = serialize_final_timing_outputs_write_started_diagnostic_fields_json(&payload)
            .expect("diagnostic fields should serialize");
        let fields: serde_json::Value =
            serde_json::from_str(&fields_text).expect("diagnostic fields should be valid JSON");

        assert_eq!(
            fields,
            serde_json::json!({
                "stage_timing_path": "timings.json",
                "profile_summary_path": "profile.summary.json",
                "run_id": "run-1",
            }),
        );
    }

    #[test]
    fn resolves_final_timing_output_context() {
        assert_eq!(
            resolve_final_timing_output_context(Some("diagnostics/timings.json"), None, None, None, false, false,),
            FinalTimingOutputContext {
                stage_timing_path: Some("diagnostics/timings.json".to_string()),
                profile_summary_path: None,
                run_id: None,
                force_stage_timing_recorder: false,
            },
        );
        assert_eq!(
            resolve_final_timing_output_context(
                Some("diagnostics/timings.json"),
                Some("telemetry/timings.json"),
                Some("telemetry/profile.summary.json"),
                Some("run-1"),
                true,
                true,
            ),
            FinalTimingOutputContext {
                stage_timing_path: Some("telemetry/timings.json".to_string()),
                profile_summary_path: Some("telemetry/profile.summary.json".to_string()),
                run_id: Some("run-1".to_string()),
                force_stage_timing_recorder: true,
            },
        );
    }

    fn create_test_directory(test_name: &str) -> PathBuf {
        let timestamp_nanoseconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let directory_path =
            std::env::temp_dir().join(format!("g-runtime-{test_name}-{}-{timestamp_nanoseconds}", std::process::id()));
        std::fs::create_dir_all(&directory_path).expect("test timing directory should be created");
        directory_path
    }
}
