//! Runtime-owned run lifecycle event payloads and rendering policy.

mod diagnostics;
mod lifecycle;
mod native_cli_diagnostics;
mod native_dispatch_diagnostics;
mod pipeline_diagnostics;
mod runner_diagnostics;
mod telemetry;

use diagnostics::{
    boolean_diagnostic_field, integer_diagnostic_field, optional_integer_diagnostic_field, text_diagnostic_field,
};

pub use diagnostics::{
    RunDiagnosticEventPayload, RunDiagnosticFieldPayload, RunDiagnosticFieldValue, serialize_run_diagnostic_fields_json,
};
pub use lifecycle::{
    RunArtifactPayload, RunArtifactTelemetryFields, RunArtifactsPayload, RunCompletedEventPayload,
    RunCompletedTelemetryFields, RunFailedEventPayload, RunFailedTelemetryFields, RunInterruptedEventPayload,
    RunInterruptedTelemetryFields, RunTelemetryStringField, attach_run_metadata_to_artifacts,
    build_artifact_telemetry_fields, build_run_completed_event_from_artifacts, build_run_completed_telemetry_fields,
    build_run_failed_event_payload, build_run_failed_telemetry_fields, build_run_interrupted_event_payload,
    build_run_interrupted_telemetry_fields, flatten_run_artifact_payloads, render_artifact_lines,
    render_run_completed_lines, render_run_failed_lines, render_run_interrupted_lines,
};
pub use native_cli_diagnostics::{
    build_native_cli_completed_line_diagnostic_payload, build_native_cli_failed_line_diagnostic_payload,
    build_native_cli_interrupted_line_diagnostic_payload, build_native_cli_stderr_diagnostic_payload,
    build_native_cli_stdout_diagnostic_payload,
};
pub use native_dispatch_diagnostics::{
    build_native_dispatch_bgen_engine_constructing_diagnostic_payload,
    build_native_dispatch_callback_drain_started_diagnostic_payload,
    build_native_dispatch_delivery_failed_diagnostic_payload,
    build_native_dispatch_delivery_finished_diagnostic_payload,
    build_native_dispatch_delivery_interrupted_diagnostic_payload,
    build_native_dispatch_delivery_started_diagnostic_payload,
    build_native_dispatch_pipeline_finished_diagnostic_payload,
    build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload,
    build_native_dispatch_writer_session_finish_started_diagnostic_payload,
    build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload,
    build_native_dispatch_writer_sessions_finish_started_diagnostic_payload,
    build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload,
};
pub use pipeline_diagnostics::{
    build_pipeline_bgen_engine_open_started_diagnostic_payload, build_pipeline_bgen_engine_opened_diagnostic_payload,
    build_pipeline_gpu_genotype_format_resolved_diagnostic_payload,
    build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload,
    build_pipeline_grouped_per_phenotype_started_diagnostic_payload,
    build_pipeline_grouped_union_delivery_selected_diagnostic_payload,
    build_pipeline_multi_group_preflight_completed_diagnostic_payload,
    build_pipeline_multi_group_preflight_started_diagnostic_payload,
    build_pipeline_multi_phenotype_sample_summary_diagnostic_payload,
    build_pipeline_multi_trait_input_aligned_diagnostic_payload,
    build_pipeline_multi_trait_input_load_started_diagnostic_payload,
    build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload,
    build_pipeline_multi_trait_started_diagnostic_payload,
    build_pipeline_output_resume_committed_chunks_diagnostic_payload,
    build_pipeline_output_writer_sessions_create_started_diagnostic_payload,
    build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload,
    build_pipeline_single_trait_input_aligned_diagnostic_payload,
    build_pipeline_single_trait_input_load_started_diagnostic_payload,
    build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload,
    build_pipeline_single_trait_preflight_completed_diagnostic_payload,
    build_pipeline_single_trait_preflight_started_diagnostic_payload,
    build_pipeline_single_trait_started_diagnostic_payload,
};
pub use runner_diagnostics::{
    build_runner_binary_engine_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_build_started_diagnostic_payload,
    build_runner_execution_plan_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_finalization_started_diagnostic_payload,
    build_runner_execution_plan_prepared_diagnostic_payload,
    build_runner_jax_runtime_configuration_started_diagnostic_payload,
    build_runner_linear_engine_dispatch_started_diagnostic_payload,
    build_runner_metadata_artifacts_finalized_diagnostic_payload,
    build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload,
    build_runner_multi_phenotype_dispatch_started_diagnostic_payload,
    build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload,
    build_runner_run_completed_diagnostic_payload, build_runner_run_failed_diagnostic_payload,
    build_runner_run_interrupted_diagnostic_payload, build_runner_run_started_diagnostic_payload,
    build_runner_single_phenotype_dispatch_started_diagnostic_payload,
};
pub use telemetry::{
    AssociationBackendSelectedTelemetryFields, BgenEngineOpenedTelemetryFields, EffectiveConfigWrittenTelemetryFields,
    ExecutionPlanPreparedTelemetryFields, GpuGenotypeFormatResolvedTelemetryFields,
    MultiPhenotypePreflightCompletedTelemetryFields, MultiPhenotypeSampleSummaryTelemetryFields,
    MultiPhenotypeWriterFinishedTelemetryFields, PhenotypeWriterFinishedTelemetryFields,
    PredictionSourceLoadedTelemetryFields, RunStartedTelemetryFields, SampleAlignmentCompletedTelemetryFields,
    SingleTraitPreflightCompletedTelemetryFields, build_association_backend_selected_telemetry_fields,
    build_bgen_engine_opened_telemetry_fields, build_effective_config_written_telemetry_fields,
    build_execution_plan_prepared_telemetry_fields, build_gpu_genotype_format_resolved_telemetry_fields,
    build_multi_phenotype_preflight_completed_telemetry_fields, build_multi_phenotype_sample_summary_telemetry_fields,
    build_multi_phenotype_writer_finished_telemetry_fields, build_phenotype_writer_finished_telemetry_fields,
    build_prediction_source_loaded_telemetry_fields, build_run_started_telemetry_fields,
    build_sample_alignment_completed_telemetry_fields, build_single_trait_preflight_completed_telemetry_fields,
};

pub const RUN_STARTED_EVENT_NAME: &str = "run_started";
pub const RUN_COMPLETED_EVENT_NAME: &str = "run_completed";
pub const RUN_FAILED_EVENT_NAME: &str = "run_failed";
pub const EXECUTION_PLAN_PREPARED_EVENT_NAME: &str = "execution_plan_prepared";
pub const EFFECTIVE_CONFIG_WRITTEN_EVENT_NAME: &str = "effective_config_written";
pub const WRITER_FINISHED_EVENT_NAME: &str = "writer_finished";
pub const PREFLIGHT_COMPLETED_EVENT_NAME: &str = "preflight_completed";
pub const SAMPLE_ALIGNMENT_COMPLETED_EVENT_NAME: &str = "sample_alignment_completed";
pub const PREDICTION_SOURCE_LOADED_EVENT_NAME: &str = "prediction_source_loaded";
pub const MULTI_PHENOTYPE_SAMPLE_SUMMARY_EVENT_NAME: &str = "multi_phenotype_sample_summary";
pub const GPU_GENOTYPE_FORMAT_RESOLVED_EVENT_NAME: &str = "gpu_genotype_format_resolved";
pub const ASSOCIATION_BACKEND_SELECTED_EVENT_NAME: &str = "association_backend_selected";
pub const BGEN_ENGINE_OPENED_EVENT_NAME: &str = "bgen_engine_opened";
pub const BINARY_CORRECTION_SUMMARY_EVENT_NAME: &str = "binary_correction_summary";
pub const CALLBACK_NULL_LOGISTIC_NONCONVERGENCE_WARNING_DIAGNOSTIC_EVENT_NAME: &str =
    "callback_null_logistic_nonconvergence_warning";
pub const RUN_LIFECYCLE_INFO_LEVEL: &str = "info";
pub const RUN_LIFECYCLE_WARN_LEVEL: &str = "warn";
pub const RUN_LIFECYCLE_ERROR_LEVEL: &str = "error";
pub const RUNNER_REGENIE_RUN_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "runner_regenie_run_started";
pub const RUNNER_REGENIE_RUN_INTERRUPTED_DIAGNOSTIC_EVENT_NAME: &str = "runner_regenie_run_interrupted";
pub const RUNNER_REGENIE_RUN_FAILED_DIAGNOSTIC_EVENT_NAME: &str = "runner_regenie_run_failed";
pub const RUNNER_REGENIE_RUN_COMPLETED_DIAGNOSTIC_EVENT_NAME: &str = "runner_regenie_run_completed";
pub const RUNNER_REGENIE_RUN_STARTED_DIAGNOSTIC_MESSAGE: &str = "Starting REGENIE run.";
pub const RUNNER_REGENIE_RUN_FAILED_DIAGNOSTIC_MESSAGE: &str = "REGENIE run failed.";
pub const RUNNER_REGENIE_RUN_COMPLETED_DIAGNOSTIC_MESSAGE: &str = "Finished REGENIE run.";
pub const NATIVE_CLI_STDOUT_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_stdout";
pub const NATIVE_CLI_STDERR_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_stderr";
pub const NATIVE_CLI_INTERRUPTED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_interrupted_line";
pub const NATIVE_CLI_FAILED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_failed_line";
pub const NATIVE_CLI_COMPLETED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_completed_line";
pub const RUNNER_JAX_RUNTIME_CONFIGURATION_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_jax_runtime_configuration_started";
pub const RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "runner_execution_plan_build_started";
pub const RUNNER_EXECUTION_PLAN_PREPARED_DIAGNOSTIC_EVENT_NAME: &str = "runner_execution_plan_prepared";
pub const RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "runner_execution_plan_dispatch_started";
pub const RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_execution_plan_finalization_started";
pub const RUNNER_JAX_RUNTIME_CONFIGURATION_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Configuring JAX runtime before backend initialization.";
pub const RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_MESSAGE: &str = "Building REGENIE execution plan.";
pub const RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str = "Dispatching REGENIE execution plan.";
pub const RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_MESSAGE: &str = "Finalizing REGENIE execution plan.";
pub const IO_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME: &str = "io_output_resume_committed_chunks";
pub const PIPELINE_GPU_GENOTYPE_FORMAT_RESOLVED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_gpu_genotype_format_resolved";
pub const PIPELINE_BGEN_ENGINE_OPEN_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_bgen_engine_open_started";
pub const PIPELINE_BGEN_ENGINE_OPENED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_bgen_engine_opened";
pub const PIPELINE_PREVALIDATED_BGEN_ENGINE_USED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_prevalidated_bgen_engine_used";
pub const PIPELINE_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_output_resume_committed_chunks";
pub const PIPELINE_OUTPUT_WRITER_SESSIONS_CREATE_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_output_writer_sessions_create_started";
pub const PIPELINE_MULTI_PHENOTYPE_SAMPLE_SUMMARY_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_phenotype_sample_summary";
pub const PIPELINE_MULTI_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_multi_trait_started";
pub const PIPELINE_MULTI_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_trait_input_load_started";
pub const PIPELINE_MULTI_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_multi_trait_input_aligned";
pub const PIPELINE_MULTI_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_trait_prediction_source_load_started";
pub const PIPELINE_GROUPED_PER_PHENOTYPE_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_grouped_per_phenotype_started";
pub const PIPELINE_GROUPED_PER_PHENOTYPE_GROUPS_PREPARED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_grouped_per_phenotype_groups_prepared";
pub const PIPELINE_GROUPED_UNION_DELIVERY_SELECTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_grouped_union_delivery_selected";
pub const PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_multi_group_preflight_started";
pub const PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Running preflight validation for multi-phenotype pipeline.";
pub const PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_group_preflight_completed";
pub const PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_MESSAGE: &str =
    "Preflight validation passed for multi-phenotype pipeline.";
pub const PIPELINE_SINGLE_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_single_trait_started";
pub const PIPELINE_SINGLE_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_input_load_started";
pub const PIPELINE_SINGLE_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_single_trait_input_aligned";
pub const PIPELINE_SINGLE_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_prediction_source_load_started";
pub const PIPELINE_SINGLE_TRAIT_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_preflight_started";
pub const PIPELINE_SINGLE_TRAIT_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_preflight_completed";
pub const NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_bgen_engine_constructing";
pub const NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_MESSAGE: &str = "Constructing native BGEN run engine.";
pub const NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_trusted_bgen_validation_started";
pub const NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Validating trusted no-missing diploid BGEN mode.";
pub const NATIVE_DISPATCH_CALLBACK_DRAIN_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_callback_drain_started";
pub const NATIVE_DISPATCH_CALLBACK_DRAIN_STARTED_DIAGNOSTIC_MESSAGE: &str = "Draining native callback worker queues.";
pub const NATIVE_DISPATCH_DELIVERY_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_delivery_started";
pub const NATIVE_DISPATCH_DELIVERY_FINISHED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_delivery_finished";
pub const NATIVE_DISPATCH_DELIVERY_INTERRUPTED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_delivery_interrupted";
pub const NATIVE_DISPATCH_DELIVERY_FAILED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_delivery_failed";
pub const NATIVE_DISPATCH_PIPELINE_FINISHED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_pipeline_finished";
pub const NATIVE_DISPATCH_WRITER_SESSION_FINISH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_writer_session_finish_started";
pub const NATIVE_DISPATCH_WRITER_SESSION_FINISH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Finishing output writer and optional Parquet finalization.";
pub const NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_writer_sessions_finish_started";
pub const NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Finishing output writer(s) and optional Parquet finalization.";
pub const NATIVE_DISPATCH_WRITER_SESSION_INTERRUPTED_FLUSH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_writer_session_interrupted_flush_started";
pub const NATIVE_DISPATCH_WRITER_SESSIONS_INTERRUPTED_FLUSH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_writer_sessions_interrupted_flush_started";
pub const NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_EVENT_NAME: &str = "native_runtime_knobs_configured";
pub const NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_MESSAGE: &str = "Configuring native runtime knobs.";
pub const PREFLIGHT_WARNING_DIAGNOSTIC_EVENT_NAME: &str = "preflight_warning";
pub const RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_multi_phenotype_dispatch_started";
pub const RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_single_phenotype_dispatch_started";
pub const RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "runner_binary_engine_dispatch_started";
pub const RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "runner_linear_engine_dispatch_started";
pub const RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_multi_phenotype_binary_engine_dispatch_started";
pub const RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_multi_phenotype_linear_engine_dispatch_started";
pub const RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching multi-phenotype native engine pipeline.";
pub const RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching single-phenotype native engine pipeline.";
pub const RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str = "Dispatching binary native engine pipeline.";
pub const RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str = "Dispatching linear native engine pipeline.";
pub const RUNNER_METADATA_ARTIFACTS_FINALIZED_DIAGNOSTIC_EVENT_NAME: &str = "runner_metadata_artifacts_finalized";
pub const RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching multi-phenotype binary native engine pipeline.";
pub const RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching multi-phenotype linear native engine pipeline.";

#[must_use]
pub fn build_native_runtime_knobs_configured_diagnostic_payload(
    bgen_decode_tile_variant_count: i64,
    threads: Option<i64>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_EVENT_NAME,
        message: NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("bgen_decode_tile_variant_count", bgen_decode_tile_variant_count),
            optional_integer_diagnostic_field("threads", threads),
        ],
    }
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_preflight_warning_diagnostic_payload(
    message: &str,
    chromosome_count: i64,
    covariate_count: i64,
    preflight_scope: &str,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    warning_index: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "warning",
        event_name: PREFLIGHT_WARNING_DIAGNOSTIC_EVENT_NAME,
        message: message.to_string(),
        fields: vec![
            integer_diagnostic_field("chromosome_count", chromosome_count),
            integer_diagnostic_field("covariate_count", covariate_count),
            text_diagnostic_field("preflight_scope", preflight_scope),
            integer_diagnostic_field("sample_count", sample_count),
            boolean_diagnostic_field("trusted_no_missing_diploid", trusted_no_missing_diploid),
            integer_diagnostic_field("warning_index", warning_index),
        ],
    }
}

#[must_use]
pub fn build_io_output_resume_committed_chunks_diagnostic_payload(
    chunks_directory: &str,
    committed_chunk_count: i64,
    run_directory: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: IO_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME,
        message: format!("Resuming run with {committed_chunk_count} previously committed chunks."),
        fields: vec![
            text_diagnostic_field("chunks_directory", chunks_directory),
            integer_diagnostic_field("committed_chunk_count", committed_chunk_count),
            text_diagnostic_field("run_directory", run_directory),
        ],
    }
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_callback_null_logistic_nonconvergence_warning_diagnostic_payload(
    message: &str,
    chromosome: &str,
    nonconverged_count: i64,
    phenotype_count: i64,
    policy: &str,
    scalar_convergence: bool,
    total_fit_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "warning",
        event_name: CALLBACK_NULL_LOGISTIC_NONCONVERGENCE_WARNING_DIAGNOSTIC_EVENT_NAME,
        message: message.to_string(),
        fields: vec![
            text_diagnostic_field("chromosome", chromosome),
            integer_diagnostic_field("nonconverged_count", nonconverged_count),
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("policy", policy),
            boolean_diagnostic_field("scalar_convergence", scalar_convergence),
            integer_diagnostic_field("total_fit_count", total_fit_count),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_artifact() -> RunArtifactPayload {
        RunArtifactPayload {
            phenotype_name: Some("height".to_string()),
            output_run_directory: Some("run".to_string()),
            final_dataset: Some("run/chunks".to_string()),
            final_parquet: Some("run/final.parquet".to_string()),
            final_regenie: None,
            effective_config: Some("run/effective_config.toml".to_string()),
        }
    }

    fn build_artifact_tree(name: &str) -> RunArtifactsPayload {
        RunArtifactsPayload {
            phenotype_name: Some(name.to_string()),
            output_run_directory: Some(format!("run/{name}")),
            final_dataset: None,
            final_parquet: Some(format!("run/{name}.parquet")),
            final_regenie: None,
            effective_config: Some(format!("run/{name}/effective_config.toml")),
            phenotype_artifacts: Vec::new(),
            association_mode: None,
            phenotype_count: None,
            run_id: None,
        }
    }

    #[test]
    fn builds_run_completed_event_from_artifact_tree() {
        let artifacts = RunArtifactsPayload {
            phenotype_name: None,
            output_run_directory: None,
            final_dataset: None,
            final_parquet: None,
            final_regenie: None,
            effective_config: None,
            phenotype_artifacts: vec![build_artifact_tree("height"), build_artifact_tree("weight")],
            association_mode: Some("regenie2_linear".to_string()),
            phenotype_count: None,
            run_id: Some("run-1".to_string()),
        };

        let event = build_run_completed_event_from_artifacts(&artifacts);

        assert_eq!(event.run_id.as_deref(), Some("run-1"));
        assert_eq!(event.association_mode.as_deref(), Some("regenie2_linear"));
        assert_eq!(event.phenotype_count, Some(2));
        assert_eq!(
            event.artifacts.iter().map(|artifact| artifact.phenotype_name.as_deref()).collect::<Vec<_>>(),
            vec![Some("height"), Some("weight")]
        );
        assert_eq!(event.artifacts[1].final_parquet.as_deref(), Some("run/weight.parquet"));
    }

    #[test]
    fn attaches_run_metadata_to_artifact_tree() {
        let artifacts = RunArtifactsPayload {
            phenotype_name: None,
            output_run_directory: None,
            final_dataset: None,
            final_parquet: None,
            final_regenie: None,
            effective_config: None,
            phenotype_artifacts: vec![build_artifact_tree("height"), build_artifact_tree("weight")],
            association_mode: None,
            phenotype_count: None,
            run_id: None,
        };

        let attached_artifacts = attach_run_metadata_to_artifacts(&artifacts, Some("run-1"), "regenie2_linear", 2);

        assert_eq!(attached_artifacts.run_id.as_deref(), Some("run-1"));
        assert_eq!(attached_artifacts.association_mode.as_deref(), Some("regenie2_linear"));
        assert_eq!(attached_artifacts.phenotype_count, Some(2));
        assert_eq!(attached_artifacts.phenotype_artifacts[0].run_id.as_deref(), Some("run-1"));
        assert_eq!(attached_artifacts.phenotype_artifacts[1].association_mode.as_deref(), Some("regenie2_linear"));
        assert_eq!(attached_artifacts.phenotype_artifacts[1].phenotype_count, Some(2));
    }

    #[test]
    fn builds_interrupted_and_failed_event_payloads() {
        assert_eq!(
            build_run_interrupted_event_payload(2, "SIGINT", 130, true),
            RunInterruptedEventPayload {
                signal_number: 2,
                signal_name: "SIGINT".to_string(),
                exit_code: 130,
                flushed_for_resume: true,
            },
        );
        assert_eq!(
            build_run_failed_event_payload("RuntimeError", "boom"),
            RunFailedEventPayload { error_type: "RuntimeError".to_string(), error_message: "boom".to_string() },
        );
    }

    #[test]
    fn builds_runner_lifecycle_diagnostic_payloads() {
        assert_eq!(
            build_runner_run_started_diagnostic_payload("regenie2_linear", "quantitative", 2),
            RunDiagnosticEventPayload {
                level: "info",
                event_name: "runner_regenie_run_started",
                message: "Starting REGENIE run.".to_string(),
                fields: vec![
                    RunDiagnosticFieldPayload {
                        name: "association_mode",
                        value: RunDiagnosticFieldValue::Text("regenie2_linear".to_string()),
                    },
                    RunDiagnosticFieldPayload {
                        name: "trait_type",
                        value: RunDiagnosticFieldValue::Text("quantitative".to_string()),
                    },
                    RunDiagnosticFieldPayload { name: "phenotype_count", value: RunDiagnosticFieldValue::Integer(2) },
                ],
            },
        );

        let interrupted_event = build_run_interrupted_event_payload(2, "SIGINT", 130, true);
        let failed_event = build_run_failed_event_payload("RuntimeError", "boom");
        let completed_event = RunCompletedEventPayload {
            run_id: Some("run-1".to_string()),
            association_mode: Some("regenie2_linear".to_string()),
            phenotype_count: Some(2),
            artifacts: Vec::new(),
        };

        assert_eq!(
            build_runner_run_interrupted_diagnostic_payload(&interrupted_event).event_name,
            "runner_regenie_run_interrupted"
        );
        assert_eq!(
            build_runner_run_interrupted_diagnostic_payload(&interrupted_event).fields[3].value,
            RunDiagnosticFieldValue::Boolean(true),
        );
        assert_eq!(
            build_runner_run_failed_diagnostic_payload(&failed_event).fields[1].value,
            RunDiagnosticFieldValue::Text("boom".to_string()),
        );
        assert_eq!(
            build_runner_run_completed_diagnostic_payload(&completed_event).fields[2].value,
            RunDiagnosticFieldValue::OptionalInteger(Some(2)),
        );
    }

    #[test]
    fn serializes_run_diagnostic_fields_json() {
        let fields = vec![
            RunDiagnosticFieldPayload { name: "flag", value: RunDiagnosticFieldValue::Boolean(true) },
            RunDiagnosticFieldPayload { name: "count", value: RunDiagnosticFieldValue::Integer(3) },
            RunDiagnosticFieldPayload { name: "maybe_count", value: RunDiagnosticFieldValue::OptionalInteger(None) },
            RunDiagnosticFieldPayload {
                name: "maybe_text",
                value: RunDiagnosticFieldValue::OptionalText(Some("present".to_string())),
            },
            RunDiagnosticFieldPayload { name: "text", value: RunDiagnosticFieldValue::Text("value".to_string()) },
        ];
        let fields_text = serialize_run_diagnostic_fields_json(&fields).expect("fields should serialize");
        let fields_payload: serde_json::Value =
            serde_json::from_str(&fields_text).expect("fields should be valid JSON");

        assert_eq!(
            fields_payload,
            serde_json::json!({
                "flag": true,
                "count": 3,
                "maybe_count": null,
                "maybe_text": "present",
                "text": "value",
            }),
        );
    }

    #[test]
    fn builds_native_cli_diagnostic_payloads() {
        let stdout_payload = build_native_cli_stdout_diagnostic_payload("abcdef", 3);
        let stderr_payload = build_native_cli_stderr_diagnostic_payload("éx", 5);
        let interrupted_payload = build_native_cli_interrupted_line_diagnostic_payload("Interrupted.");
        let failed_payload = build_native_cli_failed_line_diagnostic_payload("Error: failed.");
        let completed_payload = build_native_cli_completed_line_diagnostic_payload("Success.");

        assert_eq!(stdout_payload.event_name, "native_cli_stdout");
        assert_eq!(stdout_payload.fields[0].value, RunDiagnosticFieldValue::Integer(6));
        assert_eq!(stdout_payload.fields[1].value, RunDiagnosticFieldValue::Integer(6));
        assert_eq!(stdout_payload.fields[2].value, RunDiagnosticFieldValue::Text("abc".to_string()));
        assert_eq!(stdout_payload.fields[3].value, RunDiagnosticFieldValue::Boolean(true));
        assert_eq!(stdout_payload.fields[4].value, RunDiagnosticFieldValue::Integer(3));
        assert_eq!(stderr_payload.event_name, "native_cli_stderr");
        assert_eq!(stderr_payload.fields[0].value, RunDiagnosticFieldValue::Integer(2));
        assert_eq!(stderr_payload.fields[1].value, RunDiagnosticFieldValue::Integer(3));
        assert_eq!(stderr_payload.fields[3].value, RunDiagnosticFieldValue::Boolean(false));
        assert_eq!(interrupted_payload.event_name, "native_cli_interrupted_line");
        assert_eq!(failed_payload.event_name, "native_cli_failed_line");
        assert_eq!(completed_payload.event_name, "native_cli_completed_line");
        assert_eq!(completed_payload.fields[0].value, RunDiagnosticFieldValue::Text("Success.".to_string()));
    }

    #[test]
    fn builds_runner_execution_plan_diagnostic_payloads() {
        assert_eq!(
            build_runner_jax_runtime_configuration_started_diagnostic_payload().event_name,
            "runner_jax_runtime_configuration_started",
        );
        assert_eq!(
            build_runner_execution_plan_build_started_diagnostic_payload().message,
            "Building REGENIE execution plan.",
        );

        let prepared_payload =
            build_runner_execution_plan_prepared_diagnostic_payload("regenie2_binary", 3, 1024, Some(4096), "gpu");

        assert_eq!(prepared_payload.level, "info");
        assert_eq!(prepared_payload.event_name, "runner_execution_plan_prepared");
        assert_eq!(prepared_payload.message, "Prepared REGENIE execution plan for 3 phenotype(s).");
        assert_eq!(
            prepared_payload.fields[3],
            RunDiagnosticFieldPayload {
                name: "variant_limit",
                value: RunDiagnosticFieldValue::OptionalInteger(Some(4096)),
            },
        );
        assert_eq!(
            build_runner_execution_plan_dispatch_started_diagnostic_payload(3, "regenie2_binary").fields[1].value,
            RunDiagnosticFieldValue::Text("regenie2_binary".to_string()),
        );
        assert_eq!(
            build_runner_execution_plan_finalization_started_diagnostic_payload(3, "regenie2_binary").event_name,
            "runner_execution_plan_finalization_started",
        );
    }

    #[test]
    fn builds_runner_dispatch_diagnostic_payloads() {
        assert_eq!(
            build_runner_multi_phenotype_dispatch_started_diagnostic_payload(3, "regenie2_binary").event_name,
            "runner_multi_phenotype_dispatch_started",
        );
        assert_eq!(
            build_runner_single_phenotype_dispatch_started_diagnostic_payload("regenie2_linear", "height").fields[1]
                .value,
            RunDiagnosticFieldValue::Text("height".to_string()),
        );
        assert_eq!(
            build_runner_binary_engine_dispatch_started_diagnostic_payload("height").message,
            "Dispatching binary native engine pipeline.",
        );
        assert_eq!(
            build_runner_linear_engine_dispatch_started_diagnostic_payload("height").event_name,
            "runner_linear_engine_dispatch_started",
        );
        assert_eq!(
            build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload(3).fields[0].value,
            RunDiagnosticFieldValue::Integer(3),
        );
        assert_eq!(
            build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload(3).message,
            "Dispatching multi-phenotype linear native engine pipeline.",
        );
    }

    #[test]
    fn builds_native_runtime_knobs_diagnostic_payload() {
        let payload = build_native_runtime_knobs_configured_diagnostic_payload(32, Some(4));

        assert_eq!(payload.event_name, "native_runtime_knobs_configured");
        assert_eq!(payload.message, "Configuring native runtime knobs.");
        assert_eq!(payload.fields[0].value, RunDiagnosticFieldValue::Integer(32));
        assert_eq!(payload.fields[1].value, RunDiagnosticFieldValue::OptionalInteger(Some(4)));
    }

    #[test]
    fn builds_runner_metadata_artifacts_finalized_diagnostic_payload() {
        let payload = build_runner_metadata_artifacts_finalized_diagnostic_payload("regenie2_binary", 3);

        assert_eq!(payload.event_name, "runner_metadata_artifacts_finalized");
        assert_eq!(payload.level, "info");
        assert_eq!(payload.message, "Finalized REGENIE run artifacts for 3 phenotype(s).");
        assert_eq!(payload.fields[0].value, RunDiagnosticFieldValue::Text("regenie2_binary".to_string()));
        assert_eq!(payload.fields[1].value, RunDiagnosticFieldValue::Integer(3));
    }

    #[test]
    fn builds_preflight_warning_diagnostic_payload() {
        let payload = build_preflight_warning_diagnostic_payload("low degrees", 1, 2, "single_trait", 3, true, 0);

        assert_eq!(payload.level, "warning");
        assert_eq!(payload.event_name, "preflight_warning");
        assert_eq!(payload.message, "low degrees");
        assert_eq!(payload.fields[2].value, RunDiagnosticFieldValue::Text("single_trait".to_string()));
        assert_eq!(payload.fields[4].value, RunDiagnosticFieldValue::Boolean(true));
    }

    #[test]
    fn builds_io_output_resume_committed_chunks_diagnostic_payload() {
        let payload = build_io_output_resume_committed_chunks_diagnostic_payload("out/chunks", 2, "out/run");

        assert_eq!(payload.event_name, "io_output_resume_committed_chunks");
        assert_eq!(payload.message, "Resuming run with 2 previously committed chunks.");
        assert_eq!(payload.fields[0].value, RunDiagnosticFieldValue::Text("out/chunks".to_string()));
        assert_eq!(payload.fields[1].value, RunDiagnosticFieldValue::Integer(2));
        assert_eq!(payload.fields[2].value, RunDiagnosticFieldValue::Text("out/run".to_string()));
    }

    #[test]
    fn builds_pipeline_output_diagnostic_payloads() {
        let open_started_payload = build_pipeline_bgen_engine_open_started_diagnostic_payload(
            Some(2),
            None,
            "multi-phenotype",
            true,
            Some(100),
        );
        let opened_payload =
            build_pipeline_bgen_engine_opened_diagnostic_payload(Some(2), Some("trait"), "binary", 3, 4);
        let prevalidated_payload =
            build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload(None, Some("trait"), "binary");
        let resume_payload = build_pipeline_output_resume_committed_chunks_diagnostic_payload(5, 1);
        let writer_payload =
            build_pipeline_output_writer_sessions_create_started_diagnostic_payload("regenie2_linear", 2);

        assert_eq!(open_started_payload.event_name, "pipeline_bgen_engine_open_started");
        assert_eq!(open_started_payload.fields[0].value, RunDiagnosticFieldValue::OptionalInteger(Some(2)));
        assert_eq!(open_started_payload.fields[1].value, RunDiagnosticFieldValue::OptionalText(None));
        assert_eq!(opened_payload.event_name, "pipeline_bgen_engine_opened");
        assert_eq!(
            opened_payload.message,
            "Native BGEN engine opened for binary pipeline: sample_count=3 variant_count=4."
        );
        assert_eq!(prevalidated_payload.event_name, "pipeline_prevalidated_bgen_engine_used");
        assert_eq!(resume_payload.event_name, "pipeline_output_resume_committed_chunks");
        assert_eq!(resume_payload.fields[1].value, RunDiagnosticFieldValue::Integer(1));
        assert_eq!(writer_payload.event_name, "pipeline_output_writer_sessions_create_started");
    }

    #[test]
    fn builds_native_dispatch_engine_diagnostic_payloads() {
        let constructing_payload =
            build_native_dispatch_bgen_engine_constructing_diagnostic_payload(1024, "input.bgen", true, Some(4096));
        let validation_payload =
            build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload("input.bgen", "cache");

        assert_eq!(constructing_payload.event_name, "native_dispatch_bgen_engine_constructing");
        assert_eq!(constructing_payload.fields[3].value, RunDiagnosticFieldValue::OptionalInteger(Some(4096)));
        assert_eq!(validation_payload.event_name, "native_dispatch_trusted_bgen_validation_started");
        assert_eq!(validation_payload.fields[1].value, RunDiagnosticFieldValue::Text("cache".to_string()));
    }

    #[test]
    fn builds_native_dispatch_writer_diagnostic_payloads() {
        let callback_payload = build_native_dispatch_callback_drain_started_diagnostic_payload();
        let single_writer_payload = build_native_dispatch_writer_session_finish_started_diagnostic_payload();
        let multi_writer_payload = build_native_dispatch_writer_sessions_finish_started_diagnostic_payload(2, 3);
        let interrupted_payload =
            build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload(130, "SIGINT", 2);
        let interrupted_writers_payload =
            build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload(
                4, 143, "SIGTERM", 15, 5,
            );

        assert_eq!(callback_payload.event_name, "native_dispatch_callback_drain_started");
        assert_eq!(callback_payload.fields, Vec::new());
        assert_eq!(single_writer_payload.event_name, "native_dispatch_writer_session_finish_started");
        assert_eq!(multi_writer_payload.fields[1].value, RunDiagnosticFieldValue::Integer(3));
        assert_eq!(interrupted_payload.message, "Flushing interrupted output writer after SIGINT.");
        assert_eq!(interrupted_writers_payload.event_name, "native_dispatch_writer_sessions_interrupted_flush_started");
        assert_eq!(interrupted_writers_payload.fields[4].value, RunDiagnosticFieldValue::Integer(5));
    }

    #[test]
    fn builds_native_dispatch_delivery_diagnostic_payloads() {
        let started_payload = build_native_dispatch_delivery_started_diagnostic_payload(2, "Native BGEN", true);
        let finished_payload = build_native_dispatch_delivery_finished_diagnostic_payload("Native BGEN", 3);
        let interrupted_payload =
            build_native_dispatch_delivery_interrupted_diagnostic_payload("Native BGEN", 130, "SIGINT", 2);
        let failed_payload =
            build_native_dispatch_delivery_failed_diagnostic_payload("decode failed", "RuntimeError", "Native BGEN");
        let pipeline_finished_payload = build_native_dispatch_pipeline_finished_diagnostic_payload(1, "Native BGEN");

        assert_eq!(started_payload.event_name, "native_dispatch_delivery_started");
        assert_eq!(
            started_payload.message,
            "Starting Native BGEN delivery: committed_chunk_count=2 variant_major_packed8_probability_pairs=true."
        );
        assert_eq!(finished_payload.fields[1].value, RunDiagnosticFieldValue::Integer(3));
        assert_eq!(interrupted_payload.event_name, "native_dispatch_delivery_interrupted");
        assert_eq!(failed_payload.fields[1].value, RunDiagnosticFieldValue::Text("RuntimeError".to_string()));
        assert_eq!(pipeline_finished_payload.message, "Native BGEN pipeline finished.");
    }

    #[test]
    fn builds_pipeline_gpu_genotype_format_resolved_diagnostic_payload() {
        let payload = build_pipeline_gpu_genotype_format_resolved_diagnostic_payload(
            "auto",
            "dosage",
            "trusted_validation_failed",
            Some("packed8 incompatible"),
        );

        assert_eq!(payload.level, "info");
        assert_eq!(payload.event_name, "pipeline_gpu_genotype_format_resolved");
        assert_eq!(payload.message, "Resolved gpu_genotype_format=auto to dosage: trusted_validation_failed.");
        assert_eq!(
            payload.fields[0].value,
            RunDiagnosticFieldValue::OptionalText(Some("packed8 incompatible".to_string()))
        );
        assert_eq!(payload.fields[3].value, RunDiagnosticFieldValue::Text("dosage".to_string()));
    }

    #[test]
    fn builds_callback_null_logistic_nonconvergence_warning_diagnostic_payload() {
        let payload = build_callback_null_logistic_nonconvergence_warning_diagnostic_payload(
            "Null logistic failed.",
            "1",
            2,
            3,
            "warn",
            false,
            4,
        );

        assert_eq!(payload.level, "warning");
        assert_eq!(payload.event_name, "callback_null_logistic_nonconvergence_warning");
        assert_eq!(payload.message, "Null logistic failed.");
        assert_eq!(payload.fields[0].value, RunDiagnosticFieldValue::Text("1".to_string()));
        assert_eq!(payload.fields[5].value, RunDiagnosticFieldValue::Integer(4));
    }

    #[test]
    fn builds_pipeline_multi_phenotype_sample_summary_diagnostic_payload() {
        let complete_case_payload =
            build_pipeline_multi_phenotype_sample_summary_diagnostic_payload(2, 1, false, "complete-case");
        let per_phenotype_payload =
            build_pipeline_multi_phenotype_sample_summary_diagnostic_payload(2, 2, true, "per-phenotype");

        assert_eq!(complete_case_payload.event_name, "pipeline_multi_phenotype_sample_summary");
        assert_eq!(
            complete_case_payload.message,
            "Analyzed 2 phenotypes in complete-case sample mode; one shared sample set was used."
        );
        assert_eq!(
            per_phenotype_payload.message,
            "Analyzed 2 phenotypes in per-phenotype sample mode; sample counts differ across phenotypes."
        );
        assert_eq!(per_phenotype_payload.fields[1].value, RunDiagnosticFieldValue::Integer(2));
    }

    #[test]
    fn builds_pipeline_multi_trait_diagnostic_payloads() {
        let started_payload =
            build_pipeline_multi_trait_started_diagnostic_payload("regenie2_linear", 2, "complete-case");
        let input_payload = build_pipeline_multi_trait_input_load_started_diagnostic_payload(2);
        let aligned_payload = build_pipeline_multi_trait_input_aligned_diagnostic_payload(3, 2, 4);
        let prediction_payload = build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload(2);

        assert_eq!(started_payload.event_name, "pipeline_multi_trait_started");
        assert_eq!(started_payload.message, "Starting multi-phenotype REGENIE step 2 BGEN pipeline.");
        assert_eq!(input_payload.event_name, "pipeline_multi_trait_input_load_started");
        assert_eq!(aligned_payload.fields[0].value, RunDiagnosticFieldValue::Integer(3));
        assert_eq!(
            aligned_payload.message,
            "Aligned multi-phenotype pipeline inputs: sample_count=4 phenotype_count=2 covariate_count=3."
        );
        assert_eq!(prediction_payload.event_name, "pipeline_multi_trait_prediction_source_load_started");
    }

    #[test]
    fn builds_pipeline_grouped_diagnostic_payloads() {
        let started_payload =
            build_pipeline_grouped_per_phenotype_started_diagnostic_payload("regenie2_binary", 2, "per-phenotype");
        let prepared_payload = build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload(2, 1);
        let union_payload = build_pipeline_grouped_union_delivery_selected_diagnostic_payload(6, 2, 4);

        assert_eq!(started_payload.event_name, "pipeline_grouped_per_phenotype_started");
        assert_eq!(started_payload.message, "Starting grouped per-phenotype REGENIE step 2 BGEN pipeline.");
        assert_eq!(prepared_payload.event_name, "pipeline_grouped_per_phenotype_groups_prepared");
        assert_eq!(prepared_payload.message, "Prepared 1 compatible per-phenotype group(s) for 2 phenotype(s).");
        assert_eq!(union_payload.event_name, "pipeline_grouped_union_delivery_selected");
        assert_eq!(union_payload.fields[0].value, RunDiagnosticFieldValue::Integer(6));
        assert_eq!(
            union_payload.message,
            "Using union per-phenotype BGEN delivery: group_count=2 union_sample_count=4 grouped_sample_count=6."
        );
    }

    #[test]
    fn builds_pipeline_multi_group_preflight_diagnostic_payloads() {
        let started_payload = build_pipeline_multi_group_preflight_started_diagnostic_payload(2, 3, true, Some(100));
        let completed_payload = build_pipeline_multi_group_preflight_completed_diagnostic_payload(2, 3, true, None);

        assert_eq!(started_payload.event_name, "pipeline_multi_group_preflight_started");
        assert_eq!(started_payload.fields[3].value, RunDiagnosticFieldValue::OptionalInteger(Some(100)));
        assert_eq!(completed_payload.event_name, "pipeline_multi_group_preflight_completed");
        assert_eq!(completed_payload.fields[3].value, RunDiagnosticFieldValue::OptionalInteger(None));
    }

    #[test]
    fn builds_pipeline_single_trait_diagnostic_payloads() {
        let started_payload =
            build_pipeline_single_trait_started_diagnostic_payload("regenie2_binary", "trait", "binary");
        let input_payload = build_pipeline_single_trait_input_load_started_diagnostic_payload("trait", "binary");
        let aligned_payload = build_pipeline_single_trait_input_aligned_diagnostic_payload(2, "trait", "binary", 3);
        let prediction_payload =
            build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload("trait", "binary");
        let preflight_started_payload =
            build_pipeline_single_trait_preflight_started_diagnostic_payload("trait", "binary", true, Some(100));
        let preflight_completed_payload =
            build_pipeline_single_trait_preflight_completed_diagnostic_payload(22, 2, "trait", "binary", 3);

        assert_eq!(started_payload.event_name, "pipeline_single_trait_started");
        assert_eq!(started_payload.message, "Starting binary REGENIE step 2 BGEN pipeline.");
        assert_eq!(input_payload.event_name, "pipeline_single_trait_input_load_started");
        assert_eq!(aligned_payload.fields[0].value, RunDiagnosticFieldValue::Integer(2));
        assert_eq!(prediction_payload.event_name, "pipeline_single_trait_prediction_source_load_started");
        assert_eq!(preflight_started_payload.fields[3].value, RunDiagnosticFieldValue::OptionalInteger(Some(100)));
        assert_eq!(
            preflight_completed_payload.message,
            "Preflight validation passed for binary pipeline: sample_count=3 covariate_count=2 chromosome_count=22."
        );
    }

    #[test]
    fn builds_run_completed_telemetry_fields() {
        let event = RunCompletedEventPayload {
            run_id: Some("run-1".to_string()),
            association_mode: Some("regenie2_linear".to_string()),
            phenotype_count: Some(1),
            artifacts: vec![build_artifact()],
        };

        let fields = build_run_completed_telemetry_fields(&event);

        assert_eq!(fields.artifact_count, 1);
        assert_eq!(fields.run_id.as_deref(), Some("run-1"));
        assert_eq!(fields.phenotype_artifacts.len(), 1);
        assert_eq!(fields.single_artifact, fields.phenotype_artifacts.first().cloned());
        assert!(
            fields.phenotype_artifacts[0]
                .fields
                .iter()
                .any(|field| field.key == "final_parquet" && field.value == "run/final.parquet")
        );
    }

    #[test]
    fn builds_run_started_and_execution_plan_telemetry_fields() {
        assert_eq!(
            build_run_started_telemetry_fields("regenie2_linear", "quantitative", 2, "output.g"),
            RunStartedTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                trait_type: "quantitative".to_string(),
                phenotype_count: 2,
                output_run_root: "output.g".to_string(),
            }
        );
        assert_eq!(
            build_execution_plan_prepared_telemetry_fields("regenie2_binary", "binary", 3, 1024, Some(4096), "gpu",),
            ExecutionPlanPreparedTelemetryFields {
                association_mode: "regenie2_binary".to_string(),
                trait_type: "binary".to_string(),
                phenotype_count: 3,
                chunk_size: 1024,
                variant_limit: Some(4096),
                device: "gpu".to_string(),
            }
        );
    }

    #[test]
    fn builds_writer_lifecycle_telemetry_fields() {
        assert_eq!(
            build_effective_config_written_telemetry_fields(
                "regenie2_linear",
                "height",
                "run/height/effective_config.toml",
                "run/height",
            ),
            EffectiveConfigWrittenTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                phenotype: "height".to_string(),
                effective_config: "run/height/effective_config.toml".to_string(),
                output_run_directory: "run/height".to_string(),
            }
        );
        assert_eq!(
            build_phenotype_writer_finished_telemetry_fields(
                "regenie2_binary",
                "case_status",
                Some("run/case_status/results.parquet"),
            ),
            PhenotypeWriterFinishedTelemetryFields {
                association_mode: "regenie2_binary".to_string(),
                phenotype: "case_status".to_string(),
                final_output_path: Some("run/case_status/results.parquet".to_string()),
            }
        );
        assert_eq!(
            build_multi_phenotype_writer_finished_telemetry_fields(
                "regenie2_linear",
                2,
                &[Some("run/height.parquet".to_string()), None],
            ),
            MultiPhenotypeWriterFinishedTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                phenotype_count: 2,
                final_output_paths: vec![Some("run/height.parquet".to_string()), None],
            }
        );
    }

    #[test]
    fn builds_preflight_completed_telemetry_fields() {
        assert_eq!(
            build_single_trait_preflight_completed_telemetry_fields("regenie2_linear", "height", 2504, 3, 22),
            SingleTraitPreflightCompletedTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                phenotype: "height".to_string(),
                sample_count: 2504,
                covariate_count: 3,
                chromosome_count: 22,
            }
        );
        assert_eq!(
            build_multi_phenotype_preflight_completed_telemetry_fields("regenie2_binary", 4, 2504),
            MultiPhenotypePreflightCompletedTelemetryFields {
                association_mode: "regenie2_binary".to_string(),
                phenotype_count: 4,
                sample_count: 2504,
            }
        );
    }

    #[test]
    fn builds_pipeline_setup_telemetry_fields() {
        assert_eq!(
            build_sample_alignment_completed_telemetry_fields(
                "regenie2_linear",
                Some("height"),
                None,
                Some(2504),
                Some(3),
                None,
            ),
            SampleAlignmentCompletedTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                phenotype: Some("height".to_string()),
                phenotype_count: None,
                sample_count: Some(2504),
                covariate_count: Some(3),
                phenotype_group_count: None,
            }
        );
        assert_eq!(
            build_prediction_source_loaded_telemetry_fields("regenie2_binary", None, Some(4)),
            PredictionSourceLoadedTelemetryFields {
                association_mode: "regenie2_binary".to_string(),
                phenotype: None,
                phenotype_count: Some(4),
            }
        );
    }

    #[test]
    fn builds_multi_phenotype_sample_summary_telemetry_fields() {
        assert_eq!(
            build_multi_phenotype_sample_summary_telemetry_fields(
                "regenie2_linear",
                "per-phenotype",
                &[3, 2],
                &[Some("sample-a".to_string()), Some("sample-b".to_string())],
                2,
            ),
            MultiPhenotypeSampleSummaryTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                multi_phenotype_sample_mode: "per-phenotype".to_string(),
                phenotype_count: 2,
                phenotype_group_count: 2,
                sample_counts: vec![3, 2],
                sample_counts_differ: true,
                shared_sample_set: false,
            }
        );
        assert!(
            build_multi_phenotype_sample_summary_telemetry_fields(
                "regenie2_binary",
                "complete-case",
                &[2504, 2504],
                &[Some("shared".to_string()), Some("shared".to_string())],
                1,
            )
            .shared_sample_set
        );
    }

    #[test]
    fn builds_gpu_genotype_format_resolved_telemetry_fields() {
        assert_eq!(
            build_gpu_genotype_format_resolved_telemetry_fields(
                "auto",
                "dosage",
                "trusted_validation_failed",
                Some("packed8 incompatible"),
            ),
            GpuGenotypeFormatResolvedTelemetryFields {
                requested_gpu_genotype_format: "auto".to_string(),
                resolved_gpu_genotype_format: "dosage".to_string(),
                resolution_reason: "trusted_validation_failed".to_string(),
                fallback_error: Some("packed8 incompatible".to_string()),
            }
        );
        assert_eq!(
            build_gpu_genotype_format_resolved_telemetry_fields("auto", "packed8", "trusted_validation_passed", None,)
                .fallback_error,
            None,
        );
    }

    #[test]
    fn builds_engine_opening_telemetry_fields() {
        assert_eq!(
            build_association_backend_selected_telemetry_fields(
                "regenie2_linear",
                "jax_packed8",
                "gpu",
                "packed8",
                Some("height"),
                None,
            ),
            AssociationBackendSelectedTelemetryFields {
                association_mode: "regenie2_linear".to_string(),
                association_backend_kind: "jax_packed8".to_string(),
                device: "gpu".to_string(),
                genotype_format: "packed8".to_string(),
                phenotype: Some("height".to_string()),
                phenotype_count: None,
            }
        );
        assert_eq!(
            build_bgen_engine_opened_telemetry_fields("regenie2_binary", "jax_dosage", 2504, 12345, None, Some(3),),
            BgenEngineOpenedTelemetryFields {
                association_mode: "regenie2_binary".to_string(),
                association_backend_kind: "jax_dosage".to_string(),
                sample_count: 2504,
                variant_count: 12345,
                phenotype: None,
                phenotype_count: Some(3),
            }
        );
    }

    #[test]
    fn renders_run_lifecycle_lines() {
        let completed = RunCompletedEventPayload {
            run_id: None,
            association_mode: None,
            phenotype_count: None,
            artifacts: vec![build_artifact()],
        };
        let interrupted = RunInterruptedEventPayload {
            signal_number: 2,
            signal_name: "SIGINT".to_string(),
            exit_code: 130,
            flushed_for_resume: true,
        };
        let failed = RunFailedEventPayload { error_type: "RuntimeError".to_string(), error_message: String::new() };

        assert_eq!(render_run_completed_lines(&completed)[0], "Success. Chunked run saved to run");
        assert_eq!(
            render_run_interrupted_lines(&interrupted),
            vec!["Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume.".to_string()]
        );
        assert_eq!(render_run_failed_lines(&failed), vec!["Error: RuntimeError".to_string()]);
    }
}
