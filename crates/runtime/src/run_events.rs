//! Runtime-owned run lifecycle event payloads and rendering policy.

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
pub const RUN_LIFECYCLE_INFO_LEVEL: &str = "info";
pub const RUN_LIFECYCLE_WARN_LEVEL: &str = "warn";
pub const RUN_LIFECYCLE_ERROR_LEVEL: &str = "error";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunArtifactPayload {
    pub phenotype_name: Option<String>,
    pub output_run_directory: Option<String>,
    pub final_dataset: Option<String>,
    pub final_parquet: Option<String>,
    pub final_regenie: Option<String>,
    pub effective_config: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunArtifactsPayload {
    pub output_run_directory: Option<String>,
    pub final_dataset: Option<String>,
    pub final_parquet: Option<String>,
    pub final_regenie: Option<String>,
    pub effective_config: Option<String>,
    pub phenotype_artifacts: Vec<RunArtifactsPayload>,
    pub phenotype_name: Option<String>,
    pub association_mode: Option<String>,
    pub phenotype_count: Option<i64>,
    pub run_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunCompletedEventPayload {
    pub run_id: Option<String>,
    pub association_mode: Option<String>,
    pub phenotype_count: Option<i64>,
    pub artifacts: Vec<RunArtifactPayload>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunInterruptedEventPayload {
    pub signal_number: i64,
    pub signal_name: String,
    pub exit_code: i64,
    pub flushed_for_resume: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunFailedEventPayload {
    pub error_type: String,
    pub error_message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunTelemetryStringField {
    pub key: &'static str,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunArtifactTelemetryFields {
    pub fields: Vec<RunTelemetryStringField>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunCompletedTelemetryFields {
    pub artifact_count: usize,
    pub phenotype_artifacts: Vec<RunArtifactTelemetryFields>,
    pub run_id: Option<String>,
    pub association_mode: Option<String>,
    pub phenotype_count: Option<i64>,
    pub single_artifact: Option<RunArtifactTelemetryFields>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunInterruptedTelemetryFields {
    pub failure_kind: &'static str,
    pub signal_number: i64,
    pub signal_name: String,
    pub exit_code: i64,
    pub flushed_for_resume: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunFailedTelemetryFields {
    pub failure_kind: &'static str,
    pub error_type: String,
    pub error_message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunStartedTelemetryFields {
    pub association_mode: String,
    pub trait_type: String,
    pub phenotype_count: i64,
    pub output_run_root: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlanPreparedTelemetryFields {
    pub association_mode: String,
    pub trait_type: String,
    pub phenotype_count: i64,
    pub chunk_size: i64,
    pub variant_limit: Option<i64>,
    pub device: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectiveConfigWrittenTelemetryFields {
    pub association_mode: String,
    pub phenotype: String,
    pub effective_config: String,
    pub output_run_directory: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhenotypeWriterFinishedTelemetryFields {
    pub association_mode: String,
    pub phenotype: String,
    pub final_output_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiPhenotypeWriterFinishedTelemetryFields {
    pub association_mode: String,
    pub phenotype_count: i64,
    pub final_output_paths: Vec<Option<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SingleTraitPreflightCompletedTelemetryFields {
    pub association_mode: String,
    pub phenotype: String,
    pub sample_count: i64,
    pub covariate_count: i64,
    pub chromosome_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiPhenotypePreflightCompletedTelemetryFields {
    pub association_mode: String,
    pub phenotype_count: i64,
    pub sample_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleAlignmentCompletedTelemetryFields {
    pub association_mode: String,
    pub phenotype: Option<String>,
    pub phenotype_count: Option<i64>,
    pub sample_count: Option<i64>,
    pub covariate_count: Option<i64>,
    pub phenotype_group_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredictionSourceLoadedTelemetryFields {
    pub association_mode: String,
    pub phenotype: Option<String>,
    pub phenotype_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiPhenotypeSampleSummaryTelemetryFields {
    pub association_mode: String,
    pub multi_phenotype_sample_mode: String,
    pub phenotype_count: usize,
    pub phenotype_group_count: i64,
    pub sample_counts: Vec<i64>,
    pub sample_counts_differ: bool,
    pub shared_sample_set: bool,
}

#[must_use]
pub fn build_run_completed_event_from_artifacts(artifacts: &RunArtifactsPayload) -> RunCompletedEventPayload {
    let artifact_payloads = flatten_run_artifact_payloads(artifacts);
    let inferred_phenotype_count =
        if artifact_payloads.len() > 1 { i64::try_from(artifact_payloads.len()).ok() } else { None };
    RunCompletedEventPayload {
        run_id: artifacts.run_id.clone(),
        association_mode: artifacts.association_mode.clone(),
        phenotype_count: artifacts.phenotype_count.or(inferred_phenotype_count),
        artifacts: artifact_payloads,
    }
}

#[must_use]
pub fn attach_run_metadata_to_artifacts(
    artifacts: &RunArtifactsPayload,
    run_id: Option<&str>,
    association_mode: &str,
    phenotype_count: i64,
) -> RunArtifactsPayload {
    RunArtifactsPayload {
        output_run_directory: artifacts.output_run_directory.clone(),
        final_dataset: artifacts.final_dataset.clone(),
        final_parquet: artifacts.final_parquet.clone(),
        final_regenie: artifacts.final_regenie.clone(),
        effective_config: artifacts.effective_config.clone(),
        phenotype_artifacts: artifacts
            .phenotype_artifacts
            .iter()
            .map(|phenotype_artifact| {
                attach_run_metadata_to_artifacts(phenotype_artifact, run_id, association_mode, phenotype_count)
            })
            .collect(),
        phenotype_name: artifacts.phenotype_name.clone(),
        association_mode: Some(association_mode.to_string()),
        phenotype_count: Some(phenotype_count),
        run_id: run_id.map(str::to_string),
    }
}

#[must_use]
pub fn flatten_run_artifact_payloads(artifacts: &RunArtifactsPayload) -> Vec<RunArtifactPayload> {
    if !artifacts.phenotype_artifacts.is_empty() {
        return artifacts.phenotype_artifacts.iter().flat_map(flatten_run_artifact_payloads).collect();
    }
    vec![RunArtifactPayload {
        phenotype_name: artifacts.phenotype_name.clone(),
        output_run_directory: artifacts.output_run_directory.clone(),
        final_dataset: artifacts.final_dataset.clone(),
        final_parquet: artifacts.final_parquet.clone(),
        final_regenie: artifacts.final_regenie.clone(),
        effective_config: artifacts.effective_config.clone(),
    }]
}

#[must_use]
pub fn build_run_interrupted_event_payload(
    signal_number: i64,
    signal_name: &str,
    exit_code: i64,
    flushed_for_resume: bool,
) -> RunInterruptedEventPayload {
    RunInterruptedEventPayload { signal_number, signal_name: signal_name.to_string(), exit_code, flushed_for_resume }
}

#[must_use]
pub fn build_run_failed_event_payload(error_type: &str, error_message: &str) -> RunFailedEventPayload {
    RunFailedEventPayload { error_type: error_type.to_string(), error_message: error_message.to_string() }
}

#[must_use]
pub fn build_run_completed_telemetry_fields(event: &RunCompletedEventPayload) -> RunCompletedTelemetryFields {
    let phenotype_artifacts = event.artifacts.iter().map(build_artifact_telemetry_fields).collect::<Vec<_>>();
    let single_artifact = if phenotype_artifacts.len() == 1 { phenotype_artifacts.first().cloned() } else { None };
    RunCompletedTelemetryFields {
        artifact_count: phenotype_artifacts.len(),
        phenotype_artifacts,
        run_id: event.run_id.clone(),
        association_mode: event.association_mode.clone(),
        phenotype_count: event.phenotype_count,
        single_artifact,
    }
}

#[must_use]
pub fn build_run_interrupted_telemetry_fields(event: &RunInterruptedEventPayload) -> RunInterruptedTelemetryFields {
    RunInterruptedTelemetryFields {
        failure_kind: "graceful_shutdown",
        signal_number: event.signal_number,
        signal_name: event.signal_name.clone(),
        exit_code: event.exit_code,
        flushed_for_resume: event.flushed_for_resume,
    }
}

#[must_use]
pub fn build_run_failed_telemetry_fields(event: &RunFailedEventPayload) -> RunFailedTelemetryFields {
    RunFailedTelemetryFields {
        failure_kind: "exception",
        error_type: event.error_type.clone(),
        error_message: event.error_message.clone(),
    }
}

#[must_use]
pub fn build_run_started_telemetry_fields(
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    output_run_root: &str,
) -> RunStartedTelemetryFields {
    RunStartedTelemetryFields {
        association_mode: association_mode.to_string(),
        trait_type: trait_type.to_string(),
        phenotype_count,
        output_run_root: output_run_root.to_string(),
    }
}

#[must_use]
pub fn build_execution_plan_prepared_telemetry_fields(
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: &str,
) -> ExecutionPlanPreparedTelemetryFields {
    ExecutionPlanPreparedTelemetryFields {
        association_mode: association_mode.to_string(),
        trait_type: trait_type.to_string(),
        phenotype_count,
        chunk_size,
        variant_limit,
        device: device.to_string(),
    }
}

#[must_use]
pub fn build_effective_config_written_telemetry_fields(
    association_mode: &str,
    phenotype: &str,
    effective_config: &str,
    output_run_directory: &str,
) -> EffectiveConfigWrittenTelemetryFields {
    EffectiveConfigWrittenTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype: phenotype.to_string(),
        effective_config: effective_config.to_string(),
        output_run_directory: output_run_directory.to_string(),
    }
}

#[must_use]
pub fn build_phenotype_writer_finished_telemetry_fields(
    association_mode: &str,
    phenotype: &str,
    final_output_path: Option<&str>,
) -> PhenotypeWriterFinishedTelemetryFields {
    PhenotypeWriterFinishedTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype: phenotype.to_string(),
        final_output_path: final_output_path.map(str::to_string),
    }
}

#[must_use]
pub fn build_multi_phenotype_writer_finished_telemetry_fields(
    association_mode: &str,
    phenotype_count: i64,
    final_output_paths: &[Option<String>],
) -> MultiPhenotypeWriterFinishedTelemetryFields {
    MultiPhenotypeWriterFinishedTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype_count,
        final_output_paths: final_output_paths.to_vec(),
    }
}

#[must_use]
pub fn build_single_trait_preflight_completed_telemetry_fields(
    association_mode: &str,
    phenotype: &str,
    sample_count: i64,
    covariate_count: i64,
    chromosome_count: i64,
) -> SingleTraitPreflightCompletedTelemetryFields {
    SingleTraitPreflightCompletedTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype: phenotype.to_string(),
        sample_count,
        covariate_count,
        chromosome_count,
    }
}

#[must_use]
pub fn build_multi_phenotype_preflight_completed_telemetry_fields(
    association_mode: &str,
    phenotype_count: i64,
    sample_count: i64,
) -> MultiPhenotypePreflightCompletedTelemetryFields {
    MultiPhenotypePreflightCompletedTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype_count,
        sample_count,
    }
}

#[must_use]
pub fn build_sample_alignment_completed_telemetry_fields(
    association_mode: &str,
    phenotype: Option<&str>,
    phenotype_count: Option<i64>,
    sample_count: Option<i64>,
    covariate_count: Option<i64>,
    phenotype_group_count: Option<i64>,
) -> SampleAlignmentCompletedTelemetryFields {
    SampleAlignmentCompletedTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype: phenotype.map(str::to_string),
        phenotype_count,
        sample_count,
        covariate_count,
        phenotype_group_count,
    }
}

#[must_use]
pub fn build_prediction_source_loaded_telemetry_fields(
    association_mode: &str,
    phenotype: Option<&str>,
    phenotype_count: Option<i64>,
) -> PredictionSourceLoadedTelemetryFields {
    PredictionSourceLoadedTelemetryFields {
        association_mode: association_mode.to_string(),
        phenotype: phenotype.map(str::to_string),
        phenotype_count,
    }
}

#[must_use]
pub fn build_multi_phenotype_sample_summary_telemetry_fields(
    association_mode: &str,
    multi_phenotype_sample_mode: &str,
    sample_counts: &[i64],
    sample_set_fingerprints: &[Option<String>],
    phenotype_group_count: i64,
) -> MultiPhenotypeSampleSummaryTelemetryFields {
    let sample_counts_differ = sample_counts
        .first()
        .is_some_and(|first_sample_count| sample_counts.iter().any(|sample_count| sample_count != first_sample_count));
    let mut observed_sample_set_fingerprints =
        sample_set_fingerprints.iter().filter_map(|sample_set_fingerprint| sample_set_fingerprint.as_ref());
    let first_observed_sample_set_fingerprint = observed_sample_set_fingerprints.next();
    let shared_sample_set = first_observed_sample_set_fingerprint.is_some_and(|first_sample_set_fingerprint| {
        observed_sample_set_fingerprints
            .all(|sample_set_fingerprint| sample_set_fingerprint == first_sample_set_fingerprint)
    });

    MultiPhenotypeSampleSummaryTelemetryFields {
        association_mode: association_mode.to_string(),
        multi_phenotype_sample_mode: multi_phenotype_sample_mode.to_string(),
        phenotype_count: sample_counts.len(),
        phenotype_group_count,
        sample_counts: sample_counts.to_vec(),
        sample_counts_differ,
        shared_sample_set,
    }
}

#[must_use]
pub fn render_run_completed_lines(event: &RunCompletedEventPayload) -> Vec<String> {
    let mut lines = Vec::new();
    for artifact in &event.artifacts {
        lines.extend(render_artifact_lines(artifact));
    }
    if lines.is_empty() {
        lines.push("Success. Run completed.".to_string());
    }
    lines
}

#[must_use]
pub fn render_run_interrupted_lines(event: &RunInterruptedEventPayload) -> Vec<String> {
    vec![format!(
        "Interrupted by {}. Flushed queued chunks and saved committed output for --resume.",
        event.signal_name
    )]
}

#[must_use]
pub fn render_run_failed_lines(event: &RunFailedEventPayload) -> Vec<String> {
    if event.error_message.is_empty() {
        return vec![format!("Error: {}", event.error_type)];
    }
    vec![format!("Error: {}", event.error_message)]
}

#[must_use]
pub fn build_artifact_telemetry_fields(artifact: &RunArtifactPayload) -> RunArtifactTelemetryFields {
    let mut fields = Vec::new();
    push_optional_field(&mut fields, "phenotype", artifact.phenotype_name.as_ref());
    push_optional_field(&mut fields, "output_run_directory", artifact.output_run_directory.as_ref());
    push_optional_field(&mut fields, "final_dataset", artifact.final_dataset.as_ref());
    push_optional_field(&mut fields, "final_parquet", artifact.final_parquet.as_ref());
    push_optional_field(&mut fields, "final_regenie", artifact.final_regenie.as_ref());
    push_optional_field(&mut fields, "effective_config", artifact.effective_config.as_ref());
    RunArtifactTelemetryFields { fields }
}

#[must_use]
pub fn render_artifact_lines(artifact: &RunArtifactPayload) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(output_run_directory) = artifact.output_run_directory.as_ref() {
        lines.push(format!("Success. Chunked run saved to {output_run_directory}"));
    } else {
        lines.push("Success. Run completed.".to_string());
    }
    if let Some(final_dataset) = artifact.final_dataset.as_ref() {
        lines.push(format!("Parquet dataset saved to {final_dataset}"));
    }
    if let Some(final_parquet) = artifact.final_parquet.as_ref() {
        lines.push(format!("Finalized Parquet saved to {final_parquet}"));
    }
    if let Some(final_regenie) = artifact.final_regenie.as_ref() {
        lines.push(format!("REGENIE text output saved to {final_regenie}"));
    }
    lines
}

fn push_optional_field(fields: &mut Vec<RunTelemetryStringField>, key: &'static str, value: Option<&String>) {
    if let Some(value) = value {
        fields.push(RunTelemetryStringField { key, value: value.clone() });
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
        assert_eq!(
            build_multi_phenotype_sample_summary_telemetry_fields(
                "regenie2_binary",
                "complete-case",
                &[2504, 2504],
                &[Some("shared".to_string()), Some("shared".to_string())],
                1,
            )
            .shared_sample_set,
            true,
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
