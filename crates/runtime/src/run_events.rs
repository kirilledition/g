//! Runtime-owned run lifecycle event payloads and rendering policy.

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
