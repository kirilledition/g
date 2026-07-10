#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhenotypeRunArtifacts {
    pub output_run_directory: String,
    pub parquet_dataset_directory: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunInterruptedEventPayload {
    pub signal_name: String,
    pub exit_code: i32,
    pub flushed_for_resume: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunFailedEventPayload {
    pub error_type: String,
    pub error_message: String,
}

#[derive(serde::Serialize)]
pub(crate) struct RunFailedTelemetryFields {
    failure_kind: &'static str,
    error_type: String,
    error_message: String,
}

#[must_use]
pub fn build_run_interrupted_event_payload(
    signal_name: &str,
    exit_code: i32,
    flushed_for_resume: bool,
) -> RunInterruptedEventPayload {
    RunInterruptedEventPayload { signal_name: signal_name.to_string(), exit_code, flushed_for_resume }
}

#[must_use]
pub fn build_run_failed_event_payload(error_type: &str, error_message: &str) -> RunFailedEventPayload {
    RunFailedEventPayload { error_type: error_type.to_string(), error_message: error_message.to_string() }
}

pub(crate) fn build_run_failed_telemetry_fields(event: &RunFailedEventPayload) -> RunFailedTelemetryFields {
    RunFailedTelemetryFields {
        failure_kind: "exception",
        error_type: event.error_type.clone(),
        error_message: event.error_message.clone(),
    }
}

#[must_use]
pub fn render_run_completed_lines(artifacts: &[PhenotypeRunArtifacts]) -> Vec<String> {
    let mut lines = Vec::new();
    for artifact in artifacts {
        append_artifact_lines(artifact, &mut lines);
    }
    if lines.is_empty() {
        lines.push("Success. Run completed.".to_string());
    }
    lines
}

#[must_use]
pub fn render_run_interrupted_lines(event: &RunInterruptedEventPayload) -> Vec<String> {
    if event.flushed_for_resume {
        return vec![format!(
            "Interrupted by {}. Flushed queued chunks and saved committed output for resume.",
            event.signal_name
        )];
    }
    vec![format!("Interrupted by {}.", event.signal_name)]
}

#[must_use]
pub fn render_run_failed_lines(event: &RunFailedEventPayload) -> Vec<String> {
    if event.error_message.is_empty() {
        return vec![format!("Error: {}", event.error_type)];
    }
    vec![format!("Error: {}", event.error_message)]
}

fn append_artifact_lines(artifact: &PhenotypeRunArtifacts, lines: &mut Vec<String>) {
    lines.push(format!("Success. Run saved to {}", artifact.output_run_directory));
    lines.push(format!("Parquet dataset saved to {}", artifact.parquet_dataset_directory));
}
