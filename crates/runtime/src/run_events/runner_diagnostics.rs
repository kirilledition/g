use super::diagnostics::{
    RunDiagnosticEventPayload, integer_diagnostic_field, optional_integer_diagnostic_field, text_diagnostic_field,
};
use super::names::{
    RUN_LIFECYCLE_INFO_LEVEL, RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE, RUNNER_EXECUTION_PLAN_PREPARED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_METADATA_ARTIFACTS_COMPLETED_DIAGNOSTIC_EVENT_NAME,
};

#[must_use]
pub fn build_runner_execution_plan_build_started_diagnostic_payload() -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: Vec::new(),
    }
}

#[must_use]
pub fn build_runner_execution_plan_prepared_diagnostic_payload(
    association_mode: &str,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_INFO_LEVEL,
        event_name: RUNNER_EXECUTION_PLAN_PREPARED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Prepared REGENIE execution plan for {phenotype_count} phenotype(s)."),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            integer_diagnostic_field("phenotype_count", phenotype_count),
            integer_diagnostic_field("chunk_size", chunk_size),
            optional_integer_diagnostic_field("variant_limit", variant_limit),
            text_diagnostic_field("device", device),
        ],
    }
}

#[must_use]
pub fn build_runner_execution_plan_dispatch_started_diagnostic_payload(
    phenotype_count: i64,
    association_mode: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("association_mode", association_mode),
        ],
    }
}

#[must_use]
pub fn build_runner_metadata_artifacts_completed_diagnostic_payload(
    association_mode: &str,
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: RUNNER_METADATA_ARTIFACTS_COMPLETED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Completed REGENIE run artifacts for {phenotype_count} phenotype(s)."),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            integer_diagnostic_field("phenotype_count", phenotype_count),
        ],
    }
}
