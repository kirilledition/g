use super::diagnostics::{
    RunDiagnosticEventPayload, boolean_diagnostic_field, integer_diagnostic_field, optional_integer_diagnostic_field,
    optional_text_diagnostic_field, text_diagnostic_field,
};
use super::{
    RUN_LIFECYCLE_ERROR_LEVEL, RUN_LIFECYCLE_INFO_LEVEL, RUN_LIFECYCLE_WARN_LEVEL,
    RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_EVENT_NAME, RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_EXECUTION_PLAN_PREPARED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_JAX_RUNTIME_CONFIGURATION_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_JAX_RUNTIME_CONFIGURATION_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_METADATA_ARTIFACTS_FINALIZED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
    RUNNER_REGENIE_RUN_COMPLETED_DIAGNOSTIC_EVENT_NAME, RUNNER_REGENIE_RUN_COMPLETED_DIAGNOSTIC_MESSAGE,
    RUNNER_REGENIE_RUN_FAILED_DIAGNOSTIC_EVENT_NAME, RUNNER_REGENIE_RUN_FAILED_DIAGNOSTIC_MESSAGE,
    RUNNER_REGENIE_RUN_INTERRUPTED_DIAGNOSTIC_EVENT_NAME, RUNNER_REGENIE_RUN_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_REGENIE_RUN_STARTED_DIAGNOSTIC_MESSAGE, RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
    RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE,
};
use super::{RunCompletedEventPayload, RunFailedEventPayload, RunInterruptedEventPayload};

#[must_use]
pub fn build_runner_run_started_diagnostic_payload(
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_INFO_LEVEL,
        event_name: RUNNER_REGENIE_RUN_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_REGENIE_RUN_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            text_diagnostic_field("trait_type", trait_type),
            integer_diagnostic_field("phenotype_count", phenotype_count),
        ],
    }
}

#[must_use]
pub fn build_runner_run_interrupted_diagnostic_payload(
    event: &RunInterruptedEventPayload,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_WARN_LEVEL,
        event_name: RUNNER_REGENIE_RUN_INTERRUPTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("REGENIE run interrupted by {}.", event.signal_name),
        fields: vec![
            integer_diagnostic_field("signal_number", event.signal_number),
            text_diagnostic_field("signal_name", &event.signal_name),
            integer_diagnostic_field("exit_code", event.exit_code),
            boolean_diagnostic_field("flushed_for_resume", event.flushed_for_resume),
        ],
    }
}

#[must_use]
pub fn build_runner_run_failed_diagnostic_payload(event: &RunFailedEventPayload) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_ERROR_LEVEL,
        event_name: RUNNER_REGENIE_RUN_FAILED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_REGENIE_RUN_FAILED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            text_diagnostic_field("error_type", &event.error_type),
            text_diagnostic_field("error_message", &event.error_message),
        ],
    }
}

#[must_use]
pub fn build_runner_run_completed_diagnostic_payload(event: &RunCompletedEventPayload) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_INFO_LEVEL,
        event_name: RUNNER_REGENIE_RUN_COMPLETED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_REGENIE_RUN_COMPLETED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            optional_text_diagnostic_field("run_id", event.run_id.clone()),
            optional_text_diagnostic_field("association_mode", event.association_mode.clone()),
            optional_integer_diagnostic_field("phenotype_count", event.phenotype_count),
        ],
    }
}

#[must_use]
pub fn build_runner_jax_runtime_configuration_started_diagnostic_payload() -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_JAX_RUNTIME_CONFIGURATION_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_JAX_RUNTIME_CONFIGURATION_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: Vec::new(),
    }
}

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
pub fn build_runner_execution_plan_finalization_started_diagnostic_payload(
    phenotype_count: i64,
    association_mode: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("association_mode", association_mode),
        ],
    }
}

#[must_use]
pub fn build_runner_multi_phenotype_dispatch_started_diagnostic_payload(
    phenotype_count: i64,
    association_mode: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("association_mode", association_mode),
        ],
    }
}

#[must_use]
pub fn build_runner_single_phenotype_dispatch_started_diagnostic_payload(
    association_mode: &str,
    phenotype: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            text_diagnostic_field("phenotype", phenotype),
        ],
    }
}

#[must_use]
pub fn build_runner_binary_engine_dispatch_started_diagnostic_payload(phenotype: &str) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![text_diagnostic_field("phenotype", phenotype)],
    }
}

#[must_use]
pub fn build_runner_linear_engine_dispatch_started_diagnostic_payload(phenotype: &str) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![text_diagnostic_field("phenotype", phenotype)],
    }
}

#[must_use]
pub fn build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload(
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![integer_diagnostic_field("phenotype_count", phenotype_count)],
    }
}

#[must_use]
pub fn build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload(
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![integer_diagnostic_field("phenotype_count", phenotype_count)],
    }
}

#[must_use]
pub fn build_runner_metadata_artifacts_finalized_diagnostic_payload(
    association_mode: &str,
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: RUNNER_METADATA_ARTIFACTS_FINALIZED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Finalized REGENIE run artifacts for {phenotype_count} phenotype(s)."),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            integer_diagnostic_field("phenotype_count", phenotype_count),
        ],
    }
}
