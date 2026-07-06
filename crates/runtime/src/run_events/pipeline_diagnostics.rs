use super::diagnostics::{
    RunDiagnosticEventPayload, boolean_diagnostic_field, integer_diagnostic_field, optional_integer_diagnostic_field,
    optional_text_diagnostic_field, text_diagnostic_field,
};
use super::{
    PIPELINE_BGEN_ENGINE_OPEN_STARTED_DIAGNOSTIC_EVENT_NAME, PIPELINE_BGEN_ENGINE_OPENED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_GPU_GENOTYPE_FORMAT_RESOLVED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_GROUPED_PER_PHENOTYPE_GROUPS_PREPARED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_GROUPED_PER_PHENOTYPE_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_GROUPED_UNION_DELIVERY_SELECTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_MESSAGE,
    PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_MESSAGE,
    PIPELINE_MULTI_PHENOTYPE_SAMPLE_SUMMARY_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_MULTI_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME, PIPELINE_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_OUTPUT_WRITER_SESSIONS_CREATE_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_PREVALIDATED_BGEN_ENGINE_USED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_SINGLE_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_SINGLE_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_SINGLE_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_SINGLE_TRAIT_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME,
    PIPELINE_SINGLE_TRAIT_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME, PIPELINE_SINGLE_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME,
};

#[must_use]
pub fn build_pipeline_bgen_engine_open_started_diagnostic_payload(
    phenotype_count: Option<i64>,
    phenotype_name: Option<&str>,
    pipeline_label: &str,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_BGEN_ENGINE_OPEN_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Opening native BGEN engine for {pipeline_label} pipeline."),
        fields: vec![
            optional_integer_diagnostic_field("phenotype_count", phenotype_count),
            optional_text_diagnostic_field("phenotype_name", phenotype_name.map(str::to_string)),
            text_diagnostic_field("pipeline_label", pipeline_label),
            boolean_diagnostic_field("trusted_no_missing_diploid", trusted_no_missing_diploid),
            optional_integer_diagnostic_field("variant_limit", variant_limit),
        ],
    }
}

#[must_use]
pub fn build_pipeline_bgen_engine_opened_diagnostic_payload(
    phenotype_count: Option<i64>,
    phenotype_name: Option<&str>,
    pipeline_label: &str,
    sample_count: i64,
    variant_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_BGEN_ENGINE_OPENED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Native BGEN engine opened for {pipeline_label} pipeline: sample_count={sample_count} \
             variant_count={variant_count}."
        ),
        fields: vec![
            optional_integer_diagnostic_field("phenotype_count", phenotype_count),
            optional_text_diagnostic_field("phenotype_name", phenotype_name.map(str::to_string)),
            text_diagnostic_field("pipeline_label", pipeline_label),
            integer_diagnostic_field("sample_count", sample_count),
            integer_diagnostic_field("variant_count", variant_count),
        ],
    }
}

#[must_use]
pub fn build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload(
    phenotype_count: Option<i64>,
    phenotype_name: Option<&str>,
    pipeline_label: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_PREVALIDATED_BGEN_ENGINE_USED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Using prevalidated native BGEN engine for {pipeline_label} pipeline."),
        fields: vec![
            optional_integer_diagnostic_field("phenotype_count", phenotype_count),
            optional_text_diagnostic_field("phenotype_name", phenotype_name.map(str::to_string)),
            text_diagnostic_field("pipeline_label", pipeline_label),
        ],
    }
}

#[must_use]
pub fn build_pipeline_output_resume_committed_chunks_diagnostic_payload(
    committed_chunk_count: i64,
    output_index: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME,
        message: format!("Resuming run with {committed_chunk_count} previously committed chunks."),
        fields: vec![
            integer_diagnostic_field("committed_chunk_count", committed_chunk_count),
            integer_diagnostic_field("output_index", output_index),
        ],
    }
}

#[must_use]
pub fn build_pipeline_output_writer_sessions_create_started_diagnostic_payload(
    association_mode: &str,
    output_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_OUTPUT_WRITER_SESSIONS_CREATE_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Creating output writer(s) for {association_mode} pipeline."),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            integer_diagnostic_field("output_count", output_count),
        ],
    }
}

#[must_use]
pub fn build_pipeline_gpu_genotype_format_resolved_diagnostic_payload(
    requested_gpu_genotype_format: &str,
    resolved_gpu_genotype_format: &str,
    resolution_reason: &str,
    fallback_error: Option<&str>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_GPU_GENOTYPE_FORMAT_RESOLVED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Resolved gpu_genotype_format={requested_gpu_genotype_format} to {resolved_gpu_genotype_format}: \
             {resolution_reason}."
        ),
        fields: vec![
            optional_text_diagnostic_field("fallback_error", fallback_error.map(str::to_string)),
            text_diagnostic_field("requested_gpu_genotype_format", requested_gpu_genotype_format),
            text_diagnostic_field("resolution_reason", resolution_reason),
            text_diagnostic_field("resolved_gpu_genotype_format", resolved_gpu_genotype_format),
        ],
    }
}

#[must_use]
pub fn build_pipeline_multi_phenotype_sample_summary_diagnostic_payload(
    phenotype_count: i64,
    phenotype_group_count: i64,
    sample_counts_differ: bool,
    sample_mode: &str,
) -> RunDiagnosticEventPayload {
    let message = if sample_mode == "complete-case" {
        format!("Analyzed {phenotype_count} phenotypes in complete-case sample mode; one shared sample set was used.")
    } else {
        let sample_count_summary = if sample_counts_differ {
            "sample counts differ across phenotypes"
        } else {
            "sample counts do not differ across phenotypes"
        };
        format!("Analyzed {phenotype_count} phenotypes in per-phenotype sample mode; {sample_count_summary}.")
    };
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_MULTI_PHENOTYPE_SAMPLE_SUMMARY_DIAGNOSTIC_EVENT_NAME,
        message,
        fields: vec![
            integer_diagnostic_field("phenotype_count", phenotype_count),
            integer_diagnostic_field("phenotype_group_count", phenotype_group_count),
            boolean_diagnostic_field("sample_counts_differ", sample_counts_differ),
            text_diagnostic_field("sample_mode", sample_mode),
        ],
    }
}

#[must_use]
pub fn build_pipeline_multi_trait_started_diagnostic_payload(
    association_mode: &str,
    phenotype_count: i64,
    sample_mode: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_MULTI_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: "Starting multi-phenotype REGENIE step 2 BGEN pipeline.".to_string(),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("sample_mode", sample_mode),
        ],
    }
}

#[must_use]
pub fn build_pipeline_multi_trait_input_load_started_diagnostic_payload(
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_MULTI_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: "Loading aligned native sample, phenotype, and covariate inputs for multi-phenotype pipeline."
            .to_string(),
        fields: vec![integer_diagnostic_field("phenotype_count", phenotype_count)],
    }
}

#[must_use]
pub fn build_pipeline_multi_trait_input_aligned_diagnostic_payload(
    covariate_count: i64,
    phenotype_count: i64,
    sample_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_MULTI_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Aligned multi-phenotype pipeline inputs: sample_count={sample_count} \
             phenotype_count={phenotype_count} covariate_count={covariate_count}."
        ),
        fields: vec![
            integer_diagnostic_field("covariate_count", covariate_count),
            integer_diagnostic_field("phenotype_count", phenotype_count),
            integer_diagnostic_field("sample_count", sample_count),
        ],
    }
}

#[must_use]
pub fn build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload(
    phenotype_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_MULTI_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: "Loading REGENIE prediction source for multi-phenotype pipeline.".to_string(),
        fields: vec![integer_diagnostic_field("phenotype_count", phenotype_count)],
    }
}

#[must_use]
pub fn build_pipeline_grouped_per_phenotype_started_diagnostic_payload(
    association_mode: &str,
    phenotype_count: i64,
    sample_mode: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_GROUPED_PER_PHENOTYPE_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: "Starting grouped per-phenotype REGENIE step 2 BGEN pipeline.".to_string(),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("sample_mode", sample_mode),
        ],
    }
}

#[must_use]
pub fn build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload(
    phenotype_count: i64,
    phenotype_group_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_GROUPED_PER_PHENOTYPE_GROUPS_PREPARED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Prepared {phenotype_group_count} compatible per-phenotype group(s) for {phenotype_count} phenotype(s)."
        ),
        fields: vec![
            integer_diagnostic_field("phenotype_count", phenotype_count),
            integer_diagnostic_field("phenotype_group_count", phenotype_group_count),
        ],
    }
}

#[must_use]
pub fn build_pipeline_grouped_union_delivery_selected_diagnostic_payload(
    grouped_sample_count: i64,
    phenotype_group_count: i64,
    union_sample_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_GROUPED_UNION_DELIVERY_SELECTED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Using union per-phenotype BGEN delivery: group_count={phenotype_group_count} \
             union_sample_count={union_sample_count} grouped_sample_count={grouped_sample_count}."
        ),
        fields: vec![
            integer_diagnostic_field("grouped_sample_count", grouped_sample_count),
            integer_diagnostic_field("phenotype_group_count", phenotype_group_count),
            integer_diagnostic_field("union_sample_count", union_sample_count),
        ],
    }
}

#[must_use]
pub fn build_pipeline_multi_group_preflight_started_diagnostic_payload(
    phenotype_count: i64,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> RunDiagnosticEventPayload {
    build_pipeline_multi_group_preflight_diagnostic_payload(
        PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME,
        PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_MESSAGE,
        phenotype_count,
        sample_count,
        trusted_no_missing_diploid,
        variant_limit,
    )
}

#[must_use]
pub fn build_pipeline_multi_group_preflight_completed_diagnostic_payload(
    phenotype_count: i64,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> RunDiagnosticEventPayload {
    build_pipeline_multi_group_preflight_diagnostic_payload(
        PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME,
        PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_MESSAGE,
        phenotype_count,
        sample_count,
        trusted_no_missing_diploid,
        variant_limit,
    )
}

#[must_use]
pub fn build_pipeline_single_trait_started_diagnostic_payload(
    association_mode: &str,
    phenotype_name: &str,
    pipeline_label: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: PIPELINE_SINGLE_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Starting {pipeline_label} REGENIE step 2 BGEN pipeline."),
        fields: vec![
            text_diagnostic_field("association_mode", association_mode),
            text_diagnostic_field("phenotype_name", phenotype_name),
            text_diagnostic_field("pipeline_label", pipeline_label),
        ],
    }
}

#[must_use]
pub fn build_pipeline_single_trait_input_load_started_diagnostic_payload(
    phenotype_name: &str,
    pipeline_label: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_SINGLE_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Loading aligned native sample, phenotype, and covariate inputs for {pipeline_label} pipeline."
        ),
        fields: vec![
            text_diagnostic_field("phenotype_name", phenotype_name),
            text_diagnostic_field("pipeline_label", pipeline_label),
        ],
    }
}

#[must_use]
pub fn build_pipeline_single_trait_input_aligned_diagnostic_payload(
    covariate_count: i64,
    phenotype_name: &str,
    pipeline_label: &str,
    sample_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_SINGLE_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Aligned {pipeline_label} pipeline inputs: sample_count={sample_count} \
             covariate_count={covariate_count}."
        ),
        fields: vec![
            integer_diagnostic_field("covariate_count", covariate_count),
            text_diagnostic_field("phenotype_name", phenotype_name),
            text_diagnostic_field("pipeline_label", pipeline_label),
            integer_diagnostic_field("sample_count", sample_count),
        ],
    }
}

#[must_use]
pub fn build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload(
    phenotype_name: &str,
    pipeline_label: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_SINGLE_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Loading REGENIE prediction source for {pipeline_label} pipeline."),
        fields: vec![
            text_diagnostic_field("phenotype_name", phenotype_name),
            text_diagnostic_field("pipeline_label", pipeline_label),
        ],
    }
}

#[must_use]
pub fn build_pipeline_single_trait_preflight_started_diagnostic_payload(
    phenotype_name: &str,
    pipeline_label: &str,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_SINGLE_TRAIT_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Running preflight validation for {pipeline_label} pipeline."),
        fields: vec![
            text_diagnostic_field("phenotype_name", phenotype_name),
            text_diagnostic_field("pipeline_label", pipeline_label),
            boolean_diagnostic_field("trusted_no_missing_diploid", trusted_no_missing_diploid),
            optional_integer_diagnostic_field("variant_limit", variant_limit),
        ],
    }
}

#[must_use]
pub fn build_pipeline_single_trait_preflight_completed_diagnostic_payload(
    chromosome_count: i64,
    covariate_count: i64,
    phenotype_name: &str,
    pipeline_label: &str,
    sample_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: PIPELINE_SINGLE_TRAIT_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME,
        message: format!(
            "Preflight validation passed for {pipeline_label} pipeline: sample_count={sample_count} \
             covariate_count={covariate_count} chromosome_count={chromosome_count}."
        ),
        fields: vec![
            integer_diagnostic_field("chromosome_count", chromosome_count),
            integer_diagnostic_field("covariate_count", covariate_count),
            text_diagnostic_field("phenotype_name", phenotype_name),
            text_diagnostic_field("pipeline_label", pipeline_label),
            integer_diagnostic_field("sample_count", sample_count),
        ],
    }
}

fn build_pipeline_multi_group_preflight_diagnostic_payload(
    event_name: &'static str,
    message: &str,
    phenotype_count: i64,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name,
        message: message.to_string(),
        fields: vec![
            integer_diagnostic_field("phenotype_count", phenotype_count),
            integer_diagnostic_field("sample_count", sample_count),
            boolean_diagnostic_field("trusted_no_missing_diploid", trusted_no_missing_diploid),
            optional_integer_diagnostic_field("variant_limit", variant_limit),
        ],
    }
}
