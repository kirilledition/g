mod kind;

use serde::Serialize;

pub use kind::RunTelemetryEventKind;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunStartedTelemetryFields {
    pub association_mode: String,
    pub trait_type: String,
    pub phenotype_count: i64,
    pub output_run_root: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionPlanPreparedTelemetryFields {
    pub association_mode: String,
    pub trait_type: String,
    pub phenotype_count: i64,
    pub chunk_size: i64,
    pub variant_limit: Option<i64>,
    pub device: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EffectiveConfigWrittenTelemetryFields {
    pub association_mode: String,
    pub phenotype: String,
    pub effective_config: String,
    pub output_run_directory: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhenotypeWriterFinishedTelemetryFields {
    pub association_mode: String,
    pub phenotype: String,
    pub final_output_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MultiPhenotypeWriterFinishedTelemetryFields {
    pub association_mode: String,
    pub phenotype_count: i64,
    pub final_output_paths: Vec<Option<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SingleTraitPreflightCompletedTelemetryFields {
    pub association_mode: String,
    pub phenotype: String,
    pub sample_count: i64,
    pub covariate_count: i64,
    pub chromosome_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MultiPhenotypePreflightCompletedTelemetryFields {
    pub association_mode: String,
    pub phenotype_count: i64,
    pub sample_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SampleAlignmentCompletedTelemetryFields {
    pub association_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covariate_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype_group_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PredictionSourceLoadedTelemetryFields {
    pub association_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MultiPhenotypeSampleSummaryTelemetryFields {
    pub association_mode: String,
    pub multi_phenotype_sample_mode: String,
    pub phenotype_count: usize,
    pub phenotype_group_count: i64,
    pub sample_counts: Vec<i64>,
    pub sample_counts_differ: bool,
    pub shared_sample_set: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuGenotypeFormatResolvedTelemetryFields {
    pub requested_gpu_genotype_format: String,
    pub resolved_gpu_genotype_format: String,
    pub resolution_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AssociationBackendSelectedTelemetryFields {
    pub association_mode: String,
    pub association_backend_kind: String,
    pub device: String,
    pub genotype_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype_count: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BgenEngineOpenedTelemetryFields {
    pub association_mode: String,
    pub association_backend_kind: String,
    pub sample_count: i64,
    pub variant_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phenotype_count: Option<i64>,
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
pub fn build_gpu_genotype_format_resolved_telemetry_fields(
    requested_gpu_genotype_format: &str,
    resolved_gpu_genotype_format: &str,
    resolution_reason: &str,
    fallback_error: Option<&str>,
) -> GpuGenotypeFormatResolvedTelemetryFields {
    GpuGenotypeFormatResolvedTelemetryFields {
        requested_gpu_genotype_format: requested_gpu_genotype_format.to_string(),
        resolved_gpu_genotype_format: resolved_gpu_genotype_format.to_string(),
        resolution_reason: resolution_reason.to_string(),
        fallback_error: fallback_error.map(str::to_string),
    }
}

#[must_use]
pub fn build_association_backend_selected_telemetry_fields(
    association_mode: &str,
    association_backend_kind: &str,
    device: &str,
    genotype_format: &str,
    phenotype: Option<&str>,
    phenotype_count: Option<i64>,
) -> AssociationBackendSelectedTelemetryFields {
    AssociationBackendSelectedTelemetryFields {
        association_mode: association_mode.to_string(),
        association_backend_kind: association_backend_kind.to_string(),
        device: device.to_string(),
        genotype_format: genotype_format.to_string(),
        phenotype: phenotype.map(str::to_string),
        phenotype_count,
    }
}

#[must_use]
pub fn build_bgen_engine_opened_telemetry_fields(
    association_mode: &str,
    association_backend_kind: &str,
    sample_count: i64,
    variant_count: i64,
    phenotype: Option<&str>,
    phenotype_count: Option<i64>,
) -> BgenEngineOpenedTelemetryFields {
    BgenEngineOpenedTelemetryFields {
        association_mode: association_mode.to_string(),
        association_backend_kind: association_backend_kind.to_string(),
        sample_count,
        variant_count,
        phenotype: phenotype.map(str::to_string),
        phenotype_count,
    }
}
