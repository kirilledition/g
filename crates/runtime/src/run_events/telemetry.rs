use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RunTelemetryEventKind {
    AssociationBackendSelected,
    ExecutionPlanPrepared,
    RunFailed,
    WriterFinished,
}

impl RunTelemetryEventKind {
    pub(crate) const fn event_name(self) -> &'static str {
        match self {
            Self::AssociationBackendSelected => super::names::ASSOCIATION_BACKEND_SELECTED_EVENT_NAME,
            Self::ExecutionPlanPrepared => super::names::EXECUTION_PLAN_PREPARED_EVENT_NAME,
            Self::RunFailed => super::names::RUN_FAILED_EVENT_NAME,
            Self::WriterFinished => super::names::WRITER_FINISHED_EVENT_NAME,
        }
    }

    pub(crate) const fn level(self) -> &'static str {
        match self {
            Self::RunFailed => super::names::RUN_LIFECYCLE_ERROR_LEVEL,
            _ => super::names::RUN_LIFECYCLE_INFO_LEVEL,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ExecutionPlanPreparedTelemetryFields {
    association_mode: String,
    trait_type: String,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: String,
}

#[derive(Serialize)]
pub(crate) struct PhenotypeWriterFinishedTelemetryFields<'a> {
    association_mode: &'a str,
    phenotype: &'a str,
    parquet_dataset_path: &'a str,
}

#[derive(Serialize)]
pub(crate) struct MultiPhenotypeWriterFinishedTelemetryFields<'a> {
    association_mode: &'a str,
    phenotype_count: i64,
    parquet_dataset_paths: &'a [&'a str],
}

#[derive(Serialize)]
pub(crate) struct AssociationBackendSelectedTelemetryFields {
    association_mode: String,
    association_backend_kind: String,
    device: String,
    genotype_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    phenotype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phenotype_count: Option<i64>,
}

pub(crate) fn build_execution_plan_prepared_telemetry_fields(
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

pub(crate) const fn build_phenotype_writer_finished_telemetry_fields<'a>(
    association_mode: &'a str,
    phenotype: &'a str,
    parquet_dataset_path: &'a str,
) -> PhenotypeWriterFinishedTelemetryFields<'a> {
    PhenotypeWriterFinishedTelemetryFields { association_mode, phenotype, parquet_dataset_path }
}

pub(crate) const fn build_multi_phenotype_writer_finished_telemetry_fields<'a>(
    association_mode: &'a str,
    phenotype_count: i64,
    parquet_dataset_paths: &'a [&'a str],
) -> MultiPhenotypeWriterFinishedTelemetryFields<'a> {
    MultiPhenotypeWriterFinishedTelemetryFields { association_mode, phenotype_count, parquet_dataset_paths }
}

pub(crate) fn build_association_backend_selected_telemetry_fields(
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
