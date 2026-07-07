use std::collections::BTreeMap;

use serde::{Serialize, Serializer};

use super::queue_backpressure::QueueBackpressureSnapshot;
use super::transfer_metadata::TransferMetadataSnapshot;

#[derive(Clone, Debug, PartialEq)]
pub enum NumericDiagnosticValue {
    Integer(i64),
    Float(f64),
}

impl NumericDiagnosticValue {
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Integer(value) => *value as f64,
            Self::Float(value) => *value,
        }
    }
}

impl Serialize for NumericDiagnosticValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Integer(value) => serializer.serialize_i64(*value),
            Self::Float(value) => serializer.serialize_f64(*value),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NullLogisticDiagnosticValue {
    Integer(i64),
    Text(String),
}

impl Serialize for NullLogisticDiagnosticValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Integer(value) => serializer.serialize_i64(*value),
            Self::Text(value) => serializer.serialize_str(value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChunkStageTiming {
    pub chunk_identifier: i64,
    pub chromosome: String,
    pub variant_start_index: i64,
    pub variant_stop_index: i64,
    pub variant_count: i64,
    pub stage_name: String,
    pub duration_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChunkStageSummary {
    pub total_seconds: f64,
    pub count: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct NullLogisticSummary {
    pub chromosome_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ProfileSummaryPayload {
    pub schema_version: i64,
    pub run_id: Option<String>,
    pub stage_totals_seconds: BTreeMap<String, f64>,
    pub stage_counts: BTreeMap<String, i64>,
    pub native_bgen_profile: BTreeMap<String, i64>,
    pub derived_metrics: BTreeMap<String, f64>,
    pub chunk_stage_summary: BTreeMap<String, ChunkStageSummary>,
    pub binary_chunk_summary: BTreeMap<String, NumericDiagnosticValue>,
    pub queue_backpressure: Vec<QueueBackpressureSnapshot>,
    pub transfer_metadata: Vec<TransferMetadataSnapshot>,
    pub null_logistic_summary: NullLogisticSummary,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StageTimingSnapshotPayload {
    pub stage_totals_seconds: BTreeMap<String, f64>,
    pub stage_counts: BTreeMap<String, i64>,
    pub chunk_stage_timings: Vec<ChunkStageTiming>,
    pub native_bgen_profile: BTreeMap<String, i64>,
    pub binary_chunk_diagnostics: Vec<BTreeMap<String, NumericDiagnosticValue>>,
    pub null_logistic_diagnostics: Vec<BTreeMap<String, NullLogisticDiagnosticValue>>,
    pub queue_backpressure: Vec<QueueBackpressureSnapshot>,
    pub transfer_metadata: Vec<TransferMetadataSnapshot>,
    pub derived_metrics: BTreeMap<String, f64>,
}
