//! Native stage timing recorder state and aggregate bookkeeping.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Serialize, Serializer};

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct QueueBackpressureKey {
    pub queue_name: String,
    pub operation_name: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct TransferMetadataKey {
    pub transfer_name: String,
    pub array_role: String,
    pub dtype_name: String,
    pub dimension_count: i64,
}

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
pub struct QueueBackpressureAccumulator {
    pub observation_count: i64,
    pub max_depth: i64,
    pub max_capacity: i64,
    pub total_elapsed_seconds: f64,
    pub total_blocked_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TransferMetadataAccumulator {
    pub observation_count: i64,
    pub total_bytes: i64,
    pub max_bytes: i64,
    pub total_elements: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferMetadataObservation {
    pub key: TransferMetadataKey,
    pub byte_count: i64,
    pub element_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct QueueBackpressureSnapshot {
    pub queue_name: String,
    pub operation_name: String,
    pub observation_count: i64,
    pub max_depth: i64,
    pub max_capacity: i64,
    pub total_elapsed_seconds: f64,
    pub total_blocked_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TransferMetadataSnapshot {
    pub transfer_name: String,
    pub array_role: String,
    pub dtype_name: String,
    pub dimension_count: i64,
    pub observation_count: i64,
    pub total_bytes: i64,
    pub max_bytes: i64,
    pub total_elements: i64,
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageTimingState {
    pub stage_totals_seconds: BTreeMap<String, f64>,
    pub stage_counts: BTreeMap<String, i64>,
    pub chunk_stage_timings: Vec<ChunkStageTiming>,
    pub native_bgen_profile: BTreeMap<String, i64>,
    pub binary_chunk_diagnostics: Vec<BTreeMap<String, NumericDiagnosticValue>>,
    pub null_logistic_diagnostics: Vec<BTreeMap<String, NullLogisticDiagnosticValue>>,
    pub queue_backpressure: BTreeMap<QueueBackpressureKey, QueueBackpressureAccumulator>,
    pub transfer_metadata: BTreeMap<TransferMetadataKey, TransferMetadataAccumulator>,
}

#[derive(Debug)]
pub enum TimingFileError {
    CreateParentDirectory { path: PathBuf, source: std::io::Error },
    Serialize { source: serde_json::Error },
    WriteFile { path: PathBuf, source: std::io::Error },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransferMetadataError {
    NegativeDimension { dimension: i64 },
    NonPositiveItemSize { item_size: i64 },
    DimensionCountOverflow { dimension_count: usize },
    ElementCountOverflow,
    ByteCountOverflow,
}

impl fmt::Display for TimingFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateParentDirectory { path, source } => {
                write!(formatter, "failed to create timing file parent directory for {}: {source}", path.display())
            }
            Self::Serialize { source } => write!(formatter, "failed to serialize timing payload: {source}"),
            Self::WriteFile { path, source } => {
                write!(formatter, "failed to write timing file {}: {source}", path.display())
            }
        }
    }
}

impl Error for TimingFileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CreateParentDirectory { source, .. } | Self::WriteFile { source, .. } => Some(source),
            Self::Serialize { source } => Some(source),
        }
    }
}

impl fmt::Display for TransferMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeDimension { dimension } => {
                write!(formatter, "Transfer metadata shape dimensions must be nonnegative: {dimension}")
            }
            Self::NonPositiveItemSize { item_size } => {
                write!(formatter, "Transfer metadata dtype item size must be positive: {item_size}")
            }
            Self::DimensionCountOverflow { dimension_count } => {
                write!(formatter, "Transfer metadata dimension count exceeds platform capacity: {dimension_count}")
            }
            Self::ElementCountOverflow => write!(formatter, "Transfer metadata element count exceeds i64 capacity."),
            Self::ByteCountOverflow => write!(formatter, "Transfer metadata byte count exceeds i64 capacity."),
        }
    }
}

impl Error for TransferMetadataError {}

#[must_use]
pub const fn should_collect_exact_stage_timings(exact_stage_timings: bool) -> bool {
    exact_stage_timings
}

/// Build one transfer metadata observation from array adapter fields.
///
/// # Errors
///
/// Returns an error when the dtype item size is non-positive, any dimension is
/// negative, or the dimension/element/byte counts exceed `i64`.
pub fn build_transfer_metadata_observation(
    transfer_name: &str,
    array_role: &str,
    dtype_name: &str,
    shape_dimensions: &[i64],
    item_size: i64,
) -> Result<TransferMetadataObservation, TransferMetadataError> {
    if item_size <= 0 {
        return Err(TransferMetadataError::NonPositiveItemSize { item_size });
    }
    let dimension_count = i64::try_from(shape_dimensions.len())
        .map_err(|_| TransferMetadataError::DimensionCountOverflow { dimension_count: shape_dimensions.len() })?;
    let mut element_count = 1_i64;
    for dimension in shape_dimensions {
        if *dimension < 0 {
            return Err(TransferMetadataError::NegativeDimension { dimension: *dimension });
        }
        element_count = element_count.checked_mul(*dimension).ok_or(TransferMetadataError::ElementCountOverflow)?;
    }
    let byte_count = element_count.checked_mul(item_size).ok_or(TransferMetadataError::ByteCountOverflow)?;
    Ok(TransferMetadataObservation {
        key: TransferMetadataKey {
            transfer_name: transfer_name.to_string(),
            array_role: array_role.to_string(),
            dtype_name: dtype_name.to_string(),
            dimension_count,
        },
        byte_count,
        element_count,
    })
}

/// Write a stage timing snapshot payload as pretty JSON.
///
/// # Errors
///
/// Returns an error when the parent directory cannot be created, the payload
/// cannot be serialized, or the file cannot be written.
pub fn write_stage_timing_snapshot_payload(
    path: &Path,
    payload: &StageTimingSnapshotPayload,
) -> Result<(), TimingFileError> {
    write_pretty_json_payload(path, payload)
}

/// Write a profile summary payload as pretty JSON.
///
/// # Errors
///
/// Returns an error when the parent directory cannot be created, the payload
/// cannot be serialized, or the file cannot be written.
pub fn write_profile_summary_payload(path: &Path, payload: &ProfileSummaryPayload) -> Result<(), TimingFileError> {
    write_pretty_json_payload(path, payload)
}

fn write_pretty_json_payload<T>(path: &Path, payload: &T) -> Result<(), TimingFileError>
where
    T: Serialize,
{
    if let Some(parent_directory) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent_directory).map_err(|source| TimingFileError::CreateParentDirectory {
            path: parent_directory.to_path_buf(),
            source,
        })?;
    }
    let payload_text = serde_json::to_string_pretty(payload).map_err(|source| TimingFileError::Serialize { source })?;
    std::fs::write(path, format!("{payload_text}\n"))
        .map_err(|source| TimingFileError::WriteFile { path: path.to_path_buf(), source })
}

impl StageTimingState {
    pub fn add_stage_duration(&mut self, stage_name: String, duration_seconds: f64) {
        *self.stage_totals_seconds.entry(stage_name.clone()).or_insert(0.0) += duration_seconds;
        *self.stage_counts.entry(stage_name).or_insert(0) += 1;
    }

    pub fn add_chunk_stage_duration(&mut self, chunk_stage_timing: ChunkStageTiming) {
        self.add_stage_duration(chunk_stage_timing.stage_name.clone(), chunk_stage_timing.duration_seconds);
        self.chunk_stage_timings.push(chunk_stage_timing);
    }

    pub fn set_native_bgen_profile(&mut self, profile_snapshot: BTreeMap<String, i64>) {
        self.native_bgen_profile = profile_snapshot;
    }

    pub fn add_binary_chunk_diagnostics(&mut self, diagnostics: BTreeMap<String, NumericDiagnosticValue>) {
        self.binary_chunk_diagnostics.push(diagnostics);
    }

    pub fn add_null_logistic_diagnostics(&mut self, diagnostics: BTreeMap<String, NullLogisticDiagnosticValue>) {
        self.null_logistic_diagnostics.push(diagnostics);
    }

    pub fn add_queue_backpressure_observation(
        &mut self,
        key: QueueBackpressureKey,
        queue_depth: i64,
        queue_capacity: i64,
        elapsed_seconds: f64,
        blocked_seconds: f64,
    ) {
        let accumulator = self.queue_backpressure.entry(key).or_default();
        accumulator.observation_count += 1;
        accumulator.max_depth = accumulator.max_depth.max(queue_depth);
        accumulator.max_capacity = accumulator.max_capacity.max(queue_capacity);
        accumulator.total_elapsed_seconds += elapsed_seconds;
        accumulator.total_blocked_seconds += blocked_seconds;
    }

    pub fn add_transfer_metadata(&mut self, key: TransferMetadataKey, byte_count: i64, element_count: i64) {
        let accumulator = self.transfer_metadata.entry(key).or_default();
        accumulator.observation_count += 1;
        accumulator.total_bytes += byte_count;
        accumulator.max_bytes = accumulator.max_bytes.max(byte_count);
        accumulator.total_elements += element_count;
    }

    /// Store transfer metadata from array adapter fields.
    ///
    /// # Errors
    ///
    /// Returns an error when the adapter fields cannot form a valid transfer
    /// metadata observation.
    pub fn add_transfer_metadata_for_shape(
        &mut self,
        transfer_name: &str,
        array_role: &str,
        dtype_name: &str,
        shape_dimensions: &[i64],
        item_size: i64,
    ) -> Result<(), TransferMetadataError> {
        let observation =
            build_transfer_metadata_observation(transfer_name, array_role, dtype_name, shape_dimensions, item_size)?;
        self.add_transfer_metadata(observation.key, observation.byte_count, observation.element_count);
        Ok(())
    }

    #[must_use]
    pub fn build_profile_summary(&self, run_id: Option<String>) -> ProfileSummaryPayload {
        ProfileSummaryPayload {
            schema_version: 1,
            run_id,
            stage_totals_seconds: self.stage_totals_seconds.clone(),
            stage_counts: self.stage_counts.clone(),
            native_bgen_profile: self.native_bgen_profile.clone(),
            derived_metrics: self.build_derived_metrics(),
            chunk_stage_summary: self.build_chunk_stage_summary(),
            binary_chunk_summary: self.build_binary_chunk_summary(),
            queue_backpressure: self.build_queue_backpressure_snapshots(),
            transfer_metadata: self.build_transfer_metadata_snapshots(),
            null_logistic_summary: NullLogisticSummary {
                chromosome_count: saturating_usize_to_i64(self.null_logistic_diagnostics.len()),
            },
        }
    }

    #[must_use]
    pub fn build_stage_timing_snapshot_payload(&self) -> StageTimingSnapshotPayload {
        StageTimingSnapshotPayload {
            stage_totals_seconds: self.stage_totals_seconds.clone(),
            stage_counts: self.stage_counts.clone(),
            chunk_stage_timings: self.chunk_stage_timings.clone(),
            native_bgen_profile: self.native_bgen_profile.clone(),
            binary_chunk_diagnostics: self.binary_chunk_diagnostics.clone(),
            null_logistic_diagnostics: self.null_logistic_diagnostics.clone(),
            queue_backpressure: self.build_queue_backpressure_snapshots(),
            transfer_metadata: self.build_transfer_metadata_snapshots(),
            derived_metrics: self.build_derived_metrics(),
        }
    }

    #[must_use]
    pub fn build_derived_metrics(&self) -> BTreeMap<String, f64> {
        let mut derived_metrics = BTreeMap::new();
        let variant_decode_count = integer_counter_as_f64(&self.native_bgen_profile, "variant_decode_count");
        let native_delivery_seconds = stage_total_seconds(&self.stage_totals_seconds, "native_engine_delivery");
        if variant_decode_count > 0.0 && native_delivery_seconds > 0.0 {
            derived_metrics
                .insert("native_variant_decode_per_second".to_string(), variant_decode_count / native_delivery_seconds);
        }

        let output_write_seconds = stage_total_seconds(&self.stage_totals_seconds, "output_write");
        if variant_decode_count > 0.0 && output_write_seconds > 0.0 {
            derived_metrics
                .insert("output_variant_rows_per_second".to_string(), variant_decode_count / output_write_seconds);
        }

        let jax_compute_seconds = stage_total_seconds(&self.stage_totals_seconds, "jax_compute");
        if variant_decode_count > 0.0 && jax_compute_seconds > 0.0 {
            derived_metrics
                .insert("jax_variant_compute_per_second".to_string(), variant_decode_count / jax_compute_seconds);
        }

        let selected_sample_count = integer_counter_as_f64(&self.native_bgen_profile, "selected_sample_count");
        if variant_decode_count > 0.0 && selected_sample_count > 0.0 && native_delivery_seconds > 0.0 {
            derived_metrics.insert(
                "native_dosage_values_per_second".to_string(),
                variant_decode_count * selected_sample_count / native_delivery_seconds,
            );
        }

        let mut transfer_byte_totals = BTreeMap::new();
        for (key, accumulator) in &self.transfer_metadata {
            *transfer_byte_totals.entry(key.transfer_name.clone()).or_insert(0) += accumulator.total_bytes;
        }
        for (transfer_name, byte_count) in transfer_byte_totals {
            let transfer_seconds = stage_total_seconds(&self.stage_totals_seconds, &transfer_name);
            if byte_count > 0 && transfer_seconds > 0.0 {
                derived_metrics
                    .insert(format!("{transfer_name}_bytes_per_second"), i64_to_f64(byte_count) / transfer_seconds);
            }
        }

        derived_metrics
    }

    #[must_use]
    pub fn build_chunk_stage_summary(&self) -> BTreeMap<String, ChunkStageSummary> {
        let mut summary = BTreeMap::new();
        for chunk_stage_timing in &self.chunk_stage_timings {
            let stage_summary =
                summary.entry(chunk_stage_timing.stage_name.clone()).or_insert_with(ChunkStageSummary::default);
            stage_summary.total_seconds += chunk_stage_timing.duration_seconds;
            stage_summary.count += 1;
        }
        summary
    }

    #[must_use]
    pub fn build_binary_chunk_summary(&self) -> BTreeMap<String, NumericDiagnosticValue> {
        let mut summary = BTreeMap::new();
        let chunk_count = saturating_usize_to_i64(self.binary_chunk_diagnostics.len());
        summary.insert("chunk_count".to_string(), NumericDiagnosticValue::Integer(chunk_count));
        if self.binary_chunk_diagnostics.is_empty() {
            return summary;
        }

        for key in BINARY_CHUNK_SUM_KEYS {
            let total = self
                .binary_chunk_diagnostics
                .iter()
                .map(|diagnostics| numeric_diagnostic_or_zero(diagnostics, key))
                .sum::<f64>();
            summary.insert(format!("{key}_total"), NumericDiagnosticValue::Float(total));
        }

        summary.insert(
            "firth_iteration_min".to_string(),
            NumericDiagnosticValue::Float(binary_diagnostic_minimum(
                &self.binary_chunk_diagnostics,
                "firth_iteration_min",
            )),
        );
        summary.insert(
            "firth_iteration_max".to_string(),
            NumericDiagnosticValue::Float(binary_diagnostic_maximum(
                &self.binary_chunk_diagnostics,
                "firth_iteration_max",
            )),
        );
        summary
    }

    #[must_use]
    pub fn build_queue_backpressure_snapshots(&self) -> Vec<QueueBackpressureSnapshot> {
        self.queue_backpressure
            .iter()
            .map(|(key, accumulator)| QueueBackpressureSnapshot {
                queue_name: key.queue_name.clone(),
                operation_name: key.operation_name.clone(),
                observation_count: accumulator.observation_count,
                max_depth: accumulator.max_depth,
                max_capacity: accumulator.max_capacity,
                total_elapsed_seconds: accumulator.total_elapsed_seconds,
                total_blocked_seconds: accumulator.total_blocked_seconds,
            })
            .collect()
    }

    #[must_use]
    pub fn build_transfer_metadata_snapshots(&self) -> Vec<TransferMetadataSnapshot> {
        self.transfer_metadata
            .iter()
            .map(|(key, accumulator)| TransferMetadataSnapshot {
                transfer_name: key.transfer_name.clone(),
                array_role: key.array_role.clone(),
                dtype_name: key.dtype_name.clone(),
                dimension_count: key.dimension_count,
                observation_count: accumulator.observation_count,
                total_bytes: accumulator.total_bytes,
                max_bytes: accumulator.max_bytes,
                total_elements: accumulator.total_elements,
            })
            .collect()
    }
}

const BINARY_CHUNK_SUM_KEYS: &[&str] = &[
    "score_only_count",
    "score_test_candidate_count",
    "firth_candidate_count",
    "firth_converged_count",
    "firth_failed_count",
    "firth_numerical_failure_count",
    "firth_max_iteration_failure_count",
    "firth_invalid_statistic_failure_count",
    "firth_step_halving_failure_count",
];

fn stage_total_seconds(stage_totals_seconds: &BTreeMap<String, f64>, key: &str) -> f64 {
    stage_totals_seconds.get(key).copied().unwrap_or_default()
}

fn integer_counter_as_f64(counters: &BTreeMap<String, i64>, key: &str) -> f64 {
    counters.get(key).copied().map_or(0.0, i64_to_f64)
}

#[allow(clippy::cast_precision_loss)]
fn i64_to_f64(value: i64) -> f64 {
    value as f64
}

fn numeric_diagnostic_or_zero(diagnostics: &BTreeMap<String, NumericDiagnosticValue>, key: &str) -> f64 {
    diagnostics.get(key).map_or(0.0, NumericDiagnosticValue::as_f64)
}

fn binary_diagnostic_minimum(diagnostics: &[BTreeMap<String, NumericDiagnosticValue>], key: &str) -> f64 {
    diagnostics
        .iter()
        .map(|diagnostic_mapping| numeric_diagnostic_or_zero(diagnostic_mapping, key))
        .fold(f64::INFINITY, f64::min)
}

fn binary_diagnostic_maximum(diagnostics: &[BTreeMap<String, NumericDiagnosticValue>], key: &str) -> f64 {
    diagnostics
        .iter()
        .map(|diagnostic_mapping| numeric_diagnostic_or_zero(diagnostic_mapping, key))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn saturating_usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_stage_queue_and_transfer_timing_state() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("load".to_string(), 0.25);
        state.add_chunk_stage_duration(ChunkStageTiming {
            chunk_identifier: 1,
            chromosome: "22".to_string(),
            variant_start_index: 0,
            variant_stop_index: 4,
            variant_count: 4,
            stage_name: "load".to_string(),
            duration_seconds: 0.75,
        });
        state.add_queue_backpressure_observation(
            QueueBackpressureKey { queue_name: "writer".to_string(), operation_name: "send".to_string() },
            3,
            8,
            0.4,
            0.1,
        );
        state.add_transfer_metadata(
            TransferMetadataKey {
                transfer_name: "host_to_device".to_string(),
                array_role: "genotype".to_string(),
                dtype_name: "float32".to_string(),
                dimension_count: 2,
            },
            128,
            32,
        );

        assert_eq!(state.stage_counts["load"], 2);
        assert!((state.stage_totals_seconds["load"] - 1.0).abs() < f64::EPSILON);
        assert_eq!(state.chunk_stage_timings.len(), 1);
        assert_eq!(state.queue_backpressure.values().next().unwrap().max_depth, 3);
        assert_eq!(state.transfer_metadata.values().next().unwrap().total_bytes, 128);
    }

    #[test]
    fn builds_profile_summary_payload_from_timing_state() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("native_engine_delivery".to_string(), 2.0);
        state.add_stage_duration("output_write".to_string(), 4.0);
        state.set_native_bgen_profile(BTreeMap::from([
            ("variant_decode_count".to_string(), 8),
            ("selected_sample_count".to_string(), 10),
        ]));
        state.add_chunk_stage_duration(ChunkStageTiming {
            chunk_identifier: 0,
            chromosome: "22".to_string(),
            variant_start_index: 0,
            variant_stop_index: 8,
            variant_count: 8,
            stage_name: "python_callback".to_string(),
            duration_seconds: 0.5,
        });
        state.add_binary_chunk_diagnostics(BTreeMap::from([
            ("score_test_candidate_count".to_string(), NumericDiagnosticValue::Integer(2)),
            ("firth_iteration_min".to_string(), NumericDiagnosticValue::Integer(4)),
            ("firth_iteration_max".to_string(), NumericDiagnosticValue::Integer(8)),
        ]));

        let summary = state.build_profile_summary(Some("run-1".to_string()));

        assert_eq!(summary.schema_version, 1);
        assert_eq!(summary.run_id.as_deref(), Some("run-1"));
        assert!((summary.derived_metrics["native_variant_decode_per_second"] - 4.0).abs() < f64::EPSILON);
        assert!((summary.derived_metrics["output_variant_rows_per_second"] - 2.0).abs() < f64::EPSILON);
        assert_eq!(summary.chunk_stage_summary["python_callback"].count, 1);
        assert_eq!(summary.binary_chunk_summary["chunk_count"], NumericDiagnosticValue::Integer(1));
        assert_eq!(
            summary.binary_chunk_summary["score_test_candidate_count_total"],
            NumericDiagnosticValue::Float(2.0)
        );
        assert_eq!(summary.binary_chunk_summary["firth_iteration_max"], NumericDiagnosticValue::Float(8.0));
    }

    #[test]
    fn builds_stage_timing_snapshot_payload_with_derived_metrics() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("host_to_device_transfer".to_string(), 2.0);
        state.add_transfer_metadata(
            TransferMetadataKey {
                transfer_name: "host_to_device_transfer".to_string(),
                array_role: "genotype_matrix".to_string(),
                dtype_name: "float32".to_string(),
                dimension_count: 2,
            },
            96,
            24,
        );

        let payload = state.build_stage_timing_snapshot_payload();

        assert_eq!(payload.stage_counts["host_to_device_transfer"], 1);
        assert_eq!(payload.transfer_metadata.len(), 1);
        assert!((payload.derived_metrics["host_to_device_transfer_bytes_per_second"] - 48.0).abs() < f64::EPSILON);
    }

    #[test]
    fn builds_transfer_metadata_observation_from_shape_dimensions() {
        let observation =
            build_transfer_metadata_observation("host_to_device_transfer", "genotype_matrix", "float32", &[4, 8], 4)
                .unwrap();

        assert_eq!(
            observation,
            TransferMetadataObservation {
                key: TransferMetadataKey {
                    transfer_name: "host_to_device_transfer".to_string(),
                    array_role: "genotype_matrix".to_string(),
                    dtype_name: "float32".to_string(),
                    dimension_count: 2,
                },
                byte_count: 128,
                element_count: 32,
            },
        );

        let scalar_observation =
            build_transfer_metadata_observation("device_to_host_materialization", "beta", "float64", &[], 8).unwrap();
        assert_eq!(scalar_observation.key.dimension_count, 0);
        assert_eq!(scalar_observation.byte_count, 8);
        assert_eq!(scalar_observation.element_count, 1);
    }

    #[test]
    fn rejects_invalid_transfer_metadata_shape_inputs() {
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[1, -1], 4).unwrap_err(),
            TransferMetadataError::NegativeDimension { dimension: -1 },
        );
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[1], 0).unwrap_err(),
            TransferMetadataError::NonPositiveItemSize { item_size: 0 },
        );
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[i64::MAX, 2], 4).unwrap_err(),
            TransferMetadataError::ElementCountOverflow,
        );
        assert_eq!(
            build_transfer_metadata_observation("transfer", "array", "float32", &[i64::MAX], 2).unwrap_err(),
            TransferMetadataError::ByteCountOverflow,
        );
    }

    #[test]
    fn records_transfer_metadata_from_shape_dimensions() {
        let mut state = StageTimingState::default();
        state
            .add_transfer_metadata_for_shape("host_to_device_transfer", "genotype_matrix", "float32", &[4, 8], 4)
            .unwrap();

        let payload = state.build_stage_timing_snapshot_payload();

        assert_eq!(
            payload.transfer_metadata,
            vec![TransferMetadataSnapshot {
                transfer_name: "host_to_device_transfer".to_string(),
                array_role: "genotype_matrix".to_string(),
                dtype_name: "float32".to_string(),
                dimension_count: 2,
                observation_count: 1,
                total_bytes: 128,
                max_bytes: 128,
                total_elements: 32,
            }],
        );
    }

    #[test]
    fn writes_stage_timing_and_profile_summary_payloads() {
        let mut state = StageTimingState::default();
        state.add_stage_duration("native_engine_delivery".to_string(), 2.0);
        state.set_native_bgen_profile(BTreeMap::from([("variant_decode_count".to_string(), 8)]));
        let directory_path = create_test_directory("writes_stage_timing_and_profile_summary_payloads");
        let stage_timing_path = directory_path.join("nested").join("stage-timings.json");
        let profile_summary_path = directory_path.join("profile.summary.json");

        write_stage_timing_snapshot_payload(&stage_timing_path, &state.build_stage_timing_snapshot_payload())
            .expect("stage timing payload should be written");
        write_profile_summary_payload(&profile_summary_path, &state.build_profile_summary(Some("run-1".to_string())))
            .expect("profile summary payload should be written");

        let stage_timing_text =
            std::fs::read_to_string(&stage_timing_path).expect("stage timing payload should be readable");
        let profile_summary_text =
            std::fs::read_to_string(&profile_summary_path).expect("profile summary payload should be readable");
        assert!(stage_timing_text.ends_with('\n'));
        assert!(profile_summary_text.ends_with('\n'));
        let stage_timing_payload: serde_json::Value =
            serde_json::from_str(&stage_timing_text).expect("stage timing payload should be valid JSON");
        let profile_summary_payload: serde_json::Value =
            serde_json::from_str(&profile_summary_text).expect("profile summary payload should be valid JSON");
        assert_eq!(stage_timing_payload["derived_metrics"]["native_variant_decode_per_second"], serde_json::json!(4.0));
        assert_eq!(profile_summary_payload["run_id"], serde_json::json!("run-1"));
        assert_eq!(
            profile_summary_payload["derived_metrics"]["native_variant_decode_per_second"],
            serde_json::json!(4.0)
        );

        std::fs::remove_dir_all(directory_path).expect("test timing directory should be removed");
    }

    #[test]
    fn resolves_exact_stage_timing_collection_policy() {
        assert!(should_collect_exact_stage_timings(true));
        assert!(!should_collect_exact_stage_timings(false));
    }

    fn create_test_directory(test_name: &str) -> PathBuf {
        let timestamp_nanoseconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let directory_path =
            std::env::temp_dir().join(format!("g-runtime-{test_name}-{}-{timestamp_nanoseconds}", std::process::id()));
        std::fs::create_dir_all(&directory_path).expect("test timing directory should be created");
        directory_path
    }
}
