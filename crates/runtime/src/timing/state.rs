use std::collections::BTreeMap;

use super::payloads::{
    ChunkStageSummary, ChunkStageTiming, NullLogisticDiagnosticValue, NullLogisticSummary, NumericDiagnosticValue,
    ProfileSummaryPayload, StageTimingSnapshotPayload,
};
use super::queue_backpressure::{QueueBackpressureAccumulator, QueueBackpressureKey, QueueBackpressureSnapshot};
use super::transfer_metadata::{
    TransferMetadataAccumulator, TransferMetadataError, TransferMetadataKey, TransferMetadataSnapshot,
    build_transfer_metadata_observation,
};

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
