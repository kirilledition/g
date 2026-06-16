//! Native stage timing recorder state and aggregate bookkeeping.

use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct QueueBackpressureKey {
    pub(crate) queue_name: String,
    pub(crate) operation_name: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct TransferMetadataKey {
    pub(crate) transfer_name: String,
    pub(crate) array_role: String,
    pub(crate) dtype_name: String,
    pub(crate) dimension_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NumericDiagnosticValue {
    Integer(i64),
    Float(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NullLogisticDiagnosticValue {
    Integer(i64),
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChunkStageTiming {
    pub(crate) chunk_identifier: i64,
    pub(crate) chromosome: String,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) variant_count: i64,
    pub(crate) stage_name: String,
    pub(crate) duration_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct QueueBackpressureAccumulator {
    pub(crate) observation_count: i64,
    pub(crate) max_depth: i64,
    pub(crate) max_capacity: i64,
    pub(crate) total_elapsed_seconds: f64,
    pub(crate) total_blocked_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct TransferMetadataAccumulator {
    pub(crate) observation_count: i64,
    pub(crate) total_bytes: i64,
    pub(crate) max_bytes: i64,
    pub(crate) total_elements: i64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StageTimingState {
    pub(crate) stage_totals_seconds: BTreeMap<String, f64>,
    pub(crate) stage_counts: BTreeMap<String, i64>,
    pub(crate) chunk_stage_timings: Vec<ChunkStageTiming>,
    pub(crate) native_bgen_profile: BTreeMap<String, i64>,
    pub(crate) binary_chunk_diagnostics: Vec<BTreeMap<String, NumericDiagnosticValue>>,
    pub(crate) null_logistic_diagnostics: Vec<BTreeMap<String, NullLogisticDiagnosticValue>>,
    pub(crate) queue_backpressure: BTreeMap<QueueBackpressureKey, QueueBackpressureAccumulator>,
    pub(crate) transfer_metadata: BTreeMap<TransferMetadataKey, TransferMetadataAccumulator>,
}

impl StageTimingState {
    pub(crate) fn add_stage_duration(&mut self, stage_name: String, duration_seconds: f64) {
        *self.stage_totals_seconds.entry(stage_name.clone()).or_insert(0.0) += duration_seconds;
        *self.stage_counts.entry(stage_name).or_insert(0) += 1;
    }

    pub(crate) fn add_chunk_stage_duration(&mut self, chunk_stage_timing: ChunkStageTiming) {
        self.add_stage_duration(chunk_stage_timing.stage_name.clone(), chunk_stage_timing.duration_seconds);
        self.chunk_stage_timings.push(chunk_stage_timing);
    }

    pub(crate) fn set_native_bgen_profile(&mut self, profile_snapshot: BTreeMap<String, i64>) {
        self.native_bgen_profile = profile_snapshot;
    }

    pub(crate) fn add_binary_chunk_diagnostics(&mut self, diagnostics: BTreeMap<String, NumericDiagnosticValue>) {
        self.binary_chunk_diagnostics.push(diagnostics);
    }

    pub(crate) fn add_null_logistic_diagnostics(&mut self, diagnostics: BTreeMap<String, NullLogisticDiagnosticValue>) {
        self.null_logistic_diagnostics.push(diagnostics);
    }

    pub(crate) fn add_queue_backpressure_observation(
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

    pub(crate) fn add_transfer_metadata(&mut self, key: TransferMetadataKey, byte_count: i64, element_count: i64) {
        let accumulator = self.transfer_metadata.entry(key).or_default();
        accumulator.observation_count += 1;
        accumulator.total_bytes += byte_count;
        accumulator.max_bytes = accumulator.max_bytes.max(byte_count);
        accumulator.total_elements += element_count;
    }
}
