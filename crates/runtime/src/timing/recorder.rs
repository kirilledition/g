use std::collections::BTreeMap;
use std::path::Path;

use super::final_outputs::{
    FinalTimingOutputsWriteResultPayload, TimingFileError, plan_stage_timing_recorder, plan_timing_file_write,
    write_pretty_json_payload,
};
use super::payloads::{
    ChunkStageTiming, NullLogisticDiagnosticValue, NumericDiagnosticValue, ProfileSummaryPayload,
    StageTimingSnapshotPayload,
};
use super::queue_backpressure::QueueBackpressureKey;
use super::state::StageTimingState;
use super::transfer_metadata::{TransferMetadataError, TransferMetadataKey};

#[derive(Clone, Debug, PartialEq)]
pub struct StageTimingRecorder {
    exact_stage_timings: bool,
    state: StageTimingState,
}

impl StageTimingRecorder {
    #[must_use]
    pub fn new(exact_stage_timings: bool) -> Self {
        Self { exact_stage_timings, state: StageTimingState::default() }
    }

    #[must_use]
    pub fn from_config(stage_timing_path_configured: bool, force: bool) -> Option<Self> {
        let plan = plan_stage_timing_recorder(stage_timing_path_configured, force);
        plan.should_create.then(|| Self::new(plan.exact_stage_timings))
    }

    #[must_use]
    pub const fn exact_stage_timings(&self) -> bool {
        self.exact_stage_timings
    }

    #[must_use]
    pub const fn should_collect_exact_stage_timings(&self) -> bool {
        self.exact_stage_timings
    }

    #[must_use]
    pub const fn should_write_timing_file(&self, path_configured: bool) -> bool {
        plan_timing_file_write(true, path_configured).should_write
    }

    pub fn add_stage_duration(&mut self, stage_name: String, duration_seconds: f64) {
        self.state.add_stage_duration(stage_name, duration_seconds);
    }

    pub fn add_chunk_stage_duration(&mut self, chunk_stage_timing: ChunkStageTiming) {
        self.state.add_chunk_stage_duration(chunk_stage_timing);
    }

    pub fn set_native_bgen_profile(&mut self, profile_snapshot: BTreeMap<String, i64>) {
        self.state.set_native_bgen_profile(profile_snapshot);
    }

    pub fn add_binary_chunk_diagnostics(&mut self, diagnostics: BTreeMap<String, NumericDiagnosticValue>) {
        self.state.add_binary_chunk_diagnostics(diagnostics);
    }

    pub fn add_null_logistic_diagnostics(&mut self, diagnostics: BTreeMap<String, NullLogisticDiagnosticValue>) {
        self.state.add_null_logistic_diagnostics(diagnostics);
    }

    pub fn add_queue_backpressure_observation(
        &mut self,
        key: QueueBackpressureKey,
        queue_depth: i64,
        queue_capacity: i64,
        elapsed_seconds: f64,
        blocked_seconds: f64,
    ) {
        self.state.add_queue_backpressure_observation(
            key,
            queue_depth,
            queue_capacity,
            elapsed_seconds,
            blocked_seconds,
        );
    }

    pub fn add_transfer_metadata(&mut self, key: TransferMetadataKey, byte_count: i64, element_count: i64) {
        self.state.add_transfer_metadata(key, byte_count, element_count);
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
        self.state.add_transfer_metadata_for_shape(transfer_name, array_role, dtype_name, shape_dimensions, item_size)
    }

    #[must_use]
    pub fn build_profile_summary(&self, run_id: Option<String>) -> ProfileSummaryPayload {
        self.state.build_profile_summary(run_id)
    }

    #[must_use]
    pub fn build_stage_timing_snapshot_payload(&self) -> StageTimingSnapshotPayload {
        self.state.build_stage_timing_snapshot_payload()
    }

    #[must_use]
    pub fn build_derived_metrics(&self) -> BTreeMap<String, f64> {
        self.state.build_derived_metrics()
    }

    /// Write a stage timing snapshot as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when the timing payload cannot be written.
    pub fn write_stage_timing_snapshot(&self, path: &Path) -> Result<(), TimingFileError> {
        write_pretty_json_payload(path, &self.state.build_stage_timing_snapshot_payload())
    }

    /// Write a stage timing snapshot when a path is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the timing payload cannot be written.
    pub fn write_stage_timing_snapshot_if_configured(&self, path: Option<&Path>) -> Result<bool, TimingFileError> {
        let Some(active_path) = path else {
            return Ok(false);
        };
        if !self.should_write_timing_file(true) {
            return Ok(false);
        }
        self.write_stage_timing_snapshot(active_path)?;
        Ok(true)
    }

    /// Write a profile summary as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when the profile summary payload cannot be written.
    pub fn write_profile_summary(&self, path: &Path, run_id: Option<String>) -> Result<(), TimingFileError> {
        write_pretty_json_payload(path, &self.state.build_profile_summary(run_id))
    }

    /// Write a profile summary when a path is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the profile summary payload cannot be written.
    pub fn write_profile_summary_if_configured(
        &self,
        path: Option<&Path>,
        run_id: Option<String>,
    ) -> Result<bool, TimingFileError> {
        let Some(active_path) = path else {
            return Ok(false);
        };
        if !self.should_write_timing_file(true) {
            return Ok(false);
        }
        self.write_profile_summary(active_path, run_id)?;
        Ok(true)
    }

    /// Write all configured final timing outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when any configured timing payload cannot be written.
    pub fn write_final_timing_outputs(
        &self,
        stage_timing_path: Option<&Path>,
        profile_summary_path: Option<&Path>,
        run_id: Option<String>,
    ) -> Result<FinalTimingOutputsWriteResultPayload, TimingFileError> {
        Ok(FinalTimingOutputsWriteResultPayload {
            wrote_stage_timing_snapshot: self.write_stage_timing_snapshot_if_configured(stage_timing_path)?,
            wrote_profile_summary: self.write_profile_summary_if_configured(profile_summary_path, run_id)?,
        })
    }
}
