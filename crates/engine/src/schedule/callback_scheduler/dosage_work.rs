use crate::schedule::{
    DosageWorkDrainCompletionPlan, DosageWorkHandoffPlan, DosageWorkItemDispatchPlan, DosageWorkItemStageDurationPlan,
    ScheduleError, VariantMajorDosageBatchHandoffPlan, plan_dosage_work_handoff, plan_dosage_work_item_dispatch,
    plan_dosage_work_item_stage_duration, plan_variant_major_dosage_batch_handoff,
};

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn plan_dosage_work_drain_completion(&self, has_dosage_work_item: bool) -> DosageWorkDrainCompletionPlan {
        DosageWorkDrainCompletionPlan { should_stop: !has_dosage_work_item }
    }

    /// Plan which dosage consumer path should process a dequeued work item.
    ///
    /// # Errors
    ///
    /// Returns an error when the work-item kind is unsupported.
    pub fn plan_dosage_work_item_dispatch(
        &self,
        dosage_work_item_kind: &str,
    ) -> Result<DosageWorkItemDispatchPlan, ScheduleError> {
        plan_dosage_work_item_dispatch(dosage_work_item_kind)
    }

    /// Plan chunk-level timing attribution for one dosage work item.
    ///
    /// # Errors
    ///
    /// Returns an error when the work-item kind or chunk count is invalid.
    pub fn plan_dosage_work_item_stage_duration(
        &self,
        dosage_work_item_kind: &str,
        chunk_count: usize,
        elapsed_seconds: f64,
    ) -> Result<DosageWorkItemStageDurationPlan, ScheduleError> {
        plan_dosage_work_item_stage_duration(dosage_work_item_kind, chunk_count, elapsed_seconds)
    }

    /// Plan a variant-major dosage batch handoff into the callback queue.
    ///
    /// # Errors
    ///
    /// Returns an error when the metadata, genotype matrix, and chunk-stat
    /// batches have different lengths, or when the batch is empty.
    pub fn plan_variant_major_dosage_batch_handoff(
        &self,
        metadata_count: usize,
        genotype_matrix_by_variant_count: usize,
        chunk_stats_count: usize,
    ) -> Result<VariantMajorDosageBatchHandoffPlan, ScheduleError> {
        plan_variant_major_dosage_batch_handoff(metadata_count, genotype_matrix_by_variant_count, chunk_stats_count)
    }

    /// Plan a dosage work handoff into the callback queue.
    ///
    /// # Errors
    ///
    /// Returns an error when the handoff contains no chunks.
    pub fn plan_dosage_work_handoff(&self, chunk_count: usize) -> Result<DosageWorkHandoffPlan, ScheduleError> {
        plan_dosage_work_handoff(chunk_count)
    }
}
