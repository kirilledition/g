use crate::schedule::{
    ResultWriteDrainCompletionPlan, ResultWriteHandoffPlan, ResultWriteItemDispatchPlan, ResultWriteItemKind,
    ResultWriteItemResourceReleasePlan, ScheduleError, plan_result_write_handoff, plan_result_write_item_dispatch,
    plan_result_write_item_dispatch_for_kinds,
};

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn plan_result_write_item_pre_write_resource_release(
        &self,
        has_host_dosage_buffer: bool,
    ) -> ResultWriteItemResourceReleasePlan {
        ResultWriteItemResourceReleasePlan {
            should_release_host_buffer: has_host_dosage_buffer,
            should_release_result_in_flight_slot: false,
        }
    }

    #[must_use]
    #[allow(clippy::fn_params_excessive_bools)]
    pub const fn plan_result_write_item_final_resource_release(
        &self,
        has_host_dosage_buffer: bool,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> ResultWriteItemResourceReleasePlan {
        ResultWriteItemResourceReleasePlan {
            should_release_host_buffer: has_host_dosage_buffer && !has_released_host_dosage_buffer,
            should_release_result_in_flight_slot: release_in_flight_slot,
        }
    }

    #[must_use]
    pub const fn plan_result_write_handoff(&self, has_result_work_item: bool) -> ResultWriteHandoffPlan {
        plan_result_write_handoff(has_result_work_item)
    }

    #[must_use]
    pub const fn plan_result_write_drain_completion(
        &self,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> ResultWriteDrainCompletionPlan {
        ResultWriteDrainCompletionPlan {
            should_stop: !has_result_work_item,
            should_flush_binary_correction_diagnostics: !has_result_work_item
                && flush_binary_correction_diagnostics_on_stop,
        }
    }

    /// Plan which result consumer should process a dequeued work item.
    ///
    /// # Errors
    ///
    /// Returns an error when either work-item kind is unsupported.
    pub fn plan_result_write_item_dispatch(
        &self,
        result_work_item_kind: &str,
        expected_result_work_item_kind: &str,
    ) -> Result<ResultWriteItemDispatchPlan, ScheduleError> {
        plan_result_write_item_dispatch(result_work_item_kind, expected_result_work_item_kind)
    }

    #[must_use]
    pub fn plan_result_write_item_dispatch_for_kinds(
        &self,
        result_work_item_kind: ResultWriteItemKind,
        expected_result_work_item_kind: ResultWriteItemKind,
    ) -> ResultWriteItemDispatchPlan {
        plan_result_write_item_dispatch_for_kinds(result_work_item_kind, expected_result_work_item_kind)
    }
}
