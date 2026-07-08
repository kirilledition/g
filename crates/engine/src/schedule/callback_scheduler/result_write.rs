use crate::schedule::{ResultWriteDrainCompletionPlan, ResultWriteItemResourceReleasePlan};

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
}
