use crate::schedule::DosageWorkDrainCompletionPlan;

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn plan_dosage_work_drain_completion(&self, has_dosage_work_item: bool) -> DosageWorkDrainCompletionPlan {
        DosageWorkDrainCompletionPlan { should_stop: !has_dosage_work_item }
    }
}
