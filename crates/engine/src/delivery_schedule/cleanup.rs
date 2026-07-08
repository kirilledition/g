use super::types::{BgenDeliveryCleanupAction, BgenDeliveryCleanupOutcome, BgenDeliveryCleanupPlan};

/// Plan cleanup side effects after native BGEN delivery exits.
#[must_use]
pub fn plan_bgen_delivery_cleanup(
    cleanup_outcome: BgenDeliveryCleanupOutcome,
    callback_finished: bool,
) -> BgenDeliveryCleanupPlan {
    match cleanup_outcome {
        BgenDeliveryCleanupOutcome::Success => build_bgen_delivery_cleanup_plan(&[
            BgenDeliveryCleanupAction::DrainCallback,
            BgenDeliveryCleanupAction::FinishWriterSessions,
            BgenDeliveryCleanupAction::WriteStageTimingSnapshot,
        ]),
        BgenDeliveryCleanupOutcome::Interrupted => {
            let mut cleanup_actions =
                if callback_finished { Vec::new() } else { vec![BgenDeliveryCleanupAction::DrainCallback] };
            cleanup_actions.extend([
                BgenDeliveryCleanupAction::FinishInterruptedWriterSessions,
                BgenDeliveryCleanupAction::WriteStageTimingSnapshot,
            ]);
            build_bgen_delivery_cleanup_plan(&cleanup_actions)
        }
        BgenDeliveryCleanupOutcome::Failure | BgenDeliveryCleanupOutcome::InterruptedCleanupFailure => {
            build_bgen_delivery_cleanup_plan(&[
                BgenDeliveryCleanupAction::AbortCallback,
                BgenDeliveryCleanupAction::AbortWriterSessions,
                BgenDeliveryCleanupAction::WriteStageTimingSnapshot,
            ])
        }
    }
}

pub(crate) fn build_bgen_delivery_cleanup_plan(
    cleanup_actions: &[BgenDeliveryCleanupAction],
) -> BgenDeliveryCleanupPlan {
    BgenDeliveryCleanupPlan { cleanup_actions: cleanup_actions.to_vec() }
}
