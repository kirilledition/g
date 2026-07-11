//! Canonical run-plan views used to construct the JAX association backend.

use g_plan::{CorrectionPlan, KernelPlan, LinearKernelPlan, RunPlan};

/// Mode-specialized policy passed to the native Python host exactly once.
#[derive(Clone, Copy, Debug)]
pub enum JaxAssociationBackendPlan<'plan> {
    /// Linear association requires only its numerical kernel policy.
    Linear(&'plan LinearKernelPlan),
    /// Binary score testing does not execute Firth correction.
    BinaryScore(&'plan KernelPlan),
    /// Binary score testing with approximate-Firth candidate correction.
    BinaryFirth { correction: &'plan CorrectionPlan, kernels: &'plan KernelPlan },
}

impl<'plan> JaxAssociationBackendPlan<'plan> {
    pub(crate) const fn from_run_plan(run_plan: &'plan RunPlan) -> Self {
        match (run_plan.association_mode, run_plan.correction.method) {
            (g_plan::AssociationMode::Regenie2Linear, _) => Self::Linear(&run_plan.compute.kernels.linear),
            (g_plan::AssociationMode::Regenie2Binary, g_plan::BinaryFallbackMethod::ScoreOnly) => {
                Self::BinaryScore(&run_plan.compute.kernels)
            }
            (g_plan::AssociationMode::Regenie2Binary, g_plan::BinaryFallbackMethod::FirthApproximate) => {
                Self::BinaryFirth { correction: &run_plan.correction, kernels: &run_plan.compute.kernels }
            }
        }
    }
}
