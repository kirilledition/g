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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::JaxAssociationBackendPlan;

    #[test]
    fn backend_plan_routes_linear_score_and_firth_modes() {
        let mut run_plan =
            crate::test_support::run_plan(Path::new("runner-plan"), g_plan::AssociationMode::Regenie2Linear);
        match JaxAssociationBackendPlan::from_run_plan(&run_plan) {
            JaxAssociationBackendPlan::Linear(kernels) => {
                assert!(
                    (kernels.minimum_variance.get() - run_plan.compute.kernels.linear.minimum_variance.get()).abs()
                        < f32::EPSILON
                );
            }
            _ => panic!("linear association should select the linear backend plan"),
        }

        run_plan.association_mode = g_plan::AssociationMode::Regenie2Binary;
        match JaxAssociationBackendPlan::from_run_plan(&run_plan) {
            JaxAssociationBackendPlan::BinaryScore(kernels) => {
                assert_eq!(kernels.binary_null.maximum_iterations, 50);
            }
            _ => panic!("score-only binary association should select the score backend plan"),
        }

        run_plan.correction.method = g_plan::BinaryFallbackMethod::FirthApproximate;
        match JaxAssociationBackendPlan::from_run_plan(&run_plan) {
            JaxAssociationBackendPlan::BinaryFirth { correction, kernels } => {
                assert_eq!(correction.method, g_plan::BinaryFallbackMethod::FirthApproximate);
                assert_eq!(kernels.firth.candidate_capacity, 16);
            }
            _ => panic!("approximate-Firth association should select the Firth backend plan"),
        }
    }
}
