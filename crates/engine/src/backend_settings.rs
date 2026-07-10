//! Validated scalar settings exchanged with the Python JAX backend.

use g_plan::RunPlan;

use crate::run::{RunPreparationError, validate_jax_integer_domain};

/// Complete Python-visible policy for one JAX association backend.
#[derive(Clone, Debug)]
pub struct JaxBackendSettings {
    pub association_mode: &'static str,
    pub correction: JaxCorrectionSettings,
    pub linear: JaxLinearSettings,
    pub binary: JaxBinarySettings,
}

/// Binary correction policy used for score-test fallback.
#[derive(Clone, Copy, Debug)]
pub struct JaxCorrectionSettings {
    pub method: &'static str,
    pub p_threshold: f32,
    pub firth_se: bool,
}

/// Numerical settings for linear association kernels.
#[derive(Clone, Copy, Debug)]
pub struct JaxLinearSettings {
    pub minimum_variance: f32,
    pub relative_variance_tolerance: f32,
}

/// Numerical settings for binary association kernels.
#[derive(Clone, Debug)]
pub struct JaxBinarySettings {
    pub numerical: JaxBinaryNumericalSettings,
    pub null_logistic: JaxBinaryNullLogisticSettings,
    pub firth_candidate: JaxFirthCandidateSettings,
    pub approximate_firth: JaxApproximateFirthSettings,
    pub null_firth: JaxNullFirthSettings,
}

/// Shared numerical floor settings for binary score kernels.
#[derive(Clone, Copy, Debug)]
pub struct JaxBinaryNumericalSettings {
    pub minimum_probability: f32,
    pub minimum_variance: f32,
    pub relative_variance_tolerance: f32,
}

/// Null-logistic fitting settings for binary association.
#[derive(Clone, Copy, Debug)]
pub struct JaxBinaryNullLogisticSettings {
    pub maximum_iterations: i32,
    pub coefficient_tolerance: f32,
}

/// Capacity limits for approximate-Firth candidate selection.
#[derive(Clone, Copy, Debug)]
pub struct JaxFirthCandidateSettings {
    pub batch_size: i32,
    pub candidate_capacity: i32,
}

/// Approximate-Firth solver settings.
#[derive(Clone, Copy, Debug)]
pub struct JaxApproximateFirthSettings {
    pub maximum_iterations: i32,
    pub gradient_tolerance: f64,
    pub coefficient_tolerance: f64,
    pub likelihood_tolerance: f64,
    pub maximum_step_size: f64,
    pub pseudo_maximum_iterations: i32,
    pub pseudo_inner_maximum_iterations: i32,
    pub newton_raphson_zero_start_iterations: i32,
    pub line_search_maximum_attempts: i32,
    pub step_halving_maximum_attempts: i32,
    pub initial_response_scale: f64,
    pub sparse_carrier_dosage_threshold: f64,
    pub step_halving_scale: f64,
    pub use_block_math: bool,
}

/// Null-Firth solver and fallback settings.
#[derive(Clone, Copy, Debug)]
pub struct JaxNullFirthSettings {
    pub maximum_iterations: i32,
    pub gradient_tolerance: f64,
    pub maximum_step_size: f64,
    pub fallback_iteration_multiplier: i32,
    pub fallback_step_divisor: f64,
    pub line_search_maximum_attempts: i32,
    pub step_halving_scale: f64,
}

impl JaxBackendSettings {
    /// Project one validated run plan into Python-free JAX backend settings.
    ///
    /// # Errors
    ///
    /// Returns an error when a JAX integer setting cannot fit into `i32`.
    pub fn from_run_plan(run_plan: &RunPlan) -> Result<Self, RunPreparationError> {
        validate_jax_integer_domain(run_plan)?;
        let kernels = &run_plan.compute.kernels;
        Ok(Self {
            association_mode: run_plan.association_mode.as_str(),
            correction: JaxCorrectionSettings {
                method: run_plan.correction.method.as_str(),
                p_threshold: run_plan.correction.p_threshold.get(),
                firth_se: run_plan.correction.firth_se,
            },
            linear: JaxLinearSettings {
                minimum_variance: kernels.linear.minimum_variance.get(),
                relative_variance_tolerance: kernels.linear.relative_variance_tolerance.get(),
            },
            binary: JaxBinarySettings {
                numerical: JaxBinaryNumericalSettings {
                    minimum_probability: kernels.binary_null.minimum_probability.get(),
                    minimum_variance: kernels.binary_null.minimum_variance.get(),
                    relative_variance_tolerance: kernels.binary_null.relative_variance_tolerance.get(),
                },
                null_logistic: JaxBinaryNullLogisticSettings {
                    maximum_iterations: jax_i32(
                        kernels.binary_null.maximum_iterations,
                        "binary null maximum iterations",
                    )?,
                    coefficient_tolerance: kernels.binary_null.coefficient_tolerance.get(),
                },
                firth_candidate: JaxFirthCandidateSettings {
                    batch_size: jax_i32(kernels.firth.batch_size, "Firth batch size")?,
                    candidate_capacity: jax_i32(kernels.firth.candidate_capacity, "Firth candidate capacity")?,
                },
                approximate_firth: JaxApproximateFirthSettings {
                    maximum_iterations: jax_i32(kernels.firth.maximum_iterations, "Firth maximum iterations")?,
                    gradient_tolerance: kernels.firth.gradient_tolerance.get(),
                    coefficient_tolerance: kernels.firth.coefficient_tolerance.get(),
                    likelihood_tolerance: kernels.firth.likelihood_tolerance.get(),
                    maximum_step_size: kernels.firth.maximum_step_size.get(),
                    pseudo_maximum_iterations: jax_i32(
                        kernels.firth.pseudo_maximum_iterations,
                        "Firth pseudo maximum iterations",
                    )?,
                    pseudo_inner_maximum_iterations: jax_i32(
                        kernels.firth.pseudo_inner_maximum_iterations,
                        "Firth pseudo inner maximum iterations",
                    )?,
                    newton_raphson_zero_start_iterations: jax_i32(
                        kernels.firth.newton_raphson_zero_start_iterations,
                        "Firth Newton-Raphson zero-start iterations",
                    )?,
                    line_search_maximum_attempts: jax_i32(
                        kernels.firth.line_search_maximum_attempts,
                        "Firth line-search maximum attempts",
                    )?,
                    step_halving_maximum_attempts: jax_i32(
                        kernels.firth.step_halving_maximum_attempts,
                        "Firth step-halving maximum attempts",
                    )?,
                    initial_response_scale: kernels.firth.initial_response_scale.get(),
                    sparse_carrier_dosage_threshold: kernels.firth.sparse_carrier_dosage_threshold.get(),
                    step_halving_scale: kernels.firth.step_halving_scale.get(),
                    use_block_math: kernels.firth.use_block_math,
                },
                null_firth: JaxNullFirthSettings {
                    maximum_iterations: jax_i32(
                        kernels.null_firth.maximum_iterations,
                        "null Firth maximum iterations",
                    )?,
                    gradient_tolerance: kernels.null_firth.gradient_tolerance.get(),
                    maximum_step_size: kernels.null_firth.maximum_step_size.get(),
                    fallback_iteration_multiplier: jax_i32(
                        kernels.null_firth.fallback_iteration_multiplier,
                        "null Firth fallback iteration multiplier",
                    )?,
                    fallback_step_divisor: kernels.null_firth.fallback_step_divisor.get(),
                    line_search_maximum_attempts: jax_i32(
                        kernels.null_firth.line_search_maximum_attempts,
                        "null Firth line-search maximum attempts",
                    )?,
                    step_halving_scale: kernels.null_firth.step_halving_scale.get(),
                },
            },
        })
    }
}

fn jax_i32(value: u32, field_name: &'static str) -> Result<i32, RunPreparationError> {
    i32::try_from(value).map_err(|_| RunPreparationError::JaxIntegerOverflow { field_name })
}
