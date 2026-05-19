"""JAX pytree types for binary REGENIE step 2 compute."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class BinaryKernelConfig:
    """Static binary-kernel settings that affect traced JAX programs.

    Attributes:
        maximum_null_iterations: Maximum IRLS iterations for the null logistic model.
        null_logistic_coefficient_tolerance: Coefficient convergence tolerance for the null logistic model.
        firth_batch_size: Fixed batch size for device-resident Firth fallback lanes.
        firth_candidate_capacity: Preferred fixed candidate capacity before falling back to full chunk capacity.
        firth_maximum_iterations: Maximum Firth solver iterations.
        firth_gradient_tolerance: Firth adjusted-score convergence tolerance.
        firth_coefficient_tolerance: Firth coefficient-step convergence tolerance.
        firth_likelihood_tolerance: Firth penalized-likelihood convergence tolerance.
        firth_maximum_step_size: Maximum absolute Firth coefficient update before step scaling.
        use_block_firth_math: Whether to use the experimental block-matrix Firth path.

    """

    maximum_null_iterations: int
    null_logistic_coefficient_tolerance: float
    firth_batch_size: int
    firth_candidate_capacity: int
    firth_maximum_iterations: int
    firth_gradient_tolerance: float
    firth_coefficient_tolerance: float
    firth_likelihood_tolerance: float
    firth_maximum_step_size: float
    use_block_firth_math: bool

    def __post_init__(self) -> None:
        """Validate positive static kernel settings."""
        if self.maximum_null_iterations <= 0:
            message = "Maximum null iterations must be positive."
            raise ValueError(message)
        if self.null_logistic_coefficient_tolerance <= 0.0:
            message = "Null logistic coefficient tolerance must be positive."
            raise ValueError(message)
        if self.firth_batch_size <= 0:
            message = "Firth batch size must be positive."
            raise ValueError(message)
        if self.firth_candidate_capacity <= 0:
            message = "Firth candidate capacity must be positive."
            raise ValueError(message)
        if self.firth_maximum_iterations <= 0:
            message = "Firth maximum iterations must be positive."
            raise ValueError(message)
        if self.firth_gradient_tolerance <= 0.0:
            message = "Firth gradient tolerance must be positive."
            raise ValueError(message)
        if self.firth_coefficient_tolerance <= 0.0:
            message = "Firth coefficient tolerance must be positive."
            raise ValueError(message)
        if self.firth_likelihood_tolerance <= 0.0:
            message = "Firth likelihood tolerance must be positive."
            raise ValueError(message)
        if self.firth_maximum_step_size <= 0.0:
            message = "Firth maximum step size must be positive."
            raise ValueError(message)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryState:
    """Reusable state for REGENIE step 2 binary association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        sample_count: Number of samples.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    sample_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryChromosomeState:
    """Chromosome-specific binary null model state.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        null_logistic_coefficients: Covariate-only null logistic coefficients.
        fitted_probability: Null-model fitted probabilities.
        score_residual: Raw score residual, ``phenotype - fitted_probability``.
        loco_offset: LOCO offset in the logistic linear predictor.
        standardized_residual: Pearson residual.
        square_root_weight: Square root of Bernoulli variance.
        weighted_genotype_projection_matrix: Cholesky-whitened weighted covariate transpose.
        null_firth_penalized_log_likelihood: Covariate-only Firth null penalized log-likelihood.
        null_logistic_iteration_count: Number of IRLS updates used for the null logistic fit.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    null_logistic_coefficients: jax.Array
    fitted_probability: jax.Array
    score_residual: jax.Array
    loco_offset: jax.Array
    standardized_residual: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array
    null_logistic_iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryChunkResult:
    """Association outputs for a REGENIE step 2 binary chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        extra_code: Integer correction code for output rendering.
        valid_mask: Boolean mask for valid statistics.
        firth_iteration_count: Number of Firth iterations per variant, or zero for non-Firth rows.
        firth_failure_code: Integer Firth failure-reason code, or zero for non-failed rows.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
