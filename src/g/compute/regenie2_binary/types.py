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
        firth_pseudo_maximum_iterations: Maximum approximate pseudo-Firth outer iterations.
        firth_pseudo_inner_maximum_iterations: Maximum pseudo-response logistic inner iterations.
        firth_newton_raphson_zero_start_iterations: Maximum zero-start scalar Newton-Raphson iterations.
        firth_line_search_maximum_attempts: Maximum scalar/full-model Firth line-search attempts.
        firth_step_halving_maximum_attempts: Maximum full-model Firth step-halving attempts.
        null_firth_maximum_iterations: Maximum covariate-only null Firth iterations.
        null_firth_gradient_tolerance: Covariate-only null Firth adjusted-score tolerance.
        null_firth_maximum_step_size: Covariate-only null Firth maximum coefficient step size.
        null_firth_fallback_iteration_multiplier: Multiplier for null Firth fallback retry iterations.
        null_firth_fallback_step_divisor: Divisor for null Firth fallback retry step size.
        null_firth_line_search_maximum_attempts: Maximum covariate-only null Firth line-search attempts.
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
    firth_pseudo_maximum_iterations: int = 50
    firth_pseudo_inner_maximum_iterations: int = 25
    firth_newton_raphson_zero_start_iterations: int = 100
    firth_line_search_maximum_attempts: int = 25
    firth_step_halving_maximum_attempts: int = 12
    null_firth_maximum_iterations: int = 1000
    null_firth_gradient_tolerance: float = 50.0e-6
    null_firth_maximum_step_size: float = 25.0
    null_firth_fallback_iteration_multiplier: int = 5
    null_firth_fallback_step_divisor: float = 5.0
    null_firth_line_search_maximum_attempts: int = 25
    use_block_firth_math: bool = False

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
        if self.firth_pseudo_maximum_iterations <= 0:
            message = "Firth pseudo maximum iterations must be positive."
            raise ValueError(message)
        if self.firth_pseudo_inner_maximum_iterations <= 0:
            message = "Firth pseudo inner maximum iterations must be positive."
            raise ValueError(message)
        if self.firth_newton_raphson_zero_start_iterations <= 0:
            message = "Firth zero-start Newton-Raphson iterations must be positive."
            raise ValueError(message)
        if self.firth_line_search_maximum_attempts <= 0:
            message = "Firth line-search maximum attempts must be positive."
            raise ValueError(message)
        if self.firth_step_halving_maximum_attempts <= 0:
            message = "Firth step-halving maximum attempts must be positive."
            raise ValueError(message)
        if self.null_firth_maximum_iterations <= 0:
            message = "Null Firth maximum iterations must be positive."
            raise ValueError(message)
        if self.null_firth_gradient_tolerance <= 0.0:
            message = "Null Firth gradient tolerance must be positive."
            raise ValueError(message)
        if self.null_firth_maximum_step_size <= 0.0:
            message = "Null Firth maximum step size must be positive."
            raise ValueError(message)
        if self.null_firth_fallback_iteration_multiplier <= 0:
            message = "Null Firth fallback iteration multiplier must be positive."
            raise ValueError(message)
        if self.null_firth_fallback_step_divisor <= 0.0:
            message = "Null Firth fallback step divisor must be positive."
            raise ValueError(message)
        if self.null_firth_line_search_maximum_attempts <= 0:
            message = "Null Firth line-search maximum attempts must be positive."
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
        null_firth_coefficients: Covariate-only null Firth coefficients.
        null_firth_offset: Covariate-only null Firth linear predictor plus LOCO offset.
        fitted_probability: Null-model fitted probabilities.
        score_residual: Raw score residual, ``phenotype - fitted_probability``.
        loco_offset: LOCO offset in the logistic linear predictor.
        standardized_residual: Pearson residual.
        square_root_weight: Square root of Bernoulli variance.
        weighted_genotype_projection_matrix: Cholesky-whitened weighted covariate transpose.
        null_firth_penalized_log_likelihood: Covariate-only Firth null penalized log-likelihood.
        null_firth_iteration_count: Number of covariate-only Firth iterations.
        null_firth_convergence_reason_code: Internal covariate-only Firth termination-reason code.
        null_logistic_iteration_count: Number of IRLS updates used for the null logistic fit.
        null_logistic_converged: Whether the null logistic IRLS fit converged.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    null_logistic_coefficients: jax.Array
    null_firth_coefficients: jax.Array
    null_firth_offset: jax.Array
    fitted_probability: jax.Array
    score_residual: jax.Array
    loco_offset: jax.Array
    standardized_residual: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array
    null_firth_iteration_count: jax.Array
    null_firth_convergence_reason_code: jax.Array
    null_logistic_iteration_count: jax.Array
    null_logistic_converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryChunkResult:
    """Association outputs for a REGENIE step 2 binary chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        extra_code: Integer value from `types.BinaryExtraCode` for output rendering.
        valid_mask: Boolean mask for valid statistics.
        firth_iteration_count: Number of Firth iterations per variant, or zero for non-Firth rows.
        firth_failure_code: Integer value from `types.FirthFailureCode`, or zero for non-failed rows.
        firth_convergence_reason_code: Internal Firth termination-reason integer.
        firth_correction_code: Integer value from `types.FirthCorrectionCode`.
        firth_sparse_correction_mask: Whether the approximate correction used carrier-only sparse inputs.
        pseudo_firth_iteration_count: Scalar pseudo-Firth iterations per variant.
        nr_zero_start_iteration_count: Scalar Newton-Raphson zero-start iterations per variant.
        nr_warm_start_iteration_count: Scalar Newton-Raphson warm-start iterations per variant.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
    firth_convergence_reason_code: jax.Array
    firth_correction_code: jax.Array
    firth_sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryState:
    """Reusable state for multi-trait binary REGENIE step 2 association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_matrix: Binary phenotype matrix with shape ``traits x samples``.
        sample_count: Number of samples.

    """

    covariate_matrix: jax.Array
    phenotype_matrix: jax.Array
    sample_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryChromosomeState:
    """Trait-major chromosome-specific binary null model state.

    Attributes:
        covariate_matrix: Shared covariate design matrix including intercept.
        phenotype_matrix: Binary phenotype matrix with shape ``traits x samples``.
        null_logistic_coefficients: Per-trait null logistic coefficients.
        null_firth_coefficients: Per-trait null Firth coefficients.
        null_firth_offset_matrix: Per-trait null Firth linear predictors plus LOCO offsets.
        fitted_probability: Per-trait null-model fitted probabilities.
        score_residual: Per-trait raw score residuals.
        loco_offset_matrix: Per-trait LOCO offsets.
        standardized_residual: Per-trait Pearson residuals.
        square_root_weight: Per-trait square root Bernoulli variance.
        weighted_genotype_projection_matrix: Per-trait weighted covariate projection matrix.
        null_firth_penalized_log_likelihood: Per-trait Firth null penalized log-likelihood.
        null_firth_iteration_count: Per-trait covariate-only Firth iteration counts.
        null_firth_convergence_reason_code: Per-trait covariate-only Firth termination-reason codes.
        null_logistic_iteration_count: Per-trait null IRLS iteration counts.
        null_logistic_converged: Per-trait null IRLS convergence flags.

    """

    covariate_matrix: jax.Array
    phenotype_matrix: jax.Array
    null_logistic_coefficients: jax.Array
    null_firth_coefficients: jax.Array
    null_firth_offset_matrix: jax.Array
    fitted_probability: jax.Array
    score_residual: jax.Array
    loco_offset_matrix: jax.Array
    standardized_residual: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array
    null_firth_iteration_count: jax.Array
    null_firth_convergence_reason_code: jax.Array
    null_logistic_iteration_count: jax.Array
    null_logistic_converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryChunkResult:
    """Trait-major association outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        extra_code: Integer values from `types.BinaryExtraCode` with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.
        firth_iteration_count: Firth iteration counts with shape ``traits x variants``.
        firth_failure_code: Values from `types.FirthFailureCode` with shape ``traits x variants``.
        firth_convergence_reason_code: Internal Firth termination-reason integers with shape ``traits x variants``.
        firth_correction_code: Values from `types.FirthCorrectionCode` with shape ``traits x variants``.
        firth_sparse_correction_mask: Sparse carrier-only correction flags with shape ``traits x variants``.
        pseudo_firth_iteration_count: Scalar pseudo-Firth iteration counts with shape ``traits x variants``.
        nr_zero_start_iteration_count: Scalar zero-start NR iteration counts with shape ``traits x variants``.
        nr_warm_start_iteration_count: Scalar warm-start NR iteration counts with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
    firth_convergence_reason_code: jax.Array
    firth_correction_code: jax.Array
    firth_sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array
