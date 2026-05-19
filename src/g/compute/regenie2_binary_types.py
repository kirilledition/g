"""JAX pytree types for binary REGENIE step 2 compute."""

from __future__ import annotations

from dataclasses import dataclass

import jax


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
        fitted_probability: Per-trait null-model fitted probabilities.
        score_residual: Per-trait raw score residuals.
        loco_offset_matrix: Per-trait LOCO offsets.
        standardized_residual: Per-trait Pearson residuals.
        square_root_weight: Per-trait square root Bernoulli variance.
        weighted_genotype_projection_matrix: Per-trait weighted covariate projection matrix.
        null_firth_penalized_log_likelihood: Per-trait Firth null penalized log-likelihood.
        null_logistic_iteration_count: Per-trait null IRLS iteration counts.

    """

    covariate_matrix: jax.Array
    phenotype_matrix: jax.Array
    null_logistic_coefficients: jax.Array
    fitted_probability: jax.Array
    score_residual: jax.Array
    loco_offset_matrix: jax.Array
    standardized_residual: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array
    null_logistic_iteration_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryChunkResult:
    """Trait-major association outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        extra_code: Integer correction codes with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.
        firth_iteration_count: Firth iteration counts with shape ``traits x variants``.
        firth_failure_code: Firth failure-reason codes with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
