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

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
