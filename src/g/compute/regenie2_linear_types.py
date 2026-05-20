"""JAX pytree types for quantitative REGENIE step 2 compute."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearState:
    """Precomputed state for REGENIE step 2 linear association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        phenotype_residual: Phenotype residualized against covariates.
        sample_count: Number of samples.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    covariate_matrix: jax.Array
    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    whitened_covariate_transpose: jax.Array
    phenotype_residual: jax.Array
    sample_count: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearChromosomeState:
    """Chromosome-specific REGENIE step 2 linear state.

    Attributes:
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        stacked_score_matrix: Matrix for covariate projection coordinates and phenotype covariance.
        adjusted_residual: Phenotype residual after covariate residualization and LOCO subtraction.
        adjusted_residual_projection_coordinates: Projection of adjusted residual onto whitened covariates.
        adjusted_residual_sum_squares: Sum of squares of ``adjusted_residual``.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    stacked_score_matrix: jax.Array
    adjusted_residual: jax.Array
    adjusted_residual_projection_coordinates: jax.Array
    adjusted_residual_sum_squares: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearChunkResult:
    """Association outputs for a REGENIE step 2 linear chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        valid_mask: Boolean mask for valid statistics.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearState:
    """Precomputed state for multi-trait REGENIE step 2 linear association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        phenotype_residual_matrix: Trait-major phenotype residuals after covariate projection.
        sample_count: Number of samples.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    covariate_matrix: jax.Array
    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    whitened_covariate_transpose: jax.Array
    phenotype_residual_matrix: jax.Array
    sample_count: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearChromosomeState:
    """Chromosome-specific multi-trait linear state.

    Attributes:
        covariate_matrix_transpose: Transpose of the covariate design matrix.
        covariate_crossproduct_cholesky_factor: Lower-triangular Cholesky factor of X'X.
        whitened_covariate_transpose: Cholesky-whitened covariate transpose.
        adjusted_residual_matrix: Trait-major residuals after covariate residualization and LOCO subtraction.
        adjusted_residual_projection_coordinate_matrix: Per-trait projection onto whitened covariates.
        adjusted_residual_sum_squares: Per-trait adjusted residual sums of squares.
        degrees_of_freedom: Null-model residual degrees of freedom.

    """

    covariate_matrix_transpose: jax.Array
    covariate_crossproduct_cholesky_factor: jax.Array
    whitened_covariate_transpose: jax.Array
    adjusted_residual_matrix: jax.Array
    adjusted_residual_projection_coordinate_matrix: jax.Array
    adjusted_residual_sum_squares: jax.Array
    degrees_of_freedom: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiLinearChunkResult:
    """Trait-major association outputs for a multi-trait linear chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array
