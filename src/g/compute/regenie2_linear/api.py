"""Public linear REGENIE step 2 compute API."""

from __future__ import annotations

import functools

import jax

from g import types
from g.compute.common import genotype
from g.compute.regenie2_linear import result as regenie2_linear_result
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state

Regenie2LinearState = regenie2_linear_state.Regenie2LinearState
Regenie2LinearChromosomeState = regenie2_linear_state.Regenie2LinearChromosomeState
Regenie2MultiLinearState = regenie2_linear_state.Regenie2MultiLinearState
Regenie2MultiLinearChromosomeState = regenie2_linear_state.Regenie2MultiLinearChromosomeState
Regenie2LinearChunkResult = regenie2_linear_result.Regenie2LinearChunkResult
Regenie2MultiLinearChunkResult = regenie2_linear_result.Regenie2MultiLinearChunkResult


def prepare_regenie2_linear_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_state.Regenie2LinearState:
    """Prepare covariate projection and phenotype residual for REGENIE step 2."""
    multi_state = prepare_regenie2_multi_linear_state(
        covariate_matrix=covariate_matrix,
        phenotype_matrix=phenotype_vector[None, :],
        score_dtype=score_dtype,
    )
    return regenie2_linear_state.build_single_linear_state_from_multi(multi_state)


def prepare_regenie2_multi_linear_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_state.Regenie2MultiLinearState:
    """Prepare shared covariate projection and trait-major phenotype residuals."""
    return regenie2_linear_state.build_multi_linear_state(covariate_matrix, phenotype_matrix, score_dtype)


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def prepare_regenie2_linear_chromosome_state(
    state: regenie2_linear_state.Regenie2LinearState,
    loco_predictions: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_state.Regenie2LinearChromosomeState:
    """Prepare chromosome-specific residual state reused across chunks."""
    multi_state = regenie2_linear_state.build_multi_linear_state_from_single(state)
    multi_chromosome_state = regenie2_linear_state.build_multi_linear_chromosome_state(
        multi_state,
        loco_predictions[None, :],
        score_dtype,
    )
    return regenie2_linear_state.build_single_linear_chromosome_state_from_multi(multi_chromosome_state)


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def prepare_regenie2_multi_linear_chromosome_state(
    state: regenie2_linear_state.Regenie2MultiLinearState,
    loco_prediction_matrix: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_state.Regenie2MultiLinearChromosomeState:
    """Prepare chromosome-specific multi-trait residual state reused across chunks."""
    return regenie2_linear_state.build_multi_linear_chromosome_state(state, loco_prediction_matrix, score_dtype)


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def compute_regenie2_linear_chunk_from_chromosome_state(
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    genotype_matrix: jax.Array,
    genotype_dosage_sum: jax.Array | None = None,
    genotype_observation_count: jax.Array | None = None,
    genotype_imputed_dosage_square_sum: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_result.Regenie2LinearChunkResult:
    """Compute REGENIE step 2 linear association using chromosome-cached state."""
    multi_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual_matrix=chromosome_state.adjusted_residual[None, :],
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinates[
            None, :
        ],
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares[None],
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix, score_dtype),
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
    )
    return regenie2_linear_result.squeeze_single_trait_linear_result(multi_result)


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def compute_regenie2_multi_linear_chunk_from_chromosome_state(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    genotype_matrix: jax.Array,
    genotype_dosage_sum: jax.Array | None = None,
    genotype_observation_count: jax.Array | None = None,
    genotype_imputed_dosage_square_sum: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_result.Regenie2MultiLinearChunkResult:
    """Compute multi-trait quantitative REGENIE step 2 association."""
    return regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual_matrix=chromosome_state.adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares,
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix, score_dtype),
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
    )


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None = None,
    genotype_observation_count: jax.Array | None = None,
    genotype_imputed_dosage_square_sum: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_result.Regenie2LinearChunkResult:
    """Compute quantitative REGENIE step 2 association from variant-major genotypes."""
    multi_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual_matrix=chromosome_state.adjusted_residual[None, :],
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinates[
            None, :
        ],
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares[None],
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
    )
    return regenie2_linear_result.squeeze_single_trait_linear_result(multi_result)


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None = None,
    genotype_observation_count: jax.Array | None = None,
    genotype_imputed_dosage_square_sum: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_result.Regenie2MultiLinearChunkResult:
    """Compute multi-trait quantitative REGENIE step 2 from variant-major genotypes."""
    return regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
        whitened_covariate_transpose=chromosome_state.whitened_covariate_transpose,
        adjusted_residual_matrix=chromosome_state.adjusted_residual_matrix,
        adjusted_residual_projection_coordinate_matrix=chromosome_state.adjusted_residual_projection_coordinate_matrix,
        adjusted_residual_sum_squares=chromosome_state.adjusted_residual_sum_squares,
        degrees_of_freedom=chromosome_state.degrees_of_freedom,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
    )


def compute_regenie2_linear_chunk(
    state: regenie2_linear_state.Regenie2LinearState,
    genotype_matrix: jax.Array,
    loco_predictions: jax.Array,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_linear_result.Regenie2LinearChunkResult:
    """Compute REGENIE step 2 linear association for a genotype chunk.

    This implements the REGENIE step 2 score test for quantitative traits:
    1. Subtract LOCO predictions from the covariate-residualized phenotype
    2. Residualize genotypes against covariates
    3. Compute score test statistics

    The test statistic follows a chi-squared distribution with 1 degree of freedom.

    Args:
        state: Precomputed covariate state from prepare_regenie2_linear_state.
        genotype_matrix: Mean-imputed genotype dosage matrix.
        loco_predictions: LOCO predictions for this chromosome.
        score_dtype: Floating-point dtype for score-test computation.

    Returns:
        Association statistics for the chunk.

    Mathematical formulation:
        adjusted_residual = phenotype_residual - loco_predictions
        For each variant g:
            genotype_residual = g - X @ (X'X)^-1 @ X' @ g
            beta = (genotype_residual' @ adjusted_residual) / (genotype_residual' @ genotype_residual)
            variance = null_mean_squared_error / (genotype_residual' @ genotype_residual)
            chi_squared = beta^2 / variance
            log10_p_value = -log10(chi2_to_p(chi_squared, df=1))

    """
    chromosome_state = prepare_regenie2_linear_chromosome_state(state, loco_predictions, score_dtype)
    return compute_regenie2_linear_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        score_dtype=score_dtype,
    )
