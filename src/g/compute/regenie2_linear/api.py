"""Public linear REGENIE step 2 compute API."""

from __future__ import annotations

import functools
import typing

import jax

from g.compute.common import genotype
from g.compute.regenie2_linear import result as regenie2_linear_result
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state

if typing.TYPE_CHECKING:
    from g import types as g_types

Regenie2LinearState = regenie2_linear_state.Regenie2LinearState
Regenie2LinearChromosomeState = regenie2_linear_state.Regenie2LinearChromosomeState
Regenie2MultiLinearState = regenie2_linear_state.Regenie2MultiLinearState
Regenie2MultiLinearChromosomeState = regenie2_linear_state.Regenie2MultiLinearChromosomeState
Regenie2LinearChunkResult = regenie2_linear_result.Regenie2LinearChunkResult
Regenie2MultiLinearChunkResult = regenie2_linear_result.Regenie2MultiLinearChunkResult

LINEAR_SCORE_STATIC_ARGNAMES = (
    "score_dtype",
    "linear_minimum_variance",
    "linear_relative_variance_tolerance",
)


def prepare_regenie2_linear_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    score_dtype: g_types.FloatingPointDtype,
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
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_linear_state.Regenie2MultiLinearState:
    """Prepare shared covariate projection and trait-major phenotype residuals."""
    return regenie2_linear_state.build_multi_linear_state(covariate_matrix, phenotype_matrix, score_dtype)


@functools.partial(jax.jit, static_argnames=("score_dtype",))
def prepare_regenie2_linear_chromosome_state(
    state: regenie2_linear_state.Regenie2LinearState,
    loco_predictions: jax.Array,
    score_dtype: g_types.FloatingPointDtype,
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
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_linear_state.Regenie2MultiLinearChromosomeState:
    """Prepare chromosome-specific multi-trait residual state reused across chunks."""
    return regenie2_linear_state.build_multi_linear_chromosome_state(state, loco_prediction_matrix, score_dtype)


@functools.partial(jax.jit, static_argnames=LINEAR_SCORE_STATIC_ARGNAMES)
def compute_regenie2_linear_chunk_from_chromosome_state(
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    genotype_matrix: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
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
        score_left_hand_matrix=chromosome_state.score_left_hand_matrix,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )
    return regenie2_linear_result.squeeze_single_trait_linear_result(multi_result)


@functools.partial(jax.jit, static_argnames=LINEAR_SCORE_STATIC_ARGNAMES)
def compute_regenie2_multi_linear_chunk_from_chromosome_state(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    genotype_matrix: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
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
        score_left_hand_matrix=chromosome_state.score_left_hand_matrix,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )


@functools.partial(jax.jit, static_argnames=LINEAR_SCORE_STATIC_ARGNAMES)
def compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
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
        score_left_hand_matrix=chromosome_state.score_left_hand_matrix,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )
    return regenie2_linear_result.squeeze_single_trait_linear_result(multi_result)


@functools.partial(
    jax.jit,
    static_argnames=LINEAR_SCORE_STATIC_ARGNAMES,
    donate_argnames=(
        "packed_probability_pairs_by_variant",
        "genotype_dosage_sum",
        "genotype_observation_count",
        "genotype_imputed_dosage_square_sum",
    ),
)
def compute_regenie2_linear_chunk_from_chromosome_state_packed8_donating_inputs(
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
) -> regenie2_linear_result.Regenie2LinearChunkResult:
    """Decode packed8 probabilities on device and compute quantitative statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )


compute_linear_chunk_packed8_donating_inputs = (
    compute_regenie2_linear_chunk_from_chromosome_state_packed8_donating_inputs
)


@functools.partial(jax.jit, static_argnames=LINEAR_SCORE_STATIC_ARGNAMES)
def compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
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
        score_left_hand_matrix=chromosome_state.score_left_hand_matrix,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )


@functools.partial(
    jax.jit,
    static_argnames=LINEAR_SCORE_STATIC_ARGNAMES,
    donate_argnames=(
        "packed_probability_pairs_by_variant",
        "genotype_dosage_sum",
        "genotype_observation_count",
        "genotype_imputed_dosage_square_sum",
    ),
)
def compute_regenie2_multi_linear_chunk_from_chromosome_state_packed8_donating_inputs(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
) -> regenie2_linear_result.Regenie2MultiLinearChunkResult:
    """Decode packed8 probabilities on device and compute multi-trait quantitative statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )


compute_multi_linear_chunk_packed8_donating_inputs = (
    compute_regenie2_multi_linear_chunk_from_chromosome_state_packed8_donating_inputs
)
