"""Binary score-test kernels for REGENIE step 2."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import genotype, linalg, pvalue
from g.compute.common import result as association_result
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import state as regenie2_binary_state

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import result as regenie2_binary_result

SCORE_STATIC_ARGNAMES = (
    "firth_candidate_p_threshold",
    "minimum_variance",
    "relative_variance_tolerance",
)
DECODED_SCORE_SAMPLE_TILE_SIZE = 256


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScoreReduction:
    """Reduced score operands for one variant-major genotype chunk.

    Attributes:
        stacked_product_by_variant: Projection and score products by variant.
        weighted_genotype_sum_squares: Weighted minor-allele sums of squares.

    """

    stacked_product_by_variant: jax.Array
    weighted_genotype_sum_squares: jax.Array


def reduce_materialized_score_genotypes(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    score_genotype_matrix_by_variant: jax.Array,
) -> ScoreReduction:
    """Reduce a materialized minor-allele genotype matrix in two matrix products."""
    return ScoreReduction(
        stacked_product_by_variant=score_genotype_matrix_by_variant @ chromosome_state.score_right_hand_matrix.T,
        weighted_genotype_sum_squares=jnp.einsum(
            "vs,ts->tv",
            score_genotype_matrix_by_variant * score_genotype_matrix_by_variant,
            chromosome_state.bernoulli_weight,
        ),
    )


def reduce_tiled_score_genotypes(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    raw_genotype_matrix_by_variant: jax.Array,
    genotype_flip_mask: jax.Array,
) -> ScoreReduction:
    """Reduce decoded genotypes in bounded minor-allele-coded sample tiles."""
    variant_count = raw_genotype_matrix_by_variant.shape[0]
    sample_count = raw_genotype_matrix_by_variant.shape[1]
    sample_tile_size = min(sample_count, DECODED_SCORE_SAMPLE_TILE_SIZE)
    full_tile_count = sample_count // sample_tile_size
    tiled_sample_count = full_tile_count * sample_tile_size
    initial_reduction = ScoreReduction(
        stacked_product_by_variant=jnp.zeros(
            (variant_count, chromosome_state.score_right_hand_matrix.shape[0]),
            dtype=raw_genotype_matrix_by_variant.dtype,
        ),
        weighted_genotype_sum_squares=jnp.zeros(
            (chromosome_state.null_logistic_converged.shape[0], variant_count),
            dtype=raw_genotype_matrix_by_variant.dtype,
        ),
    )

    def reduce_tile(tile_index: jax.Array, reduction: ScoreReduction) -> ScoreReduction:
        tile_start = tile_index * sample_tile_size
        raw_genotype_tile = jax.lax.dynamic_slice_in_dim(
            raw_genotype_matrix_by_variant,
            tile_start,
            sample_tile_size,
            axis=1,
        )
        score_genotype_tile = jnp.where(
            genotype_flip_mask[:, None],
            genotype.ALLELE_COUNT_MULTIPLIER - raw_genotype_tile,
            raw_genotype_tile,
        )
        score_right_hand_tile = jax.lax.dynamic_slice_in_dim(
            chromosome_state.score_right_hand_matrix,
            tile_start,
            sample_tile_size,
            axis=1,
        )
        bernoulli_weight_tile = jax.lax.dynamic_slice_in_dim(
            chromosome_state.bernoulli_weight,
            tile_start,
            sample_tile_size,
            axis=1,
        )
        return ScoreReduction(
            stacked_product_by_variant=(
                reduction.stacked_product_by_variant + score_genotype_tile @ score_right_hand_tile.T
            ),
            weighted_genotype_sum_squares=(
                reduction.weighted_genotype_sum_squares
                + jnp.einsum(
                    "vs,ts->tv",
                    score_genotype_tile * score_genotype_tile,
                    bernoulli_weight_tile,
                )
            ),
        )

    reduction = jax.lax.fori_loop(
        0,
        full_tile_count,
        reduce_tile,
        initial_reduction,
    )
    if tiled_sample_count == sample_count:
        return reduction

    raw_genotype_tail = raw_genotype_matrix_by_variant[:, tiled_sample_count:]
    score_genotype_tail = jnp.where(
        genotype_flip_mask[:, None],
        genotype.ALLELE_COUNT_MULTIPLIER - raw_genotype_tail,
        raw_genotype_tail,
    )
    return ScoreReduction(
        stacked_product_by_variant=(
            reduction.stacked_product_by_variant
            + score_genotype_tail @ chromosome_state.score_right_hand_matrix[:, tiled_sample_count:].T
        ),
        weighted_genotype_sum_squares=(
            reduction.weighted_genotype_sum_squares
            + jnp.einsum(
                "vs,ts->tv",
                score_genotype_tail * score_genotype_tail,
                chromosome_state.bernoulli_weight[:, tiled_sample_count:],
            )
        ),
    )


def build_multi_binary_score_result(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    score_reduction: ScoreReduction,
    genotype_flip_mask: jax.Array,
    firth_candidate_p_threshold: float | None,
    minimum_variance: float,
    relative_variance_tolerance: float,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Build binary association statistics from stable minor-allele reductions."""
    trait_count = chromosome_state.null_logistic_converged.shape[0]
    covariate_count = (chromosome_state.score_right_hand_matrix.shape[0] // trait_count) - 1
    variant_count = score_reduction.stacked_product_by_variant.shape[0]
    genotype_flip_mask_by_trait_variant = genotype_flip_mask[None, :]
    projection_row_count = trait_count * covariate_count
    projection_coordinates = jnp.reshape(
        score_reduction.stacked_product_by_variant[:, :projection_row_count],
        (variant_count, trait_count, covariate_count),
    )
    projection_coordinates = jnp.transpose(projection_coordinates, (1, 0, 2))
    score = jnp.transpose(
        score_reduction.stacked_product_by_variant[:, projection_row_count:],
        (1, 0),
    )
    projection_sum_squares = jnp.einsum("tvc,tvc->tv", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(score_reduction.weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    null_logistic_converged = chromosome_state.null_logistic_converged[:, None]
    positive_variance_mask = linalg.compute_positive_residual_variance_mask(
        variance,
        score_reduction.weighted_genotype_sum_squares,
        minimum_variance,
        relative_variance_tolerance,
    )
    statistic_mask = positive_variance_mask & null_logistic_converged
    inverse_variance = jnp.where(statistic_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(
        statistic_mask,
        jnp.where(genotype_flip_mask_by_trait_variant, -score * inverse_variance, score * inverse_variance),
        jnp.nan,
    )
    standard_error = jnp.where(statistic_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(
        statistic_mask,
        score * score * inverse_variance,
        jnp.nan,
    )
    log10_p_value = jnp.where(
        statistic_mask,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    valid_mask = null_logistic_converged & jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    correction_code = regenie2_binary_correction.build_correction_code(
        log10_p_value,
        valid_mask,
        firth_candidate_p_threshold,
    )
    return association_result.AssociationResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        correction_code=correction_code,
    )


def compute_multi_binary_score_test_chunk_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    firth_candidate_p_threshold: float | None,
    minimum_variance: float,
    relative_variance_tolerance: float,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute batched binary score tests for trait-major states and variant-major genotypes.

    Args:
        chromosome_state: Trait-major chromosome-specific null model state.
        genotype_matrix_by_variant: Variant-major dosage matrix.
        firth_candidate_p_threshold: Firth candidate threshold, or ``None`` for score-only execution.
        minimum_variance: Absolute variance floor.
        relative_variance_tolerance: Relative variance floor multiplier.
        native_genotype_mean: Optional native per-variant genotype mean.

    Returns:
        Trait-major score-test result for the chunk.

    """
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_mean = genotype.compute_diploid_genotype_mean(
        raw_genotype_matrix_by_variant,
        native_genotype_mean,
    )
    genotype_flip_mask = genotype_mean > 1.0
    return build_multi_binary_score_result(
        chromosome_state,
        reduce_tiled_score_genotypes(
            chromosome_state,
            raw_genotype_matrix_by_variant,
            genotype_flip_mask,
        ),
        genotype_flip_mask,
        firth_candidate_p_threshold,
        minimum_variance,
        relative_variance_tolerance,
    )


def compute_multi_binary_score_test_packed8_with_flip_mask(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    genotype_flip_mask: jax.Array,
    firth_candidate_p_threshold: float | None,
    minimum_variance: float,
    relative_variance_tolerance: float,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute packed8 scores from an established REGENIE allele orientation."""
    return build_multi_binary_score_result(
        chromosome_state,
        reduce_materialized_score_genotypes(
            chromosome_state,
            genotype.decode_packed8_probability_pairs_to_regenie_score_genotypes(
                packed_probability_pairs_by_variant,
                genotype_flip_mask,
            ),
        ),
        genotype_flip_mask,
        firth_candidate_p_threshold,
        minimum_variance,
        relative_variance_tolerance,
    )


compute_multi_binary_score_test_variant_major = jax.jit(
    compute_multi_binary_score_test_chunk_variant_major,
    static_argnames=SCORE_STATIC_ARGNAMES,
)

compute_multi_binary_score_test_variant_major_donating_inputs = jax.jit(
    compute_multi_binary_score_test_chunk_variant_major,
    static_argnames=SCORE_STATIC_ARGNAMES,
    donate_argnames=("native_genotype_mean",),
)


def compute_multi_binary_score_test_packed8_core(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    firth_candidate_p_threshold: float | None,
    minimum_variance: float,
    relative_variance_tolerance: float,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Decode packed8 genotypes and compute binary score statistics."""
    raw_genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant
    )
    genotype_mean = genotype.compute_diploid_genotype_mean(
        raw_genotype_matrix_by_variant,
        native_genotype_mean,
    )
    return compute_multi_binary_score_test_packed8_with_flip_mask(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        genotype_flip_mask=genotype_mean > 1.0,
        firth_candidate_p_threshold=firth_candidate_p_threshold,
        minimum_variance=minimum_variance,
        relative_variance_tolerance=relative_variance_tolerance,
    )


compute_multi_binary_score_test_packed8_donating_inputs = jax.jit(
    compute_multi_binary_score_test_packed8_core,
    static_argnames=SCORE_STATIC_ARGNAMES,
    donate_argnames=("native_genotype_mean",),
)
