"""Linear-regression kernels for additive association testing."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax._src.pallas.triton import core as triton_core
from jax.experimental import pallas as pl
from jax.scipy.special import betainc

from g.models import LinearAssociationChunkResult, LinearAssociationState

PALLAS_LINEAR_VARIANT_TILE_SIZE = 128
PALLAS_LINEAR_SAMPLE_TILE_SIZE = 256


class PallasReductionStatistics(NamedTuple):
    """Column-wise reduction statistics produced by the Pallas kernel."""

    genotype_sum_squares: jax.Array
    covariance_with_phenotype: jax.Array


def solve_positive_definite_system(
    cholesky_factor: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a positive-definite linear system from its Cholesky factor."""
    return jax.scipy.linalg.cho_solve((cholesky_factor, True), right_hand_side)


def prepare_linear_association_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> LinearAssociationState:
    """Precompute covariate-only linear regression terms.

    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Continuous phenotype vector.

    Returns:
        Reusable linear-regression state for genotype chunks.

    """
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_float32 = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    covariate_matrix_transpose = covariate_matrix_float32.T
    covariate_crossproduct = covariate_matrix_transpose @ covariate_matrix_float32
    covariate_crossproduct_cholesky_factor = jnp.linalg.cholesky(covariate_crossproduct)
    phenotype_projection = solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_matrix_transpose @ phenotype_vector_float32,
    )
    phenotype_residual = phenotype_vector_float32 - covariate_matrix_float32 @ phenotype_projection
    phenotype_residual_sum_squares = jnp.dot(phenotype_residual, phenotype_residual)
    return LinearAssociationState(
        covariate_matrix=covariate_matrix_float32,
        covariate_matrix_transpose=covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=covariate_crossproduct_cholesky_factor,
        phenotype_residual=phenotype_residual,
        phenotype_residual_sum_squares=phenotype_residual_sum_squares,
    )


def compute_pallas_reduction_statistics(
    genotype_matrix: jax.Array,
    phenotype_residual: jax.Array,
) -> PallasReductionStatistics:
    """Compute genotype reduction statistics with a fused Pallas kernel."""
    sample_count, variant_count = genotype_matrix.shape
    padded_sample_count = pl.cdiv(sample_count, PALLAS_LINEAR_SAMPLE_TILE_SIZE) * PALLAS_LINEAR_SAMPLE_TILE_SIZE
    padded_variant_count = pl.cdiv(variant_count, PALLAS_LINEAR_VARIANT_TILE_SIZE) * PALLAS_LINEAR_VARIANT_TILE_SIZE
    padded_genotype_matrix = jnp.pad(
        genotype_matrix,
        (
            (0, padded_sample_count - sample_count),
            (0, padded_variant_count - variant_count),
        ),
    )
    padded_phenotype_residual = jnp.pad(
        phenotype_residual,
        (0, padded_sample_count - sample_count),
    )
    sample_tile_count = padded_sample_count // PALLAS_LINEAR_SAMPLE_TILE_SIZE

    def reduction_kernel(
        genotype_matrix_ref: jax.Array,
        phenotype_residual_ref: jax.Array,
        genotype_sum_squares_ref: jax.Array,
        covariance_with_phenotype_ref: jax.Array,
    ) -> None:
        variant_start = pl.program_id(0) * PALLAS_LINEAR_VARIANT_TILE_SIZE
        genotype_sum_squares_accumulator = jnp.zeros(
            (PALLAS_LINEAR_VARIANT_TILE_SIZE,),
            dtype=genotype_matrix.dtype,
        )
        covariance_with_phenotype_accumulator = jnp.zeros(
            (PALLAS_LINEAR_VARIANT_TILE_SIZE,),
            dtype=genotype_matrix.dtype,
        )
        for sample_tile_index in range(sample_tile_count):
            sample_start = sample_tile_index * PALLAS_LINEAR_SAMPLE_TILE_SIZE
            genotype_tile = genotype_matrix_ref[
                pl.ds(sample_start, PALLAS_LINEAR_SAMPLE_TILE_SIZE),
                pl.ds(variant_start, PALLAS_LINEAR_VARIANT_TILE_SIZE),
            ]
            phenotype_tile = phenotype_residual_ref[pl.ds(sample_start, PALLAS_LINEAR_SAMPLE_TILE_SIZE)]
            genotype_sum_squares_accumulator = genotype_sum_squares_accumulator + jnp.sum(
                genotype_tile * genotype_tile,
                axis=0,
            )
            covariance_with_phenotype_accumulator = covariance_with_phenotype_accumulator + jnp.sum(
                genotype_tile * phenotype_tile[:, None],
                axis=0,
            )
        genotype_sum_squares_ref[pl.ds(variant_start, PALLAS_LINEAR_VARIANT_TILE_SIZE)] = (
            genotype_sum_squares_accumulator
        )
        covariance_with_phenotype_ref[pl.ds(variant_start, PALLAS_LINEAR_VARIANT_TILE_SIZE)] = (
            covariance_with_phenotype_accumulator
        )

    genotype_sum_squares, covariance_with_phenotype = pl.pallas_call(
        reduction_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_variant_count,), genotype_matrix.dtype),
            jax.ShapeDtypeStruct((padded_variant_count,), genotype_matrix.dtype),
        ),
        grid=(padded_variant_count // PALLAS_LINEAR_VARIANT_TILE_SIZE,),
        name="linear_pallas_reduction_statistics",
        compiler_params=triton_core.CompilerParams(),
    )(padded_genotype_matrix, padded_phenotype_residual)
    return PallasReductionStatistics(
        genotype_sum_squares=genotype_sum_squares[:variant_count],
        covariance_with_phenotype=covariance_with_phenotype[:variant_count],
    )


@jax.jit
def compute_linear_association_chunk(
    linear_association_state: LinearAssociationState,
    genotype_matrix: jax.Array,
) -> LinearAssociationChunkResult:
    """Compute linear association statistics for a chunk of variants.

    Args:
        linear_association_state: Precomputed covariate-only linear state.
        genotype_matrix: Mean-imputed genotype matrix.

    Returns:
        Chunk-level linear association statistics.

    """
    covariate_matrix = linear_association_state.covariate_matrix
    covariate_matrix_transpose = linear_association_state.covariate_matrix_transpose
    genotype_matrix = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    sample_count = covariate_matrix.shape[0]
    covariate_parameter_count = covariate_matrix.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count - 1

    covariate_genotype_crossproduct = covariate_matrix_transpose @ genotype_matrix
    genotype_projection = solve_positive_definite_system(
        linear_association_state.covariate_crossproduct_cholesky_factor,
        covariate_genotype_crossproduct,
    )

    genotype_sum_squares = jnp.sum(genotype_matrix * genotype_matrix, axis=0)
    projection_sum_squares = jnp.sum(covariate_genotype_crossproduct * genotype_projection, axis=0)
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares - projection_sum_squares, 0.0)

    # Because phenotype_residual is orthogonal to the covariates,
    # genotype_residual.T @ phenotype_residual is mathematically equivalent to:
    covariance_with_phenotype = genotype_matrix.T @ linear_association_state.phenotype_residual

    safe_denominator = jnp.where(genotype_residual_sum_squares > 0.0, genotype_residual_sum_squares, jnp.nan)
    beta = covariance_with_phenotype / safe_denominator
    residual_sum_squares = linear_association_state.phenotype_residual_sum_squares - beta * covariance_with_phenotype
    residual_sum_squares = jnp.maximum(residual_sum_squares, 0.0)
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = jnp.sqrt(residual_variance / safe_denominator)
    test_statistic = beta / standard_error
    absolute_test_statistic = jnp.abs(test_statistic)
    degrees_of_freedom_value = jnp.asarray(degrees_of_freedom, dtype=beta.dtype)
    beta_inc_argument = degrees_of_freedom_value / (
        degrees_of_freedom_value + absolute_test_statistic * absolute_test_statistic
    )
    p_value = betainc(0.5 * degrees_of_freedom_value, 0.5, beta_inc_argument)
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)

    return LinearAssociationChunkResult(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        valid_mask=valid_mask,
    )


@jax.jit
def compute_linear_association_chunk_with_pallas(
    linear_association_state: LinearAssociationState,
    genotype_matrix: jax.Array,
) -> LinearAssociationChunkResult:
    """Compute linear association statistics with a fused Pallas reduction kernel."""
    covariate_matrix = linear_association_state.covariate_matrix
    covariate_matrix_transpose = linear_association_state.covariate_matrix_transpose
    genotype_matrix = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    sample_count = covariate_matrix.shape[0]
    covariate_parameter_count = covariate_matrix.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count - 1

    covariate_genotype_crossproduct = covariate_matrix_transpose @ genotype_matrix
    genotype_projection = solve_positive_definite_system(
        linear_association_state.covariate_crossproduct_cholesky_factor,
        covariate_genotype_crossproduct,
    )

    pallas_reduction_statistics = compute_pallas_reduction_statistics(
        genotype_matrix=genotype_matrix,
        phenotype_residual=linear_association_state.phenotype_residual,
    )
    genotype_sum_squares = pallas_reduction_statistics.genotype_sum_squares
    covariance_with_phenotype = pallas_reduction_statistics.covariance_with_phenotype
    projection_sum_squares = jnp.sum(covariate_genotype_crossproduct * genotype_projection, axis=0)
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares - projection_sum_squares, 0.0)

    safe_denominator = jnp.where(genotype_residual_sum_squares > 0.0, genotype_residual_sum_squares, jnp.nan)
    beta = covariance_with_phenotype / safe_denominator
    residual_sum_squares = linear_association_state.phenotype_residual_sum_squares - beta * covariance_with_phenotype
    residual_sum_squares = jnp.maximum(residual_sum_squares, 0.0)
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = jnp.sqrt(residual_variance / safe_denominator)
    test_statistic = beta / standard_error
    absolute_test_statistic = jnp.abs(test_statistic)
    degrees_of_freedom_value = jnp.asarray(degrees_of_freedom, dtype=beta.dtype)
    beta_inc_argument = degrees_of_freedom_value / (
        degrees_of_freedom_value + absolute_test_statistic * absolute_test_statistic
    )
    p_value = betainc(0.5 * degrees_of_freedom_value, 0.5, beta_inc_argument)
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)

    return LinearAssociationChunkResult(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        valid_mask=valid_mask,
    )
