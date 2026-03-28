"""Pure-Triton experiments for the linear association path using jax-triton."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

from g.models import LinearAssociationChunkResult, LinearAssociationState

TRITON_VARIANT_TILE_SIZE = 128


class TritonReductionStatistics(NamedTuple):
    """Fused reduction outputs for one genotype chunk."""

    covariate_genotype_crossproduct: jax.Array
    genotype_sum_squares: jax.Array
    covariance_with_phenotype: jax.Array


def round_up_to_power_of_two(value: int) -> int:
    """Round a positive integer up to the next power of two."""
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


@triton.autotune(
    configs=[
        triton.Config({"sample_tile_size": 64}, num_warps=4),
        triton.Config({"sample_tile_size": 128}, num_warps=4),
        triton.Config({"sample_tile_size": 128}, num_warps=8),
        triton.Config({"sample_tile_size": 256}, num_warps=8),
    ],
    key=["sample_count", "variant_count", "padded_covariate_count"],
)
@triton.jit
def fused_linear_statistics_kernel(
    genotype_matrix_pointer,
    padded_covariate_matrix_pointer,
    phenotype_residual_pointer,
    covariate_genotype_crossproduct_pointer,
    genotype_sum_squares_pointer,
    covariance_with_phenotype_pointer,
    sample_count,
    variant_count,
    padded_covariate_count,
    genotype_sample_stride,
    genotype_variant_stride,
    covariate_sample_stride,
    covariate_stride,
    crossproduct_covariate_stride,
    crossproduct_variant_stride,
    sample_tile_size: tl.constexpr,
    variant_tile_size: tl.constexpr,
    covariate_tile_size: tl.constexpr,
) -> None:
    """Fused linear GWAS statistics kernel: computes X^T G, sum(G^2), and G^T r in one pass."""
    variant_program_index = tl.program_id(axis=0)
    variant_offsets = variant_program_index * variant_tile_size + tl.arange(0, variant_tile_size)
    variant_mask = variant_offsets < variant_count
    covariate_offsets = tl.arange(0, covariate_tile_size)
    genotype_sum_squares_accumulator = tl.zeros((variant_tile_size,), dtype=tl.float32)
    covariance_with_phenotype_accumulator = tl.zeros((variant_tile_size,), dtype=tl.float32)
    covariate_genotype_crossproduct_accumulator = tl.zeros(
        (covariate_tile_size, variant_tile_size),
        dtype=tl.float32,
    )

    for sample_start in range(0, sample_count, sample_tile_size):
        sample_offsets = sample_start + tl.arange(0, sample_tile_size)
        sample_mask = sample_offsets < sample_count
        genotype_tile_pointer = (
            genotype_matrix_pointer
            + sample_offsets[:, None] * genotype_sample_stride
            + variant_offsets[None, :] * genotype_variant_stride
        )
        phenotype_residual_tile_pointer = phenotype_residual_pointer + sample_offsets
        covariate_tile_pointer = (
            padded_covariate_matrix_pointer
            + sample_offsets[:, None] * covariate_sample_stride
            + covariate_offsets[None, :] * covariate_stride
        )
        genotype_tile = tl.load(
            genotype_tile_pointer,
            mask=sample_mask[:, None] & variant_mask[None, :],
            other=0.0,
        )
        phenotype_residual_tile = tl.load(
            phenotype_residual_tile_pointer,
            mask=sample_mask,
            other=0.0,
        )
        covariate_tile = tl.load(
            covariate_tile_pointer,
            mask=sample_mask[:, None] & (covariate_offsets[None, :] < padded_covariate_count),
            other=0.0,
        )
        genotype_sum_squares_accumulator += tl.sum(genotype_tile * genotype_tile, axis=0)
        covariance_with_phenotype_accumulator += tl.sum(
            genotype_tile * phenotype_residual_tile[:, None],
            axis=0,
        )
        covariate_genotype_crossproduct_accumulator += tl.dot(
            tl.trans(covariate_tile),
            genotype_tile,
            input_precision="ieee",
        )

    genotype_sum_squares_pointer = genotype_sum_squares_pointer + variant_offsets
    covariance_with_phenotype_pointer = covariance_with_phenotype_pointer + variant_offsets
    tl.store(genotype_sum_squares_pointer, genotype_sum_squares_accumulator, mask=variant_mask)
    tl.store(
        covariance_with_phenotype_pointer,
        covariance_with_phenotype_accumulator,
        mask=variant_mask,
    )

    crossproduct_pointer = (
        covariate_genotype_crossproduct_pointer
        + covariate_offsets[:, None] * crossproduct_covariate_stride
        + variant_offsets[None, :] * crossproduct_variant_stride
    )
    tl.store(
        crossproduct_pointer,
        covariate_genotype_crossproduct_accumulator,
        mask=(covariate_offsets[:, None] < padded_covariate_count) & variant_mask[None, :],
    )


def prepare_triton_linear_association_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> tuple[LinearAssociationState, int]:
    """Precompute covariate-only state for the Triton linear experiment.
    
    Args:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Continuous phenotype vector.
        
    Returns:
        Tuple of (reusable linear-regression state, padded_covariate_count).
    """
    covariate_matrix = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    
    covariate_matrix_transpose = covariate_matrix.T
    covariate_crossproduct = covariate_matrix_transpose @ covariate_matrix
    covariate_crossproduct_cholesky_factor = jnp.linalg.cholesky(covariate_crossproduct)
    phenotype_projection = jax.scipy.linalg.cho_solve(
        (covariate_crossproduct_cholesky_factor, True),
        covariate_matrix_transpose @ phenotype_vector,
    )
    phenotype_residual = phenotype_vector - covariate_matrix @ phenotype_projection
    phenotype_residual_sum_squares = jnp.dot(phenotype_residual, phenotype_residual)
    
    state = LinearAssociationState(
        covariate_matrix=covariate_matrix,
        covariate_matrix_transpose=covariate_matrix_transpose,
        covariate_crossproduct_cholesky_factor=covariate_crossproduct_cholesky_factor,
        phenotype_residual=phenotype_residual,
        phenotype_residual_sum_squares=phenotype_residual_sum_squares,
    )
    
    covariate_count = int(covariate_matrix.shape[1])
    padded_covariate_count = round_up_to_power_of_two(covariate_count)
    
    return state, padded_covariate_count


def compute_triton_reduction_statistics(
    linear_association_state: LinearAssociationState,
    genotype_matrix: jax.Array,
    padded_covariate_count: int,
) -> TritonReductionStatistics:
    """Compute fused genotype statistics with a Triton kernel (non-JIT wrapper).
    
    This function is called from outside JIT context to properly compute strides
    and launch the Triton kernel via jax-triton.
    
    Args:
        linear_association_state: Precomputed covariate-only state.
        genotype_matrix: Genotype matrix (samples x variants).
        padded_covariate_count: Power-of-2 padded covariate count for Triton.
        
    Returns:
        Fused reduction statistics.
    """
    genotype_matrix = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    sample_count, variant_count = genotype_matrix.shape
    covariate_count = int(linear_association_state.covariate_matrix.shape[1])
    
    # Pad covariate matrix for the kernel
    padded_covariate_matrix = jnp.zeros(
        (sample_count, padded_covariate_count),
        dtype=jnp.float32,
    )
    padded_covariate_matrix = padded_covariate_matrix.at[:, :covariate_count].set(
        linear_association_state.covariate_matrix
    )
    
    # Launch Triton kernel via jax-triton
    grid = (triton.cdiv(variant_count, TRITON_VARIANT_TILE_SIZE),)
    
    # Compute strides manually for JAX arrays
    # For a 2D array of shape (M, N) with dtype float32 (4 bytes):
    # - stride along dimension 0 is N * 4 bytes
    # - stride along dimension 1 is 4 bytes
    genotype_sample_stride = genotype_matrix.shape[1]
    genotype_variant_stride = 1
    covariate_sample_stride = padded_covariate_count
    covariate_stride = 1
    
    result_crossproduct, result_ss, result_cov = jt.triton_call(
        genotype_matrix,
        padded_covariate_matrix,
        linear_association_state.phenotype_residual,
        kernel=fused_linear_statistics_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_covariate_count, variant_count), jnp.float32),
            jax.ShapeDtypeStruct((variant_count,), jnp.float32),
            jax.ShapeDtypeStruct((variant_count,), jnp.float32),
        ),
        grid=grid,
        sample_count=sample_count,
        variant_count=variant_count,
        padded_covariate_count=padded_covariate_count,
        genotype_sample_stride=genotype_sample_stride,
        genotype_variant_stride=genotype_variant_stride,
        covariate_sample_stride=covariate_sample_stride,
        covariate_stride=covariate_stride,
        crossproduct_covariate_stride=padded_covariate_count,
        crossproduct_variant_stride=1,
        variant_tile_size=TRITON_VARIANT_TILE_SIZE,
        covariate_tile_size=padded_covariate_count,
    )
    
    return TritonReductionStatistics(
        covariate_genotype_crossproduct=result_crossproduct[:covariate_count],
        genotype_sum_squares=result_ss,
        covariance_with_phenotype=result_cov,
    )


def compute_linear_association_chunk_with_triton_jax(
    linear_association_state: LinearAssociationState,
    genotype_matrix: jax.Array,
    padded_covariate_count: int,
) -> LinearAssociationChunkResult:
    """Compute linear association statistics with a fused Triton kernel.
    
    This function orchestrates the Triton kernel call and post-processing,
    all in JAX for JIT compatibility.
    
    Args:
        linear_association_state: Precomputed covariate-only linear state.
        genotype_matrix: Mean-imputed genotype matrix.
        padded_covariate_count: Power-of-2 padded covariate count for Triton.
        
    Returns:
        Chunk-level linear association statistics.
    """
    covariate_matrix = linear_association_state.covariate_matrix
    genotype_matrix = jnp.asarray(genotype_matrix, dtype=jnp.float32)
    sample_count = covariate_matrix.shape[0]
    covariate_parameter_count = covariate_matrix.shape[1]
    degrees_of_freedom = sample_count - covariate_parameter_count - 1
    
    # Get fused statistics from Triton kernel (called outside JIT)
    reduction_statistics = compute_triton_reduction_statistics(
        linear_association_state=linear_association_state,
        genotype_matrix=genotype_matrix,
        padded_covariate_count=padded_covariate_count,
    )
    
    # Solve for genotype projection using JAX
    genotype_projection = jax.scipy.linalg.cho_solve(
        (linear_association_state.covariate_crossproduct_cholesky_factor, True),
        reduction_statistics.covariate_genotype_crossproduct,
    )
    
    projection_sum_squares = jnp.sum(
        reduction_statistics.covariate_genotype_crossproduct * genotype_projection,
        axis=0,
    )
    genotype_residual_sum_squares = jnp.maximum(
        reduction_statistics.genotype_sum_squares - projection_sum_squares,
        0.0,
    )
    
    safe_denominator = jnp.where(
        genotype_residual_sum_squares > 0.0,
        genotype_residual_sum_squares,
        jnp.nan,
    )
    beta = reduction_statistics.covariance_with_phenotype / safe_denominator
    residual_sum_squares = (
        linear_association_state.phenotype_residual_sum_squares
        - beta * reduction_statistics.covariance_with_phenotype
    )
    residual_sum_squares = jnp.maximum(residual_sum_squares, 0.0)
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = jnp.sqrt(residual_variance / safe_denominator)
    test_statistic = beta / standard_error
    absolute_test_statistic = jnp.abs(test_statistic)
    degrees_of_freedom_value = jnp.asarray(degrees_of_freedom, dtype=beta.dtype)
    
    from jax.scipy.special import betainc
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
