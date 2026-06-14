"""Initial coefficient and candidate-stat helpers for Firth batches."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config


def build_firth_initial_coefficients(
    *,
    null_logistic_coefficients: jax.Array,
    score_beta: jax.Array,
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
    heuristic_firth_mask: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> jax.Array:
    """Build candidate-specific initial coefficients for Firth correction."""
    standard_initial_coefficients = jnp.broadcast_to(
        null_logistic_coefficients[None, :],
        (
            genotype_matrix_by_variant.shape[0],
            null_logistic_coefficients.shape[0],
        ),
    )
    standard_initial_beta = score_beta if kernel_config.approximate_firth.use_block_math else jnp.zeros_like(score_beta)
    standard_initial_coefficients = jnp.concatenate(
        [
            standard_initial_coefficients,
            standard_initial_beta[:, None],
        ],
        axis=1,
    )
    if not kernel_config.approximate_firth.use_block_math:
        return standard_initial_coefficients
    heuristic_initial_coefficients = regenie2_binary_firth_full_model.initialize_full_model_coefficients_without_mask(
        covariate_matrix=covariate_matrix,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        phenotype_vector=phenotype_vector,
        kernel_config=kernel_config,
    )
    return jnp.where(
        heuristic_firth_mask[:, None],
        heuristic_initial_coefficients,
        standard_initial_coefficients,
    )


def build_multi_firth_initial_coefficients(
    *,
    null_logistic_coefficients: jax.Array,
    score_beta: jax.Array,
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_matrix: jax.Array,
    heuristic_firth_mask: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> jax.Array:
    """Build lane-specific initial coefficients for multi-trait Firth correction."""
    standard_initial_beta = score_beta if kernel_config.approximate_firth.use_block_math else jnp.zeros_like(score_beta)
    standard_initial_coefficients = jnp.concatenate(
        [
            null_logistic_coefficients,
            standard_initial_beta[:, None],
        ],
        axis=1,
    )
    if not kernel_config.approximate_firth.use_block_math:
        return standard_initial_coefficients

    def initialize_one_lane(genotype_vector: jax.Array, phenotype_vector: jax.Array) -> jax.Array:
        return regenie2_binary_firth_full_model.initialize_full_model_coefficients_without_mask(
            covariate_matrix=covariate_matrix,
            genotype_matrix_by_variant=genotype_vector[None, :],
            phenotype_vector=phenotype_vector,
            kernel_config=kernel_config,
        )[0]

    heuristic_initial_coefficients = jax.vmap(initialize_one_lane)(genotype_matrix_by_variant, phenotype_matrix)
    return jnp.where(
        heuristic_firth_mask[:, None],
        heuristic_initial_coefficients,
        standard_initial_coefficients,
    )


def residualize_and_scale_multi_genotypes_for_approximate_firth(
    *,
    square_root_weight: jax.Array,
    weighted_genotype_projection_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
) -> jax.Array:
    """Build REGENIE approximate-Firth residualized genotypes for lane-specific traits."""
    weighted_genotype_matrix_by_variant = genotype_matrix_by_variant * square_root_weight
    projection_coordinates = jnp.einsum(
        "ls,lcs->lc",
        weighted_genotype_matrix_by_variant,
        weighted_genotype_projection_matrix,
    )
    weighted_residual_matrix_by_variant = weighted_genotype_matrix_by_variant - jnp.einsum(
        "lc,lcs->ls",
        projection_coordinates,
        weighted_genotype_projection_matrix,
    )
    return weighted_residual_matrix_by_variant / square_root_weight


def take_candidate_stat_vector(stat_vector: jax.Array | None, candidate_indices: jax.Array) -> jax.Array | None:
    """Gather an optional per-variant native statistic for candidate lanes."""
    if stat_vector is None:
        return None
    return jnp.take(jnp.asarray(stat_vector), candidate_indices, axis=0)
