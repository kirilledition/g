"""Firth candidate batching helpers for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.common import genotype as compute_genotype
from g.compute.common import linalg
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import types as regenie2_binary_types

INITIAL_RESPONSE_SCALE = 4.863891244002886
SPARSE_CARRIER_DOSAGE_THRESHOLD = 1.0e-4


def compute_firth_pre_dispatch_mask_without_mask(
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
) -> jax.Array:
    """Identify variants with obvious case-control allele-count separation."""
    case_mask = phenotype_vector > regenie2_binary_config.BINARY_CASE_THRESHOLD
    control_mask = phenotype_vector < regenie2_binary_config.BINARY_CASE_THRESHOLD
    case_mask_float = case_mask.astype(genotype_matrix_by_variant.dtype)
    control_mask_float = control_mask.astype(genotype_matrix_by_variant.dtype)
    case_sample_count = jnp.sum(case_mask_float)
    control_sample_count = jnp.sum(control_mask_float)
    case_allele_count = genotype_matrix_by_variant @ case_mask_float
    control_allele_count = genotype_matrix_by_variant @ control_mask_float
    case_reference_allele_count = compute_genotype.ALLELE_COUNT_MULTIPLIER * case_sample_count - case_allele_count
    control_reference_allele_count = (
        compute_genotype.ALLELE_COUNT_MULTIPLIER * control_sample_count - control_allele_count
    )
    return (
        (case_allele_count <= 0.0)
        | (control_allele_count <= 0.0)
        | (case_reference_allele_count <= 0.0)
        | (control_reference_allele_count <= 0.0)
    )


def initialize_full_model_coefficients_without_mask(
    covariate_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    phenotype_vector: jax.Array,
) -> jax.Array:
    """Initialize full-model coefficients with a pseudo-response regression."""
    pseudo_response_vector = INITIAL_RESPONSE_SCALE * (phenotype_vector - regenie2_binary_config.BINARY_CASE_THRESHOLD)
    covariate_information_matrix = covariate_matrix.T @ covariate_matrix
    covariate_information_matrix = jnp.broadcast_to(
        covariate_information_matrix[None, :, :],
        (genotype_matrix_by_variant.shape[0], covariate_matrix.shape[1], covariate_matrix.shape[1]),
    )
    cross_information_vector = genotype_matrix_by_variant @ covariate_matrix
    genotype_information = jnp.einsum("ij,ij->i", genotype_matrix_by_variant, genotype_matrix_by_variant)
    covariate_score = jnp.broadcast_to(
        (covariate_matrix.T @ pseudo_response_vector)[None, :],
        (genotype_matrix_by_variant.shape[0], covariate_matrix.shape[1]),
    )
    genotype_score = genotype_matrix_by_variant @ pseudo_response_vector
    stacked_right_hand_side = jnp.stack([covariate_score, cross_information_vector], axis=-1)
    covariate_and_cross_solutions = jax.vmap(linalg.solve_from_positive_definite_matrix)(
        covariate_information_matrix,
        stacked_right_hand_side,
    )
    covariate_solution = covariate_and_cross_solutions[..., 0]
    cross_solution = covariate_and_cross_solutions[..., 1]
    schur_complement = genotype_information - jnp.einsum("ij,ij->i", cross_information_vector, cross_solution)
    genotype_coefficient = (
        genotype_score - jnp.einsum("ij,ij->i", cross_information_vector, covariate_solution)
    ) / schur_complement
    covariate_coefficients = covariate_solution - cross_solution * genotype_coefficient[:, None]
    return jnp.concatenate([covariate_coefficients, genotype_coefficient[:, None]], axis=1)


def residualize_and_scale_genotypes_for_approximate_firth(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> jax.Array:
    """Build REGENIE's approximate-Firth residualized genotype vector."""
    weighted_genotype_matrix_by_variant = genotype_matrix_by_variant * chromosome_state.square_root_weight[None, :]
    projection_coordinates = (
        weighted_genotype_matrix_by_variant @ chromosome_state.weighted_genotype_projection_matrix.T
    )
    weighted_residual_matrix_by_variant = weighted_genotype_matrix_by_variant - (
        projection_coordinates @ chromosome_state.weighted_genotype_projection_matrix
    )
    return weighted_residual_matrix_by_variant / chromosome_state.square_root_weight[None, :]


def compute_firth_variantwise(
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute device-side Firth fits for a padded set of candidate lanes."""
    del null_logistic_coefficients

    scalar_offset_vector = jnp.asarray(null_firth_offset, dtype=jnp.float64)
    scalar_phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)

    def fit_variant(
        genotype_vector: jax.Array,
        raw_genotype_vector: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        if not kernel_config.use_block_firth_math:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
                phenotype_vector=scalar_phenotype_vector,
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=scalar_offset_vector,
                carrier_sample_mask=raw_genotype_vector > SPARSE_CARRIER_DOSAGE_THRESHOLD,
                sparse_correction=sparse_correction,
                warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
                skip_firth=skip_firth,
                null_failed=~jnp.isfinite(null_penalized_log_likelihood),
                kernel_config=kernel_config,
            )
        return regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            loco_offset=loco_offset,
            initial_coefficients=variant_initial_coefficients,
            skip_firth=skip_firth,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0))(
        genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
        initial_coefficients,
        skip_firth_mask,
        sparse_correction_mask,
    )


def build_empty_firth_variant_result(
    batch_size: int,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Build a placeholder Firth result for skipped padded batches."""
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        standard_error=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        chi_squared=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        log10_p_value=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        penalized_log_likelihood=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        converged_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        valid_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        failure_code=jnp.zeros((batch_size,), dtype=jnp.int32),
        convergence_reason_code=jnp.zeros((batch_size,), dtype=jnp.int32),
        correction_code=jnp.zeros((batch_size,), dtype=jnp.int32),
        sparse_correction_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros((batch_size,), dtype=jnp.int32),
    )
