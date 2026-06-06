"""Firth candidate batching helpers for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import genotype as compute_genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_binary import state as regenie2_binary_state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedFirthCandidateBatch:
    """Prepared fixed-capacity Firth candidate lanes.

    Attributes:
        batch_plan: Fixed-shape candidate index and active-lane plan.
        candidate_inputs: Ordered candidate lane inputs.
        initial_coefficients: Initial full-model coefficients for each candidate lane.

    """

    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan
    candidate_inputs: regenie2_binary_candidate_planning.FirthCandidateBatchInputs
    initial_coefficients: jax.Array


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


def prepare_firth_candidate_batch(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    candidate_mask: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    candidate_capacity: int,
    firth_batch_size: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> PreparedFirthCandidateBatch:
    """Prepare ordered fixed-capacity candidate lanes for Firth correction."""
    genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan(
        candidate_mask,
        candidate_capacity,
        firth_batch_size,
    )
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    candidate_genotype_matrix_by_variant = jnp.take(
        genotype_matrix_by_variant_float32,
        flat_fallback_indices,
        axis=0,
    )
    return prepare_firth_candidate_batch_from_raw_candidate_genotypes(
        chromosome_state=chromosome_state,
        batch_plan=batch_plan,
        raw_candidate_genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        score_beta=score_beta,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def prepare_firth_candidate_batch_from_raw_candidate_genotypes(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan,
    raw_candidate_genotype_matrix_by_variant: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> PreparedFirthCandidateBatch:
    """Prepare fixed-capacity Firth lanes from already-gathered raw candidate genotypes."""
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
    raw_candidate_genotype_matrix_by_variant = jnp.asarray(raw_candidate_genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_flip_result = compute_genotype.build_regenie_flipped_genotypes(raw_candidate_genotype_matrix_by_variant)
    if kernel_config.approximate_firth.use_block_math:
        firth_raw_candidate_genotype_matrix_by_variant = raw_candidate_genotype_matrix_by_variant
        flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
        candidate_genotype_matrix_by_variant = firth_raw_candidate_genotype_matrix_by_variant
    else:
        firth_raw_candidate_genotype_matrix_by_variant = genotype_flip_result.genotype_matrix_by_variant
        flat_genotype_flip_mask = genotype_flip_result.flip_mask
        candidate_genotype_matrix_by_variant = (
            regenie2_binary_firth_scalar_approx.residualize_and_scale_genotypes_for_approximate_firth(
                chromosome_state,
                firth_raw_candidate_genotype_matrix_by_variant,
            )
        )
    if sparse_candidate_mask is None:
        flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
    else:
        flat_sparse_candidate_mask = (
            jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
            & flat_active_mask
        )
    heuristic_firth_mask = (
        regenie2_binary_candidate_planning.compute_firth_pre_dispatch_mask_without_mask(
            genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
            phenotype_vector=chromosome_state.phenotype_vector,
        )
        | flat_sparse_candidate_mask
    ) & flat_active_mask
    candidate_inputs = regenie2_binary_candidate_planning.group_firth_candidate_batch_inputs(
        flat_fallback_indices=flat_fallback_indices,
        flat_active_mask=flat_active_mask,
        genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
        genotype_flip_mask=flat_genotype_flip_mask,
        sparse_correction_mask=flat_sparse_candidate_mask,
        heuristic_firth_mask=heuristic_firth_mask,
    )
    initial_coefficients = build_firth_initial_coefficients(
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
        score_beta=jnp.take(score_beta, candidate_inputs.flat_fallback_indices, axis=0),
        covariate_matrix=chromosome_state.covariate_matrix,
        genotype_matrix_by_variant=candidate_inputs.genotype_matrix_by_variant,
        phenotype_vector=chromosome_state.phenotype_vector,
        heuristic_firth_mask=candidate_inputs.heuristic_firth_mask,
        kernel_config=kernel_config,
    )
    return PreparedFirthCandidateBatch(
        batch_plan=batch_plan,
        candidate_inputs=candidate_inputs,
        initial_coefficients=initial_coefficients,
    )


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
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
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
        if not kernel_config.approximate_firth.use_block_math:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
                phenotype_vector=scalar_phenotype_vector,
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=scalar_offset_vector,
                carrier_sample_mask=raw_genotype_vector
                > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
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


def compute_firth_variantwise_fixed_batches(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute Firth fits for flattened candidate lanes using fixed-size batches."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    raw_genotype_batches = raw_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    sparse_correction_mask_batches = sparse_correction_mask.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return compute_firth_variantwise(
                covariate_matrix=covariate_matrix,
                null_logistic_coefficients=null_logistic_coefficients,
                null_firth_offset=null_firth_offset,
                phenotype_vector=phenotype_vector,
                genotype_matrix_by_variant=genotype_batches[batch_index],
                raw_genotype_matrix_by_variant=raw_genotype_batches[batch_index],
                loco_offset=loco_offset,
                initial_coefficients=initial_coefficient_batches[batch_index],
                skip_firth_mask=~active_mask_batches[batch_index],
                sparse_correction_mask=sparse_correction_mask_batches[batch_index],
                null_penalized_log_likelihood=null_penalized_log_likelihood,
                kernel_config=kernel_config,
            )

        batch_result = jax.lax.cond(
            batch_index < active_batch_count,
            run_active_batch,
            lambda _: empty_firth_variant_result,
            operand=None,
        )
        return None, batch_result

    _, batched_firth_result = jax.lax.scan(
        compute_firth_batch,
        None,
        jnp.arange(batch_count, dtype=jnp.int32),
    )
    return regenie2_binary_firth_types.flatten_batched_firth_variant_result(batched_firth_result)
