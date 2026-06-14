"""Candidate batch preparation helpers for Firth correction."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.common import genotype as compute_genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth.batch import initial, models

if typing.TYPE_CHECKING:
    from g import types as g_types


def prepare_firth_candidate_batch_from_candidate_genotypes(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan,
    flat_fallback_indices: jax.Array,
    flat_active_mask: jax.Array,
    candidate_genotype_matrix_by_variant: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    candidate_dosage_sum: jax.Array | None,
    candidate_observation_count: jax.Array | None,
) -> models.PreparedFirthCandidateBatch:
    """Prepare ordered fixed-capacity candidate lanes from decoded candidate genotypes."""
    raw_candidate_genotype_matrix_by_variant = candidate_genotype_matrix_by_variant
    genotype_flip_result = compute_genotype.build_regenie_flipped_genotypes(
        raw_candidate_genotype_matrix_by_variant,
        dosage_sum=candidate_dosage_sum,
        observation_count=candidate_observation_count,
    )
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
        order_candidates=order_candidates,
    )
    initial_coefficients = initial.build_firth_initial_coefficients(
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
        score_beta=jnp.take(score_beta, candidate_inputs.flat_fallback_indices, axis=0),
        covariate_matrix=chromosome_state.covariate_matrix,
        genotype_matrix_by_variant=candidate_inputs.genotype_matrix_by_variant,
        phenotype_vector=chromosome_state.phenotype_vector,
        heuristic_firth_mask=candidate_inputs.heuristic_firth_mask,
        kernel_config=kernel_config,
    )
    return models.PreparedFirthCandidateBatch(
        batch_plan=batch_plan,
        candidate_inputs=candidate_inputs,
        initial_coefficients=initial_coefficients,
        full_null_deviance=chromosome_state.full_null_deviance,
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
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> models.PreparedFirthCandidateBatch:
    """Prepare ordered fixed-capacity candidate lanes for Firth correction."""
    genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan(
        candidate_mask,
        candidate_capacity,
        firth_batch_size,
    )
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
    candidate_genotype_matrix_by_variant = jnp.take(
        genotype_matrix_by_variant_float32,
        flat_fallback_indices,
        axis=0,
    )
    candidate_dosage_sum = initial.take_candidate_stat_vector(dosage_sum, flat_fallback_indices)
    candidate_observation_count = initial.take_candidate_stat_vector(observation_count, flat_fallback_indices)
    return prepare_firth_candidate_batch_from_candidate_genotypes(
        chromosome_state=chromosome_state,
        batch_plan=batch_plan,
        flat_fallback_indices=flat_fallback_indices,
        flat_active_mask=flat_active_mask,
        candidate_genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        score_beta=score_beta,
        sparse_candidate_mask=sparse_candidate_mask,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        candidate_dosage_sum=candidate_dosage_sum,
        candidate_observation_count=candidate_observation_count,
    )


def prepare_firth_candidate_batch_from_packed8(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    candidate_mask: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    candidate_capacity: int,
    firth_batch_size: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> models.PreparedFirthCandidateBatch:
    """Prepare Firth candidate lanes by decoding only selected packed8 rows."""
    batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan(
        candidate_mask,
        candidate_capacity,
        firth_batch_size,
    )
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
    packed_candidate_probability_pairs_by_variant = jnp.take(
        packed_probability_pairs_by_variant,
        flat_fallback_indices,
        axis=0,
    )
    candidate_genotype_matrix_by_variant = compute_genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_candidate_probability_pairs_by_variant,
        score_dtype,
    )
    candidate_dosage_sum = initial.take_candidate_stat_vector(dosage_sum, flat_fallback_indices)
    candidate_observation_count = initial.take_candidate_stat_vector(observation_count, flat_fallback_indices)
    return prepare_firth_candidate_batch_from_candidate_genotypes(
        chromosome_state=chromosome_state,
        batch_plan=batch_plan,
        flat_fallback_indices=flat_fallback_indices,
        flat_active_mask=flat_active_mask,
        candidate_genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        score_beta=score_beta,
        sparse_candidate_mask=sparse_candidate_mask,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        candidate_dosage_sum=candidate_dosage_sum,
        candidate_observation_count=candidate_observation_count,
    )


def prepare_multi_firth_candidate_batch_from_candidate_genotypes(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan,
    flat_fallback_indices: jax.Array,
    flat_active_mask: jax.Array,
    flat_trait_indices: jax.Array,
    flat_variant_indices: jax.Array,
    candidate_genotype_matrix_by_variant: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    candidate_dosage_sum: jax.Array | None,
    candidate_observation_count: jax.Array | None,
) -> models.PreparedMultiFirthCandidateBatch:
    """Prepare ordered multi-trait candidate lanes from decoded candidate genotypes."""
    raw_candidate_genotype_matrix_by_variant = candidate_genotype_matrix_by_variant
    genotype_flip_result = compute_genotype.build_regenie_flipped_genotypes(
        raw_candidate_genotype_matrix_by_variant,
        dosage_sum=candidate_dosage_sum,
        observation_count=candidate_observation_count,
    )
    if kernel_config.approximate_firth.use_block_math:
        firth_raw_candidate_genotype_matrix_by_variant = raw_candidate_genotype_matrix_by_variant
        flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
        candidate_genotype_matrix_by_variant = firth_raw_candidate_genotype_matrix_by_variant
    else:
        firth_raw_candidate_genotype_matrix_by_variant = genotype_flip_result.genotype_matrix_by_variant
        flat_genotype_flip_mask = genotype_flip_result.flip_mask
        candidate_genotype_matrix_by_variant = initial.residualize_and_scale_multi_genotypes_for_approximate_firth(
            square_root_weight=jnp.take(chromosome_state.square_root_weight, flat_trait_indices, axis=0),
            weighted_genotype_projection_matrix=jnp.take(
                chromosome_state.weighted_genotype_projection_matrix,
                flat_trait_indices,
                axis=0,
            ),
            genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
        )
    if sparse_candidate_mask is None:
        flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
    else:
        flat_sparse_candidate_mask = (
            jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_variant_indices, axis=0)
            & flat_active_mask
        )
    phenotype_matrix_by_lane = jnp.take(chromosome_state.phenotype_matrix, flat_trait_indices, axis=0)
    null_logistic_coefficients_by_lane = jnp.take(
        chromosome_state.null_logistic_coefficients,
        flat_trait_indices,
        axis=0,
    )
    null_firth_offset_matrix_by_lane = jnp.take(
        chromosome_state.null_firth_offset_matrix,
        flat_trait_indices,
        axis=0,
    )
    loco_offset_matrix_by_lane = jnp.take(chromosome_state.loco_offset_matrix, flat_trait_indices, axis=0)
    null_firth_penalized_log_likelihood_by_lane = jnp.take(
        chromosome_state.null_firth_penalized_log_likelihood,
        flat_trait_indices,
        axis=0,
    )
    heuristic_firth_mask = (
        regenie2_binary_candidate_planning.compute_multi_firth_pre_dispatch_mask_without_mask(
            genotype_matrix_by_lane=firth_raw_candidate_genotype_matrix_by_variant,
            phenotype_matrix_by_lane=phenotype_matrix_by_lane,
        )
        | flat_sparse_candidate_mask
    ) & flat_active_mask
    candidate_inputs = regenie2_binary_candidate_planning.group_multi_firth_candidate_batch_inputs(
        flat_fallback_indices=flat_fallback_indices,
        flat_trait_indices=flat_trait_indices,
        flat_variant_indices=flat_variant_indices,
        flat_active_mask=flat_active_mask,
        genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
        genotype_flip_mask=flat_genotype_flip_mask,
        sparse_correction_mask=flat_sparse_candidate_mask,
        heuristic_firth_mask=heuristic_firth_mask,
        phenotype_matrix=phenotype_matrix_by_lane,
        null_logistic_coefficients=null_logistic_coefficients_by_lane,
        null_firth_offset_matrix=null_firth_offset_matrix_by_lane,
        loco_offset_matrix=loco_offset_matrix_by_lane,
        null_firth_penalized_log_likelihood=null_firth_penalized_log_likelihood_by_lane,
        order_candidates=order_candidates,
    )
    initial_coefficients = initial.build_multi_firth_initial_coefficients(
        null_logistic_coefficients=candidate_inputs.null_logistic_coefficients,
        score_beta=score_beta[candidate_inputs.flat_trait_indices, candidate_inputs.flat_variant_indices],
        covariate_matrix=chromosome_state.covariate_matrix,
        genotype_matrix_by_variant=candidate_inputs.genotype_matrix_by_variant,
        phenotype_matrix=candidate_inputs.phenotype_matrix,
        heuristic_firth_mask=candidate_inputs.heuristic_firth_mask,
        kernel_config=kernel_config,
    )

    return models.PreparedMultiFirthCandidateBatch(
        batch_plan=batch_plan,
        candidate_inputs=candidate_inputs,
        initial_coefficients=initial_coefficients,
        full_null_deviance=jnp.take(chromosome_state.full_null_deviance, candidate_inputs.flat_trait_indices, axis=0),
    )


def prepare_multi_firth_candidate_batch(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    candidate_mask: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    candidate_capacity: int,
    firth_batch_size: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> models.PreparedMultiFirthCandidateBatch:
    """Prepare ordered fixed-capacity multi-trait candidate lanes for Firth correction."""
    genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    variant_count = genotype_matrix_by_variant.shape[0]
    batch_plan = regenie2_binary_candidate_planning.build_device_multi_firth_batch_plan(
        candidate_mask,
        candidate_capacity,
        firth_batch_size,
    )
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
    flat_trait_indices = flat_fallback_indices // variant_count
    flat_variant_indices = flat_fallback_indices % variant_count
    candidate_genotype_matrix_by_variant = jnp.take(
        genotype_matrix_by_variant_float32,
        flat_variant_indices,
        axis=0,
    )
    candidate_dosage_sum = initial.take_candidate_stat_vector(dosage_sum, flat_variant_indices)
    candidate_observation_count = initial.take_candidate_stat_vector(observation_count, flat_variant_indices)
    return prepare_multi_firth_candidate_batch_from_candidate_genotypes(
        chromosome_state=chromosome_state,
        batch_plan=batch_plan,
        flat_fallback_indices=flat_fallback_indices,
        flat_active_mask=flat_active_mask,
        flat_trait_indices=flat_trait_indices,
        flat_variant_indices=flat_variant_indices,
        candidate_genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        score_beta=score_beta,
        sparse_candidate_mask=sparse_candidate_mask,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        candidate_dosage_sum=candidate_dosage_sum,
        candidate_observation_count=candidate_observation_count,
    )


def prepare_multi_firth_candidate_batch_from_packed8(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    candidate_mask: jax.Array,
    score_beta: jax.Array,
    sparse_candidate_mask: jax.Array | None,
    candidate_capacity: int,
    firth_batch_size: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> models.PreparedMultiFirthCandidateBatch:
    """Prepare multi-trait Firth candidate lanes by decoding only selected packed8 rows."""
    variant_count = packed_probability_pairs_by_variant.shape[0]
    batch_plan = regenie2_binary_candidate_planning.build_device_multi_firth_batch_plan(
        candidate_mask,
        candidate_capacity,
        firth_batch_size,
    )
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
    flat_trait_indices = flat_fallback_indices // variant_count
    flat_variant_indices = flat_fallback_indices % variant_count
    packed_candidate_probability_pairs_by_variant = jnp.take(
        packed_probability_pairs_by_variant,
        flat_variant_indices,
        axis=0,
    )
    candidate_genotype_matrix_by_variant = compute_genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_candidate_probability_pairs_by_variant,
        score_dtype,
    )
    candidate_dosage_sum = initial.take_candidate_stat_vector(dosage_sum, flat_variant_indices)
    candidate_observation_count = initial.take_candidate_stat_vector(observation_count, flat_variant_indices)
    return prepare_multi_firth_candidate_batch_from_candidate_genotypes(
        chromosome_state=chromosome_state,
        batch_plan=batch_plan,
        flat_fallback_indices=flat_fallback_indices,
        flat_active_mask=flat_active_mask,
        flat_trait_indices=flat_trait_indices,
        flat_variant_indices=flat_variant_indices,
        candidate_genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
        score_beta=score_beta,
        sparse_candidate_mask=sparse_candidate_mask,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        candidate_dosage_sum=candidate_dosage_sum,
        candidate_observation_count=candidate_observation_count,
    )
