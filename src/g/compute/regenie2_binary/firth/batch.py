"""Firth candidate batching helpers for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types as g_types
from g.compute.common import genotype as compute_genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_binary import state as regenie2_binary_state

SPARSE_FIRTH_CARRIER_CAPACITY = 64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedFirthCandidateBatch:
    """Prepared fixed-capacity Firth candidate lanes.

    Attributes:
        batch_plan: Fixed-shape candidate index and active-lane plan.
        candidate_inputs: Ordered candidate lane inputs.
        initial_coefficients: Initial full-model coefficients for each candidate lane.
        full_null_deviance: Full-sample null deviance reused by compact sparse scalar lanes.

    """

    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan
    candidate_inputs: regenie2_binary_candidate_planning.FirthCandidateBatchInputs
    initial_coefficients: jax.Array
    full_null_deviance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedMultiFirthCandidateBatch:
    """Prepared fixed-capacity multi-trait Firth candidate lanes.

    Attributes:
        batch_plan: Fixed-shape candidate index and active-lane plan.
        candidate_inputs: Ordered candidate lane inputs with trait and variant indices.
        initial_coefficients: Initial full-model coefficients for each candidate lane.
        full_null_deviance: Lane-specific full-sample null deviance.

    """

    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan
    candidate_inputs: regenie2_binary_candidate_planning.MultiFirthCandidateBatchInputs
    initial_coefficients: jax.Array
    full_null_deviance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthLaneStreamPlan:
    """Fixed-shape lane stream selected from a candidate batch.

    Attributes:
        lane_indices: Candidate-batch positions packed into stream order.
        active_mask: Active mask for the packed stream.
        active_count: Number of active lanes in the stream.

    """

    lane_indices: jax.Array
    active_mask: jax.Array
    active_count: jax.Array


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
    candidate_dosage_sum: jax.Array | None = None,
    candidate_observation_count: jax.Array | None = None,
) -> PreparedFirthCandidateBatch:
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
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> PreparedFirthCandidateBatch:
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
    candidate_dosage_sum = take_candidate_stat_vector(dosage_sum, flat_fallback_indices)
    candidate_observation_count = take_candidate_stat_vector(observation_count, flat_fallback_indices)
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
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> PreparedFirthCandidateBatch:
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
    candidate_dosage_sum = take_candidate_stat_vector(dosage_sum, flat_fallback_indices)
    candidate_observation_count = take_candidate_stat_vector(observation_count, flat_fallback_indices)
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
    candidate_dosage_sum: jax.Array | None = None,
    candidate_observation_count: jax.Array | None = None,
) -> PreparedMultiFirthCandidateBatch:
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
        candidate_genotype_matrix_by_variant = residualize_and_scale_multi_genotypes_for_approximate_firth(
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
    initial_coefficients = build_multi_firth_initial_coefficients(
        null_logistic_coefficients=candidate_inputs.null_logistic_coefficients,
        score_beta=score_beta[candidate_inputs.flat_trait_indices, candidate_inputs.flat_variant_indices],
        covariate_matrix=chromosome_state.covariate_matrix,
        genotype_matrix_by_variant=candidate_inputs.genotype_matrix_by_variant,
        phenotype_matrix=candidate_inputs.phenotype_matrix,
        heuristic_firth_mask=candidate_inputs.heuristic_firth_mask,
        kernel_config=kernel_config,
    )

    return PreparedMultiFirthCandidateBatch(
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
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> PreparedMultiFirthCandidateBatch:
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
    candidate_dosage_sum = take_candidate_stat_vector(dosage_sum, flat_variant_indices)
    candidate_observation_count = take_candidate_stat_vector(observation_count, flat_variant_indices)
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
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> PreparedMultiFirthCandidateBatch:
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
    candidate_dosage_sum = take_candidate_stat_vector(dosage_sum, flat_variant_indices)
    candidate_observation_count = take_candidate_stat_vector(observation_count, flat_variant_indices)
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


def build_firth_lane_stream_plan(active_lane_mask: jax.Array) -> FirthLaneStreamPlan:
    """Pack active candidate-lane positions into a fixed-capacity stream."""
    stream_capacity = active_lane_mask.shape[0]
    lane_indices = jnp.nonzero(active_lane_mask, size=stream_capacity, fill_value=0)[0]
    active_count = jnp.sum(active_lane_mask, dtype=jnp.int32)
    active_mask = jnp.arange(stream_capacity, dtype=jnp.int32) < active_count
    return FirthLaneStreamPlan(
        lane_indices=lane_indices,
        active_mask=active_mask,
        active_count=active_count,
    )


def scatter_firth_variant_result_by_lane_stream(
    *,
    base_result: regenie2_binary_firth_types.FirthVariantResult,
    lane_indices: jax.Array,
    active_mask: jax.Array,
    stream_result: regenie2_binary_firth_types.FirthVariantResult,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Scatter stream-ordered Firth results back into candidate-lane order."""
    result_capacity = base_result.beta.shape[0]
    inactive_index = jnp.asarray(result_capacity, dtype=jnp.int32)
    scatter_indices = jnp.where(active_mask, lane_indices, inactive_index)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=base_result.beta.at[scatter_indices].set(stream_result.beta, mode="drop"),
        standard_error=base_result.standard_error.at[scatter_indices].set(
            stream_result.standard_error,
            mode="drop",
        ),
        chi_squared=base_result.chi_squared.at[scatter_indices].set(stream_result.chi_squared, mode="drop"),
        log10_p_value=base_result.log10_p_value.at[scatter_indices].set(
            stream_result.log10_p_value,
            mode="drop",
        ),
        penalized_log_likelihood=base_result.penalized_log_likelihood.at[scatter_indices].set(
            stream_result.penalized_log_likelihood,
            mode="drop",
        ),
        converged_mask=base_result.converged_mask.at[scatter_indices].set(
            stream_result.converged_mask,
            mode="drop",
        ),
        valid_mask=base_result.valid_mask.at[scatter_indices].set(stream_result.valid_mask, mode="drop"),
        iteration_count=base_result.iteration_count.at[scatter_indices].set(
            stream_result.iteration_count,
            mode="drop",
        ),
        failure_code=base_result.failure_code.at[scatter_indices].set(stream_result.failure_code, mode="drop"),
        convergence_reason_code=base_result.convergence_reason_code.at[scatter_indices].set(
            stream_result.convergence_reason_code,
            mode="drop",
        ),
        correction_code=base_result.correction_code.at[scatter_indices].set(
            stream_result.correction_code,
            mode="drop",
        ),
        sparse_correction_mask=base_result.sparse_correction_mask.at[scatter_indices].set(
            stream_result.sparse_correction_mask,
            mode="drop",
        ),
        pseudo_firth_iteration_count=base_result.pseudo_firth_iteration_count.at[scatter_indices].set(
            stream_result.pseudo_firth_iteration_count,
            mode="drop",
        ),
        nr_zero_start_iteration_count=base_result.nr_zero_start_iteration_count.at[scatter_indices].set(
            stream_result.nr_zero_start_iteration_count,
            mode="drop",
        ),
        nr_warm_start_iteration_count=base_result.nr_warm_start_iteration_count.at[scatter_indices].set(
            stream_result.nr_warm_start_iteration_count,
            mode="drop",
        ),
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


def compute_firth_multi_variantwise(
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute device-side Firth fits for lane-specific multi-trait candidates."""
    del null_logistic_coefficients

    def fit_variant(
        phenotype_vector: jax.Array,
        null_firth_offset: jax.Array,
        genotype_vector: jax.Array,
        raw_genotype_vector: jax.Array,
        loco_offset: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
        lane_null_penalized_log_likelihood: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        if not kernel_config.approximate_firth.use_block_math:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
                phenotype_vector=jnp.asarray(phenotype_vector, dtype=jnp.float64),
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=jnp.asarray(null_firth_offset, dtype=jnp.float64),
                carrier_sample_mask=raw_genotype_vector
                > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
                sparse_correction=sparse_correction,
                warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
                skip_firth=skip_firth,
                null_failed=~jnp.isfinite(lane_null_penalized_log_likelihood),
                kernel_config=kernel_config,
            )
        return regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            loco_offset=loco_offset,
            initial_coefficients=variant_initial_coefficients,
            skip_firth=skip_firth,
            null_penalized_log_likelihood=lane_null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
        phenotype_matrix,
        null_firth_offset_matrix,
        genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
        loco_offset_matrix,
        initial_coefficients,
        skip_firth_mask,
        sparse_correction_mask,
        null_penalized_log_likelihood,
    )


def build_compact_sparse_carrier_indices(
    *,
    raw_genotype_matrix_by_variant: jax.Array,
    sparse_carrier_dosage_threshold: float,
) -> jax.Array:
    """Build fixed-capacity carrier sample indices for sparse Firth lanes."""
    carrier_sample_mask = raw_genotype_matrix_by_variant > sparse_carrier_dosage_threshold

    def build_one_lane(carrier_mask: jax.Array) -> jax.Array:
        return jnp.nonzero(carrier_mask, size=SPARSE_FIRTH_CARRIER_CAPACITY, fill_value=0)[0]

    return jax.vmap(build_one_lane)(carrier_sample_mask)


def compute_compact_sparse_firth_variantwise_fixed_batches(
    *,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    offset_matrix: jax.Array,
    active_carrier_slot_mask: jax.Array,
    full_null_deviance: jax.Array,
    active_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_failed_mask: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute compact sparse scalar Firth fits using fixed-size carrier slots."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    phenotype_batches = phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    offset_batches = offset_matrix.reshape((batch_count, firth_batch_size, -1))
    active_carrier_slot_mask_batches = active_carrier_slot_mask.reshape((batch_count, firth_batch_size, -1))
    full_null_deviance_batches = full_null_deviance.reshape((batch_count, firth_batch_size))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    null_failed_mask_batches = null_failed_mask.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def fit_variant(
            phenotype_vector: jax.Array,
            genotype_vector: jax.Array,
            offset_vector: jax.Array,
            carrier_slot_mask: jax.Array,
            lane_full_null_deviance: jax.Array,
            skip_firth: jax.Array,
            null_failed: jax.Array,
        ) -> regenie2_binary_firth_types.FirthVariantResult:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth_compact_carriers(
                phenotype_vector=phenotype_vector,
                genotype_vector=genotype_vector,
                offset_vector=offset_vector,
                active_carrier_slot_mask=carrier_slot_mask,
                full_null_deviance=lane_full_null_deviance,
                warm_start_beta=jnp.asarray(0.0, dtype=offset_vector.dtype),
                skip_firth=skip_firth,
                null_failed=null_failed,
                kernel_config=kernel_config,
            )

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0, 0, 0))(
                phenotype_batches[batch_index],
                genotype_batches[batch_index],
                offset_batches[batch_index],
                active_carrier_slot_mask_batches[batch_index],
                full_null_deviance_batches[batch_index],
                ~active_mask_batches[batch_index],
                null_failed_mask_batches[batch_index],
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


def compute_firth_variantwise_fixed_batches_without_sparse_compaction(
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


def compute_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute multi-trait Firth fits for flattened candidate lanes using fixed-size batches."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    null_logistic_coefficient_batches = null_logistic_coefficients.reshape((batch_count, firth_batch_size, -1))
    null_firth_offset_batches = null_firth_offset_matrix.reshape((batch_count, firth_batch_size, -1))
    phenotype_batches = phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    raw_genotype_batches = raw_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    loco_offset_batches = loco_offset_matrix.reshape((batch_count, firth_batch_size, -1))
    initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    sparse_correction_mask_batches = sparse_correction_mask.reshape((batch_count, firth_batch_size))
    null_penalized_log_likelihood_batches = null_penalized_log_likelihood.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return compute_firth_multi_variantwise(
                covariate_matrix=covariate_matrix,
                null_logistic_coefficients=null_logistic_coefficient_batches[batch_index],
                null_firth_offset_matrix=null_firth_offset_batches[batch_index],
                phenotype_matrix=phenotype_batches[batch_index],
                genotype_matrix_by_variant=genotype_batches[batch_index],
                raw_genotype_matrix_by_variant=raw_genotype_batches[batch_index],
                loco_offset_matrix=loco_offset_batches[batch_index],
                initial_coefficients=initial_coefficient_batches[batch_index],
                skip_firth_mask=~active_mask_batches[batch_index],
                sparse_correction_mask=sparse_correction_mask_batches[batch_index],
                null_penalized_log_likelihood=null_penalized_log_likelihood_batches[batch_index],
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
    full_null_deviance: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute single-trait Firth fits with compact sparse lanes when eligible."""

    def compute_without_sparse_compaction() -> regenie2_binary_firth_types.FirthVariantResult:
        return compute_firth_variantwise_fixed_batches_without_sparse_compaction(
            covariate_matrix=covariate_matrix,
            null_logistic_coefficients=null_logistic_coefficients,
            null_firth_offset=null_firth_offset,
            phenotype_vector=phenotype_vector,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
            loco_offset=loco_offset,
            initial_coefficients=initial_coefficients,
            active_mask=active_mask,
            sparse_correction_mask=sparse_correction_mask,
            fallback_count=fallback_count,
            firth_batch_size=firth_batch_size,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    if kernel_config.approximate_firth.use_block_math:
        return compute_without_sparse_compaction()

    def compute_with_sparse_compaction(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
        carrier_sample_mask = (
            raw_genotype_matrix_by_variant > kernel_config.approximate_firth.sparse_carrier_dosage_threshold
        )
        carrier_count = jnp.sum(carrier_sample_mask, axis=1, dtype=jnp.int32)
        compact_sparse_lane_mask = (
            active_mask & sparse_correction_mask & (carrier_count <= SPARSE_FIRTH_CARRIER_CAPACITY)
        )

        def compute_split_path(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            dense_lane_mask = active_mask & (~compact_sparse_lane_mask)
            dense_stream_plan = build_firth_lane_stream_plan(dense_lane_mask)
            compact_stream_plan = build_firth_lane_stream_plan(compact_sparse_lane_mask)

            dense_result = compute_firth_variantwise_fixed_batches_without_sparse_compaction(
                covariate_matrix=covariate_matrix,
                null_logistic_coefficients=null_logistic_coefficients,
                null_firth_offset=null_firth_offset,
                phenotype_vector=phenotype_vector,
                genotype_matrix_by_variant=jnp.take(genotype_matrix_by_variant, dense_stream_plan.lane_indices, axis=0),
                raw_genotype_matrix_by_variant=jnp.take(
                    raw_genotype_matrix_by_variant,
                    dense_stream_plan.lane_indices,
                    axis=0,
                ),
                loco_offset=loco_offset,
                initial_coefficients=jnp.take(initial_coefficients, dense_stream_plan.lane_indices, axis=0),
                active_mask=dense_stream_plan.active_mask,
                sparse_correction_mask=jnp.take(sparse_correction_mask, dense_stream_plan.lane_indices, axis=0),
                fallback_count=dense_stream_plan.active_count,
                firth_batch_size=firth_batch_size,
                null_penalized_log_likelihood=null_penalized_log_likelihood,
                kernel_config=kernel_config,
            )

            compact_lane_raw_genotype_matrix = jnp.take(
                raw_genotype_matrix_by_variant,
                compact_stream_plan.lane_indices,
                axis=0,
            )
            compact_carrier_indices = build_compact_sparse_carrier_indices(
                raw_genotype_matrix_by_variant=compact_lane_raw_genotype_matrix,
                sparse_carrier_dosage_threshold=kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
            )
            compact_carrier_count = jnp.take(carrier_count, compact_stream_plan.lane_indices, axis=0)
            compact_carrier_slot_mask = (
                jnp.arange(SPARSE_FIRTH_CARRIER_CAPACITY, dtype=jnp.int32)[None, :] < compact_carrier_count[:, None]
            ) & compact_stream_plan.active_mask[:, None]
            compact_lane_genotype_matrix = jnp.take(
                genotype_matrix_by_variant,
                compact_stream_plan.lane_indices,
                axis=0,
            )
            compact_genotype_matrix = jnp.take_along_axis(
                compact_lane_genotype_matrix,
                compact_carrier_indices,
                axis=1,
            )
            compact_phenotype_matrix = jnp.take(
                jnp.asarray(phenotype_vector, dtype=jnp.float64),
                compact_carrier_indices,
                axis=0,
            )
            compact_offset_matrix = jnp.take(
                jnp.asarray(null_firth_offset, dtype=jnp.float64),
                compact_carrier_indices,
                axis=0,
            )
            compact_result = compute_compact_sparse_firth_variantwise_fixed_batches(
                phenotype_matrix=compact_phenotype_matrix,
                genotype_matrix_by_variant=jnp.asarray(compact_genotype_matrix, dtype=jnp.float64),
                offset_matrix=compact_offset_matrix,
                active_carrier_slot_mask=compact_carrier_slot_mask,
                full_null_deviance=jnp.full(
                    (active_mask.shape[0],),
                    jnp.asarray(full_null_deviance, dtype=jnp.float64),
                    dtype=jnp.float64,
                ),
                active_mask=compact_stream_plan.active_mask,
                fallback_count=compact_stream_plan.active_count,
                firth_batch_size=firth_batch_size,
                null_failed_mask=jnp.full(
                    (active_mask.shape[0],),
                    ~jnp.isfinite(null_penalized_log_likelihood),
                    dtype=jnp.bool_,
                ),
                kernel_config=kernel_config,
            )

            empty_result = regenie2_binary_firth_types.build_empty_firth_variant_result(active_mask.shape[0])
            scattered_dense_result = scatter_firth_variant_result_by_lane_stream(
                base_result=empty_result,
                lane_indices=dense_stream_plan.lane_indices,
                active_mask=dense_stream_plan.active_mask,
                stream_result=dense_result,
            )
            return scatter_firth_variant_result_by_lane_stream(
                base_result=scattered_dense_result,
                lane_indices=compact_stream_plan.lane_indices,
                active_mask=compact_stream_plan.active_mask,
                stream_result=compact_result,
            )

        return jax.lax.cond(
            jnp.any(compact_sparse_lane_mask),
            compute_split_path,
            lambda _: compute_without_sparse_compaction(),
            operand=None,
        )

    return jax.lax.cond(
        jnp.any(active_mask & sparse_correction_mask),
        compute_with_sparse_compaction,
        lambda _: compute_without_sparse_compaction(),
        operand=None,
    )


def compute_firth_multi_variantwise_fixed_batches(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    full_null_deviance: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute multi-trait Firth fits with compact sparse lanes when eligible."""

    def compute_without_sparse_compaction() -> regenie2_binary_firth_types.FirthVariantResult:
        return compute_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
            covariate_matrix=covariate_matrix,
            null_logistic_coefficients=null_logistic_coefficients,
            null_firth_offset_matrix=null_firth_offset_matrix,
            phenotype_matrix=phenotype_matrix,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
            loco_offset_matrix=loco_offset_matrix,
            initial_coefficients=initial_coefficients,
            active_mask=active_mask,
            sparse_correction_mask=sparse_correction_mask,
            fallback_count=fallback_count,
            firth_batch_size=firth_batch_size,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    if kernel_config.approximate_firth.use_block_math:
        return compute_without_sparse_compaction()

    def compute_with_sparse_compaction(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
        carrier_sample_mask = (
            raw_genotype_matrix_by_variant > kernel_config.approximate_firth.sparse_carrier_dosage_threshold
        )
        carrier_count = jnp.sum(carrier_sample_mask, axis=1, dtype=jnp.int32)
        compact_sparse_lane_mask = (
            active_mask & sparse_correction_mask & (carrier_count <= SPARSE_FIRTH_CARRIER_CAPACITY)
        )

        def compute_split_path(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            dense_lane_mask = active_mask & (~compact_sparse_lane_mask)
            dense_stream_plan = build_firth_lane_stream_plan(dense_lane_mask)
            compact_stream_plan = build_firth_lane_stream_plan(compact_sparse_lane_mask)

            dense_result = compute_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
                covariate_matrix=covariate_matrix,
                null_logistic_coefficients=jnp.take(null_logistic_coefficients, dense_stream_plan.lane_indices, axis=0),
                null_firth_offset_matrix=jnp.take(null_firth_offset_matrix, dense_stream_plan.lane_indices, axis=0),
                phenotype_matrix=jnp.take(phenotype_matrix, dense_stream_plan.lane_indices, axis=0),
                genotype_matrix_by_variant=jnp.take(genotype_matrix_by_variant, dense_stream_plan.lane_indices, axis=0),
                raw_genotype_matrix_by_variant=jnp.take(
                    raw_genotype_matrix_by_variant,
                    dense_stream_plan.lane_indices,
                    axis=0,
                ),
                loco_offset_matrix=jnp.take(loco_offset_matrix, dense_stream_plan.lane_indices, axis=0),
                initial_coefficients=jnp.take(initial_coefficients, dense_stream_plan.lane_indices, axis=0),
                active_mask=dense_stream_plan.active_mask,
                sparse_correction_mask=jnp.take(sparse_correction_mask, dense_stream_plan.lane_indices, axis=0),
                fallback_count=dense_stream_plan.active_count,
                firth_batch_size=firth_batch_size,
                null_penalized_log_likelihood=jnp.take(
                    null_penalized_log_likelihood,
                    dense_stream_plan.lane_indices,
                    axis=0,
                ),
                kernel_config=kernel_config,
            )

            compact_lane_raw_genotype_matrix = jnp.take(
                raw_genotype_matrix_by_variant,
                compact_stream_plan.lane_indices,
                axis=0,
            )
            compact_carrier_indices = build_compact_sparse_carrier_indices(
                raw_genotype_matrix_by_variant=compact_lane_raw_genotype_matrix,
                sparse_carrier_dosage_threshold=kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
            )
            compact_carrier_count = jnp.take(carrier_count, compact_stream_plan.lane_indices, axis=0)
            compact_carrier_slot_mask = (
                jnp.arange(SPARSE_FIRTH_CARRIER_CAPACITY, dtype=jnp.int32)[None, :] < compact_carrier_count[:, None]
            ) & compact_stream_plan.active_mask[:, None]
            compact_lane_genotype_matrix = jnp.take(
                genotype_matrix_by_variant,
                compact_stream_plan.lane_indices,
                axis=0,
            )
            compact_lane_phenotype_matrix = jnp.take(phenotype_matrix, compact_stream_plan.lane_indices, axis=0)
            compact_lane_offset_matrix = jnp.take(null_firth_offset_matrix, compact_stream_plan.lane_indices, axis=0)
            compact_genotype_matrix = jnp.take_along_axis(
                compact_lane_genotype_matrix,
                compact_carrier_indices,
                axis=1,
            )
            compact_phenotype_matrix = jnp.take_along_axis(
                compact_lane_phenotype_matrix,
                compact_carrier_indices,
                axis=1,
            )
            compact_offset_matrix = jnp.take_along_axis(compact_lane_offset_matrix, compact_carrier_indices, axis=1)
            compact_result = compute_compact_sparse_firth_variantwise_fixed_batches(
                phenotype_matrix=jnp.asarray(compact_phenotype_matrix, dtype=jnp.float64),
                genotype_matrix_by_variant=jnp.asarray(compact_genotype_matrix, dtype=jnp.float64),
                offset_matrix=jnp.asarray(compact_offset_matrix, dtype=jnp.float64),
                active_carrier_slot_mask=compact_carrier_slot_mask,
                full_null_deviance=jnp.take(full_null_deviance, compact_stream_plan.lane_indices, axis=0),
                active_mask=compact_stream_plan.active_mask,
                fallback_count=compact_stream_plan.active_count,
                firth_batch_size=firth_batch_size,
                null_failed_mask=~jnp.isfinite(
                    jnp.take(null_penalized_log_likelihood, compact_stream_plan.lane_indices)
                ),
                kernel_config=kernel_config,
            )

            empty_result = regenie2_binary_firth_types.build_empty_firth_variant_result(active_mask.shape[0])
            scattered_dense_result = scatter_firth_variant_result_by_lane_stream(
                base_result=empty_result,
                lane_indices=dense_stream_plan.lane_indices,
                active_mask=dense_stream_plan.active_mask,
                stream_result=dense_result,
            )
            return scatter_firth_variant_result_by_lane_stream(
                base_result=scattered_dense_result,
                lane_indices=compact_stream_plan.lane_indices,
                active_mask=compact_stream_plan.active_mask,
                stream_result=compact_result,
            )

        return jax.lax.cond(
            jnp.any(compact_sparse_lane_mask),
            compute_split_path,
            lambda _: compute_without_sparse_compaction(),
            operand=None,
        )

    return jax.lax.cond(
        jnp.any(active_mask & sparse_correction_mask),
        compute_with_sparse_compaction,
        lambda _: compute_without_sparse_compaction(),
        operand=None,
    )
