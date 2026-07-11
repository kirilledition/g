"""Candidate planning helpers for REGENIE step 2 binary Firth fallback."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import genotype as compute_genotype
from g.compute.regenie2_binary import config as regenie2_binary_config

TINY_FIRTH_CANDIDATE_CAPACITY_PER_TRAIT = 64
SMALL_FIRTH_CANDIDATE_CAPACITY_PER_TRAIT = 256
JAX_INT32_INDEX_MAXIMUM = 2_147_483_647


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthBatchPlan:
    """Fixed-shape Firth candidate batch plan.

    Attributes:
        fallback_index_matrix: Candidate variant indices padded into fixed Firth batches.
        fallback_active_mask_matrix: Active-lane mask matching `fallback_index_matrix`.

    """

    fallback_index_matrix: jax.Array
    fallback_active_mask_matrix: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthCandidateLaneInputs:
    """Common fixed-shape candidate-lane inputs after lane ordering.

    Attributes:
        flat_trait_indices: Trait indices matching each flattened candidate lane.
        flat_variant_indices: Variant indices matching each flattened candidate lane.
        flat_active_mask: Active-lane mask in flattened batch order.
        phenotype_matrix: Lane-specific phenotype vectors.

    """

    flat_trait_indices: jax.Array
    flat_variant_indices: jax.Array
    flat_active_mask: jax.Array
    phenotype_matrix: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthCandidateBatchInputs:
    """Fixed-shape candidate inputs retained by the scalar Firth solver.

    Attributes:
        lanes: Common candidate indices, activity, and phenotypes.
        genotype_matrix_by_variant: Residualized candidate genotypes.
        carrier_sample_mask: Carrier samples in the solver's allele orientation.
        genotype_flip_mask: Whether corrected beta signs need restoration.
        sparse_correction_mask: Whether lanes use carrier-only correction.
        null_firth_offset_matrix: Lane-specific null Firth offsets.
        full_null_deviance: Lane-specific full-sample null deviances.
        null_failed_mask: Whether lane-specific null Firth fitting failed.

    """

    lanes: FirthCandidateLaneInputs
    genotype_matrix_by_variant: jax.Array
    carrier_sample_mask: jax.Array
    genotype_flip_mask: jax.Array
    sparse_correction_mask: jax.Array
    null_firth_offset_matrix: jax.Array
    full_null_deviance: jax.Array
    null_failed_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BlockFirthCandidateBatchInputs:
    """Fixed-shape candidate inputs retained by the block Firth solver.

    Attributes:
        lanes: Common candidate indices, activity, and phenotypes.
        genotype_matrix_by_variant: Candidate genotypes in original allele orientation.
        initial_coefficients: Lane-specific full-model starting coefficients.
        loco_offset_matrix: Lane-specific LOCO offsets.
        null_firth_penalized_log_likelihood: Lane-specific null penalized log-likelihoods.

    """

    lanes: FirthCandidateLaneInputs
    genotype_matrix_by_variant: jax.Array
    initial_coefficients: jax.Array
    loco_offset_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array


@dataclass(frozen=True)
class FirthCandidateCapacityPlan:
    """Static candidate capacities for tiered Firth correction paths.

    Attributes:
        tiny_candidate_capacity: Fixed capacity for very small candidate counts.
        small_candidate_capacity: Fixed capacity for small candidate counts.
        bounded_candidate_capacity: Preferred fixed candidate capacity, capped by the current chunk size.
        overflow_candidate_capacity: Full chunk capacity used when candidate count exceeds the bounded capacity.

    """

    tiny_candidate_capacity: int
    small_candidate_capacity: int
    bounded_candidate_capacity: int
    overflow_candidate_capacity: int


def build_compact_int32_indices(active_mask: jax.Array, capacity: int) -> jax.Array:
    """Compact true-mask positions into a fixed-capacity int32 index vector."""
    if active_mask.ndim != 1:
        message = "Index compaction requires a one-dimensional mask."
        raise ValueError(message)
    if capacity <= 0:
        message = "Index compaction capacity must be positive."
        raise ValueError(message)
    if active_mask.size > JAX_INT32_INDEX_MAXIMUM or capacity > JAX_INT32_INDEX_MAXIMUM:
        message = "Index compaction exceeds the JAX int32 index domain."
        raise ValueError(message)
    source_indices = jnp.arange(active_mask.size, dtype=jnp.int32)
    compact_positions = jnp.cumsum(active_mask, dtype=jnp.int32) - 1
    dropped_position = jnp.asarray(capacity, dtype=jnp.int32)
    scatter_positions = jnp.where(active_mask, compact_positions, dropped_position)
    return jnp.zeros((capacity,), dtype=jnp.int32).at[scatter_positions].set(source_indices, mode="drop")


def build_firth_candidate_capacity_plan(
    *,
    variant_count: int,
    preferred_candidate_capacity: int,
    trait_count: int,
) -> FirthCandidateCapacityPlan:
    """Build static capacities for device Firth candidate dispatch."""
    if variant_count <= 0:
        message = "Variant count must be positive."
        raise ValueError(message)
    if preferred_candidate_capacity <= 0:
        message = "Preferred Firth candidate capacity must be positive."
        raise ValueError(message)
    if trait_count <= 0:
        message = "Trait count must be positive."
        raise ValueError(message)
    if variant_count > JAX_INT32_INDEX_MAXIMUM:
        message = "Flattened Firth candidate count exceeds the JAX int32 index domain."
        raise ValueError(message)
    bounded_candidate_capacity = min(preferred_candidate_capacity, variant_count)
    small_candidate_capacity = min(SMALL_FIRTH_CANDIDATE_CAPACITY_PER_TRAIT * trait_count, bounded_candidate_capacity)
    tiny_candidate_capacity = min(TINY_FIRTH_CANDIDATE_CAPACITY_PER_TRAIT * trait_count, small_candidate_capacity)
    return FirthCandidateCapacityPlan(
        tiny_candidate_capacity=tiny_candidate_capacity,
        small_candidate_capacity=small_candidate_capacity,
        bounded_candidate_capacity=bounded_candidate_capacity,
        overflow_candidate_capacity=variant_count,
    )


def build_multi_firth_candidate_capacity_plan(
    *,
    trait_count: int,
    variant_count: int,
    preferred_candidate_capacity: int,
) -> FirthCandidateCapacityPlan:
    """Build flattened-lane capacities for multi-trait Firth correction."""
    if trait_count <= 0:
        message = "Trait count must be positive."
        raise ValueError(message)
    if variant_count <= 0:
        message = "Variant count must be positive."
        raise ValueError(message)
    flattened_candidate_count = trait_count * variant_count
    if flattened_candidate_count > JAX_INT32_INDEX_MAXIMUM:
        message = "Flattened trait-variant candidate count exceeds the JAX int32 index domain."
        raise ValueError(message)
    return build_firth_candidate_capacity_plan(
        variant_count=flattened_candidate_count,
        preferred_candidate_capacity=preferred_candidate_capacity * trait_count,
        trait_count=trait_count,
    )


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


def compute_multi_firth_pre_dispatch_mask_without_mask(
    genotype_matrix_by_lane: jax.Array,
    phenotype_matrix_by_lane: jax.Array,
) -> jax.Array:
    """Identify lane-specific separation candidates for multi-trait Firth correction."""

    def compute_one_lane(genotype_vector: jax.Array, phenotype_vector: jax.Array) -> jax.Array:
        return compute_firth_pre_dispatch_mask_without_mask(
            genotype_matrix_by_variant=genotype_vector[None, :],
            phenotype_vector=phenotype_vector,
        )[0]

    return jax.vmap(compute_one_lane)(genotype_matrix_by_lane, phenotype_matrix_by_lane)


def build_device_firth_batch_plan(
    fallback_mask: jax.Array,
    candidate_capacity: int,
    firth_batch_size: int,
) -> FirthBatchPlan:
    """Build fixed-shape Firth index batches on device."""
    if candidate_capacity <= 0:
        message = "Firth candidate capacity must be positive."
        raise ValueError(message)
    if firth_batch_size <= 0:
        message = "Firth batch size must be positive."
        raise ValueError(message)
    if fallback_mask.size > JAX_INT32_INDEX_MAXIMUM:
        message = "Firth fallback mask exceeds the JAX int32 index domain."
        raise ValueError(message)
    max_batch_count = (candidate_capacity + firth_batch_size - 1) // firth_batch_size
    padded_variant_count = max_batch_count * firth_batch_size
    if padded_variant_count > JAX_INT32_INDEX_MAXIMUM:
        message = "Padded Firth candidate count exceeds the JAX int32 index domain."
        raise ValueError(message)
    fallback_index_vector = build_compact_int32_indices(fallback_mask, candidate_capacity)
    fallback_count = jnp.sum(fallback_mask, dtype=jnp.int32)
    padded_index_vector = jnp.pad(
        fallback_index_vector,
        (0, padded_variant_count - candidate_capacity),
        constant_values=0,
    )
    active_mask_vector = jnp.arange(padded_variant_count, dtype=jnp.int32) < fallback_count
    return FirthBatchPlan(
        fallback_index_matrix=padded_index_vector.reshape((max_batch_count, firth_batch_size)),
        fallback_active_mask_matrix=active_mask_vector.reshape((max_batch_count, firth_batch_size)),
    )


def build_firth_candidate_bucket_order(
    *,
    flat_active_mask: jax.Array,
    heuristic_firth_mask: jax.Array,
) -> jax.Array:
    """Build a stable regular, heuristic, inactive lane order without a full sort."""
    candidate_count = flat_active_mask.shape[0]
    if candidate_count > JAX_INT32_INDEX_MAXIMUM:
        message = "Firth candidate bucket exceeds the JAX int32 index domain."
        raise ValueError(message)
    regular_active_mask = flat_active_mask & (~heuristic_firth_mask)
    heuristic_active_mask = flat_active_mask & heuristic_firth_mask
    inactive_mask = ~flat_active_mask
    regular_indices = build_compact_int32_indices(regular_active_mask, candidate_count)
    heuristic_indices = build_compact_int32_indices(heuristic_active_mask, candidate_count)
    inactive_indices = build_compact_int32_indices(inactive_mask, candidate_count)
    regular_count = jnp.sum(regular_active_mask, dtype=jnp.int32)
    heuristic_count = jnp.sum(heuristic_active_mask, dtype=jnp.int32)
    output_positions = jnp.arange(candidate_count, dtype=jnp.int32)
    heuristic_positions = output_positions - regular_count
    inactive_positions = output_positions - regular_count - heuristic_count
    return jnp.where(
        output_positions < regular_count,
        jnp.take(regular_indices, output_positions, axis=0),
        jnp.where(
            output_positions < regular_count + heuristic_count,
            jnp.take(heuristic_indices, heuristic_positions, axis=0),
            jnp.take(inactive_indices, inactive_positions, axis=0),
        ),
    )


def reorder_firth_candidate_lane_inputs(
    lanes: FirthCandidateLaneInputs,
    sort_order: jax.Array,
) -> FirthCandidateLaneInputs:
    """Reorder common candidate-lane inputs with one device order."""
    return FirthCandidateLaneInputs(
        flat_trait_indices=jnp.take(lanes.flat_trait_indices, sort_order, axis=0),
        flat_variant_indices=jnp.take(lanes.flat_variant_indices, sort_order, axis=0),
        flat_active_mask=jnp.take(lanes.flat_active_mask, sort_order, axis=0),
        phenotype_matrix=jnp.take(lanes.phenotype_matrix, sort_order, axis=0),
    )


def group_scalar_firth_candidate_batch_inputs(
    *,
    candidate_inputs: ScalarFirthCandidateBatchInputs,
    heuristic_firth_mask: jax.Array,
    order_candidates: bool,
) -> ScalarFirthCandidateBatchInputs:
    """Group likely long-running scalar Firth lanes before fixed-size batching."""
    if not order_candidates:
        return candidate_inputs
    sort_order = build_firth_candidate_bucket_order(
        flat_active_mask=candidate_inputs.lanes.flat_active_mask,
        heuristic_firth_mask=heuristic_firth_mask,
    )
    return ScalarFirthCandidateBatchInputs(
        lanes=reorder_firth_candidate_lane_inputs(candidate_inputs.lanes, sort_order),
        genotype_matrix_by_variant=jnp.take(candidate_inputs.genotype_matrix_by_variant, sort_order, axis=0),
        carrier_sample_mask=jnp.take(candidate_inputs.carrier_sample_mask, sort_order, axis=0),
        genotype_flip_mask=jnp.take(candidate_inputs.genotype_flip_mask, sort_order, axis=0),
        sparse_correction_mask=jnp.take(candidate_inputs.sparse_correction_mask, sort_order, axis=0),
        null_firth_offset_matrix=jnp.take(candidate_inputs.null_firth_offset_matrix, sort_order, axis=0),
        full_null_deviance=jnp.take(candidate_inputs.full_null_deviance, sort_order, axis=0),
        null_failed_mask=jnp.take(candidate_inputs.null_failed_mask, sort_order, axis=0),
    )


def group_block_firth_candidate_batch_inputs(
    *,
    candidate_inputs: BlockFirthCandidateBatchInputs,
    heuristic_firth_mask: jax.Array,
    order_candidates: bool,
) -> BlockFirthCandidateBatchInputs:
    """Group likely long-running block Firth lanes before fixed-size batching."""
    if not order_candidates:
        return candidate_inputs
    sort_order = build_firth_candidate_bucket_order(
        flat_active_mask=candidate_inputs.lanes.flat_active_mask,
        heuristic_firth_mask=heuristic_firth_mask,
    )
    return BlockFirthCandidateBatchInputs(
        lanes=reorder_firth_candidate_lane_inputs(candidate_inputs.lanes, sort_order),
        genotype_matrix_by_variant=jnp.take(candidate_inputs.genotype_matrix_by_variant, sort_order, axis=0),
        initial_coefficients=jnp.take(candidate_inputs.initial_coefficients, sort_order, axis=0),
        loco_offset_matrix=jnp.take(candidate_inputs.loco_offset_matrix, sort_order, axis=0),
        null_firth_penalized_log_likelihood=jnp.take(
            candidate_inputs.null_firth_penalized_log_likelihood,
            sort_order,
            axis=0,
        ),
    )
