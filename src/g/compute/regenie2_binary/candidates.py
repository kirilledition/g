"""Candidate planning helpers for REGENIE step 2 binary Firth fallback."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthBatchPlan:
    """Fixed-shape Firth candidate batch plan.

    Attributes:
        fallback_index_matrix: Candidate variant indices padded into fixed Firth batches.
        fallback_active_mask_matrix: Active-lane mask matching `fallback_index_matrix`.
        active_flat_position_vector: Fixed-size positions of active lanes in flattened padded batches.

    """

    fallback_index_matrix: jax.Array
    fallback_active_mask_matrix: jax.Array
    active_flat_position_vector: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthCandidateBatchInputs:
    """Fixed-shape Firth candidate inputs after optional lane ordering.

    Attributes:
        flat_fallback_indices: Candidate variant indices in flattened batch order.
        flat_active_mask: Active-lane mask in flattened batch order.
        genotype_matrix_by_variant: Candidate genotypes in flattened batch order.
        heuristic_firth_mask: Whether each lane uses the separation-oriented initializer.

    """

    flat_fallback_indices: jax.Array
    flat_active_mask: jax.Array
    genotype_matrix_by_variant: jax.Array
    heuristic_firth_mask: jax.Array


@dataclass(frozen=True)
class FirthCandidateCapacityPlan:
    """Static candidate capacities for normal and overflow Firth correction paths.

    Attributes:
        bounded_candidate_capacity: Preferred fixed candidate capacity, capped by the current chunk size.
        overflow_candidate_capacity: Full chunk capacity used when candidate count exceeds the bounded capacity.

    """

    bounded_candidate_capacity: int
    overflow_candidate_capacity: int


def build_firth_candidate_capacity_plan(
    *,
    variant_count: int,
    preferred_candidate_capacity: int,
) -> FirthCandidateCapacityPlan:
    """Build static capacities for device Firth candidate dispatch."""
    if variant_count <= 0:
        message = "Variant count must be positive."
        raise ValueError(message)
    if preferred_candidate_capacity <= 0:
        message = "Preferred Firth candidate capacity must be positive."
        raise ValueError(message)
    return FirthCandidateCapacityPlan(
        bounded_candidate_capacity=min(preferred_candidate_capacity, variant_count),
        overflow_candidate_capacity=variant_count,
    )


def count_firth_candidates_on_host(fallback_mask: jax.Array) -> int:
    """Return the Firth candidate count as a host integer for dispatch."""
    fallback_count = jnp.sum(fallback_mask, dtype=jnp.int32)
    return int(jax.device_get(fallback_count))


def select_firth_candidate_capacity(
    *,
    fallback_count: int,
    capacity_plan: FirthCandidateCapacityPlan,
) -> int:
    """Select the fixed Firth correction capacity for a candidate count."""
    if fallback_count > capacity_plan.bounded_candidate_capacity:
        return capacity_plan.overflow_candidate_capacity
    return capacity_plan.bounded_candidate_capacity


def build_device_firth_batch_plan(
    fallback_mask: jax.Array,
    candidate_capacity: int,
    firth_batch_size: int,
) -> FirthBatchPlan:
    """Build fixed-shape Firth index batches on device."""
    max_batch_count = (candidate_capacity + firth_batch_size - 1) // firth_batch_size
    padded_variant_count = max_batch_count * firth_batch_size
    fallback_index_vector = jnp.nonzero(fallback_mask, size=candidate_capacity, fill_value=0)[0]
    fallback_count = jnp.sum(fallback_mask, dtype=jnp.int32)
    padded_index_vector = jnp.pad(
        fallback_index_vector,
        (0, padded_variant_count - candidate_capacity),
        constant_values=0,
    )
    active_mask_vector = jnp.arange(padded_variant_count, dtype=jnp.int32) < fallback_count
    active_flat_position_vector = jnp.nonzero(
        active_mask_vector,
        size=candidate_capacity,
        fill_value=0,
    )[0]
    return FirthBatchPlan(
        fallback_index_matrix=padded_index_vector.reshape((max_batch_count, firth_batch_size)),
        fallback_active_mask_matrix=active_mask_vector.reshape((max_batch_count, firth_batch_size)),
        active_flat_position_vector=active_flat_position_vector,
    )


def group_firth_candidate_batch_inputs(
    *,
    flat_fallback_indices: jax.Array,
    flat_active_mask: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    heuristic_firth_mask: jax.Array,
) -> FirthCandidateBatchInputs:
    """Group likely long-running Firth lanes together before fixed-size batching."""
    inactive_sort_key = jnp.asarray(2, dtype=jnp.int32)
    sort_key = jnp.where(flat_active_mask, heuristic_firth_mask.astype(jnp.int32), inactive_sort_key)
    sort_order = jnp.argsort(sort_key, stable=True)
    return FirthCandidateBatchInputs(
        flat_fallback_indices=jnp.take(flat_fallback_indices, sort_order, axis=0),
        flat_active_mask=jnp.take(flat_active_mask, sort_order, axis=0),
        genotype_matrix_by_variant=jnp.take(genotype_matrix_by_variant, sort_order, axis=0),
        heuristic_firth_mask=jnp.take(heuristic_firth_mask, sort_order, axis=0),
    )
