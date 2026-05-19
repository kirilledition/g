"""Candidate planning helpers for REGENIE step 2 binary Firth fallback."""

from __future__ import annotations

import functools
import math
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import g.compute.regenie2_binary_diagnostics as regenie2_binary_diagnostics
import g.types as g_types

DEFAULT_FIRTH_BATCH_SIZE = 64
DEFAULT_FIRTH_CANDIDATE_CAPACITY = 1024
configured_firth_batch_size = DEFAULT_FIRTH_BATCH_SIZE
configured_firth_candidate_capacity = DEFAULT_FIRTH_CANDIDATE_CAPACITY


def configure_firth_candidate_planning(*, firth_batch_size: int, firth_candidate_capacity: int) -> None:
    """Configure fixed-shape Firth planning settings."""
    if firth_batch_size <= 0:
        message = "Firth batch size must be positive."
        raise ValueError(message)
    if firth_candidate_capacity <= 0:
        message = "Firth candidate capacity must be positive."
        raise ValueError(message)
    global configured_firth_batch_size, configured_firth_candidate_capacity
    configured_firth_batch_size = firth_batch_size
    configured_firth_candidate_capacity = firth_candidate_capacity
    get_firth_batch_size.cache_clear()
    get_firth_candidate_capacity.cache_clear()


@functools.cache
def get_firth_batch_size() -> int:
    """Resolve the active fixed Firth batch size."""
    return configured_firth_batch_size


@functools.cache
def get_firth_candidate_capacity() -> int:
    """Resolve the active fixed Firth candidate lane capacity."""
    return configured_firth_candidate_capacity


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


def build_extra_code(
    log10_p_value: jax.Array,
    valid_mask: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
) -> jax.Array:
    """Select correction labels from score-test statistics."""
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        candidate_mask = jnp.zeros_like(valid_mask, dtype=jnp.bool_)
        correction_code = regenie2_binary_diagnostics.EXTRA_CODE_SCORE
    elif correction_plan.method == g_types.BinaryFallbackMethod.FIRTH_APPROXIMATE:
        fallback_log10p_threshold = -math.log10(correction_plan.p_threshold)
        candidate_mask = log10_p_value > fallback_log10p_threshold
        correction_code = regenie2_binary_diagnostics.EXTRA_CODE_FIRTH
    elif correction_plan.method == g_types.BinaryFallbackMethod.FIRTH:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    elif correction_plan.method == g_types.BinaryFallbackMethod.SPA:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    else:
        typing.assert_never(correction_plan.method)
    return jnp.where(
        valid_mask,
        jnp.where(candidate_mask, correction_code, regenie2_binary_diagnostics.EXTRA_CODE_SCORE),
        regenie2_binary_diagnostics.EXTRA_CODE_TEST_FAIL,
    ).astype(jnp.int32)


def build_device_firth_batch_plan(
    fallback_mask: jax.Array,
    candidate_capacity: int,
) -> FirthBatchPlan:
    """Build fixed-shape Firth index batches on device."""
    firth_batch_size = get_firth_batch_size()
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
