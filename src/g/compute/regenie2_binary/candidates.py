"""Candidate planning helpers for REGENIE step 2 binary Firth fallback."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import genotype as compute_genotype
from g.compute.regenie2_binary import config as regenie2_binary_config


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
        raw_genotype_matrix_by_variant: Raw candidate genotypes matching the solver's chosen allele orientation.
        genotype_flip_mask: Whether each candidate lane needs beta sign restoration after correction.
        sparse_correction_mask: Whether each candidate lane uses sparse carrier-only correction.
        heuristic_firth_mask: Whether each lane uses the separation-oriented initializer.

    """

    flat_fallback_indices: jax.Array
    flat_active_mask: jax.Array
    genotype_matrix_by_variant: jax.Array
    raw_genotype_matrix_by_variant: jax.Array
    genotype_flip_mask: jax.Array
    sparse_correction_mask: jax.Array
    heuristic_firth_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MultiFirthCandidateBatchInputs:
    """Fixed-shape multi-trait Firth candidate inputs after lane ordering.

    Attributes:
        flat_fallback_indices: Flattened trait-variant candidate lane ids in batch order.
        flat_trait_indices: Trait indices matching each flattened candidate lane.
        flat_variant_indices: Variant indices matching each flattened candidate lane.
        flat_active_mask: Active-lane mask in flattened batch order.
        genotype_matrix_by_variant: Candidate genotypes in flattened batch order.
        raw_genotype_matrix_by_variant: Raw candidate genotypes matching the solver's chosen allele orientation.
        genotype_flip_mask: Whether each candidate lane needs beta sign restoration after correction.
        sparse_correction_mask: Whether each candidate lane uses sparse carrier-only correction.
        heuristic_firth_mask: Whether each lane uses the separation-oriented initializer.
        phenotype_matrix: Lane-specific phenotype vectors.
        null_logistic_coefficients: Lane-specific null logistic coefficients.
        null_firth_offset_matrix: Lane-specific null Firth offsets.
        loco_offset_matrix: Lane-specific LOCO offsets.
        null_firth_penalized_log_likelihood: Lane-specific null Firth penalized log-likelihoods.

    """

    flat_fallback_indices: jax.Array
    flat_trait_indices: jax.Array
    flat_variant_indices: jax.Array
    flat_active_mask: jax.Array
    genotype_matrix_by_variant: jax.Array
    raw_genotype_matrix_by_variant: jax.Array
    genotype_flip_mask: jax.Array
    sparse_correction_mask: jax.Array
    heuristic_firth_mask: jax.Array
    phenotype_matrix: jax.Array
    null_logistic_coefficients: jax.Array
    null_firth_offset_matrix: jax.Array
    loco_offset_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array


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
    return build_firth_candidate_capacity_plan(
        variant_count=trait_count * variant_count,
        preferred_candidate_capacity=preferred_candidate_capacity * trait_count,
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


def build_device_multi_firth_batch_plan(
    fallback_mask: jax.Array,
    candidate_capacity: int,
    firth_batch_size: int,
) -> FirthBatchPlan:
    """Build fixed-shape flattened trait-variant lane batches on device."""
    return build_device_firth_batch_plan(
        fallback_mask.reshape((-1,)),
        candidate_capacity=candidate_capacity,
        firth_batch_size=firth_batch_size,
    )


def group_firth_candidate_batch_inputs(
    *,
    flat_fallback_indices: jax.Array,
    flat_active_mask: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    genotype_flip_mask: jax.Array,
    sparse_correction_mask: jax.Array,
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
        raw_genotype_matrix_by_variant=jnp.take(raw_genotype_matrix_by_variant, sort_order, axis=0),
        genotype_flip_mask=jnp.take(genotype_flip_mask, sort_order, axis=0),
        sparse_correction_mask=jnp.take(sparse_correction_mask, sort_order, axis=0),
        heuristic_firth_mask=jnp.take(heuristic_firth_mask, sort_order, axis=0),
    )


def group_multi_firth_candidate_batch_inputs(
    *,
    flat_fallback_indices: jax.Array,
    flat_trait_indices: jax.Array,
    flat_variant_indices: jax.Array,
    flat_active_mask: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    genotype_flip_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    heuristic_firth_mask: jax.Array,
    phenotype_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    loco_offset_matrix: jax.Array,
    null_firth_penalized_log_likelihood: jax.Array,
) -> MultiFirthCandidateBatchInputs:
    """Group likely long-running multi-trait Firth lanes before fixed-size batching."""
    inactive_sort_key = jnp.asarray(2, dtype=jnp.int32)
    sort_key = jnp.where(flat_active_mask, heuristic_firth_mask.astype(jnp.int32), inactive_sort_key)
    sort_order = jnp.argsort(sort_key, stable=True)
    return MultiFirthCandidateBatchInputs(
        flat_fallback_indices=jnp.take(flat_fallback_indices, sort_order, axis=0),
        flat_trait_indices=jnp.take(flat_trait_indices, sort_order, axis=0),
        flat_variant_indices=jnp.take(flat_variant_indices, sort_order, axis=0),
        flat_active_mask=jnp.take(flat_active_mask, sort_order, axis=0),
        genotype_matrix_by_variant=jnp.take(genotype_matrix_by_variant, sort_order, axis=0),
        raw_genotype_matrix_by_variant=jnp.take(raw_genotype_matrix_by_variant, sort_order, axis=0),
        genotype_flip_mask=jnp.take(genotype_flip_mask, sort_order, axis=0),
        sparse_correction_mask=jnp.take(sparse_correction_mask, sort_order, axis=0),
        heuristic_firth_mask=jnp.take(heuristic_firth_mask, sort_order, axis=0),
        phenotype_matrix=jnp.take(phenotype_matrix, sort_order, axis=0),
        null_logistic_coefficients=jnp.take(null_logistic_coefficients, sort_order, axis=0),
        null_firth_offset_matrix=jnp.take(null_firth_offset_matrix, sort_order, axis=0),
        loco_offset_matrix=jnp.take(loco_offset_matrix, sort_order, axis=0),
        null_firth_penalized_log_likelihood=jnp.take(null_firth_penalized_log_likelihood, sort_order, axis=0),
    )
