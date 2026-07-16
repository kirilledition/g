"""Candidate batch preparation helpers for Firth correction."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.common import genotype as compute_genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import state as regenie2_binary_state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SelectedMultiFirthCandidateRows:
    """Selected fixed-capacity candidate rows before solver-specific preparation."""

    flat_active_mask: jax.Array
    flat_trait_indices: jax.Array
    flat_variant_indices: jax.Array
    genotype_matrix_by_variant: jax.Array


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


def select_multi_firth_candidate_rows(
    *,
    genotype_matrix_by_variant: jax.Array,
    candidate_mask: jax.Array,
    candidate_capacity: int,
    firth_batch_size: int,
) -> SelectedMultiFirthCandidateRows:
    """Select fixed-capacity candidate rows from a decoded genotype matrix."""
    variant_count = genotype_matrix_by_variant.shape[0]
    batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan(
        candidate_mask.reshape((-1,)),
        candidate_capacity=candidate_capacity,
        firth_batch_size=firth_batch_size,
    )
    flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
    flat_variant_indices = flat_fallback_indices % variant_count
    return SelectedMultiFirthCandidateRows(
        flat_active_mask=batch_plan.fallback_active_mask_matrix.reshape((-1,)),
        flat_trait_indices=flat_fallback_indices // variant_count,
        flat_variant_indices=flat_variant_indices,
        genotype_matrix_by_variant=jnp.take(
            jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32),
            flat_variant_indices,
            axis=0,
        ),
    )


def build_flat_sparse_candidate_mask(
    *,
    sparse_candidate_mask: jax.Array | None,
    flat_variant_indices: jax.Array,
    flat_active_mask: jax.Array,
) -> jax.Array:
    """Gather the active sparse-correction mask for candidate lanes."""
    if sparse_candidate_mask is None:
        flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
    else:
        flat_sparse_candidate_mask = (
            jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_variant_indices, axis=0)
            & flat_active_mask
        )
    return flat_sparse_candidate_mask


def build_firth_candidate_lane_inputs(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    selected_rows: SelectedMultiFirthCandidateRows,
) -> regenie2_binary_candidate_planning.FirthCandidateLaneInputs:
    """Build per-lane indices and activity with shared trait phenotypes."""
    return regenie2_binary_candidate_planning.FirthCandidateLaneInputs(
        flat_trait_indices=selected_rows.flat_trait_indices,
        flat_variant_indices=selected_rows.flat_variant_indices,
        flat_active_mask=selected_rows.flat_active_mask,
        phenotype_matrix=chromosome_state.phenotype_matrix,
    )


def prepare_scalar_firth_candidate_batch(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    selected_rows: SelectedMultiFirthCandidateRows,
    sparse_candidate_mask: jax.Array | None,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_candidate_planning.ScalarFirthCandidateBatchInputs:
    """Prepare only the arrays consumed by scalar approximate Firth."""
    candidate_genotype_mean = take_candidate_stat_vector(native_genotype_mean, selected_rows.flat_variant_indices)
    genotype_flip_result = compute_genotype.build_regenie_flipped_genotypes(
        selected_rows.genotype_matrix_by_variant,
        native_genotype_mean=candidate_genotype_mean,
    )
    raw_genotype_matrix_by_variant = genotype_flip_result.genotype_matrix_by_variant
    carrier_sample_mask = (
        raw_genotype_matrix_by_variant > kernel_config.approximate_firth.sparse_carrier_dosage_threshold
    )
    flat_sparse_candidate_mask = build_flat_sparse_candidate_mask(
        sparse_candidate_mask=sparse_candidate_mask,
        flat_variant_indices=selected_rows.flat_variant_indices,
        flat_active_mask=selected_rows.flat_active_mask,
    )
    lanes = build_firth_candidate_lane_inputs(
        chromosome_state=chromosome_state,
        selected_rows=selected_rows,
    )
    genotype_matrix_by_variant = residualize_and_scale_multi_genotypes_for_approximate_firth(
        square_root_weight=jnp.take(
            chromosome_state.square_root_weight,
            selected_rows.flat_trait_indices,
            axis=0,
        ),
        weighted_genotype_projection_matrix=jnp.take(
            chromosome_state.weighted_genotype_projection_matrix,
            selected_rows.flat_trait_indices,
            axis=0,
        ),
        genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
    )
    null_firth_penalized_log_likelihood = jnp.take(
        chromosome_state.null_firth_penalized_log_likelihood,
        selected_rows.flat_trait_indices,
        axis=0,
    )
    candidate_inputs = regenie2_binary_candidate_planning.ScalarFirthCandidateBatchInputs(
        lanes=lanes,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        carrier_sample_mask=carrier_sample_mask,
        genotype_flip_mask=genotype_flip_result.flip_mask,
        sparse_correction_mask=flat_sparse_candidate_mask,
        null_firth_offset_matrix=chromosome_state.null_firth_offset_matrix,
        full_null_deviance=jnp.take(
            chromosome_state.full_null_deviance,
            selected_rows.flat_trait_indices,
            axis=0,
        ),
        null_failed_mask=~jnp.isfinite(null_firth_penalized_log_likelihood),
    )
    if not order_candidates:
        return candidate_inputs
    heuristic_firth_mask = (
        regenie2_binary_candidate_planning.compute_multi_firth_pre_dispatch_mask_without_mask(
            genotype_matrix_by_lane=raw_genotype_matrix_by_variant,
            phenotype_matrix=lanes.phenotype_matrix,
            flat_trait_indices=lanes.flat_trait_indices,
        )
        | flat_sparse_candidate_mask
    ) & selected_rows.flat_active_mask
    return regenie2_binary_candidate_planning.group_scalar_firth_candidate_batch_inputs(
        candidate_inputs=candidate_inputs,
        heuristic_firth_mask=heuristic_firth_mask,
    )
