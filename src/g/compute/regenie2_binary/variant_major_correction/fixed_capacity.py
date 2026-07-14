"""Fixed-capacity Firth correction kernels for variant-major binary chunks."""

from __future__ import annotations

import typing

import jax.numpy as jnp

from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth.batch import compute as regenie2_binary_firth_batch_compute
from g.compute.regenie2_binary.firth.batch import prepare as regenie2_binary_firth_batch_prepare

if typing.TYPE_CHECKING:
    import jax

    from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


def merge_fixed_capacity_firth_result(
    *,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_result: regenie2_binary_firth_types.FirthVariantResult,
    lanes: regenie2_binary_candidate_planning.FirthCandidateLaneInputs,
    genotype_flip_mask: jax.Array,
    candidate_capacity: int,
    firth_se: bool,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Merge active fixed-capacity candidate results and drop padded lanes."""
    active_flat_positions = jnp.arange(candidate_capacity, dtype=jnp.int32)
    active_merge_mask = lanes.flat_active_mask[:candidate_capacity]
    active_trait_indices = jnp.where(
        active_merge_mask,
        lanes.flat_trait_indices[:candidate_capacity],
        jnp.asarray(result.beta.shape[0], dtype=jnp.int32),
    )
    active_variant_indices = lanes.flat_variant_indices[:candidate_capacity]
    return regenie2_binary_correction.merge_firth_variant_result_into_multi_chunk(
        result=result,
        firth_result=firth_result,
        active_flat_positions=active_flat_positions,
        active_merge_mask=active_merge_mask,
        active_trait_indices=active_trait_indices,
        active_variant_indices=active_variant_indices,
        genotype_flip_mask=genotype_flip_mask,
        firth_se=firth_se,
    )


def apply_selected_firth_candidate_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    selected_rows: regenie2_binary_firth_batch_prepare.SelectedMultiFirthCandidateRows,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_se: bool,
    fallback_count: jax.Array,
    candidate_capacity: int,
    firth_batch_size: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Prepare, compute, and merge one statically selected Firth solver payload."""
    if kernel_config.approximate_firth.use_block_math:
        candidate_inputs = regenie2_binary_firth_batch_prepare.prepare_block_firth_candidate_batch(
            chromosome_state=chromosome_state,
            selected_rows=selected_rows,
            score_beta=result.beta,
            sparse_candidate_mask=sparse_candidate_mask,
            order_candidates=order_candidates,
            kernel_config=kernel_config,
        )
        firth_result = regenie2_binary_firth_batch_compute.compute_block_firth_multi_variantwise_fixed_batches(
            covariate_matrix=chromosome_state.covariate_matrix,
            candidate_inputs=candidate_inputs,
            fallback_count=fallback_count,
            firth_batch_size=firth_batch_size,
            kernel_config=kernel_config,
        )
        return merge_fixed_capacity_firth_result(
            result=result,
            firth_result=firth_result,
            lanes=candidate_inputs.lanes,
            genotype_flip_mask=jnp.zeros_like(candidate_inputs.lanes.flat_active_mask),
            candidate_capacity=candidate_capacity,
            firth_se=firth_se,
        )

    candidate_inputs = regenie2_binary_firth_batch_prepare.prepare_scalar_firth_candidate_batch(
        chromosome_state=chromosome_state,
        selected_rows=selected_rows,
        sparse_candidate_mask=sparse_candidate_mask,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        native_genotype_mean=native_genotype_mean,
    )
    firth_result = regenie2_binary_firth_batch_compute.compute_scalar_firth_multi_variantwise_fixed_batches(
        candidate_inputs=candidate_inputs,
        fallback_count=fallback_count,
        firth_batch_size=firth_batch_size,
        kernel_config=kernel_config,
    )
    return merge_fixed_capacity_firth_result(
        result=result,
        firth_result=firth_result,
        lanes=candidate_inputs.lanes,
        genotype_flip_mask=candidate_inputs.genotype_flip_mask,
        candidate_capacity=candidate_capacity,
        firth_se=firth_se,
    )


def apply_firth_multi_variant_major_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_se: bool,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply device-resident multi-trait Firth corrections with a fixed candidate capacity."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    selected_rows = regenie2_binary_firth_batch_prepare.select_multi_firth_candidate_rows(
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        candidate_mask=candidate_mask,
        candidate_capacity=candidate_capacity,
        firth_batch_size=firth_batch_size,
    )
    return apply_selected_firth_candidate_corrections(
        chromosome_state=chromosome_state,
        selected_rows=selected_rows,
        result=result,
        firth_se=firth_se,
        fallback_count=fallback_count,
        candidate_capacity=candidate_capacity,
        firth_batch_size=firth_batch_size,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        native_genotype_mean=native_genotype_mean,
    )
