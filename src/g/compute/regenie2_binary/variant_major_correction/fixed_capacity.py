"""Fixed-capacity Firth correction kernels for variant-major binary chunks."""

from __future__ import annotations

import typing

from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth.batch import compute as regenie2_binary_firth_batch_compute
from g.compute.regenie2_binary.firth.batch import prepare as regenie2_binary_firth_batch_prepare

if typing.TYPE_CHECKING:
    import jax

    from g import types


def apply_firth_multi_variant_major_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply device-resident multi-trait Firth corrections with a fixed candidate capacity."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    prepared_batch = regenie2_binary_firth_batch_prepare.prepare_multi_firth_candidate_batch(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        candidate_mask=candidate_mask,
        score_beta=result.beta,
        sparse_candidate_mask=sparse_candidate_mask,
        candidate_capacity=candidate_capacity,
        firth_batch_size=firth_batch_size,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
    )
    firth_result = regenie2_binary_firth_batch_compute.compute_firth_multi_variantwise_fixed_batches(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_firth_offset_matrix=prepared_batch.candidate_inputs.null_firth_offset_matrix,
        phenotype_matrix=prepared_batch.candidate_inputs.phenotype_matrix,
        genotype_matrix_by_variant=prepared_batch.candidate_inputs.genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=prepared_batch.candidate_inputs.raw_genotype_matrix_by_variant,
        loco_offset_matrix=prepared_batch.candidate_inputs.loco_offset_matrix,
        initial_coefficients=prepared_batch.candidate_inputs.initial_coefficients,
        active_mask=prepared_batch.candidate_inputs.flat_active_mask,
        sparse_correction_mask=prepared_batch.candidate_inputs.sparse_correction_mask,
        fallback_count=fallback_count,
        firth_batch_size=firth_batch_size,
        null_penalized_log_likelihood=prepared_batch.candidate_inputs.null_firth_penalized_log_likelihood,
        full_null_deviance=prepared_batch.full_null_deviance,
        kernel_config=kernel_config,
    )
    active_flat_positions = prepared_batch.batch_plan.active_flat_position_vector
    active_trait_indices = prepared_batch.candidate_inputs.flat_trait_indices[active_flat_positions]
    active_variant_indices = prepared_batch.candidate_inputs.flat_variant_indices[active_flat_positions]
    return regenie2_binary_correction.merge_firth_variant_result_into_multi_chunk(
        result=result,
        firth_result=firth_result,
        active_flat_positions=active_flat_positions,
        active_trait_indices=active_trait_indices,
        active_variant_indices=active_variant_indices,
        genotype_flip_mask=prepared_batch.candidate_inputs.genotype_flip_mask,
        firth_se=correction_plan.firth_se,
    )


def apply_firth_multi_packed8_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply multi-trait Firth corrections by decoding only packed8 candidate rows."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    prepared_batch = regenie2_binary_firth_batch_prepare.prepare_multi_firth_candidate_batch_from_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        candidate_mask=candidate_mask,
        score_beta=result.beta,
        sparse_candidate_mask=sparse_candidate_mask,
        candidate_capacity=candidate_capacity,
        firth_batch_size=firth_batch_size,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    firth_result = regenie2_binary_firth_batch_compute.compute_firth_multi_variantwise_fixed_batches(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_firth_offset_matrix=prepared_batch.candidate_inputs.null_firth_offset_matrix,
        phenotype_matrix=prepared_batch.candidate_inputs.phenotype_matrix,
        genotype_matrix_by_variant=prepared_batch.candidate_inputs.genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=prepared_batch.candidate_inputs.raw_genotype_matrix_by_variant,
        loco_offset_matrix=prepared_batch.candidate_inputs.loco_offset_matrix,
        initial_coefficients=prepared_batch.candidate_inputs.initial_coefficients,
        active_mask=prepared_batch.candidate_inputs.flat_active_mask,
        sparse_correction_mask=prepared_batch.candidate_inputs.sparse_correction_mask,
        fallback_count=fallback_count,
        firth_batch_size=firth_batch_size,
        null_penalized_log_likelihood=prepared_batch.candidate_inputs.null_firth_penalized_log_likelihood,
        full_null_deviance=prepared_batch.full_null_deviance,
        kernel_config=kernel_config,
    )
    active_flat_positions = prepared_batch.batch_plan.active_flat_position_vector
    active_trait_indices = prepared_batch.candidate_inputs.flat_trait_indices[active_flat_positions]
    active_variant_indices = prepared_batch.candidate_inputs.flat_variant_indices[active_flat_positions]
    return regenie2_binary_correction.merge_firth_variant_result_into_multi_chunk(
        result=result,
        firth_result=firth_result,
        active_flat_positions=active_flat_positions,
        active_trait_indices=active_trait_indices,
        active_variant_indices=active_variant_indices,
        genotype_flip_mask=prepared_batch.candidate_inputs.genotype_flip_mask,
        firth_se=correction_plan.firth_se,
    )
