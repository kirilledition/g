"""Public variant-major candidate correction entrypoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.variant_major_correction import dispatch


def apply_device_candidate_corrections_multi_firth_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply multi-trait Firth corrections with non-blocking device-side capacity dispatch."""
    if genotype_matrix_by_variant.shape[0] == 0:
        return result
    trait_count = chromosome_state.phenotype_matrix.shape[0]
    variant_count = genotype_matrix_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_multi_firth_candidate_capacity_plan(
        trait_count=trait_count,
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    return dispatch.apply_device_candidate_corrections_multi_firth_variant_major_with_device_dispatch(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        tiny_candidate_capacity=capacity_plan.tiny_candidate_capacity,
        small_candidate_capacity=capacity_plan.small_candidate_capacity,
        bounded_candidate_capacity=capacity_plan.bounded_candidate_capacity,
        overflow_candidate_capacity=capacity_plan.overflow_candidate_capacity,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        kernel_config=kernel_config,
    )


def apply_device_candidate_corrections_multi_firth_packed8(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply multi-trait Firth corrections to packed8 chunks without dense chunk decode."""
    if packed_probability_pairs_by_variant.shape[0] == 0:
        return result
    trait_count = chromosome_state.phenotype_matrix.shape[0]
    variant_count = packed_probability_pairs_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_multi_firth_candidate_capacity_plan(
        trait_count=trait_count,
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    candidate_count = jnp.sum(
        result.correction_code == types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
        dtype=jnp.int32,
    )
    return jax.lax.cond(
        candidate_count <= capacity_plan.bounded_candidate_capacity,
        lambda _: dispatch.apply_device_candidate_corrections_multi_firth_packed8_with_device_dispatch(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            result=result,
            correction_plan=correction_plan,
            tiny_candidate_capacity=capacity_plan.tiny_candidate_capacity,
            small_candidate_capacity=capacity_plan.small_candidate_capacity,
            bounded_candidate_capacity=capacity_plan.bounded_candidate_capacity,
            sparse_candidate_mask=sparse_candidate_mask,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            score_dtype=score_dtype,
            kernel_config=kernel_config,
        ),
        lambda _: dispatch.apply_device_candidate_corrections_multi_firth_packed8_with_overflow_dispatch(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            result=result,
            correction_plan=correction_plan,
            overflow_candidate_capacity=capacity_plan.overflow_candidate_capacity,
            sparse_candidate_mask=sparse_candidate_mask,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            score_dtype=score_dtype,
            kernel_config=kernel_config,
        ),
        operand=None,
    )


def apply_device_candidate_corrections_multi_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply multi-trait binary candidate corrections for variant-major genotype chunks."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    return apply_device_candidate_corrections_multi_firth_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        kernel_config=kernel_config,
    )


def apply_device_candidate_corrections_multi_packed8(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply multi-trait binary candidate corrections for packed8 chunks."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    return apply_device_candidate_corrections_multi_firth_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
        kernel_config=kernel_config,
    )
