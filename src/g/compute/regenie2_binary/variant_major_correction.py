"""Variant-major binary candidate correction kernels for REGENIE step 2."""

from __future__ import annotations

import collections.abc
import functools
import time

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import batch as regenie2_binary_firth_batch

StageDurationRecorder = collections.abc.Callable[[str, float], None]


def apply_firth_variant_major_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply device-resident Firth corrections with a fixed candidate capacity."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    prepared_batch = regenie2_binary_firth_batch.prepare_firth_candidate_batch(
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
    firth_result = regenie2_binary_firth_batch.compute_firth_variantwise_fixed_batches(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
        null_firth_offset=chromosome_state.null_firth_offset,
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_matrix_by_variant=prepared_batch.candidate_inputs.genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=prepared_batch.candidate_inputs.raw_genotype_matrix_by_variant,
        loco_offset=chromosome_state.loco_offset,
        initial_coefficients=prepared_batch.initial_coefficients,
        active_mask=prepared_batch.candidate_inputs.flat_active_mask,
        sparse_correction_mask=prepared_batch.candidate_inputs.sparse_correction_mask,
        fallback_count=fallback_count,
        firth_batch_size=firth_batch_size,
        null_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood,
        full_null_deviance=prepared_batch.full_null_deviance,
        kernel_config=kernel_config,
    )
    active_flat_positions = prepared_batch.batch_plan.active_flat_position_vector
    active_fallback_indices = prepared_batch.candidate_inputs.flat_fallback_indices[active_flat_positions]
    return regenie2_binary_correction.merge_firth_variant_result_into_chunk(
        result=result,
        firth_result=firth_result,
        active_flat_positions=active_flat_positions,
        active_fallback_indices=active_fallback_indices,
        genotype_flip_mask=prepared_batch.candidate_inputs.genotype_flip_mask,
        firth_se=correction_plan.firth_se,
    )


def apply_firth_multi_variant_major_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply device-resident multi-trait Firth corrections with a fixed candidate capacity."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    prepared_batch = regenie2_binary_firth_batch.prepare_multi_firth_candidate_batch(
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
    firth_result = regenie2_binary_firth_batch.compute_firth_multi_variantwise_fixed_batches(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_logistic_coefficients=prepared_batch.candidate_inputs.null_logistic_coefficients,
        null_firth_offset_matrix=prepared_batch.candidate_inputs.null_firth_offset_matrix,
        phenotype_matrix=prepared_batch.candidate_inputs.phenotype_matrix,
        genotype_matrix_by_variant=prepared_batch.candidate_inputs.genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=prepared_batch.candidate_inputs.raw_genotype_matrix_by_variant,
        loco_offset_matrix=prepared_batch.candidate_inputs.loco_offset_matrix,
        initial_coefficients=prepared_batch.initial_coefficients,
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


def apply_firth_packed8_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply Firth corrections by decoding only packed8 candidate rows."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    prepared_batch = regenie2_binary_firth_batch.prepare_firth_candidate_batch_from_packed8(
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
    firth_result = regenie2_binary_firth_batch.compute_firth_variantwise_fixed_batches(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
        null_firth_offset=chromosome_state.null_firth_offset,
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_matrix_by_variant=prepared_batch.candidate_inputs.genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=prepared_batch.candidate_inputs.raw_genotype_matrix_by_variant,
        loco_offset=chromosome_state.loco_offset,
        initial_coefficients=prepared_batch.initial_coefficients,
        active_mask=prepared_batch.candidate_inputs.flat_active_mask,
        sparse_correction_mask=prepared_batch.candidate_inputs.sparse_correction_mask,
        fallback_count=fallback_count,
        firth_batch_size=firth_batch_size,
        null_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood,
        full_null_deviance=prepared_batch.full_null_deviance,
        kernel_config=kernel_config,
    )
    active_flat_positions = prepared_batch.batch_plan.active_flat_position_vector
    active_fallback_indices = prepared_batch.candidate_inputs.flat_fallback_indices[active_flat_positions]
    return regenie2_binary_correction.merge_firth_variant_result_into_chunk(
        result=result,
        firth_result=firth_result,
        active_flat_positions=active_flat_positions,
        active_fallback_indices=active_fallback_indices,
        genotype_flip_mask=prepared_batch.candidate_inputs.genotype_flip_mask,
        firth_se=correction_plan.firth_se,
    )


def apply_firth_multi_packed8_fixed_capacity_corrections(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    candidate_mask: jax.Array,
    fallback_count: jax.Array,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait Firth corrections by decoding only packed8 candidate rows."""
    firth_batch_size = min(kernel_config.firth_candidate.batch_size, candidate_capacity)
    prepared_batch = regenie2_binary_firth_batch.prepare_multi_firth_candidate_batch_from_packed8(
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
    firth_result = regenie2_binary_firth_batch.compute_firth_multi_variantwise_fixed_batches(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_logistic_coefficients=prepared_batch.candidate_inputs.null_logistic_coefficients,
        null_firth_offset_matrix=prepared_batch.candidate_inputs.null_firth_offset_matrix,
        phenotype_matrix=prepared_batch.candidate_inputs.phenotype_matrix,
        genotype_matrix_by_variant=prepared_batch.candidate_inputs.genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant=prepared_batch.candidate_inputs.raw_genotype_matrix_by_variant,
        loco_offset_matrix=prepared_batch.candidate_inputs.loco_offset_matrix,
        initial_coefficients=prepared_batch.initial_coefficients,
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


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
    ),
)
def apply_device_candidate_corrections_firth_variant_major_with_device_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply common-case Firth corrections with device-side zero and tiered dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return apply_firth_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=tiny_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
            )

        def apply_small_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return apply_firth_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=small_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
            )

        def apply_bounded_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return apply_firth_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=bounded_candidate_capacity,
                order_candidates=True,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
            )

        return jax.lax.cond(
            fallback_count <= tiny_candidate_capacity,
            apply_tiny_corrections,
            lambda _: jax.lax.cond(
                fallback_count <= small_candidate_capacity,
                apply_small_corrections,
                apply_bounded_corrections,
                operand=None,
            ),
            operand=None,
        )

    return jax.lax.cond(
        fallback_count == 0,
        return_empty_diagnostics,
        apply_candidate_corrections,
        operand=None,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "overflow_candidate_capacity",
    ),
)
def apply_device_candidate_corrections_firth_variant_major_with_overflow_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    overflow_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply rare overflow single-trait Firth corrections in a separate executable."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    return apply_firth_variant_major_fixed_capacity_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=diagnostic_result,
        correction_plan=correction_plan,
        candidate_mask=candidate_mask,
        fallback_count=fallback_count,
        candidate_capacity=overflow_candidate_capacity,
        order_candidates=True,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
        "score_dtype",
    ),
)
def apply_device_candidate_corrections_firth_packed8_with_device_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply common-case Firth corrections from packed8 rows with device-side dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return apply_firth_packed8_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=tiny_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                score_dtype=score_dtype,
            )

        def apply_small_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return apply_firth_packed8_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=small_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                score_dtype=score_dtype,
            )

        def apply_bounded_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return apply_firth_packed8_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=bounded_candidate_capacity,
                order_candidates=True,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                score_dtype=score_dtype,
            )

        return jax.lax.cond(
            fallback_count <= tiny_candidate_capacity,
            apply_tiny_corrections,
            lambda _: jax.lax.cond(
                fallback_count <= small_candidate_capacity,
                apply_small_corrections,
                apply_bounded_corrections,
                operand=None,
            ),
            operand=None,
        )

    return jax.lax.cond(
        fallback_count == 0,
        return_empty_diagnostics,
        apply_candidate_corrections,
        operand=None,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "overflow_candidate_capacity",
        "score_dtype",
    ),
)
def apply_device_candidate_corrections_firth_packed8_with_overflow_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    overflow_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply rare overflow packed8 single-trait Firth corrections in a separate executable."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    return apply_firth_packed8_fixed_capacity_corrections(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=diagnostic_result,
        correction_plan=correction_plan,
        candidate_mask=candidate_mask,
        fallback_count=fallback_count,
        candidate_capacity=overflow_candidate_capacity,
        order_candidates=True,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
    ),
)
def apply_device_candidate_corrections_multi_firth_variant_major_with_device_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply common multi-trait Firth corrections with device-side capacity dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=tiny_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
            )

        def apply_small_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=small_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
            )

        def apply_bounded_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=bounded_candidate_capacity,
                order_candidates=True,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
            )

        return jax.lax.cond(
            fallback_count <= tiny_candidate_capacity,
            apply_tiny_corrections,
            lambda _: jax.lax.cond(
                fallback_count <= small_candidate_capacity,
                apply_small_corrections,
                apply_bounded_corrections,
                operand=None,
            ),
            operand=None,
        )

    return jax.lax.cond(
        fallback_count == 0,
        return_empty_diagnostics,
        apply_candidate_corrections,
        operand=None,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "overflow_candidate_capacity",
    ),
)
def apply_device_candidate_corrections_multi_firth_variant_major_with_overflow_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    overflow_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply rare overflow multi-trait Firth corrections in a separate executable."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)
    return apply_firth_multi_variant_major_fixed_capacity_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=diagnostic_result,
        correction_plan=correction_plan,
        candidate_mask=candidate_mask,
        fallback_count=fallback_count,
        candidate_capacity=overflow_candidate_capacity,
        order_candidates=True,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
        "score_dtype",
    ),
)
def apply_device_candidate_corrections_multi_firth_packed8_with_device_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait Firth corrections from packed8 rows with device-side dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return apply_firth_multi_packed8_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=tiny_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                score_dtype=score_dtype,
            )

        def apply_small_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return apply_firth_multi_packed8_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=small_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                score_dtype=score_dtype,
            )

        def apply_bounded_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return apply_firth_multi_packed8_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                result=diagnostic_result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=bounded_candidate_capacity,
                order_candidates=True,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                score_dtype=score_dtype,
            )

        return jax.lax.cond(
            fallback_count <= tiny_candidate_capacity,
            apply_tiny_corrections,
            lambda _: jax.lax.cond(
                fallback_count <= small_candidate_capacity,
                apply_small_corrections,
                apply_bounded_corrections,
                operand=None,
            ),
            operand=None,
        )

    return jax.lax.cond(
        fallback_count == 0,
        return_empty_diagnostics,
        apply_candidate_corrections,
        operand=None,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "overflow_candidate_capacity",
        "score_dtype",
    ),
)
def apply_device_candidate_corrections_multi_firth_packed8_with_overflow_dispatch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    overflow_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply rare overflow multi-trait Firth corrections from packed8 rows."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)
    return apply_firth_multi_packed8_fixed_capacity_corrections(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=diagnostic_result,
        correction_plan=correction_plan,
        candidate_mask=candidate_mask,
        fallback_count=fallback_count,
        candidate_capacity=overflow_candidate_capacity,
        order_candidates=True,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


def apply_device_candidate_corrections_firth_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply Firth corrections with non-blocking device-side capacity dispatch."""
    if genotype_matrix_by_variant.shape[0] == 0:
        return regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    capacity_plan_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
    variant_count = genotype_matrix_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_firth_candidate_capacity_plan(
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    if stage_duration_recorder is not None:
        stage_duration_recorder("firth_candidate_dispatch_plan", capacity_plan_start_time)
    candidate_count = int(
        jax.device_get(jnp.sum(result.extra_code == types.BinaryExtraCode.FIRTH.value, dtype=jnp.int32))
    )
    if candidate_count > capacity_plan.bounded_candidate_capacity:
        return apply_device_candidate_corrections_firth_variant_major_with_overflow_dispatch(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            result=result,
            correction_plan=correction_plan,
            overflow_candidate_capacity=capacity_plan.overflow_candidate_capacity,
            sparse_candidate_mask=sparse_candidate_mask,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            kernel_config=kernel_config,
        )
    return apply_device_candidate_corrections_firth_variant_major_with_device_dispatch(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        tiny_candidate_capacity=capacity_plan.tiny_candidate_capacity,
        small_candidate_capacity=capacity_plan.small_candidate_capacity,
        bounded_candidate_capacity=capacity_plan.bounded_candidate_capacity,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        kernel_config=kernel_config,
    )


def apply_device_candidate_corrections_firth_packed8(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply Firth corrections to packed8 chunks without dense chunk decode."""
    if packed_probability_pairs_by_variant.shape[0] == 0:
        return regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    capacity_plan_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
    variant_count = packed_probability_pairs_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_firth_candidate_capacity_plan(
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    if stage_duration_recorder is not None:
        stage_duration_recorder("firth_candidate_dispatch_plan", capacity_plan_start_time)
    candidate_count = int(
        jax.device_get(jnp.sum(result.extra_code == types.BinaryExtraCode.FIRTH.value, dtype=jnp.int32))
    )
    if candidate_count > capacity_plan.bounded_candidate_capacity:
        return apply_device_candidate_corrections_firth_packed8_with_overflow_dispatch(
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
        )
    return apply_device_candidate_corrections_firth_packed8_with_device_dispatch(
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
    )


def apply_device_candidate_corrections_multi_firth_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait Firth corrections with non-blocking device-side capacity dispatch."""
    if genotype_matrix_by_variant.shape[0] == 0:
        return regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)
    capacity_plan_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
    trait_count = chromosome_state.phenotype_matrix.shape[0]
    variant_count = genotype_matrix_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_multi_firth_candidate_capacity_plan(
        trait_count=trait_count,
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    if stage_duration_recorder is not None:
        stage_duration_recorder("firth_candidate_dispatch_plan", capacity_plan_start_time)
    candidate_count = int(
        jax.device_get(jnp.sum(result.extra_code == types.BinaryExtraCode.FIRTH.value, dtype=jnp.int32))
    )
    if candidate_count > capacity_plan.bounded_candidate_capacity:
        return apply_device_candidate_corrections_multi_firth_variant_major_with_overflow_dispatch(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            result=result,
            correction_plan=correction_plan,
            overflow_candidate_capacity=capacity_plan.overflow_candidate_capacity,
            sparse_candidate_mask=sparse_candidate_mask,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            kernel_config=kernel_config,
        )
    return apply_device_candidate_corrections_multi_firth_variant_major_with_device_dispatch(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        tiny_candidate_capacity=capacity_plan.tiny_candidate_capacity,
        small_candidate_capacity=capacity_plan.small_candidate_capacity,
        bounded_candidate_capacity=capacity_plan.bounded_candidate_capacity,
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
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait Firth corrections to packed8 chunks without dense chunk decode."""
    if packed_probability_pairs_by_variant.shape[0] == 0:
        return regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)
    capacity_plan_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
    trait_count = chromosome_state.phenotype_matrix.shape[0]
    variant_count = packed_probability_pairs_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_multi_firth_candidate_capacity_plan(
        trait_count=trait_count,
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    if stage_duration_recorder is not None:
        stage_duration_recorder("firth_candidate_dispatch_plan", capacity_plan_start_time)
    candidate_count = int(
        jax.device_get(jnp.sum(result.extra_code == types.BinaryExtraCode.FIRTH.value, dtype=jnp.int32))
    )
    if candidate_count > capacity_plan.bounded_candidate_capacity:
        return apply_device_candidate_corrections_multi_firth_packed8_with_overflow_dispatch(
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
        )
    return apply_device_candidate_corrections_multi_firth_packed8_with_device_dispatch(
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
    )


def apply_device_candidate_corrections_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections for variant-major genotype chunks."""
    regenie2_binary_correction.validate_runtime_correction_plan(correction_plan)
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    return apply_device_candidate_corrections_firth_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        kernel_config=kernel_config,
        stage_duration_recorder=stage_duration_recorder,
    )


def apply_device_candidate_corrections_multi_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait binary candidate corrections for variant-major genotype chunks."""
    regenie2_binary_correction.validate_runtime_correction_plan(correction_plan)
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
        stage_duration_recorder=stage_duration_recorder,
    )


def apply_device_candidate_corrections_packed8(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections for packed8 chunks."""
    regenie2_binary_correction.validate_runtime_correction_plan(correction_plan)
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    return apply_device_candidate_corrections_firth_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
        kernel_config=kernel_config,
        stage_duration_recorder=stage_duration_recorder,
    )


def apply_device_candidate_corrections_multi_packed8(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait binary candidate corrections for packed8 chunks."""
    regenie2_binary_correction.validate_runtime_correction_plan(correction_plan)
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
        stage_duration_recorder=stage_duration_recorder,
    )
