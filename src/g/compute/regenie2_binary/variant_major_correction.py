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


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "candidate_capacity"))
def apply_device_candidate_corrections_firth_variant_major_with_capacity(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    candidate_capacity: int = regenie2_binary_config.PACKAGED_FIRTH_CANDIDATE_CAPACITY,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply device-resident Firth corrections with a fixed candidate capacity."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def apply_candidate_corrections() -> regenie2_binary_result.Regenie2BinaryChunkResult:
        firth_batch_size = kernel_config.firth_candidate.batch_size

        def apply_candidate_corrections_with_capacity(
            candidate_capacity: int,
        ) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            prepared_batch = regenie2_binary_firth_batch.prepare_firth_candidate_batch(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                candidate_mask=candidate_mask,
                score_beta=result.beta,
                sparse_candidate_mask=sparse_candidate_mask,
                candidate_capacity=candidate_capacity,
                firth_batch_size=firth_batch_size,
                kernel_config=kernel_config,
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

        return apply_candidate_corrections_with_capacity(candidate_capacity)

    return apply_candidate_corrections()


def apply_device_candidate_corrections_firth_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Select bounded or overflow Firth capacity on the host before correction."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    candidate_count_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
    fallback_count = regenie2_binary_candidate_planning.count_firth_candidates_on_host(candidate_mask)
    if stage_duration_recorder is not None:
        stage_duration_recorder("firth_candidate_count_host_sync", candidate_count_start_time)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    if fallback_count == 0:
        return diagnostic_result
    variant_count = genotype_matrix_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_firth_candidate_capacity_plan(
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    candidate_capacity = regenie2_binary_candidate_planning.select_firth_candidate_capacity(
        fallback_count=fallback_count,
        capacity_plan=capacity_plan,
    )
    return apply_device_candidate_corrections_firth_variant_major_with_capacity(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=diagnostic_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        candidate_capacity=candidate_capacity,
        kernel_config=kernel_config,
    )


def apply_device_candidate_corrections_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections for variant-major genotype chunks."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    if correction_plan.method == types.BinaryFallbackMethod.FIRTH:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    if correction_plan.method == types.BinaryFallbackMethod.SPA:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    return apply_device_candidate_corrections_firth_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        stage_duration_recorder=stage_duration_recorder,
    )
