"""Variant-major binary candidate correction kernels for REGENIE step 2."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import batch as regenie2_binary_firth_batch
from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "candidate_capacity"))
def apply_device_candidate_corrections_firth_variant_major_with_capacity(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    candidate_capacity: int = regenie2_binary_config.DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply device-resident Firth corrections with a fixed candidate capacity."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def apply_candidate_corrections() -> regenie2_binary_result.Regenie2BinaryChunkResult:
        firth_batch_size = kernel_config.firth_candidate.batch_size
        genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)

        def apply_candidate_corrections_with_capacity(
            candidate_capacity: int,
        ) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            batch_plan = regenie2_binary_candidate_planning.build_device_firth_batch_plan(
                candidate_mask,
                candidate_capacity,
                firth_batch_size,
            )
            flat_fallback_indices = batch_plan.fallback_index_matrix.reshape((-1,))
            flat_active_mask = batch_plan.fallback_active_mask_matrix.reshape((-1,))
            candidate_genotype_matrix_by_variant = jnp.take(
                genotype_matrix_by_variant_float32,
                flat_fallback_indices,
                axis=0,
            )
            raw_candidate_genotype_matrix_by_variant = candidate_genotype_matrix_by_variant
            genotype_flip_result = genotype.build_regenie_flipped_genotypes(raw_candidate_genotype_matrix_by_variant)
            if kernel_config.approximate_firth.use_block_math:
                firth_raw_candidate_genotype_matrix_by_variant = raw_candidate_genotype_matrix_by_variant
                flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
                candidate_genotype_matrix_by_variant = firth_raw_candidate_genotype_matrix_by_variant
            else:
                firth_raw_candidate_genotype_matrix_by_variant = genotype_flip_result.genotype_matrix_by_variant
                flat_genotype_flip_mask = genotype_flip_result.flip_mask
                candidate_genotype_matrix_by_variant = (
                    regenie2_binary_firth_scalar_approx.residualize_and_scale_genotypes_for_approximate_firth(
                        chromosome_state,
                        firth_raw_candidate_genotype_matrix_by_variant,
                    )
                )
            if sparse_candidate_mask is None:
                flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
            else:
                flat_sparse_candidate_mask = (
                    jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
                    & flat_active_mask
                )
            heuristic_firth_mask = (
                regenie2_binary_candidate_planning.compute_firth_pre_dispatch_mask_without_mask(
                    genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
                    phenotype_vector=chromosome_state.phenotype_vector,
                )
                | flat_sparse_candidate_mask
            ) & flat_active_mask
            ordered_candidate_inputs = regenie2_binary_candidate_planning.group_firth_candidate_batch_inputs(
                flat_fallback_indices=flat_fallback_indices,
                flat_active_mask=flat_active_mask,
                genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                raw_genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
                genotype_flip_mask=flat_genotype_flip_mask,
                sparse_correction_mask=flat_sparse_candidate_mask,
                heuristic_firth_mask=heuristic_firth_mask,
            )
            flat_fallback_indices = ordered_candidate_inputs.flat_fallback_indices
            flat_active_mask = ordered_candidate_inputs.flat_active_mask
            candidate_genotype_matrix_by_variant = ordered_candidate_inputs.genotype_matrix_by_variant
            firth_raw_candidate_genotype_matrix_by_variant = ordered_candidate_inputs.raw_genotype_matrix_by_variant
            flat_genotype_flip_mask = ordered_candidate_inputs.genotype_flip_mask
            flat_sparse_candidate_mask = ordered_candidate_inputs.sparse_correction_mask
            heuristic_firth_mask = ordered_candidate_inputs.heuristic_firth_mask
            standard_initial_coefficients = jnp.broadcast_to(
                chromosome_state.null_logistic_coefficients[None, :],
                (
                    candidate_genotype_matrix_by_variant.shape[0],
                    chromosome_state.null_logistic_coefficients.shape[0],
                ),
            )
            standard_initial_beta = (
                jnp.take(result.beta, flat_fallback_indices, axis=0)
                if kernel_config.approximate_firth.use_block_math
                else jnp.zeros_like(jnp.take(result.beta, flat_fallback_indices, axis=0))
            )
            standard_initial_coefficients = jnp.concatenate(
                [
                    standard_initial_coefficients,
                    standard_initial_beta[:, None],
                ],
                axis=1,
            )
            if kernel_config.approximate_firth.use_block_math:
                heuristic_initial_coefficients = (
                    regenie2_binary_firth_full_model.initialize_full_model_coefficients_without_mask(
                        covariate_matrix=chromosome_state.covariate_matrix,
                        genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                        phenotype_vector=chromosome_state.phenotype_vector,
                        kernel_config=kernel_config,
                    )
                )
                initial_coefficients = jnp.where(
                    heuristic_firth_mask[:, None],
                    heuristic_initial_coefficients,
                    standard_initial_coefficients,
                )
            else:
                initial_coefficients = standard_initial_coefficients
            batch_count = batch_plan.fallback_index_matrix.shape[0]
            active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
            genotype_batches = candidate_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
            raw_genotype_batches = firth_raw_candidate_genotype_matrix_by_variant.reshape(
                (batch_count, firth_batch_size, -1)
            )
            initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
            active_mask_batches = flat_active_mask.reshape((batch_count, firth_batch_size))
            sparse_correction_mask_batches = flat_sparse_candidate_mask.reshape((batch_count, firth_batch_size))
            empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

            def compute_firth_batch(
                carry: None,
                batch_index: jax.Array,
            ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
                del carry

                def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
                    return regenie2_binary_firth_batch.compute_firth_variantwise(
                        covariate_matrix=chromosome_state.covariate_matrix,
                        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
                        null_firth_offset=chromosome_state.null_firth_offset,
                        phenotype_vector=chromosome_state.phenotype_vector,
                        genotype_matrix_by_variant=genotype_batches[batch_index],
                        raw_genotype_matrix_by_variant=raw_genotype_batches[batch_index],
                        loco_offset=chromosome_state.loco_offset,
                        initial_coefficients=initial_coefficient_batches[batch_index],
                        skip_firth_mask=~active_mask_batches[batch_index],
                        sparse_correction_mask=sparse_correction_mask_batches[batch_index],
                        null_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood,
                        kernel_config=kernel_config,
                    )

                batch_result = jax.lax.cond(
                    batch_index < active_batch_count,
                    run_active_batch,
                    lambda _: empty_firth_variant_result,
                    operand=None,
                )
                return None, batch_result

            _, batched_firth_result = jax.lax.scan(
                compute_firth_batch,
                None,
                jnp.arange(batch_count, dtype=jnp.int32),
            )
            firth_result = regenie2_binary_firth_types.flatten_batched_firth_variant_result(batched_firth_result)
            active_flat_positions = batch_plan.active_flat_position_vector
            active_fallback_indices = flat_fallback_indices[active_flat_positions]
            return regenie2_binary_correction.merge_firth_variant_result_into_chunk(
                result=result,
                firth_result=firth_result,
                active_flat_positions=active_flat_positions,
                active_fallback_indices=active_fallback_indices,
                genotype_flip_mask=flat_genotype_flip_mask,
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
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Select bounded or overflow Firth capacity on the host before correction."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = regenie2_binary_candidate_planning.count_firth_candidates_on_host(candidate_mask)
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
    )
