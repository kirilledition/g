"""Variant-major binary candidate correction kernels for REGENIE step 2."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from g import types
from g.compute.common import genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import types as regenie2_binary_types
from g.compute.regenie2_binary.firth import batch as regenie2_binary_firth_batch
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "candidate_capacity"))
def apply_device_candidate_corrections_firth_variant_major_with_capacity(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    candidate_capacity: int = regenie2_binary_candidate_planning.DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Apply device-resident Firth corrections with a fixed candidate capacity."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def apply_candidate_corrections() -> regenie2_binary_types.Regenie2BinaryChunkResult:
        firth_batch_size = kernel_config.firth_batch_size
        genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)

        def apply_candidate_corrections_with_capacity(
            candidate_capacity: int,
        ) -> regenie2_binary_types.Regenie2BinaryChunkResult:
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
            if kernel_config.use_block_firth_math:
                firth_raw_candidate_genotype_matrix_by_variant = raw_candidate_genotype_matrix_by_variant
                flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
                candidate_genotype_matrix_by_variant = firth_raw_candidate_genotype_matrix_by_variant
            else:
                firth_raw_candidate_genotype_matrix_by_variant = genotype_flip_result.genotype_matrix_by_variant
                flat_genotype_flip_mask = genotype_flip_result.flip_mask
                candidate_genotype_matrix_by_variant = (
                    regenie2_binary_firth_batch.residualize_and_scale_genotypes_for_approximate_firth(
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
                regenie2_binary_firth_batch.compute_firth_pre_dispatch_mask_without_mask(
                    genotype_matrix_by_variant=firth_raw_candidate_genotype_matrix_by_variant,
                    phenotype_vector=chromosome_state.phenotype_vector,
                )
                | flat_sparse_candidate_mask
            ) & flat_active_mask
            ordered_candidate_inputs = regenie2_binary_candidate_planning.group_firth_candidate_batch_inputs(
                flat_fallback_indices=flat_fallback_indices,
                flat_active_mask=flat_active_mask,
                genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                heuristic_firth_mask=heuristic_firth_mask,
            )
            flat_fallback_indices = ordered_candidate_inputs.flat_fallback_indices
            flat_active_mask = ordered_candidate_inputs.flat_active_mask
            candidate_genotype_matrix_by_variant = ordered_candidate_inputs.genotype_matrix_by_variant
            heuristic_firth_mask = ordered_candidate_inputs.heuristic_firth_mask
            raw_candidate_genotype_matrix_by_variant = jnp.take(
                genotype_matrix_by_variant_float32,
                flat_fallback_indices,
                axis=0,
            )
            genotype_flip_result = genotype.build_regenie_flipped_genotypes(raw_candidate_genotype_matrix_by_variant)
            if kernel_config.use_block_firth_math:
                firth_raw_candidate_genotype_matrix_by_variant = raw_candidate_genotype_matrix_by_variant
                flat_genotype_flip_mask = jnp.zeros_like(flat_active_mask)
            else:
                firth_raw_candidate_genotype_matrix_by_variant = genotype_flip_result.genotype_matrix_by_variant
                flat_genotype_flip_mask = genotype_flip_result.flip_mask
            flat_sparse_candidate_mask = (
                jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
                & flat_active_mask
                if sparse_candidate_mask is not None
                else jnp.zeros_like(flat_active_mask)
            )
            standard_initial_coefficients = jnp.broadcast_to(
                chromosome_state.null_logistic_coefficients[None, :],
                (
                    candidate_genotype_matrix_by_variant.shape[0],
                    chromosome_state.null_logistic_coefficients.shape[0],
                ),
            )
            standard_initial_beta = (
                jnp.take(result.beta, flat_fallback_indices, axis=0)
                if kernel_config.use_block_firth_math
                else jnp.zeros_like(jnp.take(result.beta, flat_fallback_indices, axis=0))
            )
            standard_initial_coefficients = jnp.concatenate(
                [
                    standard_initial_coefficients,
                    standard_initial_beta[:, None],
                ],
                axis=1,
            )
            if kernel_config.use_block_firth_math:
                heuristic_initial_coefficients = (
                    regenie2_binary_firth_batch.initialize_full_model_coefficients_without_mask(
                        covariate_matrix=chromosome_state.covariate_matrix,
                        genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                        phenotype_vector=chromosome_state.phenotype_vector,
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
            empty_firth_variant_result = regenie2_binary_firth_batch.build_empty_firth_variant_result(firth_batch_size)

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
            firth_result = regenie2_binary_firth_types.FirthVariantResult(
                beta=batched_firth_result.beta.reshape((-1,)),
                standard_error=batched_firth_result.standard_error.reshape((-1,)),
                chi_squared=batched_firth_result.chi_squared.reshape((-1,)),
                log10_p_value=batched_firth_result.log10_p_value.reshape((-1,)),
                penalized_log_likelihood=batched_firth_result.penalized_log_likelihood.reshape((-1,)),
                converged_mask=batched_firth_result.converged_mask.reshape((-1,)),
                valid_mask=batched_firth_result.valid_mask.reshape((-1,)),
                iteration_count=batched_firth_result.iteration_count.reshape((-1,)),
                failure_code=batched_firth_result.failure_code.reshape((-1,)),
                convergence_reason_code=batched_firth_result.convergence_reason_code.reshape((-1,)),
                correction_code=batched_firth_result.correction_code.reshape((-1,)),
                sparse_correction_mask=batched_firth_result.sparse_correction_mask.reshape((-1,)),
                pseudo_firth_iteration_count=batched_firth_result.pseudo_firth_iteration_count.reshape((-1,)),
                nr_zero_start_iteration_count=batched_firth_result.nr_zero_start_iteration_count.reshape((-1,)),
                nr_warm_start_iteration_count=batched_firth_result.nr_warm_start_iteration_count.reshape((-1,)),
            )
            active_flat_positions = batch_plan.active_flat_position_vector
            active_fallback_indices = flat_fallback_indices[active_flat_positions]
            active_valid_mask = firth_result.valid_mask[active_flat_positions]
            active_firth_beta = jnp.where(
                flat_genotype_flip_mask[active_flat_positions],
                -firth_result.beta[active_flat_positions],
                firth_result.beta[active_flat_positions],
            )
            active_firth_chi_squared = firth_result.chi_squared[active_flat_positions]
            active_firth_standard_error = firth_result.standard_error[active_flat_positions]
            invalid_firth_statistic = jnp.full_like(active_firth_beta, jnp.nan)
            if correction_plan.firth_se:
                active_firth_standard_error = jnp.where(
                    active_firth_chi_squared > 0.0,
                    jnp.abs(active_firth_beta) / jnp.sqrt(active_firth_chi_squared),
                    active_firth_standard_error,
                )
            merged_beta = jnp.where(active_valid_mask, active_firth_beta, invalid_firth_statistic)
            merged_standard_error = jnp.where(
                active_valid_mask,
                active_firth_standard_error,
                invalid_firth_statistic,
            )
            merged_chi_squared = jnp.where(
                active_valid_mask,
                firth_result.chi_squared[active_flat_positions],
                invalid_firth_statistic,
            )
            merged_log10_p_value = jnp.where(
                active_valid_mask,
                firth_result.log10_p_value[active_flat_positions],
                invalid_firth_statistic,
            )
            merged_extra_code = jnp.where(
                active_valid_mask,
                types.BinaryExtraCode.FIRTH.value,
                types.BinaryExtraCode.TEST_FAIL.value,
            ).astype(jnp.int32)
            return regenie2_binary_types.Regenie2BinaryChunkResult(
                beta=result.beta.at[active_fallback_indices].set(jnp.asarray(merged_beta, dtype=result.beta.dtype)),
                standard_error=result.standard_error.at[active_fallback_indices].set(
                    jnp.asarray(merged_standard_error, dtype=result.standard_error.dtype)
                ),
                chi_squared=result.chi_squared.at[active_fallback_indices].set(
                    jnp.asarray(merged_chi_squared, dtype=result.chi_squared.dtype)
                ),
                log10_p_value=result.log10_p_value.at[active_fallback_indices].set(
                    jnp.asarray(merged_log10_p_value, dtype=result.log10_p_value.dtype)
                ),
                extra_code=result.extra_code.at[active_fallback_indices].set(merged_extra_code),
                valid_mask=result.valid_mask.at[active_fallback_indices].set(active_valid_mask),
                firth_iteration_count=result.firth_iteration_count.at[active_fallback_indices].set(
                    firth_result.iteration_count[active_flat_positions]
                ),
                firth_failure_code=result.firth_failure_code.at[active_fallback_indices].set(
                    firth_result.failure_code[active_flat_positions]
                ),
                firth_convergence_reason_code=result.firth_convergence_reason_code.at[active_fallback_indices].set(
                    firth_result.convergence_reason_code[active_flat_positions]
                ),
                firth_correction_code=result.firth_correction_code.at[active_fallback_indices].set(
                    firth_result.correction_code[active_flat_positions]
                ),
                firth_sparse_correction_mask=result.firth_sparse_correction_mask.at[active_fallback_indices].set(
                    firth_result.sparse_correction_mask[active_flat_positions]
                ),
                pseudo_firth_iteration_count=result.pseudo_firth_iteration_count.at[active_fallback_indices].set(
                    firth_result.pseudo_firth_iteration_count[active_flat_positions]
                ),
                nr_zero_start_iteration_count=result.nr_zero_start_iteration_count.at[active_fallback_indices].set(
                    firth_result.nr_zero_start_iteration_count[active_flat_positions]
                ),
                nr_warm_start_iteration_count=result.nr_warm_start_iteration_count.at[active_fallback_indices].set(
                    firth_result.nr_warm_start_iteration_count[active_flat_positions]
                ),
            )

        return apply_candidate_corrections_with_capacity(candidate_capacity)

    return apply_candidate_corrections()


def apply_device_candidate_corrections_firth_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Select bounded or overflow Firth capacity on the host before correction."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = regenie2_binary_candidate_planning.count_firth_candidates_on_host(candidate_mask)
    if fallback_count == 0:
        return result
    variant_count = genotype_matrix_by_variant.shape[0]
    capacity_plan = regenie2_binary_candidate_planning.build_firth_candidate_capacity_plan(
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate_capacity,
    )
    candidate_capacity = regenie2_binary_candidate_planning.select_firth_candidate_capacity(
        fallback_count=fallback_count,
        capacity_plan=capacity_plan,
    )
    return apply_device_candidate_corrections_firth_variant_major_with_capacity(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        candidate_capacity=candidate_capacity,
        kernel_config=kernel_config,
    )


def apply_device_candidate_corrections_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
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
