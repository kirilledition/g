"""Experimental direct variant-major JAX binary association path."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

import g.compute.regenie2_binary as regenie2_binary
import g.compute.regenie2_binary_types as regenie2_types
import g.compute.regenie2_linear as regenie2_linear
import g.types as g_types


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
) -> regenie2_types.Regenie2BinaryChunkResult:
    """Compute the experimental variant-major score test for one binary chunk.

    This direct variant-major JAX path is not used by the production trusted
    BGEN pipeline until full-data Firth parity is established.

    """
    genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    weighted_genotype_matrix_by_variant = (
        genotype_matrix_by_variant_float32 * chromosome_state.square_root_weight[None, :]
    )
    projection_coordinates = (
        weighted_genotype_matrix_by_variant @ chromosome_state.weighted_genotype_projection_matrix.T
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "ij,ij->i",
        weighted_genotype_matrix_by_variant,
        weighted_genotype_matrix_by_variant,
    )
    projection_sum_squares = jnp.einsum("ij,ij->i", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = genotype_matrix_by_variant_float32 @ chromosome_state.score_residual
    positive_variance_mask = variance > regenie2_binary.MINIMUM_VARIANCE
    inverse_variance = jnp.where(positive_variance_mask, jnp.reciprocal(variance), 0.0)
    beta = jnp.where(positive_variance_mask, score * inverse_variance, jnp.nan)
    standard_error = jnp.where(positive_variance_mask, jnp.sqrt(inverse_variance), jnp.nan)
    chi_squared = jnp.where(positive_variance_mask, score * score * inverse_variance, 0.0)
    log10_p_value = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    extra_code = regenie2_binary.build_extra_code(log10_p_value, valid_mask, correction_plan)
    return regenie2_types.Regenie2BinaryChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        valid_mask=valid_mask,
        firth_iteration_count=jnp.zeros_like(extra_code, dtype=jnp.int32),
        firth_failure_code=jnp.zeros_like(extra_code, dtype=jnp.int32),
    )


compute_regenie2_binary_score_test_chunk_variant_major = typing.cast(
    "regenie2_binary.BinaryVariantMajorChunkComputeFunction",
    compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major,
)


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def apply_device_candidate_corrections_firth_variant_major(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_types.Regenie2BinaryChunkResult,
    correction_plan: g_types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_types.BinaryKernelConfig = regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_types.Regenie2BinaryChunkResult:
    """Apply device-resident Firth corrections to variant-major score-test candidates."""
    candidate_mask = result.extra_code == regenie2_binary.EXTRA_CODE_FIRTH
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def no_candidate_corrections() -> regenie2_types.Regenie2BinaryChunkResult:
        return result

    def apply_candidate_corrections() -> regenie2_types.Regenie2BinaryChunkResult:
        firth_batch_size = kernel_config.firth_batch_size
        kernel_candidate_capacity = kernel_config.firth_candidate_capacity
        genotype_matrix_by_variant_float32 = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
        variant_count = genotype_matrix_by_variant_float32.shape[0]

        def apply_candidate_corrections_with_capacity(
            candidate_capacity: int,
        ) -> regenie2_types.Regenie2BinaryChunkResult:
            batch_plan = regenie2_binary.build_device_firth_batch_plan(
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
            if sparse_candidate_mask is None:
                flat_sparse_candidate_mask = jnp.zeros_like(flat_active_mask)
            else:
                flat_sparse_candidate_mask = (
                    jnp.take(jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_), flat_fallback_indices, axis=0)
                    & flat_active_mask
                )
            heuristic_firth_mask = (
                regenie2_binary.compute_firth_pre_dispatch_mask_without_mask(
                    genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                    phenotype_vector=chromosome_state.phenotype_vector,
                )
                | flat_sparse_candidate_mask
            ) & flat_active_mask
            ordered_candidate_inputs = regenie2_binary.group_firth_candidate_batch_inputs(
                flat_fallback_indices=flat_fallback_indices,
                flat_active_mask=flat_active_mask,
                genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                heuristic_firth_mask=heuristic_firth_mask,
            )
            flat_fallback_indices = ordered_candidate_inputs.flat_fallback_indices
            flat_active_mask = ordered_candidate_inputs.flat_active_mask
            candidate_genotype_matrix_by_variant = ordered_candidate_inputs.genotype_matrix_by_variant
            heuristic_firth_mask = ordered_candidate_inputs.heuristic_firth_mask
            standard_initial_coefficients = jnp.broadcast_to(
                chromosome_state.null_logistic_coefficients[None, :],
                (
                    candidate_genotype_matrix_by_variant.shape[0],
                    chromosome_state.null_logistic_coefficients.shape[0],
                ),
            )
            standard_initial_coefficients = jnp.concatenate(
                [
                    standard_initial_coefficients,
                    jnp.take(result.beta, flat_fallback_indices, axis=0)[:, None],
                ],
                axis=1,
            )
            heuristic_initial_coefficients = regenie2_binary.initialize_full_model_coefficients_without_mask(
                covariate_matrix=chromosome_state.covariate_matrix,
                genotype_matrix_by_variant=candidate_genotype_matrix_by_variant,
                phenotype_vector=chromosome_state.phenotype_vector,
            )
            initial_coefficients = jnp.where(
                heuristic_firth_mask[:, None],
                heuristic_initial_coefficients,
                standard_initial_coefficients,
            )
            batch_count = batch_plan.fallback_index_matrix.shape[0]
            active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
            genotype_batches = candidate_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
            initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
            active_mask_batches = flat_active_mask.reshape((batch_count, firth_batch_size))
            empty_firth_variant_result = regenie2_binary.build_empty_firth_variant_result(firth_batch_size)

            def compute_firth_batch(
                carry: None,
                batch_index: jax.Array,
            ) -> tuple[None, regenie2_binary.FirthVariantResult]:
                del carry

                def run_active_batch(_: None) -> regenie2_binary.FirthVariantResult:
                    return regenie2_binary.compute_firth_variantwise(
                        covariate_matrix=chromosome_state.covariate_matrix,
                        phenotype_vector=chromosome_state.phenotype_vector,
                        genotype_matrix_by_variant=genotype_batches[batch_index],
                        loco_offset=chromosome_state.loco_offset,
                        initial_coefficients=initial_coefficient_batches[batch_index],
                        skip_firth_mask=~active_mask_batches[batch_index],
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
            firth_result = regenie2_binary.FirthVariantResult(
                beta=batched_firth_result.beta.reshape((-1,)),
                standard_error=batched_firth_result.standard_error.reshape((-1,)),
                chi_squared=batched_firth_result.chi_squared.reshape((-1,)),
                log10_p_value=batched_firth_result.log10_p_value.reshape((-1,)),
                penalized_log_likelihood=batched_firth_result.penalized_log_likelihood.reshape((-1,)),
                converged_mask=batched_firth_result.converged_mask.reshape((-1,)),
                valid_mask=batched_firth_result.valid_mask.reshape((-1,)),
                iteration_count=batched_firth_result.iteration_count.reshape((-1,)),
                failure_code=batched_firth_result.failure_code.reshape((-1,)),
            )
            active_flat_positions = batch_plan.active_flat_position_vector
            active_fallback_indices = flat_fallback_indices[active_flat_positions]
            current_beta = jnp.take(result.beta, active_fallback_indices, axis=0)
            current_standard_error = jnp.take(result.standard_error, active_fallback_indices, axis=0)
            current_chi_squared = jnp.take(result.chi_squared, active_fallback_indices, axis=0)
            current_log10_p_value = jnp.take(result.log10_p_value, active_fallback_indices, axis=0)
            active_valid_mask = firth_result.valid_mask[active_flat_positions]
            active_firth_beta = firth_result.beta[active_flat_positions]
            active_firth_chi_squared = firth_result.chi_squared[active_flat_positions]
            active_firth_standard_error = firth_result.standard_error[active_flat_positions]
            if correction_plan.firth_se:
                active_firth_standard_error = jnp.where(
                    active_firth_chi_squared > 0.0,
                    jnp.abs(active_firth_beta) / jnp.sqrt(active_firth_chi_squared),
                    active_firth_standard_error,
                )
            merged_beta = jnp.where(active_valid_mask, firth_result.beta[active_flat_positions], current_beta)
            merged_standard_error = jnp.where(
                active_valid_mask,
                active_firth_standard_error,
                current_standard_error,
            )
            merged_chi_squared = jnp.where(
                active_valid_mask,
                firth_result.chi_squared[active_flat_positions],
                current_chi_squared,
            )
            merged_log10_p_value = jnp.where(
                active_valid_mask,
                firth_result.log10_p_value[active_flat_positions],
                current_log10_p_value,
            )
            merged_extra_code = jnp.where(
                active_valid_mask,
                regenie2_binary.EXTRA_CODE_FIRTH,
                regenie2_binary.EXTRA_CODE_TEST_FAIL,
            ).astype(jnp.int32)
            return regenie2_types.Regenie2BinaryChunkResult(
                beta=result.beta.at[active_fallback_indices].set(merged_beta),
                standard_error=result.standard_error.at[active_fallback_indices].set(merged_standard_error),
                chi_squared=result.chi_squared.at[active_fallback_indices].set(merged_chi_squared),
                log10_p_value=result.log10_p_value.at[active_fallback_indices].set(merged_log10_p_value),
                extra_code=result.extra_code.at[active_fallback_indices].set(merged_extra_code),
                valid_mask=result.valid_mask.at[active_fallback_indices].set(active_valid_mask),
                firth_iteration_count=result.firth_iteration_count.at[active_fallback_indices].set(
                    firth_result.iteration_count[active_flat_positions]
                ),
                firth_failure_code=result.firth_failure_code.at[active_fallback_indices].set(
                    firth_result.failure_code[active_flat_positions]
                ),
            )

        bounded_candidate_capacity = min(kernel_candidate_capacity, variant_count)
        return jax.lax.cond(
            fallback_count <= bounded_candidate_capacity,
            lambda _: apply_candidate_corrections_with_capacity(bounded_candidate_capacity),
            lambda _: apply_candidate_corrections_with_capacity(variant_count),
            operand=None,
        )

    return jax.lax.cond(fallback_count > 0, apply_candidate_corrections, no_candidate_corrections)


def apply_device_candidate_corrections_variant_major(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_types.Regenie2BinaryChunkResult,
    correction_plan: g_types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_types.BinaryKernelConfig = regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_types.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections for variant-major genotype chunks."""
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    if correction_plan.method == g_types.BinaryFallbackMethod.FIRTH:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    if correction_plan.method == g_types.BinaryFallbackMethod.SPA:
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


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_types.BinaryKernelConfig = regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_types.Regenie2BinaryChunkResult:
    """Compute experimental binary association from a variant-major chunk.

    This direct variant-major JAX path is not used by the production trusted
    BGEN pipeline until full-data Firth parity is established.

    """
    score_test_result = compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
        chromosome_state,
        genotype_matrix_by_variant,
        correction_plan,
    )
    return apply_device_candidate_corrections_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )
