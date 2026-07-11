"""Fixed-batch Firth computation helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config


SPARSE_FIRTH_CARRIER_CAPACITY = 64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthLaneStreamPlan:
    """Fixed-shape lane stream selected from a candidate batch."""

    lane_indices: jax.Array
    active_mask: jax.Array
    active_count: jax.Array


def compute_scalar_firth_multi_variantwise(
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    carrier_sample_mask: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_failed_mask: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute scalar Firth fits for lane-specific multi-trait candidates."""
    solver_parameters = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        kernel_config
    )

    def fit_variant(
        phenotype_vector: jax.Array,
        null_firth_offset: jax.Array,
        genotype_vector: jax.Array,
        lane_carrier_sample_mask: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
        null_failed: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth_with_solver_parameters(
            phenotype_vector=jnp.asarray(phenotype_vector, dtype=jnp.float64),
            genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
            offset_vector=jnp.asarray(null_firth_offset, dtype=jnp.float64),
            carrier_sample_mask=lane_carrier_sample_mask,
            sparse_correction=sparse_correction,
            warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
            skip_firth=skip_firth,
            null_failed=null_failed,
            solver_parameters=solver_parameters,
        )

    return jax.vmap(fit_variant)(
        phenotype_matrix,
        null_firth_offset_matrix,
        genotype_matrix_by_variant,
        carrier_sample_mask,
        skip_firth_mask,
        sparse_correction_mask,
        null_failed_mask,
    )


def compute_block_firth_multi_variantwise(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute block full-model Firth fits for lane-specific candidates."""

    def fit_variant(
        phenotype_vector: jax.Array,
        genotype_vector: jax.Array,
        loco_offset: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        lane_null_penalized_log_likelihood: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        return regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            loco_offset=loco_offset,
            initial_coefficients=variant_initial_coefficients,
            skip_firth=skip_firth,
            null_penalized_log_likelihood=lane_null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    return jax.vmap(fit_variant)(
        phenotype_matrix,
        genotype_matrix_by_variant,
        loco_offset_matrix,
        initial_coefficients,
        skip_firth_mask,
        null_penalized_log_likelihood,
    )


def build_compact_sparse_carrier_indices(*, carrier_sample_mask: jax.Array) -> jax.Array:
    """Build fixed-capacity carrier sample indices for sparse Firth lanes."""
    sample_count = carrier_sample_mask.shape[1]
    if sample_count > regenie2_binary_candidate_planning.JAX_INT32_INDEX_MAXIMUM:
        message = "Sparse Firth sample count exceeds the JAX int32 index domain."
        raise ValueError(message)
    return jax.vmap(
        lambda carrier_mask: regenie2_binary_candidate_planning.build_compact_int32_indices(
            carrier_mask,
            SPARSE_FIRTH_CARRIER_CAPACITY,
        )
    )(carrier_sample_mask)


def build_firth_lane_stream_plan(active_lane_mask: jax.Array) -> FirthLaneStreamPlan:
    """Pack active candidate-lane positions into a fixed-capacity stream."""
    stream_capacity = active_lane_mask.shape[0]
    if stream_capacity > regenie2_binary_candidate_planning.JAX_INT32_INDEX_MAXIMUM:
        message = "Firth lane stream exceeds the JAX int32 index domain."
        raise ValueError(message)
    lane_indices = regenie2_binary_candidate_planning.build_compact_int32_indices(
        active_lane_mask,
        stream_capacity,
    )
    active_count = jnp.sum(active_lane_mask, dtype=jnp.int32)
    active_mask = jnp.arange(stream_capacity, dtype=jnp.int32) < active_count
    return FirthLaneStreamPlan(
        lane_indices=lane_indices,
        active_mask=active_mask,
        active_count=active_count,
    )


def scatter_firth_variant_result_by_lane_stream(
    *,
    base_result: regenie2_binary_firth_types.FirthVariantResult,
    lane_indices: jax.Array,
    active_mask: jax.Array,
    stream_result: regenie2_binary_firth_types.FirthVariantResult,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Scatter stream-ordered Firth results back into candidate-lane order."""
    result_capacity = base_result.beta.shape[0]
    inactive_index = jnp.asarray(result_capacity, dtype=jnp.int32)
    scatter_indices = jnp.where(active_mask, lane_indices, inactive_index)
    return regenie2_binary_firth_types.FirthVariantResult(
        beta=base_result.beta.at[scatter_indices].set(stream_result.beta, mode="drop"),
        standard_error=base_result.standard_error.at[scatter_indices].set(
            stream_result.standard_error,
            mode="drop",
        ),
        chi_squared=base_result.chi_squared.at[scatter_indices].set(stream_result.chi_squared, mode="drop"),
        log10_p_value=base_result.log10_p_value.at[scatter_indices].set(
            stream_result.log10_p_value,
            mode="drop",
        ),
        valid_mask=base_result.valid_mask.at[scatter_indices].set(stream_result.valid_mask, mode="drop"),
    )


def compute_compact_sparse_firth_variantwise_fixed_batches_with_solver_parameters(
    *,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    offset_matrix: jax.Array,
    active_carrier_slot_mask: jax.Array,
    full_null_deviance: jax.Array,
    active_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_failed_mask: jax.Array,
    solver_parameters: regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute compact sparse fixed batches with explicit solver policy."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    phenotype_batches = phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    offset_batches = offset_matrix.reshape((batch_count, firth_batch_size, -1))
    active_carrier_slot_mask_batches = active_carrier_slot_mask.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    full_null_deviance_batches = full_null_deviance.reshape((batch_count, firth_batch_size))
    null_failed_mask_batches = null_failed_mask.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def fit_compact_lane(
        phenotype_vector: jax.Array,
        genotype_vector: jax.Array,
        offset_vector: jax.Array,
        carrier_slot_mask: jax.Array,
        lane_full_null_deviance: jax.Array,
        skip_firth: jax.Array,
        null_failed: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        return regenie2_binary_firth_scalar_approx.fit_compact_carrier_regenie_approximate_firth_with_solver_parameters(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_carrier_slot_mask=carrier_slot_mask,
            full_null_deviance=lane_full_null_deviance,
            warm_start_beta=jnp.asarray(0.0, dtype=offset_vector.dtype),
            skip_firth=skip_firth,
            null_failed=null_failed,
            solver_parameters=solver_parameters,
        )

    fit_compact_batch = jax.vmap(fit_compact_lane)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def compute_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return fit_compact_batch(
                phenotype_batches[batch_index],
                genotype_batches[batch_index],
                offset_batches[batch_index],
                active_carrier_slot_mask_batches[batch_index],
                full_null_deviance_batches[batch_index],
                ~active_mask_batches[batch_index],
                null_failed_mask_batches[batch_index],
            )

        batch_result = jax.lax.cond(
            batch_index < active_batch_count,
            compute_active_batch,
            lambda _: empty_firth_variant_result,
            operand=None,
        )
        return None, batch_result

    _, batched_firth_result = jax.lax.scan(
        compute_firth_batch,
        None,
        jnp.arange(batch_count, dtype=jnp.int32),
    )
    return regenie2_binary_firth_types.flatten_batched_firth_variant_result(batched_firth_result)


def compute_scalar_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
    *,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    carrier_sample_mask: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_failed_mask: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute scalar Firth fits for flattened candidate lanes using fixed-size batches."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    null_firth_offset_batches = null_firth_offset_matrix.reshape((batch_count, firth_batch_size, -1))
    phenotype_batches = phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    carrier_sample_mask_batches = carrier_sample_mask.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    sparse_correction_mask_batches = sparse_correction_mask.reshape((batch_count, firth_batch_size))
    null_failed_mask_batches = null_failed_mask.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return compute_scalar_firth_multi_variantwise(
                null_firth_offset_matrix=null_firth_offset_batches[batch_index],
                phenotype_matrix=phenotype_batches[batch_index],
                genotype_matrix_by_variant=genotype_batches[batch_index],
                carrier_sample_mask=carrier_sample_mask_batches[batch_index],
                skip_firth_mask=~active_mask_batches[batch_index],
                sparse_correction_mask=sparse_correction_mask_batches[batch_index],
                null_failed_mask=null_failed_mask_batches[batch_index],
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
    return regenie2_binary_firth_types.flatten_batched_firth_variant_result(batched_firth_result)


def compute_block_firth_multi_variantwise_fixed_batches(
    *,
    covariate_matrix: jax.Array,
    candidate_inputs: regenie2_binary_candidate_planning.BlockFirthCandidateBatchInputs,
    fallback_count: jax.Array,
    firth_batch_size: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute block full-model Firth fits using fixed-size candidate batches."""
    active_mask = candidate_inputs.lanes.flat_active_mask
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    phenotype_batches = candidate_inputs.lanes.phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = candidate_inputs.genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    loco_offset_batches = candidate_inputs.loco_offset_matrix.reshape((batch_count, firth_batch_size, -1))
    initial_coefficient_batches = candidate_inputs.initial_coefficients.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    null_penalized_log_likelihood_batches = candidate_inputs.null_firth_penalized_log_likelihood.reshape(
        (batch_count, firth_batch_size)
    )
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return compute_block_firth_multi_variantwise(
                covariate_matrix=covariate_matrix,
                phenotype_matrix=phenotype_batches[batch_index],
                genotype_matrix_by_variant=genotype_batches[batch_index],
                loco_offset_matrix=loco_offset_batches[batch_index],
                initial_coefficients=initial_coefficient_batches[batch_index],
                skip_firth_mask=~active_mask_batches[batch_index],
                null_penalized_log_likelihood=null_penalized_log_likelihood_batches[batch_index],
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
    return regenie2_binary_firth_types.flatten_batched_firth_variant_result(batched_firth_result)


def compute_scalar_firth_multi_variantwise_fixed_batches(
    *,
    candidate_inputs: regenie2_binary_candidate_planning.ScalarFirthCandidateBatchInputs,
    fallback_count: jax.Array,
    firth_batch_size: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute scalar Firth fits with compact sparse lanes when eligible."""
    active_mask = candidate_inputs.lanes.flat_active_mask

    def compute_without_sparse_compaction() -> regenie2_binary_firth_types.FirthVariantResult:
        return compute_scalar_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
            null_firth_offset_matrix=candidate_inputs.null_firth_offset_matrix,
            phenotype_matrix=candidate_inputs.lanes.phenotype_matrix,
            genotype_matrix_by_variant=candidate_inputs.genotype_matrix_by_variant,
            carrier_sample_mask=candidate_inputs.carrier_sample_mask,
            active_mask=active_mask,
            sparse_correction_mask=candidate_inputs.sparse_correction_mask,
            fallback_count=fallback_count,
            firth_batch_size=firth_batch_size,
            null_failed_mask=candidate_inputs.null_failed_mask,
            kernel_config=kernel_config,
        )

    def compute_with_sparse_compaction(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
        carrier_count = jnp.sum(candidate_inputs.carrier_sample_mask, axis=1, dtype=jnp.int32)
        compact_sparse_lane_mask = (
            active_mask & candidate_inputs.sparse_correction_mask & (carrier_count <= SPARSE_FIRTH_CARRIER_CAPACITY)
        )

        def compute_split_path(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            dense_lane_mask = active_mask & (~compact_sparse_lane_mask)
            dense_stream_plan = build_firth_lane_stream_plan(dense_lane_mask)
            compact_stream_plan = build_firth_lane_stream_plan(compact_sparse_lane_mask)
            dense_fallback_result = regenie2_binary_firth_types.build_empty_firth_variant_result(
                dense_stream_plan.active_mask.shape[0]
            )
            empty_result = regenie2_binary_firth_types.build_empty_firth_variant_result(active_mask.shape[0])
            dense_stream_has_lanes = dense_stream_plan.active_count > 0
            compact_stream_has_lanes = compact_stream_plan.active_count > 0

            def compute_dense_stream(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
                return compute_scalar_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
                    null_firth_offset_matrix=jnp.take(
                        candidate_inputs.null_firth_offset_matrix,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    phenotype_matrix=jnp.take(
                        candidate_inputs.lanes.phenotype_matrix,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    genotype_matrix_by_variant=jnp.take(
                        candidate_inputs.genotype_matrix_by_variant,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    carrier_sample_mask=jnp.take(
                        candidate_inputs.carrier_sample_mask,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    active_mask=dense_stream_plan.active_mask,
                    sparse_correction_mask=jnp.take(
                        candidate_inputs.sparse_correction_mask,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    fallback_count=dense_stream_plan.active_count,
                    firth_batch_size=firth_batch_size,
                    null_failed_mask=jnp.take(
                        candidate_inputs.null_failed_mask,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    kernel_config=kernel_config,
                )

            dense_result = jax.lax.cond(
                dense_stream_has_lanes,
                compute_dense_stream,
                lambda _: dense_fallback_result,
                operand=None,
            )

            def compute_compact_stream(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
                compact_carrier_sample_mask = jnp.take(
                    candidate_inputs.carrier_sample_mask,
                    compact_stream_plan.lane_indices,
                    axis=0,
                )
                compact_carrier_indices = build_compact_sparse_carrier_indices(
                    carrier_sample_mask=compact_carrier_sample_mask,
                )
                compact_carrier_count = jnp.take(carrier_count, compact_stream_plan.lane_indices, axis=0)
                compact_carrier_slot_mask = (
                    jnp.arange(SPARSE_FIRTH_CARRIER_CAPACITY, dtype=jnp.int32)[None, :] < compact_carrier_count[:, None]
                ) & compact_stream_plan.active_mask[:, None]
                compact_lane_indices = compact_stream_plan.lane_indices[:, None]
                compact_genotype_matrix = candidate_inputs.genotype_matrix_by_variant[
                    compact_lane_indices,
                    compact_carrier_indices,
                ]
                compact_phenotype_matrix = candidate_inputs.lanes.phenotype_matrix[
                    compact_lane_indices,
                    compact_carrier_indices,
                ]
                compact_offset_matrix = candidate_inputs.null_firth_offset_matrix[
                    compact_lane_indices,
                    compact_carrier_indices,
                ]
                return compute_compact_sparse_firth_variantwise_fixed_batches_with_solver_parameters(
                    phenotype_matrix=jnp.asarray(compact_phenotype_matrix, dtype=jnp.float64),
                    genotype_matrix_by_variant=jnp.asarray(compact_genotype_matrix, dtype=jnp.float64),
                    offset_matrix=jnp.asarray(compact_offset_matrix, dtype=jnp.float64),
                    active_carrier_slot_mask=compact_carrier_slot_mask,
                    full_null_deviance=jnp.take(
                        candidate_inputs.full_null_deviance,
                        compact_stream_plan.lane_indices,
                        axis=0,
                    ),
                    active_mask=compact_stream_plan.active_mask,
                    fallback_count=compact_stream_plan.active_count,
                    firth_batch_size=firth_batch_size,
                    null_failed_mask=jnp.take(
                        candidate_inputs.null_failed_mask,
                        compact_stream_plan.lane_indices,
                        axis=0,
                    ),
                    solver_parameters=regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
                        kernel_config
                    ),
                )

            compact_fallback_result = regenie2_binary_firth_types.build_empty_firth_variant_result(
                compact_stream_plan.active_mask.shape[0]
            )

            compact_result = jax.lax.cond(
                compact_stream_has_lanes,
                compute_compact_stream,
                lambda _: compact_fallback_result,
                operand=None,
            )

            scattered_dense_result = jax.lax.cond(
                dense_stream_has_lanes,
                lambda base: scatter_firth_variant_result_by_lane_stream(
                    base_result=base,
                    lane_indices=dense_stream_plan.lane_indices,
                    active_mask=dense_stream_plan.active_mask,
                    stream_result=dense_result,
                ),
                lambda base: base,
                empty_result,
            )
            return jax.lax.cond(
                compact_stream_has_lanes,
                lambda base: scatter_firth_variant_result_by_lane_stream(
                    base_result=base,
                    lane_indices=compact_stream_plan.lane_indices,
                    active_mask=compact_stream_plan.active_mask,
                    stream_result=compact_result,
                ),
                lambda base: base,
                scattered_dense_result,
            )

        return jax.lax.cond(
            jnp.any(compact_sparse_lane_mask),
            compute_split_path,
            lambda _: compute_without_sparse_compaction(),
            operand=None,
        )

    return jax.lax.cond(
        jnp.any(active_mask & candidate_inputs.sparse_correction_mask),
        compute_with_sparse_compaction,
        lambda _: compute_without_sparse_compaction(),
        operand=None,
    )
