"""Fixed-batch Firth computation helpers."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.compute.regenie2_binary.firth.batch import models, streams

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config


def compute_firth_variantwise(
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute device-side Firth fits for a padded set of candidate lanes."""
    del null_logistic_coefficients

    scalar_offset_vector = jnp.asarray(null_firth_offset, dtype=jnp.float64)
    scalar_phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)

    def fit_variant(
        genotype_vector: jax.Array,
        raw_genotype_vector: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        if not kernel_config.approximate_firth.use_block_math:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
                phenotype_vector=scalar_phenotype_vector,
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=scalar_offset_vector,
                carrier_sample_mask=raw_genotype_vector
                > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
                sparse_correction=sparse_correction,
                warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
                skip_firth=skip_firth,
                null_failed=~jnp.isfinite(null_penalized_log_likelihood),
                kernel_config=kernel_config,
            )
        return regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            loco_offset=loco_offset,
            initial_coefficients=variant_initial_coefficients,
            skip_firth=skip_firth,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0))(
        genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
        initial_coefficients,
        skip_firth_mask,
        sparse_correction_mask,
    )


def compute_firth_multi_variantwise(
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute device-side Firth fits for lane-specific multi-trait candidates."""
    del null_logistic_coefficients

    def fit_variant(
        phenotype_vector: jax.Array,
        null_firth_offset: jax.Array,
        genotype_vector: jax.Array,
        raw_genotype_vector: jax.Array,
        loco_offset: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
        lane_null_penalized_log_likelihood: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        if not kernel_config.approximate_firth.use_block_math:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
                phenotype_vector=jnp.asarray(phenotype_vector, dtype=jnp.float64),
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=jnp.asarray(null_firth_offset, dtype=jnp.float64),
                carrier_sample_mask=raw_genotype_vector
                > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
                sparse_correction=sparse_correction,
                warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
                skip_firth=skip_firth,
                null_failed=~jnp.isfinite(lane_null_penalized_log_likelihood),
                kernel_config=kernel_config,
            )
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

    return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
        phenotype_matrix,
        null_firth_offset_matrix,
        genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
        loco_offset_matrix,
        initial_coefficients,
        skip_firth_mask,
        sparse_correction_mask,
        null_penalized_log_likelihood,
    )


def build_compact_sparse_carrier_indices(
    *,
    raw_genotype_matrix_by_variant: jax.Array,
    sparse_carrier_dosage_threshold: float,
) -> jax.Array:
    """Build fixed-capacity carrier sample indices for sparse Firth lanes."""
    carrier_sample_mask = raw_genotype_matrix_by_variant > sparse_carrier_dosage_threshold

    def build_one_lane(carrier_mask: jax.Array) -> jax.Array:
        return jnp.nonzero(carrier_mask, size=models.SPARSE_FIRTH_CARRIER_CAPACITY, fill_value=0)[0]

    return jax.vmap(build_one_lane)(carrier_sample_mask)


def compute_compact_sparse_firth_variantwise_fixed_batches(
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
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute compact sparse scalar Firth fits using fixed-size carrier slots."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    phenotype_batches = phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    offset_batches = offset_matrix.reshape((batch_count, firth_batch_size, -1))
    active_carrier_slot_mask_batches = active_carrier_slot_mask.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    full_null_deviance_vector = jnp.broadcast_to(full_null_deviance, active_mask.shape)
    null_failed_mask_vector = jnp.broadcast_to(null_failed_mask, active_mask.shape)
    full_null_deviance_batches = full_null_deviance_vector.reshape((batch_count, firth_batch_size))
    null_failed_mask_batches = null_failed_mask_vector.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def fit_variant(
            phenotype_vector: jax.Array,
            genotype_vector: jax.Array,
            offset_vector: jax.Array,
            carrier_slot_mask: jax.Array,
            lane_full_null_deviance: jax.Array,
            skip_firth: jax.Array,
            null_failed: jax.Array,
        ) -> regenie2_binary_firth_types.FirthVariantResult:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth_compact_carriers(
                phenotype_vector=phenotype_vector,
                genotype_vector=genotype_vector,
                offset_vector=offset_vector,
                active_carrier_slot_mask=carrier_slot_mask,
                full_null_deviance=lane_full_null_deviance,
                warm_start_beta=jnp.asarray(0.0, dtype=offset_vector.dtype),
                skip_firth=skip_firth,
                null_failed=null_failed,
                kernel_config=kernel_config,
            )

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return jax.vmap(
                fit_variant,
                in_axes=(0, 0, 0, 0, 0, 0, 0),
            )(
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


def compute_firth_variantwise_fixed_batches_without_sparse_compaction(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute Firth fits for flattened candidate lanes using fixed-size batches."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    raw_genotype_batches = raw_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    sparse_correction_mask_batches = sparse_correction_mask.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return compute_firth_variantwise(
                covariate_matrix=covariate_matrix,
                null_logistic_coefficients=null_logistic_coefficients,
                null_firth_offset=null_firth_offset,
                phenotype_vector=phenotype_vector,
                genotype_matrix_by_variant=genotype_batches[batch_index],
                raw_genotype_matrix_by_variant=raw_genotype_batches[batch_index],
                loco_offset=loco_offset,
                initial_coefficients=initial_coefficient_batches[batch_index],
                skip_firth_mask=~active_mask_batches[batch_index],
                sparse_correction_mask=sparse_correction_mask_batches[batch_index],
                null_penalized_log_likelihood=null_penalized_log_likelihood,
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


def compute_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute multi-trait Firth fits for flattened candidate lanes using fixed-size batches."""
    batch_count = active_mask.shape[0] // firth_batch_size
    active_batch_count = (fallback_count + firth_batch_size - 1) // firth_batch_size
    null_logistic_coefficient_batches = null_logistic_coefficients.reshape((batch_count, firth_batch_size, -1))
    null_firth_offset_batches = null_firth_offset_matrix.reshape((batch_count, firth_batch_size, -1))
    phenotype_batches = phenotype_matrix.reshape((batch_count, firth_batch_size, -1))
    genotype_batches = genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    raw_genotype_batches = raw_genotype_matrix_by_variant.reshape((batch_count, firth_batch_size, -1))
    loco_offset_batches = loco_offset_matrix.reshape((batch_count, firth_batch_size, -1))
    initial_coefficient_batches = initial_coefficients.reshape((batch_count, firth_batch_size, -1))
    active_mask_batches = active_mask.reshape((batch_count, firth_batch_size))
    sparse_correction_mask_batches = sparse_correction_mask.reshape((batch_count, firth_batch_size))
    null_penalized_log_likelihood_batches = null_penalized_log_likelihood.reshape((batch_count, firth_batch_size))
    empty_firth_variant_result = regenie2_binary_firth_types.build_empty_firth_variant_result(firth_batch_size)

    def compute_firth_batch(
        carry: None,
        batch_index: jax.Array,
    ) -> tuple[None, regenie2_binary_firth_types.FirthVariantResult]:
        del carry

        def run_active_batch(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            return compute_firth_multi_variantwise(
                covariate_matrix=covariate_matrix,
                null_logistic_coefficients=null_logistic_coefficient_batches[batch_index],
                null_firth_offset_matrix=null_firth_offset_batches[batch_index],
                phenotype_matrix=phenotype_batches[batch_index],
                genotype_matrix_by_variant=genotype_batches[batch_index],
                raw_genotype_matrix_by_variant=raw_genotype_batches[batch_index],
                loco_offset_matrix=loco_offset_batches[batch_index],
                initial_coefficients=initial_coefficient_batches[batch_index],
                skip_firth_mask=~active_mask_batches[batch_index],
                sparse_correction_mask=sparse_correction_mask_batches[batch_index],
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


def compute_firth_variantwise_fixed_batches(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    full_null_deviance: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute single-trait Firth fits with compact sparse lanes when eligible."""

    def compute_without_sparse_compaction() -> regenie2_binary_firth_types.FirthVariantResult:
        return compute_firth_variantwise_fixed_batches_without_sparse_compaction(
            covariate_matrix=covariate_matrix,
            null_logistic_coefficients=null_logistic_coefficients,
            null_firth_offset=null_firth_offset,
            phenotype_vector=phenotype_vector,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
            loco_offset=loco_offset,
            initial_coefficients=initial_coefficients,
            active_mask=active_mask,
            sparse_correction_mask=sparse_correction_mask,
            fallback_count=fallback_count,
            firth_batch_size=firth_batch_size,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    if kernel_config.approximate_firth.use_block_math:
        return compute_without_sparse_compaction()

    def compute_with_sparse_compaction(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
        carrier_sample_mask = (
            raw_genotype_matrix_by_variant > kernel_config.approximate_firth.sparse_carrier_dosage_threshold
        )
        carrier_count = jnp.sum(carrier_sample_mask, axis=1, dtype=jnp.int32)
        compact_sparse_lane_mask = (
            active_mask & sparse_correction_mask & (carrier_count <= models.SPARSE_FIRTH_CARRIER_CAPACITY)
        )

        def compute_split_path(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            dense_lane_mask = active_mask & (~compact_sparse_lane_mask)
            dense_stream_plan = streams.build_firth_lane_stream_plan(dense_lane_mask)
            compact_stream_plan = streams.build_firth_lane_stream_plan(compact_sparse_lane_mask)
            dense_fallback_result = regenie2_binary_firth_types.build_empty_firth_variant_result(
                dense_stream_plan.active_mask.shape[0]
            )
            empty_result = regenie2_binary_firth_types.build_empty_firth_variant_result(active_mask.shape[0])
            dense_stream_has_lanes = dense_stream_plan.active_count > 0
            compact_stream_has_lanes = compact_stream_plan.active_count > 0

            def compute_dense_stream(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
                dense_null_penalized_log_likelihood = (
                    null_penalized_log_likelihood
                    if null_penalized_log_likelihood.ndim == 0
                    else jnp.take(
                        null_penalized_log_likelihood,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    )
                )
                return compute_firth_variantwise_fixed_batches_without_sparse_compaction(
                    covariate_matrix=covariate_matrix,
                    null_logistic_coefficients=null_logistic_coefficients,
                    null_firth_offset=null_firth_offset,
                    phenotype_vector=phenotype_vector,
                    genotype_matrix_by_variant=jnp.take(
                        genotype_matrix_by_variant,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    raw_genotype_matrix_by_variant=jnp.take(
                        raw_genotype_matrix_by_variant,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    loco_offset=loco_offset,
                    initial_coefficients=jnp.take(initial_coefficients, dense_stream_plan.lane_indices, axis=0),
                    active_mask=dense_stream_plan.active_mask,
                    sparse_correction_mask=jnp.take(sparse_correction_mask, dense_stream_plan.lane_indices, axis=0),
                    fallback_count=dense_stream_plan.active_count,
                    firth_batch_size=firth_batch_size,
                    null_penalized_log_likelihood=dense_null_penalized_log_likelihood,
                    kernel_config=kernel_config,
                )

            dense_result = jax.lax.cond(
                dense_stream_has_lanes,
                compute_dense_stream,
                lambda _: dense_fallback_result,
                operand=None,
            )

            def compute_compact_stream(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
                compact_null_failed_mask = (
                    ~jnp.isfinite(null_penalized_log_likelihood)
                    if null_penalized_log_likelihood.ndim == 0
                    else ~jnp.isfinite(
                        jnp.take(null_penalized_log_likelihood, compact_stream_plan.lane_indices, axis=0)
                    )
                )
                compact_lane_raw_genotype_matrix = jnp.take(
                    raw_genotype_matrix_by_variant,
                    compact_stream_plan.lane_indices,
                    axis=0,
                )
                compact_carrier_indices = build_compact_sparse_carrier_indices(
                    raw_genotype_matrix_by_variant=compact_lane_raw_genotype_matrix,
                    sparse_carrier_dosage_threshold=kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
                )
                compact_carrier_count = jnp.take(carrier_count, compact_stream_plan.lane_indices, axis=0)
                compact_carrier_slot_mask = (
                    jnp.arange(models.SPARSE_FIRTH_CARRIER_CAPACITY, dtype=jnp.int32)[None, :]
                    < compact_carrier_count[:, None]
                ) & compact_stream_plan.active_mask[:, None]
                compact_lane_genotype_matrix = jnp.take(
                    genotype_matrix_by_variant,
                    compact_stream_plan.lane_indices,
                    axis=0,
                )
                compact_genotype_matrix = jnp.take_along_axis(
                    compact_lane_genotype_matrix,
                    compact_carrier_indices,
                    axis=1,
                )
                compact_phenotype_matrix = jnp.take(
                    jnp.asarray(phenotype_vector, dtype=jnp.float64),
                    compact_carrier_indices,
                    axis=0,
                )
                compact_offset_matrix = jnp.take(
                    jnp.asarray(null_firth_offset, dtype=jnp.float64),
                    compact_carrier_indices,
                    axis=0,
                )
                return compute_compact_sparse_firth_variantwise_fixed_batches(
                    phenotype_matrix=compact_phenotype_matrix,
                    genotype_matrix_by_variant=jnp.asarray(compact_genotype_matrix, dtype=jnp.float64),
                    offset_matrix=compact_offset_matrix,
                    active_carrier_slot_mask=compact_carrier_slot_mask,
                    full_null_deviance=jnp.asarray(full_null_deviance, dtype=jnp.float64),
                    active_mask=compact_stream_plan.active_mask,
                    fallback_count=compact_stream_plan.active_count,
                    firth_batch_size=firth_batch_size,
                    null_failed_mask=compact_null_failed_mask,
                    kernel_config=kernel_config,
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
                lambda base: streams.scatter_firth_variant_result_by_lane_stream(
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
                lambda base: streams.scatter_firth_variant_result_by_lane_stream(
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
        jnp.any(active_mask & sparse_correction_mask),
        compute_with_sparse_compaction,
        lambda _: compute_without_sparse_compaction(),
        operand=None,
    )


def compute_firth_multi_variantwise_fixed_batches(
    *,
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset_matrix: jax.Array,
    initial_coefficients: jax.Array,
    active_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    fallback_count: jax.Array,
    firth_batch_size: int,
    null_penalized_log_likelihood: jax.Array,
    full_null_deviance: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute multi-trait Firth fits with compact sparse lanes when eligible."""

    def compute_without_sparse_compaction() -> regenie2_binary_firth_types.FirthVariantResult:
        return compute_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
            covariate_matrix=covariate_matrix,
            null_logistic_coefficients=null_logistic_coefficients,
            null_firth_offset_matrix=null_firth_offset_matrix,
            phenotype_matrix=phenotype_matrix,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
            loco_offset_matrix=loco_offset_matrix,
            initial_coefficients=initial_coefficients,
            active_mask=active_mask,
            sparse_correction_mask=sparse_correction_mask,
            fallback_count=fallback_count,
            firth_batch_size=firth_batch_size,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    if kernel_config.approximate_firth.use_block_math:
        return compute_without_sparse_compaction()

    def compute_with_sparse_compaction(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
        carrier_sample_mask = (
            raw_genotype_matrix_by_variant > kernel_config.approximate_firth.sparse_carrier_dosage_threshold
        )
        carrier_count = jnp.sum(carrier_sample_mask, axis=1, dtype=jnp.int32)
        compact_sparse_lane_mask = (
            active_mask & sparse_correction_mask & (carrier_count <= models.SPARSE_FIRTH_CARRIER_CAPACITY)
        )

        def compute_split_path(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
            dense_lane_mask = active_mask & (~compact_sparse_lane_mask)
            dense_stream_plan = streams.build_firth_lane_stream_plan(dense_lane_mask)
            compact_stream_plan = streams.build_firth_lane_stream_plan(compact_sparse_lane_mask)
            dense_fallback_result = regenie2_binary_firth_types.build_empty_firth_variant_result(
                dense_stream_plan.active_mask.shape[0]
            )
            empty_result = regenie2_binary_firth_types.build_empty_firth_variant_result(active_mask.shape[0])
            dense_stream_has_lanes = dense_stream_plan.active_count > 0
            compact_stream_has_lanes = compact_stream_plan.active_count > 0

            def compute_dense_stream(_: None) -> regenie2_binary_firth_types.FirthVariantResult:
                return compute_firth_multi_variantwise_fixed_batches_without_sparse_compaction(
                    covariate_matrix=covariate_matrix,
                    null_logistic_coefficients=jnp.take(
                        null_logistic_coefficients, dense_stream_plan.lane_indices, axis=0
                    ),
                    null_firth_offset_matrix=jnp.take(null_firth_offset_matrix, dense_stream_plan.lane_indices, axis=0),
                    phenotype_matrix=jnp.take(phenotype_matrix, dense_stream_plan.lane_indices, axis=0),
                    genotype_matrix_by_variant=jnp.take(
                        genotype_matrix_by_variant, dense_stream_plan.lane_indices, axis=0
                    ),
                    raw_genotype_matrix_by_variant=jnp.take(
                        raw_genotype_matrix_by_variant,
                        dense_stream_plan.lane_indices,
                        axis=0,
                    ),
                    loco_offset_matrix=jnp.take(loco_offset_matrix, dense_stream_plan.lane_indices, axis=0),
                    initial_coefficients=jnp.take(initial_coefficients, dense_stream_plan.lane_indices, axis=0),
                    active_mask=dense_stream_plan.active_mask,
                    sparse_correction_mask=jnp.take(sparse_correction_mask, dense_stream_plan.lane_indices, axis=0),
                    fallback_count=dense_stream_plan.active_count,
                    firth_batch_size=firth_batch_size,
                    null_penalized_log_likelihood=jnp.take(
                        null_penalized_log_likelihood,
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
                compact_null_failed_mask = ~jnp.isfinite(
                    jnp.take(null_penalized_log_likelihood, compact_stream_plan.lane_indices)
                )
                compact_lane_raw_genotype_matrix = jnp.take(
                    raw_genotype_matrix_by_variant,
                    compact_stream_plan.lane_indices,
                    axis=0,
                )
                compact_carrier_indices = build_compact_sparse_carrier_indices(
                    raw_genotype_matrix_by_variant=compact_lane_raw_genotype_matrix,
                    sparse_carrier_dosage_threshold=kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
                )
                compact_carrier_count = jnp.take(carrier_count, compact_stream_plan.lane_indices, axis=0)
                compact_carrier_slot_mask = (
                    jnp.arange(models.SPARSE_FIRTH_CARRIER_CAPACITY, dtype=jnp.int32)[None, :]
                    < compact_carrier_count[:, None]
                ) & compact_stream_plan.active_mask[:, None]
                compact_lane_genotype_matrix = jnp.take(
                    genotype_matrix_by_variant,
                    compact_stream_plan.lane_indices,
                    axis=0,
                )
                compact_lane_phenotype_matrix = jnp.take(phenotype_matrix, compact_stream_plan.lane_indices, axis=0)
                compact_lane_offset_matrix = jnp.take(
                    null_firth_offset_matrix, compact_stream_plan.lane_indices, axis=0
                )
                compact_genotype_matrix = jnp.take_along_axis(
                    compact_lane_genotype_matrix,
                    compact_carrier_indices,
                    axis=1,
                )
                compact_phenotype_matrix = jnp.take_along_axis(
                    compact_lane_phenotype_matrix,
                    compact_carrier_indices,
                    axis=1,
                )
                compact_offset_matrix = jnp.take_along_axis(compact_lane_offset_matrix, compact_carrier_indices, axis=1)
                return compute_compact_sparse_firth_variantwise_fixed_batches(
                    phenotype_matrix=jnp.asarray(compact_phenotype_matrix, dtype=jnp.float64),
                    genotype_matrix_by_variant=jnp.asarray(compact_genotype_matrix, dtype=jnp.float64),
                    offset_matrix=jnp.asarray(compact_offset_matrix, dtype=jnp.float64),
                    active_carrier_slot_mask=compact_carrier_slot_mask,
                    full_null_deviance=jnp.take(full_null_deviance, compact_stream_plan.lane_indices, axis=0),
                    active_mask=compact_stream_plan.active_mask,
                    fallback_count=compact_stream_plan.active_count,
                    firth_batch_size=firth_batch_size,
                    null_failed_mask=compact_null_failed_mask,
                    kernel_config=kernel_config,
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
                lambda base: streams.scatter_firth_variant_result_by_lane_stream(
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
                lambda base: streams.scatter_firth_variant_result_by_lane_stream(
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
        jnp.any(active_mask & sparse_correction_mask),
        compute_with_sparse_compaction,
        lambda _: compute_without_sparse_compaction(),
        operand=None,
    )
