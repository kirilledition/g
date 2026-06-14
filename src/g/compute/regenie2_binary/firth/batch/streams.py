"""Lane-stream helpers for Firth candidate batches."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.compute.regenie2_binary.firth.batch import models


def build_firth_lane_stream_plan(active_lane_mask: jax.Array) -> models.FirthLaneStreamPlan:
    """Pack active candidate-lane positions into a fixed-capacity stream."""
    stream_capacity = active_lane_mask.shape[0]
    lane_indices = jnp.nonzero(active_lane_mask, size=stream_capacity, fill_value=0)[0]
    active_count = jnp.sum(active_lane_mask, dtype=jnp.int32)
    active_mask = jnp.arange(stream_capacity, dtype=jnp.int32) < active_count
    return models.FirthLaneStreamPlan(
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
        penalized_log_likelihood=base_result.penalized_log_likelihood.at[scatter_indices].set(
            stream_result.penalized_log_likelihood,
            mode="drop",
        ),
        converged_mask=base_result.converged_mask.at[scatter_indices].set(
            stream_result.converged_mask,
            mode="drop",
        ),
        valid_mask=base_result.valid_mask.at[scatter_indices].set(stream_result.valid_mask, mode="drop"),
        iteration_count=base_result.iteration_count.at[scatter_indices].set(
            stream_result.iteration_count,
            mode="drop",
        ),
        failure_code=base_result.failure_code.at[scatter_indices].set(stream_result.failure_code, mode="drop"),
        convergence_reason_code=base_result.convergence_reason_code.at[scatter_indices].set(
            stream_result.convergence_reason_code,
            mode="drop",
        ),
        correction_code=base_result.correction_code.at[scatter_indices].set(
            stream_result.correction_code,
            mode="drop",
        ),
        sparse_correction_mask=base_result.sparse_correction_mask.at[scatter_indices].set(
            stream_result.sparse_correction_mask,
            mode="drop",
        ),
        pseudo_firth_iteration_count=base_result.pseudo_firth_iteration_count.at[scatter_indices].set(
            stream_result.pseudo_firth_iteration_count,
            mode="drop",
        ),
        nr_zero_start_iteration_count=base_result.nr_zero_start_iteration_count.at[scatter_indices].set(
            stream_result.nr_zero_start_iteration_count,
            mode="drop",
        ),
        nr_warm_start_iteration_count=base_result.nr_warm_start_iteration_count.at[scatter_indices].set(
            stream_result.nr_warm_start_iteration_count,
            mode="drop",
        ),
    )
