"""JIT dispatch kernels for variant-major Firth corrections."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary.variant_major_correction import fixed_capacity

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_binary import result as regenie2_binary_result
    from g.compute.regenie2_binary import state as regenie2_binary_state


@jax.jit
def count_firth_candidates(correction_code: jax.Array) -> jax.Array:
    """Count Firth candidates without changing the score executable."""
    return jnp.sum(
        correction_code == types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
        dtype=jnp.int32,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "firth_se",
        "kernel_config",
        "candidate_capacity",
        "order_candidates",
    ),
    donate_argnames=("result",),
)
def apply_fixed_capacity_corrections_multi_firth_variant_major_donating_result(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_candidate_count: jax.Array,
    firth_se: bool,
    candidate_capacity: int,
    order_candidates: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply one statically selected fixed-capacity multi-trait Firth correction."""
    candidate_mask = result.correction_code == types.BinaryCorrectionCode.FIRTH_SUCCESS.value
    return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        firth_se=firth_se,
        candidate_mask=candidate_mask,
        fallback_count=firth_candidate_count,
        candidate_capacity=candidate_capacity,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        native_genotype_mean=native_genotype_mean,
    )


def apply_host_selected_corrections_multi_firth_variant_major(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_se: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Select one static Firth capacity after materializing the device count."""
    capacity_plan = regenie2_binary_candidate_planning.build_multi_firth_candidate_capacity_plan(
        trait_count=result.beta.shape[0],
        variant_count=genotype_matrix_by_variant.shape[0],
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )
    firth_candidate_count = count_firth_candidates(result.correction_code)
    host_candidate_count = int(jax.device_get(firth_candidate_count))
    if host_candidate_count == 0:
        return result

    if host_candidate_count <= capacity_plan.tiny_candidate_capacity:
        candidate_capacity = capacity_plan.tiny_candidate_capacity
        order_candidates = False
    elif host_candidate_count <= capacity_plan.small_candidate_capacity:
        candidate_capacity = capacity_plan.small_candidate_capacity
        order_candidates = candidate_capacity > kernel_config.firth_candidate.batch_size
    elif host_candidate_count <= capacity_plan.bounded_candidate_capacity:
        candidate_capacity = capacity_plan.bounded_candidate_capacity
        order_candidates = True
    else:
        candidate_capacity = capacity_plan.overflow_candidate_capacity
        order_candidates = True

    return apply_fixed_capacity_corrections_multi_firth_variant_major_donating_result(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        firth_candidate_count=firth_candidate_count,
        firth_se=firth_se,
        candidate_capacity=candidate_capacity,
        order_candidates=order_candidates,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        native_genotype_mean=native_genotype_mean,
    )
