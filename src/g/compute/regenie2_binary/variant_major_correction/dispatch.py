"""JIT dispatch kernels for variant-major Firth corrections."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary.variant_major_correction import fixed_capacity

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_binary import result as regenie2_binary_result
    from g.compute.regenie2_binary import state as regenie2_binary_state


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
        "overflow_candidate_capacity",
    ),
    donate_argnames=("result",),
)
def apply_device_candidate_corrections_multi_firth_variant_major_donating_result(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    overflow_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Apply common multi-trait Firth corrections with device-side capacity dispatch."""
    candidate_mask = result.correction_code == types.BinaryCorrectionCode.FIRTH_SUCCESS.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)

    def return_score_result(_: None) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
        return result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=tiny_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                native_genotype_mean=native_genotype_mean,
            )

        def apply_small_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=small_candidate_capacity,
                order_candidates=False,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                native_genotype_mean=native_genotype_mean,
            )

        def apply_bounded_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=bounded_candidate_capacity,
                order_candidates=True,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                native_genotype_mean=native_genotype_mean,
            )

        def apply_overflow_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                result=result,
                correction_plan=correction_plan,
                candidate_mask=candidate_mask,
                fallback_count=fallback_count,
                candidate_capacity=overflow_candidate_capacity,
                order_candidates=True,
                kernel_config=kernel_config,
                sparse_candidate_mask=sparse_candidate_mask,
                native_genotype_mean=native_genotype_mean,
            )

        def apply_bounded_or_overflow_corrections(
            _: None,
        ) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
            if bounded_candidate_capacity == overflow_candidate_capacity:
                return apply_bounded_corrections(None)
            return jax.lax.cond(
                fallback_count <= bounded_candidate_capacity,
                apply_bounded_corrections,
                apply_overflow_corrections,
                operand=None,
            )

        return jax.lax.cond(
            fallback_count <= tiny_candidate_capacity,
            apply_tiny_corrections,
            lambda _: jax.lax.cond(
                fallback_count <= small_candidate_capacity,
                apply_small_corrections,
                apply_bounded_or_overflow_corrections,
                operand=None,
            ),
            operand=None,
        )

    return jax.lax.cond(
        fallback_count == 0,
        return_score_result,
        apply_candidate_corrections,
        operand=None,
    )
