"""JIT dispatch kernels for variant-major Firth corrections."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.variant_major_correction import fixed_capacity


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "chromosome_state",
        "genotype_matrix_by_variant",
        "diagnostic_result",
        "candidate_mask",
        "fallback_count",
        "sparse_candidate_mask",
        "dosage_sum",
        "observation_count",
    ),
    meta_fields=(
        "correction_plan",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
        "kernel_config",
    ),
)
@dataclass(frozen=True)
class FirthVariantMajorDeviceDispatchOperands:
    """Operands for single-trait variant-major Firth capacity dispatch."""

    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState
    genotype_matrix_by_variant: jax.Array
    diagnostic_result: regenie2_binary_result.Regenie2BinaryChunkResult
    candidate_mask: jax.Array
    fallback_count: jax.Array
    sparse_candidate_mask: jax.Array | None
    dosage_sum: jax.Array | None
    observation_count: jax.Array | None
    correction_plan: types.BinaryCorrectionPlan
    tiny_candidate_capacity: int
    small_candidate_capacity: int
    bounded_candidate_capacity: int
    kernel_config: regenie2_binary_config.BinaryKernelConfig


def return_empty_firth_variant_major_diagnostics(
    operands: FirthVariantMajorDeviceDispatchOperands,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Return score results with empty Firth diagnostics."""
    return operands.diagnostic_result


def apply_tiny_firth_variant_major_corrections(
    operands: FirthVariantMajorDeviceDispatchOperands,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply tiny-capacity single-trait variant-major Firth corrections."""
    return fixed_capacity.apply_firth_variant_major_fixed_capacity_corrections(
        chromosome_state=operands.chromosome_state,
        genotype_matrix_by_variant=operands.genotype_matrix_by_variant,
        result=operands.diagnostic_result,
        correction_plan=operands.correction_plan,
        candidate_mask=operands.candidate_mask,
        fallback_count=operands.fallback_count,
        candidate_capacity=operands.tiny_candidate_capacity,
        order_candidates=False,
        kernel_config=operands.kernel_config,
        sparse_candidate_mask=operands.sparse_candidate_mask,
        dosage_sum=operands.dosage_sum,
        observation_count=operands.observation_count,
    )


def apply_small_firth_variant_major_corrections(
    operands: FirthVariantMajorDeviceDispatchOperands,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply small-capacity single-trait variant-major Firth corrections."""
    return fixed_capacity.apply_firth_variant_major_fixed_capacity_corrections(
        chromosome_state=operands.chromosome_state,
        genotype_matrix_by_variant=operands.genotype_matrix_by_variant,
        result=operands.diagnostic_result,
        correction_plan=operands.correction_plan,
        candidate_mask=operands.candidate_mask,
        fallback_count=operands.fallback_count,
        candidate_capacity=operands.small_candidate_capacity,
        order_candidates=False,
        kernel_config=operands.kernel_config,
        sparse_candidate_mask=operands.sparse_candidate_mask,
        dosage_sum=operands.dosage_sum,
        observation_count=operands.observation_count,
    )


def apply_bounded_firth_variant_major_corrections(
    operands: FirthVariantMajorDeviceDispatchOperands,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply bounded-capacity single-trait variant-major Firth corrections."""
    return fixed_capacity.apply_firth_variant_major_fixed_capacity_corrections(
        chromosome_state=operands.chromosome_state,
        genotype_matrix_by_variant=operands.genotype_matrix_by_variant,
        result=operands.diagnostic_result,
        correction_plan=operands.correction_plan,
        candidate_mask=operands.candidate_mask,
        fallback_count=operands.fallback_count,
        candidate_capacity=operands.bounded_candidate_capacity,
        order_candidates=True,
        kernel_config=operands.kernel_config,
        sparse_candidate_mask=operands.sparse_candidate_mask,
        dosage_sum=operands.dosage_sum,
        observation_count=operands.observation_count,
    )


def apply_small_or_bounded_firth_variant_major_corrections(
    operands: FirthVariantMajorDeviceDispatchOperands,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply small or bounded single-trait corrections by fallback count."""
    return jax.lax.cond(
        operands.fallback_count <= operands.small_candidate_capacity,
        apply_small_firth_variant_major_corrections,
        apply_bounded_firth_variant_major_corrections,
        operands,
    )


def apply_tiered_firth_variant_major_corrections(
    operands: FirthVariantMajorDeviceDispatchOperands,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply tiered single-trait variant-major Firth corrections."""
    return jax.lax.cond(
        operands.fallback_count <= operands.tiny_candidate_capacity,
        apply_tiny_firth_variant_major_corrections,
        apply_small_or_bounded_firth_variant_major_corrections,
        operands,
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply common-case Firth corrections with device-side zero and tiered dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    operands = FirthVariantMajorDeviceDispatchOperands(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        diagnostic_result=diagnostic_result,
        candidate_mask=candidate_mask,
        fallback_count=fallback_count,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        correction_plan=correction_plan,
        tiny_candidate_capacity=tiny_candidate_capacity,
        small_candidate_capacity=small_candidate_capacity,
        bounded_candidate_capacity=bounded_candidate_capacity,
        kernel_config=kernel_config,
    )
    return jax.lax.cond(
        fallback_count == 0,
        return_empty_firth_variant_major_diagnostics,
        apply_tiered_firth_variant_major_corrections,
        operands,
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply rare overflow single-trait Firth corrections in a separate executable."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    return fixed_capacity.apply_firth_variant_major_fixed_capacity_corrections(
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply common-case Firth corrections from packed8 rows with device-side dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2BinaryChunkResult:
            return fixed_capacity.apply_firth_packed8_fixed_capacity_corrections(
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
            return fixed_capacity.apply_firth_packed8_fixed_capacity_corrections(
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
            return fixed_capacity.apply_firth_packed8_fixed_capacity_corrections(
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    """Apply rare overflow packed8 single-trait Firth corrections in a separate executable."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_score_result_with_empty_firth_diagnostics(result)
    return fixed_capacity.apply_firth_packed8_fixed_capacity_corrections(
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
        "overflow_candidate_capacity",
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
    overflow_candidate_capacity: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply common multi-trait Firth corrections with device-side capacity dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
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
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
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
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
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

        def apply_overflow_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
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

        return jax.lax.cond(
            fallback_count <= tiny_candidate_capacity,
            apply_tiny_corrections,
            lambda _: jax.lax.cond(
                fallback_count <= small_candidate_capacity,
                apply_small_corrections,
                lambda __: jax.lax.cond(
                    fallback_count <= bounded_candidate_capacity,
                    apply_bounded_corrections,
                    apply_overflow_corrections,
                    operand=None,
                ),
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply rare overflow multi-trait Firth corrections in a separate executable."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)
    return fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply multi-trait Firth corrections from packed8 rows with device-side dispatch."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)

    def return_empty_diagnostics(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        return diagnostic_result

    def apply_candidate_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
        def apply_tiny_corrections(_: None) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
            return fixed_capacity.apply_firth_multi_packed8_fixed_capacity_corrections(
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
            return fixed_capacity.apply_firth_multi_packed8_fixed_capacity_corrections(
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
            return fixed_capacity.apply_firth_multi_packed8_fixed_capacity_corrections(
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
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Apply rare overflow multi-trait Firth corrections from packed8 rows."""
    candidate_mask = result.extra_code == types.BinaryExtraCode.FIRTH.value
    fallback_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    diagnostic_result = regenie2_binary_result.expand_multi_score_result_with_empty_firth_diagnostics(result)
    return fixed_capacity.apply_firth_multi_packed8_fixed_capacity_corrections(
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
