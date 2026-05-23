"""Public binary REGENIE step 2 compute API."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g import types as g_types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary import types as regenie2_binary_types
from g.compute.regenie2_binary import variant_major as regenie2_binary_variant_major

BinaryScoreTestChunkComputeFunction = typing.Callable[
    [regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, g_types.BinaryCorrectionPlan],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
BinaryChunkComputeFunction = typing.Callable[
    [
        regenie2_binary_types.Regenie2BinaryChromosomeState,
        jax.Array,
        g_types.BinaryCorrectionPlan,
        jax.Array | None,
        regenie2_binary_types.BinaryKernelConfig,
    ],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
BinaryVariantMajorChunkComputeFunction = typing.Callable[
    [
        regenie2_binary_types.Regenie2BinaryChromosomeState,
        jax.Array,
        g_types.BinaryCorrectionPlan,
        jax.Array | None,
        regenie2_binary_types.BinaryKernelConfig,
    ],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute the uncorrected score-test result for one binary chunk."""
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
        correction_plan=correction_plan,
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary REGENIE step 2 association using one genotype chunk."""
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
            correction_plan=correction_plan,
        )

    def compute_one_trait(trait_index: jax.Array) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        single_chromosome_state = regenie2_binary_state.build_single_binary_chromosome_state_from_multi(
            chromosome_state,
            trait_index,
        )
        return compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=single_chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return regenie2_binary_result.stack_binary_chunk_results(
        [compute_one_trait(trait_index) for trait_index in range(trait_count)]
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes."""
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
        )

    def compute_one_trait(trait_index: jax.Array) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        single_chromosome_state = regenie2_binary_state.build_single_binary_chromosome_state_from_multi(
            chromosome_state,
            trait_index,
        )
        compute_variant_major_chunk = (
            regenie2_binary_variant_major.compute_regenie2_binary_chunk_from_chromosome_state_variant_major
        )
        return compute_variant_major_chunk(
            chromosome_state=single_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return regenie2_binary_result.stack_binary_chunk_results(
        [compute_one_trait(trait_index) for trait_index in range(trait_count)]
    )


def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association using cached null state."""
    score_test_result = compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state,
        genotype_matrix,
        correction_plan,
    )
    return regenie2_binary_correction.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def compute_regenie2_binary_chunk(
    state: regenie2_binary_types.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = regenie2_binary_state.prepare_regenie2_binary_chromosome_state(
        state, loco_offset, correction_plan, kernel_config
    )
    compute_regenie2_binary_chunk_from_state = typing.cast(
        "BinaryChunkComputeFunction",
        compute_regenie2_binary_chunk_from_chromosome_state,
    )
    return compute_regenie2_binary_chunk_from_state(
        chromosome_state,
        genotype_matrix,
        correction_plan,
        sparse_candidate_mask,
        kernel_config,
    )
