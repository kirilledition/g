"""Direct variant-major JAX binary association path."""

from __future__ import annotations

import functools

import jax

from g import types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import types as regenie2_binary_types
from g.compute.regenie2_binary import variant_major_correction as regenie2_binary_variant_major_correction


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute binary association from a variant-major chunk."""
    score_test_result = regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
    )
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )
