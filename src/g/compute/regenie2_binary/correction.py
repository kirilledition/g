"""Sample-major binary candidate correction adapters for REGENIE step 2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import types as regenie2_binary_types
from g.compute.regenie2_binary import variant_major_correction as regenie2_binary_variant_major_correction


def apply_device_candidate_corrections_firth(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Apply Firth corrections to a sample-major genotype chunk."""
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_firth_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def apply_device_candidate_corrections(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
    correction_plan: types.BinaryCorrectionPlan,
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Apply binary candidate corrections to a sample-major genotype chunk."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return result
    if correction_plan.method == types.BinaryFallbackMethod.FIRTH:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    if correction_plan.method == types.BinaryFallbackMethod.SPA:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    return apply_device_candidate_corrections_firth(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )
