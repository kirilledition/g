"""JIT dispatch kernels for variant-major Firth corrections."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary.variant_major_correction import fixed_capacity

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_binary import state as regenie2_binary_state


@functools.partial(
    jax.jit,
    static_argnames=("firth_se", "kernel_config"),
    donate_argnames=("result",),
)
def apply_static_capacity_corrections_multi_firth_variant_major_donating_result(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    result: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    firth_se: bool,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    native_genotype_mean: jax.Array | None,
) -> regenie2_binary_result.CorrectedMultiBinaryScoreChunkResult:
    """Apply one hard-capacity correction without host candidate-count synchronization."""
    candidate_mask = result.correction_code == types.BinaryCorrectionCode.FIRTH_SUCCESS.value
    firth_candidate_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    candidate_capacity = (
        min(kernel_config.firth_candidate.candidate_capacity, genotype_matrix_by_variant.shape[0])
        * result.beta.shape[0]
    )
    corrected_result = fixed_capacity.apply_firth_multi_variant_major_fixed_capacity_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=result,
        firth_se=firth_se,
        candidate_mask=candidate_mask,
        fallback_count=firth_candidate_count,
        candidate_capacity=candidate_capacity,
        order_candidates=candidate_capacity > kernel_config.firth_candidate.batch_size,
        kernel_config=kernel_config,
        sparse_candidate_mask=sparse_candidate_mask,
        native_genotype_mean=native_genotype_mean,
    )
    return regenie2_binary_result.CorrectedMultiBinaryScoreChunkResult(
        association=corrected_result,
        firth_candidate_count=firth_candidate_count,
        firth_candidate_capacity=candidate_capacity,
    )
