"""Public linear REGENIE step 2 compute API."""

from __future__ import annotations

import functools
import typing

import jax

from g.compute.common import genotype
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state

if typing.TYPE_CHECKING:
    from g import types as g_types

LINEAR_SCORE_STATIC_ARGNAMES = (
    "score_dtype",
    "linear_minimum_variance",
    "linear_relative_variance_tolerance",
)


@functools.partial(
    jax.jit,
    static_argnames=LINEAR_SCORE_STATIC_ARGNAMES,
    donate_argnames=(
        "packed_probability_pairs_by_variant",
        "genotype_dosage_sum",
        "genotype_observation_count",
        "genotype_imputed_dosage_square_sum",
    ),
)
def compute_multi_linear_chunk_packed8_donating_inputs(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    genotype_dosage_sum: jax.Array | None,
    genotype_observation_count: jax.Array | None,
    genotype_imputed_dosage_square_sum: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    linear_minimum_variance: float,
    linear_relative_variance_tolerance: float,
) -> regenie2_linear_score.Regenie2MultiLinearChunkResult:
    """Decode packed8 probabilities on device and compute multi-trait quantitative statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        genotype_dosage_sum=genotype_dosage_sum,
        genotype_observation_count=genotype_observation_count,
        genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
        score_dtype=score_dtype,
        linear_minimum_variance=linear_minimum_variance,
        linear_relative_variance_tolerance=linear_relative_variance_tolerance,
    )
