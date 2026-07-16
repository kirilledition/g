"""Binary association result type for REGENIE step 2 compute."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax

from g.compute.common import result as association_result

type Regenie2MultiBinaryScoreChunkResult = association_result.AssociationResult[jax.Array, jax.Array]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("association", "firth_candidate_count"),
    meta_fields=("firth_candidate_capacity",),
)
@dataclass(frozen=True)
class CorrectedMultiBinaryScoreChunkResult:
    """Corrected association values with the device-resident Firth candidate count.

    Attributes:
        association: Trait-major score and approximate-Firth results.
        firth_candidate_count: Number of candidate lanes before fixed-capacity selection.
        firth_candidate_capacity: Static aggregate capacity for this executable shape.

    """

    association: Regenie2MultiBinaryScoreChunkResult
    firth_candidate_count: jax.Array
    firth_candidate_capacity: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DecodedMultiBinaryScoreChunkResult:
    """Packed8 score result retaining its decoded device genotypes.

    Attributes:
        genotype_matrix_by_variant: Decoded variant-major dosage matrix.
        score_result: Trait-major score-test result.

    """

    genotype_matrix_by_variant: jax.Array
    score_result: Regenie2MultiBinaryScoreChunkResult
