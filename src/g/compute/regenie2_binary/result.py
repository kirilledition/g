"""Binary association result type for REGENIE step 2 compute."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from g.compute.common import result as association_result

type Regenie2MultiBinaryScoreChunkResult = association_result.AssociationResult[jax.Array, jax.Array]


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
