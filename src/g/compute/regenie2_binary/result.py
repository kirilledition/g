"""Binary association result type for REGENIE step 2 compute."""

from __future__ import annotations

import jax

from g.compute.common import result as association_result

type Regenie2MultiBinaryScoreChunkResult = association_result.AssociationResult[jax.Array, jax.Array]
