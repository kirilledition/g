"""Compute-domain association chunk results."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import jax

    from g import models


@dataclass(frozen=True)
class AssociationChunkResult:
    """Device-resident association chunk result used by the output layer."""

    metadata: models.VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None
