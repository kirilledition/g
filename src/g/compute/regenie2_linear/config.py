"""Default linear REGENIE step 2 kernel policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearNumericalConfig:
    """Shared linear numerical floors and tolerances.

    Attributes:
        minimum_variance: Residualized genotype variance floor.
        relative_variance_tolerance: Relative residualized genotype variance floor multiplier.

    """

    minimum_variance: float
    relative_variance_tolerance: float
