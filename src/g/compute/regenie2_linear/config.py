"""Default linear REGENIE step 2 kernel policy."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LINEAR_MINIMUM_VARIANCE = 1.0e-8
DEFAULT_LINEAR_RELATIVE_VARIANCE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class LinearNumericalConfig:
    """Shared linear numerical floors and tolerances.

    Attributes:
        minimum_variance: Residualized genotype variance floor.
        relative_variance_tolerance: Relative residualized genotype variance floor multiplier.

    """

    minimum_variance: float = DEFAULT_LINEAR_MINIMUM_VARIANCE
    relative_variance_tolerance: float = DEFAULT_LINEAR_RELATIVE_VARIANCE_TOLERANCE

    def __post_init__(self) -> None:
        """Validate positive numerical settings."""
        if self.minimum_variance <= 0.0:
            message = "Minimum variance must be positive."
            raise ValueError(message)
        if self.relative_variance_tolerance <= 0.0:
            message = "Relative variance tolerance must be positive."
            raise ValueError(message)


DEFAULT_LINEAR_NUMERICAL_CONFIG = LinearNumericalConfig()
