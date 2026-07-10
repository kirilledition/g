"""Enumerated types for configuration and mode selection."""

import enum
from dataclasses import dataclass


class AssociationMode(enum.StrEnum):
    """Statistical association model."""

    REGENIE2_LINEAR = "regenie2_linear"
    REGENIE2_BINARY = "regenie2_binary"


class FloatingPointDtype(enum.StrEnum):
    """Floating-point dtype selector for JAX compute kernels."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class BinaryFallbackMethod(enum.StrEnum):
    """Internal binary fallback method."""

    SCORE_ONLY = "score_only"
    FIRTH_APPROXIMATE = "firth_approximate"


class BinaryCorrectionCode(enum.IntEnum):
    """Binary association correction method and outcome."""

    SCORE_SUCCESS = 0
    SCORE_FAILED = 1
    FIRTH_SUCCESS = 2
    FIRTH_FAILED = 3


@dataclass(frozen=True)
class BinaryCorrectionPlan:
    """Normalized binary fallback execution plan.

    Attributes:
        method: Binary fallback method to run.
        p_threshold: Score-test p-value threshold for fallback candidates.
        firth_se: Whether successful Firth rows use LRT-derived standard errors.

    """

    method: BinaryFallbackMethod
    p_threshold: float
    firth_se: bool
