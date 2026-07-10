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
    FIRTH = "firth"
    FIRTH_APPROXIMATE = "firth_approximate"
    SPA = "spa"


class BinaryExtraCode(enum.IntEnum):
    """Integer correction labels used by binary REGENIE step 2 output."""

    SCORE = 0
    FIRTH = 1
    SPA = 2
    TEST_FAIL = 3


class FirthFailureCode(enum.IntEnum):
    """Integer failure labels for binary Firth fallback rows."""

    NONE = 0
    NUMERICAL = 1
    MAX_ITERATIONS = 2
    INVALID_STATISTIC = 3
    STEP_HALVING = 4


class FirthCorrectionCode(enum.IntEnum):
    """Integer labels for the final binary approximate-Firth branch."""

    NONE = 0
    PSEUDO_FIRTH = 1
    NEWTON_RAPHSON_ZERO_START = 2
    NEWTON_RAPHSON_WARM_START = 3


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
