"""Enumerated types for configuration and mode selection."""

import enum
from dataclasses import dataclass


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
        p_threshold: Score-test p-value threshold for fallback candidates.
        firth_se: Whether successful Firth rows use LRT-derived standard errors.

    """

    p_threshold: float
    firth_se: bool
