"""Enumerated types for configuration and mode selection."""

import enum
from dataclasses import dataclass


class Device(enum.StrEnum):
    """JAX execution device."""

    CPU = "cpu"
    GPU = "gpu"


class AssociationMode(enum.StrEnum):
    """Statistical association model."""

    REGENIE2_LINEAR = "regenie2_linear"
    REGENIE2_BINARY = "regenie2_binary"


class RegenieTraitType(enum.StrEnum):
    """REGENIE trait family."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class BinaryFallbackMethod(enum.StrEnum):
    """Internal binary fallback method."""

    SCORE_ONLY = "score_only"
    FIRTH = "firth"
    FIRTH_APPROXIMATE = "firth_approximate"
    SPA = "spa"


@dataclass(frozen=True)
class BinaryCorrectionPlan:
    """Normalized binary fallback execution plan.

    Attributes:
        method: Binary fallback method to run.
        p_threshold: Score-test p-value threshold for fallback candidates.
        firth_se: Whether successful Firth rows use LRT-derived standard errors.

    """

    method: BinaryFallbackMethod = BinaryFallbackMethod.SCORE_ONLY
    p_threshold: float = 0.05
    firth_se: bool = False


class SampleIdentifierSource(enum.StrEnum):
    """Origin of BGEN sample identifiers."""

    EMBEDDED = "embedded"
    EXTERNAL = "external"
    GENERATED = "generated"


class ArrayMemoryOrder(enum.StrEnum):
    """NumPy array memory layout selector."""

    KEEP = "K"
    ANY = "A"
    C_CONTIGUOUS = "C"
    FORTRAN_CONTIGUOUS = "F"
