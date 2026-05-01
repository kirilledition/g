"""Enumerated types for configuration and mode selection."""

import enum


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


class RegenieBinaryCorrection(enum.StrEnum):
    """Binary step 2 correction mode."""

    FIRTH_APPROXIMATE = "firth_approximate"
    SPA = "spa"


class GenotypeSourceFormat(enum.StrEnum):
    """Supported genotype file formats."""

    PLINK = "plink"
    BGEN = "bgen"


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
