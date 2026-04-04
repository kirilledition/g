"""Enumerated types for configuration and mode selection."""

import enum


class Device(enum.StrEnum):
    """JAX execution device."""

    CPU = "cpu"
    GPU = "gpu"


class OutputMode(enum.StrEnum):
    """Output format mode."""

    TSV = "tsv"
    ARROW_CHUNKS = "arrow_chunks"


class AssociationMode(enum.StrEnum):
    """Statistical association model."""

    LINEAR = "linear"
    LOGISTIC = "logistic"
    REGENIE2_LINEAR = "regenie2_linear"


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
