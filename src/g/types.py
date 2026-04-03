"""Enumerated types for configuration and mode selection."""

from enum import StrEnum


class Device(StrEnum):
    """JAX execution device."""

    CPU = "cpu"
    GPU = "gpu"


class OutputMode(StrEnum):
    """Output format mode."""

    TSV = "tsv"
    ARROW_CHUNKS = "arrow_chunks"


class AssociationMode(StrEnum):
    """Statistical association model."""

    LINEAR = "linear"
    LOGISTIC = "logistic"


class GenotypeSourceFormat(StrEnum):
    """Supported genotype file formats."""

    PLINK = "plink"
    BGEN = "bgen"


class SampleIdentifierSource(StrEnum):
    """Origin of BGEN sample identifiers."""

    EMBEDDED = "embedded"
    EXTERNAL = "external"
    GENERATED = "generated"


class ArrayMemoryOrder(StrEnum):
    """NumPy array memory layout selector."""

    KEEP = "K"
    ANY = "A"
    C_CONTIGUOUS = "C"
    FORTRAN_CONTIGUOUS = "F"
