"""Public configuration models for GWAS execution."""

from __future__ import annotations

import dataclasses

DEFAULT_LINEAR_CHUNK_SIZE = 2048
DEFAULT_LOGISTIC_CHUNK_SIZE = 1024


@dataclasses.dataclass(frozen=True)
class ComputeConfig:
    """Hardware and batching settings shared across association methods."""

    chunk_size: int = DEFAULT_LINEAR_CHUNK_SIZE
    device: str = "cpu"
    variant_limit: int | None = None


@dataclasses.dataclass(frozen=True)
class LogisticConfig:
    """Mathematical settings for logistic regression."""

    max_iterations: int = 50
    tolerance: float = 1.0e-8
    firth_fallback: bool = True


@dataclasses.dataclass(frozen=True)
class LinearConfig:
    """Mathematical settings for linear regression."""
