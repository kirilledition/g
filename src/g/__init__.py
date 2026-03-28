"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

from g.api import RunArtifacts, linear, logistic
from g.cli import main
from g.config import ComputeConfig, LinearConfig, LogisticConfig

__all__ = [
    "ComputeConfig",
    "LinearConfig",
    "LogisticConfig",
    "RunArtifacts",
    "linear",
    "logistic",
    "main",
]
