"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

from g.api import ComputeConfig, LinearConfig, LogisticConfig, RunArtifacts, linear, logistic
from g.cli import main

__all__ = [
    "ComputeConfig",
    "LinearConfig",
    "LogisticConfig",
    "RunArtifacts",
    "linear",
    "logistic",
    "main",
]
