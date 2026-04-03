"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from g.api import ComputeConfig, LinearConfig, LogisticConfig, RunArtifacts


def __getattr__(name: str) -> Any:
    """Resolve public package attributes lazily to avoid eager heavy imports."""
    if name == "main":
        from g.cli import main

        return main
    if name in {"ComputeConfig", "LinearConfig", "LogisticConfig", "RunArtifacts", "linear", "logistic"}:
        from g import api

        return getattr(api, name)
    message = f"module 'g' has no attribute {name!r}"
    raise AttributeError(message)


__all__ = [
    "ComputeConfig",
    "LinearConfig",
    "LogisticConfig",
    "RunArtifacts",
    "linear",
    "logistic",
    "main",
]
