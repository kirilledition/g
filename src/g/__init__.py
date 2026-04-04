"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

import typing

from g import api, cli, types


def __getattr__(name: str) -> typing.Any:
    """Resolve public package attributes lazily to avoid eager heavy imports."""
    if name == "main":
        return cli.main
    if name in {"ComputeConfig", "LinearConfig", "LogisticConfig", "RunArtifacts", "linear", "logistic"}:
        return getattr(api, name)
    if name in {
        "ArrayMemoryOrder",
        "AssociationMode",
        "Device",
        "GenotypeSourceFormat",
        "OutputMode",
        "SampleIdentifierSource",
    }:
        return getattr(types, name)
    message = f"module 'g' has no attribute {name!r}"
    raise AttributeError(message)


__all__ = [
    "ArrayMemoryOrder",
    "AssociationMode",
    "ComputeConfig",
    "Device",
    "GenotypeSourceFormat",
    "LinearConfig",
    "LogisticConfig",
    "OutputMode",
    "RunArtifacts",
    "SampleIdentifierSource",
    "linear",
    "logistic",
    "main",
]
