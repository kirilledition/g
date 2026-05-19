"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

import importlib
import typing


def __getattr__(name: str) -> typing.Any:
    """Resolve public package attributes lazily to avoid eager heavy imports."""
    if name == "main":
        cli = importlib.import_module("g.cli")
        return cli.main
    if name in {
        "ComputeConfig",
        "Regenie2BinaryConfig",
        "Regenie2LinearConfig",
        "RunArtifacts",
        "SampleAlignmentConfig",
        "regenie2",
        "regenie2_linear",
    }:
        api = importlib.import_module("g.api")
        return getattr(api, name)
    if name in {
        "ArrayMemoryOrder",
        "AssociationMode",
        "Device",
        "RegenieTraitType",
        "SampleKeyMode",
        "SampleIdentifierSource",
    }:
        types = importlib.import_module("g.types")
        return getattr(types, name)
    message = f"module 'g' has no attribute {name!r}"
    raise AttributeError(message)


__all__ = [
    "ArrayMemoryOrder",
    "AssociationMode",
    "ComputeConfig",
    "Device",
    "Regenie2BinaryConfig",
    "Regenie2LinearConfig",
    "RegenieTraitType",
    "RunArtifacts",
    "SampleAlignmentConfig",
    "SampleIdentifierSource",
    "SampleKeyMode",
    "main",
    "regenie2",
    "regenie2_linear",
]
