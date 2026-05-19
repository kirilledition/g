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
        "BinaryConfig",
        "GComputeConfig",
        "GDiagnosticsConfig",
        "GOutputConfig",
        "InputConfig",
        "RegenieConfig",
        "RunArtifacts",
        "TraitConfig",
        "regenie",
    }:
        api = importlib.import_module("g.api")
        return getattr(api, name)
    if name in {
        "ArrayMemoryOrder",
        "AssociationMode",
        "Device",
        "JaxMatmulPrecision",
        "OutputFormat",
        "ArrowCompression",
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
    "ArrowCompression",
    "AssociationMode",
    "BinaryConfig",
    "Device",
    "GComputeConfig",
    "GDiagnosticsConfig",
    "GOutputConfig",
    "InputConfig",
    "JaxMatmulPrecision",
    "OutputFormat",
    "RegenieConfig",
    "RegenieTraitType",
    "RunArtifacts",
    "SampleIdentifierSource",
    "SampleKeyMode",
    "TraitConfig",
    "main",
    "regenie",
]
