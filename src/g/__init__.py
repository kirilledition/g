"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

import typing

from g import api, cli, types


def __getattr__(name: str) -> typing.Any:
    """Resolve public package attributes lazily to avoid eager heavy imports."""
    if name == "main":
        return cli.main
    if name in {
        "ComputeConfig",
        "Regenie2BinaryConfig",
        "Regenie2LinearConfig",
        "RunArtifacts",
        "regenie2",
        "regenie2_linear",
    }:
        return getattr(api, name)
    if name in {
        "ArrayMemoryOrder",
        "AssociationMode",
        "Device",
        "GenotypeSourceFormat",
        "OutputWriterBackend",
        "RegenieBinaryCorrection",
        "RegenieTraitType",
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
    "OutputWriterBackend",
    "Regenie2BinaryConfig",
    "Regenie2LinearConfig",
    "RegenieBinaryCorrection",
    "RegenieTraitType",
    "RunArtifacts",
    "SampleIdentifierSource",
    "main",
    "regenie2",
    "regenie2_linear",
]
