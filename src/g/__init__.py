"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Resolve public package attributes lazily to keep plain ``import g`` lightweight."""
    if name == "main":
        from g import cli

        return cli.main
    if name == "RuntimeState":
        from g import api

        return api.RuntimeState
    if name == "RunArtifacts":
        from g import api

        return api.RunArtifacts
    if name == "describe_runtime_state":
        from g import api

        return api.describe_runtime_state
    if name == "regenie":
        from g import api

        return api.regenie
    if name == "ArrayMemoryOrder":
        from g import types

        return types.ArrayMemoryOrder
    if name == "AssociationMode":
        from g import types

        return types.AssociationMode
    if name == "Device":
        from g import types

        return types.Device
    if name == "JaxMatmulPrecision":
        from g import types

        return types.JaxMatmulPrecision
    if name == "OutputFormat":
        from g import types

        return types.OutputFormat
    if name == "ArrowCompression":
        from g import types

        return types.ArrowCompression
    if name == "RegenieTraitType":
        from g import types

        return types.RegenieTraitType
    if name == "SampleKeyMode":
        from g import types

        return types.SampleKeyMode
    if name == "SampleIdentifierSource":
        from g import types

        return types.SampleIdentifierSource
    message = f"module 'g' has no attribute {name!r}"
    raise AttributeError(message)
