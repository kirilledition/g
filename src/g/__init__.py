"""Python entrypoints for the GWAS engine package."""

from __future__ import annotations

import importlib
import typing


def __getattr__(name: str) -> typing.Any:
    """Resolve public package attributes lazily to keep plain ``import g`` lightweight."""
    if name == "main":
        cli = importlib.import_module("g.cli")
        return cli.main
    if name in {
        "RunArtifacts",
        "regenie",
    }:
        api = importlib.import_module("g.api")
        return getattr(api, name)
    if name in {
        "AssociationMode",
        "Device",
        "JaxMatmulPrecision",
        "OutputFormat",
        "ArrowCompression",
        "RegenieTraitType",
        "SampleKeyMode",
    }:
        types = importlib.import_module("g.types")
        return getattr(types, name)
    message = f"module 'g' has no attribute {name!r}"
    raise AttributeError(message)
