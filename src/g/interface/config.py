"""Thin Python boundary for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import typing

import g._core

InputConfig = g._core.InputConfig
TraitConfig = g._core.TraitConfig
BinaryConfig = g._core.BinaryConfig
GComputeConfig = g._core.GComputeConfig
GOutputConfig = g._core.GOutputConfig
GDiagnosticsConfig = g._core.GDiagnosticsConfig
RegenieConfig = g._core.RegenieConfig


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from Python option dictionaries."""
    return g._core.config_from_options(raw_options)


typing.cast("typing.Any", RegenieConfig).from_options = staticmethod(from_options)


def validate_config_for_run(config: RegenieConfig) -> None:
    """Validate a complete normalized config at the execution boundary."""
    g._core.validate_regenie_config_for_run(config)
