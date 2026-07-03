"""Thin Python boundary for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass

import g._core

if typing.TYPE_CHECKING:
    from pathlib import Path

InputConfig = g._core.InputConfig
TraitConfig = g._core.TraitConfig
BinaryConfig = g._core.BinaryConfig
GComputeConfig = g._core.GComputeConfig
GOutputConfig = g._core.GOutputConfig
GDiagnosticsConfig = g._core.GDiagnosticsConfig
RegenieConfig = g._core.RegenieConfig


@dataclass(frozen=True)
class FlatOptionTarget:
    """Native config target for one flat Python option name.

    Attributes:
        section_name: Native TOML section name.
        option_name: Native TOML option name inside the section.

    """

    section_name: str
    option_name: str


def build_flat_option_sections() -> dict[str, FlatOptionTarget]:
    """Build Python flat-option targets from Rust-owned metadata."""
    flat_option_sections: dict[str, FlatOptionTarget] = {}
    for option_metadata in g._core.config_option_schema():
        section_name = option_metadata["section"]
        toml_name = option_metadata["toml_name"]
        for python_name in option_metadata["flat_python_names"]:
            flat_option_sections[python_name] = FlatOptionTarget(section_name=section_name, option_name=toml_name)
    return flat_option_sections


def build_boolean_python_options() -> frozenset[str]:
    """Build boolean Python flat-option names from Rust-owned metadata."""
    boolean_option_names: set[str] = set()
    for option_metadata in g._core.config_option_schema():
        if option_metadata["value_kind"] != "boolean":
            continue
        boolean_option_names.update(option_metadata["flat_python_names"])
    return frozenset(boolean_option_names)


FLAT_OPTION_SECTIONS: dict[str, FlatOptionTarget] = build_flat_option_sections()
BOOLEAN_PYTHON_OPTIONS: frozenset[str] = build_boolean_python_options()

NATIVE_CONFIG_SECTION_NAMES = frozenset(("input", "trait", "binary", "compute", "output", "diagnostics", "metadata"))


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from Python option dictionaries."""
    return g._core.config_from_options(raw_options)


typing.cast("typing.Any", RegenieConfig).from_options = staticmethod(from_options)


@functools.cache
def load_packaged_config() -> RegenieConfig:
    """Load packaged TOML defaults as a complete unvalidated runtime config."""
    return g._core.load_packaged_config()


def validate_config_for_run(config: RegenieConfig) -> None:
    """Validate a complete normalized config at the execution boundary."""
    g._core.validate_regenie_config_for_run(config)


def write_toml(config: RegenieConfig, path: Path | str) -> None:
    """Write deterministic TOML."""
    g._core.write_config_toml(config, path)


def dumps_toml(config: RegenieConfig) -> str:
    """Serialize a configuration to TOML."""
    return g._core.dumps_config_toml(config)
