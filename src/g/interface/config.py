"""Compatibility wrappers for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import functools
import os
import typing
from pathlib import Path

import g._core
from g import types

InputConfig = g._core.InputConfig
TraitConfig = g._core.TraitConfig
BinaryConfig = g._core.BinaryConfig
GComputeConfig = g._core.GComputeConfig
GOutputConfig = g._core.GOutputConfig
GDiagnosticsConfig = g._core.GDiagnosticsConfig
RegenieConfig = g._core.RegenieConfig

QUANTITATIVE_BINARY_ONLY_OPTION_NAMES = ("firth", "approx", "firth-se", "spa", "pThresh")


def split_name_list(raw_names: str | typing.Iterable[str] | None) -> tuple[str, ...]:
    """Normalize comma-delimited or iterable column names."""
    if raw_names is None:
        return ()
    if isinstance(raw_names, str):
        return tuple(stripped_name for name in raw_names.split(",") if (stripped_name := name.strip()))
    return tuple(stripped_name for name in raw_names if (stripped_name := str(name).strip()))


def optional_string(raw_value: typing.Any) -> str | None:
    """Convert an optional string value."""
    if raw_value is None:
        return None
    return str(raw_value)


def normalize_trait_type(*, qt: bool | None, bt: bool | None) -> types.RegenieTraitType:
    """Resolve REGENIE trait flags into one trait type."""
    if qt and bt:
        message = "--qt and --bt are mutually exclusive."
        raise ValueError(message)
    if bt:
        return types.RegenieTraitType.BINARY
    return types.RegenieTraitType.QUANTITATIVE


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from CLI/TOML/Python option dictionaries."""
    return g._core.config_from_options(raw_options)


@functools.cache
def load_packaged_config() -> RegenieConfig:
    """Load packaged TOML defaults as a complete unvalidated runtime config."""
    return g._core.load_packaged_config()


def load_toml(path: Path) -> RegenieConfig:
    """Load a configuration from a TOML file."""
    return g._core.config_from_toml(path)


def validate_config(config: RegenieConfig) -> None:
    """Validate a complete normalized config."""
    g._core.validate_regenie_config(config)


def write_toml(config: RegenieConfig, path: Path | str) -> None:
    """Write a deterministic TOML file."""
    g._core.write_config_toml(config, path)


def dumps_toml(config: RegenieConfig) -> str:
    """Serialize a configuration to TOML."""
    return g._core.dumps_config_toml(config)


def build_template() -> str:
    """Return a starter config with placeholders and packaged defaults."""
    return g._core.build_config_template()


def validate_positive_integer(option_name: str, value: int) -> None:
    """Validate that an integer config value is positive."""
    if value <= 0:
        message = f"{option_name} must be positive."
        raise ValueError(message)


def validate_non_negative_integer(option_name: str, value: int) -> None:
    """Validate that an integer config value is non-negative."""
    if value < 0:
        message = f"{option_name} must be non-negative."
        raise ValueError(message)


def validate_positive_float(option_name: str, value: float) -> None:
    """Validate that a floating-point config value is positive."""
    if value <= 0.0:
        message = f"{option_name} must be positive."
        raise ValueError(message)


def validate_probability_floor(option_name: str, value: float) -> None:
    """Validate that a probability floor remains below a symmetric midpoint."""
    validate_positive_float(option_name, value)
    if value >= 0.5:
        message = f"{option_name} must be less than 0.5."
        raise ValueError(message)


def format_toml_value(value: typing.Any) -> str:
    """Format one TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Path):
        return format_toml_string(os.fspath(value))
    if isinstance(value, list | tuple):
        return f"[{', '.join(format_toml_value(item_value) for item_value in value)}]"
    return format_toml_string(str(value))


def format_toml_string(value: str) -> str:
    """Format a TOML basic string."""
    escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_value}"'
