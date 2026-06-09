"""Compatibility wrappers for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import functools
import typing

import g._core
from g import types

if typing.TYPE_CHECKING:
    from pathlib import Path

InputConfig = g._core.InputConfig
TraitConfig = g._core.TraitConfig
BinaryConfig = g._core.BinaryConfig
GComputeConfig = g._core.GComputeConfig
GOutputConfig = g._core.GOutputConfig
GDiagnosticsConfig = g._core.GDiagnosticsConfig
RegenieConfig = g._core.RegenieConfig

QUANTITATIVE_BINARY_ONLY_OPTION_NAMES = ("firth", "approx", "firth-se", "pThresh")


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
    if config.is_validated:
        return
    g._core.validate_regenie_config(config)


def write_toml(config: RegenieConfig, path: Path | str) -> None:
    """Write a deterministic TOML file."""
    g._core.write_config_toml(config, path)


def dumps_toml(config: RegenieConfig) -> str:
    """Serialize a configuration to TOML."""
    return g._core.dumps_config_toml(config)


def explain_option(name: str) -> str:
    """Return the Rust-owned explanation for a config option."""
    return g._core.explain_config_option(name)


def iter_explanations() -> tuple[str, ...]:
    """Return Rust-owned explanations for all config options."""
    return tuple(g._core.iter_config_explanations())


def decode_toml_mapping(toml_data: bytes | str, *, source: str) -> dict[str, typing.Any]:
    """Decode TOML into built-in containers through Rust."""
    return g._core.decode_config_toml_mapping(toml_data, source)


def flatten_toml_mapping(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-shaped sections into canonical option names through Rust."""
    return g._core.flatten_config_toml_mapping(raw_options)


def normalize_option_name(option_name: str) -> str:
    """Map Pythonic names to canonical option names through Rust."""
    return g._core.normalize_config_option_name(option_name)


def normalize_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize option names through Rust."""
    return g._core.normalize_config_option_dictionary(raw_options)
