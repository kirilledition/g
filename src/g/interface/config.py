"""Thin Python boundary for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import collections.abc
import functools
import typing
from dataclasses import dataclass

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

BOOLEAN_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
BOOLEAN_FALSE_VALUES = frozenset(("0", "false", "no", "off"))

NATIVE_CONFIG_SECTION_NAMES = frozenset(("input", "trait", "binary", "compute", "output", "diagnostics", "metadata"))


def normalize_python_options(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize flat Python option dictionaries into native TOML sections."""
    normalized_options: dict[str, typing.Any] = {}
    for option_name, option_value in raw_options.items():
        option_target = FLAT_OPTION_SECTIONS.get(option_name)
        if option_target is None:
            if option_name in NATIVE_CONFIG_SECTION_NAMES:
                if isinstance(option_value, collections.abc.Mapping):
                    section_options = normalized_options.setdefault(option_name, {})
                    if isinstance(section_options, dict):
                        section_options.update(option_value)
                    else:
                        normalized_options[option_name] = dict(option_value)
                else:
                    normalized_options[option_name] = option_value
                continue
            if isinstance(option_value, collections.abc.Mapping):
                message = f"Unknown g regenie option: {flatten_unknown_option_name(option_name, option_value)}"
            else:
                message = f"Unknown g regenie option: {option_name}"
            raise ValueError(message)
        if option_value is None:
            message = f"Option {option_name} does not accept None; omit the key to leave it unset."
            raise ValueError(message)
        section_options = normalized_options.setdefault(option_target.section_name, {})
        if not isinstance(section_options, dict):
            normalized_options[option_name] = option_value
            continue
        section_options[option_target.option_name] = normalize_python_option_value(option_name, option_value)
    return normalized_options


def normalize_python_option_value(option_name: str, option_value: typing.Any) -> typing.Any:
    """Normalize Python option values before native TOML conversion."""
    if option_name not in BOOLEAN_PYTHON_OPTIONS:
        return option_value
    if isinstance(option_value, bool):
        return option_value
    if isinstance(option_value, str):
        normalized_value = option_value.strip().lower()
        if normalized_value in BOOLEAN_TRUE_VALUES:
            return True
        if normalized_value in BOOLEAN_FALSE_VALUES:
            return False
    message = "Boolean option value must be a bool or one of true/false/on/off/yes/no/1/0."
    raise ValueError(message)


def flatten_unknown_option_name(option_name: str, option_value: collections.abc.Mapping[str, typing.Any]) -> str:
    """Build a dotted option name for unknown nested Python options."""
    if not option_value:
        return option_name
    nested_key = next(iter(option_value))
    nested_value = option_value[nested_key]
    if isinstance(nested_value, collections.abc.Mapping):
        return f"{option_name}.{flatten_unknown_option_name(str(nested_key), nested_value)}"
    return f"{option_name}.{nested_key}"


def split_name_list(value: str | None) -> tuple[str, ...]:
    """Split a comma-delimited REGENIE name list."""
    if value is None:
        return ()
    names: list[str] = []
    for zero_based_index, raw_name in enumerate(value.split(",")):
        name = raw_name.strip()
        if not name:
            message = f"Name list contains an empty entry at position {zero_based_index + 1}."
            raise ValueError(message)
        names.append(name)
    return tuple(names)


def optional_string(value: object | None) -> str | None:
    """Normalize optional string-like config values."""
    if value is None:
        return None
    return str(value)


def normalize_trait_type(
    *,
    qt: bool | None,
    bt: bool | None,
    trait_type: types.RegenieTraitType | str | None,
) -> types.RegenieTraitType:
    """Normalize quantitative/binary trait selectors."""
    if qt is True and bt is True:
        message = "--qt and --bt are mutually exclusive."
        raise ValueError(message)
    if bt is True:
        return types.RegenieTraitType.BINARY
    if qt is True:
        return types.RegenieTraitType.QUANTITATIVE
    if trait_type is None:
        return types.RegenieTraitType.QUANTITATIVE
    return types.RegenieTraitType(trait_type)


def flatten_toml_mapping(raw_mapping: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-shaped config mappings into Python option names."""
    flattened_options: dict[str, typing.Any] = {}
    for key, value in raw_mapping.items():
        if isinstance(value, collections.abc.Mapping):
            flatten_mapping_section(prefix=key, raw_mapping=value, flattened_options=flattened_options)
        else:
            flattened_options[key] = value
    return flattened_options


def flatten_mapping_section(
    *,
    prefix: str,
    raw_mapping: typing.Mapping[str, typing.Any],
    flattened_options: dict[str, typing.Any],
) -> None:
    """Flatten an unknown TOML section using dotted keys."""
    for key, value in raw_mapping.items():
        flattened_key = f"{prefix}.{key}"
        if isinstance(value, collections.abc.Mapping):
            flatten_mapping_section(prefix=flattened_key, raw_mapping=value, flattened_options=flattened_options)
        else:
            flattened_options[flattened_key] = value


def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig:
    """Build a normalized config from Python option dictionaries."""
    return g._core.config_from_options(normalize_python_options(raw_options))


typing.cast("typing.Any", RegenieConfig).from_options = staticmethod(from_options)


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


def validate_config_for_run(config: RegenieConfig) -> None:
    """Validate a complete normalized config at the execution boundary."""
    g._core.validate_regenie_config_for_run(config)


def write_toml(config: RegenieConfig, path: Path | str) -> None:
    """Write deterministic TOML."""
    g._core.write_config_toml(config, path)


def dumps_toml(config: RegenieConfig) -> str:
    """Serialize a configuration to TOML."""
    return g._core.dumps_config_toml(config)
