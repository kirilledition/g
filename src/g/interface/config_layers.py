"""Compatibility wrappers for Rust-owned TOML configuration layers."""

from __future__ import annotations

import dataclasses
import functools
import typing
from dataclasses import dataclass

import msgspec
import msgspec.inspect

import g._core
from g.interface import toml_schema

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class TomlConfigLayer:
    """One TOML-shaped configuration layer and its explicit option names."""

    toml_config: toml_schema.TomlConfig
    explicit_options: frozenset[str] = dataclasses.field(default_factory=frozenset)


def decode_toml_bytes(toml_data: bytes | str, *, source: str) -> toml_schema.TomlConfig:
    """Decode TOML bytes into the legacy typed config schema."""
    return convert_toml_mapping(decode_toml_builtin_mapping(toml_data, source=source), source=source)


def decode_toml_builtin_mapping(toml_data: bytes | str, *, source: str) -> dict[str, typing.Any]:
    """Decode TOML bytes into built-in containers through Rust."""
    return g._core.decode_config_toml_mapping(toml_data, source)


def decode_toml_file(path: Path) -> toml_schema.TomlConfig:
    """Decode a TOML file into the legacy typed config schema."""
    return decode_toml_bytes(path.read_bytes(), source=str(path))


def decode_toml_file_layer(path: Path | None) -> TomlConfigLayer:
    """Decode an optional TOML file into a typed explicit config layer."""
    if path is None:
        return TomlConfigLayer(toml_config=toml_schema.TomlConfig())
    raw_toml = decode_toml_builtin_mapping(path.read_bytes(), source=str(path))
    return TomlConfigLayer(
        toml_config=convert_toml_mapping(raw_toml, source=str(path)),
        explicit_options=frozenset(flatten_toml_mapping(raw_toml)),
    )


def toml_config_to_builtin_mapping(toml_config: toml_schema.TomlConfig) -> dict[str, typing.Any]:
    """Convert a typed TOML config to built-in TOML-shaped containers."""
    return typing.cast("dict[str, typing.Any]", msgspec.to_builtins(toml_config))


def convert_toml_mapping(
    toml_mapping: typing.Mapping[str, typing.Any],
    *,
    source: str,
) -> toml_schema.TomlConfig:
    """Convert a TOML-shaped mapping into the legacy typed config schema."""
    try:
        return msgspec.convert(toml_mapping, type=toml_schema.TomlConfig, strict=True)
    except msgspec.ValidationError as error:
        message = f"Invalid config layer {source}: {error}"
        raise ValueError(message) from error


def overlay_toml_configs(
    base_config: toml_schema.TomlConfig,
    override_config: toml_schema.TomlConfig,
) -> toml_schema.TomlConfig:
    """Overlay one typed config over another, ignoring unset override fields."""
    return typing.cast(
        "toml_schema.TomlConfig",
        overlay_struct_values(base_config, override_config),
    )


def overlay_struct_values(
    base_value: msgspec.Struct,
    override_value: msgspec.Struct,
) -> msgspec.Struct:
    """Overlay one msgspec struct over another, ignoring unset override fields."""
    struct_values: dict[str, typing.Any] = {}
    for field_name in struct_field_names(type(base_value)):
        base_field_value = getattr(base_value, field_name)
        override_field_value = getattr(override_value, field_name)
        if override_field_value is msgspec.UNSET:
            struct_values[field_name] = base_field_value
        elif isinstance(base_field_value, msgspec.Struct) and isinstance(override_field_value, msgspec.Struct):
            struct_values[field_name] = overlay_struct_values(base_field_value, override_field_value)
        else:
            struct_values[field_name] = override_field_value
    return type(base_value)(**struct_values)


@functools.cache
def struct_field_names(struct_type: type[msgspec.Struct]) -> tuple[str, ...]:
    """Return cached msgspec struct field names."""
    struct_information = typing.cast("msgspec.inspect.StructType", msgspec.inspect.type_info(struct_type))
    return tuple(field_information.name for field_information in struct_information.fields)


def replace_struct_values(
    struct_value: msgspec.Struct,
    updated_values: typing.Mapping[str, typing.Any],
) -> msgspec.Struct:
    """Return a msgspec struct with selected fields replaced."""
    struct_values: dict[str, typing.Any] = {}
    for field_name in struct_field_names(type(struct_value)):
        struct_values[field_name] = updated_values.get(field_name, getattr(struct_value, field_name))
    return type(struct_value)(**struct_values)


def toml_config_to_option_dictionary(toml_config: toml_schema.TomlConfig) -> dict[str, typing.Any]:
    """Flatten a typed TOML config to canonical option names."""
    return flatten_toml_mapping(toml_config_to_builtin_mapping(toml_config))


def flatten_toml_mapping(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-shaped sections into canonical option names through Rust."""
    return g._core.flatten_config_toml_mapping(raw_options)


def option_dictionary_to_toml_config_layer(
    raw_options: typing.Mapping[str, typing.Any],
    *,
    source: str,
) -> TomlConfigLayer:
    """Convert Python or CLI options into a typed partial TOML layer through Rust."""
    raw_layer = g._core.option_dictionary_to_config_toml_layer(raw_options, source)
    raw_toml_config = typing.cast("dict[str, typing.Any]", raw_layer["toml_config"])
    explicit_options = typing.cast("typing.Iterable[str]", raw_layer["explicit_options"])
    return TomlConfigLayer(
        toml_config=convert_toml_mapping(raw_toml_config, source=source),
        explicit_options=frozenset(explicit_options),
    )


def normalize_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize snake-case aliases and nested dictionaries into canonical names through Rust."""
    return g._core.normalize_config_option_dictionary(raw_options)


def normalize_option_name(option_name: str) -> str:
    """Map Pythonic names to canonical option names through Rust."""
    return g._core.normalize_config_option_name(option_name)
