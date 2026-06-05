"""Typed configuration layer helpers."""

from __future__ import annotations

import dataclasses
import enum
import os
import typing
from dataclasses import dataclass

import msgspec
import msgspec.inspect

from g import types
from g.interface import options, toml_schema

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class TomlConfigLayer:
    """One typed configuration layer and the options explicitly supplied by it.

    Attributes:
        toml_config: Typed partial TOML config.
        explicit_options: Canonical option names present in this layer.

    """

    toml_config: toml_schema.TomlConfig
    explicit_options: frozenset[str] = dataclasses.field(default_factory=frozenset)


def empty_toml_config() -> toml_schema.TomlConfig:
    """Return an empty typed config layer."""
    return toml_schema.TomlConfig()


def empty_toml_config_layer() -> TomlConfigLayer:
    """Return an empty typed config layer with no explicit options."""
    return TomlConfigLayer(toml_config=empty_toml_config())


def decode_toml_bytes(toml_data: bytes | str, *, source: str) -> toml_schema.TomlConfig:
    """Decode TOML bytes into the typed config schema.

    Args:
        toml_data: TOML document bytes or text.
        source: Human-readable source name for error messages.

    Returns:
        Decoded TOML config.

    Raises:
        ValueError: If parsing or schema validation fails.

    """
    try:
        return msgspec.toml.decode(toml_data, type=toml_schema.TomlConfig, strict=True)
    except msgspec.DecodeError as error:
        message = f"Invalid TOML config {source}: {error}"
        raise ValueError(message) from error


def decode_toml_file(path: Path) -> toml_schema.TomlConfig:
    """Decode a TOML file into the typed config schema."""
    return decode_toml_bytes(path.read_bytes(), source=str(path))


def decode_toml_file_layer(path: Path | None) -> TomlConfigLayer:
    """Decode an optional TOML file into a typed explicit config layer."""
    if path is None:
        return empty_toml_config_layer()
    toml_config = decode_toml_file(path)
    explicit_options = frozenset(toml_config_to_option_dictionary(toml_config))
    return TomlConfigLayer(toml_config=toml_config, explicit_options=explicit_options)


def toml_config_to_builtin_mapping(toml_config: toml_schema.TomlConfig) -> dict[str, typing.Any]:
    """Convert a typed TOML config to built-in TOML-shaped containers."""
    return typing.cast("dict[str, typing.Any]", msgspec.to_builtins(toml_config))


def convert_toml_mapping(
    toml_mapping: typing.Mapping[str, typing.Any],
    *,
    source: str,
) -> toml_schema.TomlConfig:
    """Convert a TOML-shaped mapping into the typed config schema.

    Args:
        toml_mapping: TOML-shaped mapping.
        source: Human-readable source name for error messages.

    Returns:
        Typed TOML config.

    Raises:
        ValueError: If the mapping does not match the TOML schema.

    """
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
    merged_config = typing.cast(
        "toml_schema.TomlConfig",
        overlay_struct_values(base_config, override_config),
    )
    return apply_trait_flag_overlay_precedence(merged_config, override_config)


def overlay_struct_values(
    base_value: msgspec.Struct,
    override_value: msgspec.Struct,
) -> msgspec.Struct:
    """Overlay one msgspec struct over another, ignoring unset override fields."""
    struct_values: dict[str, typing.Any] = {}
    struct_information = typing.cast("msgspec.inspect.StructType", msgspec.inspect.type_info(type(base_value)))
    for field_information in struct_information.fields:
        base_field_value = getattr(base_value, field_information.name)
        override_field_value = getattr(override_value, field_information.name)
        if override_field_value is msgspec.UNSET:
            struct_values[field_information.name] = base_field_value
        elif isinstance(base_field_value, msgspec.Struct) and isinstance(override_field_value, msgspec.Struct):
            struct_values[field_information.name] = overlay_struct_values(base_field_value, override_field_value)
        else:
            struct_values[field_information.name] = override_field_value
    return type(base_value)(**struct_values)


def apply_trait_flag_overlay_precedence(
    merged_config: toml_schema.TomlConfig,
    override_config: toml_schema.TomlConfig,
) -> toml_schema.TomlConfig:
    """Apply cross-layer trait flag exclusivity after a typed overlay."""
    override_trait_section = override_config.trait
    merged_trait_section = merged_config.trait
    if override_trait_section is msgspec.UNSET or merged_trait_section is msgspec.UNSET:
        return merged_config

    trait_updates: dict[str, typing.Any] = {}
    if override_trait_section.qt is True:
        trait_updates["bt"] = False
    if override_trait_section.bt is True:
        trait_updates["qt"] = False
    if not trait_updates:
        return merged_config

    updated_trait_section = replace_struct_values(merged_trait_section, trait_updates)
    return typing.cast(
        "toml_schema.TomlConfig",
        replace_struct_values(merged_config, {"trait": updated_trait_section}),
    )


def replace_struct_values(
    struct_value: msgspec.Struct,
    updated_values: typing.Mapping[str, typing.Any],
) -> msgspec.Struct:
    """Return a msgspec struct with selected fields replaced."""
    struct_values: dict[str, typing.Any] = {}
    struct_information = typing.cast("msgspec.inspect.StructType", msgspec.inspect.type_info(type(struct_value)))
    for field_information in struct_information.fields:
        struct_values[field_information.name] = updated_values.get(
            field_information.name,
            getattr(struct_value, field_information.name),
        )
    return type(struct_value)(**struct_values)


def toml_config_to_option_dictionary(toml_config: toml_schema.TomlConfig) -> dict[str, typing.Any]:
    """Flatten a typed TOML config to canonical option names."""
    return flatten_toml_mapping(toml_config_to_builtin_mapping(toml_config))


def flatten_toml_mapping(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML-shaped sections into canonical option names where possible."""
    flattened_options: dict[str, typing.Any] = {}
    config_options = {
        option_name: option_value for option_name, option_value in raw_options.items() if option_name != "metadata"
    }
    for section_name, section_value in config_options.items():
        if isinstance(section_value, dict):
            if section_name == "g":
                flattened_options.update(flatten_g_toml_section(section_value))
            else:
                flattened_options.update(flatten_toml_section(section_name, section_value))
        else:
            flattened_options[section_name] = section_value
    return flattened_options


def flatten_g_toml_section(raw_g_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten TOML tables below the reserved [g.*] namespace."""
    flattened_options: dict[str, typing.Any] = {}
    for section_name, section_value in raw_g_options.items():
        if isinstance(section_value, dict):
            flattened_options.update(flatten_toml_section(f"g.{section_name}", section_value))
        else:
            flattened_options[f"g.{section_name}"] = section_value
    return flattened_options


def flatten_toml_section(section_name: str, section_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Flatten one TOML section through the option registry."""
    flattened_options: dict[str, typing.Any] = {}
    for toml_key, option_value in section_options.items():
        option_spec = options.OPTION_SPEC_BY_TOML_PATH.get((section_name, toml_key))
        if option_spec is None:
            flattened_options[f"{section_name}.{toml_key}"] = option_value
        else:
            flattened_options[option_spec.name] = option_value
    return flattened_options


def option_dictionary_to_toml_config_layer(
    raw_options: typing.Mapping[str, typing.Any],
    *,
    source: str,
) -> TomlConfigLayer:
    """Convert Python or CLI options into a typed partial TOML layer."""
    normalized_options = normalize_option_dictionary(raw_options)
    toml_mapping: dict[str, typing.Any] = {}
    for option_name, option_value in normalized_options.items():
        if option_value is None or option_name == "trait_type":
            continue
        option_spec = options.OPTION_SPEC_BY_NAME.get(option_name)
        if option_spec is None:
            message = f"Unknown g regenie option: {option_name}"
            raise ValueError(message)
        set_toml_option_value(toml_mapping, option_spec, coerce_option_value(option_value, option_spec))

    apply_trait_type_alias(toml_mapping, normalized_options.get("trait_type"))
    apply_explicit_trait_flag_precedence(toml_mapping, normalized_options)
    toml_config = convert_toml_mapping(toml_mapping, source=source)
    return TomlConfigLayer(
        toml_config=toml_config,
        explicit_options=frozenset(normalized_options),
    )


def normalize_option_dictionary(raw_options: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize snake-case aliases and nested dictionaries into canonical names."""
    normalized_options: dict[str, typing.Any] = {}
    for option_name, option_value in flatten_toml_mapping(raw_options).items():
        normalized_options[normalize_option_name(option_name)] = option_value
    return normalized_options


def normalize_option_name(option_name: str) -> str:
    """Map Pythonic names to canonical option names."""
    if option_name == "trait_type":
        return option_name
    if option_name in options.OPTION_SPEC_BY_NAME:
        return option_name
    destination_option_spec = options.OPTION_SPEC_BY_DESTINATION.get(option_name)
    if destination_option_spec is not None:
        return destination_option_spec.name
    alias_option_spec = options.OPTION_SPEC_BY_PYTHON_ALIAS.get(option_name)
    if alias_option_spec is not None:
        return alias_option_spec.name
    if option_name.startswith("g_"):
        return option_name.replace("_", "-")
    return option_name


def coerce_option_value(option_value: typing.Any, option_spec: options.OptionSpec) -> typing.Any:
    """Coerce Python and CLI option values to TOML-compatible schema values."""
    if option_spec.multiple:
        return coerce_string_list_value(option_value)
    if isinstance(option_value, enum.Enum):
        return option_value.value
    if option_spec.type == options.OptionValueType.PATH:
        return os.fspath(option_value)
    if option_spec.type == options.OptionValueType.INTEGER:
        return int(option_value)
    if option_spec.type == options.OptionValueType.FLOAT:
        return float(option_value)
    if option_spec.type == options.OptionValueType.BOOLEAN:
        return bool(option_value)
    if isinstance(option_value, list | tuple):
        return [str(item_value) for item_value in option_value]
    return str(option_value)


def coerce_string_list_value(option_value: typing.Any) -> str | list[str]:
    """Coerce repeated string options to TOML-compatible values."""
    if isinstance(option_value, str):
        return option_value
    return [str(item_value) for item_value in option_value]


def set_toml_option_value(
    toml_mapping: dict[str, typing.Any],
    option_spec: options.OptionSpec,
    option_value: typing.Any,
) -> None:
    """Set one canonical option value in a TOML-shaped mapping."""
    toml_key = typing.cast("str", option_spec.toml_key)
    if "." not in option_spec.section:
        section_mapping = toml_mapping.setdefault(option_spec.section, {})
        typing.cast("dict[str, typing.Any]", section_mapping)[toml_key] = option_value
        return
    namespace_name, section_name = option_spec.section.split(".", maxsplit=1)
    namespace_mapping = toml_mapping.setdefault(namespace_name, {})
    section_mapping = typing.cast("dict[str, typing.Any]", namespace_mapping).setdefault(section_name, {})
    typing.cast("dict[str, typing.Any]", section_mapping)[toml_key] = option_value


def apply_trait_type_alias(
    toml_mapping: dict[str, typing.Any],
    raw_trait_type: typing.Any,
) -> None:
    """Apply the Python-only trait_type alias to the TOML-shaped trait section."""
    if raw_trait_type is None:
        return
    trait_type = types.RegenieTraitType(str(raw_trait_type))
    trait_mapping = typing.cast("dict[str, typing.Any]", toml_mapping.setdefault("trait", {}))
    trait_mapping["qt"] = trait_type == types.RegenieTraitType.QUANTITATIVE
    trait_mapping["bt"] = trait_type == types.RegenieTraitType.BINARY


def apply_explicit_trait_flag_precedence(
    toml_mapping: dict[str, typing.Any],
    normalized_options: typing.Mapping[str, typing.Any],
) -> None:
    """Preserve the CLI trait flag precedence rules for typed option layers."""
    trait_mapping = typing.cast("dict[str, typing.Any]", toml_mapping.setdefault("trait", {}))
    if normalized_options.get("qt") is True:
        trait_mapping["bt"] = False
    if normalized_options.get("bt") is True:
        trait_mapping["qt"] = False
