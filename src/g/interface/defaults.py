"""Packaged default configuration catalog."""

from __future__ import annotations

import functools
import hashlib
import importlib.resources
import json
import typing
from dataclasses import dataclass

from g.interface import config_layers, options, toml_schema

DEFAULT_CONFIG_RESOURCE = "config.default.toml"
OPTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DefaultOptionCatalog:
    """Validated packaged defaults.

    Attributes:
        toml_config: Typed packaged TOML config.
        raw_toml: Packaged TOML dictionary.
        normalized_options: Defaults keyed by canonical option name.
        default_config_hash: Stable hash of the packaged defaults.

    """

    toml_config: toml_schema.TomlConfig
    raw_toml: typing.Mapping[str, typing.Any]
    normalized_options: typing.Mapping[str, typing.Any]
    default_config_hash: str


@functools.cache
def load_default_option_catalog() -> DefaultOptionCatalog:
    """Load Rust-owned packaged defaults through the compatibility catalog."""
    ensure_rust_default_config_loads()
    default_toml_bytes = load_default_toml_bytes()
    toml_config = config_layers.decode_toml_bytes(default_toml_bytes, source=DEFAULT_CONFIG_RESOURCE)
    raw_toml = config_layers.decode_toml_builtin_mapping(default_toml_bytes, source=DEFAULT_CONFIG_RESOURCE)
    normalized_options = normalize_default_toml(raw_toml)
    validate_default_catalog(normalized_options)
    return DefaultOptionCatalog(
        toml_config=toml_config,
        raw_toml=raw_toml,
        normalized_options=normalized_options,
        default_config_hash=build_default_config_hash(raw_toml),
    )


def load_default_toml_bytes() -> bytes:
    """Load packaged default TOML file bytes."""
    default_config_resource = importlib.resources.files("g").joinpath(DEFAULT_CONFIG_RESOURCE)
    return default_config_resource.read_bytes()


@functools.cache
def ensure_rust_default_config_loads() -> None:
    """Assert Rust can load the packaged default config."""
    import g._core

    g._core.load_packaged_config()


def normalize_default_toml(raw_toml: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Normalize default TOML paths to canonical option names."""
    normalized_options: dict[str, typing.Any] = {}
    for canonical_name, option_value in config_layers.flatten_toml_mapping(raw_toml).items():
        if canonical_name in normalized_options:
            message = f"Default config contains duplicate default for {canonical_name!r}."
            raise ValueError(message)
        normalized_options[canonical_name] = option_value
    return normalized_options


def validate_default_catalog(normalized_options: typing.Mapping[str, typing.Any]) -> None:
    """Validate packaged default coverage and policy compliance."""
    unknown_option_names = sorted(
        option_name for option_name in normalized_options if option_name not in options.OPTION_SPEC_BY_NAME
    )
    if unknown_option_names:
        formatted_names = ", ".join(unknown_option_names)
        message = f"Default config contains unknown option(s): {formatted_names}."
        raise ValueError(message)

    missing_default_names = sorted(
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy == options.DefaultPolicy.VALUE and option_spec.name not in normalized_options
    )
    if missing_default_names:
        formatted_names = ", ".join(missing_default_names)
        message = f"Default config is missing required default option(s): {formatted_names}."
        raise ValueError(message)

    invalid_default_names = sorted(
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy
        in {
            options.DefaultPolicy.REQUIRED_AT_RUNTIME,
            options.DefaultPolicy.UNSUPPORTED,
            options.DefaultPolicy.DERIVED,
        }
        and option_spec.name in normalized_options
    )
    if invalid_default_names:
        formatted_names = ", ".join(invalid_default_names)
        message = f"Default config contains non-defaultable option(s): {formatted_names}."
        raise ValueError(message)


def build_default_config_hash(raw_toml: typing.Mapping[str, typing.Any]) -> str:
    """Build a stable SHA-256 hash for the packaged default config."""
    embedded_hash = metadata_default_config_hash(raw_toml)
    if embedded_hash is not None:
        return embedded_hash
    normalized_payload = normalize_hash_value(default_hash_payload(raw_toml))
    encoded_payload = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def metadata_default_config_hash(raw_toml: typing.Mapping[str, typing.Any]) -> str | None:
    """Return the Rust-rendered default hash embedded in TOML metadata."""
    raw_metadata = raw_toml.get("metadata")
    if not isinstance(raw_metadata, dict):
        return None
    raw_hash = raw_metadata.get("default-config-hash")
    if isinstance(raw_hash, str):
        return raw_hash
    return None


def default_hash_payload(raw_toml: typing.Mapping[str, typing.Any]) -> typing.Mapping[str, typing.Any]:
    """Return TOML payload fields covered by the default hash."""
    if "metadata" not in raw_toml:
        return raw_toml
    return {
        section_name: section_value for section_name, section_value in raw_toml.items() if section_name != "metadata"
    }


def normalize_hash_value(value: typing.Any) -> typing.Any:
    """Normalize TOML values into a stable JSON-compatible shape."""
    if isinstance(value, dict):
        return {
            str(item_key): normalize_hash_value(item_value)
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [normalize_hash_value(item_value) for item_value in value]
    return value
