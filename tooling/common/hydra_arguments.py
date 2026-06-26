"""Helpers for Hydra-backed tooling entrypoints."""

from __future__ import annotations

import collections.abc
import typing
from pathlib import Path

import omegaconf


def tool_config_to_dictionary(config: omegaconf.DictConfig) -> dict[str, typing.Any]:
    """Return the resolved ``tool`` node from a composed Hydra config.

    Args:
        config: Composed Hydra config containing a ``tool`` mapping.

    Returns:
        Plain resolved dictionary for the tool-specific parameters.

    Raises:
        KeyError: If the composed config does not contain a ``tool`` node.

    """
    if "tool" not in config:
        message = "Hydra tooling config must contain a 'tool' node."
        raise KeyError(message)
    return typing.cast(
        "dict[str, typing.Any]",
        omegaconf.OmegaConf.to_container(config.tool, resolve=True),
    )


def comma_join(values: typing.Any) -> str:
    """Serialize Hydra scalar or list values into the legacy comma-list format.

    Args:
        values: Scalar, list, tuple, or ``None`` value from a tool config.

    Returns:
        Comma-separated string representation.

    """
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, collections.abc.Sequence) and not isinstance(values, bytes):
        return ",".join(format_scalar_value(value) for value in values)
    return format_scalar_value(values)


def path_or_none(value: typing.Any) -> Path | None:
    """Convert a resolved config value into an optional path."""
    if value is None:
        return None
    return Path(str(value))


def integer_or_none(value: typing.Any) -> int | None:
    """Convert a resolved config value into an optional integer."""
    if value is None:
        return None
    return int(value)


def float_or_none(value: typing.Any) -> float | None:
    """Convert a resolved config value into an optional float."""
    if value is None:
        return None
    return float(value)


def boolean_value(value: typing.Any) -> bool:
    """Convert a resolved config value into a boolean.

    Args:
        value: Resolved config value.

    Returns:
        Boolean value.

    Raises:
        TypeError: If the value is not a boolean or explicit boolean string.

    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in {"true", "1", "yes", "on"}:
            return True
        if normalized_value in {"false", "0", "no", "off"}:
            return False
    message = f"Expected a boolean value, got {value!r}."
    raise TypeError(message)


def format_scalar_value(value: typing.Any) -> str:
    """Format one scalar for CLI-compatible comma lists or Hydra overrides."""
    if value is None:
        return "default"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_override_value(value: typing.Any) -> str:
    """Format a Python value as a Hydra command-line override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return quote_string_value(value)
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, str | bytes):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    return str(value)


def quote_string_value(value: str) -> str:
    """Quote Hydra override strings that contain grammar-significant characters."""
    if value == "" or any(character in value for character in ", []{}:=\t\n"):
        escaped_value = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped_value}'"
    return value


def build_overrides(values_by_key: collections.abc.Mapping[str, typing.Any]) -> list[str]:
    """Build Hydra override strings from key-value pairs.

    Args:
        values_by_key: Mapping from Hydra override key to Python value.

    Returns:
        List of ``key=value`` override strings.

    """
    return [f"{key}={format_override_value(value)}" for key, value in values_by_key.items()]
