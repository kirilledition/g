"""Sweep parsing helpers for development tooling."""

from __future__ import annotations


def split_comma_separated_values(raw_value: str, option_name: str) -> list[str]:
    """Split a comma-separated CLI value into stripped entries.

    Args:
        raw_value: Raw comma-separated value.
        option_name: Option name used in validation messages.

    Returns:
        Stripped values.

    Raises:
        ValueError: If no values are present or an entry is empty.

    """
    values = [value.strip() for value in raw_value.split(",")]
    if not values:
        message = f"{option_name} must contain at least one value."
        raise ValueError(message)
    for value_index, value in enumerate(values, start=1):
        if not value:
            message = f"{option_name} contains an empty entry at position {value_index}."
            raise ValueError(message)
    return values


def parse_optional_integer_list(raw_values: str) -> list[int | None]:
    """Parse a comma-separated integer list with optional default sentinels.

    Args:
        raw_values: Raw comma-separated value.

    Returns:
        Parsed integers, using None for default sentinels.

    """
    parsed_values: list[int | None] = []
    for stripped_value in split_comma_separated_values(raw_values, "optional integer list"):
        if stripped_value.lower() in {"none", "default"}:
            parsed_values.append(None)
            continue
        parsed_values.append(int(stripped_value))
    return parsed_values


def parse_positive_integer_list(raw_value: str, option_name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers.

    Args:
        raw_value: Raw comma-separated value.
        option_name: Option name used in validation messages.

    Returns:
        Parsed positive integers.

    Raises:
        ValueError: If any parsed value is not positive.

    """
    parsed_values: list[int] = []
    for value in split_comma_separated_values(raw_value, option_name):
        parsed_value = int(value)
        if parsed_value <= 0:
            message = f"{option_name} values must be positive."
            raise ValueError(message)
        parsed_values.append(parsed_value)
    return tuple(parsed_values)


def parse_boolean_mode_list(raw_values: str) -> list[bool]:
    """Parse a comma-separated boolean sweep value.

    Args:
        raw_values: Raw comma-separated value.

    Returns:
        Parsed booleans.

    Raises:
        ValueError: If a value is not recognized.

    """
    parsed_values: list[bool] = []
    for raw_value in split_comma_separated_values(raw_values, "boolean mode list"):
        stripped_value = raw_value.lower()
        if stripped_value in {"true", "trusted", "on", "1", "yes"}:
            parsed_values.append(True)
            continue
        if stripped_value in {"false", "safe", "off", "0", "no"}:
            parsed_values.append(False)
            continue
        message = f"Unrecognized boolean sweep value: {raw_value}."
        raise ValueError(message)
    return parsed_values


def build_queue_depths(thread_counts: tuple[int, ...], depth_multipliers: tuple[int, ...]) -> tuple[int, ...]:
    """Build queue depths from writer thread counts and multipliers.

    Args:
        thread_counts: Writer thread counts.
        depth_multipliers: Queue-depth multipliers.

    Returns:
        Sorted unique queue depths.

    """
    return tuple(
        sorted({thread_count * multiplier for thread_count in thread_counts for multiplier in depth_multipliers})
    )
