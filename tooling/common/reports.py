"""Report serialization helpers for development tooling."""

from __future__ import annotations

import dataclasses
import enum
import json
import typing
from pathlib import Path


def to_jsonable(value: typing.Any) -> typing.Any:
    """Convert common tooling values into JSON-serializable structures.

    Args:
        value: Value to convert.

    Returns:
        JSON-serializable value.

    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


def to_json_text(value: typing.Any, *, sort_keys: bool = False) -> str:
    """Serialize a report payload as pretty JSON text.

    Args:
        value: Report payload.
        sort_keys: Whether to sort dictionary keys.

    Returns:
        JSON text with a trailing newline.

    """
    return json.dumps(to_jsonable(value), indent=2, sort_keys=sort_keys) + "\n"


def write_json_report(path: Path, value: typing.Any, *, sort_keys: bool = False) -> None:
    """Write a JSON report, creating parent directories as needed.

    Args:
        path: Output JSON path.
        value: Report payload.
        sort_keys: Whether to sort dictionary keys.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_text(value, sort_keys=sort_keys), encoding="utf-8")


def write_markdown_report(path: Path, markdown_text: str) -> None:
    """Write a Markdown report, creating parent directories as needed.

    Args:
        path: Output Markdown path.
        markdown_text: Markdown report body.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")
