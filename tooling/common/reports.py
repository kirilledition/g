"""Report serialization helpers for development tooling."""

from __future__ import annotations

import dataclasses
import enum
import json
import typing
from dataclasses import dataclass
from pathlib import Path


class ReportSchemaError(ValueError):
    """Raised when a durable report does not match its schema contract."""


@dataclass(frozen=True)
class VersionedReportContract:
    """Schema contract for a durable JSON report.

    Attributes:
        schema_version: Required integer schema version.
        required_fields: Required top-level field names.
        optional_fields: Optional top-level field names.
        schema_field_name: Field that stores the schema version.
        reject_unknown_fields: Whether unexpected top-level fields are rejected.

    """

    schema_version: int
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    schema_field_name: str
    reject_unknown_fields: bool


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


def read_json_report(path: Path) -> dict[str, typing.Any]:
    """Read a JSON report payload.

    Args:
        path: JSON report path.

    Returns:
        Parsed JSON object.

    Raises:
        ReportSchemaError: If the file cannot be read, decoded, or is not an object.

    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as error:
        message = f"Could not read JSON report `{path}`: {error}"
        raise ReportSchemaError(message) from error
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        message = f"JSON report `{path}` is invalid: {error.msg} at line {error.lineno} column {error.colno}."
        raise ReportSchemaError(message) from error
    if not isinstance(payload, dict):
        message = f"JSON report `{path}` must contain a top-level object."
        raise ReportSchemaError(message)
    return typing.cast("dict[str, typing.Any]", payload)


def validate_report_shape(payload: dict[str, typing.Any], contract: VersionedReportContract) -> None:
    """Validate a report payload against a versioned top-level contract.

    Args:
        payload: Report payload.
        contract: Expected report contract.

    Raises:
        ReportSchemaError: If the payload violates the contract.

    """
    schema_version = payload.get(contract.schema_field_name)
    if schema_version != contract.schema_version:
        message = f"Expected {contract.schema_field_name}={contract.schema_version}, got {schema_version!r}."
        raise ReportSchemaError(message)
    required_fields = set(contract.required_fields) | {contract.schema_field_name}
    missing_fields = sorted(required_fields - set(payload))
    if missing_fields:
        message = f"Report is missing required fields: {', '.join(missing_fields)}."
        raise ReportSchemaError(message)
    if contract.reject_unknown_fields:
        allowed_fields = required_fields | set(contract.optional_fields)
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            message = f"Report contains unknown fields: {', '.join(unknown_fields)}."
            raise ReportSchemaError(message)


def write_json_report(path: Path, value: typing.Any, *, sort_keys: bool = False) -> None:
    """Write a JSON report, creating parent directories as needed.

    Args:
        path: Output JSON path.
        value: Report payload.
        sort_keys: Whether to sort dictionary keys.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_text(value, sort_keys=sort_keys), encoding="utf-8")


def write_versioned_json_report(
    path: Path,
    value: typing.Any,
    contract: VersionedReportContract,
    *,
    sort_keys: bool = False,
) -> None:
    """Write a JSON report after validating its versioned contract.

    Args:
        path: Output JSON path.
        value: Report payload.
        contract: Expected report contract.
        sort_keys: Whether to sort dictionary keys.

    Raises:
        ReportSchemaError: If the payload violates the contract.

    """
    payload = typing.cast("dict[str, typing.Any]", to_jsonable(value))
    if not isinstance(payload, dict):
        message = "Versioned JSON report payload must be an object."
        raise ReportSchemaError(message)
    validate_report_shape(payload, contract)
    write_json_report(path, payload, sort_keys=sort_keys)


def read_versioned_json_report(path: Path, contract: VersionedReportContract) -> dict[str, typing.Any]:
    """Read and validate a versioned JSON report.

    Args:
        path: JSON report path.
        contract: Expected report contract.

    Returns:
        Validated report payload.

    """
    payload = read_json_report(path)
    validate_report_shape(payload, contract)
    return payload


def write_markdown_report(path: Path, markdown_text: str) -> None:
    """Write a Markdown report, creating parent directories as needed.

    Args:
        path: Output Markdown path.
        markdown_text: Markdown report body.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")
