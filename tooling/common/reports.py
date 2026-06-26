"""Report serialization helpers for development tooling."""

from __future__ import annotations

import dataclasses
import enum
import json
import shutil
import typing
from dataclasses import dataclass
from pathlib import Path

import omegaconf


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


def validate_report_envelope(payload: dict[str, typing.Any], *, schema_name: str) -> None:
    """Validate common Tooling Artifact Format envelope fields.

    Args:
        payload: Parsed payload.
        schema_name: Expected schema name.

    Raises:
        ReportSchemaError: If the payload violates the envelope contract.

    """
    if payload.get("schema_name") != schema_name:
        message = f"Expected schema_name={schema_name!r}, got {payload.get('schema_name')!r}."
        raise ReportSchemaError(message)
    if payload.get("schema_version") != 1:
        message = f"Expected schema_version=1, got {payload.get('schema_version')!r}."
        raise ReportSchemaError(message)
    for field_name in ("producer", "run"):
        if not isinstance(payload.get(field_name), dict):
            message = f"Report envelope requires object field `{field_name}`."
            raise ReportSchemaError(message)


def write_report_envelope(
    path: Path,
    value: typing.Any,
    *,
    schema_name: str,
    sort_keys: bool = True,
) -> None:
    """Write and validate a Tooling Artifact Format envelope.

    Args:
        path: Output JSON path.
        value: Envelope payload.
        schema_name: Expected schema name.
        sort_keys: Whether to sort dictionary keys.

    """
    payload = typing.cast("dict[str, typing.Any]", to_jsonable(value))
    if not isinstance(payload, dict):
        message = "Report envelope payload must be an object."
        raise ReportSchemaError(message)
    validate_report_envelope(payload, schema_name=schema_name)
    write_json_report(path, payload, sort_keys=sort_keys)


def write_jsonl(path: Path, records: typing.Iterable[typing.Any], *, sort_keys: bool = True) -> None:
    """Write JSON Lines records.

    Args:
        path: Output JSONL path.
        records: Records to serialize.
        sort_keys: Whether to sort dictionary keys.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(to_jsonable(record), sort_keys=sort_keys) for record in records]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, typing.Any]]:
    """Read JSON Lines records as objects.

    Args:
        path: Input JSONL path.

    Returns:
        Parsed record objects.

    Raises:
        ReportSchemaError: If a line is invalid or not an object.

    """
    records: list[dict[str, typing.Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            message = f"JSONL file `{path}` line {line_number} is invalid: {error.msg}."
            raise ReportSchemaError(message) from error
        if not isinstance(payload, dict):
            message = f"JSONL file `{path}` line {line_number} must contain an object."
            raise ReportSchemaError(message)
        records.append(typing.cast("dict[str, typing.Any]", payload))
    return records


def write_config_snapshots(
    output_directory: Path,
    *,
    hydra_config: omegaconf.DictConfig | None = None,
    tool_payload: typing.Any | None = None,
) -> dict[str, Path | None]:
    """Write resolved config snapshots for an artifact directory.

    Args:
        output_directory: Artifact output directory.
        hydra_config: Optional composed Hydra config.
        tool_payload: Optional resolved tool argument payload.

    Returns:
        Paths for written snapshots.

    """
    config_directory = output_directory / "config"
    config_directory.mkdir(parents=True, exist_ok=True)
    hydra_path = None
    tool_path = None
    if hydra_config is not None:
        hydra_path = config_directory / "resolved_hydra.yaml"
        hydra_path.write_text(omegaconf.OmegaConf.to_yaml(hydra_config, resolve=True), encoding="utf-8")
    if tool_payload is not None:
        tool_path = config_directory / "resolved_tool.json"
        write_json_report(tool_path, tool_payload, sort_keys=True)
    return {"resolved_hydra": hydra_path, "resolved_tool": tool_path}


def copy_json_alias(source_path: Path, alias_path: Path) -> None:
    """Copy a JSON artifact to a legacy alias path.

    Args:
        source_path: Canonical source path.
        alias_path: Compatibility alias path.

    """
    if source_path.resolve() == alias_path.resolve():
        return
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, alias_path)


def write_markdown_report(path: Path, markdown_text: str) -> None:
    """Write a Markdown report, creating parent directories as needed.

    Args:
        path: Output Markdown path.
        markdown_text: Markdown report body.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")
