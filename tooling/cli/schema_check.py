"""Validate Tooling Artifact Format v1 artifact directories."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass

import hydra

import tooling.configuration as tooling_configuration
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    from pathlib import Path

    import omegaconf


@dataclass(frozen=True)
class SchemaCheckArguments:
    """Resolved schema-check arguments.

    Attributes:
        path: Artifact directory or JSON/JSONL file to validate.
        require_optional_files: Whether optional standard files must exist.

    """

    path: Path
    require_optional_files: bool


@dataclasses.dataclass(frozen=True)
class SchemaCheckResult:
    """Schema-check result."""

    checked_paths: tuple[str, ...]
    error_messages: tuple[str, ...]


def validate_artifact_directory(path: Path, *, require_optional_files: bool) -> list[str]:
    """Validate a Tooling Artifact Format directory.

    Args:
        path: Artifact directory.
        require_optional_files: Whether optional standard files must exist.

    Returns:
        Validation error messages.

    """
    error_messages: list[str] = []
    required_files = ("artifact_manifest.json", "report.json", "summary.md")
    optional_files = ("events.jsonl", "metrics.jsonl", "commands/commands.jsonl")
    for relative_path in required_files:
        if not (path / relative_path).is_file():
            error_messages.append(f"Missing required artifact file: {relative_path}")
    if require_optional_files:
        for relative_path in optional_files:
            if not (path / relative_path).is_file():
                error_messages.append(f"Missing optional artifact file required by check: {relative_path}")
    for relative_path, schema_name in (
        ("artifact_manifest.json", "g.tooling.artifact_manifest"),
        ("report.json", "g.tooling.report"),
        ("comparisons.json", "g.tooling.comparison"),
    ):
        file_path = path / relative_path
        if file_path.exists():
            error_messages.extend(validate_json_file(file_path, schema_name=schema_name))
    for relative_path, schema_name in (
        ("events.jsonl", "g.tooling.event"),
        ("metrics.jsonl", "g.tooling.metric"),
        ("commands/commands.jsonl", "g.tooling.command"),
    ):
        file_path = path / relative_path
        if file_path.exists():
            error_messages.extend(validate_jsonl_file(file_path, schema_name=schema_name))
    return error_messages


def validate_json_file(path: Path, *, schema_name: str) -> list[str]:
    """Validate one JSON envelope file."""
    try:
        payload = tooling_reports.read_json_report(path)
        tooling_artifact_format.validate_schema_payload(payload, schema_name)
    except tooling_reports.ReportSchemaError as error:
        return [f"{path}: {error}"]
    return []


def validate_jsonl_file(path: Path, *, schema_name: str) -> list[str]:
    """Validate one JSONL file."""
    error_messages: list[str] = []
    try:
        records = tooling_reports.read_jsonl(path)
    except tooling_reports.ReportSchemaError as error:
        return [f"{path}: {error}"]
    for record_index, payload in enumerate(records, start=1):
        try:
            tooling_artifact_format.validate_schema_payload(payload, schema_name)
        except tooling_reports.ReportSchemaError as error:
            error_messages.append(f"{path}:{record_index}: {error}")
    return error_messages


def run_schema_check(arguments: SchemaCheckArguments) -> SchemaCheckResult:
    """Run schema validation and return structured results."""
    path = arguments.path
    if path.is_dir():
        error_messages = validate_artifact_directory(path, require_optional_files=arguments.require_optional_files)
        return SchemaCheckResult(checked_paths=(str(path),), error_messages=tuple(error_messages))
    if path.suffix == ".jsonl":
        schema_name = "g.tooling.event"
        if path.name == "metrics.jsonl":
            schema_name = "g.tooling.metric"
        elif path.name == "commands.jsonl":
            schema_name = "g.tooling.command"
        return SchemaCheckResult(
            checked_paths=(str(path),),
            error_messages=tuple(validate_jsonl_file(path, schema_name=schema_name)),
        )
    if path.suffix == ".json":
        schema_name_by_file = {
            "artifact_manifest.json": "g.tooling.artifact_manifest",
            "report.json": "g.tooling.report",
            "comparisons.json": "g.tooling.comparison",
        }
        schema_name = schema_name_by_file.get(path.name, "g.tooling.report")
        return SchemaCheckResult(
            checked_paths=(str(path),),
            error_messages=tuple(validate_json_file(path, schema_name=schema_name)),
        )
    return SchemaCheckResult(checked_paths=(str(path),), error_messages=(f"Unsupported path: {path}",))


def run_tool(arguments: SchemaCheckArguments) -> None:
    """Run the schema checker CLI."""
    result = run_schema_check(arguments)
    if result.error_messages:
        for error_message in result.error_messages:
            print(error_message)
        raise SystemExit(1)
    for checked_path in result.checked_paths:
        print(f"OK {checked_path}")


def build_arguments_from_config(config: omegaconf.DictConfig) -> SchemaCheckArguments:
    """Build schema-check arguments from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    path = tooling_hydra_arguments.path_or_none(tool_values["path"])
    if path is None:
        message = "tool.path is required."
        raise ValueError(message)
    return SchemaCheckArguments(
        path=path,
        require_optional_files=tooling_hydra_arguments.boolean_value(tool_values["require_optional_files"]),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> SchemaCheckArguments:
    """Compose schema-check arguments from Hydra overrides."""
    config = tooling_configuration.compose_config(config_name="schema_check", overrides=overrides)
    return build_arguments_from_config(config)


@hydra.main(version_base=None, config_path="../configs", config_name="schema_check")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run schema-check from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run schema-check from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
