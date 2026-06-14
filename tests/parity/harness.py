from __future__ import annotations

import dataclasses
import enum
import json
import typing
from pathlib import Path

import numpy as np
import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = Path(__file__).with_name("golden_metadata.json")
REQUIRED_WORKFLOW_IDENTIFIERS = frozenset(
    {
        "quantitative_single_bgen_loco",
        "quantitative_multi_per_phenotype",
        "quantitative_multi_complete_case",
        "binary_score_only",
        "binary_approximate_firth",
        "missing_sample_filtering",
        "variant_missingness_imputation",
        "resume_clean_equivalence",
    }
)


class ParityWorkflowStatus(enum.StrEnum):
    EXTERNAL_GOLDEN = "external_golden"
    CONTRACT = "contract"
    EXPERIMENTAL = "experimental"


@dataclasses.dataclass(frozen=True)
class StatisticTolerance:
    observed_column: str
    baseline_column: str
    absolute_tolerance: float


@dataclasses.dataclass(frozen=True)
class StatisticComparison:
    observed_column: str
    baseline_column: str
    row_count: int
    maximum_absolute_difference: float


@dataclasses.dataclass(frozen=True)
class GoldenWorkflow:
    identifier: str
    title: str
    status: ParityWorkflowStatus
    description: str
    regenie_version: str | None
    regenie_commands: tuple[str, ...]
    expected_output_path: Path | None
    g_command_options: dict[str, object]
    validation_nodes: tuple[str, ...]
    documentation_paths: tuple[Path, ...]
    tolerances: tuple[StatisticTolerance, ...]


@dataclasses.dataclass(frozen=True)
class ParityMetadata:
    schema_version: int
    regenie_reference: dict[str, object]
    workflows: tuple[GoldenWorkflow, ...]

    @property
    def workflow_identifiers(self) -> frozenset[str]:
        return frozenset(workflow.identifier for workflow in self.workflows)

    def workflow_by_identifier(self, workflow_identifier: str) -> GoldenWorkflow:
        for workflow in self.workflows:
            if workflow.identifier == workflow_identifier:
                return workflow
        raise KeyError(f"Unknown REGENIE parity workflow: {workflow_identifier}")


def read_regenie_table(table_path: Path) -> pl.DataFrame:
    table_lines = [line.strip() for line in table_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not table_lines:
        raise ValueError(f"Results table is empty: {table_path}")
    header_columns = table_lines[0].split()
    rows: list[dict[str, str]] = []
    for line in table_lines[1:]:
        values = line.split()
        if len(values) != len(header_columns):
            raise ValueError(
                f"Unexpected column count in {table_path}: expected {len(header_columns)}, got {len(values)}"
            )
        rows.append(dict(zip(header_columns, values, strict=True)))
    return pl.DataFrame(rows)


def assert_statistic_columns_match(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    join_column: str,
    tolerance: StatisticTolerance,
    expected_row_count: int,
) -> StatisticComparison:
    merged_results = observed_results.join(
        baseline_results.select(join_column, tolerance.baseline_column),
        on=join_column,
        how="inner",
    )

    if merged_results.height != expected_row_count:
        raise AssertionError(
            f"Expected {expected_row_count} joined rows for {tolerance.observed_column}, got {merged_results.height}"
        )

    observed_values = np.asarray(merged_results.get_column(tolerance.observed_column).to_numpy(), dtype=np.float64)
    baseline_values = np.asarray(merged_results.get_column(tolerance.baseline_column).to_numpy(), dtype=np.float64)
    absolute_differences = np.abs(observed_values - baseline_values)
    maximum_absolute_difference = float(np.nanmax(absolute_differences)) if absolute_differences.size else 0.0

    np.testing.assert_allclose(
        observed_values,
        baseline_values,
        atol=tolerance.absolute_tolerance,
        rtol=0.0,
    )
    return StatisticComparison(
        observed_column=tolerance.observed_column,
        baseline_column=tolerance.baseline_column,
        row_count=merged_results.height,
        maximum_absolute_difference=maximum_absolute_difference,
    )


def assert_metadata_covers_required_workflows(metadata: ParityMetadata) -> None:
    missing_identifiers = REQUIRED_WORKFLOW_IDENTIFIERS.difference(metadata.workflow_identifiers)
    if missing_identifiers:
        missing_text = ", ".join(sorted(missing_identifiers))
        raise AssertionError(f"Missing REGENIE parity workflow metadata: {missing_text}")

    for workflow in metadata.workflows:
        if workflow.status != ParityWorkflowStatus.EXPERIMENTAL and not workflow.validation_nodes:
            raise AssertionError(f"Parity workflow has no validation nodes: {workflow.identifier}")


def load_golden_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> ParityMetadata:
    payload = typing.cast("dict[str, object]", json.loads(metadata_path.read_text(encoding="utf-8")))
    workflow_payloads = typing.cast("list[dict[str, object]]", payload["workflows"])
    regenie_reference = typing.cast("dict[str, object]", payload["regenie_reference"])
    workflows = tuple(parse_workflow_payload(workflow_payload) for workflow_payload in workflow_payloads)
    return ParityMetadata(
        schema_version=int(typing.cast("int | str", payload["schema_version"])),
        regenie_reference=regenie_reference,
        workflows=workflows,
    )


def parse_workflow_payload(workflow_payload: dict[str, object]) -> GoldenWorkflow:
    return GoldenWorkflow(
        identifier=str(workflow_payload["identifier"]),
        title=str(workflow_payload["title"]),
        status=ParityWorkflowStatus(str(workflow_payload["status"])),
        description=str(workflow_payload["description"]),
        regenie_version=parse_optional_string(workflow_payload["regenie_version"]),
        regenie_commands=parse_string_tuple(workflow_payload["regenie_commands"]),
        expected_output_path=parse_optional_repository_path(workflow_payload["expected_output_path"]),
        g_command_options=typing.cast("dict[str, object]", workflow_payload["g_command_options"]),
        validation_nodes=parse_string_tuple(workflow_payload["validation_nodes"]),
        documentation_paths=parse_repository_paths(workflow_payload["documentation_paths"]),
        tolerances=parse_tolerances(workflow_payload["tolerances"]),
    )


def parse_tolerances(tolerances_payload: object) -> tuple[StatisticTolerance, ...]:
    tolerance_payloads = typing.cast("list[dict[str, object]]", tolerances_payload)
    return tuple(
        StatisticTolerance(
            observed_column=str(tolerance_payload["observed_column"]),
            baseline_column=str(tolerance_payload["baseline_column"]),
            absolute_tolerance=float(typing.cast("float | int | str", tolerance_payload["absolute_tolerance"])),
        )
        for tolerance_payload in tolerance_payloads
    )


def parse_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_optional_repository_path(value: object) -> Path | None:
    if value is None:
        return None
    return REPOSITORY_ROOT / str(value)


def parse_repository_paths(value: object) -> tuple[Path, ...]:
    return tuple(REPOSITORY_ROOT / path for path in parse_string_tuple(value))


def parse_string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in typing.cast("list[object]", value))
