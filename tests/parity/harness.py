"""Shared external-REGENIE parity metadata and comparison helpers."""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import tests.numerical

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = Path(__file__).with_name("golden_metadata.json")
VARIANT_KEY_COLUMNS = ("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1")
REQUIRED_INPUT_OPTION_NAMES = ("bgen", "sample", "phenotype_file", "covariate_file", "prediction_list")
REQUIRED_WORKFLOW_IDENTIFIERS = frozenset(
    {
        "quantitative_single_bgen_loco",
        "binary_score_only",
        "binary_approximate_firth",
    }
)
SIGNIFICANCE_P_VALUE_THRESHOLDS = (0.05, 5.0e-8)
REGENIE_CORRECTION_COUNT_PATTERN = re.compile(
    r"^\s*Number of tests with Firth correction\s*:\s*(\d+)\s*$",
    flags=re.MULTILINE,
)
REGENIE_CORRECTION_FAILURE_PATTERN = re.compile(
    r"^\s*Number of failed tests\s*:\s*\((\d+)/(\d+)\)\s*$",
    flags=re.MULTILINE,
)


class ParityWorkflowStatus(enum.StrEnum):
    """Lifecycle state of one external parity workflow."""

    EXTERNAL_GOLDEN = "external_golden"


class ParityGateStatus(enum.StrEnum):
    """Enforcement state of one external parity workflow."""

    BLOCKING = "blocking"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class StatisticTolerance:
    """Absolute tolerance for one externally observable statistic."""

    observed_column: str
    baseline_column: str
    absolute_tolerance: float


@dataclass(frozen=True)
class StatisticComparison:
    """Summary of one successful statistic comparison."""

    observed_column: str
    baseline_column: str
    row_count: int
    maximum_absolute_difference: float


@dataclass(frozen=True)
class RegenieCorrectionSummary:
    """Aggregate approximate-Firth counts parsed from an upstream log."""

    correction_count: int
    correction_failure_count: int


@dataclass(frozen=True)
class GoldenWorkflow:
    """One production CLI workflow with an upstream REGENIE oracle."""

    identifier: str
    title: str
    status: ParityWorkflowStatus
    gate_status: ParityGateStatus
    description: str
    regenie_version: str
    regenie_commands: tuple[str, ...]
    expected_output_relative_path: Path
    expected_output_sha256: str
    expected_log_relative_path: Path
    expected_log_sha256: str
    expected_row_count: int
    expected_correction_count: int | None
    expected_correction_failure_count: int | None
    g_cli_options: dict[str, object]
    input_sha256: dict[str, str]
    prediction_file_sha256: dict[str, str]
    validation_nodes: tuple[str, ...]
    documentation_paths: tuple[Path, ...]
    tolerances: tuple[StatisticTolerance, ...]
    qualification: dict[str, object]


@dataclass(frozen=True)
class ParityMetadata:
    """Parsed external parity metadata."""

    schema_version: int
    regenie_reference: dict[str, object]
    workflows: tuple[GoldenWorkflow, ...]

    @property
    def workflow_identifiers(self) -> frozenset[str]:
        """Return every declared workflow identifier."""
        return frozenset(workflow.identifier for workflow in self.workflows)

    def workflow_by_identifier(self, workflow_identifier: str) -> GoldenWorkflow:
        """Return the workflow with the requested stable identifier."""
        for workflow in self.workflows:
            if workflow.identifier == workflow_identifier:
                return workflow
        raise KeyError(f"Unknown REGENIE parity workflow: {workflow_identifier}")


def read_regenie_table(table_path: Path) -> pl.DataFrame:
    """Read an upstream REGENIE whitespace-delimited results table."""
    return pl.read_csv(table_path, separator=" ", null_values="NA")


def read_direct_parquet_dataset(output_root: Path) -> pl.DataFrame:
    """Read the sole phenotype's production Parquet-parts dataset.

    Args:
        output_root: Native CLI output root containing one phenotype run.

    Raises:
        AssertionError: If the run or direct Parquet-parts contract is absent.
    """
    parquet_paths = direct_parquet_paths(output_root)
    return pl.concat((pl.read_parquet(path) for path in parquet_paths), how="vertical")


def direct_parquet_paths(output_root: Path) -> tuple[Path, ...]:
    """Return the sole phenotype's ordered production Parquet parts."""
    run_directories = sorted(path for path in output_root.rglob("*.run") if path.is_dir())
    if len(run_directories) != 1:
        raise AssertionError(f"Expected one phenotype run below {output_root}, found {len(run_directories)}")
    parquet_paths = sorted((run_directories[0] / "parts").glob("*.parquet"))
    if not parquet_paths:
        raise AssertionError(f"No direct Parquet parts found below {run_directories[0]}")
    return tuple(parquet_paths)


def sha256_file(file_path: Path) -> str:
    """Return the SHA-256 digest of one external-oracle artifact."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_file_sha256(file_path: Path, expected_sha256: str) -> None:
    """Require an external-oracle artifact to match its recorded digest."""
    observed_sha256 = sha256_file(file_path)
    if observed_sha256 != expected_sha256:
        raise AssertionError(
            f"SHA-256 mismatch for {file_path}: expected {expected_sha256}, observed {observed_sha256}"
        )


def sha256_file_set(file_paths: tuple[Path, ...], *, root: Path) -> str:
    """Hash relative names and contents for an ordered file set."""
    digest = hashlib.sha256()
    for file_path in file_paths:
        digest.update(file_path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def parse_regenie_correction_summary(log_path: Path) -> RegenieCorrectionSummary | None:
    """Parse aggregate approximate-Firth counts from a pinned REGENIE log.

    Returns:
        Parsed counts, or `None` when the workflow did not request Firth.

    Raises:
        AssertionError: If the log contains an incomplete, repeated, or
            internally inconsistent correction summary.
    """
    log_text = log_path.read_text(encoding="utf-8")
    correction_matches = REGENIE_CORRECTION_COUNT_PATTERN.findall(log_text)
    failure_matches = REGENIE_CORRECTION_FAILURE_PATTERN.findall(log_text)
    if not correction_matches and not failure_matches:
        return None
    if len(correction_matches) != 1 or len(failure_matches) != 1:
        raise AssertionError(f"Expected one complete Firth correction summary in {log_path}")
    correction_count = int(correction_matches[0])
    correction_failure_count = int(failure_matches[0][0])
    failure_denominator = int(failure_matches[0][1])
    if failure_denominator != correction_count:
        raise AssertionError(
            f"REGENIE correction summary denominator mismatch in {log_path}: "
            f"{failure_denominator} != {correction_count}"
        )
    if correction_failure_count > correction_count:
        raise AssertionError(
            f"REGENIE correction failures exceed corrections in {log_path}: "
            f"{correction_failure_count} > {correction_count}"
        )
    return RegenieCorrectionSummary(
        correction_count=correction_count,
        correction_failure_count=correction_failure_count,
    )


def resolve_prediction_files(prediction_list_path: Path, *, data_directory: Path) -> dict[str, Path]:
    """Resolve prediction-list members without allowing fixture-root escapes.

    Relative member paths are interpreted relative to the prediction list.

    Raises:
        AssertionError: If a row is malformed, a member is absent, two rows
            resolve to the same data-relative path, or a member escapes the
            configured data directory.
    """
    data_root = data_directory.resolve(strict=True)
    resolved_paths: dict[str, Path] = {}
    for line_number, line in enumerate(prediction_list_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise AssertionError(f"Malformed prediction-list row {line_number} in {prediction_list_path}")
        member_path = Path(fields[1])
        candidate_path = member_path if member_path.is_absolute() else prediction_list_path.parent / member_path
        try:
            resolved_path = candidate_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise AssertionError(f"Missing prediction file from {prediction_list_path}: {candidate_path}") from error
        try:
            relative_path = resolved_path.relative_to(data_root).as_posix()
        except ValueError as error:
            raise AssertionError(
                f"Prediction file escapes configured data directory {data_root}: {resolved_path}"
            ) from error
        if relative_path in resolved_paths:
            raise AssertionError(f"Duplicate prediction file in {prediction_list_path}: {relative_path}")
        resolved_paths[relative_path] = resolved_path
    if not resolved_paths:
        raise AssertionError(f"Prediction list is empty: {prediction_list_path}")
    return resolved_paths


def cast_variant_key_columns(data_frame: pl.DataFrame) -> pl.DataFrame:
    """Cast variant identity columns to stable comparison types."""
    return data_frame.with_columns(
        pl.col("CHROM").cast(pl.String),
        pl.col("GENPOS").cast(pl.Int64),
        pl.col("ID").cast(pl.String),
        pl.col("ALLELE0").cast(pl.String),
        pl.col("ALLELE1").cast(pl.String),
    )


def assert_unique_variant_keys(data_frame: pl.DataFrame, *, label: str) -> None:
    """Require one row per composite variant identity."""
    missing_columns = set(VARIANT_KEY_COLUMNS).difference(data_frame.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise AssertionError(f"{label} results are missing variant key columns: {missing_text}")
    duplicate_keys = data_frame.group_by(*VARIANT_KEY_COLUMNS).len().filter(pl.col("len") != 1)
    if duplicate_keys.height != 0:
        raise AssertionError(f"{label} results contain {duplicate_keys.height} duplicate composite variant keys")


def assert_variant_key_order_match(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    expected_row_count: int,
) -> None:
    """Require upstream and production rows to have identical variant order."""
    if observed_results.height != expected_row_count:
        raise AssertionError(f"Expected {expected_row_count} observed rows, got {observed_results.height}")
    if baseline_results.height != expected_row_count:
        raise AssertionError(f"Expected {expected_row_count} REGENIE rows, got {baseline_results.height}")
    observed_keys = cast_variant_key_columns(observed_results).select(*VARIANT_KEY_COLUMNS)
    baseline_keys = cast_variant_key_columns(baseline_results).select(*VARIANT_KEY_COLUMNS)
    if not observed_keys.equals(baseline_keys):
        raise AssertionError("g and REGENIE variant keys differ in identity or row order")


def align_result_columns(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    observed_column: str,
    baseline_column: str,
    expected_row_count: int,
) -> pl.DataFrame:
    """Align one observed/reference column by full variant identity."""
    if observed_results.height != expected_row_count:
        raise AssertionError(f"Expected {expected_row_count} observed rows, got {observed_results.height}")
    if baseline_results.height != expected_row_count:
        raise AssertionError(f"Expected {expected_row_count} REGENIE rows, got {baseline_results.height}")
    assert_unique_variant_keys(observed_results, label="g")
    assert_unique_variant_keys(baseline_results, label="REGENIE")

    observed_frame = cast_variant_key_columns(observed_results).select(
        *VARIANT_KEY_COLUMNS,
        pl.col(observed_column).alias("actual_value"),
    )
    baseline_frame = cast_variant_key_columns(baseline_results).select(
        *VARIANT_KEY_COLUMNS,
        pl.col(baseline_column).alias("reference_value"),
    )
    aligned_results = observed_frame.join(
        baseline_frame,
        on=list(VARIANT_KEY_COLUMNS),
        how="inner",
        validate="1:1",
    )
    if aligned_results.height != expected_row_count:
        raise AssertionError(
            f"Expected {expected_row_count} composite-key matches for {observed_column}, got {aligned_results.height}"
        )
    return aligned_results.sort(*VARIANT_KEY_COLUMNS)


def assert_statistic_columns_match(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    tolerance: StatisticTolerance,
    expected_row_count: int,
) -> StatisticComparison:
    """Require strict absolute agreement with an upstream statistic."""
    aligned_results = align_result_columns(
        observed_results,
        baseline_results,
        observed_column=tolerance.observed_column,
        baseline_column=tolerance.baseline_column,
        expected_row_count=expected_row_count,
    )
    observed_values = np.asarray(aligned_results.get_column("actual_value").to_numpy(), dtype=np.float64)
    baseline_values = np.asarray(aligned_results.get_column("reference_value").to_numpy(), dtype=np.float64)
    tests.numerical.assert_absolute_difference_less_than(
        observed_values,
        baseline_values,
        tolerance.absolute_tolerance,
    )
    finite_mask = np.isfinite(observed_values)
    finite_differences = np.abs(observed_values[finite_mask] - baseline_values[finite_mask])
    maximum_absolute_difference = float(np.max(finite_differences)) if finite_differences.size else 0.0
    return StatisticComparison(
        observed_column=tolerance.observed_column,
        baseline_column=tolerance.baseline_column,
        row_count=aligned_results.height,
        maximum_absolute_difference=maximum_absolute_difference,
    )


def measure_statistic_columns(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    tolerance: StatisticTolerance,
    expected_row_count: int,
) -> StatisticComparison:
    """Measure a statistic without applying its numerical tolerance."""
    aligned_results = align_result_columns(
        observed_results,
        baseline_results,
        observed_column=tolerance.observed_column,
        baseline_column=tolerance.baseline_column,
        expected_row_count=expected_row_count,
    )
    observed_values = np.asarray(aligned_results.get_column("actual_value").to_numpy(), dtype=np.float64)
    baseline_values = np.asarray(aligned_results.get_column("reference_value").to_numpy(), dtype=np.float64)
    jointly_finite_mask = np.isfinite(observed_values) & np.isfinite(baseline_values)
    finite_differences = np.abs(observed_values[jointly_finite_mask] - baseline_values[jointly_finite_mask])
    maximum_absolute_difference = float(np.max(finite_differences)) if finite_differences.size else 0.0
    return StatisticComparison(
        observed_column=tolerance.observed_column,
        baseline_column=tolerance.baseline_column,
        row_count=aligned_results.height,
        maximum_absolute_difference=maximum_absolute_difference,
    )


def assert_exact_column_match(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    observed_column: str,
    baseline_column: str,
    expected_row_count: int,
) -> None:
    """Require exact equality for one aligned non-floating contract column."""
    aligned_results = align_result_columns(
        observed_results,
        baseline_results,
        observed_column=observed_column,
        baseline_column=baseline_column,
        expected_row_count=expected_row_count,
    )
    np.testing.assert_array_equal(
        aligned_results.get_column("actual_value").to_numpy(),
        aligned_results.get_column("reference_value").to_numpy(),
    )


def assert_significance_classifications_match(
    observed_results: pl.DataFrame,
    baseline_results: pl.DataFrame,
    *,
    expected_row_count: int,
) -> None:
    """Require exact public significance decisions at both release thresholds."""
    aligned_results = align_result_columns(
        observed_results,
        baseline_results,
        observed_column="LOG10P",
        baseline_column="LOG10P",
        expected_row_count=expected_row_count,
    )
    observed_values = np.asarray(aligned_results.get_column("actual_value").to_numpy(), dtype=np.float64)
    baseline_values = np.asarray(aligned_results.get_column("reference_value").to_numpy(), dtype=np.float64)
    for p_value_threshold in SIGNIFICANCE_P_VALUE_THRESHOLDS:
        log10p_threshold = -math.log10(p_value_threshold)
        np.testing.assert_array_equal(
            observed_values > log10p_threshold,
            baseline_values > log10p_threshold,
        )


def assert_metadata_covers_required_workflows(metadata: ParityMetadata) -> None:
    """Validate the required external-oracle coverage contract."""
    missing_identifiers = REQUIRED_WORKFLOW_IDENTIFIERS.difference(metadata.workflow_identifiers)
    if missing_identifiers:
        missing_text = ", ".join(sorted(missing_identifiers))
        raise AssertionError(f"Missing REGENIE parity workflow metadata: {missing_text}")
    for workflow in metadata.workflows:
        if workflow.status != ParityWorkflowStatus.EXTERNAL_GOLDEN:
            raise AssertionError(f"Parity workflow is not backed by an external golden: {workflow.identifier}")
        if not workflow.validation_nodes:
            raise AssertionError(f"Parity workflow has no validation nodes: {workflow.identifier}")
        if not workflow.tolerances:
            raise AssertionError(f"Parity workflow has no numerical tolerances: {workflow.identifier}")
        if set(workflow.input_sha256) != set(REQUIRED_INPUT_OPTION_NAMES):
            raise AssertionError(f"Parity workflow has incomplete input hashes: {workflow.identifier}")
        if not workflow.prediction_file_sha256:
            raise AssertionError(f"Parity workflow has no referenced prediction hashes: {workflow.identifier}")


def load_golden_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> ParityMetadata:
    """Load and type the checked-in external parity metadata."""
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
    """Parse one workflow object from the metadata document."""
    return GoldenWorkflow(
        identifier=str(workflow_payload["identifier"]),
        title=str(workflow_payload["title"]),
        status=ParityWorkflowStatus(str(workflow_payload["status"])),
        gate_status=ParityGateStatus(str(workflow_payload["gate_status"])),
        description=str(workflow_payload["description"]),
        regenie_version=str(workflow_payload["regenie_version"]),
        regenie_commands=parse_string_tuple(workflow_payload["regenie_commands"]),
        expected_output_relative_path=Path(str(workflow_payload["expected_output_relative_path"])),
        expected_output_sha256=str(workflow_payload["expected_output_sha256"]),
        expected_log_relative_path=Path(str(workflow_payload["expected_log_relative_path"])),
        expected_log_sha256=str(workflow_payload["expected_log_sha256"]),
        expected_row_count=int(typing.cast("int | str", workflow_payload["expected_row_count"])),
        expected_correction_count=parse_optional_integer(workflow_payload["expected_correction_count"]),
        expected_correction_failure_count=parse_optional_integer(workflow_payload["expected_correction_failure_count"]),
        g_cli_options=typing.cast("dict[str, object]", workflow_payload["g_cli_options"]),
        input_sha256=parse_string_mapping(workflow_payload["input_sha256"]),
        prediction_file_sha256=parse_string_mapping(workflow_payload["prediction_file_sha256"]),
        validation_nodes=parse_string_tuple(workflow_payload["validation_nodes"]),
        documentation_paths=parse_repository_paths(workflow_payload["documentation_paths"]),
        tolerances=parse_tolerances(workflow_payload["tolerances"]),
        qualification=typing.cast("dict[str, object]", workflow_payload["qualification"]),
    )


def parse_tolerances(tolerances_payload: object) -> tuple[StatisticTolerance, ...]:
    """Parse statistic-tolerance records."""
    tolerance_payloads = typing.cast("list[dict[str, object]]", tolerances_payload)
    return tuple(
        StatisticTolerance(
            observed_column=str(tolerance_payload["observed_column"]),
            baseline_column=str(tolerance_payload["baseline_column"]),
            absolute_tolerance=float(typing.cast("float | int | str", tolerance_payload["absolute_tolerance"])),
        )
        for tolerance_payload in tolerance_payloads
    )


def parse_optional_integer(value: object) -> int | None:
    """Parse a nullable integer metadata field."""
    if value is None:
        return None
    return int(typing.cast("int | str", value))


def parse_repository_paths(value: object) -> tuple[Path, ...]:
    """Resolve repository-relative documentation paths."""
    return tuple(REPOSITORY_ROOT / path for path in parse_string_tuple(value))


def parse_string_tuple(value: object) -> tuple[str, ...]:
    """Parse a JSON string array as an immutable tuple."""
    return tuple(str(item) for item in typing.cast("list[object]", value))


def parse_string_mapping(value: object) -> dict[str, str]:
    """Parse a JSON string-to-string object."""
    payload = typing.cast("dict[str, object]", value)
    return {str(key): str(item) for key, item in payload.items()}
