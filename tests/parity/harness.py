"""Shared external-REGENIE parity metadata and comparison helpers."""

from __future__ import annotations

import datetime
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
import pyarrow.parquet

import tests.numerical
import tooling.science_gate

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = Path(__file__).with_name("golden_metadata.json")
PARITY_METADATA_SCHEMA_VERSION = 3
VARIANT_KEY_COLUMNS = ("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1")
REQUIRED_INPUT_OPTION_NAMES = ("bgen", "sample", "phenotype_file", "covariate_file", "prediction_list")
REQUIRED_WORKFLOW_IDENTIFIERS = frozenset(
    {
        "quantitative_single_bgen_loco",
        "binary_score_only",
        "binary_approximate_firth",
    }
)
REQUIRED_QUALIFICATION_HOSTS = ("landau",)
QUALIFICATION_BOOTSTRAP_RELATIVE_PATH = "tooling/server/exact_parity_bootstrap.sh"
QUALIFICATION_CLOCK_SKEW = datetime.timedelta(minutes=5)
PARQUET_DATASET_COMPLETION_PREFIX = "Parquet dataset saved to "
SIGNIFICANCE_P_VALUE_THRESHOLDS = (0.05, 5.0e-8)
REQUIRED_JAX_VERSION = "0.11.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_NONCE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SLURM_JOB_ID_PATTERN = re.compile(r"^[0-9]+$")
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

    DIAGNOSTIC = "diagnostic"
    REQUIRED = "required"


class QualificationOutputDataType(enum.StrEnum):
    """Closed output data-type names stored in qualification metadata."""

    STRING = "String"
    INT64 = "Int64"
    FLOAT32 = "Float32"
    INT32 = "Int32"


class NativeBuildProfile(enum.StrEnum):
    """Native build profiles eligible for qualification."""

    RELEASE = "release"


@dataclass(frozen=True)
class QualificationOutputField:
    """One ordered field in the qualified production output."""

    name: str
    data_type: QualificationOutputDataType
    nullable: bool


PRODUCTION_OUTPUT_FIELDS = (
    QualificationOutputField(
        name="CHROM",
        data_type=QualificationOutputDataType.STRING,
        nullable=False,
    ),
    QualificationOutputField(
        name="GENPOS",
        data_type=QualificationOutputDataType.INT64,
        nullable=False,
    ),
    QualificationOutputField(
        name="ID",
        data_type=QualificationOutputDataType.STRING,
        nullable=False,
    ),
    QualificationOutputField(
        name="ALLELE0",
        data_type=QualificationOutputDataType.STRING,
        nullable=False,
    ),
    QualificationOutputField(
        name="ALLELE1",
        data_type=QualificationOutputDataType.STRING,
        nullable=False,
    ),
    QualificationOutputField(
        name="A1FREQ",
        data_type=QualificationOutputDataType.FLOAT32,
        nullable=False,
    ),
    QualificationOutputField(
        name="INFO",
        data_type=QualificationOutputDataType.FLOAT32,
        nullable=True,
    ),
    QualificationOutputField(
        name="N",
        data_type=QualificationOutputDataType.INT32,
        nullable=False,
    ),
    QualificationOutputField(
        name="BETA",
        data_type=QualificationOutputDataType.FLOAT32,
        nullable=False,
    ),
    QualificationOutputField(
        name="SE",
        data_type=QualificationOutputDataType.FLOAT32,
        nullable=False,
    ),
    QualificationOutputField(
        name="CHISQ",
        data_type=QualificationOutputDataType.FLOAT32,
        nullable=False,
    ),
    QualificationOutputField(
        name="LOG10P",
        data_type=QualificationOutputDataType.FLOAT32,
        nullable=False,
    ),
    QualificationOutputField(
        name="CORRECTION_METHOD",
        data_type=QualificationOutputDataType.STRING,
        nullable=False,
    ),
    QualificationOutputField(
        name="CORRECTION_STATUS",
        data_type=QualificationOutputDataType.STRING,
        nullable=False,
    ),
)
BASELINE_RESULT_SCHEMA = (
    ("CHROM", "Int64"),
    ("GENPOS", "Int64"),
    ("ID", "String"),
    ("ALLELE0", "String"),
    ("ALLELE1", "String"),
    ("A1FREQ", "Float64"),
    ("INFO", "Int64"),
    ("N", "Int64"),
    ("TEST", "String"),
    ("BETA", "Float64"),
    ("SE", "Float64"),
    ("CHISQ", "Float64"),
    ("LOG10P", "Float64"),
    ("EXTRA", "String"),
)


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
class CompletedParquetDataset:
    """One unambiguous production dataset selected from native CLI output."""

    output_root: Path
    directory: Path
    completion_line: str
    parquet_paths: tuple[Path, ...]


@dataclass(frozen=True)
class RegenieCorrectionSummary:
    """Aggregate approximate-Firth counts parsed from an upstream log."""

    correction_count: int
    correction_failure_count: int


@dataclass(frozen=True)
class QualificationNativeBuild:
    """Native extension identity required by an exact-source qualification."""

    git_commit: str
    science_source_sha256: str
    source_clean: bool
    profile: NativeBuildProfile
    run_nonce: str
    library_sha256: str
    library_size_bytes: int


@dataclass(frozen=True)
class QualificationToolEvidence:
    """Absolute executable identity supplied by the trusted bootstrap."""

    path: str
    sha256: str
    version: str


@dataclass(frozen=True)
class QualificationToolchainEvidence:
    """Host tools trusted to create and execute the qualification checkout."""

    bash: QualificationToolEvidence
    ar: QualificationToolEvidence
    assembler: QualificationToolEvidence
    cc: QualificationToolEvidence
    cc1: QualificationToolEvidence
    cc1plus: QualificationToolEvidence
    cargo: QualificationToolEvidence
    collect2: QualificationToolEvidence
    cxx: QualificationToolEvidence
    environment: QualificationToolEvidence
    git: QualificationToolEvidence
    just: QualificationToolEvidence
    maturin: QualificationToolEvidence
    mold: QualificationToolEvidence
    python: QualificationToolEvidence
    ranlib: QualificationToolEvidence
    rustc: QualificationToolEvidence
    scontrol: QualificationToolEvidence
    uv: QualificationToolEvidence
    venv_python: QualificationToolEvidence


@dataclass(frozen=True)
class QualificationDeviceEvidence:
    """Sanitized identity of the JAX CUDA devices used for qualification."""

    platform: str
    device_kind: str
    device_count: int
    backend_platform_version: str
    nvidia_driver_version: str
    cuda_runtime_version: str


@dataclass(frozen=True)
class QualificationStatisticEvidence:
    """Observed numerical evidence for one exclusive upstream comparison."""

    observed_column: str
    baseline_column: str
    maximum_absolute_difference: float
    absolute_tolerance: float


@dataclass(frozen=True)
class QualificationEvidence:
    """Sanitized evidence required to qualify an exact source externally."""

    passed: bool
    qualified_git_commit: str
    science_source_sha256: str
    working_tree_clean: bool
    qualification_generated_at_utc: str
    qualification_node: str
    slurm_job_id: str
    slurm_step_id: str
    run_nonce: str
    run_started_at_utc: str
    bootstrap_relative_path: str
    bootstrap_sha256: str
    toolchain: QualificationToolchainEvidence
    cargo_lock_sha256: str
    cargo_configuration_sha256: str
    rust_toolchain_sha256: str
    rustflags_environment: str
    cargo_encoded_rustflags_environment: str
    rustc_wrapper_environment: str
    cargo_build_rustc_wrapper_environment: str
    uv_lock_sha256: str
    jax_version: str
    jaxlib_version: str
    configured_device: str
    actual_device: QualificationDeviceEvidence
    native_build: QualificationNativeBuild
    observed_row_count: int
    observed_output_sha256: str
    output_fields: tuple[QualificationOutputField, ...]
    statistics: tuple[QualificationStatisticEvidence, ...]
    observed_correction_count: int
    observed_correction_failure_count: int
    exact_variant_keys: bool
    exact_sample_counts: bool
    exact_nonfinite_classes: bool
    exact_significance_classifications: bool


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
    qualification_hosts: tuple[str, ...]
    documentation_paths: tuple[Path, ...]
    tolerances: tuple[StatisticTolerance, ...]
    qualification: QualificationEvidence | None


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
    data_frame = pl.read_csv(table_path, separator=" ", null_values="NA")
    assert_data_frame_schema(data_frame, expected_schema=BASELINE_RESULT_SCHEMA, label="REGENIE")
    return data_frame


def read_direct_parquet_dataset(dataset: CompletedParquetDataset) -> pl.DataFrame:
    """Read the sole phenotype's production Parquet-parts dataset.

    Args:
        dataset: Dataset selected from the native CLI completion contract.

    Raises:
        AssertionError: If the direct Parquet-parts contract is invalid.
    """
    assert_direct_parquet_schema(dataset.parquet_paths)
    data_frame = pl.concat((pl.read_parquet(path) for path in dataset.parquet_paths), how="vertical")
    expected_schema = tuple((field.name, field.data_type.value) for field in PRODUCTION_OUTPUT_FIELDS)
    assert_data_frame_schema(data_frame, expected_schema=expected_schema, label="g")
    return data_frame


def completed_parquet_dataset(
    output_root: Path,
    stdout_chunks: tuple[str, ...],
) -> CompletedParquetDataset:
    """Select exactly one direct dataset path from native completion output."""
    completion_lines = tuple(
        line for line in "".join(stdout_chunks).splitlines() if line.startswith(PARQUET_DATASET_COMPLETION_PREFIX)
    )
    if len(completion_lines) != 1:
        raise AssertionError(f"Expected one native Parquet completion line, found {len(completion_lines)}")
    completion_line = completion_lines[0]
    dataset_path_text = completion_line.removeprefix(PARQUET_DATASET_COMPLETION_PREFIX).strip()
    dataset_path = Path(dataset_path_text)
    if not dataset_path_text or not dataset_path.is_absolute():
        raise AssertionError(f"Native Parquet completion path must be absolute: {completion_line}")
    resolved_output_root = output_root.resolve(strict=True)
    resolved_dataset_path = dataset_path.resolve(strict=True)
    if not resolved_dataset_path.is_dir():
        raise AssertionError(f"Native Parquet completion path is not a directory: {resolved_dataset_path}")
    try:
        dataset_relative_path = resolved_dataset_path.relative_to(resolved_output_root)
    except ValueError as error:
        raise AssertionError(
            f"Native Parquet completion path escapes the requested output root: {resolved_dataset_path}"
        ) from error
    if dataset_relative_path == Path():
        raise AssertionError("Native Parquet completion path must be below the requested output root")
    parquet_paths = direct_parquet_paths(resolved_dataset_path)
    return CompletedParquetDataset(
        output_root=resolved_output_root,
        directory=resolved_dataset_path,
        completion_line=completion_line,
        parquet_paths=parquet_paths,
    )


def direct_parquet_paths(dataset_directory: Path) -> tuple[Path, ...]:
    """Return ordered parts directly below one selected production dataset."""
    parquet_paths = tuple(sorted(dataset_directory.glob("*.parquet")))
    if not parquet_paths:
        raise AssertionError(f"No direct Parquet parts found below {dataset_directory}")
    for parquet_path in parquet_paths:
        if parquet_path.is_symlink() or not parquet_path.is_file():
            raise AssertionError(f"Parquet part is not a direct regular file: {parquet_path}")
    return parquet_paths


def assert_data_frame_schema(
    data_frame: pl.DataFrame,
    *,
    expected_schema: tuple[tuple[str, str], ...],
    label: str,
) -> None:
    """Require exact logical column order and data types."""
    observed_schema = tuple((column_name, str(data_type)) for column_name, data_type in data_frame.schema.items())
    if observed_schema != expected_schema:
        raise AssertionError(f"{label} result schema mismatch: expected {expected_schema}, observed {observed_schema}")


def assert_direct_parquet_schema(parquet_paths: tuple[Path, ...]) -> None:
    """Require every production part to retain the ordered Arrow contract."""
    arrow_data_type_names = {
        QualificationOutputDataType.STRING: "string",
        QualificationOutputDataType.INT64: "int64",
        QualificationOutputDataType.FLOAT32: "float",
        QualificationOutputDataType.INT32: "int32",
    }
    expected_schema = tuple(
        (field.name, arrow_data_type_names[field.data_type], field.nullable) for field in PRODUCTION_OUTPUT_FIELDS
    )
    for parquet_path in parquet_paths:
        arrow_schema = pyarrow.parquet.read_schema(parquet_path)
        observed_schema = tuple((field.name, str(field.type), field.nullable) for field in arrow_schema)
        if observed_schema != expected_schema:
            raise AssertionError(
                f"Production Parquet schema mismatch in {parquet_path}: "
                f"expected {expected_schema}, observed {observed_schema}"
            )


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
    if metadata.schema_version != PARITY_METADATA_SCHEMA_VERSION:
        raise AssertionError(
            f"Expected parity metadata schema {PARITY_METADATA_SCHEMA_VERSION}, got {metadata.schema_version}"
        )
    missing_identifiers = REQUIRED_WORKFLOW_IDENTIFIERS.difference(metadata.workflow_identifiers)
    unexpected_identifiers = metadata.workflow_identifiers.difference(REQUIRED_WORKFLOW_IDENTIFIERS)
    if missing_identifiers:
        missing_text = ", ".join(sorted(missing_identifiers))
        raise AssertionError(f"Missing REGENIE parity workflow metadata: {missing_text}")
    if unexpected_identifiers:
        unexpected_text = ", ".join(sorted(unexpected_identifiers))
        raise AssertionError(f"Unexpected REGENIE parity workflow metadata: {unexpected_text}")
    for workflow in metadata.workflows:
        if workflow.status != ParityWorkflowStatus.EXTERNAL_GOLDEN:
            raise AssertionError(f"Parity workflow is not backed by an external golden: {workflow.identifier}")
        if workflow.gate_status != ParityGateStatus.REQUIRED:
            raise AssertionError(f"Required REGENIE parity workflow lost required status: {workflow.identifier}")
        if workflow.qualification is not None:
            raise AssertionError(
                f"Required parity evidence must remain external to checked-in metadata: {workflow.identifier}"
            )
        if not workflow.validation_nodes:
            raise AssertionError(f"Parity workflow has no validation nodes: {workflow.identifier}")
        if workflow.qualification_hosts != REQUIRED_QUALIFICATION_HOSTS:
            raise AssertionError(
                f"Parity workflow has wrong qualification hosts: "
                f"{workflow.identifier} requires {REQUIRED_QUALIFICATION_HOSTS}"
            )
        if not workflow.tolerances:
            raise AssertionError(f"Parity workflow has no numerical tolerances: {workflow.identifier}")
        if set(workflow.input_sha256) != set(REQUIRED_INPUT_OPTION_NAMES):
            raise AssertionError(f"Parity workflow has incomplete input hashes: {workflow.identifier}")
        if not workflow.prediction_file_sha256:
            raise AssertionError(f"Parity workflow has no referenced prediction hashes: {workflow.identifier}")


def assert_workflow_qualification_is_current(
    workflow: GoldenWorkflow,
    qualification: QualificationEvidence,
    *,
    git_commit: str,
    science_source_sha256: str,
) -> None:
    """Validate one required workflow's exact-source qualification evidence."""
    if not qualification.passed:
        raise AssertionError(f"Qualification did not pass for {workflow.identifier}")
    if not qualification.working_tree_clean:
        raise AssertionError(f"Qualification used a dirty working tree for {workflow.identifier}")
    if qualification.qualified_git_commit != git_commit:
        raise AssertionError(
            f"Stale Git commit qualification for {workflow.identifier}: "
            f"expected {git_commit}, observed {qualification.qualified_git_commit}"
        )
    if qualification.qualification_node not in workflow.qualification_hosts:
        raise AssertionError(
            f"Qualification host is not allowed for {workflow.identifier}: "
            f"{qualification.qualification_node} not in {workflow.qualification_hosts}"
        )
    if SLURM_JOB_ID_PATTERN.fullmatch(qualification.slurm_job_id) is None:
        raise AssertionError(f"Qualification omitted a valid Slurm job ID for {workflow.identifier}")
    if SLURM_JOB_ID_PATTERN.fullmatch(qualification.slurm_step_id) is None:
        raise AssertionError(f"Qualification omitted a valid Slurm step ID for {workflow.identifier}")
    if RUN_NONCE_PATTERN.fullmatch(qualification.run_nonce) is None:
        raise AssertionError(f"Qualification omitted a valid run nonce for {workflow.identifier}")
    if qualification.bootstrap_relative_path != QUALIFICATION_BOOTSTRAP_RELATIVE_PATH:
        raise AssertionError(f"Qualification used an unknown bootstrap for {workflow.identifier}")
    if SHA256_PATTERN.fullmatch(qualification.bootstrap_sha256) is None:
        raise AssertionError(f"Qualification omitted a valid bootstrap digest for {workflow.identifier}")
    for tool_name, tool_evidence in (
        ("bash", qualification.toolchain.bash),
        ("ar", qualification.toolchain.ar),
        ("as", qualification.toolchain.assembler),
        ("cc", qualification.toolchain.cc),
        ("cargo", qualification.toolchain.cargo),
        ("cxx", qualification.toolchain.cxx),
        ("git", qualification.toolchain.git),
        ("just", qualification.toolchain.just),
        ("mold", qualification.toolchain.mold),
        ("python", qualification.toolchain.python),
        ("ranlib", qualification.toolchain.ranlib),
        ("rustc", qualification.toolchain.rustc),
        ("scontrol", qualification.toolchain.scontrol),
        ("uv", qualification.toolchain.uv),
    ):
        if not Path(tool_evidence.path).is_absolute():
            raise AssertionError(f"Qualification {tool_name} path is not absolute for {workflow.identifier}")
        if SHA256_PATTERN.fullmatch(tool_evidence.sha256) is None or not tool_evidence.version:
            raise AssertionError(f"Qualification omitted {tool_name} identity for {workflow.identifier}")
    try:
        run_started_at = parse_utc_datetime(
            qualification.run_started_at_utc,
            label="qualification.run_started_at_utc",
        )
        generated_at = parse_utc_datetime(
            qualification.qualification_generated_at_utc,
            label="qualification.qualification_generated_at_utc",
        )
    except ValueError as error:
        raise AssertionError(f"Qualification has invalid run timestamps for {workflow.identifier}") from error
    if generated_at < run_started_at:
        raise AssertionError(f"Qualification predates its run for {workflow.identifier}")
    if generated_at > datetime.datetime.now(datetime.UTC) + QUALIFICATION_CLOCK_SKEW:
        raise AssertionError(f"Qualification timestamp is implausibly in the future for {workflow.identifier}")
    if qualification.science_source_sha256 != science_source_sha256:
        raise AssertionError(
            f"Stale science-source qualification for {workflow.identifier}: "
            f"expected {science_source_sha256}, observed {qualification.science_source_sha256}"
        )
    native_build = qualification.native_build
    if native_build.git_commit != qualification.qualified_git_commit:
        raise AssertionError(f"Native extension commit differs from qualified commit for {workflow.identifier}")
    if native_build.science_source_sha256 != science_source_sha256:
        raise AssertionError(f"Native extension is stale for {workflow.identifier}")
    if not native_build.source_clean:
        raise AssertionError(f"Native extension was built from dirty source for {workflow.identifier}")
    if native_build.profile != NativeBuildProfile.RELEASE:
        raise AssertionError(f"Native extension is not a release build for {workflow.identifier}")
    if native_build.run_nonce != qualification.run_nonce:
        raise AssertionError(f"Native extension run nonce differs for {workflow.identifier}")
    if qualification.cargo_lock_sha256 != sha256_file(REPOSITORY_ROOT / "Cargo.lock"):
        raise AssertionError(f"Cargo.lock changed since qualification for {workflow.identifier}")
    if qualification.cargo_configuration_sha256 != sha256_file(REPOSITORY_ROOT / ".cargo" / "config.toml"):
        raise AssertionError(f"Cargo configuration changed since qualification for {workflow.identifier}")
    if qualification.rust_toolchain_sha256 != sha256_file(REPOSITORY_ROOT / "rust-toolchain.toml"):
        raise AssertionError(f"Rust toolchain selection changed since qualification for {workflow.identifier}")
    inherited_rust_overrides = (
        qualification.rustflags_environment,
        qualification.cargo_encoded_rustflags_environment,
        qualification.rustc_wrapper_environment,
        qualification.cargo_build_rustc_wrapper_environment,
    )
    if any(inherited_rust_overrides):
        raise AssertionError(f"Qualification inherited Rust flags or wrappers for {workflow.identifier}")
    if qualification.uv_lock_sha256 != sha256_file(REPOSITORY_ROOT / "uv.lock"):
        raise AssertionError(f"uv.lock changed since qualification for {workflow.identifier}")
    if qualification.jax_version != REQUIRED_JAX_VERSION or qualification.jaxlib_version != REQUIRED_JAX_VERSION:
        raise AssertionError(f"Unsupported JAX qualification versions for {workflow.identifier}")
    configured_device = str(workflow.g_cli_options["device"])
    if qualification.configured_device != configured_device:
        raise AssertionError(f"Qualified device changed for {workflow.identifier}")
    actual_device = qualification.actual_device
    if actual_device.platform not in {"cuda", "gpu"}:
        raise AssertionError(f"Qualification did not use a JAX GPU platform for {workflow.identifier}")
    if "cuda" not in actual_device.backend_platform_version.lower():
        raise AssertionError(f"Qualification did not use the JAX CUDA backend for {workflow.identifier}")
    if actual_device.device_count <= 0:
        raise AssertionError(f"Qualification observed no JAX CUDA devices for {workflow.identifier}")
    required_device_strings = (
        actual_device.device_kind,
        actual_device.nvidia_driver_version,
        actual_device.cuda_runtime_version,
    )
    if not all(required_device_strings):
        raise AssertionError(f"Qualification omitted CUDA device/runtime identity for {workflow.identifier}")
    if qualification.observed_row_count != workflow.expected_row_count:
        raise AssertionError(f"Qualified row count changed for {workflow.identifier}")
    if SHA256_PATTERN.fullmatch(qualification.observed_output_sha256) is None:
        raise AssertionError(f"Qualification omitted the output digest for {workflow.identifier}")
    if qualification.output_fields != PRODUCTION_OUTPUT_FIELDS:
        raise AssertionError(f"Qualified output schema/order/dtypes changed for {workflow.identifier}")
    expected_statistics = tuple(
        (
            tolerance.observed_column,
            tolerance.baseline_column,
            tolerance.absolute_tolerance,
        )
        for tolerance in workflow.tolerances
    )
    observed_statistics = tuple(
        (
            statistic.observed_column,
            statistic.baseline_column,
            statistic.absolute_tolerance,
        )
        for statistic in qualification.statistics
    )
    if observed_statistics != expected_statistics:
        raise AssertionError(f"Qualified statistic contract changed for {workflow.identifier}")
    for statistic in qualification.statistics:
        if not math.isfinite(statistic.maximum_absolute_difference):
            raise AssertionError(f"Qualified statistic is non-finite for {workflow.identifier}")
        if statistic.maximum_absolute_difference < 0.0:
            raise AssertionError(f"Qualified statistic difference is negative for {workflow.identifier}")
        if statistic.maximum_absolute_difference >= statistic.absolute_tolerance:
            raise AssertionError(
                f"Qualified statistic reached its exclusive tolerance for "
                f"{workflow.identifier}/{statistic.observed_column}"
            )
    expected_correction_count = workflow.expected_correction_count or 0
    expected_correction_failure_count = workflow.expected_correction_failure_count or 0
    if qualification.observed_correction_count != expected_correction_count:
        raise AssertionError(f"Qualified correction count changed for {workflow.identifier}")
    if qualification.observed_correction_failure_count != expected_correction_failure_count:
        raise AssertionError(f"Qualified correction failure count changed for {workflow.identifier}")
    exact_contracts = (
        qualification.exact_variant_keys,
        qualification.exact_sample_counts,
        qualification.exact_nonfinite_classes,
        qualification.exact_significance_classifications,
    )
    if not all(exact_contracts):
        raise AssertionError(f"Qualification omitted an exact comparison contract for {workflow.identifier}")


def load_golden_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> ParityMetadata:
    """Load and type the checked-in external parity metadata."""
    payload = typing.cast("dict[str, object]", json.loads(metadata_path.read_text(encoding="utf-8")))
    schema_version = parse_integer(payload["schema_version"], label="schema_version")
    if schema_version != PARITY_METADATA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported parity metadata schema version: {schema_version}")
    workflow_payloads = typing.cast("list[dict[str, object]]", payload["workflows"])
    regenie_reference = typing.cast("dict[str, object]", payload["regenie_reference"])
    workflows = tuple(parse_workflow_payload(workflow_payload) for workflow_payload in workflow_payloads)
    return ParityMetadata(
        schema_version=schema_version,
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
        qualification_hosts=parse_string_tuple(workflow_payload["qualification_hosts"]),
        documentation_paths=parse_repository_paths(workflow_payload["documentation_paths"]),
        tolerances=parse_tolerances(workflow_payload["tolerances"]),
        qualification=parse_qualification_evidence(workflow_payload["qualification"]),
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


def parse_qualification_evidence(value: object) -> QualificationEvidence | None:
    """Parse complete promotable evidence or an explicit pending value."""
    if value is None:
        return None
    payload = parse_mapping(value, label="qualification")
    require_mapping_fields(
        payload,
        label="qualification",
        expected_fields={
            "passed",
            "qualified_git_commit",
            "science_source_sha256",
            "working_tree_clean",
            "qualification_generated_at_utc",
            "qualification_node",
            "slurm_job_id",
            "slurm_step_id",
            "run_nonce",
            "run_started_at_utc",
            "bootstrap_relative_path",
            "bootstrap_sha256",
            "toolchain",
            "cargo_lock_sha256",
            "cargo_configuration_sha256",
            "rust_toolchain_sha256",
            "rustflags_environment",
            "cargo_encoded_rustflags_environment",
            "rustc_wrapper_environment",
            "cargo_build_rustc_wrapper_environment",
            "uv_lock_sha256",
            "jax_version",
            "jaxlib_version",
            "configured_device",
            "actual_device",
            "native_build",
            "observed_row_count",
            "observed_output_sha256",
            "output_fields",
            "statistics",
            "observed_correction_count",
            "observed_correction_failure_count",
            "exact_variant_keys",
            "exact_sample_counts",
            "exact_nonfinite_classes",
            "exact_significance_classifications",
        },
    )
    generated_at = parse_nonempty_string(
        payload["qualification_generated_at_utc"],
        label="qualification.qualification_generated_at_utc",
    )
    parsed_generated_at = parse_utc_datetime(
        generated_at,
        label="qualification.qualification_generated_at_utc",
    )
    run_started_at = parse_nonempty_string(
        payload["run_started_at_utc"],
        label="qualification.run_started_at_utc",
    )
    parsed_run_started_at = parse_utc_datetime(
        run_started_at,
        label="qualification.run_started_at_utc",
    )
    if parsed_generated_at < parsed_run_started_at:
        raise ValueError("qualification.qualification_generated_at_utc predates the run start")
    return QualificationEvidence(
        passed=parse_boolean(payload["passed"], label="qualification.passed"),
        qualified_git_commit=parse_git_commit(
            payload["qualified_git_commit"],
            label="qualification.qualified_git_commit",
        ),
        science_source_sha256=parse_sha256(
            payload["science_source_sha256"],
            label="qualification.science_source_sha256",
        ),
        working_tree_clean=parse_boolean(
            payload["working_tree_clean"],
            label="qualification.working_tree_clean",
        ),
        qualification_generated_at_utc=generated_at,
        qualification_node=parse_nonempty_string(
            payload["qualification_node"],
            label="qualification.qualification_node",
        ),
        slurm_job_id=parse_pattern_string(
            payload["slurm_job_id"],
            pattern=SLURM_JOB_ID_PATTERN,
            label="qualification.slurm_job_id",
        ),
        slurm_step_id=parse_pattern_string(
            payload["slurm_step_id"],
            pattern=SLURM_JOB_ID_PATTERN,
            label="qualification.slurm_step_id",
        ),
        run_nonce=parse_pattern_string(
            payload["run_nonce"],
            pattern=RUN_NONCE_PATTERN,
            label="qualification.run_nonce",
        ),
        run_started_at_utc=run_started_at,
        bootstrap_relative_path=parse_nonempty_string(
            payload["bootstrap_relative_path"],
            label="qualification.bootstrap_relative_path",
        ),
        bootstrap_sha256=parse_sha256(
            payload["bootstrap_sha256"],
            label="qualification.bootstrap_sha256",
        ),
        toolchain=parse_qualification_toolchain_evidence(payload["toolchain"]),
        cargo_lock_sha256=parse_sha256(
            payload["cargo_lock_sha256"],
            label="qualification.cargo_lock_sha256",
        ),
        cargo_configuration_sha256=parse_sha256(
            payload["cargo_configuration_sha256"],
            label="qualification.cargo_configuration_sha256",
        ),
        rust_toolchain_sha256=parse_sha256(
            payload["rust_toolchain_sha256"],
            label="qualification.rust_toolchain_sha256",
        ),
        rustflags_environment=parse_string(
            payload["rustflags_environment"],
            label="qualification.rustflags_environment",
        ),
        cargo_encoded_rustflags_environment=parse_string(
            payload["cargo_encoded_rustflags_environment"],
            label="qualification.cargo_encoded_rustflags_environment",
        ),
        rustc_wrapper_environment=parse_string(
            payload["rustc_wrapper_environment"],
            label="qualification.rustc_wrapper_environment",
        ),
        cargo_build_rustc_wrapper_environment=parse_string(
            payload["cargo_build_rustc_wrapper_environment"],
            label="qualification.cargo_build_rustc_wrapper_environment",
        ),
        uv_lock_sha256=parse_sha256(
            payload["uv_lock_sha256"],
            label="qualification.uv_lock_sha256",
        ),
        jax_version=parse_nonempty_string(payload["jax_version"], label="qualification.jax_version"),
        jaxlib_version=parse_nonempty_string(
            payload["jaxlib_version"],
            label="qualification.jaxlib_version",
        ),
        configured_device=parse_nonempty_string(
            payload["configured_device"],
            label="qualification.configured_device",
        ),
        actual_device=parse_qualification_device_evidence(payload["actual_device"]),
        native_build=parse_qualification_native_build(payload["native_build"]),
        observed_row_count=parse_nonnegative_integer(
            payload["observed_row_count"],
            label="qualification.observed_row_count",
        ),
        observed_output_sha256=parse_sha256(
            payload["observed_output_sha256"],
            label="qualification.observed_output_sha256",
        ),
        output_fields=parse_qualification_output_fields(payload["output_fields"]),
        statistics=parse_qualification_statistics(payload["statistics"]),
        observed_correction_count=parse_nonnegative_integer(
            payload["observed_correction_count"],
            label="qualification.observed_correction_count",
        ),
        observed_correction_failure_count=parse_nonnegative_integer(
            payload["observed_correction_failure_count"],
            label="qualification.observed_correction_failure_count",
        ),
        exact_variant_keys=parse_boolean(
            payload["exact_variant_keys"],
            label="qualification.exact_variant_keys",
        ),
        exact_sample_counts=parse_boolean(
            payload["exact_sample_counts"],
            label="qualification.exact_sample_counts",
        ),
        exact_nonfinite_classes=parse_boolean(
            payload["exact_nonfinite_classes"],
            label="qualification.exact_nonfinite_classes",
        ),
        exact_significance_classifications=parse_boolean(
            payload["exact_significance_classifications"],
            label="qualification.exact_significance_classifications",
        ),
    )


def qualification_evidence_payload(qualification: QualificationEvidence) -> dict[str, object]:
    """Serialize typed qualification evidence for checked-in metadata."""
    return {
        "passed": qualification.passed,
        "qualified_git_commit": qualification.qualified_git_commit,
        "science_source_sha256": qualification.science_source_sha256,
        "working_tree_clean": qualification.working_tree_clean,
        "qualification_generated_at_utc": qualification.qualification_generated_at_utc,
        "qualification_node": qualification.qualification_node,
        "slurm_job_id": qualification.slurm_job_id,
        "slurm_step_id": qualification.slurm_step_id,
        "run_nonce": qualification.run_nonce,
        "run_started_at_utc": qualification.run_started_at_utc,
        "bootstrap_relative_path": qualification.bootstrap_relative_path,
        "bootstrap_sha256": qualification.bootstrap_sha256,
        "toolchain": qualification_toolchain_evidence_payload(qualification.toolchain),
        "cargo_lock_sha256": qualification.cargo_lock_sha256,
        "cargo_configuration_sha256": qualification.cargo_configuration_sha256,
        "rust_toolchain_sha256": qualification.rust_toolchain_sha256,
        "rustflags_environment": qualification.rustflags_environment,
        "cargo_encoded_rustflags_environment": qualification.cargo_encoded_rustflags_environment,
        "rustc_wrapper_environment": qualification.rustc_wrapper_environment,
        "cargo_build_rustc_wrapper_environment": qualification.cargo_build_rustc_wrapper_environment,
        "uv_lock_sha256": qualification.uv_lock_sha256,
        "jax_version": qualification.jax_version,
        "jaxlib_version": qualification.jaxlib_version,
        "configured_device": qualification.configured_device,
        "actual_device": {
            "platform": qualification.actual_device.platform,
            "device_kind": qualification.actual_device.device_kind,
            "device_count": qualification.actual_device.device_count,
            "backend_platform_version": qualification.actual_device.backend_platform_version,
            "nvidia_driver_version": qualification.actual_device.nvidia_driver_version,
            "cuda_runtime_version": qualification.actual_device.cuda_runtime_version,
        },
        "native_build": {
            "git_commit": qualification.native_build.git_commit,
            "science_source_sha256": qualification.native_build.science_source_sha256,
            "source_clean": qualification.native_build.source_clean,
            "profile": qualification.native_build.profile.value,
            "run_nonce": qualification.native_build.run_nonce,
            "library_sha256": qualification.native_build.library_sha256,
            "library_size_bytes": qualification.native_build.library_size_bytes,
        },
        "observed_row_count": qualification.observed_row_count,
        "observed_output_sha256": qualification.observed_output_sha256,
        "output_fields": [
            {
                "name": field.name,
                "data_type": field.data_type.value,
                "nullable": field.nullable,
            }
            for field in qualification.output_fields
        ],
        "statistics": [
            {
                "observed_column": statistic.observed_column,
                "baseline_column": statistic.baseline_column,
                "maximum_absolute_difference": statistic.maximum_absolute_difference,
                "absolute_tolerance": statistic.absolute_tolerance,
            }
            for statistic in qualification.statistics
        ],
        "observed_correction_count": qualification.observed_correction_count,
        "observed_correction_failure_count": qualification.observed_correction_failure_count,
        "exact_variant_keys": qualification.exact_variant_keys,
        "exact_sample_counts": qualification.exact_sample_counts,
        "exact_nonfinite_classes": qualification.exact_nonfinite_classes,
        "exact_significance_classifications": qualification.exact_significance_classifications,
    }


def qualification_tool_evidence_payload(tool: QualificationToolEvidence) -> dict[str, str]:
    """Serialize one trusted host executable identity."""
    return {
        "path": tool.path,
        "sha256": tool.sha256,
        "version": tool.version,
    }


def qualification_toolchain_evidence_payload(
    toolchain: QualificationToolchainEvidence,
) -> dict[str, object]:
    """Serialize the fixed qualification toolchain identity."""
    return {
        "bash": qualification_tool_evidence_payload(toolchain.bash),
        "ar": qualification_tool_evidence_payload(toolchain.ar),
        "as": qualification_tool_evidence_payload(toolchain.assembler),
        "cc": qualification_tool_evidence_payload(toolchain.cc),
        "cc1": qualification_tool_evidence_payload(toolchain.cc1),
        "cc1plus": qualification_tool_evidence_payload(toolchain.cc1plus),
        "cargo": qualification_tool_evidence_payload(toolchain.cargo),
        "collect2": qualification_tool_evidence_payload(toolchain.collect2),
        "cxx": qualification_tool_evidence_payload(toolchain.cxx),
        "env": qualification_tool_evidence_payload(toolchain.environment),
        "git": qualification_tool_evidence_payload(toolchain.git),
        "just": qualification_tool_evidence_payload(toolchain.just),
        "maturin": qualification_tool_evidence_payload(toolchain.maturin),
        "mold": qualification_tool_evidence_payload(toolchain.mold),
        "python": qualification_tool_evidence_payload(toolchain.python),
        "ranlib": qualification_tool_evidence_payload(toolchain.ranlib),
        "rustc": qualification_tool_evidence_payload(toolchain.rustc),
        "scontrol": qualification_tool_evidence_payload(toolchain.scontrol),
        "uv": qualification_tool_evidence_payload(toolchain.uv),
        "venv_python": qualification_tool_evidence_payload(toolchain.venv_python),
    }


def parse_qualification_tool_evidence(value: object, *, label: str) -> QualificationToolEvidence:
    """Parse one absolute executable identity."""
    payload = parse_mapping(value, label=label)
    require_mapping_fields(
        payload,
        label=label,
        expected_fields={"path", "sha256", "version"},
    )
    path = parse_nonempty_string(payload["path"], label=f"{label}.path")
    if not Path(path).is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    return QualificationToolEvidence(
        path=path,
        sha256=parse_sha256(payload["sha256"], label=f"{label}.sha256"),
        version=parse_nonempty_string(payload["version"], label=f"{label}.version"),
    )


def parse_qualification_toolchain_evidence(value: object) -> QualificationToolchainEvidence:
    """Parse the fixed host toolchain trusted by qualification."""
    payload = parse_mapping(value, label="qualification.toolchain")
    require_mapping_fields(
        payload,
        label="qualification.toolchain",
        expected_fields={
            "bash",
            "ar",
            "as",
            "cc",
            "cc1",
            "cc1plus",
            "cargo",
            "collect2",
            "cxx",
            "env",
            "git",
            "just",
            "maturin",
            "mold",
            "python",
            "ranlib",
            "rustc",
            "scontrol",
            "uv",
            "venv_python",
        },
    )
    return QualificationToolchainEvidence(
        bash=parse_qualification_tool_evidence(
            payload["bash"],
            label="qualification.toolchain.bash",
        ),
        ar=parse_qualification_tool_evidence(
            payload["ar"],
            label="qualification.toolchain.ar",
        ),
        assembler=parse_qualification_tool_evidence(
            payload["as"],
            label="qualification.toolchain.as",
        ),
        cc=parse_qualification_tool_evidence(
            payload["cc"],
            label="qualification.toolchain.cc",
        ),
        cc1=parse_qualification_tool_evidence(
            payload["cc1"],
            label="qualification.toolchain.cc1",
        ),
        cc1plus=parse_qualification_tool_evidence(
            payload["cc1plus"],
            label="qualification.toolchain.cc1plus",
        ),
        cargo=parse_qualification_tool_evidence(
            payload["cargo"],
            label="qualification.toolchain.cargo",
        ),
        collect2=parse_qualification_tool_evidence(
            payload["collect2"],
            label="qualification.toolchain.collect2",
        ),
        cxx=parse_qualification_tool_evidence(
            payload["cxx"],
            label="qualification.toolchain.cxx",
        ),
        environment=parse_qualification_tool_evidence(
            payload["env"],
            label="qualification.toolchain.env",
        ),
        git=parse_qualification_tool_evidence(
            payload["git"],
            label="qualification.toolchain.git",
        ),
        just=parse_qualification_tool_evidence(
            payload["just"],
            label="qualification.toolchain.just",
        ),
        maturin=parse_qualification_tool_evidence(
            payload["maturin"],
            label="qualification.toolchain.maturin",
        ),
        mold=parse_qualification_tool_evidence(
            payload["mold"],
            label="qualification.toolchain.mold",
        ),
        python=parse_qualification_tool_evidence(
            payload["python"],
            label="qualification.toolchain.python",
        ),
        ranlib=parse_qualification_tool_evidence(
            payload["ranlib"],
            label="qualification.toolchain.ranlib",
        ),
        rustc=parse_qualification_tool_evidence(
            payload["rustc"],
            label="qualification.toolchain.rustc",
        ),
        scontrol=parse_qualification_tool_evidence(
            payload["scontrol"],
            label="qualification.toolchain.scontrol",
        ),
        uv=parse_qualification_tool_evidence(
            payload["uv"],
            label="qualification.toolchain.uv",
        ),
        venv_python=parse_qualification_tool_evidence(
            payload["venv_python"],
            label="qualification.toolchain.venv_python",
        ),
    )


def parse_qualification_device_evidence(value: object) -> QualificationDeviceEvidence:
    """Parse observed JAX CUDA device identity from qualification evidence."""
    payload = parse_mapping(value, label="qualification.actual_device")
    require_mapping_fields(
        payload,
        label="qualification.actual_device",
        expected_fields={
            "platform",
            "device_kind",
            "device_count",
            "backend_platform_version",
            "nvidia_driver_version",
            "cuda_runtime_version",
        },
    )
    return QualificationDeviceEvidence(
        platform=parse_nonempty_string(
            payload["platform"],
            label="qualification.actual_device.platform",
        ),
        device_kind=parse_nonempty_string(
            payload["device_kind"],
            label="qualification.actual_device.device_kind",
        ),
        device_count=parse_nonnegative_integer(
            payload["device_count"],
            label="qualification.actual_device.device_count",
        ),
        backend_platform_version=parse_nonempty_string(
            payload["backend_platform_version"],
            label="qualification.actual_device.backend_platform_version",
        ),
        nvidia_driver_version=parse_nonempty_string(
            payload["nvidia_driver_version"],
            label="qualification.actual_device.nvidia_driver_version",
        ),
        cuda_runtime_version=parse_nonempty_string(
            payload["cuda_runtime_version"],
            label="qualification.actual_device.cuda_runtime_version",
        ),
    )


def parse_qualification_native_build(value: object) -> QualificationNativeBuild:
    """Parse native build identity from qualification evidence."""
    payload = parse_mapping(value, label="qualification.native_build")
    require_mapping_fields(
        payload,
        label="qualification.native_build",
        expected_fields={
            "git_commit",
            "science_source_sha256",
            "source_clean",
            "profile",
            "run_nonce",
            "library_sha256",
            "library_size_bytes",
        },
    )
    library_size_bytes = parse_nonnegative_integer(
        payload["library_size_bytes"],
        label="qualification.native_build.library_size_bytes",
    )
    if library_size_bytes == 0:
        raise ValueError("qualification.native_build.library_size_bytes must be positive")
    return QualificationNativeBuild(
        git_commit=parse_git_commit(
            payload["git_commit"],
            label="qualification.native_build.git_commit",
        ),
        science_source_sha256=parse_sha256(
            payload["science_source_sha256"],
            label="qualification.native_build.science_source_sha256",
        ),
        source_clean=parse_boolean(
            payload["source_clean"],
            label="qualification.native_build.source_clean",
        ),
        profile=NativeBuildProfile(
            parse_nonempty_string(payload["profile"], label="qualification.native_build.profile")
        ),
        run_nonce=parse_pattern_string(
            payload["run_nonce"],
            pattern=RUN_NONCE_PATTERN,
            label="qualification.native_build.run_nonce",
        ),
        library_sha256=parse_sha256(
            payload["library_sha256"],
            label="qualification.native_build.library_sha256",
        ),
        library_size_bytes=library_size_bytes,
    )


def parse_qualification_output_fields(value: object) -> tuple[QualificationOutputField, ...]:
    """Parse ordered qualified output schema fields."""
    payloads = parse_list(value, label="qualification.output_fields")
    fields: list[QualificationOutputField] = []
    for field_index, field_value in enumerate(payloads):
        label = f"qualification.output_fields[{field_index}]"
        payload = parse_mapping(field_value, label=label)
        require_mapping_fields(payload, label=label, expected_fields={"name", "data_type", "nullable"})
        fields.append(
            QualificationOutputField(
                name=parse_nonempty_string(payload["name"], label=f"{label}.name"),
                data_type=QualificationOutputDataType(
                    parse_nonempty_string(payload["data_type"], label=f"{label}.data_type")
                ),
                nullable=parse_boolean(payload["nullable"], label=f"{label}.nullable"),
            )
        )
    if not fields:
        raise ValueError("qualification.output_fields must not be empty")
    return tuple(fields)


def parse_qualification_statistics(value: object) -> tuple[QualificationStatisticEvidence, ...]:
    """Parse ordered observed numerical evidence."""
    payloads = parse_list(value, label="qualification.statistics")
    statistics: list[QualificationStatisticEvidence] = []
    for statistic_index, statistic_value in enumerate(payloads):
        label = f"qualification.statistics[{statistic_index}]"
        payload = parse_mapping(statistic_value, label=label)
        require_mapping_fields(
            payload,
            label=label,
            expected_fields={
                "observed_column",
                "baseline_column",
                "maximum_absolute_difference",
                "absolute_tolerance",
            },
        )
        statistics.append(
            QualificationStatisticEvidence(
                observed_column=parse_nonempty_string(
                    payload["observed_column"],
                    label=f"{label}.observed_column",
                ),
                baseline_column=parse_nonempty_string(
                    payload["baseline_column"],
                    label=f"{label}.baseline_column",
                ),
                maximum_absolute_difference=parse_finite_float(
                    payload["maximum_absolute_difference"],
                    label=f"{label}.maximum_absolute_difference",
                ),
                absolute_tolerance=parse_finite_float(
                    payload["absolute_tolerance"],
                    label=f"{label}.absolute_tolerance",
                ),
            )
        )
    if not statistics:
        raise ValueError("qualification.statistics must not be empty")
    return tuple(statistics)


def parse_mapping(value: object, *, label: str) -> dict[str, object]:
    """Require and type one JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return typing.cast("dict[str, object]", value)


def parse_list(value: object, *, label: str) -> list[object]:
    """Require and type one JSON array."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return typing.cast("list[object]", value)


def require_mapping_fields(
    payload: dict[str, object],
    *,
    label: str,
    expected_fields: set[str],
) -> None:
    """Reject missing or unknown fields in qualification evidence."""
    observed_fields = set(payload)
    missing_fields = expected_fields.difference(observed_fields)
    unknown_fields = observed_fields.difference(expected_fields)
    if missing_fields:
        raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing_fields))}")
    if unknown_fields:
        raise ValueError(f"{label} has unknown fields: {', '.join(sorted(unknown_fields))}")


def parse_boolean(value: object, *, label: str) -> bool:
    """Require one JSON boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def parse_integer(value: object, *, label: str) -> int:
    """Require one JSON integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def parse_nonnegative_integer(value: object, *, label: str) -> int:
    """Require one nonnegative JSON integer."""
    parsed_value = parse_integer(value, label=label)
    if parsed_value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed_value


def parse_finite_float(value: object, *, label: str) -> float:
    """Require one finite JSON number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{label} must be finite")
    return parsed_value


def parse_string(value: object, *, label: str) -> str:
    """Require one JSON string, including an explicitly empty value."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def parse_nonempty_string(value: object, *, label: str) -> str:
    """Require one nonempty JSON string."""
    parsed_value = parse_string(value, label=label)
    if not parsed_value:
        raise ValueError(f"{label} must be a nonempty string")
    return parsed_value


def parse_pattern_string(value: object, *, pattern: re.Pattern[str], label: str) -> str:
    """Require one nonempty string matching a closed lexical contract."""
    parsed_value = parse_nonempty_string(value, label=label)
    if pattern.fullmatch(parsed_value) is None:
        raise ValueError(f"{label} has an invalid format")
    return parsed_value


def parse_utc_datetime(value: str, *, label: str) -> datetime.datetime:
    """Parse an aware ISO-8601 timestamp whose offset is exactly UTC."""
    try:
        parsed_value = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed_value.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"{label} must include UTC offset")
    return parsed_value


def parse_sha256(value: object, *, label: str) -> str:
    """Require one lowercase hexadecimal SHA-256 digest."""
    parsed_value = parse_nonempty_string(value, label=label)
    if SHA256_PATTERN.fullmatch(parsed_value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return parsed_value


def parse_git_commit(value: object, *, label: str) -> str:
    """Require one full lowercase Git commit identifier."""
    parsed_value = parse_nonempty_string(value, label=label)
    if tooling.science_gate.GIT_COMMIT_PATTERN.fullmatch(parsed_value) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return parsed_value


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
