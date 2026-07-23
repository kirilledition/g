#!/usr/bin/env python3
"""Validate authoritative Python and Rust coverage reports."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import typing
import xml.etree.ElementTree as ET
from pathlib import Path

PYTHON_MINIMUM_PERCENT = 95.0
RUST_LINE_MINIMUM_PERCENT = 78.0
RUST_REGION_MINIMUM_PERCENT = 77.0
RUST_FUNCTION_MINIMUM_PERCENT = 72.0
REQUIRED_BINDING_PATHS = (
    Path("src/lib.rs"),
    Path("src/binding/mod.rs"),
    Path("src/binding/cli.rs"),
    Path("src/binding/engine.rs"),
    Path("src/binding/jax_runtime.rs"),
    Path("src/binding/logging.rs"),
)


class CoverageValidationError(ValueError):
    """An authoritative coverage report violates its contract."""


class RustCoverageMetricPayload(typing.TypedDict):
    """One LLVM coverage metric from an exported JSON summary."""

    count: int
    covered: int
    percent: float


class RustCoverageSummaryPayload(typing.TypedDict):
    """LLVM coverage metrics required by the gate."""

    functions: RustCoverageMetricPayload
    lines: RustCoverageMetricPayload
    regions: RustCoverageMetricPayload


class RustCoverageFilePayload(typing.TypedDict):
    """One file entry from an LLVM coverage JSON export."""

    filename: str
    summary: RustCoverageSummaryPayload


class RustCoverageDataPayload(typing.TypedDict):
    """One data entry from an LLVM coverage JSON export."""

    files: list[RustCoverageFilePayload]
    totals: RustCoverageSummaryPayload


class RustCoverageReportPayload(typing.TypedDict):
    """Top-level LLVM coverage JSON export."""

    data: list[RustCoverageDataPayload]
    type: str


@dataclasses.dataclass(frozen=True)
class CoverageMetric:
    """One measured coverage metric."""

    covered: int
    count: int

    @property
    def percent(self) -> float:
        """Return the exact covered percentage."""
        return 100.0 * self.covered / self.count


@dataclasses.dataclass(frozen=True)
class PythonCoverageSummary:
    """Branch-aware Python coverage totals."""

    lines: CoverageMetric
    branches: CoverageMetric

    @property
    def combined_percent(self) -> float:
        """Return the combined line-and-branch percentage used by coverage.py."""
        covered = self.lines.covered + self.branches.covered
        count = self.lines.count + self.branches.count
        return 100.0 * covered / count


@dataclasses.dataclass(frozen=True)
class RustCoverageSummary:
    """Rust coverage totals enforced by the gate."""

    lines: CoverageMetric
    regions: CoverageMetric
    functions: CoverageMetric


@dataclasses.dataclass(frozen=True)
class LcovSummary:
    """Nonempty record counts from one LCOV report."""

    source_files: int
    line_records: int
    covered_line_records: int


@dataclasses.dataclass(frozen=True)
class BindingCoverage:
    """Observed line coverage for one required PyO3 binding source."""

    path: Path
    lines: CoverageMetric


def coverage_metric(covered: int, count: int, *, label: str) -> CoverageMetric:
    """Build and validate one nonempty coverage metric."""
    if count <= 0:
        raise CoverageValidationError(f"{label} has no measurable items")
    if covered < 0 or covered > count:
        raise CoverageValidationError(f"{label} has invalid covered/count values: {covered}/{count}")
    return CoverageMetric(covered=covered, count=count)


def require_minimum(metric_percent: float, minimum_percent: float, *, label: str) -> None:
    """Require an exact percentage to meet its configured floor."""
    if not math.isfinite(metric_percent):
        raise CoverageValidationError(f"{label} is not finite: {metric_percent}")
    if metric_percent < minimum_percent:
        raise CoverageValidationError(
            f"{label} {metric_percent:.2f}% is below the required {minimum_percent:.2f}%",
        )


def parse_integer_attribute(root: ET.Element, attribute_name: str, *, report_path: Path) -> int:
    """Parse one required integer attribute from a coverage.py XML root."""
    value = root.get(attribute_name)
    if value is None:
        raise CoverageValidationError(f"{report_path} has no {attribute_name!r} attribute")
    try:
        return int(value)
    except ValueError as error:
        raise CoverageValidationError(
            f"{report_path} has a non-integer {attribute_name!r} attribute: {value!r}",
        ) from error


def validate_python_report(
    report_path: Path,
    *,
    minimum_percent: float = PYTHON_MINIMUM_PERCENT,
) -> PythonCoverageSummary:
    """Validate a branch-aware coverage.py XML report and its total floor."""
    try:
        root = ET.parse(report_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise CoverageValidationError(f"Cannot parse Python coverage report {report_path}: {error}") from error
    if root.tag != "coverage":
        raise CoverageValidationError(f"{report_path} has unexpected root element {root.tag!r}")
    summary = PythonCoverageSummary(
        lines=coverage_metric(
            parse_integer_attribute(root, "lines-covered", report_path=report_path),
            parse_integer_attribute(root, "lines-valid", report_path=report_path),
            label="Python line coverage",
        ),
        branches=coverage_metric(
            parse_integer_attribute(root, "branches-covered", report_path=report_path),
            parse_integer_attribute(root, "branches-valid", report_path=report_path),
            label="Python branch coverage",
        ),
    )
    require_minimum(summary.combined_percent, minimum_percent, label="Python branch-aware coverage")
    return summary


def load_rust_report(report_path: Path) -> RustCoverageReportPayload:
    """Load and minimally validate one LLVM coverage JSON report."""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageValidationError(f"Cannot parse Rust coverage report {report_path}: {error}") from error
    if not isinstance(payload, dict):
        raise CoverageValidationError(f"{report_path} must contain a JSON object")
    report = typing.cast("RustCoverageReportPayload", payload)
    if report.get("type") != "llvm.coverage.json.export":
        raise CoverageValidationError(f"{report_path} is not an LLVM coverage JSON export")
    data = report.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise CoverageValidationError(f"{report_path} must contain exactly one LLVM coverage data entry")
    if not isinstance(data[0], dict):
        raise CoverageValidationError(f"{report_path} has a non-object LLVM coverage data entry")
    return report


def rust_metric(
    summary: RustCoverageSummaryPayload,
    metric_name: typing.Literal["functions", "lines", "regions"],
    *,
    label: str,
) -> CoverageMetric:
    """Read one metric from an LLVM coverage summary."""
    payload = summary.get(metric_name)
    if not isinstance(payload, dict):
        raise CoverageValidationError(f"{label} is missing from the LLVM coverage summary")
    covered = payload.get("covered")
    count = payload.get("count")
    if not isinstance(covered, int) or not isinstance(count, int):
        raise CoverageValidationError(f"{label} must contain integer covered/count values")
    return coverage_metric(covered, count, label=label)


def validate_rust_report(report_path: Path) -> RustCoverageSummary:
    """Validate an LLVM coverage JSON report and all Rust floors."""
    report = load_rust_report(report_path)
    totals = report["data"][0].get("totals")
    if not isinstance(totals, dict):
        raise CoverageValidationError(f"{report_path} has no LLVM coverage totals")
    summary = RustCoverageSummary(
        lines=rust_metric(totals, "lines", label="Rust line coverage"),
        regions=rust_metric(totals, "regions", label="Rust region coverage"),
        functions=rust_metric(totals, "functions", label="Rust function coverage"),
    )
    require_minimum(summary.lines.percent, RUST_LINE_MINIMUM_PERCENT, label="Rust line coverage")
    require_minimum(summary.regions.percent, RUST_REGION_MINIMUM_PERCENT, label="Rust region coverage")
    require_minimum(summary.functions.percent, RUST_FUNCTION_MINIMUM_PERCENT, label="Rust function coverage")
    return summary


def validate_lcov_report(report_path: Path) -> LcovSummary:
    """Require a parseable, nonempty LCOV report with executed lines."""
    try:
        lines = report_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CoverageValidationError(f"Cannot read LCOV report {report_path}: {error}") from error
    source_file_count = sum(1 for line in lines if line.startswith("SF:") and bool(line.removeprefix("SF:")))
    line_records = [line.removeprefix("DA:") for line in lines if line.startswith("DA:")]
    covered_line_records = 0
    for line_record in line_records:
        fields = line_record.split(",", maxsplit=2)
        if len(fields) < 2:
            raise CoverageValidationError(f"{report_path} contains malformed LCOV line record {line_record!r}")
        try:
            execution_count = int(fields[1])
        except ValueError as error:
            raise CoverageValidationError(
                f"{report_path} contains non-integer LCOV execution count {fields[1]!r}",
            ) from error
        covered_line_records += int(execution_count > 0)
    summary = LcovSummary(
        source_files=source_file_count,
        line_records=len(line_records),
        covered_line_records=covered_line_records,
    )
    if summary.source_files == 0 or summary.line_records == 0 or summary.covered_line_records == 0:
        raise CoverageValidationError(f"{report_path} contains no executed source-line coverage")
    return summary


def normalized_source_path(source_path: str) -> str:
    """Normalize path separators for portable suffix matching."""
    return source_path.replace("\\", "/")


def matching_binding_files(
    files: list[RustCoverageFilePayload],
    required_path: Path,
) -> list[RustCoverageFilePayload]:
    """Return LLVM file entries matching one repository-relative binding path."""
    required_text = required_path.as_posix()
    return [
        file_payload
        for file_payload in files
        if normalized_source_path(file_payload.get("filename", "")).endswith(f"/{required_text}")
        or normalized_source_path(file_payload.get("filename", "")) == required_text
    ]


def validate_binding_report(
    report_path: Path,
    *,
    required_paths: tuple[Path, ...] = REQUIRED_BINDING_PATHS,
) -> tuple[BindingCoverage, ...]:
    """Require nonzero line execution in every supported PyO3 binding file."""
    report = load_rust_report(report_path)
    files = report["data"][0].get("files")
    if not isinstance(files, list) or not files:
        raise CoverageValidationError(f"{report_path} contains no LLVM source-file entries")
    if any(
        not isinstance(file_payload, dict)
        or not isinstance(file_payload.get("filename"), str)
        or not isinstance(file_payload.get("summary"), dict)
        for file_payload in files
    ):
        raise CoverageValidationError(f"{report_path} contains a malformed LLVM source-file entry")
    typed_files = typing.cast("list[RustCoverageFilePayload]", files)
    observed: list[BindingCoverage] = []
    for required_path in required_paths:
        matching_files = matching_binding_files(typed_files, required_path)
        if len(matching_files) != 1:
            raise CoverageValidationError(
                f"{report_path} contains {len(matching_files)} entries for required binding {required_path}",
            )
        lines = rust_metric(
            matching_files[0]["summary"],
            "lines",
            label=f"binding line coverage for {required_path}",
        )
        if lines.covered == 0:
            raise CoverageValidationError(f"required binding {required_path} executed zero lines")
        observed.append(BindingCoverage(path=required_path, lines=lines))
    return tuple(observed)


def usage() -> str:
    """Return command-line usage text."""
    return (
        "Usage:\n"
        "  python -m tooling.debug.check_coverage_reports python REPORT.xml\n"
        "  python -m tooling.debug.check_coverage_reports rust REPORT.json REPORT.lcov\n"
        "  python -m tooling.debug.check_coverage_reports bindings REPORT.json"
    )


def main(arguments: list[str] | None = None) -> int:
    """Validate coverage reports selected by the command line."""
    resolved_arguments = list(sys.argv[1:] if arguments is None else arguments)
    try:
        if len(resolved_arguments) == 2 and resolved_arguments[0] == "python":
            summary = validate_python_report(Path(resolved_arguments[1]))
            print(f"Python branch-aware coverage: {summary.combined_percent:.2f}%")
            return 0
        if len(resolved_arguments) == 3 and resolved_arguments[0] == "rust":
            summary = validate_rust_report(Path(resolved_arguments[1]))
            lcov_summary = validate_lcov_report(Path(resolved_arguments[2]))
            print(
                "Rust coverage: "
                f"lines {summary.lines.percent:.2f}%, "
                f"regions {summary.regions.percent:.2f}%, "
                f"functions {summary.functions.percent:.2f}%; "
                f"LCOV sources {lcov_summary.source_files}",
            )
            return 0
        if len(resolved_arguments) == 2 and resolved_arguments[0] == "bindings":
            binding_coverage = validate_binding_report(Path(resolved_arguments[1]))
            for binding in binding_coverage:
                print(f"Binding coverage: {binding.path} {binding.lines.percent:.2f}%")
            return 0
    except CoverageValidationError as error:
        print(f"Coverage validation failed: {error}", file=sys.stderr)
        return 1
    print(usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
