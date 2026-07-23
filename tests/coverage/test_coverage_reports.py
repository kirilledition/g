from __future__ import annotations

import json
import typing

import pytest

from tooling.debug import check_coverage_reports

if typing.TYPE_CHECKING:
    from pathlib import Path


def write_python_report(
    report_path: Path,
    *,
    lines_covered: int = 96,
    lines_valid: int = 100,
    branches_covered: int = 19,
    branches_valid: int = 20,
) -> None:
    report_path.write_text(
        (
            '<?xml version="1.0" ?>\n'
            f'<coverage lines-valid="{lines_valid}" lines-covered="{lines_covered}" '
            f'branches-valid="{branches_valid}" branches-covered="{branches_covered}"/>\n'
        ),
        encoding="utf-8",
    )


def rust_metric(*, covered: int, count: int) -> dict[str, int | float]:
    return {"covered": covered, "count": count, "percent": 100.0 * covered / count}


def write_rust_report(
    report_path: Path,
    *,
    line_covered: int = 80,
    region_covered: int = 78,
    function_covered: int = 73,
    binding_line_covered: int = 1,
) -> None:
    summary = {
        "functions": rust_metric(covered=function_covered, count=100),
        "lines": rust_metric(covered=line_covered, count=100),
        "regions": rust_metric(covered=region_covered, count=100),
    }
    files = [
        {
            "filename": f"/checkout/g/{binding_path.as_posix()}",
            "summary": {
                "functions": rust_metric(covered=1, count=1),
                "lines": rust_metric(covered=binding_line_covered, count=2),
                "regions": rust_metric(covered=1, count=2),
            },
        }
        for binding_path in check_coverage_reports.REQUIRED_BINDING_PATHS
    ]
    report_path.write_text(
        json.dumps(
            {
                "data": [{"files": files, "totals": summary}],
                "type": "llvm.coverage.json.export",
                "version": "2.0.1",
            },
        ),
        encoding="utf-8",
    )


def test_python_report_requires_branch_aware_floor(tmp_path: Path) -> None:
    report_path = tmp_path / "python.xml"
    write_python_report(report_path)

    summary = check_coverage_reports.validate_python_report(report_path)

    assert summary.combined_percent == pytest.approx(95.83333333333333)
    assert summary.branches.count == 20


def test_python_report_deliberately_rejects_below_floor(tmp_path: Path) -> None:
    report_path = tmp_path / "python.xml"
    write_python_report(report_path, lines_covered=94, branches_covered=19)

    with pytest.raises(check_coverage_reports.CoverageValidationError, match=r"below the required 95\.00%"):
        check_coverage_reports.validate_python_report(report_path)


def test_python_report_rejects_missing_branch_measurements(tmp_path: Path) -> None:
    report_path = tmp_path / "python.xml"
    write_python_report(report_path, branches_covered=0, branches_valid=0)

    with pytest.raises(check_coverage_reports.CoverageValidationError, match="branch coverage has no measurable items"):
        check_coverage_reports.validate_python_report(report_path)


def test_rust_report_requires_all_floors(tmp_path: Path) -> None:
    report_path = tmp_path / "rust.json"
    write_rust_report(report_path)

    summary = check_coverage_reports.validate_rust_report(report_path)

    assert summary.lines.percent == 80.0
    assert summary.regions.percent == 78.0
    assert summary.functions.percent == 73.0


@pytest.mark.parametrize(
    ("metric_name", "overrides", "expected_floor"),
    [
        ("line", {"line_covered": 77}, "78.00%"),
        ("region", {"region_covered": 76}, "77.00%"),
        ("function", {"function_covered": 71}, "72.00%"),
    ],
)
def test_rust_report_deliberately_rejects_each_below_floor(
    tmp_path: Path,
    metric_name: str,
    overrides: dict[str, int],
    expected_floor: str,
) -> None:
    report_path = tmp_path / "rust.json"
    write_rust_report(report_path, **overrides)

    with pytest.raises(
        check_coverage_reports.CoverageValidationError,
        match=rf"Rust {metric_name} coverage .* below the required {expected_floor}",
    ):
        check_coverage_reports.validate_rust_report(report_path)


def test_lcov_report_must_be_parseable_and_nonempty(tmp_path: Path) -> None:
    report_path = tmp_path / "rust.lcov"
    report_path.write_text("TN:\nSF:/checkout/g/src/lib.rs\nDA:1,1\nDA:2,0\nend_of_record\n", encoding="utf-8")

    summary = check_coverage_reports.validate_lcov_report(report_path)

    assert summary.source_files == 1
    assert summary.line_records == 2
    assert summary.covered_line_records == 1


def test_lcov_report_rejects_empty_execution(tmp_path: Path) -> None:
    report_path = tmp_path / "rust.lcov"
    report_path.write_text("TN:\nSF:/checkout/g/src/lib.rs\nDA:1,0\nend_of_record\n", encoding="utf-8")

    with pytest.raises(check_coverage_reports.CoverageValidationError, match="no executed source-line coverage"):
        check_coverage_reports.validate_lcov_report(report_path)


def test_binding_report_requires_every_file_to_execute(tmp_path: Path) -> None:
    report_path = tmp_path / "bindings.json"
    write_rust_report(report_path)

    observed = check_coverage_reports.validate_binding_report(report_path)

    assert tuple(binding.path for binding in observed) == check_coverage_reports.REQUIRED_BINDING_PATHS
    assert all(binding.lines.covered == 1 for binding in observed)


def test_binding_report_rejects_zero_line_execution(tmp_path: Path) -> None:
    report_path = tmp_path / "bindings.json"
    write_rust_report(report_path, binding_line_covered=0)

    with pytest.raises(check_coverage_reports.CoverageValidationError, match="executed zero lines"):
        check_coverage_reports.validate_binding_report(report_path)


def test_report_parsers_reject_malformed_payloads(tmp_path: Path) -> None:
    python_path = tmp_path / "python.xml"
    rust_path = tmp_path / "rust.json"
    python_path.write_text("<coverage>", encoding="utf-8")
    rust_path.write_text("{", encoding="utf-8")

    with pytest.raises(check_coverage_reports.CoverageValidationError, match="Cannot parse Python"):
        check_coverage_reports.validate_python_report(python_path)
    with pytest.raises(check_coverage_reports.CoverageValidationError, match="Cannot parse Rust"):
        check_coverage_reports.validate_rust_report(rust_path)


def test_binding_report_rejects_malformed_file_entries(tmp_path: Path) -> None:
    report_path = tmp_path / "bindings.json"
    report_path.write_text(
        json.dumps(
            {
                "data": [{"files": [None], "totals": {}}],
                "type": "llvm.coverage.json.export",
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(check_coverage_reports.CoverageValidationError, match="malformed LLVM source-file entry"):
        check_coverage_reports.validate_binding_report(report_path)
