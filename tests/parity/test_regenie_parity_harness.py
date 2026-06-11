from __future__ import annotations

import polars as pl
import pytest

import tests.parity.harness


def test_golden_metadata_covers_required_workflows() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    tests.parity.harness.assert_metadata_covers_required_workflows(metadata)
    assert metadata.schema_version == 1
    assert metadata.regenie_reference["version"] == "v4.1"


def test_upstream_golden_workflows_record_commands_outputs_and_tolerances() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    expected_statistic_columns = {"BETA", "SE", "CHISQ", "LOG10P"}

    golden_workflows = [
        workflow
        for workflow in metadata.workflows
        if workflow.status == tests.parity.harness.ParityWorkflowStatus.EXTERNAL_GOLDEN
    ]
    assert golden_workflows
    for workflow in golden_workflows:
        assert workflow.regenie_version == "v4.1"
        assert workflow.regenie_commands
        assert workflow.expected_output_path is not None
        assert {tolerance.observed_column for tolerance in workflow.tolerances} == expected_statistic_columns
        assert all(tolerance.absolute_tolerance > 0.0 for tolerance in workflow.tolerances)


def test_validation_node_files_exist() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    for workflow in metadata.workflows:
        for validation_node in workflow.validation_nodes:
            validation_path_text = validation_node.split("::", maxsplit=1)[0]
            validation_path = tests.parity.harness.REPOSITORY_ROOT / validation_path_text
            assert validation_path.exists(), validation_node


def test_approximate_firth_is_experimental_until_upstream_golden_exists() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    workflow = metadata.workflow_by_identifier("binary_approximate_firth")

    assert workflow.status == tests.parity.harness.ParityWorkflowStatus.EXPERIMENTAL
    assert workflow.expected_output_path is None
    assert any(path.match("*/documentation/public/compatibility.md") for path in workflow.documentation_paths)


def test_statistic_comparison_detects_drift() -> None:
    observed_results = pl.DataFrame({"ID": ["variant_a"], "BETA": [1.20]})
    baseline_results = pl.DataFrame({"ID": ["variant_a"], "baseline_beta": [1.00]})
    tolerance = tests.parity.harness.StatisticTolerance(
        observed_column="BETA",
        baseline_column="baseline_beta",
        absolute_tolerance=0.01,
    )

    with pytest.raises(AssertionError):
        tests.parity.harness.assert_statistic_columns_match(
            observed_results,
            baseline_results,
            join_column="ID",
            tolerance=tolerance,
            expected_row_count=1,
        )
