from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import tests.numerical
import tests.parity.harness


def variant_frame(*, beta_values: list[float], log10p_values: list[float]) -> pl.DataFrame:
    """Build two rows whose repeated ID still has unique full variant keys."""
    return pl.DataFrame(
        {
            "CHROM": ["22", "22"],
            "GENPOS": [100, 200],
            "ID": ["repeated_identifier", "repeated_identifier"],
            "ALLELE0": ["A", "C"],
            "ALLELE1": ["G", "T"],
            "BETA": beta_values,
            "LOG10P": log10p_values,
        }
    )


def test_golden_metadata_covers_only_required_external_workflows() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    tests.parity.harness.assert_metadata_covers_required_workflows(metadata)
    assert metadata.schema_version == 2
    assert metadata.regenie_reference["version"] == "v4.1"
    assert metadata.workflow_identifiers == tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS


def test_upstream_golden_workflows_record_commands_hashes_rows_and_tolerances() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    expected_statistic_columns = {"BETA", "SE", "CHISQ", "LOG10P"}

    for workflow in metadata.workflows:
        assert workflow.status == tests.parity.harness.ParityWorkflowStatus.EXTERNAL_GOLDEN
        assert workflow.regenie_version == "v4.1"
        assert workflow.regenie_commands
        assert len(workflow.expected_output_sha256) == 64
        assert len(workflow.expected_log_sha256) == 64
        assert workflow.expected_row_count == 418_943
        assert set(workflow.input_sha256) == set(tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES)
        assert all(len(input_sha256) == 64 for input_sha256 in workflow.input_sha256.values())
        assert all(int(input_sha256, 16) >= 0 for input_sha256 in workflow.input_sha256.values())
        assert workflow.prediction_file_sha256
        assert all(len(file_sha256) == 64 for file_sha256 in workflow.prediction_file_sha256.values())
        assert all(int(file_sha256, 16) >= 0 for file_sha256 in workflow.prediction_file_sha256.values())
        assert {tolerance.observed_column for tolerance in workflow.tolerances} == expected_statistic_columns
        assert all(tolerance.absolute_tolerance > 0.0 for tolerance in workflow.tolerances)

    assert (
        metadata.workflow_by_identifier("quantitative_single_bgen_loco").gate_status
        == tests.parity.harness.ParityGateStatus.BLOCKING
    )
    assert (
        metadata.workflow_by_identifier("binary_score_only").gate_status
        == tests.parity.harness.ParityGateStatus.BLOCKING
    )
    assert (
        metadata.workflow_by_identifier("binary_approximate_firth").gate_status
        == tests.parity.harness.ParityGateStatus.BLOCKING
    )


def test_native_cli_config_emits_binary_section_only_for_binary_traits(tmp_path: Path) -> None:
    import tests.test_regenie2_parity

    metadata = tests.parity.harness.load_golden_metadata()
    quantitative_config_path = tests.test_regenie2_parity.write_native_cli_config(
        metadata.workflow_by_identifier("quantitative_single_bgen_loco"),
        output_root=tmp_path / "quantitative-output",
        jax_cache_directory=tmp_path / "jax-cache",
    )
    binary_config_path = tests.test_regenie2_parity.write_native_cli_config(
        metadata.workflow_by_identifier("binary_score_only"),
        output_root=tmp_path / "binary-output",
        jax_cache_directory=tmp_path / "jax-cache",
    )

    quantitative_config = quantitative_config_path.read_text(encoding="utf-8")
    binary_config = binary_config_path.read_text(encoding="utf-8")
    assert "\n[binary]\n" not in quantitative_config
    assert "\n[binary]\n" in binary_config
    assert 'fallback_method = "score_only"' in binary_config
    assert "p_threshold = 0.05" in binary_config
    assert "firth_se = false" in binary_config


def test_validation_node_files_and_documentation_exist() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    for workflow in metadata.workflows:
        for validation_node in workflow.validation_nodes:
            validation_path_text, test_name = validation_node.split("::", maxsplit=1)
            validation_path = tests.parity.harness.REPOSITORY_ROOT / validation_path_text
            assert validation_path.exists(), validation_node
            assert f"def {test_name}(" in validation_path.read_text(encoding="utf-8"), validation_node
        for documentation_path in workflow.documentation_paths:
            assert documentation_path.exists(), documentation_path


def test_approximate_firth_records_existing_full_upstream_golden() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    workflow = metadata.workflow_by_identifier("binary_approximate_firth")

    assert workflow.status == tests.parity.harness.ParityWorkflowStatus.EXTERNAL_GOLDEN
    assert workflow.expected_output_relative_path == Path("baselines/regenie_step2_phenotype_binary.regenie")
    assert workflow.expected_output_sha256 == "0b9dc124525b6fec63e1b0d3f446263c05f690862235bd84f51b1b3c77b6ed72"
    assert workflow.expected_correction_count == 17_938
    assert workflow.expected_correction_failure_count == 0
    assert workflow.gate_status == tests.parity.harness.ParityGateStatus.BLOCKING
    assert workflow.qualification["g_commit"] == "68f831f9ba51e28140b281c786555e3af6c36d4f"
    assert workflow.qualification["native_library_sha256"] == (
        "8265e3ad6f5a59cf607941ec67c78f5529b56e82cf1f9caca9a8d43e129b378c"
    )
    assert workflow.qualification["artifact_retained"] is False
    observed_differences = workflow.qualification["observed_maximum_absolute_differences"]
    assert isinstance(observed_differences, dict)
    configured_tolerances = {
        tolerance.observed_column: tolerance.absolute_tolerance for tolerance in workflow.tolerances
    }
    for statistic_name, observed_difference in observed_differences.items():
        assert isinstance(statistic_name, str)
        assert isinstance(observed_difference, int | float)
        assert float(observed_difference) < configured_tolerances[statistic_name]


def test_score_only_records_new_full_upstream_golden() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    workflow = metadata.workflow_by_identifier("binary_score_only")

    assert workflow.gate_status == tests.parity.harness.ParityGateStatus.BLOCKING
    assert workflow.expected_output_relative_path == Path("baselines/regenie_step2_score_only_phenotype_binary.regenie")
    assert workflow.expected_output_sha256 == "ba7278541d211a8ca446f5af3d45beba06030ad40f8124651db3038c196dac33"
    assert workflow.expected_log_sha256 == "c4002866c86dd67ebe23fcb563f17488635b59547cc30baa3a8566730e2e0e5b"
    assert workflow.expected_correction_count is None
    assert workflow.expected_correction_failure_count is None
    assert workflow.qualification["reference_generated_on_node"] == "hilbert"
    assert workflow.qualification["reference_generation_start_time_from_log"] == "Mon Jul 20 13:45:00 2026"
    assert workflow.qualification["g_commit"] == "68f831f9ba51e28140b281c786555e3af6c36d4f"
    assert workflow.qualification["native_library_sha256"] == (
        "8265e3ad6f5a59cf607941ec67c78f5529b56e82cf1f9caca9a8d43e129b378c"
    )
    assert workflow.qualification["artifact_retained"] is False
    observed_differences = workflow.qualification["observed_maximum_absolute_differences"]
    assert isinstance(observed_differences, dict)
    configured_tolerances = {
        tolerance.observed_column: tolerance.absolute_tolerance for tolerance in workflow.tolerances
    }
    for statistic_name, observed_difference in observed_differences.items():
        assert isinstance(statistic_name, str)
        assert isinstance(observed_difference, int | float)
        assert float(observed_difference) < configured_tolerances[statistic_name]


def test_statistic_comparison_uses_full_variant_key_and_strict_tolerance() -> None:
    observed_results = variant_frame(beta_values=[1.25, 2.25], log10p_values=[2.0, 8.0])
    baseline_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    tolerance = tests.parity.harness.StatisticTolerance(
        observed_column="BETA",
        baseline_column="BETA",
        absolute_tolerance=0.5,
    )

    comparison = tests.parity.harness.assert_statistic_columns_match(
        observed_results,
        baseline_results,
        tolerance=tolerance,
        expected_row_count=2,
    )

    assert comparison.row_count == 2
    assert comparison.maximum_absolute_difference < tolerance.absolute_tolerance

    boundary_results = variant_frame(beta_values=[1.5, 2.0], log10p_values=[2.0, 8.0])
    with pytest.raises(AssertionError):
        tests.parity.harness.assert_statistic_columns_match(
            boundary_results,
            baseline_results,
            tolerance=tolerance,
            expected_row_count=2,
        )


def test_negative_differences_use_the_same_strict_absolute_boundary() -> None:
    baseline_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    tolerance = tests.parity.harness.StatisticTolerance(
        observed_column="BETA",
        baseline_column="BETA",
        absolute_tolerance=0.5,
    )
    within_results = variant_frame(beta_values=[0.75, 1.75], log10p_values=[2.0, 8.0])
    tests.parity.harness.assert_statistic_columns_match(
        within_results,
        baseline_results,
        tolerance=tolerance,
        expected_row_count=2,
    )

    for beta_value in (0.5, 0.49):
        outside_results = variant_frame(beta_values=[beta_value, 2.0], log10p_values=[2.0, 8.0])
        with pytest.raises(AssertionError):
            tests.parity.harness.assert_statistic_columns_match(
                outside_results,
                baseline_results,
                tolerance=tolerance,
                expected_row_count=2,
            )


def test_nonfinite_masks_match_exactly() -> None:
    reference_values = np.asarray([np.nan, np.inf, -np.inf, 1.0])
    tests.numerical.assert_absolute_difference_less_than(
        reference_values.copy(),
        reference_values,
        0.5,
    )

    mismatches = (
        np.asarray([0.0, np.inf, -np.inf, 1.0]),
        np.asarray([np.nan, -np.inf, -np.inf, 1.0]),
        np.asarray([np.nan, np.inf, np.inf, 1.0]),
    )
    for actual_values in mismatches:
        with pytest.raises(AssertionError):
            tests.numerical.assert_absolute_difference_less_than(
                actual_values,
                reference_values,
                0.5,
            )


def test_regenie_correction_summary_is_parsed_from_log(tmp_path: Path) -> None:
    log_path = tmp_path / "regenie.log"
    log_path.write_text(
        "Number of tests with Firth correction : 17938\nNumber of failed tests : (0/17938)\n",
        encoding="utf-8",
    )

    summary = tests.parity.harness.parse_regenie_correction_summary(log_path)

    assert summary == tests.parity.harness.RegenieCorrectionSummary(
        correction_count=17_938,
        correction_failure_count=0,
    )


def test_regenie_correction_summary_rejects_inconsistent_denominator(tmp_path: Path) -> None:
    log_path = tmp_path / "regenie.log"
    log_path.write_text(
        "Number of tests with Firth correction : 10\nNumber of failed tests : (1/9)\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="denominator mismatch"):
        tests.parity.harness.parse_regenie_correction_summary(log_path)


def test_regenie_score_only_log_has_no_correction_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "regenie.log"
    log_path.write_text("Number of ignored tests due to low MAC : 0\n", encoding="utf-8")

    assert tests.parity.harness.parse_regenie_correction_summary(log_path) is None


def test_prediction_list_members_are_resolved_inside_data_root(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    prediction_directory = data_directory / "baselines"
    prediction_directory.mkdir(parents=True)
    prediction_path = prediction_directory / "step1_1.loco"
    prediction_path.write_text("prediction data\n", encoding="utf-8")
    prediction_list_path = prediction_directory / "step1_pred.list"
    prediction_list_path.write_text(f"phenotype {prediction_path}\n", encoding="utf-8")

    resolved_paths = tests.parity.harness.resolve_prediction_files(
        prediction_list_path,
        data_directory=data_directory,
    )

    assert resolved_paths == {"baselines/step1_1.loco": prediction_path.resolve()}


def test_prediction_list_members_cannot_escape_data_root(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    outside_path = tmp_path / "outside.loco"
    outside_path.write_text("prediction data\n", encoding="utf-8")
    prediction_list_path = data_directory / "step1_pred.list"
    prediction_list_path.write_text(f"phenotype {outside_path}\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="escapes configured data directory"):
        tests.parity.harness.resolve_prediction_files(
            prediction_list_path,
            data_directory=data_directory,
        )


def test_missing_prediction_member_skips_optional_run_and_fails_required_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    workflow = dataclasses.replace(
        tests.parity.harness.load_golden_metadata().workflow_by_identifier("quantitative_single_bgen_loco"),
        expected_output_relative_path=Path("baselines/reference.regenie"),
        expected_log_relative_path=Path("baselines/reference.log"),
    )
    data_directory = tmp_path / "data"
    for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES:
        input_path = data_directory / str(workflow.g_cli_options[option_name])
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.touch()
    prediction_list_path = data_directory / str(workflow.g_cli_options["prediction_list"])
    prediction_list_path.write_text("phenotype missing_1.loco\n", encoding="utf-8")
    for relative_path in (workflow.expected_output_relative_path, workflow.expected_log_relative_path):
        artifact_path = data_directory / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.touch()

    monkeypatch.setattr(tests.test_regenie2_parity, "DATA_DIRECTORY", data_directory)
    monkeypatch.delenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(pytest.skip.Exception, match="Missing prediction file"):
        tests.test_regenie2_parity.require_or_skip_workflow_data(workflow)

    monkeypatch.setenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, "1")
    with pytest.raises(pytest.fail.Exception, match="Missing prediction file"):
        tests.test_regenie2_parity.require_or_skip_workflow_data(workflow)


def test_failure_reporting_never_masks_original_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    workflow = dataclasses.replace(
        tests.parity.harness.load_golden_metadata().workflow_by_identifier("quantitative_single_bgen_loco"),
        expected_row_count=2,
    )
    baseline_results = pl.DataFrame(
        {
            "CHROM": ["22", "22"],
            "GENPOS": [100, 200],
            "ID": ["first", "second"],
            "ALLELE0": ["A", "C"],
            "ALLELE1": ["G", "T"],
            "BETA": [1.0, 2.0],
            "SE": [0.1, 0.2],
            "CHISQ": [10.0, 20.0],
            "LOG10P": [2.0, 8.0],
            "N": [100, 100],
        }
    )
    observed_results = baseline_results.head(1).with_columns(
        pl.lit("score").alias("CORRECTION_METHOD"),
        pl.lit("success").alias("CORRECTION_STATUS"),
    )
    parity_results = tests.test_regenie2_parity.RegenieParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
        output_root=tmp_path / "output",
        config_path=tmp_path / "config.toml",
        observed_input_sha256={},
        observed_prediction_file_sha256={},
        reference_correction_summary=None,
    )

    def fail_report(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("report writer failed")

    monkeypatch.setattr(tests.test_regenie2_parity, "write_qualification_report", fail_report)
    with pytest.raises(AssertionError, match="Expected 2 observed rows"):
        tests.test_regenie2_parity.assert_and_record_workflow_qualification(workflow, parity_results)


def test_variant_key_order_must_match_exactly() -> None:
    observed_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    baseline_results = observed_results.reverse()

    with pytest.raises(AssertionError, match="row order"):
        tests.parity.harness.assert_variant_key_order_match(
            observed_results,
            baseline_results,
            expected_row_count=2,
        )


def test_significance_classifications_are_exact() -> None:
    baseline_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    matching_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[3.0, 9.0])

    tests.parity.harness.assert_significance_classifications_match(
        matching_results,
        baseline_results,
        expected_row_count=2,
    )

    mismatching_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[1.0, 9.0])
    with pytest.raises(AssertionError):
        tests.parity.harness.assert_significance_classifications_match(
            mismatching_results,
            baseline_results,
            expected_row_count=2,
        )
