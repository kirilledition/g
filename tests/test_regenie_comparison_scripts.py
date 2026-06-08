from __future__ import annotations

import dataclasses
import json
import sys
import typing
from pathlib import Path

import polars as pl

import scripts.benchmark as baseline_benchmark
import scripts.benchmark_regenie2_linear_fresh_process as fresh_process_benchmark
import scripts.benchmark_regenie_comparison as comparison_benchmark
import scripts.compare_binary_firth_paths as binary_firth_parity
import scripts.debug_binary_regenie_parity as binary_regenie_debug
import scripts.debug_linear_regenie_parity as linear_regenie_debug
import scripts.profile_regenie_comparison as comparison_profile
import tooling.cli.benchmark_bgen_reader as bgen_reader_benchmark
import tooling.cli.benchmark_output_stages as output_stage_benchmark
import tooling.cli.benchmark_regenie2_binary_hot as binary_hot_benchmark
import tooling.cli.profile_regenie2_deep as deep_profile
import tooling.cli.tune_regenie2_gpu as tuning_benchmark

if typing.TYPE_CHECKING:
    import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def test_regenie_command_builders_shape() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    command_specs = comparison_benchmark.build_regenie_program_specs("regenie", baseline_paths)
    assert len(command_specs) == 4
    assert command_specs[0][0] == "regenie_step1_binary"
    assert "--step" in command_specs[0][3]
    assert "--bt" in command_specs[0][3]
    assert command_specs[1][0] == "regenie_step2_binary"
    assert "--bgen" in command_specs[1][3]
    assert command_specs[2][0] == "regenie_step1_quantitative"
    assert "--qt" in command_specs[2][3]
    assert command_specs[3][0] == "regenie_step2_quantitative"
    assert "--pred" in command_specs[3][3]


def test_bgen_reader_benchmark_parses_sweep_lists() -> None:
    assert bgen_reader_benchmark.parse_optional_int_list("8192,16384") == [8192, 16384]
    assert bgen_reader_benchmark.parse_optional_int_list("default,4") == [None, 4]


def test_bgen_reader_benchmark_parses_path_modes() -> None:
    path_modes = bgen_reader_benchmark.parse_path_modes("variant_major_buffered")
    assert [path_mode.value for path_mode in path_modes] == [
        "variant_major_buffered",
    ]


def test_bgen_reader_benchmark_parses_boolean_modes() -> None:
    assert bgen_reader_benchmark.parse_boolean_mode_list("trusted,safe") == [True, False]


def test_binary_regenie_debug_selector_matches_ids_and_indices() -> None:
    selector = binary_regenie_debug.VariantSelector(
        variant_identifiers=frozenset({"rs1"}),
        variant_indices=frozenset({3}),
    )

    assert selector.matches(variant_identifier="rs1", variant_index=99)
    assert selector.matches(variant_identifier="rs2", variant_index=3)
    assert not selector.matches(variant_identifier="rs2", variant_index=4)


def test_linear_regenie_debug_selector_matches_ids_and_indices() -> None:
    selector = linear_regenie_debug.VariantSelector(
        variant_identifiers=frozenset({"rs1"}),
        variant_indices=frozenset({3}),
    )

    assert selector.matches(variant_identifier="rs1", variant_index=99)
    assert selector.matches(variant_identifier="rs2", variant_index=3)
    assert not selector.matches(variant_identifier="rs2", variant_index=4)


def test_linear_regenie_debug_comparison_reports_nested_numeric_differences() -> None:
    record = linear_regenie_debug.VariantDebugRecord(
        variant_index=0,
        chromosome="22",
        position=100,
        variant_identifier="rs1",
        allele_zero="A",
        allele_one="G",
        allele_count=2.0,
        allele_one_frequency=0.1,
        minor_allele_count=2.0,
        info_score=0.9,
        observation_count=10,
        sparse_candidate=False,
        normalization_offset=0.0,
        normalized_genotype_sum_squares=4.0,
        projection_sum_squares=1.0,
        genotype_residual_sum_squares=3.0,
        covariance_with_phenotype=0.5,
        null_mean_squared_error=1.5,
        adjusted_residual_sum_squares=15.0,
        adjusted_residual={"sum": 1.0},
        adjusted_residual_projection={"sum": 2.0},
        beta=0.1,
        standard_error=0.2,
        chi_squared=0.3,
        log10_p_value=0.4,
        valid=True,
    )

    comparisons = linear_regenie_debug.build_comparisons(
        records=[record],
        reference_records={
            "rs1": {
                "variant_identifier": "rs1",
                "beta": 0.2,
                "adjusted_residual": {"sum": 1.0},
                "genotype_residual_sum_squares": 3.0,
            }
        },
        tolerance=1.0e-8,
    )

    assert len(comparisons) == 1
    assert comparisons[0].variant_identifier == "rs1"
    assert not comparisons[0].missing_reference
    assert [difference.path for difference in comparisons[0].differences] == ["beta"]


def test_binary_regenie_debug_comparison_reports_nested_numeric_differences() -> None:
    record = binary_regenie_debug.VariantDebugRecord(
        variant_index=0,
        chromosome="22",
        position=100,
        variant_identifier="rs1",
        allele_zero="A",
        allele_one="G",
        allele_count=2.0,
        flipped_allele_count=2.0,
        flip_mask=False,
        minor_allele_count=2.0,
        sparse_candidate=False,
        rare_sparse_firth_candidate=False,
        carrier_count=1,
        score=0.5,
        score_variance=1.5,
        score_beta=0.1,
        score_standard_error=0.2,
        score_chi_squared=0.3,
        score_log10_p_value=0.4,
        score_extra_code="score",
        null_logistic_offset={"sum": 1.0},
        null_firth_offset={"sum": 2.0},
        null_logistic_iteration_count=4,
        null_logistic_converged=True,
        null_firth_iteration_count=5,
        null_firth_convergence_reason="converged",
        firth_correction_branch="pseudo_firth",
        firth_iteration_count=6,
        pseudo_firth_iteration_count=6,
        nr_zero_start_iteration_count=0,
        nr_warm_start_iteration_count=0,
        final_beta=0.11,
        final_standard_error=0.21,
        final_chi_squared=0.31,
        final_log10_p_value=0.41,
        final_extra_code="firth",
        final_valid=True,
        firth_failure_code="none",
        firth_convergence_reason="converged",
    )

    comparisons = binary_regenie_debug.build_comparisons(
        records=[record],
        reference_records={
            "rs1": {
                "variant_identifier": "rs1",
                "final_beta": 0.2,
                "null_firth_offset": {"sum": 2.0},
                "score_variance": 1.5,
            }
        },
        tolerance=1.0e-8,
    )

    assert len(comparisons) == 1
    assert comparisons[0].variant_identifier == "rs1"
    assert not comparisons[0].missing_reference
    assert [difference.path for difference in comparisons[0].differences] == ["final_beta"]


def test_binary_regenie_debug_missing_selection_count_handles_id_and_index_match() -> None:
    selector = binary_regenie_debug.VariantSelector(
        variant_identifiers=frozenset({"rs1", "missing_rs"}),
        variant_indices=frozenset({0, 99}),
    )
    record = binary_regenie_debug.VariantDebugRecord(
        variant_index=0,
        chromosome="22",
        position=100,
        variant_identifier="rs1",
        allele_zero="A",
        allele_one="G",
        allele_count=2.0,
        flipped_allele_count=2.0,
        flip_mask=False,
        minor_allele_count=2.0,
        sparse_candidate=False,
        rare_sparse_firth_candidate=False,
        carrier_count=1,
        score=0.5,
        score_variance=1.5,
        score_beta=0.1,
        score_standard_error=0.2,
        score_chi_squared=0.3,
        score_log10_p_value=0.4,
        score_extra_code="score",
        null_logistic_offset={"sum": 1.0},
        null_firth_offset={"sum": 2.0},
        null_logistic_iteration_count=4,
        null_logistic_converged=True,
        null_firth_iteration_count=5,
        null_firth_convergence_reason="converged",
        firth_correction_branch="pseudo_firth",
        firth_iteration_count=6,
        pseudo_firth_iteration_count=6,
        nr_zero_start_iteration_count=0,
        nr_warm_start_iteration_count=0,
        final_beta=0.11,
        final_standard_error=0.21,
        final_chi_squared=0.31,
        final_log10_p_value=0.41,
        final_extra_code="firth",
        final_valid=True,
        firth_failure_code="none",
        firth_convergence_reason="converged",
    )

    assert binary_regenie_debug.count_missing_selections(records=[record], selector=selector) == 2


def test_tuning_benchmark_builds_queue_depth_values() -> None:
    assert tuning_benchmark.build_queue_depth_values(4, (1, 2)) == (4, 8)


def test_output_stage_benchmark_builds_handoff_timing_metrics(tmp_path: Path) -> None:
    python_stage_path = tmp_path / "python_stage_timings.json"
    first_rust_stage_path = tmp_path / "first_output_stage_timings.json"
    second_rust_stage_path = tmp_path / "second_output_stage_timings.json"
    python_stage_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "device_to_host_materialization": 2.0,
                    "output_write": 5.0,
                }
            }
        ),
        encoding="utf-8",
    )
    first_rust_stage_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "rust_output_metadata_clone": 0.5,
                    "rust_output_result_buffer_copy": 1.0,
                    "rust_output_enqueue": 0.25,
                    "rust_output_writer_record_batch_build": 2.0,
                    "rust_output_writer_arrow_file_write": 3.0,
                    "rust_output_writer_total": 6.0,
                }
            }
        ),
        encoding="utf-8",
    )
    second_rust_stage_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "rust_output_metadata_clone": 0.25,
                    "rust_output_result_buffer_copy": 0.5,
                    "rust_output_enqueue": 0.25,
                    "rust_output_writer_record_batch_build": 1.0,
                    "rust_output_writer_arrow_file_write": 1.5,
                    "rust_output_writer_total": 3.0,
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = output_stage_benchmark.build_output_handoff_timing_metrics(
        python_stage_timing_path=python_stage_path,
        rust_stage_timing_paths=(first_rust_stage_path, second_rust_stage_path),
        wall_time_seconds=20.0,
    )

    assert metrics.seconds_by_metric["device_to_host_materialization"] == 2.0
    assert metrics.seconds_by_metric["python_output_write"] == 5.0
    assert metrics.seconds_by_metric["rust_output_result_buffer_copy"] == 1.5
    assert metrics.seconds_by_metric["rust_output_writer_record_batch_build"] == 3.0
    assert metrics.seconds_by_metric["rust_output_writer_arrow_file_write"] == 4.5
    assert metrics.seconds_by_metric["bridge_residual"] == 2.25
    assert metrics.seconds_by_metric["measured_output_path"] == 16.0
    assert metrics.wall_time_percentage_by_metric["bridge_residual"] == 11.25
    assert metrics.output_path_percentage_by_metric["rust_output_result_buffer_copy"] == 9.375


def test_output_stage_benchmark_summarizes_handoff_metrics() -> None:
    first_timing = output_stage_benchmark.OutputHandoffTimingMetrics(
        seconds_by_metric={"bridge_residual": 2.0},
        wall_time_percentage_by_metric={"bridge_residual": 10.0},
        output_path_percentage_by_metric={"bridge_residual": 20.0},
    )
    second_timing = output_stage_benchmark.OutputHandoffTimingMetrics(
        seconds_by_metric={"bridge_residual": 4.0},
        wall_time_percentage_by_metric={"bridge_residual": 20.0},
        output_path_percentage_by_metric={"bridge_residual": 40.0},
    )
    trial_results = (
        output_stage_benchmark.TrialResult(
            case_name="case",
            trial_index=0,
            finalize_parquet=False,
            phenotype_count=1,
            chunk_size=1024,
            writer_thread_count=1,
            writer_queue_depth=1,
            chunks_per_arrow_file=4,
            arrow_compression="none",
            wall_time_seconds=20.0,
            python_stage_timing_path="python0.json",
            rust_stage_timing_paths=("rust0.json",),
            output_run_directories=("run0",),
            final_parquet_paths=(),
            chunk_file_count=1,
            chunk_bytes=100,
            final_parquet_bytes=None,
            handoff_timing=first_timing,
        ),
        output_stage_benchmark.TrialResult(
            case_name="case",
            trial_index=1,
            finalize_parquet=False,
            phenotype_count=1,
            chunk_size=1024,
            writer_thread_count=1,
            writer_queue_depth=1,
            chunks_per_arrow_file=4,
            arrow_compression="none",
            wall_time_seconds=30.0,
            python_stage_timing_path="python1.json",
            rust_stage_timing_paths=("rust1.json",),
            output_run_directories=("run1",),
            final_parquet_paths=(),
            chunk_file_count=1,
            chunk_bytes=200,
            final_parquet_bytes=None,
            handoff_timing=second_timing,
        ),
    )

    summary = output_stage_benchmark.summarize_trial_group(trial_results)

    assert summary["mean_handoff_timing_seconds"]["bridge_residual"] == 3.0
    assert summary["mean_handoff_wall_time_percentages"]["bridge_residual"] == 15.0
    assert summary["mean_handoff_output_path_percentages"]["bridge_residual"] == 30.0


def test_deep_profile_collects_regenie_and_g_stage_totals(tmp_path: Path) -> None:
    g_stage_path = tmp_path / "g.stage_timings.json"
    regenie_profile_path = tmp_path / "regenie.profile.json"
    g_stage_path.write_text(
        json.dumps({"stage_totals_seconds": {"native_engine_delivery": 2.0}}),
        encoding="utf-8",
    )
    regenie_profile_path.write_text(
        json.dumps({"stage_totals_seconds": {"bgen_decode_impute_filter": 3.0}}),
        encoding="utf-8",
    )
    g_trial = deep_profile.TrialResult(
        name="g_trial",
        implementation="g",
        trait_type="quantitative",
        device="gpu",
        status="success",
        wall_time_seconds=5.0,
        output_row_count=10,
        stdout_log_path="stdout",
        stderr_log_path="stderr",
        command_arguments=[],
        environment_overrides={},
        stage_timing_path=str(g_stage_path),
    )
    regenie_trial = deep_profile.TrialResult(
        name="regenie_trial",
        implementation="regenie",
        trait_type="quantitative",
        device="external_cpu",
        status="success",
        wall_time_seconds=6.0,
        output_row_count=10,
        stdout_log_path="stdout",
        stderr_log_path="stderr",
        command_arguments=[],
        environment_overrides={},
        regenie_profile_path=str(regenie_profile_path),
    )
    aggregates = [
        deep_profile.aggregate_trial_results(
            name="headline_g_quantitative_gpu",
            implementation="g",
            trait_type="quantitative",
            device="gpu",
            warmup_count=0,
            trial_results=[g_trial],
        ),
        deep_profile.aggregate_trial_results(
            name="headline_regenie_quantitative",
            implementation="regenie",
            trait_type="quantitative",
            device="external_cpu",
            warmup_count=0,
            trial_results=[regenie_trial],
        ),
    ]

    stage_totals = deep_profile.collect_stage_totals(aggregates)

    assert stage_totals["headline_g_quantitative_gpu:native_engine_delivery"] == 2.0
    assert stage_totals["headline_regenie_quantitative:bgen_decode_impute_filter"] == 3.0


def test_deep_profile_builds_stage_comparison_rows(tmp_path: Path) -> None:
    g_stage_path = tmp_path / "g.stage_timings.json"
    regenie_profile_path = tmp_path / "regenie.profile.json"
    g_stage_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "native_engine_delivery": 2.0,
                    "jax_compute": 4.0,
                    "output_write": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    regenie_profile_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "block_read": 4.0,
                    "association_compute": 8.0,
                    "block_output": 2.0,
                }
            }
        ),
        encoding="utf-8",
    )
    g_trial = deep_profile.TrialResult(
        name="g_trial",
        implementation="g",
        trait_type="binary",
        device="gpu",
        status="success",
        wall_time_seconds=5.0,
        output_row_count=10,
        stdout_log_path="stdout",
        stderr_log_path="stderr",
        command_arguments=[],
        environment_overrides={},
        stage_timing_path=str(g_stage_path),
    )
    regenie_trial = deep_profile.TrialResult(
        name="regenie_trial",
        implementation="regenie",
        trait_type="binary",
        device="external_cpu",
        status="success",
        wall_time_seconds=10.0,
        output_row_count=10,
        stdout_log_path="stdout",
        stderr_log_path="stderr",
        command_arguments=[],
        environment_overrides={},
        regenie_profile_path=str(regenie_profile_path),
    )
    aggregates = [
        deep_profile.aggregate_trial_results(
            name="headline_regenie_binary",
            implementation="regenie",
            trait_type="binary",
            device="external_cpu",
            warmup_count=0,
            trial_results=[regenie_trial],
        ),
        deep_profile.aggregate_trial_results(
            name="headline_g_binary_gpu",
            implementation="g",
            trait_type="binary",
            device="gpu",
            warmup_count=0,
            trial_results=[g_trial],
        ),
    ]

    rows = deep_profile.build_stage_comparison_rows(aggregates)
    bgen_row = next(row for row in rows if row["stage_group"] == "bgen_decode")
    findings = deep_profile.build_algorithmic_findings(rows)

    assert bgen_row["regenie_seconds"] == 4.0
    assert bgen_row["g_seconds"] == 2.0
    assert bgen_row["g_speedup_ratio"] == 2.0
    assert any("BGEN delivery" in finding for finding in findings)


def test_deep_profile_builds_binary_correction_diagnostics(tmp_path: Path) -> None:
    stage_timing_path = tmp_path / "binary.stage_timings.json"
    stage_timing_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "jax_compute": 3.0,
                    "output_write": 1.0,
                },
                "stage_counts": {
                    "jax_compute": 2,
                    "output_write": 1,
                },
                "chunk_stage_timings": [
                    {
                        "chunk_identifier": 10,
                        "chromosome": "22",
                        "variant_start_index": 0,
                        "variant_stop_index": 2,
                        "variant_count": 2,
                        "stage_name": "jax_compute",
                        "duration_seconds": 1.0,
                    },
                    {
                        "chunk_identifier": 11,
                        "chromosome": "22",
                        "variant_start_index": 2,
                        "variant_stop_index": 5,
                        "variant_count": 3,
                        "stage_name": "jax_compute",
                        "duration_seconds": 2.0,
                    },
                ],
                "binary_chunk_diagnostics": [
                    {
                        "score_test_candidate_count": 2,
                        "firth_candidate_count": 1,
                        "firth_iteration_min": 4,
                        "firth_iteration_median": 4.0,
                        "firth_iteration_max": 4,
                        "firth_converged_count": 1,
                        "firth_failed_count": 0,
                        "firth_numerical_failure_count": 0,
                        "firth_max_iteration_failure_count": 0,
                        "firth_invalid_statistic_failure_count": 0,
                        "firth_step_halving_failure_count": 0,
                        "pseudo_firth_attempt_count": 1,
                        "pseudo_firth_success_count": 1,
                        "nr_zero_start_attempt_count": 0,
                        "nr_zero_start_success_count": 0,
                        "nr_warm_start_attempt_count": 0,
                        "nr_warm_start_success_count": 0,
                        "sparse_correction_count": 0,
                        "dense_correction_count": 1,
                    },
                    {
                        "score_test_candidate_count": 3,
                        "firth_candidate_count": 2,
                        "firth_iteration_min": 5,
                        "firth_iteration_median": 6.0,
                        "firth_iteration_max": 7,
                        "firth_converged_count": 1,
                        "firth_failed_count": 1,
                        "firth_numerical_failure_count": 0,
                        "firth_max_iteration_failure_count": 1,
                        "firth_invalid_statistic_failure_count": 0,
                        "firth_step_halving_failure_count": 0,
                        "pseudo_firth_attempt_count": 2,
                        "pseudo_firth_success_count": 1,
                        "nr_zero_start_attempt_count": 1,
                        "nr_zero_start_success_count": 0,
                        "nr_warm_start_attempt_count": 1,
                        "nr_warm_start_success_count": 1,
                        "sparse_correction_count": 1,
                        "dense_correction_count": 1,
                    },
                ],
                "null_logistic_diagnostics": [
                    {
                        "chromosome": "22",
                        "iteration_count": 3,
                        "converged": 1,
                        "firth_iteration_count": 2,
                        "firth_convergence_reason_code": 0,
                        "correction_method": "firth",
                    }
                ],
                "queue_backpressure": [
                    {
                        "queue_name": "writer",
                        "operation_name": "enqueue",
                        "observation_count": 2,
                        "max_depth": 4,
                        "max_capacity": 8,
                        "total_elapsed_seconds": 1.0,
                        "total_blocked_seconds": 0.25,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trial = deep_profile.TrialResult(
        name="headline_g_binary_gpu_trial00",
        implementation="g",
        trait_type="binary",
        device="gpu",
        status="success",
        wall_time_seconds=2.0,
        output_row_count=10,
        stdout_log_path="stdout",
        stderr_log_path="stderr",
        command_arguments=[],
        environment_overrides={},
        stage_timing_path=str(stage_timing_path),
    )
    headline = deep_profile.aggregate_trial_results(
        name="headline_g_binary_gpu",
        implementation="g",
        trait_type="binary",
        device="gpu",
        warmup_count=0,
        trial_results=[trial],
    )
    finalist = dataclasses.replace(headline, name="finalist_binary_gpu")

    diagnostics = deep_profile.build_binary_correction_diagnostics(
        headline_results=[headline],
        finalist_results_by_key={"binary_gpu": [finalist]},
        stage_timing_mode=deep_profile.ProfileStageTimingMode.EXACT,
    )
    headline_diagnostics = diagnostics["headline"]["headline_g_binary_gpu"]
    finalist_diagnostics = diagnostics["finalists"]["binary_gpu"]["finalist_binary_gpu"]

    assert headline_diagnostics["available"] is True
    assert headline_diagnostics["candidate_counts"]["score_test"] == 5
    assert headline_diagnostics["candidate_counts"]["firth"] == 3
    assert headline_diagnostics["correction_outcome_counts"] == {
        "corrected": 2,
        "failed": 1,
        "score_test_or_uncorrected": 2,
    }
    assert headline_diagnostics["failure_code_counts"]["max_iterations"] == 1
    assert headline_diagnostics["firth_iteration_counts"] == {
        "active_chunk_count": 2,
        "minimum": 4.0,
        "median_per_chunk_mean": 5.0,
        "maximum": 7.0,
    }
    assert headline_diagnostics["correction_input_counts"] == {"sparse": 1, "dense": 2}
    assert headline_diagnostics["fallback_density"]["firth_candidates_per_output_row"] == 0.3
    assert headline_diagnostics["stage_counts"]["jax_compute"] == 2.0
    assert headline_diagnostics["null_logistic"]["converged_count"] == 1
    assert headline_diagnostics["queue_backpressure"][0]["blocked_fraction"] == 0.25
    assert headline_diagnostics["chunk_outliers"][0]["chunk_index"] == 1
    assert headline_diagnostics["chunk_outliers"][0]["chunk_identity"]["chunk_identifier"] == 11
    assert finalist_diagnostics["available"] is True


def test_deep_profile_binary_correction_diagnostics_report_stage_timing_off() -> None:
    trial = deep_profile.TrialResult(
        name="headline_g_binary_cpu_trial00",
        implementation="g",
        trait_type="binary",
        device="cpu",
        status="success",
        wall_time_seconds=2.0,
        output_row_count=10,
        stdout_log_path="stdout",
        stderr_log_path="stderr",
        command_arguments=[],
        environment_overrides={},
        stage_timing_path=None,
    )
    headline = deep_profile.aggregate_trial_results(
        name="headline_g_binary_cpu",
        implementation="g",
        trait_type="binary",
        device="cpu",
        warmup_count=0,
        trial_results=[trial],
    )

    diagnostics = deep_profile.build_binary_correction_diagnostics(
        headline_results=[headline],
        finalist_results_by_key={},
        stage_timing_mode=deep_profile.ProfileStageTimingMode.OFF,
    )
    markdown = deep_profile.build_summary_markdown(
        aggregate_results=[headline],
        comparisons={},
        stage_totals={},
        stage_comparison_rows=[],
        algorithmic_findings=[],
        binary_correction_diagnostics=diagnostics,
    )
    headline_diagnostics = diagnostics["headline"]["headline_g_binary_cpu"]

    assert headline_diagnostics["available"] is False
    assert headline_diagnostics["reason"] == "exact_stage_timings_disabled"
    assert "## Binary Correction Diagnostics" in markdown
    assert "telemetry.stage_timing_mode=off" in markdown
    assert "unavailable: stage timing mode off" in markdown


def test_deep_profile_algorithmic_findings_respect_speedup_direction() -> None:
    rows: list[dict[str, float | str]] = [
        {
            "trait_type": "quantitative",
            "g_device": "cpu",
            "stage_group": "input_setup",
            "regenie_seconds": 4.0,
            "g_seconds": 2.0,
            "g_speedup_ratio": 2.0,
        },
        {
            "trait_type": "binary",
            "g_device": "cpu",
            "stage_group": "output",
            "regenie_seconds": 1.0,
            "g_seconds": 4.0,
            "g_speedup_ratio": 0.25,
        },
    ]

    findings = deep_profile.build_algorithmic_findings(rows)

    assert any("quantitative/cpu: g is faster in input_setup" in finding for finding in findings)
    assert any("binary/cpu: REGENIE remains faster in output" in finding for finding in findings)


def test_binary_firth_parity_harness_synthetic_fixture_passes() -> None:
    comparison = binary_firth_parity.compare_binary_paths(
        inputs=binary_firth_parity.build_synthetic_inputs(),
        correction_plan=binary_firth_parity.types.BinaryCorrectionPlan(
            method=binary_firth_parity.types.BinaryFallbackMethod.FIRTH_APPROXIMATE
        ),
    )

    assert comparison.passed is True
    assert comparison.production_metrics == comparison.variant_major_metrics
    assert comparison.production_metrics.firth_candidate_count >= 0


def test_binary_firth_parity_harness_loads_npz_fixture(tmp_path: Path) -> None:
    inputs = binary_firth_parity.build_synthetic_inputs()
    fixture_path = tmp_path / "binary_fixture.npz"
    binary_firth_parity.np.savez(
        fixture_path,
        covariate_matrix=binary_firth_parity.np.asarray(inputs.covariate_matrix),
        phenotype_vector=binary_firth_parity.np.asarray(inputs.phenotype_vector),
        genotype_matrix=binary_firth_parity.np.asarray(inputs.genotype_matrix),
        loco_offset=binary_firth_parity.np.asarray(inputs.loco_offset),
    )

    loaded_inputs = binary_firth_parity.load_npz_inputs(fixture_path)

    assert loaded_inputs.genotype_matrix.shape == inputs.genotype_matrix.shape


def test_tuning_benchmark_builds_trial_environment_from_low_level_knobs() -> None:
    candidate = tuning_benchmark.Step2TuningCandidate(
        trait_type=tuning_benchmark.types.RegenieTraitType.BINARY,
        chunk_size=8192,
        staging_depth=1,
        output_writer_thread_count=8,
        output_writer_queue_depth=16,
        bgen_decode_tile_variant_count=128,
        rayon_thread_count=4,
        firth_batch_size=64,
    )
    environment = tuning_benchmark.build_step2_trial_environment(candidate)
    assert "G_BGEN_DECODE_TILE_VARIANT_COUNT" not in environment
    assert "RAYON_NUM_THREADS" not in environment
    assert "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE" not in environment


def test_tuning_benchmark_builds_shared_compute_candidates() -> None:
    bgen_candidate_summary = tuning_benchmark.BgenCandidateSummary(
        candidate=tuning_benchmark.BgenCandidate(
            decode_tile_variant_count=64,
            rayon_thread_count=2,
            benchmark_chunk_size=8192,
        ),
        median_seconds=0.1,
        mean_seconds=0.1,
        repeat_count=3,
    )
    candidates = tuning_benchmark.build_compute_stage_candidates(
        trait_type=tuning_benchmark.types.RegenieTraitType.QUANTITATIVE,
        chunk_sizes=(4096, 8192),
        staging_depth_values=(1, 2),
        bgen_candidates=(bgen_candidate_summary,),
        firth_batch_sizes=(32, 64),
    )
    assert len(candidates) == 4
    assert all(candidate.firth_batch_size is None for candidate in candidates)


def test_tuning_benchmark_builds_binary_compute_candidates_with_firth_sizes() -> None:
    bgen_candidate_summary = tuning_benchmark.BgenCandidateSummary(
        candidate=tuning_benchmark.BgenCandidate(
            decode_tile_variant_count=64,
            rayon_thread_count=2,
            benchmark_chunk_size=8192,
        ),
        median_seconds=0.1,
        mean_seconds=0.1,
        repeat_count=3,
    )
    candidates = tuning_benchmark.build_compute_stage_candidates(
        trait_type=tuning_benchmark.types.RegenieTraitType.BINARY,
        chunk_sizes=(8192,),
        staging_depth_values=(1,),
        bgen_candidates=(bgen_candidate_summary,),
        firth_batch_sizes=(32, 64),
    )
    assert [candidate.firth_batch_size for candidate in candidates] == [32, 64]


def test_regenie_command_builders_can_focus_quantitative_step2() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    command_specs = comparison_benchmark.build_regenie_program_specs(
        "regenie",
        baseline_paths,
        only_quantitative_step2=True,
    )
    assert len(command_specs) == 1
    assert command_specs[0][0] == "regenie_step2_quantitative"
    assert "--step" in command_specs[0][3]
    assert command_specs[0][3][command_specs[0][3].index("--step") + 1] == "2"
    assert "--qt" in command_specs[0][3]


def test_g_comparison_runner_builds_cpu_and_gpu_commands() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    cpu_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_cpu"),
        device="cpu",
        chunk_size=512,
        variant_limit=1024,
    )
    gpu_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_gpu"),
        device="gpu",
        chunk_size=2048,
        variant_limit=None,
    )
    binary_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_bin"),
        device="cpu",
        chunk_size=8192,
        variant_limit=None,
        trait_type="binary",
    )
    assert cpu_command[:4] == ["uv", "run", "g", "regenie"]
    assert "--step" in cpu_command
    assert cpu_command[cpu_command.index("--step") + 1] == "2"
    assert "--qt" in cpu_command
    assert "--g-device" in cpu_command
    assert cpu_command[cpu_command.index("--g-device") + 1] == "cpu"
    assert "--g-output-format" in cpu_command
    assert cpu_command[cpu_command.index("--g-output-format") + 1] == "parquet"
    assert "--g-variant-limit" in cpu_command
    assert "--variant-limit" not in cpu_command
    assert gpu_command[gpu_command.index("--g-device") + 1] == "gpu"
    assert "--g-variant-limit" not in gpu_command
    assert "--bt" in binary_command
    assert "--firth" in binary_command
    assert "--approx" in binary_command
    assert "phenotype_binary" in binary_command


def test_g_comparison_runner_resolves_current_output_layout(tmp_path: Path) -> None:
    output_root_directory = tmp_path / "out.g"
    output_association_directory = output_root_directory / "trait_0001_phenotype_binary.regenie2_binary.run"
    output_association_directory.mkdir(parents=True)
    final_parquet_path = output_association_directory / "final.parquet"
    final_parquet_path.touch()

    resolved_path = comparison_benchmark.resolve_g_step2_final_parquet_path(
        output_root_directory=output_root_directory,
        association_suffix=".regenie2_binary.run",
    )

    assert resolved_path == final_parquet_path


def test_unsupported_g_program_result_marked_not_implemented() -> None:
    result = comparison_benchmark.build_not_implemented_result(
        program_name="g_regenie2_binary_step1",
        trait_type="binary",
        step=1,
        device="cpu",
    )
    assert result.status == "not_implemented"
    assert result.implementation == "g"
    assert result.notes is not None


def test_profiled_subprocess_wrapper_metadata(tmp_path: Path) -> None:
    stdout_log_path = tmp_path / "stdout.log"
    stderr_log_path = tmp_path / "stderr.log"
    success, wall_time_seconds, peak_rss_megabytes, cpu_user_seconds, cpu_system_seconds, error_message = (
        comparison_profile.run_profiled_subprocess(
            command_arguments=[sys.executable, "-c", "import time; print('ok'); time.sleep(0.05)"],
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            sample_interval_seconds=0.01,
        )
    )
    assert success
    assert wall_time_seconds > 0.0
    assert peak_rss_megabytes is not None
    assert peak_rss_megabytes >= 0.0
    assert cpu_user_seconds >= 0.0
    assert cpu_system_seconds >= 0.0
    assert error_message is None
    assert "ok" in stdout_log_path.read_text()


def test_summary_serializer_json_shape() -> None:
    result = comparison_benchmark.ComparisonProgramResult(
        program_name="regenie_step2_quantitative",
        implementation="regenie",
        trait_type="quantitative",
        step=2,
        device="external_cpu",
        status="success",
        wall_time_seconds=12.3,
        variants_per_second=1000.0,
        peak_memory_megabytes=None,
        stdout_log_path="stdout.log",
        stderr_log_path="stderr.log",
        output_paths=["out.regenie"],
        output_row_count=100,
        prediction_list_present=None,
    )
    payload = {"results": [result.__dict__]}
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert isinstance(decoded["results"], list)
    assert decoded["results"][0]["program_name"] == "regenie_step2_quantitative"
    assert decoded["results"][0]["status"] == "success"


def test_text_summary_includes_required_sections(tmp_path: Path) -> None:
    results = [
        comparison_benchmark.ComparisonProgramResult(
            program_name="regenie_step2_quantitative",
            implementation="regenie",
            trait_type="quantitative",
            step=2,
            device="external_cpu",
            status="success",
            wall_time_seconds=20.0,
            variants_per_second=100.0,
            peak_memory_megabytes=None,
            stdout_log_path=None,
            stderr_log_path=None,
            output_paths=[],
            output_row_count=1000,
            prediction_list_present=None,
        ),
        comparison_benchmark.ComparisonProgramResult(
            program_name="g_regenie2_quantitative_step2_cpu",
            implementation="g",
            trait_type="quantitative",
            step=2,
            device="cpu",
            status="success",
            wall_time_seconds=10.0,
            variants_per_second=200.0,
            peak_memory_megabytes=None,
            stdout_log_path=None,
            stderr_log_path=None,
            output_paths=[],
            output_row_count=1000,
            prediction_list_present=None,
        ),
        comparison_benchmark.ComparisonProgramResult(
            program_name="g_regenie2_quantitative_step2_gpu",
            implementation="g",
            trait_type="quantitative",
            step=2,
            device="gpu",
            status="not_implemented",
            wall_time_seconds=None,
            variants_per_second=None,
            peak_memory_megabytes=None,
            stdout_log_path=None,
            stderr_log_path=None,
            output_paths=[],
            output_row_count=None,
            prediction_list_present=None,
            notes="not_implemented",
        ),
    ]
    agreement = comparison_benchmark.QuantitativeStep2Agreement(
        comparable=True,
        merged_variant_count=1000,
        beta_max_abs_error=1.0e-4,
        beta_mean_abs_error=1.0e-5,
        beta_allclose_within_tolerance=True,
        log10p_max_abs_error=1.0e-4,
        log10p_mean_abs_error=1.0e-5,
        log10p_allclose_within_tolerance=True,
    )
    report_path = tmp_path / "summary.txt"
    comparison_benchmark.write_text_summary(
        report_path=report_path,
        results=results,
        agreement_cpu=agreement,
        agreement_gpu=None,
    )
    summary = report_path.read_text()
    assert "regenie_step2_quantitative" in summary
    assert "g_regenie2_quantitative_step2_cpu" in summary
    assert "Direct Runtime Comparisons" in summary
    assert "Numeric Agreement" in summary


def test_quantitative_step2_comparison_wires_parity_logic(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text(
        "CHROM GENPOS ID BETA SE CHISQ LOG10P EXTRA\n1 100 rs1 0.1 0.3 0.11 1.0 NA\n1 200 rs2 0.2 0.4 0.25 2.0 NA\n"
    )
    pl.DataFrame(
        {
            "ID": ["rs1", "rs2"],
            "BETA": [0.1, 0.2],
            "SE": [0.3, 0.4],
            "CHISQ": [0.11, 0.25],
            "LOG10P": [1.0, 2.0],
            "EXTRA": [None, None],
        }
    ).write_parquet(g_output)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 2
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.standard_error_allclose_within_tolerance is True
    assert agreement.chi_squared_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True
    assert agreement.extra_match_rate == 1.0
    assert agreement.top_variant_differences is not None


def test_fresh_process_benchmark_parser_accepts_output_writer_options() -> None:
    arguments = fresh_process_benchmark.build_argument_parser().parse_args(
        [
            "--output-writer-thread-count",
            "2",
        ]
    )
    assert arguments.output_writer_thread_count == 2


def test_fresh_process_benchmark_generates_multi_phenotype_inputs(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    baseline_directory = data_directory / "baselines"
    baseline_directory.mkdir(parents=True)
    (data_directory / "pheno_cont.txt").write_text(
        "FID\tIID\tphenotype_continuous\nF1\tI1\t1.5\nF2\tI2\t2.5\n",
        encoding="utf-8",
    )
    (baseline_directory / "shared.loco").write_text("FID_IID F1_I1 F2_I2\n22 0.1 0.2\n", encoding="utf-8")
    (baseline_directory / "regenie_step1_qt_pred.list").write_text(
        "phenotype_continuous shared.loco\n",
        encoding="utf-8",
    )

    benchmark_inputs = fresh_process_benchmark.prepare_benchmark_inputs(
        data_directory=data_directory,
        output_directory=tmp_path / "output",
        phenotype_count=2,
    )

    assert benchmark_inputs.phenotype_names == ("phenotype_continuous_1", "phenotype_continuous_2")
    assert benchmark_inputs.phenotype_path.read_text(encoding="utf-8") == (
        "FID\tIID\tphenotype_continuous_1\tphenotype_continuous_2\nF1\tI1\t1.5\t1.5\nF2\tI2\t2.5\t2.5\n"
    )
    assert benchmark_inputs.prediction_list_path.read_text(encoding="utf-8") == (
        f"phenotype_continuous_1 {baseline_directory / 'shared.loco'}\n"
        f"phenotype_continuous_2 {baseline_directory / 'shared.loco'}\n"
    )


def test_fresh_process_benchmark_child_command_wires_multi_phenotype_options(tmp_path: Path) -> None:
    benchmark_inputs = fresh_process_benchmark.BenchmarkInputs(
        bgen_path=tmp_path / "input.bgen",
        sample_path=tmp_path / "input.sample",
        phenotype_path=tmp_path / "phenotypes.tsv",
        phenotype_names=("trait_a", "trait_b"),
        covariate_path=tmp_path / "covariates.tsv",
        prediction_list_path=tmp_path / "pred.list",
    )

    command_arguments = fresh_process_benchmark.build_child_command(
        benchmark_inputs=benchmark_inputs,
        output_path=tmp_path / "out",
        device="gpu",
        chunk_size=2048,
        finalize_parquet=True,
        output_writer_thread_count=4,
        stage_timing_path=tmp_path / "stage.json",
        multi_phenotype_sample_mode="complete-case",
    )
    child_code = command_arguments[2]

    assert "'phenoColList': 'trait_a,trait_b'" in child_code
    assert "'g-multi-phenotype-sample-mode': 'complete-case'" in child_code
    assert "'g-stage-timings-json':" in child_code


def test_fresh_process_benchmark_summary_tracks_output_metrics() -> None:
    trial_results = [
        fresh_process_benchmark.TrialResult(
            trial_index=0,
            wall_time_seconds=2.0,
            output_path="out0",
            output_row_count=100,
            chunk_file_count=2,
            chunk_bytes=1024,
            final_parquet_bytes=512,
        ),
        fresh_process_benchmark.TrialResult(
            trial_index=1,
            wall_time_seconds=1.0,
            output_path="out1",
            output_row_count=100,
            chunk_file_count=2,
            chunk_bytes=2048,
            final_parquet_bytes=1024,
        ),
    ]
    summary = fresh_process_benchmark.build_summary(
        device="gpu",
        chunk_size=8192,
        finalize_parquet=True,
        output_writer_thread_count=2,
        warmup_count=1,
        trial_results=trial_results,
    )
    assert summary.mean_rows_per_second == 75.0
    assert summary.mean_chunk_bytes == 1536.0
    assert summary.mean_final_parquet_bytes == 768.0


def test_fresh_process_benchmark_summary_tracks_startup_fields() -> None:
    trial_results = [
        fresh_process_benchmark.TrialResult(
            trial_index=0,
            wall_time_seconds=4.0,
            output_path="out0",
            output_row_count=100,
            chunk_file_count=1,
            chunk_bytes=128,
            final_parquet_bytes=64,
            mode="fresh_process",
            phenotype_count=2,
            child_wall_time_seconds=3.0,
            stage_timing_path="stage0.json",
        ),
        fresh_process_benchmark.TrialResult(
            trial_index=1,
            wall_time_seconds=2.0,
            output_path="out1",
            output_row_count=100,
            chunk_file_count=1,
            chunk_bytes=128,
            final_parquet_bytes=64,
            mode="fresh_process",
            phenotype_count=2,
            child_wall_time_seconds=1.0,
            stage_timing_path="stage1.json",
        ),
    ]

    summary = fresh_process_benchmark.build_summary(
        device="gpu",
        chunk_size=2048,
        finalize_parquet=True,
        output_writer_thread_count=4,
        warmup_count=1,
        trial_results=trial_results,
        mode="fresh_process",
        phenotype_count=2,
    )

    assert summary.phenotype_count == 2
    assert summary.mean_child_wall_time_seconds == 2.0
    assert summary.stage_timing_paths == ["stage0.json", "stage1.json"]


def test_binary_hot_benchmark_defaults_to_comparable_modes() -> None:
    arguments = binary_hot_benchmark.build_arguments_from_overrides()
    assert arguments.device == "gpu"
    assert arguments.chunk_size == binary_hot_benchmark.config.load_packaged_config().trait.bsize
    assert arguments.output_writer_thread_count == 8
    assert arguments.trusted_no_missing_diploid is True
    assert arguments.assume_trusted_validated is False
    trial_specs = binary_hot_benchmark.build_trial_specs(
        include_cold_process=arguments.include_cold_process,
        include_no_final_hot=arguments.include_no_final_hot,
        include_finalized_hot=arguments.include_finalized_hot,
    )
    assert [trial_spec.mode.value for trial_spec in trial_specs] == [
        "cold_process_finalized",
        "warm_same_process_no_final",
        "hot_same_process_no_final",
        "warm_same_process_finalized",
        "hot_same_process_finalized",
    ]
    configuration = binary_hot_benchmark.build_configuration(arguments)
    benchmark_cases = binary_hot_benchmark.build_benchmark_cases(configuration)
    assert configuration.bgen_path == Path("data/1kg_chr22_full.bgen")
    assert configuration.sample_path == Path("data/1kg_chr22_full.sample")
    assert configuration.expected_variant_count == binary_hot_benchmark.DEFAULT_VARIANT_COUNT
    assert configuration.stage_timing_mode == binary_hot_benchmark.StageTimingMode.EXACT
    assert [benchmark_case.name for benchmark_case in benchmark_cases] == [
        "traits1_variant_major_default_batch1024_capacity2048"
    ]
    assert benchmark_cases[0].phenotype_columns == ("phenotype_binary",)
    assert benchmark_cases[0].gpu_genotype_format == binary_hot_benchmark.types.GpuGenotypeFormat.DOSAGE


def test_binary_hot_benchmark_expands_multi_binary_firth_sweep(tmp_path: Path) -> None:
    arguments = binary_hot_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.phenotype_columns=[trait_one,trait_two,trait_three,trait_four]",
            "tool.binary_trait_counts=[1,2]",
            "tool.firth_batch_sizes=[32,64]",
            "tool.firth_candidate_capacities=[128,512]",
            "tool.storage_modes=[variant_major,packed8]",
            "tool.fallback_density_scenarios=[low,high]",
            "tool.low_fallback_p_threshold=1e-6",
            "tool.high_fallback_p_threshold=0.5",
        ]
    )

    configuration = binary_hot_benchmark.build_configuration(arguments)
    benchmark_cases = binary_hot_benchmark.build_benchmark_cases(configuration)

    assert len(benchmark_cases) == 32
    assert {benchmark_case.binary_trait_count for benchmark_case in benchmark_cases} == {1, 2}
    assert {benchmark_case.firth_batch_size for benchmark_case in benchmark_cases} == {32, 64}
    assert {benchmark_case.firth_candidate_capacity for benchmark_case in benchmark_cases} == {128, 512}
    assert {benchmark_case.storage_mode.value for benchmark_case in benchmark_cases} == {"variant_major", "packed8"}
    assert {benchmark_case.fallback_density.value for benchmark_case in benchmark_cases} == {"low", "high"}
    assert next(
        benchmark_case for benchmark_case in benchmark_cases if benchmark_case.binary_trait_count == 2
    ).phenotype_columns == (
        "trait_one",
        "trait_two",
    )

    packed_high_case = next(
        benchmark_case
        for benchmark_case in benchmark_cases
        if benchmark_case.storage_mode == binary_hot_benchmark.BenchmarkStorageMode.PACKED8
        and benchmark_case.fallback_density == binary_hot_benchmark.FallbackDensityScenario.HIGH
        and benchmark_case.firth_batch_size == 32
        and benchmark_case.firth_candidate_capacity == 128
    )
    compute_config = binary_hot_benchmark.build_compute_config(
        configuration=configuration,
        benchmark_case=packed_high_case,
        output_root=tmp_path / "out",
        finalize_parquet=False,
        stage_timing_path=tmp_path / "stage.json",
    )
    assert compute_config["g-firth-batch-size"] == 32
    assert compute_config["g-firth-candidate-capacity"] == 128
    assert compute_config["g-gpu-genotype-format"] == "packed8"
    assert compute_config["g-telemetry"] == "off"
    assert packed_high_case.firth_p_threshold == 0.5


def test_binary_hot_benchmark_can_disable_exact_stage_timings(tmp_path: Path) -> None:
    arguments = binary_hot_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.stage_timing_mode=off",
            f"tool.jax_cache_dir={tmp_path / 'jax-cache'}",
        ]
    )
    configuration = binary_hot_benchmark.build_configuration(arguments)
    benchmark_case = binary_hot_benchmark.build_benchmark_cases(configuration)[0]
    trial_spec = binary_hot_benchmark.TrialSpec(
        name="hot_same_process_no_final",
        mode=binary_hot_benchmark.BenchmarkMode.HOT_SAME_PROCESS_NO_FINAL,
        finalize_parquet=False,
        fresh_process=False,
        same_process_group="no_final",
    )
    compute_config = binary_hot_benchmark.build_compute_config(
        configuration=configuration,
        benchmark_case=benchmark_case,
        output_root=tmp_path / "out",
        finalize_parquet=False,
        stage_timing_path=None,
    )
    child_command = binary_hot_benchmark.build_fresh_process_command(
        configuration=configuration,
        benchmark_case=benchmark_case,
        trial_spec=trial_spec,
        stage_timing_path=None,
    )
    serialized_configuration = binary_hot_benchmark.configuration_to_json_dict(configuration)
    restored_configuration = binary_hot_benchmark.configuration_from_json_dict(serialized_configuration)
    trial_result = binary_hot_benchmark.TrialResult(
        name=f"{benchmark_case.name}_{trial_spec.name}",
        benchmark_case=benchmark_case,
        mode=trial_spec.mode,
        fresh_process=trial_spec.fresh_process,
        finalize_parquet=trial_spec.finalize_parquet,
        same_process_group=trial_spec.same_process_group,
        wall_time_seconds=1.0,
        stage_timing_path=None,
        output_metrics=binary_hot_benchmark.OutputMetrics(
            output_run_directory="run",
            final_parquet=None,
            output_row_count=1,
            info_non_null_count=1,
            chunk_file_count=1,
            chunk_bytes=1,
            final_parquet_bytes=None,
        ),
    )

    assert configuration.stage_timing_mode == binary_hot_benchmark.StageTimingMode.OFF
    assert restored_configuration.stage_timing_mode == binary_hot_benchmark.StageTimingMode.OFF
    assert compute_config["g-stage-timings-json"] is None
    assert "stage_timing_path_value = None" in child_command.command_arguments[2]
    assert (
        binary_hot_benchmark.trial_result_from_json_dict(
            binary_hot_benchmark.trial_result_to_json_dict(trial_result)
        ).stage_timing_path
        is None
    )
    summary = binary_hot_benchmark.build_summary(configuration=configuration, trial_results=[trial_result])
    diagnostics = summary["binary_diagnostics_by_case"][benchmark_case.name]["hot_same_process_no_final"]
    assert diagnostics["available"] is False
    assert diagnostics["reason"] == binary_hot_benchmark.BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED
    assert diagnostics["stage_timing_path"] is None
    assert diagnostics["stage_timing_mode"] == "off"
    assert diagnostics["candidate_counts"] == {"score_test": None, "firth": None}
    assert diagnostics["failure_code_counts"]["none"] is None
    assert diagnostics["correction_branch_counts"]["pseudo_firth"] is None
    assert diagnostics["stage_totals_seconds"] is None


def test_binary_hot_benchmark_accepts_custom_genotype_inputs(tmp_path: Path) -> None:
    arguments = binary_hot_benchmark.build_arguments_from_overrides(
        [
            f"tool.data_dir={tmp_path / 'data'}",
            "tool.bgen=1kg_chr10_full.bgen",
            "tool.sample=1kg_chr10_full.sample",
            "tool.prediction_list=baselines_chr10/regenie_step1_pred.list",
            "tool.expected_variant_count=1200000",
            f"tool.output_dir={tmp_path / 'profile'}",
            f"tool.jax_cache_dir={tmp_path / 'jax-cache'}",
        ]
    )

    configuration = binary_hot_benchmark.build_configuration(arguments)
    serialized_configuration = binary_hot_benchmark.configuration_to_json_dict(configuration)
    restored_configuration = binary_hot_benchmark.configuration_from_json_dict(serialized_configuration)

    assert configuration.bgen_path == tmp_path / "data" / "1kg_chr10_full.bgen"
    assert configuration.sample_path == tmp_path / "data" / "1kg_chr10_full.sample"
    assert configuration.prediction_list == tmp_path / "data" / "baselines_chr10" / "regenie_step1_pred.list"
    assert configuration.expected_variant_count == 1_200_000
    assert restored_configuration.bgen_path == configuration.bgen_path
    assert restored_configuration.sample_path == configuration.sample_path
    assert restored_configuration.expected_variant_count == configuration.expected_variant_count


def test_binary_hot_child_process_command_contains_binary_controls(tmp_path: Path) -> None:
    arguments = binary_hot_benchmark.build_arguments_from_overrides(
        [
            "tool.data_dir=data",
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.device=cpu",
            "tool.chunk_size=4096",
            "tool.staging_depth=2",
            "tool.output_writer_thread_count=4",
            "tool.output_writer_queue_depth=8",
            "tool.assume_trusted_validated=true",
            "tool.firth_batch_size=64",
            "tool.firth_candidate_capacity=256",
            "tool.variant_limit=1000",
            f"tool.python_executable={sys.executable}",
            f"tool.jax_cache_dir={tmp_path / 'jax-cache'}",
        ]
    )
    configuration = binary_hot_benchmark.build_configuration(arguments)
    benchmark_case = binary_hot_benchmark.build_benchmark_cases(configuration)[0]
    trial_spec = binary_hot_benchmark.TrialSpec(
        name="cold_process_finalized",
        mode=binary_hot_benchmark.BenchmarkMode.COLD_PROCESS_FINALIZED,
        finalize_parquet=True,
        fresh_process=True,
        same_process_group=None,
    )
    child_command = binary_hot_benchmark.build_fresh_process_command(
        configuration=configuration,
        benchmark_case=benchmark_case,
        trial_spec=trial_spec,
        stage_timing_path=tmp_path / "stages.json",
    )
    command_text = child_command.command_arguments[2]
    assert child_command.command_arguments[:2] == [sys.executable, "-c"]
    assert "benchmark_regenie2_binary_hot" in command_text
    assert "trusted_no_missing_diploid" in command_text
    assert "variant_limit" in command_text
    assert "benchmark_case" in command_text
    assert "firth_candidate_capacity" in command_text
    assert "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE" not in child_command.environment_overrides
    assert "G_REGENIE2_ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED" not in child_command.environment_overrides
    assert "JAX_PLATFORMS" not in child_command.environment_overrides


def test_binary_hot_output_metrics_aggregate_multi_phenotype_artifacts(tmp_path: Path) -> None:
    first_run_directory = tmp_path / "first.run"
    second_run_directory = tmp_path / "second.run"
    first_chunk_directory = first_run_directory / "chunks"
    second_chunk_directory = second_run_directory / "chunks"
    first_chunk_directory.mkdir(parents=True)
    second_chunk_directory.mkdir(parents=True)
    pl.DataFrame({"INFO": [0.9, None], "BETA": [0.1, 0.2]}).write_ipc(first_chunk_directory / "chunk_000.arrow")
    pl.DataFrame({"INFO": [0.7], "BETA": [0.3]}).write_ipc(second_chunk_directory / "chunk_000.arrow")

    output_metrics = binary_hot_benchmark.measure_output_metrics(
        binary_hot_benchmark.api.RunArtifacts(
            phenotype_artifacts=(
                binary_hot_benchmark.api.RunArtifacts(output_run_directory=first_run_directory),
                binary_hot_benchmark.api.RunArtifacts(output_run_directory=second_run_directory),
            )
        )
    )

    assert output_metrics.output_run_directory is None
    assert output_metrics.output_row_count == 3
    assert output_metrics.info_non_null_count == 2
    assert output_metrics.chunk_file_count == 2
    assert output_metrics.chunk_bytes > 0


def test_binary_hot_summary_records_headline_modes(tmp_path: Path) -> None:
    arguments = binary_hot_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.assume_trusted_validated=true",
            f"tool.jax_cache_dir={tmp_path / 'jax-cache'}",
        ]
    )
    configuration = binary_hot_benchmark.build_configuration(arguments)
    benchmark_case = binary_hot_benchmark.build_benchmark_cases(configuration)[0]
    output_metrics = binary_hot_benchmark.OutputMetrics(
        output_run_directory="run",
        final_parquet=None,
        output_row_count=100,
        info_non_null_count=100,
        chunk_file_count=2,
        chunk_bytes=1024,
        final_parquet_bytes=None,
    )
    stage_timing_path = tmp_path / "hot_no_final.json"
    stage_timing_path.write_text(
        json.dumps(
            {
                "stage_totals_seconds": {
                    "jax_compute": 3.0,
                    "output_write": 1.0,
                },
                "stage_counts": {
                    "jax_compute": 2,
                    "output_write": 1,
                },
                "derived_metrics": {
                    "jax_variant_compute_per_second": 50.0,
                },
                "binary_chunk_diagnostics": [
                    {
                        "score_test_candidate_count": 2,
                        "firth_candidate_count": 1,
                        "firth_iteration_min": 4,
                        "firth_iteration_median": 4.0,
                        "firth_iteration_max": 4,
                        "firth_converged_count": 1,
                        "firth_failed_count": 0,
                        "firth_numerical_failure_count": 0,
                        "firth_max_iteration_failure_count": 0,
                        "firth_invalid_statistic_failure_count": 0,
                        "firth_step_halving_failure_count": 0,
                        "pseudo_firth_attempt_count": 1,
                        "pseudo_firth_success_count": 1,
                        "nr_zero_start_attempt_count": 0,
                        "nr_zero_start_success_count": 0,
                        "nr_warm_start_attempt_count": 0,
                        "nr_warm_start_success_count": 0,
                        "sparse_correction_count": 0,
                        "dense_correction_count": 1,
                    },
                    {
                        "score_test_candidate_count": 3,
                        "firth_candidate_count": 2,
                        "firth_iteration_min": 5,
                        "firth_iteration_median": 6.0,
                        "firth_iteration_max": 7,
                        "firth_converged_count": 1,
                        "firth_failed_count": 1,
                        "firth_numerical_failure_count": 0,
                        "firth_max_iteration_failure_count": 1,
                        "firth_invalid_statistic_failure_count": 0,
                        "firth_step_halving_failure_count": 0,
                        "pseudo_firth_attempt_count": 2,
                        "pseudo_firth_success_count": 1,
                        "nr_zero_start_attempt_count": 1,
                        "nr_zero_start_success_count": 0,
                        "nr_warm_start_attempt_count": 1,
                        "nr_warm_start_success_count": 1,
                        "sparse_correction_count": 1,
                        "dense_correction_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    trial_results = [
        binary_hot_benchmark.TrialResult(
            name=f"{benchmark_case.name}_hot_same_process_no_final",
            benchmark_case=benchmark_case,
            mode=binary_hot_benchmark.BenchmarkMode.HOT_SAME_PROCESS_NO_FINAL,
            fresh_process=False,
            finalize_parquet=False,
            same_process_group="no_final",
            wall_time_seconds=7.25,
            stage_timing_path=str(stage_timing_path),
            output_metrics=output_metrics,
        ),
        binary_hot_benchmark.TrialResult(
            name=f"{benchmark_case.name}_hot_same_process_finalized",
            benchmark_case=benchmark_case,
            mode=binary_hot_benchmark.BenchmarkMode.HOT_SAME_PROCESS_FINALIZED,
            fresh_process=False,
            finalize_parquet=True,
            same_process_group="finalized",
            wall_time_seconds=7.85,
            stage_timing_path="hot_finalized.json",
            output_metrics=output_metrics,
        ),
    ]
    summary = binary_hot_benchmark.build_summary(configuration=configuration, trial_results=trial_results)
    assert summary["headline"]["hot_same_process_no_final_seconds"] == 7.25
    assert summary["headline"]["hot_same_process_finalized_seconds"] == 7.85
    assert summary["metadata"]["configuration"]["trusted_no_missing_diploid"] is True
    assert summary["metadata"]["configuration"]["firth_candidate_capacities"] == [2048]
    assert summary["headline_by_case"][benchmark_case.name]["hot_same_process_finalized_seconds"] == 7.85
    diagnostics = summary["binary_diagnostics_by_case"][benchmark_case.name]["hot_same_process_no_final"]
    assert diagnostics["available"] is True
    assert diagnostics["candidate_counts"] == {"score_test": 5, "firth": 3}
    assert diagnostics["firth_outcome_counts"] == {"converged": 2, "failed": 1}
    assert diagnostics["failure_code_counts"]["none"] == 2
    assert diagnostics["failure_code_counts"]["max_iterations"] == 1
    assert diagnostics["correction_branch_counts"] == {
        "pseudo_firth": 2,
        "newton_raphson_zero_start": 0,
        "newton_raphson_warm_start": 1,
    }
    assert diagnostics["correction_attempt_counts"]["newton_raphson_zero_start"] == 1
    assert diagnostics["correction_input_counts"] == {"sparse": 1, "dense": 2}
    assert diagnostics["firth_iteration_counts"] == {"minimum": 4.0, "median_per_chunk_mean": 5.0, "maximum": 7.0}
    assert diagnostics["code_values"]["firth_failure"]["max_iterations"] == 2
    assert diagnostics["stage_totals_seconds"]["jax_compute"] == 3.0
    assert diagnostics["stage_counts"]["jax_compute"] == 2
    assert diagnostics["derived_metrics"]["jax_variant_compute_per_second"] == 50.0


def test_output_stage_benchmark_builds_recommended_matrix() -> None:
    cases = output_stage_benchmark.build_benchmark_cases(
        small_chunk_size=1024,
        large_chunk_size=8192,
        many_phenotype_count=8,
        writer_thread_counts=(4,),
        writer_queue_depth_multipliers=(4,),
        chunks_per_arrow_file_values=(16,),
        arrow_compressions=(output_stage_benchmark.types.ArrowCompression.ZSTD,),
    )

    assert len(cases) == 8
    assert {benchmark_case.finalize_parquet for benchmark_case in cases} == {False, True}
    assert {benchmark_case.phenotype_count for benchmark_case in cases} == {1, 8}
    assert {benchmark_case.chunk_size for benchmark_case in cases} == {1024, 8192}
    assert "arrow_chunks_single_phenotype_small_bsize_1024_writer4_queue16_chunks16_zstd" in {
        benchmark_case.name for benchmark_case in cases
    }
    assert "parquet_final_8_phenotypes_large_bsize_8192_writer4_queue16_chunks16_zstd" in {
        benchmark_case.name for benchmark_case in cases
    }


def test_output_stage_benchmark_prepares_multi_phenotype_resources(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    baseline_directory = data_directory / "baselines"
    baseline_directory.mkdir(parents=True)
    phenotype_path = data_directory / "pheno_cont.txt"
    phenotype_path.write_text(
        "FID\tIID\tphenotype_continuous\nf1\ti1\t1.0\nf2\ti2\t2.0\n",
        encoding="utf-8",
    )
    loco_path = baseline_directory / "trait.loco"
    loco_path.write_text("FID_IID 0_i1 0_i2\n22 0.1 0.2\n", encoding="utf-8")
    (baseline_directory / "regenie_step1_qt_pred.list").write_text(
        f"phenotype_continuous {loco_path}\n",
        encoding="utf-8",
    )

    resources = output_stage_benchmark.prepare_phenotype_resources(
        data_directory=data_directory,
        output_directory=tmp_path / "benchmark",
        phenotype_count=3,
    )

    assert resources.phenotype_names == ("output_trait_00", "output_trait_01", "output_trait_02")
    assert resources.phenotype_path.exists()
    assert resources.prediction_list_path.read_text(encoding="utf-8").count(str(loco_path)) == 3


def test_deep_profile_builds_cache_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G_PROFILE_GPU_JAX_CACHE_PARENT", str(tmp_path / "gpu_cache"))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    candidate = deep_profile.Step2Candidate(
        trait_type="binary",
        device="gpu",
        chunk_size=8192,
        staging_depth=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
        output_writer_thread_count=4,
        output_writer_queue_depth=8,
        bgen_decode_tile_variant_count=128,
        rayon_thread_count=2,
        firth_batch_size=64,
    )
    environment = deep_profile.build_g_trial_environment(
        candidate=candidate,
        cache_directory=tmp_path / "jax_cache",
        stage_timing_path=tmp_path / "stages.json",
    )
    assert "JAX_COMPILATION_CACHE_DIR" not in environment
    assert "G_REGENIE2_STAGE_TIMINGS_JSON" not in environment
    assert "G_BGEN_DECODE_TILE_VARIANT_COUNT" not in environment
    assert "RAYON_NUM_THREADS" not in environment
    assert "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE" not in environment
    assert "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in environment


def test_deep_profile_arguments_can_focus_workload_grid() -> None:
    arguments = deep_profile.build_arguments_from_overrides(
        [
            "tool.workload_keys=[binary_gpu]",
        ]
    )

    assert arguments.workload_keys == "binary_gpu"
    assert deep_profile.parse_profile_workload_keys(arguments.workload_keys) == (
        deep_profile.ProfileWorkloadKey.BINARY_GPU,
    )


def test_deep_profile_child_command_contains_binary_controls() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    candidate = deep_profile.Step2Candidate(
        trait_type="binary",
        device="cpu",
        chunk_size=4096,
        staging_depth=2,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
        output_writer_thread_count=1,
        output_writer_queue_depth=2,
        bgen_decode_tile_variant_count=None,
        rayon_thread_count=None,
        firth_batch_size=32,
    )
    command = deep_profile.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=Path("data/profiles/out"),
        variant_limit=1000,
    )
    command_text = command[2]
    assert command[:2] == [sys.executable, "-c"]
    assert "phenotype_binary" in command_text
    assert "\"g-device\": 'cpu'" in command_text
    assert '"bsize": 4096' in command_text
    assert '"g-variant-limit": 1000' in command_text
    assert '"firth": True' in command_text
    assert "count_artifact_rows" in command_text
    assert "parts" in command_text
    assert "jax_probe_device_platform" in command_text


def test_deep_profile_artifact_manifest_records_tools_and_skips(tmp_path: Path) -> None:
    output_directory = tmp_path / "profile"
    output_directory.mkdir()
    (output_directory / "summary.json").write_text("{}\n", encoding="utf-8")
    profiler_tool_status = {
        "py_spy": deep_profile.ProfilerToolStatus(
            tool_name="py_spy",
            enabled=True,
            available=False,
            executable_path=None,
            notes="py-spy is not on PATH.",
        )
    }
    summary_payload = {
        "deep_profiles": {
            "sampling_profiles": [
                {
                    "name": "profile_binary_gpu_py_spy",
                    "status": "skipped",
                    "notes": "py-spy is not on PATH.",
                },
                {
                    "name": "profile_binary_gpu_scalene",
                    "implementation": "Scalene",
                    "status": "failed",
                    "profiler_artifact_path": str(output_directory / "deep_profiles" / "binary_gpu.scalene.json"),
                    "application_output_prefix": str(output_directory / "deep_profiles" / "profile_binary_gpu_scalene"),
                    "application_output_run_directory": str(
                        output_directory / "deep_profiles" / "profile_binary_gpu_scalene.g"
                    ),
                    "stage_timing_path": str(
                        output_directory / "deep_profiles" / "profile_binary_gpu_scalene.stage_timings.json"
                    ),
                },
            ]
        }
    }

    manifest = deep_profile.collect_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
    )

    assert manifest["artifact_paths"] == ["summary.json"]
    assert manifest["profiler_tools"]["py_spy"]["available"] is False
    assert manifest["profiler_runs"] == [
        {
            "name": "profile_binary_gpu_scalene",
            "implementation": "Scalene",
            "status": "failed",
            "profiler_artifact_path": "deep_profiles/binary_gpu.scalene.json",
            "application_output_prefix": "deep_profiles/profile_binary_gpu_scalene",
            "application_output_run_directory": "deep_profiles/profile_binary_gpu_scalene.g",
            "stage_timing_path": "deep_profiles/profile_binary_gpu_scalene.stage_timings.json",
        }
    ]
    assert manifest["skipped_profiles"] == [summary_payload["deep_profiles"]["sampling_profiles"][0]]


def test_deep_profile_bounded_regenie_baseline_uses_extract_list(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    baseline_directory = data_directory / "baselines"
    baseline_directory.mkdir(parents=True)
    bgen_path = data_directory / "1kg_chr22_full.bgen"
    bgen_path.write_text("", encoding="utf-8")
    pvar_path = data_directory / "1kg_chr22_full.pvar"
    pvar_path.write_text(
        "#CHROM POS ID REF ALT\n22 100 rs1 A G\n22 200 rs2 C T\n22 300 rs3 G A\n",
        encoding="utf-8",
    )
    baseline_paths = dataclasses.replace(
        baseline_benchmark.build_baseline_paths(),
        data_directory=data_directory,
        baseline_directory=baseline_directory,
        bed_prefix=data_directory / "1kg_chr22_full",
        bgen_path=bgen_path,
        sample_path=data_directory / "1kg_chr22_full.sample",
        continuous_phenotype_path=data_directory / "pheno_cont.txt",
        binary_phenotype_path=data_directory / "pheno_bin.txt",
        covariate_path=data_directory / "covariates.txt",
        regenie_prediction_list_path=baseline_directory / "regenie_step1_pred.list",
        regenie_qt_prediction_list_path=baseline_directory / "regenie_step1_qt_pred.list",
    )
    arguments = deep_profile.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.variant_limit=2",
        ]
    )

    scope = deep_profile.build_regenie_baseline_scope(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=tmp_path / "profile",
    )
    deep_profile.write_regenie_baseline_extract_file(scope)
    command_arguments = deep_profile.build_regenie_step2_command(
        trait_type="quantitative",
        regenie_executable="regenie",
        baseline_paths=baseline_paths,
        output_prefix=tmp_path / "profile" / "headline_runs" / "baseline",
        baseline_scope=scope,
    )

    assert scope.status == deep_profile.RegenieBaselineScopeStatus.BOUNDED
    assert scope.selected_variant_count == 2
    assert scope.extract_path is not None
    assert scope.extract_path.read_text(encoding="utf-8") == "rs1\nrs2\n"
    assert "--extract" in command_arguments
    assert str(scope.extract_path) in command_arguments


def test_deep_profile_manifest_records_regenie_commands_and_inputs(tmp_path: Path) -> None:
    output_directory = tmp_path / "profile"
    output_directory.mkdir()
    command_arguments = [
        "regenie",
        "--step",
        "2",
        "--bgen",
        "data/input.bgen",
        "--sample",
        "data/input.sample",
        "--phenoFile",
        "data/pheno.txt",
        "--covarFile",
        "data/covar.txt",
        "--pred",
        "data/pred.list",
        "--extract",
        "data/extract.txt",
        "--out",
        "data/out",
    ]
    summary_payload = {
        "preflight": {"input_file_sizes": {"data/input.bgen": 123}},
        "setup_results": [],
        "headline_results": [
            dataclasses.asdict(
                deep_profile.AggregateResult(
                    name="headline_regenie_quantitative",
                    implementation="regenie",
                    trait_type="quantitative",
                    device="external_cpu",
                    status="success",
                    trial_count=1,
                    warmup_count=0,
                    median_wall_time_seconds=1.0,
                    mean_wall_time_seconds=1.0,
                    min_wall_time_seconds=1.0,
                    max_wall_time_seconds=1.0,
                    standard_deviation_seconds=0.0,
                    rows_per_second=10.0,
                    trials=[
                        deep_profile.TrialResult(
                            name="headline_regenie_quantitative_trial00",
                            implementation="regenie",
                            trait_type="quantitative",
                            device="external_cpu",
                            status="success",
                            wall_time_seconds=1.0,
                            output_row_count=10,
                            stdout_log_path="stdout.log",
                            stderr_log_path="stderr.log",
                            command_arguments=command_arguments,
                            environment_overrides={},
                        )
                    ],
                )
            )
        ],
        "regenie_baseline_scope": {"status": "bounded"},
    }

    manifest = deep_profile.collect_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status={},
        summary_payload=summary_payload,
    )

    baseline_command = manifest["regenie_baseline_commands"][0]
    assert baseline_command["name"] == "headline_regenie_quantitative_trial00"
    assert baseline_command["binary"] is not None
    assert baseline_command["command_arguments"] == command_arguments
    assert baseline_command["input_files"] == [
        "data/covar.txt",
        "data/extract.txt",
        "data/input.bgen",
        "data/input.sample",
        "data/pheno.txt",
        "data/pred.list",
    ]
    assert manifest["input_files"] == [{"path": "data/input.bgen", "size_bytes": 123}]
    assert manifest["regenie_baseline_scope"] == {"status": "bounded"}


def test_deep_profile_scalene_command_uses_run_subcommand(tmp_path: Path) -> None:
    tool_status = deep_profile.ProfilerToolStatus(
        tool_name="scalene",
        enabled=True,
        available=True,
        executable_path="/usr/bin/uv",
        notes="scalene will run through uv.",
    )
    profile_script_path = tmp_path / "profile_child.py"
    output_path = tmp_path / "profile.scalene.json"

    command_arguments = deep_profile.build_scalene_command_arguments(
        tool_status=tool_status,
        output_path=output_path,
        profile_script_path=profile_script_path,
    )

    assert command_arguments == [
        "/usr/bin/uv",
        "run",
        "--no-sync",
        "--with",
        "scalene",
        "scalene",
        "run",
        "--outfile",
        str(output_path),
        str(profile_script_path),
    ]


def test_deep_profile_memray_command_uses_project_environment(tmp_path: Path) -> None:
    tool_status = deep_profile.ProfilerToolStatus(
        tool_name="memray",
        enabled=True,
        available=True,
        executable_path="/usr/bin/uv",
        notes="memray will run through uv.",
    )
    profile_script_path = tmp_path / "profile_child.py"
    output_path = tmp_path / "profile.memray.bin"

    command_arguments = deep_profile.build_memray_command_arguments(
        tool_status=tool_status,
        output_path=output_path,
        profile_script_path=profile_script_path,
    )

    assert command_arguments == [
        "/usr/bin/uv",
        "run",
        "--no-sync",
        "--with",
        "memray",
        "python",
        "-m",
        "memray",
        "run",
        "--force",
        "--native",
        "--output",
        str(output_path),
        str(profile_script_path),
    ]


def test_deep_profile_logging_perturbation_rows_compare_against_off() -> None:
    rows = deep_profile.build_logging_perturbation_rows(
        [
            {
                "winner_key": "binary_gpu",
                "case": {"name": "telemetry_off"},
                "trial": {"status": "success", "wall_time_seconds": 2.0},
            },
            {
                "winner_key": "binary_gpu",
                "case": {"name": "trace_file_lossy_capped"},
                "trial": {"status": "success", "wall_time_seconds": 2.5},
            },
        ]
    )

    assert rows == [
        {
            "winner_key": "binary_gpu",
            "case_name": "telemetry_off",
            "wall_time_seconds": 2.0,
            "delta_vs_off_seconds": 0.0,
            "ratio_vs_off": 1.0,
            "status": "success",
        },
        {
            "winner_key": "binary_gpu",
            "case_name": "trace_file_lossy_capped",
            "wall_time_seconds": 2.5,
            "delta_vs_off_seconds": 0.5,
            "ratio_vs_off": 1.25,
            "status": "success",
        },
    ]


def test_deep_profile_full_bundle_builds_profiler_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    candidate = deep_profile.Step2Candidate(
        trait_type="binary",
        device="gpu",
        chunk_size=8192,
        staging_depth=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
        output_writer_thread_count=4,
        output_writer_queue_depth=8,
        bgen_decode_tile_variant_count=128,
        rayon_thread_count=2,
        firth_batch_size=64,
    )
    winner_command = deep_profile.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=tmp_path / "winner",
        variant_limit=1000,
    )
    winner_trial = deep_profile.TrialResult(
        name="winner",
        implementation="g",
        trait_type="binary",
        device="gpu",
        status="success",
        wall_time_seconds=1.0,
        output_row_count=1000,
        stdout_log_path="stdout.log",
        stderr_log_path="stderr.log",
        command_arguments=winner_command,
        environment_overrides={},
    )
    winner = deep_profile.AggregateResult(
        name="winner",
        implementation="g",
        trait_type="binary",
        device="gpu",
        status="success",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=1.0,
        mean_wall_time_seconds=1.0,
        min_wall_time_seconds=1.0,
        max_wall_time_seconds=1.0,
        standard_deviation_seconds=0.0,
        rows_per_second=1000.0,
        trials=[winner_trial],
    )
    arguments = deep_profile.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.variant_limit=1000",
            "tool.enable_scalene=true",
            "tool.enable_memray=true",
        ]
    )
    logged_commands: list[tuple[str, list[str]]] = []
    metadata_commands: list[list[str]] = []
    jax_profile_names: list[str] = []

    def fake_run_g_trial(**keyword_arguments: typing.Any) -> deep_profile.TrialResult:
        jax_profile_names.append(str(keyword_arguments["name"]))
        output_prefix = typing.cast("Path", keyword_arguments["output_directory"]) / str(keyword_arguments["name"])
        stage_timing_path = output_prefix.parent / f"{output_prefix.name}.stage_timings.json"
        profile_command = deep_profile.build_g_step2_child_command(
            baseline_paths=typing.cast("baseline_benchmark.BaselinePaths", keyword_arguments["baseline_paths"]),
            candidate=typing.cast("deep_profile.Step2Candidate", keyword_arguments["candidate"]),
            output_prefix=output_prefix,
            variant_limit=typing.cast("int | None", keyword_arguments["variant_limit"]),
            cache_directory=typing.cast("Path", keyword_arguments["cache_directory"]),
            stage_timing_path=stage_timing_path,
            trace_directory=typing.cast("Path | None", keyword_arguments.get("trace_directory")),
            memory_profile_path=typing.cast("Path | None", keyword_arguments.get("memory_profile_path")),
        )
        return deep_profile.TrialResult(
            name=str(keyword_arguments["name"]),
            implementation="g",
            trait_type=candidate.trait_type,
            device=candidate.device,
            status="success",
            wall_time_seconds=1.0,
            output_row_count=1000,
            stdout_log_path="stdout.log",
            stderr_log_path="stderr.log",
            command_arguments=profile_command,
            environment_overrides={},
            stage_timing_path=str(stage_timing_path),
            application_output_prefix=str(output_prefix),
            application_output_run_directory=str(deep_profile.build_application_output_run_directory(output_prefix)),
        )

    def fake_run_logged_command(**keyword_arguments: typing.Any) -> deep_profile.TrialResult:
        logged_commands.append(
            (
                str(keyword_arguments["implementation"]),
                [str(value) for value in typing.cast("list[object]", keyword_arguments["command_arguments"])],
            )
        )
        return deep_profile.TrialResult(
            name=str(keyword_arguments["name"]),
            implementation=str(keyword_arguments["implementation"]),
            trait_type=str(keyword_arguments["trait_type"]),
            device=str(keyword_arguments["device"]),
            status="success",
            wall_time_seconds=1.0,
            output_row_count=None,
            stdout_log_path="stdout.log",
            stderr_log_path="stderr.log",
            command_arguments=typing.cast("list[str]", keyword_arguments["command_arguments"]),
            environment_overrides=typing.cast("dict[str, str]", keyword_arguments["environment_overrides"]),
        )

    def fake_command_output(
        command_arguments: list[str],
        environment_overrides: dict[str, str] | None = None,
    ) -> dict[str, typing.Any]:
        del environment_overrides
        metadata_commands.append(command_arguments)
        return {"command": command_arguments, "returncode": 0, "stdout": "profile\n", "stderr": ""}

    def fake_which(command_name: str) -> str | None:
        if command_name in {"cargo", "py-spy", "perf", "uv"}:
            return f"/usr/bin/{command_name}"
        return None

    def fake_python_module_is_available(module_name: str) -> bool:
        del module_name
        return False

    monkeypatch.setattr(deep_profile, "run_g_trial", fake_run_g_trial)
    monkeypatch.setattr(deep_profile, "run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(deep_profile, "command_output", fake_command_output)
    monkeypatch.setattr(deep_profile, "python_module_is_available", fake_python_module_is_available)
    monkeypatch.setattr(deep_profile.shutil, "which", fake_which)

    results = deep_profile.run_deep_profiles(
        arguments=arguments,
        baseline_paths=baseline_paths,
        winners={"binary_gpu": winner},
        output_directory=tmp_path / "profile",
        cache_directory=tmp_path / "profile" / "jax_cache",
    )

    implementations = [implementation for implementation, _command in logged_commands]
    assert jax_profile_names == ["profile_binary_gpu_jax"]
    assert implementations == ["cProfile", "py-spy", "Scalene", "Memray", "perf"]
    assert metadata_commands[:2] == [
        ["cargo", "bench", "--bench", "bgen_read"],
        ["cargo", "bench", "--bench", "preprocess"],
    ]
    assert any(command[0].endswith("py-spy") and "--format" in command for _implementation, command in logged_commands)
    assert any(command[0].endswith("perf") and "record" in command for _implementation, command in logged_commands)
    profile_directory = tmp_path / "profile" / "deep_profiles"
    sampling_profiles = typing.cast("list[dict[str, object]]", results["sampling_profiles"])
    sampling_profile_by_name = {str(profile["name"]): profile for profile in sampling_profiles}
    expected_profiler_artifacts = {
        "cprofile": "binary_gpu.cprofile",
        "py_spy": "binary_gpu.speedscope.json",
        "scalene": "binary_gpu.scalene.json",
        "memray": "binary_gpu.memray.bin",
        "perf": "binary_gpu.perf.data",
    }
    for profiler_suffix, artifact_name in expected_profiler_artifacts.items():
        profile_name = f"profile_binary_gpu_{profiler_suffix}"
        expected_stage_timing_path = profile_directory / f"{profile_name}.stage_timings.json"
        profile_result = sampling_profile_by_name[profile_name]
        assert profile_result["profiler_artifact_path"] == str(profile_directory / artifact_name)
        assert profile_result["application_output_prefix"] == str(profile_directory / profile_name)
        assert profile_result["application_output_run_directory"] == str(profile_directory / f"{profile_name}.g")
        assert profile_result["stage_timing_path"] == str(expected_stage_timing_path)
        script_text = (profile_directory / f"{profile_name}_child.py").read_text(encoding="utf-8")
        assert f"\"out\": '{profile_directory / profile_name}'" in script_text
        assert f"\"g-stage-timings-json\": '{expected_stage_timing_path}'" in script_text
        assert "profile_binary_gpu_profiler" not in script_text
    application_output_run_directories = {
        str(profile["application_output_run_directory"])
        for profile in sampling_profiles
        if profile.get("application_output_run_directory") is not None
    }
    assert len(application_output_run_directories) == 6
    assert len(results["sampling_profiles"]) == 6


def test_deep_profile_aggregates_trial_results() -> None:
    trial_results = [
        deep_profile.TrialResult(
            name="trial0",
            implementation="g",
            trait_type="quantitative",
            device="gpu",
            status="success",
            wall_time_seconds=2.0,
            output_row_count=100,
            stdout_log_path="stdout0",
            stderr_log_path="stderr0",
            command_arguments=["python"],
            environment_overrides={},
        ),
        deep_profile.TrialResult(
            name="trial1",
            implementation="g",
            trait_type="quantitative",
            device="gpu",
            status="success",
            wall_time_seconds=1.0,
            output_row_count=100,
            stdout_log_path="stdout1",
            stderr_log_path="stderr1",
            command_arguments=["python"],
            environment_overrides={},
        ),
    ]
    aggregate = deep_profile.aggregate_trial_results(
        name="headline_g_quantitative_gpu",
        implementation="g",
        trait_type="quantitative",
        device="gpu",
        warmup_count=1,
        trial_results=trial_results,
    )
    assert aggregate.status == "success"
    assert aggregate.median_wall_time_seconds == 1.5
    assert aggregate.rows_per_second == 100 / 1.5


def test_deep_profile_runtime_comparison_uses_regenie_baseline() -> None:
    regenie_result = deep_profile.AggregateResult(
        name="headline_regenie_quantitative",
        implementation="regenie",
        trait_type="quantitative",
        device="external_cpu",
        status="success",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=10.0,
        mean_wall_time_seconds=10.0,
        min_wall_time_seconds=10.0,
        max_wall_time_seconds=10.0,
        standard_deviation_seconds=0.0,
        rows_per_second=10.0,
        trials=[],
    )
    g_result = deep_profile.AggregateResult(
        name="headline_g_quantitative_gpu",
        implementation="g",
        trait_type="quantitative",
        device="gpu",
        status="success",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=2.5,
        mean_wall_time_seconds=2.5,
        min_wall_time_seconds=2.5,
        max_wall_time_seconds=2.5,
        standard_deviation_seconds=0.0,
        rows_per_second=40.0,
        trials=[],
    )
    comparisons = deep_profile.build_runtime_comparisons([regenie_result, g_result])
    comparison = comparisons["headline_g_quantitative_gpu_vs_regenie_quantitative"]
    assert comparison["speedup_ratio"] == 4.0
    assert comparison["absolute_delta_seconds"] == -7.5


def test_deep_profile_runtime_comparison_notes_separate_unsupported_and_failed(tmp_path: Path) -> None:
    unsupported_regenie = deep_profile.unsupported_aggregate_result(
        name="headline_regenie_quantitative",
        trait_type="quantitative",
        device="external_cpu",
        log_directory=tmp_path / "logs",
        notes="REGENIE executable is unavailable.",
    )
    failed_regenie_trial = deep_profile.TrialResult(
        name="headline_regenie_binary_trial00",
        implementation="regenie",
        trait_type="binary",
        device="external_cpu",
        status="failed",
        wall_time_seconds=1.0,
        output_row_count=None,
        stdout_log_path="stdout.log",
        stderr_log_path="stderr.log",
        command_arguments=["regenie"],
        environment_overrides={},
        notes="Command exited with code 1.",
    )
    failed_regenie = deep_profile.aggregate_trial_results(
        name="headline_regenie_binary",
        implementation="regenie",
        trait_type="binary",
        device="external_cpu",
        warmup_count=0,
        trial_results=[failed_regenie_trial],
    )
    g_quantitative = deep_profile.AggregateResult(
        name="headline_g_quantitative_gpu",
        implementation="g",
        trait_type="quantitative",
        device="gpu",
        status="success",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=2.0,
        mean_wall_time_seconds=2.0,
        min_wall_time_seconds=2.0,
        max_wall_time_seconds=2.0,
        standard_deviation_seconds=0.0,
        rows_per_second=10.0,
        trials=[],
    )
    g_binary = dataclasses.replace(g_quantitative, name="headline_g_binary_gpu", trait_type="binary")

    notes = deep_profile.build_runtime_comparison_notes([unsupported_regenie, failed_regenie, g_quantitative, g_binary])

    assert len(notes.unsupported) == 1
    assert "unsupported" in notes.unsupported[0]
    assert "REGENIE executable is unavailable" in notes.unsupported[0]
    assert len(notes.failed) == 1
    assert "did not produce a measured runtime" in notes.failed[0]
    assert "Command exited with code 1" in notes.failed[0]


def test_quantitative_step2_comparison_uses_full_variant_identity_when_available(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text(
        "\n".join(
            [
                "CHROM GENPOS ID ALLELE0 ALLELE1 BETA LOG10P",
                "1 100 rs1 A G 0.1 1.0",
                "1 101 rs1 A T 0.9 9.0",
            ]
        )
        + "\n"
    )
    pl.DataFrame(
        {
            "CHROM": [1],
            "GENPOS": [100],
            "ID": ["rs1"],
            "ALLELE0": ["A"],
            "ALLELE1": ["G"],
            "BETA": [0.1],
            "LOG10P": [1.0],
        }
    ).write_parquet(g_output)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 1
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True


def test_quantitative_step2_comparison_coerces_merge_key_types(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text(
        "\n".join(
            [
                "CHROM GENPOS ID ALLELE0 ALLELE1 BETA LOG10P",
                "22 100 rs1 A G 0.1 1.0",
            ]
        )
        + "\n"
    )
    pl.DataFrame(
        {
            "CHROM": ["22"],
            "GENPOS": [100],
            "ID": ["rs1"],
            "ALLELE0": ["A"],
            "ALLELE1": ["G"],
            "BETA": [0.1],
            "LOG10P": [1.0],
        }
    ).write_parquet(g_output)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 1


def test_quantitative_step2_comparison_reads_parquet_outputs(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text("CHROM GENPOS ID BETA LOG10P\n1 100 rs1 0.1 1.0\n1 200 rs2 0.2 2.0\n")
    pl.DataFrame(
        {
            "ID": ["rs1", "rs2"],
            "BETA": [0.1, 0.2],
            "LOG10P": [1.0, 2.0],
        }
    ).write_parquet(g_output)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 2
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True
