from __future__ import annotations

import json
import typing

import pytest

import tooling.performance_compare as performance_compare

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path


def write_json(path: Path, payload: collections.abc.Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def comparisons_by_name(
    report: performance_compare.ComparisonReport,
) -> dict[str, performance_compare.MetricComparison]:
    return {comparison.name: comparison for comparison in report.comparisons}


def test_explicit_smoke_metrics_compare_speed_memory_and_numerical_deltas(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    new_path = tmp_path / "new.json"
    write_json(
        baseline_path,
        {
            "schema": "g.performance_smoke.v1",
            "metrics": {
                "smoke.wall_time_seconds": {"category": "speed", "unit": "seconds", "value": 10.0},
                "smoke.peak_memory_bytes": {"category": "memory", "unit": "bytes", "value": 2000},
                "smoke.checksum": {"category": "numerical", "unit": "checksum", "value": 40.0},
            },
        },
    )
    write_json(
        new_path,
        {
            "schema": "g.performance_smoke.v1",
            "metrics": {
                "smoke.wall_time_seconds": {"category": "speed", "unit": "seconds", "value": 5.0},
                "smoke.peak_memory_bytes": {"category": "memory", "unit": "bytes", "value": 2500},
                "smoke.checksum": {"category": "numerical", "unit": "checksum", "value": 42.0},
            },
        },
    )

    report = performance_compare.compare_summary_paths(baseline_path, new_path)
    comparison_map = comparisons_by_name(report)

    assert comparison_map["smoke.wall_time_seconds"].ratio == 0.5
    assert comparison_map["smoke.peak_memory_bytes"].delta == 500.0
    assert comparison_map["smoke.checksum"].delta == 2.0
    rendered_report = performance_compare.render_comparison_report(report)
    assert "2x faster" in rendered_report
    assert "smoke.peak_memory_bytes" in rendered_report


def test_binary_hot_summary_metrics_are_extracted(tmp_path: Path) -> None:
    baseline_path = tmp_path / "binary-hot-baseline.json"
    new_path = tmp_path / "binary-hot-new.json"
    baseline_payload = {
        "headline": {"hot_same_process_no_final_seconds": 8.0},
        "headline_by_case": {
            "traits1_variant_major_default_batch64_capacity1024": {
                "hot_same_process_no_final_seconds": 8.0,
            },
        },
        "binary_diagnostics_by_case": {
            "traits1_variant_major_default_batch64_capacity1024": {
                "hot_same_process_no_final": {
                    "available": True,
                    "reason": None,
                    "stage_timing_path": "baseline.json",
                    "stage_timing_mode": "exact",
                    "chunk_count": 2,
                    "candidate_counts": {
                        "score_test": 8,
                        "firth": 4,
                    },
                    "correction_branch_counts": {
                        "pseudo_firth": 3,
                        "newton_raphson_zero_start": 0,
                        "newton_raphson_warm_start": 1,
                    },
                    "stage_totals_seconds": {
                        "jax_compute": 4.0,
                    },
                },
            },
        },
        "results": [
            {
                "name": "hot_same_process_no_final",
                "benchmark_case": {"name": "traits1_variant_major_default_batch64_capacity1024"},
                "wall_time_seconds": 8.0,
                "output_metrics": {
                    "chunk_bytes": 1000,
                    "final_parquet_bytes": None,
                    "output_row_count": 100,
                    "chunk_file_count": 2,
                },
            },
        ],
    }
    new_payload = {
        "headline": {"hot_same_process_no_final_seconds": 6.0},
        "headline_by_case": {
            "traits1_variant_major_default_batch64_capacity1024": {
                "hot_same_process_no_final_seconds": 6.0,
            },
        },
        "binary_diagnostics_by_case": {
            "traits1_variant_major_default_batch64_capacity1024": {
                "hot_same_process_no_final": {
                    "available": True,
                    "reason": None,
                    "stage_timing_path": "new.json",
                    "stage_timing_mode": "exact",
                    "chunk_count": 2,
                    "candidate_counts": {
                        "score_test": 9,
                        "firth": 5,
                    },
                    "correction_branch_counts": {
                        "pseudo_firth": 4,
                        "newton_raphson_zero_start": 0,
                        "newton_raphson_warm_start": 1,
                    },
                    "stage_totals_seconds": {
                        "jax_compute": 3.0,
                    },
                },
            },
        },
        "results": [
            {
                "name": "hot_same_process_no_final",
                "benchmark_case": {"name": "traits1_variant_major_default_batch64_capacity1024"},
                "wall_time_seconds": 6.0,
                "output_metrics": {
                    "chunk_bytes": 1200,
                    "final_parquet_bytes": None,
                    "output_row_count": 100,
                    "chunk_file_count": 2,
                },
            },
        ],
    }
    write_json(baseline_path, baseline_payload)
    write_json(new_path, new_payload)

    report = performance_compare.compare_summary_paths(baseline_path, new_path)
    comparison_map = comparisons_by_name(report)

    assert comparison_map["headline.hot_same_process_no_final_seconds"].delta == -2.0
    assert (
        comparison_map[
            "results.traits1_variant_major_default_batch64_capacity1024.hot_same_process_no_final."
            "output_metrics.chunk_bytes"
        ].ratio
        == 1.2
    )
    assert (
        comparison_map[
            "binary_diagnostics_by_case.traits1_variant_major_default_batch64_capacity1024."
            "hot_same_process_no_final.candidate_counts.firth"
        ].delta
        == 1.0
    )
    assert (
        comparison_map[
            "binary_diagnostics_by_case.traits1_variant_major_default_batch64_capacity1024."
            "hot_same_process_no_final.stage_totals_seconds.jax_compute"
        ].delta
        == -1.0
    )


def test_bgen_reader_case_metrics_are_extracted(tmp_path: Path) -> None:
    baseline_path = tmp_path / "bgen-baseline.json"
    new_path = tmp_path / "bgen-new.json"
    baseline_payload = {
        "cases": [
            {
                "chunk_size": 8192,
                "sample_selection_mode": "full",
                "trusted_no_missing_diploid": False,
                "path_results": [
                    {
                        "path_mode": "variant_major_buffered",
                        "mean_seconds": 2.0,
                        "median_seconds": 1.8,
                        "checksum": 123.0,
                    },
                ],
            },
        ],
    }
    new_payload = {
        "cases": [
            {
                "chunk_size": 8192,
                "sample_selection_mode": "full",
                "trusted_no_missing_diploid": False,
                "path_results": [
                    {
                        "path_mode": "variant_major_buffered",
                        "mean_seconds": 1.0,
                        "median_seconds": 0.9,
                        "checksum": 123.5,
                    },
                ],
            },
        ],
    }
    write_json(baseline_path, baseline_payload)
    write_json(new_path, new_payload)

    report = performance_compare.compare_summary_paths(baseline_path, new_path)
    comparison_map = comparisons_by_name(report)

    assert (
        comparison_map["cases.case0.chunk8192.selectionfull.trustedfalse.variant_major_buffered.median_seconds"].ratio
        == 0.5
    )
    assert (
        comparison_map["cases.case0.chunk8192.selectionfull.trustedfalse.variant_major_buffered.checksum"].delta == 0.5
    )


def test_malformed_json_fails(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    new_path = tmp_path / "new.json"
    baseline_path.write_text("{", encoding="utf-8")
    write_json(new_path, {"metrics": {"wall_time_seconds": 1.0}})

    with pytest.raises(performance_compare.PerformanceComparisonError):
        performance_compare.compare_summary_paths(baseline_path, new_path)


def test_nonnumeric_metric_fails(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    new_path = tmp_path / "new.json"
    write_json(baseline_path, {"metrics": {"wall_time_seconds": "fast"}})
    write_json(new_path, {"metrics": {"wall_time_seconds": 1.0}})

    with pytest.raises(performance_compare.PerformanceComparisonError):
        performance_compare.compare_summary_paths(baseline_path, new_path)
