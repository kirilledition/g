"""Tests for the supported binary lifecycle benchmark harness."""

from __future__ import annotations

import typing

import pytest

from tooling.cli import benchmark_regenie2_binary_hot
from tooling.common import g_regenie as tooling_g_regenie

if typing.TYPE_CHECKING:
    from pathlib import Path


def benchmark_arguments(tmp_path: Path) -> benchmark_regenie2_binary_hot.BenchmarkArguments:
    """Build a minimal binary lifecycle configuration."""
    return benchmark_regenie2_binary_hot.BenchmarkArguments(
        data_directory=tmp_path,
        bgen_path=tmp_path / "input.bgen",
        sample_path=tmp_path / "input.sample",
        phenotype_path=tmp_path / "phenotypes.tsv",
        covariate_path=tmp_path / "covariates.tsv",
        prediction_list_path=tmp_path / "predictions.list",
        phenotype_column="phenotype_binary",
        covariate_columns=("age", "sex"),
        output_directory=tmp_path / "output",
        device=tooling_g_regenie.RegenieDevice.GPU,
        chunk_size=16_384,
        firth_batch_size=512,
        firth_candidate_capacity=1_024,
        writer_thread_count=8,
        p_threshold=0.05,
        expected_variant_count=418_943,
        jax_cache_directory=tmp_path / "cache",
        include_fresh_process=True,
        hot_run_count=3,
        diagnostic_run_count=2,
        python_executable="python",
        summary_path=None,
    )


def test_binary_trial_plan_has_unique_isolated_diagnostics(tmp_path: Path) -> None:
    """Profile diagnostics are unique fresh processes outside headline timing."""
    plans = benchmark_regenie2_binary_hot.build_trial_plans(benchmark_arguments(tmp_path))

    assert len(plans) == 7
    assert len({plan.name for plan in plans}) == len(plans)
    diagnostic_plans = [plan for plan in plans if plan.telemetry == tooling_g_regenie.RegenieTelemetry.PROFILE]
    assert len(diagnostic_plans) == 2
    assert all(plan.fresh_process and not plan.headline for plan in diagnostic_plans)
    assert all(plan.telemetry == tooling_g_regenie.RegenieTelemetry.OFF for plan in plans if plan.headline)


def test_binary_default_cache_is_campaign_local(tmp_path: Path) -> None:
    """A null cache setting resolves below the new output campaign."""
    output_directory = tmp_path / "campaign"

    arguments = benchmark_regenie2_binary_hot.build_arguments_from_overrides([f"tool.output_dir={output_directory}"])

    assert arguments.jax_cache_directory == output_directory / "jax-cache"


def test_binary_invalid_device_is_rejected(tmp_path: Path) -> None:
    """Unsupported devices fail during typed configuration adaptation."""
    with pytest.raises(ValueError, match="not a valid RegenieDevice"):
        benchmark_regenie2_binary_hot.build_arguments_from_overrides(
            [f"tool.output_dir={tmp_path / 'campaign'}", "tool.device=tpu"]
        )


def test_binary_campaign_rejects_prepopulated_cache(tmp_path: Path) -> None:
    """The discarded compile warm must start from an empty campaign cache."""
    arguments = benchmark_arguments(tmp_path)
    arguments.jax_cache_directory.mkdir()
    (arguments.jax_cache_directory / "old-entry").write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires an empty campaign cache"):
        benchmark_regenie2_binary_hot.run_benchmark(arguments)
