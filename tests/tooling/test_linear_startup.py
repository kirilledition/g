"""Tests for the quantitative native lifecycle benchmark harness."""

from __future__ import annotations

import tomllib
import typing

import pytest

from tooling.benchmark import linear_startup
from tooling.common import g_regenie as tooling_g_regenie

if typing.TYPE_CHECKING:
    from pathlib import Path


def test_clone_helpers_generate_current_multi_trait_inputs(tmp_path: Path) -> None:
    """Cloned traits retain sample identifiers and resolve the LOCO path."""
    phenotype_source = tmp_path / "phenotypes.tsv"
    phenotype_source.write_text(
        "FID\tIID\tphenotype_continuous\nfamily\tindividual\t1.5\n",
        encoding="utf-8",
    )
    prediction_source = tmp_path / "predictions.list"
    prediction_source.write_text("phenotype_continuous loco.tsv\n", encoding="utf-8")
    phenotype_destination = tmp_path / "generated" / "phenotypes.tsv"
    prediction_destination = tmp_path / "prediction-generated" / "nested" / "predictions.list"
    phenotype_names = ("trait_1", "trait_2")

    linear_startup.write_cloned_phenotype_table(
        source_path=phenotype_source,
        destination_path=phenotype_destination,
        phenotype_names=phenotype_names,
    )
    linear_startup.write_cloned_prediction_list(
        source_path=prediction_source,
        destination_path=prediction_destination,
        phenotype_names=phenotype_names,
    )

    assert phenotype_destination.read_text(encoding="utf-8") == (
        "FID\tIID\ttrait_1\ttrait_2\nfamily\tindividual\t1.5\t1.5\n"
    )
    expected_loco_path = (tmp_path / "loco.tsv").resolve()
    assert prediction_destination.read_text(encoding="utf-8") == (
        f"trait_1 {expected_loco_path}\ntrait_2 {expected_loco_path}\n"
    )


def test_prediction_clone_selects_named_row_independent_of_order(tmp_path: Path) -> None:
    """Only the requested trait supplies the cloned LOCO path."""
    prediction_source = tmp_path / "predictions.list"
    prediction_source.write_text(
        "unrelated other.tsv\nphenotype_continuous wanted.tsv\nanother ignored.tsv\n",
        encoding="utf-8",
    )
    prediction_destination = tmp_path / "generated" / "predictions.list"

    linear_startup.write_cloned_prediction_list(
        source_path=prediction_source,
        destination_path=prediction_destination,
        phenotype_names=("trait_1",),
    )

    assert prediction_destination.read_text(encoding="utf-8") == (f"trait_1 {(tmp_path / 'wanted.tsv').resolve()}\n")


@pytest.mark.parametrize(
    "source_text, expected_message",
    [
        pytest.param("other loco.tsv\n", "exactly one", id="missing-trait"),
        pytest.param(
            "phenotype_continuous one.tsv\nphenotype_continuous two.tsv\n",
            "exactly one",
            id="duplicate-trait",
        ),
        pytest.param("phenotype_continuous\n", "row 1", id="malformed-row"),
    ],
)
def test_prediction_clone_rejects_ambiguous_or_malformed_sources(
    tmp_path: Path, source_text: str, expected_message: str
) -> None:
    """Malformed prediction lists fail before benchmark execution."""
    prediction_source = tmp_path / "predictions.list"
    prediction_source.write_text(source_text, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        linear_startup.write_cloned_prediction_list(
            source_path=prediction_source,
            destination_path=tmp_path / "generated" / "predictions.list",
            phenotype_names=("trait_1",),
        )


def test_linear_default_cache_is_campaign_local(tmp_path: Path) -> None:
    """A null cache setting resolves below the new output campaign."""
    output_directory = tmp_path / "campaign"

    arguments = linear_startup.build_arguments_from_overrides([f"tool.output_dir={output_directory}"])

    assert arguments.jax_cache_dir == output_directory / "jax-cache"


def test_linear_invalid_device_is_rejected(tmp_path: Path) -> None:
    """Unsupported devices fail during typed configuration adaptation."""
    with pytest.raises(ValueError, match="not a valid RegenieDevice"):
        linear_startup.build_arguments_from_overrides([f"tool.output_dir={tmp_path / 'campaign'}", "tool.device=tpu"])


def test_build_run_spec_renders_production_parquet_config(tmp_path: Path) -> None:
    """The lifecycle harness uses current TOML and telemetry-off headlines."""
    arguments = linear_startup.LinearStartupArguments(
        device=tooling_g_regenie.RegenieDevice.GPU,
        chunk_size=16_384,
        cpu_threads=8,
        output_writer_thread_count=8,
        include_fresh_process=True,
        hot_run_count=5,
        diagnostic_run_count=0,
        multi_phenotype_count=1,
        multi_phenotype_sample_mode=tooling_g_regenie.RegenieMultiPhenotypeSampleMode.COMPLETE_CASE,
        expected_variant_count=418_943,
        data_dir=tmp_path,
        output_dir=tmp_path / "output",
        jax_cache_dir=tmp_path / "cache",
        python_executable="python",
        json_summary_path=None,
    )
    inputs = linear_startup.BenchmarkInputs(
        bgen_path=tmp_path / "input.bgen",
        sample_path=tmp_path / "input.sample",
        phenotype_path=tmp_path / "phenotypes.tsv",
        phenotype_names=("phenotype_continuous",),
        covariate_path=tmp_path / "covariates.tsv",
        prediction_list_path=tmp_path / "predictions.list",
    )

    spec = linear_startup.build_run_spec(
        arguments,
        inputs,
        output_root=tmp_path / "runs" / "hot_01",
        telemetry=tooling_g_regenie.RegenieTelemetry.OFF,
    )
    parsed = tomllib.loads(tooling_g_regenie.render_regenie_toml(spec))

    assert parsed["trait"] == {"trait_type": "quantitative", "bsize": 16_384}
    assert parsed["compute"]["jax_cache_dir"] == str(tmp_path / "cache")
    assert parsed["output"]["writer_threads"] == 8
    assert parsed["diagnostics"] == {"telemetry": "off"}


def test_profile_diagnostics_use_fresh_processes(tmp_path: Path) -> None:
    """Profile telemetry never follows telemetry-off work in the hot process."""
    arguments = linear_startup.LinearStartupArguments(
        device=tooling_g_regenie.RegenieDevice.GPU,
        chunk_size=16_384,
        cpu_threads=8,
        output_writer_thread_count=8,
        include_fresh_process=True,
        hot_run_count=2,
        diagnostic_run_count=2,
        multi_phenotype_count=1,
        multi_phenotype_sample_mode=tooling_g_regenie.RegenieMultiPhenotypeSampleMode.COMPLETE_CASE,
        expected_variant_count=418_943,
        data_dir=tmp_path,
        output_dir=tmp_path / "output",
        jax_cache_dir=tmp_path / "cache",
        python_executable="python",
        json_summary_path=None,
    )

    plans = linear_startup.build_trial_plans(arguments)

    profile_plans = [plan for plan in plans if plan.telemetry == tooling_g_regenie.RegenieTelemetry.PROFILE]
    assert len(profile_plans) == 2
    assert all(plan.fresh_process for plan in profile_plans)
    assert all(
        not plan.fresh_process
        for plan in plans
        if plan.role in {"discarded_compile_warmup", "same_process_hot_production"}
    )
