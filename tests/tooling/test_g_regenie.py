"""Tests for the supported tooling REGENIE config boundary."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from tooling.common import g_regenie as tooling_g_regenie


def quantitative_spec() -> tooling_g_regenie.RegenieRunSpec:
    """Build a representative current production run spec."""
    return tooling_g_regenie.RegenieRunSpec(
        trait_kind=tooling_g_regenie.RegenieTraitKind.QUANTITATIVE,
        command_prefix=("uv", "run", "g", "regenie"),
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=Path("input.bgen"),
            sample_path=Path("input.sample"),
            phenotype_path=Path("phenotypes.tsv"),
            phenotype_columns=("trait_a", "trait_b"),
            covariate_path=Path("covariates.tsv"),
            covariate_columns=("age", "sex"),
            prediction_list_path=Path("predictions.list"),
            output_prefix=Path("results/linear"),
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=tooling_g_regenie.RegenieDevice.GPU,
            bsize=16_384,
            cpu_threads=8,
            multi_phenotype_sample_mode=tooling_g_regenie.RegenieMultiPhenotypeSampleMode.COMPLETE_CASE,
            jax_cache_dir=Path("cache/jax"),
        ),
        output=tooling_g_regenie.RegenieOutputOptions(writer_threads=8, resume=False),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(telemetry=tooling_g_regenie.RegenieTelemetry.OFF),
        binary=None,
    )


def test_render_regenie_toml_uses_current_schema() -> None:
    """The renderer emits canonical sections and no removed runtime knobs."""
    rendered = tooling_g_regenie.render_regenie_toml(quantitative_spec())
    parsed = tomllib.loads(rendered)

    assert parsed["input"]["pheno_columns"] == ["trait_a", "trait_b"]
    assert parsed["trait"] == {"trait_type": "quantitative", "bsize": 16_384}
    assert parsed["compute"] == {
        "device": "gpu",
        "cpu_threads": 8,
        "multi_phenotype_sample_mode": "complete-case",
        "jax_cache_dir": "cache/jax",
    }
    assert parsed["output"] == {"out": "results/linear", "writer_threads": 8, "resume": False}
    assert parsed["diagnostics"] == {"telemetry": "off"}
    for removed_option in ("staging_depth", "variant_limit", "writer_queue_depth", "finalize_parquet"):
        assert removed_option not in rendered


def test_write_config_and_render_g_regenie_command(tmp_path: Path) -> None:
    """Config persistence is explicit and the command exposes only --config."""
    config_path = tmp_path / "run.toml"

    tooling_g_regenie.write_regenie_toml(quantitative_spec(), config_path)
    command = tooling_g_regenie.render_g_regenie_command(quantitative_spec(), config_path)

    assert command == ["uv", "run", "g", "regenie", "--config", str(config_path)]
    assert tomllib.loads(config_path.read_text(encoding="utf-8"))["compute"]["device"] == "gpu"
    assert tooling_g_regenie.render_native_cli_arguments(config_path) == [
        "regenie",
        "--config",
        str(config_path),
    ]


def test_binary_spec_requires_binary_options() -> None:
    """A binary trait cannot silently inherit quantitative defaults."""
    invalid_spec = quantitative_spec()
    invalid_spec = tooling_g_regenie.RegenieRunSpec(
        trait_kind=tooling_g_regenie.RegenieTraitKind.BINARY,
        command_prefix=invalid_spec.command_prefix,
        inputs=invalid_spec.inputs,
        compute=invalid_spec.compute,
        output=invalid_spec.output,
        diagnostics=invalid_spec.diagnostics,
        binary=None,
    )

    with pytest.raises(ValueError, match="requires binary options"):
        tooling_g_regenie.render_regenie_toml(invalid_spec)


def test_binary_spec_renders_current_fallback_contract() -> None:
    """Binary options render through the current TOML schema."""
    base_spec = quantitative_spec()
    binary_spec = dataclasses.replace(
        base_spec,
        trait_kind=tooling_g_regenie.RegenieTraitKind.BINARY,
        binary=tooling_g_regenie.RegenieBinaryOptions(
            fallback_method=tooling_g_regenie.RegenieBinaryFallback.FIRTH_APPROXIMATE,
            p_threshold=0.05,
            firth_se=False,
        ),
    )

    parsed = tomllib.loads(tooling_g_regenie.render_regenie_toml(binary_spec))

    assert parsed["binary"] == {
        "fallback_method": "firth_approximate",
        "p_threshold": 0.05,
        "firth_se": False,
    }


def test_score_only_rejects_firth_standard_errors() -> None:
    """Score-only runs cannot request a Firth-derived standard error."""
    base_spec = quantitative_spec()
    invalid_spec = dataclasses.replace(
        base_spec,
        trait_kind=tooling_g_regenie.RegenieTraitKind.BINARY,
        binary=tooling_g_regenie.RegenieBinaryOptions(
            fallback_method=tooling_g_regenie.RegenieBinaryFallback.SCORE_ONLY,
            firth_se=True,
        ),
    )

    with pytest.raises(ValueError, match="cannot request Firth standard errors"):
        tooling_g_regenie.render_regenie_toml(invalid_spec)
