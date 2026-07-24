"""Tests for deep-profile production and diagnostic separation."""

from __future__ import annotations

import tomllib
import types
import typing

from tooling.cli import profile_regenie2_deep
from tooling.profile_deep import commands as profile_deep_commands
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest


def candidate() -> profile_deep_models.Step2Candidate:
    """Build one representative production candidate."""
    return profile_deep_models.Step2Candidate(
        trait_type="binary",
        device="gpu",
        chunk_size=16_384,
        output_writer_thread_count=8,
        rayon_thread_count=8,
        firth_batch_size=512,
    )


def trial_result(name: str) -> profile_deep_models.TrialResult:
    """Build one successful synthetic g trial."""
    return profile_deep_models.TrialResult(
        name=name,
        implementation="g",
        trait_type="binary",
        device="gpu",
        status="success",
        wall_time_seconds=1.0,
        process_wall_time_seconds=1.0,
        output_row_count=4,
        stdout_log_path="stdout.log",
        stderr_log_path="stderr.log",
        command_arguments=[],
        environment_overrides={},
    )


def test_trial_config_defaults_to_production_telemetry_off(tmp_path: Path) -> None:
    """Uninstrumented timing configs are off; profile is an explicit opt-in."""
    production_path = profile_deep_commands.write_trial_config(
        candidate=candidate(),
        output_prefix=tmp_path / "production",
        jax_cache_directory=tmp_path / "cache",
        diagnostic_options=None,
    )
    diagnostic_path = profile_deep_commands.write_trial_config(
        candidate=candidate(),
        output_prefix=tmp_path / "diagnostic",
        jax_cache_directory=tmp_path / "cache",
        diagnostic_options={"telemetry": "profile"},
    )

    assert tomllib.loads(production_path.read_text(encoding="utf-8"))["diagnostics"] == {"telemetry": "off"}
    assert tomllib.loads(diagnostic_path.read_text(encoding="utf-8"))["diagnostics"] == {"telemetry": "profile"}


def test_headline_environment_omits_debug_and_allocator_overrides() -> None:
    """Production timings do not inherit profiler logging or allocator policy."""
    headline_environment = profile_regenie2_deep.build_g_trial_environment(enable_jax_debug_logging=False)
    diagnostic_environment = profile_regenie2_deep.build_g_trial_environment(enable_jax_debug_logging=True)

    assert headline_environment == {}
    assert diagnostic_environment["JAX_LOGGING_LEVEL"] == "DEBUG"
    assert diagnostic_environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_profile_child_retains_native_artifact_lines(tmp_path: Path) -> None:
    """The isolated child verifies native stdout without filesystem discovery."""
    baseline_paths = types.SimpleNamespace(
        binary_phenotype_path=tmp_path / "binary.tsv",
        continuous_phenotype_path=tmp_path / "continuous.tsv",
        regenie_prediction_list_path=tmp_path / "binary.list",
        regenie_qt_prediction_list_path=tmp_path / "linear.list",
        bgen_path=tmp_path / "input.bgen",
        sample_path=tmp_path / "input.sample",
        covariate_path=tmp_path / "covariates.tsv",
    )

    command = profile_deep_commands.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate(),
        output_prefix=tmp_path / "profile",
    )

    child_source = command[2]
    assert "g._core.cli.run" in child_source
    assert "collect_completed_output_evidence" in child_source
    assert '"cli_stdout_chunks": list(native_result.stdout_chunks)' in child_source
    assert "discover_completed_run_directory" not in child_source


def test_repeated_trials_keep_profile_run_out_of_headline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact timing adds one explicit diagnostic without instrumenting measured trials."""
    observed_calls: list[dict[str, typing.Any]] = []

    def fake_run_g_trial(**arguments: typing.Any) -> profile_deep_models.TrialResult:
        observed_calls.append(arguments)
        return trial_result(str(arguments["name"]))

    monkeypatch.setattr(profile_regenie2_deep, "run_g_trial", fake_run_g_trial)

    aggregate = profile_regenie2_deep.run_repeated_g_trials(
        name="binary_gpu",
        baseline_paths=object(),
        candidate=candidate(),
        output_directory=tmp_path / "runs",
        log_directory=tmp_path / "logs",
        cache_directory=tmp_path / "cache",
        warmup_count=1,
        trial_count=2,
        emit_stage_timings=True,
    )

    assert len(aggregate.trials) == 2
    assert len(aggregate.diagnostic_trials) == 1
    measured_calls = [call for call in observed_calls if "_trial" in str(call["name"])]
    assert all(call["emit_stage_timings"] is False for call in measured_calls)
    assert all(call.get("diagnostic_options") is None for call in measured_calls)
    diagnostic_call = next(call for call in observed_calls if str(call["name"]).endswith("_stage_diagnostic"))
    assert diagnostic_call["diagnostic_options"] == {"telemetry": "profile"}
