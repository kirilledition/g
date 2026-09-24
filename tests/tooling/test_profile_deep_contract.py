"""Tests for deep-profile production and diagnostic separation."""

from __future__ import annotations

import dataclasses
import json
import sys
import tomllib
import types
import typing

import pytest

from tooling.benchmark import native_lifecycle
from tooling.cli import profile_regenie2_deep
from tooling.profile_deep import commands as profile_deep_commands
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path


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


@pytest.mark.parametrize("profiler_status", ["success", "partial"])
@pytest.mark.parametrize("output_row_count", [None, 0, 4])
def test_external_profiler_success_requires_completed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_row_count: int | None,
    profiler_status: str,
) -> None:
    """A stopped sampler can return zero while its application is incomplete."""
    application_metadata = profile_regenie2_deep.GTrialApplicationMetadata(
        wall_time_seconds=None,
        output_row_count=output_row_count,
        output_path=None,
        application_output_run_directory=str(tmp_path / "application.g"),
        profile_summary_path=None,
        device_diagnostics=None,
        child_reported_cache_directory=None,
    )
    monkeypatch.setattr(
        profile_regenie2_deep,
        "collect_g_trial_application_metadata",
        lambda **arguments: application_metadata,
    )
    run_paths = profile_deep_models.DeepProfilerRunPaths(
        application_output_prefix=tmp_path / "application",
        application_output_run_directory=tmp_path / "application.g",
        stage_timing_path=None,
        profile_script_path=tmp_path / "child.py",
    )
    retained_trace = tmp_path / "partial-trace.json"
    result = profile_regenie2_deep.attach_deep_profiler_metadata(
        result=dataclasses.replace(
            trial_result("sampler"),
            status=profiler_status,
            output_row_count=None,
            notes="Original profiler diagnostic.",
        ),
        run_paths=run_paths,
        profiler_artifact_path=retained_trace,
    )

    assert result.status == ("failed" if output_row_count is None else profiler_status)
    assert result.notes is not None and "Original profiler diagnostic." in result.notes
    assert result.profiler_artifact_path == str(retained_trace)
    if output_row_count is None:
        assert result.notes is not None and "partial run" in result.notes


@pytest.mark.parametrize("capture_trace", [False, True])
def test_trace_warmup_is_separate_from_captured_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    capture_trace: bool,
) -> None:
    """Only trace diagnostics warm up, using distinct output and explicit scope."""
    events: list[str] = []
    output_prefixes: list[str] = []

    def run_application(arguments: list[str]) -> int:
        events.append("application")
        output_prefixes.append(arguments[arguments.index("--out") + 1])
        return 0

    def start_trace(directory: str, *, profiler_options: typing.Any) -> None:
        assert profiler_options.python_tracer_level == 0
        events.append("start_trace")

    fake_package: typing.Any = types.ModuleType("g")
    fake_cli: typing.Any = types.ModuleType("g.cli")
    fake_cli.run = run_application
    fake_package.cli = fake_cli
    fake_jax: typing.Any = types.ModuleType("jax")
    fake_jax.__version__ = "test"
    fake_jax.profiler = types.SimpleNamespace(
        ProfileOptions=lambda: types.SimpleNamespace(python_tracer_level=1),
        start_trace=start_trace,
        stop_trace=lambda: events.append("stop_trace"),
    )
    monkeypatch.setitem(sys.modules, "g", fake_package)
    monkeypatch.setitem(sys.modules, "g.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setattr(native_lifecycle, "discover_completed_run_directory", lambda **arguments: tmp_path / "run")
    monkeypatch.setattr(
        native_lifecycle,
        "measure_completed_output_run",
        lambda directory: types.SimpleNamespace(parquet_paths=[str(tmp_path / "part.parquet")]),
    )
    paths = types.SimpleNamespace(
        binary_phenotype_path=tmp_path / "phenotype.txt",
        regenie_prediction_list_path=tmp_path / "predictions.list",
        bgen_path=tmp_path / "genotypes.bgen",
        sample_path=tmp_path / "genotypes.sample",
        covariate_path=tmp_path / "covariates.txt",
    )
    command = profile_deep_commands.build_g_step2_child_command(
        baseline_paths=paths,
        candidate=candidate(),
        output_prefix=tmp_path / "profile",
        trace_directory=tmp_path / "trace" if capture_trace else None,
    )

    exec(command[2], {})

    if capture_trace:
        assert events == ["application", "start_trace", "application", "stop_trace"]
        assert output_prefixes == [str(tmp_path / "profile.trace_warmup"), str(tmp_path / "profile")]
        scope = json.loads((tmp_path / "trace" / "capture_scope.json").read_text(encoding="utf-8"))
        assert scope["includes_cold_initialization"] is False
        assert scope["warmup_output_root"] == str(tmp_path / "profile.trace_warmup.g")
    else:
        assert events == ["application"]
        assert output_prefixes == [str(tmp_path / "profile")]
