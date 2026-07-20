"""Tests for the current REGENIE matrix harness."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from tooling.benchmark import native_lifecycle
from tooling.cli import run_regenie2_matrix


@pytest.mark.parametrize("config_name", ["matrix_chr10_dry", "matrix_chr22_dry"])
def test_canonical_dry_configs_build_current_commands(tmp_path: Path, config_name: str) -> None:
    """Both chromosome configs render six current-schema CPU/GPU/cache cases."""
    arguments = run_regenie2_matrix.build_arguments_from_overrides(
        [f"tool.output_dir={tmp_path / config_name}"],
        config_name=config_name,
    )

    run_specs = run_regenie2_matrix.build_run_specs(arguments)

    assert len(run_specs) == 6
    assert len({run_spec.name for run_spec in run_specs}) == 6
    assert {run_spec.cache_state for run_spec in run_specs} == {
        run_regenie2_matrix.CacheState.DISABLED,
        run_regenie2_matrix.CacheState.COLD,
        run_regenie2_matrix.CacheState.WARM,
    }
    for run_spec in run_specs:
        assert run_spec.command_arguments.count("--config") == 1
        assert run_spec.command_arguments[-2] == "--config"
        parsed = tomllib.loads(Path(run_spec.command_arguments[-1]).read_text(encoding="utf-8"))
        assert "variant_limit" not in parsed
        assert parsed["output"]["resume"] is False
        if run_spec.trait == run_regenie2_matrix.TraitKind.BINARY:
            assert parsed["binary"]["fallback_method"] == "firth_approximate"


@pytest.mark.parametrize(
    ("telemetry_mode", "expected_artifacts"),
    [
        ("off", (False, False)),
        ("progress", (False, True)),
        ("profile", (True, True)),
    ],
)
def test_matrix_telemetry_paths_match_mode(
    tmp_path: Path,
    telemetry_mode: str,
    expected_artifacts: tuple[bool, bool],
) -> None:
    """Headline-off and explicit diagnostic modes advertise only real artifacts."""
    arguments = run_regenie2_matrix.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / telemetry_mode}",
            f"tool.telemetry_mode={telemetry_mode}",
        ],
        config_name="matrix_chr10_dry",
    )

    has_profile, has_events = expected_artifacts
    for run_spec in run_regenie2_matrix.build_run_specs(arguments):
        assert (run_spec.profile_summary_path is not None) is has_profile
        assert (run_spec.event_log_path is not None) is has_events


def test_matrix_omits_cached_cases_when_gpu_cache_is_disabled(tmp_path: Path) -> None:
    """A disabled GPU cache cannot produce a misleading warm-cache case."""
    arguments = run_regenie2_matrix.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'matrix'}",
            "tool.gpu_jax_persistent_cache=false",
        ],
        config_name="matrix_chr10_dry",
    )

    run_specs = run_regenie2_matrix.build_run_specs(arguments)

    assert len(run_specs) == 4
    assert all(run_spec.mode != run_regenie2_matrix.ExecutionMode.GPU_CACHED for run_spec in run_specs)


def test_manifest_compatibility_ignores_implementation_provenance() -> None:
    """A candidate implementation remains comparable under an identical workload scope."""
    compatibility_scope = {"sha256": "scope", "configuration": {"chunk_size": 16_384}}
    payload = {
        "dry_run": False,
        "compatibility_scope": compatibility_scope,
        "implementation_provenance": {"git_commit": "candidate"},
    }

    assert run_regenie2_matrix.manifest_matches_compatibility_scope(compatibility_scope, payload)
    assert not run_regenie2_matrix.manifest_matches_compatibility_scope({"sha256": "different"}, payload)
    assert not run_regenie2_matrix.manifest_matches_compatibility_scope(
        compatibility_scope, {**payload, "dry_run": True}
    )


def test_numeric_metrics_request_only_current_native_stages() -> None:
    """Matrix reports cannot silently request removed callback-era stage names."""
    current_stages = {
        stage_name: float(index + 1) for index, stage_name in enumerate(native_lifecycle.NATIVE_PROFILE_STAGE_NAMES)
    }
    result = run_regenie2_matrix.RunResult(
        name="binary_gpu",
        trait=run_regenie2_matrix.TraitKind.BINARY,
        mode=run_regenie2_matrix.ExecutionMode.GPU,
        status=run_regenie2_matrix.RunStatus.SUCCESS,
        return_code=0,
        wall_time_seconds=1.0,
        command_arguments=[],
        output_prefix="output",
        output_run_directory="run",
        profile_summary_path="profile.json",
        event_log_path="events.jsonl",
        cache_enabled=True,
        cache_state=run_regenie2_matrix.CacheState.COLD,
        cache_before=native_lifecycle.CacheSnapshot(file_count=0, total_size_bytes=0, sha256="before"),
        cache_after=native_lifecycle.CacheSnapshot(file_count=1, total_size_bytes=1, sha256="after"),
        output_row_count=4,
        committed_chunk_count=1,
        output_file_count=1,
        output_total_bytes=1,
        stage_seconds=current_stages,
    )

    metrics = run_regenie2_matrix.numeric_result_metrics(result)

    assert {
        metric_name.removeprefix("stage.")
        for metric_name, value in metrics.items()
        if metric_name.startswith("stage.") and value is not None
    } == set(native_lifecycle.NATIVE_PROFILE_STAGE_NAMES)


def test_warm_matrix_run_requires_byte_identical_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A labeled warm run cannot compile or write new persistent-cache entries."""
    arguments = run_regenie2_matrix.build_arguments_from_overrides(
        [f"tool.output_dir={tmp_path / 'matrix'}"],
        config_name="matrix_chr10_dry",
    )
    arguments = dataclasses.replace(arguments, dry_run=False)
    warm_spec = next(
        run_spec
        for run_spec in run_regenie2_matrix.build_run_specs(arguments)
        if run_spec.cache_state == run_regenie2_matrix.CacheState.WARM
    )
    before = native_lifecycle.CacheSnapshot(file_count=1, total_size_bytes=1, sha256="before")
    after = native_lifecycle.CacheSnapshot(file_count=2, total_size_bytes=2, sha256="after")
    snapshots = iter((before, after))
    monkeypatch.setattr(native_lifecycle, "snapshot_tree", lambda _: next(snapshots))
    monkeypatch.setattr(run_regenie2_matrix, "run_streaming_command", lambda _: 0)
    monkeypatch.setattr(
        run_regenie2_matrix,
        "measure_run_outputs",
        lambda *_: {
            "output_run_directory": "run",
            "output_row_count": 4,
            "committed_chunk_count": 1,
            "output_file_count": 1,
            "output_total_bytes": 1,
            "stage_seconds": {},
        },
    )

    with pytest.raises(RuntimeError, match="changed its cache tree"):
        run_regenie2_matrix.run_one_spec(arguments, warm_spec)


def test_invalid_matrix_telemetry_mode_is_rejected(tmp_path: Path) -> None:
    """Closed configuration choices fail during adaptation."""
    with pytest.raises(ValueError, match="not a valid RegenieTelemetry"):
        run_regenie2_matrix.build_arguments_from_overrides(
            [f"tool.output_dir={tmp_path / 'matrix'}", "tool.telemetry_mode=verbose"],
            config_name="matrix_chr10_dry",
        )
