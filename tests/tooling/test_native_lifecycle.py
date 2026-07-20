"""Tests for shared native lifecycle evidence validation."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import types
import typing

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tooling.benchmark import native_lifecycle
from tooling.common import g_regenie as tooling_g_regenie

if typing.TYPE_CHECKING:
    from pathlib import Path


def default_chunks() -> list[dict[str, object]]:
    """Return two contiguous chunks intentionally batched into one part."""
    return [
        {
            "chunk_identifier": 0,
            "variant_start_index": 0,
            "variant_stop_index": 2,
            "row_count": 2,
            "chunk_file_name": "part_000000000_000000002.parquet",
        },
        {
            "chunk_identifier": 2,
            "variant_start_index": 2,
            "variant_stop_index": 4,
            "row_count": 2,
            "chunk_file_name": "part_000000000_000000002.parquet",
        },
    ]


def write_output_run(
    root: Path,
    *,
    committed_chunks: list[dict[str, object]] | None = None,
    variant_count: object = 4,
    parquet_row_count: int = 4,
) -> Path:
    """Write one manifest-backed direct-Parquet output fixture."""
    run_directory = root / "trait_0001_trait.regenie2_binary.run"
    parts_directory = run_directory / "parts"
    parts_directory.mkdir(parents=True)
    schema = pa.schema([pa.field("variant_index", pa.int64())], metadata={b"contract": b"0"})
    table = pa.Table.from_arrays([pa.array(range(parquet_row_count), type=pa.int64())], schema=schema)
    pq.write_table(table, parts_directory / "part_000000000_000000002.parquet")
    manifest = {
        "status": "completed",
        "execution_plan": {"variant_count": variant_count},
        "committed_chunks": default_chunks() if committed_chunks is None else committed_chunks,
    }
    (run_directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_directory


def native_profile_totals() -> dict[str, float]:
    """Return every current native runtime stage."""
    return {
        stage_name: float(index + 1) for index, stage_name in enumerate(native_lifecycle.NATIVE_PROFILE_STAGE_NAMES)
    }


def test_prediction_dependencies_resolve_relative_paths(tmp_path: Path) -> None:
    """Prediction lists resolve every uniquely named LOCO dependency."""
    prediction_list = tmp_path / "lists" / "predictions.list"
    prediction_list.parent.mkdir()
    prediction_list.write_text("trait_b ../b.loco\ntrait_a a.loco\n", encoding="utf-8")

    dependencies = native_lifecycle.prediction_dependency_paths(prediction_list)

    assert dependencies == {
        "loco:trait_b": (tmp_path / "b.loco").resolve(),
        "loco:trait_a": (prediction_list.parent / "a.loco").resolve(),
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "empty"),
        ("trait only-one-field extra-field\n", "must contain"),
        ("trait first.loco\ntrait second.loco\n", "duplicate phenotype"),
    ],
)
def test_prediction_dependencies_reject_invalid_lists(tmp_path: Path, contents: str, message: str) -> None:
    """Malformed, empty, and duplicate prediction rows fail closed."""
    prediction_list = tmp_path / "predictions.list"
    prediction_list.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        native_lifecycle.prediction_dependency_paths(prediction_list)


def test_completed_output_accepts_batched_chunk_file_names(tmp_path: Path) -> None:
    """Multiple contiguous chunk commits may intentionally share one Parquet part."""
    run_directory = write_output_run(tmp_path)

    evidence = native_lifecycle.measure_completed_output_run(run_directory)

    assert evidence.row_count == 4
    assert evidence.committed_chunk_count == 2
    assert evidence.parquet_file_count == 1
    assert evidence.schema_metadata["contract"] == "0"
    assert evidence.manifest["status"] == "completed"
    assert evidence.manifest_sha256 == native_lifecycle.sha256_file(run_directory / "run_manifest.json")


@pytest.mark.parametrize(
    "committed_chunks",
    [
        [
            *default_chunks()[:1],
            {
                **default_chunks()[1],
                "chunk_identifier": 0,
            },
        ],
        [
            {**default_chunks()[0], "variant_stop_index": 1, "row_count": 1},
            default_chunks()[1],
        ],
        [
            {**default_chunks()[0], "variant_stop_index": 3, "row_count": 3},
            default_chunks()[1],
        ],
        [
            default_chunks()[0],
            {**default_chunks()[1], "row_count": 1},
        ],
        [
            default_chunks()[0],
            {**default_chunks()[1], "chunk_identifier": 3},
        ],
    ],
)
def test_completed_output_rejects_corrupt_chunk_coverage(
    tmp_path: Path, committed_chunks: list[dict[str, object]]
) -> None:
    """Duplicate, gap, overlap, row-count, and identifier corruption fail closed."""
    run_directory = write_output_run(tmp_path, committed_chunks=committed_chunks)

    with pytest.raises(RuntimeError):
        native_lifecycle.measure_completed_output_run(run_directory)


@pytest.mark.parametrize("variant_count", [True, "4", 4.0])
def test_completed_output_requires_integer_variant_count(tmp_path: Path, variant_count: object) -> None:
    """Corrupted manifest scalar types are not coerced into valid counts."""
    run_directory = write_output_run(tmp_path, variant_count=variant_count)

    with pytest.raises(RuntimeError, match="variant count"):
        native_lifecycle.measure_completed_output_run(run_directory)


def test_completed_output_rejects_part_path_traversal(tmp_path: Path) -> None:
    """Manifest part names cannot escape the parts directory."""
    chunks = [{**chunk, "chunk_file_name": "../part.parquet"} for chunk in default_chunks()]
    run_directory = write_output_run(tmp_path, committed_chunks=chunks)

    with pytest.raises(RuntimeError, match="invalid Parquet part name"):
        native_lifecycle.measure_completed_output_run(run_directory)


def test_completed_output_rejects_missing_and_unexpected_parts(tmp_path: Path) -> None:
    """The observed Parquet part set must exactly equal the manifest set."""
    missing_run = write_output_run(tmp_path / "missing")
    next((missing_run / "parts").glob("*.parquet")).unlink()
    with pytest.raises(RuntimeError, match="differ from its manifest"):
        native_lifecycle.measure_completed_output_run(missing_run)

    unexpected_run = write_output_run(tmp_path / "unexpected")
    pq.write_table(pa.table({"variant_index": [1]}), unexpected_run / "parts" / "unexpected.parquet")
    with pytest.raises(RuntimeError, match="differ from its manifest"):
        native_lifecycle.measure_completed_output_run(unexpected_run)


def test_completed_output_rejects_corrupt_or_wrong_row_count_parquet(tmp_path: Path) -> None:
    """Unreadable parts and Parquet row-count mismatches fail validation."""
    corrupt_run = write_output_run(tmp_path / "corrupt")
    next((corrupt_run / "parts").glob("*.parquet")).write_bytes(b"not parquet")
    with pytest.raises(pa.ArrowInvalid, match=r"Parquet|magic bytes"):
        native_lifecycle.measure_completed_output_run(corrupt_run)

    wrong_rows_run = write_output_run(tmp_path / "wrong-rows", parquet_row_count=3)
    with pytest.raises(RuntimeError, match="Parquet rows differ"):
        native_lifecycle.measure_completed_output_run(wrong_rows_run)


def test_completed_run_discovery_requires_exactly_one_manifest(tmp_path: Path) -> None:
    """Output discovery rejects zero or ambiguous manifest-backed runs."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    with pytest.raises(RuntimeError, match="found 0"):
        native_lifecycle.discover_completed_run_directory(
            expected_run_directory=None,
            output_root=output_root,
            glob_pattern="*.run",
            run_label="test",
        )
    first = output_root / "first.run"
    first.mkdir()
    (first / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert (
        native_lifecycle.discover_completed_run_directory(
            expected_run_directory=None,
            output_root=output_root,
            glob_pattern="*.run",
            run_label="test",
        )
        == first
    )
    second = output_root / "second.run"
    second.mkdir()
    (second / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="found 2"):
        native_lifecycle.discover_completed_run_directory(
            expected_run_directory=None,
            output_root=output_root,
            glob_pattern="*.run",
            run_label="test",
        )


def test_diagnostic_evidence_enforces_telemetry_contracts(tmp_path: Path) -> None:
    """Off, progress, and profile modes require exactly their promised artifacts."""
    telemetry_root = tmp_path / "telemetry"
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    off = native_lifecycle.collect_diagnostic_evidence(
        telemetry=tooling_g_regenie.RegenieTelemetry.OFF,
        telemetry_root=telemetry_root,
        run_directories=(run_directory,),
    )
    assert off.events_path is None

    with pytest.raises(RuntimeError, match="no events"):
        native_lifecycle.collect_diagnostic_evidence(
            telemetry=tooling_g_regenie.RegenieTelemetry.PROGRESS,
            telemetry_root=telemetry_root,
            run_directories=(run_directory,),
        )
    logs_directory = telemetry_root / "logs"
    logs_directory.mkdir(parents=True)
    (logs_directory / "events.jsonl").write_text('{"event":"complete"}\n', encoding="utf-8")
    progress = native_lifecycle.collect_diagnostic_evidence(
        telemetry=tooling_g_regenie.RegenieTelemetry.PROGRESS,
        telemetry_root=telemetry_root,
        run_directories=(run_directory,),
    )
    assert progress.events_sha256 is not None
    assert progress.profile_summary_path is None

    (logs_directory / "profile.summary.json").write_text(
        json.dumps({"stage_totals_seconds": {"runner_total": 1.0}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="missing current stages"):
        native_lifecycle.collect_diagnostic_evidence(
            telemetry=tooling_g_regenie.RegenieTelemetry.PROFILE,
            telemetry_root=telemetry_root,
            run_directories=(run_directory,),
        )
    (logs_directory / "profile.summary.json").write_text(
        json.dumps({"stage_totals_seconds": native_profile_totals()}), encoding="utf-8"
    )
    (run_directory / "output_stage_timings.json").write_text(
        json.dumps({"stage_totals_seconds": {"rust_output_writer_total": 0.5}}), encoding="utf-8"
    )
    profile = native_lifecycle.collect_diagnostic_evidence(
        telemetry=tooling_g_regenie.RegenieTelemetry.PROFILE,
        telemetry_root=telemetry_root,
        run_directories=(run_directory,),
    )
    assert profile.profile_stage_totals_seconds == native_profile_totals()
    assert profile.output_stage_timings[0].stage_totals_seconds == {"rust_output_writer_total": 0.5}


def test_cache_snapshots_and_state_transitions(tmp_path: Path) -> None:
    """Cache evidence distinguishes empty, populated, unchanged, and changed trees."""
    empty = native_lifecycle.snapshot_tree(tmp_path)
    cache_file = tmp_path / "entry"
    cache_file.write_text("first", encoding="utf-8")
    populated = native_lifecycle.snapshot_tree(tmp_path)
    unchanged = native_lifecycle.snapshot_tree(tmp_path)
    cache_file.write_text("second", encoding="utf-8")
    changed = native_lifecycle.snapshot_tree(tmp_path)

    assert native_lifecycle.cache_state(empty, populated) == "cache_populated"
    assert native_lifecycle.cache_state(populated, unchanged) == "populated_tree_unchanged"
    assert native_lifecycle.cache_state(unchanged, changed) == "cache_tree_changed"


def test_fresh_process_parses_child_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh-process execution validates the interpreter and parses native results."""
    observed_commands: list[list[str]] = []

    def fake_run(command: list[str], **_: typing.Any) -> subprocess.CompletedProcess[str]:
        observed_commands.append(command)
        payload = {
            "elapsed_seconds": 1.25,
            "exit_code": 0,
            "stdout_chunks": ["out"],
            "stderr_chunks": [],
        }
        return subprocess.CompletedProcess(command, 0, stdout=f"diagnostic noise\n{json.dumps(payload)}\n", stderr="")

    monkeypatch.setattr(native_lifecycle.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(native_lifecycle.subprocess, "run", fake_run)

    result = native_lifecycle.run_fresh_process(sys.executable, tmp_path / "run.toml")

    assert result.elapsed_seconds == 1.25
    assert result.stdout_chunks == ("out",)
    assert observed_commands[0][0] == sys.executable


def test_same_process_restores_info_after_native_logging_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discarded native warm cannot leave headline runs at root NOTSET/DEBUG."""
    observed_disable_levels: list[int] = []

    def fake_native_run(_: list[str]) -> typing.Any:
        observed_disable_levels.append(logging.root.manager.disable)
        logging.getLogger().setLevel(logging.NOTSET)
        return types.SimpleNamespace(exit_code=0, stdout_chunks=[], stderr_chunks=[])

    previous_root_level = logging.getLogger().level
    previous_jax_level = logging.getLogger("jax").level
    previous_disable_level = logging.root.manager.disable
    g_module = types.ModuleType("g")
    setattr(g_module, "__path__", [])
    native_module = types.ModuleType("g._core")
    setattr(native_module, "cli", types.SimpleNamespace(run=fake_native_run))
    setattr(g_module, "_core", native_module)
    monkeypatch.setitem(sys.modules, "g", g_module)
    monkeypatch.setitem(sys.modules, "g._core", native_module)
    try:
        native_lifecycle.run_same_process(tmp_path / "run.toml")

        assert observed_disable_levels == [logging.DEBUG]
        assert logging.getLogger().level == logging.INFO
        assert logging.getLogger("jax").level == logging.INFO
        assert logging.root.manager.disable == previous_disable_level
    finally:
        logging.getLogger().setLevel(previous_root_level)
        logging.getLogger("jax").setLevel(previous_jax_level)
        logging.disable(previous_disable_level)


def test_fresh_child_source_guards_logging_before_native_import() -> None:
    """The fresh child suppresses debug logs before loading the native module."""
    guard_index = native_lifecycle.CHILD_RUN_SOURCE.index("logging.disable(logging.DEBUG)")
    native_import_index = native_lifecycle.CHILD_RUN_SOURCE.index("import g._core")

    assert guard_index < native_import_index


def test_fresh_process_rejects_unattributed_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A different Python environment cannot inherit the parent's evidence envelope."""
    monkeypatch.setattr(native_lifecycle.shutil, "which", lambda _: "/different/python")

    with pytest.raises(ValueError, match="current Python environment"):
        native_lifecycle.run_fresh_process("python", tmp_path / "run.toml")
