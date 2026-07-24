"""Tests for shared native lifecycle evidence validation."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import types
import typing
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tooling.benchmark import native_lifecycle
from tooling.common import g_regenie as tooling_g_regenie

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class OutputTransactionFixture:
    """Paths and CLI evidence for one completed schema-v0 transaction."""

    output_root: Path
    stdout_chunks: tuple[str, ...]
    run_directories: tuple[Path, ...]
    part_paths: tuple[Path, ...]
    receipt_paths: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]
    outcome_path: Path
    finalization_path: Path
    owner_claim_path: Path
    owner_transition_path: Path
    owner_claim_id: str
    released_state_id: str
    attempt_id: str
    ancestor_attempt_id: str | None


def default_chunk_geometries() -> list[dict[str, int]]:
    """Return the canonical geometry used by the transaction fixture."""
    return [
        {
            "chunk_identifier": 0,
            "variant_start_index": 0,
            "variant_stop_index": 2,
            "row_count": 2,
        },
        {
            "chunk_identifier": 2,
            "variant_start_index": 2,
            "variant_stop_index": 4,
            "row_count": 2,
        },
    ]


def default_chunks(part_file_name: str) -> list[dict[str, object]]:
    """Return two contiguous chunks intentionally batched into one part."""
    return [
        {
            **chunk_geometry,
            "chunk_file_name": part_file_name,
        }
        for chunk_geometry in default_chunk_geometries()
    ]


def write_json_mapping(path: Path, payload: dict[str, typing.Any]) -> None:
    """Write deterministic JSON bytes used by raw-hash bindings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def read_json_mapping(path: Path) -> dict[str, typing.Any]:
    """Read one test fixture JSON object."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return typing.cast("dict[str, typing.Any]", payload)


def immutable_files_aggregate_sha256(paths: tuple[Path, ...]) -> str:
    """Independently hash length-prefixed paths and current raw bytes."""
    digest = hashlib.sha256()
    for path in paths:
        encoded_path = str(path).encode()
        digest.update(len(encoded_path).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded_path)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def canonical_output_table(row_count: int, footer: dict[str, typing.Any]) -> pa.Table:
    """Return rows with the exact native schema and embedded part binding."""
    column_values: dict[str, list[object]] = {
        "CHROM": ["1"] * row_count,
        "GENPOS": list(range(1, row_count + 1)),
        "ID": [f"variant-{index}" for index in range(row_count)],
        "ALLELE0": ["A"] * row_count,
        "ALLELE1": ["G"] * row_count,
        "A1FREQ": [0.25] * row_count,
        "INFO": [0.95] * row_count,
        "N": [100] * row_count,
        "BETA": [0.1] * row_count,
        "SE": [0.05] * row_count,
        "CHISQ": [4.0] * row_count,
        "LOG10P": [1.3] * row_count,
        "CORRECTION_METHOD": ["none"] * row_count,
        "CORRECTION_STATUS": ["not_required"] * row_count,
    }
    metadata = {
        b"contract": b"0",
        native_lifecycle.PART_BINDING_METADATA_KEY: json.dumps(
            footer,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    }
    schema = native_lifecycle.CANONICAL_OUTPUT_SCHEMA.with_metadata(metadata)
    return pa.Table.from_pydict(column_values, schema=schema)


def write_terminal_claim(
    control_directory: Path,
    *,
    run_set_id: str,
    attempt_id: str,
    status: str,
    phenotypes: list[dict[str, str]],
) -> None:
    """Write a terminal outcome and its raw-byte finalization."""
    terminal = {
        "record_kind": "terminal",
        "schema_version": 0,
        "run_set_id": run_set_id,
        "attempt_id": attempt_id,
        "status": status,
        "interrupted_signal": "SIGTERM" if status == "interrupted" else None,
        "failure_reason": None,
        "phenotypes": phenotypes,
    }
    outcome_path = control_directory / "outcomes" / f"{attempt_id}.json"
    write_json_mapping(outcome_path, {"outcome_kind": "terminal_claim", "record": terminal})
    finalization = {
        "record_kind": "terminal_finalization",
        "schema_version": 0,
        "run_set_id": run_set_id,
        "attempt_id": attempt_id,
        "terminal_claim_sha256": native_lifecycle.sha256_file(outcome_path),
    }
    write_json_mapping(control_directory / "terminal-finalizations" / f"{attempt_id}.json", finalization)


def write_output_transaction(
    output_root: Path,
    *,
    phenotype_names: tuple[str, ...],
    with_interrupted_ancestor: bool,
) -> OutputTransactionFixture:
    """Write a strict append-only schema-v0 output transaction."""
    output_root.mkdir(parents=True)
    control_directory = output_root / ".g-output"
    run_set_id = "run-set-0001"
    owner_claim_id = "owner-claim-0001"
    released_state_id = "owner-released-0001"
    attempt_id = "attempt-current"
    ancestor_attempt_id = "attempt-previous" if with_interrupted_ancestor else None
    initial_attempt_id = ancestor_attempt_id or attempt_id
    producer_attempt_id = initial_attempt_id
    part_id = "part-000000000-000000002"
    part_file_name = f"{part_id}.parquet"
    receipt_file_name = f"{part_id}.json"
    chunks = default_chunks(part_file_name)
    chunk_plan = {"algorithm": "sha256", "chunks": default_chunk_geometries()}
    chunk_plan_sha256 = native_lifecycle.sha256_json_value(chunk_plan)

    owner_claim_path = control_directory / "session.claim.json"
    owner_claim = {
        "schema_version": 0,
        "claim_id": owner_claim_id,
        "host_name": "test-host",
        "process_id": 123,
    }
    write_json_mapping(owner_claim_path, owner_claim)
    owner_transition_path = control_directory / "owner-transitions" / f"{owner_claim_id}.json"
    owner_transition = {
        "transition_kind": "graceful_release",
        "schema_version": 0,
        "predecessor_claim_id": owner_claim_id,
        "released_state_id": released_state_id,
    }
    write_json_mapping(owner_transition_path, owner_transition)

    execution_plans = [{"phenotype_name": phenotype_name, "variant_count": 4} for phenotype_name in phenotype_names]
    execution_plan_hashes = [native_lifecycle.sha256_json_value(execution_plan) for execution_plan in execution_plans]
    output_directory_names = [
        f"trait_{index:04d}_{phenotype_name}" for index, phenotype_name in enumerate(phenotype_names, start=1)
    ]
    genesis_phenotypes = [
        {
            "phenotype_name": phenotype_name,
            "output_directory_name": output_directory_name,
            "execution_plan_sha256": execution_plan_sha256,
        }
        for phenotype_name, output_directory_name, execution_plan_sha256 in zip(
            phenotype_names,
            output_directory_names,
            execution_plan_hashes,
            strict=True,
        )
    ]
    genesis = {
        "record_kind": "genesis",
        "schema_version": 0,
        "run_set_id": run_set_id,
        "attempt_id": initial_attempt_id,
        "chunk_plan_sha256": chunk_plan_sha256,
        "phenotypes": genesis_phenotypes,
    }
    write_json_mapping(control_directory / "genesis.json", genesis)

    run_directories: list[Path] = []
    part_paths: list[Path] = []
    receipt_paths: list[Path] = []
    manifest_paths: list[Path] = []
    terminal_phenotypes: list[dict[str, str]] = []
    stdout_chunks: list[str] = []
    for phenotype_name, output_directory_name, execution_plan, execution_plan_sha256 in zip(
        phenotype_names,
        output_directory_names,
        execution_plans,
        execution_plan_hashes,
        strict=True,
    ):
        run_directory = output_root / "attempts" / attempt_id / output_directory_name
        parts_directory = run_directory / "parts"
        commits_directory = run_directory / "commits"
        parts_directory.mkdir(parents=True)
        commits_directory.mkdir()
        footer = {
            "schema_version": 0,
            "run_set_id": run_set_id,
            "attempt_id": producer_attempt_id,
            "phenotype_name": phenotype_name,
            "execution_plan_sha256": execution_plan_sha256,
            "chunk_plan_sha256": chunk_plan_sha256,
            "part_id": part_id,
            "part_file_name": part_file_name,
            "receipt_id": part_id,
            "receipt_file_name": receipt_file_name,
            "chunks": chunks,
        }
        part_path = parts_directory / part_file_name
        pq.write_table(canonical_output_table(4, footer), part_path)
        receipt = {
            "footer": footer,
            "part_size_bytes": part_path.stat().st_size,
            "part_sha256": native_lifecycle.sha256_file(part_path),
        }
        receipt_path = commits_directory / receipt_file_name
        write_json_mapping(receipt_path, receipt)
        manifest = {
            "schema_version": 0,
            "output_schema_version": 0,
            "execution_plan": execution_plan,
            "execution_plan_hash": execution_plan_sha256,
            "attempt_manifest_schema_version": 0,
            "run_set_id": run_set_id,
            "attempt_id": attempt_id,
            "phenotype_name": phenotype_name,
            "output_directory_name": output_directory_name,
            "chunk_plan_hash": chunk_plan_sha256,
            "status": "completed",
            "committed_parts": [receipt],
            "committed_chunks": chunks,
            "command": {"name": "regenie"},
            "runtime": {"engine": "native"},
        }
        manifest_path = run_directory / "run_manifest.json"
        write_json_mapping(manifest_path, manifest)
        terminal_phenotypes.append(
            {
                "phenotype_name": phenotype_name,
                "output_directory_name": output_directory_name,
                "run_manifest_sha256": native_lifecycle.sha256_file(manifest_path),
            }
        )
        run_directories.append(run_directory)
        part_paths.append(part_path)
        receipt_paths.append(receipt_path)
        manifest_paths.append(manifest_path)
        stdout_chunks.append(f"{native_lifecycle.PARQUET_DIRECTORY_LINE_PREFIX}{parts_directory}\n")

    if ancestor_attempt_id is not None:
        (output_root / "attempts" / ancestor_attempt_id).mkdir(parents=True)
        ancestor_phenotypes = [
            {
                "phenotype_name": phenotype_name,
                "output_directory_name": output_directory_name,
                "run_manifest_sha256": "0" * 64,
            }
            for phenotype_name, output_directory_name in zip(
                phenotype_names,
                output_directory_names,
                strict=True,
            )
        ]
        write_terminal_claim(
            control_directory,
            run_set_id=run_set_id,
            attempt_id=ancestor_attempt_id,
            status="interrupted",
            phenotypes=ancestor_phenotypes,
        )
        ancestor_outcome_path = control_directory / "outcomes" / f"{ancestor_attempt_id}.json"
        successor = {
            "record_kind": "successor",
            "schema_version": 0,
            "run_set_id": run_set_id,
            "parent_attempt_id": ancestor_attempt_id,
            "attempt_id": attempt_id,
            "recovery_kind": "terminal_resume",
            "parent_terminal_sha256": native_lifecycle.sha256_file(ancestor_outcome_path),
        }
        write_json_mapping(control_directory / "successors" / f"{ancestor_attempt_id}.json", successor)

    write_terminal_claim(
        control_directory,
        run_set_id=run_set_id,
        attempt_id=attempt_id,
        status="completed",
        phenotypes=terminal_phenotypes,
    )
    return OutputTransactionFixture(
        output_root=output_root,
        stdout_chunks=tuple(stdout_chunks),
        run_directories=tuple(run_directories),
        part_paths=tuple(part_paths),
        receipt_paths=tuple(receipt_paths),
        manifest_paths=tuple(manifest_paths),
        outcome_path=control_directory / "outcomes" / f"{attempt_id}.json",
        finalization_path=control_directory / "terminal-finalizations" / f"{attempt_id}.json",
        owner_claim_path=owner_claim_path,
        owner_transition_path=owner_transition_path,
        owner_claim_id=owner_claim_id,
        released_state_id=released_state_id,
        attempt_id=attempt_id,
        ancestor_attempt_id=ancestor_attempt_id,
    )


def refresh_completed_terminal_hashes(fixture: OutputTransactionFixture) -> None:
    """Refresh terminal and finalization hashes after an intentional manifest rewrite."""
    outcome = read_json_mapping(fixture.outcome_path)
    terminal = typing.cast("dict[str, typing.Any]", outcome["record"])
    terminal_phenotypes = typing.cast("list[dict[str, typing.Any]]", terminal["phenotypes"])
    for terminal_phenotype, manifest_path in zip(
        terminal_phenotypes,
        fixture.manifest_paths,
        strict=True,
    ):
        terminal_phenotype["run_manifest_sha256"] = native_lifecycle.sha256_file(manifest_path)
    write_json_mapping(fixture.outcome_path, outcome)
    finalization = read_json_mapping(fixture.finalization_path)
    finalization["terminal_claim_sha256"] = native_lifecycle.sha256_file(fixture.outcome_path)
    write_json_mapping(fixture.finalization_path, finalization)


def rewrite_first_part_footer(
    fixture: OutputTransactionFixture,
    embedded_footer: dict[str, typing.Any],
) -> None:
    """Rewrite a part footer while keeping its receipt raw-byte binding current."""
    part_path = fixture.part_paths[0]
    table = pq.read_table(part_path)
    metadata = dict(table.schema.metadata or {})
    metadata[native_lifecycle.PART_BINDING_METADATA_KEY] = json.dumps(
        embedded_footer,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    pq.write_table(table.replace_schema_metadata(metadata), part_path)
    refresh_first_part_raw_binding(fixture)


def refresh_first_part_raw_binding(fixture: OutputTransactionFixture) -> None:
    """Refresh receipt, manifest, terminal, and finalization raw hashes."""
    part_path = fixture.part_paths[0]
    receipt = read_json_mapping(fixture.receipt_paths[0])
    receipt["part_size_bytes"] = part_path.stat().st_size
    receipt["part_sha256"] = native_lifecycle.sha256_file(part_path)
    write_json_mapping(fixture.receipt_paths[0], receipt)
    manifest = read_json_mapping(fixture.manifest_paths[0])
    manifest_receipts = typing.cast("list[dict[str, typing.Any]]", manifest["committed_parts"])
    manifest_receipts[0] = receipt
    write_json_mapping(fixture.manifest_paths[0], manifest)
    refresh_completed_terminal_hashes(fixture)


def append_owner_reacquisition_and_release(fixture: OutputTransactionFixture) -> None:
    """Append one valid acquire-after-release cycle with a new released head."""
    replacement_claim_id = "owner-claim-0002"
    replacement_released_state_id = "owner-released-0002"
    replacement_claim = {
        "schema_version": 0,
        "claim_id": replacement_claim_id,
        "host_name": "test-host",
        "process_id": 124,
    }
    transitions_directory = fixture.output_root / ".g-output" / "owner-transitions"
    write_json_mapping(
        transitions_directory / f"{fixture.released_state_id}.json",
        {
            "transition_kind": "acquire_after_release",
            "schema_version": 0,
            "predecessor_released_state_id": fixture.released_state_id,
            "claim": replacement_claim,
        },
    )
    write_json_mapping(
        transitions_directory / f"{replacement_claim_id}.json",
        {
            "transition_kind": "graceful_release",
            "schema_version": 0,
            "predecessor_claim_id": replacement_claim_id,
            "released_state_id": replacement_released_state_id,
        },
    )


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


def test_completed_output_accepts_gracefully_released_owner_authority(tmp_path: Path) -> None:
    """A persistent root claim is inactive after its bound graceful release."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )

    evidence_set = native_lifecycle.collect_completed_output_evidence(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=1,
        run_label="released-owner",
    )

    assert fixture.owner_claim_path.is_file()
    assert fixture.owner_transition_path.is_file()
    assert evidence_set.runs[0].manifest["status"] == "completed"
    expected_paths = (fixture.owner_claim_path, fixture.owner_transition_path)
    assert tuple(file_evidence.absolute_path for file_evidence in evidence_set.owner_authority.files) == tuple(
        str(path) for path in expected_paths
    )
    assert tuple(file_evidence.raw_sha256 for file_evidence in evidence_set.owner_authority.files) == tuple(
        native_lifecycle.sha256_file(path) for path in expected_paths
    )
    assert evidence_set.owner_authority.aggregate_sha256 == immutable_files_aggregate_sha256(expected_paths)
    assert evidence_set.owner_authority.released_state_id == fixture.released_state_id
    control_directory = fixture.output_root / ".g-output"
    expected_immutable_paths = (
        *expected_paths,
        control_directory / "genesis.json",
        fixture.outcome_path,
        fixture.finalization_path,
        fixture.manifest_paths[0],
        fixture.receipt_paths[0],
    )
    assert tuple(file_evidence.absolute_path for file_evidence in evidence_set.immutable_authority.files) == tuple(
        str(path) for path in expected_immutable_paths
    )
    assert tuple(file_evidence.raw_sha256 for file_evidence in evidence_set.immutable_authority.files) == tuple(
        native_lifecycle.sha256_file(path) for path in expected_immutable_paths
    )
    assert evidence_set.immutable_authority.aggregate_sha256 == immutable_files_aggregate_sha256(
        expected_immutable_paths
    )


def test_completed_output_accepts_reacquired_then_released_owner_authority(tmp_path: Path) -> None:
    """A completed no-op can reacquire a released root and release its new claim."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    append_owner_reacquisition_and_release(fixture)

    evidence_set = native_lifecycle.collect_completed_output_evidence(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=1,
        run_label="reacquired-owner",
    )

    assert evidence_set.runs[0].manifest["status"] == "completed"
    transitions_directory = fixture.output_root / ".g-output" / "owner-transitions"
    expected_paths = (
        fixture.owner_claim_path,
        fixture.owner_transition_path,
        transitions_directory / f"{fixture.released_state_id}.json",
        transitions_directory / "owner-claim-0002.json",
    )
    assert tuple(file_evidence.absolute_path for file_evidence in evidence_set.owner_authority.files) == tuple(
        str(path) for path in expected_paths
    )
    assert evidence_set.owner_authority.aggregate_sha256 == immutable_files_aggregate_sha256(expected_paths)
    assert evidence_set.owner_authority.released_state_id == "owner-released-0002"


def test_completed_output_rejects_owner_head_change_during_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence collection requires the same released owner head before and after reads."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    original_measure = native_lifecycle.measure_completed_output_artifact

    def measure_then_change_owner(
        artifact: native_lifecycle.CompletedRunArtifact,
        phenotype: native_lifecycle.LineagePhenotypeBinding,
        lineage: native_lifecycle.CompletedLineage,
    ) -> native_lifecycle.MeasuredCompletedOutput:
        measured_output = original_measure(artifact, phenotype, lineage)
        append_owner_reacquisition_and_release(fixture)
        return measured_output

    monkeypatch.setattr(native_lifecycle, "measure_completed_output_artifact", measure_then_change_owner)

    with pytest.raises(RuntimeError, match="authority changed"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="owner-race",
        )


def test_completed_output_rejects_raw_owner_evidence_tamper_during_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantically unchanged authority rewrite still invalidates the raw proof."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    original_measure = native_lifecycle.measure_completed_output_artifact

    def measure_then_rewrite_owner_transition(
        artifact: native_lifecycle.CompletedRunArtifact,
        phenotype: native_lifecycle.LineagePhenotypeBinding,
        lineage: native_lifecycle.CompletedLineage,
    ) -> native_lifecycle.MeasuredCompletedOutput:
        measured_output = original_measure(artifact, phenotype, lineage)
        transition_bytes = fixture.owner_transition_path.read_bytes()
        fixture.owner_transition_path.write_bytes(transition_bytes + b"\n")
        return measured_output

    monkeypatch.setattr(
        native_lifecycle,
        "measure_completed_output_artifact",
        measure_then_rewrite_owner_transition,
    )

    with pytest.raises(RuntimeError, match="authority changed"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="owner-raw-tamper",
        )


@pytest.mark.parametrize("target_role", ["receipt", "finalization"])
def test_completed_output_rejects_raw_immutable_history_tamper_during_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_role: str,
) -> None:
    """Semantic-equivalent receipt and finalization rewrites invalidate the proof."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    target_path = fixture.receipt_paths[0] if target_role == "receipt" else fixture.finalization_path
    original_verify = native_lifecycle.verify_completed_lineage
    verification_count = 0

    def verify_then_tamper(
        output_root: Path,
        artifacts: native_lifecycle.CompletedRunArtifacts,
        *,
        run_label: str,
    ) -> native_lifecycle.CompletedLineage:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            target_path.write_bytes(target_path.read_bytes() + b"\n")
        return original_verify(output_root, artifacts, run_label=run_label)

    monkeypatch.setattr(native_lifecycle, "verify_completed_lineage", verify_then_tamper)

    with pytest.raises(RuntimeError, match="Immutable output authority changed"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label=f"{target_role}-raw-tamper",
        )


def test_completed_output_rejects_active_owner_authority(tmp_path: Path) -> None:
    """A root claim without a release transition remains authoritative."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    fixture.owner_transition_path.unlink()

    with pytest.raises(RuntimeError, match=r"(?:active.*owner|owner.*active)"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="active-owner",
        )


def test_completed_output_fails_closed_on_unreadable_optional_owner_transition(tmp_path: Path) -> None:
    """A non-file transition path cannot be treated as an absent released head."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    next_transition_path = fixture.output_root / ".g-output" / "owner-transitions" / f"{fixture.released_state_id}.json"
    next_transition_path.mkdir()

    with pytest.raises(RuntimeError, match="optional output owner transition"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="owner-transition-io",
        )


def test_completed_output_rejects_malformed_owner_transition(tmp_path: Path) -> None:
    """Owner authority traversal requires the exact schema-v0 transition fields."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    transition = read_json_mapping(fixture.owner_transition_path)
    transition["legacy_field"] = True
    write_json_mapping(fixture.owner_transition_path, transition)

    with pytest.raises(RuntimeError, match=r"owner.*transition"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="malformed-owner-transition",
        )


def test_completed_output_rejects_stale_owner_transition_binding(tmp_path: Path) -> None:
    """A transition file must name the authority state used to reach it."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    transition = read_json_mapping(fixture.owner_transition_path)
    transition["predecessor_claim_id"] = "owner-stale-claim"
    write_json_mapping(fixture.owner_transition_path, transition)

    with pytest.raises(RuntimeError, match=r"owner.*transition"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="stale-owner-transition",
        )


def test_completed_output_accepts_one_exact_cli_artifact(tmp_path: Path) -> None:
    """One exact dataset line resolves and verifies the full schema-v0 transaction."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )

    artifacts = native_lifecycle.parse_completed_run_artifacts(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=1,
        run_label="single",
    )
    evidence_set = native_lifecycle.collect_completed_output_evidence(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=1,
        run_label="single",
    )

    assert artifacts.artifacts[0].run_directory == fixture.run_directories[0]
    assert artifacts.artifacts[0].parts_directory == fixture.part_paths[0].parent
    assert artifacts.artifacts[0].attempt_id == fixture.attempt_id
    assert evidence_set.runs[0].row_count == 4
    assert evidence_set.runs[0].committed_chunk_count == 2
    assert evidence_set.runs[0].parquet_file_count == 1
    assert evidence_set.runs[0].schema == str(native_lifecycle.CANONICAL_OUTPUT_SCHEMA)
    assert evidence_set.runs[0].schema_metadata["contract"] == "0"
    assert evidence_set.runs[0].manifest["status"] == "completed"
    assert evidence_set.runs[0].manifest_sha256 == native_lifecycle.sha256_file(fixture.manifest_paths[0])


def test_completed_output_preserves_exact_multi_phenotype_cli_order(tmp_path: Path) -> None:
    """Dataset lines and evidence remain in completed terminal phenotype order."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait_b", "trait_a"),
        with_interrupted_ancestor=False,
    )

    evidence_set = native_lifecycle.collect_completed_output_evidence(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=2,
        run_label="multiple",
    )

    assert tuple(evidence.run_directory for evidence in evidence_set.runs) == tuple(
        str(run_directory) for run_directory in fixture.run_directories
    )
    assert tuple(evidence.manifest["phenotype_name"] for evidence in evidence_set.runs) == (
        "trait_b",
        "trait_a",
    )
    control_directory = fixture.output_root / ".g-output"
    expected_immutable_paths = (
        fixture.owner_claim_path,
        fixture.owner_transition_path,
        control_directory / "genesis.json",
        fixture.outcome_path,
        fixture.finalization_path,
        *fixture.manifest_paths,
        *fixture.receipt_paths,
    )
    assert tuple(file_evidence.absolute_path for file_evidence in evidence_set.immutable_authority.files) == tuple(
        str(path) for path in expected_immutable_paths
    )
    with pytest.raises(RuntimeError, match="phenotype order"):
        native_lifecycle.collect_completed_output_evidence(
            tuple(reversed(fixture.stdout_chunks)),
            output_root=fixture.output_root,
            expected_phenotype_count=2,
            run_label="reversed",
        )


def test_completed_output_uses_reported_current_leaf_after_interrupted_attempt(tmp_path: Path) -> None:
    """A stale attempt directory cannot override the finalized CLI-reported successor."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=True,
    )
    assert fixture.ancestor_attempt_id is not None
    stale_run_directory = (
        fixture.output_root / "attempts" / fixture.ancestor_attempt_id / fixture.run_directories[0].name
    )
    (stale_run_directory / "parts").mkdir(parents=True)
    write_json_mapping(stale_run_directory / "run_manifest.json", {"status": "stale"})

    evidence_set = native_lifecycle.collect_completed_output_evidence(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=1,
        run_label="resumed",
    )

    assert evidence_set.runs[0].run_directory == str(fixture.run_directories[0])
    assert evidence_set.runs[0].manifest["attempt_id"] == fixture.attempt_id
    control_directory = fixture.output_root / ".g-output"
    expected_immutable_paths = (
        fixture.owner_claim_path,
        fixture.owner_transition_path,
        control_directory / "genesis.json",
        control_directory / "outcomes" / f"{fixture.ancestor_attempt_id}.json",
        control_directory / "successors" / f"{fixture.ancestor_attempt_id}.json",
        control_directory / "terminal-finalizations" / f"{fixture.ancestor_attempt_id}.json",
        fixture.outcome_path,
        fixture.finalization_path,
        fixture.manifest_paths[0],
        fixture.receipt_paths[0],
    )
    assert tuple(file_evidence.absolute_path for file_evidence in evidence_set.immutable_authority.files) == tuple(
        str(path) for path in expected_immutable_paths
    )


def test_completed_output_rejects_missing_terminal_ancestor_attempt_directory(tmp_path: Path) -> None:
    """Every materialized terminal-resume ancestor remains present in the attempt tree."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=True,
    )
    assert fixture.ancestor_attempt_id is not None
    ancestor_attempt_directory = fixture.output_root / "attempts" / fixture.ancestor_attempt_id
    ancestor_attempt_directory.rmdir()

    with pytest.raises(RuntimeError, match="missing attempt directory"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="missing-ancestor",
        )


def test_completed_output_rejects_symlinked_terminal_ancestor_attempt_directory(tmp_path: Path) -> None:
    """Visited materialized attempts must be real directories rather than symlinks."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=True,
    )
    assert fixture.ancestor_attempt_id is not None
    ancestor_attempt_directory = fixture.output_root / "attempts" / fixture.ancestor_attempt_id
    ancestor_attempt_directory.rmdir()
    ancestor_attempt_directory.symlink_to(
        fixture.output_root / "attempts" / fixture.attempt_id,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="missing attempt directory"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="symlinked-ancestor",
        )


def test_exact_recovery_parent_exempts_only_an_absent_attempt_path(tmp_path: Path) -> None:
    """An authorized unmaterialized parent may be absent but cannot be a file."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    exact_parent_id = "attempt-unmaterialized"
    genesis_path = fixture.output_root / ".g-output" / "genesis.json"
    genesis = read_json_mapping(genesis_path)
    genesis["attempt_id"] = exact_parent_id
    write_json_mapping(genesis_path, genesis)
    write_json_mapping(
        fixture.output_root / ".g-output" / "outcomes" / f"{exact_parent_id}.json",
        {
            "outcome_kind": "exact_recovery_claim",
            "record": {
                "record_kind": "successor",
                "schema_version": 0,
                "run_set_id": "run-set-0001",
                "parent_attempt_id": exact_parent_id,
                "attempt_id": fixture.attempt_id,
                "recovery_kind": "exact_nonterminal_recovery",
                "parent_terminal_sha256": None,
            },
        },
    )

    evidence_set = native_lifecycle.collect_completed_output_evidence(
        fixture.stdout_chunks,
        output_root=fixture.output_root,
        expected_phenotype_count=1,
        run_label="absent-exact-parent",
    )
    assert evidence_set.runs[0].manifest["status"] == "completed"

    exact_parent_path = fixture.output_root / "attempts" / exact_parent_id
    exact_parent_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a directory"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="file-exact-parent",
        )


@pytest.mark.parametrize(
    "stdout_chunks",
    [
        ("Parquet dataset saved to",),
        ("Parquet dataset saved to ",),
        ("Success. Run saved to /tmp/legacy-attempt",),
    ],
)
def test_cli_artifacts_reject_malformed_or_legacy_unpaired_lines(
    tmp_path: Path,
    stdout_chunks: tuple[str, ...],
) -> None:
    """Malformed dataset lines and legacy unpaired run lines fail closed."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    with pytest.raises(RuntimeError, match="CLI"):
        native_lifecycle.parse_completed_run_artifacts(
            stdout_chunks,
            output_root=output_root,
            expected_phenotype_count=1,
            run_label="malformed",
        )


def test_cli_artifacts_reject_duplicate_dataset_paths(tmp_path: Path) -> None:
    """The same reported dataset cannot satisfy two phenotype outputs."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )

    with pytest.raises(RuntimeError, match="duplicate"):
        native_lifecycle.parse_completed_run_artifacts(
            fixture.stdout_chunks * 2,
            output_root=fixture.output_root,
            expected_phenotype_count=2,
            run_label="duplicate",
        )


def test_cli_artifacts_reject_paths_outside_output_root(tmp_path: Path) -> None:
    """A resolved dataset path cannot escape the requested transaction root."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    outside_parts_directory = tmp_path / "outside" / "attempts" / "attempt-current" / "trait_0001_trait" / "parts"
    outside_parts_directory.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="escapes"):
        native_lifecycle.parse_completed_run_artifacts(
            (f"{native_lifecycle.PARQUET_DIRECTORY_LINE_PREFIX}{outside_parts_directory}",),
            output_root=output_root,
            expected_phenotype_count=1,
            run_label="outside",
        )


def test_cli_artifacts_reject_relative_dataset_paths(tmp_path: Path) -> None:
    """CLI dataset authority must use an absolute path."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    with pytest.raises(RuntimeError, match="absolute"):
        native_lifecycle.parse_completed_run_artifacts(
            (f"{native_lifecycle.PARQUET_DIRECTORY_LINE_PREFIX}attempts/attempt-current/trait/parts",),
            output_root=output_root,
            expected_phenotype_count=1,
            run_label="relative",
        )


def test_cli_artifacts_reject_wrong_transaction_path_shape(tmp_path: Path) -> None:
    """Reported datasets must have shape attempts/<attempt>/<phenotype>/parts."""
    output_root = tmp_path / "outputs"
    wrong_parts_directory = output_root / "transactions" / "attempt-current" / "trait_0001_trait" / "parts"
    wrong_parts_directory.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="shape"):
        native_lifecycle.parse_completed_run_artifacts(
            (f"{native_lifecycle.PARQUET_DIRECTORY_LINE_PREFIX}{wrong_parts_directory}",),
            output_root=output_root,
            expected_phenotype_count=1,
            run_label="shape",
        )


@pytest.mark.parametrize("line_separator", ["\u2028", "\u2029"])
def test_cli_artifacts_reject_unicode_line_separators_in_paths(
    tmp_path: Path,
    line_separator: str,
) -> None:
    """Unicode line separators cannot truncate a reported path into a valid artifact."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    parts_directory = fixture.part_paths[0].parent
    forged_line = f"{native_lifecycle.PARQUET_DIRECTORY_LINE_PREFIX}{parts_directory}{line_separator}continuation\n"

    with pytest.raises(RuntimeError, match="CLI artifact"):
        native_lifecycle.parse_completed_run_artifacts(
            (forged_line,),
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="unicode-separator",
        )


def test_completed_output_rejects_non_schema_v0_manifest_fields(tmp_path: Path) -> None:
    """A completed manifest must contain exactly the schema-v0 field set."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    manifest = read_json_mapping(fixture.manifest_paths[0])
    manifest["legacy_field"] = True
    write_json_mapping(fixture.manifest_paths[0], manifest)
    refresh_completed_terminal_hashes(fixture)

    with pytest.raises(RuntimeError, match="fields differ from schema v0"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="manifest-schema",
        )


def test_completed_output_rejects_duplicate_manifest_json_fields(tmp_path: Path) -> None:
    """Immutable manifest JSON cannot rely on last-key-wins parsing."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    manifest_path = fixture.manifest_paths[0]
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert manifest_text.startswith("{")
    manifest_path.write_text(f'{{"status":"completed",{manifest_text[1:]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate object field"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="duplicate-manifest-field",
        )


def test_completed_output_rejects_nonfinite_json_number_overflow(tmp_path: Path) -> None:
    """JSON float overflow cannot enter evidence as a Python infinity."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    manifest_path = fixture.manifest_paths[0]
    manifest_text = manifest_path.read_text(encoding="utf-8")
    overflow_manifest_text = manifest_text.replace('"variant_count":4', '"variant_count":1e309')
    assert overflow_manifest_text != manifest_text
    manifest_path.write_text(overflow_manifest_text, encoding="utf-8")
    refresh_completed_terminal_hashes(fixture)

    with pytest.raises(RuntimeError, match="finite float range"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="nonfinite-json",
        )


def test_completed_output_rejects_lone_surrogate_json_string(tmp_path: Path) -> None:
    """Escaped lone surrogates are not valid Rust-compatible JSON strings."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    manifest_path = fixture.manifest_paths[0]
    manifest_text = manifest_path.read_text(encoding="utf-8")
    surrogate_manifest_text = manifest_text.replace('"engine":"native"', '"engine":"\\ud800"')
    assert surrogate_manifest_text != manifest_text
    manifest_path.write_text(surrogate_manifest_text, encoding="utf-8")
    refresh_completed_terminal_hashes(fixture)

    with pytest.raises(RuntimeError, match="lone surrogate"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="surrogate-json",
        )


def test_completed_output_rejects_missing_terminal_finalization(tmp_path: Path) -> None:
    """A terminal outcome is not authoritative without its raw-hash finalization."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    fixture.finalization_path.unlink()

    with pytest.raises(RuntimeError, match="terminal finalization"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="missing-finalization",
        )


def test_completed_output_rejects_tampered_terminal_outcome_raw_hash(tmp_path: Path) -> None:
    """Finalization must bind the exact raw bytes of the terminal outcome."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    outcome = read_json_mapping(fixture.outcome_path)
    terminal = typing.cast("dict[str, typing.Any]", outcome["record"])
    terminal_phenotypes = typing.cast("list[dict[str, typing.Any]]", terminal["phenotypes"])
    terminal_phenotypes[0]["run_manifest_sha256"] = "f" * 64
    write_json_mapping(fixture.outcome_path, outcome)

    with pytest.raises(RuntimeError, match="stale terminal binding"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="terminal-hash",
        )


def test_completed_output_rejects_tampered_manifest_raw_hash(tmp_path: Path) -> None:
    """Raw manifest bytes must equal the digest in the completed terminal."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    manifest_path = fixture.manifest_paths[0]
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="terminal hash"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="manifest-hash",
        )


def test_completed_output_rejects_tampered_receipt_file(tmp_path: Path) -> None:
    """The immutable receipt file must exactly equal its completed manifest copy."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    receipt = read_json_mapping(fixture.receipt_paths[0])
    receipt["part_size_bytes"] = typing.cast("int", receipt["part_size_bytes"]) + 1
    write_json_mapping(fixture.receipt_paths[0], receipt)

    with pytest.raises(RuntimeError, match="receipt differs"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="receipt",
        )


def test_completed_output_rejects_type_coercing_receipt_tamper(tmp_path: Path) -> None:
    """Boolean and float equality cannot bypass typed receipt validation."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    receipt = read_json_mapping(fixture.receipt_paths[0])
    receipt_footer = typing.cast("dict[str, typing.Any]", receipt["footer"])
    receipt_footer["schema_version"] = 0.0
    write_json_mapping(fixture.receipt_paths[0], receipt)

    with pytest.raises(RuntimeError, match="must be an integer"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="typed-receipt",
        )


def test_completed_output_rejects_tampered_embedded_footer(tmp_path: Path) -> None:
    """The embedded Parquet footer must exactly equal the immutable receipt footer."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    receipt = read_json_mapping(fixture.receipt_paths[0])
    embedded_footer = typing.cast(
        "dict[str, typing.Any]",
        json.loads(json.dumps(receipt["footer"])),
    )
    embedded_footer["attempt_id"] = "attempt-forged"
    rewrite_first_part_footer(fixture, embedded_footer)

    with pytest.raises(RuntimeError, match="footer"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="footer",
        )


def test_completed_output_rejects_type_coercing_embedded_footer(tmp_path: Path) -> None:
    """Parsed JSON equality cannot normalize a malformed embedded footer."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    receipt = read_json_mapping(fixture.receipt_paths[0])
    embedded_footer = typing.cast(
        "dict[str, typing.Any]",
        json.loads(json.dumps(receipt["footer"])),
    )
    embedded_footer["schema_version"] = False
    rewrite_first_part_footer(fixture, embedded_footer)

    with pytest.raises(RuntimeError, match="must be an integer"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="typed-footer",
        )


def test_completed_output_rejects_whitespace_only_footer_phenotype(tmp_path: Path) -> None:
    """Footer phenotype validation matches the native non-whitespace contract."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("   ",),
        with_interrupted_ancestor=False,
    )

    with pytest.raises(RuntimeError, match="whitespace-only"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="whitespace-phenotype",
        )


def test_completed_output_rejects_tampered_parquet_raw_hash(tmp_path: Path) -> None:
    """Raw Parquet bytes must retain both the immutable receipt size and digest."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    part_path = fixture.part_paths[0]
    part_path.write_bytes(part_path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="raw bytes"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="part-hash",
        )


def test_completed_output_rejects_wrong_parquet_schema(tmp_path: Path) -> None:
    """Raw-byte-valid Parquet still must use the canonical 14-column schema."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    receipt = read_json_mapping(fixture.receipt_paths[0])
    metadata = {
        native_lifecycle.PART_BINDING_METADATA_KEY: json.dumps(
            receipt["footer"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    }
    wrong_table = pa.table({"variant_index": [0, 1, 2, 3]}).replace_schema_metadata(metadata)
    pq.write_table(wrong_table, fixture.part_paths[0])
    refresh_first_part_raw_binding(fixture)

    with pytest.raises(RuntimeError, match="canonical output schema"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="schema",
        )


def test_completed_output_rejects_parquet_field_metadata(tmp_path: Path) -> None:
    """Canonical field equality includes field-level metadata."""
    fixture = write_output_transaction(
        tmp_path / "outputs",
        phenotype_names=("trait",),
        with_interrupted_ancestor=False,
    )
    part_path = fixture.part_paths[0]
    table = pq.read_table(part_path)
    fields = list(table.schema)
    fields[0] = fields[0].with_metadata({b"tampered": b"true"})
    schema = pa.schema(fields, metadata=table.schema.metadata)
    pq.write_table(pa.Table.from_arrays(table.columns, schema=schema), part_path)
    refresh_first_part_raw_binding(fixture)

    with pytest.raises(RuntimeError, match="canonical output schema"):
        native_lifecycle.collect_completed_output_evidence(
            fixture.stdout_chunks,
            output_root=fixture.output_root,
            expected_phenotype_count=1,
            run_label="field-metadata",
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
