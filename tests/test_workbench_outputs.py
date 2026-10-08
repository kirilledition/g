"""Publishing requires complete native manifests and valid Parquet footers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import typing
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet
import pytest

from tooling.workbench import outputs

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class OutputFixture:
    output_root: Path
    manifest_path: Path
    part_path: Path
    manifest: dict[str, object]
    commits: list[dict[str, object]]


def write_part(path: Path, commits: list[dict[str, object]], row_count: int = 4) -> None:
    table = pa.table({"GENPOS": list(range(row_count))}).replace_schema_metadata(
        {b"g.output.chunk_commits": json.dumps(commits).encode("utf-8")},
    )
    pyarrow.parquet.write_table(table, path)


def write_manifest(fixture: OutputFixture) -> None:
    fixture.manifest_path.write_text(json.dumps(fixture.manifest), encoding="utf-8")


def completed_fixture(output_root: Path, phenotype_name: str = "trait_a") -> OutputFixture:
    run_directory = output_root / f"{phenotype_name}.run"
    parts_directory = run_directory / "parts"
    parts_directory.mkdir(parents=True)
    commits: list[dict[str, object]] = [
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
    fixture = OutputFixture(
        output_root=output_root,
        manifest_path=run_directory / "run_manifest.json",
        part_path=parts_directory / "part_000000000_000000002.parquet",
        manifest={
            "status": "completed",
            "execution_plan": {"phenotype_name": phenotype_name, "variant_count": 4},
            "committed_chunks": commits,
        },
        commits=commits,
    )
    write_part(fixture.part_path, fixture.commits)
    write_manifest(fixture)
    return fixture


def test_valid_grouped_parts_hash_all_files_and_both_traits(tmp_path: Path) -> None:
    first = completed_fixture(tmp_path)
    second = completed_fixture(tmp_path, "trait_b")
    log_path = tmp_path / "execution.log"
    log_path.write_text("completed\n", encoding="utf-8")

    identities = outputs.validate_completed_outputs(tmp_path, ("trait_a", "trait_b"), 4)

    expected_paths = (first.manifest_path, first.part_path, second.manifest_path, second.part_path, log_path)
    assert {identity.relative_path for identity in identities} == {
        path.relative_to(tmp_path).as_posix() for path in expected_paths
    }
    assert [identity.relative_path for identity in identities] == sorted(
        identity.relative_path for identity in identities
    )
    for identity in identities:
        content = (tmp_path / identity.relative_path).read_bytes()
        assert identity.sha256 == hashlib.sha256(content).hexdigest()
        assert identity.size_bytes == len(content)


def test_module_import_does_not_load_jax_or_pyarrow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tooling.workbench.outputs; "
            "assert 'jax' not in sys.modules; assert 'pyarrow' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_trait_manifest_is_rejected(tmp_path: Path) -> None:
    completed_fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest count"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a", "trait_b"), 4)


def test_running_attempt_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.manifest["status"] = "running"
    write_manifest(fixture)
    with pytest.raises(ValueError, match="not completed"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_unexpected_or_duplicate_trait_is_rejected(tmp_path: Path) -> None:
    completed_fixture(tmp_path)
    duplicate = completed_fixture(tmp_path, "trait_b")
    duplicate.manifest["execution_plan"] = {"phenotype_name": "trait_a", "variant_count": 4}
    write_manifest(duplicate)
    with pytest.raises(ValueError, match="unexpected or duplicate phenotype"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a", "trait_b"), 4)


@pytest.mark.parametrize("variant_count", [True, 5, "4"])
def test_manifest_variant_count_must_match_input(tmp_path: Path, variant_count: object) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.manifest["execution_plan"] = {"phenotype_name": "trait_a", "variant_count": variant_count}
    write_manifest(fixture)
    with pytest.raises(ValueError, match="variant_count"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


@pytest.mark.parametrize(
    "file_name", ["../outside.parquet", "/tmp/outside.parquet", "nested\\part.parquet", "bad\x00.parquet"]
)
def test_commit_path_traversal_is_rejected(tmp_path: Path, file_name: str) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.commits[0]["chunk_file_name"] = file_name
    write_manifest(fixture)
    with pytest.raises(ValueError, match="unsafe Parquet part"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_missing_committed_part_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.part_path.unlink()
    with pytest.raises(ValueError, match="missing or unexplained"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_unexplained_parquet_outside_trait_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    (tmp_path / "unreferenced.parquet").write_bytes(fixture.part_path.read_bytes())
    with pytest.raises(ValueError, match="unexplained Parquet"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_uncommitted_staging_file_in_parts_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    (fixture.part_path.parent / "unfinished.tmp").write_bytes(b"in progress")
    with pytest.raises(ValueError, match="missing or unexplained"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_chunk_gap_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.commits[1].update({"chunk_identifier": 3, "variant_start_index": 3, "row_count": 1})
    write_manifest(fixture)
    with pytest.raises(ValueError, match="gaps or overlaps"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_chunk_overlap_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.commits[1].update({"chunk_identifier": 1, "variant_start_index": 1, "row_count": 3})
    write_manifest(fixture)
    with pytest.raises(ValueError, match="gaps or overlaps"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_truncated_chunk_range_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.commits.pop()
    write_manifest(fixture)
    with pytest.raises(ValueError, match="full variant range"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_duplicate_chunk_identifier_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.commits.append(fixture.commits[0].copy())
    write_manifest(fixture)
    with pytest.raises(ValueError, match="duplicate chunk identifiers"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_manifest_row_count_must_match_chunk_range(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.commits[0]["row_count"] = 1
    write_manifest(fixture)
    with pytest.raises(ValueError, match="inconsistent chunk range or row_count"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_actual_parquet_rows_must_match_manifest(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    write_part(fixture.part_path, fixture.commits, row_count=3)
    with pytest.raises(ValueError, match="footer row count"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_actual_parquet_chunk_commits_must_match_manifest(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    changed_commits = [entry.copy() for entry in fixture.commits]
    changed_commits[0]["chunk_file_name"] = "different.parquet"
    write_part(fixture.part_path, changed_commits)
    with pytest.raises(ValueError, match="footer chunk commits differ"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_parquet_without_native_commit_metadata_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    pyarrow.parquet.write_table(pa.table({"GENPOS": [0, 1, 2, 3]}), fixture.part_path)
    with pytest.raises(ValueError, match=r"missing g\.output\.chunk_commits"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


@pytest.mark.parametrize("content", [b"", b"PAR1", b"BAD!data\x04\x00\x00\x00PAR1", b"PAR1data\xff\xff\xff\xffPAR1"])
def test_truncated_or_invalid_parquet_is_rejected(tmp_path: Path, content: bytes) -> None:
    fixture = completed_fixture(tmp_path)
    fixture.part_path.write_bytes(content)
    with pytest.raises(ValueError, match="Parquet part"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_output_file_symlink_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    (tmp_path / "linked.log").symlink_to(fixture.manifest_path)
    with pytest.raises(ValueError, match="symlink"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_output_directory_symlink_is_rejected(tmp_path: Path) -> None:
    fixture = completed_fixture(tmp_path)
    (tmp_path / "linked-directory").symlink_to(fixture.part_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        outputs.validate_completed_outputs(tmp_path, ("trait_a",), 4)


def test_output_root_symlink_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "actual"
    completed_fixture(output_root)
    alias = tmp_path / "alias"
    alias.symlink_to(output_root, target_is_directory=True)
    with pytest.raises(ValueError, match="without symlinks"):
        outputs.validate_completed_outputs(alias, ("trait_a",), 4)
