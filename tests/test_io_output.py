"""Tests for Rust-backed output persistence."""

from __future__ import annotations

import typing
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest

from g import _core
from g.io import output
from g.types import AssociationMode

EXPECTED_FINAL_COLUMNS = [
    "CHROM",
    "GENPOS",
    "ID",
    "ALLELE0",
    "ALLELE1",
    "A1FREQ",
    "INFO",
    "N",
    "TEST",
    "BETA",
    "SE",
    "CHISQ",
    "LOG10P",
    "EXTRA",
]
EXPECTED_CHUNK_COLUMNS = [
    "chunk_identifier",
    "variant_start_index",
    "variant_stop_index",
    *EXPECTED_FINAL_COLUMNS,
]
TEST_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"


class NativeChunkWritingCallback:
    """Callback that writes deterministic association values for native chunks."""

    def __init__(self, writer_session: typing.Any, extra_code_value: int | None = None) -> None:
        self.writer_session = writer_session
        self.extra_code_value = extra_code_value
        self.free_buffers: list[np.ndarray] = []

    def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> np.ndarray:
        if self.free_buffers:
            dosage_buffer = self.free_buffers.pop()
            if dosage_buffer.shape == (sample_count, variant_count):
                return dosage_buffer
        return np.empty((sample_count, variant_count), dtype=np.float32, order="C")

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        variant_count = metadata.variant_stop_index - metadata.variant_start_index
        extra_code = (
            np.full(variant_count, self.extra_code_value, dtype=np.int32) if self.extra_code_value is not None else None
        )
        self.writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=np.full(variant_count, 0.1, dtype=np.float32),
            standard_error=np.full(variant_count, 0.01, dtype=np.float32),
            chi_squared=np.full(variant_count, 10.0, dtype=np.float32),
            log10_p_value=np.full(variant_count, 5.0, dtype=np.float32),
            extra_code=extra_code,
        )
        self.free_buffers.append(genotype_matrix)


def write_native_chunks(
    output_run_paths: output.OutputRunPaths,
    association_mode: AssociationMode,
    *,
    extra_code_value: int | None = None,
) -> None:
    writer_session = output.create_output_writer_session(
        output_run_paths,
        association_mode,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
    )
    callback = NativeChunkWritingCallback(writer_session, extra_code_value)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
        writer_session.finish()
    except Exception:
        writer_session.abort()
        raise


def test_resolve_output_run_paths_appends_mode_suffix(tmp_path: Path) -> None:
    output_run_paths = output.resolve_output_run_paths(tmp_path / "results/output", AssociationMode.REGENIE2_LINEAR)
    assert output_run_paths.run_directory == tmp_path / "results/output.regenie2_linear.run"
    assert output_run_paths.chunks_directory == tmp_path / "results/output.regenie2_linear.run/chunks"


def test_scan_committed_chunk_identifiers_discovers_single_chunk_files(tmp_path: Path) -> None:
    (tmp_path / "chunk_000000000.arrow").write_bytes(b"")
    (tmp_path / "chunk_000000512.arrow").write_bytes(b"")
    assert output.scan_committed_chunk_identifiers(tmp_path) == frozenset({0, 512})


def test_prepare_output_run_rejects_non_empty_directory_without_resume(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    run_directory.mkdir(parents=True)
    (run_directory / "stale_file.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        output.prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=False,
        )


def test_native_writer_uses_shared_schema_and_null_placeholders(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(output_run_paths, AssociationMode.REGENIE2_LINEAR)

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("TEST").to_list() == ["ADD", "ADD", "ADD", "ADD"]
    assert frame.get_column("INFO").to_list() == [1.0, 1.0, 1.0, 1.0]
    assert frame.get_column("EXTRA").to_list() == [None, None, None, None]


def test_native_binary_writer_maps_extra_code_to_label(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=1)

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("EXTRA").to_list() == [None, None, None, None]


def test_native_binary_writer_maps_test_fail_extra_code_to_label(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=3)

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("EXTRA").to_list() == ["TEST_FAIL", "TEST_FAIL", "TEST_FAIL", "TEST_FAIL"]


def test_prepare_output_run_resume_detects_native_chunks(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    write_native_chunks(prepared_output_run.output_run_paths, AssociationMode.REGENIE2_LINEAR)

    resumed_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    assert resumed_output_run.committed_chunk_identifiers == frozenset({0, 2})


def test_prepare_output_run_strict_resume_validates_manifest_chunks(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    write_native_chunks(prepared_output_run.output_run_paths, AssociationMode.REGENIE2_LINEAR)

    resumed_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
        resume_mode=output.types.ResumeMode.STRICT,
    )

    assert resumed_output_run.committed_chunk_identifiers == frozenset({0, 2})


def test_prepare_output_run_rejects_incompatible_manifest_even_in_fast_mode(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    manifest_path = output.get_run_manifest_path(prepared_output_run.output_run_paths)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["association_mode"] = AssociationMode.REGENIE2_BINARY.value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="association mode is incompatible"):
        output.prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=True,
        )


def test_prepare_output_run_strict_resume_requires_manifest(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)

    with pytest.raises(ValueError, match="Strict resume requires run_manifest.json"):
        output.prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=True,
            resume_mode=output.types.ResumeMode.STRICT,
        )


def test_chunk_arrow_schema_is_shared_between_linear_and_binary(tmp_path: Path) -> None:
    linear_run_paths = output.OutputRunPaths(tmp_path / "linear", tmp_path / "linear")
    binary_run_paths = output.OutputRunPaths(tmp_path / "binary", tmp_path / "binary")
    linear_run_paths.chunks_directory.mkdir()
    binary_run_paths.chunks_directory.mkdir()
    write_native_chunks(linear_run_paths, AssociationMode.REGENIE2_LINEAR)
    write_native_chunks(binary_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=3)

    linear_schema = pyarrow.ipc.open_file(
        output.iter_sorted_chunk_file_paths(linear_run_paths.chunks_directory)[0],
    ).schema
    binary_schema = pyarrow.ipc.open_file(
        output.iter_sorted_chunk_file_paths(binary_run_paths.chunks_directory)[0],
    ).schema
    assert linear_schema == binary_schema
    assert linear_schema.names == EXPECTED_CHUNK_COLUMNS
    assert linear_schema.field("INFO").nullable
    assert linear_schema.field("EXTRA").nullable


def test_finalize_chunks_to_parquet_projects_technical_columns_away(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        resume=False,
    )
    write_native_chunks(prepared_output_run.output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=1)

    parquet_path = output.finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
    assert parquet_frame.get_column("EXTRA").to_list() == [None, None, None, None]
    parquet_schema = pq.ParquetFile(parquet_path).schema_arrow
    assert parquet_schema.names == EXPECTED_FINAL_COLUMNS
    assert parquet_schema.field("INFO").nullable
    assert parquet_schema.field("EXTRA").nullable
    parquet_metadata = pq.ParquetFile(parquet_path).metadata.metadata
    assert parquet_metadata is not None
    assert parquet_metadata[b"g.output.schema_version"] == b"1"
    assert parquet_metadata[b"g.output.association_mode"] == b"regenie2_binary"
    assert parquet_metadata[b"g.output.chunk_file_count"] == b"1"
    assert parquet_metadata[b"g.output.row_count"] == b"4"
    assert parquet_metadata[b"g.output.writer"] == b"rust"
    manifest = json.loads(
        output.get_run_manifest_path(prepared_output_run.output_run_paths).read_text(encoding="utf-8")
    )
    assert manifest["finalized"] is True
    assert manifest["final_row_count"] == 4


def test_finalize_chunks_to_parquet_writes_empty_schema_when_no_chunks_exist(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )

    parquet_path = output.finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.height == 0
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
