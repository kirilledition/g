"""Tests for output persistence and schema helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.engine import LinearChunkAccumulator
from g.io.output import (
    build_chunk_file_name,
    cast_frame_to_schema,
    finalize_chunks_to_parquet,
    get_output_schema,
    persist_chunked_results,
    prepare_output_run,
    resolve_output_run_paths,
    scan_committed_chunk_identifiers,
    write_chunk_to_disk,
)
from g.models import LinearAssociationChunkResult, VariantMetadata
from g.types import AssociationMode

if TYPE_CHECKING:
    from pathlib import Path

    from g.engine import LinearChunkPayload


def create_linear_chunk_payload(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
) -> LinearChunkPayload:
    """Build a minimal linear chunk payload for testing."""
    from g.engine import LinearChunkPayload as Payload

    return Payload(
        chunk_identifier=chunk_identifier,
        variant_start_index=chunk_identifier,
        variant_stop_index=variant_stop_index,
        chromosome=np.asarray(["1"]),
        position=np.asarray([123 + chunk_identifier], dtype=np.int64),
        variant_identifier=np.asarray([variant_identifier]),
        allele_one=np.asarray(["A"]),
        allele_two=np.asarray(["C"]),
        allele_one_frequency=np.asarray([0.5], dtype=np.float32),
        observation_count=np.asarray([100], dtype=np.int32),
        beta=np.asarray([0.1], dtype=np.float32),
        standard_error=np.asarray([0.01], dtype=np.float32),
        t_statistic=np.asarray([10.0], dtype=np.float32),
        p_value=np.asarray([1.0e-5], dtype=np.float32),
        is_valid=np.asarray([True], dtype=np.bool_),
    )


def create_linear_chunk_accumulator(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
) -> LinearChunkAccumulator:
    """Build a minimal linear chunk accumulator for testing."""
    return LinearChunkAccumulator(
        metadata=VariantMetadata(
            variant_start_index=chunk_identifier,
            variant_stop_index=variant_stop_index,
            chromosome=np.asarray(["1"]),
            variant_identifiers=np.asarray([variant_identifier]),
            position=np.asarray([123 + chunk_identifier], dtype=np.int64),
            allele_one=np.asarray(["A"]),
            allele_two=np.asarray(["C"]),
        ),
        allele_one_frequency=jnp.asarray([0.5], dtype=jnp.float32),
        observation_count=jnp.asarray([100], dtype=jnp.int32),
        linear_result=LinearAssociationChunkResult(
            beta=jnp.asarray([0.1], dtype=jnp.float32),
            standard_error=jnp.asarray([0.01], dtype=jnp.float32),
            test_statistic=jnp.asarray([10.0], dtype=jnp.float32),
            p_value=jnp.asarray([1.0e-5], dtype=jnp.float32),
            valid_mask=jnp.asarray([True], dtype=jnp.bool_),
        ),
    )


class TestOutputSchema:
    """Tests for output schema helpers."""

    def test_get_output_schema_returns_logistic_columns(self) -> None:
        """Ensure schema lookup returns the logistic output schema."""
        logistic_schema = get_output_schema(AssociationMode.LOGISTIC)

        assert "firth_flag" in logistic_schema
        assert "iteration_count" in logistic_schema

    def test_cast_frame_to_schema_reorders_and_casts_columns(self) -> None:
        """Ensure output frames are cast and reordered to the fixed linear schema."""
        data_frame = pl.DataFrame(
            {
                "beta": [1.5],
                "chromosome": [1],
                "chunk_identifier": [7],
                "variant_start_index": [0],
                "variant_stop_index": [1],
                "position": [123],
                "variant_identifier": ["variant1"],
                "allele_one": ["A"],
                "allele_two": ["G"],
                "allele_one_frequency": [0.25],
                "observation_count": [100],
                "standard_error": [0.1],
                "t_statistic": [15.0],
                "p_value": [1.0e-5],
                "is_valid": [True],
            }
        )

        cast_data_frame = cast_frame_to_schema(data_frame, AssociationMode.LINEAR)

        assert cast_data_frame.schema == get_output_schema(AssociationMode.LINEAR)
        assert cast_data_frame.columns == list(get_output_schema(AssociationMode.LINEAR))
        assert cast_data_frame.get_column("chromosome").to_list() == ["1"]


class TestResolveOutputRunPaths:
    """Tests for output path resolution."""

    def test_appends_mode_suffix(self, tmp_path: Path) -> None:
        """Ensure run directory gets association mode in its suffix."""
        output_run_paths = resolve_output_run_paths(tmp_path / "results/output", AssociationMode.LINEAR)
        assert output_run_paths.run_directory == tmp_path / "results/output.linear.run"
        assert output_run_paths.chunks_directory == tmp_path / "results/output.linear.run/chunks"

    def test_preserves_explicit_run_suffix(self, tmp_path: Path) -> None:
        """Ensure a .run suffix is kept as-is."""
        output_run_paths = resolve_output_run_paths(tmp_path / "my_output.run", AssociationMode.LOGISTIC)
        assert output_run_paths.run_directory == tmp_path / "my_output.run"


class TestScanCommittedChunkIdentifiers:
    """Tests for filesystem-based chunk discovery."""

    def test_returns_empty_for_missing_directory(self, tmp_path: Path) -> None:
        """Ensure a missing directory returns an empty set."""
        assert scan_committed_chunk_identifiers(tmp_path / "nonexistent") == frozenset()

    def test_discovers_committed_chunks(self, tmp_path: Path) -> None:
        """Ensure completed chunk files are discovered by name."""
        (tmp_path / "chunk_000000000.arrow").write_bytes(b"")
        (tmp_path / "chunk_000000512.arrow").write_bytes(b"")
        assert scan_committed_chunk_identifiers(tmp_path) == frozenset({0, 512})

    def test_ignores_temporary_files(self, tmp_path: Path) -> None:
        """Ensure .arrow.tmp files are not counted as committed."""
        (tmp_path / "chunk_000000000.arrow").write_bytes(b"")
        (tmp_path / "chunk_000000512.arrow.tmp").write_bytes(b"")
        assert scan_committed_chunk_identifiers(tmp_path) == frozenset({0})


class TestPrepareOutputRun:
    """Tests for output run preparation and resume."""

    def test_creates_directories(self, tmp_path: Path) -> None:
        """Ensure fresh run creates the chunks directory."""
        prepared_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=False,
        )
        assert prepared_output_run.output_run_paths.chunks_directory.exists()
        assert prepared_output_run.committed_chunk_identifiers == frozenset()

    def test_rejects_non_empty_directory_without_resume(self, tmp_path: Path) -> None:
        """Ensure a non-empty run directory fails without --resume."""
        run_dir = tmp_path / "output.linear.run"
        run_dir.mkdir(parents=True)
        (run_dir / "stale_file.txt").write_text("stale", encoding="utf-8")
        with pytest.raises(ValueError, match="already exists and is not empty"):
            prepare_output_run(
                output_root=tmp_path / "output",
                association_mode=AssociationMode.LINEAR,
                resume=False,
            )

    def test_resumes_with_committed_chunks(self, tmp_path: Path) -> None:
        """Ensure resume scans and returns existing chunk identifiers."""
        run_dir = tmp_path / "output.linear.run"
        chunks_dir = run_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        payload = create_linear_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=2,
            variant_identifier="v0",
        )
        write_chunk_to_disk(payload, chunks_dir, AssociationMode.LINEAR)

        prepared_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=True,
        )
        assert prepared_output_run.committed_chunk_identifiers == frozenset({0})


class TestWriteChunkToDisk:
    """Tests for atomic chunk file writes."""

    def test_produces_readable_arrow_file(self, tmp_path: Path) -> None:
        """Ensure chunk files are valid Arrow IPC."""
        payload = create_linear_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=2,
            variant_identifier="v0",
        )
        write_chunk_to_disk(payload, tmp_path, AssociationMode.LINEAR)
        chunk_path = tmp_path / build_chunk_file_name(0)
        assert chunk_path.exists()
        frame = pl.read_ipc(chunk_path)
        assert frame.height == 1
        assert frame.schema == get_output_schema(AssociationMode.LINEAR)
        assert frame.get_column("chunk_identifier").to_list() == [0]

    def test_atomic_write_leaves_no_tmp_files(self, tmp_path: Path) -> None:
        """Ensure the temporary file is cleaned up after a successful write."""
        payload = create_linear_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=2,
            variant_identifier="v0",
        )
        write_chunk_to_disk(payload, tmp_path, AssociationMode.LINEAR)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestPersistChunkedResults:
    """Tests for the end-to-end background writer pipeline."""

    def test_persists_multiple_chunks(self, tmp_path: Path) -> None:
        """Ensure all chunks from an iterator are written to disk."""
        prepared_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=False,
        )
        accumulators = [
            create_linear_chunk_accumulator(
                chunk_identifier=0,
                variant_stop_index=2,
                variant_identifier="v0",
            ),
            create_linear_chunk_accumulator(
                chunk_identifier=2,
                variant_stop_index=4,
                variant_identifier="v2",
            ),
        ]
        persist_chunked_results(
            frame_iterator=iter(accumulators),
            output_run_paths=prepared_output_run.output_run_paths,
            association_mode=AssociationMode.LINEAR,
        )
        committed = scan_committed_chunk_identifiers(prepared_output_run.output_run_paths.chunks_directory)
        assert committed == frozenset({0, 2})

    def test_resume_skips_committed_chunks(self, tmp_path: Path) -> None:
        """Ensure resume correctly skips already-committed chunks."""
        prepared_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=False,
        )
        first_payload = create_linear_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=2,
            variant_identifier="v0",
        )
        write_chunk_to_disk(
            first_payload,
            prepared_output_run.output_run_paths.chunks_directory,
            AssociationMode.LINEAR,
        )

        resumed_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=True,
        )
        assert resumed_output_run.committed_chunk_identifiers == frozenset({0})

        accumulators = [
            create_linear_chunk_accumulator(
                chunk_identifier=2,
                variant_stop_index=4,
                variant_identifier="v2",
            ),
        ]
        persist_chunked_results(
            frame_iterator=iter(accumulators),
            output_run_paths=prepared_output_run.output_run_paths,
            association_mode=AssociationMode.LINEAR,
        )
        final_committed = scan_committed_chunk_identifiers(prepared_output_run.output_run_paths.chunks_directory)
        assert final_committed == frozenset({0, 2})


class TestFinalizeChunksToParquet:
    """Tests for Arrow chunk compaction to Parquet."""

    def test_compacts_valid_chunks(self, tmp_path: Path) -> None:
        """Ensure finalization creates a valid Parquet file."""
        prepared_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=False,
        )
        for chunk_identifier, variant_identifier in [(0, "v0"), (2, "v2")]:
            payload = create_linear_chunk_payload(
                chunk_identifier=chunk_identifier,
                variant_stop_index=chunk_identifier + 2,
                variant_identifier=variant_identifier,
            )
            write_chunk_to_disk(
                payload,
                prepared_output_run.output_run_paths.chunks_directory,
                AssociationMode.LINEAR,
            )

        parquet_path = finalize_chunks_to_parquet(prepared_output_run.output_run_paths, AssociationMode.LINEAR)
        frame = pl.read_parquet(parquet_path)
        assert frame.height == 2
        assert frame.schema == get_output_schema(AssociationMode.LINEAR)

    def test_produces_empty_parquet_with_no_chunks(self, tmp_path: Path) -> None:
        """Ensure finalization handles empty chunk directories gracefully."""
        prepared_output_run = prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.LINEAR,
            resume=False,
        )
        parquet_path = finalize_chunks_to_parquet(prepared_output_run.output_run_paths, AssociationMode.LINEAR)
        frame = pl.read_parquet(parquet_path)
        assert frame.height == 0
        assert frame.schema == get_output_schema(AssociationMode.LINEAR)
