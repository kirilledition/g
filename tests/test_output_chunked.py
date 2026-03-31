from __future__ import annotations

from pathlib import Path

import polars as pl

from g.output.chunked import finalize_chunks_to_parquet, resolve_output_run_paths, write_chunk_file
from g.output.manifest import ManifestChunkRecord, OutputManifest
from g.output.schema import cast_frame_to_schema, get_output_schema, write_schema_file


def test_resolve_output_run_paths_uses_mode_suffix() -> None:
    output_run_paths = resolve_output_run_paths(Path("results/output"), "linear")
    assert output_run_paths.run_directory == Path("results/output.linear.run")
    assert output_run_paths.chunks_directory == Path("results/output.linear.run/chunks")


def test_write_schema_file_persists_column_metadata(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.arrow.json"
    write_schema_file(schema_path, "linear")
    schema_contents = schema_path.read_text(encoding="utf-8")
    assert "schema_version" in schema_contents
    assert "allele_one_frequency" in schema_contents


def test_cast_frame_to_schema_enforces_linear_schema() -> None:
    input_frame = pl.DataFrame(
        {
            "chromosome": ["1"],
            "position": [123],
            "variant_identifier": ["variant1"],
            "allele_one": ["A"],
            "allele_two": ["C"],
            "allele_one_frequency": [0.5],
            "observation_count": [100],
            "beta": [0.1],
            "standard_error": [0.01],
            "t_statistic": [10.0],
            "p_value": [1.0e-5],
            "is_valid": [True],
        }
    )
    cast_frame = cast_frame_to_schema(input_frame, "linear")
    assert cast_frame.schema == get_output_schema("linear")


def test_manifest_records_and_reads_committed_chunks(tmp_path: Path) -> None:
    manifest = OutputManifest(tmp_path / "manifest.sqlite")
    manifest.register_run("run_identifier", "linear", "1")
    manifest.insert_committed_chunk(
        ManifestChunkRecord(
            chunk_identifier=0,
            chunk_label="linear_chunk_000000",
            row_count=8,
            file_path="chunks/000000.arrow",
            checksum_sha256="checksum",
            status="committed",
        )
    )
    committed_chunk_identifiers = manifest.load_committed_chunk_identifiers()
    manifest.close()
    assert committed_chunk_identifiers == {0}


def test_finalize_chunks_to_parquet_compacts_rows(tmp_path: Path) -> None:
    output_run_paths = resolve_output_run_paths(tmp_path / "results", "linear")
    output_run_paths.chunks_directory.mkdir(parents=True, exist_ok=True)

    chunk_frame = pl.DataFrame(
        {
            "chromosome": ["1"],
            "position": [123],
            "variant_identifier": ["variant1"],
            "allele_one": ["A"],
            "allele_two": ["C"],
            "allele_one_frequency": [0.5],
            "observation_count": [100],
            "beta": [0.1],
            "standard_error": [0.01],
            "t_statistic": [10.0],
            "p_value": [1.0e-5],
            "is_valid": [True],
        }
    )
    chunk_file_path = output_run_paths.chunks_directory / "000000.arrow"
    write_chunk_file(chunk_frame, chunk_file_path)

    manifest = OutputManifest(output_run_paths.manifest_path)
    manifest.register_run("run_identifier", "linear", "1")
    manifest.insert_committed_chunk(
        ManifestChunkRecord(
            chunk_identifier=0,
            chunk_label="linear_chunk_000000",
            row_count=1,
            file_path="chunks/000000.arrow",
            checksum_sha256="checksum",
            status="committed",
        )
    )
    manifest.close()

    final_parquet_path = finalize_chunks_to_parquet(output_run_paths)
    parquet_frame = pl.read_parquet(final_parquet_path)
    assert parquet_frame.height == 1
