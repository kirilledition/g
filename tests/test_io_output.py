"""Tests for Rust-backed output persistence."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest

from g import engine
from g.io import output
from g.models import VariantMetadata
from g.types import AssociationMode

if typing.TYPE_CHECKING:
    from pathlib import Path


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


def create_regenie_chunk_payload(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
    chromosome: str = "1",
    allele_zero: str = "C",
    allele_one: str = "A",
    extra_code: npt.NDArray[np.int32] | None = None,
) -> engine.Regenie2ChunkPayload:
    return engine.Regenie2ChunkPayload(
        chunk_identifier=chunk_identifier,
        variant_start_index=chunk_identifier,
        variant_stop_index=variant_stop_index,
        chromosome=np.asarray([chromosome]),
        position=np.asarray([123 + chunk_identifier], dtype=np.int64),
        variant_identifier=np.asarray([variant_identifier]),
        allele_zero=np.asarray([allele_zero]),
        allele_one=np.asarray([allele_one]),
        allele_one_frequency=np.asarray([0.5], dtype=np.float32),
        observation_count=np.asarray([100], dtype=np.int32),
        beta=np.asarray([0.1], dtype=np.float32),
        standard_error=np.asarray([0.01], dtype=np.float32),
        chi_squared=np.asarray([10.0], dtype=np.float32),
        log10_p_value=np.asarray([5.0], dtype=np.float32),
        extra_code=extra_code,
    )


def create_regenie_chunk_accumulator(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
    chromosome: str = "1",
    allele_zero: str = "C",
    allele_one: str = "A",
    extra_code: jax.Array | None = None,
) -> engine.Regenie2ChunkAccumulator:
    return engine.Regenie2ChunkAccumulator(
        metadata=VariantMetadata(
            variant_start_index=chunk_identifier,
            variant_stop_index=variant_stop_index,
            chromosome=np.asarray([chromosome]),
            variant_identifiers=np.asarray([variant_identifier]),
            position=np.asarray([123 + chunk_identifier], dtype=np.int64),
            allele_one=np.asarray([allele_one]),
            allele_two=np.asarray([allele_zero]),
        ),
        allele_one_frequency=jnp.asarray([0.5], dtype=jnp.float32),
        observation_count=jnp.asarray([100], dtype=jnp.int32),
        beta=jnp.asarray([0.1], dtype=jnp.float32),
        standard_error=jnp.asarray([0.01], dtype=jnp.float32),
        chi_squared=jnp.asarray([10.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([5.0], dtype=jnp.float32),
        extra_code=extra_code,
    )


def write_reference_chunk(chunk_payload: engine.Regenie2ChunkPayload, chunk_path: Path) -> None:
    reference_output_frame = pl.DataFrame(
        {
            "chunk_identifier": [chunk_payload.chunk_identifier],
            "variant_start_index": [chunk_payload.variant_start_index],
            "variant_stop_index": [chunk_payload.variant_stop_index],
            "CHROM": chunk_payload.chromosome.tolist(),
            "GENPOS": chunk_payload.position.tolist(),
            "ID": chunk_payload.variant_identifier.tolist(),
            "ALLELE0": chunk_payload.allele_zero.tolist(),
            "ALLELE1": chunk_payload.allele_one.tolist(),
            "A1FREQ": chunk_payload.allele_one_frequency.tolist(),
            "INFO": [None],
            "N": chunk_payload.observation_count.tolist(),
            "TEST": ["ADD"],
            "BETA": chunk_payload.beta.tolist(),
            "SE": chunk_payload.standard_error.tolist(),
            "CHISQ": chunk_payload.chi_squared.tolist(),
            "LOG10P": chunk_payload.log10_p_value.tolist(),
            "EXTRA": [None],
        }
    )
    reference_output_frame.write_ipc(chunk_path, compression="zstd")


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


def test_write_linear_chunk_to_disk_uses_shared_schema_and_null_placeholders(tmp_path: Path) -> None:
    payload = create_regenie_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="v0",
    )

    output.write_chunk_to_disk(payload, tmp_path, AssociationMode.REGENIE2_LINEAR)

    chunk_path = tmp_path / output.build_chunk_file_name(0)
    frame = pl.read_ipc(chunk_path)
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("TEST").to_list() == ["ADD"]
    assert frame.get_column("INFO").to_list() == [None]
    assert frame.get_column("EXTRA").to_list() == [None]


def test_write_binary_chunk_to_disk_maps_extra_code_to_label(tmp_path: Path) -> None:
    payload = create_regenie_chunk_payload(
        chunk_identifier=7,
        variant_stop_index=8,
        variant_identifier="variant1",
        chromosome="22",
        allele_zero="G",
        allele_one="A",
        extra_code=np.asarray([1], dtype=np.int32),
    )

    output.write_chunk_to_disk(payload, tmp_path, AssociationMode.REGENIE2_BINARY)

    frame = pl.read_ipc(tmp_path / "chunk_000000007.arrow")
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.select("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "TEST", "EXTRA").row(0) == (
        "22",
        130,
        "variant1",
        "G",
        "A",
        "ADD",
        "FIRTH",
    )


def test_persist_chunked_results_batches_multiple_chunks_into_one_arrow_file(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    accumulators = iter(
        [
            create_regenie_chunk_accumulator(chunk_identifier=0, variant_stop_index=1, variant_identifier="v0"),
            create_regenie_chunk_accumulator(chunk_identifier=1, variant_stop_index=2, variant_identifier="v1"),
        ]
    )

    output.persist_chunked_results(
        frame_iterator=accumulators,
        output_run_paths=prepared_output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_LINEAR,
        payload_batch_size=2,
    )

    chunk_paths = tuple(output.iter_sorted_chunk_file_paths(prepared_output_run.output_run_paths.chunks_directory))
    assert [path.name for path in chunk_paths] == ["chunk_000000000_000000001.arrow"]
    assert output.scan_committed_chunk_identifiers(prepared_output_run.output_run_paths.chunks_directory) == frozenset(
        {0, 1}
    )


def test_prepare_output_run_resume_detects_new_and_reference_chunks(tmp_path: Path) -> None:
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    chunks_directory = prepared_output_run.output_run_paths.chunks_directory
    output.write_chunk_to_disk(
        create_regenie_chunk_payload(chunk_identifier=0, variant_stop_index=1, variant_identifier="v0"),
        chunks_directory,
        AssociationMode.REGENIE2_LINEAR,
    )
    write_reference_chunk(
        create_regenie_chunk_payload(chunk_identifier=1, variant_stop_index=2, variant_identifier="v1"),
        chunks_directory / output.build_chunk_file_name(1),
    )
    write_reference_chunk(
        create_regenie_chunk_payload(chunk_identifier=2, variant_stop_index=3, variant_identifier="v2"),
        chunks_directory / output.build_chunk_file_name(2),
    )

    resumed_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    assert resumed_output_run.committed_chunk_identifiers == frozenset({0, 1, 2})


def test_chunk_arrow_schema_is_shared_between_linear_and_binary(tmp_path: Path) -> None:
    linear_chunks_directory = tmp_path / "linear"
    binary_chunks_directory = tmp_path / "binary"
    linear_chunks_directory.mkdir()
    binary_chunks_directory.mkdir()
    linear_payload = create_regenie_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=1,
        variant_identifier="linear_variant",
    )
    binary_payload = create_regenie_chunk_payload(
        chunk_identifier=1,
        variant_stop_index=2,
        variant_identifier="binary_variant",
        chromosome="22",
        allele_zero="G",
        extra_code=np.asarray([3], dtype=np.int32),
    )

    output.write_chunk_to_disk(linear_payload, linear_chunks_directory, AssociationMode.REGENIE2_LINEAR)
    output.write_chunk_to_disk(binary_payload, binary_chunks_directory, AssociationMode.REGENIE2_BINARY)

    linear_schema = pyarrow.ipc.open_file(linear_chunks_directory / "chunk_000000000.arrow").schema
    binary_schema = pyarrow.ipc.open_file(binary_chunks_directory / "chunk_000000001.arrow").schema
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
    output.write_chunk_to_disk(
        create_regenie_chunk_payload(
            chunk_identifier=7,
            variant_stop_index=8,
            variant_identifier="variant1",
            chromosome="22",
            allele_zero="G",
            allele_one="A",
            extra_code=np.asarray([1], dtype=np.int32),
        ),
        prepared_output_run.output_run_paths.chunks_directory,
        AssociationMode.REGENIE2_BINARY,
    )

    parquet_path = output.finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
    assert parquet_frame.select("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "TEST", "EXTRA").row(0) == (
        "22",
        130,
        "variant1",
        "G",
        "A",
        "ADD",
        "FIRTH",
    )
    parquet_schema = pq.ParquetFile(parquet_path).schema_arrow
    assert parquet_schema.names == EXPECTED_FINAL_COLUMNS
    assert parquet_schema.field("INFO").nullable
    assert parquet_schema.field("EXTRA").nullable


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


def test_persist_chunked_results_finalizes_binary_output_with_nullable_extra(tmp_path: Path) -> None:
    output_run = output.prepare_output_run(
        output_root=tmp_path / "binary_output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        resume=False,
    )
    accumulators = iter(
        [
            create_regenie_chunk_accumulator(
                chunk_identifier=7,
                variant_stop_index=8,
                variant_identifier="variant1",
                chromosome="22",
                allele_zero="G",
                allele_one="A",
                extra_code=jnp.asarray([1], dtype=jnp.int32),
            ),
            create_regenie_chunk_accumulator(
                chunk_identifier=8,
                variant_stop_index=9,
                variant_identifier="variant2",
                chromosome="22",
                allele_zero="G",
                allele_one="A",
                extra_code=jnp.asarray([0], dtype=jnp.int32),
            ),
        ]
    )

    parquet_path = output.persist_chunked_results(
        frame_iterator=accumulators,
        output_run_paths=output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_BINARY,
        finalize_parquet=True,
        writer_thread_count=1,
        payload_batch_size=2,
    )

    assert parquet_path is not None
    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
    assert parquet_frame.get_column("EXTRA").to_list() == ["FIRTH", None]
