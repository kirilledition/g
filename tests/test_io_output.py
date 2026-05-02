"""Tests for output persistence and schema helpers."""

from __future__ import annotations

import typing

import jax.numpy as jnp
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from g.engine import (
    Regenie2BinaryChunkAccumulator,
    Regenie2BinaryChunkPayload,
    Regenie2LinearChunkAccumulator,
    Regenie2LinearChunkPayload,
)
from g.io.output import (
    LEGACY_REGENIE2_LINEAR_OUTPUT_SCHEMA,
    build_chunk_file_name,
    cast_frame_to_schema,
    finalize_chunks_to_parquet,
    get_output_schema,
    iter_sorted_chunk_file_paths,
    persist_chunked_results,
    prepare_output_run,
    resolve_output_run_paths,
    scan_committed_chunk_identifiers,
    write_chunk_to_disk,
)
from g.models import Regenie2BinaryChunkResult, Regenie2LinearChunkResult, VariantMetadata
from g.types import AssociationMode, OutputWriterBackend

if typing.TYPE_CHECKING:
    from pathlib import Path


def create_regenie_chunk_payload(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
) -> Regenie2LinearChunkPayload:
    return Regenie2LinearChunkPayload(
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
        chi_squared=np.asarray([10.0], dtype=np.float32),
        log10_p_value=np.asarray([5.0], dtype=np.float32),
        is_valid=np.asarray([True], dtype=np.bool_),
    )


def create_regenie_chunk_accumulator(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
) -> Regenie2LinearChunkAccumulator:
    return Regenie2LinearChunkAccumulator(
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
        regenie2_linear_result=Regenie2LinearChunkResult(
            beta=jnp.asarray([0.1], dtype=jnp.float32),
            standard_error=jnp.asarray([0.01], dtype=jnp.float32),
            chi_squared=jnp.asarray([10.0], dtype=jnp.float32),
            log10_p_value=jnp.asarray([5.0], dtype=jnp.float32),
            valid_mask=jnp.asarray([True], dtype=jnp.bool_),
        ),
    )


def create_regenie_binary_chunk_accumulator(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
) -> Regenie2BinaryChunkAccumulator:
    return Regenie2BinaryChunkAccumulator(
        metadata=VariantMetadata(
            variant_start_index=chunk_identifier,
            variant_stop_index=variant_stop_index,
            chromosome=np.asarray(["22"]),
            variant_identifiers=np.asarray([variant_identifier]),
            position=np.asarray([123 + chunk_identifier], dtype=np.int64),
            allele_one=np.asarray(["A"]),
            allele_two=np.asarray(["G"]),
        ),
        allele_one_frequency=jnp.asarray([0.25], dtype=jnp.float32),
        observation_count=jnp.asarray([100], dtype=jnp.int32),
        regenie2_binary_result=Regenie2BinaryChunkResult(
            beta=jnp.asarray([0.1], dtype=jnp.float32),
            standard_error=jnp.asarray([0.2], dtype=jnp.float32),
            chi_squared=jnp.asarray([0.25], dtype=jnp.float32),
            log10_p_value=jnp.asarray([0.5], dtype=jnp.float32),
            extra_code=jnp.asarray([1], dtype=jnp.int32),
            valid_mask=jnp.asarray([True], dtype=jnp.bool_),
        ),
    )


def write_legacy_string_chunk(
    payload: Regenie2LinearChunkPayload,
    chunk_path: Path,
) -> None:
    legacy_output_frame = pl.DataFrame(
        {
            "chunk_identifier": [payload.chunk_identifier],
            "variant_start_index": [payload.variant_start_index],
            "variant_stop_index": [payload.variant_stop_index],
            "chromosome": payload.chromosome.tolist(),
            "position": payload.position.tolist(),
            "variant_identifier": payload.variant_identifier.tolist(),
            "allele_one": payload.allele_one.tolist(),
            "allele_two": payload.allele_two.tolist(),
            "allele_one_frequency": payload.allele_one_frequency.tolist(),
            "observation_count": payload.observation_count.tolist(),
            "beta": payload.beta.tolist(),
            "standard_error": payload.standard_error.tolist(),
            "chi_squared": payload.chi_squared.tolist(),
            "log10_p_value": payload.log10_p_value.tolist(),
            "is_valid": payload.is_valid.tolist(),
        },
        schema=LEGACY_REGENIE2_LINEAR_OUTPUT_SCHEMA,
    )
    legacy_output_frame.write_ipc(chunk_path, compression="zstd")


def test_get_output_schema_returns_regenie_columns() -> None:
    output_schema = get_output_schema(AssociationMode.REGENIE2_LINEAR)
    assert output_schema["chromosome"] == pl.Categorical()
    assert output_schema["variant_identifier"] == pl.String()
    assert output_schema["allele_one"] == pl.Categorical()
    assert output_schema["allele_two"] == pl.Categorical()


def test_get_output_schema_returns_binary_regenie_columns() -> None:
    output_schema = get_output_schema(AssociationMode.REGENIE2_BINARY)
    assert list(output_schema) == [
        "chunk_identifier",
        "variant_start_index",
        "variant_stop_index",
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
    assert output_schema["CHROM"] == pl.Categorical()
    assert output_schema["TEST"] == pl.Categorical()
    assert output_schema["EXTRA"] == pl.Categorical()


def test_binary_payload_writes_regenie_like_schema(tmp_path: Path) -> None:
    payload = Regenie2BinaryChunkPayload(
        chunk_identifier=7,
        variant_start_index=7,
        variant_stop_index=8,
        chromosome=np.asarray(["22"]),
        position=np.asarray([12345], dtype=np.int64),
        variant_identifier=np.asarray(["variant1"]),
        allele_one=np.asarray(["A"]),
        allele_two=np.asarray(["G"]),
        allele_one_frequency=np.asarray([0.25], dtype=np.float32),
        observation_count=np.asarray([100], dtype=np.int32),
        beta=np.asarray([0.1], dtype=np.float32),
        standard_error=np.asarray([0.2], dtype=np.float32),
        chi_squared=np.asarray([0.25], dtype=np.float32),
        log10_p_value=np.asarray([0.5], dtype=np.float32),
        extra_code=np.asarray([1], dtype=np.int32),
        is_valid=np.asarray([True]),
    )

    write_chunk_to_disk(payload, tmp_path, AssociationMode.REGENIE2_BINARY)

    frame = pl.read_ipc(tmp_path / "chunk_000000007.arrow")
    assert frame.columns == list(get_output_schema(AssociationMode.REGENIE2_BINARY))
    assert frame.select("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "TEST", "EXTRA").row(0) == (
        "22",
        12345,
        "variant1",
        "G",
        "A",
        "ADD",
        "FIRTH",
    )
    assert frame.get_column("INFO").to_list() == [1.0]


def test_get_output_schema_rejects_non_regenie_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported association mode"):
        get_output_schema(typing.cast("AssociationMode", "linear"))


def test_cast_frame_to_schema_reorders_and_casts_columns() -> None:
    data_frame = pl.DataFrame(
        {
            "beta": [1.5],
            "chromosome": ["1"],
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
            "chi_squared": [15.0],
            "log10_p_value": [3.0],
            "is_valid": [True],
        }
    )
    cast_data_frame = cast_frame_to_schema(data_frame, AssociationMode.REGENIE2_LINEAR)
    assert cast_data_frame.schema == get_output_schema(AssociationMode.REGENIE2_LINEAR)
    assert cast_data_frame.columns == list(get_output_schema(AssociationMode.REGENIE2_LINEAR))
    assert cast_data_frame.get_column("chromosome").to_list() == ["1"]


def test_resolve_output_run_paths_appends_mode_suffix(tmp_path: Path) -> None:
    output_run_paths = resolve_output_run_paths(tmp_path / "results/output", AssociationMode.REGENIE2_LINEAR)
    assert output_run_paths.run_directory == tmp_path / "results/output.regenie2_linear.run"
    assert output_run_paths.chunks_directory == tmp_path / "results/output.regenie2_linear.run/chunks"


def test_scan_committed_chunk_identifiers_discovers_chunks(tmp_path: Path) -> None:
    (tmp_path / "chunk_000000000.arrow").write_bytes(b"")
    (tmp_path / "chunk_000000512.arrow").write_bytes(b"")
    assert scan_committed_chunk_identifiers(tmp_path) == frozenset({0, 512})


def test_prepare_output_run_rejects_non_empty_directory_without_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "output.regenie2_linear.run"
    run_dir.mkdir(parents=True)
    (run_dir / "stale_file.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=False,
        )


def test_write_chunk_to_disk_produces_readable_arrow_file(tmp_path: Path) -> None:
    payload = create_regenie_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="v0",
    )
    write_chunk_to_disk(payload, tmp_path, AssociationMode.REGENIE2_LINEAR)
    chunk_path = tmp_path / build_chunk_file_name(0)
    assert chunk_path.exists()
    frame = pl.read_ipc(chunk_path)
    assert frame.height == 1
    assert frame.schema == get_output_schema(AssociationMode.REGENIE2_LINEAR)
    assert frame.get_column("chunk_identifier").to_list() == [0]


def test_persist_chunked_results_writes_chunks(tmp_path: Path) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    accumulators = iter(
        [
            create_regenie_chunk_accumulator(
                chunk_identifier=0,
                variant_stop_index=1,
                variant_identifier="v0",
            ),
            create_regenie_chunk_accumulator(
                chunk_identifier=1,
                variant_stop_index=2,
                variant_identifier="v1",
            ),
        ]
    )

    persist_chunked_results(
        frame_iterator=accumulators,
        output_run_paths=prepared_output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_LINEAR,
        payload_batch_size=1,
    )

    chunk_paths = sorted(prepared_output_run.output_run_paths.chunks_directory.glob("chunk_*.arrow"))
    assert [path.name for path in chunk_paths] == ["chunk_000000000.arrow", "chunk_000000001.arrow"]


def test_persist_chunked_results_batches_multiple_payloads_into_one_arrow_file(tmp_path: Path) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    accumulators = iter(
        [
            create_regenie_chunk_accumulator(
                chunk_identifier=0,
                variant_stop_index=1,
                variant_identifier="v0",
            ),
            create_regenie_chunk_accumulator(
                chunk_identifier=1,
                variant_stop_index=2,
                variant_identifier="v1",
            ),
        ]
    )

    persist_chunked_results(
        frame_iterator=accumulators,
        output_run_paths=prepared_output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_LINEAR,
        payload_batch_size=2,
    )

    chunk_paths = tuple(iter_sorted_chunk_file_paths(prepared_output_run.output_run_paths.chunks_directory))
    assert [path.name for path in chunk_paths] == ["chunk_000000000_000000001.arrow"]
    assert scan_committed_chunk_identifiers(prepared_output_run.output_run_paths.chunks_directory) == frozenset({0, 1})


def test_prepare_output_run_resume_detects_old_and_new_arrow_chunks(tmp_path: Path) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    chunks_directory = prepared_output_run.output_run_paths.chunks_directory
    write_chunk_to_disk(
        create_regenie_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=1,
            variant_identifier="v0",
        ),
        chunks_directory,
        AssociationMode.REGENIE2_LINEAR,
    )
    payload_batch = (
        create_regenie_chunk_payload(
            chunk_identifier=1,
            variant_stop_index=2,
            variant_identifier="v1",
        ),
        create_regenie_chunk_payload(
            chunk_identifier=2,
            variant_stop_index=3,
            variant_identifier="v2",
        ),
    )
    write_legacy_string_chunk(payload_batch[0], chunks_directory / build_chunk_file_name(1))
    write_legacy_string_chunk(payload_batch[1], chunks_directory / build_chunk_file_name(2))

    resumed_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    assert resumed_output_run.committed_chunk_identifiers == frozenset({0, 1, 2})


def test_finalize_chunks_to_parquet_writes_expected_schema(tmp_path: Path) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    payload = create_regenie_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="v0",
    )
    write_chunk_to_disk(
        payload,
        prepared_output_run.output_run_paths.chunks_directory,
        AssociationMode.REGENIE2_LINEAR,
    )

    parquet_path = finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
    )

    frame = pl.read_parquet(parquet_path)
    assert frame.height == 1
    assert frame.schema == get_output_schema(AssociationMode.REGENIE2_LINEAR)


def test_finalize_chunks_to_parquet_reads_mixed_old_and_new_chunks_in_sorted_order(tmp_path: Path) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    write_legacy_string_chunk(
        create_regenie_chunk_payload(
            chunk_identifier=2,
            variant_stop_index=3,
            variant_identifier="v2",
        ),
        prepared_output_run.output_run_paths.chunks_directory / build_chunk_file_name(2),
    )
    write_chunk_to_disk(
        create_regenie_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=1,
            variant_identifier="v0",
        ),
        prepared_output_run.output_run_paths.chunks_directory,
        AssociationMode.REGENIE2_LINEAR,
    )
    write_legacy_string_chunk(
        create_regenie_chunk_payload(
            chunk_identifier=1,
            variant_stop_index=2,
            variant_identifier="v1",
        ),
        prepared_output_run.output_run_paths.chunks_directory / build_chunk_file_name(1),
    )

    parquet_path = finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
    )

    frame = pl.read_parquet(parquet_path)
    assert frame.schema == get_output_schema(AssociationMode.REGENIE2_LINEAR)
    assert frame.get_column("chunk_identifier").to_list() == [0, 1, 2]
    assert frame.get_column("variant_identifier").to_list() == ["v0", "v1", "v2"]


def test_finalize_chunks_to_parquet_writes_empty_schema_when_no_chunks_exist(tmp_path: Path) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )

    parquet_path = finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
    )

    frame = pl.read_parquet(parquet_path)
    assert frame.height == 0
    assert frame.schema == get_output_schema(AssociationMode.REGENIE2_LINEAR)


def test_write_chunk_to_disk_replaces_temporary_file_atomically(tmp_path: Path) -> None:
    payload = create_regenie_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="v0",
    )
    write_chunk_to_disk(payload, tmp_path, AssociationMode.REGENIE2_LINEAR)

    assert (tmp_path / build_chunk_file_name(0)).exists()
    assert not (tmp_path / "chunk_000000000.arrow.tmp").exists()


@pytest.mark.parametrize(
    "output_writer_backend",
    [OutputWriterBackend.PYTHON, OutputWriterBackend.RUST],
)
def test_binary_persist_chunked_results_writes_expected_chunk_schema(
    tmp_path: Path,
    output_writer_backend: OutputWriterBackend,
) -> None:
    prepared_output_run = prepare_output_run(
        output_root=tmp_path / f"output_{output_writer_backend}",
        association_mode=AssociationMode.REGENIE2_BINARY,
        resume=False,
    )
    accumulators = iter(
        [
            create_regenie_binary_chunk_accumulator(
                chunk_identifier=7,
                variant_stop_index=8,
                variant_identifier="variant1",
            )
        ]
    )

    persist_chunked_results(
        frame_iterator=accumulators,
        output_run_paths=prepared_output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_writer_backend=output_writer_backend,
        writer_thread_count=1,
        payload_batch_size=1,
    )

    frame = pl.read_ipc(prepared_output_run.output_run_paths.chunks_directory / "chunk_000000007.arrow")
    assert frame.columns == list(get_output_schema(AssociationMode.REGENIE2_BINARY))
    assert frame.select("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "TEST", "EXTRA").row(0) == (
        "22",
        130,
        "variant1",
        "G",
        "A",
        "ADD",
        "FIRTH",
    )


def test_binary_python_and_rust_output_backends_write_matching_parquet(tmp_path: Path) -> None:
    python_output_run = prepare_output_run(
        output_root=tmp_path / "python_output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        resume=False,
    )
    rust_output_run = prepare_output_run(
        output_root=tmp_path / "rust_output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        resume=False,
    )

    python_accumulators = iter(
        [
            create_regenie_binary_chunk_accumulator(
                chunk_identifier=7,
                variant_stop_index=8,
                variant_identifier="variant1",
            )
        ]
    )
    rust_accumulators = iter(
        [
            create_regenie_binary_chunk_accumulator(
                chunk_identifier=7,
                variant_stop_index=8,
                variant_identifier="variant1",
            )
        ]
    )

    python_parquet_path = persist_chunked_results(
        frame_iterator=python_accumulators,
        output_run_paths=python_output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_writer_backend=OutputWriterBackend.PYTHON,
        finalize_parquet=True,
        writer_thread_count=1,
        payload_batch_size=1,
    )
    rust_parquet_path = persist_chunked_results(
        frame_iterator=rust_accumulators,
        output_run_paths=rust_output_run.output_run_paths,
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_writer_backend=OutputWriterBackend.RUST,
        finalize_parquet=True,
        writer_thread_count=1,
        payload_batch_size=1,
    )

    assert python_parquet_path is not None
    assert rust_parquet_path is not None
    assert pl.read_parquet(python_parquet_path).to_dict(as_series=False) == pl.read_parquet(
        rust_parquet_path
    ).to_dict(as_series=False)
    python_parquet_file = pq.ParquetFile(python_parquet_path)
    rust_parquet_file = pq.ParquetFile(rust_parquet_path)
    assert python_parquet_path.stat().st_size == rust_parquet_path.stat().st_size
    assert python_parquet_file.metadata.created_by == rust_parquet_file.metadata.created_by
    assert python_parquet_file.metadata.num_row_groups == rust_parquet_file.metadata.num_row_groups
    assert python_parquet_file.schema_arrow.equals(rust_parquet_file.schema_arrow)
