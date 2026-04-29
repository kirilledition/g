"""Tests for output persistence and schema helpers."""

from __future__ import annotations

import typing

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.engine import Regenie2LinearChunkAccumulator, Regenie2LinearChunkPayload
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
from g.models import Regenie2LinearChunkResult, VariantMetadata
from g.types import AssociationMode

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


def test_get_output_schema_returns_regenie_columns() -> None:
    output_schema = get_output_schema(AssociationMode.REGENIE2_LINEAR)
    assert "chi_squared" in output_schema
    assert "log10_p_value" in output_schema
    assert "is_valid" in output_schema


def test_get_output_schema_rejects_non_regenie_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported association mode"):
        get_output_schema(typing.cast("AssociationMode", "linear"))


def test_cast_frame_to_schema_reorders_and_casts_columns() -> None:
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
    )

    chunk_paths = sorted(prepared_output_run.output_run_paths.chunks_directory.glob("chunk_*.arrow"))
    assert [path.name for path in chunk_paths] == ["chunk_000000000.arrow", "chunk_000000001.arrow"]


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
