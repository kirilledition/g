from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.engine import LinearChunkAccumulator, LinearChunkPayload
from g.io.source import build_plink_source_config
from g.models import LinearAssociationChunkResult, VariantMetadata
from g.output.chunked import (
    OutputRunConfiguration,
    build_chunk_file_name,
    build_output_run_configuration,
    calculate_file_checksum,
    finalize_chunks_to_parquet,
    persist_chunked_results,
    prepare_output_run,
    resolve_output_run_paths,
    write_chunk_file,
)
from g.output.manifest import ManifestChunkRecord, OutputManifest
from g.output.schema import get_output_schema

if TYPE_CHECKING:
    from pathlib import Path


def create_input_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    bed_prefix = tmp_path / "study"
    bed_prefix.with_suffix(".bed").write_bytes(b"bed")
    bed_prefix.with_suffix(".bim").write_text("1\tvariant1\t0\t123\tA\tC\n", encoding="utf-8")
    bed_prefix.with_suffix(".fam").write_text("family individual 0 0 1 1\n", encoding="utf-8")
    phenotype_path = tmp_path / "phenotype.txt"
    phenotype_path.write_text("family individual phenotype\nfamily individual 1\n", encoding="utf-8")
    covariate_path = tmp_path / "covariate.txt"
    covariate_path.write_text("family individual age\nfamily individual 40\n", encoding="utf-8")
    return bed_prefix, phenotype_path, covariate_path


def create_linear_output_run_configuration(
    tmp_path: Path,
    *,
    chunk_size: int = 2,
) -> tuple[Path, OutputRunConfiguration]:
    bed_prefix, phenotype_path, covariate_path = create_input_files(tmp_path)
    output_root = tmp_path / "results"
    output_run_configuration = build_output_run_configuration(
        association_mode="linear",
        genotype_source_config=build_plink_source_config(bed_prefix),
        phenotype_path=phenotype_path,
        phenotype_name="phenotype",
        covariate_path=covariate_path,
        covariate_names=("age",),
        chunk_size=chunk_size,
        variant_limit=4,
    )
    return output_root, output_run_configuration


def create_linear_chunk_payload(
    *,
    chunk_identifier: int,
    variant_stop_index: int,
    variant_identifier: str,
) -> LinearChunkPayload:
    return LinearChunkPayload(
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


def register_committed_chunk(
    output_run_paths: Path,
    chunk_payload: LinearChunkPayload,
) -> None:
    chunk_file_name = build_chunk_file_name(chunk_payload)
    chunk_file_path = output_run_paths / "chunks" / chunk_file_name
    chunk_file_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_checksum = write_chunk_file(chunk_payload, chunk_file_path, "linear")
    manifest = OutputManifest(output_run_paths / "manifest.sqlite")
    manifest.insert_committed_chunk(
        ManifestChunkRecord(
            chunk_identifier=chunk_payload.chunk_identifier,
            chunk_label=f"linear_chunk_{chunk_payload.chunk_identifier:06d}_{chunk_payload.variant_stop_index:06d}",
            variant_start_index=chunk_payload.variant_start_index,
            variant_stop_index=chunk_payload.variant_stop_index,
            row_count=1,
            file_path=f"chunks/{chunk_file_name}",
            checksum_sha256=chunk_checksum,
            status="committed",
        )
    )
    manifest.close()


def test_prepare_output_run_uses_mode_suffix(tmp_path: Path) -> None:
    output_run_paths = resolve_output_run_paths(tmp_path / "results/output", "linear")
    assert output_run_paths.run_directory == tmp_path / "results/output.linear.run"
    assert output_run_paths.chunks_directory == tmp_path / "results/output.linear.run/chunks"


def test_prepare_output_run_rejects_configuration_mismatch(tmp_path: Path) -> None:
    output_root, output_run_configuration = create_linear_output_run_configuration(tmp_path, chunk_size=2)
    prepared_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=False,
    )
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_run_record(prepared_output_run.run_manifest_record)
    manifest.close()

    mismatched_output_run_configuration = build_output_run_configuration(
        association_mode="linear",
        genotype_source_config=build_plink_source_config(tmp_path / "study"),
        phenotype_path=tmp_path / "phenotype.txt",
        phenotype_name="phenotype",
        covariate_path=tmp_path / "covariate.txt",
        covariate_names=("age",),
        chunk_size=4,
        variant_limit=4,
    )
    with pytest.raises(ValueError, match="does not match the requested configuration"):
        prepare_output_run(
            output_root=output_root,
            output_run_configuration=mismatched_output_run_configuration,
            resume=True,
        )


def test_prepare_output_run_resumes_only_valid_chunks(tmp_path: Path) -> None:
    output_root, output_run_configuration = create_linear_output_run_configuration(tmp_path)
    prepared_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=False,
    )
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_run_record(prepared_output_run.run_manifest_record)
    manifest.close()

    valid_chunk_payload = create_linear_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="variant0",
    )
    invalid_chunk_payload = create_linear_chunk_payload(
        chunk_identifier=2,
        variant_stop_index=4,
        variant_identifier="variant2",
    )
    register_committed_chunk(
        prepared_output_run.output_run_paths.run_directory,
        valid_chunk_payload,
    )
    invalid_chunk_file_name = build_chunk_file_name(invalid_chunk_payload)
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_committed_chunk(
        ManifestChunkRecord(
            chunk_identifier=invalid_chunk_payload.chunk_identifier,
            chunk_label="linear_chunk_000002_000004",
            variant_start_index=invalid_chunk_payload.variant_start_index,
            variant_stop_index=invalid_chunk_payload.variant_stop_index,
            row_count=1,
            file_path=f"chunks/{invalid_chunk_file_name}",
            checksum_sha256="incorrect-checksum",
            status="committed",
        )
    )
    manifest.close()

    resumed_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=True,
    )
    assert resumed_output_run.committed_chunk_identifiers == frozenset({0})


def test_persist_chunked_results_preserves_chunk_identifier_on_resume(tmp_path: Path) -> None:
    output_root, output_run_configuration = create_linear_output_run_configuration(tmp_path)
    prepared_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=False,
    )
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_run_record(prepared_output_run.run_manifest_record)
    manifest.close()
    register_committed_chunk(
        prepared_output_run.output_run_paths.run_directory,
        create_linear_chunk_payload(
            chunk_identifier=0,
            variant_stop_index=2,
            variant_identifier="variant0",
        ),
    )

    resumed_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=True,
    )
    persisted_output_run_paths = persist_chunked_results(
        frame_iterator=iter(
            [
                create_linear_chunk_accumulator(
                    chunk_identifier=2,
                    variant_stop_index=4,
                    variant_identifier="variant2",
                )
            ]
        ),
        prepared_output_run=resumed_output_run,
        resume=True,
    )

    manifest = OutputManifest(persisted_output_run_paths.manifest_path)
    committed_chunk_records = manifest.load_committed_chunk_records()
    manifest.close()
    assert [chunk_record.chunk_identifier for chunk_record in committed_chunk_records] == [0, 2]
    assert [chunk_record.file_path for chunk_record in committed_chunk_records] == [
        "chunks/000000_000002.arrow",
        "chunks/000002_000004.arrow",
    ]


def test_finalize_chunks_to_parquet_raises_for_missing_chunks(tmp_path: Path) -> None:
    output_root, output_run_configuration = create_linear_output_run_configuration(tmp_path)
    prepared_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=False,
    )
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_run_record(prepared_output_run.run_manifest_record)
    manifest.insert_committed_chunk(
        ManifestChunkRecord(
            chunk_identifier=0,
            chunk_label="linear_chunk_000000_000002",
            variant_start_index=0,
            variant_stop_index=2,
            row_count=1,
            file_path="chunks/000000_000002.arrow",
            checksum_sha256="missing",
            status="committed",
        )
    )
    manifest.close()

    with pytest.raises(ValueError, match="incomplete or corrupted"):
        finalize_chunks_to_parquet(prepared_output_run.output_run_paths)


def test_finalize_chunks_to_parquet_raises_for_checksum_mismatch(tmp_path: Path) -> None:
    output_root, output_run_configuration = create_linear_output_run_configuration(tmp_path)
    prepared_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=False,
    )
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_run_record(prepared_output_run.run_manifest_record)
    manifest.close()

    chunk_payload = create_linear_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="variant0",
    )
    register_committed_chunk(
        prepared_output_run.output_run_paths.run_directory,
        chunk_payload,
    )

    chunk_file_path = prepared_output_run.output_run_paths.chunks_directory / build_chunk_file_name(chunk_payload)
    chunk_file_path.write_bytes(b"corrupted-arrow-bytes")

    with pytest.raises(ValueError, match="checksum mismatch"):
        finalize_chunks_to_parquet(prepared_output_run.output_run_paths)


def test_finalize_chunks_to_parquet_compacts_valid_rows(tmp_path: Path) -> None:
    output_root, output_run_configuration = create_linear_output_run_configuration(tmp_path)
    prepared_output_run = prepare_output_run(
        output_root=output_root,
        output_run_configuration=output_run_configuration,
        resume=False,
    )
    manifest = OutputManifest(prepared_output_run.output_run_paths.manifest_path)
    manifest.insert_run_record(prepared_output_run.run_manifest_record)
    manifest.close()

    first_chunk_payload = create_linear_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="variant0",
    )
    second_chunk_payload = create_linear_chunk_payload(
        chunk_identifier=2,
        variant_stop_index=4,
        variant_identifier="variant2",
    )
    register_committed_chunk(
        prepared_output_run.output_run_paths.run_directory,
        first_chunk_payload,
    )
    register_committed_chunk(
        prepared_output_run.output_run_paths.run_directory,
        second_chunk_payload,
    )

    final_parquet_path = finalize_chunks_to_parquet(prepared_output_run.output_run_paths)
    parquet_frame = pl.read_parquet(final_parquet_path)
    assert parquet_frame.height == 2
    assert parquet_frame.schema == get_output_schema("linear")
    assert parquet_frame.get_column("chunk_identifier").to_list() == [0, 2]


def test_write_chunk_file_produces_stable_checksum(tmp_path: Path) -> None:
    chunk_payload = create_linear_chunk_payload(
        chunk_identifier=0,
        variant_stop_index=2,
        variant_identifier="variant0",
    )
    chunk_file_path = tmp_path / "000000_000002.arrow"
    manifest_checksum = write_chunk_file(chunk_payload, chunk_file_path, "linear")
    assert manifest_checksum == calculate_file_checksum(chunk_file_path)
