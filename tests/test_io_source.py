from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from g.io.source import (
    GenotypeSourceConfig,
    build_bgen_source_config,
    build_genotype_source_signature_paths,
    build_plink_source_config,
    iter_genotype_chunks_from_source,
    iter_linear_genotype_chunks_from_source,
    load_aligned_sample_data_from_source,
    resolve_genotype_source_config,
    validate_genotype_source_config,
)
from g.models import GenotypeChunk, VariantMetadata
from g.types import ArrayMemoryOrder, GenotypeSourceFormat, SampleIdentifierSource

if typing.TYPE_CHECKING:
    import polars as pl

    from g.io.reader import GenotypeReader, VariantTableArrays


class FakeSourceReader:
    """Minimal protocol-compatible reader used in source tests."""

    sample_identifier_source = SampleIdentifierSource.EXTERNAL

    @property
    def sample_count(self) -> int:
        return 2

    @property
    def variant_count(self) -> int:
        return 1

    @property
    def samples(self) -> np.ndarray:
        return np.array(["sample0", "sample1"], dtype=np.str_)

    @property
    def variant_table(self) -> pl.DataFrame:
        raise AssertionError

    def get_variant_table_arrays(self, variant_start: int, variant_stop: int) -> VariantTableArrays:
        raise AssertionError

    def read(
        self,
        index: object = None,
        dtype: type[np.float32] | type[np.float64] = np.float32,
        order: ArrayMemoryOrder = ArrayMemoryOrder.C_CONTIGUOUS,
    ) -> np.ndarray:
        raise AssertionError

    def close(self) -> None:
        return

    def __enter__(self) -> FakeSourceReader:
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        return


def build_chunk(variant_start_index: int) -> GenotypeChunk:
    """Build a small chunk fixture for source tests."""
    return GenotypeChunk(
        genotypes=jnp.array([[0.0], [1.0]]),
        missing_mask=jnp.array([[False], [False]]),
        has_missing_values=False,
        metadata=VariantMetadata(
            variant_start_index=variant_start_index,
            variant_stop_index=variant_start_index + 1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array([f"variant{variant_start_index}"]),
            position=np.array([100 + variant_start_index], dtype=np.int64),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25], dtype=jnp.float32),
        observation_count=jnp.array([2], dtype=jnp.int32),
    )


def test_resolve_genotype_source_config_requires_exactly_one_source() -> None:
    """Ensure the public source resolver rejects ambiguous inputs."""
    with pytest.raises(ValueError, match="Exactly one genotype source"):
        resolve_genotype_source_config(None, None)
    with pytest.raises(ValueError, match="Exactly one genotype source"):
        resolve_genotype_source_config("dataset", "dataset.bgen")


def test_build_genotype_source_signature_paths_supports_both_formats() -> None:
    """Ensure reproducibility signatures include the right source files."""
    plink_paths = build_genotype_source_signature_paths(build_plink_source_config(Path("dataset")))
    bgen_paths = build_genotype_source_signature_paths(build_bgen_source_config(Path("dataset.bgen")))

    assert plink_paths == (Path("dataset.bed"), Path("dataset.bim"), Path("dataset.fam"))
    assert bgen_paths == (Path("dataset.bgen"),)


def test_iter_genotype_chunks_from_source_dispatches_to_bgen_reader() -> None:
    """Ensure the shared source iterator dispatches through the BGEN backend."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"))
    expected_chunk = build_chunk(0)

    with patch("g.io.source.iter_bgen_genotype_chunks", return_value=iter([expected_chunk])) as mock_iter_bgen:
        chunks = list(
            iter_genotype_chunks_from_source(
                genotype_source_config=bgen_source_config,
                sample_indices=np.array([0, 1], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample0", "sample1"]),
                chunk_size=64,
                variant_limit=1,
            )
        )

    assert [chunk.metadata.variant_identifiers.tolist() for chunk in chunks] == [["variant0"]]
    mock_iter_bgen.assert_called_once()


def test_iter_genotype_chunks_from_source_dispatches_to_plink_reader() -> None:
    """Ensure the shared source iterator dispatches through the PLINK backend."""
    plink_source_config = build_plink_source_config(Path("study"))
    expected_chunk = build_chunk(0)

    with patch("g.io.source.iter_plink_genotype_chunks", return_value=iter([expected_chunk])) as mock_iter_plink:
        chunks = list(
            iter_genotype_chunks_from_source(
                genotype_source_config=plink_source_config,
                sample_indices=np.array([0, 1], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample0", "sample1"]),
                chunk_size=64,
                variant_limit=1,
            )
        )

    assert [chunk.metadata.variant_identifiers.tolist() for chunk in chunks] == [["variant0"]]
    mock_iter_plink.assert_called_once()


def test_validate_genotype_source_config_rejects_unknown_format() -> None:
    """Ensure unsupported source formats fail fast."""
    with pytest.raises(ValueError, match="Unsupported genotype source format"):
        validate_genotype_source_config(
            GenotypeSourceConfig(
                source_format=typing.cast("GenotypeSourceFormat", "vcf"),
                source_path=Path("study.vcf"),
            )
        )


def test_load_aligned_sample_data_from_source_dispatches_to_plink_loader() -> None:
    """Ensure sample loading uses the PLINK backend for PLINK configs."""
    plink_source_config = build_plink_source_config(Path("study"))
    expected_aligned_sample_data = object()

    with patch("g.io.source.load_aligned_sample_data", return_value=expected_aligned_sample_data) as mock_load:
        aligned_sample_data = load_aligned_sample_data_from_source(
            genotype_source_config=plink_source_config,
            phenotype_path=Path("pheno.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covar.tsv"),
            covariate_names=("age",),
            is_binary_trait=False,
        )

    assert aligned_sample_data is expected_aligned_sample_data
    mock_load.assert_called_once()


def test_load_aligned_sample_data_from_source_dispatches_to_bgen_loader() -> None:
    """Ensure sample loading uses embedded BGEN sample identifiers for BGEN configs."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"))
    sample_table = object()
    expected_aligned_sample_data = object()

    with (
        patch("g.io.source.load_bgen_sample_table", return_value=sample_table) as mock_load_bgen_sample_table,
        patch(
            "g.io.source.load_aligned_sample_data_from_individual_identifier_table",
            return_value=expected_aligned_sample_data,
        ) as mock_load_from_sample_table,
    ):
        aligned_sample_data = load_aligned_sample_data_from_source(
            genotype_source_config=bgen_source_config,
            phenotype_path=Path("pheno.tsv"),
            phenotype_name="trait",
            covariate_path=None,
            covariate_names=None,
            is_binary_trait=True,
        )

    assert aligned_sample_data is expected_aligned_sample_data
    mock_load_bgen_sample_table.assert_called_once_with(Path("study.bgen"), None)
    mock_load_from_sample_table.assert_called_once()


def test_load_aligned_sample_data_from_source_reuses_open_bgen_reader() -> None:
    """Ensure BGEN sample alignment can reuse an already-open reader."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"))
    genotype_reader = typing.cast("GenotypeReader", FakeSourceReader())
    expected_aligned_sample_data = object()

    with (
        patch("g.io.source.load_bgen_sample_table") as mock_load_bgen_sample_table,
        patch(
            "g.io.source.load_aligned_sample_data_from_individual_identifier_table",
            return_value=expected_aligned_sample_data,
        ) as mock_load_from_sample_table,
    ):
        aligned_sample_data = load_aligned_sample_data_from_source(
            genotype_source_config=bgen_source_config,
            phenotype_path=Path("pheno.tsv"),
            phenotype_name="trait",
            covariate_path=None,
            covariate_names=None,
            is_binary_trait=True,
            genotype_reader=genotype_reader,
        )

    assert aligned_sample_data is expected_aligned_sample_data
    mock_load_bgen_sample_table.assert_not_called()
    sample_table = mock_load_from_sample_table.call_args.kwargs["sample_table"]
    assert sample_table.get_column("individual_identifier").to_list() == ["sample0", "sample1"]


def test_build_bgen_source_config_preserves_sample_path() -> None:
    """Ensure BGEN source configs keep the optional sample-file path."""
    genotype_source_config = build_bgen_source_config(Path("study.bgen"), sample_path=Path("study.sample"))

    assert genotype_source_config.sample_path == Path("study.sample")


def test_resolve_genotype_source_config_rejects_sample_for_plink() -> None:
    """Ensure explicit sample files are only accepted for BGEN configs."""
    with pytest.raises(ValueError, match="can only be provided together with `bgen`"):
        resolve_genotype_source_config("dataset", None, "dataset.sample")


def test_iter_linear_genotype_chunks_from_source_dispatches_to_plink_reader() -> None:
    """Ensure the linear source iterator dispatches through the PLINK backend."""
    plink_source_config = build_plink_source_config(Path("study"))

    with patch("g.io.source.iter_plink_linear_genotype_chunks", return_value=iter(())) as mock_iter_plink:
        linear_chunks = list(
            iter_linear_genotype_chunks_from_source(
                genotype_source_config=plink_source_config,
                sample_indices=np.array([0, 1], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample0", "sample1"]),
                chunk_size=64,
                variant_limit=1,
            )
        )

    assert linear_chunks == []
    mock_iter_plink.assert_called_once()


def test_iter_linear_genotype_chunks_from_source_dispatches_to_bgen_reader() -> None:
    """Ensure the linear source iterator dispatches through the BGEN backend."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"))

    with patch("g.io.source.iter_bgen_linear_genotype_chunks", return_value=iter(())) as mock_iter_bgen:
        linear_chunks = list(
            iter_linear_genotype_chunks_from_source(
                genotype_source_config=bgen_source_config,
                sample_indices=np.array([0, 1], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample0", "sample1"]),
                chunk_size=64,
                variant_limit=1,
            )
        )

    assert linear_chunks == []
    mock_iter_bgen.assert_called_once()


def test_iter_linear_genotype_chunks_from_source_reuses_open_reader() -> None:
    """Ensure source iteration can reuse one already-open genotype reader."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"))
    genotype_reader = typing.cast("GenotypeReader", FakeSourceReader())

    with patch("g.io.source.iter_linear_genotype_chunks_from_reader", return_value=iter(())) as mock_iter_reader:
        linear_chunks = list(
            iter_linear_genotype_chunks_from_source(
                genotype_source_config=bgen_source_config,
                sample_indices=np.array([0, 1], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample0", "sample1"]),
                chunk_size=64,
                variant_limit=1,
                genotype_reader=genotype_reader,
            )
        )

    assert linear_chunks == []
    mock_iter_reader.assert_called_once()
    assert mock_iter_reader.call_args.kwargs["genotype_reader"] is genotype_reader
    assert mock_iter_reader.call_args.kwargs["source_name"] == "BGEN"
    np.testing.assert_array_equal(mock_iter_reader.call_args.kwargs["sample_indices"], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        mock_iter_reader.call_args.kwargs["expected_individual_identifiers"],
        np.array(["sample0", "sample1"]),
    )
    assert mock_iter_reader.call_args.kwargs["chunk_size"] == 64
    assert mock_iter_reader.call_args.kwargs["variant_limit"] == 1
    assert mock_iter_reader.call_args.kwargs["validate_sample_order_flag"] is True
