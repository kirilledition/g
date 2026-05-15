from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from g.io.source import (
    GenotypeSourceConfig,
    build_bgen_source_config,
    build_genotype_source_signature_paths,
    build_plink_source_config,
    load_aligned_sample_data_from_source,
    resolve_genotype_source_config,
    validate_genotype_source_config,
)
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


def test_load_aligned_sample_data_from_source_uses_explicit_sample_file_with_open_reader() -> None:
    """Ensure explicit BGEN sample files are still honored when a reader is already open."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"), sample_path=Path("study.sample"))
    genotype_reader = typing.cast("GenotypeReader", FakeSourceReader())
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
            genotype_reader=genotype_reader,
        )

    assert aligned_sample_data is expected_aligned_sample_data
    mock_load_bgen_sample_table.assert_called_once_with(Path("study.bgen"), Path("study.sample"))
    mock_load_from_sample_table.assert_called_once_with(
        sample_table=sample_table,
        phenotype_path=Path("pheno.tsv"),
        phenotype_name="trait",
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=True,
    )


def test_build_bgen_source_config_preserves_sample_path() -> None:
    """Ensure BGEN source configs keep the optional sample-file path."""
    genotype_source_config = build_bgen_source_config(Path("study.bgen"), sample_path=Path("study.sample"))

    assert genotype_source_config.sample_path == Path("study.sample")


def test_resolve_genotype_source_config_rejects_sample_for_plink() -> None:
    """Ensure explicit sample files are only accepted for BGEN configs."""
    with pytest.raises(ValueError, match="can only be provided together with `bgen`"):
        resolve_genotype_source_config("dataset", None, "dataset.sample")
