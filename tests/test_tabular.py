from __future__ import annotations

import typing

import numpy as np
import pytest

from g import _core
from g.io import source

if typing.TYPE_CHECKING:
    from pathlib import Path


def align_sample_data(
    sample_indices: np.ndarray,
    family_identifiers: list[str],
    individual_identifiers: list[str],
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: list[str] | None,
    *,
    is_binary_trait: bool,
) -> _core.NativeAlignedSampleData:
    """Run the native alignment helper with test-friendly arguments."""
    return _core.align_sample_data(
        np.ascontiguousarray(sample_indices, dtype=np.int64),
        family_identifiers,
        individual_identifiers,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        covariate_names,
        is_binary_trait,
    )


def test_native_aligned_sample_data_continuous(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.5\nf2\ts2\t2.5\nf3\ts3\t3.5\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\tsex\nf1\ts1\t25\t1\nf2\ts2\t30\t2\nf3\ts3\t35\t1\n")

    result = align_sample_data(
        np.asarray([0, 1, 2], dtype=np.int64),
        ["s1", "s2", "s3"],
        ["s1", "s2", "s3"],
        phenotype_path,
        "trait",
        covariate_path,
        ["age", "sex"],
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(result.sample_indices, np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(result.phenotype_vector, np.asarray([1.5, 2.5, 3.5], dtype=np.float32))
    np.testing.assert_allclose(
        result.covariate_matrix,
        np.asarray([[1.0, 25.0, 1.0], [1.0, 30.0, 2.0], [1.0, 35.0, 1.0]], dtype=np.float32),
    )
    assert result.covariate_names == ["intercept", "age", "sex"]
    assert result.is_binary_trait is False


def test_native_aligned_sample_data_binary_intercept_only(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1\nf2\ts2\t2\n")

    result = align_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["s1", "s2"],
        ["s1", "s2"],
        phenotype_path,
        "trait",
        None,
        None,
        is_binary_trait=True,
    )

    np.testing.assert_array_equal(result.sample_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(result.phenotype_vector, np.asarray([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(result.covariate_matrix, np.asarray([[1.0], [1.0]], dtype=np.float32))
    assert result.covariate_names == ["intercept"]
    assert result.is_binary_trait is True


def test_native_alignment_matches_iid_only_and_sorts_by_sample_order(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nfamily1\ts3\t3.0\nfamily1\ts1\t1.0\nfamily1\ts2\t2.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\tsex\nfamily1\ts1\t25\t1\nfamily1\ts2\tNA\t2\nfamily1\ts3\t35\t1\n")

    result = align_sample_data(
        np.asarray([0, 1, 2], dtype=np.int64),
        ["s2", "s1", "s3"],
        ["s2", "s1", "s3"],
        phenotype_path,
        "trait",
        covariate_path,
        None,
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(result.sample_indices, np.asarray([1, 2], dtype=np.int64))
    assert result.individual_identifiers == ["s1", "s3"]
    np.testing.assert_allclose(result.phenotype_vector, np.asarray([1.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(
        result.covariate_matrix,
        np.asarray([[1.0, 25.0, 1.0], [1.0, 35.0, 1.0]], dtype=np.float32),
    )


def test_native_alignment_rejects_invalid_binary_values(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1\nf2\ts2\t0\n")

    with pytest.raises(ValueError, match="Binary phenotype"):
        align_sample_data(
            np.asarray([0, 1], dtype=np.int64),
            ["s1", "s2"],
            ["s1", "s2"],
            phenotype_path,
            "trait",
            None,
            None,
            is_binary_trait=True,
        )


def test_native_alignment_rejects_missing_covariate(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\n")

    with pytest.raises(ValueError, match="Covariate columns are missing"):
        align_sample_data(
            np.asarray([0], dtype=np.int64),
            ["s1"],
            ["s1"],
            phenotype_path,
            "trait",
            covariate_path,
            ["age", "sex"],
            is_binary_trait=False,
        )


def test_native_sample_file_alignment_uses_oxford_id_2_column(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID_1 ID_2 missing\n0 0 0\nf2 s2 0\nf1 s1 0\nf3 s3 0\n")
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts3\t3.0\nf1\ts1\t1.0\nf1\ts2\t2.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\tsex\nf1\ts1\t25\t1\nf1\ts2\tNA\t2\nf1\ts3\t35\t1\n")

    sample_table = source.load_sample_identifier_table(sample_path)
    native_aligned_sample_data = _core.align_sample_data_from_sample_file(
        str(sample_path),
        3,
        str(phenotype_path),
        "trait",
        covariate_path=str(covariate_path),
        covariate_names=None,
        is_binary_trait=False,
    )

    assert sample_table.get_column("individual_identifier").to_list() == ["s2", "s1", "s3"]
    np.testing.assert_array_equal(native_aligned_sample_data.sample_indices, np.asarray([1, 2], dtype=np.int64))
    assert native_aligned_sample_data.individual_identifiers == ["s1", "s3"]
    np.testing.assert_allclose(
        native_aligned_sample_data.covariate_matrix,
        np.asarray([[1.0, 25.0, 1.0], [1.0, 35.0, 1.0]], dtype=np.float32),
    )
