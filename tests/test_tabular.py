from __future__ import annotations

import typing

import numpy as np
import pytest

from g import _core

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
    sample_key_mode: str = "iid",
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
        sample_key_mode=sample_key_mode,
    )


def align_multi_sample_data(
    sample_indices: np.ndarray,
    family_identifiers: list[str],
    individual_identifiers: list[str],
    phenotype_path: Path,
    phenotype_names: list[str],
    covariate_path: Path | None,
    covariate_names: list[str] | None,
    *,
    is_binary_trait: bool,
    sample_key_mode: str = "iid",
) -> _core.NativeMultiAlignedSampleData:
    """Run the native multi-phenotype alignment helper with test-friendly arguments."""
    return _core.align_multi_sample_data(
        np.ascontiguousarray(sample_indices, dtype=np.int64),
        family_identifiers,
        individual_identifiers,
        str(phenotype_path),
        phenotype_names,
        str(covariate_path) if covariate_path is not None else None,
        covariate_names,
        is_binary_trait,
        sample_key_mode=sample_key_mode,
    )


def align_grouped_sample_data(
    sample_indices: np.ndarray,
    family_identifiers: list[str],
    individual_identifiers: list[str],
    phenotype_path: Path,
    phenotype_names: list[str],
    covariate_path: Path | None,
    covariate_names: list[str] | None,
    *,
    is_binary_trait: bool,
    sample_key_mode: str = "iid",
) -> _core.NativeGroupedAlignedSampleData:
    """Run the native grouped per-phenotype alignment helper with test-friendly arguments."""
    return _core.align_grouped_sample_data(
        np.ascontiguousarray(sample_indices, dtype=np.int64),
        family_identifiers,
        individual_identifiers,
        str(phenotype_path),
        phenotype_names,
        str(covariate_path) if covariate_path is not None else None,
        covariate_names,
        is_binary_trait,
        sample_key_mode=sample_key_mode,
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


def test_native_alignment_streaming_parser_handles_crlf_and_quoted_tsv(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_bytes(b'FID\tIID\ttrait\r\nf1\t"s1"\t"1.5"\r\nf2\t"s2"\t"2.5"\r\n')
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_bytes(b'FID\tIID\tage\r\nf1\t"s1"\t"25"\r\nf2\t"s2"\t"30"\r\n')

    result = align_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["s1", "s2"],
        ["s1", "s2"],
        phenotype_path,
        "trait",
        covariate_path,
        ["age"],
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(result.sample_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(result.phenotype_vector, np.asarray([1.5, 2.5], dtype=np.float32))
    np.testing.assert_allclose(result.covariate_matrix, np.asarray([[1.0, 25.0], [1.0, 30.0]], dtype=np.float32))


def test_native_multi_alignment_is_explicit_complete_case_trait_major_matrix(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait_a\ttrait_b\nf1\ts1\t1.0\tNA\nf2\ts2\t2.0\t20.0\nf3\ts3\t3.0\t30.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts2\t30\nf3\ts3\t35\n")

    single_trait_result = align_sample_data(
        np.asarray([0, 1, 2], dtype=np.int64),
        ["f1", "f2", "f3"],
        ["s1", "s2", "s3"],
        phenotype_path,
        "trait_a",
        covariate_path,
        ["age"],
        is_binary_trait=False,
        sample_key_mode="fid_iid",
    )
    result = align_multi_sample_data(
        np.asarray([0, 1, 2], dtype=np.int64),
        ["f1", "f2", "f3"],
        ["s1", "s2", "s3"],
        phenotype_path,
        ["trait_a", "trait_b"],
        covariate_path,
        ["age"],
        is_binary_trait=False,
        sample_key_mode="fid_iid",
    )

    np.testing.assert_array_equal(single_trait_result.sample_indices, np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(result.sample_indices, np.asarray([1, 2], dtype=np.int64))
    assert result.phenotype_names == ["trait_a", "trait_b"]
    assert result.family_identifiers == ["f2", "f3"]
    assert result.individual_identifiers == ["s2", "s3"]
    np.testing.assert_allclose(
        result.phenotype_matrix,
        np.asarray([[2.0, 3.0], [20.0, 30.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.covariate_matrix,
        np.asarray([[1.0, 30.0], [1.0, 35.0]], dtype=np.float32),
    )


def test_native_grouped_alignment_batches_traits_with_identical_sample_sets(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text(
        "FID\tIID\ttrait_a\ttrait_b\ttrait_c\nf1\ts1\t1.0\t10.0\t100.0\nf2\ts2\t2.0\t20.0\tNA\nf3\ts3\tNA\tNA\t300.0\n"
    )
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts2\t30\nf3\ts3\t35\n")

    result = align_grouped_sample_data(
        np.asarray([2, 0, 1], dtype=np.int64),
        ["f3", "f1", "f2"],
        ["s3", "s1", "s2"],
        phenotype_path,
        ["trait_a", "trait_b", "trait_c"],
        covariate_path,
        ["age"],
        is_binary_trait=False,
        sample_key_mode="fid_iid",
    )

    assert len(result.groups) == 2
    first_group = result.groups[0]
    second_group = result.groups[1]
    assert first_group.phenotype_indices == [0, 1]
    assert first_group.aligned_sample_data.phenotype_names == ["trait_a", "trait_b"]
    np.testing.assert_array_equal(first_group.aligned_sample_data.sample_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(
        first_group.aligned_sample_data.phenotype_matrix,
        np.asarray([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        first_group.aligned_sample_data.covariate_matrix,
        np.asarray([[1.0, 25.0], [1.0, 30.0]], dtype=np.float32),
    )
    assert second_group.phenotype_indices == [2]
    assert second_group.aligned_sample_data.phenotype_names == ["trait_c"]
    np.testing.assert_array_equal(second_group.aligned_sample_data.sample_indices, np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_allclose(
        second_group.aligned_sample_data.phenotype_matrix,
        np.asarray([[100.0, 300.0]], dtype=np.float32),
    )


def test_native_multi_alignment_binary_encoding_applies_to_all_traits(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\tcase_a\tcase_b\nf1\ts1\t1\t2\nf2\ts2\t2\t1\n")

    result = align_multi_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["s1", "s2"],
        ["s1", "s2"],
        phenotype_path,
        ["case_a", "case_b"],
        None,
        None,
        is_binary_trait=True,
    )

    np.testing.assert_allclose(
        result.phenotype_matrix,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert result.is_binary_trait is True


def test_native_multi_fid_iid_alignment_allows_repeated_iid_across_families(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait_a\ttrait_b\nf1\ts1\t1.0\t10.0\nf2\ts1\t2.0\t20.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts1\t35\n")

    result = align_multi_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["f2", "f1"],
        ["s1", "s1"],
        phenotype_path,
        ["trait_a", "trait_b"],
        covariate_path,
        ["age"],
        is_binary_trait=False,
        sample_key_mode="fid_iid",
    )

    np.testing.assert_array_equal(result.sample_indices, np.asarray([0, 1], dtype=np.int64))
    assert result.family_identifiers == ["f2", "f1"]
    np.testing.assert_allclose(result.phenotype_matrix, np.asarray([[2.0, 1.0], [20.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(result.covariate_matrix, np.asarray([[1.0, 35.0], [1.0, 25.0]], dtype=np.float32))


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

    native_aligned_sample_data = _core.align_sample_data_from_sample_file(
        str(sample_path),
        3,
        str(phenotype_path),
        "trait",
        covariate_path=str(covariate_path),
        covariate_names=None,
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(native_aligned_sample_data.sample_indices, np.asarray([1, 2], dtype=np.int64))
    assert native_aligned_sample_data.family_identifiers == ["f1", "f3"]
    assert native_aligned_sample_data.individual_identifiers == ["s1", "s3"]
    np.testing.assert_allclose(
        native_aligned_sample_data.covariate_matrix,
        np.asarray([[1.0, 25.0, 1.0], [1.0, 35.0, 1.0]], dtype=np.float32),
    )


def test_iid_alignment_rejects_duplicate_bgen_iid_by_default(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\n")

    with pytest.raises(ValueError, match=r"Duplicate IID 's1'.*BGEN/sample identifiers"):
        align_sample_data(
            np.asarray([0, 1], dtype=np.int64),
            ["f1", "f2"],
            ["s1", "s1"],
            phenotype_path,
            "trait",
            None,
            None,
            is_binary_trait=False,
        )


def test_iid_alignment_rejects_duplicate_phenotype_iid_by_default(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\nf2\ts1\t2.0\n")

    with pytest.raises(ValueError, match=r"Duplicate IID 's1'.*phenotype table"):
        align_sample_data(
            np.asarray([0], dtype=np.int64),
            ["f1"],
            ["s1"],
            phenotype_path,
            "trait",
            None,
            None,
            is_binary_trait=False,
        )


def test_iid_alignment_rejects_duplicate_covariate_iid_by_default(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts1\t26\n")

    with pytest.raises(ValueError, match=r"Duplicate IID 's1'.*covariate table"):
        align_sample_data(
            np.asarray([0], dtype=np.int64),
            ["f1"],
            ["s1"],
            phenotype_path,
            "trait",
            covariate_path,
            ["age"],
            is_binary_trait=False,
        )


def test_iid_alignment_rejects_duplicate_phenotype_iid_before_missing_filter(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\tNA\nf2\ts1\t2.0\n")

    with pytest.raises(ValueError, match=r"Duplicate IID 's1'.*phenotype table"):
        align_sample_data(
            np.asarray([0], dtype=np.int64),
            ["f1"],
            ["s1"],
            phenotype_path,
            "trait",
            None,
            None,
            is_binary_trait=False,
        )


def test_iid_alignment_rejects_duplicate_covariate_iid_before_missing_filter(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\tNA\nf2\ts1\t26\n")

    with pytest.raises(ValueError, match=r"Duplicate IID 's1'.*covariate table"):
        align_sample_data(
            np.asarray([0], dtype=np.int64),
            ["f1"],
            ["s1"],
            phenotype_path,
            "trait",
            covariate_path,
            ["age"],
            is_binary_trait=False,
        )


def test_fid_iid_alignment_allows_repeated_iid_across_families(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\nf2\ts1\t2.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts1\t35\n")

    result = align_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["f2", "f1"],
        ["s1", "s1"],
        phenotype_path,
        "trait",
        covariate_path,
        ["age"],
        is_binary_trait=False,
        sample_key_mode="fid_iid",
    )

    np.testing.assert_array_equal(result.sample_indices, np.asarray([0, 1], dtype=np.int64))
    assert result.family_identifiers == ["f2", "f1"]
    np.testing.assert_allclose(result.phenotype_vector, np.asarray([2.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(result.covariate_matrix, np.asarray([[1.0, 35.0], [1.0, 25.0]], dtype=np.float32))


def test_fid_iid_alignment_requires_fid_columns(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("IID\ttrait\ns1\t1.0\n")

    with pytest.raises(ValueError, match="Identifier column 'FID'"):
        align_sample_data(
            np.asarray([0], dtype=np.int64),
            ["f1"],
            ["s1"],
            phenotype_path,
            "trait",
            None,
            None,
            is_binary_trait=False,
            sample_key_mode="fid_iid",
        )
