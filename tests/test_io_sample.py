from __future__ import annotations

import typing

import pytest

from g import _core

if typing.TYPE_CHECKING:
    from pathlib import Path


def align_from_sample_file(sample_path: Path, phenotype_path: Path) -> _core.NativeAlignedSampleData:
    return _core.align_sample_data_from_sample_file(
        str(sample_path),
        2,
        str(phenotype_path),
        "trait",
        is_binary_trait=False,
    )


def test_native_sample_file_alignment_with_single_identifier_column(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID   missing\n0   0\nalpha   0\nbeta   0\n", encoding="utf-8")
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nalpha\talpha\t1.0\nbeta\tbeta\t2.0\n", encoding="utf-8")

    native_aligned_sample_data = align_from_sample_file(sample_path, phenotype_path)

    assert native_aligned_sample_data.family_identifiers == ["alpha", "beta"]
    assert native_aligned_sample_data.individual_identifiers == ["alpha", "beta"]


def test_native_sample_file_alignment_prefers_id_2_column(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text(
        "ID_1 ID_2 missing sex\n0 0 0 D\nfam1 ind1 0 F\nfam2 ind2 0 M\n",
        encoding="utf-8",
    )
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nfam1\tind1\t1.0\nfam2\tind2\t2.0\n", encoding="utf-8")

    native_aligned_sample_data = align_from_sample_file(sample_path, phenotype_path)

    assert native_aligned_sample_data.family_identifiers == ["fam1", "fam2"]
    assert native_aligned_sample_data.individual_identifiers == ["ind1", "ind2"]


def test_native_sample_file_alignment_accepts_mixed_whitespace(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text(
        "ID_1\tID_2   missing\tsex\n0\t0  0\tD\nfam1\tind1   0\tF\nfam2   ind2\t0 M\n",
        encoding="utf-8",
    )
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nfam1\tind1\t1.0\nfam2\tind2\t2.0\n", encoding="utf-8")

    native_aligned_sample_data = align_from_sample_file(sample_path, phenotype_path)

    assert native_aligned_sample_data.family_identifiers == ["fam1", "fam2"]
    assert native_aligned_sample_data.individual_identifiers == ["ind1", "ind2"]


def test_native_sample_file_alignment_rejects_invalid_identifier_type(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID missing\nD 0\nalpha 0\n", encoding="utf-8")
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nalpha\talpha\t1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must mark the first identifier column with type '0'"):
        _core.align_sample_data_from_sample_file(str(sample_path), 1, str(phenotype_path), "trait")


def test_native_sample_file_alignment_rejects_ragged_rows(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID missing\n0 0\nalpha\n", encoding="utf-8")
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nalpha\talpha\t1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has 1 values, but the header declares 2 columns"):
        _core.align_sample_data_from_sample_file(str(sample_path), 1, str(phenotype_path), "trait")
