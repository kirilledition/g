from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from g.io.sample import load_sample_identifier_table, resolve_bgen_sample_path

if TYPE_CHECKING:
    from pathlib import Path


def test_load_sample_identifier_table_with_single_identifier_column(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID missing\n0 0\nalpha 0\nbeta 0\n", encoding="utf-8")

    sample_table = load_sample_identifier_table(sample_path)

    assert sample_table.get_column("family_identifier").to_list() == ["alpha", "beta"]
    assert sample_table.get_column("individual_identifier").to_list() == ["alpha", "beta"]


def test_load_sample_identifier_table_prefers_id_2_column(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text(
        "ID_1 ID_2 missing sex\n0 0 0 D\nfam1 ind1 0 F\nfam2 ind2 0 M\n",
        encoding="utf-8",
    )

    sample_table = load_sample_identifier_table(sample_path)

    assert sample_table.get_column("family_identifier").to_list() == ["fam1", "fam2"]
    assert sample_table.get_column("individual_identifier").to_list() == ["ind1", "ind2"]


def test_load_sample_identifier_table_rejects_invalid_identifier_type(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID missing\nD 0\nalpha 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must mark the first identifier column with type '0'"):
        load_sample_identifier_table(sample_path)


def test_load_sample_identifier_table_rejects_ragged_rows(tmp_path: Path) -> None:
    sample_path = tmp_path / "study.sample"
    sample_path.write_text("ID missing\n0 0\nalpha\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has 1 values, but the header declares 2 columns"):
        load_sample_identifier_table(sample_path)


def test_resolve_bgen_sample_path_prefers_explicit_path(tmp_path: Path) -> None:
    bgen_path = tmp_path / "study.bgen"
    explicit_sample_path = tmp_path / "explicit.sample"
    adjacent_sample_path = tmp_path / "study.sample"
    bgen_path.write_text("", encoding="utf-8")
    explicit_sample_path.write_text("", encoding="utf-8")
    adjacent_sample_path.write_text("", encoding="utf-8")

    assert resolve_bgen_sample_path(bgen_path, explicit_sample_path) == explicit_sample_path


def test_resolve_bgen_sample_path_finds_adjacent_sample_file(tmp_path: Path) -> None:
    bgen_path = tmp_path / "study.bgen"
    adjacent_sample_path = tmp_path / "study.sample"
    bgen_path.write_text("", encoding="utf-8")
    adjacent_sample_path.write_text("", encoding="utf-8")

    assert resolve_bgen_sample_path(bgen_path) == adjacent_sample_path


def test_resolve_bgen_sample_path_returns_none_without_match(tmp_path: Path) -> None:
    bgen_path = tmp_path / "study.bgen"
    bgen_path.write_text("", encoding="utf-8")

    assert resolve_bgen_sample_path(bgen_path) is None
