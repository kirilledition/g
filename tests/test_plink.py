from pathlib import Path

import polars as pl
import pytest

from g.io.plink import VARIANT_TABLE_COLUMNS, load_variant_table


def test_load_variant_table(tmp_path: Path) -> None:
    """Ensure load_variant_table parses a valid BIM file correctly."""
    bed_prefix = tmp_path / "test_dataset"
    bim_path = bed_prefix.with_suffix(".bim")

    bim_content = (
        "1\tvariant1\t0.0\t1000\tA\tC\n"
        "1\tvariant2\t0.0\t2000\tG\tT\n"
        "2\tvariant3\t0.0\t3000\tC\tG\n"
    )
    bim_path.write_text(bim_content)

    df = load_variant_table(bed_prefix)

    assert df.height == 3
    assert df.columns == list(VARIANT_TABLE_COLUMNS)
    assert df.get_column("chromosome").to_list() == [1, 1, 2]
    assert df.get_column("variant_identifier").to_list() == ["variant1", "variant2", "variant3"]
    assert df.get_column("genetic_distance").to_list() == [0.0, 0.0, 0.0]
    assert df.get_column("position").to_list() == [1000, 2000, 3000]
    assert df.get_column("allele_one").to_list() == ["A", "G", "C"]
    assert df.get_column("allele_two").to_list() == ["C", "T", "G"]


def test_load_variant_table_empty_file(tmp_path: Path) -> None:
    """Ensure load_variant_table raises NoDataError for an empty BIM file."""
    bed_prefix = tmp_path / "test_dataset"
    bim_path = bed_prefix.with_suffix(".bim")
    bim_path.write_text("")

    with pytest.raises(pl.exceptions.NoDataError):
        load_variant_table(bed_prefix)


def test_load_variant_table_missing_file(tmp_path: Path) -> None:
    """Ensure load_variant_table raises FileNotFoundError for missing files."""
    bed_prefix = tmp_path / "test_dataset"

    with pytest.raises(FileNotFoundError):
        load_variant_table(bed_prefix)
