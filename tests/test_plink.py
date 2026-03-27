from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from g.io.plink import VARIANT_TABLE_COLUMNS, load_variant_table, read_bed_chunk_host


def test_read_bed_chunk_host():
    """Test reading a BED chunk into a host NumPy array."""
    mock_bed_handle = Mock()
    mock_bed_handle.cloud_options = {"option": "value"}
    mock_bed_handle.iid_count = 100
    mock_bed_handle.sid_count = 500
    mock_bed_handle.count_A1 = True

    bed_path = Path("fake/path/to/file.bed")
    sample_index_array = np.array([0, 1, 2, 5, 10], dtype=np.intp)
    variant_start = 10
    variant_stop = 20
    num_threads = 4

    with patch("g.io.plink.read_f64") as mock_read_f64:
        # Define a side effect to populate the array
        def side_effect(
            path, cloud_options, iid_count, sid_count, is_a1_counted, iid_index, sid_index, val, num_threads
        ):
            # Populate val with ones to simulate reading
            val.fill(1.0)

        mock_read_f64.side_effect = side_effect

        result = read_bed_chunk_host(
            bed_handle=mock_bed_handle,
            bed_path=bed_path,
            sample_index_array=sample_index_array,
            variant_start=variant_start,
            variant_stop=variant_stop,
            num_threads=num_threads,
        )

        # Verify the shape
        expected_shape = (5, 10)  # 5 samples, 10 variants (20 - 10)
        assert result.shape == expected_shape
        assert result.dtype == np.float64

        # Verify side effect works
        assert np.all(result == 1.0)

        # Verify read_f64 was called correctly
        mock_read_f64.assert_called_once()
        args, kwargs = mock_read_f64.call_args

        assert args[0] == "fake/path/to/file.bed"
        assert args[1] == {"option": "value"}
        assert kwargs["iid_count"] == 100
        assert kwargs["sid_count"] == 500
        assert kwargs["is_a1_counted"] is True
        np.testing.assert_array_equal(kwargs["iid_index"], sample_index_array)
        np.testing.assert_array_equal(kwargs["sid_index"], np.arange(10, 20, dtype=np.intp))
        assert kwargs["val"].shape == expected_shape
        assert kwargs["num_threads"] == 4


def test_load_variant_table(tmp_path: Path) -> None:
    """Ensure load_variant_table parses a valid BIM file correctly."""
    bed_prefix = tmp_path / "test_dataset"
    bim_path = bed_prefix.with_suffix(".bim")

    bim_content = "1\tvariant1\t0.0\t1000\tA\tC\n1\tvariant2\t0.0\t2000\tG\tT\n2\tvariant3\t0.0\t3000\tC\tG\n"
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
