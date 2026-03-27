from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from g.io.plink import (
    build_genotype_chunk,
    preprocess_genotype_matrix,
    read_bed_chunk,
    validate_bed_sample_order,
)
from g.models import PreprocessedGenotypeChunkData


def test_preprocess_genotype_matrix_no_missing() -> None:
    """Test preprocessing without missing values."""
    genotype_matrix = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )

    result = preprocess_genotype_matrix(genotype_matrix)

    assert result.has_missing_values is False
    assert result.genotypes.shape == (3, 3)
    np.testing.assert_allclose(result.allele_one_frequency, jnp.array([0.5, 0.5, 0.5]))
    np.testing.assert_array_equal(result.observation_count, jnp.array([3, 3, 3]))


def test_preprocess_genotype_matrix_with_missing() -> None:
    """Test preprocessing with missing values (mean imputation)."""
    genotype_matrix = jnp.array(
        [
            [0.0, jnp.nan, 2.0],
            [1.0, 1.0, jnp.nan],
            [jnp.nan, 2.0, 1.0],
        ]
    )

    result = preprocess_genotype_matrix(genotype_matrix)

    assert result.has_missing_values is True
    assert result.genotypes.shape == (3, 3)

    expected_freq_0 = 0.5 / 2.0
    expected_freq_1 = 1.5 / 2.0
    expected_freq_2 = 1.5 / 2.0
    np.testing.assert_allclose(
        result.allele_one_frequency,
        jnp.array([expected_freq_0, expected_freq_1, expected_freq_2]),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(result.observation_count, jnp.array([2, 2, 2]))


def test_preprocess_genotype_matrix_all_missing() -> None:
    """Test preprocessing when all values are missing for a variant."""
    genotype_matrix = jnp.array(
        [
            [jnp.nan, 1.0],
            [jnp.nan, 2.0],
            [jnp.nan, 0.0],
        ]
    )

    result = preprocess_genotype_matrix(genotype_matrix)

    np.testing.assert_allclose(result.allele_one_frequency[0], 0.0)
    np.testing.assert_array_equal(result.observation_count[0], 0)


def test_build_genotype_chunk() -> None:
    """Test building a genotype chunk from preprocessed data."""
    preprocessed_data = PreprocessedGenotypeChunkData(
        genotypes=jnp.array([[0.0, 1.0], [1.0, 2.0]]),
        missing_mask=jnp.array([[False, False], [False, False]]),
        has_missing_values=False,
        allele_one_frequency=jnp.array([0.25, 0.75]),
        observation_count=jnp.array([2, 2]),
    )

    chunk = build_genotype_chunk(
        preprocessed_chunk_data=preprocessed_data,
        chromosome_values=np.array(["1", "1", "2", "2"]),
        variant_identifier_values=np.array(["var1", "var2", "var3", "var4"]),
        position_values=np.array([100, 200, 300, 400]),
        allele_one_values=np.array(["A", "C", "G", "T"]),
        allele_two_values=np.array(["G", "T", "A", "C"]),
        variant_start=1,
        variant_stop=3,
    )

    assert chunk.genotypes.shape == (2, 2)
    assert chunk.metadata.chromosome.tolist() == ["1", "2"]
    assert chunk.metadata.variant_identifiers.tolist() == ["var2", "var3"]
    assert chunk.metadata.position.tolist() == [200, 300]
    assert chunk.metadata.allele_one.tolist() == ["C", "G"]
    assert chunk.metadata.allele_two.tolist() == ["T", "A"]


def test_read_bed_chunk() -> None:
    """Test read_bed_chunk wraps read_bed_chunk_host and device_put."""
    mock_bed_handle = Mock()
    mock_bed_handle.cloud_options = {}
    mock_bed_handle.iid_count = 3
    mock_bed_handle.sid_count = 10
    mock_bed_handle.count_A1 = True

    bed_path = Path("test.bed")
    sample_indices = np.array([0, 1, 2], dtype=np.intp)

    with patch("g.io.plink.read_bed_chunk_host") as mock_host:
        mock_host.return_value = np.ones((3, 5), dtype=np.float64)

        result = read_bed_chunk(
            bed_handle=mock_bed_handle,
            bed_path=bed_path,
            sample_index_array=sample_indices,
            variant_start=0,
            variant_stop=5,
            num_threads=4,
        )

        assert result.shape == (3, 5)
        mock_host.assert_called_once()


def test_validate_bed_sample_order_matches(tmp_path: Path) -> None:
    """Test validate_bed_sample_order passes when order matches."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\nf1\ts2\t0\t0\t2\t-9\nf1\ts3\t0\t0\t1\t-9\n")

    sample_index_array = np.array([0, 1, 2], dtype=np.intp)
    expected_identifiers = np.array(["s1", "s2", "s3"])

    validate_bed_sample_order(tmp_path / "test", sample_index_array, expected_identifiers)


def test_validate_bed_sample_order_mismatch(tmp_path: Path) -> None:
    """Test validate_bed_sample_order raises when order mismatches."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\nf1\ts2\t0\t0\t2\t-9\nf1\ts3\t0\t0\t1\t-9\n")

    sample_index_array = np.array([0, 1, 2], dtype=np.intp)
    expected_identifiers = np.array(["s3", "s2", "s1"])

    with pytest.raises(ValueError, match="BED sample order does not match"):
        validate_bed_sample_order(tmp_path / "test", sample_index_array, expected_identifiers)
