from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from g import _core

if TYPE_CHECKING:
    from pathlib import Path


def build_test_bed_prefix(tmp_path: Path) -> Path:
    """Create a tiny PLINK BED/FAM pair for native-extension tests."""
    bed_prefix = tmp_path / "tiny_dataset"
    family_path = bed_prefix.with_suffix(".fam")
    family_path.write_text("f1\ts1\t0\t0\t1\t-9\nf1\ts2\t0\t0\t1\t-9\nf1\ts3\t0\t0\t1\t-9\nf1\ts4\t0\t0\t1\t-9\n")

    bed_path = bed_prefix.with_suffix(".bed")
    bed_path.write_bytes(bytes([0x6C, 0x1B, 0x01, 0x4B, 0x2C]))
    return bed_prefix


def test_hello_from_bin_returns_expected_message() -> None:
    """Ensure the extension module exports a simple health-check string."""
    assert _core.hello_from_bin() == "Hello from g!"


def test_read_bed_chunk_f64_decodes_expected_values(tmp_path: Path) -> None:
    """Ensure the Rust BED reader decodes packed genotypes correctly."""
    bed_prefix = build_test_bed_prefix(tmp_path)

    native_chunk = _core.read_bed_chunk_f64(
        bed_path=bed_prefix.with_suffix(".bed"),
        sample_indices=[3, 1, 0],
        variant_start=0,
        variant_stop=2,
    )

    result_matrix = np.frombuffer(native_chunk.genotype_values_le, dtype=np.float64).reshape((3, 2), order="C")
    expected_matrix = np.array(
        [
            [np.nan, 2.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ]
    )

    assert native_chunk.sample_count == 3
    assert native_chunk.variant_count == 2
    np.testing.assert_allclose(result_matrix, expected_matrix, atol=0.0, equal_nan=True)


def test_read_bed_chunk_f64_rejects_invalid_variant_range(tmp_path: Path) -> None:
    """Ensure the Rust BED reader rejects empty requested ranges."""
    bed_prefix = build_test_bed_prefix(tmp_path)

    with pytest.raises(ValueError, match="must contain at least one variant"):
        _core.read_bed_chunk_f64(
            bed_path=bed_prefix.with_suffix(".bed"),
            sample_indices=[0, 1],
            variant_start=1,
            variant_stop=1,
        )


def test_read_bed_chunk_f64_rejects_out_of_range_sample_index(tmp_path: Path) -> None:
    """Ensure the Rust BED reader validates sample indices against the FAM size."""
    bed_prefix = build_test_bed_prefix(tmp_path)

    with pytest.raises(ValueError, match="sample index is out of range"):
        _core.read_bed_chunk_f64(
            bed_path=bed_prefix.with_suffix(".bed"),
            sample_indices=[4],
            variant_start=0,
            variant_stop=1,
        )


def test_read_bed_chunk_f64_rejects_invalid_header(tmp_path: Path) -> None:
    """Ensure the Rust BED reader rejects non-PLINK headers."""
    bed_prefix = build_test_bed_prefix(tmp_path)
    bed_prefix.with_suffix(".bed").write_bytes(bytes([0x00, 0x00, 0x00, 0x4B]))

    with pytest.raises(ValueError, match="expected SNP-major PLINK header"):
        _core.read_bed_chunk_f64(
            bed_path=bed_prefix.with_suffix(".bed"),
            sample_indices=[0, 1],
            variant_start=0,
            variant_stop=1,
        )


def test_preprocess_genotype_matrix_f64_returns_expected_buffers() -> None:
    """Ensure native preprocessing exposes correct imputed values and summaries."""
    genotype_matrix = np.array(
        [
            [0.0, np.nan],
            [1.0, 2.0],
            [np.nan, 0.0],
        ],
        dtype=np.float64,
    )

    native_result = _core.preprocess_genotype_matrix_f64(genotype_matrix)

    imputed_matrix = np.frombuffer(memoryview(native_result.imputed_genotype_values), dtype=np.float64).reshape(
        (native_result.sample_count, native_result.variant_count),
        order="C",
    )
    missing_mask = np.frombuffer(memoryview(native_result.missing_mask_values), dtype=np.uint8).reshape(
        (native_result.sample_count, native_result.variant_count),
        order="C",
    )
    allele_one_frequency = np.frombuffer(memoryview(native_result.allele_one_frequency_values), dtype=np.float64)
    observation_count = np.frombuffer(memoryview(native_result.observation_count_values), dtype=np.int64)

    np.testing.assert_allclose(imputed_matrix, np.array([[0.0, 1.0], [1.0, 2.0], [0.5, 0.0]]), atol=0.0)
    np.testing.assert_array_equal(missing_mask, np.array([[0, 1], [0, 0], [1, 0]], dtype=np.uint8))
    np.testing.assert_allclose(allele_one_frequency, np.array([0.25, 0.5]), atol=0.0)
    np.testing.assert_array_equal(observation_count, np.array([2, 2], dtype=np.int64))


def test_preprocess_genotype_matrix_f64_rejects_one_dimensional_input() -> None:
    """Ensure native preprocessing requires a two-dimensional matrix."""
    with pytest.raises(ValueError, match="two-dimensional"):
        _core.preprocess_genotype_matrix_f64(np.array([0.0, 1.0, 2.0], dtype=np.float64))


def test_preprocess_genotype_matrix_f64_rejects_non_contiguous_input() -> None:
    """Ensure native preprocessing requires C-contiguous float64 input."""
    genotype_matrix = np.arange(12, dtype=np.float64).reshape((3, 4))[:, ::2]

    with pytest.raises(ValueError, match="C-contiguous float64"):
        _core.preprocess_genotype_matrix_f64(genotype_matrix)
