from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from g.io.bgen import (
    build_bgen_variant_table,
    build_bgen_variant_table_arrays,
    build_core_variant_metadata,
    convert_probability_matrix_to_dosage,
    convert_probability_tensor_to_dosage,
    iter_dosage_genotype_chunks,
    iter_genotype_chunks,
    load_backend_core,
    load_bgen_sample_table,
    open_bgen,
    read_bgen_chunk,
    read_bgen_chunk_host,
    validate_bgen_sample_order,
)
from g.types import ArrayMemoryOrder, SampleIdentifierSource

TEST_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"
COMPLEX_BGEN_PATH = TEST_DATA_DIRECTORY / "complex.23bits.no.samples.bgen"


def write_sample_file(sample_path: Path, sample_identifiers: list[str]) -> None:
    sample_lines = ["ID", "0"]
    sample_lines.extend(sample_identifier for sample_identifier in sample_identifiers)
    sample_path.write_text("\n".join(sample_lines) + "\n", encoding="utf-8")


def test_convert_probability_tensor_to_dosage_for_unphased_layout() -> None:
    probability_tensor = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )

    dosage_matrix = convert_probability_tensor_to_dosage(
        probability_tensor=probability_tensor,
        combination_count=3,
        is_phased=False,
        dtype=np.float32,
        order=ArrayMemoryOrder.C_CONTIGUOUS,
    )

    expected_dosage_matrix = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    np.testing.assert_allclose(dosage_matrix, expected_dosage_matrix, equal_nan=True)


def test_convert_probability_tensor_to_dosage_for_phased_layout() -> None:
    probability_tensor = np.array(
        [
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    dosage_matrix = convert_probability_tensor_to_dosage(
        probability_tensor=probability_tensor,
        combination_count=4,
        is_phased=True,
        dtype=np.float32,
        order=ArrayMemoryOrder.C_CONTIGUOUS,
    )

    expected_dosage_matrix = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(dosage_matrix, expected_dosage_matrix)


def test_convert_probability_matrix_to_dosage_for_unphased_layout() -> None:
    probability_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    dosage_vector = convert_probability_matrix_to_dosage(
        probability_matrix,
        combination_count=3,
        is_phased=False,
    )

    np.testing.assert_allclose(dosage_vector, np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_convert_probability_matrix_to_dosage_for_phased_layout() -> None:
    probability_matrix = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    dosage_vector = convert_probability_matrix_to_dosage(
        probability_matrix,
        combination_count=4,
        is_phased=True,
    )

    np.testing.assert_allclose(dosage_vector, np.array([0.0, 1.0, 1.0], dtype=np.float32))


def test_convert_probability_tensor_to_dosage_rejects_unsupported_layout() -> None:
    probability_tensor = np.zeros((2, 2, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported BGEN probability layout"):
        convert_probability_tensor_to_dosage(
            probability_tensor=probability_tensor,
            combination_count=5,
            is_phased=False,
            dtype=np.float32,
            order=ArrayMemoryOrder.C_CONTIGUOUS,
        )


def test_load_backend_core_reports_missing_dependency() -> None:
    with (
        patch(
            "g.io.bgen.importlib.import_module",
            side_effect=ModuleNotFoundError("missing"),
        ),
        pytest.raises(ModuleNotFoundError, match="Rust core helpers are unavailable"),
    ):
        load_backend_core()


def test_build_bgen_variant_table_counts_last_allele() -> None:
    mock_bgen_handle = SimpleNamespace(
        allele_ids=np.array(["A,G", "C,T"], dtype=np.str_),
        ids=np.array(["variant_1", "variant_2"], dtype=np.str_),
        rsids=np.array(["rs1", ""], dtype=np.str_),
        chromosomes=np.array(["1", "2"], dtype=np.str_),
        positions=np.array([123, 456], dtype=np.int64),
        nvariants=2,
    )

    variant_table = build_bgen_variant_table(mock_bgen_handle)

    assert variant_table.get_column("variant_identifier").to_list() == ["rs1", "variant_2"]
    assert variant_table.get_column("allele_one").to_list() == ["G", "T"]
    assert variant_table.get_column("allele_two").to_list() == ["A", "C"]


def test_build_bgen_variant_table_arrays_counts_last_allele() -> None:
    mock_bgen_handle = SimpleNamespace(
        allele_ids=np.array(["A,G", "C,T"], dtype=np.str_),
        ids=np.array(["variant_1", "variant_2"], dtype=np.str_),
        rsids=np.array(["rs1", ""], dtype=np.str_),
        chromosomes=np.array(["1", "2"], dtype=np.str_),
        positions=np.array([123, 456], dtype=np.int64),
    )

    variant_table_arrays = build_bgen_variant_table_arrays(mock_bgen_handle)

    assert variant_table_arrays.variant_identifier_values.tolist() == ["rs1", "variant_2"]
    assert variant_table_arrays.allele_one_values.tolist() == ["G", "T"]
    assert variant_table_arrays.allele_two_values.tolist() == ["A", "C"]


def test_open_bgen_reads_phased_haplotype_example_as_dosage() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with open_bgen(bgen_path) as bgen_reader:
        assert bgen_reader.sample_count == 4
        assert bgen_reader.variant_count == 4
        genotype_matrix = bgen_reader.read(dtype=np.float32, order=ArrayMemoryOrder.C_CONTIGUOUS)

    expected_genotype_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(genotype_matrix, expected_genotype_matrix)


def test_open_bgen_direct_subset_read_matches_probability_tensor_conversion() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_indices = np.array([0, 2], dtype=np.intp)
    variant_slice = slice(1, 4)

    with open_bgen(bgen_path) as bgen_reader:
        direct_dosage_matrix = bgen_reader.read(
            index=(sample_indices, variant_slice),
            dtype=np.float32,
            order=ArrayMemoryOrder.C_CONTIGUOUS,
        )
    expected_dosage_matrix = np.array(
        [
            [1.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(direct_dosage_matrix, expected_dosage_matrix)


def test_open_bgen_read_float32_matches_read() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_indices = np.array([0, 2, 3], dtype=np.intp)
    variant_start = 1
    variant_stop = 4

    with open_bgen(bgen_path) as bgen_reader:
        strict_dosage_matrix = bgen_reader.read_float32(sample_indices, variant_start, variant_stop)
        compatibility_dosage_matrix = bgen_reader.read(
            index=(sample_indices, slice(variant_start, variant_stop)),
            dtype=np.float32,
            order=ArrayMemoryOrder.C_CONTIGUOUS,
        )

    np.testing.assert_allclose(strict_dosage_matrix, compatibility_dosage_matrix)


def test_open_bgen_read_float32_into_fills_reusable_output_array() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_indices = np.array([0, 2, 3], dtype=np.intp)
    variant_start = 1
    variant_stop = 4

    with open_bgen(bgen_path) as bgen_reader:
        output_array = np.empty((3, 3), dtype=np.float32, order="C")
        filled_output_array = bgen_reader.read_float32_into(output_array, sample_indices, variant_start, variant_stop)
        compatibility_dosage_matrix = bgen_reader.read(
            index=(sample_indices, slice(variant_start, variant_stop)),
            dtype=np.float32,
            order=ArrayMemoryOrder.C_CONTIGUOUS,
        )

    assert filled_output_array is output_array
    np.testing.assert_allclose(output_array, compatibility_dosage_matrix)


def test_open_bgen_read_float32_rejects_invalid_variant_bounds() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with open_bgen(bgen_path) as bgen_reader, pytest.raises(ValueError, match="Variant bounds must satisfy"):
        _ = bgen_reader.read_float32(np.array([0, 1], dtype=np.intp), 3, 2)


def test_open_bgen_read_float32_into_rejects_non_contiguous_output_array() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with open_bgen(bgen_path) as bgen_reader, pytest.raises(ValueError, match="C-contiguous"):
        output_array = np.empty((3, 3), dtype=np.float32, order="F")
        _ = bgen_reader.read_float32_into(output_array, np.array([0, 1, 2], dtype=np.intp), 1, 4)


def test_open_bgen_split_variant_slice_by_chromosome_returns_whole_chunk_for_single_chromosome() -> None:
    with open_bgen(HAPLOTYPES_BGEN_PATH) as bgen_reader:
        chromosome_slices = bgen_reader.split_variant_slice_by_chromosome(0, bgen_reader.variant_count)

    assert chromosome_slices == ((0, 4),)


def test_open_bgen_float32_read_uses_native_core_reader_for_contiguous_variant_slice() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with (
        open_bgen(bgen_path) as bgen_reader,
        patch.object(
            bgen_reader,
            "read_float32",
            wraps=bgen_reader.read_float32,
        ) as mock_read_float32,
    ):
        dosage_matrix = bgen_reader.read(
            index=(np.array([0, 1], dtype=np.intp), slice(0, 2)),
            dtype=np.float32,
            order=ArrayMemoryOrder.C_CONTIGUOUS,
        )

    assert dosage_matrix.shape == (2, 2)
    assert mock_read_float32.call_count == 1


def test_open_bgen_defers_variant_table_materialization() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with (
        patch(
            "g.io.bgen.build_variant_table_from_core_metadata",
            return_value=pl.DataFrame(),
        ) as mock_build_variant_table,
        open_bgen(bgen_path) as bgen_reader,
    ):
        assert mock_build_variant_table.call_count == 0
        _ = bgen_reader.variant_table
        assert mock_build_variant_table.call_count == 1


def test_bgen_variant_slice_metadata_does_not_materialize_full_variant_table() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with (
        patch(
            "g.io.bgen.build_variant_table_from_core_metadata",
            return_value=pl.DataFrame(),
        ) as mock_build_variant_table,
        open_bgen(bgen_path) as bgen_reader,
    ):
        variant_table_arrays = bgen_reader.get_variant_table_arrays(0, 2)

    assert mock_build_variant_table.call_count == 0
    assert variant_table_arrays.variant_identifier_values.shape == (2,)


def test_bgen_variant_slice_metadata_caches_full_array_build() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with (
        patch(
            "g.io.bgen.build_core_variant_metadata",
            wraps=build_core_variant_metadata,
        ) as mock_build_core_variant_metadata,
        open_bgen(bgen_path) as bgen_reader,
    ):
        first_slice = bgen_reader.get_variant_table_arrays(0, 2)
        second_slice = bgen_reader.get_variant_table_arrays(2, 4)

    assert mock_build_core_variant_metadata.call_count == 1
    assert first_slice.variant_identifier_values.shape == (2,)
    assert second_slice.variant_identifier_values.shape == (2,)


def test_open_bgen_uses_external_sample_file(tmp_path: Path) -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_path = tmp_path / "custom_ids.sample"
    custom_identifiers = ["person_a", "person_b", "person_c", "person_d"]
    write_sample_file(sample_path, custom_identifiers)

    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        assert bgen_reader.sample_identifier_source == SampleIdentifierSource.EXTERNAL
        assert bgen_reader.samples.tolist() == custom_identifiers


def test_open_bgen_uses_id_2_values_from_oxford_sample_file(tmp_path: Path) -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_path = tmp_path / "oxford.sample"
    sample_path.write_text(
        "ID_1 ID_2 missing sex\n0 0 0 D\n0 person_a 0 1\n0 person_b 0 2\n0 person_c 0 1\n0 person_d 0 2\n",
        encoding="utf-8",
    )

    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        assert bgen_reader.samples.tolist() == ["person_a", "person_b", "person_c", "person_d"]


def test_load_bgen_sample_table_uses_embedded_samples() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    sample_table = load_bgen_sample_table(bgen_path)

    assert sample_table.get_column("individual_identifier").to_list() == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
    ]


def test_load_bgen_sample_table_uses_external_sample_file(tmp_path: Path) -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_path = tmp_path / "custom_ids.sample"
    custom_identifiers = ["person_a", "person_b", "person_c", "person_d"]
    write_sample_file(sample_path, custom_identifiers)

    sample_table = load_bgen_sample_table(bgen_path, sample_path)

    assert sample_table.get_column("family_identifier").to_list() == custom_identifiers
    assert sample_table.get_column("individual_identifier").to_list() == custom_identifiers


def test_load_bgen_sample_table_rejects_sample_count_mismatch(tmp_path: Path) -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_path = tmp_path / "mismatch.sample"
    write_sample_file(sample_path, ["person_a", "person_b", "person_c"])

    with pytest.raises(ValueError, match="Expect number of samples in file to match"):
        load_bgen_sample_table(bgen_path, sample_path)


def test_load_bgen_sample_table_requires_sample_file_when_identifiers_are_missing() -> None:
    bgen_path = COMPLEX_BGEN_PATH

    with pytest.raises(ValueError, match="Only diploid biallelic BGEN variants are supported"):
        load_bgen_sample_table(bgen_path)


def test_validate_bgen_sample_order_failure() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_indices = np.arange(4, dtype=np.intp)
    expected_identifiers = np.array(["wrong1", "wrong2", "wrong3", "wrong4"], dtype=np.str_)

    with pytest.raises(ValueError, match="BGEN sample order does not match"), open_bgen(bgen_path) as bgen_reader:
        validate_bgen_sample_order(bgen_reader, sample_indices, expected_identifiers, bgen_path)


def test_validate_bgen_sample_order_requires_real_identifiers() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with (
        open_bgen(bgen_path) as bgen_reader,
        pytest.raises(
            ValueError,
            match=r"does not contain samples and no \.sample file was found",
        ),
    ):
        bgen_reader.sample_identifier_source = SampleIdentifierSource.GENERATED
        validate_bgen_sample_order(
            bgen_reader,
            np.arange(2, dtype=np.intp),
            np.array(["sample_1", "sample_2"], dtype=np.str_),
            Path("study.bgen"),
        )


def test_read_bgen_chunk_helpers_return_expected_shapes() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH

    with open_bgen(bgen_path) as bgen_reader:
        host_chunk = read_bgen_chunk_host(
            bgen_reader=bgen_reader,
            sample_index_array=np.arange(4, dtype=np.intp),
            variant_start=0,
            variant_stop=2,
        )
        device_chunk = read_bgen_chunk(
            bgen_reader=bgen_reader,
            sample_index_array=np.arange(4, dtype=np.intp),
            variant_start=0,
            variant_stop=2,
        )

    assert host_chunk.shape == (4, 2)
    assert device_chunk.shape == (4, 2)


def test_bgen_reader_rejects_unsupported_complex_layout() -> None:
    bgen_path = COMPLEX_BGEN_PATH

    with pytest.raises(ValueError, match="Only diploid biallelic BGEN variants are supported"):
        open_bgen(bgen_path)


class FakeGeneratedSampleReader:
    """Context manager returning a reader without real sample identifiers."""

    sample_identifier_source = SampleIdentifierSource.GENERATED

    def __enter__(self) -> FakeGeneratedSampleReader:
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        return


def test_iter_genotype_chunks_requires_real_sample_identifiers() -> None:
    with (
        patch(
            "g.io.bgen.open_bgen",
            return_value=FakeGeneratedSampleReader(),
        ),
        pytest.raises(ValueError, match=r"does not contain samples and no \.sample file was found"),
    ):
        list(
            iter_genotype_chunks(
                bgen_path=Path("study.bgen"),
                sample_indices=np.arange(2, dtype=np.int64),
                expected_individual_identifiers=np.array(["sample_1", "sample_2"], dtype=np.str_),
                chunk_size=1,
            )
        )


def test_iter_dosage_genotype_chunks_requires_real_sample_identifiers() -> None:
    with (
        patch(
            "g.io.bgen.open_bgen",
            return_value=FakeGeneratedSampleReader(),
        ),
        pytest.raises(ValueError, match=r"does not contain samples and no \.sample file was found"),
    ):
        list(
            iter_dosage_genotype_chunks(
                bgen_path=Path("study.bgen"),
                sample_indices=np.arange(2, dtype=np.int64),
                expected_individual_identifiers=np.array(["sample_1", "sample_2"], dtype=np.str_),
                chunk_size=1,
            )
        )


def test_iter_genotype_chunks() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_indices = np.arange(4, dtype=np.int64)
    expected_ids = np.array([f"sample_{sample_index}" for sample_index in range(4)], dtype=np.str_)

    chunks = list(
        iter_genotype_chunks(
            bgen_path=bgen_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_ids,
            chunk_size=2,
            variant_limit=4,
        )
    )

    assert len(chunks) == 2
    assert chunks[0].genotypes.shape == (4, 2)
    assert chunks[0].metadata.allele_one.tolist() == ["G", "G"]
    assert chunks[0].metadata.allele_two.tolist() == ["A", "A"]
    np.testing.assert_allclose(chunks[0].allele_one_frequency, np.array([0.5, 0.5], dtype=np.float32))


def test_iter_dosage_genotype_chunks() -> None:
    bgen_path = HAPLOTYPES_BGEN_PATH
    sample_indices = np.arange(4, dtype=np.int64)
    expected_ids = np.array([f"sample_{sample_index}" for sample_index in range(4)], dtype=np.str_)

    chunks = list(
        iter_dosage_genotype_chunks(
            bgen_path=bgen_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_ids,
            chunk_size=3,
            variant_limit=4,
        )
    )

    assert len(chunks) == 2
    assert chunks[0].genotypes.shape == (4, 3)
    assert chunks[1].genotypes.shape == (4, 1)
