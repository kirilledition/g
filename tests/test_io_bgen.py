from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

if importlib.util.find_spec("cbgen") is None or importlib.util.find_spec("bgen_reader") is None:
    pytest.skip("BGEN dependencies are unavailable in this environment.", allow_module_level=True)

example = importlib.import_module("cbgen").example

from g.io.bgen import (  # noqa: E402
    build_bgen_variant_table,
    convert_probability_tensor_to_dosage,
    iter_genotype_chunks,
    iter_linear_genotype_chunks,
    load_backend_open_bgen,
    load_bgen_sample_table,
    open_bgen,
    read_bgen_chunk,
    read_bgen_chunk_host,
    validate_bgen_sample_order,
)


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
        order="C",
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
        order="C",
    )

    expected_dosage_matrix = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(dosage_matrix, expected_dosage_matrix)


def test_convert_probability_tensor_to_dosage_rejects_unsupported_layout() -> None:
    probability_tensor = np.zeros((2, 2, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported BGEN probability layout"):
        convert_probability_tensor_to_dosage(
            probability_tensor=probability_tensor,
            combination_count=5,
            is_phased=False,
            dtype=np.float32,
            order="C",
        )


def test_load_backend_open_bgen_reports_missing_dependency() -> None:
    with (
        patch(
            "g.io.bgen.importlib.import_module",
            side_effect=ModuleNotFoundError("missing"),
        ),
        pytest.raises(ModuleNotFoundError, match="requires the `bgen-reader` stack"),
    ):
        load_backend_open_bgen()


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


def test_open_bgen_reads_phased_haplotype_example_as_dosage() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))

    with open_bgen(bgen_path) as bgen_reader:
        assert bgen_reader.sample_count == 4
        assert bgen_reader.variant_count == 4
        genotype_matrix = bgen_reader.read(dtype=np.float32, order="C")

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


def test_open_bgen_defers_variant_table_materialization() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))

    with (
        patch("g.io.bgen.build_bgen_variant_table", return_value=pl.DataFrame()) as mock_build_variant_table,
        open_bgen(bgen_path) as bgen_reader,
    ):
        assert mock_build_variant_table.call_count == 0
        _ = bgen_reader.variant_table
        assert mock_build_variant_table.call_count == 1


def test_bgen_variant_slice_metadata_does_not_materialize_full_variant_table() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))

    with (
        patch("g.io.bgen.build_bgen_variant_table", return_value=pl.DataFrame()) as mock_build_variant_table,
        open_bgen(bgen_path) as bgen_reader,
    ):
        variant_table_arrays = bgen_reader.get_variant_table_arrays(0, 2)

    assert mock_build_variant_table.call_count == 0
    assert variant_table_arrays.variant_identifier_values.shape == (2,)


def test_open_bgen_uses_external_sample_file(tmp_path: Path) -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_path = tmp_path / "custom_ids.sample"
    custom_identifiers = ["person_a", "person_b", "person_c", "person_d"]
    write_sample_file(sample_path, custom_identifiers)

    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        assert bgen_reader.sample_identifier_source == "external"
        assert bgen_reader.samples.tolist() == custom_identifiers


def test_open_bgen_uses_id_2_values_from_oxford_sample_file(tmp_path: Path) -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_path = tmp_path / "oxford.sample"
    sample_path.write_text(
        "ID_1 ID_2 missing sex\n0 0 0 D\n0 person_a 0 1\n0 person_b 0 2\n0 person_c 0 1\n0 person_d 0 2\n",
        encoding="utf-8",
    )

    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        assert bgen_reader.samples.tolist() == ["person_a", "person_b", "person_c", "person_d"]


def test_load_bgen_sample_table_uses_embedded_samples() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))

    sample_table = load_bgen_sample_table(bgen_path)

    assert sample_table.get_column("individual_identifier").to_list() == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
    ]


def test_load_bgen_sample_table_uses_external_sample_file(tmp_path: Path) -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_path = tmp_path / "custom_ids.sample"
    custom_identifiers = ["person_a", "person_b", "person_c", "person_d"]
    write_sample_file(sample_path, custom_identifiers)

    sample_table = load_bgen_sample_table(bgen_path, sample_path)

    assert sample_table.get_column("family_identifier").to_list() == custom_identifiers
    assert sample_table.get_column("individual_identifier").to_list() == custom_identifiers


def test_load_bgen_sample_table_rejects_sample_count_mismatch(tmp_path: Path) -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_path = tmp_path / "mismatch.sample"
    write_sample_file(sample_path, ["person_a", "person_b", "person_c"])

    with pytest.raises(ValueError, match="Expect number of samples in file to match"):
        load_bgen_sample_table(bgen_path, sample_path)


def test_load_bgen_sample_table_requires_sample_file_when_identifiers_are_missing() -> None:
    bgen_path = Path(example.get("complex.23bits.no.samples.bgen"))

    with pytest.raises(ValueError, match=r"does not contain samples and no \.sample file was found"):
        load_bgen_sample_table(bgen_path)


def test_validate_bgen_sample_order_failure() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_indices = np.arange(4, dtype=np.intp)
    expected_identifiers = np.array(["wrong1", "wrong2", "wrong3", "wrong4"], dtype=np.str_)

    with pytest.raises(ValueError, match="BGEN sample order does not match"), open_bgen(bgen_path) as bgen_reader:
        validate_bgen_sample_order(bgen_reader, sample_indices, expected_identifiers, bgen_path)


def test_validate_bgen_sample_order_requires_real_identifiers() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))

    with (
        open_bgen(bgen_path) as bgen_reader,
        pytest.raises(
            ValueError,
            match=r"does not contain samples and no \.sample file was found",
        ),
    ):
        bgen_reader.sample_identifier_source = "generated"
        validate_bgen_sample_order(
            bgen_reader,
            np.arange(2, dtype=np.intp),
            np.array(["sample_1", "sample_2"], dtype=np.str_),
            Path("study.bgen"),
        )


def test_read_bgen_chunk_helpers_return_expected_shapes() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))

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


def test_bgen_reader_rejects_non_biallelic_layout() -> None:
    fake_bgen_handle = SimpleNamespace(
        allele_ids=np.array(["A,C,G"], dtype=np.str_),
        ids=np.array(["variant_1"], dtype=np.str_),
        rsids=np.array(["rs1"], dtype=np.str_),
        chromosomes=np.array(["1"], dtype=np.str_),
        positions=np.array([123], dtype=np.int64),
        nvariants=1,
        nsamples=2,
        nalleles=np.array([3], dtype=np.int32),
        ncombinations=np.array([3], dtype=np.int32),
        phased=np.array([False]),
        samples=np.array(["sample_1", "sample_2"], dtype=np.str_),
        _cbgen=SimpleNamespace(contain_samples=True),
        close=lambda: None,
    )

    with (
        patch(
            "g.io.bgen.load_backend_open_bgen",
            return_value=lambda *args, **kwargs: fake_bgen_handle,
        ),
        pytest.raises(ValueError, match="Only diploid biallelic BGEN variants are supported"),
    ):
        open_bgen("study.bgen")


class FakeGeneratedSampleReader:
    """Context manager returning a reader without real sample identifiers."""

    sample_identifier_source = "generated"

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


def test_iter_linear_genotype_chunks_requires_real_sample_identifiers() -> None:
    with (
        patch(
            "g.io.bgen.open_bgen",
            return_value=FakeGeneratedSampleReader(),
        ),
        pytest.raises(ValueError, match=r"does not contain samples and no \.sample file was found"),
    ):
        list(
            iter_linear_genotype_chunks(
                bgen_path=Path("study.bgen"),
                sample_indices=np.arange(2, dtype=np.int64),
                expected_individual_identifiers=np.array(["sample_1", "sample_2"], dtype=np.str_),
                chunk_size=1,
            )
        )


def test_iter_genotype_chunks() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
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


def test_iter_linear_genotype_chunks() -> None:
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_indices = np.arange(4, dtype=np.int64)
    expected_ids = np.array([f"sample_{sample_index}" for sample_index in range(4)], dtype=np.str_)

    chunks = list(
        iter_linear_genotype_chunks(
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
