from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import polars as pl

from g.io.plink import iter_genotype_chunks
from g.models import GenotypeChunk, PreprocessedGenotypeChunkData, VariantMetadata


class MockBedHandle:
    """Minimal context-managed BED handle for iterator tests."""

    def __init__(self) -> None:
        """Store a thread-count attribute that mimics the bed-reader handle."""
        self._num_threads = 2

    def __enter__(self) -> MockBedHandle:
        """Enter the mock context manager."""
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        """Exit the mock context manager without suppressing exceptions."""
        return


def build_variant_table() -> pl.DataFrame:
    """Build a small BIM-like variant table."""
    return pl.DataFrame(
        {
            "chromosome": [1, 1, 1],
            "variant_identifier": ["variant1", "variant2", "variant3"],
            "genetic_distance": [0.0, 0.0, 0.0],
            "position": [10, 20, 30],
            "allele_one": ["A", "C", "G"],
            "allele_two": ["G", "T", "A"],
        }
    )


def build_preprocessed_chunk() -> PreprocessedGenotypeChunkData:
    """Build a preprocessed chunk fixture."""
    return PreprocessedGenotypeChunkData(
        genotypes=jnp.array([[0.0, 1.0], [1.0, 2.0]]),
        missing_mask=jnp.array([[False, False], [False, False]]),
        has_missing_values=False,
        allele_one_frequency=jnp.array([0.25, 0.75]),
        observation_count=jnp.array([2, 2], dtype=jnp.int32),
    )


def build_output_chunk(variant_start: int, variant_stop: int) -> GenotypeChunk:
    """Build an iterator output chunk with sliced metadata."""
    variant_identifiers = np.array(["variant1", "variant2", "variant3"])[variant_start:variant_stop]
    return GenotypeChunk(
        genotypes=jnp.ones((2, variant_stop - variant_start)),
        missing_mask=jnp.zeros((2, variant_stop - variant_start), dtype=bool),
        has_missing_values=False,
        metadata=VariantMetadata(
            chromosome=np.array(["1"] * (variant_stop - variant_start)),
            variant_identifiers=variant_identifiers,
            position=np.array([10, 20, 30], dtype=np.int64)[variant_start:variant_stop],
            allele_one=np.array(["A", "C", "G"])[variant_start:variant_stop],
            allele_two=np.array(["G", "T", "A"])[variant_start:variant_stop],
        ),
        allele_one_frequency=jnp.full((variant_stop - variant_start,), 0.5),
        observation_count=jnp.full((variant_stop - variant_start,), 2, dtype=jnp.int32),
    )


def test_iter_genotype_chunks_uses_python_reader_chunk_boundaries() -> None:
    """Ensure the Python reader path respects variant limits and chunk slicing."""
    preprocessed_chunk = build_preprocessed_chunk()

    with (
        patch("g.io.plink.load_variant_table", return_value=build_variant_table()),
        patch("g.io.plink.validate_bed_sample_order") as mock_validate_sample_order,
        patch("g.io.plink.open_bed", return_value=MockBedHandle()),
        patch("g.io.plink.get_num_threads", return_value=3),
        patch("g.io.plink.read_bed_chunk_host", return_value=np.ones((2, 2), dtype=np.float64)) as mock_read_chunk_host,
        patch("g.io.plink.preprocess_genotype_matrix", return_value=preprocessed_chunk) as mock_preprocess,
        patch(
            "g.io.plink.build_genotype_chunk",
            side_effect=lambda **kwargs: build_output_chunk(kwargs["variant_start"], kwargs["variant_stop"]),
        ) as mock_build_chunk,
    ):
        chunks = list(
            iter_genotype_chunks(
                bed_prefix=Path("dataset"),
                sample_indices=np.array([0, 2], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample1", "sample2"]),
                chunk_size=2,
                variant_limit=3,
            )
        )

    assert [chunk.metadata.variant_identifiers.tolist() for chunk in chunks] == [["variant1", "variant2"], ["variant3"]]
    mock_validate_sample_order.assert_called_once()
    assert mock_read_chunk_host.call_count == 2
    assert mock_preprocess.call_count == 2
    assert mock_build_chunk.call_count == 2


def test_iter_genotype_chunks_uses_native_reader_and_preprocessing() -> None:
    """Ensure the native iterator path wires native decode and preprocessing together."""
    preprocessed_chunk = build_preprocessed_chunk()

    with (
        patch("g.io.plink.load_variant_table", return_value=build_variant_table()),
        patch("g.io.plink.validate_bed_sample_order"),
        patch("g.io.plink.read_bed_chunk_native", return_value=jnp.ones((2, 2))) as mock_read_bed_chunk_native,
        patch(
            "g.io.plink.preprocess_genotype_matrix_native", return_value=preprocessed_chunk
        ) as mock_preprocess_native,
        patch(
            "g.io.plink.build_genotype_chunk",
            side_effect=lambda **kwargs: build_output_chunk(kwargs["variant_start"], kwargs["variant_stop"]),
        ),
    ):
        chunks = list(
            iter_genotype_chunks(
                bed_prefix=Path("dataset"),
                sample_indices=np.array([0, 2], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample1", "sample2"]),
                chunk_size=5,
                variant_limit=2,
                use_native_reader=True,
                use_native_preprocessing=True,
            )
        )

    assert len(chunks) == 1
    assert chunks[0].metadata.variant_identifiers.tolist() == ["variant1", "variant2"]
    mock_read_bed_chunk_native.assert_called_once()
    mock_preprocess_native.assert_called_once()
