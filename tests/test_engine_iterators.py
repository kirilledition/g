from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.engine import (
    Regenie2LinearChunkAccumulator,
    iter_regenie2_linear_output_frames,
    split_dosage_genotype_chunk_by_absolute_variant_slices,
    split_dosage_genotype_chunk_by_chromosome,
    split_dosage_genotype_chunk_with_reader_metadata,
    write_frame_iterator_to_tsv,
)
from g.io.source import build_bgen_source_config
from g.models import (
    AlignedSampleData,
    DosageGenotypeChunk,
    Regenie2LinearChunkResult,
    VariantMetadata,
)


class FakeContextReader:
    def __enter__(self) -> FakeContextReader:
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        return


class FakeChromosomePartitionReader(FakeContextReader):
    def split_variant_slice_by_chromosome(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[tuple[int, int], ...]:
        del variant_stop
        return ((variant_start, variant_start + 2), (variant_start + 2, variant_start + 4))


class FakePredictionSource:
    def get_chromosome_predictions(
        self,
        chromosome: str,
        sample_family_identifiers: np.ndarray,
        sample_individual_identifiers: np.ndarray,
    ) -> jnp.ndarray:
        del chromosome, sample_family_identifiers, sample_individual_identifiers
        return jnp.array([0.0, 0.0], dtype=jnp.float32)


def build_aligned_sample_data() -> AlignedSampleData:
    return AlignedSampleData(
        sample_indices=np.array([0, 1], dtype=np.int64),
        family_identifiers=np.array(["family1", "family2"]),
        individual_identifiers=np.array(["sample1", "sample2"]),
        phenotype_name="trait",
        phenotype_vector=jnp.array([0.0, 1.0]),
        covariate_names=("intercept", "age"),
        covariate_matrix=jnp.array([[1.0, 25.0], [1.0, 30.0]]),
        is_binary_trait=False,
    )


def build_dosage_chunk(*, chromosome_values: list[str]) -> DosageGenotypeChunk:
    variant_count = len(chromosome_values)
    return DosageGenotypeChunk(
        genotypes=jnp.arange(variant_count * 2, dtype=jnp.float32).reshape(2, variant_count),
        metadata=VariantMetadata(
            variant_start_index=10,
            variant_stop_index=10 + variant_count,
            chromosome=np.asarray(chromosome_values, dtype=np.str_),
            variant_identifiers=np.asarray([f"variant{variant_index}" for variant_index in range(variant_count)]),
            position=np.asarray([100 + variant_index for variant_index in range(variant_count)], dtype=np.int64),
            allele_one=np.asarray(["A"] * variant_count),
            allele_two=np.asarray(["G"] * variant_count),
        ),
        allele_one_frequency=jnp.linspace(0.1, 0.4, num=variant_count, dtype=jnp.float32),
        observation_count=jnp.full((variant_count,), 2, dtype=jnp.int32),
    )


def build_chunk_accumulator(
    *,
    variant_start_index: int,
    chromosome_values: list[str],
) -> Regenie2LinearChunkAccumulator:
    dosage_chunk = build_dosage_chunk(chromosome_values=chromosome_values)
    variant_count = len(chromosome_values)
    metadata = VariantMetadata(
        variant_start_index=variant_start_index,
        variant_stop_index=variant_start_index + variant_count,
        chromosome=dosage_chunk.metadata.chromosome,
        variant_identifiers=np.asarray(
            [f"variant{variant_start_index + variant_offset}" for variant_offset in range(variant_count)]
        ),
        position=np.asarray([200 + variant_start_index + variant_offset for variant_offset in range(variant_count)]),
        allele_one=dosage_chunk.metadata.allele_one,
        allele_two=dosage_chunk.metadata.allele_two,
    )
    regenie_result = Regenie2LinearChunkResult(
        beta=jnp.linspace(0.1, 0.1 * variant_count, num=variant_count, dtype=jnp.float32),
        standard_error=jnp.full((variant_count,), 0.2, dtype=jnp.float32),
        chi_squared=jnp.full((variant_count,), 5.0, dtype=jnp.float32),
        log10_p_value=jnp.full((variant_count,), 2.0, dtype=jnp.float32),
        valid_mask=jnp.ones((variant_count,), dtype=bool),
    )
    return Regenie2LinearChunkAccumulator(
        metadata=metadata,
        allele_one_frequency=dosage_chunk.allele_one_frequency,
        observation_count=dosage_chunk.observation_count,
        regenie2_linear_result=regenie_result,
    )


def test_split_dosage_genotype_chunk_by_chromosome_returns_original_for_homogeneous_chunk() -> None:
    genotype_chunk = build_dosage_chunk(chromosome_values=["22", "22", "22"])
    chromosome_subchunks = split_dosage_genotype_chunk_by_chromosome(genotype_chunk)
    assert chromosome_subchunks == (genotype_chunk,)


def test_split_dosage_genotype_chunk_by_chromosome_splits_heterogeneous_chunk() -> None:
    genotype_chunk = build_dosage_chunk(chromosome_values=["22", "22", "X", "X"])
    chromosome_subchunks = split_dosage_genotype_chunk_by_chromosome(genotype_chunk)
    assert len(chromosome_subchunks) == 2
    assert chromosome_subchunks[0].metadata.chromosome.tolist() == ["22", "22"]
    assert chromosome_subchunks[1].metadata.chromosome.tolist() == ["X", "X"]
    assert chromosome_subchunks[0].metadata.variant_start_index == 10
    assert chromosome_subchunks[1].metadata.variant_start_index == 12


def test_split_dosage_genotype_chunk_by_absolute_variant_slices_preserves_metadata_offsets() -> None:
    genotype_chunk = build_dosage_chunk(chromosome_values=["22", "22", "X", "X"])
    chromosome_subchunks = split_dosage_genotype_chunk_by_absolute_variant_slices(
        genotype_chunk,
        ((10, 12), (12, 14)),
    )

    assert len(chromosome_subchunks) == 2
    assert chromosome_subchunks[0].metadata.variant_start_index == 10
    assert chromosome_subchunks[0].metadata.variant_stop_index == 12
    assert chromosome_subchunks[1].metadata.variant_start_index == 12
    assert chromosome_subchunks[1].metadata.variant_stop_index == 14


def test_split_dosage_genotype_chunk_with_reader_metadata_uses_reader_partitioning() -> None:
    genotype_chunk = build_dosage_chunk(chromosome_values=["22", "22", "X", "X"])
    chromosome_subchunks = split_dosage_genotype_chunk_with_reader_metadata(
        genotype_chunk,
        FakeChromosomePartitionReader(),
    )

    assert len(chromosome_subchunks) == 2
    assert chromosome_subchunks[0].metadata.variant_start_index == 10
    assert chromosome_subchunks[1].metadata.variant_start_index == 12


def test_iter_regenie2_linear_output_frames_reuses_open_bgen_reader() -> None:
    aligned_sample_data = build_aligned_sample_data()
    source_chunk = build_dosage_chunk(chromosome_values=["22", "22"])
    regenie_result = Regenie2LinearChunkResult(
        beta=jnp.array([0.5, 0.3], dtype=jnp.float32),
        standard_error=jnp.array([0.1, 0.1], dtype=jnp.float32),
        chi_squared=jnp.array([25.0, 9.0], dtype=jnp.float32),
        log10_p_value=jnp.array([5.0, 2.0], dtype=jnp.float32),
        valid_mask=jnp.array([True, True]),
    )
    genotype_reader = FakeContextReader()

    with (
        patch("g.engine.open_genotype_reader", return_value=genotype_reader) as mock_open_genotype_reader,
        patch("g.engine.load_aligned_sample_data_from_source", return_value=aligned_sample_data) as mock_load,
        patch("g.engine.prepare_regenie2_linear_state", return_value="regenie-state"),
        patch("g.engine.load_prediction_source", return_value=FakePredictionSource()),
        patch("g.engine.iter_dosage_genotype_chunks_from_source", return_value=iter([source_chunk])) as mock_iter,
        patch("g.engine.compute_regenie2_linear_chunk", return_value=regenie_result),
    ):
        accumulators = list(
            iter_regenie2_linear_output_frames(
                genotype_source_config=build_bgen_source_config(Path("study.bgen")),
                phenotype_path=Path("phenotype.tsv"),
                phenotype_name="trait",
                prediction_list_path=Path("predictions.list"),
                covariate_path=Path("covariates.tsv"),
                covariate_names=("age",),
                chunk_size=32,
                variant_limit=1,
            )
        )

    assert len(accumulators) == 1
    mock_open_genotype_reader.assert_called_once()
    assert mock_load.call_args.kwargs["genotype_reader"] is genotype_reader
    assert mock_iter.call_args.kwargs["genotype_reader"] is genotype_reader


def test_write_frame_iterator_to_tsv_batches_multiple_chunks(tmp_path: Path) -> None:
    output_path = tmp_path / "results.tsv"
    accumulators = [
        build_chunk_accumulator(variant_start_index=0, chromosome_values=["22", "22"]),
        build_chunk_accumulator(variant_start_index=2, chromosome_values=["22", "22"]),
    ]

    write_frame_iterator_to_tsv(iter(accumulators), output_path, frame_batch_size=2)

    output_frame = pl.read_csv(output_path, separator="\t")
    assert output_frame.height == 4
    assert output_frame.get_column("variant_identifier").to_list() == ["variant0", "variant1", "variant2", "variant3"]


def test_write_frame_iterator_to_tsv_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TSV frame batch size must be positive"):
        write_frame_iterator_to_tsv(iter(()), tmp_path / "results.tsv", frame_batch_size=0)
