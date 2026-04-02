from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import polars as pl

from g.engine import (
    iter_linear_output_frames,
    iter_logistic_output_frames,
    run_linear_association,
    run_logistic_association,
)
from g.io.source import build_plink_source_config
from g.models import (
    AlignedSampleData,
    GenotypeChunk,
    LinearAssociationChunkResult,
    LogisticAssociationChunkResult,
    LogisticAssociationEvaluation,
    VariantMetadata,
)


def build_aligned_sample_data(*, is_binary_trait: bool) -> AlignedSampleData:
    """Build a small aligned-sample fixture for engine iterator tests."""
    return AlignedSampleData(
        sample_indices=np.array([0, 2], dtype=np.int64),
        family_identifiers=np.array(["family1", "family2"]),
        individual_identifiers=np.array(["sample1", "sample2"]),
        phenotype_name="trait",
        phenotype_vector=jnp.array([0.0, 1.0]),
        covariate_names=("intercept", "age"),
        covariate_matrix=jnp.array([[1.0, 25.0], [1.0, 30.0]]),
        is_binary_trait=is_binary_trait,
    )


def build_genotype_chunk(variant_identifier: str) -> GenotypeChunk:
    """Build a single-variant chunk fixture."""
    return GenotypeChunk(
        genotypes=jnp.array([[0.0], [1.0]]),
        missing_mask=jnp.array([[False], [False]]),
        has_missing_values=False,
        metadata=VariantMetadata(
            variant_start_index=0,
            variant_stop_index=1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array([variant_identifier]),
            position=np.array([123], dtype=np.int64),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([2]),
    )


def test_iter_linear_output_frames_yields_one_accumulator_per_chunk() -> None:
    """Ensure linear iteration preserves chunk order and propagated summary fields."""
    aligned_sample_data = build_aligned_sample_data(is_binary_trait=False)
    genotype_chunks = [build_genotype_chunk("variant1"), build_genotype_chunk("variant2")]
    linear_result = LinearAssociationChunkResult(
        beta=jnp.array([0.5]),
        standard_error=jnp.array([0.1]),
        test_statistic=jnp.array([5.0]),
        p_value=jnp.array([0.01]),
        valid_mask=jnp.array([True]),
    )

    with (
        patch("g.engine.load_aligned_sample_data_from_source", return_value=aligned_sample_data)
        as mock_load_aligned_sample_data,
        patch("g.engine.prepare_linear_association_state", return_value="linear-state") as mock_prepare_state,
        patch(
            "g.engine.iter_linear_genotype_chunks_from_source",
            return_value=iter(genotype_chunks),
        ) as mock_iter_genotype_chunks,
        patch("g.engine.compute_linear_association_chunk", return_value=linear_result) as mock_compute_chunk,
    ):
        accumulators = list(
            iter_linear_output_frames(
                genotype_source_config=build_plink_source_config(Path("dataset")),
                phenotype_path=Path("phenotype.tsv"),
                phenotype_name="trait",
                covariate_path=Path("covariates.tsv"),
                covariate_names=("age",),
                chunk_size=32,
                variant_limit=2,
            )
        )

    assert len(accumulators) == 2
    assert accumulators[0].metadata.variant_identifiers.tolist() == ["variant1"]
    assert accumulators[1].metadata.variant_identifiers.tolist() == ["variant2"]
    np.testing.assert_allclose(accumulators[0].linear_result.beta, np.array([0.5]), atol=0.0)
    mock_load_aligned_sample_data.assert_called_once()
    mock_prepare_state.assert_called_once()
    mock_iter_genotype_chunks.assert_called_once()
    assert mock_compute_chunk.call_count == 2


def test_iter_logistic_output_frames_passes_no_missing_constants() -> None:
    """Ensure logistic iteration reuses precomputed no-missing constants across chunks."""
    aligned_sample_data = build_aligned_sample_data(is_binary_trait=True)
    genotype_chunks = [build_genotype_chunk("variant1"), build_genotype_chunk("variant2")]
    logistic_evaluation = LogisticAssociationEvaluation(
        logistic_result=LogisticAssociationChunkResult(
            beta=jnp.array([0.25]),
            standard_error=jnp.array([0.05]),
            test_statistic=jnp.array([5.0]),
            p_value=jnp.array([0.02]),
            method_code=jnp.array([0], dtype=jnp.int32),
            error_code=jnp.array([0], dtype=jnp.int32),
            converged_mask=jnp.array([True]),
            valid_mask=jnp.array([True]),
            iteration_count=jnp.array([3], dtype=jnp.int32),
        ),
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([2], dtype=jnp.int64),
    )
    no_missing_constants = object()

    with (
        patch("g.engine.load_aligned_sample_data_from_source", return_value=aligned_sample_data),
        patch(
            "g.engine.prepare_no_missing_logistic_constants", return_value=no_missing_constants
        ) as mock_prepare_constants,
        patch("g.engine.iter_genotype_chunks_from_source", return_value=iter(genotype_chunks)),
        patch(
            "g.engine.compute_logistic_association_with_missing_exclusion",
            return_value=logistic_evaluation,
        ) as mock_compute_chunk,
    ):
        accumulators = list(
            iter_logistic_output_frames(
                genotype_source_config=build_plink_source_config(Path("dataset")),
                phenotype_path=Path("phenotype.tsv"),
                phenotype_name="trait",
                covariate_path=Path("covariates.tsv"),
                covariate_names=None,
                chunk_size=16,
                variant_limit=2,
                max_iterations=25,
                tolerance=1.0e-8,
            )
        )

    assert len(accumulators) == 2
    assert accumulators[0].logistic_result.iteration_count.tolist() == [3]
    mock_prepare_constants.assert_called_once()
    assert mock_compute_chunk.call_count == 2
    for call in mock_compute_chunk.call_args_list:
        assert call.kwargs["no_missing_constants"] is no_missing_constants


def test_run_linear_association_concatenates_iterator_results() -> None:
    """Ensure the linear runner materializes iterator output then concatenates it."""
    expected_frame = pl.DataFrame({"variant_identifier": ["variant1"]})
    accumulator = object()

    with (
        patch("g.engine.iter_linear_output_frames", return_value=iter([accumulator])) as mock_iter_linear_output_frames,
        patch("g.engine.concatenate_linear_results", return_value=expected_frame) as mock_concatenate,
    ):
        result_frame = run_linear_association(
            genotype_source_config=build_plink_source_config(Path("dataset")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=64,
            variant_limit=5,
        )

    assert result_frame.equals(expected_frame)
    mock_iter_linear_output_frames.assert_called_once()
    mock_concatenate.assert_called_once_with([accumulator])


def test_run_logistic_association_concatenates_iterator_results() -> None:
    """Ensure the logistic runner materializes iterator output then concatenates it."""
    expected_frame = pl.DataFrame({"variant_identifier": ["variant1"]})
    accumulator = object()

    with (
        patch(
            "g.engine.iter_logistic_output_frames", return_value=iter([accumulator])
        ) as mock_iter_logistic_output_frames,
        patch("g.engine.concatenate_logistic_results", return_value=expected_frame) as mock_concatenate,
    ):
        result_frame = run_logistic_association(
            genotype_source_config=build_plink_source_config(Path("dataset")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=64,
            variant_limit=5,
            max_iterations=25,
            tolerance=1.0e-8,
        )

    assert result_frame.equals(expected_frame)
    mock_iter_logistic_output_frames.assert_called_once()
    mock_concatenate.assert_called_once_with([accumulator])
