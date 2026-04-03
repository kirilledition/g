from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl

from g.compute.logistic import LogisticErrorCode, LogisticMethod
from g.engine import (
    LinearChunkAccumulator,
    LogisticChunkAccumulator,
    build_linear_output_frame,
    build_logistic_output_frame,
    compute_logistic_association_with_missing_exclusion,
    concatenate_linear_results,
    concatenate_logistic_results,
    write_frame_iterator_to_tsv,
)
from g.models import (
    GenotypeChunk,
    LinearAssociationChunkResult,
    LogisticAssociationChunkResult,
    VariantMetadata,
)


def test_build_linear_output_frame() -> None:
    """Test building a linear output DataFrame from results."""
    metadata = VariantMetadata(variant_start_index=0, variant_stop_index=1,
        chromosome=np.array(["1", "1"]),
        variant_identifiers=np.array(["var1", "var2"]),
        position=np.array([100, 200]),
        allele_one=np.array(["A", "C"]),
        allele_two=np.array(["G", "T"]),
    )
    allele_one_frequency = jnp.array([0.25, 0.5])
    observation_count = jnp.array([100, 100])
    linear_result = LinearAssociationChunkResult(
        beta=jnp.array([0.1, 0.2]),
        standard_error=jnp.array([0.01, 0.02]),
        test_statistic=jnp.array([10.0, 10.0]),
        p_value=jnp.array([0.001, 0.001]),
        valid_mask=jnp.array([True, True]),
    )

    df = build_linear_output_frame(metadata, allele_one_frequency, observation_count, linear_result)

    assert df.height == 2
    assert "variant_identifier" in df.columns
    assert "beta" in df.columns
    assert "p_value" in df.columns
    assert df.get_column("variant_identifier").to_list() == ["var1", "var2"]


def test_build_logistic_output_frame() -> None:
    """Test building a logistic output DataFrame from results."""
    metadata = VariantMetadata(variant_start_index=0, variant_stop_index=1,
        chromosome=np.array(["1", "1"]),
        variant_identifiers=np.array(["var1", "var2"]),
        position=np.array([100, 200]),
        allele_one=np.array(["A", "C"]),
        allele_two=np.array(["G", "T"]),
    )
    allele_one_frequency = jnp.array([0.25, 0.5])
    observation_count = jnp.array([100, 100])
    logistic_result = LogisticAssociationChunkResult(
        beta=jnp.array([0.1, 0.2]),
        standard_error=jnp.array([0.01, 0.02]),
        test_statistic=jnp.array([10.0, 10.0]),
        p_value=jnp.array([0.001, 0.001]),
        method_code=jnp.array([LogisticMethod.FIRTH, LogisticMethod.STANDARD]),
        error_code=jnp.array([LogisticErrorCode.NONE, LogisticErrorCode.UNFINISHED]),
        converged_mask=jnp.array([True, False]),
        iteration_count=jnp.array([10, 50]),
        valid_mask=jnp.array([True, True]),
    )

    df = build_logistic_output_frame(metadata, allele_one_frequency, observation_count, logistic_result)

    assert df.height == 2
    assert "variant_identifier" in df.columns
    assert "beta" in df.columns
    assert "firth_flag" in df.columns
    assert "error_code" in df.columns
    assert df.get_column("firth_flag").to_list() == ["Y", "N"]
    assert df.get_column("error_code").to_list() == [".", "UNFINISHED"]


def test_concatenate_linear_results_empty() -> None:
    """Test concatenating empty linear results."""
    result = concatenate_linear_results([])

    assert result.is_empty()


def test_concatenate_linear_results_single_chunk() -> None:
    """Test concatenating single linear chunk."""
    metadata = VariantMetadata(variant_start_index=0, variant_stop_index=1,
        chromosome=np.array(["1"]),
        variant_identifiers=np.array(["var1"]),
        position=np.array([100]),
        allele_one=np.array(["A"]),
        allele_two=np.array(["G"]),
    )
    accumulator = LinearChunkAccumulator(
        metadata=metadata,
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([100]),
        linear_result=LinearAssociationChunkResult(
            beta=jnp.array([0.1]),
            standard_error=jnp.array([0.01]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            valid_mask=jnp.array([True]),
        ),
    )

    result = concatenate_linear_results([accumulator])

    assert result.height == 1
    assert result.get_column("variant_identifier").to_list() == ["var1"]


def test_concatenate_linear_results_multiple_chunks() -> None:
    """Test concatenating multiple linear chunks."""
    acc1 = LinearChunkAccumulator(
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array(["var1"]),
            position=np.array([100]),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([100]),
        linear_result=LinearAssociationChunkResult(
            beta=jnp.array([0.1]),
            standard_error=jnp.array([0.01]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            valid_mask=jnp.array([True]),
        ),
    )
    acc2 = LinearChunkAccumulator(
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["2"]),
            variant_identifiers=np.array(["var2"]),
            position=np.array([200]),
            allele_one=np.array(["C"]),
            allele_two=np.array(["T"]),
        ),
        allele_one_frequency=jnp.array([0.5]),
        observation_count=jnp.array([100]),
        linear_result=LinearAssociationChunkResult(
            beta=jnp.array([0.2]),
            standard_error=jnp.array([0.02]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            valid_mask=jnp.array([True]),
        ),
    )

    result = concatenate_linear_results([acc1, acc2])

    assert result.height == 2
    assert result.get_column("variant_identifier").to_list() == ["var1", "var2"]


def test_concatenate_logistic_results_empty() -> None:
    """Test concatenating empty logistic results."""
    result = concatenate_logistic_results([])

    assert result.is_empty()


def test_concatenate_logistic_results_single_chunk() -> None:
    """Test concatenating single logistic chunk."""
    metadata = VariantMetadata(variant_start_index=0, variant_stop_index=1,
        chromosome=np.array(["1"]),
        variant_identifiers=np.array(["var1"]),
        position=np.array([100]),
        allele_one=np.array(["A"]),
        allele_two=np.array(["G"]),
    )
    accumulator = LogisticChunkAccumulator(
        metadata=metadata,
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([100]),
        logistic_result=LogisticAssociationChunkResult(
            beta=jnp.array([0.1]),
            standard_error=jnp.array([0.01]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            method_code=jnp.array([LogisticMethod.STANDARD]),
            error_code=jnp.array([LogisticErrorCode.NONE]),
            converged_mask=jnp.array([True]),
            iteration_count=jnp.array([10]),
            valid_mask=jnp.array([True]),
        ),
    )

    result = concatenate_logistic_results([accumulator])

    assert result.height == 1
    assert result.get_column("variant_identifier").to_list() == ["var1"]


def test_concatenate_logistic_results_multiple_chunks() -> None:
    """Test concatenating multiple logistic chunks."""
    acc1 = LogisticChunkAccumulator(
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array(["var1"]),
            position=np.array([100]),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([100]),
        logistic_result=LogisticAssociationChunkResult(
            beta=jnp.array([0.1]),
            standard_error=jnp.array([0.01]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            method_code=jnp.array([LogisticMethod.STANDARD]),
            error_code=jnp.array([LogisticErrorCode.NONE]),
            converged_mask=jnp.array([True]),
            iteration_count=jnp.array([10]),
            valid_mask=jnp.array([True]),
        ),
    )
    acc2 = LogisticChunkAccumulator(
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["2"]),
            variant_identifiers=np.array(["var2"]),
            position=np.array([200]),
            allele_one=np.array(["C"]),
            allele_two=np.array(["T"]),
        ),
        allele_one_frequency=jnp.array([0.5]),
        observation_count=jnp.array([100]),
        logistic_result=LogisticAssociationChunkResult(
            beta=jnp.array([0.2]),
            standard_error=jnp.array([0.02]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            method_code=jnp.array([LogisticMethod.FIRTH]),
            error_code=jnp.array([LogisticErrorCode.NONE]),
            converged_mask=jnp.array([True]),
            iteration_count=jnp.array([20]),
            valid_mask=jnp.array([True]),
        ),
    )

    result = concatenate_logistic_results([acc1, acc2])

    assert result.height == 2
    assert result.get_column("variant_identifier").to_list() == ["var1", "var2"]
    assert result.get_column("firth_flag").to_list() == ["N", "Y"]


def test_write_frame_iterator_to_tsv_linear(tmp_path: Path) -> None:
    """Test writing linear results to TSV."""
    accumulator = LinearChunkAccumulator(
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array(["var1"]),
            position=np.array([100]),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([100]),
        linear_result=LinearAssociationChunkResult(
            beta=jnp.array([0.1]),
            standard_error=jnp.array([0.01]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            valid_mask=jnp.array([True]),
        ),
    )

    output_path = tmp_path / "results.linear.tsv"
    write_frame_iterator_to_tsv(iter([accumulator]), output_path)

    assert output_path.exists()
    df = pl.read_csv(output_path, separator="\t")
    assert df.height == 1
    assert "variant_identifier" in df.columns


def test_write_frame_iterator_to_tsv_logistic(tmp_path: Path) -> None:
    """Test writing logistic results to TSV."""
    accumulator = LogisticChunkAccumulator(
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array(["var1"]),
            position=np.array([100]),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25]),
        observation_count=jnp.array([100]),
        logistic_result=LogisticAssociationChunkResult(
            beta=jnp.array([0.1]),
            standard_error=jnp.array([0.01]),
            test_statistic=jnp.array([10.0]),
            p_value=jnp.array([0.001]),
            method_code=jnp.array([LogisticMethod.STANDARD]),
            error_code=jnp.array([LogisticErrorCode.NONE]),
            converged_mask=jnp.array([True]),
            iteration_count=jnp.array([10]),
            valid_mask=jnp.array([True]),
        ),
    )

    output_path = tmp_path / "results.logistic.tsv"
    write_frame_iterator_to_tsv(iter([accumulator]), output_path)

    assert output_path.exists()
    df = pl.read_csv(output_path, separator="\t")
    assert df.height == 1
    assert "variant_identifier" in df.columns


def test_write_frame_iterator_to_tsv_empty(tmp_path: Path) -> None:
    """Test writing empty results to TSV."""
    output_path = tmp_path / "results.tsv"
    write_frame_iterator_to_tsv(iter([]), output_path)

    assert output_path.exists()


def test_compute_logistic_association_with_missing_exclusion_no_missing() -> None:
    """Test logistic association when no missing values."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    phenotype_vector = jnp.array([0.0, 1.0, 0.0])
    genotype_chunk = GenotypeChunk(
        genotypes=jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]]).T,
        missing_mask=jnp.array([[False, False], [False, False], [False, False]]),
        has_missing_values=False,
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["1", "2"]),
            variant_identifiers=np.array(["var1", "var2"]),
            position=np.array([100, 200]),
            allele_one=np.array(["A", "C"]),
            allele_two=np.array(["G", "T"]),
        ),
        allele_one_frequency=jnp.array([0.25, 0.5]),
        observation_count=jnp.array([3, 3]),
    )

    result = compute_logistic_association_with_missing_exclusion(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_chunk=genotype_chunk,
        max_iterations=50,
        tolerance=1e-8,
    )

    assert result.logistic_result.beta.shape == (2,)
    assert result.logistic_result.valid_mask.shape == (2,)
    np.testing.assert_array_equal(result.allele_one_frequency, jnp.array([0.25, 0.5]))


def test_compute_logistic_association_with_missing_exclusion_with_missing() -> None:
    """Test logistic association when there are missing values."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    phenotype_vector = jnp.array([0.0, 1.0, 0.0])
    genotype_chunk = GenotypeChunk(
        genotypes=jnp.array([[0.0, 1.0], [jnp.nan, 0.0], [2.0, 1.0]]),
        missing_mask=jnp.array([[False, False], [True, False], [False, False]]),
        has_missing_values=True,
        metadata=VariantMetadata(variant_start_index=0, variant_stop_index=1,
            chromosome=np.array(["1", "2"]),
            variant_identifiers=np.array(["var1", "var2"]),
            position=np.array([100, 200]),
            allele_one=np.array(["A", "C"]),
            allele_two=np.array(["G", "T"]),
        ),
        allele_one_frequency=jnp.array([0.25, 0.5]),
        observation_count=jnp.array([3, 3]),
    )

    result = compute_logistic_association_with_missing_exclusion(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_chunk=genotype_chunk,
        max_iterations=50,
        tolerance=1e-8,
    )

    assert result.logistic_result.beta.shape == (2,)
    assert result.logistic_result.valid_mask.shape == (2,)
