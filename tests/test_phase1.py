from __future__ import annotations

import math
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.cli import main as cli_main
from g.engine import (
    compute_logistic_association_with_missing_exclusion,
    run_linear_association,
    run_logistic_association,
)
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data
from g.models import GenotypeChunk, VariantMetadata

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", str(REPOSITORY_ROOT / "data")))
BED_PREFIX = DATA_DIRECTORY / "1kg_chr22_full"
CONTINUOUS_BASELINE_PATH = DATA_DIRECTORY / "baselines" / "plink_cont.phenotype_continuous.glm.linear"
BINARY_BASELINE_PATH = DATA_DIRECTORY / "baselines" / "plink_bin.phenotype_binary.glm.logistic.hybrid"


BETA_MAX_ABSOLUTE_ERROR = 1.0e-3
P_VALUE_MAX_LOG10_ERROR = 1.0e-2


def assert_beta_parity(observed_beta: np.ndarray, expected_beta: np.ndarray) -> None:
    """Assert that beta values stay within the allowed absolute error budget."""
    absolute_error = np.abs(observed_beta - expected_beta)
    max_absolute_error = float(np.max(absolute_error, initial=0.0))
    assert max_absolute_error < BETA_MAX_ABSOLUTE_ERROR, (
        f"beta max absolute error {max_absolute_error:.6g} exceeds {BETA_MAX_ABSOLUTE_ERROR:.6g}"
    )


def assert_log10_p_value_parity(
    observed_p_values: np.ndarray,
    expected_p_values: np.ndarray,
    max_log10_difference: float,
) -> None:
    """Assert that p-values match in log10 space within the requested bound."""
    minimum_positive_value = np.finfo(np.float64).tiny
    observed_log10_p_values = np.log10(np.clip(observed_p_values, minimum_positive_value, None))
    expected_log10_p_values = np.log10(np.clip(expected_p_values, minimum_positive_value, None))
    log10_difference = np.abs(observed_log10_p_values - expected_log10_p_values)
    max_observed_difference = float(np.max(log10_difference, initial=0.0))
    assert max_observed_difference < max_log10_difference, (
        f"max log10(p) error {max_observed_difference:.6g} exceeds {max_log10_difference:.6g}"
    )


def require_phase_zero_inputs() -> None:
    """Skip tests if Phase 0 artifacts are not present."""
    required_paths = [
        BED_PREFIX.with_suffix(".bed"),
        BED_PREFIX.with_suffix(".bim"),
        BED_PREFIX.with_suffix(".fam"),
        DATA_DIRECTORY / "pheno_cont.txt",
        DATA_DIRECTORY / "pheno_bin.txt",
        DATA_DIRECTORY / "covariates.txt",
        CONTINUOUS_BASELINE_PATH,
        BINARY_BASELINE_PATH,
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("Phase 0 data and baseline files are required for Phase 1 tests.")


def test_sample_alignment_recode_binary_and_intercept() -> None:
    """Ensure aligned binary inputs preserve order and build an intercept."""
    require_phase_zero_inputs()

    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=BED_PREFIX,
        phenotype_path=DATA_DIRECTORY / "pheno_bin.txt",
        phenotype_name="phenotype_binary",
        covariate_path=DATA_DIRECTORY / "covariates.txt",
        covariate_names=None,
        is_binary_trait=True,
    )

    assert aligned_sample_data.sample_indices.shape[0] == 2504
    assert aligned_sample_data.individual_identifiers[0] == "HG00096"
    np.testing.assert_allclose(np.asarray(aligned_sample_data.covariate_matrix)[:, 0], 1.0, atol=0.0)
    assert set(np.unique(np.asarray(aligned_sample_data.phenotype_vector)).tolist()) == {0.0, 1.0}


def test_genotype_chunk_reader_matches_expected_metadata() -> None:
    """Ensure chunked BED reads align with BIM metadata and sample order."""
    require_phase_zero_inputs()

    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=BED_PREFIX,
        phenotype_path=DATA_DIRECTORY / "pheno_cont.txt",
        phenotype_name="phenotype_continuous",
        covariate_path=DATA_DIRECTORY / "covariates.txt",
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )

    first_chunk = next(
        iter_genotype_chunks(
            bed_prefix=BED_PREFIX,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=8,
            variant_limit=8,
        )
    )

    assert first_chunk.metadata.variant_identifiers.tolist()[:3] == [
        "rs587755077",
        "rs587654921",
        "rs587720402",
    ]
    assert first_chunk.genotypes.shape == (2504, 8)
    assert math.isclose(float(np.asarray(first_chunk.allele_one_frequency)[0]), 0.00638978, rel_tol=0.0, abs_tol=1e-6)


def test_logistic_missing_rows_are_excluded_per_variant() -> None:
    """Ensure logistic regression excludes missing genotype rows per variant."""
    covariate_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    phenotype_vector = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    genotype_chunk = GenotypeChunk(
        genotypes=jnp.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ]
        ),
        missing_mask=jnp.asarray(
            [
                [False, False],
                [False, False],
                [True, False],
                [False, False],
            ]
        ),
        has_missing_values=True,
        metadata=VariantMetadata(
            chromosome=np.asarray(["1", "1"]),
            variant_identifiers=np.asarray(["variant_one", "variant_two"]),
            position=np.asarray([1, 2]),
            allele_one=np.asarray(["A", "C"]),
            allele_two=np.asarray(["G", "T"]),
        ),
        allele_one_frequency=jnp.zeros((2,), dtype=jnp.float32),
        observation_count=jnp.asarray([4, 4]),
    )

    logistic_result, allele_frequency, observation_count = compute_logistic_association_with_missing_exclusion(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_chunk=genotype_chunk,
        max_iterations=50,
        tolerance=1.0e-8,
    )

    assert np.asarray(logistic_result.beta).shape == (2,)
    np.testing.assert_array_equal(observation_count, np.asarray([3, 4]))
    np.testing.assert_allclose(allele_frequency, np.asarray([1.0 / 3.0, 0.5]), atol=1e-8)


def test_linear_parity_matches_plink_baseline_subset() -> None:
    """Ensure linear regression matches PLINK on an initial variant subset."""
    require_phase_zero_inputs()

    variant_limit = 64
    result_frame = run_linear_association(
        bed_prefix=BED_PREFIX,
        phenotype_path=DATA_DIRECTORY / "pheno_cont.txt",
        phenotype_name="phenotype_continuous",
        covariate_path=DATA_DIRECTORY / "covariates.txt",
        covariate_names=("age", "sex"),
        chunk_size=32,
        variant_limit=variant_limit,
    )
    baseline_frame = (
        pl.read_csv(CONTINUOUS_BASELINE_PATH, separator="\t")
        .filter(pl.col("TEST") == "ADD")
        .select("ID", "A1", "BETA", "SE", "T_STAT", "P")
        .head(variant_limit)
        .rename({"ID": "variant_identifier", "SE": "baseline_standard_error", "P": "baseline_p_value"})
    )

    joined_frame = result_frame.join(baseline_frame, on="variant_identifier", how="inner").with_columns(
        pl.when(pl.col("A1") == pl.col("allele_one")).then(pl.lit(1.0)).otherwise(pl.lit(-1.0)).alias("alignment_sign"),
    )
    assert joined_frame.height == variant_limit
    beta_values = (joined_frame.get_column("beta") * joined_frame.get_column("alignment_sign")).to_numpy()
    baseline_beta_values = joined_frame.get_column("BETA").to_numpy()
    p_values = joined_frame.get_column("p_value").to_numpy()
    baseline_p_values = joined_frame.get_column("baseline_p_value").to_numpy()
    assert_beta_parity(beta_values, baseline_beta_values)
    assert_log10_p_value_parity(
        observed_p_values=p_values,
        expected_p_values=baseline_p_values,
        max_log10_difference=P_VALUE_MAX_LOG10_ERROR,
    )


def test_logistic_hybrid_parity_matches_plink_baseline_subset() -> None:
    """Ensure logistic hybrid regression matches PLINK baseline rows."""
    require_phase_zero_inputs()

    variant_limit = 64
    result_frame = run_logistic_association(
        bed_prefix=BED_PREFIX,
        phenotype_path=DATA_DIRECTORY / "pheno_bin.txt",
        phenotype_name="phenotype_binary",
        covariate_path=DATA_DIRECTORY / "covariates.txt",
        covariate_names=("age", "sex"),
        chunk_size=32,
        variant_limit=variant_limit,
        max_iterations=50,
        tolerance=1.0e-8,
    )
    baseline_frame = (
        pl.read_csv(BINARY_BASELINE_PATH, separator="\t")
        .filter(pl.col("TEST") == "ADD")
        .with_row_index("row_index")
        .select("row_index", "ID", "A1", "OR", "LOG(OR)_SE", "Z_STAT", "P", "FIRTH?", "ERRCODE")
        .head(variant_limit)
        .with_columns(pl.col("OR").log().alias("baseline_beta"))
        .rename({"ID": "variant_identifier", "LOG(OR)_SE": "baseline_standard_error", "P": "baseline_p_value"})
    )

    joined_frame = (
        result_frame.with_row_index("row_index")
        .join(baseline_frame, on="row_index", how="inner")
        .with_columns(
            pl.when(pl.col("A1") == pl.col("allele_one"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(-1.0))
            .alias("alignment_sign"),
        )
    )
    assert joined_frame.height == variant_limit
    joined_frame = joined_frame.with_columns(
        (pl.col("beta") * pl.col("alignment_sign")).alias("aligned_beta"),
    )
    assert_beta_parity(
        joined_frame.get_column("aligned_beta").to_numpy(),
        joined_frame.get_column("baseline_beta").to_numpy(),
    )
    assert_log10_p_value_parity(
        observed_p_values=joined_frame.get_column("p_value").to_numpy(),
        expected_p_values=joined_frame.get_column("baseline_p_value").to_numpy(),
        max_log10_difference=P_VALUE_MAX_LOG10_ERROR,
    )


def test_cli_writes_linear_output_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the CLI writes a linear result file."""
    require_phase_zero_inputs()

    output_prefix = tmp_path / "phase1_result"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g",
            "--bfile",
            str(BED_PREFIX),
            "--pheno",
            str(DATA_DIRECTORY / "pheno_cont.txt"),
            "--pheno-name",
            "phenotype_continuous",
            "--covar",
            str(DATA_DIRECTORY / "covariates.txt"),
            "--covar-names",
            "age,sex",
            "--glm",
            "linear",
            "--out",
            str(output_prefix),
            "--chunk-size",
            "16",
            "--variant-limit",
            "16",
        ],
    )

    cli_main()

    output_path = output_prefix.with_suffix(".linear.tsv")
    assert output_path.exists()
    output_frame = pl.read_csv(output_path, separator="\t")
    assert output_frame.height == 16
    assert {"variant_identifier", "beta", "p_value"}.issubset(set(output_frame.columns))
