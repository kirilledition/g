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


def test_native_genotype_chunk_reader_matches_python_path() -> None:
    """Ensure Rust BED chunk ingestion matches the Python reader path."""
    require_phase_zero_inputs()

    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=BED_PREFIX,
        phenotype_path=DATA_DIRECTORY / "pheno_cont.txt",
        phenotype_name="phenotype_continuous",
        covariate_path=DATA_DIRECTORY / "covariates.txt",
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )

    python_chunks = list(
        iter_genotype_chunks(
            bed_prefix=BED_PREFIX,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=16,
            variant_limit=64,
            use_native_reader=False,
        )
    )
    native_chunks = list(
        iter_genotype_chunks(
            bed_prefix=BED_PREFIX,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=16,
            variant_limit=64,
            use_native_reader=True,
        )
    )

    assert len(python_chunks) == len(native_chunks)
    for python_chunk, native_chunk in zip(python_chunks, native_chunks, strict=True):
        np.testing.assert_allclose(np.asarray(native_chunk.genotypes), np.asarray(python_chunk.genotypes), atol=0.0)
        np.testing.assert_array_equal(np.asarray(native_chunk.missing_mask), np.asarray(python_chunk.missing_mask))
        np.testing.assert_array_equal(native_chunk.metadata.chromosome, python_chunk.metadata.chromosome)
        np.testing.assert_array_equal(
            native_chunk.metadata.variant_identifiers, python_chunk.metadata.variant_identifiers
        )
        np.testing.assert_array_equal(native_chunk.metadata.position, python_chunk.metadata.position)
        np.testing.assert_array_equal(native_chunk.metadata.allele_one, python_chunk.metadata.allele_one)
        np.testing.assert_array_equal(native_chunk.metadata.allele_two, python_chunk.metadata.allele_two)
        np.testing.assert_allclose(
            np.asarray(native_chunk.allele_one_frequency),
            np.asarray(python_chunk.allele_one_frequency),
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.asarray(native_chunk.observation_count),
            np.asarray(python_chunk.observation_count),
        )


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
        metadata=VariantMetadata(
            chromosome=np.asarray(["1", "1"]),
            variant_identifiers=np.asarray(["variant_one", "variant_two"]),
            position=np.asarray([1, 2]),
            allele_one=np.asarray(["A", "C"]),
            allele_two=np.asarray(["G", "T"]),
        ),
        allele_one_frequency=jnp.asarray([0.0, 0.0]),
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
    t_statistic_values = (joined_frame.get_column("t_statistic") * joined_frame.get_column("alignment_sign")).to_numpy()
    baseline_t_statistic_values = joined_frame.get_column("T_STAT").to_numpy()
    p_values = joined_frame.get_column("p_value").to_numpy()
    baseline_p_values = joined_frame.get_column("baseline_p_value").to_numpy()
    np.testing.assert_allclose(
        beta_values,
        baseline_beta_values,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        joined_frame.get_column("standard_error").to_numpy(),
        joined_frame.get_column("baseline_standard_error").to_numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        t_statistic_values,
        baseline_t_statistic_values,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        p_values,
        baseline_p_values,
        atol=1e-5,
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
    assert joined_frame.get_column("firth_flag").to_list() == joined_frame.get_column("FIRTH?").to_list()
    assert joined_frame.get_column("error_code").to_list() == joined_frame.get_column("ERRCODE").to_list()
    joined_frame = joined_frame.with_columns(
        (pl.col("beta") * pl.col("alignment_sign")).alias("aligned_beta"),
        (pl.col("z_statistic") * pl.col("alignment_sign")).alias("aligned_z"),
    )
    non_firth_joined_frame = joined_frame.filter(pl.col("FIRTH?") == "N")
    firth_joined_frame = joined_frame.filter(pl.col("FIRTH?") == "Y")
    np.testing.assert_allclose(
        non_firth_joined_frame.get_column("aligned_beta").to_numpy(),
        non_firth_joined_frame.get_column("baseline_beta").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        non_firth_joined_frame.get_column("standard_error").to_numpy(),
        non_firth_joined_frame.get_column("baseline_standard_error").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        non_firth_joined_frame.get_column("aligned_z").to_numpy(),
        non_firth_joined_frame.get_column("Z_STAT").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        non_firth_joined_frame.get_column("p_value").to_numpy(),
        non_firth_joined_frame.get_column("baseline_p_value").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        firth_joined_frame.get_column("aligned_beta").to_numpy(),
        firth_joined_frame.get_column("baseline_beta").to_numpy(),
        atol=5e-4,
    )
    np.testing.assert_allclose(
        firth_joined_frame.get_column("standard_error").to_numpy(),
        firth_joined_frame.get_column("baseline_standard_error").to_numpy(),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        firth_joined_frame.get_column("aligned_z").to_numpy(),
        firth_joined_frame.get_column("Z_STAT").to_numpy(),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        firth_joined_frame.get_column("p_value").to_numpy(),
        firth_joined_frame.get_column("baseline_p_value").to_numpy(),
        atol=1e-3,
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
