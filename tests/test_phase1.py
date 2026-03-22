from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from g.cli import main as cli_main
from g.engine import run_linear_association, run_logistic_association
from g.io.plink import iter_genotype_chunks
from g.io.tabular import load_aligned_sample_data

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = REPOSITORY_ROOT / "data"
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
        .select("ID", "BETA", "SE", "T_STAT", "P")
        .head(variant_limit)
        .rename({"ID": "variant_identifier", "SE": "baseline_standard_error", "P": "baseline_p_value"})
    )

    joined_frame = result_frame.join(baseline_frame, on="variant_identifier", how="inner")
    assert joined_frame.height == variant_limit
    beta_values = joined_frame.get_column("beta").to_numpy()
    baseline_beta_values = joined_frame.get_column("BETA").to_numpy()
    t_statistic_values = joined_frame.get_column("t_statistic").to_numpy()
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


def test_logistic_parity_matches_non_firth_plink_baseline_subset() -> None:
    """Ensure logistic regression matches non-Firth PLINK baseline rows."""
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
        .filter((pl.col("TEST") == "ADD") & (pl.col("FIRTH?") == "N"))
        .select("ID", "OR", "LOG(OR)_SE", "Z_STAT", "P")
        .head(variant_limit)
        .with_columns(pl.col("OR").log().alias("baseline_beta"))
        .rename({"ID": "variant_identifier", "LOG(OR)_SE": "baseline_standard_error", "P": "baseline_p_value"})
    )

    joined_frame = result_frame.join(baseline_frame, on="variant_identifier", how="inner")
    assert joined_frame.height >= 48
    np.testing.assert_allclose(
        joined_frame.get_column("beta").to_numpy(),
        joined_frame.get_column("baseline_beta").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        joined_frame.get_column("standard_error").to_numpy(),
        joined_frame.get_column("baseline_standard_error").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        joined_frame.get_column("z_statistic").to_numpy(),
        joined_frame.get_column("Z_STAT").to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        joined_frame.get_column("p_value").to_numpy(),
        joined_frame.get_column("baseline_p_value").to_numpy(),
        atol=1e-4,
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
