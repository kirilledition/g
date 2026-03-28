from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.io.tabular import (
    FAMILY_TABLE_COLUMNS,
    convert_frame_to_float64_jax,
    infer_covariate_names,
    load_aligned_sample_data,
    load_family_table,
    load_phenotype_or_covariate_table,
    recode_binary_phenotype,
)


def test_load_family_table(tmp_path: Path) -> None:
    """Ensure load_family_table parses a valid FAM file correctly."""
    fam_path = tmp_path / "test_dataset.fam"
    fam_content = "family1\tsample1\t0\t0\t1\t-9\nfamily2\tsample2\t0\t0\t2\t-9\nfamily3\tsample3\t0\t0\t1\t-9\n"
    fam_path.write_text(fam_content)

    df = load_family_table(fam_path)

    assert df.height == 3
    assert "sample_index" in df.columns
    assert list(FAMILY_TABLE_COLUMNS) == [c for c in df.columns if c != "sample_index"]
    assert df.get_column("family_identifier").to_list() == ["family1", "family2", "family3"]
    assert df.get_column("individual_identifier").to_list() == ["sample1", "sample2", "sample3"]


def test_load_family_table_missing_file(tmp_path: Path) -> None:
    """Ensure load_family_table raises FileNotFoundError for missing files."""
    fam_path = tmp_path / "missing.fam"

    with pytest.raises(FileNotFoundError):
        load_family_table(fam_path)


def test_load_phenotype_or_covariate_table(tmp_path: Path) -> None:
    """Ensure load_phenotype_or_covariate_table parses a valid table."""
    table_path = tmp_path / "phenotypes.txt"
    table_content = "FID\tIID\tphenotype1\tphenotype2\nf1\ts1\t1.5\t2.0\nf2\ts2\t2.5\tNA\n"
    table_path.write_text(table_content)

    df = load_phenotype_or_covariate_table(table_path)

    assert df.height == 2
    assert df.columns == ["FID", "IID", "phenotype1", "phenotype2"]
    assert df.get_column("phenotype1").to_list() == [1.5, 2.5]
    assert df.get_column("phenotype2").null_count() == 1


def test_infer_covariate_names_basic() -> None:
    """Ensure infer_covariate_names excludes FID and IID columns."""
    covariate_table = pl.DataFrame(
        {
            "FID": ["f1", "f2"],
            "IID": ["s1", "s2"],
            "age": [25, 30],
            "sex": [1, 2],
        }
    )

    names = infer_covariate_names(covariate_table)

    assert names == ("age", "sex")


def test_infer_covariate_names_no_identifiers() -> None:
    """Ensure infer_covariate_names works when no FID/IID columns."""
    covariate_table = pl.DataFrame(
        {
            "age": [25, 30],
            "sex": [1, 2],
        }
    )

    names = infer_covariate_names(covariate_table)

    assert names == ("age", "sex")


def test_infer_covariate_names_empty_raises() -> None:
    """Ensure infer_covariate_names raises when no covariate columns."""
    covariate_table = pl.DataFrame(
        {
            "FID": ["f1", "f2"],
            "IID": ["s1", "s2"],
        }
    )

    with pytest.raises(ValueError, match="at least one non-identifier"):
        infer_covariate_names(covariate_table)


def test_convert_frame_to_float64_jax() -> None:
    """Ensure convert_frame_to_float64_jax converts DataFrame correctly."""
    df = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, 5.0, 6.0],
        }
    )

    result = convert_frame_to_float64_jax(df)

    assert result.shape == (3, 2)
    assert result.dtype == jnp.float64
    np.testing.assert_allclose(result[:, 0], jnp.array([1.0, 2.0, 3.0]))


def test_recode_binary_phenotype_valid() -> None:
    """Ensure recode_binary_phenotype converts PLINK 1/2 to 0/1."""
    phenotype_values = np.array([1.0, 2.0, 1.0, 2.0])

    result = recode_binary_phenotype(phenotype_values)

    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 0.0, 1.0]))


def test_recode_binary_phenotype_invalid_values() -> None:
    """Ensure recode_binary_phenotype raises for invalid values."""
    phenotype_values = np.array([1.0, 2.0, 0.0])

    with pytest.raises(ValueError, match="PLINK values 1 and 2"):
        recode_binary_phenotype(phenotype_values)


def test_recode_binary_phenotype_nan_allowed() -> None:
    """Ensure recode_binary_phenotype handles NaN values."""
    phenotype_values = np.array([1.0, 2.0, np.nan, 1.0])

    with pytest.raises(ValueError):
        recode_binary_phenotype(phenotype_values)


def test_load_aligned_sample_data_continuous(tmp_path: Path) -> None:
    """Test load_aligned_sample_data with continuous phenotype."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\nf2\ts2\t0\t0\t2\t-9\nf3\ts3\t0\t0\t1\t-9\n")

    pheno_path = tmp_path / "pheno.txt"
    pheno_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.5\nf2\ts2\t2.5\nf3\ts3\t3.5\n")

    covar_path = tmp_path / "covar.txt"
    covar_path.write_text("FID\tIID\tage\tsex\nf1\ts1\t25\t1\nf2\ts2\t30\t2\nf3\ts3\t35\t1\n")

    result = load_aligned_sample_data(
        bed_prefix=tmp_path / "test",
        phenotype_path=pheno_path,
        phenotype_name="trait",
        covariate_path=covar_path,
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )

    assert result.sample_indices.shape == (3,)
    assert result.phenotype_vector.shape == (3,)
    assert result.covariate_matrix.shape == (3, 3)
    assert result.covariate_names == ("intercept", "age", "sex")
    assert result.is_binary_trait is False


def test_load_aligned_sample_data_binary(tmp_path: Path) -> None:
    """Test load_aligned_sample_data with binary phenotype."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\nf2\ts2\t0\t0\t2\t-9\n")

    pheno_path = tmp_path / "pheno.txt"
    pheno_path.write_text("FID\tIID\ttrait\nf1\ts1\t1\nf2\ts2\t2\n")

    covar_path = tmp_path / "covar.txt"
    covar_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts2\t30\n")

    result = load_aligned_sample_data(
        bed_prefix=tmp_path / "test",
        phenotype_path=pheno_path,
        phenotype_name="trait",
        covariate_path=covar_path,
        covariate_names=None,
        is_binary_trait=True,
    )

    np.testing.assert_array_equal(result.phenotype_vector, np.array([0.0, 1.0]))
    assert result.is_binary_trait is True


def test_load_aligned_sample_data_missing_phenotype_column(tmp_path: Path) -> None:
    """Test load_aligned_sample_data raises for missing phenotype column."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\n")

    pheno_path = tmp_path / "pheno.txt"
    pheno_path.write_text("FID\tIID\tother\nf1\ts1\t1.0\n")

    covar_path = tmp_path / "covar.txt"
    covar_path.write_text("FID\tIID\tage\nf1\ts1\t25\n")

    with pytest.raises(ValueError, match="Phenotype column"):
        load_aligned_sample_data(
            bed_prefix=tmp_path / "test",
            phenotype_path=pheno_path,
            phenotype_name="trait",
            covariate_path=covar_path,
            covariate_names=("age",),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_missing_covariate_column(tmp_path: Path) -> None:
    """Test load_aligned_sample_data raises for missing covariate columns."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\n")

    pheno_path = tmp_path / "pheno.txt"
    pheno_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\n")

    covar_path = tmp_path / "covar.txt"
    covar_path.write_text("FID\tIID\tage\nf1\ts1\t25\n")

    with pytest.raises(ValueError, match="Covariate columns are missing"):
        load_aligned_sample_data(
            bed_prefix=tmp_path / "test",
            phenotype_path=pheno_path,
            phenotype_name="trait",
            covariate_path=covar_path,
            covariate_names=("age", "sex"),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_no_aligned_samples(tmp_path: Path) -> None:
    """Test load_aligned_sample_data raises when no samples align."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts1\t0\t0\t1\t-9\n")

    pheno_path = tmp_path / "pheno.txt"
    pheno_path.write_text("FID\tIID\ttrait\nf2\ts2\t1.0\n")

    covar_path = tmp_path / "covar.txt"
    covar_path.write_text("FID\tIID\tage\nf2\ts2\t25\n")

    with pytest.raises(ValueError, match="No aligned samples"):
        load_aligned_sample_data(
            bed_prefix=tmp_path / "test",
            phenotype_path=pheno_path,
            phenotype_name="trait",
            covariate_path=covar_path,
            covariate_names=("age",),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_sorts_by_family_order_and_drops_null_rows(tmp_path: Path) -> None:
    """Ensure alignment follows FAM order and excludes rows with null phenotype or covariates."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text("f1\ts2\t0\t0\t1\t-9\nf1\ts1\t0\t0\t1\t-9\nf1\ts3\t0\t0\t1\t-9\n")

    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts3\t3.0\nf1\ts1\t1.0\nf1\ts2\t2.0\n")

    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\tsex\nf1\ts1\t25\t1\nf1\ts2\tNA\t2\nf1\ts3\t35\t1\n")

    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=tmp_path / "test",
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=covariate_path,
        covariate_names=None,
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(aligned_sample_data.sample_indices, np.array([1, 2]))
    np.testing.assert_array_equal(aligned_sample_data.individual_identifiers, np.array(["s1", "s3"]))
    np.testing.assert_allclose(np.asarray(aligned_sample_data.phenotype_vector), np.array([1.0, 3.0]), atol=0.0)
    np.testing.assert_allclose(
        np.asarray(aligned_sample_data.covariate_matrix),
        np.array([[1.0, 25.0, 1.0], [1.0, 35.0, 1.0]]),
        atol=0.0,
    )
