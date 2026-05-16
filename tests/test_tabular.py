from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from g.io import samples


def build_sample_table(sample_identifiers: tuple[str, ...]) -> pl.DataFrame:
    """Build a normalized sample table for alignment tests."""
    return pl.DataFrame(
        {
            "family_identifier": list(sample_identifiers),
            "individual_identifier": list(sample_identifiers),
        }
    ).with_row_index("sample_index")


def test_load_phenotype_or_covariate_table(tmp_path: Path) -> None:
    """Ensure load_phenotype_or_covariate_table parses a valid table."""
    table_path = tmp_path / "phenotypes.txt"
    table_content = "FID\tIID\tphenotype1\tphenotype2\nf1\ts1\t1.5\t2.0\nf2\ts2\t2.5\tNA\n"
    table_path.write_text(table_content)

    data_frame = samples.load_phenotype_or_covariate_table(table_path)

    assert data_frame.height == 2
    assert data_frame.columns == ["FID", "IID", "phenotype1", "phenotype2"]
    assert data_frame.get_column("phenotype1").to_list() == [1.5, 2.5]
    assert data_frame.get_column("phenotype2").null_count() == 1


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

    names = samples.infer_covariate_names(covariate_table)

    assert names == ("age", "sex")


def test_infer_covariate_names_no_identifiers() -> None:
    """Ensure infer_covariate_names works when no FID/IID columns."""
    covariate_table = pl.DataFrame(
        {
            "age": [25, 30],
            "sex": [1, 2],
        }
    )

    names = samples.infer_covariate_names(covariate_table)

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
        samples.infer_covariate_names(covariate_table)


def test_convert_frame_to_float32_jax() -> None:
    """Ensure convert_frame_to_float32_jax converts DataFrame correctly."""
    data_frame = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, 5.0, 6.0],
        }
    )

    result = samples.convert_frame_to_float32_jax(data_frame)

    assert result.shape == (3, 2)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result[:, 0], jnp.array([1.0, 2.0, 3.0]))


def test_recode_binary_phenotype_valid() -> None:
    """Ensure recode_binary_phenotype converts 1/2 to 0/1."""
    phenotype_values = np.array([1.0, 2.0, 1.0, 2.0])

    result = samples.recode_binary_phenotype(phenotype_values)

    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 0.0, 1.0]))


def test_recode_binary_phenotype_invalid_values() -> None:
    """Ensure recode_binary_phenotype raises for invalid values."""
    phenotype_values = np.array([1.0, 2.0, 0.0])

    with pytest.raises(ValueError, match="values 1 and 2"):
        samples.recode_binary_phenotype(phenotype_values)


def test_recode_binary_phenotype_nan_rejected() -> None:
    """Ensure recode_binary_phenotype rejects NaN values."""
    phenotype_values = np.array([1.0, 2.0, np.nan, 1.0])

    with pytest.raises(ValueError):
        samples.recode_binary_phenotype(phenotype_values)


def test_load_aligned_sample_data_continuous(tmp_path: Path) -> None:
    """Test sample alignment with continuous phenotype."""
    sample_table = build_sample_table(("s1", "s2", "s3"))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.5\nf2\ts2\t2.5\nf3\ts3\t3.5\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\tsex\nf1\ts1\t25\t1\nf2\ts2\t30\t2\nf3\ts3\t35\t1\n")

    result = samples.load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=covariate_path,
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )

    assert result.sample_indices.shape == (3,)
    assert result.phenotype_vector.shape == (3,)
    assert result.covariate_matrix.shape == (3, 3)
    assert result.covariate_names == ("intercept", "age", "sex")
    assert result.is_binary_trait is False


def test_load_aligned_sample_data_binary(tmp_path: Path) -> None:
    """Test sample alignment with binary phenotype."""
    sample_table = build_sample_table(("s1", "s2"))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1\nf2\ts2\t2\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\nf2\ts2\t30\n")

    result = samples.load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=covariate_path,
        covariate_names=None,
        is_binary_trait=True,
    )

    np.testing.assert_array_equal(result.phenotype_vector, np.array([0.0, 1.0]))
    assert result.is_binary_trait is True


def test_load_aligned_sample_data_from_individual_identifier_table_matches_iid_only(tmp_path: Path) -> None:
    """Ensure sample tables can align on IID without matching FID."""
    sample_table = build_sample_table(("sample1", "sample2"))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nfamily1\tsample1\t1.5\nfamily2\tsample2\t2.5\n")

    result = samples.load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(result.sample_indices, np.array([0, 1]))
    np.testing.assert_array_equal(result.individual_identifiers, np.array(["sample1", "sample2"]))
    np.testing.assert_allclose(np.asarray(result.phenotype_vector), np.array([1.5, 2.5]), atol=0.0)


def test_load_aligned_sample_data_missing_phenotype_column(tmp_path: Path) -> None:
    """Test aligned sample loading raises for missing phenotype column."""
    sample_table = build_sample_table(("s1",))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\tother\nf1\ts1\t1.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\n")

    with pytest.raises(ValueError, match="Phenotype column"):
        samples.load_aligned_sample_data_from_individual_identifier_table(
            sample_table=sample_table,
            phenotype_path=phenotype_path,
            phenotype_name="trait",
            covariate_path=covariate_path,
            covariate_names=("age",),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_missing_covariate_column(tmp_path: Path) -> None:
    """Test aligned sample loading raises for missing covariate columns."""
    sample_table = build_sample_table(("s1",))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf1\ts1\t25\n")

    with pytest.raises(ValueError, match="Covariate columns are missing"):
        samples.load_aligned_sample_data_from_individual_identifier_table(
            sample_table=sample_table,
            phenotype_path=phenotype_path,
            phenotype_name="trait",
            covariate_path=covariate_path,
            covariate_names=("age", "sex"),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_no_aligned_samples(tmp_path: Path) -> None:
    """Test aligned sample loading raises when no samples align."""
    sample_table = build_sample_table(("s1",))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf2\ts2\t1.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\nf2\ts2\t25\n")

    with pytest.raises(ValueError, match="No aligned samples"):
        samples.load_aligned_sample_data_from_individual_identifier_table(
            sample_table=sample_table,
            phenotype_path=phenotype_path,
            phenotype_name="trait",
            covariate_path=covariate_path,
            covariate_names=("age",),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_supports_intercept_only_runs(tmp_path: Path) -> None:
    """Ensure aligned sample loading supports runs without an external covariate table."""
    sample_table = build_sample_table(("s1", "s2"))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.5\nf2\ts2\t2.5\n")

    aligned_sample_data = samples.load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=False,
    )

    assert aligned_sample_data.covariate_names == ("intercept",)
    np.testing.assert_allclose(np.asarray(aligned_sample_data.covariate_matrix), np.array([[1.0], [1.0]]), atol=0.0)


def test_load_aligned_sample_data_rejects_covariate_names_without_table(tmp_path: Path) -> None:
    """Ensure explicit covariate names require a covariate table."""
    sample_table = build_sample_table(("s1",))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts1\t1.5\n")

    with pytest.raises(ValueError, match="Covariate names cannot be provided without a covariate table"):
        samples.load_aligned_sample_data_from_individual_identifier_table(
            sample_table=sample_table,
            phenotype_path=phenotype_path,
            phenotype_name="trait",
            covariate_path=None,
            covariate_names=("age",),
            is_binary_trait=False,
        )


def test_load_aligned_sample_data_sorts_by_sample_order_and_drops_null_rows(tmp_path: Path) -> None:
    """Ensure alignment follows source order and excludes rows with null phenotype or covariates."""
    sample_table = build_sample_table(("s2", "s1", "s3"))
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("FID\tIID\ttrait\nf1\ts3\t3.0\nf1\ts1\t1.0\nf1\ts2\t2.0\n")
    covariate_path = tmp_path / "covar.txt"
    covariate_path.write_text("FID\tIID\tage\tsex\nf1\ts1\t25\t1\nf1\ts2\tNA\t2\nf1\ts3\t35\t1\n")

    aligned_sample_data = samples.load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
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
