from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from g.cli import build_argument_parser, parse_covariate_names


def test_build_argument_parser() -> None:
    parser = build_argument_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description == "Run GWAS association testing with JAX and Polars."


def test_parse_arguments_success() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--bfile",
            "data/test",
            "--pheno",
            "data/pheno.tsv",
            "--pheno-name",
            "trait1",
            "--covar",
            "data/covar.tsv",
            "--glm",
            "linear",
            "--out",
            "output/results",
        ]
    )

    assert args.bfile == Path("data/test")
    assert args.pheno == Path("data/pheno.tsv")
    assert args.pheno_name == "trait1"
    assert args.covar == Path("data/covar.tsv")
    assert args.glm == "linear"
    assert args.out == Path("output/results")
    assert args.chunk_size == 512
    assert args.device == "cpu"
    assert args.max_iterations == 50
    assert args.tolerance == 1.0e-8
    assert args.variant_limit is None
    assert args.covar_names is None


def test_parse_arguments_missing_required() -> None:
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--bfile",
                "data/test",
                "--pheno",
                "data/pheno.tsv",
                "--pheno-name",
                "trait1",
                "--covar",
                "data/covar.tsv",
                "--out",
                "output/results",
            ]
        )  # missing --glm


def test_parse_arguments_invalid_choice() -> None:
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--bfile",
                "data/test",
                "--pheno",
                "data/pheno.tsv",
                "--pheno-name",
                "trait1",
                "--covar",
                "data/covar.tsv",
                "--glm",
                "poisson",
                "--out",
                "output/results",
            ]
        )  # invalid glm choice


def test_parse_arguments_all_options() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--bfile",
            "data/test",
            "--pheno",
            "data/pheno.tsv",
            "--pheno-name",
            "trait1",
            "--covar",
            "data/covar.tsv",
            "--covar-names",
            "age,sex",
            "--glm",
            "logistic",
            "--out",
            "output/results",
            "--chunk-size",
            "1024",
            "--variant-limit",
            "100",
            "--max-iterations",
            "100",
            "--tolerance",
            "1e-6",
            "--device",
            "gpu",
        ]
    )

    assert args.covar_names == "age,sex"
    assert args.glm == "logistic"
    assert args.chunk_size == 1024
    assert args.variant_limit == 100
    assert args.max_iterations == 100
    assert args.tolerance == 1e-6
    assert args.device == "gpu"


def test_parse_covariate_names_none() -> None:
    """Test parse_covariate_names returns None for None input."""
    result = parse_covariate_names(None)
    assert result is None


def test_parse_covariate_names_empty_string() -> None:
    """Test parse_covariate_names returns None for empty string."""
    result = parse_covariate_names("")
    assert result is None


def test_parse_covariate_names_whitespace_only() -> None:
    """Test parse_covariate_names returns None for whitespace-only string."""
    result = parse_covariate_names("   ,  ,  ")
    assert result is None


def test_parse_covariate_names_single() -> None:
    """Test parse_covariate_names with single name."""
    result = parse_covariate_names("age")
    assert result == ("age",)


def test_parse_covariate_names_multiple() -> None:
    """Test parse_covariate_names with multiple names."""
    result = parse_covariate_names("age,sex,bmi")
    assert result == ("age", "sex", "bmi")


def test_parse_covariate_names_with_whitespace() -> None:
    """Test parse_covariate_names strips whitespace."""
    result = parse_covariate_names("  age , sex ,  bmi  ")
    assert result == ("age", "sex", "bmi")


def test_parse_covariate_names_skips_empty() -> None:
    """Test parse_covariate_names skips empty entries."""
    result = parse_covariate_names("age,,sex,")
    assert result == ("age", "sex")
