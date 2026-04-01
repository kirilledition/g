from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from g.api import DEFAULT_LINEAR_CHUNK_SIZE, DEFAULT_LOGISTIC_CHUNK_SIZE, RunArtifacts
from g.cli import app, resolve_chunk_size

runner = CliRunner()


def test_root_command_without_arguments_shows_help() -> None:
    """Ensure the CLI shows help when invoked without arguments."""
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Blazing fast GWAS engine" in result.output
    assert "linear" in result.output
    assert "logistic" in result.output


def test_root_help_renders_without_style_errors() -> None:
    """Ensure help rendering stays plain and free of Rich panel boxes."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MissingStyle" not in result.output
    assert "Usage:" in result.output
    assert "╭" not in result.output
    assert "╰" not in result.output


def test_resolve_chunk_size_uses_linear_default() -> None:
    """Ensure the linear path uses the tuned default chunk size."""
    assert resolve_chunk_size(None, "linear") == DEFAULT_LINEAR_CHUNK_SIZE


def test_resolve_chunk_size_uses_logistic_default() -> None:
    """Ensure the logistic path keeps the tuned default chunk size."""
    assert resolve_chunk_size(None, "logistic") == DEFAULT_LOGISTIC_CHUNK_SIZE


def test_resolve_chunk_size_preserves_explicit_override() -> None:
    """Ensure explicit chunk sizes override model-specific defaults."""
    assert resolve_chunk_size(1024, "linear") == 1024


def test_linear_command_dispatches_api_call() -> None:
    """Ensure the linear subcommand forwards arguments to the public API."""
    with patch(
        "g.cli.run_linear_api",
        return_value=RunArtifacts(sumstats_tsv=Path("results/output.linear.tsv")),
    ) as mock_run_linear_api:
        result = runner.invoke(
            app,
            [
                "linear",
                "--bfile",
                "dataset",
                "--pheno",
                "phenotype.tsv",
                "--pheno-name",
                "trait",
                "--covar",
                "covariates.tsv",
                "--covar-names",
                "age,sex",
                "--out",
                "results/output",
                "--device",
                "gpu",
            ],
        )

    assert result.exit_code == 0
    assert str(Path("results/output.linear.tsv")) in result.output
    assert mock_run_linear_api.call_args.kwargs["covar_names"] == ("age", "sex")
    compute_config = mock_run_linear_api.call_args.kwargs["compute"]
    assert compute_config.device == "gpu"
    assert compute_config.chunk_size == DEFAULT_LINEAR_CHUNK_SIZE


def test_linear_command_supports_intercept_only_run() -> None:
    """Ensure the linear subcommand allows runs without a covariate table."""
    with patch(
        "g.cli.run_linear_api",
        return_value=RunArtifacts(sumstats_tsv=Path("results/output.linear.tsv")),
    ) as mock_run_linear_api:
        result = runner.invoke(
            app,
            [
                "linear",
                "--bfile",
                "dataset",
                "--pheno",
                "phenotype.tsv",
                "--pheno-name",
                "trait",
                "--out",
                "results/output",
            ],
        )

    assert result.exit_code == 0
    assert mock_run_linear_api.call_args.kwargs["covar"] is None
    assert mock_run_linear_api.call_args.kwargs["covar_names"] is None


def test_linear_command_supports_bgen_input() -> None:
    """Ensure the linear subcommand can dispatch a BGEN-backed run."""
    with patch(
        "g.cli.run_linear_api",
        return_value=RunArtifacts(sumstats_tsv=Path("results/output.linear.tsv")),
    ) as mock_run_linear_api:
        result = runner.invoke(
            app,
            [
                "linear",
                "--bgen",
                "dataset.bgen",
                "--pheno",
                "phenotype.tsv",
                "--pheno-name",
                "trait",
                "--out",
                "results/output",
            ],
        )

    assert result.exit_code == 0
    assert mock_run_linear_api.call_args.kwargs["bfile"] is None
    assert mock_run_linear_api.call_args.kwargs["bgen"] == Path("dataset.bgen")


def test_linear_subcommand_without_options_shows_help() -> None:
    """Ensure the linear subcommand shows help instead of a usage error."""
    result = runner.invoke(app, ["linear"])
    assert result.exit_code == 2
    assert "Run a linear association scan" in result.output
    assert "--bfile" in result.output
    assert "--pheno-name" in result.output


def test_logistic_subcommand_without_options_shows_help() -> None:
    """Ensure the logistic subcommand shows help instead of a usage error."""
    result = runner.invoke(app, ["logistic"])
    assert result.exit_code == 2
    assert "Run a logistic association scan" in result.output
    assert "--max-iterations" in result.output
    assert "--firth" in result.output


def test_logistic_command_dispatches_api_call() -> None:
    """Ensure the logistic subcommand forwards model-specific options to the public API."""
    with patch(
        "g.cli.run_logistic_api",
        return_value=RunArtifacts(sumstats_tsv=Path("results/output.logistic.tsv")),
    ) as mock_run_logistic_api:
        result = runner.invoke(
            app,
            [
                "logistic",
                "--bfile",
                "dataset",
                "--pheno",
                "phenotype.tsv",
                "--pheno-name",
                "trait",
                "--covar",
                "covariates.tsv",
                "--out",
                "results/output",
                "--max-iterations",
                "75",
                "--tolerance",
                "1e-6",
                "--no-firth",
            ],
        )

    assert result.exit_code == 0
    solver_config = mock_run_logistic_api.call_args.kwargs["solver"]
    assert solver_config.max_iterations == 75
    assert solver_config.tolerance == 1.0e-6
    assert solver_config.firth_fallback is False
    compute_config = mock_run_logistic_api.call_args.kwargs["compute"]
    assert compute_config.chunk_size == DEFAULT_LOGISTIC_CHUNK_SIZE


def test_subcommand_requires_known_name() -> None:
    """Ensure invalid subcommands fail with a helpful error."""
    result = runner.invoke(app, ["poisson"])
    assert result.exit_code != 0
    assert "No such command 'poisson'" in result.output
