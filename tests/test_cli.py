from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from g.api import DEFAULT_ARROW_PAYLOAD_BATCH_SIZE, DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE, RunArtifacts
from g.cli import app, main, print_success_message, resolve_arrow_payload_batch_size, resolve_chunk_size
from g.types import Device

runner = CliRunner()


def test_root_command_without_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Blazing fast REGENIE step 2 GWAS engine." in result.output
    assert "regenie2-linear" in result.output
    assert "\n  linear" not in result.output
    assert "\n  logistic" not in result.output


def test_root_help_renders_without_style_errors() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MissingStyle" not in result.output
    assert "Usage:" in result.output
    assert "╭" not in result.output
    assert "╰" not in result.output


def test_resolve_chunk_size_uses_regenie_default() -> None:
    assert resolve_chunk_size(None) == DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE


def test_resolve_chunk_size_preserves_explicit_override() -> None:
    assert resolve_chunk_size(1024) == 1024


def test_resolve_arrow_payload_batch_size_uses_default() -> None:
    assert resolve_arrow_payload_batch_size(None) == DEFAULT_ARROW_PAYLOAD_BATCH_SIZE


def test_resolve_arrow_payload_batch_size_preserves_explicit_override() -> None:
    assert resolve_arrow_payload_batch_size(4) == 4


def test_removed_linear_command_is_unknown() -> None:
    result = runner.invoke(app, ["linear", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_removed_logistic_command_is_unknown() -> None:
    result = runner.invoke(app, ["logistic", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_regenie2_linear_subcommand_without_options_shows_help() -> None:
    result = runner.invoke(app, ["regenie2-linear"])
    assert result.exit_code == 2
    assert "Run a REGENIE step 2 linear association scan" in result.output
    assert "--pred" in result.output
    assert "--bgen" in result.output


def test_regenie2_linear_command_dispatches_api_call() -> None:
    with patch(
        "g.cli.run_regenie2_linear_api",
        return_value=RunArtifacts(
            output_run_directory=Path("results/output.regenie2_linear.run"),
            final_parquet=Path("results/output.regenie2_linear.run/final.parquet"),
        ),
    ) as mock_run_regenie2_linear_api:
        result = runner.invoke(
            app,
            [
                "regenie2-linear",
                "--bgen",
                "dataset.bgen",
                "--sample",
                "dataset.sample",
                "--pheno",
                "phenotype.tsv",
                "--pheno-name",
                "trait",
                "--covar",
                "covariates.tsv",
                "--covar-names",
                "age,sex",
                "--pred",
                "predictions.list",
                "--out",
                "results/output",
                "--device",
                "gpu",
            ],
        )

    assert result.exit_code == 0
    assert str(Path("results/output.regenie2_linear.run")) in result.output
    assert str(Path("results/output.regenie2_linear.run/final.parquet")) in result.output
    assert mock_run_regenie2_linear_api.call_args.kwargs["covar_names"] == ("age", "sex")
    compute_config = mock_run_regenie2_linear_api.call_args.kwargs["compute"]
    assert compute_config.device == Device.GPU
    assert compute_config.chunk_size == DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE


def test_print_success_message_reports_run_directory_and_parquet(capsys: typing.Any) -> None:
    print_success_message(
        RunArtifacts(
            output_run_directory=Path("results.regenie2_linear.run"),
            final_parquet=Path("results.regenie2_linear.run/final.parquet"),
        )
    )
    captured = capsys.readouterr()
    assert "results.regenie2_linear.run" in captured.out
    assert "final.parquet" in captured.out


def test_main_dispatches_to_typer_app() -> None:
    with patch("g.cli.app") as mock_app:
        main()
    mock_app.assert_called_once_with()
