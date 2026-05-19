from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from g import api, types
from g.cli import app, main, print_success_message

runner = CliRunner()


def test_root_command_without_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Blazing fast REGENIE step 2 GWAS engine." in result.output
    assert "regenie" in result.output
    assert "config" in result.output
    assert "regenie2" not in result.output
    assert "\n  linear" not in result.output
    assert "\n  logistic" not in result.output


def test_root_help_renders_without_style_errors() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "MissingStyle" not in result.output
    assert "Usage:" in result.output
    assert "╭" not in result.output
    assert "╰" not in result.output


def test_regenie_command_dispatches_config_api() -> None:
    with patch(
        "g.cli.run_regenie_api",
        return_value=api.RunArtifacts(
            output_run_directory=Path("results/output.g/trait.regenie2_linear.run"),
            final_regenie=Path("results/output_trait.regenie"),
        ),
    ) as mock_run_regenie_api:
        result = runner.invoke(
            app,
            [
                "regenie",
                "--step",
                "2",
                "--bgen",
                "dataset.bgen",
                "--sample",
                "dataset.sample",
                "--phenoFile",
                "phenotype.tsv",
                "--phenoCol",
                "trait",
                "--covarFile",
                "covariates.tsv",
                "--covarColList",
                "age,sex",
                "--pred",
                "predictions.list",
                "--out",
                "results/output",
                "--qt",
                "--bsize",
                "4096",
                "--g-device",
                "gpu",
                "--g-output-format",
                "regenie",
            ],
        )

    assert result.exit_code == 0
    regenie_config = mock_run_regenie_api.call_args.args[0]
    assert regenie_config.input.pheno_columns == ("trait",)
    assert regenie_config.input.covar_columns == ("age", "sex")
    assert regenie_config.trait.bsize == 4096
    assert regenie_config.g_compute.device == types.Device.GPU
    assert "output_trait.regenie" in result.output


def test_regenie_command_rejects_unsupported_regenie_flag() -> None:
    result = runner.invoke(
        app,
        [
            "regenie",
            "--step",
            "2",
            "--pgen",
            "dataset",
            "--phenoFile",
            "phenotype.tsv",
            "--phenoCol",
            "trait",
            "--pred",
            "predictions.list",
            "--out",
            "results/output",
        ],
    )

    assert result.exit_code != 0
    assert "--pgen is a valid REGENIE option" in result.output


def test_regenie_command_rejects_removed_duplicate_iid_flag() -> None:
    result = runner.invoke(app, ["regenie", "--g-allow-duplicate-iid-alignment"])

    assert result.exit_code != 0
    assert "No such option: --g-allow-duplicate-iid-alignment" in result.output


def test_regenie_command_applies_toml_then_explicit_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[input]",
                'bgen = "dataset.bgen"',
                'phenoFile = "phenotype.tsv"',
                'phenoCol = "trait"',
                'pred = "predictions.list"',
                "[trait]",
                "step = 2",
                "bt = true",
                "bsize = 1024",
                "[binary]",
                "firth = true",
                "approx = true",
                "[output]",
                'out = "results/output"',
                "[g.output]",
                'format = "parquet"',
            ]
        ),
        encoding="utf-8",
    )

    with patch("g.cli.run_regenie_api", return_value=api.RunArtifacts()) as mock_run_regenie_api:
        result = runner.invoke(app, ["regenie", "--config", str(config_path), "--qt", "--bsize", "4096"])

    assert result.exit_code == 0
    regenie_config = mock_run_regenie_api.call_args.args[0]
    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert regenie_config.trait.bsize == 4096
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET


def test_config_subcommands_render_and_validate(tmp_path: Path) -> None:
    config_path = tmp_path / "regenie.toml"

    init_result = runner.invoke(app, ["config", "init", "--out", str(config_path)])

    assert init_result.exit_code == 0
    assert config_path.exists()
    validate_result = runner.invoke(app, ["config", "validate", str(config_path)])
    assert validate_result.exit_code == 0
    explain_result = runner.invoke(app, ["config", "explain", "bgen"])
    assert explain_result.exit_code == 0
    assert "supported" in explain_result.output


def test_legacy_commands_are_not_registered() -> None:
    for command_name in ["regenie2", "regenie2-linear", "regenie2-warm-cache", "linear", "logistic"]:
        result = runner.invoke(app, [command_name, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_print_success_message_reports_run_directory_outputs(capsys: typing.Any) -> None:
    print_success_message(
        api.RunArtifacts(
            output_run_directory=Path("results/output.g/trait.regenie2_linear.run"),
            final_regenie=Path("results/output_trait.regenie"),
            final_parquet=Path("results/output.g/trait.regenie2_linear.run/final.parquet"),
        )
    )
    captured = capsys.readouterr()
    assert "results/output.g/trait.regenie2_linear.run" in captured.out
    assert "output_trait.regenie" in captured.out
    assert "final.parquet" in captured.out


def test_main_dispatches_to_click_app() -> None:
    with patch("g.cli.app") as mock_app:
        main()
    mock_app.assert_called_once_with()
