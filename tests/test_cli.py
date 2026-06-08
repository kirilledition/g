from __future__ import annotations

import signal
import typing
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from g import api, types
from g.cli import (
    app,
    main,
    print_success_message,
    print_warm_cache_message,
    regenie_main,
    resolve_trusted_bgen_validation_mode,
)
from g.engine import shutdown
from g.interface import config, config_layers, options

runner = CliRunner()


@dataclass(frozen=True)
class WarmShape:
    sample_count: int
    variant_count: int


@dataclass(frozen=True)
class WarmReport:
    warmed_shapes: tuple[WarmShape, ...]


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
        "g.cli.api.regenie",
        return_value=api.RunArtifacts(
            output_run_directory=Path("results/output.g/trait.regenie2_linear.run"),
            final_dataset=Path("results/output.g/trait.regenie2_linear.run/parts"),
            final_parquet=Path("results/output.g/trait.regenie2_linear.run/final.parquet"),
        ),
    ) as mock_regenie_api:
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
                "parquet",
                "--g-log-filter",
                "g=info",
                "--g-log-file",
                "logs/g.jsonl",
                "--no-g-log-stderr",
                "--g-telemetry",
                "profile",
                "--g-log-dir",
                "logs",
                "--g-progress-interval-seconds",
                "2",
                "--g-progress-interval-chunks",
                "3",
                "--g-profile-summary-json",
                "logs/profile.summary.json",
                "--g-trace-file",
                "logs/trace.jsonl",
                "--g-trace-filter",
                "g=trace",
                "--g-trace-event-cap",
                "2048",
                "--g-log-queue-size",
                "1024",
                "--no-g-log-lossy",
                "--g-include-source-location",
                "--g-include-span-events",
            ],
        )

    assert result.exit_code == 0
    regenie_config = mock_regenie_api.call_args.args[0]
    assert regenie_config.input.pheno_columns == ("trait",)
    assert regenie_config.input.covar_columns == ("age", "sex")
    assert regenie_config.trait.bsize == 4096
    assert regenie_config.g_compute.device == types.Device.GPU
    assert regenie_config.g_diagnostics.log_filter == "g=info"
    assert regenie_config.g_diagnostics.log_file == Path("logs/g.jsonl")
    assert regenie_config.g_diagnostics.log_stderr is False
    assert regenie_config.g_diagnostics.telemetry == types.TelemetryMode.PROFILE
    assert regenie_config.g_diagnostics.log_dir == Path("logs")
    assert regenie_config.g_diagnostics.progress_interval_seconds == 2
    assert regenie_config.g_diagnostics.progress_interval_chunks == 3
    assert regenie_config.g_diagnostics.profile_summary_json == Path("logs/profile.summary.json")
    assert regenie_config.g_diagnostics.trace_file == Path("logs/trace.jsonl")
    assert regenie_config.g_diagnostics.trace_filter == "g=trace"
    assert regenie_config.g_diagnostics.trace_event_cap == 2048
    assert regenie_config.g_diagnostics.log_queue_size == 1024
    assert regenie_config.g_diagnostics.log_lossy is False
    assert regenie_config.g_diagnostics.include_source_location is True
    assert regenie_config.g_diagnostics.include_span_events is True
    assert "Parquet dataset saved" in result.output
    assert "final.parquet" in result.output


def test_regenie_command_accepts_regenie_text_output_format() -> None:
    with patch(
        "g.cli.api.regenie",
        return_value=api.RunArtifacts(
            output_run_directory=Path("results/output.g/trait.regenie2_linear.run"),
            final_regenie=Path("results/output.g/trait.regenie2_linear.run/final.regenie"),
        ),
    ) as mock_regenie_api:
        result = runner.invoke(
            app,
            [
                "regenie",
                "--step",
                "2",
                "--bgen",
                "dataset.bgen",
                "--phenoFile",
                "phenotype.tsv",
                "--phenoCol",
                "trait",
                "--pred",
                "predictions.list",
                "--out",
                "results/output",
                "--qt",
                "--g-output-format",
                "regenie",
            ],
        )

    assert result.exit_code == 0
    regenie_config = mock_regenie_api.call_args.args[0]
    assert regenie_config.g_output.format == types.OutputFormat.REGENIE
    assert "REGENIE text output saved" in result.output
    assert "final.regenie" in result.output


def test_regenie_command_loads_packaged_default_toml() -> None:
    with patch("g.cli.api.regenie", return_value=api.RunArtifacts()) as mock_regenie_api:
        result = runner.invoke(
            app,
            [
                "regenie",
                "--bgen",
                "dataset.bgen",
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

    assert result.exit_code == 0
    regenie_config = mock_regenie_api.call_args.args[0]
    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert regenie_config.trait.bsize == config.load_packaged_config().trait.bsize
    assert regenie_config.g_compute.device == types.Device.CPU
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET


def test_regenie_command_returns_signal_exit_code_for_graceful_shutdown() -> None:
    shutdown_request = shutdown.GracefulShutdownRequested(
        shutdown.ShutdownSignal(number=int(signal.SIGINT), name="SIGINT", exit_code=130)
    )
    with patch("g.cli.api.regenie", side_effect=shutdown_request):
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
                "--pred",
                "predictions.list",
                "--out",
                "results/output",
                "--qt",
            ],
        )

    assert result.exit_code == 130
    assert "Interrupted by SIGINT" in result.output
    assert "saved committed output for --resume" in result.output
    assert "Traceback" not in result.output


def test_graceful_shutdown_controller_escalates_second_signal() -> None:
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    controller = shutdown.GracefulShutdownController()

    with controller:
        with pytest.raises(shutdown.GracefulShutdownRequested):
            controller.handle_signal(int(signal.SIGINT), None)
        with pytest.raises(KeyboardInterrupt):
            controller.handle_signal(int(signal.SIGINT), None)

    assert signal.getsignal(signal.SIGINT) == previous_sigint_handler


def test_regenie_command_options_are_generated_from_specs() -> None:
    regenie_command = app.commands["regenie"]
    click_options = {
        click_option.name: click_option
        for click_option in regenie_command.params
        if isinstance(click_option, click.Option)
    }

    for option_spec in options.OPTION_SPECS:
        click_option = click_options[option_spec.destination]
        assert click_option.opts[0].split("/")[0] == f"--{option_spec.name}"
        assert click_option.help == option_spec.help_text


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


def test_regenie_command_rejects_binary_only_flag_under_quantitative_trait() -> None:
    result = runner.invoke(
        app,
        [
            "regenie",
            "--step",
            "2",
            "--bgen",
            "dataset.bgen",
            "--phenoFile",
            "phenotype.tsv",
            "--phenoCol",
            "trait",
            "--pred",
            "predictions.list",
            "--out",
            "results/output",
            "--qt",
            "--firth",
        ],
    )

    assert result.exit_code != 0
    assert "--firth can only be used with --bt" in result.output


def test_regenie_command_rejects_explicit_binary_threshold_under_quantitative_trait() -> None:
    result = runner.invoke(
        app,
        [
            "regenie",
            "--step",
            "2",
            "--bgen",
            "dataset.bgen",
            "--phenoFile",
            "phenotype.tsv",
            "--phenoCol",
            "trait",
            "--pred",
            "predictions.list",
            "--out",
            "results/output",
            "--qt",
            "--pThresh",
            "0.05",
        ],
    )

    assert result.exit_code != 0
    assert "--pThresh can only be used with --bt" in result.output


def test_regenie_command_rejects_removed_duplicate_iid_flag() -> None:
    result = runner.invoke(app, ["regenie", "--g-allow-duplicate-iid-alignment"])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--g-allow-duplicate-iid-alignment" in result.output


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
                "[output]",
                'out = "results/output"',
                "[g.output]",
                'format = "arrow"',
            ]
        ),
        encoding="utf-8",
    )

    with patch("g.cli.api.regenie", return_value=api.RunArtifacts()) as mock_regenie_api:
        result = runner.invoke(app, ["regenie", "--config", str(config_path), "--qt", "--bsize", "4096"])

    assert result.exit_code == 0
    regenie_config = mock_regenie_api.call_args.args[0]
    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert regenie_config.trait.bsize == 4096
    assert regenie_config.g_output.format == types.OutputFormat.ARROW


def test_regenie_command_applies_explicit_cli_override_above_both_toml_layers(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[input]",
                'bgen = "dataset.bgen"',
                'phenoFile = "phenotype.tsv"',
                'phenoCol = "trait"',
                'pred = "predictions.list"',
                "[output]",
                'out = "results/output"',
                "[g.compute]",
                'device = "gpu"',
                "[g.output]",
                'format = "arrow"',
            ]
        ),
        encoding="utf-8",
    )

    with patch("g.cli.api.regenie", return_value=api.RunArtifacts()) as mock_regenie_api:
        result = runner.invoke(
            app,
            [
                "regenie",
                "--config",
                str(config_path),
                "--g-device",
                "cpu",
                "--g-output-format",
                "parquet",
            ],
        )

    assert result.exit_code == 0
    regenie_config = mock_regenie_api.call_args.args[0]
    assert regenie_config.g_compute.device == types.Device.CPU
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET


def test_regenie_command_applies_explicit_binary_override(tmp_path: Path) -> None:
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
                "qt = true",
                "[output]",
                'out = "results/output"',
            ]
        ),
        encoding="utf-8",
    )

    with patch("g.cli.api.regenie", return_value=api.RunArtifacts()) as mock_regenie_api:
        result = runner.invoke(app, ["regenie", "--config", str(config_path), "--bt"])

    assert result.exit_code == 0
    regenie_config = mock_regenie_api.call_args.args[0]
    assert regenie_config.trait.trait_type == types.RegenieTraitType.BINARY


def test_config_subcommands_render_and_validate(tmp_path: Path) -> None:
    config_path = tmp_path / "regenie.toml"

    init_result = runner.invoke(app, ["config", "init", "--out", str(config_path)])

    assert init_result.exit_code == 0
    assert config_path.exists()
    toml_mapping = config_layers.toml_config_to_builtin_mapping(config_layers.decode_toml_file(config_path))
    assert toml_mapping["trait"]["step"] == 2
    validate_result = runner.invoke(app, ["config", "validate", str(config_path)])
    assert validate_result.exit_code != 0
    assert "Exactly one genotype source" in validate_result.output
    explain_result = runner.invoke(app, ["config", "explain", "bgen"])
    assert explain_result.exit_code == 0
    assert "supported" in explain_result.output


def test_config_init_writes_to_stdout() -> None:
    result = runner.invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert "[input]" in result.output
    assert "[trait]" in result.output


def test_config_validate_reports_invalid_toml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text('[input]\nphenoFile = "phenotype.tsv"\n', encoding="utf-8")

    result = runner.invoke(app, ["config", "validate", str(config_path)])

    assert result.exit_code != 0
    assert "Exactly one genotype source" in result.output


def test_config_explain_lists_all_options() -> None:
    result = runner.invoke(app, ["config", "explain"])

    assert result.exit_code == 0
    assert "bgen: supported" in result.output
    assert "g-output-format: g_extension" in result.output


def test_config_explain_reports_unknown_option() -> None:
    result = runner.invoke(app, ["config", "explain", "not-a-real-option"])

    assert result.exit_code != 0
    assert "Unknown option: not-a-real-option" in result.output


def test_legacy_commands_are_not_registered() -> None:
    for command_name in ["regenie2", "regenie2-linear", "regenie2-warm-cache", "linear", "logistic"]:
        result = runner.invoke(app, [command_name, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_print_success_message_reports_run_directory_outputs(capsys: typing.Any) -> None:
    print_success_message(
        api.RunArtifacts(
            output_run_directory=Path("results/output.g/trait.regenie2_linear.run"),
            final_dataset=Path("results/output.g/trait.regenie2_linear.run/parts"),
            final_parquet=Path("results/output.g/trait.regenie2_linear.run/final.parquet"),
        )
    )
    captured = capsys.readouterr()
    assert "results/output.g/trait.regenie2_linear.run" in captured.out
    assert "Parquet dataset" in captured.out
    assert "final.parquet" in captured.out


def test_print_success_message_reports_nested_phenotype_artifacts(capsys: typing.Any) -> None:
    print_success_message(
        api.RunArtifacts(
            phenotype_artifacts=(
                api.RunArtifacts(output_run_directory=Path("results/trait_a.run")),
                api.RunArtifacts(
                    output_run_directory=Path("results/trait_b.run"), final_parquet=Path("trait_b.parquet")
                ),
            )
        )
    )

    captured = capsys.readouterr()
    assert "results/trait_a.run" in captured.out
    assert "results/trait_b.run" in captured.out
    assert "trait_b.parquet" in captured.out


def test_print_warm_cache_message_lists_warmed_shapes(capsys: typing.Any) -> None:
    print_warm_cache_message(WarmReport(warmed_shapes=(WarmShape(sample_count=12, variant_count=512),)))

    captured = capsys.readouterr()
    assert "(12, 512)" in captured.out


def test_resolve_trusted_bgen_validation_mode_rejects_conflicts() -> None:
    assert (
        resolve_trusted_bgen_validation_mode(validate_trusted_bgen=False, assume_trusted_bgen_validated=False)
        == types.TrustedBgenValidationMode.CACHE_ON_MISS
    )
    assert (
        resolve_trusted_bgen_validation_mode(validate_trusted_bgen=True, assume_trusted_bgen_validated=False)
        == types.TrustedBgenValidationMode.FORCE_VALIDATE
    )
    assert (
        resolve_trusted_bgen_validation_mode(validate_trusted_bgen=False, assume_trusted_bgen_validated=True)
        == types.TrustedBgenValidationMode.ASSUME_VALIDATED
    )
    with pytest.raises(click.BadParameter, match="mutually exclusive"):
        resolve_trusted_bgen_validation_mode(validate_trusted_bgen=True, assume_trusted_bgen_validated=True)


def test_main_dispatches_to_click_app() -> None:
    with patch("g.cli.app") as mock_app:
        main()
    mock_app.assert_called_once_with()


def test_regenie_main_dispatches_direct_entrypoint() -> None:
    with patch("g.cli.run_regenie_command.main") as mock_main:
        regenie_main()

    mock_main.assert_called_once()
    assert mock_main.call_args.kwargs["prog_name"] == "g-regenie"
    assert mock_main.call_args.kwargs["standalone_mode"] is True
