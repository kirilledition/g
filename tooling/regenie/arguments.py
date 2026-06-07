"""Config-to-argument adapters for REGENIE development tooling."""

from __future__ import annotations

import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from tooling import configuration as tooling_configuration


def append_optional_path_argument(command_arguments: list[str], option_name: str, path_value: str | None) -> None:
    """Append a path-valued argument when a value is configured.

    Args:
        command_arguments: Mutable command argument list.
        option_name: CLI option name.
        path_value: Optional path value.

    """
    if path_value is not None:
        command_arguments.extend([option_name, str(Path(path_value))])


def build_bgen_reader_arguments(configuration: tooling_configuration.ToolingConfig) -> list[str]:
    """Convert typed tooling config into BGEN reader CLI arguments.

    Args:
        configuration: Typed tooling configuration.

    Returns:
        CLI arguments for ``tooling.cli.benchmark_bgen_reader``.

    """
    command_arguments = [
        "--bgen",
        str(Path(configuration.dataset.data_directory) / configuration.dataset.bgen_file),
        "--sample",
        str(Path(configuration.dataset.data_directory) / configuration.dataset.sample_file),
        "--chunk-size",
        str(configuration.workload.chunk_size),
        "--repeat-count",
        str(configuration.workload.repeat_count),
        "--path-modes",
        ",".join(configuration.sweep.path_modes),
    ]
    if configuration.workload.variant_limit is not None:
        command_arguments.extend(["--variant-limit", str(configuration.workload.variant_limit)])
    if configuration.sweep.chunk_sizes:
        command_arguments.extend(["--chunk-sizes", ",".join(str(value) for value in configuration.sweep.chunk_sizes)])
    if configuration.sweep.sample_selection_modes:
        command_arguments.extend(
            [
                "--sample-selection-modes",
                ",".join(configuration.sweep.sample_selection_modes),
            ]
        )
    if configuration.sweep.decode_tile_variant_counts:
        command_arguments.extend(
            [
                "--decode-tile-variant-counts",
                ",".join(
                    "default" if value is None else str(value)
                    for value in configuration.sweep.decode_tile_variant_counts
                ),
            ]
        )
    if configuration.sweep.rayon_thread_counts:
        command_arguments.extend(
            [
                "--rayon-thread-counts",
                ",".join(
                    "default" if value is None else str(value) for value in configuration.sweep.rayon_thread_counts
                ),
            ]
        )
    if configuration.sweep.trusted_no_missing_diploid_modes:
        command_arguments.extend(
            [
                "--trusted-no-missing-diploid-modes",
                ",".join(
                    "trusted" if value else "safe" for value in configuration.sweep.trusted_no_missing_diploid_modes
                ),
            ]
        )
    append_optional_path_argument(command_arguments, "--json-summary-path", configuration.telemetry.json_summary_path)
    append_optional_path_argument(
        command_arguments, "--markdown-summary-path", configuration.telemetry.markdown_summary_path
    )
    return command_arguments


def build_regenie2_binary_hot_arguments(configuration: tooling_configuration.ToolingConfig) -> list[str]:
    """Convert typed tooling config into binary-hot benchmark CLI arguments.

    Args:
        configuration: Typed tooling configuration.

    Returns:
        CLI arguments for ``tooling.cli.benchmark_regenie2_binary_hot``.

    """
    command_arguments = [
        "--data-dir",
        configuration.dataset.data_directory,
        "--bgen",
        configuration.dataset.bgen_file,
        "--sample",
        configuration.dataset.sample_file,
        "--phenotype-file",
        configuration.dataset.phenotype_file,
        "--prediction-list",
        configuration.dataset.prediction_list,
        "--device",
        configuration.machine.device,
        "--chunk-size",
        str(configuration.workload.chunk_size),
        "--staging-depth",
        str(configuration.workload.staging_depth),
        "--output-writer-thread-count",
        str(configuration.workload.output_writer_thread_count),
        "--output-writer-queue-depth",
        str(configuration.workload.output_writer_queue_depth),
        "--phenotype-columns",
        ",".join(configuration.dataset.phenotype_columns),
        "--storage-modes",
        ",".join(configuration.sweep.storage_modes),
        "--fallback-density-scenarios",
        ",".join(configuration.sweep.fallback_density_scenarios),
        "--stage-timing-mode",
        configuration.telemetry.stage_timing_mode,
    ]
    if configuration.workload.variant_limit is not None:
        command_arguments.extend(["--variant-limit", str(configuration.workload.variant_limit)])
    append_optional_path_argument(command_arguments, "--json-summary-path", configuration.telemetry.json_summary_path)
    return command_arguments
