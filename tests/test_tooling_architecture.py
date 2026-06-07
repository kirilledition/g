from __future__ import annotations

import dataclasses
import enum
import importlib
import json
import typing
from pathlib import Path

import tooling.configuration as tooling_configuration
import tooling.regenie.arguments as regenie_arguments
import tooling.regenie.bgen_reader as regenie_bgen_reader
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports
from tooling.common import sweeps as tooling_sweeps


class ReportMode(enum.StrEnum):
    TEST = "test"


@dataclasses.dataclass(frozen=True)
class ReportPayload:
    path: Path
    mode: ReportMode


def test_hydra_tooling_config_composes_with_repo_relative_behavior() -> None:
    config = tooling_configuration.compose_config()
    typed_config = tooling_configuration.instantiate_config(config)

    assert typed_config.dataset.data_directory == "data"
    assert typed_config.workload.chunk_size == 8192
    assert typed_config.sweep.path_modes == ["variant_major_buffered"]

    hydra_config = tooling_configuration.compose_config(include_hydra_config=True)
    assert hydra_config.hydra.job.chdir is False


def test_hydra_tooling_config_accepts_group_overrides() -> None:
    config = tooling_configuration.compose_config(
        overrides=[
            "machine=landau_gpu",
            "workload=regenie2_binary_hot",
            "sweep=regenie2_binary_hot_default",
        ]
    )
    typed_config = tooling_configuration.instantiate_config(config)

    assert typed_config.machine.device == "gpu"
    assert typed_config.machine.slurm_node == "landau"
    assert typed_config.workload.name == "regenie2_binary_hot"
    assert typed_config.sweep.trusted_no_missing_diploid_modes == [True]


def test_tooling_config_converts_to_script_arguments() -> None:
    config = tooling_configuration.ToolingConfig(
        dataset=tooling_configuration.DatasetConfig(
            data_directory="custom-data",
            bgen_file="input.bgen",
            sample_file="input.sample",
            phenotype_columns=["trait_a", "trait_b"],
        ),
        machine=tooling_configuration.MachineConfig(device="gpu"),
        workload=tooling_configuration.WorkloadConfig(chunk_size=4096, variant_limit=100, repeat_count=2),
        telemetry=tooling_configuration.TelemetryConfig(
            json_summary_path="reports/summary.json",
            markdown_summary_path="reports/summary.md",
        ),
        sweep=tooling_configuration.SweepConfig(
            chunk_sizes=[4096, 8192],
            path_modes=["variant_major_buffered", "variant_major_packed8_buffered"],
            sample_selection_modes=["full", "strided_half"],
            trusted_no_missing_diploid_modes=[True, False],
            storage_modes=["packed8"],
            fallback_density_scenarios=["low"],
        ),
    )

    bgen_arguments = regenie_arguments.build_bgen_reader_arguments(config)
    assert bgen_arguments[bgen_arguments.index("--bgen") + 1] == "custom-data/input.bgen"
    assert bgen_arguments[bgen_arguments.index("--chunk-sizes") + 1] == "4096,8192"
    assert bgen_arguments[bgen_arguments.index("--trusted-no-missing-diploid-modes") + 1] == "trusted,safe"

    binary_hot_arguments = regenie_arguments.build_regenie2_binary_hot_arguments(config)
    assert binary_hot_arguments[binary_hot_arguments.index("--device") + 1] == "gpu"
    assert binary_hot_arguments[binary_hot_arguments.index("--storage-modes") + 1] == "packed8"
    assert binary_hot_arguments[binary_hot_arguments.index("--phenotype-columns") + 1] == "trait_a,trait_b"


def test_path_resolution_honors_data_directory_environment(tmp_path: Path) -> None:
    relative_directory = tooling_paths.resolve_data_directory(
        repository_root=tmp_path,
        environment={"GWAS_ENGINE_DATA_DIR": "alternate-data"},
    )
    absolute_override = tmp_path / "absolute-data"
    absolute_directory = tooling_paths.resolve_data_directory(
        repository_root=tmp_path,
        environment={"GWAS_ENGINE_DATA_DIR": str(absolute_override)},
    )

    assert relative_directory == tmp_path / "alternate-data"
    assert absolute_directory == absolute_override


def test_report_serialization_handles_dataclasses_paths_and_enums(tmp_path: Path) -> None:
    report_path = tmp_path / "nested" / "report.json"
    tooling_reports.write_json_report(report_path, ReportPayload(path=Path("data/input.bgen"), mode=ReportMode.TEST))

    payload = typing.cast("dict[str, str]", json.loads(report_path.read_text(encoding="utf-8")))
    assert payload == {"path": "data/input.bgen", "mode": "test"}


def test_sweep_and_bgen_mode_parsing() -> None:
    assert tooling_sweeps.parse_optional_integer_list("default,4") == [None, 4]
    assert tooling_sweeps.parse_boolean_mode_list("trusted,safe") == [True, False]
    assert tooling_sweeps.build_queue_depths((1, 4), (1, 2)) == (1, 2, 4, 8)

    path_modes = regenie_bgen_reader.parse_path_modes("variant_major_buffered,variant_major_packed8_buffered")
    assert path_modes == [
        regenie_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_BUFFERED,
        regenie_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_PACKED8_BUFFERED,
    ]
    assert regenie_bgen_reader.supported_path_modes(path_modes, trusted_no_missing_diploid=False) == [
        regenie_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_BUFFERED
    ]
    assert regenie_bgen_reader.parse_sample_selection_modes("full,strided_half") == [
        regenie_bgen_reader.SampleSelectionMode.FULL,
        regenie_bgen_reader.SampleSelectionMode.STRIDED_HALF,
    ]


def test_script_wrappers_expose_migrated_entrypoints() -> None:
    wrapper_module = importlib.import_module("scripts.benchmark_bgen_reader")
    tooling_module = importlib.import_module("tooling.cli.benchmark_bgen_reader")

    assert wrapper_module.build_argument_parser is tooling_module.build_argument_parser
    assert wrapper_module.BenchmarkPathMode is tooling_module.BenchmarkPathMode
