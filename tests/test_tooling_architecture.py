from __future__ import annotations

import dataclasses
import enum
import json
import os
import typing
from pathlib import Path

import tooling.cli.benchmark_bgen_reader as benchmark_bgen_reader
import tooling.cli.benchmark_regenie2_binary_hot as binary_hot_benchmark
import tooling.cli.profile_regenie2_deep as deep_profile
import tooling.cli.run_regenie2_matrix as regenie2_matrix
import tooling.configuration as tooling_configuration
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


def test_hydra_chr10_matrix_config_composes() -> None:
    config = tooling_configuration.compose_config(
        config_name="run_regenie2_chr10_matrix",
        overrides=[
            "tool.dry_run=true",
            "tool.output_dir=data/benchmarks/test_chr10_matrix",
            "tool.variant_limit=1000",
        ],
        include_hydra_config=True,
    )

    assert config.dataset.bgen_file == "1kg_chr10_full.bgen"
    assert config.dataset.prediction_list == "baselines_chr10/regenie_step1_pred.list"
    assert config.workload.name == "regenie2_chr10_matrix"
    assert config.tool.dry_run is True
    assert config.tool.variant_limit == 1000
    assert config.hydra.job.chdir is False


def test_hydra_chr22_matrix_config_composes() -> None:
    config = tooling_configuration.compose_config(
        config_name="run_regenie2_chr22_matrix",
        overrides=[
            "tool.dry_run=true",
            "tool.output_dir=data/benchmarks/test_chr22_matrix",
            "tool.variant_limit=1000",
        ],
        include_hydra_config=True,
    )

    assert config.dataset.bgen_file == "1kg_chr22_full.bgen"
    assert config.dataset.prediction_list == "baselines/regenie_step1_pred.list"
    assert config.workload.name == "regenie2_chr22_matrix"
    assert config.tool.chromosome_label == "chr22"
    assert config.tool.run_directory_prefix == "regenie2_chr22_matrix"
    assert config.tool.linear_prediction_list == "baselines/regenie_step1_qt_pred.list"
    assert config.tool.dry_run is True
    assert config.tool.variant_limit == 1000
    assert config.hydra.job.chdir is False


def test_hydra_deep_profile_config_converts_to_tool_arguments(tmp_path: Path) -> None:
    arguments = deep_profile.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.dry_run=true",
            "tool.variant_limit=1000",
        ]
    )
    baseline_paths = deep_profile.build_baseline_paths(arguments)
    profile_plan = deep_profile.build_profile_plan(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=tmp_path / "profile",
    )

    assert arguments.chromosome_label == "chr22"
    assert arguments.bgen_path.name == "1kg_chr22_full.bgen"
    assert arguments.regenie_prediction_list_path == arguments.data_directory / "baselines" / "regenie_step1_pred.list"
    assert arguments.regenie_qt_prediction_list_path == (
        arguments.data_directory / "baselines" / "regenie_step1_qt_pred.list"
    )
    assert arguments.enable_jax_trace is True
    assert arguments.enable_python_cprofile is True
    assert arguments.enable_rust_criterion is True
    assert arguments.include_regenie_baseline is True
    assert profile_plan.profiler_modes == {
        "regenie_baseline": True,
        "jax_trace": True,
        "jax_memory_profile": True,
        "python_cprofile": True,
        "py_spy": True,
        "linux_perf": True,
        "rust_criterion": True,
    }
    assert profile_plan.rust_benchmark_commands == [
        ["cargo", "bench", "--bench", "bgen_read"],
        ["cargo", "bench", "--bench", "preprocess"],
    ]


def test_hydra_tooling_config_converts_to_tool_arguments() -> None:
    bgen_arguments = benchmark_bgen_reader.build_arguments_from_overrides(
        [
            "dataset.data_directory=custom-data",
            "dataset.bgen_file=input.bgen",
            "dataset.sample_file=input.sample",
            "workload.chunk_size=4096",
            "workload.variant_limit=100",
            "workload.repeat_count=2",
            "telemetry.json_summary_path=reports/summary.json",
            "telemetry.markdown_summary_path=reports/summary.md",
            "sweep.chunk_sizes=[4096,8192]",
            "sweep.path_modes=[variant_major_buffered,variant_major_packed8_buffered]",
            "sweep.sample_selection_modes=[full,strided_half]",
            "sweep.trusted_no_missing_diploid_modes=[true,false]",
        ]
    )
    assert bgen_arguments.bgen == Path("custom-data/input.bgen")
    assert bgen_arguments.chunk_sizes == "4096,8192"
    assert bgen_arguments.trusted_no_missing_diploid_modes == "true,false"
    assert bgen_arguments.json_summary_path == Path("reports/summary.json")

    binary_hot_arguments = binary_hot_benchmark.build_arguments_from_overrides(
        [
            "machine.device=gpu",
            "dataset.phenotype_columns=[trait_a,trait_b]",
            "sweep.storage_modes=[packed8]",
            "sweep.fallback_density_scenarios=[low]",
        ]
    )
    assert binary_hot_arguments.device == "gpu"
    assert binary_hot_arguments.storage_modes == "packed8"
    assert binary_hot_arguments.phenotype_columns == "trait_a,trait_b"

    chr10_arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            "tool.dry_run=true",
            "tool.output_dir=data/benchmarks/test_chr10_matrix",
            "tool.variant_limit=1000",
            "tool.binary_firth_batch_size=64",
            "tool.binary_firth_candidate_capacity=1024",
        ]
    )
    chr10_run_specs = regenie2_matrix.build_run_specs(chr10_arguments)
    run_names = [run_spec.name for run_spec in chr10_run_specs]
    gpu_cache_arguments = chr10_run_specs[1].command_arguments
    cached_gpu_arguments = chr10_run_specs[2].command_arguments

    assert chr10_arguments.bgen_path.name == "1kg_chr10_full.bgen"
    assert run_names == [
        "binary_cpu",
        "binary_gpu",
        "binary_gpu_cached",
        "linear_cpu",
        "linear_gpu",
        "linear_gpu_cached",
    ]
    assert "--g-variant-limit" in chr10_run_specs[0].command_arguments
    assert "1000" in chr10_run_specs[0].command_arguments
    assert "--g-firth-batch-size" in chr10_run_specs[0].command_arguments
    assert "64" in chr10_run_specs[0].command_arguments
    assert "--g-firth-candidate-capacity" in chr10_run_specs[0].command_arguments
    assert "1024" in chr10_run_specs[0].command_arguments
    assert "--g-firth-batch-size" not in chr10_run_specs[3].command_arguments
    assert "--g-firth-candidate-capacity" not in chr10_run_specs[3].command_arguments
    assert "--no-g-jax-persistent-cache" in chr10_run_specs[0].command_arguments
    assert "--g-jax-cache-dir" in gpu_cache_arguments
    assert "--g-jax-cache-dir" in cached_gpu_arguments
    assert gpu_cache_arguments[gpu_cache_arguments.index("--g-jax-cache-dir") + 1] == cached_gpu_arguments[
        cached_gpu_arguments.index("--g-jax-cache-dir") + 1
    ]

    chr22_arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            "tool.dry_run=true",
            "tool.output_dir=data/benchmarks/test_chr22_matrix",
            "tool.variant_limit=1000",
        ],
        config_name="run_regenie2_chr22_matrix",
    )
    chr22_run_specs = regenie2_matrix.build_run_specs(chr22_arguments)

    assert chr22_arguments.chromosome_label == "chr22"
    assert chr22_arguments.run_directory_prefix == "regenie2_chr22_matrix"
    assert chr22_arguments.bgen_path.name == "1kg_chr22_full.bgen"
    assert chr22_arguments.linear_prediction_list_path == (
        chr22_arguments.data_directory / "baselines" / "regenie_step1_qt_pred.list"
    )
    assert "--g-variant-limit" in chr22_run_specs[0].command_arguments
    assert "1000" in chr22_run_specs[0].command_arguments


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


def test_chr10_matrix_previous_run_comparison() -> None:
    previous_result = regenie2_matrix.RunResult(
        name="binary_gpu",
        trait=regenie2_matrix.TraitKind.BINARY,
        mode=regenie2_matrix.ExecutionMode.GPU,
        status=regenie2_matrix.RunStatus.SUCCESS,
        return_code=0,
        wall_time_seconds=20.0,
        command_arguments=["g", "regenie"],
        output_prefix="previous",
        output_run_directory="previous.g/phenotype_binary.regenie2_binary.run",
        stage_timing_path="previous_stage.json",
        profile_summary_path="previous_profile.json",
        event_log_path="previous_events.jsonl",
        output_row_count=100,
        committed_chunk_count=10,
        output_file_count=2,
        output_total_bytes=1000,
        final_parquet_path=None,
        final_parquet_bytes=None,
        stage_seconds={"jax_compute": 5.0},
    )
    current_result = dataclasses.replace(
        previous_result,
        wall_time_seconds=10.0,
        output_total_bytes=1200,
        stage_seconds={"jax_compute": 4.0},
    )

    comparisons = regenie2_matrix.compare_run_results(
        current_results=[current_result],
        previous_results_by_name={"binary_gpu": previous_result},
    )
    comparisons_by_metric = {comparison.metric: comparison for comparison in comparisons}

    assert comparisons_by_metric["wall_time_seconds"].delta == -10.0
    assert comparisons_by_metric["wall_time_seconds"].ratio == 0.5
    assert comparisons_by_metric["output_total_bytes"].delta == 200.0
    assert comparisons_by_metric["stage.jax_compute"].ratio == 0.8


def test_chr10_matrix_explicit_output_directory_finds_sibling_previous_manifest(tmp_path: Path) -> None:
    previous_directory = tmp_path / "regenie2_chr10_matrix_previous"
    dry_run_directory = tmp_path / "regenie2_chr10_matrix_dry_run"
    mismatched_variant_limit_directory = tmp_path / "regenie2_chr10_matrix_full"
    current_directory = tmp_path / "regenie2_chr10_matrix_current"
    previous_manifest_path = previous_directory / "manifest.json"
    dry_run_manifest_path = dry_run_directory / "manifest.json"
    mismatched_variant_limit_manifest_path = mismatched_variant_limit_directory / "manifest.json"
    previous_directory.mkdir()
    dry_run_directory.mkdir()
    mismatched_variant_limit_directory.mkdir()
    current_directory.mkdir()
    previous_manifest_path.write_text(
        json.dumps(
            {
                "dry_run": False,
                "configuration": {
                    "variant_limit": 1000,
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    dry_run_manifest_path.write_text(
        json.dumps(
            {
                "dry_run": True,
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    mismatched_variant_limit_manifest_path.write_text(
        json.dumps(
            {
                "dry_run": False,
                "configuration": {
                    "variant_limit": None,
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    os.utime(previous_manifest_path, (100.0, 100.0))
    os.utime(dry_run_manifest_path, (200.0, 200.0))
    os.utime(mismatched_variant_limit_manifest_path, (300.0, 300.0))
    arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            f"tool.output_dir={current_directory}",
            "tool.dry_run=true",
            "tool.variant_limit=1000",
        ]
    )

    discovered_manifest_path = regenie2_matrix.find_previous_manifest(arguments)

    assert arguments.output_parent == tmp_path
    assert discovered_manifest_path == previous_manifest_path


def test_chr22_matrix_discovers_chr22_previous_manifest_only(tmp_path: Path) -> None:
    previous_chr10_directory = tmp_path / "regenie2_chr10_matrix_previous"
    previous_chr22_directory = tmp_path / "regenie2_chr22_matrix_previous"
    current_directory = tmp_path / "regenie2_chr22_matrix_current"
    previous_chr10_manifest_path = previous_chr10_directory / "manifest.json"
    previous_chr22_manifest_path = previous_chr22_directory / "manifest.json"
    previous_chr10_directory.mkdir()
    previous_chr22_directory.mkdir()
    current_directory.mkdir()
    previous_chr10_manifest_path.write_text(
        json.dumps(
            {
                "dry_run": False,
                "configuration": {
                    "variant_limit": 1000,
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    previous_chr22_manifest_path.write_text(
        json.dumps(
            {
                "dry_run": False,
                "configuration": {
                    "variant_limit": 1000,
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    os.utime(previous_chr10_manifest_path, (300.0, 300.0))
    os.utime(previous_chr22_manifest_path, (200.0, 200.0))
    arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            f"tool.output_dir={current_directory}",
            "tool.dry_run=true",
            "tool.variant_limit=1000",
        ],
        config_name="run_regenie2_chr22_matrix",
    )

    discovered_manifest_path = regenie2_matrix.find_previous_manifest(arguments)

    assert arguments.output_parent == tmp_path
    assert discovered_manifest_path == previous_chr22_manifest_path


def test_chr10_matrix_discovers_trait_prefixed_output_directory(tmp_path: Path) -> None:
    output_prefix = tmp_path / "runs" / "binary_cpu"
    actual_output_directory = output_prefix.parent / "binary_cpu.g" / "trait_0001_phenotype_binary.regenie2_binary.run"
    parts_directory = actual_output_directory / "parts"
    parts_directory.mkdir(parents=True)
    (parts_directory / "part-0.parquet").write_bytes(b"parquet")
    (actual_output_directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "committed_chunks": [
                    {
                        "row_count": 1000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run_spec = regenie2_matrix.RunSpec(
        name="binary_cpu",
        trait=regenie2_matrix.TraitKind.BINARY,
        mode=regenie2_matrix.ExecutionMode.CPU,
        command_arguments=["g", "regenie"],
        output_prefix=output_prefix,
        output_run_directory=output_prefix.parent / "binary_cpu.g" / "phenotype_binary.regenie2_binary.run",
        stage_timing_path=tmp_path / "stage.json",
        profile_summary_path=tmp_path / "profile.json",
        event_log_path=tmp_path / "events.jsonl",
        environment_overrides={},
    )

    output_metrics = regenie2_matrix.measure_run_outputs(run_spec)

    assert output_metrics["output_run_directory"] == str(actual_output_directory)
    assert output_metrics["output_row_count"] == 1000
    assert output_metrics["committed_chunk_count"] == 1
    assert output_metrics["output_file_count"] == 1


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


def test_tooling_entrypoint_exposes_cli_surface() -> None:
    assert benchmark_bgen_reader.build_arguments_from_overrides is not None
    assert benchmark_bgen_reader.hydra_main is not None
    assert benchmark_bgen_reader.BenchmarkPathMode is regenie_bgen_reader.BenchmarkPathMode
