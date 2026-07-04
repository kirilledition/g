from __future__ import annotations

import dataclasses
import enum
import gzip
import json
import os
import sys
import typing
from pathlib import Path

import numpy as np

import tooling.cli.benchmark as grouped_benchmark
import tooling.cli.benchmark_bgen_reader as benchmark_bgen_reader
import tooling.cli.benchmark_regenie2_binary_hot as binary_hot_benchmark
import tooling.cli.benchmark_tensorqtl_chr22 as tensorqtl_benchmark
import tooling.cli.benchmark_torchgwas_chr22 as torchgwas_benchmark
import tooling.cli.data as grouped_data
import tooling.cli.debug as grouped_debug
import tooling.cli.performance as grouped_performance
import tooling.cli.profile_regenie2_deep as deep_profile
import tooling.cli.run_regenie2_matrix as regenie2_matrix
import tooling.cli.rust_build_profiles as rust_build_profiles
import tooling.cli.schema_check as schema_check
import tooling.cli.server as grouped_server
import tooling.configuration as tooling_configuration
import tooling.debug.check_pyo3_stub as check_pyo3_stub
import tooling.regenie.bgen_reader as regenie_bgen_reader
from g.interface import config as interface_config
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import commands as tooling_commands
from tooling.common import g_regenie as tooling_g_regenie
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import jax_cache as tooling_jax_cache
from tooling.common import paths as tooling_paths
from tooling.common import registry as tooling_registry
from tooling.common import reports as tooling_reports
from tooling.common import sweeps as tooling_sweeps
from tooling.profile_deep import budget as profile_deep_budget
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import jax_cache as profile_deep_jax_cache
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    import pytest


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


def test_grouped_hydra_tooling_configs_compose() -> None:
    config_names = ["benchmark", "data", "debug", "performance", "rust_build_profiles", "server"]

    for config_name in config_names:
        config = tooling_configuration.compose_config(config_name=config_name, include_hydra_config=True)
        assert "tool" in config
        assert config.hydra.job.chdir is False


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


def test_hydra_torchgwas_chr22_config_converts_to_arguments(tmp_path: Path) -> None:
    arguments = torchgwas_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'torchgwas'}",
            "tool.dry_run=true",
            "tool.variant_limit=1000",
        ]
    )

    assert arguments.bgen_path.name == "1kg_chr22_full.bgen"
    assert arguments.sample_path.name == "1kg_chr22_full.sample"
    assert arguments.bim_path.name == "1kg_chr22_full.bim"
    assert arguments.torchgwas_full_genotype_path.name == "1kg_chr22_full.bed"
    assert arguments.torchgwas_full_genotype_format == torchgwas_benchmark.TorchgwasGenotypeFormat.PLINK
    assert arguments.torchgwas_full_bim_path is not None
    assert arguments.torchgwas_full_bim_path.name == "1kg_chr22_full.bim"
    assert arguments.torchgwas_full_fam_path is not None
    assert arguments.torchgwas_full_fam_path.name == "1kg_chr22_full.fam"
    assert arguments.phenotype_column == "phenotype_continuous"
    assert arguments.covariate_columns == ("age", "sex")
    assert arguments.prediction_list_path == arguments.data_directory / "baselines" / "regenie_step1_qt_pred.list"
    assert arguments.chunk_size == 16384
    assert arguments.variant_limit == 1000
    assert arguments.torchgwas_commit == torchgwas_benchmark.TORCHGWAS_MAIN_COMMIT
    assert arguments.torchgwas_python == "/usr/bin/python3.12"
    assert arguments.torch_package == "torch==2.5.1+cu121"
    assert arguments.torch_package_index_url == "https://download.pytorch.org/whl/cu121"
    assert torchgwas_benchmark.torchgwas_venv_directory(arguments).name == "venv-python3_12"
    assert arguments.output_directory == tmp_path / "torchgwas"


def test_torchgwas_chr22_cases_render_g_and_competitor_commands(tmp_path: Path) -> None:
    arguments = torchgwas_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'torchgwas'}",
            "tool.dry_run=true",
            "tool.variant_limit=null",
        ]
    )
    input_spec = torchgwas_benchmark.TorchgwasInputSpec(
        genotype_path=arguments.torchgwas_full_genotype_path,
        genotype_format=arguments.torchgwas_full_genotype_format,
        sample_file=None,
        sample_ids_path=None,
        marker_ids_path=None,
        bim_path=arguments.torchgwas_full_bim_path,
        fam_path=arguments.torchgwas_full_fam_path,
    )

    cases = torchgwas_benchmark.build_cases(arguments, input_spec)
    case_by_identifier = {benchmark_case.case_id: benchmark_case for benchmark_case in cases}
    g_command = list(case_by_identifier["g_linear_gpu_cold"].command_arguments)
    torchgwas_command = list(case_by_identifier["torchgwas_linear_gpu_cold"].command_arguments)
    assert arguments.torchgwas_full_bim_path is not None
    assert arguments.torchgwas_full_fam_path is not None

    assert [benchmark_case.case_id for benchmark_case in cases] == [
        "g_linear_gpu_cold",
        "torchgwas_linear_gpu_cold",
        "g_linear_gpu_warm",
        "torchgwas_linear_gpu_warm",
    ]
    assert "--qt" in g_command
    assert g_command[g_command.index("--bgen") + 1] == str(arguments.bgen_path)
    assert g_command[g_command.index("--phenoCol") + 1] == "phenotype_continuous"
    assert g_command[g_command.index("--pred") + 1] == str(arguments.prediction_list_path)
    assert g_command[g_command.index("--device") + 1] == "gpu"
    assert g_command[g_command.index("--jax_cache_dir") + 1] == str(tmp_path / "torchgwas" / "caches" / "g")

    assert Path(torchgwas_command[0]).name == "torchgwas"
    assert torchgwas_command[1] == "linear"
    assert torchgwas_command[torchgwas_command.index("--genotype") + 1] == str(arguments.torchgwas_full_genotype_path)
    assert torchgwas_command[torchgwas_command.index("--genotype-format") + 1] == "plink"
    assert "--sample-file" not in torchgwas_command
    assert torchgwas_command[torchgwas_command.index("--bim") + 1] == str(arguments.torchgwas_full_bim_path)
    assert torchgwas_command[torchgwas_command.index("--fam") + 1] == str(arguments.torchgwas_full_fam_path)
    assert torchgwas_command[torchgwas_command.index("--compute-dtype") + 1] == "float32"
    assert torchgwas_command[torchgwas_command.index("--chunk-size") + 1] == "16384"
    assert torchgwas_command[torchgwas_command.index("--trait-columns") + 1] == "phenotype_continuous"


def test_torchgwas_chr22_dry_run_writes_standard_artifacts(tmp_path: Path) -> None:
    arguments = torchgwas_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'torchgwas'}",
            "tool.dry_run=true",
            "tool.validate_inputs=false",
            "tool.variant_limit=1000",
        ]
    )

    case_results = torchgwas_benchmark.run_benchmark(arguments)

    assert {case_result.status for case_result in case_results} == {torchgwas_benchmark.RunStatus.DRY_RUN}
    assert (arguments.output_directory / "report.json").is_file()
    assert (arguments.output_directory / "summary.md").is_file()
    assert (arguments.output_directory / "metrics.jsonl").is_file()
    assert (arguments.output_directory / "commands" / "commands.jsonl").is_file()
    assert (arguments.output_directory / "artifact_manifest.json").is_file()
    report_payload = tooling_reports.read_json_report(arguments.output_directory / "report.json")
    assert report_payload["run"]["status"] == "dry_run"
    assert len(report_payload["trials"]) == 4
    torchgwas_trial = next(
        trial
        for trial in typing.cast("list[dict[str, object]]", report_payload["trials"])
        if trial["case_id"] == "torchgwas_linear_gpu_cold"
    )
    command_arguments = typing.cast("list[str]", torchgwas_trial["command_arguments"])
    assert "torchgwas_subset/genotype.npy" in command_arguments[command_arguments.index("--genotype") + 1]


def test_torchgwas_count_gzip_table_rows(tmp_path: Path) -> None:
    table_path = tmp_path / "results.tsv.gz"
    with gzip.open(table_path, "wt", encoding="utf-8") as table_file:
        table_file.write("marker_id\ttrait\tp_value\n")
        table_file.write("rs1\tphenotype\t0.5\n")
        table_file.write("rs2\tphenotype\t0.1\n")

    assert torchgwas_benchmark.count_gzip_table_rows(table_path) == 2
    assert torchgwas_benchmark.count_gzip_table_rows(tmp_path / "missing.tsv.gz") is None


def test_torchgwas_converts_tab_separated_plink2_raw(tmp_path: Path) -> None:
    raw_path = tmp_path / "plink2_subset.raw"
    genotype_path = tmp_path / "genotype.npy"
    sample_ids_path = tmp_path / "sample_ids.txt"
    marker_ids_path = tmp_path / "marker_ids.txt"
    raw_path.write_text(
        "\t".join(["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "rs1_A", "rs2_G"]) + "\n"
        "0\tHG00096\t0\t0\t1\t-9\t2\t1\n"
        "0\tHG00097\t0\t0\t2\t-9\t0\tNA\n",
        encoding="utf-8",
    )

    torchgwas_benchmark.convert_plink2_raw_to_numpy(
        raw_path=raw_path,
        genotype_path=genotype_path,
        sample_ids_path=sample_ids_path,
        marker_ids_path=marker_ids_path,
    )

    genotype_array = np.load(genotype_path)
    assert genotype_array.shape == (2, 2)
    assert genotype_array[0].tolist() == [2.0, 1.0]
    assert np.isnan(genotype_array[1, 1])
    assert sample_ids_path.read_text(encoding="utf-8").splitlines() == ["HG00096", "HG00097"]
    assert marker_ids_path.read_text(encoding="utf-8").splitlines() == ["rs1_A", "rs2_G"]


def test_hydra_tensorqtl_chr22_config_converts_to_arguments(tmp_path: Path) -> None:
    arguments = tensorqtl_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'tensorqtl'}",
            "tool.dry_run=true",
            "tool.variant_limit=1000",
        ]
    )

    assert arguments.bgen_path.name == "1kg_chr22_full.bgen"
    assert arguments.sample_path.name == "1kg_chr22_full.sample"
    assert arguments.plink_prefix_path.name == "1kg_chr22_full"
    assert arguments.bed_path.name == "1kg_chr22_full.bed"
    assert arguments.bim_path.name == "1kg_chr22_full.bim"
    assert arguments.fam_path.name == "1kg_chr22_full.fam"
    assert arguments.phenotype_column == "phenotype_continuous"
    assert arguments.covariate_columns == ("age", "sex")
    assert arguments.prediction_list_path == arguments.data_directory / "baselines" / "regenie_step1_qt_pred.list"
    assert arguments.chunk_size == 16384
    assert arguments.variant_limit == 1000
    assert arguments.tensorqtl_batch_size == 20000
    assert arguments.tensorqtl_commit == tensorqtl_benchmark.TENSORQTL_MAIN_COMMIT
    assert arguments.tensorqtl_python == "3.12"
    assert arguments.torch_package == "torch==2.5.1+cu121"
    assert arguments.torch_package_index_url == "https://download.pytorch.org/whl/cu121"
    assert tensorqtl_benchmark.tensorqtl_venv_directory(arguments).name == "venv-plink-3_12"
    assert arguments.output_directory == tmp_path / "tensorqtl"


def test_tensorqtl_prepares_sample_ordered_matrices(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "genotypes.fam").write_text(
        "0\tHG00097\t0\t0\t2\t-9\n0\tHG00096\t0\t0\t1\t-9\n",
        encoding="utf-8",
    )
    (data_directory / "pheno_cont.txt").write_text(
        "FID\tIID\tphenotype_continuous\n0\tHG00096\t1.5\n0\tHG00097\t2.5\n",
        encoding="utf-8",
    )
    (data_directory / "covariates.txt").write_text(
        "FID\tIID\tage\tsex\n0\tHG00096\t42\t1\n0\tHG00097\t48\t2\n",
        encoding="utf-8",
    )
    arguments = tensorqtl_benchmark.build_arguments_from_overrides(
        [
            f"tool.data_dir={data_directory}",
            "tool.plink_prefix=genotypes",
            "tool.bgen=missing.bgen",
            "tool.sample=missing.sample",
            "tool.prediction_list=missing.list",
            f"tool.output_dir={tmp_path / 'tensorqtl'}",
            "tool.dry_run=true",
            "tool.validate_inputs=false",
            "tool.variant_limit=null",
        ]
    )

    input_spec = tensorqtl_benchmark.prepare_tensorqtl_matrices(arguments=arguments, command_results=[])

    assert input_spec.genotype_prefix_path == tmp_path / "tensorqtl" / "tensorqtl_inputs" / "plink_genotypes"
    assert input_spec.phenotype_matrix_path.read_text(encoding="utf-8").splitlines() == [
        "phenotype_id\tHG00097\tHG00096",
        "phenotype_continuous\t2.5\t1.5",
    ]
    assert input_spec.covariate_matrix_path.read_text(encoding="utf-8").splitlines() == [
        "covariate_id\tHG00097\tHG00096",
        "age\t48\t42",
        "sex\t2\t1",
    ]


def test_tensorqtl_chr22_cases_render_g_and_competitor_commands(tmp_path: Path) -> None:
    arguments = tensorqtl_benchmark.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'tensorqtl'}",
            "tool.dry_run=true",
            "tool.variant_limit=null",
        ]
    )
    input_spec = tensorqtl_benchmark.TensorqtlInputSpec(
        genotype_prefix_path=arguments.plink_prefix_path,
        phenotype_matrix_path=tmp_path / "phenotypes.tsv",
        covariate_matrix_path=tmp_path / "covariates.tsv",
    )

    cases = tensorqtl_benchmark.build_cases(arguments, input_spec)
    case_by_identifier = {benchmark_case.case_id: benchmark_case for benchmark_case in cases}
    g_command = list(case_by_identifier["g_linear_gpu_cold"].command_arguments)
    tensorqtl_command = list(case_by_identifier["tensorqtl_trans_dense_gpu_cold"].command_arguments)

    assert [benchmark_case.case_id for benchmark_case in cases] == [
        "g_linear_gpu_cold",
        "tensorqtl_trans_dense_gpu_cold",
        "g_linear_gpu_warm",
        "tensorqtl_trans_dense_gpu_warm",
    ]
    assert "--qt" in g_command
    assert g_command[g_command.index("--bgen") + 1] == str(arguments.bgen_path)
    assert g_command[g_command.index("--phenoCol") + 1] == "phenotype_continuous"
    assert g_command[g_command.index("--pred") + 1] == str(arguments.prediction_list_path)
    assert g_command[g_command.index("--device") + 1] == "gpu"

    assert Path(tensorqtl_command[0]).name == "python"
    assert tensorqtl_command[1:3] == ["-m", "tensorqtl"]
    assert tensorqtl_command[3] == str(arguments.plink_prefix_path)
    assert tensorqtl_command[4] == str(input_spec.phenotype_matrix_path)
    assert tensorqtl_command[tensorqtl_command.index("--covariates") + 1] == str(input_spec.covariate_matrix_path)
    assert tensorqtl_command[tensorqtl_command.index("--mode") + 1] == "trans"
    assert "--return_dense" in tensorqtl_command
    assert tensorqtl_command[tensorqtl_command.index("--batch_size") + 1] == "20000"


def test_tensorqtl_chr22_dry_run_writes_standard_artifacts(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "genotypes.fam").write_text(
        "0\tHG00096\t0\t0\t1\t-9\n",
        encoding="utf-8",
    )
    (data_directory / "pheno_cont.txt").write_text(
        "FID\tIID\tphenotype_continuous\n0\tHG00096\t1.5\n",
        encoding="utf-8",
    )
    (data_directory / "covariates.txt").write_text(
        "FID\tIID\tage\tsex\n0\tHG00096\t42\t1\n",
        encoding="utf-8",
    )
    arguments = tensorqtl_benchmark.build_arguments_from_overrides(
        [
            f"tool.data_dir={data_directory}",
            "tool.plink_prefix=genotypes",
            "tool.bgen=missing.bgen",
            "tool.sample=missing.sample",
            "tool.prediction_list=missing.list",
            f"tool.output_dir={tmp_path / 'tensorqtl'}",
            "tool.dry_run=true",
            "tool.validate_inputs=false",
            "tool.variant_limit=null",
        ]
    )

    case_results = tensorqtl_benchmark.run_benchmark(arguments)

    assert {case_result.status for case_result in case_results} == {tensorqtl_benchmark.RunStatus.DRY_RUN}
    assert (arguments.output_directory / "report.json").is_file()
    assert (arguments.output_directory / "summary.md").is_file()
    assert (arguments.output_directory / "metrics.jsonl").is_file()
    assert (arguments.output_directory / "commands" / "commands.jsonl").is_file()
    assert (arguments.output_directory / "artifact_manifest.json").is_file()
    report_payload = tooling_reports.read_json_report(arguments.output_directory / "report.json")
    assert report_payload["run"]["status"] == "dry_run"
    assert len(report_payload["trials"]) == 4
    tensorqtl_trial = next(
        trial
        for trial in typing.cast("list[dict[str, object]]", report_payload["trials"])
        if trial["case_id"] == "tensorqtl_trans_dense_gpu_cold"
    )
    command_arguments = typing.cast("list[str]", tensorqtl_trial["command_arguments"])
    assert command_arguments[command_arguments.index("--mode") + 1] == "trans"
    assert "--return_dense" in command_arguments


def test_named_justfile_workflow_configs_compose() -> None:
    config_names = [
        "data_baseline_binary",
        "data_baseline_quantitative",
        "data_verify_binary_gpu_inputs",
        "bench_bgen_reader",
        "bench_callback_overhead",
        "bench_callback_overhead_gpu",
        "bench_binary_hot_gpu",
        "bench_binary_hot_gpu_smoke",
        "bench_output_stages_gpu",
        "benchmark_tensorqtl_chr22",
        "benchmark_torchgwas_chr22",
        "bench_linear_startup_gpu",
        "bench_linear_startup_gpu_parquet",
        "perf_cpu",
        "perf_gpu",
        "rust_build_profiles",
        "schema_check",
        "matrix_chr10",
        "matrix_chr10_dry",
        "matrix_chr10_smoke",
        "matrix_chr22",
        "matrix_chr22_dry",
        "matrix_chr22_smoke",
        "profile_app_full",
        "profile_app_full_dry",
        "profile_app_full_smoke",
        "profile_chr10_binary_gpu_full",
        "profile_chr10_binary_gpu_dry",
        "profile_chr10_binary_gpu_smoke",
        "debug_check_internal_defaults",
        "debug_check_internal_init_exports",
        "debug_check_pyo3_stub",
        "debug_check_justfile",
        "debug_check_native_cli_frontend",
        "debug_schema_check",
    ]

    for config_name in config_names:
        config = tooling_configuration.compose_config(config_name=config_name, include_hydra_config=True)
        assert "tool" in config
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
    campaign_budget = deep_profile.build_campaign_budget(
        arguments=arguments,
        output_directory=tmp_path / "profile",
    )
    profile_plan = deep_profile.build_profile_plan(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=tmp_path / "profile",
        campaign_budget=campaign_budget,
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
    assert arguments.regenie_executable is None
    assert arguments.regenie_baseline_trait_types == "quantitative"
    assert arguments.regenie_baseline_variant_limit is None
    assert arguments.regenie_baseline_warmups == 0
    assert arguments.regenie_baseline_trials == 1
    assert arguments.stage_timing_mode == deep_profile.ProfileStageTimingMode.EXACT
    assert arguments.workload_keys == "quantitative_cpu,quantitative_gpu,binary_cpu,binary_gpu"
    assert arguments.max_subprocess_runs == 1000
    assert arguments.max_major_profiler_runs == 64
    assert arguments.allow_over_budget is False
    assert arguments.py_spy_timeout_seconds == 1800
    assert arguments.scalene_timeout_seconds == 1800
    assert arguments.memray_timeout_seconds == 1800
    assert arguments.linux_perf_timeout_seconds == 1200
    assert arguments.nsight_systems_timeout_seconds == 1800
    assert arguments.nsight_compute_timeout_seconds == 1800
    assert campaign_budget.workload_keys == (
        "quantitative_cpu",
        "quantitative_gpu",
        "binary_cpu",
        "binary_gpu",
    )
    assert campaign_budget.total_subprocess_run_count == 6335
    assert campaign_budget.over_subprocess_budget is True
    assert profile_plan.workload_keys == [
        "quantitative_cpu",
        "quantitative_gpu",
        "binary_cpu",
        "binary_gpu",
    ]
    assert [section.name for section in profile_plan.campaign_budget.sections] == [
        "bgen_pre_sweep",
        "tuning",
        "finalists",
        "headline_trials",
        "deep_profilers",
        "logging_perturbation",
        "rust_criterion",
    ]
    assert profile_plan.profiler_modes == {
        "regenie_baseline": True,
        "jax_trace": True,
        "jax_memory_profile": True,
        "python_cprofile": True,
        "py_spy": True,
        "scalene": False,
        "memray": False,
        "linux_perf": True,
        "nsight_systems": False,
        "nsight_compute": False,
        "rust_criterion": True,
        "logging_perturbation": True,
    }
    assert profile_plan.logging_perturbation_cases
    assert profile_plan.regenie_baseline_scope is not None
    assert "py_spy" in profile_plan.profiler_tools
    assert profile_plan.rust_benchmark_commands == [
        ["cargo", "bench", "--bench", "bgen_read"],
        ["cargo", "bench", "--bench", "preprocess"],
    ]
    off_arguments = deep_profile.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile-off'}",
            "tool.dry_run=true",
            "telemetry.stage_timing_mode=off",
        ]
    )
    assert off_arguments.stage_timing_mode == deep_profile.ProfileStageTimingMode.OFF


def test_cpu_feature_aware_jax_cache_directory_uses_host_and_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpuinfo_path = tmp_path / "cpuinfo"
    cpuinfo_path.write_text(
        "\n".join(
            (
                "processor   : 0",
                "vendor_id   : GenuineIntel",
                "cpu family  : 6",
                "model       : 143",
                "model name  : Test CPU",
                "stepping    : 8",
                "flags       : avx512f avx2 sse4_2",
                "",
            )
        ),
        encoding="utf-8",
    )
    cache_parent = tmp_path / "cpu-cache"
    monkeypatch.setenv("G_PROFILE_CPU_JAX_CACHE_PARENT", str(cache_parent))
    monkeypatch.setattr(tooling_jax_cache.socket, "gethostname", lambda: "cantor/node")

    cache_directory = tooling_jax_cache.resolve_cpu_feature_aware_cache_directory(
        Path("/mnt/beegfs/profiles/current/jax_cache"),
        cpuinfo_path=cpuinfo_path,
    )

    assert cache_directory.parent.parent == cache_parent / "host-cantor-node"
    assert cache_directory.parent.name == f"features-{tooling_jax_cache.cpu_feature_fingerprint(cpuinfo_path)}"
    assert cache_directory.name.startswith("jax_cache-")


def test_deep_profile_resolves_cpu_cache_by_node_features_and_keeps_gpu_job_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_cpu_feature_fingerprint(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str:
        del cpuinfo_path
        return "abc123"

    monkeypatch.setenv("G_PROFILE_CPU_JAX_CACHE_PARENT", str(tmp_path / "cpu-cache"))
    monkeypatch.setenv("G_PROFILE_GPU_JAX_CACHE_PARENT", str(tmp_path / "gpu-cache"))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(tooling_jax_cache.socket, "gethostname", lambda: "cantor")
    monkeypatch.setattr(tooling_jax_cache, "cpu_feature_fingerprint", fake_cpu_feature_fingerprint)
    cpu_candidate = deep_profile.Step2Candidate(
        trait_type="binary",
        device="cpu",
        chunk_size=8192,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
        output_writer_thread_count=4,
        output_writer_queue_depth=8,
        bgen_decode_tile_variant_count=128,
        rayon_thread_count=2,
        firth_batch_size=64,
    )
    gpu_candidate = dataclasses.replace(cpu_candidate, device="gpu")
    base_cache_directory = tmp_path / "profile" / "jax_cache"

    cpu_cache_directory = deep_profile.resolve_profile_jax_cache_directory(cpu_candidate, base_cache_directory)
    gpu_cache_directory = deep_profile.resolve_profile_jax_cache_directory(gpu_candidate, base_cache_directory)

    assert cpu_cache_directory is not None
    assert cpu_cache_directory.parent.parent == tmp_path / "cpu-cache" / "host-cantor"
    assert cpu_cache_directory.parent.name == "features-abc123"
    assert cpu_cache_directory.name.startswith("jax_cache-")
    assert gpu_cache_directory == tmp_path / "gpu-cache" / "12345" / "jax_cache"


def test_deep_profile_smoke_overrides_profiler_timeouts(tmp_path: Path) -> None:
    base_arguments = deep_profile.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.dry_run=true",
            "tool.smoke=true",
        ]
    )
    arguments = deep_profile.apply_smoke_overrides(base_arguments)

    assert arguments.py_spy_timeout_seconds == 15
    assert arguments.scalene_timeout_seconds == 15
    assert arguments.memray_timeout_seconds == 15
    assert arguments.linux_perf_timeout_seconds == 15
    assert arguments.nsight_systems_timeout_seconds == 15
    assert arguments.nsight_compute_timeout_seconds == 15


def test_deep_profile_budget_respects_workload_subset_and_bounded_grid(tmp_path: Path) -> None:
    arguments = deep_profile.build_arguments_from_overrides(
        [
            f"tool.output_dir={tmp_path / 'profile'}",
            "tool.dry_run=true",
            "tool.include_regenie_baseline=false",
            "tool.workload_keys=[binary_gpu]",
            "tool.chunk_sizes=[2048,4096]",
            "tool.staging_depths=[1,2]",
            "tool.output_writer_thread_counts=[1,4]",
            "tool.writer_queue_depth_multipliers=[1,2]",
            "tool.firth_batch_sizes=[32]",
            "tool.bgen_decode_tile_variant_counts=[64,128]",
            "tool.rayon_thread_counts=[4,8]",
            "tool.top_bgen_candidates=1",
            "tool.top_finalists=2",
            "tool.tuning_warmups=0",
            "tool.tuning_trials=1",
            "tool.finalist_warmups=0",
            "tool.finalist_trials=2",
            "tool.headline_warmups=0",
            "tool.headline_trials=3",
        ]
    )

    campaign_budget = deep_profile.build_campaign_budget(
        arguments=arguments,
        output_directory=tmp_path / "profile",
    )
    sections_by_name = {section.name: section for section in campaign_budget.sections}

    assert campaign_budget.workload_keys == ("binary_gpu",)
    assert campaign_budget.total_subprocess_run_count == 37
    assert campaign_budget.over_subprocess_budget is False
    assert sections_by_name["bgen_pre_sweep"].candidate_count == 4
    assert sections_by_name["tuning"].candidate_count == 16
    assert sections_by_name["finalists"].candidate_count == 2
    assert sections_by_name["headline_trials"].subprocess_run_count == 3
    assert sections_by_name["deep_profilers"].major_profiler_run_count == 4
    assert sections_by_name["logging_perturbation"].subprocess_run_count == 4
    assert sections_by_name["rust_criterion"].subprocess_run_count == 2


def test_deep_profile_workload_selectors_expand_groups() -> None:
    assert deep_profile.parse_profile_workload_keys("gpu") == (
        deep_profile.ProfileWorkloadKey.QUANTITATIVE_GPU,
        deep_profile.ProfileWorkloadKey.BINARY_GPU,
    )
    assert deep_profile.parse_profile_workload_keys("binary_cpu,binary_cpu") == (
        deep_profile.ProfileWorkloadKey.BINARY_CPU,
    )


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

    rust_build_arguments = rust_build_profiles.build_arguments_from_overrides(
        [
            "tool.labels=[dev-fast,dev-opt,perf-thin-cgu1]",
            "tool.output_parent=results/perf/test-rust-build-profiles",
            "tool.incremental_touch_paths=[src/python/mod.rs]",
            "tool.run_bgen_reader_smoke=true",
        ]
    )
    assert rust_build_arguments.labels == (
        rust_build_profiles.BuildProfileLabel.DEV_FAST,
        rust_build_profiles.BuildProfileLabel.DEV_OPT,
        rust_build_profiles.BuildProfileLabel.PERF_THIN_CGU1,
    )
    assert rust_build_arguments.output_parent == Path("results/perf/test-rust-build-profiles")
    assert rust_build_arguments.incremental_touch_paths == (Path("src/python/mod.rs"),)
    assert rust_build_arguments.run_bgen_reader_smoke is True

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
    assert "--variant_limit" in chr10_run_specs[0].command_arguments
    assert "1000" in chr10_run_specs[0].command_arguments
    assert "--firth_batch_size" in chr10_run_specs[0].command_arguments
    assert "64" in chr10_run_specs[0].command_arguments
    assert "--firth_candidate_capacity" in chr10_run_specs[0].command_arguments
    assert "1024" in chr10_run_specs[0].command_arguments
    assert "--firth_batch_size" not in chr10_run_specs[3].command_arguments
    assert "--firth_candidate_capacity" not in chr10_run_specs[3].command_arguments
    assert "--no-jax_persistent_cache" in chr10_run_specs[0].command_arguments
    assert "--jax_cache_dir" in gpu_cache_arguments
    assert "--jax_cache_dir" in cached_gpu_arguments
    assert (
        gpu_cache_arguments[gpu_cache_arguments.index("--jax_cache_dir") + 1]
        == cached_gpu_arguments[cached_gpu_arguments.index("--jax_cache_dir") + 1]
    )

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
    assert "--variant_limit" in chr22_run_specs[0].command_arguments
    assert "1000" in chr22_run_specs[0].command_arguments


def test_matrix_cpu_persistent_cache_uses_feature_aware_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_cpu_feature_fingerprint(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str:
        del cpuinfo_path
        return "abc123"

    monkeypatch.setenv("G_PROFILE_CPU_JAX_CACHE_PARENT", str(tmp_path / "cpu-cache"))
    monkeypatch.setattr(tooling_jax_cache.socket, "gethostname", lambda: "cantor")
    monkeypatch.setattr(tooling_jax_cache, "cpu_feature_fingerprint", fake_cpu_feature_fingerprint)
    arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            "tool.dry_run=true",
            f"tool.output_dir={tmp_path / 'matrix'}",
            "tool.cpu_jax_persistent_cache=true",
            f"tool.jax_cache_dir={tmp_path / 'shared-jax-cache'}",
        ]
    )
    run_specs = regenie2_matrix.build_run_specs(arguments)
    cpu_arguments = run_specs[0].command_arguments
    gpu_arguments = run_specs[1].command_arguments

    assert "--g-jax-cache-dir" not in cpu_arguments
    assert "--g-jax-cache-dir" not in gpu_arguments
    cpu_cache_directory = Path(cpu_arguments[cpu_arguments.index("--jax_cache_dir") + 1])
    gpu_cache_directory = Path(gpu_arguments[gpu_arguments.index("--jax_cache_dir") + 1])

    assert cpu_cache_directory.parent.parent == tmp_path / "cpu-cache" / "host-cantor"
    assert cpu_cache_directory.parent.name == "features-abc123"
    assert cpu_cache_directory.name.startswith("cpu-")
    assert gpu_cache_directory == tmp_path / "shared-jax-cache" / "gpu"


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


def test_sweep_and_boolean_helpers_reject_ambiguous_values() -> None:
    assert tooling_hydra_arguments.boolean_value("false") is False
    assert tooling_hydra_arguments.boolean_value("true") is True

    for raw_value in ("default,,4", ",4", "4,"):
        try:
            tooling_sweeps.parse_optional_integer_list(raw_value)
        except ValueError as error:
            assert "empty entry" in str(error)
        else:
            raise AssertionError(f"Expected {raw_value!r} to be rejected.")

    try:
        tooling_hydra_arguments.boolean_value("maybe")
    except TypeError as error:
        assert "Expected a boolean value" in str(error)
    else:
        raise AssertionError("Expected ambiguous boolean string to be rejected.")


def test_shared_regenie_renderer_produces_valid_binary_cli_and_python_options() -> None:
    run_spec = tooling_g_regenie.RegenieRunSpec(
        trait_kind=tooling_g_regenie.RegenieTraitKind.BINARY,
        command_prefix=("g", "regenie"),
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=Path("data/input.bgen"),
            sample_path=Path("data/input.sample"),
            phenotype_path=Path("data/pheno.tsv"),
            phenotype_columns=("trait",),
            covariate_path=Path("data/covar.tsv"),
            covariate_columns=("age", "sex"),
            prediction_list_path=Path("data/pred.list"),
            output_prefix=Path("results/output"),
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=tooling_g_regenie.RegenieDevice.CPU,
            bsize=4096,
            threads=2,
            staging_depth=1,
            native_callback_batch_size=2,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            variant_limit=100,
            trusted_no_missing_diploid=True,
            trusted_bgen_validation_mode="cache_on_miss",
            bgen_decode_tile_variant_count=64,
            firth_batch_size=32,
            firth_candidate_capacity=128,
            gpu_genotype_format=None,
            jax_cache_dir=Path("cache/jax"),
            jax_persistent_cache=True,
            jax_persistent_cache_min_entry_size_bytes=-1,
            jax_persistent_cache_min_compile_time_seconds=0,
            jax_xla_autotune_cache=None,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            output_format="parquet",
            output_run_directory=None,
            writer_threads=2,
            writer_queue_depth=4,
            chunks_per_arrow_file=None,
            arrow_compression=None,
            parquet_compression=None,
            output_statistic_dtype=None,
            finalize_parquet=False,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(
            telemetry="off",
            log_dir=Path("logs"),
            stage_timings_json=Path("logs/stage.json"),
            profile_summary_json=Path("logs/profile.json"),
            log_file=Path("logs/events.jsonl"),
            log_filter="info",
            log_stderr=False,
            progress_interval_seconds=1.0,
            progress_interval_chunks=2,
        ),
        binary=tooling_g_regenie.RegenieBinaryOptions(
            firth=True,
            approx=True,
            firth_se=None,
            p_threshold=0.05,
        ),
    )

    command_arguments = tooling_g_regenie.render_g_regenie_cli(run_spec)
    python_options = tooling_g_regenie.render_python_api_options(run_spec)
    rendered_config = interface_config.RegenieConfig.from_options(python_options)

    assert "--out" in command_arguments
    assert not any(argument.startswith("--g-") for argument in command_arguments)
    assert "--bt" in command_arguments
    assert "--qt" not in command_arguments
    assert "--firth" in command_arguments
    assert "--pred" in command_arguments
    assert rendered_config.g_output.out == Path("results/output")
    assert tooling_g_regenie.expected_output_run_directory(run_spec) == (
        Path("results/output.g") / "trait.regenie2_binary.run"
    )
    explicit_output_spec = dataclasses.replace(
        run_spec,
        output=dataclasses.replace(
            run_spec.output,
            output_run_directory=Path("explicit/run-directory"),
        ),
    )
    assert tooling_g_regenie.expected_output_run_directory(explicit_output_spec) == Path("explicit/run-directory")


def test_matrix_commands_use_shared_regenie_contract() -> None:
    arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            "tool.dry_run=true",
            "tool.variant_limit=1000",
        ]
    )
    for run_spec in regenie2_matrix.build_run_specs(arguments):
        assert "--out" in run_spec.command_arguments
        assert not any(argument.startswith("--g-") for argument in run_spec.command_arguments)
        if run_spec.trait == regenie2_matrix.TraitKind.BINARY:
            assert "--bt" in run_spec.command_arguments
            assert "--firth" in run_spec.command_arguments
        else:
            assert "--qt" in run_spec.command_arguments
            assert "--firth" not in run_spec.command_arguments


def test_command_runner_records_redacted_environment_and_missing_executable(tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    command_spec = tooling_commands.build_command_spec(
        [sys.executable, "-c", "import os; print(os.environ['VISIBLE_VALUE'])"],
        env={"VISIBLE_VALUE": "shown", "SECRET_VALUE": "hidden"},
        stdout_path=stdout_path,
        sensitive_env_keys=("SECRET_VALUE",),
    )

    result = tooling_commands.run_command(command_spec)
    missing_result = tooling_commands.run_command(
        tooling_commands.build_command_spec(["definitely-not-a-real-gwas-command"]),
    )
    streaming_timeout_result = tooling_commands.run_command(
        tooling_commands.build_command_spec(
            ["/bin/sh", "-c", "printf partial; sleep 2"],
            timeout_seconds=1.0,
            stream=True,
        )
    )

    assert result.return_code == 0
    assert result.stdout.strip() == "shown"
    assert stdout_path.read_text(encoding="utf-8").strip() == "shown"
    assert result.environment_overrides["SECRET_VALUE"] == tooling_commands.REDACTED_ENVIRONMENT_VALUE
    assert missing_result.missing_executable is True
    assert missing_result.return_code is None
    assert streaming_timeout_result.timed_out is True
    assert streaming_timeout_result.stdout == "partial"


def test_versioned_report_contract_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    contract = tooling_reports.VersionedReportContract(
        schema_version=1,
        required_fields=("name",),
        optional_fields=("notes",),
        schema_field_name="schema_version",
        reject_unknown_fields=True,
    )
    report_path = tmp_path / "report.json"

    tooling_reports.write_versioned_json_report(
        report_path,
        {"schema_version": 1, "name": "ok"},
        contract,
    )
    assert tooling_reports.read_versioned_json_report(report_path, contract)["name"] == "ok"

    for payload in ({"schema_version": 1}, {"schema_version": 1, "name": "ok", "extra": True}):
        try:
            tooling_reports.validate_report_shape(payload, contract)
        except tooling_reports.ReportSchemaError:
            pass
        else:
            raise AssertionError(f"Expected report payload to be rejected: {payload!r}")


def test_tooling_artifact_bundle_writes_standard_files(tmp_path: Path) -> None:
    output_directory = tmp_path / "artifact"
    producer = tooling_artifact_format.ToolProducer(
        tool_name="test_tool",
        tool_version=1,
        repository="kirilledition/g",
        git_head="abc123",
        dirty=False,
        dirty_diff_sha256=None,
    )
    run = tooling_artifact_format.ToolRunIdentity(
        run_id="test-run",
        created_at="2026-06-26T00:00:00Z",
        status=tooling_artifact_format.ToolArtifactStatus.SUCCESS,
        status_reason=None,
        output_directory=str(output_directory),
    )
    context_snapshot = tooling_artifact_format.ToolContextSnapshot(
        repository_root=str(tmp_path),
        data_directory=None,
        output_directory=str(output_directory),
        cwd=str(tmp_path),
        hydra_chdir=False,
        machine_profile=None,
        hostname="test-host",
        slurm_job_id=None,
    )
    metric_record = tooling_artifact_format.build_metric_record(
        run_id=run.run_id,
        case_id="case",
        metric_name="wall_time_seconds",
        value=1.25,
        unit=tooling_artifact_format.MetricUnit.SECONDS.value,
        aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
        higher_is_better=False,
    )
    command_record = tooling_artifact_format.build_command_record(
        command_id="inline_python",
        tool_name=producer.tool_name,
        run_id=run.run_id,
        phase="test",
        args=[sys.executable, "-c", "print('hello')"],
        output_directory=output_directory,
        status=tooling_artifact_format.ToolArtifactStatus.SUCCESS,
    )
    report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title="Test Tool",
        configuration={"mode": "test"},
        metrics=[metric_record],
    )

    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=output_directory,
        report=report,
        events=[
            tooling_artifact_format.build_tool_event(
                tool_name=producer.tool_name,
                run_id=run.run_id,
                phase="test",
                event="completed",
                message="completed",
            )
        ],
        commands=[command_record],
    )

    assert (output_directory / "artifact_manifest.json").is_file()
    assert (output_directory / "report.json").is_file()
    assert (output_directory / "summary.md").is_file()
    assert (output_directory / "metrics.jsonl").is_file()
    assert (output_directory / "events.jsonl").is_file()
    assert (output_directory / "commands" / "commands.jsonl").is_file()
    assert (output_directory / "commands" / "scripts" / "inline_python.py").is_file()
    assert (
        schema_check.run_schema_check(
            schema_check.SchemaCheckArguments(path=output_directory, require_optional_files=True)
        ).error_messages
        == ()
    )


def test_schema_check_rejects_wrong_schema_version(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    tooling_reports.write_json_report(
        report_path,
        {
            "schema_name": "g.tooling.report",
            "schema_version": 99,
            "producer": {},
            "run": {},
        },
    )

    result = schema_check.run_schema_check(
        schema_check.SchemaCheckArguments(path=report_path, require_optional_files=False)
    )

    assert result.error_messages
    assert "Expected schema_version=1" in result.error_messages[0]


def test_matrix_dry_run_writes_tooling_artifact_format(tmp_path: Path) -> None:
    output_directory = tmp_path / "matrix"
    arguments = regenie2_matrix.build_arguments_from_overrides(
        [
            "tool.dry_run=true",
            f"tool.output_dir={output_directory}",
            "tool.variant_limit=1000",
        ]
    )

    run_results = regenie2_matrix.run_matrix(arguments)

    assert len(run_results) == 6
    assert {run_result.status for run_result in run_results} == {regenie2_matrix.RunStatus.DRY_RUN}
    assert (output_directory / "manifest.json").is_file()
    assert (output_directory / "report.md").is_file()
    assert (output_directory / "artifact_manifest.json").is_file()
    assert (output_directory / "report.json").is_file()
    report_payload = tooling_reports.read_json_report(output_directory / "report.json")
    assert report_payload["schema_name"] == "g.tooling.report"
    assert report_payload["run"]["status"] == "dry_run"
    assert len(tooling_reports.read_jsonl(output_directory / "commands" / "commands.jsonl")) == 6
    assert (
        schema_check.run_schema_check(
            schema_check.SchemaCheckArguments(path=output_directory, require_optional_files=True)
        ).error_messages
        == ()
    )


def test_grouped_cli_registries_document_tool_names() -> None:
    assert tooling_registry.registered_tool_names(grouped_benchmark.TOOLS) == (
        "baselines",
        "linear_startup",
        "profile_comparison",
        "regenie_comparison",
    )
    assert tooling_registry.registered_tool_names(grouped_data.TOOLS) == (
        "fetch",
        "regenie_baseline",
        "simulate",
    )
    assert "check_pyo3_stub" in tooling_registry.registered_tool_names(grouped_debug.TOOLS)
    assert "schema_check" in tooling_registry.registered_tool_names(grouped_debug.TOOLS)
    assert "schema_check" in tooling_registry.registered_tool_names(grouped_debug.TOOLS)
    assert tooling_registry.registered_tool_names(grouped_performance.TOOLS) == (
        "compare",
        "jax_runtime",
        "smoke",
    )
    assert tooling_registry.registered_tool_names(grouped_server.TOOLS) == ("bootstrap_tools", "nsight_tools")


def test_pyo3_stub_checker_reads_rustfmt_wrapped_pyfunction_registration() -> None:
    registration_text = """
    module.add_function(wrap_pyfunction!(
        build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_runner_run_completed_diagnostic_payload, module)?)?;
    """

    assert check_pyo3_stub.FUNC_PATTERN.findall(registration_text) == [
        "build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload",
        "build_runner_run_completed_diagnostic_payload",
    ]


def test_pyo3_stub_checker_scans_delegated_registration_modules() -> None:
    assert Path("src/python/association_backend.rs") not in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/callback_diagnostics.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/callback_progress.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/callback_queue.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/callback_runtime_resources.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/callback_summary.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/jax_runtime.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/logging.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/genotype.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/host_policy.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/output.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/preflight.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/preparation.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/prediction_sources.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/run_events.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/run_engine.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/run_metadata.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/runtime.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/runtime_state.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/sample_alignment.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/schedule.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/shutdown.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/telemetry_policy.rs") in check_pyo3_stub.RUST_EXPORT_FILES
    assert Path("src/python/timing.rs") in check_pyo3_stub.RUST_EXPORT_FILES


def test_tooling_entrypoint_exposes_cli_surface() -> None:
    assert benchmark_bgen_reader.build_arguments_from_overrides is not None
    assert benchmark_bgen_reader.hydra_main is not None
    assert benchmark_bgen_reader.BenchmarkPathMode is regenie_bgen_reader.BenchmarkPathMode
    assert rust_build_profiles.build_arguments_from_overrides is not None
    assert rust_build_profiles.hydra_main is not None


def test_deep_profile_models_live_in_profile_package() -> None:
    assert profile_deep_models.ProfileArguments.__module__ == "tooling.profile_deep.models"
    assert profile_deep_models.Step2Candidate.__module__ == "tooling.profile_deep.models"
    assert profile_deep_models.CampaignBudget.__module__ == "tooling.profile_deep.models"
    assert deep_profile.ProfileArguments is profile_deep_models.ProfileArguments
    assert deep_profile.Step2Candidate is profile_deep_models.Step2Candidate
    assert deep_profile.CampaignBudget is profile_deep_models.CampaignBudget


def test_deep_profile_budget_helpers_live_in_profile_package() -> None:
    assert profile_deep_budget.build_campaign_budget.__module__ == "tooling.profile_deep.budget"
    assert profile_deep_budget.enforce_campaign_budget.__module__ == "tooling.profile_deep.budget"
    assert profile_deep_budget.parse_profile_workload_keys.__module__ == "tooling.profile_deep.budget"
    assert profile_deep_budget.build_logging_perturbation_cases.__module__ == "tooling.profile_deep.budget"
    assert deep_profile.build_campaign_budget is profile_deep_budget.build_campaign_budget
    assert deep_profile.enforce_campaign_budget is profile_deep_budget.enforce_campaign_budget
    assert deep_profile.parse_profile_workload_keys is profile_deep_budget.parse_profile_workload_keys
    assert deep_profile.build_logging_perturbation_cases is profile_deep_budget.build_logging_perturbation_cases


def test_deep_profile_config_helpers_live_in_profile_package() -> None:
    assert profile_deep_config.build_arguments_from_config.__module__ == "tooling.profile_deep.config"
    assert profile_deep_config.build_arguments_from_overrides.__module__ == "tooling.profile_deep.config"
    assert profile_deep_config.apply_smoke_overrides.__module__ == "tooling.profile_deep.config"
    assert profile_deep_config.profile_configuration_payload.__module__ == "tooling.profile_deep.config"
    assert deep_profile.build_arguments_from_config is profile_deep_config.build_arguments_from_config
    assert deep_profile.build_arguments_from_overrides is profile_deep_config.build_arguments_from_overrides
    assert deep_profile.apply_smoke_overrides is profile_deep_config.apply_smoke_overrides
    assert deep_profile.profile_configuration_payload is profile_deep_config.profile_configuration_payload


def test_deep_profile_jax_cache_helpers_live_in_profile_package() -> None:
    assert profile_deep_jax_cache.resolve_profile_jax_cache_directory.__module__ == "tooling.profile_deep.jax_cache"
    assert profile_deep_jax_cache.collect_jax_cache_snapshot.__module__ == "tooling.profile_deep.jax_cache"
    assert profile_deep_jax_cache.build_jax_cache_diagnostics.__module__ == "tooling.profile_deep.jax_cache"
    assert profile_deep_jax_cache.build_jax_cold_warm_diagnostics.__module__ == "tooling.profile_deep.jax_cache"
    assert (
        deep_profile.resolve_profile_jax_cache_directory is profile_deep_jax_cache.resolve_profile_jax_cache_directory
    )
    assert deep_profile.collect_jax_cache_snapshot is profile_deep_jax_cache.collect_jax_cache_snapshot
    assert deep_profile.build_jax_cache_diagnostics is profile_deep_jax_cache.build_jax_cache_diagnostics
    assert deep_profile.build_jax_cold_warm_diagnostics is profile_deep_jax_cache.build_jax_cold_warm_diagnostics


def test_rust_build_profile_specs_map_expected_cargo_profiles() -> None:
    assert rust_build_profiles.PROFILE_SPECS[rust_build_profiles.BuildProfileLabel.DEV_FAST].cargo_profile == "dev-fast"
    assert (
        rust_build_profiles.PROFILE_SPECS[rust_build_profiles.BuildProfileLabel.PERF_THIN_CGU8].cargo_profile == "perf"
    )
    assert (
        rust_build_profiles.PROFILE_SPECS[rust_build_profiles.BuildProfileLabel.PERF_FAT_CGU1].cargo_profile
        == "perf-max"
    )
    assert (
        rust_build_profiles.PROFILE_SPECS[rust_build_profiles.BuildProfileLabel.PERF_O2_THIN_CGU8].cargo_profile
        == "perf-o2"
    )


def test_rust_build_profile_command_environment_contains_target_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = rust_build_profiles.PROFILE_SPECS[rust_build_profiles.BuildProfileLabel.PERF_THIN_CGU1]
    environment = rust_build_profiles.build_environment(spec, Path("target/rust-build-profiles/perf-thin-cgu1"))

    assert environment["CARGO_TARGET_DIR"] == "target/rust-build-profiles/perf-thin-cgu1"
    assert "RUSTFLAGS" not in environment
    monkeypatch.delenv("CARGO_BUILD_JOBS", raising=False)
    assert rust_build_profiles.maturin_develop_command(spec) == (
        "uv",
        "run",
        "--no-sync",
        "maturin",
        "develop",
        "--profile",
        "perf-thin-cgu1",
        "--uv",
    )
    monkeypatch.setenv("CARGO_BUILD_JOBS", "30")
    assert rust_build_profiles.maturin_develop_command(spec) == (
        "uv",
        "run",
        "--no-sync",
        "maturin",
        "develop",
        "-j",
        "30",
        "--profile",
        "perf-thin-cgu1",
        "--uv",
    )


def test_rust_build_profile_specs_do_not_set_rustflags() -> None:
    assert all(profile_spec.rustflags == "" for profile_spec in rust_build_profiles.PROFILE_SPECS.values())


def test_rust_build_profile_touch_restores_source_timestamp(tmp_path: Path) -> None:
    source_path = tmp_path / "source.rs"
    source_path.write_text("fn main() {}\n", encoding="utf-8")
    original_access_time_nanoseconds = 1_700_000_000_000_000_000
    original_modification_time_nanoseconds = 1_700_000_000_100_000_000
    os.utime(source_path, ns=(original_access_time_nanoseconds, original_modification_time_nanoseconds))
    original_stat = source_path.stat()

    timestamp = rust_build_profiles.touch_source_path(source_path)
    touched_stat = source_path.stat()
    rust_build_profiles.restore_timestamp(timestamp)
    restored_stat = source_path.stat()

    assert touched_stat.st_mtime_ns != original_stat.st_mtime_ns
    assert restored_stat.st_atime_ns == original_stat.st_atime_ns
    assert restored_stat.st_mtime_ns == original_stat.st_mtime_ns
