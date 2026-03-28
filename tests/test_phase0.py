from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", str(REPOSITORY_ROOT / "data")))


def load_module(module_name: str, relative_path: str) -> ModuleType:
    """Load a Python module directly from a repository-relative path.

    Args:
        module_name: Synthetic module name for the loaded file.
        relative_path: Repository-relative file path.

    Returns:
        Imported module object.

    """
    module_path = REPOSITORY_ROOT / relative_path
    module_specification = importlib.util.spec_from_file_location(module_name, module_path)
    assert module_specification is not None
    assert module_specification.loader is not None

    module = importlib.util.module_from_spec(module_specification)
    sys.modules[module_name] = module
    module_specification.loader.exec_module(module)
    return module


fetch_1kg_module = load_module("fetch_1kg_module", "scripts/fetch_1kg.py")
benchmark_module = load_module("benchmark_module", "scripts/benchmark.py")
simulate_phenos_module = load_module("simulate_phenos_module", "scripts/simulate_phenos.py")


def test_fetch_paths_match_phase_zero_layout() -> None:
    """Ensure Phase 0 uses the documented dataset prefixes."""
    dataset_paths = fetch_1kg_module.build_dataset_paths()
    assert dataset_paths.data_directory == Path("data")
    assert dataset_paths.full_dataset_prefix == Path("data/1kg_chr22_full")
    assert dataset_paths.toy_dataset_prefix == Path("data/1kg_chr22_toy")


def test_simulated_tables_are_deterministic_and_valid() -> None:
    """Ensure phenotype generation is deterministic and covariates stay valid."""
    family_table = pd.DataFrame(
        {
            "family_identifier": ["family1", "family2", "family3"],
            "individual_identifier": ["sample1", "sample2", "sample3"],
            "paternal_identifier": [0, 0, 0],
            "maternal_identifier": [0, 0, 0],
            "reported_sex": [1, 0, 2],
            "placeholder_phenotype": [-9, -9, -9],
        }
    )

    first_tables = simulate_phenos_module.create_phenotype_and_covariate_tables(family_table)
    second_tables = simulate_phenos_module.create_phenotype_and_covariate_tables(family_table)

    np.testing.assert_allclose(
        first_tables.continuous_table["phenotype_continuous"].to_numpy(),
        second_tables.continuous_table["phenotype_continuous"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_tables.binary_table["phenotype_binary"].to_numpy(),
        second_tables.binary_table["phenotype_binary"].to_numpy(),
        atol=0,
    )
    np.testing.assert_allclose(
        first_tables.covariate_table[["age", "sex"]].to_numpy(),
        second_tables.covariate_table[["age", "sex"]].to_numpy(),
        atol=0,
    )
    assert (first_tables.covariate_table["age"] >= simulate_phenos_module.MINIMUM_AGE_YEARS).all()
    assert set(first_tables.covariate_table["sex"].to_list()).issubset({1, 2})


def test_regenie_step_two_command_uses_required_bgen_inputs() -> None:
    """Ensure Regenie step 2 uses both sample and prediction inputs."""
    baseline_paths = benchmark_module.build_baseline_paths()
    command_arguments = benchmark_module.build_regenie_step2_command("regenie", baseline_paths)

    assert "--sample" in command_arguments
    assert str(baseline_paths.sample_path) in command_arguments
    assert "--pred" in command_arguments
    assert str(baseline_paths.regenie_prediction_list_path) in command_arguments
    assert "--ref-first" in command_arguments


def test_hail_commands_use_managed_runner_and_expected_outputs() -> None:
    """Ensure Hail baseline commands point at the managed runner and output paths."""
    baseline_paths = benchmark_module.build_baseline_paths()

    linear_command_arguments = benchmark_module.build_hail_linear_command("/tmp/hail-python", baseline_paths)
    logistic_command_arguments = benchmark_module.build_hail_logistic_command(
        "/tmp/hail-python",
        baseline_paths,
        test_name="wald",
    )
    suite_command_arguments = benchmark_module.build_hail_suite_command(
        "/tmp/hail-python",
        baseline_paths,
        cache_mode="require",
    )

    assert linear_command_arguments[0] == "/tmp/hail-python"
    assert linear_command_arguments[1].endswith("scripts/run_hail_baseline.py")
    assert "--matrix-table-cache" in linear_command_arguments
    assert str(baseline_paths.hail_matrix_table_path) in linear_command_arguments
    assert "--cache-mode" in linear_command_arguments
    assert "require" in linear_command_arguments
    assert "--glm" in linear_command_arguments
    assert "linear" in linear_command_arguments
    assert str(benchmark_module.hail_output_path(baseline_paths, "hail_cont")) in linear_command_arguments
    assert str(benchmark_module.hail_log_path(baseline_paths, "hail_cont")) in linear_command_arguments

    assert logistic_command_arguments[0] == "/tmp/hail-python"
    assert logistic_command_arguments[1].endswith("scripts/run_hail_baseline.py")
    assert "--matrix-table-cache" in logistic_command_arguments
    assert str(baseline_paths.hail_matrix_table_path) in logistic_command_arguments
    assert "--glm" in logistic_command_arguments
    assert "logistic" in logistic_command_arguments
    assert "--logistic-test" in logistic_command_arguments
    assert "wald" in logistic_command_arguments
    assert str(benchmark_module.hail_output_path(baseline_paths, "hail_bin_wald")) in logistic_command_arguments
    assert str(benchmark_module.hail_log_path(baseline_paths, "hail_bin_wald")) in logistic_command_arguments

    assert suite_command_arguments[0] == "/tmp/hail-python"
    assert suite_command_arguments[1].endswith("scripts/run_hail_benchmark_suite.py")
    assert "--report-path" in suite_command_arguments
    assert str(baseline_paths.hail_suite_report_path) in suite_command_arguments
    assert str(benchmark_module.hail_output_path(baseline_paths, "hail_cont")) in suite_command_arguments
    assert str(benchmark_module.hail_output_path(baseline_paths, "hail_bin_wald")) in suite_command_arguments
    assert str(benchmark_module.hail_output_path(baseline_paths, "hail_bin_firth")) in suite_command_arguments


def test_validate_input_files_reports_all_missing_paths(tmp_path: Path) -> None:
    """Ensure benchmark preflight lists every missing required file."""
    baseline_paths = benchmark_module.BaselinePaths(
        data_directory=tmp_path,
        baseline_directory=tmp_path / "baselines",
        bed_prefix=tmp_path / "missing_dataset",
        bgen_path=tmp_path / "missing_dataset.bgen",
        sample_path=tmp_path / "missing_dataset.sample",
        continuous_phenotype_path=tmp_path / "pheno_cont.txt",
        binary_phenotype_path=tmp_path / "pheno_bin.txt",
        covariate_path=tmp_path / "covariates.txt",
        hail_directory=tmp_path / "hail",
        hail_matrix_table_path=tmp_path / "hail/missing_dataset.mt",
        hail_suite_report_path=tmp_path / "baselines/hail_suite_report.json",
        regenie_prediction_list_path=tmp_path / "baselines/regenie_step1_pred.list",
    )

    with pytest.raises(FileNotFoundError) as exception_information:
        benchmark_module.validate_input_files(baseline_paths)

    error_message = str(exception_information.value)
    assert str(tmp_path / "missing_dataset.bed") in error_message
    assert str(tmp_path / "missing_dataset.bim") in error_message
    assert str(tmp_path / "missing_dataset.fam") in error_message
    assert str(tmp_path / "missing_dataset.bgen") in error_message
    assert str(tmp_path / "missing_dataset.sample") in error_message


def test_saved_phenotypes_match_regenerated_outputs_when_data_exists() -> None:
    """Validate on-disk simulated phenotype files against regenerated tables."""
    family_path = DATA_DIRECTORY / "1kg_chr22_full.fam"
    continuous_path = DATA_DIRECTORY / "pheno_cont.txt"
    binary_path = DATA_DIRECTORY / "pheno_bin.txt"
    covariate_path = DATA_DIRECTORY / "covariates.txt"
    if not all(path.exists() for path in [family_path, continuous_path, binary_path, covariate_path]):
        pytest.skip("Phase 0 data files are not available yet.")

    family_table = simulate_phenos_module.load_family_table(family_path)
    regenerated_tables = simulate_phenos_module.create_phenotype_and_covariate_tables(family_table)
    saved_continuous_table = pd.read_csv(continuous_path, sep="\t")
    saved_binary_table = pd.read_csv(binary_path, sep="\t")
    saved_covariate_table = pd.read_csv(covariate_path, sep="\t")

    np.testing.assert_allclose(
        saved_continuous_table["phenotype_continuous"].to_numpy(),
        regenerated_tables.continuous_table["phenotype_continuous"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        saved_binary_table["phenotype_binary"].to_numpy(),
        regenerated_tables.binary_table["phenotype_binary"].to_numpy(),
        atol=0,
    )
    np.testing.assert_allclose(
        saved_covariate_table[["age", "sex"]].to_numpy(),
        regenerated_tables.covariate_table[["age", "sex"]].to_numpy(),
        atol=0,
    )


def test_benchmark_report_and_baselines_are_parseable_when_present() -> None:
    """Validate saved Phase 0 benchmark artifacts when they have been generated."""
    benchmark_report_path = DATA_DIRECTORY / "benchmark_report.json"
    if not benchmark_report_path.exists():
        pytest.skip("Benchmark report has not been generated yet.")

    report = json.loads(benchmark_report_path.read_text())
    assert "hardware" in report
    assert "results" in report
    assert {"plink_cont", "plink_bin", "regenie_step1", "regenie_step2"}.issubset(report["results"])
    if "plink1_cont" in report["results"] or "plink1_bin" in report["results"]:
        assert {"plink1_cont", "plink1_bin"}.issubset(report["results"])

    hail_result_paths = [
        DATA_DIRECTORY / "baselines" / "hail_cont.tsv",
        DATA_DIRECTORY / "baselines" / "hail_bin_wald.tsv",
        DATA_DIRECTORY / "baselines" / "hail_bin_firth.tsv",
    ]
    if any(path.exists() for path in hail_result_paths):
        assert {
            "hail_matrix_table_prepare",
            "hail_suite_cached",
            "hail_cont",
            "hail_bin_wald",
            "hail_bin_firth",
        }.issubset(report["results"])

    continuous_result_paths = sorted((DATA_DIRECTORY / "baselines").glob("plink_cont*.glm.linear"))
    binary_result_paths = sorted((DATA_DIRECTORY / "baselines").glob("plink_bin*.glm.logistic*"))
    plink1_continuous_result_paths = sorted((DATA_DIRECTORY / "baselines").glob("plink1_cont*.assoc.linear"))
    plink1_binary_result_paths = sorted((DATA_DIRECTORY / "baselines").glob("plink1_bin*.assoc.logistic"))
    regenie_result_paths = sorted((DATA_DIRECTORY / "baselines").glob("regenie_step2*.regenie"))
    if not continuous_result_paths or not binary_result_paths or not regenie_result_paths:
        pytest.skip("One or more baseline result files are not available yet.")

    continuous_frame = pd.read_csv(continuous_result_paths[0], sep=r"\s+")
    binary_frame = pd.read_csv(binary_result_paths[0], sep=r"\s+")
    regenie_frame = pd.read_csv(regenie_result_paths[0], sep=r"\s+")

    assert not continuous_frame.empty
    assert not binary_frame.empty
    assert not regenie_frame.empty
    assert {"ID", "P"}.issubset(continuous_frame.columns)
    assert {"ID", "P"}.issubset(binary_frame.columns)
    assert {"ID", "LOG10P"}.issubset(regenie_frame.columns)

    if plink1_continuous_result_paths and plink1_binary_result_paths:
        plink1_continuous_frame = pd.read_csv(plink1_continuous_result_paths[0], sep=r"\s+")
        plink1_binary_frame = pd.read_csv(plink1_binary_result_paths[0], sep=r"\s+")
        assert not plink1_continuous_frame.empty
        assert not plink1_binary_frame.empty
        assert {"P"}.issubset(plink1_continuous_frame.columns)
        assert {"P"}.issubset(plink1_binary_frame.columns)

    hail_existing_result_paths = [path for path in hail_result_paths if path.exists()]
    if hail_existing_result_paths:
        hail_frame = pd.read_csv(hail_existing_result_paths[0], sep="\t")
        assert not hail_frame.empty
        assert {"variant_identifier", "beta", "p_value", "hail_test"}.issubset(hail_frame.columns)
