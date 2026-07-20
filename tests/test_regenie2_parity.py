"""Full chromosome 22 comparisons against upstream REGENIE outputs."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

import pytest

import g._core
import tests.parity.harness

if typing.TYPE_CHECKING:
    import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", str(REPOSITORY_ROOT / "data")))
REQUIRE_DATA_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_REQUIRE_DATA"
DEVICE_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_DEVICE"
REPORT_DIRECTORY_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_REPORT_DIRECTORY"
DEFAULT_REPORT_DIRECTORY = REPOSITORY_ROOT / "results" / "parity" / "qualification"
PARITY_METADATA = tests.parity.harness.load_golden_metadata()
QUANTITATIVE_WORKFLOW = PARITY_METADATA.workflow_by_identifier("quantitative_single_bgen_loco")
BINARY_SCORE_ONLY_WORKFLOW = PARITY_METADATA.workflow_by_identifier("binary_score_only")
BINARY_APPROXIMATE_FIRTH_WORKFLOW = PARITY_METADATA.workflow_by_identifier("binary_approximate_firth")

pytestmark = [pytest.mark.phase0_data, pytest.mark.phase1_parity]


@dataclass(frozen=True)
class RegenieParityResults:
    """Materialized production and upstream result tables."""

    observed_results: pl.DataFrame
    baseline_results: pl.DataFrame
    output_root: Path
    config_path: Path
    observed_input_sha256: dict[str, str]
    observed_prediction_file_sha256: dict[str, str]
    reference_correction_summary: tests.parity.harness.RegenieCorrectionSummary | None


def metadata_string(options: dict[str, object], key: str) -> str:
    """Return one required string from workflow metadata."""
    return str(options[key])


def metadata_integer(options: dict[str, object], key: str) -> int:
    """Return one required integer from workflow metadata."""
    return int(typing.cast("int | str", options[key]))


def metadata_string_list(options: dict[str, object], key: str) -> tuple[str, ...]:
    """Return one required string sequence from workflow metadata."""
    return tuple(str(value) for value in typing.cast("list[object]", options[key]))


def toml_string(value: str | Path) -> str:
    """Encode a path or string as a TOML basic string."""
    return json.dumps(str(value))


def parity_data_is_required() -> bool:
    """Return whether a scheduled parity lane forbids missing-data skips."""
    return os.environ.get(REQUIRE_DATA_ENVIRONMENT_VARIABLE, "").lower() in {"1", "true", "yes"}


def configured_workflow_device(workflow: tests.parity.harness.GoldenWorkflow) -> str:
    """Return the environment-adjusted execution device for one workflow."""
    return os.environ.get(
        DEVICE_ENVIRONMENT_VARIABLE,
        metadata_string(workflow.g_cli_options, "device"),
    )


def workflow_input_paths(workflow: tests.parity.harness.GoldenWorkflow) -> dict[str, Path]:
    """Resolve every hashed workflow input from the configured data root."""
    return {
        option_name: DATA_DIRECTORY / metadata_string(workflow.g_cli_options, option_name)
        for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES
    }


def workflow_prediction_file_paths(workflow: tests.parity.harness.GoldenWorkflow) -> dict[str, Path]:
    """Resolve every prediction file referenced by the workflow's list."""
    prediction_list_path = workflow_input_paths(workflow)["prediction_list"]
    return tests.parity.harness.resolve_prediction_files(
        prediction_list_path,
        data_directory=DATA_DIRECTORY,
    )


def required_workflow_paths(workflow: tests.parity.harness.GoldenWorkflow) -> tuple[Path, ...]:
    """Return every local input and oracle artifact required by one workflow."""
    return (
        *workflow_input_paths(workflow).values(),
        DATA_DIRECTORY / workflow.expected_output_relative_path,
        DATA_DIRECTORY / workflow.expected_log_relative_path,
    )


def require_or_skip_workflow_data(workflow: tests.parity.harness.GoldenWorkflow) -> None:
    """Skip an optional local run or fail a scheduled required-data run."""
    missing_paths = [path for path in required_workflow_paths(workflow) if not path.is_file()]
    if not missing_paths:
        try:
            workflow_prediction_file_paths(workflow)
        except AssertionError as error:
            message = f"Invalid external REGENIE prediction fixture for {workflow.identifier}:\n{error}"
            if parity_data_is_required():
                pytest.fail(message)
            pytest.skip(f"{message}\nSet {REQUIRE_DATA_ENVIRONMENT_VARIABLE}=1 in scheduled required-data lanes.")
        return
    missing_text = "\n".join(str(path) for path in missing_paths)
    message = f"Missing external REGENIE parity data for {workflow.identifier}:\n{missing_text}"
    if parity_data_is_required():
        pytest.fail(message)
    pytest.skip(f"{message}\nSet {REQUIRE_DATA_ENVIRONMENT_VARIABLE}=1 in scheduled required-data lanes.")


def assert_workflow_input_hashes(workflow: tests.parity.harness.GoldenWorkflow) -> dict[str, str]:
    """Verify and return hashes for every production workflow input."""
    observed_hashes: dict[str, str] = {}
    for option_name, input_path in workflow_input_paths(workflow).items():
        expected_sha256 = workflow.input_sha256[option_name]
        observed_sha256 = tests.parity.harness.sha256_file(input_path)
        if observed_sha256 != expected_sha256:
            raise AssertionError(
                f"SHA-256 mismatch for {option_name} input {input_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        observed_hashes[option_name] = observed_sha256
    return observed_hashes


def assert_workflow_prediction_file_hashes(workflow: tests.parity.harness.GoldenWorkflow) -> dict[str, str]:
    """Verify every prediction file referenced by the hashed list."""
    prediction_file_paths = workflow_prediction_file_paths(workflow)
    if set(prediction_file_paths) != set(workflow.prediction_file_sha256):
        raise AssertionError(f"Prediction-file manifest changed for {workflow.identifier}")
    observed_hashes: dict[str, str] = {}
    for relative_path, prediction_file_path in prediction_file_paths.items():
        expected_sha256 = workflow.prediction_file_sha256[relative_path]
        observed_sha256 = tests.parity.harness.sha256_file(prediction_file_path)
        if observed_sha256 != expected_sha256:
            raise AssertionError(
                f"SHA-256 mismatch for prediction file {prediction_file_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        observed_hashes[relative_path] = observed_sha256
    return observed_hashes


def parse_and_validate_reference_corrections(
    workflow: tests.parity.harness.GoldenWorkflow,
    log_path: Path,
) -> tests.parity.harness.RegenieCorrectionSummary | None:
    """Parse the pinned log and require metadata to describe its counts."""
    summary = tests.parity.harness.parse_regenie_correction_summary(log_path)
    expected_count = workflow.expected_correction_count
    expected_failure_count = workflow.expected_correction_failure_count
    if (expected_count is None) != (expected_failure_count is None):
        raise AssertionError(f"Incomplete correction-count metadata for {workflow.identifier}")
    if expected_count is None:
        if summary is not None:
            raise AssertionError(f"Unexpected Firth correction summary for {workflow.identifier}")
        return None
    if summary is None:
        raise AssertionError(f"Pinned REGENIE log has no correction summary for {workflow.identifier}")
    if summary.correction_count != expected_count:
        raise AssertionError(
            f"Pinned REGENIE correction count changed for {workflow.identifier}: "
            f"{summary.correction_count} != {expected_count}"
        )
    if summary.correction_failure_count != expected_failure_count:
        raise AssertionError(
            f"Pinned REGENIE correction failure count changed for {workflow.identifier}: "
            f"{summary.correction_failure_count} != {expected_failure_count}"
        )
    return summary


def write_native_cli_config(
    workflow: tests.parity.harness.GoldenWorkflow,
    *,
    output_root: Path,
    jax_cache_directory: Path,
) -> Path:
    """Write one supported native CLI configuration for full parity."""
    options = workflow.g_cli_options
    trait_type = metadata_string(options, "trait_type")
    phenotype_columns = toml_string(metadata_string(options, "phenotype_column"))
    covariate_columns = ", ".join(toml_string(value) for value in metadata_string_list(options, "covariate_columns"))
    configured_device = configured_workflow_device(workflow)
    lines = [
        "[input]",
        f"bgen = {toml_string(DATA_DIRECTORY / metadata_string(options, 'bgen'))}",
        f"sample = {toml_string(DATA_DIRECTORY / metadata_string(options, 'sample'))}",
        f"pheno_file = {toml_string(DATA_DIRECTORY / metadata_string(options, 'phenotype_file'))}",
        f"pheno_columns = [{phenotype_columns}]",
        f"covar_file = {toml_string(DATA_DIRECTORY / metadata_string(options, 'covariate_file'))}",
        f"covar_columns = [{covariate_columns}]",
        f"pred = {toml_string(DATA_DIRECTORY / metadata_string(options, 'prediction_list'))}",
        "",
        "[trait]",
        f"trait_type = {toml_string(trait_type)}",
        f"bsize = {metadata_integer(options, 'chunk_size')}",
        "",
    ]
    if trait_type == "binary":
        lines.extend(
            [
                "[binary]",
                f"fallback_method = {toml_string(metadata_string(options, 'binary_fallback_method'))}",
                "p_threshold = 0.05",
                "firth_se = false",
                "",
            ]
        )
    lines.extend(
        [
            "[compute]",
            f"device = {toml_string(configured_device)}",
            "firth_batch_size = 512",
            "firth_candidate_capacity = 1024",
            f"jax_cache_dir = {toml_string(jax_cache_directory)}",
            "",
            "[output]",
            f"out = {toml_string(output_root)}",
            f"output_run_directory = {toml_string(output_root)}",
            f"writer_threads = {metadata_integer(options, 'writer_thread_count')}",
            "resume = false",
            "",
            "[diagnostics]",
            'telemetry = "off"',
            "",
        ]
    )
    config_path = output_root.with_suffix(".toml")
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def run_workflow(
    workflow: tests.parity.harness.GoldenWorkflow,
    tmp_path_factory: pytest.TempPathFactory,
    *,
    jax_cache_directory: Path,
) -> RegenieParityResults:
    """Run one full production workflow and load its upstream oracle."""
    require_or_skip_workflow_data(workflow)
    observed_input_sha256 = assert_workflow_input_hashes(workflow)
    observed_prediction_file_sha256 = assert_workflow_prediction_file_hashes(workflow)
    baseline_path = DATA_DIRECTORY / workflow.expected_output_relative_path
    baseline_log_path = DATA_DIRECTORY / workflow.expected_log_relative_path
    tests.parity.harness.assert_file_sha256(baseline_path, workflow.expected_output_sha256)
    tests.parity.harness.assert_file_sha256(baseline_log_path, workflow.expected_log_sha256)
    reference_correction_summary = parse_and_validate_reference_corrections(workflow, baseline_log_path)

    temporary_directory = tmp_path_factory.mktemp(workflow.identifier)
    output_root = temporary_directory / "output"
    config_path = write_native_cli_config(
        workflow,
        output_root=output_root,
        jax_cache_directory=jax_cache_directory,
    )
    native_result = g._core.cli.run(["regenie", "--config", str(config_path)])
    if native_result.exit_code != 0:
        native_output = "".join((*native_result.stderr_chunks, *native_result.stdout_chunks))
        raise AssertionError(f"Native CLI parity run failed for {workflow.identifier}:\n{native_output}")

    return RegenieParityResults(
        observed_results=tests.parity.harness.read_direct_parquet_dataset(output_root),
        baseline_results=tests.parity.harness.read_regenie_table(baseline_path),
        output_root=output_root,
        config_path=config_path,
        observed_input_sha256=observed_input_sha256,
        observed_prediction_file_sha256=observed_prediction_file_sha256,
        reference_correction_summary=reference_correction_summary,
    )


def assert_external_result_contract(
    workflow: tests.parity.harness.GoldenWorkflow,
    parity_results: RegenieParityResults,
) -> tuple[tests.parity.harness.StatisticComparison, ...]:
    """Require numerical, validity-mask, identity, and classification parity."""
    tests.parity.harness.assert_variant_key_order_match(
        parity_results.observed_results,
        parity_results.baseline_results,
        expected_row_count=workflow.expected_row_count,
    )
    comparisons: list[tests.parity.harness.StatisticComparison] = []
    for tolerance in workflow.tolerances:
        comparisons.append(
            tests.parity.harness.assert_statistic_columns_match(
                parity_results.observed_results,
                parity_results.baseline_results,
                tolerance=tolerance,
                expected_row_count=workflow.expected_row_count,
            )
        )
    tests.parity.harness.assert_exact_column_match(
        parity_results.observed_results,
        parity_results.baseline_results,
        observed_column="N",
        baseline_column="N",
        expected_row_count=workflow.expected_row_count,
    )
    tests.parity.harness.assert_significance_classifications_match(
        parity_results.observed_results,
        parity_results.baseline_results,
        expected_row_count=workflow.expected_row_count,
    )
    return tuple(comparisons)


def summarize_observed_corrections(observed_results: pl.DataFrame) -> tests.parity.harness.RegenieCorrectionSummary:
    """Count approximate-Firth corrections and failures in production output."""
    firth_mask = observed_results.get_column("CORRECTION_METHOD") == "firth_approximate"
    failed_mask = observed_results.get_column("CORRECTION_STATUS") == "failed"
    return tests.parity.harness.RegenieCorrectionSummary(
        correction_count=int(firth_mask.sum()),
        correction_failure_count=int((firth_mask & failed_mask).sum()),
    )


def assert_correction_contract(
    observed_results: pl.DataFrame,
    reference_summary: tests.parity.harness.RegenieCorrectionSummary | None,
) -> tests.parity.harness.RegenieCorrectionSummary:
    """Require correction labels to match upstream aggregate decisions exactly."""
    observed_pairs = observed_results.select("CORRECTION_METHOD", "CORRECTION_STATUS")
    allowed_pairs = {
        ("score", "success"),
        ("score", "failed"),
        ("firth_approximate", "success"),
        ("firth_approximate", "failed"),
    }
    assert set(observed_pairs.iter_rows()) <= allowed_pairs

    firth_mask = observed_results.get_column("CORRECTION_METHOD") == "firth_approximate"
    failed_mask = observed_results.get_column("CORRECTION_STATUS") == "failed"
    observed_summary = summarize_observed_corrections(observed_results)
    assert not ((~firth_mask) & failed_mask).any()
    reference_correction_count = 0 if reference_summary is None else reference_summary.correction_count
    reference_failure_count = 0 if reference_summary is None else reference_summary.correction_failure_count
    assert observed_summary.correction_count == reference_correction_count
    assert observed_summary.correction_failure_count == reference_failure_count
    return observed_summary


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of an in-memory evidence payload."""
    return hashlib.sha256(payload).hexdigest()


def git_output(*arguments: str) -> bytes:
    """Return bytes emitted by one read-only repository query."""
    completed_process = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return completed_process.stdout


def correction_summary_payload(
    summary: tests.parity.harness.RegenieCorrectionSummary | None,
) -> dict[str, object]:
    """Serialize an optional correction summary."""
    return {
        "summary_present": summary is not None,
        "correction_count": 0 if summary is None else summary.correction_count,
        "correction_failure_count": 0 if summary is None else summary.correction_failure_count,
    }


def qualification_report_directory() -> Path:
    """Resolve the ignored directory for persistent parity evidence."""
    configured_directory = Path(
        os.environ.get(
            REPORT_DIRECTORY_ENVIRONMENT_VARIABLE,
            str(DEFAULT_REPORT_DIRECTORY),
        )
    )
    if configured_directory.is_absolute():
        return configured_directory
    return REPOSITORY_ROOT / configured_directory


def write_qualification_report(
    workflow: tests.parity.harness.GoldenWorkflow,
    parity_results: RegenieParityResults,
    comparisons: tuple[tests.parity.harness.StatisticComparison, ...],
    observed_correction_summary: tests.parity.harness.RegenieCorrectionSummary,
    *,
    qualification_passed: bool,
    qualification_failure: str | None,
) -> Path:
    """Write ignored provenance and numerical evidence for a completed run."""
    native_library_value = g._core.__file__
    if native_library_value is None:
        raise AssertionError("The loaded native extension has no filesystem path")
    native_library_path = Path(native_library_value)
    parquet_paths = tests.parity.harness.direct_parquet_paths(parity_results.output_root)
    run_directory = parquet_paths[0].parent.parent
    required_run_metadata_paths = (run_directory / "run_manifest.json", run_directory / "effective_config.toml")
    missing_run_metadata_paths = [path for path in required_run_metadata_paths if not path.is_file()]
    if missing_run_metadata_paths:
        missing_text = ", ".join(str(path) for path in missing_run_metadata_paths)
        raise AssertionError(f"Completed parity run is missing metadata files: {missing_text}")
    run_metadata_paths = tuple(sorted(path for path in run_directory.iterdir() if path.is_file()))
    input_paths = workflow_input_paths(workflow)
    prediction_file_paths = workflow_prediction_file_paths(workflow)
    configured_tolerances = {
        tolerance.observed_column: tolerance.absolute_tolerance for tolerance in workflow.tolerances
    }
    git_status = git_output("status", "--short")
    git_diff = git_output("diff", "--binary", "HEAD")
    generated_at = datetime.datetime.now(datetime.UTC)
    report_payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "workflow": {
            "identifier": workflow.identifier,
            "gate_status": workflow.gate_status.value,
            "regenie_version": workflow.regenie_version,
        },
        "qualification": {
            "passed": qualification_passed,
            "failure": qualification_failure,
        },
        "source": {
            "git_commit": git_output("rev-parse", "HEAD").decode("utf-8").strip(),
            "working_tree_dirty": bool(git_status),
            "git_status_sha256": sha256_bytes(git_status),
            "git_diff_sha256": sha256_bytes(git_diff),
            "native_library_path": str(native_library_path),
            "native_library_sha256": tests.parity.harness.sha256_file(native_library_path),
            "cargo_lock_sha256": tests.parity.harness.sha256_file(REPOSITORY_ROOT / "Cargo.lock"),
            "uv_lock_sha256": tests.parity.harness.sha256_file(REPOSITORY_ROOT / "uv.lock"),
        },
        "runtime": {
            "jax_version": importlib.metadata.version("jax"),
            "jaxlib_version": importlib.metadata.version("jaxlib"),
            "configured_device": configured_workflow_device(workflow),
            "jax_platforms_environment": os.environ.get("JAX_PLATFORMS"),
        },
        "configuration": {
            "metadata_options": workflow.g_cli_options,
            "toml_path": str(parity_results.config_path),
            "toml_sha256": tests.parity.harness.sha256_file(parity_results.config_path),
        },
        "inputs": {
            option_name: {
                "path": str(input_paths[option_name]),
                "sha256": parity_results.observed_input_sha256[option_name],
            }
            for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES
        },
        "prediction_files": {
            relative_path: {
                "path": str(prediction_file_paths[relative_path]),
                "sha256": parity_results.observed_prediction_file_sha256[relative_path],
            }
            for relative_path in sorted(prediction_file_paths)
        },
        "reference": {
            "output_path": str(DATA_DIRECTORY / workflow.expected_output_relative_path),
            "output_sha256": workflow.expected_output_sha256,
            "log_path": str(DATA_DIRECTORY / workflow.expected_log_relative_path),
            "log_sha256": workflow.expected_log_sha256,
            "corrections": correction_summary_payload(parity_results.reference_correction_summary),
        },
        "output": {
            "root": str(parity_results.output_root),
            "row_count": parity_results.observed_results.height,
            "column_order": parity_results.observed_results.columns,
            "schema": {
                column_name: str(data_type) for column_name, data_type in parity_results.observed_results.schema.items()
            },
            "parquet_dataset_sha256": tests.parity.harness.sha256_file_set(
                parquet_paths,
                root=parity_results.output_root,
            ),
            "parquet_files": [
                {
                    "relative_path": parquet_path.relative_to(parity_results.output_root).as_posix(),
                    "sha256": tests.parity.harness.sha256_file(parquet_path),
                }
                for parquet_path in parquet_paths
            ],
            "run_metadata_files": [
                {
                    "relative_path": metadata_path.relative_to(parity_results.output_root).as_posix(),
                    "sha256": tests.parity.harness.sha256_file(metadata_path),
                }
                for metadata_path in run_metadata_paths
            ],
            "corrections": correction_summary_payload(observed_correction_summary),
        },
        "statistics": [
            {
                "observed_column": comparison.observed_column,
                "reference_column": comparison.baseline_column,
                "row_count": comparison.row_count,
                "maximum_absolute_difference": comparison.maximum_absolute_difference,
                "absolute_tolerance": configured_tolerances[comparison.observed_column],
            }
            for comparison in comparisons
        ],
    }
    workflow_report_directory = qualification_report_directory() / workflow.identifier
    workflow_report_directory.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    report_path = workflow_report_directory / f"{timestamp}_{os.getpid()}.json"
    report_path.write_text(
        f"{json.dumps(report_payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return report_path


def assert_and_record_workflow_qualification(
    workflow: tests.parity.harness.GoldenWorkflow,
    parity_results: RegenieParityResults,
) -> Path:
    """Apply external contracts and persist evidence from a completed run."""
    comparisons: tuple[tests.parity.harness.StatisticComparison, ...] = ()
    try:
        comparisons = assert_external_result_contract(workflow, parity_results)
        observed_correction_summary = assert_correction_contract(
            parity_results.observed_results,
            parity_results.reference_correction_summary,
        )
    except AssertionError as error:
        original_traceback = error.__traceback__
        try:
            if not comparisons:
                available_comparisons: list[tests.parity.harness.StatisticComparison] = []
                for tolerance in workflow.tolerances:
                    try:
                        comparison = tests.parity.harness.measure_statistic_columns(
                            parity_results.observed_results,
                            parity_results.baseline_results,
                            tolerance=tolerance,
                            expected_row_count=workflow.expected_row_count,
                        )
                    except AssertionError:
                        continue
                    available_comparisons.append(comparison)
                comparisons = tuple(available_comparisons)
            observed_correction_summary = summarize_observed_corrections(parity_results.observed_results)
            write_qualification_report(
                workflow,
                parity_results,
                comparisons,
                observed_correction_summary,
                qualification_passed=False,
                qualification_failure=str(error),
            )
        finally:
            raise error.with_traceback(original_traceback) from None
    return write_qualification_report(
        workflow,
        parity_results,
        comparisons,
        observed_correction_summary,
        qualification_passed=True,
        qualification_failure=None,
    )


@pytest.fixture(scope="module")
def parity_jax_cache_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the sole explicit JAX cache path used by this Python process."""
    return tmp_path_factory.mktemp("parity-jax-cache")


@pytest.fixture(scope="module")
def quantitative_parity_results(
    tmp_path_factory: pytest.TempPathFactory,
    parity_jax_cache_directory: Path,
) -> RegenieParityResults:
    """Run the full quantitative external-oracle workflow once."""
    return run_workflow(
        QUANTITATIVE_WORKFLOW,
        tmp_path_factory,
        jax_cache_directory=parity_jax_cache_directory,
    )


@pytest.fixture(scope="module")
def binary_score_only_parity_results(
    tmp_path_factory: pytest.TempPathFactory,
    parity_jax_cache_directory: Path,
) -> RegenieParityResults:
    """Run the full binary score-only external-oracle workflow once."""
    return run_workflow(
        BINARY_SCORE_ONLY_WORKFLOW,
        tmp_path_factory,
        jax_cache_directory=parity_jax_cache_directory,
    )


@pytest.fixture(scope="module")
def binary_approximate_firth_parity_results(
    tmp_path_factory: pytest.TempPathFactory,
    parity_jax_cache_directory: Path,
) -> RegenieParityResults:
    """Run the full binary approximate-Firth external-oracle workflow once."""
    return run_workflow(
        BINARY_APPROXIMATE_FIRTH_WORKFLOW,
        tmp_path_factory,
        jax_cache_directory=parity_jax_cache_directory,
    )


@pytest.mark.parity_blocking
def test_quantitative_full_chr22_matches_upstream_regenie(
    quantitative_parity_results: RegenieParityResults,
) -> None:
    """Compare every quantitative chr22 row with upstream REGENIE v4.1."""
    assert_and_record_workflow_qualification(QUANTITATIVE_WORKFLOW, quantitative_parity_results)


@pytest.mark.parity_diagnostic
def test_binary_score_only_full_chr22_matches_upstream_regenie(
    binary_score_only_parity_results: RegenieParityResults,
) -> None:
    """Compare every binary score-only chr22 row with upstream REGENIE v4.1."""
    assert_and_record_workflow_qualification(BINARY_SCORE_ONLY_WORKFLOW, binary_score_only_parity_results)


@pytest.mark.parity_diagnostic
def test_binary_approximate_firth_full_chr22_matches_upstream_regenie(
    binary_approximate_firth_parity_results: RegenieParityResults,
) -> None:
    """Compare every binary approximate-Firth chr22 row with upstream REGENIE v4.1."""
    assert_and_record_workflow_qualification(
        BINARY_APPROXIMATE_FIRTH_WORKFLOW,
        binary_approximate_firth_parity_results,
    )
