"""Full chromosome 22 comparisons against upstream REGENIE outputs."""

from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
import socket
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

import pytest

import tests.parity.harness
import tooling.science_gate

if typing.TYPE_CHECKING:
    import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", str(REPOSITORY_ROOT / "data")))
REQUIRE_DATA_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_REQUIRE_DATA"
DEVICE_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_DEVICE"
REPORT_DIRECTORY_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_REPORT_DIRECTORY"
JAX_CACHE_DIRECTORY_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_JAX_CACHE_DIRECTORY"
EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_EXPECTED_GIT_COMMIT"
EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_EXPECTED_SCIENCE_SOURCE_SHA256"
DEFAULT_REPORT_DIRECTORY = REPOSITORY_ROOT / "results" / "parity" / "qualification"
PARITY_METADATA = tests.parity.harness.load_golden_metadata()
QUANTITATIVE_WORKFLOW = PARITY_METADATA.workflow_by_identifier("quantitative_single_bgen_loco")
BINARY_SCORE_ONLY_WORKFLOW = PARITY_METADATA.workflow_by_identifier("binary_score_only")
BINARY_APPROXIMATE_FIRTH_WORKFLOW = PARITY_METADATA.workflow_by_identifier("binary_approximate_firth")

pytestmark = [pytest.mark.phase0_data, pytest.mark.phase1_parity]


class NativeCliRunResultProtocol(typing.Protocol):
    """Typed subset of the native CLI result used by parity."""

    exit_code: int
    stdout_chunks: tuple[str, ...]
    stderr_chunks: tuple[str, ...]


class NativeCliProtocol(typing.Protocol):
    """Typed native CLI submodule used by parity."""

    def run(self, arguments: list[str]) -> NativeCliRunResultProtocol:
        """Execute one native CLI command."""
        ...


class NativeCoreProtocol(typing.Protocol):
    """Build identity and CLI exported by the native extension."""

    __file__: str | None
    __build_git_commit__: str
    __build_science_source_sha256__: str
    __build_source_clean__: bool
    __build_profile__: str
    cli: NativeCliProtocol


class JaxClientProtocol(typing.Protocol):
    """Typed JAX client identity used by qualification evidence."""

    platform_version: str


class JaxDeviceProtocol(typing.Protocol):
    """Typed JAX device identity used by qualification evidence."""

    platform: str
    device_kind: str
    client: JaxClientProtocol


class JaxModuleProtocol(typing.Protocol):
    """Typed subset of JAX imported after native runtime initialization."""

    def devices(self) -> list[JaxDeviceProtocol]:
        """Return devices visible to the initialized JAX backend."""
        ...


@dataclass(frozen=True)
class ExactQualificationSource:
    """Clean source and matching release extension accepted by the gate."""

    git_commit: str
    science_source_sha256: str
    native_library_path: Path
    native_build: tests.parity.harness.QualificationNativeBuild


@dataclass(frozen=True)
class WorkflowQualificationReport:
    """One completed workflow report selected for the sanitized bundle."""

    workflow_identifier: str
    report_path: Path


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
    exact_qualification_source: ExactQualificationSource | None


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


def load_native_core() -> NativeCoreProtocol:
    """Load the extension only at the native execution boundary."""
    native_module = importlib.import_module("g._core")
    return typing.cast("NativeCoreProtocol", native_module)


def assert_exact_qualification_source(native_core: NativeCoreProtocol) -> ExactQualificationSource:
    """Reject dirty, wrong-commit, or stale-extension qualification runs."""
    expected_git_commit = os.environ.get(EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE)
    if expected_git_commit is None:
        raise AssertionError(
            f"Required parity must set {EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE} to the scheduler-selected commit"
        )
    expected_science_source_sha256 = os.environ.get(EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE)
    if expected_science_source_sha256 is None:
        raise AssertionError(
            f"Required parity must set {EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE} before building the extension"
        )
    source_state = tooling.science_gate.assert_clean_exact_source(REPOSITORY_ROOT, expected_git_commit)
    if source_state.science_source_sha256 != expected_science_source_sha256:
        raise AssertionError(
            f"Qualification science-source fingerprint changed: "
            f"expected {expected_science_source_sha256}, observed {source_state.science_source_sha256}"
        )
    if native_core.__build_git_commit__ != source_state.git_commit:
        raise AssertionError(
            f"Loaded native extension is stale or from the wrong commit: "
            f"expected {source_state.git_commit}, observed {native_core.__build_git_commit__}"
        )
    if native_core.__build_science_source_sha256__ != source_state.science_source_sha256:
        raise AssertionError(
            f"Loaded native extension has stale science source: "
            f"expected {source_state.science_source_sha256}, "
            f"observed {native_core.__build_science_source_sha256__}"
        )
    if not native_core.__build_source_clean__:
        raise AssertionError("Loaded native extension was not built from a clean qualification checkout")
    try:
        native_build_profile = tests.parity.harness.NativeBuildProfile(native_core.__build_profile__)
    except ValueError as error:
        raise AssertionError(
            f"Loaded native extension has unknown build profile: {native_core.__build_profile__}"
        ) from error
    if native_build_profile != tests.parity.harness.NativeBuildProfile.RELEASE:
        raise AssertionError(f"Loaded native extension is not a release build: {native_build_profile.value}")
    native_library_value = native_core.__file__
    if native_library_value is None:
        raise AssertionError("The loaded native extension has no filesystem path")
    native_library_path = Path(native_library_value).resolve(strict=True)
    return ExactQualificationSource(
        git_commit=source_state.git_commit,
        science_source_sha256=source_state.science_source_sha256,
        native_library_path=native_library_path,
        native_build=tests.parity.harness.QualificationNativeBuild(
            git_commit=native_core.__build_git_commit__,
            science_source_sha256=native_core.__build_science_source_sha256__,
            source_clean=native_core.__build_source_clean__,
            profile=native_build_profile,
            library_sha256=tests.parity.harness.sha256_file(native_library_path),
            library_size_bytes=native_library_path.stat().st_size,
        ),
    )


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
    native_core = load_native_core()
    exact_qualification_source = assert_exact_qualification_source(native_core) if parity_data_is_required() else None
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
    native_result = native_core.cli.run(["regenie", "--config", str(config_path)])
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
        exact_qualification_source=exact_qualification_source,
    )


def assert_external_result_contract(
    workflow: tests.parity.harness.GoldenWorkflow,
    parity_results: RegenieParityResults,
) -> tuple[tests.parity.harness.StatisticComparison, ...]:
    """Require numerical, validity-mask, identity, and classification parity."""
    observed_schema = tuple(
        (field.name, field.data_type.value) for field in tests.parity.harness.PRODUCTION_OUTPUT_FIELDS
    )
    tests.parity.harness.assert_data_frame_schema(
        parity_results.observed_results,
        expected_schema=observed_schema,
        label="g",
    )
    tests.parity.harness.assert_data_frame_schema(
        parity_results.baseline_results,
        expected_schema=tests.parity.harness.BASELINE_RESULT_SCHEMA,
        label="REGENIE",
    )
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


def qualification_node_name() -> str:
    """Return the scheduler node or host that executed qualification."""
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname()


def nvidia_driver_version() -> str:
    """Return the sole NVIDIA driver version visible on the qualification host."""
    try:
        completed_process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise AssertionError("Required CUDA qualification could not query the NVIDIA driver") from error
    versions = {line.strip() for line in completed_process.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise AssertionError(f"Required CUDA qualification observed ambiguous NVIDIA drivers: {sorted(versions)}")
    return versions.pop()


def observe_qualification_device() -> tests.parity.harness.QualificationDeviceEvidence:
    """Observe and require one homogeneous JAX CUDA device family."""
    jax_module = typing.cast("JaxModuleProtocol", importlib.import_module("jax"))
    devices = tuple(jax_module.devices())
    if not devices:
        raise AssertionError("Required CUDA qualification observed no JAX devices")
    platforms = {device.platform for device in devices}
    device_kinds = {device.device_kind for device in devices}
    backend_platform_versions = {device.client.platform_version for device in devices}
    if len(platforms) != 1 or len(device_kinds) != 1 or len(backend_platform_versions) != 1:
        raise AssertionError("Required CUDA qualification observed heterogeneous JAX devices")
    evidence = tests.parity.harness.QualificationDeviceEvidence(
        platform=platforms.pop(),
        device_kind=device_kinds.pop(),
        device_count=len(devices),
        backend_platform_version=backend_platform_versions.pop(),
        nvidia_driver_version=nvidia_driver_version(),
        cuda_runtime_version=importlib.metadata.version("nvidia-cuda-runtime-cu12"),
    )
    if evidence.platform not in {"cuda", "gpu"} or "cuda" not in evidence.backend_platform_version.lower():
        raise AssertionError(
            f"Required qualification must use JAX CUDA, observed "
            f"{evidence.platform}/{evidence.backend_platform_version}"
        )
    return evidence


def build_qualification_evidence(
    workflow: tests.parity.harness.GoldenWorkflow,
    parity_results: RegenieParityResults,
    comparisons: tuple[tests.parity.harness.StatisticComparison, ...],
    observed_correction_summary: tests.parity.harness.RegenieCorrectionSummary,
    *,
    generated_at: datetime.datetime,
) -> tests.parity.harness.QualificationEvidence | None:
    """Build promotable evidence only for an exact required-source run."""
    exact_source = parity_results.exact_qualification_source
    if exact_source is None:
        return None
    configured_tolerances = {
        tolerance.observed_column: tolerance.absolute_tolerance for tolerance in workflow.tolerances
    }
    return tests.parity.harness.QualificationEvidence(
        passed=True,
        qualified_git_commit=exact_source.git_commit,
        science_source_sha256=exact_source.science_source_sha256,
        working_tree_clean=True,
        qualification_generated_at_utc=generated_at.isoformat(),
        qualification_node=qualification_node_name(),
        cargo_lock_sha256=tests.parity.harness.sha256_file(REPOSITORY_ROOT / "Cargo.lock"),
        uv_lock_sha256=tests.parity.harness.sha256_file(REPOSITORY_ROOT / "uv.lock"),
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
        configured_device=configured_workflow_device(workflow),
        actual_device=observe_qualification_device(),
        native_build=exact_source.native_build,
        observed_row_count=parity_results.observed_results.height,
        output_fields=tests.parity.harness.PRODUCTION_OUTPUT_FIELDS,
        statistics=tuple(
            tests.parity.harness.QualificationStatisticEvidence(
                observed_column=comparison.observed_column,
                baseline_column=comparison.baseline_column,
                maximum_absolute_difference=comparison.maximum_absolute_difference,
                absolute_tolerance=configured_tolerances[comparison.observed_column],
            )
            for comparison in comparisons
        ),
        observed_correction_count=observed_correction_summary.correction_count,
        observed_correction_failure_count=observed_correction_summary.correction_failure_count,
        exact_variant_keys=True,
        exact_sample_counts=True,
        exact_nonfinite_classes=True,
        exact_significance_classifications=True,
    )


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
    native_core = load_native_core()
    native_library_value = native_core.__file__
    if native_library_value is None:
        raise AssertionError("The loaded native extension has no filesystem path")
    native_library_path = Path(native_library_value).resolve(strict=True)
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
    qualification_evidence = (
        build_qualification_evidence(
            workflow,
            parity_results,
            comparisons,
            observed_correction_summary,
            generated_at=generated_at,
        )
        if qualification_passed
        else None
    )
    report_payload: dict[str, object] = {
        "schema_version": 2,
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
        "qualification_evidence": (
            None
            if qualification_evidence is None
            else tests.parity.harness.qualification_evidence_payload(qualification_evidence)
        ),
        "source": {
            "git_commit": git_output("rev-parse", "HEAD").decode("utf-8").strip(),
            "working_tree_dirty": bool(git_status),
            "git_status_sha256": sha256_bytes(git_status),
            "git_diff_sha256": sha256_bytes(git_diff),
            "science_source_sha256": tooling.science_gate.repository_science_source_fingerprint(REPOSITORY_ROOT),
            "native_library_path": str(native_library_path),
            "native_library_sha256": tests.parity.harness.sha256_file(native_library_path),
            "native_build_git_commit": native_core.__build_git_commit__,
            "native_build_science_source_sha256": native_core.__build_science_source_sha256__,
            "native_build_source_clean": native_core.__build_source_clean__,
            "native_build_profile": native_core.__build_profile__,
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
            "schema": [
                {
                    "name": field.name,
                    "data_type": field.data_type.value,
                    "nullable": field.nullable,
                }
                for field in tests.parity.harness.PRODUCTION_OUTPUT_FIELDS
            ],
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


def load_report_qualification_evidence(
    report: WorkflowQualificationReport,
) -> tests.parity.harness.QualificationEvidence:
    """Load and validate promotable evidence from one ignored report."""
    payload = typing.cast(
        "dict[str, object]",
        json.loads(report.report_path.read_text(encoding="utf-8")),
    )
    if payload.get("schema_version") != 2:
        raise AssertionError(f"Unsupported qualification report schema: {report.report_path}")
    workflow_payload = typing.cast("dict[str, object]", payload["workflow"])
    if workflow_payload.get("identifier") != report.workflow_identifier:
        raise AssertionError(f"Qualification report workflow mismatch: {report.report_path}")
    qualification_payload = typing.cast("dict[str, object]", payload["qualification"])
    if qualification_payload.get("passed") is not True or qualification_payload.get("failure") is not None:
        raise AssertionError(f"Qualification report did not pass: {report.report_path}")
    evidence = tests.parity.harness.parse_qualification_evidence(payload["qualification_evidence"])
    if evidence is None:
        raise AssertionError(f"Qualification report has no exact-source evidence: {report.report_path}")
    return evidence


def write_qualification_bundle(reports: tuple[WorkflowQualificationReport, ...]) -> Path:
    """Write one sanitized exact-source bundle covering every required workflow."""
    report_identifiers = {report.workflow_identifier for report in reports}
    if report_identifiers != tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS:
        missing_identifiers = tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS.difference(report_identifiers)
        unexpected_identifiers = report_identifiers.difference(tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS)
        raise AssertionError(
            f"Qualification bundle workflow mismatch: "
            f"missing={sorted(missing_identifiers)}, unexpected={sorted(unexpected_identifiers)}"
        )
    if len(reports) != len(report_identifiers):
        raise AssertionError("Qualification bundle contains duplicate workflows")

    source_state: tooling.science_gate.ScienceSourceState | None = None
    shared_run_identity: tuple[object, ...] | None = None
    workflow_payloads: list[dict[str, object]] = []
    for report in sorted(reports, key=lambda item: item.workflow_identifier):
        workflow = PARITY_METADATA.workflow_by_identifier(report.workflow_identifier)
        evidence = load_report_qualification_evidence(report)
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=evidence.science_source_sha256,
        )
        report_source_state = tooling.science_gate.ScienceSourceState(
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=evidence.science_source_sha256,
        )
        if source_state is None:
            source_state = report_source_state
        elif source_state != report_source_state:
            raise AssertionError("Qualification reports do not describe one exact source")
        report_run_identity = (
            evidence.qualification_node,
            evidence.cargo_lock_sha256,
            evidence.uv_lock_sha256,
            evidence.jax_version,
            evidence.jaxlib_version,
            evidence.configured_device,
            evidence.actual_device,
            evidence.native_build,
        )
        if shared_run_identity is None:
            shared_run_identity = report_run_identity
        elif shared_run_identity != report_run_identity:
            raise AssertionError("Qualification reports do not describe one native/runtime build")
        workflow_payloads.append(
            {
                "identifier": workflow.identifier,
                "regenie_version": workflow.regenie_version,
                "reference_output_sha256": workflow.expected_output_sha256,
                "reference_log_sha256": workflow.expected_log_sha256,
                "input_sha256": workflow.input_sha256,
                "prediction_file_sha256": workflow.prediction_file_sha256,
                "qualification": tests.parity.harness.qualification_evidence_payload(evidence),
            }
        )
    if source_state is None:
        raise AssertionError("Qualification bundle has no workflow evidence")
    current_source_state = tooling.science_gate.assert_clean_exact_source(
        REPOSITORY_ROOT,
        source_state.git_commit,
    )
    if current_source_state.science_source_sha256 != source_state.science_source_sha256:
        raise AssertionError("Qualification bundle science-source fingerprint is stale")

    bundle_payload = {
        "schema_version": 1,
        "qualified_git_commit": source_state.git_commit,
        "science_source_sha256": source_state.science_source_sha256,
        "workflows": workflow_payloads,
    }
    report_directory = qualification_report_directory()
    report_directory.mkdir(parents=True, exist_ok=True)
    bundle_path = report_directory / f"qualification_bundle_{source_state.git_commit}.json"
    bundle_path.write_text(
        f"{json.dumps(bundle_payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return bundle_path


@pytest.fixture(scope="module")
def parity_jax_cache_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the sole explicit JAX cache path used by this Python process."""
    configured_directory = os.environ.get(JAX_CACHE_DIRECTORY_ENVIRONMENT_VARIABLE)
    if configured_directory is None:
        return tmp_path_factory.mktemp("parity-jax-cache")
    cache_directory = Path(configured_directory).resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    return cache_directory


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


@pytest.fixture(scope="module")
def quantitative_qualification_report(
    quantitative_parity_results: RegenieParityResults,
) -> WorkflowQualificationReport:
    """Record exact-source evidence for the quantitative workflow."""
    report_path = assert_and_record_workflow_qualification(
        QUANTITATIVE_WORKFLOW,
        quantitative_parity_results,
    )
    return WorkflowQualificationReport(
        workflow_identifier=QUANTITATIVE_WORKFLOW.identifier,
        report_path=report_path,
    )


@pytest.fixture(scope="module")
def binary_score_only_qualification_report(
    binary_score_only_parity_results: RegenieParityResults,
) -> WorkflowQualificationReport:
    """Record exact-source evidence for the binary score-only workflow."""
    report_path = assert_and_record_workflow_qualification(
        BINARY_SCORE_ONLY_WORKFLOW,
        binary_score_only_parity_results,
    )
    return WorkflowQualificationReport(
        workflow_identifier=BINARY_SCORE_ONLY_WORKFLOW.identifier,
        report_path=report_path,
    )


@pytest.fixture(scope="module")
def binary_approximate_firth_qualification_report(
    binary_approximate_firth_parity_results: RegenieParityResults,
) -> WorkflowQualificationReport:
    """Record exact-source evidence for the binary approximate-Firth workflow."""
    report_path = assert_and_record_workflow_qualification(
        BINARY_APPROXIMATE_FIRTH_WORKFLOW,
        binary_approximate_firth_parity_results,
    )
    return WorkflowQualificationReport(
        workflow_identifier=BINARY_APPROXIMATE_FIRTH_WORKFLOW.identifier,
        report_path=report_path,
    )


@pytest.mark.parity_required
def test_quantitative_full_chr22_matches_upstream_regenie(
    quantitative_qualification_report: WorkflowQualificationReport,
) -> None:
    """Compare every quantitative chr22 row with upstream REGENIE v4.1."""
    assert quantitative_qualification_report.report_path.is_file()


@pytest.mark.parity_required
def test_binary_score_only_full_chr22_matches_upstream_regenie(
    binary_score_only_qualification_report: WorkflowQualificationReport,
) -> None:
    """Compare every binary score-only chr22 row with upstream REGENIE v4.1."""
    assert binary_score_only_qualification_report.report_path.is_file()


@pytest.mark.parity_required
def test_binary_approximate_firth_full_chr22_matches_upstream_regenie(
    binary_approximate_firth_qualification_report: WorkflowQualificationReport,
) -> None:
    """Compare every binary approximate-Firth chr22 row with upstream REGENIE v4.1."""
    assert binary_approximate_firth_qualification_report.report_path.is_file()


@pytest.mark.parity_required
def test_exact_head_qualification_bundle_covers_every_workflow(
    quantitative_qualification_report: WorkflowQualificationReport,
    binary_score_only_qualification_report: WorkflowQualificationReport,
    binary_approximate_firth_qualification_report: WorkflowQualificationReport,
) -> None:
    """Write the sanitized bundle consumed by the trusted status publisher."""
    bundle_path = write_qualification_bundle(
        (
            quantitative_qualification_report,
            binary_score_only_qualification_report,
            binary_approximate_firth_qualification_report,
        )
    )
    assert bundle_path.is_file()
