"""Full chromosome 22 comparisons against upstream REGENIE outputs."""

from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.metadata
import importlib.util
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
EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_EXPECTED_NATIVE_LIBRARY_PATH"
EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_EXPECTED_NATIVE_LIBRARY_SHA256"
EXPECTED_BUNDLE_PATH_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_EXPECTED_BUNDLE_PATH"
RUN_NONCE_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_RUN_NONCE"
RUN_STARTED_AT_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_RUN_STARTED_AT_UTC"
SLURM_JOB_ID_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_SLURM_JOB_ID"
SLURM_STEP_ID_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_SLURM_STEP_ID"
BOOTSTRAP_RELATIVE_PATH_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_BOOTSTRAP_RELATIVE_PATH"
BOOTSTRAP_SHA256_ENVIRONMENT_VARIABLE = "G_REGENIE_PARITY_BOOTSTRAP_SHA256"
QUALIFICATION_TOOL_ENVIRONMENT_PREFIX = "G_REGENIE_PARITY_TOOL"
TRUSTED_SYSTEM_TOOL_PATHS = {
    "bash": "/usr/bin/bash",
    "ar": "/usr/bin/ar",
    "as": "/usr/bin/as",
    "cc": "/usr/bin/cc",
    "cxx": "/usr/bin/c++",
    "env": "/usr/bin/env",
    "git": "/usr/bin/git",
    "ranlib": "/usr/bin/ranlib",
    "scontrol": "/usr/bin/scontrol",
}
DEFAULT_REPORT_DIRECTORY = REPOSITORY_ROOT / "results" / "parity" / "qualification"
PARITY_METADATA = tests.parity.harness.load_golden_metadata()
tests.parity.harness.assert_metadata_covers_required_workflows(PARITY_METADATA)
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
    __build_run_nonce__: str
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
class ExpectedNativeArtifact:
    """Trusted path and digest of the extension built for this run."""

    library_path: Path
    library_sha256: str


@dataclass(frozen=True)
class QualificationBootstrapIdentity:
    """Committed entrypoint selected by the trusted scheduler launcher."""

    relative_path: str
    sha256: str


@dataclass(frozen=True)
class WorkflowQualificationReport:
    """One completed workflow report selected for the sanitized bundle."""

    workflow_identifier: str
    report_path: Path


@dataclass(frozen=True)
class WorkflowArtifactSnapshot:
    """Hashes of every external artifact used by one workflow."""

    input_sha256: dict[str, str]
    prediction_file_sha256: dict[str, str]
    reference_output_sha256: str
    reference_log_sha256: str


@dataclass(frozen=True)
class OutputArtifactSnapshot:
    """Hashes of the exact production bytes parsed for comparison."""

    parquet_file_sha256: dict[str, str]
    parquet_dataset_sha256: str


@dataclass(frozen=True)
class LoadedQualificationReport:
    """Validated promotable report plus its publication digests."""

    evidence: tests.parity.harness.QualificationEvidence
    report_relative_path: Path
    report_sha256: str
    observed_output_sha256: str


@dataclass(frozen=True)
class QualificationRunIdentity:
    """Fields that bind reports to one scheduler execution and native build."""

    qualification_node: str
    slurm_job_id: str
    slurm_step_id: str
    run_nonce: str
    run_started_at_utc: str
    bootstrap_relative_path: str
    bootstrap_sha256: str
    toolchain: tests.parity.harness.QualificationToolchainEvidence
    cargo_lock_sha256: str
    cargo_configuration_sha256: str
    rust_toolchain_sha256: str
    rustflags_environment: str
    cargo_encoded_rustflags_environment: str
    rustc_wrapper_environment: str
    cargo_build_rustc_wrapper_environment: str
    uv_lock_sha256: str
    jax_version: str
    jaxlib_version: str
    configured_device: str
    actual_device: tests.parity.harness.QualificationDeviceEvidence
    native_build: tests.parity.harness.QualificationNativeBuild


@dataclass(frozen=True)
class RegenieParityResults:
    """Materialized production and upstream result tables."""

    observed_results: pl.DataFrame
    baseline_results: pl.DataFrame
    output_root: Path
    output_dataset: tests.parity.harness.CompletedParquetDataset
    config_path: Path
    artifact_snapshot: WorkflowArtifactSnapshot
    output_artifact_snapshot: OutputArtifactSnapshot
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


def require_exact_bundle_mode() -> None:
    """Skip bundle publication during optional local diagnostic parity."""
    if not parity_data_is_required():
        pytest.skip("Exact-source bundle publication applies only to the required parity recipe.")


def load_native_core() -> NativeCoreProtocol:
    """Load the extension only at the native execution boundary."""
    if parity_data_is_required():
        expected_artifact = expected_native_artifact()
        native_specification = importlib.util.find_spec("g._core")
        if native_specification is None or native_specification.origin is None:
            raise AssertionError("Required parity could not locate the native extension")
        discovered_library_path = Path(native_specification.origin).resolve(strict=True)
        if discovered_library_path != expected_artifact.library_path:
            raise AssertionError(
                f"Native extension import path differs from the just-built artifact: "
                f"expected {expected_artifact.library_path}, observed {discovered_library_path}"
            )
        discovered_library_sha256 = tests.parity.harness.sha256_file(discovered_library_path)
        if discovered_library_sha256 != expected_artifact.library_sha256:
            raise AssertionError(
                f"Native extension import bytes differ from the just-built artifact: "
                f"expected {expected_artifact.library_sha256}, observed {discovered_library_sha256}"
            )
    native_module = importlib.import_module("g._core")
    return typing.cast("NativeCoreProtocol", native_module)


def required_environment_value(variable_name: str) -> str:
    """Return one nonempty qualification environment value."""
    value = os.environ.get(variable_name)
    if not value:
        raise AssertionError(f"Required parity must set {variable_name}")
    return value


def expected_native_artifact() -> ExpectedNativeArtifact:
    """Validate the trusted path and digest before importing native code."""
    library_path_value = required_environment_value(EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE)
    library_sha256 = required_environment_value(EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE)
    if tests.parity.harness.SHA256_PATTERN.fullmatch(library_sha256) is None:
        raise AssertionError("Required parity expected native library SHA-256 is malformed")
    library_path = Path(library_path_value)
    if not library_path.is_absolute():
        raise AssertionError("Required parity expected native library path must be absolute")
    library_path = library_path.resolve(strict=True)
    observed_library_sha256 = tests.parity.harness.sha256_file(library_path)
    if observed_library_sha256 != library_sha256:
        raise AssertionError(
            f"Just-built native extension bytes changed before import: "
            f"expected {library_sha256}, observed {observed_library_sha256}"
        )
    return ExpectedNativeArtifact(
        library_path=library_path,
        library_sha256=library_sha256,
    )


def assert_exact_qualification_source(native_core: NativeCoreProtocol) -> ExactQualificationSource:
    """Reject dirty, wrong-commit, or stale-extension qualification runs."""
    expected_git_commit = required_environment_value(EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE)
    expected_science_source_sha256 = required_environment_value(EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE)
    expected_artifact = expected_native_artifact()
    run_nonce = required_environment_value(RUN_NONCE_ENVIRONMENT_VARIABLE)
    if tests.parity.harness.RUN_NONCE_PATTERN.fullmatch(run_nonce) is None:
        raise AssertionError("Required parity run nonce is malformed")
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
    if native_core.__build_run_nonce__ != run_nonce:
        raise AssertionError(
            f"Loaded native extension has the wrong qualification nonce: "
            f"expected {run_nonce}, observed {native_core.__build_run_nonce__}"
        )
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
    if native_library_path != expected_artifact.library_path:
        raise AssertionError(
            f"Loaded native extension path differs from the just-built artifact: "
            f"expected {expected_artifact.library_path}, observed {native_library_path}"
        )
    observed_native_library_sha256 = tests.parity.harness.sha256_file(native_library_path)
    if observed_native_library_sha256 != expected_artifact.library_sha256:
        raise AssertionError(
            f"Loaded native extension bytes differ from the just-built artifact: "
            f"expected {expected_artifact.library_sha256}, observed {observed_native_library_sha256}"
        )
    return ExactQualificationSource(
        git_commit=source_state.git_commit,
        science_source_sha256=source_state.science_source_sha256,
        native_library_path=native_library_path,
        native_build=tests.parity.harness.QualificationNativeBuild(
            git_commit=native_core.__build_git_commit__,
            science_source_sha256=native_core.__build_science_source_sha256__,
            source_clean=native_core.__build_source_clean__,
            profile=native_build_profile,
            run_nonce=native_core.__build_run_nonce__,
            library_sha256=observed_native_library_sha256,
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


def snapshot_workflow_artifacts(
    workflow: tests.parity.harness.GoldenWorkflow,
) -> WorkflowArtifactSnapshot:
    """Hash and validate every input, prediction, and upstream oracle."""
    reference_output_path = DATA_DIRECTORY / workflow.expected_output_relative_path
    reference_log_path = DATA_DIRECTORY / workflow.expected_log_relative_path
    reference_output_sha256 = tests.parity.harness.sha256_file(reference_output_path)
    reference_log_sha256 = tests.parity.harness.sha256_file(reference_log_path)
    if reference_output_sha256 != workflow.expected_output_sha256:
        raise AssertionError(
            f"SHA-256 mismatch for upstream output {reference_output_path}: "
            f"expected {workflow.expected_output_sha256}, observed {reference_output_sha256}"
        )
    if reference_log_sha256 != workflow.expected_log_sha256:
        raise AssertionError(
            f"SHA-256 mismatch for upstream log {reference_log_path}: "
            f"expected {workflow.expected_log_sha256}, observed {reference_log_sha256}"
        )
    return WorkflowArtifactSnapshot(
        input_sha256=assert_workflow_input_hashes(workflow),
        prediction_file_sha256=assert_workflow_prediction_file_hashes(workflow),
        reference_output_sha256=reference_output_sha256,
        reference_log_sha256=reference_log_sha256,
    )


def assert_workflow_artifact_snapshot_unchanged(
    workflow: tests.parity.harness.GoldenWorkflow,
    expected_snapshot: WorkflowArtifactSnapshot,
) -> None:
    """Reject external artifacts changed while a long qualification was running."""
    try:
        observed_snapshot = snapshot_workflow_artifacts(workflow)
    except AssertionError as error:
        raise AssertionError(f"External artifacts changed during qualification for {workflow.identifier}") from error
    if observed_snapshot != expected_snapshot:
        raise AssertionError(f"External artifacts changed during qualification for {workflow.identifier}")


def snapshot_output_artifacts(
    output_dataset: tests.parity.harness.CompletedParquetDataset,
) -> OutputArtifactSnapshot:
    """Hash the complete direct Parquet file set selected for comparison."""
    parquet_paths = tests.parity.harness.direct_parquet_paths(output_dataset.directory)
    parquet_file_sha256 = {
        parquet_path.relative_to(output_dataset.output_root).as_posix(): tests.parity.harness.sha256_file(parquet_path)
        for parquet_path in parquet_paths
    }
    return OutputArtifactSnapshot(
        parquet_file_sha256=parquet_file_sha256,
        parquet_dataset_sha256=tests.parity.harness.sha256_file_set(
            parquet_paths,
            root=output_dataset.output_root,
        ),
    )


def assert_output_artifact_snapshot_unchanged(
    output_dataset: tests.parity.harness.CompletedParquetDataset,
    expected_snapshot: OutputArtifactSnapshot,
) -> None:
    """Reject output file-set or byte changes after the pre-read snapshot."""
    try:
        observed_snapshot = snapshot_output_artifacts(output_dataset)
    except (AssertionError, OSError) as error:
        raise AssertionError("Production output changed during qualification") from error
    if observed_snapshot != expected_snapshot:
        raise AssertionError("Production output changed during qualification")


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
    artifact_snapshot = snapshot_workflow_artifacts(workflow)
    baseline_path = DATA_DIRECTORY / workflow.expected_output_relative_path
    baseline_log_path = DATA_DIRECTORY / workflow.expected_log_relative_path

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

    output_dataset = tests.parity.harness.completed_parquet_dataset(
        output_root,
        native_result.stdout_chunks,
    )
    output_artifact_snapshot = snapshot_output_artifacts(output_dataset)
    observed_results = tests.parity.harness.read_direct_parquet_dataset(output_dataset)
    assert_output_artifact_snapshot_unchanged(output_dataset, output_artifact_snapshot)
    baseline_results = tests.parity.harness.read_regenie_table(baseline_path)
    reference_correction_summary = parse_and_validate_reference_corrections(workflow, baseline_log_path)
    assert_workflow_artifact_snapshot_unchanged(workflow, artifact_snapshot)
    assert_output_artifact_snapshot_unchanged(output_dataset, output_artifact_snapshot)
    return RegenieParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
        output_root=output_root,
        output_dataset=output_dataset,
        config_path=config_path,
        artifact_snapshot=artifact_snapshot,
        output_artifact_snapshot=output_artifact_snapshot,
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


def fsync_directory(directory_path: Path) -> None:
    """Persist directory-entry changes below one existing directory."""
    directory_descriptor = os.open(directory_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def git_output(*arguments: str) -> bytes:
    """Return bytes emitted by one read-only repository query."""
    completed_process = subprocess.run(
        ["git", "--no-replace-objects", *arguments],
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


def validated_slurm_job_id() -> str:
    """Return the numeric Slurm job identity supplied by the trusted recipe."""
    slurm_job_id = required_environment_value(SLURM_JOB_ID_ENVIRONMENT_VARIABLE)
    if tests.parity.harness.SLURM_JOB_ID_PATTERN.fullmatch(slurm_job_id) is None:
        raise AssertionError("Required parity Slurm job ID is malformed")
    return slurm_job_id


def validated_slurm_step_id() -> str:
    """Return the numeric Slurm step identity supplied by the trusted recipe."""
    slurm_step_id = required_environment_value(SLURM_STEP_ID_ENVIRONMENT_VARIABLE)
    if tests.parity.harness.SLURM_JOB_ID_PATTERN.fullmatch(slurm_step_id) is None:
        raise AssertionError("Required parity Slurm step ID is malformed")
    return slurm_step_id


def validated_bootstrap_identity() -> QualificationBootstrapIdentity:
    """Return the exact committed bootstrap path and digest for this run."""
    bootstrap_relative_path = required_environment_value(BOOTSTRAP_RELATIVE_PATH_ENVIRONMENT_VARIABLE)
    if bootstrap_relative_path != tests.parity.harness.QUALIFICATION_BOOTSTRAP_RELATIVE_PATH:
        raise AssertionError(f"Required parity bootstrap path is unknown: {bootstrap_relative_path}")
    bootstrap_sha256 = required_environment_value(BOOTSTRAP_SHA256_ENVIRONMENT_VARIABLE)
    if tests.parity.harness.SHA256_PATTERN.fullmatch(bootstrap_sha256) is None:
        raise AssertionError("Required parity bootstrap SHA-256 is malformed")
    return QualificationBootstrapIdentity(
        relative_path=bootstrap_relative_path,
        sha256=bootstrap_sha256,
    )


def qualification_tool_evidence(tool_name: str) -> tests.parity.harness.QualificationToolEvidence:
    """Return one bootstrap-recorded executable identity."""
    environment_tool_name = tool_name.upper()
    path = required_environment_value(f"{QUALIFICATION_TOOL_ENVIRONMENT_PREFIX}_{environment_tool_name}_PATH")
    sha256 = required_environment_value(f"{QUALIFICATION_TOOL_ENVIRONMENT_PREFIX}_{environment_tool_name}_SHA256")
    version = required_environment_value(f"{QUALIFICATION_TOOL_ENVIRONMENT_PREFIX}_{environment_tool_name}_VERSION")
    if not Path(path).is_absolute():
        raise AssertionError(f"Required parity {tool_name} path must be absolute")
    trusted_system_path = TRUSTED_SYSTEM_TOOL_PATHS.get(tool_name)
    if trusted_system_path is not None and path != trusted_system_path:
        raise AssertionError(f"Required parity {tool_name} path must be {trusted_system_path}")
    expected_generated_path: Path | None = None
    if tool_name == "maturin":
        expected_generated_path = REPOSITORY_ROOT / ".venv" / "bin" / "maturin"
    elif tool_name == "venv_python":
        expected_generated_path = Path(required_environment_value("G_REGENIE_PARITY_VENV_PYTHON_PATH"))
    if expected_generated_path is not None and Path(path) != expected_generated_path:
        raise AssertionError(f"Required parity {tool_name} path must be {expected_generated_path}")
    if tests.parity.harness.SHA256_PATTERN.fullmatch(sha256) is None:
        raise AssertionError(f"Required parity {tool_name} SHA-256 is malformed")
    if tests.parity.harness.sha256_file(Path(path)) != sha256:
        raise AssertionError(f"Required parity {tool_name} executable changed after bootstrap")
    return tests.parity.harness.QualificationToolEvidence(
        path=path,
        sha256=sha256,
        version=version,
    )


def qualification_toolchain_evidence() -> tests.parity.harness.QualificationToolchainEvidence:
    """Return the fixed executable set trusted for this qualification."""
    return tests.parity.harness.QualificationToolchainEvidence(
        bash=qualification_tool_evidence("bash"),
        ar=qualification_tool_evidence("ar"),
        assembler=qualification_tool_evidence("as"),
        cc=qualification_tool_evidence("cc"),
        cc1=qualification_tool_evidence("cc1"),
        cc1plus=qualification_tool_evidence("cc1plus"),
        cargo=qualification_tool_evidence("cargo"),
        collect2=qualification_tool_evidence("collect2"),
        cxx=qualification_tool_evidence("cxx"),
        environment=qualification_tool_evidence("env"),
        git=qualification_tool_evidence("git"),
        just=qualification_tool_evidence("just"),
        maturin=qualification_tool_evidence("maturin"),
        mold=qualification_tool_evidence("mold"),
        python=qualification_tool_evidence("python"),
        ranlib=qualification_tool_evidence("ranlib"),
        rustc=qualification_tool_evidence("rustc"),
        scontrol=qualification_tool_evidence("scontrol"),
        uv=qualification_tool_evidence("uv"),
        venv_python=qualification_tool_evidence("venv_python"),
    )


def validated_run_nonce() -> str:
    """Return the cryptographic lexical nonce supplied by the trusted recipe."""
    run_nonce = required_environment_value(RUN_NONCE_ENVIRONMENT_VARIABLE)
    if tests.parity.harness.RUN_NONCE_PATTERN.fullmatch(run_nonce) is None:
        raise AssertionError("Required parity run nonce is malformed")
    return run_nonce


def validated_run_started_at_utc() -> str:
    """Return the UTC start timestamp supplied before checkout and build."""
    run_started_at_utc = required_environment_value(RUN_STARTED_AT_ENVIRONMENT_VARIABLE)
    try:
        tests.parity.harness.parse_utc_datetime(
            run_started_at_utc,
            label=RUN_STARTED_AT_ENVIRONMENT_VARIABLE,
        )
    except ValueError as error:
        raise AssertionError("Required parity run start timestamp is malformed") from error
    return run_started_at_utc


def nvidia_driver_version() -> str:
    """Return the sole NVIDIA driver version visible on the qualification host."""
    try:
        completed_process = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
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
    bootstrap_identity = validated_bootstrap_identity()
    return tests.parity.harness.QualificationEvidence(
        passed=True,
        qualified_git_commit=exact_source.git_commit,
        science_source_sha256=exact_source.science_source_sha256,
        working_tree_clean=True,
        qualification_generated_at_utc=generated_at.isoformat(),
        qualification_node=qualification_node_name(),
        slurm_job_id=validated_slurm_job_id(),
        slurm_step_id=validated_slurm_step_id(),
        run_nonce=validated_run_nonce(),
        run_started_at_utc=validated_run_started_at_utc(),
        bootstrap_relative_path=bootstrap_identity.relative_path,
        bootstrap_sha256=bootstrap_identity.sha256,
        toolchain=qualification_toolchain_evidence(),
        cargo_lock_sha256=tests.parity.harness.sha256_file(REPOSITORY_ROOT / "Cargo.lock"),
        cargo_configuration_sha256=tests.parity.harness.sha256_file(REPOSITORY_ROOT / ".cargo" / "config.toml"),
        rust_toolchain_sha256=tests.parity.harness.sha256_file(REPOSITORY_ROOT / "rust-toolchain.toml"),
        rustflags_environment=os.environ.get("RUSTFLAGS", ""),
        cargo_encoded_rustflags_environment=os.environ.get("CARGO_ENCODED_RUSTFLAGS", ""),
        rustc_wrapper_environment=os.environ.get("RUSTC_WRAPPER", ""),
        cargo_build_rustc_wrapper_environment=os.environ.get("CARGO_BUILD_RUSTC_WRAPPER", ""),
        uv_lock_sha256=tests.parity.harness.sha256_file(REPOSITORY_ROOT / "uv.lock"),
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
        configured_device=configured_workflow_device(workflow),
        actual_device=observe_qualification_device(),
        native_build=exact_source.native_build,
        observed_row_count=parity_results.observed_results.height,
        observed_output_sha256=parity_results.output_artifact_snapshot.parquet_dataset_sha256,
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
    assert_workflow_artifact_snapshot_unchanged(workflow, parity_results.artifact_snapshot)
    assert_output_artifact_snapshot_unchanged(
        parity_results.output_dataset,
        parity_results.output_artifact_snapshot,
    )
    native_core = load_native_core()
    native_library_value = native_core.__file__
    if native_library_value is None:
        raise AssertionError("The loaded native extension has no filesystem path")
    native_library_path = Path(native_library_value).resolve(strict=True)
    parquet_paths = tests.parity.harness.direct_parquet_paths(parity_results.output_dataset.directory)
    run_directory = parity_results.output_dataset.directory.parent
    required_run_metadata_paths = (run_directory / "run_manifest.json", run_directory / "effective_config.toml")
    missing_run_metadata_paths = [path for path in required_run_metadata_paths if not path.is_file()]
    if missing_run_metadata_paths:
        missing_text = ", ".join(str(path) for path in missing_run_metadata_paths)
        raise AssertionError(f"Completed parity run is missing metadata files: {missing_text}")
    run_metadata_paths = tuple(sorted(path for path in run_directory.iterdir() if path.is_file()))
    input_paths = workflow_input_paths(workflow)
    prediction_file_paths = workflow_prediction_file_paths(workflow)
    observed_output_sha256 = parity_results.output_artifact_snapshot.parquet_dataset_sha256
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
        "schema_version": 3,
        "generated_at_utc": generated_at.isoformat(),
        "run": (
            None
            if qualification_evidence is None
            else {
                "qualification_node": qualification_evidence.qualification_node,
                "slurm_job_id": qualification_evidence.slurm_job_id,
                "slurm_step_id": qualification_evidence.slurm_step_id,
                "run_nonce": qualification_evidence.run_nonce,
                "run_started_at_utc": qualification_evidence.run_started_at_utc,
                "bootstrap_relative_path": qualification_evidence.bootstrap_relative_path,
                "bootstrap_sha256": qualification_evidence.bootstrap_sha256,
                "toolchain": tests.parity.harness.qualification_toolchain_evidence_payload(
                    qualification_evidence.toolchain
                ),
            }
        ),
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
            "native_build_run_nonce": native_core.__build_run_nonce__,
            "cargo_lock_sha256": tests.parity.harness.sha256_file(REPOSITORY_ROOT / "Cargo.lock"),
            "cargo_configuration_sha256": tests.parity.harness.sha256_file(REPOSITORY_ROOT / ".cargo" / "config.toml"),
            "rust_toolchain_sha256": tests.parity.harness.sha256_file(REPOSITORY_ROOT / "rust-toolchain.toml"),
            "rustflags_environment": os.environ.get("RUSTFLAGS", ""),
            "cargo_encoded_rustflags_environment": os.environ.get("CARGO_ENCODED_RUSTFLAGS", ""),
            "rustc_wrapper_environment": os.environ.get("RUSTC_WRAPPER", ""),
            "cargo_build_rustc_wrapper_environment": os.environ.get("CARGO_BUILD_RUSTC_WRAPPER", ""),
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
                "sha256": parity_results.artifact_snapshot.input_sha256[option_name],
            }
            for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES
        },
        "prediction_files": {
            relative_path: {
                "path": str(prediction_file_paths[relative_path]),
                "sha256": parity_results.artifact_snapshot.prediction_file_sha256[relative_path],
            }
            for relative_path in sorted(prediction_file_paths)
        },
        "reference": {
            "output_path": str(DATA_DIRECTORY / workflow.expected_output_relative_path),
            "output_sha256": parity_results.artifact_snapshot.reference_output_sha256,
            "log_path": str(DATA_DIRECTORY / workflow.expected_log_relative_path),
            "log_sha256": parity_results.artifact_snapshot.reference_log_sha256,
            "corrections": correction_summary_payload(parity_results.reference_correction_summary),
        },
        "output": {
            "root": str(parity_results.output_root),
            "dataset_directory": str(parity_results.output_dataset.directory),
            "completion_line": parity_results.output_dataset.completion_line,
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
            "parquet_dataset_sha256": observed_output_sha256,
            "parquet_files": [
                {
                    "relative_path": parquet_path.relative_to(parity_results.output_root).as_posix(),
                    "sha256": parity_results.output_artifact_snapshot.parquet_file_sha256[
                        parquet_path.relative_to(parity_results.output_root).as_posix()
                    ],
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
    assert_workflow_artifact_snapshot_unchanged(workflow, parity_results.artifact_snapshot)
    assert_output_artifact_snapshot_unchanged(
        parity_results.output_dataset,
        parity_results.output_artifact_snapshot,
    )
    workflow_report_directory = qualification_report_directory() / workflow.identifier
    workflow_report_directory.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    report_run_identifier = "diagnostic" if qualification_evidence is None else qualification_evidence.run_nonce
    report_path = workflow_report_directory / f"{timestamp}_{report_run_identifier}_{os.getpid()}.json"
    report_bytes = f"{json.dumps(report_payload, indent=2, sort_keys=True)}\n".encode()
    with report_path.open("xb") as report_file:
        report_file.write(report_bytes)
        report_file.flush()
        os.fsync(report_file.fileno())
    fsync_directory(workflow_report_directory)
    assert_workflow_artifact_snapshot_unchanged(workflow, parity_results.artifact_snapshot)
    assert_output_artifact_snapshot_unchanged(
        parity_results.output_dataset,
        parity_results.output_artifact_snapshot,
    )
    return report_path


def assert_and_record_workflow_qualification(
    workflow: tests.parity.harness.GoldenWorkflow,
    parity_results: RegenieParityResults,
) -> Path:
    """Apply external contracts and persist evidence from a completed run."""
    assert_workflow_artifact_snapshot_unchanged(workflow, parity_results.artifact_snapshot)
    assert_output_artifact_snapshot_unchanged(
        parity_results.output_dataset,
        parity_results.output_artifact_snapshot,
    )
    comparisons: tuple[tests.parity.harness.StatisticComparison, ...] = ()
    try:
        comparisons = assert_external_result_contract(workflow, parity_results)
        observed_correction_summary = assert_correction_contract(
            parity_results.observed_results,
            parity_results.reference_correction_summary,
        )
        assert_workflow_artifact_snapshot_unchanged(workflow, parity_results.artifact_snapshot)
        assert_output_artifact_snapshot_unchanged(
            parity_results.output_dataset,
            parity_results.output_artifact_snapshot,
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


def parse_report_artifact_sha256_mapping(
    value: object,
    *,
    expected_keys: set[str],
    label: str,
) -> dict[str, str]:
    """Parse a strict report mapping from artifact name to path and digest."""
    payload = tests.parity.harness.parse_mapping(value, label=label)
    if set(payload) != expected_keys:
        raise AssertionError(f"{label} artifact set mismatch")
    observed_sha256: dict[str, str] = {}
    for artifact_name, artifact_value in payload.items():
        artifact_label = f"{label}.{artifact_name}"
        artifact_payload = tests.parity.harness.parse_mapping(
            artifact_value,
            label=artifact_label,
        )
        tests.parity.harness.require_mapping_fields(
            artifact_payload,
            label=artifact_label,
            expected_fields={"path", "sha256"},
        )
        tests.parity.harness.parse_nonempty_string(
            artifact_payload["path"],
            label=f"{artifact_label}.path",
        )
        observed_sha256[artifact_name] = tests.parity.harness.parse_sha256(
            artifact_payload["sha256"],
            label=f"{artifact_label}.sha256",
        )
    return observed_sha256


def parse_report_file_sha256_list(
    value: object,
    *,
    root_directory: Path,
    label: str,
) -> dict[str, str]:
    """Parse and rehash a strict list of files below one trusted root."""
    if not isinstance(value, list):
        raise AssertionError(f"{label} must be a list")
    observed_sha256: dict[str, str] = {}
    for file_index, file_value in enumerate(value):
        file_label = f"{label}[{file_index}]"
        file_payload = tests.parity.harness.parse_mapping(file_value, label=file_label)
        tests.parity.harness.require_mapping_fields(
            file_payload,
            label=file_label,
            expected_fields={"relative_path", "sha256"},
        )
        relative_path_text = tests.parity.harness.parse_nonempty_string(
            file_payload["relative_path"],
            label=f"{file_label}.relative_path",
        )
        relative_path = Path(relative_path_text)
        if relative_path.is_absolute() or relative_path.as_posix() != relative_path_text or ".." in relative_path.parts:
            raise AssertionError(f"{file_label}.relative_path is not canonical")
        if relative_path_text in observed_sha256:
            raise AssertionError(f"{label} contains duplicate paths")
        expected_sha256 = tests.parity.harness.parse_sha256(
            file_payload["sha256"],
            label=f"{file_label}.sha256",
        )
        file_path = root_directory / relative_path
        if file_path.is_symlink() or not file_path.is_file():
            raise AssertionError(f"{file_label} is not a direct regular file")
        try:
            file_path.resolve(strict=True).relative_to(root_directory)
        except ValueError as error:
            raise AssertionError(f"{file_label} escapes its root") from error
        if tests.parity.harness.sha256_file(file_path) != expected_sha256:
            raise AssertionError(f"{file_label} digest no longer matches")
        observed_sha256[relative_path_text] = expected_sha256
    return observed_sha256


def load_report_qualification_evidence(
    report: WorkflowQualificationReport,
) -> LoadedQualificationReport:
    """Load and validate promotable evidence from one ignored report."""
    report_directory = qualification_report_directory().resolve(strict=True)
    if report.report_path.is_symlink() or not report.report_path.is_file():
        raise AssertionError(f"Qualification report is not a regular file: {report.report_path}")
    report_path = report.report_path.resolve(strict=True)
    try:
        report_relative_path = report_path.relative_to(report_directory)
    except ValueError as error:
        raise AssertionError(f"Qualification report escapes the report directory: {report_path}") from error
    report_bytes = report_path.read_bytes()
    report_sha256 = sha256_bytes(report_bytes)
    payload = typing.cast("dict[str, object]", json.loads(report_bytes))
    tests.parity.harness.require_mapping_fields(
        payload,
        label="qualification report",
        expected_fields={
            "schema_version",
            "generated_at_utc",
            "run",
            "workflow",
            "qualification",
            "qualification_evidence",
            "source",
            "runtime",
            "configuration",
            "inputs",
            "prediction_files",
            "reference",
            "output",
            "statistics",
        },
    )
    if payload.get("schema_version") != 3:
        raise AssertionError(f"Unsupported qualification report schema: {report_path}")
    workflow_payload = tests.parity.harness.parse_mapping(payload["workflow"], label="qualification report.workflow")
    tests.parity.harness.require_mapping_fields(
        workflow_payload,
        label="qualification report.workflow",
        expected_fields={"identifier", "gate_status", "regenie_version"},
    )
    if workflow_payload.get("identifier") != report.workflow_identifier:
        raise AssertionError(f"Qualification report workflow mismatch: {report_path}")
    qualification_payload = tests.parity.harness.parse_mapping(
        payload["qualification"],
        label="qualification report.qualification",
    )
    tests.parity.harness.require_mapping_fields(
        qualification_payload,
        label="qualification report.qualification",
        expected_fields={"passed", "failure"},
    )
    if qualification_payload.get("passed") is not True or qualification_payload.get("failure") is not None:
        raise AssertionError(f"Qualification report did not pass: {report_path}")
    evidence = tests.parity.harness.parse_qualification_evidence(payload["qualification_evidence"])
    if evidence is None:
        raise AssertionError(f"Qualification report has no exact-source evidence: {report_path}")
    if payload["generated_at_utc"] != evidence.qualification_generated_at_utc:
        raise AssertionError(f"Qualification report generation timestamp mismatch: {report_path}")
    run_payload = tests.parity.harness.parse_mapping(payload["run"], label="qualification report.run")
    tests.parity.harness.require_mapping_fields(
        run_payload,
        label="qualification report.run",
        expected_fields={
            "qualification_node",
            "slurm_job_id",
            "slurm_step_id",
            "run_nonce",
            "run_started_at_utc",
            "bootstrap_relative_path",
            "bootstrap_sha256",
            "toolchain",
        },
    )
    expected_run_payload = {
        "qualification_node": evidence.qualification_node,
        "slurm_job_id": evidence.slurm_job_id,
        "slurm_step_id": evidence.slurm_step_id,
        "run_nonce": evidence.run_nonce,
        "run_started_at_utc": evidence.run_started_at_utc,
        "bootstrap_relative_path": evidence.bootstrap_relative_path,
        "bootstrap_sha256": evidence.bootstrap_sha256,
        "toolchain": tests.parity.harness.qualification_toolchain_evidence_payload(evidence.toolchain),
    }
    if run_payload != expected_run_payload:
        raise AssertionError(f"Qualification report run identity mismatch: {report_path}")
    output_payload = tests.parity.harness.parse_mapping(
        payload["output"],
        label="qualification report.output",
    )
    tests.parity.harness.require_mapping_fields(
        output_payload,
        label="qualification report.output",
        expected_fields={
            "root",
            "dataset_directory",
            "completion_line",
            "row_count",
            "column_order",
            "schema",
            "parquet_dataset_sha256",
            "parquet_files",
            "run_metadata_files",
            "corrections",
        },
    )
    observed_output_sha256 = tests.parity.harness.parse_sha256(
        output_payload["parquet_dataset_sha256"],
        label="qualification report.output.parquet_dataset_sha256",
    )
    if observed_output_sha256 != evidence.observed_output_sha256:
        raise AssertionError(f"Qualification report output digest mismatch: {report_path}")
    workflow = PARITY_METADATA.workflow_by_identifier(report.workflow_identifier)
    if (
        workflow_payload["gate_status"] != workflow.gate_status.value
        or workflow_payload["regenie_version"] != workflow.regenie_version
    ):
        raise AssertionError(f"Qualification report workflow contract mismatch: {report_path}")
    if output_payload["row_count"] != evidence.observed_row_count:
        raise AssertionError(f"Qualification report output row count mismatch: {report_path}")
    expected_column_order = [field.name for field in evidence.output_fields]
    if output_payload["column_order"] != expected_column_order:
        raise AssertionError(f"Qualification report output column order mismatch: {report_path}")
    expected_output_schema = [
        {
            "name": field.name,
            "data_type": field.data_type.value,
            "nullable": field.nullable,
        }
        for field in evidence.output_fields
    ]
    if output_payload["schema"] != expected_output_schema:
        raise AssertionError(f"Qualification report output schema mismatch: {report_path}")
    output_root_text = tests.parity.harness.parse_nonempty_string(
        output_payload["root"],
        label="qualification report.output.root",
    )
    dataset_directory_text = tests.parity.harness.parse_nonempty_string(
        output_payload["dataset_directory"],
        label="qualification report.output.dataset_directory",
    )
    output_root_path = Path(output_root_text)
    dataset_directory_path = Path(dataset_directory_text)
    if not output_root_path.is_absolute() or not dataset_directory_path.is_absolute():
        raise AssertionError(f"Qualification report output paths must be absolute: {report_path}")
    resolved_output_root = output_root_path.resolve(strict=True)
    resolved_dataset_directory = dataset_directory_path.resolve(strict=True)
    if output_root_path != resolved_output_root or dataset_directory_path != resolved_dataset_directory:
        raise AssertionError(f"Qualification report output paths must be canonical: {report_path}")
    try:
        dataset_relative_path = resolved_dataset_directory.relative_to(resolved_output_root)
    except ValueError as error:
        raise AssertionError(f"Qualification report dataset escapes its output root: {report_path}") from error
    if dataset_relative_path == Path() or not resolved_dataset_directory.is_dir():
        raise AssertionError(f"Qualification report dataset path is invalid: {report_path}")
    expected_completion_line = f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}{resolved_dataset_directory}"
    if output_payload["completion_line"] != expected_completion_line:
        raise AssertionError(f"Qualification report completion line mismatch: {report_path}")
    observed_parquet_sha256 = parse_report_file_sha256_list(
        output_payload["parquet_files"],
        root_directory=resolved_output_root,
        label="qualification report.output.parquet_files",
    )
    current_parquet_paths = tests.parity.harness.direct_parquet_paths(resolved_dataset_directory)
    expected_parquet_relative_paths = {
        parquet_path.relative_to(resolved_output_root).as_posix() for parquet_path in current_parquet_paths
    }
    if set(observed_parquet_sha256) != expected_parquet_relative_paths:
        raise AssertionError(f"Qualification report Parquet file set mismatch: {report_path}")
    if (
        tests.parity.harness.sha256_file_set(
            current_parquet_paths,
            root=resolved_output_root,
        )
        != observed_output_sha256
    ):
        raise AssertionError(f"Qualification report Parquet bytes changed: {report_path}")
    observed_run_metadata_sha256 = parse_report_file_sha256_list(
        output_payload["run_metadata_files"],
        root_directory=resolved_output_root,
        label="qualification report.output.run_metadata_files",
    )
    run_metadata_names = {Path(relative_path).name for relative_path in observed_run_metadata_sha256}
    if not {"run_manifest.json", "effective_config.toml"}.issubset(run_metadata_names):
        raise AssertionError(f"Qualification report run metadata set mismatch: {report_path}")
    run_directory = resolved_dataset_directory.parent
    current_run_metadata_paths = tuple(sorted(path for path in run_directory.iterdir() if path.is_file()))
    if any(path.is_symlink() for path in current_run_metadata_paths):
        raise AssertionError(f"Qualification report run metadata contains a symbolic link: {report_path}")
    current_run_metadata_sha256 = {
        metadata_path.relative_to(resolved_output_root).as_posix(): tests.parity.harness.sha256_file(metadata_path)
        for metadata_path in current_run_metadata_paths
    }
    if observed_run_metadata_sha256 != current_run_metadata_sha256:
        raise AssertionError(f"Qualification report run metadata changed: {report_path}")
    expected_output_corrections = correction_summary_payload(
        tests.parity.harness.RegenieCorrectionSummary(
            correction_count=evidence.observed_correction_count,
            correction_failure_count=evidence.observed_correction_failure_count,
        )
    )
    if output_payload["corrections"] != expected_output_corrections:
        raise AssertionError(f"Qualification report output correction summary mismatch: {report_path}")
    observed_input_sha256 = parse_report_artifact_sha256_mapping(
        payload["inputs"],
        expected_keys=set(tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES),
        label="qualification report.inputs",
    )
    current_artifact_snapshot = snapshot_workflow_artifacts(workflow)
    if (
        observed_input_sha256 != workflow.input_sha256
        or observed_input_sha256 != current_artifact_snapshot.input_sha256
    ):
        raise AssertionError(f"Qualification report input digests mismatch: {report_path}")
    input_payload = tests.parity.harness.parse_mapping(
        payload["inputs"],
        label="qualification report.inputs",
    )
    expected_input_paths = workflow_input_paths(workflow)
    for option_name, expected_input_path in expected_input_paths.items():
        input_artifact_payload = tests.parity.harness.parse_mapping(
            input_payload[option_name],
            label=f"qualification report.inputs.{option_name}",
        )
        if input_artifact_payload["path"] != str(expected_input_path):
            raise AssertionError(f"Qualification report input path mismatch: {report_path}")
    observed_prediction_sha256 = parse_report_artifact_sha256_mapping(
        payload["prediction_files"],
        expected_keys=set(workflow.prediction_file_sha256),
        label="qualification report.prediction_files",
    )
    if (
        observed_prediction_sha256 != workflow.prediction_file_sha256
        or observed_prediction_sha256 != current_artifact_snapshot.prediction_file_sha256
    ):
        raise AssertionError(f"Qualification report prediction digests mismatch: {report_path}")
    prediction_payload = tests.parity.harness.parse_mapping(
        payload["prediction_files"],
        label="qualification report.prediction_files",
    )
    expected_prediction_paths = workflow_prediction_file_paths(workflow)
    for relative_path, expected_prediction_path in expected_prediction_paths.items():
        prediction_artifact_payload = tests.parity.harness.parse_mapping(
            prediction_payload[relative_path],
            label=f"qualification report.prediction_files.{relative_path}",
        )
        if prediction_artifact_payload["path"] != str(expected_prediction_path):
            raise AssertionError(f"Qualification report prediction path mismatch: {report_path}")
    reference_payload = tests.parity.harness.parse_mapping(
        payload["reference"],
        label="qualification report.reference",
    )
    tests.parity.harness.require_mapping_fields(
        reference_payload,
        label="qualification report.reference",
        expected_fields={"output_path", "output_sha256", "log_path", "log_sha256", "corrections"},
    )
    expected_reference_output_path = DATA_DIRECTORY / workflow.expected_output_relative_path
    expected_reference_log_path = DATA_DIRECTORY / workflow.expected_log_relative_path
    if (
        reference_payload.get("output_path") != str(expected_reference_output_path)
        or reference_payload.get("output_sha256") != workflow.expected_output_sha256
        or reference_payload.get("output_sha256") != current_artifact_snapshot.reference_output_sha256
    ):
        raise AssertionError(f"Qualification report reference output digest mismatch: {report_path}")
    if (
        reference_payload.get("log_path") != str(expected_reference_log_path)
        or reference_payload.get("log_sha256") != workflow.expected_log_sha256
        or reference_payload.get("log_sha256") != current_artifact_snapshot.reference_log_sha256
    ):
        raise AssertionError(f"Qualification report reference log digest mismatch: {report_path}")
    expected_reference_corrections = correction_summary_payload(
        None
        if workflow.expected_correction_count is None
        else tests.parity.harness.RegenieCorrectionSummary(
            correction_count=workflow.expected_correction_count,
            correction_failure_count=typing.cast("int", workflow.expected_correction_failure_count),
        )
    )
    if reference_payload["corrections"] != expected_reference_corrections:
        raise AssertionError(f"Qualification report reference correction summary mismatch: {report_path}")
    source_payload = tests.parity.harness.parse_mapping(
        payload["source"],
        label="qualification report.source",
    )
    tests.parity.harness.require_mapping_fields(
        source_payload,
        label="qualification report.source",
        expected_fields={
            "git_commit",
            "working_tree_dirty",
            "git_status_sha256",
            "git_diff_sha256",
            "science_source_sha256",
            "native_library_path",
            "native_library_sha256",
            "native_build_git_commit",
            "native_build_science_source_sha256",
            "native_build_source_clean",
            "native_build_profile",
            "native_build_run_nonce",
            "cargo_lock_sha256",
            "cargo_configuration_sha256",
            "rust_toolchain_sha256",
            "rustflags_environment",
            "cargo_encoded_rustflags_environment",
            "rustc_wrapper_environment",
            "cargo_build_rustc_wrapper_environment",
            "uv_lock_sha256",
        },
    )
    expected_source_values = {
        "git_commit": evidence.qualified_git_commit,
        "working_tree_dirty": False,
        "git_status_sha256": sha256_bytes(b""),
        "git_diff_sha256": sha256_bytes(b""),
        "science_source_sha256": evidence.science_source_sha256,
        "native_library_sha256": evidence.native_build.library_sha256,
        "native_build_git_commit": evidence.native_build.git_commit,
        "native_build_science_source_sha256": evidence.native_build.science_source_sha256,
        "native_build_source_clean": evidence.native_build.source_clean,
        "native_build_profile": evidence.native_build.profile.value,
        "native_build_run_nonce": evidence.native_build.run_nonce,
        "cargo_lock_sha256": evidence.cargo_lock_sha256,
        "cargo_configuration_sha256": evidence.cargo_configuration_sha256,
        "rust_toolchain_sha256": evidence.rust_toolchain_sha256,
        "rustflags_environment": evidence.rustflags_environment,
        "cargo_encoded_rustflags_environment": evidence.cargo_encoded_rustflags_environment,
        "rustc_wrapper_environment": evidence.rustc_wrapper_environment,
        "cargo_build_rustc_wrapper_environment": evidence.cargo_build_rustc_wrapper_environment,
        "uv_lock_sha256": evidence.uv_lock_sha256,
    }
    for source_name, expected_value in expected_source_values.items():
        if source_payload[source_name] != expected_value:
            raise AssertionError(f"Qualification report source mismatch for {source_name}: {report_path}")
    native_library_path_text = tests.parity.harness.parse_nonempty_string(
        source_payload["native_library_path"],
        label="qualification report.source.native_library_path",
    )
    native_library_path = Path(native_library_path_text)
    if (
        not native_library_path.is_absolute()
        or native_library_path.is_symlink()
        or not native_library_path.is_file()
        or native_library_path.resolve(strict=True) != native_library_path
        or tests.parity.harness.sha256_file(native_library_path) != evidence.native_build.library_sha256
        or native_library_path.stat().st_size != evidence.native_build.library_size_bytes
    ):
        raise AssertionError(f"Qualification report native library artifact mismatch: {report_path}")
    runtime_payload = tests.parity.harness.parse_mapping(
        payload["runtime"],
        label="qualification report.runtime",
    )
    tests.parity.harness.require_mapping_fields(
        runtime_payload,
        label="qualification report.runtime",
        expected_fields={
            "jax_version",
            "jaxlib_version",
            "configured_device",
            "jax_platforms_environment",
        },
    )
    if (
        runtime_payload["jax_version"] != evidence.jax_version
        or runtime_payload["jaxlib_version"] != evidence.jaxlib_version
        or runtime_payload["configured_device"] != evidence.configured_device
        or runtime_payload["jax_platforms_environment"] != "cuda"
    ):
        raise AssertionError(f"Qualification report runtime mismatch: {report_path}")
    configuration_payload = tests.parity.harness.parse_mapping(
        payload["configuration"],
        label="qualification report.configuration",
    )
    tests.parity.harness.require_mapping_fields(
        configuration_payload,
        label="qualification report.configuration",
        expected_fields={"metadata_options", "toml_path", "toml_sha256"},
    )
    if configuration_payload["metadata_options"] != workflow.g_cli_options:
        raise AssertionError(f"Qualification report metadata options mismatch: {report_path}")
    configuration_path_text = tests.parity.harness.parse_nonempty_string(
        configuration_payload["toml_path"],
        label="qualification report.configuration.toml_path",
    )
    configuration_path = Path(configuration_path_text)
    if not configuration_path.is_absolute() or configuration_path.is_symlink() or not configuration_path.is_file():
        raise AssertionError(f"Qualification report configuration path is invalid: {report_path}")
    expected_configuration_sha256 = tests.parity.harness.parse_sha256(
        configuration_payload["toml_sha256"],
        label="qualification report.configuration.toml_sha256",
    )
    if tests.parity.harness.sha256_file(configuration_path) != expected_configuration_sha256:
        raise AssertionError(f"Qualification report configuration changed: {report_path}")
    expected_statistics = [
        {
            "observed_column": statistic.observed_column,
            "reference_column": statistic.baseline_column,
            "row_count": evidence.observed_row_count,
            "maximum_absolute_difference": statistic.maximum_absolute_difference,
            "absolute_tolerance": statistic.absolute_tolerance,
        }
        for statistic in evidence.statistics
    ]
    if payload["statistics"] != expected_statistics:
        raise AssertionError(f"Qualification report statistics mismatch: {report_path}")
    return LoadedQualificationReport(
        evidence=evidence,
        report_relative_path=report_relative_path,
        report_sha256=report_sha256,
        observed_output_sha256=observed_output_sha256,
    )


def qualification_run_identity(
    evidence: tests.parity.harness.QualificationEvidence,
) -> QualificationRunIdentity:
    """Extract the fields that every report in one run must share."""
    return QualificationRunIdentity(
        qualification_node=evidence.qualification_node,
        slurm_job_id=evidence.slurm_job_id,
        slurm_step_id=evidence.slurm_step_id,
        run_nonce=evidence.run_nonce,
        run_started_at_utc=evidence.run_started_at_utc,
        bootstrap_relative_path=evidence.bootstrap_relative_path,
        bootstrap_sha256=evidence.bootstrap_sha256,
        toolchain=evidence.toolchain,
        cargo_lock_sha256=evidence.cargo_lock_sha256,
        cargo_configuration_sha256=evidence.cargo_configuration_sha256,
        rust_toolchain_sha256=evidence.rust_toolchain_sha256,
        rustflags_environment=evidence.rustflags_environment,
        cargo_encoded_rustflags_environment=evidence.cargo_encoded_rustflags_environment,
        rustc_wrapper_environment=evidence.rustc_wrapper_environment,
        cargo_build_rustc_wrapper_environment=evidence.cargo_build_rustc_wrapper_environment,
        uv_lock_sha256=evidence.uv_lock_sha256,
        jax_version=evidence.jax_version,
        jaxlib_version=evidence.jaxlib_version,
        configured_device=evidence.configured_device,
        actual_device=evidence.actual_device,
        native_build=evidence.native_build,
    )


def qualification_bundle_path(
    source_state: tooling.science_gate.ScienceSourceState,
    run_identity: QualificationRunIdentity,
) -> Path:
    """Return the unique immutable bundle path for one exact scheduler run."""
    return qualification_report_directory() / (
        f"qualification_bundle_{source_state.git_commit}_{run_identity.slurm_job_id}_"
        f"{run_identity.slurm_step_id}_{run_identity.run_nonce}.json"
    )


def publish_unique_json(
    file_path: Path,
    payload: dict[str, object],
    *,
    validate_temporary: typing.Callable[[Path], None] | None = None,
) -> None:
    """Publish one JSON file atomically without replacing an existing path."""
    payload_bytes = f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()
    temporary_path = file_path.parent / f".{file_path.name}.{os.getpid()}.tmp"
    try:
        with temporary_path.open("xb") as temporary_file:
            temporary_file.write(payload_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if validate_temporary is not None:
            validate_temporary(temporary_path)
        os.link(temporary_path, file_path)
        temporary_path.unlink()
        fsync_directory(file_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    shared_run_identity: QualificationRunIdentity | None = None
    workflow_payloads: list[dict[str, object]] = []
    for report in sorted(reports, key=lambda item: item.workflow_identifier):
        workflow = PARITY_METADATA.workflow_by_identifier(report.workflow_identifier)
        loaded_report = load_report_qualification_evidence(report)
        evidence = loaded_report.evidence
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
        report_run_identity = qualification_run_identity(evidence)
        if shared_run_identity is None:
            shared_run_identity = report_run_identity
        elif shared_run_identity != report_run_identity:
            raise AssertionError("Qualification reports do not describe one native/runtime build")
        snapshot_workflow_artifacts(workflow)
        workflow_payloads.append(
            {
                "identifier": workflow.identifier,
                "regenie_version": workflow.regenie_version,
                "reference_output_sha256": workflow.expected_output_sha256,
                "reference_log_sha256": workflow.expected_log_sha256,
                "input_sha256": workflow.input_sha256,
                "prediction_file_sha256": workflow.prediction_file_sha256,
                "qualification_report_relative_path": loaded_report.report_relative_path.as_posix(),
                "qualification_report_sha256": loaded_report.report_sha256,
                "observed_output_sha256": loaded_report.observed_output_sha256,
                "qualification": tests.parity.harness.qualification_evidence_payload(evidence),
            }
        )
    if source_state is None or shared_run_identity is None:
        raise AssertionError("Qualification bundle has no workflow evidence")
    current_source_state = tooling.science_gate.assert_clean_exact_source(
        REPOSITORY_ROOT,
        source_state.git_commit,
    )
    if current_source_state.science_source_sha256 != source_state.science_source_sha256:
        raise AssertionError("Qualification bundle science-source fingerprint is stale")

    generated_at = datetime.datetime.now(datetime.UTC)
    run_started_at = tests.parity.harness.parse_utc_datetime(
        shared_run_identity.run_started_at_utc,
        label="qualification bundle.run.run_started_at_utc",
    )
    if generated_at < run_started_at:
        raise AssertionError("Qualification bundle predates its scheduler run")
    bundle_payload: dict[str, object] = {
        "schema_version": 2,
        "generated_at_utc": generated_at.isoformat(),
        "qualified_git_commit": source_state.git_commit,
        "science_source_sha256": source_state.science_source_sha256,
        "run": {
            "qualification_node": shared_run_identity.qualification_node,
            "slurm_job_id": shared_run_identity.slurm_job_id,
            "slurm_step_id": shared_run_identity.slurm_step_id,
            "run_nonce": shared_run_identity.run_nonce,
            "run_started_at_utc": shared_run_identity.run_started_at_utc,
            "bootstrap_relative_path": shared_run_identity.bootstrap_relative_path,
            "bootstrap_sha256": shared_run_identity.bootstrap_sha256,
            "toolchain": tests.parity.harness.qualification_toolchain_evidence_payload(shared_run_identity.toolchain),
        },
        "workflows": workflow_payloads,
    }
    report_directory = qualification_report_directory()
    report_directory.mkdir(parents=True, exist_ok=True)
    bundle_path = qualification_bundle_path(source_state, shared_run_identity)
    expected_bundle_path = Path(required_environment_value(EXPECTED_BUNDLE_PATH_ENVIRONMENT_VARIABLE))
    if not expected_bundle_path.is_absolute():
        raise AssertionError("Required parity expected bundle path must be absolute")
    if bundle_path.resolve(strict=False) != expected_bundle_path.resolve(strict=False):
        raise AssertionError(
            f"Qualification bundle path differs from the trusted recipe: "
            f"expected {expected_bundle_path}, observed {bundle_path}"
        )

    def validate_bundle_candidate(candidate_path: Path) -> None:
        validate_published_qualification_bundle(
            candidate_path,
            expected_git_commit=source_state.git_commit,
            expected_science_source_sha256=source_state.science_source_sha256,
            expected_slurm_job_id=shared_run_identity.slurm_job_id,
            expected_slurm_step_id=shared_run_identity.slurm_step_id,
            expected_run_nonce=shared_run_identity.run_nonce,
            expected_run_started_at_utc=shared_run_identity.run_started_at_utc,
            expected_bootstrap_sha256=shared_run_identity.bootstrap_sha256,
        )

    publish_unique_json(
        bundle_path,
        bundle_payload,
        validate_temporary=validate_bundle_candidate,
    )
    try:
        validate_bundle_candidate(bundle_path)
    except Exception:
        bundle_path.unlink()
        fsync_directory(bundle_path.parent)
        raise
    return bundle_path


def validate_published_qualification_bundle(
    bundle_path: Path,
    *,
    expected_git_commit: str,
    expected_science_source_sha256: str,
    expected_slurm_job_id: str,
    expected_slurm_step_id: str,
    expected_run_nonce: str,
    expected_run_started_at_utc: str,
    expected_bootstrap_sha256: str,
) -> None:
    """Independently validate one freshly published sanitized bundle."""
    report_directory = qualification_report_directory().resolve(strict=True)
    resolved_bundle_path = bundle_path.resolve(strict=True)
    try:
        resolved_bundle_path.relative_to(report_directory)
    except ValueError as error:
        raise AssertionError(f"Qualification bundle escapes the report directory: {bundle_path}") from error
    payload = typing.cast(
        "dict[str, object]",
        json.loads(resolved_bundle_path.read_text(encoding="utf-8")),
    )
    tests.parity.harness.require_mapping_fields(
        payload,
        label="qualification bundle",
        expected_fields={
            "schema_version",
            "generated_at_utc",
            "qualified_git_commit",
            "science_source_sha256",
            "run",
            "workflows",
        },
    )
    if payload["schema_version"] != 2:
        raise AssertionError("Unsupported qualification bundle schema")
    if payload["qualified_git_commit"] != expected_git_commit:
        raise AssertionError("Qualification bundle Git commit mismatch")
    if payload["science_source_sha256"] != expected_science_source_sha256:
        raise AssertionError("Qualification bundle science-source fingerprint mismatch")
    expected_run_payload = {
        "qualification_node": "landau",
        "slurm_job_id": expected_slurm_job_id,
        "slurm_step_id": expected_slurm_step_id,
        "run_nonce": expected_run_nonce,
        "run_started_at_utc": expected_run_started_at_utc,
        "bootstrap_relative_path": tests.parity.harness.QUALIFICATION_BOOTSTRAP_RELATIVE_PATH,
        "bootstrap_sha256": expected_bootstrap_sha256,
        "toolchain": None,
    }
    run_payload = tests.parity.harness.parse_mapping(payload["run"], label="qualification bundle.run")
    tests.parity.harness.require_mapping_fields(
        run_payload,
        label="qualification bundle.run",
        expected_fields=set(expected_run_payload),
    )
    expected_run_payload["toolchain"] = run_payload.get("toolchain")
    if run_payload != expected_run_payload:
        raise AssertionError("Qualification bundle scheduler run identity mismatch")
    expected_toolchain = tests.parity.harness.parse_qualification_toolchain_evidence(run_payload["toolchain"])
    if parity_data_is_required() and expected_toolchain != qualification_toolchain_evidence():
        raise AssertionError("Qualification bundle host toolchain differs from the trusted environment")
    generated_at = tests.parity.harness.parse_utc_datetime(
        tests.parity.harness.parse_nonempty_string(
            payload["generated_at_utc"],
            label="qualification bundle.generated_at_utc",
        ),
        label="qualification bundle.generated_at_utc",
    )
    run_started_at = tests.parity.harness.parse_utc_datetime(
        expected_run_started_at_utc,
        label="expected run start",
    )
    if generated_at < run_started_at:
        raise AssertionError("Qualification bundle predates the expected scheduler run")
    if generated_at > datetime.datetime.now(datetime.UTC) + tests.parity.harness.QUALIFICATION_CLOCK_SKEW:
        raise AssertionError("Qualification bundle timestamp is implausibly in the future")
    workflow_values = tests.parity.harness.parse_list(
        payload["workflows"],
        label="qualification bundle.workflows",
    )
    observed_identifiers: set[str] = set()
    for workflow_value in workflow_values:
        workflow_payload = tests.parity.harness.parse_mapping(
            workflow_value,
            label="qualification bundle.workflow",
        )
        tests.parity.harness.require_mapping_fields(
            workflow_payload,
            label="qualification bundle.workflow",
            expected_fields={
                "identifier",
                "regenie_version",
                "reference_output_sha256",
                "reference_log_sha256",
                "input_sha256",
                "prediction_file_sha256",
                "qualification_report_relative_path",
                "qualification_report_sha256",
                "observed_output_sha256",
                "qualification",
            },
        )
        workflow_identifier = tests.parity.harness.parse_nonempty_string(
            workflow_payload["identifier"],
            label="qualification bundle.workflow.identifier",
        )
        if workflow_identifier in observed_identifiers:
            raise AssertionError(f"Qualification bundle duplicates workflow {workflow_identifier}")
        observed_identifiers.add(workflow_identifier)
        workflow = PARITY_METADATA.workflow_by_identifier(workflow_identifier)
        if workflow_payload["regenie_version"] != workflow.regenie_version:
            raise AssertionError(f"Qualification bundle REGENIE version mismatch for {workflow_identifier}")
        if workflow_payload["reference_output_sha256"] != workflow.expected_output_sha256:
            raise AssertionError(f"Qualification bundle reference output mismatch for {workflow_identifier}")
        if workflow_payload["reference_log_sha256"] != workflow.expected_log_sha256:
            raise AssertionError(f"Qualification bundle reference log mismatch for {workflow_identifier}")
        if workflow_payload["input_sha256"] != workflow.input_sha256:
            raise AssertionError(f"Qualification bundle input digests mismatch for {workflow_identifier}")
        if workflow_payload["prediction_file_sha256"] != workflow.prediction_file_sha256:
            raise AssertionError(f"Qualification bundle prediction digests mismatch for {workflow_identifier}")
        evidence = tests.parity.harness.parse_qualification_evidence(workflow_payload["qualification"])
        if evidence is None:
            raise AssertionError(f"Qualification bundle lacks evidence for {workflow_identifier}")
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            evidence,
            git_commit=expected_git_commit,
            science_source_sha256=expected_science_source_sha256,
        )
        if (
            evidence.qualification_node != "landau"
            or evidence.slurm_job_id != expected_slurm_job_id
            or evidence.slurm_step_id != expected_slurm_step_id
            or evidence.run_nonce != expected_run_nonce
            or evidence.run_started_at_utc != expected_run_started_at_utc
            or evidence.bootstrap_relative_path != tests.parity.harness.QUALIFICATION_BOOTSTRAP_RELATIVE_PATH
            or evidence.bootstrap_sha256 != expected_bootstrap_sha256
            or evidence.toolchain != expected_toolchain
        ):
            raise AssertionError(f"Qualification bundle has stale run evidence for {workflow_identifier}")
        evidence_generated_at = tests.parity.harness.parse_utc_datetime(
            evidence.qualification_generated_at_utc,
            label=f"qualification bundle.workflow[{workflow_identifier}].qualification_generated_at_utc",
        )
        if evidence_generated_at > generated_at:
            raise AssertionError(f"Qualification evidence postdates its bundle for {workflow_identifier}")
        report_relative_path = Path(
            tests.parity.harness.parse_nonempty_string(
                workflow_payload["qualification_report_relative_path"],
                label="qualification bundle.workflow.qualification_report_relative_path",
            )
        )
        report_path = (report_directory / report_relative_path).resolve(strict=True)
        try:
            report_path.relative_to(report_directory)
        except ValueError as error:
            raise AssertionError(
                f"Qualification report path escapes the report directory: {report_relative_path}"
            ) from error
        expected_report_sha256 = tests.parity.harness.parse_sha256(
            workflow_payload["qualification_report_sha256"],
            label="qualification bundle.workflow.qualification_report_sha256",
        )
        loaded_report = load_report_qualification_evidence(
            WorkflowQualificationReport(
                workflow_identifier=workflow_identifier,
                report_path=report_path,
            )
        )
        if loaded_report.report_sha256 != expected_report_sha256:
            raise AssertionError(f"Qualification report digest mismatch for {workflow_identifier}")
        if loaded_report.evidence != evidence:
            raise AssertionError(f"Qualification report evidence mismatch for {workflow_identifier}")
        expected_output_sha256 = tests.parity.harness.parse_sha256(
            workflow_payload["observed_output_sha256"],
            label="qualification bundle.workflow.observed_output_sha256",
        )
        if (
            expected_output_sha256 != evidence.observed_output_sha256
            or expected_output_sha256 != loaded_report.observed_output_sha256
        ):
            raise AssertionError(f"Qualification output digest mismatch for {workflow_identifier}")
        snapshot_workflow_artifacts(workflow)
    if observed_identifiers != tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS:
        raise AssertionError(f"Qualification bundle workflow mismatch: observed={sorted(observed_identifiers)}")
    source_state = tooling.science_gate.assert_clean_exact_source(
        REPOSITORY_ROOT,
        expected_git_commit,
    )
    if source_state.science_source_sha256 != expected_science_source_sha256:
        raise AssertionError("Qualification bundle source fingerprint is stale")


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
    require_exact_bundle_mode()
    bundle_path = write_qualification_bundle(
        (
            quantitative_qualification_report,
            binary_score_only_qualification_report,
            binary_approximate_firth_qualification_report,
        )
    )
    assert bundle_path.is_file()
