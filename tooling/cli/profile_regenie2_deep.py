#!/usr/bin/env python3
"""Deep landau profiling harness for original REGENIE and g REGENIE step 2."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import typing
from datetime import UTC, datetime
from pathlib import Path

import hydra

import tooling.cli.benchmark_bgen_reader as benchmark_bgen_reader
from tooling.benchmark import benchmark as baseline_benchmark
from tooling.benchmark import comparison as comparison_benchmark
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import g_regenie as tooling_g_regenie
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import logging as tooling_logging
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports
from tooling.profile_deep import budget as profile_deep_budget
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import jax_cache as profile_deep_jax_cache
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    import omegaconf

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
ARTIFACT_MANIFEST_SCHEMA_VERSION = 2
ARTIFACT_MANIFEST_CONTRACT = tooling_reports.VersionedReportContract(
    schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
    required_fields=(
        "generated_at",
        "output_directory",
        "profiler_tools",
        "input_files",
        "regenie_baseline_scope",
        "regenie_baseline_commands",
        "artifact_paths",
        "profiler_runs",
        "skipped_profiles",
    ),
    optional_fields=(),
    schema_field_name="schema_version",
    reject_unknown_fields=True,
)
DEFAULT_OUTPUT_PARENT = profile_deep_config.DEFAULT_OUTPUT_PARENT
DEFAULT_VARIANT_COUNT = 418_943
JAX_XLA_AUTOTUNE_CACHE = "xla_gpu_per_fusion_autotune_cache_dir"
ENABLE_XLA_AUTOTUNE_CACHE = os.environ.get("G_PROFILE_ENABLE_XLA_AUTOTUNE_CACHE") == "1"
GPU_JAX_CACHE_PARENT_DEFAULT = profile_deep_jax_cache.GPU_JAX_CACHE_PARENT_DEFAULT
JAX_DEBUG_LOG_MODULES = "jax._src.compiler,jax._src.lru_cache"
JAX_LOG_SAMPLE_LINE_LIMIT = profile_deep_jax_cache.JAX_LOG_SAMPLE_LINE_LIMIT
BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED = "exact_stage_timings_disabled"
BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING = "stage_timing_file_missing"
BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID = "stage_timing_file_invalid"
BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_MISSING = "binary_chunk_diagnostics_missing"
BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_INVALID = "binary_chunk_diagnostics_invalid"
BINARY_DIAGNOSTIC_COUNT_FIELDS = (
    "score_test_candidate_count",
    "firth_candidate_count",
    "firth_converged_count",
    "firth_failed_count",
    "firth_numerical_failure_count",
    "firth_max_iteration_failure_count",
    "firth_invalid_statistic_failure_count",
    "firth_step_halving_failure_count",
    "pseudo_firth_attempt_count",
    "pseudo_firth_success_count",
    "nr_zero_start_attempt_count",
    "nr_zero_start_success_count",
    "nr_warm_start_attempt_count",
    "nr_warm_start_success_count",
    "sparse_correction_count",
    "dense_correction_count",
)
BINARY_CHUNK_OUTLIER_LIMIT = 5


ProfileStageTimingMode = profile_deep_models.ProfileStageTimingMode
ProfileWorkloadKey = profile_deep_models.ProfileWorkloadKey
ProfileWorkloadSelector = profile_deep_models.ProfileWorkloadSelector
PROFILE_WORKLOAD_KEYS = profile_deep_models.PROFILE_WORKLOAD_KEYS
CampaignBudgetSectionName = profile_deep_models.CampaignBudgetSectionName
CAMPAIGN_BUDGET_SECTION_DISPLAY_NAMES = profile_deep_models.CAMPAIGN_BUDGET_SECTION_DISPLAY_NAMES
ProfileArguments = profile_deep_models.ProfileArguments
ProfilerToolStatus = profile_deep_models.ProfilerToolStatus
LoggingPerturbationCase = profile_deep_models.LoggingPerturbationCase
Step2Candidate = profile_deep_models.Step2Candidate
BgenCandidateSummary = profile_deep_models.BgenCandidateSummary
JaxCacheSnapshot = profile_deep_models.JaxCacheSnapshot
JaxCompileLogSummary = profile_deep_models.JaxCompileLogSummary
JaxCacheDiagnostics = profile_deep_models.JaxCacheDiagnostics
JaxColdWarmDiagnostics = profile_deep_models.JaxColdWarmDiagnostics
CampaignBudgetSection = profile_deep_models.CampaignBudgetSection
CampaignBudget = profile_deep_models.CampaignBudget
TrialResult = profile_deep_models.TrialResult
DeepProfilerRunPaths = profile_deep_models.DeepProfilerRunPaths
DeepProfilerChildCommand = profile_deep_models.DeepProfilerChildCommand
AggregateResult = profile_deep_models.AggregateResult
CandidateTuningResults = profile_deep_models.CandidateTuningResults
BinaryDiagnosticTrialPayload = profile_deep_models.BinaryDiagnosticTrialPayload
ProfilePlan = profile_deep_models.ProfilePlan
RegenieBaselineScopeStatus = profile_deep_models.RegenieBaselineScopeStatus
RegenieBaselineScope = profile_deep_models.RegenieBaselineScope
RuntimeComparisonNotes = profile_deep_models.RuntimeComparisonNotes
parse_int_list = profile_deep_budget.parse_int_list
parse_optional_int_list = profile_deep_budget.parse_optional_int_list
parse_string_list = profile_deep_budget.parse_string_list
parse_regenie_baseline_trait_types = profile_deep_budget.parse_regenie_baseline_trait_types
parse_profile_workload_keys = profile_deep_budget.parse_profile_workload_keys
selected_regenie_baseline_trait_types = profile_deep_budget.selected_regenie_baseline_trait_types
build_queue_depth_values = profile_deep_budget.build_queue_depth_values
build_logging_perturbation_cases = profile_deep_budget.build_logging_perturbation_cases
build_campaign_budget_section = profile_deep_budget.build_campaign_budget_section
count_queue_depth_grid = profile_deep_budget.count_queue_depth_grid
count_step2_tuning_candidates = profile_deep_budget.count_step2_tuning_candidates
count_enabled_deep_profiler_modes = profile_deep_budget.count_enabled_deep_profiler_modes
campaign_budget_is_over_limit = profile_deep_budget.campaign_budget_is_over_limit
build_campaign_budget = profile_deep_budget.build_campaign_budget
log_campaign_budget = profile_deep_budget.log_campaign_budget
enforce_campaign_budget = profile_deep_budget.enforce_campaign_budget
resolve_repo_path = profile_deep_config.resolve_repo_path
resolve_data_path = profile_deep_config.resolve_data_path
should_emit_stage_timings = profile_deep_config.should_emit_stage_timings
build_output_directory = profile_deep_config.build_output_directory
configured_regenie_executable = profile_deep_config.configured_regenie_executable
profile_configuration_payload = profile_deep_config.profile_configuration_payload
apply_smoke_overrides = profile_deep_config.apply_smoke_overrides
build_arguments_from_config = profile_deep_config.build_arguments_from_config
build_arguments_from_overrides = profile_deep_config.build_arguments_from_overrides
resolve_profile_jax_cache_directory = profile_deep_jax_cache.resolve_profile_jax_cache_directory
collect_jax_cache_snapshot = profile_deep_jax_cache.collect_jax_cache_snapshot
parse_jax_compile_log = profile_deep_jax_cache.parse_jax_compile_log
read_jax_compile_log_summary = profile_deep_jax_cache.read_jax_compile_log_summary
snapshot_delta = profile_deep_jax_cache.snapshot_delta
build_jax_cache_diagnostics = profile_deep_jax_cache.build_jax_cache_diagnostics
successful_trials_with_jax_diagnostics = profile_deep_jax_cache.successful_trials_with_jax_diagnostics
sum_optional_integer_values = profile_deep_jax_cache.sum_optional_integer_values
compile_log_summary_for_trial = profile_deep_jax_cache.compile_log_summary_for_trial
build_jax_cold_warm_diagnostics = profile_deep_jax_cache.build_jax_cold_warm_diagnostics
collect_jax_cache_diagnostics = profile_deep_jax_cache.collect_jax_cache_diagnostics


def build_baseline_paths(arguments: ProfileArguments) -> baseline_benchmark.BaselinePaths:
    """Build baseline paths from Hydra-resolved profile arguments."""
    return baseline_benchmark.BaselinePaths(
        data_directory=arguments.data_directory,
        baseline_directory=arguments.baseline_directory,
        bed_prefix=arguments.bed_prefix,
        bgen_path=arguments.bgen_path,
        sample_path=arguments.sample_path,
        continuous_phenotype_path=arguments.continuous_phenotype_path,
        binary_phenotype_path=arguments.binary_phenotype_path,
        covariate_path=arguments.covariate_path,
        hail_directory=arguments.data_directory / "hail",
        hail_matrix_table_path=arguments.data_directory / "hail" / f"{arguments.bed_prefix.name}.mt",
        hail_suite_report_path=arguments.baseline_directory / "hail_suite_report.json",
        regenie_prediction_list_path=arguments.regenie_prediction_list_path,
        regenie_qt_prediction_list_path=arguments.regenie_qt_prediction_list_path,
    )


def executable_is_available(executable_name: str) -> bool:
    """Return whether a command or explicit executable path is available."""
    executable_path = Path(executable_name)
    if executable_path.is_absolute() or executable_path.parent != Path():
        return executable_path.exists() and os.access(executable_path, os.X_OK)
    return shutil.which(executable_name) is not None


def resolve_available_regenie_executable(arguments: ProfileArguments) -> str | None:
    """Resolve REGENIE for optional baseline runs without failing the campaign."""
    executable_name = configured_regenie_executable(arguments)
    if executable_is_available(executable_name):
        return executable_name
    return None


def resolved_binary_path(executable_name: str | None) -> str | None:
    """Resolve a command name to an absolute executable path when possible."""
    if executable_name is None:
        return None
    executable_path = Path(executable_name)
    if executable_path.is_absolute() or executable_path.parent != Path():
        if executable_path.exists():
            return str(executable_path.resolve())
        return executable_name
    resolved_path = shutil.which(executable_name)
    if resolved_path is not None:
        return resolved_path
    return executable_name


def python_module_is_available(module_name: str) -> bool:
    """Return whether a module is importable in the active Python environment."""
    return importlib.util.find_spec(module_name) is not None


def build_uv_injected_profiler_status(
    *,
    tool_name: str,
    executable_name: str,
    module_name: str,
    enabled: bool,
) -> ProfilerToolStatus:
    """Build availability for Python profilers that must see project dependencies."""
    if python_module_is_available(module_name):
        return ProfilerToolStatus(
            tool_name=tool_name,
            enabled=enabled,
            available=True,
            executable_path=sys.executable,
            notes=f"{module_name} is importable in the project Python environment.",
        )
    uv_executable_path = shutil.which("uv")
    if uv_executable_path is not None:
        return ProfilerToolStatus(
            tool_name=tool_name,
            enabled=enabled,
            available=True,
            executable_path=uv_executable_path,
            notes=(
                f"{executable_name} will run through uv --no-sync --with {module_name} "
                "to preserve the project Python environment."
            ),
        )
    return ProfilerToolStatus(
        tool_name=tool_name,
        enabled=enabled,
        available=False,
        executable_path=None,
        notes=f"{module_name} is not importable in the project Python environment and uv is not on PATH.",
    )


def build_profiler_tool_status(arguments: ProfileArguments) -> dict[str, ProfilerToolStatus]:
    """Build profiler tool availability records for the current host."""
    optional_executable_tools = {
        "py_spy": ("py-spy", arguments.enable_py_spy),
        "linux_perf": ("perf", arguments.enable_linux_perf),
        "nsight_systems": ("nsys", arguments.enable_nsight_systems),
        "nsight_compute": ("ncu", arguments.enable_nsight_compute),
    }
    tool_status = {
        "python_cprofile": ProfilerToolStatus(
            tool_name="python_cprofile",
            enabled=arguments.enable_python_cprofile,
            available=True,
            executable_path=sys.executable,
            notes="Python cProfile is part of the standard library.",
        ),
        "jax_trace": ProfilerToolStatus(
            tool_name="jax_trace",
            enabled=arguments.enable_jax_trace,
            available=True,
            executable_path=None,
            notes="JAX profiler trace capture is provided by the installed JAX package.",
        ),
        "jax_memory_profile": ProfilerToolStatus(
            tool_name="jax_memory_profile",
            enabled=arguments.enable_jax_memory_profile,
            available=True,
            executable_path=None,
            notes="JAX device memory capture is provided by the installed JAX package.",
        ),
        "rust_criterion": ProfilerToolStatus(
            tool_name="rust_criterion",
            enabled=arguments.enable_rust_criterion,
            available=shutil.which("cargo") is not None,
            executable_path=shutil.which("cargo"),
            notes="Rust Criterion benches run through cargo.",
        ),
        "scalene": build_uv_injected_profiler_status(
            tool_name="scalene",
            executable_name="scalene",
            module_name="scalene",
            enabled=arguments.enable_scalene,
        ),
        "memray": build_uv_injected_profiler_status(
            tool_name="memray",
            executable_name="memray",
            module_name="memray",
            enabled=arguments.enable_memray,
        ),
    }
    for tool_name, (executable_name, enabled) in optional_executable_tools.items():
        executable_path = shutil.which(executable_name)
        available = executable_path is not None
        notes = f"{executable_name} is available on PATH." if available else f"{executable_name} is not on PATH."
        tool_status[tool_name] = ProfilerToolStatus(
            tool_name=tool_name,
            enabled=enabled,
            available=available,
            executable_path=executable_path,
            notes=notes,
        )
    return tool_status


def serialize_profiler_tool_status(tool_status: dict[str, ProfilerToolStatus]) -> dict[str, dict[str, object]]:
    """Serialize profiler tool availability records."""
    return {tool_name: dataclasses.asdict(status) for tool_name, status in sorted(tool_status.items())}


def manifest_path_value(*, output_directory: Path, path_value: typing.Any) -> str | None:
    """Return a manifest path relative to the campaign directory when possible."""
    if path_value is None:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        try:
            return str(path.relative_to(output_directory))
        except ValueError:
            return str(path)
    return str(path)


def collect_profiler_run_manifest_entries(
    *,
    output_directory: Path,
    sampling_profiles: list[dict[str, typing.Any]],
) -> list[dict[str, str | None]]:
    """Collect per-profiler artifact and application output paths for the manifest."""
    profiler_runs: list[dict[str, str | None]] = []
    for profile in sampling_profiles:
        profiler_artifact_path = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("profiler_artifact_path"),
        )
        application_output_prefix = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("application_output_prefix"),
        )
        application_output_run_directory = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("application_output_run_directory"),
        )
        stage_timing_path = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("stage_timing_path"),
        )
        if (
            profiler_artifact_path is None
            and application_output_prefix is None
            and application_output_run_directory is None
            and stage_timing_path is None
        ):
            continue
        profiler_runs.append(
            {
                "name": str(profile.get("name", "")),
                "implementation": str(profile.get("implementation", "")),
                "status": str(profile.get("status", "")),
                "profiler_artifact_path": profiler_artifact_path,
                "application_output_prefix": application_output_prefix,
                "application_output_run_directory": application_output_run_directory,
                "stage_timing_path": stage_timing_path,
            }
        )
    return profiler_runs


def serialize_regenie_baseline_scope(scope: RegenieBaselineScope) -> dict[str, object]:
    """Serialize baseline scope without embedding long extract lists."""
    return {
        "status": scope.status.value,
        "variant_limit": scope.variant_limit,
        "extract_path": str(scope.extract_path) if scope.extract_path is not None else None,
        "metadata_path": str(scope.metadata_path) if scope.metadata_path is not None else None,
        "selected_variant_count": scope.selected_variant_count,
        "notes": scope.notes,
    }


def command_input_paths(command_arguments: list[str]) -> list[str]:
    """Extract input file paths from a REGENIE command line."""
    input_flags = {"--bgen", "--sample", "--phenoFile", "--covarFile", "--pred", "--extract"}
    input_paths: list[str] = []
    argument_index = 0
    while argument_index < len(command_arguments):
        argument = command_arguments[argument_index]
        if argument in input_flags and argument_index + 1 < len(command_arguments):
            input_paths.append(command_arguments[argument_index + 1])
            argument_index += 2
            continue
        if argument == "--bed" and argument_index + 1 < len(command_arguments):
            bed_prefix = Path(command_arguments[argument_index + 1])
            input_paths.extend(str(bed_prefix.with_suffix(suffix)) for suffix in (".bed", ".bim", ".fam"))
            argument_index += 2
            continue
        argument_index += 1
    return sorted(set(input_paths))


def build_command_manifest(command_name: str, status: str, command_arguments: list[str]) -> dict[str, object]:
    """Build manifest metadata for one baseline command."""
    executable_name = command_arguments[0] if command_arguments else None
    return {
        "name": command_name,
        "status": status,
        "binary": resolved_binary_path(executable_name),
        "command_arguments": command_arguments,
        "input_files": command_input_paths(command_arguments),
    }


def collect_summary_baseline_commands(summary_payload: dict[str, typing.Any] | None) -> list[dict[str, object]]:
    """Collect actual baseline commands from a run summary payload."""
    if summary_payload is None:
        return []
    command_manifests: list[dict[str, object]] = []
    result_groups = [
        typing.cast("list[dict[str, typing.Any]]", summary_payload.get("setup_results", [])),
        typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])),
    ]
    for result_group in result_groups:
        for result_payload in result_group:
            trial_payloads = typing.cast("list[dict[str, typing.Any]]", result_payload.get("trials", []))
            if not trial_payloads:
                trial_payloads = [result_payload]
            for trial_payload in trial_payloads:
                if trial_payload.get("implementation") != "regenie":
                    continue
                command_manifests.append(
                    build_command_manifest(
                        command_name=str(trial_payload.get("name", "")),
                        status=str(trial_payload.get("status", "")),
                        command_arguments=[str(value) for value in trial_payload.get("command_arguments", [])],
                    )
                )
    return command_manifests


def collect_manifest_input_files(
    *,
    profile_plan: ProfilePlan | None,
    summary_payload: dict[str, typing.Any] | None,
) -> list[dict[str, object]]:
    """Collect profile input files for the artifact manifest."""
    if summary_payload is not None:
        preflight_payload = typing.cast("dict[str, typing.Any]", summary_payload.get("preflight", {}))
        file_sizes = typing.cast("dict[str, int]", preflight_payload.get("input_file_sizes", {}))
        if file_sizes:
            return [
                {"path": path, "size_bytes": size_bytes}
                for path, size_bytes in sorted(file_sizes.items(), key=lambda item: item[0])
            ]
    if profile_plan is None:
        return []
    return [{"path": input_path, "size_bytes": None} for input_path in profile_plan.required_inputs]


def collect_artifact_manifest(
    *,
    output_directory: Path,
    profiler_tool_status: dict[str, ProfilerToolStatus],
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: ProfilePlan | None = None,
) -> dict[str, typing.Any]:
    """Build a structured artifact manifest for one profile campaign."""
    artifact_paths = sorted(
        str(path.relative_to(output_directory))
        for path in output_directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    skipped_profiles: list[dict[str, typing.Any]] = []
    if summary_payload is not None:
        deep_profile_results = typing.cast("dict[str, typing.Any]", summary_payload.get("deep_profiles", {}))
        sampling_profiles = typing.cast(
            "list[dict[str, typing.Any]]", deep_profile_results.get("sampling_profiles", [])
        )
        skipped_profiles = [profile for profile in sampling_profiles if profile.get("status") == "skipped"]
    else:
        sampling_profiles = []
    baseline_commands = collect_summary_baseline_commands(summary_payload)
    regenie_baseline_scope = None
    if summary_payload is not None:
        regenie_baseline_scope = summary_payload.get("regenie_baseline_scope")
    if profile_plan is not None:
        baseline_commands = profile_plan.regenie_baseline_commands
        regenie_baseline_scope = profile_plan.regenie_baseline_scope
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_directory": str(output_directory),
        "profiler_tools": serialize_profiler_tool_status(profiler_tool_status),
        "input_files": collect_manifest_input_files(profile_plan=profile_plan, summary_payload=summary_payload),
        "regenie_baseline_scope": regenie_baseline_scope,
        "regenie_baseline_commands": baseline_commands,
        "artifact_paths": artifact_paths,
        "profiler_runs": collect_profiler_run_manifest_entries(
            output_directory=output_directory,
            sampling_profiles=sampling_profiles,
        ),
        "skipped_profiles": skipped_profiles,
    }


def write_artifact_manifest(
    *,
    output_directory: Path,
    profiler_tool_status: dict[str, ProfilerToolStatus],
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: ProfilePlan | None = None,
) -> Path:
    """Write the legacy profile artifact manifest."""
    manifest = collect_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
        profile_plan=profile_plan,
    )
    manifest_path = output_directory / "legacy_artifact_manifest.v2.json"
    tooling_reports.write_versioned_json_report(
        manifest_path,
        manifest,
        ARTIFACT_MANIFEST_CONTRACT,
        sort_keys=True,
    )
    return manifest_path


def profile_status_to_artifact_status(status: str) -> tooling_artifact_format.ToolArtifactStatus:
    """Convert a profile-local status string to the shared artifact status enum."""
    if status == "success":
        return tooling_artifact_format.ToolArtifactStatus.SUCCESS
    if status == "partial":
        return tooling_artifact_format.ToolArtifactStatus.PARTIAL
    if status == "failed":
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    if status == "skipped":
        return tooling_artifact_format.ToolArtifactStatus.SKIPPED
    if status == "unsupported":
        return tooling_artifact_format.ToolArtifactStatus.UNSUPPORTED
    if status == "dry_run" or status == "planned":
        return tooling_artifact_format.ToolArtifactStatus.DRY_RUN
    return tooling_artifact_format.ToolArtifactStatus.INVALID


def profile_command_identifier(raw_name: str, index: int) -> str:
    """Build a filesystem-safe command identifier."""
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in raw_name)
    if normalized:
        return normalized
    return f"profile_command_{index:04d}"


def collect_summary_trial_payloads(summary_payload: dict[str, typing.Any] | None) -> list[dict[str, typing.Any]]:
    """Collect trial-like payloads from a deep-profile summary."""
    if summary_payload is None:
        return []
    trial_payloads: list[dict[str, typing.Any]] = []
    for payload in typing.cast("list[dict[str, typing.Any]]", summary_payload.get("setup_results", [])):
        trial_payloads.append(payload)
    for aggregate_payload in typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])):
        for field_name in ("warmup_trials", "trials"):
            trial_payloads.extend(typing.cast("list[dict[str, typing.Any]]", aggregate_payload.get(field_name, [])))
    deep_profile_results = typing.cast("dict[str, typing.Any]", summary_payload.get("deep_profiles", {}))
    for profile_payload in typing.cast(
        "list[dict[str, typing.Any]]", deep_profile_results.get("sampling_profiles", [])
    ):
        if isinstance(profile_payload.get("command_arguments"), list):
            trial_payloads.append(profile_payload)
    for perturbation_payload in typing.cast(
        "list[dict[str, typing.Any]]",
        summary_payload.get("logging_perturbation_results", []),
    ):
        trial_payload = perturbation_payload.get("trial")
        if isinstance(trial_payload, dict):
            trial_payloads.append(typing.cast("dict[str, typing.Any]", trial_payload))
    return trial_payloads


def build_profile_command_records(
    *,
    output_directory: Path,
    run_id: str,
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: ProfilePlan | None = None,
) -> list[tooling_artifact_format.CommandRecord]:
    """Build command ledger records for deep-profile subprocesses."""
    command_records: list[tooling_artifact_format.CommandRecord] = []
    for command_index, trial_payload in enumerate(collect_summary_trial_payloads(summary_payload), start=1):
        command_arguments = [str(value) for value in trial_payload.get("command_arguments", [])]
        if not command_arguments:
            continue
        raw_name = str(trial_payload.get("name", f"profile_command_{command_index:04d}"))
        status = profile_status_to_artifact_status(str(trial_payload.get("status", "invalid")))
        command_records.append(
            tooling_artifact_format.build_command_record(
                command_id=profile_command_identifier(raw_name, command_index),
                tool_name="profile_regenie2_deep",
                run_id=run_id,
                phase=str(trial_payload.get("trait_type", "profile")),
                args=command_arguments,
                output_directory=output_directory,
                cwd=REPOSITORY_ROOT,
                environment_overrides=typing.cast(
                    "dict[str, str]",
                    trial_payload.get("environment_overrides", {}),
                ),
                stdout_log=Path(str(trial_payload["stdout_log_path"]))
                if trial_payload.get("stdout_log_path") is not None
                else None,
                stderr_log=Path(str(trial_payload["stderr_log_path"]))
                if trial_payload.get("stderr_log_path") is not None
                else None,
                status=status,
                return_code=None,
                wall_time_seconds=(
                    float(trial_payload["wall_time_seconds"])
                    if isinstance(trial_payload.get("wall_time_seconds"), (int, float))
                    else None
                ),
            )
        )
    if profile_plan is not None:
        next_index = len(command_records) + 1
        for command_manifest in profile_plan.regenie_baseline_commands:
            raw_name = str(command_manifest["name"])
            command_arguments = typing.cast("list[object]", command_manifest["command_arguments"])
            command_records.append(
                tooling_artifact_format.build_command_record(
                    command_id=profile_command_identifier(raw_name, next_index),
                    tool_name="profile_regenie2_deep",
                    run_id=run_id,
                    phase="regenie_baseline",
                    args=[str(value) for value in command_arguments],
                    output_directory=output_directory,
                    cwd=REPOSITORY_ROOT,
                    status=tooling_artifact_format.ToolArtifactStatus.DRY_RUN,
                )
            )
            next_index += 1
        for command_arguments in profile_plan.rust_benchmark_commands:
            command_records.append(
                tooling_artifact_format.build_command_record(
                    command_id=profile_command_identifier(f"rust_criterion_{next_index}", next_index),
                    tool_name="profile_regenie2_deep",
                    run_id=run_id,
                    phase="rust_criterion",
                    args=command_arguments,
                    output_directory=output_directory,
                    cwd=REPOSITORY_ROOT,
                    status=tooling_artifact_format.ToolArtifactStatus.DRY_RUN,
                )
            )
            next_index += 1
    return command_records


def build_profile_input_file_records(
    *,
    profile_plan: ProfilePlan | None,
    summary_payload: dict[str, typing.Any] | None,
) -> list[tooling_artifact_format.InputFileRecord]:
    """Build input-file records for profile artifacts."""
    input_records: list[tooling_artifact_format.InputFileRecord] = []
    for input_payload in collect_manifest_input_files(profile_plan=profile_plan, summary_payload=summary_payload):
        path_value = input_payload.get("path")
        if path_value is None:
            continue
        input_records.append(
            tooling_artifact_format.build_input_file_record(
                path=Path(str(path_value)),
                kind="profile_input",
            )
        )
    return input_records


def profile_metric_dimensions(aggregate_payload: dict[str, typing.Any]) -> dict[str, object]:
    """Build metric dimensions for a profile aggregate."""
    return {
        "implementation": str(aggregate_payload.get("implementation", "")),
        "trait_type": str(aggregate_payload.get("trait_type", "")),
        "device": str(aggregate_payload.get("device", "")),
        "status": str(aggregate_payload.get("status", "")),
    }


def optional_float(raw_value: typing.Any) -> float | None:
    """Return a float for numeric values."""
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return None


def append_profile_summary_metrics(
    *,
    metric_records: list[tooling_artifact_format.MetricRecord],
    run_id: str,
    summary_payload: dict[str, typing.Any],
) -> None:
    """Append normalized metrics from a profile summary payload."""
    metric_specs = (
        ("median_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MEDIAN.value),
        ("mean_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MEAN.value),
        ("min_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MINIMUM.value),
        ("max_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MAXIMUM.value),
        (
            "standard_deviation_seconds",
            "wall_time_seconds",
            tooling_artifact_format.MetricAggregation.STANDARD_DEVIATION.value,
        ),
        (
            "rows_per_second",
            "throughput_rows_per_second",
            tooling_artifact_format.MetricAggregation.MEDIAN.value,
        ),
    )
    for aggregate_index, aggregate_payload in enumerate(
        typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", []))
    ):
        case_id = str(aggregate_payload.get("name", f"headline_{aggregate_index}"))
        for source_field, metric_name, aggregation in metric_specs:
            unit = (
                tooling_artifact_format.MetricUnit.ROW.value
                if metric_name == "throughput_rows_per_second"
                else tooling_artifact_format.MetricUnit.SECONDS.value
            )
            metric_records.append(
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=case_id,
                    metric_name=metric_name,
                    value=optional_float(aggregate_payload.get(source_field)),
                    unit=unit,
                    aggregation=aggregation,
                    higher_is_better=metric_name == "throughput_rows_per_second",
                    dimensions=profile_metric_dimensions(aggregate_payload),
                    phase="headline_trials",
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="summary.json",
                        json_pointer=f"/headline_results/{aggregate_index}/{source_field}",
                    ),
                )
            )
    stage_totals = typing.cast("dict[str, typing.Any]", summary_payload.get("stage_totals", {}))
    for stage_name, seconds in sorted(stage_totals.items()):
        metric_records.append(
            tooling_artifact_format.build_metric_record(
                run_id=run_id,
                case_id=None,
                metric_name=f"stage.{stage_name}.seconds",
                value=optional_float(seconds),
                unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                higher_is_better=False,
                dimensions={},
                phase="stage_totals",
                source=tooling_artifact_format.MetricSource(
                    artifact_path="summary.json",
                    json_pointer=f"/stage_totals/{stage_name}",
                ),
            )
        )


def append_profile_plan_metrics(
    *,
    metric_records: list[tooling_artifact_format.MetricRecord],
    run_id: str,
    profile_plan: ProfilePlan,
) -> None:
    """Append normalized metrics from a dry-run profile plan."""
    budget_metrics = {
        "candidate_count": profile_plan.campaign_budget.total_candidate_count,
        "subprocess_run_count": profile_plan.campaign_budget.total_subprocess_run_count,
        "major_profiler_run_count": profile_plan.campaign_budget.total_major_profiler_run_count,
    }
    for metric_name, metric_value in budget_metrics.items():
        metric_records.append(
            tooling_artifact_format.build_metric_record(
                run_id=run_id,
                case_id="campaign_budget",
                metric_name=metric_name,
                value=metric_value,
                unit=tooling_artifact_format.MetricUnit.COUNT.value,
                aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                higher_is_better=None,
                dimensions={"chromosome_label": profile_plan.chromosome_label},
                phase="planning",
                source=tooling_artifact_format.MetricSource(
                    artifact_path="profile_plan.json",
                    json_pointer=f"/campaign_budget/{metric_name}",
                ),
            )
        )


def build_profile_metrics(
    *,
    run_id: str,
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: ProfilePlan | None = None,
) -> list[tooling_artifact_format.MetricRecord]:
    """Build normalized profile metrics."""
    metric_records: list[tooling_artifact_format.MetricRecord] = []
    if summary_payload is not None:
        append_profile_summary_metrics(
            metric_records=metric_records,
            run_id=run_id,
            summary_payload=summary_payload,
        )
    if profile_plan is not None:
        append_profile_plan_metrics(
            metric_records=metric_records,
            run_id=run_id,
            profile_plan=profile_plan,
        )
    return metric_records


def build_profile_failure_records(
    summary_payload: dict[str, typing.Any] | None,
) -> list[tooling_artifact_format.FailureRecord]:
    """Build structured failure records for profile trials."""
    failure_records: list[tooling_artifact_format.FailureRecord] = []
    for failure_index, trial_payload in enumerate(
        (
            payload
            for payload in collect_summary_trial_payloads(summary_payload)
            if str(payload.get("status", "")) == "failed"
        ),
        start=1,
    ):
        failure_records.append(
            tooling_artifact_format.FailureRecord(
                failure_id=f"F{failure_index:03d}",
                phase=str(trial_payload.get("trait_type", "profile")),
                status=tooling_artifact_format.ToolArtifactStatus.FAILED,
                message=f"Profile trial {trial_payload.get('name', failure_index)} failed.",
                exception_type=None,
                stderr_excerpt=None,
                stdout_log=str(trial_payload.get("stdout_log_path"))
                if trial_payload.get("stdout_log_path") is not None
                else None,
                stderr_log=str(trial_payload.get("stderr_log_path"))
                if trial_payload.get("stderr_log_path") is not None
                else None,
                command_id=profile_command_identifier(str(trial_payload.get("name", "")), failure_index),
            )
        )
    return failure_records


def build_profile_cases(
    summary_payload: dict[str, typing.Any] | None, profile_plan: ProfilePlan | None
) -> list[dict[str, object]]:
    """Build report case records for profile artifacts."""
    if summary_payload is not None:
        cases: list[dict[str, object]] = []
        for aggregate_payload in typing.cast(
            "list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])
        ):
            case_payload = dict(aggregate_payload)
            case_payload.pop("trials", None)
            case_payload.pop("warmup_trials", None)
            cases.append(typing.cast("dict[str, object]", case_payload))
        return cases
    if profile_plan is None:
        return []
    return [
        {
            "case_id": section.name,
            "display_name": section.display_name,
            "candidate_count": section.candidate_count,
            "subprocess_run_count": section.subprocess_run_count,
            "major_profiler_run_count": section.major_profiler_run_count,
        }
        for section in profile_plan.campaign_budget.sections
    ]


def build_profile_agent_summary(
    *,
    status: tooling_artifact_format.ToolArtifactStatus,
    summary_payload: dict[str, typing.Any] | None,
    profile_plan: ProfilePlan | None,
) -> dict[str, object]:
    """Build a concise agent-oriented profile summary."""
    if summary_payload is not None:
        headline_count = len(typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])))
        failures = build_profile_failure_records(summary_payload)
        return {
            "one_sentence": f"Deep profile completed with {headline_count} headline aggregate results.",
            "key_observations": [
                f"Status: {status.value}.",
                f"Headline aggregate count: {headline_count}.",
                f"Structured failure count: {len(failures)}.",
            ],
            "risks": [failure.message for failure in failures[:5]],
            "next_actions": [],
        }
    if profile_plan is not None:
        return {
            "one_sentence": "Deep profile plan was written without executing workloads.",
            "key_observations": [
                f"Status: {status.value}.",
                f"Estimated subprocess runs: {profile_plan.campaign_budget.total_subprocess_run_count}.",
                f"Estimated major profiler runs: {profile_plan.campaign_budget.total_major_profiler_run_count}.",
            ],
            "risks": list(profile_plan.campaign_budget.guidance),
            "next_actions": [],
        }
    return {
        "one_sentence": f"Deep profile finished with status {status.value}.",
        "key_observations": [f"Status: {status.value}."],
        "risks": [],
        "next_actions": [],
    }


def profile_summary_artifact_status(
    summary_payload: dict[str, typing.Any],
) -> tooling_artifact_format.ToolArtifactStatus:
    """Determine the overall standard status for a completed profile summary."""
    headline_results = typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", []))
    if not headline_results:
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    aggregate_statuses = {str(result.get("status", "")) for result in headline_results}
    if aggregate_statuses == {"success"}:
        return tooling_artifact_format.ToolArtifactStatus.SUCCESS
    if "success" in aggregate_statuses or "partial" in aggregate_statuses:
        return tooling_artifact_format.ToolArtifactStatus.PARTIAL
    return tooling_artifact_format.ToolArtifactStatus.FAILED


def write_standard_profile_artifacts(
    *,
    arguments: ProfileArguments,
    output_directory: Path,
    profiler_tool_status: dict[str, ProfilerToolStatus],
    status: tooling_artifact_format.ToolArtifactStatus,
    status_reason: str | None = None,
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: ProfilePlan | None = None,
    summary_markdown: str | None = None,
    hydra_config: omegaconf.DictConfig | None = None,
) -> None:
    """Write Tooling Artifact Format v1 artifacts for the deep profiler."""
    producer = tooling_artifact_format.build_producer(
        tool_name="profile_regenie2_deep",
        repository_root=REPOSITORY_ROOT,
    )
    run = tooling_artifact_format.build_run_identity(
        tool_name="profile_regenie2_deep",
        output_directory=output_directory,
        status=status,
        status_reason=status_reason,
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=output_directory,
        repository_root=REPOSITORY_ROOT,
    )
    report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title=f"{arguments.chromosome_label} Deep REGENIE Step 2 Profile",
        configuration=profile_configuration_payload(arguments),
        summary={
            "headline": f"Deep profile finished with status {status.value}.",
            "agent_summary": build_profile_agent_summary(
                status=status,
                summary_payload=summary_payload,
                profile_plan=profile_plan,
            ),
            "legacy_summary_path": "summary.json" if summary_payload is not None else None,
            "profile_plan_path": "profile_plan.json" if profile_plan is not None else None,
        },
        cases=build_profile_cases(summary_payload, profile_plan),
        trials=typing.cast("list[dict[str, object]]", collect_summary_trial_payloads(summary_payload)),
        metrics=build_profile_metrics(
            run_id=run.run_id,
            summary_payload=summary_payload,
            profile_plan=profile_plan,
        ),
        diagnostics={
            "profiler_tools": serialize_profiler_tool_status(profiler_tool_status),
            "legacy_artifact_manifest": "legacy_artifact_manifest.v2.json",
        },
        failures=build_profile_failure_records(summary_payload),
    )
    events = [
        tooling_artifact_format.build_tool_event(
            tool_name="profile_regenie2_deep",
            run_id=run.run_id,
            phase="profile",
            event="profile_artifacts_written",
            message=f"Deep profile artifacts written with status {status.value}.",
            fields={
                "chromosome_label": arguments.chromosome_label,
                "dry_run": arguments.dry_run,
                "status_reason": status_reason,
            },
        )
    ]
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=output_directory,
        report=report,
        events=events,
        commands=build_profile_command_records(
            output_directory=output_directory,
            run_id=run.run_id,
            summary_payload=summary_payload,
            profile_plan=profile_plan,
        ),
        input_files=build_profile_input_file_records(
            profile_plan=profile_plan,
            summary_payload=summary_payload,
        ),
        summary_markdown=summary_markdown,
        hydra_config=hydra_config,
        tool_payload=profile_configuration_payload(arguments),
        notes=["legacy_artifact_manifest.v2.json preserves the pre-v1 deep-profile manifest shape."],
    )


def command_output(
    command_arguments: list[str],
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, typing.Any]:
    """Run a metadata command and return captured output."""
    environment = dict(os.environ)
    if environment_overrides is not None:
        environment.update(environment_overrides)
    try:
        completed_process = subprocess.run(
            command_arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as error:
        return {
            "command": command_arguments,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command_arguments,
        "returncode": completed_process.returncode,
        "stdout": completed_process.stdout,
        "stderr": completed_process.stderr,
    }


def dirty_diff_sha256() -> str:
    """Hash the current dirty diff without writing it into the report."""
    completed_process = subprocess.run(["git", "diff"], check=False, capture_output=True)
    return hashlib.sha256(completed_process.stdout).hexdigest()


def collect_environment_metadata(
    baseline_paths: typing.Any,
    regenie_executable: str | None = None,
) -> dict[str, typing.Any]:
    """Collect reproducibility metadata for a profiling campaign."""
    input_paths = [
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.continuous_phenotype_path,
        baseline_paths.binary_phenotype_path,
        baseline_paths.covariate_path,
        baseline_paths.regenie_prediction_list_path,
        baseline_paths.regenie_qt_prediction_list_path,
    ]
    file_sizes = {
        str(input_path): input_path.stat().st_size
        for input_path in input_paths
        if input_path is not None and input_path.exists()
    }
    relevant_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("G_", "GWAS_ENGINE_", "JAX_", "XLA_", "CUDA_", "RAYON_", "SLURM_"))
    }
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "dirty_diff_sha256": dirty_diff_sha256(),
        "lscpu": command_output(["lscpu"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python": command_output([sys.executable, "--version"]),
        "jax": command_output([sys.executable, "-c", "import jax; print(jax.__version__); print(jax.devices())"]),
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "regenie": command_output([regenie_executable or "regenie", "--version"]),
        "hardware": dataclasses.asdict(baseline_benchmark.collect_hardware_summary()),
        "environment": relevant_environment,
        "input_file_sizes": file_sizes,
        "expected_full_variant_count": DEFAULT_VARIANT_COUNT,
    }


def ensure_prediction_lists(
    *,
    baseline_paths: typing.Any,
    regenie_executable: str,
    log_directory: Path,
) -> list[TrialResult]:
    """Generate missing REGENIE step 1 prediction lists before profiling."""
    setup_results: list[TrialResult] = []
    prediction_specs = [
        (
            baseline_paths.regenie_prediction_list_path,
            "regenie_step1_binary_setup",
            baseline_benchmark.build_regenie_step1_command(regenie_executable, baseline_paths),
        ),
        (
            baseline_paths.regenie_qt_prediction_list_path,
            "regenie_step1_quantitative_setup",
            baseline_benchmark.build_regenie_step1_continuous_command(regenie_executable, baseline_paths),
        ),
    ]
    for prediction_path, name, command_arguments in prediction_specs:
        if prediction_path is not None and prediction_path.exists():
            continue
        setup_results.append(
            run_logged_command(
                name=name,
                implementation="regenie",
                trait_type="setup",
                device="external_cpu",
                command_arguments=command_arguments,
                environment_overrides={},
                log_directory=log_directory,
            )
        )
    return setup_results


def replace_command_output_prefix(command_arguments: list[str], output_prefix: Path) -> list[str]:
    """Return a command with its --out value replaced."""
    updated_arguments = list(command_arguments)
    output_index = updated_arguments.index("--out")
    updated_arguments[output_index + 1] = str(output_prefix)
    return updated_arguments


def variant_metadata_candidate_paths(baseline_paths: baseline_benchmark.BaselinePaths) -> list[Path]:
    """Return metadata files that can provide BGEN-order variant identifiers."""
    return [
        baseline_paths.bgen_path.with_suffix(".pvar"),
        baseline_paths.bed_prefix.with_suffix(".bim"),
    ]


def read_pvar_variant_identifiers(metadata_path: Path, variant_limit: int) -> tuple[str, ...]:
    """Read the first variant identifiers from a PVAR file."""
    variant_identifiers: list[str] = []
    identifier_index = 2
    with metadata_path.open(encoding="utf-8") as metadata_file:
        for raw_line in metadata_file:
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            columns = line.split()
            if columns[0].startswith("#"):
                header_columns = [column.lstrip("#") for column in columns]
                if "ID" in header_columns:
                    identifier_index = header_columns.index("ID")
                continue
            if len(columns) <= identifier_index:
                continue
            variant_identifier = columns[identifier_index]
            if variant_identifier and variant_identifier != ".":
                variant_identifiers.append(variant_identifier)
            if len(variant_identifiers) >= variant_limit:
                break
    return tuple(variant_identifiers)


def read_bim_variant_identifiers(metadata_path: Path, variant_limit: int) -> tuple[str, ...]:
    """Read the first variant identifiers from a BIM file."""
    variant_identifiers: list[str] = []
    with metadata_path.open(encoding="utf-8") as metadata_file:
        for raw_line in metadata_file:
            columns = raw_line.strip().split()
            if len(columns) < 2:
                continue
            variant_identifier = columns[1]
            if variant_identifier and variant_identifier != ".":
                variant_identifiers.append(variant_identifier)
            if len(variant_identifiers) >= variant_limit:
                break
    return tuple(variant_identifiers)


def read_variant_identifiers(metadata_path: Path, variant_limit: int) -> tuple[str, ...]:
    """Read first variant identifiers from a supported metadata file."""
    if metadata_path.suffix == ".pvar":
        return read_pvar_variant_identifiers(metadata_path, variant_limit)
    if metadata_path.suffix == ".bim":
        return read_bim_variant_identifiers(metadata_path, variant_limit)
    message = f"Unsupported variant metadata file: {metadata_path}"
    raise ValueError(message)


def build_regenie_baseline_scope(
    *,
    arguments: ProfileArguments,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_directory: Path,
) -> RegenieBaselineScope:
    """Build original REGENIE workload scope for direct paired comparisons."""
    variant_limit = arguments.regenie_baseline_variant_limit
    if variant_limit is None:
        variant_limit = arguments.variant_limit
    if variant_limit is None:
        return RegenieBaselineScope(
            status=RegenieBaselineScopeStatus.FULL,
            variant_limit=None,
            extract_path=None,
            metadata_path=None,
            selected_variant_count=None,
            variant_identifiers=(),
            notes="Original REGENIE baseline uses the full configured BGEN workload.",
        )
    if variant_limit <= 0:
        return RegenieBaselineScope(
            status=RegenieBaselineScopeStatus.UNSUPPORTED,
            variant_limit=variant_limit,
            extract_path=None,
            metadata_path=None,
            selected_variant_count=None,
            variant_identifiers=(),
            notes="Bounded REGENIE baseline requires a positive variant limit.",
        )
    for metadata_path in variant_metadata_candidate_paths(baseline_paths):
        if not metadata_path.exists():
            continue
        variant_identifiers = read_variant_identifiers(metadata_path, variant_limit)
        if variant_identifiers:
            extract_path = output_directory / "headline_runs" / f"regenie_first_{len(variant_identifiers)}_variants.txt"
            return RegenieBaselineScope(
                status=RegenieBaselineScopeStatus.BOUNDED,
                variant_limit=variant_limit,
                extract_path=extract_path,
                metadata_path=metadata_path,
                selected_variant_count=len(variant_identifiers),
                variant_identifiers=variant_identifiers,
                notes=(
                    "Original REGENIE baseline is bounded with an --extract list derived from the first "
                    f"{len(variant_identifiers)} variants in {metadata_path}."
                ),
            )
    metadata_paths = ", ".join(str(path) for path in variant_metadata_candidate_paths(baseline_paths))
    return RegenieBaselineScope(
        status=RegenieBaselineScopeStatus.UNSUPPORTED,
        variant_limit=variant_limit,
        extract_path=None,
        metadata_path=None,
        selected_variant_count=None,
        variant_identifiers=(),
        notes=f"Bounded REGENIE baseline needs a .pvar or .bim metadata file; checked {metadata_paths}.",
    )


def write_regenie_baseline_extract_file(scope: RegenieBaselineScope) -> None:
    """Write the REGENIE extract list for a bounded baseline scope."""
    if scope.status != RegenieBaselineScopeStatus.BOUNDED or scope.extract_path is None:
        return
    scope.extract_path.parent.mkdir(parents=True, exist_ok=True)
    scope.extract_path.write_text("\n".join(scope.variant_identifiers) + "\n", encoding="utf-8")


def apply_regenie_baseline_scope(
    command_arguments: list[str],
    baseline_scope: RegenieBaselineScope,
) -> list[str]:
    """Apply bounded baseline filters to a REGENIE command."""
    updated_arguments = list(command_arguments)
    if baseline_scope.extract_path is not None:
        updated_arguments.extend(["--extract", str(baseline_scope.extract_path)])
    return updated_arguments


def build_regenie_step2_command(
    *,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_prefix: Path,
    baseline_scope: RegenieBaselineScope,
) -> list[str]:
    """Build one original REGENIE step 2 command with an isolated output prefix."""
    if trait_type == "binary":
        base_command = baseline_benchmark.build_regenie_step2_command(regenie_executable, baseline_paths)
    else:
        base_command = baseline_benchmark.build_regenie_step2_continuous_command(regenie_executable, baseline_paths)
    return apply_regenie_baseline_scope(
        replace_command_output_prefix(base_command, output_prefix),
        baseline_scope,
    )


def build_candidate_slug(candidate: Step2Candidate) -> str:
    """Build a stable filename slug for a tuning candidate."""
    candidate_parts = [
        candidate.trait_type,
        candidate.device,
        f"chunk{candidate.chunk_size}",
        f"staging{candidate.staging_depth}",
        f"callbackbatch{candidate.native_callback_batch_size}",
        f"inflight{candidate.result_in_flight_limit if candidate.result_in_flight_limit is not None else 'default'}",
        f"buffer{candidate.dosage_buffer_limit if candidate.dosage_buffer_limit is not None else 'default'}",
        f"writer{candidate.output_writer_thread_count}",
        f"queue{candidate.output_writer_queue_depth}",
        (
            f"tile{candidate.bgen_decode_tile_variant_count}"
            if candidate.bgen_decode_tile_variant_count is not None
            else "tiledefault"
        ),
        f"rayon{candidate.rayon_thread_count if candidate.rayon_thread_count is not None else 'default'}",
    ]
    if candidate.firth_batch_size is not None:
        candidate_parts.append(f"firth{candidate.firth_batch_size}")
    return "_".join(candidate_parts)


def build_step2_candidates(
    *,
    trait_type: str,
    device: str,
    bgen_candidates: tuple[BgenCandidateSummary, ...],
    chunk_sizes: tuple[int, ...],
    staging_depths: tuple[int, ...],
    native_callback_batch_sizes: tuple[int, ...],
    result_in_flight_limits: tuple[int | None, ...],
    dosage_buffer_limits: tuple[int | None, ...],
    writer_thread_counts: tuple[int, ...],
    queue_depth_multipliers: tuple[int, ...],
    firth_batch_sizes: tuple[int, ...],
) -> tuple[Step2Candidate, ...]:
    """Build the g step 2 candidate grid."""
    candidates: list[Step2Candidate] = []
    for bgen_candidate in bgen_candidates:
        for chunk_size in chunk_sizes:
            for staging_depth in staging_depths:
                for native_callback_batch_size in native_callback_batch_sizes:
                    for result_in_flight_limit in result_in_flight_limits:
                        for dosage_buffer_limit in dosage_buffer_limits:
                            for writer_thread_count in writer_thread_counts:
                                for queue_depth in build_queue_depth_values(
                                    writer_thread_count,
                                    queue_depth_multipliers,
                                ):
                                    if trait_type == "binary":
                                        for firth_batch_size in firth_batch_sizes:
                                            candidates.append(
                                                Step2Candidate(
                                                    trait_type=trait_type,
                                                    device=device,
                                                    chunk_size=chunk_size,
                                                    staging_depth=staging_depth,
                                                    native_callback_batch_size=native_callback_batch_size,
                                                    result_in_flight_limit=result_in_flight_limit,
                                                    dosage_buffer_limit=dosage_buffer_limit,
                                                    output_writer_thread_count=writer_thread_count,
                                                    output_writer_queue_depth=queue_depth,
                                                    bgen_decode_tile_variant_count=(
                                                        bgen_candidate.decode_tile_variant_count
                                                    ),
                                                    rayon_thread_count=bgen_candidate.rayon_thread_count,
                                                    firth_batch_size=firth_batch_size,
                                                )
                                            )
                                        continue
                                    candidates.append(
                                        Step2Candidate(
                                            trait_type=trait_type,
                                            device=device,
                                            chunk_size=chunk_size,
                                            staging_depth=staging_depth,
                                            native_callback_batch_size=native_callback_batch_size,
                                            result_in_flight_limit=result_in_flight_limit,
                                            dosage_buffer_limit=dosage_buffer_limit,
                                            output_writer_thread_count=writer_thread_count,
                                            output_writer_queue_depth=queue_depth,
                                            bgen_decode_tile_variant_count=bgen_candidate.decode_tile_variant_count,
                                            rayon_thread_count=bgen_candidate.rayon_thread_count,
                                            firth_batch_size=None,
                                        )
                                    )
    return tuple(candidates)


def build_g_trial_environment(
    *,
    candidate: Step2Candidate,
    cache_directory: Path,
    stage_timing_path: Path | None,
) -> dict[str, str]:
    """Build child process environment overrides for one g trial."""
    del cache_directory, stage_timing_path
    return {
        "JAX_DEBUG_LOG_MODULES": JAX_DEBUG_LOG_MODULES,
        "JAX_LOGGING_LEVEL": "DEBUG",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
    }


def build_g_step2_child_command(
    *,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_prefix: Path,
    variant_limit: int | None,
    cache_directory: Path | None = None,
    stage_timing_path: Path | None = None,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
    diagnostic_options: dict[str, object] | None = None,
) -> list[str]:
    """Build one isolated Python child command for a g REGENIE step 2 run."""
    jax_cache_directory = resolve_profile_jax_cache_directory(candidate, cache_directory)
    regenie_run_spec = build_g_step2_regenie_run_spec(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        variant_limit=variant_limit,
        jax_cache_directory=jax_cache_directory,
        stage_timing_path=stage_timing_path,
    )
    regenie_options = tooling_g_regenie.render_python_api_options(regenie_run_spec)
    rendered_diagnostics = typing.cast("dict[str, object]", regenie_options.setdefault("diagnostics", {}))
    rendered_diagnostics.update(diagnostic_options or {})
    regenie_options_payload = json.dumps(tooling_reports.to_jsonable(regenie_options), sort_keys=True)
    jax_cache_directory_payload = str(jax_cache_directory) if jax_cache_directory is not None else None
    child_code = textwrap.dedent(
        """
        import json
        import time
        from pathlib import Path

        import jax
        import polars as pl

        from g import api, types

        def configure_jax_profile_diagnostics():
            for setting_name, value in (
                ("jax_explain_cache_misses", True),
                ("jax_logging_level", "DEBUG"),
                ("jax_log_compiles", True),
            ):
                try:
                    jax.config.update(setting_name, value)
                except Exception:
                    pass

        def count_artifact_rows(artifacts):
            artifact_values = artifacts.phenotype_artifacts or (artifacts,)
            output_paths = []
            output_row_count = 0
            for artifact in artifact_values:
                if artifact.final_parquet is not None:
                    output_paths.append(str(artifact.final_parquet))
                    output_row_count += pl.scan_parquet(artifact.final_parquet).select(pl.len()).collect().item()
                    continue
                if artifact.output_run_directory is None:
                    continue
                output_run_directory = Path(artifact.output_run_directory)
                parquet_paths = sorted((output_run_directory / "parts").glob("*.parquet"))
                arrow_paths = sorted((output_run_directory / "chunks").glob("*.arrow"))
                for parquet_path in parquet_paths:
                    output_paths.append(str(parquet_path))
                    output_row_count += pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
                for arrow_path in arrow_paths:
                    output_paths.append(str(arrow_path))
                    output_row_count += pl.scan_ipc(arrow_path).select(pl.len()).collect().item()
            if not output_paths:
                raise RuntimeError("No readable output artifacts were produced.")
            return output_row_count, output_paths

        configure_jax_profile_diagnostics()
        trace_directory = {trace_directory!r}
        memory_profile_path = {memory_profile_path!r}
        if trace_directory is not None:
            jax.profiler.start_trace(trace_directory)
        try:
            start_time = time.perf_counter()
            regenie_options = json.loads({regenie_options_payload!r})
            artifacts = api.regenie.from_options(regenie_options)
            wall_time_seconds = time.perf_counter() - start_time
            output_row_count, output_paths = count_artifact_rows(artifacts)
            probe_array = jax.device_put(0)
            probe_device = next(iter(probe_array.devices()))
            if memory_profile_path is not None:
                jax.profiler.save_device_memory_profile(memory_profile_path)
            print(json.dumps({{
                "wall_time_seconds": wall_time_seconds,
                "output_path": output_paths[0],
                "output_paths": output_paths,
                "output_row_count": int(output_row_count),
                "jax_devices": [str(device) for device in jax.devices()],
                "jax_probe_device": str(probe_device),
                "jax_probe_device_platform": getattr(probe_device, "platform", None),
                "jax_cache_directory": {jax_cache_directory_payload!r},
                "jax_persistent_cache_used": True,
            }}))
        finally:
            if trace_directory is not None:
                jax.profiler.stop_trace()
        """
    ).format(
        trace_directory=str(trace_directory) if trace_directory is not None else None,
        memory_profile_path=str(memory_profile_path) if memory_profile_path is not None else None,
        regenie_options_payload=regenie_options_payload,
        jax_cache_directory_payload=jax_cache_directory_payload,
    )
    return [sys.executable, "-c", child_code]


def build_g_step2_regenie_run_spec(
    *,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_prefix: Path,
    variant_limit: int | None,
    jax_cache_directory: Path | None,
    stage_timing_path: Path | None,
) -> tooling_g_regenie.RegenieRunSpec:
    """Build the shared REGENIE run spec for a deep-profile child command."""
    is_binary_trait = candidate.trait_type == "binary"
    phenotype_path = (
        baseline_paths.binary_phenotype_path if is_binary_trait else baseline_paths.continuous_phenotype_path
    )
    phenotype_name = "phenotype_binary" if is_binary_trait else "phenotype_continuous"
    prediction_path = (
        baseline_paths.regenie_prediction_list_path
        if is_binary_trait
        else baseline_paths.regenie_qt_prediction_list_path
    )
    return tooling_g_regenie.RegenieRunSpec(
        trait_kind=(
            tooling_g_regenie.RegenieTraitKind.BINARY
            if is_binary_trait
            else tooling_g_regenie.RegenieTraitKind.QUANTITATIVE
        ),
        command_prefix=(sys.executable, "-m", "g", "regenie"),
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=baseline_paths.bgen_path,
            sample_path=baseline_paths.sample_path,
            phenotype_path=phenotype_path,
            phenotype_columns=(phenotype_name,),
            covariate_path=baseline_paths.covariate_path,
            covariate_columns=("age", "sex"),
            prediction_list_path=prediction_path,
            output_prefix=output_prefix,
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=tooling_g_regenie.RegenieDevice(candidate.device),
            bsize=candidate.chunk_size,
            threads=candidate.rayon_thread_count,
            staging_depth=candidate.staging_depth,
            native_callback_batch_size=candidate.native_callback_batch_size,
            result_in_flight_limit=candidate.result_in_flight_limit,
            dosage_buffer_limit=candidate.dosage_buffer_limit,
            variant_limit=variant_limit,
            trusted_no_missing_diploid=None,
            trusted_bgen_validation_mode=None,
            bgen_decode_tile_variant_count=candidate.bgen_decode_tile_variant_count or 64,
            firth_batch_size=candidate.firth_batch_size or 1024,
            firth_candidate_capacity=None,
            gpu_genotype_format=None,
            jax_cache_dir=jax_cache_directory,
            jax_persistent_cache=True,
            jax_persistent_cache_min_entry_size_bytes=-1,
            jax_persistent_cache_min_compile_time_seconds=0,
            jax_xla_autotune_cache=ENABLE_XLA_AUTOTUNE_CACHE,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            output_format="parquet",
            output_run_directory=None,
            writer_threads=candidate.output_writer_thread_count,
            writer_queue_depth=candidate.output_writer_queue_depth,
            chunks_per_arrow_file=None,
            arrow_compression=None,
            parquet_compression=None,
            output_statistic_dtype=None,
            finalize_parquet=None,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(
            telemetry=None,
            log_dir=None,
            stage_timings_json=stage_timing_path,
            profile_summary_json=None,
            log_file=None,
            log_filter=None,
            log_stderr=None,
            progress_interval_seconds=None,
            progress_interval_chunks=None,
        ),
        binary=(
            tooling_g_regenie.RegenieBinaryOptions(
                firth=True,
                approx=True,
                firth_se=None,
                p_threshold=None,
            )
            if is_binary_trait
            else None
        ),
    )


def build_application_output_run_directory(output_prefix: Path) -> Path:
    """Return the default g output run directory for an output prefix."""
    return output_prefix.with_name(f"{output_prefix.name}.g")


def subprocess_output_text(output: str | bytes | None) -> str:
    """Return subprocess output as text for logs and diagnostics."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def run_logged_command(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    environment_overrides: dict[str, str],
    log_directory: Path,
    timeout_seconds: int | None = None,
) -> TrialResult:
    """Run one command and persist stdout/stderr logs."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_directory / f"{name}.stdout.log"
    stderr_log_path = log_directory / f"{name}.stderr.log"
    environment = dict(os.environ)
    environment.update(environment_overrides)
    logger.info("Starting %s profiler/workload command", name)
    logger.debug("Command for %s: %s", name, shlex.join(command_arguments))
    if timeout_seconds is not None:
        logger.debug("Timeout for %s set to %.1fs", name, float(timeout_seconds))
    start_time = time.perf_counter()
    command_stdout: str = ""
    command_stderr: str = ""
    status = "success"
    notes: str | None = None
    timeout_reached = False
    try:
        completed_process = subprocess.run(
            command_arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
        command_stdout = subprocess_output_text(completed_process.stdout)
        command_stderr = subprocess_output_text(completed_process.stderr)
        status = "success" if completed_process.returncode == 0 else "failed"
        if completed_process.returncode != 0:
            notes = command_stderr.strip() or command_stdout.strip()
    except subprocess.TimeoutExpired as error:
        timeout_reached = True
        command_stdout = subprocess_output_text(error.stdout)
        command_stderr = subprocess_output_text(error.stderr)
        notes = (
            f"{name} timed out after {float(timeout_seconds or 0):.3f}s with no completion (limit={timeout_seconds}s)."
        )
        status = "failed"
    wall_time_seconds = time.perf_counter() - start_time
    stdout_log_path.write_text(command_stdout, encoding="utf-8")
    stderr_log_path.write_text(command_stderr, encoding="utf-8")
    permission_block_note = permission_blocked_profiler_note(
        stdout=command_stdout,
        stderr=command_stderr,
    )
    if permission_block_note is not None:
        status = "skipped"
        notes = permission_block_note
    if notes is None and not timeout_reached:
        notes = command_stderr.strip() or command_stdout.strip()
    logger.info(
        "Finished %s with status=%s in %.3fs",
        name,
        status,
        wall_time_seconds,
    )
    return TrialResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status=status,
        wall_time_seconds=wall_time_seconds,
        output_row_count=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        notes=notes,
    )


def permission_blocked_profiler_note(*, stdout: str, stderr: str) -> str | None:
    """Return an actionable note for known profiler permission failures."""
    combined_output = f"{stderr}\n{stdout}"
    if "ERR_NVGPUCTRPERM" in combined_output:
        return (
            "Nsight Compute connected to the CUDA process, but the NVIDIA driver restricts GPU performance "
            "counter access to admin users. Ask the cluster administrator to allow non-admin GPU performance "
            "counters on the GPU nodes, or keep using Nsight Systems/JAX traces for CUDA timelines."
        )
    return None


def skipped_profile_result(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    log_directory: Path,
    notes: str,
) -> TrialResult:
    """Build a skipped profiler result and persist the skip reason."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_directory / f"{name}.stdout.log"
    stderr_log_path = log_directory / f"{name}.stderr.log"
    stdout_log_path.write_text("", encoding="utf-8")
    stderr_log_path.write_text(notes + "\n", encoding="utf-8")
    logger.info("Skipping %s: %s", name, notes)
    return TrialResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status="skipped",
        wall_time_seconds=None,
        output_row_count=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        command_arguments=[],
        environment_overrides={},
        notes=notes,
    )


def unsupported_aggregate_result(
    *,
    name: str,
    trait_type: str,
    device: str,
    log_directory: Path,
    notes: str,
) -> AggregateResult:
    """Build an unsupported aggregate result and persist the reason."""
    trial_result = skipped_profile_result(
        name=f"{name}_unsupported",
        implementation="regenie",
        trait_type=trait_type,
        device=device,
        log_directory=log_directory,
        notes=notes,
    )
    unsupported_trial_result = dataclasses.replace(trial_result, status="unsupported")
    return AggregateResult(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device=device,
        status="unsupported",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=None,
        mean_wall_time_seconds=None,
        min_wall_time_seconds=None,
        max_wall_time_seconds=None,
        standard_deviation_seconds=None,
        rows_per_second=None,
        trials=[unsupported_trial_result],
    )


def write_inline_python_profile_script(command_arguments: list[str], script_path: Path) -> Path:
    """Write an inline Python command to a script file for external profilers."""
    if len(command_arguments) < 3 or command_arguments[0] != sys.executable or command_arguments[1] != "-c":
        message = "Expected an inline Python child command."
        raise ValueError(message)
    script_path.write_text(command_arguments[2], encoding="utf-8")
    return script_path


def build_deep_profiler_run_paths(
    *,
    profile_directory: Path,
    profile_name: str,
    emit_stage_timings: bool,
) -> DeepProfilerRunPaths:
    """Build isolated application paths for one deep profiler implementation."""
    application_output_prefix = profile_directory / profile_name
    stage_timing_path = profile_directory / f"{profile_name}.stage_timings.json" if emit_stage_timings else None
    return DeepProfilerRunPaths(
        application_output_prefix=application_output_prefix,
        application_output_run_directory=build_application_output_run_directory(application_output_prefix),
        stage_timing_path=stage_timing_path,
        profile_script_path=profile_directory / f"{profile_name}_child.py",
    )


def build_deep_profiler_child_command(
    *,
    profile_directory: Path,
    profile_name: str,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    cache_directory: Path,
    variant_limit: int | None,
    emit_stage_timings: bool,
) -> DeepProfilerChildCommand:
    """Build an isolated child command for one deep profiler implementation."""
    run_paths = build_deep_profiler_run_paths(
        profile_directory=profile_directory,
        profile_name=profile_name,
        emit_stage_timings=emit_stage_timings,
    )
    inline_command_arguments = build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=run_paths.application_output_prefix,
        variant_limit=variant_limit,
        cache_directory=cache_directory,
        stage_timing_path=run_paths.stage_timing_path,
    )
    write_inline_python_profile_script(inline_command_arguments, run_paths.profile_script_path)
    return DeepProfilerChildCommand(
        command_arguments=[sys.executable, str(run_paths.profile_script_path)],
        environment_overrides=build_g_trial_environment(
            candidate=candidate,
            cache_directory=cache_directory,
            stage_timing_path=run_paths.stage_timing_path,
        ),
        run_paths=run_paths,
    )


def attach_deep_profiler_metadata(
    *,
    result: TrialResult,
    run_paths: DeepProfilerRunPaths,
    profiler_artifact_path: Path | None,
) -> TrialResult:
    """Attach profiler artifact and application output metadata to a result."""
    return dataclasses.replace(
        result,
        profiler_artifact_path=str(profiler_artifact_path) if profiler_artifact_path is not None else None,
        application_output_prefix=str(run_paths.application_output_prefix),
        application_output_run_directory=str(run_paths.application_output_run_directory),
        stage_timing_path=str(run_paths.stage_timing_path) if run_paths.stage_timing_path is not None else None,
    )


def executable_name(executable_path: str | None) -> str:
    """Return the command basename for an optional executable path."""
    if executable_path is None:
        return ""
    return Path(executable_path).name


def build_scalene_command_arguments(
    *,
    tool_status: ProfilerToolStatus,
    output_path: Path,
    profile_script_path: Path,
) -> list[str]:
    """Build a Scalene command that preserves project dependencies."""
    if tool_status.executable_path == sys.executable:
        return [
            sys.executable,
            "-m",
            "scalene",
            "run",
            "--outfile",
            str(output_path),
            str(profile_script_path),
        ]
    if executable_name(tool_status.executable_path) == "uv":
        return [
            tool_status.executable_path or "uv",
            "run",
            "--no-sync",
            "--with",
            "scalene",
            "scalene",
            "run",
            "--outfile",
            str(output_path),
            str(profile_script_path),
        ]
    return [
        tool_status.executable_path or "scalene",
        "run",
        "--outfile",
        str(output_path),
        str(profile_script_path),
    ]


def build_memray_command_arguments(
    *,
    tool_status: ProfilerToolStatus,
    output_path: Path,
    profile_script_path: Path,
) -> list[str]:
    """Build a Memray command that preserves project dependencies."""
    memray_arguments = [
        "-m",
        "memray",
        "run",
        "--force",
        "--native",
        "--output",
        str(output_path),
        str(profile_script_path),
    ]
    if tool_status.executable_path == sys.executable:
        return [sys.executable, *memray_arguments]
    if executable_name(tool_status.executable_path) == "uv":
        return [
            tool_status.executable_path or "uv",
            "run",
            "--no-sync",
            "--with",
            "memray",
            "python",
            *memray_arguments,
        ]
    return [
        tool_status.executable_path or "memray",
        "run",
        "--force",
        "--native",
        "--output",
        str(output_path),
        str(profile_script_path),
    ]


def append_skipped_executable_profile(
    *,
    results: dict[str, typing.Any],
    tool_status: ProfilerToolStatus,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    log_directory: Path,
) -> None:
    """Append a skipped profiler result for a missing executable."""
    results["sampling_profiles"].append(
        dataclasses.asdict(
            skipped_profile_result(
                name=name,
                implementation=implementation,
                trait_type=trait_type,
                device=device,
                log_directory=log_directory,
                notes=tool_status.notes,
            )
        )
    )


def append_logged_profile_result(
    *,
    results: dict[str, typing.Any],
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    environment_overrides: dict[str, str],
    log_directory: Path,
    run_paths: DeepProfilerRunPaths,
    profiler_artifact_path: Path | None,
    timeout_seconds: int | None = None,
) -> None:
    """Run and append one external profiler result."""
    results["sampling_profiles"].append(
        dataclasses.asdict(
            attach_deep_profiler_metadata(
                result=run_logged_command(
                    name=name,
                    implementation=implementation,
                    trait_type=trait_type,
                    device=device,
                    command_arguments=command_arguments,
                    environment_overrides=environment_overrides,
                    log_directory=log_directory,
                    timeout_seconds=timeout_seconds,
                ),
                run_paths=run_paths,
                profiler_artifact_path=profiler_artifact_path,
            )
        )
    )


def run_g_trial(
    *,
    name: str,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_directory: Path,
    log_directory: Path,
    cache_directory: Path,
    variant_limit: int | None,
    emit_stage_timings: bool,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
    diagnostic_options: dict[str, object] | None = None,
) -> TrialResult:
    """Run one g trial in a fresh Python process."""
    output_prefix = output_directory / name
    stage_timing_path = output_directory / f"{name}.stage_timings.json" if emit_stage_timings else None
    resolved_cache_directory = resolve_profile_jax_cache_directory(candidate, cache_directory)
    before_cache_snapshot = collect_jax_cache_snapshot(resolved_cache_directory)
    command_arguments = build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        variant_limit=variant_limit,
        cache_directory=cache_directory,
        stage_timing_path=stage_timing_path,
        trace_directory=trace_directory,
        memory_profile_path=memory_profile_path,
        diagnostic_options=diagnostic_options,
    )
    environment_overrides = build_g_trial_environment(
        candidate=candidate,
        cache_directory=cache_directory,
        stage_timing_path=stage_timing_path,
    )
    result = run_logged_command(
        name=name,
        implementation="g",
        trait_type=candidate.trait_type,
        device=candidate.device,
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        log_directory=log_directory,
    )
    after_cache_snapshot = collect_jax_cache_snapshot(resolved_cache_directory)
    output_row_count = None
    output_path = None
    device_diagnostics = None
    child_reported_cache_directory = None
    if result.status == "success":
        output_payload = json.loads(Path(result.stdout_log_path).read_text(encoding="utf-8").strip().splitlines()[-1])
        output_row_count = int(output_payload["output_row_count"])
        output_path = str(output_payload["output_path"])
        child_reported_cache_directory = typing.cast("str | None", output_payload.get("jax_cache_directory"))
        device_diagnostics = {
            "jax_devices": output_payload.get("jax_devices"),
            "jax_probe_device": output_payload.get("jax_probe_device"),
            "jax_probe_device_platform": output_payload.get("jax_probe_device_platform"),
        }
    jax_cache_diagnostics = build_jax_cache_diagnostics(
        cache_directory=resolved_cache_directory,
        child_reported_cache_directory=child_reported_cache_directory,
        persistent_cache_used=True,
        before_snapshot=before_cache_snapshot,
        after_snapshot=after_cache_snapshot,
        stderr_log_path=result.stderr_log_path,
    )
    return dataclasses.replace(
        result,
        output_row_count=output_row_count,
        output_path=output_path,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        application_output_prefix=str(output_prefix),
        application_output_run_directory=str(build_application_output_run_directory(output_prefix)),
        device_diagnostics=device_diagnostics,
        jax_cache_diagnostics=jax_cache_diagnostics,
    )


def run_regenie_trial(
    *,
    name: str,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_directory: Path,
    log_directory: Path,
    baseline_scope: RegenieBaselineScope,
) -> TrialResult:
    """Run one original REGENIE step 2 trial."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_prefix = output_directory / name
    regenie_profile_path = output_directory / f"{name}.regenie_profile.json"
    command_arguments = build_regenie_step2_command(
        trait_type=trait_type,
        regenie_executable=regenie_executable,
        baseline_paths=baseline_paths,
        output_prefix=output_prefix,
        baseline_scope=baseline_scope,
    )
    result = run_logged_command(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device="external_cpu",
        command_arguments=command_arguments,
        environment_overrides={"REGENIE_PROFILE_JSON": str(regenie_profile_path)},
        log_directory=log_directory,
    )
    output_row_count = comparison_benchmark.count_regenie_step2_rows(output_prefix)
    output_suffix = "phenotype_binary" if trait_type == "binary" else "phenotype_continuous"
    output_path = output_prefix.parent / f"{output_prefix.name}_{output_suffix}.regenie"
    return dataclasses.replace(
        result,
        output_row_count=output_row_count,
        output_path=str(output_path) if output_path.exists() else None,
        regenie_profile_path=str(regenie_profile_path) if regenie_profile_path.exists() else None,
    )


def aggregate_trial_results(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    warmup_count: int,
    trial_results: list[TrialResult],
    warmup_trials: list[TrialResult] | None = None,
) -> AggregateResult:
    """Aggregate successful measured trial results."""
    observed_warmup_trials = [] if warmup_trials is None else warmup_trials
    jax_cold_warm_summary = build_jax_cold_warm_diagnostics(
        warmup_trials=observed_warmup_trials,
        trial_results=trial_results,
    )
    successful_trials = [
        trial_result
        for trial_result in trial_results
        if trial_result.status == "success" and trial_result.wall_time_seconds is not None
    ]
    if not successful_trials:
        return AggregateResult(
            name=name,
            implementation=implementation,
            trait_type=trait_type,
            device=device,
            status="failed",
            trial_count=len(trial_results),
            warmup_count=warmup_count,
            median_wall_time_seconds=None,
            mean_wall_time_seconds=None,
            min_wall_time_seconds=None,
            max_wall_time_seconds=None,
            standard_deviation_seconds=None,
            rows_per_second=None,
            trials=trial_results,
            warmup_trials=observed_warmup_trials,
            jax_cold_warm_summary=jax_cold_warm_summary,
        )
    wall_times = [typing.cast("float", trial_result.wall_time_seconds) for trial_result in successful_trials]
    row_counts = [
        trial_result.output_row_count for trial_result in successful_trials if trial_result.output_row_count is not None
    ]
    median_wall_time = statistics.median(wall_times)
    rows_per_second = None
    if row_counts and median_wall_time > 0.0:
        rows_per_second = statistics.median(row_counts) / median_wall_time
    return AggregateResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status="success" if len(successful_trials) == len(trial_results) else "partial",
        trial_count=len(trial_results),
        warmup_count=warmup_count,
        median_wall_time_seconds=median_wall_time,
        mean_wall_time_seconds=statistics.fmean(wall_times),
        min_wall_time_seconds=min(wall_times),
        max_wall_time_seconds=max(wall_times),
        standard_deviation_seconds=statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
        rows_per_second=rows_per_second,
        trials=trial_results,
        warmup_trials=observed_warmup_trials,
        jax_cold_warm_summary=jax_cold_warm_summary,
    )


def run_repeated_g_trials(
    *,
    name: str,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_directory: Path,
    log_directory: Path,
    cache_directory: Path,
    variant_limit: int | None,
    warmup_count: int,
    trial_count: int,
    emit_stage_timings: bool,
) -> AggregateResult:
    """Warm and measure one g candidate in fresh child processes."""
    warmup_results: list[TrialResult] = []
    for warmup_index in range(warmup_count):
        warmup_results.append(
            run_g_trial(
                name=f"{name}_warmup{warmup_index:02d}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory,
                log_directory=log_directory,
                cache_directory=cache_directory,
                variant_limit=variant_limit,
                emit_stage_timings=False,
            )
        )
    trial_results = [
        run_g_trial(
            name=f"{name}_trial{trial_index:02d}",
            baseline_paths=baseline_paths,
            candidate=candidate,
            output_directory=output_directory,
            log_directory=log_directory,
            cache_directory=cache_directory,
            variant_limit=variant_limit,
            emit_stage_timings=emit_stage_timings,
        )
        for trial_index in range(trial_count)
    ]
    return aggregate_trial_results(
        name=name,
        implementation="g",
        trait_type=candidate.trait_type,
        device=candidate.device,
        warmup_count=warmup_count,
        trial_results=trial_results,
        warmup_trials=warmup_results,
    )


def run_repeated_regenie_trials(
    *,
    name: str,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_directory: Path,
    log_directory: Path,
    baseline_scope: RegenieBaselineScope,
    warmup_count: int,
    trial_count: int,
) -> AggregateResult:
    """Warm and measure original REGENIE step 2."""
    write_regenie_baseline_extract_file(baseline_scope)
    warmup_results: list[TrialResult] = []
    for warmup_index in range(warmup_count):
        warmup_results.append(
            run_regenie_trial(
                name=f"{name}_warmup{warmup_index:02d}",
                trait_type=trait_type,
                regenie_executable=regenie_executable,
                baseline_paths=baseline_paths,
                output_directory=output_directory,
                log_directory=log_directory,
                baseline_scope=baseline_scope,
            )
        )
    trial_results = [
        run_regenie_trial(
            name=f"{name}_trial{trial_index:02d}",
            trait_type=trait_type,
            regenie_executable=regenie_executable,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            log_directory=log_directory,
            baseline_scope=baseline_scope,
        )
        for trial_index in range(trial_count)
    ]
    return aggregate_trial_results(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device="external_cpu",
        warmup_count=warmup_count,
        trial_results=trial_results,
        warmup_trials=warmup_results,
    )


def summarize_bgen_case(case_report: typing.Any) -> BgenCandidateSummary:
    """Summarize one BGEN reader benchmark case."""
    matching_results = [
        path_result
        for path_result in case_report.path_results
        if path_result.path_mode == benchmark_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_BUFFERED.value
    ]
    if len(matching_results) != 1:
        message = "Expected exactly one variant-major buffered BGEN result."
        raise ValueError(message)
    path_result = matching_results[0]
    return BgenCandidateSummary(
        decode_tile_variant_count=case_report.decode_tile_variant_count,
        rayon_thread_count=case_report.rayon_thread_count,
        median_seconds=statistics.median(path_result.durations_seconds),
        mean_seconds=path_result.mean_seconds,
        durations_seconds=list(path_result.durations_seconds),
    )


def run_bgen_sweep(
    *,
    arguments: ProfileArguments,
    baseline_paths: typing.Any,
    output_directory: Path,
) -> tuple[BgenCandidateSummary, ...]:
    """Run BGEN reader sweeps over decode tile size and Rayon threads."""
    summaries: list[BgenCandidateSummary] = []
    variant_limit = arguments.variant_limit or 16_384
    sweep_directory = output_directory / "bgen_sweep"
    sweep_directory.mkdir(parents=True, exist_ok=True)
    for decode_tile_variant_count in parse_int_list(arguments.bgen_decode_tile_variant_counts):
        for rayon_thread_count in parse_int_list(arguments.rayon_thread_counts):
            benchmark_arguments = benchmark_bgen_reader.BenchmarkArguments(
                bgen=baseline_paths.bgen_path,
                sample=baseline_paths.sample_path,
                chunk_size=arguments.bgen_benchmark_chunk_size,
                chunk_sizes=str(arguments.bgen_benchmark_chunk_size),
                variant_limit=variant_limit,
                repeat_count=arguments.tuning_trials,
                path_modes=benchmark_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_BUFFERED.value,
                sample_selection_mode=benchmark_bgen_reader.SampleSelectionMode.FULL.value,
                sample_selection_modes="",
                decode_tile_variant_count=decode_tile_variant_count,
                decode_tile_variant_counts="",
                rayon_thread_count=rayon_thread_count,
                rayon_thread_counts="",
                trusted_no_missing_diploid=False,
                trusted_no_missing_diploid_modes="",
                emit_case_json=True,
                json_summary_path=None,
                markdown_summary_path=None,
            )
            case_report = benchmark_bgen_reader.run_case_subprocess(
                benchmark_arguments,
                arguments.bgen_benchmark_chunk_size,
                decode_tile_variant_count,
                rayon_thread_count,
                trusted_no_missing_diploid=False,
                sample_selection_mode=benchmark_bgen_reader.SampleSelectionMode.FULL,
            )
            summaries.append(summarize_bgen_case(case_report))
    summaries = sorted(summaries, key=lambda summary: (summary.median_seconds, summary.mean_seconds))
    (sweep_directory / "bgen_sweep.json").write_text(
        json.dumps([dataclasses.asdict(summary) for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )
    return tuple(summaries)


def run_candidate_tuning(
    *,
    arguments: ProfileArguments,
    baseline_paths: typing.Any,
    bgen_summaries: tuple[BgenCandidateSummary, ...],
    output_directory: Path,
    cache_directory: Path,
) -> CandidateTuningResults:
    """Tune g candidates for each trait/device and return winners."""
    winners: dict[str, AggregateResult] = {}
    finalist_results_by_key: dict[str, list[AggregateResult]] = {}
    emit_stage_timings = should_emit_stage_timings(arguments)
    chunk_sizes = parse_int_list(arguments.chunk_sizes)
    staging_depths = parse_int_list(arguments.staging_depths)
    native_callback_batch_sizes = parse_int_list(arguments.native_callback_batch_sizes)
    result_in_flight_limits = parse_optional_int_list(arguments.result_in_flight_limits)
    dosage_buffer_limits = parse_optional_int_list(arguments.dosage_buffer_limits)
    writer_thread_counts = parse_int_list(arguments.output_writer_thread_counts)
    queue_depth_multipliers = parse_int_list(arguments.writer_queue_depth_multipliers)
    firth_batch_sizes = parse_int_list(arguments.firth_batch_sizes)
    selected_bgen_summaries = bgen_summaries[: arguments.top_bgen_candidates]
    for workload_key in parse_profile_workload_keys(arguments.workload_keys):
        candidates = build_step2_candidates(
            trait_type=workload_key.trait_type,
            device=workload_key.device,
            bgen_candidates=selected_bgen_summaries,
            chunk_sizes=chunk_sizes,
            staging_depths=staging_depths,
            native_callback_batch_sizes=native_callback_batch_sizes,
            result_in_flight_limits=result_in_flight_limits,
            dosage_buffer_limits=dosage_buffer_limits,
            writer_thread_counts=writer_thread_counts,
            queue_depth_multipliers=queue_depth_multipliers,
            firth_batch_sizes=firth_batch_sizes,
        )
        if arguments.smoke:
            candidates = candidates[:1]
        initial_results = [
            run_repeated_g_trials(
                name=f"tune_{build_candidate_slug(candidate)}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory / "tuning_runs",
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                warmup_count=arguments.tuning_warmups,
                trial_count=arguments.tuning_trials,
                emit_stage_timings=False,
            )
            for candidate in candidates
        ]
        successful_initial_results = [
            result for result in initial_results if result.median_wall_time_seconds is not None
        ]
        finalists = sorted(
            successful_initial_results,
            key=lambda result: typing.cast("float", result.median_wall_time_seconds),
        )[: arguments.top_finalists]
        finalist_results: list[AggregateResult] = []
        for finalist in finalists:
            candidate = recover_candidate_from_trial(finalist.trials[0], candidates)
            finalist_results.append(
                run_repeated_g_trials(
                    name=f"finalist_{build_candidate_slug(candidate)}",
                    baseline_paths=baseline_paths,
                    candidate=candidate,
                    output_directory=output_directory / "finalist_runs",
                    log_directory=output_directory / "logs",
                    cache_directory=cache_directory,
                    variant_limit=arguments.variant_limit,
                    warmup_count=arguments.finalist_warmups,
                    trial_count=arguments.finalist_trials,
                    emit_stage_timings=emit_stage_timings,
                )
            )
        if finalist_results:
            winner = sorted(
                finalist_results,
                key=lambda result: typing.cast("float", result.median_wall_time_seconds),
            )[0]
            winners[workload_key.value] = winner
            finalist_results_by_key[workload_key.value] = finalist_results
        tuning_path = output_directory / f"tuning_{workload_key.value}.json"
        tuning_path.write_text(
            json.dumps(
                {
                    "initial_results": [dataclasses.asdict(result) for result in initial_results],
                    "finalist_results": [dataclasses.asdict(result) for result in finalist_results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return CandidateTuningResults(winners=winners, finalist_results_by_key=finalist_results_by_key)


def recover_candidate_from_trial(trial_result: TrialResult, candidates: tuple[Step2Candidate, ...]) -> Step2Candidate:
    """Recover the tuning candidate that produced a trial by matching its command and env."""
    for candidate in candidates:
        if build_candidate_slug(candidate) in trial_result.name:
            return candidate
    message = f"Could not recover candidate from trial {trial_result.name}."
    raise ValueError(message)


def run_headline_trials(
    *,
    arguments: ProfileArguments,
    baseline_paths: typing.Any,
    regenie_executable: str | None,
    regenie_baseline_scope: RegenieBaselineScope,
    winners: dict[str, AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> list[AggregateResult]:
    """Run headline original REGENIE and winning g configurations."""
    headline_results: list[AggregateResult] = []
    emit_stage_timings = should_emit_stage_timings(arguments)
    if arguments.include_regenie_baseline:
        regenie_trait_types = selected_regenie_baseline_trait_types(arguments)
        if regenie_executable is None:
            for trait_type in regenie_trait_types:
                headline_results.append(
                    unsupported_aggregate_result(
                        name=f"headline_regenie_{trait_type}",
                        trait_type=trait_type,
                        device="external_cpu",
                        log_directory=output_directory / "logs",
                        notes=(
                            "Original REGENIE baseline was requested, but no executable was resolved from "
                            f"{configured_regenie_executable(arguments)!r}."
                        ),
                    )
                )
        elif regenie_baseline_scope.status == RegenieBaselineScopeStatus.UNSUPPORTED:
            for trait_type in regenie_trait_types:
                headline_results.append(
                    unsupported_aggregate_result(
                        name=f"headline_regenie_{trait_type}",
                        trait_type=trait_type,
                        device="external_cpu",
                        log_directory=output_directory / "logs",
                        notes=regenie_baseline_scope.notes,
                    )
                )
        for trait_type in regenie_trait_types:
            if regenie_executable is None or regenie_baseline_scope.status == RegenieBaselineScopeStatus.UNSUPPORTED:
                continue
            headline_results.append(
                run_repeated_regenie_trials(
                    name=f"headline_regenie_{trait_type}",
                    trait_type=trait_type,
                    regenie_executable=regenie_executable,
                    baseline_paths=baseline_paths,
                    output_directory=output_directory / "headline_runs",
                    log_directory=output_directory / "logs",
                    baseline_scope=regenie_baseline_scope,
                    warmup_count=arguments.regenie_baseline_warmups,
                    trial_count=arguments.regenie_baseline_trials,
                )
            )
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = candidate_from_aggregate_name(winner_key, winner)
        headline_results.append(
            run_repeated_g_trials(
                name=f"headline_g_{winner_key}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory / "headline_runs",
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                warmup_count=arguments.headline_warmups,
                trial_count=arguments.headline_trials,
                emit_stage_timings=emit_stage_timings,
            )
        )
    return headline_results


def candidate_from_aggregate_name(winner_key: str, aggregate_result: AggregateResult) -> Step2Candidate:
    """Reconstruct a winner candidate from its child environment and command."""
    trial = aggregate_result.trials[0]
    trait_type, device = winner_key.rsplit("_", maxsplit=1)
    command = trial.command_arguments
    code = command[2] if len(command) >= 3 and command[1] == "-c" else ""

    def read_scalar(marker: str) -> str | None:
        marker_index = code.find(marker)
        if marker_index < 0:
            return None
        value_start = marker_index + len(marker)
        value_end_candidates = [
            candidate_index
            for candidate_index in (
                code.find(",", value_start),
                code.find("}", value_start),
                code.find("\n", value_start),
            )
            if candidate_index >= 0
        ]
        value_end = min(value_end_candidates) if value_end_candidates else len(code)
        return code[value_start:value_end].strip()

    def read_int(marker: str, default_value: int) -> int:
        raw_value = read_scalar(marker)
        if raw_value is None:
            return default_value
        return int(raw_value)

    def read_optional_int(marker: str) -> int | None:
        raw_value = read_scalar(marker)
        if raw_value is None or raw_value in {"None", "null"}:
            return None
        return int(raw_value)

    return Step2Candidate(
        trait_type=trait_type,
        device=device,
        chunk_size=read_int('"bsize": ', 8192),
        staging_depth=read_int('"staging_depth": ', 1),
        native_callback_batch_size=read_int('"native_callback_batch_size": ', 1),
        result_in_flight_limit=read_optional_int('"result_in_flight_limit": '),
        dosage_buffer_limit=read_optional_int('"dosage_buffer_limit": '),
        output_writer_thread_count=read_int('"writer_threads": ', 8),
        output_writer_queue_depth=read_int('"writer_queue_depth": ', 4),
        bgen_decode_tile_variant_count=read_int('"bgen_decode_tile_variant_count": ', 64),
        rayon_thread_count=read_int('"threads": ', 0) or None,
        firth_batch_size=read_int('"firth_batch_size": ', 1024),
    )


def build_runtime_comparisons(aggregate_results: list[AggregateResult]) -> dict[str, dict[str, float]]:
    """Build speedup/slowdown comparisons against original REGENIE."""
    by_name = {result.name: result for result in aggregate_results}
    comparisons: dict[str, dict[str, float]] = {}
    for trait_type in ("quantitative", "binary"):
        baseline = by_name.get(f"headline_regenie_{trait_type}")
        if baseline is None or baseline.median_wall_time_seconds is None:
            continue
        for result in aggregate_results:
            if (
                result.implementation != "g"
                or result.trait_type != trait_type
                or result.median_wall_time_seconds is None
            ):
                continue
            comparison_name = f"{result.name}_vs_regenie_{trait_type}"
            comparisons[comparison_name] = {
                "speedup_ratio": baseline.median_wall_time_seconds / result.median_wall_time_seconds,
                "absolute_delta_seconds": result.median_wall_time_seconds - baseline.median_wall_time_seconds,
            }
    return comparisons


def build_runtime_comparison_notes(aggregate_results: list[AggregateResult]) -> RuntimeComparisonNotes:
    """Build explicit non-success direct comparison notes."""
    unsupported: list[str] = []
    failed: list[str] = []
    regenie_results = {result.trait_type: result for result in aggregate_results if result.implementation == "regenie"}
    for g_result in aggregate_results:
        if g_result.implementation != "g":
            continue
        comparison_name = f"{g_result.name}_vs_regenie_{g_result.trait_type}"
        regenie_result = regenie_results.get(g_result.trait_type)
        if regenie_result is None:
            unsupported.append(f"{comparison_name}: no original REGENIE baseline was scheduled for this trait.")
            continue
        if regenie_result.status == "unsupported":
            notes = next((trial.notes for trial in regenie_result.trials if trial.notes), None)
            suffix = f" {notes}" if notes is not None else ""
            unsupported.append(f"{comparison_name}: original REGENIE baseline is unsupported.{suffix}")
            continue
        if regenie_result.median_wall_time_seconds is None:
            notes = next((trial.notes for trial in regenie_result.trials if trial.notes), None)
            suffix = f" {notes}" if notes is not None else ""
            failed.append(f"{comparison_name}: original REGENIE baseline did not produce a measured runtime.{suffix}")
            continue
        if g_result.median_wall_time_seconds is None:
            failed.append(f"{comparison_name}: g result did not produce a measured runtime.")
    return RuntimeComparisonNotes(unsupported=unsupported, failed=failed)


REGENIE_STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "input_setup": (
        "input_file_initialization",
        "phenotype_covariate_load",
        "prediction_load",
        "step2_setup",
        "null_residual_setup",
    ),
    "bgen_decode": ("block_read", "bgen_raw_read", "bgen_decode_impute_filter"),
    "association_compute": (
        "association_compute",
        "sparse_check",
        "covariate_projection",
        "qt_score",
        "bt_score",
        "correction_candidate_check",
        "firth_correction",
        "spa_correction",
    ),
    "output": ("output_setup", "block_output"),
}

G_STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "input_setup": (
        "bgen_engine_open_index_setup",
        "sample_phenotype_covariate_alignment",
        "prediction_source_load",
        "preflight_validation",
        "chromosome_state_preparation",
        "output_run_preparation",
        "output_writer_preparation",
    ),
    "bgen_decode": ("native_engine_delivery",),
    "association_compute": (
        "host_to_device_transfer",
        "jax_compute",
        "device_to_host_materialization",
    ),
    "output": (
        "output_write",
        "single_trait_output_write",
        "multi_trait_output_write_total",
        "writer_finish_and_parquet_finalization",
        "callback_drain",
    ),
}


def read_json_file(path: str | None) -> dict[str, typing.Any] | None:
    """Read a JSON file when the path exists."""
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return typing.cast("dict[str, typing.Any]", json.loads(json_path.read_text(encoding="utf-8")))


def collect_trial_stage_totals(trial: TrialResult) -> dict[str, float]:
    """Collect raw stage totals for one g or REGENIE trial."""
    stage_payload = read_json_file(trial.stage_timing_path)
    if stage_payload is None:
        stage_payload = read_json_file(trial.regenie_profile_path)
    if stage_payload is None:
        return {}
    return {stage_name: float(seconds) for stage_name, seconds in stage_payload.get("stage_totals_seconds", {}).items()}


def collect_stage_totals(aggregate_results: list[AggregateResult]) -> dict[str, float]:
    """Collect representative stage totals from g and REGENIE trials."""
    stage_totals: dict[str, float] = {}
    for aggregate_result in aggregate_results:
        for trial in aggregate_result.trials:
            for stage_name, seconds in collect_trial_stage_totals(trial).items():
                key = f"{aggregate_result.name}:{stage_name}"
                stage_totals[key] = seconds
    return stage_totals


def build_grouped_stage_totals(
    stage_totals: dict[str, float],
    stage_groups: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    """Map raw stage totals into common comparison groups."""
    grouped_totals: dict[str, float] = {}
    for group_name, raw_stage_names in stage_groups.items():
        grouped_totals[group_name] = sum(stage_totals.get(stage_name, 0.0) for stage_name in raw_stage_names)
    return grouped_totals


def representative_stage_totals(aggregate_result: AggregateResult) -> dict[str, float]:
    """Return stage totals from the median-ish successful trial for an aggregate."""
    successful_trials = [
        trial for trial in aggregate_result.trials if trial.status == "success" and trial.wall_time_seconds is not None
    ]
    if not successful_trials:
        return {}
    sorted_trials = sorted(successful_trials, key=lambda trial: typing.cast("float", trial.wall_time_seconds))
    median_index = len(sorted_trials) // 2
    return collect_trial_stage_totals(sorted_trials[median_index])


def build_stage_comparison_rows(aggregate_results: list[AggregateResult]) -> list[dict[str, float | str]]:
    """Build stage-by-stage REGENIE versus g comparison rows."""
    rows: list[dict[str, float | str]] = []
    results_by_trait = {
        result.trait_type: result
        for result in aggregate_results
        if result.implementation == "regenie" and result.median_wall_time_seconds is not None
    }
    for regenie_trait_type, regenie_result in results_by_trait.items():
        regenie_grouped_totals = build_grouped_stage_totals(
            representative_stage_totals(regenie_result),
            REGENIE_STAGE_GROUPS,
        )
        for g_result in aggregate_results:
            if (
                g_result.implementation != "g"
                or g_result.trait_type != regenie_trait_type
                or g_result.median_wall_time_seconds is None
            ):
                continue
            g_grouped_totals = build_grouped_stage_totals(representative_stage_totals(g_result), G_STAGE_GROUPS)
            for stage_group in REGENIE_STAGE_GROUPS:
                regenie_seconds = regenie_grouped_totals.get(stage_group, 0.0)
                g_seconds = g_grouped_totals.get(stage_group, 0.0)
                speedup_ratio = regenie_seconds / g_seconds if g_seconds > 0.0 else 0.0
                rows.append(
                    {
                        "trait_type": regenie_trait_type,
                        "g_device": g_result.device,
                        "stage_group": stage_group,
                        "regenie_seconds": regenie_seconds,
                        "g_seconds": g_seconds,
                        "g_speedup_ratio": speedup_ratio,
                    }
                )
    return rows


def build_algorithmic_findings(stage_comparison_rows: list[dict[str, float | str]]) -> list[str]:
    """Explain likely source-level reasons for measured stage gaps."""
    findings: list[str] = []
    for row in stage_comparison_rows:
        stage_group = str(row["stage_group"])
        speedup_ratio = float(row["g_speedup_ratio"])
        trait_type = str(row["trait_type"])
        device = str(row["g_device"])
        if stage_group == "bgen_decode" and speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in BGEN delivery, consistent with its native buffered decoder "
                "and larger chunk pipeline versus REGENIE's per-block BGEN read/decode path."
            )
        elif stage_group == "association_compute" and speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in association compute, consistent with vectorized chunk scoring "
                "and GPU/JAX execution when available."
            )
        elif stage_group == "output" and speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in output, consistent with chunked Arrow/Parquet writes instead "
                "of REGENIE's per-variant text formatting loop."
            )
        elif stage_group == "bgen_decode" and speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in measured BGEN delivery; the g timing group maps "
                "to inclusive native engine delivery, so it can also include downstream compute and correction work. "
                "Split that timer before attributing the whole gap to decoding."
            )
        elif stage_group == "association_compute" and speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in association compute; likely gaps include "
                "REGENIE's sparse genotype checks, OpenMP score/correction scheduling, and correction thresholds "
                "that avoid expensive fallback work on most variants."
            )
        elif stage_group == "output" and speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in output; likely gaps include g callback draining, "
                "Parquet finalization, and writer queue overhead for this workload."
            )
        elif speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in {stage_group}; this points to lower orchestration overhead "
                "or more compact data movement in the measured g path."
            )
        elif speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in {stage_group}; likely gaps include REGENIE's "
                "sparse genotype paths, OpenMP scheduling, null-model reuse, correction thresholds, or lower "
                "formatting overhead for this workload."
            )
        elif speedup_ratio == 1.0:
            findings.append(f"{trait_type}/{device}: {stage_group} is effectively tied between g and REGENIE.")
    return sorted(set(findings))


def numeric_diagnostic_value(raw_value: typing.Any) -> float:
    """Convert a diagnostic JSON value into a numeric value."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return 0.0
    return float(raw_value)


def optional_numeric_value(raw_value: typing.Any) -> float | None:
    """Convert a JSON value into a float when it is numeric."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return None
    return float(raw_value)


def sum_binary_diagnostic_count(binary_chunk_diagnostics: list[dict[str, typing.Any]], field_name: str) -> int:
    """Sum one integer diagnostic field across binary chunks."""
    return int(
        sum(numeric_diagnostic_value(diagnostics.get(field_name, 0)) for diagnostics in binary_chunk_diagnostics)
    )


def mean_binary_diagnostic_value(binary_chunk_diagnostics: list[dict[str, typing.Any]], field_name: str) -> float:
    """Average one diagnostic field across chunks."""
    if not binary_chunk_diagnostics:
        return 0.0
    total = sum(numeric_diagnostic_value(diagnostics.get(field_name, 0)) for diagnostics in binary_chunk_diagnostics)
    return total / len(binary_chunk_diagnostics)


def active_firth_iteration_values(
    binary_chunk_diagnostics: list[dict[str, typing.Any]],
    field_name: str,
) -> list[float]:
    """Return a Firth iteration field for chunks with attempted Firth correction."""
    return [
        numeric_diagnostic_value(diagnostics.get(field_name, 0))
        for diagnostics in binary_chunk_diagnostics
        if numeric_diagnostic_value(diagnostics.get("firth_candidate_count", 0)) > 0.0
    ]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Divide only when the denominator is positive."""
    if denominator <= 0.0:
        return None
    return numerator / denominator


def load_binary_diagnostic_trial_payload(
    *,
    stage_timing_mode: ProfileStageTimingMode,
    trial: TrialResult,
) -> BinaryDiagnosticTrialPayload:
    """Load one trial's stage timing payload with an explicit unavailable reason."""
    if trial.stage_timing_path is None:
        reason = (
            BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED
            if stage_timing_mode == ProfileStageTimingMode.OFF
            else BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING
        )
        return BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=None,
            unavailable_reason=reason,
            payload=None,
        )
    stage_timing_path = Path(trial.stage_timing_path)
    if not stage_timing_path.exists():
        return BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING,
            payload=None,
        )
    try:
        raw_payload = json.loads(stage_timing_path.read_text(encoding="utf-8"))
    except OSError:
        return BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID,
            payload=None,
        )
    except json.JSONDecodeError:
        return BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID,
            payload=None,
        )
    if not isinstance(raw_payload, dict):
        return BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID,
            payload=None,
        )
    return BinaryDiagnosticTrialPayload(
        trial_name=trial.name,
        stage_timing_path=trial.stage_timing_path,
        unavailable_reason=None,
        payload=typing.cast("dict[str, typing.Any]", raw_payload),
    )


def extract_binary_chunk_diagnostics(
    loaded_payload: BinaryDiagnosticTrialPayload,
) -> list[dict[str, typing.Any]] | None:
    """Extract valid binary chunk diagnostic mappings from one loaded payload."""
    if loaded_payload.payload is None:
        return None
    raw_binary_chunk_diagnostics = loaded_payload.payload.get("binary_chunk_diagnostics")
    if raw_binary_chunk_diagnostics is None or not isinstance(raw_binary_chunk_diagnostics, list):
        return None
    binary_chunk_diagnostics: list[dict[str, typing.Any]] = []
    for raw_chunk_diagnostics in raw_binary_chunk_diagnostics:
        if not isinstance(raw_chunk_diagnostics, dict):
            return None
        binary_chunk_diagnostics.append(typing.cast("dict[str, typing.Any]", raw_chunk_diagnostics))
    return binary_chunk_diagnostics


def summarize_stage_mapping(
    stage_timing_payloads: list[dict[str, typing.Any]],
    field_name: str,
) -> dict[str, float]:
    """Sum numeric values from a stage timing mapping field across trials."""
    summary: dict[str, float] = {}
    for stage_timing_payload in stage_timing_payloads:
        raw_mapping = stage_timing_payload.get(field_name)
        if not isinstance(raw_mapping, dict):
            continue
        for raw_key, raw_value in raw_mapping.items():
            numeric_value = optional_numeric_value(raw_value)
            if numeric_value is None:
                continue
            key = str(raw_key)
            summary[key] = summary.get(key, 0.0) + numeric_value
    return summary


def summarize_null_logistic_diagnostics(stage_timing_payloads: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    """Aggregate null logistic diagnostics across available binary trials."""
    diagnostics: list[dict[str, typing.Any]] = []
    for stage_timing_payload in stage_timing_payloads:
        raw_diagnostics = stage_timing_payload.get("null_logistic_diagnostics")
        if not isinstance(raw_diagnostics, list):
            continue
        for raw_diagnostic in raw_diagnostics:
            if isinstance(raw_diagnostic, dict):
                diagnostics.append(typing.cast("dict[str, typing.Any]", raw_diagnostic))
    iteration_counts = [
        numeric_diagnostic_value(diagnostic.get("iteration_count", diagnostic.get("null_logistic_iteration_count", 0)))
        for diagnostic in diagnostics
    ]
    firth_iteration_counts = [
        numeric_diagnostic_value(diagnostic.get("firth_iteration_count", 0)) for diagnostic in diagnostics
    ]
    correction_method_counts: dict[str, int] = {}
    convergence_reason_counts: dict[str, int] = {}
    converged_count = 0
    for diagnostic in diagnostics:
        if numeric_diagnostic_value(diagnostic.get("converged", 0)) > 0.0:
            converged_count += 1
        correction_method = diagnostic.get("correction_method")
        if correction_method is not None:
            correction_method_key = str(correction_method)
            correction_method_counts[correction_method_key] = correction_method_counts.get(correction_method_key, 0) + 1
        convergence_reason_code = diagnostic.get("firth_convergence_reason_code")
        if convergence_reason_code is not None:
            convergence_reason_key = str(convergence_reason_code)
            convergence_reason_counts[convergence_reason_key] = (
                convergence_reason_counts.get(convergence_reason_key, 0) + 1
            )
    return {
        "chromosome_count": len(diagnostics),
        "converged_count": converged_count,
        "failed_count": max(len(diagnostics) - converged_count, 0),
        "iteration_counts": summarize_numeric_values(iteration_counts),
        "firth_iteration_counts": summarize_numeric_values(firth_iteration_counts),
        "correction_method_counts": correction_method_counts,
        "firth_convergence_reason_code_counts": convergence_reason_counts,
    }


def summarize_numeric_values(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a numeric vector for JSON output."""
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def summarize_queue_backpressure(stage_timing_payloads: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
    """Aggregate queue/backpressure observations across available trials."""
    summaries: dict[str, dict[str, typing.Any]] = {}
    for stage_timing_payload in stage_timing_payloads:
        raw_queue_backpressure = stage_timing_payload.get("queue_backpressure")
        if not isinstance(raw_queue_backpressure, list):
            continue
        for raw_queue_snapshot in raw_queue_backpressure:
            if not isinstance(raw_queue_snapshot, dict):
                continue
            queue_name = str(raw_queue_snapshot.get("queue_name", ""))
            operation_name = str(raw_queue_snapshot.get("operation_name", ""))
            summary_key = f"{queue_name}:{operation_name}"
            summary = summaries.setdefault(
                summary_key,
                {
                    "queue_name": queue_name,
                    "operation_name": operation_name,
                    "observation_count": 0,
                    "max_depth": 0,
                    "max_capacity": 0,
                    "total_elapsed_seconds": 0.0,
                    "total_blocked_seconds": 0.0,
                },
            )
            summary["observation_count"] = int(summary["observation_count"]) + int(
                numeric_diagnostic_value(raw_queue_snapshot.get("observation_count", 0))
            )
            summary["max_depth"] = max(
                int(summary["max_depth"]),
                int(numeric_diagnostic_value(raw_queue_snapshot.get("max_depth", 0))),
            )
            summary["max_capacity"] = max(
                int(summary["max_capacity"]),
                int(numeric_diagnostic_value(raw_queue_snapshot.get("max_capacity", 0))),
            )
            summary["total_elapsed_seconds"] = float(summary["total_elapsed_seconds"]) + numeric_diagnostic_value(
                raw_queue_snapshot.get("total_elapsed_seconds", 0.0)
            )
            summary["total_blocked_seconds"] = float(summary["total_blocked_seconds"]) + numeric_diagnostic_value(
                raw_queue_snapshot.get("total_blocked_seconds", 0.0)
            )
    rows = list(summaries.values())
    for row in rows:
        row["blocked_fraction"] = safe_ratio(
            float(row["total_blocked_seconds"]),
            float(row["total_elapsed_seconds"]),
        )
    return sorted(rows, key=lambda row: float(row["total_blocked_seconds"]), reverse=True)


def collect_chunk_identities(stage_timing_payload: dict[str, typing.Any]) -> list[dict[str, typing.Any]]:
    """Collect first-seen chunk identities from exact chunk stage timings."""
    raw_chunk_stage_timings = stage_timing_payload.get("chunk_stage_timings")
    if not isinstance(raw_chunk_stage_timings, list):
        return []
    identities: list[dict[str, typing.Any]] = []
    seen_chunk_identifiers: set[str] = set()
    for raw_chunk_stage_timing in raw_chunk_stage_timings:
        if not isinstance(raw_chunk_stage_timing, dict):
            continue
        chunk_identifier = str(raw_chunk_stage_timing.get("chunk_identifier", len(identities)))
        if chunk_identifier in seen_chunk_identifiers:
            continue
        seen_chunk_identifiers.add(chunk_identifier)
        identities.append(
            {
                "chunk_identifier": raw_chunk_stage_timing.get("chunk_identifier"),
                "chromosome": raw_chunk_stage_timing.get("chromosome"),
                "variant_start_index": raw_chunk_stage_timing.get("variant_start_index"),
                "variant_stop_index": raw_chunk_stage_timing.get("variant_stop_index"),
                "variant_count": raw_chunk_stage_timing.get("variant_count"),
            }
        )
    return identities


def build_binary_chunk_outliers(
    available_trials: list[dict[str, typing.Any]],
) -> list[dict[str, typing.Any]]:
    """Build a compact top-N list of per-chunk binary correction outliers."""
    outliers: list[dict[str, typing.Any]] = []
    for available_trial in available_trials:
        trial = typing.cast("BinaryDiagnosticTrialPayload", available_trial["trial"])
        stage_timing_payload = typing.cast("dict[str, typing.Any]", available_trial["payload"])
        binary_chunk_diagnostics = typing.cast(
            "list[dict[str, typing.Any]]",
            available_trial["binary_chunk_diagnostics"],
        )
        chunk_identities = collect_chunk_identities(stage_timing_payload)
        for chunk_index, diagnostics in enumerate(binary_chunk_diagnostics):
            firth_candidate_count = sum_binary_diagnostic_count([diagnostics], "firth_candidate_count")
            firth_failed_count = sum_binary_diagnostic_count([diagnostics], "firth_failed_count")
            score_test_candidate_count = sum_binary_diagnostic_count([diagnostics], "score_test_candidate_count")
            if firth_candidate_count == 0 and score_test_candidate_count == 0:
                continue
            outlier = {
                "trial_name": trial.trial_name,
                "chunk_index": chunk_index,
                "rank_fields": {
                    "firth_candidate_count": firth_candidate_count,
                    "firth_failed_count": firth_failed_count,
                    "firth_iteration_max": numeric_diagnostic_value(diagnostics.get("firth_iteration_max", 0)),
                    "score_test_candidate_count": score_test_candidate_count,
                },
                "diagnostics": {
                    "score_test_candidate_count": score_test_candidate_count,
                    "firth_candidate_count": firth_candidate_count,
                    "firth_converged_count": sum_binary_diagnostic_count([diagnostics], "firth_converged_count"),
                    "firth_failed_count": firth_failed_count,
                    "firth_iteration_min": numeric_diagnostic_value(diagnostics.get("firth_iteration_min", 0)),
                    "firth_iteration_median": numeric_diagnostic_value(diagnostics.get("firth_iteration_median", 0)),
                    "firth_iteration_max": numeric_diagnostic_value(diagnostics.get("firth_iteration_max", 0)),
                    "sparse_correction_count": sum_binary_diagnostic_count([diagnostics], "sparse_correction_count"),
                    "dense_correction_count": sum_binary_diagnostic_count([diagnostics], "dense_correction_count"),
                },
                "chunk_identity": chunk_identities[chunk_index] if chunk_index < len(chunk_identities) else None,
            }
            outliers.append(outlier)
    return sorted(
        outliers,
        key=lambda outlier: (
            int(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["firth_candidate_count"]),
            int(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["firth_failed_count"]),
            float(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["firth_iteration_max"]),
            int(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["score_test_candidate_count"]),
        ),
        reverse=True,
    )[:BINARY_CHUNK_OUTLIER_LIMIT]


def unavailable_binary_correction_diagnostics(
    *,
    aggregate_result: AggregateResult,
    stage_timing_mode: ProfileStageTimingMode,
    reason: str,
    unavailable_trials: list[dict[str, str | None]],
) -> dict[str, typing.Any]:
    """Build an explicit unavailable binary correction diagnostic payload."""
    return {
        "available": False,
        "reason": reason,
        "aggregate_name": aggregate_result.name,
        "trait_type": aggregate_result.trait_type,
        "device": aggregate_result.device,
        "status": aggregate_result.status,
        "stage_timing_mode": stage_timing_mode.value,
        "trial_count": aggregate_result.trial_count,
        "available_trial_count": 0,
        "unavailable_trials": unavailable_trials,
        "chunk_count": None,
        "candidate_counts": {
            "score_test": None,
            "firth": None,
        },
        "correction_outcome_counts": {
            "corrected": None,
            "failed": None,
            "score_test_or_uncorrected": None,
        },
        "failure_code_counts": {
            "none": None,
            "numerical": None,
            "max_iterations": None,
            "invalid_statistic": None,
            "step_halving": None,
        },
        "firth_iteration_counts": {
            "active_chunk_count": None,
            "minimum": None,
            "median_per_chunk_mean": None,
            "maximum": None,
        },
        "correction_branch_counts": {
            "pseudo_firth": None,
            "newton_raphson_zero_start": None,
            "newton_raphson_warm_start": None,
        },
        "correction_attempt_counts": {
            "pseudo_firth": None,
            "newton_raphson_zero_start": None,
            "newton_raphson_warm_start": None,
        },
        "correction_input_counts": {
            "sparse": None,
            "dense": None,
        },
        "fallback_density": {
            "firth_candidates_per_output_row": None,
            "firth_candidates_per_score_test_candidate": None,
        },
        "stage_counts": None,
        "stage_totals_seconds": None,
        "null_logistic": None,
        "queue_backpressure": None,
        "chunk_outliers": [],
    }


def build_binary_correction_diagnostics_for_aggregate(
    *,
    aggregate_result: AggregateResult,
    stage_timing_mode: ProfileStageTimingMode,
) -> dict[str, typing.Any]:
    """Build aggregate binary correction diagnostics for one g binary result."""
    loaded_payloads = [
        load_binary_diagnostic_trial_payload(stage_timing_mode=stage_timing_mode, trial=trial)
        for trial in aggregate_result.trials
        if trial.status == "success"
    ]
    unavailable_trials: list[dict[str, str | None]] = []
    available_trials: list[dict[str, typing.Any]] = []
    for loaded_payload in loaded_payloads:
        if loaded_payload.unavailable_reason is not None:
            unavailable_trials.append(
                {
                    "trial_name": loaded_payload.trial_name,
                    "stage_timing_path": loaded_payload.stage_timing_path,
                    "reason": loaded_payload.unavailable_reason,
                }
            )
            continue
        binary_chunk_diagnostics = extract_binary_chunk_diagnostics(loaded_payload)
        if binary_chunk_diagnostics is None:
            reason = BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_MISSING
            if (
                loaded_payload.payload is not None
                and "binary_chunk_diagnostics" in loaded_payload.payload
                and not isinstance(loaded_payload.payload["binary_chunk_diagnostics"], list)
            ):
                reason = BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_INVALID
            unavailable_trials.append(
                {
                    "trial_name": loaded_payload.trial_name,
                    "stage_timing_path": loaded_payload.stage_timing_path,
                    "reason": reason,
                }
            )
            continue
        available_trials.append(
            {
                "trial": loaded_payload,
                "payload": typing.cast("dict[str, typing.Any]", loaded_payload.payload),
                "binary_chunk_diagnostics": binary_chunk_diagnostics,
            }
        )
    if not available_trials:
        reason = BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING
        if unavailable_trials:
            reason = str(unavailable_trials[0]["reason"])
        return unavailable_binary_correction_diagnostics(
            aggregate_result=aggregate_result,
            stage_timing_mode=stage_timing_mode,
            reason=reason,
            unavailable_trials=unavailable_trials,
        )
    all_binary_chunk_diagnostics: list[dict[str, typing.Any]] = []
    for available_trial in available_trials:
        all_binary_chunk_diagnostics.extend(
            typing.cast("list[dict[str, typing.Any]]", available_trial["binary_chunk_diagnostics"])
        )
    diagnostic_counts = {
        field_name: sum_binary_diagnostic_count(all_binary_chunk_diagnostics, field_name)
        for field_name in BINARY_DIAGNOSTIC_COUNT_FIELDS
    }
    non_none_failure_count = (
        diagnostic_counts["firth_numerical_failure_count"]
        + diagnostic_counts["firth_max_iteration_failure_count"]
        + diagnostic_counts["firth_invalid_statistic_failure_count"]
        + diagnostic_counts["firth_step_halving_failure_count"]
    )
    stage_timing_payloads = [
        typing.cast("dict[str, typing.Any]", available_trial["payload"]) for available_trial in available_trials
    ]
    output_row_count_by_trial = {
        trial.name: trial.output_row_count
        for trial in aggregate_result.trials
        if trial.status == "success" and trial.output_row_count is not None
    }
    available_output_row_count = sum(
        output_row_count_by_trial.get(
            typing.cast("BinaryDiagnosticTrialPayload", available_trial["trial"]).trial_name,
            0,
        )
        or 0
        for available_trial in available_trials
    )
    minimum_iteration_values = active_firth_iteration_values(all_binary_chunk_diagnostics, "firth_iteration_min")
    maximum_iteration_values = active_firth_iteration_values(all_binary_chunk_diagnostics, "firth_iteration_max")
    score_test_candidate_count = diagnostic_counts["score_test_candidate_count"]
    firth_candidate_count = diagnostic_counts["firth_candidate_count"]
    firth_converged_count = diagnostic_counts["firth_converged_count"]
    firth_failed_count = diagnostic_counts["firth_failed_count"]
    return {
        "available": True,
        "reason": None,
        "aggregate_name": aggregate_result.name,
        "trait_type": aggregate_result.trait_type,
        "device": aggregate_result.device,
        "status": aggregate_result.status,
        "stage_timing_mode": stage_timing_mode.value,
        "trial_count": aggregate_result.trial_count,
        "available_trial_count": len(available_trials),
        "unavailable_trials": unavailable_trials,
        "chunk_count": len(all_binary_chunk_diagnostics),
        "candidate_counts": {
            "score_test": score_test_candidate_count,
            "firth": firth_candidate_count,
            "score_test_per_available_trial_mean": score_test_candidate_count / len(available_trials),
            "firth_per_available_trial_mean": firth_candidate_count / len(available_trials),
        },
        "correction_outcome_counts": {
            "corrected": firth_converged_count,
            "failed": firth_failed_count,
            "score_test_or_uncorrected": max(
                score_test_candidate_count - firth_converged_count - firth_failed_count, 0
            ),
        },
        "failure_code_counts": {
            "none": max(firth_candidate_count - non_none_failure_count, 0),
            "numerical": diagnostic_counts["firth_numerical_failure_count"],
            "max_iterations": diagnostic_counts["firth_max_iteration_failure_count"],
            "invalid_statistic": diagnostic_counts["firth_invalid_statistic_failure_count"],
            "step_halving": diagnostic_counts["firth_step_halving_failure_count"],
        },
        "firth_iteration_counts": {
            "active_chunk_count": len(minimum_iteration_values),
            "minimum": min(minimum_iteration_values) if minimum_iteration_values else 0,
            "median_per_chunk_mean": mean_binary_diagnostic_value(
                all_binary_chunk_diagnostics,
                "firth_iteration_median",
            ),
            "maximum": max(maximum_iteration_values) if maximum_iteration_values else 0,
        },
        "correction_branch_counts": {
            "pseudo_firth": diagnostic_counts["pseudo_firth_success_count"],
            "newton_raphson_zero_start": diagnostic_counts["nr_zero_start_success_count"],
            "newton_raphson_warm_start": diagnostic_counts["nr_warm_start_success_count"],
        },
        "correction_attempt_counts": {
            "pseudo_firth": diagnostic_counts["pseudo_firth_attempt_count"],
            "newton_raphson_zero_start": diagnostic_counts["nr_zero_start_attempt_count"],
            "newton_raphson_warm_start": diagnostic_counts["nr_warm_start_attempt_count"],
        },
        "correction_input_counts": {
            "sparse": diagnostic_counts["sparse_correction_count"],
            "dense": diagnostic_counts["dense_correction_count"],
        },
        "fallback_density": {
            "firth_candidates_per_output_row": safe_ratio(
                float(firth_candidate_count), float(available_output_row_count)
            ),
            "firth_candidates_per_score_test_candidate": safe_ratio(
                float(firth_candidate_count),
                float(score_test_candidate_count),
            ),
        },
        "stage_counts": summarize_stage_mapping(stage_timing_payloads, "stage_counts"),
        "stage_totals_seconds": summarize_stage_mapping(stage_timing_payloads, "stage_totals_seconds"),
        "null_logistic": summarize_null_logistic_diagnostics(stage_timing_payloads),
        "queue_backpressure": summarize_queue_backpressure(stage_timing_payloads),
        "chunk_outliers": build_binary_chunk_outliers(available_trials),
    }


def build_binary_correction_diagnostics(
    *,
    headline_results: list[AggregateResult],
    finalist_results_by_key: dict[str, list[AggregateResult]],
    stage_timing_mode: ProfileStageTimingMode,
) -> dict[str, typing.Any]:
    """Build binary correction diagnostics for headline and finalist g runs."""
    headline_diagnostics = {
        aggregate_result.name: build_binary_correction_diagnostics_for_aggregate(
            aggregate_result=aggregate_result,
            stage_timing_mode=stage_timing_mode,
        )
        for aggregate_result in headline_results
        if aggregate_result.implementation == "g" and aggregate_result.trait_type == "binary"
    }
    finalist_diagnostics: dict[str, dict[str, typing.Any]] = {}
    for winner_key, finalist_results in sorted(finalist_results_by_key.items()):
        if not winner_key.startswith("binary_"):
            continue
        finalist_diagnostics[winner_key] = {
            aggregate_result.name: build_binary_correction_diagnostics_for_aggregate(
                aggregate_result=aggregate_result,
                stage_timing_mode=stage_timing_mode,
            )
            for aggregate_result in finalist_results
            if aggregate_result.implementation == "g" and aggregate_result.trait_type == "binary"
        }
    return {
        "stage_timing_mode": stage_timing_mode.value,
        "headline": headline_diagnostics,
        "finalists": finalist_diagnostics,
    }


def build_summary_markdown(
    *,
    aggregate_results: list[AggregateResult],
    comparisons: dict[str, dict[str, float]],
    stage_totals: dict[str, float],
    stage_comparison_rows: list[dict[str, float | str]],
    algorithmic_findings: list[str],
    comparison_notes: RuntimeComparisonNotes | None = None,
    regenie_baseline_scope: RegenieBaselineScope | None = None,
    logging_perturbation_results: list[dict[str, typing.Any]] | None = None,
    binary_correction_diagnostics: dict[str, typing.Any] | None = None,
) -> str:
    """Build the human-readable campaign summary."""
    lines = ["# Landau Deep REGENIE Step 2 Profile", ""]
    lines.append("## Headline Runtimes")
    lines.append("")
    lines.append("| name | trait | device | median s | mean s | min s | max s | std s | rows/s |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in aggregate_results:
        lines.append(
            "| "
            f"{result.name} | {result.trait_type} | {result.device} | "
            f"{format_optional_float(result.median_wall_time_seconds)} | "
            f"{format_optional_float(result.mean_wall_time_seconds)} | "
            f"{format_optional_float(result.min_wall_time_seconds)} | "
            f"{format_optional_float(result.max_wall_time_seconds)} | "
            f"{format_optional_float(result.standard_deviation_seconds)} | "
            f"{format_optional_float(result.rows_per_second)} |"
        )
    lines.extend(["", "## JAX Compile And Cache Diagnostics", ""])
    jax_cache_diagnostics = collect_jax_cache_diagnostics(aggregate_results)
    if jax_cache_diagnostics:
        lines.append(
            "_Cold is the first successful g subprocess for an aggregate; warm summarizes later successful "
            "subprocesses sharing the same persistent cache directory._"
        )
        lines.append("")
        lines.append(
            "| name | persistent cache | cache dir | cold s | warm median s | cold/warm | "
            "cache files Δ cold/warm | cache bytes Δ cold/warm | compiles cold/warm | "
            "persistent hits cold/warm | persistent misses cold/warm | trace misses cold/warm |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result_name, diagnostics in jax_cache_diagnostics.items():
            cache_directory = diagnostics.get("cache_directory")
            lines.append(
                "| "
                f"{result_name} | "
                f"{str(diagnostics['persistent_cache_used']).lower()} | "
                f"`{cache_directory}` | "
                f"{format_optional_float(typing.cast('float | None', diagnostics['cold_wall_time_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', diagnostics['warm_median_wall_time_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', diagnostics['cold_to_warm_speedup_ratio']))} | "
                f"{format_optional_integer(typing.cast('int | None', diagnostics['cold_cache_file_count_delta']))}/"
                f"{format_optional_integer(typing.cast('int | None', diagnostics['warm_cache_file_count_delta']))} | "
                f"{format_optional_integer(typing.cast('int | None', diagnostics['cold_cache_size_bytes_delta']))}/"
                f"{format_optional_integer(typing.cast('int | None', diagnostics['warm_cache_size_bytes_delta']))} | "
                f"{diagnostics['cold_compilation_event_count']}/{diagnostics['warm_compilation_event_count']} | "
                f"{diagnostics['cold_cache_hit_count']}/{diagnostics['warm_cache_hit_count']} | "
                f"{diagnostics['cold_cache_miss_count']}/{diagnostics['warm_cache_miss_count']} | "
                f"{diagnostics['cold_tracing_cache_miss_count']}/{diagnostics['warm_tracing_cache_miss_count']} |"
            )
    else:
        lines.append("- No JAX cache diagnostics were available.")
    lines.extend(["", "## Runtime Comparisons", ""])
    lines.append("### Successful")
    lines.append("")
    if comparisons:
        for comparison_name, comparison in comparisons.items():
            lines.append(
                f"- {comparison_name}: speedup={comparison['speedup_ratio']:.4f}x, "
                f"delta={comparison['absolute_delta_seconds']:.4f}s"
            )
    else:
        lines.append("- No successful direct comparisons were available.")
    comparison_details = comparison_notes or RuntimeComparisonNotes(unsupported=[], failed=[])
    lines.extend(["", "### Unsupported", ""])
    if comparison_details.unsupported:
        for note in comparison_details.unsupported:
            lines.append(f"- {note}")
    else:
        lines.append("- No unsupported direct comparisons were recorded.")
    lines.extend(["", "### Failed", ""])
    if comparison_details.failed:
        for note in comparison_details.failed:
            lines.append(f"- {note}")
    else:
        lines.append("- No failed direct comparisons were recorded.")
    lines.extend(["", "## REGENIE Baseline Scope", ""])
    if regenie_baseline_scope is None:
        lines.append("- Original REGENIE baseline scope was not requested.")
    else:
        lines.append(f"- Status: `{regenie_baseline_scope.status.value}`")
        lines.append(f"- Variant limit: `{regenie_baseline_scope.variant_limit}`")
        if regenie_baseline_scope.extract_path is not None:
            lines.append(f"- Extract list: `{regenie_baseline_scope.extract_path}`")
        if regenie_baseline_scope.metadata_path is not None:
            lines.append(f"- Variant metadata: `{regenie_baseline_scope.metadata_path}`")
        if regenie_baseline_scope.selected_variant_count is not None:
            lines.append(f"- Selected variants: `{regenie_baseline_scope.selected_variant_count}`")
        lines.append(f"- Notes: {regenie_baseline_scope.notes}")
    lines.extend(["", "## Stage Comparisons", ""])
    if stage_comparison_rows:
        lines.append(
            "_Stage timings are inclusive cumulative seconds from each profiler, so nested stages can overlap and "
            "can exceed headline wall time._"
        )
        lines.append("")
        lines.append("| trait | g device | stage | REGENIE s | g s | g speedup |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in stage_comparison_rows:
            lines.append(
                "| "
                f"{row['trait_type']} | {row['g_device']} | {row['stage_group']} | "
                f"{float(row['regenie_seconds']):.6f} | "
                f"{float(row['g_seconds']):.6f} | "
                f"{float(row['g_speedup_ratio']):.4f}x |"
            )
    else:
        lines.append("- No paired stage timing JSON files were available.")
    lines.extend(["", "## Algorithmic Findings", ""])
    if algorithmic_findings:
        for finding in algorithmic_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- Re-run with successful REGENIE and g profile JSON files to generate source-level findings.")
    append_binary_correction_diagnostics_markdown(lines, binary_correction_diagnostics or {})
    lines.extend(["", "## Logging And Telemetry Perturbation", ""])
    logging_rows = build_logging_perturbation_rows(logging_perturbation_results or [])
    if logging_rows:
        lines.append("| winner | case | wall s | delta vs off s | ratio vs off | status |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in logging_rows:
            lines.append(
                "| "
                f"{row['winner_key']} | {row['case_name']} | "
                f"{format_optional_float(typing.cast('float | None', row['wall_time_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', row['delta_vs_off_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', row['ratio_vs_off']))} | "
                f"{row['status']} |"
            )
    else:
        lines.append("- No logging perturbation results were available.")
    lines.extend(["", "## Ranked Bottlenecks", ""])
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)[:20]:
            lines.append(f"- {stage_name}: {seconds:.6f}s")
    else:
        lines.append("- No stage timing JSON files were available.")
    lines.extend(["", "## Next Optimization Targets", ""])
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)[:5]:
            lines.append(f"- Reduce `{stage_name}` first; it is one of the largest measured wall-time shares.")
    else:
        lines.append("- Re-run with successful g diagnostic trials to rank measured stage shares.")
    return "\n".join(lines) + "\n"


def diagnostic_mapping(raw_value: typing.Any) -> dict[str, typing.Any]:
    """Return a diagnostic mapping or an empty mapping."""
    if isinstance(raw_value, dict):
        return typing.cast("dict[str, typing.Any]", raw_value)
    return {}


def binary_diagnostic_markdown_rows(
    binary_correction_diagnostics: dict[str, typing.Any],
    group_name: str,
) -> list[dict[str, typing.Any]]:
    """Flatten headline or finalist diagnostic payloads into Markdown rows."""
    rows: list[dict[str, typing.Any]] = []
    raw_group = binary_correction_diagnostics.get(group_name)
    if group_name == "headline":
        group_payload = diagnostic_mapping(raw_group)
        for aggregate_name, raw_diagnostics in sorted(group_payload.items()):
            diagnostics = diagnostic_mapping(raw_diagnostics)
            if diagnostics:
                row = dict(diagnostics)
                row["display_name"] = aggregate_name
                rows.append(row)
        return rows
    nested_payload = diagnostic_mapping(raw_group)
    for winner_key, raw_aggregate_payload in sorted(nested_payload.items()):
        aggregate_payload = diagnostic_mapping(raw_aggregate_payload)
        for aggregate_name, raw_diagnostics in sorted(aggregate_payload.items()):
            diagnostics = diagnostic_mapping(raw_diagnostics)
            if diagnostics:
                row = dict(diagnostics)
                row["display_name"] = f"{winner_key}/{aggregate_name}"
                rows.append(row)
    return rows


def format_diagnostic_integer(raw_value: typing.Any) -> str:
    """Format an optional diagnostic integer."""
    numeric_value = optional_numeric_value(raw_value)
    if numeric_value is None:
        return ""
    return str(int(numeric_value))


def format_diagnostic_ratio(raw_value: typing.Any) -> str:
    """Format an optional diagnostic ratio as a percentage."""
    numeric_value = optional_numeric_value(raw_value)
    if numeric_value is None:
        return ""
    return f"{numeric_value * 100.0:.2f}%"


def format_binary_diagnostic_status(diagnostics: dict[str, typing.Any]) -> str:
    """Format binary diagnostic availability for Markdown."""
    if diagnostics.get("available") is True:
        return "available"
    reason = str(diagnostics.get("reason", "unavailable"))
    if reason == BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED:
        return "unavailable: stage timing mode off"
    return f"unavailable: {reason}"


def format_binary_failure_counts(failure_counts: dict[str, typing.Any]) -> str:
    """Format compact failure-code counts."""
    if not failure_counts:
        return ""
    formatted_counts = []
    for field_name in ("numerical", "max_iterations", "invalid_statistic", "step_halving"):
        numeric_value = optional_numeric_value(failure_counts.get(field_name))
        if numeric_value is not None and numeric_value > 0.0:
            formatted_counts.append(f"{field_name}={int(numeric_value)}")
    if formatted_counts:
        return ", ".join(formatted_counts)
    none_count = optional_numeric_value(failure_counts.get("none"))
    if none_count is None:
        return ""
    return f"none={int(none_count)}"


def format_binary_branch_mix(branch_counts: dict[str, typing.Any]) -> str:
    """Format compact Firth correction branch counts."""
    if not branch_counts:
        return ""
    return (
        "pseudo="
        f"{format_diagnostic_integer(branch_counts.get('pseudo_firth'))}, "
        "zero="
        f"{format_diagnostic_integer(branch_counts.get('newton_raphson_zero_start'))}, "
        "warm="
        f"{format_diagnostic_integer(branch_counts.get('newton_raphson_warm_start'))}"
    )


def format_binary_firth_iterations(iteration_counts: dict[str, typing.Any]) -> str:
    """Format compact Firth iteration summary."""
    if not iteration_counts:
        return ""
    minimum = optional_numeric_value(iteration_counts.get("minimum"))
    median_mean = optional_numeric_value(iteration_counts.get("median_per_chunk_mean"))
    maximum = optional_numeric_value(iteration_counts.get("maximum"))
    if minimum is None or median_mean is None or maximum is None:
        return ""
    return f"{minimum:.0f}/{median_mean:.1f}/{maximum:.0f}"


def append_binary_diagnostic_table(
    lines: list[str],
    *,
    title: str,
    rows: list[dict[str, typing.Any]],
) -> None:
    """Append one compact binary diagnostic Markdown table."""
    lines.extend(["", f"### {title}", ""])
    if not rows:
        lines.append("- No binary correction diagnostics were available.")
        return
    lines.append(
        "| run | device | status | trials | chunks | score cand | Firth cand | corrected/failed | failures | "
        "iters min/mean/max | branch mix | sparse/dense | Firth density |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |")
    for diagnostics in rows:
        candidate_counts = diagnostic_mapping(diagnostics.get("candidate_counts"))
        outcome_counts = diagnostic_mapping(diagnostics.get("correction_outcome_counts"))
        failure_counts = diagnostic_mapping(diagnostics.get("failure_code_counts"))
        iteration_counts = diagnostic_mapping(diagnostics.get("firth_iteration_counts"))
        branch_counts = diagnostic_mapping(diagnostics.get("correction_branch_counts"))
        input_counts = diagnostic_mapping(diagnostics.get("correction_input_counts"))
        fallback_density = diagnostic_mapping(diagnostics.get("fallback_density"))
        lines.append(
            "| "
            f"{diagnostics.get('display_name', diagnostics.get('aggregate_name', ''))} | "
            f"{diagnostics.get('device', '')} | "
            f"{format_binary_diagnostic_status(diagnostics)} | "
            f"{format_diagnostic_integer(diagnostics.get('available_trial_count'))} | "
            f"{format_diagnostic_integer(diagnostics.get('chunk_count'))} | "
            f"{format_diagnostic_integer(candidate_counts.get('score_test'))} | "
            f"{format_diagnostic_integer(candidate_counts.get('firth'))} | "
            f"{format_diagnostic_integer(outcome_counts.get('corrected'))}/"
            f"{format_diagnostic_integer(outcome_counts.get('failed'))} | "
            f"{format_binary_failure_counts(failure_counts)} | "
            f"{format_binary_firth_iterations(iteration_counts)} | "
            f"{format_binary_branch_mix(branch_counts)} | "
            f"{format_diagnostic_integer(input_counts.get('sparse'))}/"
            f"{format_diagnostic_integer(input_counts.get('dense'))} | "
            f"{format_diagnostic_ratio(fallback_density.get('firth_candidates_per_output_row'))} |"
        )


def append_binary_correction_diagnostics_markdown(
    lines: list[str],
    binary_correction_diagnostics: dict[str, typing.Any],
) -> None:
    """Append compact binary correction diagnostics to the summary report."""
    lines.extend(["", "## Binary Correction Diagnostics", ""])
    if not binary_correction_diagnostics:
        lines.append("- No binary correction diagnostics were computed.")
        return
    stage_timing_mode = str(binary_correction_diagnostics.get("stage_timing_mode", "unknown"))
    lines.append(
        "_Exact stage timing JSON is required for correction diagnostics; bounded per-chunk outliers remain in "
        "`summary.json` and raw stage timing artifacts._"
    )
    if stage_timing_mode == ProfileStageTimingMode.OFF.value:
        lines.append("- Diagnostics are unavailable because `telemetry.stage_timing_mode=off`.")
    append_binary_diagnostic_table(
        lines,
        title="Headline Winners",
        rows=binary_diagnostic_markdown_rows(binary_correction_diagnostics, "headline"),
    )
    append_binary_diagnostic_table(
        lines,
        title="Finalists",
        rows=binary_diagnostic_markdown_rows(binary_correction_diagnostics, "finalists"),
    )


def build_logging_perturbation_rows(
    logging_perturbation_results: list[dict[str, typing.Any]],
) -> list[dict[str, float | str | None]]:
    """Build comparable telemetry/logging perturbation rows."""
    baseline_times: dict[str, float] = {}
    for result in logging_perturbation_results:
        winner_key = str(result["winner_key"])
        case_payload = typing.cast("dict[str, typing.Any]", result["case"])
        trial_payload = typing.cast("dict[str, typing.Any]", result["trial"])
        wall_time = trial_payload.get("wall_time_seconds")
        if case_payload.get("name") == "telemetry_off" and isinstance(wall_time, (int, float)):
            baseline_times[winner_key] = float(wall_time)
    rows: list[dict[str, float | str | None]] = []
    for result in logging_perturbation_results:
        winner_key = str(result["winner_key"])
        case_payload = typing.cast("dict[str, typing.Any]", result["case"])
        trial_payload = typing.cast("dict[str, typing.Any]", result["trial"])
        wall_time_value = trial_payload.get("wall_time_seconds")
        wall_time = float(wall_time_value) if isinstance(wall_time_value, (int, float)) else None
        baseline_time = baseline_times.get(winner_key)
        delta_vs_off = None
        ratio_vs_off = None
        if wall_time is not None and baseline_time is not None and baseline_time > 0.0:
            delta_vs_off = wall_time - baseline_time
            ratio_vs_off = wall_time / baseline_time
        rows.append(
            {
                "winner_key": winner_key,
                "case_name": str(case_payload.get("name", "")),
                "wall_time_seconds": wall_time,
                "delta_vs_off_seconds": delta_vs_off,
                "ratio_vs_off": ratio_vs_off,
                "status": str(trial_payload.get("status", "")),
            }
        )
    return rows


def format_optional_float(value: float | None) -> str:
    """Format optional floats for markdown tables."""
    if value is None:
        return ""
    return f"{value:.6f}"


def format_optional_integer(value: int | None) -> str:
    """Format optional integers for markdown tables."""
    if value is None:
        return ""
    return str(value)


def run_deep_profiles(
    *,
    arguments: ProfileArguments,
    baseline_paths: typing.Any,
    winners: dict[str, AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> dict[str, typing.Any]:
    """Run optional profiler commands for representative g winners."""
    profile_directory = output_directory / "deep_profiles"
    profile_directory.mkdir(parents=True, exist_ok=True)
    profiler_tool_status = build_profiler_tool_status(arguments)
    emit_stage_timings = should_emit_stage_timings(arguments)
    results: dict[str, typing.Any] = {
        "profiler_tools": serialize_profiler_tool_status(profiler_tool_status),
        "rust_criterion": [],
        "sampling_profiles": [],
    }
    if arguments.enable_rust_criterion and profiler_tool_status["rust_criterion"].available:
        for benchmark_name in parse_string_list(arguments.rust_benchmarks):
            logger.info("Running Rust Criterion benchmark %s", benchmark_name)
            results["rust_criterion"].append(
                command_output(
                    ["cargo", "bench", "--bench", benchmark_name],
                    environment_overrides={"RUSTFLAGS": "-C target-cpu=native"},
                )
            )
    elif arguments.enable_rust_criterion:
        results["rust_criterion"].append(dataclasses.asdict(profiler_tool_status["rust_criterion"]))
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = candidate_from_aggregate_name(winner_key, winner)
        if arguments.enable_jax_trace or arguments.enable_jax_memory_profile:
            trace_directory = profile_directory / f"{winner_key}_jax_trace" if arguments.enable_jax_trace else None
            memory_profile_path = (
                profile_directory / f"{winner_key}_device_memory.prof" if arguments.enable_jax_memory_profile else None
            )
            profiler_artifact_path = trace_directory if trace_directory is not None else memory_profile_path
            logger.info("Running JAX profiler capture for %s", winner_key)
            profile_result = run_g_trial(
                name=f"profile_{winner_key}_jax",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=profile_directory,
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
                trace_directory=trace_directory,
                memory_profile_path=memory_profile_path,
            )
            results["sampling_profiles"].append(
                dataclasses.asdict(
                    dataclasses.replace(
                        profile_result,
                        profiler_artifact_path=(
                            str(profiler_artifact_path) if profiler_artifact_path is not None else None
                        ),
                    )
                )
            )
        if arguments.enable_python_cprofile:
            cprofile_output_path = profile_directory / f"{winner_key}.cprofile"
            cprofile_text_path = profile_directory / f"{winner_key}.cprofile.txt"
            cprofile_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_cprofile",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            cprofile_result = attach_deep_profiler_metadata(
                result=run_logged_command(
                    name=f"profile_{winner_key}_cprofile",
                    implementation="cProfile",
                    trait_type=candidate.trait_type,
                    device=candidate.device,
                    command_arguments=[
                        sys.executable,
                        "-m",
                        "cProfile",
                        "-o",
                        str(cprofile_output_path),
                        str(cprofile_child_command.run_paths.profile_script_path),
                    ],
                    environment_overrides=cprofile_child_command.environment_overrides,
                    log_directory=output_directory / "logs",
                ),
                run_paths=cprofile_child_command.run_paths,
                profiler_artifact_path=cprofile_output_path,
            )
            results["sampling_profiles"].append(dataclasses.asdict(cprofile_result))
            if cprofile_result.status == "success":
                cprofile_text_result = command_output(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import pstats, sys; "
                            "pstats.Stats(sys.argv[1]).strip_dirs().sort_stats('cumtime').print_stats(80)"
                        ),
                        str(cprofile_output_path),
                    ]
                )
                cprofile_text_path.write_text(cprofile_text_result["stdout"], encoding="utf-8")
        py_spy_status = profiler_tool_status["py_spy"]
        if arguments.enable_py_spy and py_spy_status.available:
            speedscope_path = profile_directory / f"{winner_key}.speedscope.json"
            py_spy_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_py_spy",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            command_arguments = [
                py_spy_status.executable_path or "py-spy",
                "record",
                "--format",
                "speedscope",
                "--output",
                str(speedscope_path),
                "--",
                *py_spy_child_command.command_arguments,
            ]
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_py_spy",
                implementation="py-spy",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=command_arguments,
                environment_overrides=py_spy_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=py_spy_child_command.run_paths,
                profiler_artifact_path=speedscope_path,
                timeout_seconds=arguments.py_spy_timeout_seconds,
            )
        elif arguments.enable_py_spy:
            append_skipped_executable_profile(
                results=results,
                tool_status=py_spy_status,
                name=f"profile_{winner_key}_py_spy",
                implementation="py-spy",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        scalene_status = profiler_tool_status["scalene"]
        if arguments.enable_scalene and scalene_status.available:
            scalene_json_path = profile_directory / f"{winner_key}.scalene.json"
            scalene_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_scalene",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_scalene",
                implementation="Scalene",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=build_scalene_command_arguments(
                    tool_status=scalene_status,
                    output_path=scalene_json_path,
                    profile_script_path=scalene_child_command.run_paths.profile_script_path,
                ),
                environment_overrides=scalene_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=scalene_child_command.run_paths,
                profiler_artifact_path=scalene_json_path,
                timeout_seconds=arguments.scalene_timeout_seconds,
            )
        elif arguments.enable_scalene:
            append_skipped_executable_profile(
                results=results,
                tool_status=scalene_status,
                name=f"profile_{winner_key}_scalene",
                implementation="Scalene",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        memray_status = profiler_tool_status["memray"]
        if arguments.enable_memray and memray_status.available:
            memray_output_path = profile_directory / f"{winner_key}.memray.bin"
            memray_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_memray",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_memray",
                implementation="Memray",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=build_memray_command_arguments(
                    tool_status=memray_status,
                    output_path=memray_output_path,
                    profile_script_path=memray_child_command.run_paths.profile_script_path,
                ),
                environment_overrides=memray_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=memray_child_command.run_paths,
                profiler_artifact_path=memray_output_path,
                timeout_seconds=arguments.memray_timeout_seconds,
            )
        elif arguments.enable_memray:
            append_skipped_executable_profile(
                results=results,
                tool_status=memray_status,
                name=f"profile_{winner_key}_memray",
                implementation="Memray",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        nsight_systems_status = profiler_tool_status["nsight_systems"]
        if arguments.enable_nsight_systems and nsight_systems_status.available:
            nsight_report_prefix = profile_directory / f"{winner_key}_nsys"
            nsight_systems_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_nsys",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_nsys",
                implementation="Nsight Systems",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=[
                    nsight_systems_status.executable_path or "nsys",
                    "profile",
                    "--trace=cuda,cudnn,cublas,osrt,nvtx",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--stats=true",
                    "--force-overwrite=true",
                    "--output",
                    str(nsight_report_prefix),
                    *nsight_systems_child_command.command_arguments,
                ],
                environment_overrides=nsight_systems_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=nsight_systems_child_command.run_paths,
                profiler_artifact_path=nsight_report_prefix,
                timeout_seconds=arguments.nsight_systems_timeout_seconds,
            )
        elif arguments.enable_nsight_systems:
            append_skipped_executable_profile(
                results=results,
                tool_status=nsight_systems_status,
                name=f"profile_{winner_key}_nsys",
                implementation="Nsight Systems",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        nsight_compute_status = profiler_tool_status["nsight_compute"]
        if arguments.enable_nsight_compute and nsight_compute_status.available:
            nsight_compute_report_path = profile_directory / f"{winner_key}_ncu"
            nsight_compute_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_ncu",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_ncu",
                implementation="Nsight Compute",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=[
                    nsight_compute_status.executable_path or "ncu",
                    "--target-processes",
                    "all",
                    "--set",
                    "default",
                    "--export",
                    str(nsight_compute_report_path),
                    *nsight_compute_child_command.command_arguments,
                ],
                environment_overrides=nsight_compute_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=nsight_compute_child_command.run_paths,
                profiler_artifact_path=nsight_compute_report_path,
                timeout_seconds=arguments.nsight_compute_timeout_seconds,
            )
        elif arguments.enable_nsight_compute:
            append_skipped_executable_profile(
                results=results,
                tool_status=nsight_compute_status,
                name=f"profile_{winner_key}_ncu",
                implementation="Nsight Compute",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
        perf_status = profiler_tool_status["linux_perf"]
        if arguments.enable_linux_perf and perf_status.available:
            perf_path = profile_directory / f"{winner_key}.perf.data"
            perf_child_command = build_deep_profiler_child_command(
                profile_directory=profile_directory,
                profile_name=f"profile_{winner_key}_perf",
                baseline_paths=baseline_paths,
                candidate=candidate,
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
            )
            command_arguments = [
                perf_status.executable_path or "perf",
                "record",
                "-g",
                "-o",
                str(perf_path),
                "--",
                *perf_child_command.command_arguments,
            ]
            append_logged_profile_result(
                results=results,
                name=f"profile_{winner_key}_perf",
                implementation="perf",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=command_arguments,
                environment_overrides=perf_child_command.environment_overrides,
                log_directory=output_directory / "logs",
                run_paths=perf_child_command.run_paths,
                profiler_artifact_path=perf_path,
                timeout_seconds=arguments.linux_perf_timeout_seconds,
            )
        elif arguments.enable_linux_perf:
            append_skipped_executable_profile(
                results=results,
                tool_status=perf_status,
                name=f"profile_{winner_key}_perf",
                implementation="perf",
                trait_type=candidate.trait_type,
                device=candidate.device,
                log_directory=output_directory / "logs",
            )
    return results


def run_logging_perturbation_profiles(
    *,
    arguments: ProfileArguments,
    baseline_paths: typing.Any,
    winners: dict[str, AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> list[dict[str, typing.Any]]:
    """Run representative winners under telemetry/logging perturbation cases."""
    if not arguments.enable_logging_perturbation:
        return []
    perturbation_directory = output_directory / "logging_perturbation"
    perturbation_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, typing.Any]] = []
    emit_stage_timings = should_emit_stage_timings(arguments)
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = candidate_from_aggregate_name(winner_key, winner)
        for perturbation_case in build_logging_perturbation_cases(
            output_directory=output_directory,
            smoke=arguments.smoke,
        ):
            diagnostic_options = dict(perturbation_case.diagnostic_options)
            if diagnostic_options.get("telemetry") != "off":
                diagnostic_options["log_dir"] = str(
                    perturbation_directory / f"{winner_key}_{perturbation_case.name}_logs"
                )
            trial_result = run_g_trial(
                name=f"logging_{winner_key}_{perturbation_case.name}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=perturbation_directory,
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                emit_stage_timings=emit_stage_timings,
                diagnostic_options=diagnostic_options,
            )
            results.append(
                {
                    "winner_key": winner_key,
                    "case": {
                        "name": perturbation_case.name,
                        "diagnostic_options": diagnostic_options,
                    },
                    "trial": dataclasses.asdict(trial_result),
                }
            )
    (perturbation_directory / "logging_perturbation.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def required_profile_input_paths(baseline_paths: baseline_benchmark.BaselinePaths) -> list[Path]:
    """Return input and prediction paths used by real profile runs."""
    return [
        baseline_paths.bed_prefix.with_suffix(".bed"),
        baseline_paths.bed_prefix.with_suffix(".bim"),
        baseline_paths.bed_prefix.with_suffix(".fam"),
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.continuous_phenotype_path,
        baseline_paths.binary_phenotype_path,
        baseline_paths.covariate_path,
        baseline_paths.regenie_prediction_list_path,
        typing.cast("Path", baseline_paths.regenie_qt_prediction_list_path),
    ]


def build_profile_plan(
    *,
    arguments: ProfileArguments,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_directory: Path,
    campaign_budget: CampaignBudget,
) -> ProfilePlan:
    """Build a dry-run plan for the full profile campaign."""
    rust_benchmark_commands: list[list[str]] = []
    if arguments.enable_rust_criterion and not arguments.skip_deep_profiles:
        rust_benchmark_commands = [
            ["cargo", "bench", "--bench", benchmark_name]
            for benchmark_name in parse_string_list(arguments.rust_benchmarks)
        ]
    profiler_modes = {
        "regenie_baseline": arguments.include_regenie_baseline,
        "jax_trace": arguments.enable_jax_trace,
        "jax_memory_profile": arguments.enable_jax_memory_profile,
        "python_cprofile": arguments.enable_python_cprofile,
        "py_spy": arguments.enable_py_spy,
        "scalene": arguments.enable_scalene,
        "memray": arguments.enable_memray,
        "linux_perf": arguments.enable_linux_perf,
        "nsight_systems": arguments.enable_nsight_systems,
        "nsight_compute": arguments.enable_nsight_compute,
        "rust_criterion": arguments.enable_rust_criterion and not arguments.skip_deep_profiles,
        "logging_perturbation": arguments.enable_logging_perturbation,
    }
    profiler_tools = serialize_profiler_tool_status(build_profiler_tool_status(arguments))
    logging_perturbation_cases = []
    if arguments.enable_logging_perturbation:
        logging_perturbation_cases = [
            dataclasses.asdict(perturbation_case)
            for perturbation_case in build_logging_perturbation_cases(
                output_directory=output_directory,
                smoke=arguments.smoke,
            )
        ]
    regenie_baseline_scope = None
    regenie_baseline_commands: list[dict[str, object]] = []
    if arguments.include_regenie_baseline:
        scope = build_regenie_baseline_scope(
            arguments=arguments,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
        )
        regenie_baseline_scope = serialize_regenie_baseline_scope(scope)
        if scope.status != RegenieBaselineScopeStatus.UNSUPPORTED:
            regenie_executable = configured_regenie_executable(arguments)
            for trait_type in selected_regenie_baseline_trait_types(arguments):
                command_arguments = build_regenie_step2_command(
                    trait_type=trait_type,
                    regenie_executable=regenie_executable,
                    baseline_paths=baseline_paths,
                    output_prefix=output_directory / "headline_runs" / f"headline_regenie_{trait_type}_trial00",
                    baseline_scope=scope,
                )
                regenie_baseline_commands.append(
                    build_command_manifest(
                        command_name=f"headline_regenie_{trait_type}_trial00",
                        status="planned",
                        command_arguments=command_arguments,
                    )
                )
    notes = [
        "Dry run only: no workloads, profilers, or setup commands were executed.",
        "Real runs generate summary.json, summary.md, preflight.json, subprocess logs, stage timings, "
        "deep_profiles artifacts, logging perturbation results, and artifact_manifest.json.",
    ]
    if arguments.skip_deep_profiles:
        notes.append("Deep profiler captures are disabled by tool.skip_deep_profiles=true.")
    if not should_emit_stage_timings(arguments):
        notes.append("Exact stage timing diagnostics are disabled by telemetry.stage_timing_mode=off.")
    if not arguments.include_regenie_baseline:
        notes.append("Original REGENIE headline trials are disabled by tool.include_regenie_baseline=false.")
    elif regenie_baseline_scope is not None:
        notes.append(str(regenie_baseline_scope["notes"]))
    return ProfilePlan(
        chromosome_label=arguments.chromosome_label,
        output_directory=str(output_directory),
        required_inputs=[str(path) for path in required_profile_input_paths(baseline_paths)],
        workload_keys=list(campaign_budget.workload_keys),
        campaign_budget=campaign_budget,
        profiler_modes=profiler_modes,
        profiler_tools=profiler_tools,
        logging_perturbation_cases=logging_perturbation_cases,
        regenie_baseline_scope=regenie_baseline_scope,
        regenie_baseline_commands=regenie_baseline_commands,
        rust_benchmark_commands=rust_benchmark_commands,
        notes=notes,
    )


def write_profile_plan(plan: ProfilePlan, output_directory: Path) -> None:
    """Persist dry-run profile plan artifacts."""
    (output_directory / "profile_plan.json").write_text(
        json.dumps(dataclasses.asdict(plan), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Full App Profile Plan",
        "",
        f"- Chromosome: `{plan.chromosome_label}`",
        f"- Output directory: `{plan.output_directory}`",
        f"- Workloads: `{', '.join(plan.workload_keys)}`",
        "",
        "## Campaign Budget",
        "",
        f"- Total candidates/cases: `{plan.campaign_budget.total_candidate_count}`",
        f"- Estimated subprocess runs: `{plan.campaign_budget.total_subprocess_run_count}`",
        f"- Estimated major profiler runs: `{plan.campaign_budget.total_major_profiler_run_count}`",
        f"- Max subprocess runs: `{plan.campaign_budget.max_subprocess_runs}`",
        f"- Max major profiler runs: `{plan.campaign_budget.max_major_profiler_runs}`",
        f"- Over subprocess budget: `{str(plan.campaign_budget.over_subprocess_budget).lower()}`",
        f"- Over major profiler budget: `{str(plan.campaign_budget.over_major_profiler_budget).lower()}`",
        "",
        "| section | candidates/cases | subprocess runs | major profiler runs | notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for section in plan.campaign_budget.sections:
        lines.append(
            "| "
            f"{section.display_name} | "
            f"{section.candidate_count} | "
            f"{section.subprocess_run_count} | "
            f"{section.major_profiler_run_count} | "
            f"{section.notes} |"
        )
    lines.extend(["", "### Budget Guidance", ""])
    for guidance_item in plan.campaign_budget.guidance:
        lines.append(f"- {guidance_item}")
    lines.extend(["", "## Profiler Modes", ""])
    for mode_name, enabled in plan.profiler_modes.items():
        lines.append(f"- `{mode_name}`: `{str(enabled).lower()}`")
    lines.extend(["", "## Profiler Tool Availability", ""])
    for tool_name, tool_status in plan.profiler_tools.items():
        available = str(tool_status["available"]).lower()
        enabled = str(tool_status["enabled"]).lower()
        notes = str(tool_status["notes"])
        lines.append(f"- `{tool_name}`: enabled=`{enabled}`, available=`{available}`; {notes}")
    lines.extend(["", "## Logging Perturbation Cases", ""])
    if plan.logging_perturbation_cases:
        for perturbation_case in plan.logging_perturbation_cases:
            lines.append(f"- `{perturbation_case['name']}`: `{perturbation_case['diagnostic_options']}`")
    else:
        lines.append("- Logging perturbation profiling is disabled.")
    lines.extend(["", "## Inputs And Step 1 Prediction Lists", ""])
    for input_path in plan.required_inputs:
        lines.append(f"- `{input_path}`")
    lines.extend(["", "## REGENIE Baseline Scope", ""])
    if plan.regenie_baseline_scope is None:
        lines.append("- Original REGENIE baseline profiling is disabled.")
    else:
        for key, value in plan.regenie_baseline_scope.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## REGENIE Baseline Commands", ""])
    if plan.regenie_baseline_commands:
        for command_manifest in plan.regenie_baseline_commands:
            command_arguments = typing.cast("list[str]", command_manifest["command_arguments"])
            lines.append(f"- `{command_manifest['name']}`: `{shlex.join(command_arguments)}`")
    else:
        lines.append("- No REGENIE baseline commands are planned.")
    lines.extend(["", "## Rust Benchmark Commands", ""])
    if plan.rust_benchmark_commands:
        for command_arguments in plan.rust_benchmark_commands:
            lines.append(f"- `{shlex.join(command_arguments)}`")
    else:
        lines.append("- Rust Criterion profiling is disabled.")
    lines.extend(["", "## Notes", ""])
    for note in plan.notes:
        lines.append(f"- {note}")
    (output_directory / "profile_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tool(arguments: ProfileArguments, hydra_config: omegaconf.DictConfig | None = None) -> None:
    """Run the landau deep profiling campaign."""
    arguments = apply_smoke_overrides(arguments)
    output_directory = build_output_directory(arguments)
    output_directory.mkdir(parents=True, exist_ok=True)
    log_directory = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = output_directory / "jax_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    tooling_logging.configure_tool_logging(output_directory / "tooling.log")

    logger.info("Starting %s deep profile campaign", arguments.chromosome_label)
    logger.info("Writing profile artifacts under %s", output_directory)
    baseline_paths = build_baseline_paths(arguments)
    profiler_tool_status = build_profiler_tool_status(arguments)
    regenie_baseline_scope = build_regenie_baseline_scope(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=output_directory,
    )
    campaign_budget = build_campaign_budget(arguments=arguments, output_directory=output_directory)
    log_campaign_budget(campaign_budget)
    if arguments.dry_run:
        profile_plan = build_profile_plan(
            arguments=arguments,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            campaign_budget=campaign_budget,
        )
        write_profile_plan(profile_plan, output_directory)
        write_artifact_manifest(
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            profile_plan=profile_plan,
        )
        write_standard_profile_artifacts(
            arguments=arguments,
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            status=tooling_artifact_format.ToolArtifactStatus.DRY_RUN,
            profile_plan=profile_plan,
            summary_markdown=(output_directory / "profile_plan.md").read_text(encoding="utf-8"),
            hydra_config=hydra_config,
        )
        logger.info("Wrote dry-run profile plan under %s", output_directory)
        return
    if campaign_budget_is_over_limit(campaign_budget) and not arguments.allow_over_budget:
        profile_plan = build_profile_plan(
            arguments=arguments,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            campaign_budget=campaign_budget,
        )
        write_profile_plan(profile_plan, output_directory)
        write_artifact_manifest(
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            profile_plan=profile_plan,
        )
        write_standard_profile_artifacts(
            arguments=arguments,
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            status=tooling_artifact_format.ToolArtifactStatus.INVALID,
            status_reason="Campaign budget exceeds configured limits.",
            profile_plan=profile_plan,
            summary_markdown=(output_directory / "profile_plan.md").read_text(encoding="utf-8"),
            hydra_config=hydra_config,
        )
    enforce_campaign_budget(arguments, campaign_budget)
    logger.info("Validating profile inputs")
    baseline_benchmark.validate_input_files(baseline_paths)
    prediction_list_paths = [
        baseline_paths.regenie_prediction_list_path,
        baseline_paths.regenie_qt_prediction_list_path,
    ]
    missing_prediction_list_paths = [path for path in prediction_list_paths if path is not None and not path.exists()]
    regenie_executable: str | None = None
    setup_results: list[TrialResult] = []
    if arguments.include_regenie_baseline:
        regenie_executable = resolve_available_regenie_executable(arguments)
        if regenie_executable is not None:
            logger.info("Ensuring REGENIE step 1 prediction lists")
            setup_results = ensure_prediction_lists(
                baseline_paths=baseline_paths,
                regenie_executable=regenie_executable,
                log_directory=log_directory,
            )
        elif missing_prediction_list_paths:
            formatted_paths = "\n".join(str(path) for path in missing_prediction_list_paths)
            message = (
                "Step 1 prediction lists are required before g runs and REGENIE setup cannot run because "
                f"{configured_regenie_executable(arguments)!r} is unavailable:\n{formatted_paths}"
            )
            raise FileNotFoundError(message)
        else:
            logger.warning("Original REGENIE baseline requested, but executable is unavailable")
    elif missing_prediction_list_paths:
        formatted_paths = "\n".join(str(path) for path in missing_prediction_list_paths)
        message = f"Step 1 prediction lists are required when REGENIE setup is disabled:\n{formatted_paths}"
        raise FileNotFoundError(message)
    else:
        logger.info("Using existing REGENIE step 1 prediction lists")
    logger.info("Collecting preflight metadata")
    preflight_metadata = collect_environment_metadata(baseline_paths, regenie_executable)
    (output_directory / "preflight.json").write_text(json.dumps(preflight_metadata, indent=2) + "\n", encoding="utf-8")

    logger.info("Running BGEN reader pre-sweep")
    bgen_summaries = run_bgen_sweep(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=output_directory,
    )
    logger.info("Running candidate tuning")
    tuning_results = run_candidate_tuning(
        arguments=arguments,
        baseline_paths=baseline_paths,
        bgen_summaries=bgen_summaries,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    winners = tuning_results.winners
    logger.info("Running headline trials")
    headline_results = run_headline_trials(
        arguments=arguments,
        baseline_paths=baseline_paths,
        regenie_executable=regenie_executable,
        regenie_baseline_scope=regenie_baseline_scope,
        winners=winners,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    deep_profile_results: dict[str, typing.Any] = {}
    if not arguments.skip_deep_profiles:
        logger.info("Running full profiler bundle")
        deep_profile_results = run_deep_profiles(
            arguments=arguments,
            baseline_paths=baseline_paths,
            winners=winners,
            output_directory=output_directory,
            cache_directory=cache_directory,
        )
    else:
        logger.info("Skipping full profiler bundle")
    logger.info("Running logging perturbation profiles")
    logging_perturbation_results = run_logging_perturbation_profiles(
        arguments=arguments,
        baseline_paths=baseline_paths,
        winners=winners,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    comparisons = build_runtime_comparisons(headline_results)
    comparison_notes = build_runtime_comparison_notes(headline_results)
    jax_cache_diagnostics = collect_jax_cache_diagnostics(headline_results)
    stage_totals = collect_stage_totals(headline_results)
    stage_comparison_rows = build_stage_comparison_rows(headline_results)
    algorithmic_findings = build_algorithmic_findings(stage_comparison_rows)
    binary_correction_diagnostics = build_binary_correction_diagnostics(
        headline_results=headline_results,
        finalist_results_by_key=tuning_results.finalist_results_by_key,
        stage_timing_mode=arguments.stage_timing_mode,
    )
    summary_payload = {
        "stage_timing_mode": arguments.stage_timing_mode.value,
        "preflight": preflight_metadata,
        "campaign_budget": dataclasses.asdict(campaign_budget),
        "setup_results": [dataclasses.asdict(result) for result in setup_results],
        "bgen_summaries": [dataclasses.asdict(summary) for summary in bgen_summaries],
        "winners": {key: dataclasses.asdict(value) for key, value in winners.items()},
        "headline_results": [dataclasses.asdict(result) for result in headline_results],
        "regenie_baseline_scope": serialize_regenie_baseline_scope(regenie_baseline_scope),
        "comparisons": comparisons,
        "runtime_comparison_notes": dataclasses.asdict(comparison_notes),
        "jax_cache_diagnostics": jax_cache_diagnostics,
        "stage_totals": stage_totals,
        "stage_comparisons": stage_comparison_rows,
        "algorithmic_findings": algorithmic_findings,
        "binary_correction_diagnostics": binary_correction_diagnostics,
        "deep_profiles": deep_profile_results,
        "logging_perturbation_results": logging_perturbation_results,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    summary_markdown = build_summary_markdown(
        aggregate_results=headline_results,
        comparisons=comparisons,
        comparison_notes=comparison_notes,
        regenie_baseline_scope=regenie_baseline_scope,
        stage_totals=stage_totals,
        stage_comparison_rows=stage_comparison_rows,
        algorithmic_findings=algorithmic_findings,
        logging_perturbation_results=logging_perturbation_results,
        binary_correction_diagnostics=binary_correction_diagnostics,
    )
    (output_directory / "summary.md").write_text(summary_markdown, encoding="utf-8")
    write_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
    )
    write_standard_profile_artifacts(
        arguments=arguments,
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        status=profile_summary_artifact_status(summary_payload),
        summary_payload=summary_payload,
        summary_markdown=summary_markdown,
        hydra_config=hydra_config,
    )
    logger.info("Wrote deep profile artifacts under %s", output_directory)


@hydra.main(version_base=None, config_path="../configs", config_name="profile_regenie2_deep")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the deep profiling campaign through Hydra."""
    run_tool(build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run the landau deep profiling campaign."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
