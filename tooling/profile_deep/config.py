"""Deep-profile Hydra configuration adaptation."""

from __future__ import annotations

import dataclasses
import os
import typing
from datetime import UTC, datetime
from pathlib import Path

import tooling.configuration as tooling_configuration
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    import omegaconf

REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/profiles")


def resolve_repo_path(value: typing.Any) -> Path:
    """Resolve a path relative to the repository root."""
    return tooling_paths.resolve_repo_relative_path(Path(str(value)), REPOSITORY_ROOT)


def resolve_data_path(data_directory: Path, value: typing.Any) -> Path:
    """Resolve one input path relative to the data directory."""
    return tooling_paths.resolve_data_path(data_directory, Path(str(value)))


def should_emit_stage_timings(arguments: profile_deep_models.ProfileArguments) -> bool:
    """Return whether exact stage timing artifacts should be emitted."""
    return arguments.stage_timing_mode == profile_deep_models.ProfileStageTimingMode.EXACT


def build_output_directory(arguments: profile_deep_models.ProfileArguments) -> Path:
    """Resolve the campaign output directory."""
    if arguments.output_dir is not None:
        return arguments.output_dir
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return arguments.output_parent / f"landau_deep_{arguments.chromosome_label}_{timestamp}"


def configured_regenie_executable(arguments: profile_deep_models.ProfileArguments) -> str:
    """Return the configured original or patched REGENIE executable name."""
    if arguments.regenie_executable is not None:
        return arguments.regenie_executable
    return os.environ.get("REGENIE_BIN", "regenie")


def profile_configuration_payload(arguments: profile_deep_models.ProfileArguments) -> dict[str, object]:
    """Build a JSON-ready profile configuration snapshot."""
    return typing.cast("dict[str, object]", tooling_reports.to_jsonable(dataclasses.asdict(arguments)))


def apply_smoke_overrides(
    arguments: profile_deep_models.ProfileArguments,
) -> profile_deep_models.ProfileArguments:
    """Reduce the campaign size for a landau smoke profile."""
    if not arguments.smoke:
        return arguments
    return dataclasses.replace(
        arguments,
        variant_limit=None,
        chunk_sizes="2048",
        output_writer_thread_counts="1",
        firth_batch_sizes="32",
        rayon_thread_counts="1",
        top_finalists=1,
        tuning_warmups=0,
        tuning_trials=1,
        finalist_warmups=0,
        finalist_trials=1,
        headline_warmups=0,
        headline_trials=1,
        regenie_baseline_warmups=0,
        regenie_baseline_trials=1,
        py_spy_timeout_seconds=15,
        scalene_timeout_seconds=15,
        memray_timeout_seconds=15,
        linux_perf_timeout_seconds=15,
        nsight_systems_timeout_seconds=15,
        nsight_compute_timeout_seconds=15,
    )


def build_arguments_from_config(config: omegaconf.DictConfig) -> profile_deep_models.ProfileArguments:
    """Build profile parameters from a composed Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    stage_timing_mode = profile_deep_models.ProfileStageTimingMode.EXACT
    if "telemetry" in config:
        stage_timing_mode = profile_deep_models.ProfileStageTimingMode(str(config.telemetry.stage_timing_mode))
    data_directory = resolve_repo_path(tool_values["data_dir"])
    output_parent = resolve_repo_path(tool_values.get("output_parent", DEFAULT_OUTPUT_PARENT))
    explicit_output_directory = tooling_hydra_arguments.path_or_none(tool_values.get("output_dir"))
    if explicit_output_directory is not None:
        explicit_output_directory = tooling_paths.resolve_repo_relative_path(
            explicit_output_directory,
            REPOSITORY_ROOT,
        )
        output_parent = explicit_output_directory.parent
    return profile_deep_models.ProfileArguments(
        chromosome_label=str(tool_values["chromosome_label"]),
        data_directory=data_directory,
        baseline_directory=resolve_data_path(data_directory, tool_values["baseline_dir"]),
        bed_prefix=resolve_data_path(data_directory, tool_values["bed_prefix"]),
        bgen_path=resolve_data_path(data_directory, tool_values["bgen"]),
        sample_path=resolve_data_path(data_directory, tool_values["sample"]),
        continuous_phenotype_path=resolve_data_path(data_directory, tool_values["linear_phenotype_file"]),
        binary_phenotype_path=resolve_data_path(data_directory, tool_values["binary_phenotype_file"]),
        covariate_path=resolve_data_path(data_directory, tool_values["covariate_file"]),
        regenie_prediction_list_path=resolve_data_path(data_directory, tool_values["binary_prediction_list"]),
        regenie_qt_prediction_list_path=resolve_data_path(data_directory, tool_values["linear_prediction_list"]),
        output_dir=explicit_output_directory,
        output_parent=output_parent,
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values.get("variant_limit")),
        dry_run=bool(tool_values["dry_run"]),
        include_regenie_baseline=bool(tool_values["include_regenie_baseline"]),
        regenie_executable=(
            None if tool_values.get("regenie_executable") is None else str(tool_values["regenie_executable"])
        ),
        regenie_baseline_trait_types=tooling_hydra_arguments.comma_join(tool_values["regenie_baseline_trait_types"]),
        regenie_baseline_variant_limit=tooling_hydra_arguments.integer_or_none(
            tool_values.get("regenie_baseline_variant_limit")
        ),
        regenie_baseline_warmups=int(tool_values["regenie_baseline_warmups"]),
        regenie_baseline_trials=int(tool_values["regenie_baseline_trials"]),
        workload_keys=tooling_hydra_arguments.comma_join(tool_values["workload_keys"]),
        max_subprocess_runs=tooling_hydra_arguments.integer_or_none(tool_values.get("max_subprocess_runs")),
        max_major_profiler_runs=tooling_hydra_arguments.integer_or_none(tool_values.get("max_major_profiler_runs")),
        allow_over_budget=bool(tool_values["allow_over_budget"]),
        smoke=bool(tool_values["smoke"]),
        skip_deep_profiles=bool(tool_values["skip_deep_profiles"]),
        enable_jax_trace=bool(tool_values["enable_jax_trace"]),
        enable_jax_memory_profile=bool(tool_values["enable_jax_memory_profile"]),
        enable_python_cprofile=bool(tool_values["enable_python_cprofile"]),
        enable_py_spy=bool(tool_values["enable_py_spy"]),
        enable_scalene=bool(tool_values["enable_scalene"]),
        enable_memray=bool(tool_values["enable_memray"]),
        enable_linux_perf=bool(tool_values["enable_linux_perf"]),
        enable_nsight_systems=bool(tool_values["enable_nsight_systems"]),
        enable_nsight_compute=bool(tool_values["enable_nsight_compute"]),
        py_spy_timeout_seconds=int(tool_values["py_spy_timeout_seconds"]),
        scalene_timeout_seconds=int(tool_values["scalene_timeout_seconds"]),
        memray_timeout_seconds=int(tool_values["memray_timeout_seconds"]),
        linux_perf_timeout_seconds=int(tool_values["linux_perf_timeout_seconds"]),
        nsight_systems_timeout_seconds=int(tool_values["nsight_systems_timeout_seconds"]),
        nsight_compute_timeout_seconds=int(tool_values["nsight_compute_timeout_seconds"]),
        enable_rust_criterion=bool(tool_values["enable_rust_criterion"]),
        enable_logging_perturbation=bool(tool_values["enable_logging_perturbation"]),
        rust_benchmarks=tooling_hydra_arguments.comma_join(tool_values["rust_benchmarks"]),
        chunk_sizes=tooling_hydra_arguments.comma_join(tool_values["chunk_sizes"]),
        output_writer_thread_counts=tooling_hydra_arguments.comma_join(tool_values["output_writer_thread_counts"]),
        firth_batch_sizes=tooling_hydra_arguments.comma_join(tool_values["firth_batch_sizes"]),
        rayon_thread_counts=tooling_hydra_arguments.comma_join(tool_values["rayon_thread_counts"]),
        top_finalists=int(tool_values["top_finalists"]),
        tuning_warmups=int(tool_values["tuning_warmups"]),
        tuning_trials=int(tool_values["tuning_trials"]),
        finalist_warmups=int(tool_values["finalist_warmups"]),
        finalist_trials=int(tool_values["finalist_trials"]),
        headline_warmups=int(tool_values["headline_warmups"]),
        headline_trials=int(tool_values["headline_trials"]),
        stage_timing_mode=stage_timing_mode,
    )


def build_arguments_from_overrides(
    overrides: typing.Sequence[str] | None = None,
) -> profile_deep_models.ProfileArguments:
    """Compose the deep-profile config and return resolved parameters."""
    config = tooling_configuration.compose_config(config_name="profile_regenie2_deep", overrides=overrides)
    return build_arguments_from_config(config)
