#!/usr/bin/env python3
"""Deep landau profiling harness for original REGENIE and g REGENIE step 2."""

from __future__ import annotations

import dataclasses
import enum
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

import scripts.benchmark as baseline_benchmark
import scripts.benchmark_regenie_comparison as comparison_benchmark
import tooling.cli.benchmark_bgen_reader as benchmark_bgen_reader
import tooling.configuration as tooling_configuration
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import logging as tooling_logging
from tooling.common import paths as tooling_paths

if typing.TYPE_CHECKING:
    import omegaconf

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/profiles")
DEFAULT_VARIANT_COUNT = 418_943
JAX_XLA_AUTOTUNE_CACHE = "xla_gpu_per_fusion_autotune_cache_dir"
ENABLE_XLA_AUTOTUNE_CACHE = os.environ.get("G_PROFILE_ENABLE_XLA_AUTOTUNE_CACHE") == "1"
GPU_JAX_CACHE_PARENT_DEFAULT = "/tmp/g-jax-profile-cache"
JAX_DEBUG_LOG_MODULES = "jax._src.compiler,jax._src.lru_cache"
JAX_LOG_SAMPLE_LINE_LIMIT = 20


class ProfileWorkloadKey(enum.StrEnum):
    """Trait/device workload keys for splittable deep profile campaigns."""

    QUANTITATIVE_CPU = "quantitative_cpu"
    QUANTITATIVE_GPU = "quantitative_gpu"
    BINARY_CPU = "binary_cpu"
    BINARY_GPU = "binary_gpu"

    @property
    def trait_type(self) -> str:
        """Return the workload trait type."""
        return self.value.rsplit("_", maxsplit=1)[0]

    @property
    def device(self) -> str:
        """Return the workload device."""
        return self.value.rsplit("_", maxsplit=1)[1]


class ProfileWorkloadSelector(enum.StrEnum):
    """Accepted workload selection tokens."""

    ALL = "all"
    QUANTITATIVE = "quantitative"
    BINARY = "binary"
    CPU = "cpu"
    GPU = "gpu"
    QUANTITATIVE_CPU = "quantitative_cpu"
    QUANTITATIVE_GPU = "quantitative_gpu"
    BINARY_CPU = "binary_cpu"
    BINARY_GPU = "binary_gpu"


PROFILE_WORKLOAD_KEYS: tuple[ProfileWorkloadKey, ...] = (
    ProfileWorkloadKey.QUANTITATIVE_CPU,
    ProfileWorkloadKey.QUANTITATIVE_GPU,
    ProfileWorkloadKey.BINARY_CPU,
    ProfileWorkloadKey.BINARY_GPU,
)


class CampaignBudgetSectionName(enum.StrEnum):
    """Budget accounting sections in execution order."""

    BGEN_PRE_SWEEP = "bgen_pre_sweep"
    TUNING = "tuning"
    FINALISTS = "finalists"
    HEADLINE_TRIALS = "headline_trials"
    DEEP_PROFILERS = "deep_profilers"
    LOGGING_PERTURBATION = "logging_perturbation"
    RUST_CRITERION = "rust_criterion"


CAMPAIGN_BUDGET_SECTION_DISPLAY_NAMES: dict[CampaignBudgetSectionName, str] = {
    CampaignBudgetSectionName.BGEN_PRE_SWEEP: "BGEN pre-sweep",
    CampaignBudgetSectionName.TUNING: "Tuning",
    CampaignBudgetSectionName.FINALISTS: "Finalists",
    CampaignBudgetSectionName.HEADLINE_TRIALS: "Headline trials",
    CampaignBudgetSectionName.DEEP_PROFILERS: "Deep profilers",
    CampaignBudgetSectionName.LOGGING_PERTURBATION: "Logging perturbation",
    CampaignBudgetSectionName.RUST_CRITERION: "Rust Criterion",
}


@dataclasses.dataclass(frozen=True)
class ProfileArguments:
    """Resolved deep profile campaign parameters.

    Attributes:
        chromosome_label: Chromosome label used in reports and logs.
        data_directory: Directory containing benchmark inputs.
        baseline_directory: Directory containing baseline step 1 prediction lists.
        bed_prefix: PLINK BED prefix used by original REGENIE step 1 setup.
        bgen_path: BGEN path used by step 2 runs.
        sample_path: Sample path used by step 2 runs.
        continuous_phenotype_path: Quantitative phenotype table path.
        binary_phenotype_path: Binary phenotype table path.
        covariate_path: Covariate table path.
        regenie_prediction_list_path: Binary step 1 prediction list.
        regenie_qt_prediction_list_path: Quantitative step 1 prediction list.
        output_dir: Optional explicit output directory.
        output_parent: Parent directory for timestamped output directories.
        variant_limit: Optional variant cap for smoke runs.
        dry_run: Whether to write a profile plan without running workloads.
        include_regenie_baseline: Whether headline trials include original REGENIE.
        regenie_executable: Optional original or patched REGENIE executable for baseline runs.
        regenie_baseline_trait_types: Comma-separated REGENIE baseline trait types.
        regenie_baseline_variant_limit: Optional baseline variant cap. Defaults to variant_limit when unset.
        regenie_baseline_warmups: Warmup count for original REGENIE baseline trials.
        regenie_baseline_trials: Measured count for original REGENIE baseline trials.
        workload_keys: Comma-separated trait/device workloads to include.
        max_subprocess_runs: Maximum planned subprocess runs allowed without override.
        max_major_profiler_runs: Maximum major profiler runs allowed without override.
        allow_over_budget: Whether to allow execution over the configured budget.
        smoke: Whether to use the reduced smoke campaign.
        skip_deep_profiles: Whether to skip sampling and trace profiles.
        enable_jax_trace: Whether deep profiles capture JAX profiler traces.
        enable_jax_memory_profile: Whether deep profiles capture JAX memory profiles.
        enable_python_cprofile: Whether deep profiles capture cProfile output.
        enable_py_spy: Whether deep profiles capture py-spy speedscope output when available.
        enable_scalene: Whether deep profiles capture Scalene CPU/memory output when available.
        enable_memray: Whether deep profiles capture Memray allocation output when available.
        enable_linux_perf: Whether deep profiles capture Linux perf native stack data when available.
        enable_nsight_systems: Whether deep profiles capture Nsight Systems CUDA timelines when available.
        enable_nsight_compute: Whether deep profiles capture Nsight Compute kernel reports when available.
        enable_rust_criterion: Whether deep profiles run Rust Criterion benches.
        enable_logging_perturbation: Whether the profile runs telemetry/logging perturbation trials.
        rust_benchmarks: Comma-separated Rust Criterion benchmark names.
        chunk_sizes: Comma-separated step 2 chunk-size values.
        staging_depths: Comma-separated staging-depth values.
        output_writer_thread_counts: Comma-separated writer thread-count values.
        writer_queue_depth_multipliers: Comma-separated queue-depth multipliers.
        firth_batch_sizes: Comma-separated binary Firth batch sizes.
        bgen_decode_tile_variant_counts: Comma-separated BGEN decode tile sizes.
        rayon_thread_counts: Comma-separated Rayon thread-count values.
        bgen_benchmark_chunk_size: Chunk size for BGEN pre-sweep cases.
        top_bgen_candidates: Number of BGEN candidates kept.
        top_finalists: Number of finalists kept.
        tuning_warmups: Warmup count for tuning trials.
        tuning_trials: Measured count for tuning trials.
        finalist_warmups: Warmup count for finalist trials.
        finalist_trials: Measured count for finalist trials.
        headline_warmups: Warmup count for headline trials.
        headline_trials: Measured count for headline trials.

    """

    chromosome_label: str
    data_directory: Path
    baseline_directory: Path
    bed_prefix: Path
    bgen_path: Path
    sample_path: Path
    continuous_phenotype_path: Path
    binary_phenotype_path: Path
    covariate_path: Path
    regenie_prediction_list_path: Path
    regenie_qt_prediction_list_path: Path
    output_dir: Path | None
    output_parent: Path
    variant_limit: int | None
    dry_run: bool
    include_regenie_baseline: bool
    regenie_executable: str | None
    regenie_baseline_trait_types: str
    regenie_baseline_variant_limit: int | None
    regenie_baseline_warmups: int
    regenie_baseline_trials: int
    workload_keys: str
    max_subprocess_runs: int | None
    max_major_profiler_runs: int | None
    allow_over_budget: bool
    smoke: bool
    skip_deep_profiles: bool
    enable_jax_trace: bool
    enable_jax_memory_profile: bool
    enable_python_cprofile: bool
    enable_py_spy: bool
    enable_scalene: bool
    enable_memray: bool
    enable_linux_perf: bool
    enable_nsight_systems: bool
    enable_nsight_compute: bool
    enable_rust_criterion: bool
    enable_logging_perturbation: bool
    rust_benchmarks: str
    chunk_sizes: str
    staging_depths: str
    output_writer_thread_counts: str
    writer_queue_depth_multipliers: str
    firth_batch_sizes: str
    bgen_decode_tile_variant_counts: str
    rayon_thread_counts: str
    bgen_benchmark_chunk_size: int
    top_bgen_candidates: int
    top_finalists: int
    tuning_warmups: int
    tuning_trials: int
    finalist_warmups: int
    finalist_trials: int
    headline_warmups: int
    headline_trials: int


@dataclasses.dataclass(frozen=True)
class ProfilerToolStatus:
    """Availability status for one optional profiling tool.

    Attributes:
        tool_name: Stable profiler tool name.
        enabled: Whether the current profile config requests the tool.
        available: Whether the local executable or built-in profiler is available.
        executable_path: Resolved executable path when one is required.
        notes: Human-readable status details.

    """

    tool_name: str
    enabled: bool
    available: bool
    executable_path: str | None
    notes: str


@dataclasses.dataclass(frozen=True)
class LoggingPerturbationCase:
    """One telemetry/logging perturbation case.

    Attributes:
        name: Stable case name used in artifact paths.
        diagnostic_options: Extra g diagnostics options for the child run.

    """

    name: str
    diagnostic_options: dict[str, object]


@dataclasses.dataclass(frozen=True)
class Step2Candidate:
    """One g REGENIE step 2 tuning candidate."""

    trait_type: str
    device: str
    chunk_size: int
    staging_depth: int
    output_writer_thread_count: int
    output_writer_queue_depth: int
    bgen_decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    firth_batch_size: int | None


@dataclasses.dataclass(frozen=True)
class BgenCandidateSummary:
    """Measured BGEN reader candidate summary."""

    decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    median_seconds: float
    mean_seconds: float
    durations_seconds: list[float]


@dataclasses.dataclass(frozen=True)
class JaxCacheSnapshot:
    """Filesystem snapshot for a JAX persistent-cache directory.

    Attributes:
        path: Cache directory path.
        exists: Whether the directory existed when sampled.
        file_count: Number of regular files below the directory.
        total_size_bytes: Sum of regular-file sizes below the directory.
        error: Optional error encountered while walking the directory.

    """

    path: str
    exists: bool
    file_count: int
    total_size_bytes: int
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class JaxCompileLogSummary:
    """Parsed JAX compile and persistent-cache log counters.

    Attributes:
        compilation_event_count: Lines that look like JAX tracing, lowering, or compilation events.
        persistent_cache_event_count: Lines mentioning the persistent compilation cache.
        persistent_cache_hit_count: Lines that look like persistent-cache hits.
        persistent_cache_miss_count: Lines that look like persistent-cache misses.
        tracing_cache_miss_count: Lines that look like JAX tracing-cache misses.
        cache_miss_explanation_count: Miss lines that include an explanatory reason.
        sample_log_lines: Bounded sample of relevant log lines for manual inspection.

    """

    compilation_event_count: int
    persistent_cache_event_count: int
    persistent_cache_hit_count: int
    persistent_cache_miss_count: int
    tracing_cache_miss_count: int
    cache_miss_explanation_count: int
    sample_log_lines: list[str]


@dataclasses.dataclass(frozen=True)
class JaxCacheDiagnostics:
    """JAX cache and compile diagnostics for one g subprocess.

    Attributes:
        cache_directory: Cache directory observed by the profiling harness.
        child_reported_cache_directory: Cache directory echoed by the child process.
        persistent_cache_used: Whether the run requested the JAX persistent compilation cache.
        before: Cache snapshot before the subprocess.
        after: Cache snapshot after the subprocess.
        file_count_delta: Cache file-count delta from before to after.
        size_bytes_delta: Cache byte-size delta from before to after.
        compile_log_summary: Parsed compile/cache counters from stderr.

    """

    cache_directory: str | None
    child_reported_cache_directory: str | None
    persistent_cache_used: bool
    before: JaxCacheSnapshot | None
    after: JaxCacheSnapshot | None
    file_count_delta: int | None
    size_bytes_delta: int | None
    compile_log_summary: JaxCompileLogSummary


@dataclasses.dataclass(frozen=True)
class JaxColdWarmDiagnostics:
    """Cold-versus-warm JAX diagnostics for one aggregate result.

    Attributes:
        cache_directory: Cache directory shared by the subprocess trials.
        persistent_cache_used: Whether any trial requested the persistent compilation cache.
        cold_trial_name: First successful g subprocess used as the cold reference.
        cold_wall_time_seconds: Wall time for the cold reference.
        warm_trial_count: Number of later successful subprocesses treated as warm trials.
        warm_median_wall_time_seconds: Median wall time across warm trials.
        warm_mean_wall_time_seconds: Mean wall time across warm trials.
        cold_to_warm_speedup_ratio: Cold wall time divided by warm median wall time.
        cold_cache_file_count_delta: Cache file-count delta for the cold trial.
        warm_cache_file_count_delta: Sum of cache file-count deltas across warm trials.
        cold_cache_size_bytes_delta: Cache byte-size delta for the cold trial.
        warm_cache_size_bytes_delta: Sum of cache byte-size deltas across warm trials.
        cold_compilation_event_count: Compile/tracing log-event count for the cold trial.
        warm_compilation_event_count: Sum of compile/tracing log-event counts across warm trials.
        cold_cache_hit_count: Persistent-cache hit count for the cold trial.
        warm_cache_hit_count: Sum of persistent-cache hit counts across warm trials.
        cold_cache_miss_count: Persistent-cache miss count for the cold trial.
        warm_cache_miss_count: Sum of persistent-cache miss counts across warm trials.
        cold_tracing_cache_miss_count: Tracing-cache miss count for the cold trial.
        warm_tracing_cache_miss_count: Sum of tracing-cache miss counts across warm trials.

    """

    cache_directory: str | None
    persistent_cache_used: bool
    cold_trial_name: str
    cold_wall_time_seconds: float | None
    warm_trial_count: int
    warm_median_wall_time_seconds: float | None
    warm_mean_wall_time_seconds: float | None
    cold_to_warm_speedup_ratio: float | None
    cold_cache_file_count_delta: int | None
    warm_cache_file_count_delta: int | None
    cold_cache_size_bytes_delta: int | None
    warm_cache_size_bytes_delta: int | None
    cold_compilation_event_count: int
    warm_compilation_event_count: int
    cold_cache_hit_count: int
    warm_cache_hit_count: int
    cold_cache_miss_count: int
    warm_cache_miss_count: int
    cold_tracing_cache_miss_count: int
    warm_tracing_cache_miss_count: int


@dataclasses.dataclass(frozen=True)
class CampaignBudgetSection:
    """Budget estimate for one campaign section.

    Attributes:
        name: Stable machine-readable section name.
        display_name: Human-readable section name.
        candidate_count: Planned cases or configurations in the section.
        subprocess_run_count: Estimated subprocess executions in the section.
        major_profiler_run_count: Estimated heavy profiler executions in the section.
        notes: Explanation for the estimate.

    """

    name: str
    display_name: str
    candidate_count: int
    subprocess_run_count: int
    major_profiler_run_count: int
    notes: str


@dataclasses.dataclass(frozen=True)
class CampaignBudget:
    """Budget estimate for a deep profile campaign.

    Attributes:
        workload_keys: Trait/device workloads selected for this campaign.
        max_subprocess_runs: Configured subprocess budget.
        max_major_profiler_runs: Configured major-profiler budget.
        total_candidate_count: Total planned cases or configurations.
        total_subprocess_run_count: Total estimated subprocess executions.
        total_major_profiler_run_count: Total estimated heavy profiler executions.
        over_subprocess_budget: Whether the subprocess estimate exceeds the configured budget.
        over_major_profiler_budget: Whether the profiler estimate exceeds the configured budget.
        sections: Section-level budget estimates.
        guidance: Human-readable guidance for reducing or overriding the campaign.

    """

    workload_keys: tuple[str, ...]
    max_subprocess_runs: int | None
    max_major_profiler_runs: int | None
    total_candidate_count: int
    total_subprocess_run_count: int
    total_major_profiler_run_count: int
    over_subprocess_budget: bool
    over_major_profiler_budget: bool
    sections: tuple[CampaignBudgetSection, ...]
    guidance: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """One measured process execution."""

    name: str
    implementation: str
    trait_type: str
    device: str
    status: str
    wall_time_seconds: float | None
    output_row_count: int | None
    stdout_log_path: str
    stderr_log_path: str
    command_arguments: list[str]
    environment_overrides: dict[str, str]
    output_path: str | None = None
    stage_timing_path: str | None = None
    regenie_profile_path: str | None = None
    profiler_artifact_path: str | None = None
    application_output_prefix: str | None = None
    application_output_run_directory: str | None = None
    device_diagnostics: dict[str, typing.Any] | None = None
    jax_cache_diagnostics: JaxCacheDiagnostics | None = None
    notes: str | None = None


@dataclasses.dataclass(frozen=True)
class DeepProfilerRunPaths:
    """Paths dedicated to one deep profiler's application child process.

    Attributes:
        application_output_prefix: Prefix passed to the profiled g child as `out`.
        application_output_run_directory: Expected chunked g output run directory.
        stage_timing_path: Stage timing JSON path for the profiled child run.
        profile_script_path: Python script path executed by the profiler wrapper.

    """

    application_output_prefix: Path
    application_output_run_directory: Path
    stage_timing_path: Path
    profile_script_path: Path


@dataclasses.dataclass(frozen=True)
class DeepProfilerChildCommand:
    """Prepared child command and metadata for one deep profiler wrapper.

    Attributes:
        command_arguments: Python command used as the profiler wrapper target.
        environment_overrides: Environment overrides for the profiler wrapper.
        run_paths: Paths dedicated to this profiler's application child process.

    """

    command_arguments: list[str]
    environment_overrides: dict[str, str]
    run_paths: DeepProfilerRunPaths


@dataclasses.dataclass(frozen=True)
class AggregateResult:
    """Aggregate runtime statistics for one benchmark cell."""

    name: str
    implementation: str
    trait_type: str
    device: str
    status: str
    trial_count: int
    warmup_count: int
    median_wall_time_seconds: float | None
    mean_wall_time_seconds: float | None
    min_wall_time_seconds: float | None
    max_wall_time_seconds: float | None
    standard_deviation_seconds: float | None
    rows_per_second: float | None
    trials: list[TrialResult]
    warmup_trials: list[TrialResult] = dataclasses.field(default_factory=list)
    jax_cold_warm_summary: JaxColdWarmDiagnostics | None = None


@dataclasses.dataclass(frozen=True)
class ProfilePlan:
    """Dry-run profile campaign plan.

    Attributes:
        chromosome_label: Chromosome label selected by Hydra.
        output_directory: Planned output directory.
        required_inputs: Input paths and step 1 prediction paths used by real runs.
        workload_keys: Trait/device workloads selected for this campaign.
        campaign_budget: Estimated campaign budget and section counts.
        profiler_modes: Profiler modes requested by config.
        profiler_tools: Profiler tool availability records.
        logging_perturbation_cases: Planned telemetry/logging perturbation cases.
        regenie_baseline_scope: Planned original REGENIE baseline scope.
        regenie_baseline_commands: Planned original REGENIE baseline commands.
        rust_benchmark_commands: Rust Criterion benchmark commands.
        notes: Human-readable plan notes.

    """

    chromosome_label: str
    output_directory: str
    required_inputs: list[str]
    workload_keys: list[str]
    campaign_budget: CampaignBudget
    profiler_modes: dict[str, bool]
    profiler_tools: dict[str, dict[str, object]]
    logging_perturbation_cases: list[dict[str, object]]
    regenie_baseline_scope: dict[str, object] | None
    regenie_baseline_commands: list[dict[str, object]]
    rust_benchmark_commands: list[list[str]]
    notes: list[str]


class RegenieBaselineScopeStatus(enum.StrEnum):
    """Status for original REGENIE baseline workload scoping."""

    FULL = "full"
    BOUNDED = "bounded"
    UNSUPPORTED = "unsupported"


@dataclasses.dataclass(frozen=True)
class RegenieBaselineScope:
    """Original REGENIE baseline workload scope.

    Attributes:
        status: Whether the baseline is full, bounded, or unsupported.
        variant_limit: Requested bounded variant count.
        extract_path: Extract-list path used for bounded REGENIE trials.
        metadata_path: Variant metadata file used to build the extract list.
        selected_variant_count: Number of variants selected for a bounded run.
        variant_identifiers: Selected variant identifiers, omitted from serialized reports.
        notes: Human-readable scoping notes.

    """

    status: RegenieBaselineScopeStatus
    variant_limit: int | None
    extract_path: Path | None
    metadata_path: Path | None
    selected_variant_count: int | None
    variant_identifiers: tuple[str, ...]
    notes: str


@dataclasses.dataclass(frozen=True)
class RuntimeComparisonNotes:
    """Non-success runtime comparison details.

    Attributes:
        unsupported: Comparisons skipped because no compatible baseline was available.
        failed: Comparisons where at least one paired run failed.

    """

    unsupported: list[str]
    failed: list[str]


def resolve_repo_path(value: typing.Any) -> Path:
    """Resolve a path relative to the repository root."""
    return tooling_paths.resolve_repo_relative_path(Path(str(value)), REPOSITORY_ROOT)


def resolve_data_path(data_directory: Path, value: typing.Any) -> Path:
    """Resolve one input path relative to the data directory."""
    return tooling_paths.resolve_data_path(data_directory, Path(str(value)))


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


def parse_int_list(raw_values: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers."""
    parsed_values = tuple(int(value.strip()) for value in raw_values.split(",") if value.strip())
    if not parsed_values:
        message = "At least one integer is required."
        raise ValueError(message)
    return parsed_values


def parse_string_list(raw_values: str) -> tuple[str, ...]:
    """Parse a comma-separated list of strings."""
    parsed_values = tuple(value.strip() for value in raw_values.split(",") if value.strip())
    if not parsed_values:
        message = "At least one string value is required."
        raise ValueError(message)
    return parsed_values


def parse_regenie_baseline_trait_types(raw_values: str) -> tuple[str, ...]:
    """Parse and validate original REGENIE baseline trait types."""
    trait_types = parse_string_list(raw_values)
    valid_trait_types = {"quantitative", "binary"}
    invalid_trait_types = sorted(set(trait_types) - valid_trait_types)
    if invalid_trait_types:
        message = f"Unsupported REGENIE baseline trait types: {', '.join(invalid_trait_types)}"
        raise ValueError(message)
    return trait_types


def parse_profile_workload_keys(raw_values: str) -> tuple[ProfileWorkloadKey, ...]:
    """Parse and expand workload selection tokens."""
    selected_workload_keys: list[ProfileWorkloadKey] = []
    invalid_selectors: list[str] = []
    for raw_selector in parse_string_list(raw_values):
        try:
            selector = ProfileWorkloadSelector(raw_selector)
        except ValueError:
            invalid_selectors.append(raw_selector)
            continue
        if selector == ProfileWorkloadSelector.ALL:
            selected_workload_keys.extend(PROFILE_WORKLOAD_KEYS)
        elif selector == ProfileWorkloadSelector.QUANTITATIVE:
            selected_workload_keys.extend(
                workload_key for workload_key in PROFILE_WORKLOAD_KEYS if workload_key.trait_type == "quantitative"
            )
        elif selector == ProfileWorkloadSelector.BINARY:
            selected_workload_keys.extend(
                workload_key for workload_key in PROFILE_WORKLOAD_KEYS if workload_key.trait_type == "binary"
            )
        elif selector == ProfileWorkloadSelector.CPU:
            selected_workload_keys.extend(
                workload_key for workload_key in PROFILE_WORKLOAD_KEYS if workload_key.device == "cpu"
            )
        elif selector == ProfileWorkloadSelector.GPU:
            selected_workload_keys.extend(
                workload_key for workload_key in PROFILE_WORKLOAD_KEYS if workload_key.device == "gpu"
            )
        else:
            selected_workload_keys.append(ProfileWorkloadKey(selector.value))
    if invalid_selectors:
        valid_values = ", ".join(selector.value for selector in ProfileWorkloadSelector)
        message = (
            f"Unsupported deep-profile workload selectors: {', '.join(invalid_selectors)}. "
            f"Valid selectors: {valid_values}."
        )
        raise ValueError(message)
    deduplicated_workload_keys = tuple(dict.fromkeys(selected_workload_keys))
    if not deduplicated_workload_keys:
        message = "At least one deep-profile workload key is required."
        raise ValueError(message)
    return deduplicated_workload_keys


def selected_regenie_baseline_trait_types(arguments: ProfileArguments) -> tuple[str, ...]:
    """Return REGENIE baseline traits that match the selected workload traits."""
    requested_trait_types = parse_regenie_baseline_trait_types(arguments.regenie_baseline_trait_types)
    selected_trait_types = {
        workload_key.trait_type for workload_key in parse_profile_workload_keys(arguments.workload_keys)
    }
    return tuple(trait_type for trait_type in requested_trait_types if trait_type in selected_trait_types)


def build_output_directory(arguments: ProfileArguments) -> Path:
    """Resolve the campaign output directory."""
    if arguments.output_dir is not None:
        return arguments.output_dir
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return arguments.output_parent / f"landau_deep_{arguments.chromosome_label}_{timestamp}"


def configured_regenie_executable(arguments: ProfileArguments) -> str:
    """Return the configured original or patched REGENIE executable name."""
    if arguments.regenie_executable is not None:
        return arguments.regenie_executable
    return os.environ.get("REGENIE_BIN", "regenie")


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
        "schema_version": 2,
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
) -> None:
    """Write the profile artifact manifest."""
    manifest = collect_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
        profile_plan=profile_plan,
    )
    (output_directory / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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


def build_queue_depth_values(writer_thread_count: int, queue_depth_multipliers: tuple[int, ...]) -> tuple[int, ...]:
    """Build queue depths from writer thread count and multipliers."""
    return tuple(sorted({max(1, writer_thread_count * multiplier) for multiplier in queue_depth_multipliers}))


def build_candidate_slug(candidate: Step2Candidate) -> str:
    """Build a stable filename slug for a tuning candidate."""
    candidate_parts = [
        candidate.trait_type,
        candidate.device,
        f"chunk{candidate.chunk_size}",
        f"staging{candidate.staging_depth}",
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
    writer_thread_counts: tuple[int, ...],
    queue_depth_multipliers: tuple[int, ...],
    firth_batch_sizes: tuple[int, ...],
) -> tuple[Step2Candidate, ...]:
    """Build the g step 2 candidate grid."""
    candidates: list[Step2Candidate] = []
    for bgen_candidate in bgen_candidates:
        for chunk_size in chunk_sizes:
            for staging_depth in staging_depths:
                for writer_thread_count in writer_thread_counts:
                    for queue_depth in build_queue_depth_values(writer_thread_count, queue_depth_multipliers):
                        if trait_type == "binary":
                            for firth_batch_size in firth_batch_sizes:
                                candidates.append(
                                    Step2Candidate(
                                        trait_type=trait_type,
                                        device=device,
                                        chunk_size=chunk_size,
                                        staging_depth=staging_depth,
                                        output_writer_thread_count=writer_thread_count,
                                        output_writer_queue_depth=queue_depth,
                                        bgen_decode_tile_variant_count=bgen_candidate.decode_tile_variant_count,
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
                                output_writer_thread_count=writer_thread_count,
                                output_writer_queue_depth=queue_depth,
                                bgen_decode_tile_variant_count=bgen_candidate.decode_tile_variant_count,
                                rayon_thread_count=bgen_candidate.rayon_thread_count,
                                firth_batch_size=None,
                            )
                        )
    return tuple(candidates)


def build_campaign_budget_section(
    *,
    section_name: CampaignBudgetSectionName,
    candidate_count: int,
    subprocess_run_count: int,
    major_profiler_run_count: int = 0,
    notes: str,
) -> CampaignBudgetSection:
    """Build one campaign budget section."""
    return CampaignBudgetSection(
        name=section_name.value,
        display_name=CAMPAIGN_BUDGET_SECTION_DISPLAY_NAMES[section_name],
        candidate_count=candidate_count,
        subprocess_run_count=subprocess_run_count,
        major_profiler_run_count=major_profiler_run_count,
        notes=notes,
    )


def count_queue_depth_grid(writer_thread_counts: tuple[int, ...], queue_depth_multipliers: tuple[int, ...]) -> int:
    """Count distinct writer queue-depth settings across writer thread counts."""
    return sum(
        len(build_queue_depth_values(writer_thread_count, queue_depth_multipliers))
        for writer_thread_count in writer_thread_counts
    )


def count_step2_tuning_candidates(
    *,
    workload_key: ProfileWorkloadKey,
    selected_bgen_candidate_count: int,
    chunk_sizes: tuple[int, ...],
    staging_depths: tuple[int, ...],
    writer_thread_counts: tuple[int, ...],
    queue_depth_multipliers: tuple[int, ...],
    firth_batch_sizes: tuple[int, ...],
    smoke: bool,
) -> int:
    """Count step 2 tuning candidates for one selected workload."""
    queue_depth_count = count_queue_depth_grid(writer_thread_counts, queue_depth_multipliers)
    candidate_count = selected_bgen_candidate_count * len(chunk_sizes) * len(staging_depths) * queue_depth_count
    if workload_key.trait_type == "binary":
        candidate_count *= len(firth_batch_sizes)
    if smoke:
        return min(candidate_count, 1)
    return candidate_count


def count_enabled_deep_profiler_modes(arguments: ProfileArguments) -> int:
    """Count profiler subprocess modes run for each selected winner."""
    mode_count = 0
    if arguments.enable_jax_trace or arguments.enable_jax_memory_profile:
        mode_count += 1
    enabled_modes = (
        arguments.enable_python_cprofile,
        arguments.enable_py_spy,
        arguments.enable_scalene,
        arguments.enable_memray,
        arguments.enable_linux_perf,
        arguments.enable_nsight_systems,
        arguments.enable_nsight_compute,
    )
    return mode_count + sum(1 for enabled in enabled_modes if enabled)


def campaign_budget_is_over_limit(campaign_budget: CampaignBudget) -> bool:
    """Return whether a campaign exceeds either configured budget."""
    return campaign_budget.over_subprocess_budget or campaign_budget.over_major_profiler_budget


def build_campaign_budget(
    *,
    arguments: ProfileArguments,
    output_directory: Path,
) -> CampaignBudget:
    """Estimate campaign section counts before executing workloads."""
    workload_keys = parse_profile_workload_keys(arguments.workload_keys)
    chunk_sizes = parse_int_list(arguments.chunk_sizes)
    staging_depths = parse_int_list(arguments.staging_depths)
    writer_thread_counts = parse_int_list(arguments.output_writer_thread_counts)
    queue_depth_multipliers = parse_int_list(arguments.writer_queue_depth_multipliers)
    firth_batch_sizes = parse_int_list(arguments.firth_batch_sizes)
    bgen_decode_tile_variant_counts = parse_int_list(arguments.bgen_decode_tile_variant_counts)
    rayon_thread_counts = parse_int_list(arguments.rayon_thread_counts)
    bgen_candidate_count = len(bgen_decode_tile_variant_counts) * len(rayon_thread_counts)
    selected_bgen_candidate_count = min(arguments.top_bgen_candidates, bgen_candidate_count)
    tuning_candidate_counts = [
        count_step2_tuning_candidates(
            workload_key=workload_key,
            selected_bgen_candidate_count=selected_bgen_candidate_count,
            chunk_sizes=chunk_sizes,
            staging_depths=staging_depths,
            writer_thread_counts=writer_thread_counts,
            queue_depth_multipliers=queue_depth_multipliers,
            firth_batch_sizes=firth_batch_sizes,
            smoke=arguments.smoke,
        )
        for workload_key in workload_keys
    ]
    tuning_candidate_count = sum(tuning_candidate_counts)
    finalist_candidate_counts = [
        min(arguments.top_finalists, tuning_candidate_count_for_workload)
        for tuning_candidate_count_for_workload in tuning_candidate_counts
    ]
    finalist_candidate_count = sum(finalist_candidate_counts)
    expected_winner_count = sum(
        1
        for finalist_count in finalist_candidate_counts
        if finalist_count > 0 and arguments.tuning_trials > 0 and arguments.finalist_trials > 0
    )
    regenie_baseline_trait_count = 0
    if arguments.include_regenie_baseline:
        regenie_baseline_trait_count = len(selected_regenie_baseline_trait_types(arguments))
    g_headline_run_count = expected_winner_count * (arguments.headline_warmups + arguments.headline_trials)
    regenie_headline_run_count = regenie_baseline_trait_count * (
        arguments.regenie_baseline_warmups + arguments.regenie_baseline_trials
    )
    deep_profiler_mode_count = 0 if arguments.skip_deep_profiles else count_enabled_deep_profiler_modes(arguments)
    deep_profiler_run_count = expected_winner_count * deep_profiler_mode_count
    logging_case_count = 0
    if arguments.enable_logging_perturbation:
        logging_case_count = len(
            build_logging_perturbation_cases(output_directory=output_directory, smoke=arguments.smoke)
        )
    logging_run_count = expected_winner_count * logging_case_count
    rust_benchmark_count = 0
    if arguments.enable_rust_criterion and not arguments.skip_deep_profiles:
        rust_benchmark_count = len(parse_string_list(arguments.rust_benchmarks))
    sections = (
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.BGEN_PRE_SWEEP,
            candidate_count=bgen_candidate_count,
            subprocess_run_count=bgen_candidate_count,
            notes=(
                f"{len(bgen_decode_tile_variant_counts)} BGEN tile values x "
                f"{len(rayon_thread_counts)} Rayon thread values; each case repeats internally "
                f"{arguments.tuning_trials} time(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.TUNING,
            candidate_count=tuning_candidate_count,
            subprocess_run_count=tuning_candidate_count * (arguments.tuning_warmups + arguments.tuning_trials),
            notes=(
                f"{len(workload_keys)} selected workload(s), top {selected_bgen_candidate_count} BGEN candidate(s), "
                f"{arguments.tuning_warmups} warmup(s), and {arguments.tuning_trials} measured trial(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.FINALISTS,
            candidate_count=finalist_candidate_count,
            subprocess_run_count=finalist_candidate_count * (arguments.finalist_warmups + arguments.finalist_trials),
            notes=(
                f"Up to {arguments.top_finalists} finalist(s) per selected workload, "
                f"{arguments.finalist_warmups} warmup(s), and {arguments.finalist_trials} measured trial(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.HEADLINE_TRIALS,
            candidate_count=expected_winner_count + regenie_baseline_trait_count,
            subprocess_run_count=g_headline_run_count + regenie_headline_run_count,
            notes=(
                f"{expected_winner_count} expected g winner(s) and "
                f"{regenie_baseline_trait_count} selected REGENIE baseline trait(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.DEEP_PROFILERS,
            candidate_count=deep_profiler_run_count,
            subprocess_run_count=deep_profiler_run_count,
            major_profiler_run_count=deep_profiler_run_count,
            notes=(
                "Skipped by tool.skip_deep_profiles=true."
                if arguments.skip_deep_profiles
                else f"{deep_profiler_mode_count} profiler mode(s) per expected g winner."
            ),
        ),
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.LOGGING_PERTURBATION,
            candidate_count=logging_run_count,
            subprocess_run_count=logging_run_count,
            notes=(
                "Disabled by tool.enable_logging_perturbation=false."
                if not arguments.enable_logging_perturbation
                else f"{logging_case_count} logging case(s) per expected g winner."
            ),
        ),
        build_campaign_budget_section(
            section_name=CampaignBudgetSectionName.RUST_CRITERION,
            candidate_count=rust_benchmark_count,
            subprocess_run_count=rust_benchmark_count,
            major_profiler_run_count=rust_benchmark_count,
            notes=(
                "Skipped because Rust Criterion is disabled or tool.skip_deep_profiles=true."
                if rust_benchmark_count == 0
                else "Each configured Criterion benchmark is one cargo bench subprocess."
            ),
        ),
    )
    total_candidate_count = sum(section.candidate_count for section in sections)
    total_subprocess_run_count = sum(section.subprocess_run_count for section in sections)
    total_major_profiler_run_count = sum(section.major_profiler_run_count for section in sections)
    over_subprocess_budget = (
        arguments.max_subprocess_runs is not None and total_subprocess_run_count > arguments.max_subprocess_runs
    )
    over_major_profiler_budget = (
        arguments.max_major_profiler_runs is not None
        and total_major_profiler_run_count > arguments.max_major_profiler_runs
    )
    guidance = (
        "Run a dry run first and inspect profile_plan.md for the section counts.",
        "Reduce tool.workload_keys, tool.top_bgen_candidates, tool.top_finalists, trial counts, "
        "Firth batch sizes, writer counts, BGEN tile values, or Rayon thread counts to fit the budget.",
        "For an intentional huge campaign, pass tool.allow_over_budget=true and keep the run on an appropriate "
        "SLURM node.",
    )
    return CampaignBudget(
        workload_keys=tuple(workload_key.value for workload_key in workload_keys),
        max_subprocess_runs=arguments.max_subprocess_runs,
        max_major_profiler_runs=arguments.max_major_profiler_runs,
        total_candidate_count=total_candidate_count,
        total_subprocess_run_count=total_subprocess_run_count,
        total_major_profiler_run_count=total_major_profiler_run_count,
        over_subprocess_budget=over_subprocess_budget,
        over_major_profiler_budget=over_major_profiler_budget,
        sections=sections,
        guidance=guidance,
    )


def log_campaign_budget(campaign_budget: CampaignBudget) -> None:
    """Log section-level campaign budget estimates."""
    logger.info(
        "Estimated campaign budget: candidates=%s subprocess_runs=%s major_profiler_runs=%s",
        campaign_budget.total_candidate_count,
        campaign_budget.total_subprocess_run_count,
        campaign_budget.total_major_profiler_run_count,
    )
    for section in campaign_budget.sections:
        logger.info(
            "Budget section %s: candidates=%s subprocess_runs=%s major_profiler_runs=%s",
            section.display_name,
            section.candidate_count,
            section.subprocess_run_count,
            section.major_profiler_run_count,
        )


def enforce_campaign_budget(arguments: ProfileArguments, campaign_budget: CampaignBudget) -> None:
    """Fail early when a non-dry-run campaign exceeds the configured budget."""
    if arguments.allow_over_budget or not campaign_budget_is_over_limit(campaign_budget):
        return
    budget_messages = [
        "Deep profile campaign exceeds the configured budget.",
        (
            f"Estimated subprocess runs: {campaign_budget.total_subprocess_run_count} "
            f"(limit: {campaign_budget.max_subprocess_runs})."
        ),
        (
            f"Estimated major profiler runs: {campaign_budget.total_major_profiler_run_count} "
            f"(limit: {campaign_budget.max_major_profiler_runs})."
        ),
        "Section counts:",
    ]
    for section in campaign_budget.sections:
        budget_messages.append(
            f"- {section.display_name}: candidates={section.candidate_count}, "
            f"subprocess_runs={section.subprocess_run_count}, "
            f"major_profiler_runs={section.major_profiler_run_count}"
        )
    budget_messages.extend(campaign_budget.guidance)
    raise ValueError("\n".join(budget_messages))


def resolve_profile_jax_cache_directory(candidate: Step2Candidate, cache_directory: Path | None) -> Path | None:
    """Resolve the actual JAX cache directory used by one profile child."""
    if cache_directory is None:
        return None
    if candidate.device != "gpu":
        return cache_directory
    job_identifier = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    gpu_cache_parent = os.environ.get("G_PROFILE_GPU_JAX_CACHE_PARENT", GPU_JAX_CACHE_PARENT_DEFAULT)
    return Path(gpu_cache_parent) / job_identifier / cache_directory.name


def collect_jax_cache_snapshot(cache_directory: Path | None) -> JaxCacheSnapshot | None:
    """Collect lightweight file-count and byte-size stats for a JAX cache directory."""
    if cache_directory is None:
        return None
    resolved_cache_directory = cache_directory.expanduser()
    if not resolved_cache_directory.exists():
        return JaxCacheSnapshot(
            path=str(resolved_cache_directory),
            exists=False,
            file_count=0,
            total_size_bytes=0,
        )
    file_count = 0
    total_size_bytes = 0
    try:
        for cache_path in resolved_cache_directory.rglob("*"):
            if not cache_path.is_file():
                continue
            file_count += 1
            total_size_bytes += cache_path.stat().st_size
    except OSError as error:
        return JaxCacheSnapshot(
            path=str(resolved_cache_directory),
            exists=True,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            error=str(error),
        )
    return JaxCacheSnapshot(
        path=str(resolved_cache_directory),
        exists=True,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
    )


def parse_jax_compile_log(log_text: str) -> JaxCompileLogSummary:
    """Parse supported JAX compile and persistent-cache log lines from stderr text."""
    compilation_event_count = 0
    persistent_cache_event_count = 0
    persistent_cache_hit_count = 0
    persistent_cache_miss_count = 0
    tracing_cache_miss_count = 0
    cache_miss_explanation_count = 0
    sample_log_lines: list[str] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        lower_line = line.lower()
        has_compilation_event = (
            "compiling " in lower_line
            or "finished xla compilation" in lower_line
            or "tracing + transforming" in lower_line
            or "lowering " in lower_line
        )
        has_persistent_cache_event = "persistent compilation cache" in lower_line
        has_persistent_cache_hit = has_persistent_cache_event and (
            "cache hit" in lower_line or "cache_hit" in lower_line
        )
        has_persistent_cache_miss = has_persistent_cache_event and (
            "cache miss" in lower_line or "cache_miss" in lower_line or "not found" in lower_line
        )
        has_tracing_cache_miss = "tracing cache miss" in lower_line
        if has_compilation_event:
            compilation_event_count += 1
        if has_persistent_cache_event:
            persistent_cache_event_count += 1
        if has_persistent_cache_hit:
            persistent_cache_hit_count += 1
        if has_persistent_cache_miss:
            persistent_cache_miss_count += 1
        if has_tracing_cache_miss:
            tracing_cache_miss_count += 1
        if (has_persistent_cache_miss or has_tracing_cache_miss) and (
            "because" in lower_line or "explain" in lower_line
        ):
            cache_miss_explanation_count += 1
        if (
            has_compilation_event
            or has_persistent_cache_event
            or has_persistent_cache_hit
            or has_persistent_cache_miss
            or has_tracing_cache_miss
        ) and len(sample_log_lines) < JAX_LOG_SAMPLE_LINE_LIMIT:
            sample_log_lines.append(line[:500])
    return JaxCompileLogSummary(
        compilation_event_count=compilation_event_count,
        persistent_cache_event_count=persistent_cache_event_count,
        persistent_cache_hit_count=persistent_cache_hit_count,
        persistent_cache_miss_count=persistent_cache_miss_count,
        tracing_cache_miss_count=tracing_cache_miss_count,
        cache_miss_explanation_count=cache_miss_explanation_count,
        sample_log_lines=sample_log_lines,
    )


def read_jax_compile_log_summary(stderr_log_path: str) -> JaxCompileLogSummary:
    """Read a subprocess stderr file and parse JAX compile/cache log counters."""
    log_path = Path(stderr_log_path)
    if not log_path.exists():
        return parse_jax_compile_log("")
    return parse_jax_compile_log(log_path.read_text(encoding="utf-8", errors="replace"))


def snapshot_delta(
    *,
    before_snapshot: JaxCacheSnapshot | None,
    after_snapshot: JaxCacheSnapshot | None,
    field_name: str,
) -> int | None:
    """Return an integer delta for a cache snapshot field."""
    if before_snapshot is None or after_snapshot is None:
        return None
    before_value = getattr(before_snapshot, field_name)
    after_value = getattr(after_snapshot, field_name)
    if not isinstance(before_value, int) or not isinstance(after_value, int):
        return None
    return after_value - before_value


def build_jax_cache_diagnostics(
    *,
    cache_directory: Path | None,
    child_reported_cache_directory: str | None,
    persistent_cache_used: bool,
    before_snapshot: JaxCacheSnapshot | None,
    after_snapshot: JaxCacheSnapshot | None,
    stderr_log_path: str,
) -> JaxCacheDiagnostics:
    """Build one subprocess JAX cache diagnostic payload."""
    return JaxCacheDiagnostics(
        cache_directory=str(cache_directory) if cache_directory is not None else None,
        child_reported_cache_directory=child_reported_cache_directory,
        persistent_cache_used=persistent_cache_used,
        before=before_snapshot,
        after=after_snapshot,
        file_count_delta=snapshot_delta(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            field_name="file_count",
        ),
        size_bytes_delta=snapshot_delta(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            field_name="total_size_bytes",
        ),
        compile_log_summary=read_jax_compile_log_summary(stderr_log_path),
    )


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
    phenotype_path = baseline_paths.continuous_phenotype_path
    phenotype_name = "phenotype_continuous"
    prediction_path = baseline_paths.regenie_qt_prediction_list_path
    binary_options_expression = "{}"
    if candidate.trait_type == "binary":
        phenotype_path = baseline_paths.binary_phenotype_path
        phenotype_name = "phenotype_binary"
        prediction_path = baseline_paths.regenie_prediction_list_path
        binary_options_expression = '{"firth": True, "approx": True}'
    variant_limit_expression = "None" if variant_limit is None else str(variant_limit)
    jax_cache_directory = resolve_profile_jax_cache_directory(candidate, cache_directory)
    jax_cache_directory_expression = "None" if jax_cache_directory is None else repr(str(jax_cache_directory))
    bgen_tile_expression = (
        "64" if candidate.bgen_decode_tile_variant_count is None else str(candidate.bgen_decode_tile_variant_count)
    )
    firth_batch_expression = "1024" if candidate.firth_batch_size is None else str(candidate.firth_batch_size)
    rayon_thread_expression = "None" if candidate.rayon_thread_count is None else str(candidate.rayon_thread_count)
    diagnostic_options_expression = repr(diagnostic_options or {})
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
            artifacts = api.regenie.from_options({{
                "step": 2,
                "bt" if {trait_type!r} == "binary" else "qt": True,
                "bgen": {bgen_path!r},
                "sample": {sample_path!r},
                "phenoFile": {phenotype_path!r},
                "phenoCol": {phenotype_name!r},
                "out": {output_prefix!r},
                "covarFile": {covariate_path!r},
                "covarColList": "age,sex",
                "pred": {prediction_path!r},
                "g-device": {device!r},
                "bsize": {chunk_size},
                "g-variant-limit": {variant_limit_expression},
                "g-staging-depth": {staging_depth},
                "g-output-format": "parquet",
                "g-writer-threads": {writer_thread_count},
                "g-writer-queue-depth": {writer_queue_depth},
                "g-bgen-decode-tile-variant-count": {bgen_tile_expression},
                "g-firth-batch-size": {firth_batch_expression},
                "threads": {rayon_thread_expression},
                "g-jax-cache-dir": {jax_cache_directory_expression},
                "g-jax-persistent-cache": True,
                "g-jax-persistent-cache-min-entry-size-bytes": -1,
                "g-jax-persistent-cache-min-compile-time-seconds": 0,
                "g-jax-xla-autotune-cache": {enable_xla_autotune_cache},
                "g-stage-timings-json": {stage_timing_path!r},
                **{diagnostic_options_expression},
                **{binary_options_expression},
            }})
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
                "jax_cache_directory": {jax_cache_directory_expression},
                "jax_persistent_cache_used": True,
            }}))
        finally:
            if trace_directory is not None:
                jax.profiler.stop_trace()
        """
    ).format(
        trace_directory=str(trace_directory) if trace_directory is not None else None,
        memory_profile_path=str(memory_profile_path) if memory_profile_path is not None else None,
        bgen_path=str(baseline_paths.bgen_path),
        sample_path=str(baseline_paths.sample_path),
        phenotype_path=str(phenotype_path),
        phenotype_name=phenotype_name,
        output_prefix=str(output_prefix),
        covariate_path=str(baseline_paths.covariate_path),
        prediction_path=str(prediction_path),
        trait_type=candidate.trait_type,
        device=candidate.device,
        chunk_size=candidate.chunk_size,
        variant_limit_expression=variant_limit_expression,
        staging_depth=candidate.staging_depth,
        writer_thread_count=candidate.output_writer_thread_count,
        writer_queue_depth=candidate.output_writer_queue_depth,
        bgen_tile_expression=bgen_tile_expression,
        firth_batch_expression=firth_batch_expression,
        rayon_thread_expression=rayon_thread_expression,
        jax_cache_directory_expression=jax_cache_directory_expression,
        enable_xla_autotune_cache=ENABLE_XLA_AUTOTUNE_CACHE,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        diagnostic_options_expression=diagnostic_options_expression,
        binary_options_expression=binary_options_expression,
    )
    return [sys.executable, "-c", child_code]


def build_application_output_run_directory(output_prefix: Path) -> Path:
    """Return the default g output run directory for an output prefix."""
    return output_prefix.with_name(f"{output_prefix.name}.g")


def run_logged_command(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    environment_overrides: dict[str, str],
    log_directory: Path,
) -> TrialResult:
    """Run one command and persist stdout/stderr logs."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_directory / f"{name}.stdout.log"
    stderr_log_path = log_directory / f"{name}.stderr.log"
    environment = dict(os.environ)
    environment.update(environment_overrides)
    logger.info("Starting %s profiler/workload command", name)
    logger.debug("Command for %s: %s", name, shlex.join(command_arguments))
    start_time = time.perf_counter()
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    wall_time_seconds = time.perf_counter() - start_time
    stdout_log_path.write_text(completed_process.stdout, encoding="utf-8")
    stderr_log_path.write_text(completed_process.stderr, encoding="utf-8")
    status = "success" if completed_process.returncode == 0 else "failed"
    notes = None
    if completed_process.returncode != 0:
        notes = completed_process.stderr.strip() or completed_process.stdout.strip()
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


def build_deep_profiler_run_paths(*, profile_directory: Path, profile_name: str) -> DeepProfilerRunPaths:
    """Build isolated application paths for one deep profiler implementation."""
    application_output_prefix = profile_directory / profile_name
    return DeepProfilerRunPaths(
        application_output_prefix=application_output_prefix,
        application_output_run_directory=build_application_output_run_directory(application_output_prefix),
        stage_timing_path=profile_directory / f"{profile_name}.stage_timings.json",
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
) -> DeepProfilerChildCommand:
    """Build an isolated child command for one deep profiler implementation."""
    run_paths = build_deep_profiler_run_paths(profile_directory=profile_directory, profile_name=profile_name)
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
        stage_timing_path=str(run_paths.stage_timing_path),
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


def successful_trials_with_jax_diagnostics(trials: list[TrialResult]) -> list[TrialResult]:
    """Return successful trials that include JAX cache diagnostics."""
    return [
        trial
        for trial in trials
        if trial.status == "success" and trial.wall_time_seconds is not None and trial.jax_cache_diagnostics is not None
    ]


def sum_optional_integer_values(values: typing.Iterable[int | None]) -> int | None:
    """Sum integer values when at least one value is available."""
    observed_values = [value for value in values if value is not None]
    if not observed_values:
        return None
    return sum(observed_values)


def compile_log_summary_for_trial(trial: TrialResult) -> JaxCompileLogSummary:
    """Return the parsed JAX compile log summary for a diagnostic trial."""
    if trial.jax_cache_diagnostics is None:
        return parse_jax_compile_log("")
    return trial.jax_cache_diagnostics.compile_log_summary


def build_jax_cold_warm_diagnostics(
    *,
    warmup_trials: list[TrialResult],
    trial_results: list[TrialResult],
) -> JaxColdWarmDiagnostics | None:
    """Build cold-versus-warm JAX diagnostics for one aggregate result."""
    successful_trials = successful_trials_with_jax_diagnostics([*warmup_trials, *trial_results])
    if not successful_trials:
        return None
    successful_diagnostics = [
        trial.jax_cache_diagnostics for trial in successful_trials if trial.jax_cache_diagnostics is not None
    ]
    cold_trial = successful_trials[0]
    warm_trials = successful_trials[1:]
    cold_diagnostics = successful_diagnostics[0]
    warm_wall_times = [trial.wall_time_seconds for trial in warm_trials if trial.wall_time_seconds is not None]
    warm_median_wall_time = statistics.median(warm_wall_times) if warm_wall_times else None
    warm_mean_wall_time = statistics.fmean(warm_wall_times) if warm_wall_times else None
    cold_to_warm_speedup_ratio = None
    if cold_trial.wall_time_seconds is not None and warm_median_wall_time is not None and warm_median_wall_time > 0.0:
        cold_to_warm_speedup_ratio = cold_trial.wall_time_seconds / warm_median_wall_time
    warm_diagnostics = [trial.jax_cache_diagnostics for trial in warm_trials if trial.jax_cache_diagnostics is not None]
    return JaxColdWarmDiagnostics(
        cache_directory=cold_diagnostics.cache_directory,
        persistent_cache_used=any(diagnostic.persistent_cache_used for diagnostic in successful_diagnostics),
        cold_trial_name=cold_trial.name,
        cold_wall_time_seconds=cold_trial.wall_time_seconds,
        warm_trial_count=len(warm_trials),
        warm_median_wall_time_seconds=warm_median_wall_time,
        warm_mean_wall_time_seconds=warm_mean_wall_time,
        cold_to_warm_speedup_ratio=cold_to_warm_speedup_ratio,
        cold_cache_file_count_delta=cold_diagnostics.file_count_delta,
        warm_cache_file_count_delta=sum_optional_integer_values(
            diagnostic.file_count_delta for diagnostic in warm_diagnostics
        ),
        cold_cache_size_bytes_delta=cold_diagnostics.size_bytes_delta,
        warm_cache_size_bytes_delta=sum_optional_integer_values(
            diagnostic.size_bytes_delta for diagnostic in warm_diagnostics
        ),
        cold_compilation_event_count=compile_log_summary_for_trial(cold_trial).compilation_event_count,
        warm_compilation_event_count=sum(
            compile_log_summary_for_trial(trial).compilation_event_count for trial in warm_trials
        ),
        cold_cache_hit_count=compile_log_summary_for_trial(cold_trial).persistent_cache_hit_count,
        warm_cache_hit_count=sum(
            compile_log_summary_for_trial(trial).persistent_cache_hit_count for trial in warm_trials
        ),
        cold_cache_miss_count=compile_log_summary_for_trial(cold_trial).persistent_cache_miss_count,
        warm_cache_miss_count=sum(
            compile_log_summary_for_trial(trial).persistent_cache_miss_count for trial in warm_trials
        ),
        cold_tracing_cache_miss_count=compile_log_summary_for_trial(cold_trial).tracing_cache_miss_count,
        warm_tracing_cache_miss_count=sum(
            compile_log_summary_for_trial(trial).tracing_cache_miss_count for trial in warm_trials
        ),
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
) -> dict[str, AggregateResult]:
    """Tune g candidates for each trait/device and return winners."""
    winners: dict[str, AggregateResult] = {}
    chunk_sizes = parse_int_list(arguments.chunk_sizes)
    staging_depths = parse_int_list(arguments.staging_depths)
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
                    emit_stage_timings=True,
                )
            )
        if finalist_results:
            winner = sorted(
                finalist_results,
                key=lambda result: typing.cast("float", result.median_wall_time_seconds),
            )[0]
            winners[workload_key.value] = winner
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
    return winners


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
                emit_stage_timings=True,
            )
        )
    return headline_results


def candidate_from_aggregate_name(winner_key: str, aggregate_result: AggregateResult) -> Step2Candidate:
    """Reconstruct a winner candidate from its child environment and command."""
    trial = aggregate_result.trials[0]
    trait_type, device = winner_key.rsplit("_", maxsplit=1)
    command = trial.command_arguments
    code = command[2] if len(command) >= 3 and command[1] == "-c" else ""

    def read_int(marker: str, default_value: int) -> int:
        marker_index = code.find(marker)
        if marker_index < 0:
            return default_value
        value_start = marker_index + len(marker)
        value_end = code.find(",", value_start)
        return int(code[value_start:value_end].strip())

    return Step2Candidate(
        trait_type=trait_type,
        device=device,
        chunk_size=read_int('"bsize": ', 8192),
        staging_depth=read_int('"g-staging-depth": ', 1),
        output_writer_thread_count=read_int('"g-writer-threads": ', 8),
        output_writer_queue_depth=read_int('"g-writer-queue-depth": ', 4),
        bgen_decode_tile_variant_count=read_int('"g-bgen-decode-tile-variant-count": ', 64),
        rayon_thread_count=read_int('"threads": ', 0) or None,
        firth_batch_size=read_int('"g-firth-batch-size": ', 1024),
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


def collect_jax_cache_diagnostics(aggregate_results: list[AggregateResult]) -> dict[str, dict[str, object]]:
    """Collect aggregate JAX cache diagnostics keyed by result name."""
    diagnostics: dict[str, dict[str, object]] = {}
    for aggregate_result in aggregate_results:
        if aggregate_result.jax_cold_warm_summary is None:
            continue
        diagnostics[aggregate_result.name] = dataclasses.asdict(aggregate_result.jax_cold_warm_summary)
    return diagnostics


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
                emit_stage_timings=True,
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


def build_logging_perturbation_cases(
    *,
    output_directory: Path,
    smoke: bool,
) -> tuple[LoggingPerturbationCase, ...]:
    """Build telemetry/logging perturbation cases for representative winners."""
    perturbation_directory = output_directory / "logging_perturbation"
    cases = (
        LoggingPerturbationCase(
            name="telemetry_off",
            diagnostic_options={
                "g-telemetry": "off",
                "g-log-stderr": False,
            },
        ),
        LoggingPerturbationCase(
            name="progress_file_lossy",
            diagnostic_options={
                "g-telemetry": "progress",
                "g-log-dir": str(perturbation_directory / "progress_file_lossy_logs"),
                "g-log-stderr": False,
                "g-log-lossy": True,
                "g-log-queue-size": 8192,
            },
        ),
        LoggingPerturbationCase(
            name="profile_file_lossy",
            diagnostic_options={
                "g-telemetry": "profile",
                "g-log-dir": str(perturbation_directory / "profile_file_lossy_logs"),
                "g-log-stderr": False,
                "g-log-lossy": True,
                "g-log-queue-size": 8192,
            },
        ),
        LoggingPerturbationCase(
            name="trace_file_lossy_capped",
            diagnostic_options={
                "g-telemetry": "trace",
                "g-log-dir": str(perturbation_directory / "trace_file_lossy_capped_logs"),
                "g-log-stderr": False,
                "g-log-lossy": True,
                "g-log-queue-size": 8192,
                "g-trace-event-cap": 100_000,
            },
        ),
    )
    if smoke:
        return cases[:2]
    return cases


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
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = candidate_from_aggregate_name(winner_key, winner)
        for perturbation_case in build_logging_perturbation_cases(
            output_directory=output_directory,
            smoke=arguments.smoke,
        ):
            diagnostic_options = dict(perturbation_case.diagnostic_options)
            if diagnostic_options.get("g-telemetry") != "off":
                diagnostic_options["g-log-dir"] = str(
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
                emit_stage_timings=True,
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


def apply_smoke_overrides(arguments: ProfileArguments) -> ProfileArguments:
    """Reduce the campaign size for a landau smoke profile."""
    if not arguments.smoke:
        return arguments
    return dataclasses.replace(
        arguments,
        variant_limit=1000 if arguments.variant_limit is None else arguments.variant_limit,
        chunk_sizes="2048",
        staging_depths="1",
        output_writer_thread_counts="1",
        writer_queue_depth_multipliers="1",
        firth_batch_sizes="32",
        bgen_decode_tile_variant_counts="64",
        rayon_thread_counts="1",
        top_bgen_candidates=1,
        top_finalists=1,
        tuning_warmups=0,
        tuning_trials=1,
        finalist_warmups=0,
        finalist_trials=1,
        headline_warmups=0,
        headline_trials=1,
        regenie_baseline_warmups=0,
        regenie_baseline_trials=1,
    )


def build_arguments_from_config(config: omegaconf.DictConfig) -> ProfileArguments:
    """Build profile parameters from a composed Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    data_directory = resolve_repo_path(tool_values["data_dir"])
    output_parent = resolve_repo_path(tool_values.get("output_parent", DEFAULT_OUTPUT_PARENT))
    explicit_output_directory = tooling_hydra_arguments.path_or_none(tool_values.get("output_dir"))
    if explicit_output_directory is not None:
        explicit_output_directory = tooling_paths.resolve_repo_relative_path(
            explicit_output_directory,
            REPOSITORY_ROOT,
        )
        output_parent = explicit_output_directory.parent
    return ProfileArguments(
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
        enable_rust_criterion=bool(tool_values["enable_rust_criterion"]),
        enable_logging_perturbation=bool(tool_values["enable_logging_perturbation"]),
        rust_benchmarks=tooling_hydra_arguments.comma_join(tool_values["rust_benchmarks"]),
        chunk_sizes=tooling_hydra_arguments.comma_join(tool_values["chunk_sizes"]),
        staging_depths=tooling_hydra_arguments.comma_join(tool_values["staging_depths"]),
        output_writer_thread_counts=tooling_hydra_arguments.comma_join(tool_values["output_writer_thread_counts"]),
        writer_queue_depth_multipliers=tooling_hydra_arguments.comma_join(
            tool_values["writer_queue_depth_multipliers"]
        ),
        firth_batch_sizes=tooling_hydra_arguments.comma_join(tool_values["firth_batch_sizes"]),
        bgen_decode_tile_variant_counts=tooling_hydra_arguments.comma_join(
            tool_values["bgen_decode_tile_variant_counts"]
        ),
        rayon_thread_counts=tooling_hydra_arguments.comma_join(tool_values["rayon_thread_counts"]),
        bgen_benchmark_chunk_size=int(tool_values["bgen_benchmark_chunk_size"]),
        top_bgen_candidates=int(tool_values["top_bgen_candidates"]),
        top_finalists=int(tool_values["top_finalists"]),
        tuning_warmups=int(tool_values["tuning_warmups"]),
        tuning_trials=int(tool_values["tuning_trials"]),
        finalist_warmups=int(tool_values["finalist_warmups"]),
        finalist_trials=int(tool_values["finalist_trials"]),
        headline_warmups=int(tool_values["headline_warmups"]),
        headline_trials=int(tool_values["headline_trials"]),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> ProfileArguments:
    """Compose the deep-profile config and return resolved parameters."""
    config = tooling_configuration.compose_config(config_name="profile_regenie2_deep", overrides=overrides)
    return build_arguments_from_config(config)


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


def run_tool(arguments: ProfileArguments) -> None:
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
    winners = run_candidate_tuning(
        arguments=arguments,
        baseline_paths=baseline_paths,
        bgen_summaries=bgen_summaries,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
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
    summary_payload = {
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
        "deep_profiles": deep_profile_results,
        "logging_perturbation_results": logging_perturbation_results,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    (output_directory / "summary.md").write_text(
        build_summary_markdown(
            aggregate_results=headline_results,
            comparisons=comparisons,
            comparison_notes=comparison_notes,
            regenie_baseline_scope=regenie_baseline_scope,
            stage_totals=stage_totals,
            stage_comparison_rows=stage_comparison_rows,
            algorithmic_findings=algorithmic_findings,
            logging_perturbation_results=logging_perturbation_results,
        ),
        encoding="utf-8",
    )
    write_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
    )
    logger.info("Wrote deep profile artifacts under %s", output_directory)


@hydra.main(version_base=None, config_path="../configs", config_name="profile_regenie2_deep")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the deep profiling campaign through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the landau deep profiling campaign."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
