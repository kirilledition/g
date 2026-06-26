"""Deep-profile dataclasses, enums, and constants."""

from __future__ import annotations

import dataclasses
import enum
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path


class ProfileStageTimingMode(enum.StrEnum):
    """Stage timing collection mode for deep profile runs."""

    EXACT = "exact"
    OFF = "off"


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
        py_spy_timeout_seconds: Timeout seconds for optional py-spy profiler execution.
        scalene_timeout_seconds: Timeout seconds for optional Scalene profiler execution.
        memray_timeout_seconds: Timeout seconds for optional Memray profiler execution.
        linux_perf_timeout_seconds: Timeout seconds for optional Linux perf execution.
        nsight_systems_timeout_seconds: Timeout seconds for optional Nsight Systems execution.
        nsight_compute_timeout_seconds: Timeout seconds for optional Nsight Compute execution.
        enable_rust_criterion: Whether deep profiles run Rust Criterion benches.
        enable_logging_perturbation: Whether the profile runs telemetry/logging perturbation trials.
        rust_benchmarks: Comma-separated Rust Criterion benchmark names.
        chunk_sizes: Comma-separated step 2 chunk-size values.
        staging_depths: Comma-separated staging-depth values.
        native_callback_batch_sizes: Comma-separated native callback batch sizes.
        result_in_flight_limits: Comma-separated result in-flight limits, or default.
        dosage_buffer_limits: Comma-separated dosage buffer limits, or default.
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
        stage_timing_mode: Whether exact stage timing JSON artifacts are emitted.

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
    py_spy_timeout_seconds: int
    scalene_timeout_seconds: int
    memray_timeout_seconds: int
    linux_perf_timeout_seconds: int
    nsight_systems_timeout_seconds: int
    nsight_compute_timeout_seconds: int
    enable_rust_criterion: bool
    enable_logging_perturbation: bool
    rust_benchmarks: str
    chunk_sizes: str
    staging_depths: str
    native_callback_batch_sizes: str
    result_in_flight_limits: str
    dosage_buffer_limits: str
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
    stage_timing_mode: ProfileStageTimingMode


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
    native_callback_batch_size: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
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
        stage_timing_path: Stage timing JSON path for the profiled child run when exact timing is enabled.
        profile_script_path: Python script path executed by the profiler wrapper.

    """

    application_output_prefix: Path
    application_output_run_directory: Path
    stage_timing_path: Path | None
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
class CandidateTuningResults:
    """Candidate tuning output retained for final reporting.

    Attributes:
        winners: Fastest finalist aggregate keyed by trait and device.
        finalist_results_by_key: All measured finalist aggregates keyed by trait and device.

    """

    winners: dict[str, AggregateResult]
    finalist_results_by_key: dict[str, list[AggregateResult]]


@dataclasses.dataclass(frozen=True)
class BinaryDiagnosticTrialPayload:
    """Loaded stage timing diagnostics for one binary trial.

    Attributes:
        trial_name: Trial name from the aggregate result.
        stage_timing_path: Stage timing JSON path when one was requested.
        unavailable_reason: Reason diagnostics could not be read.
        payload: Parsed stage timing payload when available.

    """

    trial_name: str
    stage_timing_path: str | None
    unavailable_reason: str | None
    payload: dict[str, typing.Any] | None


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
