#!/usr/bin/env python3
"""Benchmark an already-compiled approximate-Firth JAX executable."""

from __future__ import annotations

import dataclasses
import enum
import gzip
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import time
import typing
from pathlib import Path

import hydra
import jax
import jaxlib
import numpy as np
import numpy.typing as npt

import tooling.configuration as tooling_configuration
from g.compute import cuda_ffi
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary.firth.batch import compute as firth_batch_compute
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import omegaconf

SUMMARY_SCHEMA_VERSION = 0
DEFAULT_OUTPUT_PARENT = Path("data/profiles")


class BenchmarkDevice(enum.StrEnum):
    """Logical execution device accepted by benchmark configuration."""

    CPU = "cpu"
    GPU = "gpu"


class BenchmarkImplementation(enum.StrEnum):
    """Approximate-Firth component implementation measured by the executable."""

    JAX = "jax"
    RAW_CUDA = "raw_cuda"


class CudaFfiTestSupport(typing.Protocol):
    """Feature-gated native registration used by focused CUDA evidence."""

    def register_firth_components_ffi(self) -> str:
        """Register and return the raw-CUDA Firth typed-XLA target."""


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved focused Firth benchmark settings."""

    device: BenchmarkDevice
    implementation: BenchmarkImplementation
    sample_count: int
    candidate_capacity: int
    firth_batch_size: int
    active_candidate_counts: tuple[int, ...]
    warmup_trial_count: int
    measured_trial_count: int
    trace_active_candidate_count: int | None
    output_directory: Path
    jax_cache_directory: Path


@dataclasses.dataclass(frozen=True)
class BenchmarkInputs:
    """Fixed-capacity device inputs shared by all active-count cases."""

    null_firth_offset_matrix: jax.Array
    phenotype_matrix: jax.Array
    flat_trait_indices: jax.Array
    genotype_matrix_by_variant: jax.Array
    carrier_sample_mask: jax.Array
    full_null_deviance: jax.Array
    sparse_correction_mask: jax.Array
    null_failed_mask: jax.Array


@dataclasses.dataclass(frozen=True)
class HostBenchmarkInputs:
    """Deterministic host fixture transferred before timing."""

    null_firth_offset_matrix: npt.NDArray[np.float64]
    phenotype_matrix: npt.NDArray[np.float64]
    flat_trait_indices: npt.NDArray[np.int32]
    genotype_matrix_by_variant: npt.NDArray[np.float32]
    carrier_sample_mask: npt.NDArray[np.bool_]
    full_null_deviance: npt.NDArray[np.float64]
    sparse_correction_mask: npt.NDArray[np.bool_]
    null_failed_mask: npt.NDArray[np.bool_]


@dataclasses.dataclass(frozen=True)
class CacheSnapshot:
    """Stable summary of one persistent JAX cache tree."""

    file_count: int
    total_size_bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ExecutableEvidence:
    """Compile, IR, and memory evidence for the fixed-capacity executable."""

    lowering_seconds: float
    compilation_seconds: float
    stablehlo_text_bytes: int
    stablehlo_sha256: str
    executable_text_bytes: int | None
    executable_text_sha256: str | None
    memory_analysis: dict[str, int] | None


@dataclasses.dataclass(frozen=True)
class CompiledBenchmark:
    """Compiled callable and its durable evidence."""

    executable: jax.stages.Compiled
    evidence: ExecutableEvidence


@dataclasses.dataclass(frozen=True)
class ResultDigest:
    """Correctness digest and valid-lane count for one result."""

    sha256: str
    valid_result_count: int


@dataclasses.dataclass(frozen=True)
class DeviceEventSummary:
    """Aggregated count and duration for one device event name."""

    name: str
    event_count: int
    duration_milliseconds: float


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """Synchronized hot timings and correctness digest for one active count."""

    active_candidate_count: int
    elapsed_milliseconds: tuple[float, ...]
    median_milliseconds: float
    mean_milliseconds: float
    minimum_milliseconds: float
    maximum_milliseconds: float
    result_sha256: str
    valid_result_count: int


@dataclasses.dataclass(frozen=True)
class DeviceTraceSummary:
    """Device-event summary from one isolated post-timing JAX trace."""

    active_candidate_count: int
    trace_path: str
    device_event_count: int
    device_duration_milliseconds: float
    top_device_events: tuple[DeviceEventSummary, ...]


def default_output_directory() -> Path:
    """Return a timestamped ignored output directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_PARENT / f"firth_compute_{timestamp}_{os.getpid()}"


def _validate_summary_schema_version(payload: dict[str, typing.Any]) -> None:
    """Validate that CUDA qualification/report schema versions are strict integer zero."""
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError(
            f"schema_version for CUDA qualification report must be integer 0, got {schema_version!r}."
        )
    if schema_version != SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"Expected CUDA qualification schema_version={SUMMARY_SCHEMA_VERSION}, got {schema_version!r}."
        )


def validate_arguments(arguments: BenchmarkArguments) -> None:
    """Reject configurations that cannot represent the production executable."""
    if arguments.sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if arguments.implementation is BenchmarkImplementation.RAW_CUDA and arguments.device is not BenchmarkDevice.GPU:
        raise ValueError("raw_cuda Firth components require device=gpu.")
    if arguments.candidate_capacity <= 0:
        raise ValueError("candidate_capacity must be positive.")
    if arguments.firth_batch_size <= 0:
        raise ValueError("firth_batch_size must be positive.")
    if arguments.candidate_capacity % arguments.firth_batch_size != 0:
        raise ValueError("candidate_capacity must be divisible by firth_batch_size.")
    if arguments.warmup_trial_count <= 0:
        raise ValueError("warmup_trial_count must be positive.")
    if arguments.measured_trial_count <= 0:
        raise ValueError("measured_trial_count must be positive.")
    if not arguments.active_candidate_counts:
        raise ValueError("active_candidate_counts must not be empty.")
    invalid_counts = [
        count for count in arguments.active_candidate_counts if count <= 0 or count > arguments.candidate_capacity
    ]
    if invalid_counts:
        raise ValueError(f"active candidate counts must be in [1, {arguments.candidate_capacity}]: {invalid_counts}.")
    if (
        arguments.trace_active_candidate_count is not None
        and arguments.trace_active_candidate_count not in arguments.active_candidate_counts
    ):
        raise ValueError("trace_active_candidate_count must be one of active_candidate_counts.")


def jax_platform_name(device: BenchmarkDevice) -> str:
    """Translate logical machine devices to concrete JAX platform names."""
    if device is BenchmarkDevice.GPU:
        return "cuda"
    return device.value


def build_kernel_config(arguments: BenchmarkArguments) -> regenie2_binary_config.BinaryKernelConfig:
    """Build the production approximate-Firth kernel policy."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-6,
            minimum_variance=1.0e-8,
            relative_variance_tolerance=1.0e-6,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=50,
            coefficient_tolerance=1.0e-6,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=arguments.firth_batch_size,
            candidate_capacity=arguments.candidate_capacity,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=250,
            gradient_tolerance=2.5e-4,
            maximum_step_size=5.0,
            pseudo_maximum_iterations=50,
            pseudo_inner_maximum_iterations=25,
            line_search_maximum_attempts=25,
            sparse_carrier_dosage_threshold=1.0e-4,
            use_cuda_components=arguments.implementation is BenchmarkImplementation.RAW_CUDA,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=1_000,
            gradient_tolerance=50.0e-6,
            maximum_step_size=25.0,
            fallback_iteration_multiplier=5,
            fallback_step_divisor=5.0,
            line_search_maximum_attempts=25,
            step_halving_scale=0.5,
        ),
    )


def build_host_inputs(arguments: BenchmarkArguments) -> HostBenchmarkInputs:
    """Build deterministic dense candidate inputs without random generators."""
    sample_indices = np.arange(arguments.sample_count, dtype=np.int64)[None, :]
    candidate_indices = np.arange(arguments.candidate_capacity, dtype=np.int64)[:, None]
    phenotype = (((sample_indices * 17 + 11) % 101) < 43).astype(np.float64)
    first_allele = ((sample_indices * 13 + candidate_indices * 29 + 3) % 23 == 0).astype(np.float32)
    second_allele = ((sample_indices * 31 + candidate_indices * 7 + 5) % 47 == 0).astype(np.float32)
    genotype = first_allele + second_allele
    prevalence = float(np.mean(phenotype))
    offset_value = math.log(prevalence / (1.0 - prevalence))
    offset = np.full((1, arguments.sample_count), offset_value, dtype=np.float64)
    null_deviance = -2.0 * np.sum(
        phenotype * math.log(prevalence) + (1.0 - phenotype) * math.log(1.0 - prevalence),
        dtype=np.float64,
    )
    return HostBenchmarkInputs(
        null_firth_offset_matrix=offset,
        phenotype_matrix=phenotype,
        flat_trait_indices=np.zeros(arguments.candidate_capacity, dtype=np.int32),
        genotype_matrix_by_variant=genotype,
        carrier_sample_mask=genotype > 0.0,
        full_null_deviance=np.full(arguments.candidate_capacity, null_deviance, dtype=np.float64),
        sparse_correction_mask=np.zeros(arguments.candidate_capacity, dtype=np.bool_),
        null_failed_mask=np.zeros(arguments.candidate_capacity, dtype=np.bool_),
    )


def put_inputs_on_device(arguments: BenchmarkArguments) -> BenchmarkInputs:
    """Transfer the fixed input fixture once, outside every hot timing."""
    host_inputs = build_host_inputs(arguments)
    device_inputs = BenchmarkInputs(
        null_firth_offset_matrix=jax.device_put(host_inputs.null_firth_offset_matrix),
        phenotype_matrix=jax.device_put(host_inputs.phenotype_matrix),
        flat_trait_indices=jax.device_put(host_inputs.flat_trait_indices),
        genotype_matrix_by_variant=jax.device_put(host_inputs.genotype_matrix_by_variant),
        carrier_sample_mask=jax.device_put(host_inputs.carrier_sample_mask),
        full_null_deviance=jax.device_put(host_inputs.full_null_deviance),
        sparse_correction_mask=jax.device_put(host_inputs.sparse_correction_mask),
        null_failed_mask=jax.device_put(host_inputs.null_failed_mask),
    )
    jax.block_until_ready(dataclasses.astuple(device_inputs))
    return device_inputs


def build_active_mask(arguments: BenchmarkArguments, active_candidate_count: int) -> jax.Array:
    """Build one device-resident prefix mask for a dynamic fallback count."""
    return jax.device_put(np.arange(arguments.candidate_capacity, dtype=np.int32) < active_candidate_count)


def kernel_keyword_arguments(
    arguments: BenchmarkArguments,
    inputs: BenchmarkInputs,
    active_candidate_count: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> dict[str, typing.Any]:
    """Return one complete keyword mapping for the jitted dense Firth executable."""
    return {
        "null_firth_offset_matrix": inputs.null_firth_offset_matrix,
        "phenotype_matrix": inputs.phenotype_matrix,
        "flat_trait_indices": inputs.flat_trait_indices,
        "genotype_matrix_by_variant": inputs.genotype_matrix_by_variant,
        "carrier_sample_mask": inputs.carrier_sample_mask,
        "full_null_deviance": inputs.full_null_deviance,
        "sparse_correction_mask": inputs.sparse_correction_mask,
        "null_failed_mask": inputs.null_failed_mask,
        "active_mask": build_active_mask(arguments, active_candidate_count),
        "fallback_count": jax.device_put(np.asarray(active_candidate_count, dtype=np.int32)),
        "firth_batch_size": arguments.firth_batch_size,
        "kernel_config": kernel_config,
    }


def compiled_keyword_arguments(keyword_arguments: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Remove static lowering arguments from an ahead-of-time compiled call."""
    return {
        name: value for name, value in keyword_arguments.items() if name not in {"firth_batch_size", "kernel_config"}
    }


def snapshot_cache_tree(path: Path) -> CacheSnapshot:
    """Hash a persistent cache tree in relative-path order."""
    digest = hashlib.sha256()
    file_count = 0
    total_size_bytes = 0
    if path.exists():
        for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative_path = file_path.relative_to(path).as_posix()
            file_bytes = file_path.read_bytes()
            digest.update(relative_path.encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(file_bytes).digest())
            file_count += 1
            total_size_bytes += len(file_bytes)
    return CacheSnapshot(file_count=file_count, total_size_bytes=total_size_bytes, sha256=digest.hexdigest())


def memory_analysis_to_dictionary(memory_analysis: typing.Any) -> dict[str, int] | None:
    """Normalize JAX compiled-memory statistics without depending on repr text."""
    if memory_analysis is None:
        return None
    field_names = (
        "generated_code_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {field_name: int(getattr(memory_analysis, field_name)) for field_name in field_names}


def compile_executable(keyword_arguments: dict[str, typing.Any]) -> CompiledBenchmark:
    """Lower and compile once while retaining stable size and memory evidence."""
    started_at = time.perf_counter()
    lowered = firth_batch_compute.compute_scalar_firth_multi_variantwise_fixed_batches_without_sparse_compaction.lower(
        **keyword_arguments
    )
    lowering_seconds = time.perf_counter() - started_at
    stablehlo_text = lowered.as_text(dialect="stablehlo")
    started_at = time.perf_counter()
    compiled = lowered.compile()
    compilation_seconds = time.perf_counter() - started_at
    executable_text = compiled.as_text()
    return CompiledBenchmark(
        executable=compiled,
        evidence=ExecutableEvidence(
            lowering_seconds=lowering_seconds,
            compilation_seconds=compilation_seconds,
            stablehlo_text_bytes=len(stablehlo_text.encode()),
            stablehlo_sha256=hashlib.sha256(stablehlo_text.encode()).hexdigest(),
            executable_text_bytes=None if executable_text is None else len(executable_text.encode()),
            executable_text_sha256=(
                None if executable_text is None else hashlib.sha256(executable_text.encode()).hexdigest()
            ),
            memory_analysis=memory_analysis_to_dictionary(compiled.memory_analysis()),
        ),
    )


def digest_result(result: typing.Any, active_candidate_count: int) -> ResultDigest:
    """Hash all result leaves and count valid candidate outputs."""
    host_result = jax.device_get(result)
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(host_result):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    valid_mask = np.asarray(host_result.valid_mask)
    expected_valid_mask = np.arange(valid_mask.size) < active_candidate_count
    if not np.array_equal(valid_mask, expected_valid_mask):
        raise RuntimeError(f"Firth valid-mask oracle failed for {active_candidate_count} active candidates.")
    valid_result_count = int(np.count_nonzero(valid_mask))
    return ResultDigest(sha256=digest.hexdigest(), valid_result_count=valid_result_count)


def run_case(
    arguments: BenchmarkArguments,
    executable: jax.stages.Compiled,
    keyword_arguments: dict[str, typing.Any],
    active_candidate_count: int,
) -> CaseResult:
    """Measure synchronized executions after discarded warmups."""
    for _ in range(arguments.warmup_trial_count):
        jax.block_until_ready(executable(**keyword_arguments))
    elapsed_milliseconds: list[float] = []
    result: typing.Any = None
    for _ in range(arguments.measured_trial_count):
        started_at = time.perf_counter_ns()
        result = executable(**keyword_arguments)
        jax.block_until_ready(result)
        elapsed_milliseconds.append((time.perf_counter_ns() - started_at) / 1_000_000.0)
    result_digest = digest_result(result, active_candidate_count)
    return CaseResult(
        active_candidate_count=active_candidate_count,
        elapsed_milliseconds=tuple(elapsed_milliseconds),
        median_milliseconds=statistics.median(elapsed_milliseconds),
        mean_milliseconds=statistics.fmean(elapsed_milliseconds),
        minimum_milliseconds=min(elapsed_milliseconds),
        maximum_milliseconds=max(elapsed_milliseconds),
        result_sha256=result_digest.sha256,
        valid_result_count=result_digest.valid_result_count,
    )


def find_trace_path(trace_directory: Path) -> Path:
    """Return the single TensorBoard gzip trace below a fresh trace root."""
    trace_paths = sorted(trace_directory.rglob("*.trace.json.gz"))
    if len(trace_paths) != 1:
        raise RuntimeError(f"Expected one JAX trace under {trace_directory}, found {len(trace_paths)}.")
    return trace_paths[0]


def summarize_device_trace(trace_path: Path, active_candidate_count: int) -> DeviceTraceSummary:
    """Summarize duration events belonging to trace device processes."""
    with gzip.open(trace_path, "rt", encoding="utf-8") as trace_file:
        payload = json.load(trace_file)
    events = payload.get("traceEvents", [])
    device_process_identifiers = {
        event.get("pid")
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and str(event.get("args", {}).get("name", "")).startswith("/device:")
    }
    device_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("pid") in device_process_identifiers and float(event.get("dur", 0)) > 0
    ]
    event_totals: dict[str, tuple[int, float]] = {}
    for event in device_events:
        name = str(event.get("name", "<unknown>"))
        event_count, duration_microseconds = event_totals.get(name, (0, 0.0))
        event_totals[name] = (event_count + 1, duration_microseconds + float(event["dur"]))
    top_device_events = tuple(
        DeviceEventSummary(
            name=name,
            event_count=event_count,
            duration_milliseconds=duration_microseconds / 1_000.0,
        )
        for name, (event_count, duration_microseconds) in sorted(
            event_totals.items(), key=lambda item: item[1][1], reverse=True
        )[:20]
    )
    return DeviceTraceSummary(
        active_candidate_count=active_candidate_count,
        trace_path=str(trace_path),
        device_event_count=len(device_events),
        device_duration_milliseconds=sum(float(event["dur"]) for event in device_events) / 1_000.0,
        top_device_events=top_device_events,
    )


def capture_device_trace(
    arguments: BenchmarkArguments,
    executable: jax.stages.Compiled,
    keyword_arguments: dict[str, typing.Any],
    active_candidate_count: int,
) -> DeviceTraceSummary:
    """Capture one isolated execution with Python tracing disabled."""
    trace_directory = arguments.output_directory / "jax_trace"
    profile_options = jax.profiler.ProfileOptions()
    profile_options.python_tracer_level = 0
    jax.profiler.start_trace(trace_directory, profiler_options=profile_options)
    try:
        result = executable(**keyword_arguments)
        jax.block_until_ready(result)
    finally:
        jax.profiler.stop_trace()
    return summarize_device_trace(find_trace_path(trace_directory), active_candidate_count)


def collect_environment(arguments: BenchmarkArguments) -> dict[str, typing.Any]:
    """Collect runtime identity needed to interpret focused evidence."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "requested_device": arguments.device.value,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def register_firth_components_implementation(arguments: BenchmarkArguments) -> str | None:
    """Register the explicitly requested raw-CUDA implementation.

    The focused benchmark never falls back: a raw-CUDA request either registers
    the native target or fails before lowering.
    """
    if arguments.implementation is BenchmarkImplementation.JAX:
        return None
    try:
        test_support = typing.cast(
            "CudaFfiTestSupport",
            importlib.import_module("g._core._testing"),
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "raw_cuda benchmarking requires an extension built with the private-test-support Cargo feature."
        ) from error
    registered_target = test_support.register_firth_components_ffi()
    if registered_target != cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET:
        raise RuntimeError(
            "Native raw-CUDA registration returned an FFI target that does not match the Python call site."
        )
    return registered_target


def run_benchmark(arguments: BenchmarkArguments) -> dict[str, typing.Any]:
    """Compile once, measure all active counts, and capture one isolated trace."""
    validate_arguments(arguments)
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    arguments.jax_cache_directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - JAX requires the literal enable flag.
    jax.config.update("jax_platforms", jax_platform_name(arguments.device))
    jax.config.update("jax_compilation_cache_dir", str(arguments.jax_cache_directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    registered_ffi_target = register_firth_components_implementation(arguments)
    inputs = put_inputs_on_device(arguments)
    kernel_config = build_kernel_config(arguments)
    keyword_arguments_by_count = {
        active_candidate_count: kernel_keyword_arguments(
            arguments,
            inputs,
            active_candidate_count,
            kernel_config,
        )
        for active_candidate_count in arguments.active_candidate_counts
    }
    compiled_keyword_arguments_by_count = {
        active_candidate_count: compiled_keyword_arguments(keyword_arguments)
        for active_candidate_count, keyword_arguments in keyword_arguments_by_count.items()
    }
    cache_before = snapshot_cache_tree(arguments.jax_cache_directory)
    compiled_benchmark = compile_executable(keyword_arguments_by_count[arguments.active_candidate_counts[0]])
    cache_after_compile = snapshot_cache_tree(arguments.jax_cache_directory)
    case_results = [
        run_case(
            arguments,
            compiled_benchmark.executable,
            compiled_keyword_arguments_by_count[active_candidate_count],
            active_candidate_count,
        )
        for active_candidate_count in arguments.active_candidate_counts
    ]
    trace_summary = (
        None
        if arguments.trace_active_candidate_count is None
        else capture_device_trace(
            arguments,
            compiled_benchmark.executable,
            compiled_keyword_arguments_by_count[arguments.trace_active_candidate_count],
            arguments.trace_active_candidate_count,
        )
    )
    cache_after_measurement = snapshot_cache_tree(arguments.jax_cache_directory)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "benchmark": "dense_approximate_firth_fixed_capacity",
        "arguments": dataclasses.asdict(arguments),
        "implementation": {
            "requested": arguments.implementation.value,
            "effective": arguments.implementation.value,
            "registered_ffi_target": registered_ffi_target,
            "fallback": None,
        },
        "environment": collect_environment(arguments),
        "cache": {
            "before": dataclasses.asdict(cache_before),
            "after_compile": dataclasses.asdict(cache_after_compile),
            "after_measurement": dataclasses.asdict(cache_after_measurement),
            "measurement_tree_unchanged": cache_after_compile == cache_after_measurement,
        },
        "executable": dataclasses.asdict(compiled_benchmark.evidence),
        "cases": [dataclasses.asdict(case_result) for case_result in case_results],
        "device_trace": None if trace_summary is None else dataclasses.asdict(trace_summary),
    }


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Adapt Hydra configuration into the fixed benchmark contract."""
    values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    configured_output_directory = tooling_hydra_arguments.path_or_none(values.get("output_dir"))
    trace_active_candidate_count = tooling_hydra_arguments.integer_or_none(values.get("trace_active_candidate_count"))
    return BenchmarkArguments(
        device=BenchmarkDevice(str(values["device"])),
        implementation=BenchmarkImplementation(str(values["implementation"])),
        sample_count=int(values["sample_count"]),
        candidate_capacity=int(values["candidate_capacity"]),
        firth_batch_size=int(values["firth_batch_size"]),
        active_candidate_counts=tuple(int(value) for value in values["active_candidate_counts"]),
        warmup_trial_count=int(values["warmup_trial_count"]),
        measured_trial_count=int(values["measured_trial_count"]),
        trace_active_candidate_count=trace_active_candidate_count,
        output_directory=configured_output_directory or default_output_directory(),
        jax_cache_directory=Path(str(values["jax_cache_dir"])),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Compose focused Firth configuration and return resolved arguments."""
    config = tooling_configuration.compose_config(config_name="benchmark_firth_compute", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: BenchmarkArguments) -> Path:
    """Run the focused benchmark and write its versioned summary."""
    summary = run_benchmark(arguments)
    summary_path = arguments.output_directory / "summary.json"
    _validate_summary_schema_version(summary)
    tooling_reports.write_json_report(summary_path, summary, sort_keys=True)
    print(f"Wrote focused Firth evidence: {summary_path}")
    return summary_path


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_firth_compute")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the focused benchmark through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the already-compiled Firth compute benchmark."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
