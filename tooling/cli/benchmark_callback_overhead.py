#!/usr/bin/env python3
"""Benchmark native callback overhead without BGEN decode work."""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import statistics
import time
import typing

import hydra
import jax
import numpy as np
import numpy.typing as npt

import tooling.configuration as tooling_configuration
from g import types
from g.engine.callbacks import diagnostics as callback_diagnostics
from g.engine.callbacks import runtime as callback_runtime
from g.engine.callbacks import transfers as callback_transfers
from g.runner import timing
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import omegaconf

    from g import _core


class StageTimingMode(enum.StrEnum):
    """Stage timing mode exercised by the callback benchmark."""

    OFF = "off"
    AGGREGATE = "aggregate"
    EXACT = "exact"


class CallbackWorkloadMode(enum.StrEnum):
    """Synthetic callback workload independent of native BGEN decode."""

    QUEUE_ONLY = "queue_only"
    HOST_TO_DEVICE = "host_to_device"


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved callback-overhead benchmark settings.

    Attributes:
        device: JAX platform to request.
        chunk_count: Number of callback chunks per trial.
        sample_count: Simulated sample count in each chunk.
        variants_per_chunk: Simulated variant count in each chunk.
        staging_depth: Native callback queue depth.
        trials: Measured trials per case.
        warmup_trials: Warmup trials per case.
        stage_timing_modes: Comma-separated stage timing modes.
        workload_modes: Comma-separated workload modes.
        synchronize_after_trial: Whether to block on the last device transfer before stopping the timer.
        json_summary_path: Optional JSON summary path.

    """

    device: str
    chunk_count: int
    sample_count: int
    variants_per_chunk: int
    staging_depth: int
    trials: int
    warmup_trials: int
    stage_timing_modes: str
    workload_modes: str
    synchronize_after_trial: bool
    json_summary_path: pathlib.Path | None


@dataclasses.dataclass(frozen=True)
class BenchmarkChunkMetadata:
    """Minimal native metadata identity for one synthetic callback chunk.

    Attributes:
        chromosome: Chromosome label tuple matching native metadata shape.
        variant_start_index: Inclusive chunk start index.
        variant_stop_index: Exclusive chunk stop index.

    """

    chromosome: tuple[str, ...]
    variant_start_index: int
    variant_stop_index: int


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    """One callback benchmark case.

    Attributes:
        workload_mode: Synthetic workload mode.
        stage_timing_mode: Stage timing mode.

    """

    workload_mode: CallbackWorkloadMode
    stage_timing_mode: StageTimingMode


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measured callback-overhead result.

    Attributes:
        workload_mode: Synthetic workload mode.
        stage_timing_mode: Stage timing mode.
        trial_index: Measured trial index.
        chunk_count: Number of callback chunks in the trial.
        wall_time_seconds: Trial wall time.
        chunks_per_second: Processed chunks per second.
        nanoseconds_per_chunk: Mean wall time per chunk.
        requested_device: Requested JAX platform.
        jax_backend: Active JAX backend.
        jax_devices: Active JAX device descriptions.
        stage_totals_seconds: Aggregate stage timings recorded in this trial.
        stage_counts: Aggregate stage timing counts recorded in this trial.
        queue_observation_count: Number of queue/backpressure observations recorded in this trial.
        transfer_observation_count: Number of transfer metadata observations recorded in this trial.

    """

    workload_mode: CallbackWorkloadMode
    stage_timing_mode: StageTimingMode
    trial_index: int
    chunk_count: int
    wall_time_seconds: float
    chunks_per_second: float
    nanoseconds_per_chunk: float
    requested_device: str
    jax_backend: str
    jax_devices: tuple[str, ...]
    stage_totals_seconds: dict[str, float]
    stage_counts: dict[str, int]
    queue_observation_count: int
    transfer_observation_count: int


@dataclasses.dataclass(frozen=True)
class RecorderSummary:
    """Compact stage timing recorder summary for one trial.

    Attributes:
        stage_totals_seconds: Aggregate stage timings recorded in this trial.
        stage_counts: Aggregate stage timing counts recorded in this trial.
        queue_observation_count: Number of queue/backpressure observations.
        transfer_observation_count: Number of transfer metadata observations.

    """

    stage_totals_seconds: dict[str, float]
    stage_counts: dict[str, int]
    queue_observation_count: int
    transfer_observation_count: int


@dataclasses.dataclass(frozen=True)
class CaseSummary:
    """Aggregate callback-overhead summary for one benchmark case.

    Attributes:
        workload_mode: Synthetic workload mode.
        stage_timing_mode: Stage timing mode.
        trial_count: Number of measured trials.
        mean_wall_time_seconds: Mean wall time.
        stdev_wall_time_seconds: Sample standard deviation of wall time.
        mean_chunks_per_second: Mean chunk throughput.
        mean_nanoseconds_per_chunk: Mean per-chunk wall time.

    """

    workload_mode: CallbackWorkloadMode
    stage_timing_mode: StageTimingMode
    trial_count: int
    mean_wall_time_seconds: float
    stdev_wall_time_seconds: float
    mean_chunks_per_second: float
    mean_nanoseconds_per_chunk: float


class CallbackOverheadBenchmarkRunner(callback_runtime.NativeBgenCallbackRunner):
    """Concrete callback runner used by the synthetic benchmark."""

    def __init__(
        self,
        *,
        workload_mode: CallbackWorkloadMode,
        staging_depth: int,
        stage_timing_recorder: timing.StageTimingRecorder | None,
    ) -> None:
        """Initialize a callback benchmark runner."""
        super().__init__(
            worker_name="callback-overhead-benchmark",
            staging_depth=staging_depth,
            native_callback_batch_size=1,
            expected_result_work_item_kind=callback_runtime.ResultWriteItemKind.SINGLE_RESULT,
            flush_binary_correction_diagnostics_on_result_stop=False,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )
        self.workload_mode = workload_mode
        self.last_device_array: jax.Array | None = None

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Process one synthetic sample-major callback chunk."""
        del chunk_stats
        if self.workload_mode == CallbackWorkloadMode.HOST_TO_DEVICE:
            self.last_device_array = callback_transfers.put_genotype_matrix_on_device(
                genotype_matrix,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="benchmark_genotype",
            )

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Process one synthetic variant-major callback chunk."""
        self.compute_preprocessed_chunk(
            variant_metadata=variant_metadata,
            genotype_matrix=genotype_matrix_by_variant,
            chunk_stats=chunk_stats,
        )

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Process one synthetic packed8 callback chunk."""
        del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    def synchronize_last_device_array(self) -> None:
        """Synchronize the final device transfer when the trial requests it."""
        if self.last_device_array is None:
            return
        callback_diagnostics.block_until_ready(self.last_device_array)


def parse_stage_timing_modes(raw_value: str) -> tuple[StageTimingMode, ...]:
    """Parse stage timing mode names."""
    stage_timing_modes = tuple(StageTimingMode(value.strip()) for value in raw_value.split(",") if value.strip())
    if not stage_timing_modes:
        message = "At least one stage timing mode is required."
        raise ValueError(message)
    return stage_timing_modes


def parse_workload_modes(raw_value: str) -> tuple[CallbackWorkloadMode, ...]:
    """Parse synthetic callback workload mode names."""
    workload_modes = tuple(CallbackWorkloadMode(value.strip()) for value in raw_value.split(",") if value.strip())
    if not workload_modes:
        message = "At least one workload mode is required."
        raise ValueError(message)
    return workload_modes


def validate_arguments(arguments: BenchmarkArguments) -> None:
    """Validate benchmark argument ranges."""
    if arguments.chunk_count <= 0:
        message = "chunk_count must be positive."
        raise ValueError(message)
    if arguments.sample_count <= 0:
        message = "sample_count must be positive."
        raise ValueError(message)
    if arguments.variants_per_chunk <= 0:
        message = "variants_per_chunk must be positive."
        raise ValueError(message)
    if arguments.staging_depth <= 0:
        message = "staging_depth must be positive."
        raise ValueError(message)
    if arguments.trials <= 0:
        message = "trials must be positive."
        raise ValueError(message)
    if arguments.warmup_trials < 0:
        message = "warmup_trials must be non-negative."
        raise ValueError(message)


def build_stage_timing_recorder(stage_timing_mode: StageTimingMode) -> timing.StageTimingRecorder | None:
    """Build the recorder requested by one benchmark case."""
    if stage_timing_mode == StageTimingMode.OFF:
        return None
    return timing.StageTimingRecorder(exact_stage_timings=stage_timing_mode == StageTimingMode.EXACT)


def build_chunk_metadata(arguments: BenchmarkArguments) -> tuple[BenchmarkChunkMetadata, ...]:
    """Prebuild synthetic metadata so timed trials isolate callback overhead."""
    return tuple(
        BenchmarkChunkMetadata(
            chromosome=("chr1",),
            variant_start_index=chunk_index * arguments.variants_per_chunk,
            variant_stop_index=(chunk_index + 1) * arguments.variants_per_chunk,
        )
        for chunk_index in range(arguments.chunk_count)
    )


def build_benchmark_cases(arguments: BenchmarkArguments) -> tuple[BenchmarkCase, ...]:
    """Expand configured workload and timing modes into benchmark cases."""
    return tuple(
        BenchmarkCase(workload_mode=workload_mode, stage_timing_mode=stage_timing_mode)
        for workload_mode in parse_workload_modes(arguments.workload_modes)
        for stage_timing_mode in parse_stage_timing_modes(arguments.stage_timing_modes)
    )


def summarize_recorder(
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> RecorderSummary:
    """Summarize recorder output without serializing per-chunk timing rows."""
    if stage_timing_recorder is None:
        return RecorderSummary(
            stage_totals_seconds={},
            stage_counts={},
            queue_observation_count=0,
            transfer_observation_count=0,
        )
    snapshot = stage_timing_recorder.snapshot()
    queue_observation_count = sum(item.observation_count for item in snapshot.queue_backpressure)
    transfer_observation_count = sum(item.observation_count for item in snapshot.transfer_metadata)
    return RecorderSummary(
        stage_totals_seconds=snapshot.stage_totals_seconds,
        stage_counts=snapshot.stage_counts,
        queue_observation_count=queue_observation_count,
        transfer_observation_count=transfer_observation_count,
    )


def run_one_trial(
    *,
    arguments: BenchmarkArguments,
    benchmark_case: BenchmarkCase,
    trial_index: int,
    chunk_metadata: tuple[BenchmarkChunkMetadata, ...],
    genotype_matrix: npt.NDArray[np.float32],
    chunk_stats: _core.ChunkStats,
) -> TrialResult:
    """Run one measured callback-overhead trial."""
    stage_timing_recorder = build_stage_timing_recorder(benchmark_case.stage_timing_mode)
    callback = CallbackOverheadBenchmarkRunner(
        workload_mode=benchmark_case.workload_mode,
        staging_depth=arguments.staging_depth,
        stage_timing_recorder=stage_timing_recorder,
    )
    callback.start()
    start_time = time.perf_counter()
    for metadata in chunk_metadata:
        callback.compute_preprocessed_dosage_chunk(
            metadata=typing.cast("_core.VariantMetadata", metadata),
            genotype_matrix=genotype_matrix,
            chunk_stats=chunk_stats,
        )
    callback.finish()
    if arguments.synchronize_after_trial:
        callback.synchronize_last_device_array()
    wall_time_seconds = time.perf_counter() - start_time
    recorder_summary = summarize_recorder(stage_timing_recorder)
    chunks_per_second = arguments.chunk_count / wall_time_seconds
    return TrialResult(
        workload_mode=benchmark_case.workload_mode,
        stage_timing_mode=benchmark_case.stage_timing_mode,
        trial_index=trial_index,
        chunk_count=arguments.chunk_count,
        wall_time_seconds=wall_time_seconds,
        chunks_per_second=chunks_per_second,
        nanoseconds_per_chunk=wall_time_seconds * 1_000_000_000.0 / arguments.chunk_count,
        requested_device=arguments.device,
        jax_backend=jax.default_backend(),
        jax_devices=tuple(str(device) for device in jax.devices()),
        stage_totals_seconds=recorder_summary.stage_totals_seconds,
        stage_counts=recorder_summary.stage_counts,
        queue_observation_count=recorder_summary.queue_observation_count,
        transfer_observation_count=recorder_summary.transfer_observation_count,
    )


def run_warmup_trials(
    *,
    arguments: BenchmarkArguments,
    benchmark_case: BenchmarkCase,
    chunk_metadata: tuple[BenchmarkChunkMetadata, ...],
    genotype_matrix: npt.NDArray[np.float32],
    chunk_stats: _core.ChunkStats,
) -> None:
    """Run unreported warmup trials before measured benchmarking."""
    for warmup_index in range(arguments.warmup_trials):
        run_one_trial(
            arguments=arguments,
            benchmark_case=benchmark_case,
            trial_index=warmup_index,
            chunk_metadata=chunk_metadata,
            genotype_matrix=genotype_matrix,
            chunk_stats=chunk_stats,
        )


def run_benchmark(arguments: BenchmarkArguments) -> list[TrialResult]:
    """Run all configured callback-overhead benchmark cases."""
    validate_arguments(arguments)
    jax.config.update("jax_platform_name", arguments.device)
    chunk_metadata = build_chunk_metadata(arguments)
    genotype_matrix = np.zeros((arguments.sample_count, arguments.variants_per_chunk), dtype=np.float32, order="C")
    chunk_stats = typing.cast("_core.ChunkStats", object())
    trial_results: list[TrialResult] = []
    for benchmark_case in build_benchmark_cases(arguments):
        run_warmup_trials(
            arguments=arguments,
            benchmark_case=benchmark_case,
            chunk_metadata=chunk_metadata,
            genotype_matrix=genotype_matrix,
            chunk_stats=chunk_stats,
        )
        for trial_index in range(arguments.trials):
            trial_result = run_one_trial(
                arguments=arguments,
                benchmark_case=benchmark_case,
                trial_index=trial_index,
                chunk_metadata=chunk_metadata,
                genotype_matrix=genotype_matrix,
                chunk_stats=chunk_stats,
            )
            trial_results.append(trial_result)
            print(json.dumps(tooling_reports.to_jsonable(trial_result), sort_keys=True))
    return trial_results


def build_case_summary(
    *,
    benchmark_case: BenchmarkCase,
    trial_results: list[TrialResult],
) -> CaseSummary:
    """Build aggregate metrics for one benchmark case."""
    matching_results = [
        trial_result
        for trial_result in trial_results
        if trial_result.workload_mode == benchmark_case.workload_mode
        and trial_result.stage_timing_mode == benchmark_case.stage_timing_mode
    ]
    wall_times = [trial_result.wall_time_seconds for trial_result in matching_results]
    chunks_per_second_values = [trial_result.chunks_per_second for trial_result in matching_results]
    nanoseconds_per_chunk_values = [trial_result.nanoseconds_per_chunk for trial_result in matching_results]
    return CaseSummary(
        workload_mode=benchmark_case.workload_mode,
        stage_timing_mode=benchmark_case.stage_timing_mode,
        trial_count=len(matching_results),
        mean_wall_time_seconds=statistics.fmean(wall_times),
        stdev_wall_time_seconds=statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
        mean_chunks_per_second=statistics.fmean(chunks_per_second_values),
        mean_nanoseconds_per_chunk=statistics.fmean(nanoseconds_per_chunk_values),
    )


def build_summary(arguments: BenchmarkArguments, trial_results: list[TrialResult]) -> dict[str, typing.Any]:
    """Build the benchmark summary JSON payload."""
    benchmark_cases = build_benchmark_cases(arguments)
    return {
        "configuration": arguments,
        "case_summaries": [
            build_case_summary(benchmark_case=benchmark_case, trial_results=trial_results)
            for benchmark_case in benchmark_cases
        ],
        "trial_results": trial_results,
    }


def callback_arguments_payload(arguments: BenchmarkArguments) -> dict[str, object]:
    """Build a JSON-ready callback benchmark configuration."""
    return typing.cast("dict[str, object]", tooling_reports.to_jsonable(dataclasses.asdict(arguments)))


def callback_case_identifier(case_summary: CaseSummary) -> str:
    """Build a stable callback benchmark case identifier."""
    return f"{case_summary.workload_mode.value}_{case_summary.stage_timing_mode.value}"


def build_callback_metrics(
    *,
    run_id: str,
    case_summaries: list[CaseSummary],
    trial_results: list[TrialResult],
) -> list[tooling_artifact_format.MetricRecord]:
    """Build normalized callback-overhead benchmark metrics."""
    metric_records: list[tooling_artifact_format.MetricRecord] = []
    for case_index, case_summary in enumerate(case_summaries):
        case_id = callback_case_identifier(case_summary)
        dimensions: dict[str, object] = {
            "workload_mode": case_summary.workload_mode.value,
            "stage_timing_mode": case_summary.stage_timing_mode.value,
        }
        metric_records.extend(
            [
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=case_id,
                    metric_name="wall_time_seconds",
                    value=case_summary.mean_wall_time_seconds,
                    unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                    aggregation=tooling_artifact_format.MetricAggregation.MEAN.value,
                    higher_is_better=False,
                    dimensions=dimensions,
                    phase="callback_overhead",
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="report.json",
                        json_pointer=f"/cases/{case_index}/mean_wall_time_seconds",
                    ),
                ),
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=case_id,
                    metric_name="throughput_chunks_per_second",
                    value=case_summary.mean_chunks_per_second,
                    unit=tooling_artifact_format.MetricUnit.COUNT.value,
                    aggregation=tooling_artifact_format.MetricAggregation.MEAN.value,
                    higher_is_better=True,
                    dimensions=dimensions,
                    phase="callback_overhead",
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="report.json",
                        json_pointer=f"/cases/{case_index}/mean_chunks_per_second",
                    ),
                ),
            ]
        )
    for trial_index, trial_result in enumerate(trial_results):
        trial_id = (
            f"{trial_result.workload_mode.value}_{trial_result.stage_timing_mode.value}_{trial_result.trial_index}"
        )
        trial_dimensions: dict[str, object] = {
            "workload_mode": trial_result.workload_mode.value,
            "stage_timing_mode": trial_result.stage_timing_mode.value,
            "requested_device": trial_result.requested_device,
            "jax_backend": trial_result.jax_backend,
            "chunk_count": trial_result.chunk_count,
        }
        metric_records.append(
            tooling_artifact_format.build_metric_record(
                run_id=run_id,
                case_id=f"{trial_result.workload_mode.value}_{trial_result.stage_timing_mode.value}",
                trial_id=trial_id,
                metric_name="wall_time_seconds",
                value=trial_result.wall_time_seconds,
                unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                higher_is_better=False,
                dimensions=trial_dimensions,
                phase="callback_overhead",
                source=tooling_artifact_format.MetricSource(
                    artifact_path="report.json",
                    json_pointer=f"/trials/{trial_index}/wall_time_seconds",
                ),
            )
        )
        for stage_name, seconds in sorted(trial_result.stage_totals_seconds.items()):
            metric_records.append(
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=f"{trial_result.workload_mode.value}_{trial_result.stage_timing_mode.value}",
                    trial_id=trial_id,
                    metric_name=f"stage.{stage_name}.seconds",
                    value=seconds,
                    unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                    aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                    higher_is_better=False,
                    dimensions=trial_dimensions,
                    phase="callback_overhead",
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="report.json",
                        json_pointer=f"/trials/{trial_index}/stage_totals_seconds/{stage_name}",
                    ),
                )
            )
    return metric_records


def write_standard_callback_artifacts(
    *,
    arguments: BenchmarkArguments,
    summary: dict[str, typing.Any],
    hydra_config: omegaconf.DictConfig | None = None,
) -> None:
    """Write Tooling Artifact Format v1 outputs for callback-overhead benchmarks."""
    if arguments.json_summary_path is None:
        return
    output_directory = arguments.json_summary_path.parent
    case_summaries = typing.cast("list[CaseSummary]", summary["case_summaries"])
    trial_results = typing.cast("list[TrialResult]", summary["trial_results"])
    producer = tooling_artifact_format.build_producer(
        tool_name="benchmark_callback_overhead",
        repository_root=pathlib.Path.cwd(),
    )
    run = tooling_artifact_format.build_run_identity(
        tool_name="benchmark_callback_overhead",
        output_directory=output_directory,
        status=tooling_artifact_format.ToolArtifactStatus.SUCCESS,
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=output_directory,
        repository_root=pathlib.Path.cwd(),
    )
    report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title="Callback Overhead Benchmark",
        configuration=callback_arguments_payload(arguments),
        summary={
            "headline": "Callback-overhead benchmark completed.",
            "legacy_summary_path": str(arguments.json_summary_path),
        },
        cases=typing.cast("list[dict[str, object]]", tooling_reports.to_jsonable(case_summaries)),
        trials=typing.cast("list[dict[str, object]]", tooling_reports.to_jsonable(trial_results)),
        metrics=build_callback_metrics(
            run_id=run.run_id,
            case_summaries=case_summaries,
            trial_results=trial_results,
        ),
    )
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=output_directory,
        report=report,
        events=[
            tooling_artifact_format.build_tool_event(
                tool_name="benchmark_callback_overhead",
                run_id=run.run_id,
                phase="callback_overhead",
                event="benchmark_completed",
                message="Callback-overhead benchmark completed.",
                fields={"trial_count": len(trial_results), "case_count": len(case_summaries)},
            )
        ],
        hydra_config=hydra_config,
        tool_payload=callback_arguments_payload(arguments),
        notes=["Configured JSON summary path preserves the pre-v1 callback benchmark shape."],
    )


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Build benchmark parameters from a composed Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return BenchmarkArguments(
        device=str(tool_values["device"]),
        chunk_count=int(tool_values["chunk_count"]),
        sample_count=int(tool_values["sample_count"]),
        variants_per_chunk=int(tool_values["variants_per_chunk"]),
        staging_depth=int(tool_values["staging_depth"]),
        trials=int(tool_values["trials"]),
        warmup_trials=int(tool_values["warmup_trials"]),
        stage_timing_modes=tooling_hydra_arguments.comma_join(tool_values["stage_timing_modes"]),
        workload_modes=tooling_hydra_arguments.comma_join(tool_values["workload_modes"]),
        synchronize_after_trial=bool(tool_values["synchronize_after_trial"]),
        json_summary_path=tooling_hydra_arguments.path_or_none(tool_values.get("json_summary_path")),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Compose the callback-overhead config and return resolved parameters."""
    config = tooling_configuration.compose_config(config_name="benchmark_callback_overhead", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: BenchmarkArguments, hydra_config: omegaconf.DictConfig | None = None) -> None:
    """Run the callback-overhead benchmark."""
    trial_results = run_benchmark(arguments)
    summary = build_summary(arguments, trial_results)
    print(json.dumps(tooling_reports.to_jsonable(summary), indent=2, sort_keys=True))
    if arguments.json_summary_path is not None:
        tooling_reports.write_json_report(arguments.json_summary_path, summary, sort_keys=True)
        write_standard_callback_artifacts(arguments=arguments, summary=summary, hydra_config=hydra_config)
        print(f"Wrote summary: {arguments.json_summary_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_callback_overhead")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the callback-overhead benchmark through Hydra."""
    run_tool(build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run the callback-overhead benchmark."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
