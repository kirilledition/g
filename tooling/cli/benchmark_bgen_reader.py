#!/usr/bin/env python3
"""Benchmark native REGENIE2 BGEN chunk delivery paths."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import typing
from pathlib import Path

import hydra
import numpy as np

import tooling.configuration as tooling_configuration
from g import _core
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import reports as tooling_reports
from tooling.common import sweeps as tooling_sweeps
from tooling.regenie import bgen_reader as regenie_bgen_reader

if typing.TYPE_CHECKING:
    import omegaconf

BenchmarkPathMode = regenie_bgen_reader.BenchmarkPathMode
SampleSelectionMode = regenie_bgen_reader.SampleSelectionMode
PathResult = regenie_bgen_reader.PathResult
BenchmarkCaseReport = regenie_bgen_reader.BenchmarkCaseReport
BenchmarkSweepReport = regenie_bgen_reader.BenchmarkSweepReport


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved BGEN reader benchmark parameters.

    Attributes:
        bgen: Input BGEN path.
        sample: Optional sample file path.
        chunk_size: Single chunk size for case execution.
        chunk_sizes: Comma-separated chunk-size sweep values.
        variant_limit: Variant cap for each case.
        repeat_count: Measured repeat count.
        path_modes: Comma-separated native path modes.
        sample_selection_mode: Single sample-selection mode for case execution.
        sample_selection_modes: Comma-separated sample-selection sweep values.
        decode_tile_variant_count: Single native decode tile size for case metadata.
        decode_tile_variant_counts: Comma-separated decode tile sweep values.
        rayon_thread_count: Single Rayon thread count for case metadata.
        rayon_thread_counts: Comma-separated Rayon thread-count sweep values.
        trusted_no_missing_diploid: Whether this case uses the trusted decode path.
        trusted_no_missing_diploid_modes: Comma-separated trusted-mode sweep values.
        emit_case_json: Whether to emit only one case as JSON.
        json_summary_path: Optional JSON summary path.
        markdown_summary_path: Optional Markdown summary path.

    """

    bgen: Path
    sample: Path | None
    chunk_size: int
    chunk_sizes: str
    variant_limit: int
    repeat_count: int
    path_modes: str
    sample_selection_mode: str
    sample_selection_modes: str
    decode_tile_variant_count: int | None
    decode_tile_variant_counts: str
    rayon_thread_count: int | None
    rayon_thread_counts: str
    trusted_no_missing_diploid: bool
    trusted_no_missing_diploid_modes: str
    emit_case_json: bool
    json_summary_path: Path | None
    markdown_summary_path: Path | None


class ChecksumCallback:
    """Native chunk callback that accumulates finite dosage checksums."""

    def __init__(self) -> None:
        """Initialize checksum state and reusable native callback buffers."""
        self.checksum = 0.0
        self.free_dosage_buffers: list[np.ndarray] = []
        self.free_packed8_buffers: list[np.ndarray] = []

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
        """Return a reusable variant-major dosage buffer."""
        expected_shape = (variant_count, sample_count)
        if self.free_dosage_buffers:
            buffer = self.free_dosage_buffers.pop()
            if buffer.shape == expected_shape:
                return buffer
        return np.empty(expected_shape, dtype=np.float32, order="C")

    def acquire_variant_major_packed8_probability_pair_buffer(
        self, variant_count: int, sample_count: int
    ) -> np.ndarray:
        """Return a reusable packed8 probability-pair buffer."""
        expected_shape = (variant_count, sample_count, 2)
        if self.free_packed8_buffers:
            buffer = self.free_packed8_buffers.pop()
            if buffer.shape == expected_shape:
                return buffer
        return np.empty(expected_shape, dtype=np.uint8, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Accumulate a checksum for a dosage chunk."""
        del metadata, chunk_stats
        self.checksum += float(np.nansum(genotype_matrix_by_variant))
        self.free_dosage_buffers.append(genotype_matrix_by_variant)

    def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        self,
        metadata: _core.VariantMetadata,
        probability_pairs_by_variant: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Accumulate a checksum for a packed8 probability-pair chunk."""
        del metadata, chunk_stats
        probability_pairs_as_i16 = probability_pairs_by_variant.astype(np.int16, copy=False)
        raw_dosage = 510 - (2 * probability_pairs_as_i16[:, :, 0]) - probability_pairs_as_i16[:, :, 1]
        self.checksum += float(np.sum(raw_dosage, dtype=np.float64) / 255.0)
        self.free_packed8_buffers.append(probability_pairs_by_variant)


def parse_optional_int_list(raw_values: str) -> list[int | None]:
    """Parse a comma-separated integer list with an optional empty sentinel."""
    return tooling_sweeps.parse_optional_integer_list(raw_values)


def parse_path_modes(raw_path_modes: str) -> list[BenchmarkPathMode]:
    """Parse the requested native benchmark paths."""
    return regenie_bgen_reader.parse_path_modes(raw_path_modes)


def parse_sample_selection_modes(raw_sample_selection_modes: str) -> list[SampleSelectionMode]:
    """Parse requested sample-selection benchmark shapes."""
    return regenie_bgen_reader.parse_sample_selection_modes(raw_sample_selection_modes)


def parse_boolean_mode_list(raw_values: str) -> list[bool]:
    """Parse a comma-separated boolean list."""
    return tooling_sweeps.parse_boolean_mode_list(raw_values)


def build_sample_indices(sample_count: int, sample_selection_mode: SampleSelectionMode) -> np.ndarray:
    """Build the selected sample index vector for one benchmark case."""
    return regenie_bgen_reader.build_sample_indices(sample_count, sample_selection_mode)


def supported_path_modes(
    path_modes: list[BenchmarkPathMode], *, trusted_no_missing_diploid: bool
) -> list[BenchmarkPathMode]:
    """Return path modes that are valid for the current trusted-mode case."""
    return regenie_bgen_reader.supported_path_modes(path_modes, trusted_no_missing_diploid=trusted_no_missing_diploid)


def run_native_delivery(arguments: BenchmarkArguments, path_mode: BenchmarkPathMode, variant_limit: int) -> float:
    """Run one native delivery path and return its checksum."""
    engine = _core.Regenie2RunEngine(
        str(arguments.bgen),
        chunk_size=arguments.chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=arguments.trusted_no_missing_diploid,
    )
    if arguments.trusted_no_missing_diploid:
        engine.validate_trusted_no_missing_diploid()
    callback = ChecksumCallback()
    sample_selection_mode = SampleSelectionMode(arguments.sample_selection_mode)
    sample_indices = build_sample_indices(int(engine.sample_count), sample_selection_mode)
    if path_mode == BenchmarkPathMode.VARIANT_MAJOR_BUFFERED:
        engine.run_bgen_variant_major_dosage_buffered_chunks(sample_indices, callback)
    elif path_mode == BenchmarkPathMode.VARIANT_MAJOR_PACKED8_BUFFERED:
        engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks(sample_indices, callback)
    else:
        typing.assert_never(path_mode)
    return callback.checksum


def time_operation(
    operation: typing.Callable[[], float], repeat_count: int, path_mode: BenchmarkPathMode
) -> PathResult:
    """Warm once and repeatedly time one benchmark operation."""
    warmup_checksum = operation()
    duration_seconds: list[float] = []
    checksum = warmup_checksum
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return PathResult(
        path_mode=path_mode.value,
        durations_seconds=duration_seconds,
        mean_seconds=sum(duration_seconds) / len(duration_seconds),
        median_seconds=float(np.median(duration_seconds)),
        checksum=checksum,
    )


def build_case_report(arguments: BenchmarkArguments) -> BenchmarkCaseReport:
    """Run one benchmark case in-process."""
    path_modes = supported_path_modes(
        parse_path_modes(arguments.path_modes),
        trusted_no_missing_diploid=bool(arguments.trusted_no_missing_diploid),
    )
    if not path_modes:
        message = "No supported benchmark path modes remain for this case."
        raise ValueError(message)
    variant_limit = arguments.variant_limit
    sample_selection_mode = SampleSelectionMode(arguments.sample_selection_mode)
    engine_for_shape = _core.Regenie2RunEngine(
        str(arguments.bgen),
        chunk_size=arguments.chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=arguments.trusted_no_missing_diploid,
    )
    selected_sample_count = len(build_sample_indices(int(engine_for_shape.sample_count), sample_selection_mode))
    path_results = [
        time_operation(
            lambda path_mode=path_mode: run_native_delivery(arguments, path_mode, variant_limit),
            arguments.repeat_count,
            path_mode,
        )
        for path_mode in path_modes
    ]
    checksum_reference_path = path_results[0].path_mode
    checksum_reference_value = path_results[0].checksum
    for path_result in path_results[1:]:
        if not np.isclose(checksum_reference_value, path_result.checksum, rtol=1.0e-6, atol=1.0e-3):
            message = (
                "Checksum mismatch between benchmark paths: "
                f"{checksum_reference_path}={checksum_reference_value} vs "
                f"{path_result.path_mode}={path_result.checksum}."
            )
            raise ValueError(message)
    return BenchmarkCaseReport(
        bgen_path=str(arguments.bgen),
        sample_path=str(arguments.sample) if arguments.sample is not None else None,
        chunk_size=arguments.chunk_size,
        variant_limit=variant_limit,
        repeat_count=arguments.repeat_count,
        decode_tile_variant_count=arguments.decode_tile_variant_count,
        rayon_thread_count=arguments.rayon_thread_count,
        trusted_no_missing_diploid=bool(arguments.trusted_no_missing_diploid),
        sample_selection_mode=sample_selection_mode.value,
        selected_sample_count=selected_sample_count,
        path_results=path_results,
        checksum_reference_path=checksum_reference_path,
    )


def run_case_subprocess(
    arguments: BenchmarkArguments,
    chunk_size: int,
    decode_tile_variant_count: int | None,
    rayon_thread_count: int | None,
    *,
    trusted_no_missing_diploid: bool,
    sample_selection_mode: SampleSelectionMode,
) -> BenchmarkCaseReport:
    """Run one benchmark case in a fresh process with low-level env knobs."""
    command = [sys.executable, "-m", "tooling.cli.benchmark_bgen_reader"]
    command.extend(
        tooling_hydra_arguments.build_overrides(
            {
                "tool.bgen": str(arguments.bgen),
                "tool.sample": str(arguments.sample) if arguments.sample is not None else None,
                "tool.chunk_size": chunk_size,
                "tool.variant_limit": arguments.variant_limit,
                "tool.repeat_count": arguments.repeat_count,
                "tool.path_modes": arguments.path_modes,
                "tool.sample_selection_mode": sample_selection_mode.value,
                "tool.trusted_no_missing_diploid": trusted_no_missing_diploid,
                "tool.emit_case_json": True,
                "tool.json_summary_path": None,
                "tool.markdown_summary_path": None,
            }
        )
    )
    environment = os.environ.copy()
    if decode_tile_variant_count is not None:
        environment["G_BGEN_DECODE_TILE_VARIANT_COUNT"] = str(decode_tile_variant_count)
        command.append(f"tool.decode_tile_variant_count={decode_tile_variant_count}")
    if rayon_thread_count is not None:
        environment["RAYON_NUM_THREADS"] = str(rayon_thread_count)
        command.append(f"tool.rayon_thread_count={rayon_thread_count}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    except subprocess.CalledProcessError as error:
        message = (
            "BGEN reader benchmark case subprocess failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{error.stdout}\n"
            f"stderr:\n{error.stderr}"
        )
        raise RuntimeError(message) from error
    payload = json.loads(result.stdout)
    return BenchmarkCaseReport(
        bgen_path=payload["bgen_path"],
        sample_path=payload["sample_path"],
        chunk_size=int(payload["chunk_size"]),
        variant_limit=int(payload["variant_limit"]),
        repeat_count=int(payload["repeat_count"]),
        decode_tile_variant_count=payload["decode_tile_variant_count"],
        rayon_thread_count=payload["rayon_thread_count"],
        trusted_no_missing_diploid=bool(payload["trusted_no_missing_diploid"]),
        sample_selection_mode=payload["sample_selection_mode"],
        selected_sample_count=int(payload["selected_sample_count"]),
        path_results=[PathResult(**path_result) for path_result in payload["path_results"]],
        checksum_reference_path=payload["checksum_reference_path"],
    )


def build_sweep_report(arguments: BenchmarkArguments) -> BenchmarkSweepReport:
    """Run all requested native BGEN benchmark cases."""
    chunk_sizes = parse_optional_int_list(arguments.chunk_sizes) or [arguments.chunk_size]
    decode_tile_variant_counts = parse_optional_int_list(arguments.decode_tile_variant_counts) or [
        arguments.decode_tile_variant_count
    ]
    rayon_thread_counts = parse_optional_int_list(arguments.rayon_thread_counts) or [arguments.rayon_thread_count]
    trusted_modes = parse_boolean_mode_list(arguments.trusted_no_missing_diploid_modes) or [
        bool(arguments.trusted_no_missing_diploid)
    ]
    sample_selection_modes = (
        parse_sample_selection_modes(arguments.sample_selection_modes)
        if arguments.sample_selection_modes
        else [SampleSelectionMode(arguments.sample_selection_mode)]
    )
    cases = [
        run_case_subprocess(
            arguments,
            chunk_size=int(chunk_size),
            decode_tile_variant_count=decode_tile_variant_count,
            rayon_thread_count=rayon_thread_count,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
            sample_selection_mode=sample_selection_mode,
        )
        for chunk_size in chunk_sizes
        if chunk_size is not None
        for decode_tile_variant_count in decode_tile_variant_counts
        for rayon_thread_count in rayon_thread_counts
        for trusted_no_missing_diploid in trusted_modes
        for sample_selection_mode in sample_selection_modes
    ]
    return BenchmarkSweepReport(cases=cases)


def write_text_report(path: Path, report: BenchmarkSweepReport) -> None:
    """Write a compact Markdown benchmark report."""
    lines = [
        "# BGEN Reader Benchmark",
        "",
        "| trusted | selection | samples | chunk | path | median_s | mean_s | checksum |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for benchmark_case in report.cases:
        for path_result in benchmark_case.path_results:
            lines.append(
                "| "
                f"{benchmark_case.trusted_no_missing_diploid} | "
                f"{benchmark_case.sample_selection_mode} | "
                f"{benchmark_case.selected_sample_count} | "
                f"{benchmark_case.chunk_size} | "
                f"{path_result.path_mode} | "
                f"{path_result.median_seconds:.6f} | "
                f"{path_result.mean_seconds:.6f} | "
                f"{path_result.checksum:.6f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_arguments_payload(arguments: BenchmarkArguments) -> dict[str, object]:
    """Build a JSON-ready argument snapshot."""
    return typing.cast("dict[str, object]", tooling_reports.to_jsonable(dataclasses.asdict(arguments)))


def bgen_case_identifier(benchmark_case: BenchmarkCaseReport, case_index: int) -> str:
    """Build a stable case identifier for one BGEN reader benchmark case."""
    return (
        f"bgen_case_{case_index:04d}_chunk{benchmark_case.chunk_size}_"
        f"trusted{str(benchmark_case.trusted_no_missing_diploid).lower()}_"
        f"{benchmark_case.sample_selection_mode}"
    )


def build_bgen_metrics(
    *,
    run_id: str,
    report: BenchmarkSweepReport,
) -> list[tooling_artifact_format.MetricRecord]:
    """Build normalized metrics for a BGEN reader benchmark report."""
    metric_records: list[tooling_artifact_format.MetricRecord] = []
    for case_index, benchmark_case in enumerate(report.cases):
        case_id = bgen_case_identifier(benchmark_case, case_index)
        for path_index, path_result in enumerate(benchmark_case.path_results):
            dimensions: dict[str, object] = {
                "path_mode": path_result.path_mode,
                "chunk_size": benchmark_case.chunk_size,
                "variant_limit": benchmark_case.variant_limit,
                "repeat_count": benchmark_case.repeat_count,
                "decode_tile_variant_count": benchmark_case.decode_tile_variant_count,
                "rayon_thread_count": benchmark_case.rayon_thread_count,
                "trusted_no_missing_diploid": benchmark_case.trusted_no_missing_diploid,
                "sample_selection_mode": benchmark_case.sample_selection_mode,
                "selected_sample_count": benchmark_case.selected_sample_count,
            }
            metric_records.extend(
                [
                    tooling_artifact_format.build_metric_record(
                        run_id=run_id,
                        case_id=case_id,
                        metric_name="wall_time_seconds",
                        value=path_result.median_seconds,
                        unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                        aggregation=tooling_artifact_format.MetricAggregation.MEDIAN.value,
                        higher_is_better=False,
                        dimensions=dimensions,
                        phase="bgen_reader",
                        source=tooling_artifact_format.MetricSource(
                            artifact_path="report.json",
                            json_pointer=f"/cases/{case_index}/path_results/{path_index}/median_seconds",
                        ),
                    ),
                    tooling_artifact_format.build_metric_record(
                        run_id=run_id,
                        case_id=case_id,
                        metric_name="wall_time_seconds",
                        value=path_result.mean_seconds,
                        unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                        aggregation=tooling_artifact_format.MetricAggregation.MEAN.value,
                        higher_is_better=False,
                        dimensions=dimensions,
                        phase="bgen_reader",
                        source=tooling_artifact_format.MetricSource(
                            artifact_path="report.json",
                            json_pointer=f"/cases/{case_index}/path_results/{path_index}/mean_seconds",
                        ),
                    ),
                ]
            )
    return metric_records


def build_bgen_command_records(
    *,
    arguments: BenchmarkArguments,
    output_directory: Path,
    run_id: str,
    report: BenchmarkSweepReport,
) -> list[tooling_artifact_format.CommandRecord]:
    """Build command ledger records for BGEN reader subprocess cases."""
    command_records: list[tooling_artifact_format.CommandRecord] = []
    for case_index, benchmark_case in enumerate(report.cases):
        command_arguments = [sys.executable, "-m", "tooling.cli.benchmark_bgen_reader"]
        command_arguments.extend(
            tooling_hydra_arguments.build_overrides(
                {
                    "tool.bgen": str(arguments.bgen),
                    "tool.sample": str(arguments.sample) if arguments.sample is not None else None,
                    "tool.chunk_size": benchmark_case.chunk_size,
                    "tool.variant_limit": benchmark_case.variant_limit,
                    "tool.repeat_count": benchmark_case.repeat_count,
                    "tool.path_modes": arguments.path_modes,
                    "tool.sample_selection_mode": benchmark_case.sample_selection_mode,
                    "tool.trusted_no_missing_diploid": benchmark_case.trusted_no_missing_diploid,
                    "tool.emit_case_json": True,
                    "tool.json_summary_path": None,
                    "tool.markdown_summary_path": None,
                }
            )
        )
        environment_overrides: dict[str, str] = {}
        if benchmark_case.decode_tile_variant_count is not None:
            environment_overrides["G_BGEN_DECODE_TILE_VARIANT_COUNT"] = str(benchmark_case.decode_tile_variant_count)
            command_arguments.append(f"tool.decode_tile_variant_count={benchmark_case.decode_tile_variant_count}")
        if benchmark_case.rayon_thread_count is not None:
            environment_overrides["RAYON_NUM_THREADS"] = str(benchmark_case.rayon_thread_count)
            command_arguments.append(f"tool.rayon_thread_count={benchmark_case.rayon_thread_count}")
        command_records.append(
            tooling_artifact_format.build_command_record(
                command_id=bgen_case_identifier(benchmark_case, case_index),
                tool_name="benchmark_bgen_reader",
                run_id=run_id,
                phase="bgen_reader",
                args=command_arguments,
                output_directory=output_directory,
                cwd=Path.cwd(),
                environment_overrides=environment_overrides,
                status=tooling_artifact_format.ToolArtifactStatus.SUCCESS,
            )
        )
    return command_records


def resolve_bgen_artifact_output_directory(arguments: BenchmarkArguments) -> Path | None:
    """Resolve the artifact directory for BGEN reader summary outputs."""
    if arguments.json_summary_path is not None:
        return arguments.json_summary_path.parent
    if arguments.markdown_summary_path is not None:
        return arguments.markdown_summary_path.parent
    return None


def write_standard_bgen_artifacts(
    *,
    arguments: BenchmarkArguments,
    report: BenchmarkSweepReport,
    hydra_config: omegaconf.DictConfig | None = None,
) -> None:
    """Write Tooling Artifact Format v1 outputs for BGEN reader benchmarks."""
    output_directory = resolve_bgen_artifact_output_directory(arguments)
    if output_directory is None:
        return
    producer = tooling_artifact_format.build_producer(
        tool_name="benchmark_bgen_reader",
        repository_root=Path.cwd(),
    )
    run = tooling_artifact_format.build_run_identity(
        tool_name="benchmark_bgen_reader",
        output_directory=output_directory,
        status=tooling_artifact_format.ToolArtifactStatus.SUCCESS,
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=output_directory,
        repository_root=Path.cwd(),
    )
    standard_report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title="BGEN Reader Benchmark",
        configuration=benchmark_arguments_payload(arguments),
        summary={
            "headline": "BGEN reader benchmark completed.",
            "legacy_json_summary_path": str(arguments.json_summary_path) if arguments.json_summary_path else None,
        },
        cases=typing.cast("list[dict[str, object]]", tooling_reports.to_jsonable(report.cases)),
        metrics=build_bgen_metrics(run_id=run.run_id, report=report),
    )
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=output_directory,
        report=standard_report,
        events=[
            tooling_artifact_format.build_tool_event(
                tool_name="benchmark_bgen_reader",
                run_id=run.run_id,
                phase="bgen_reader",
                event="benchmark_completed",
                message="BGEN reader benchmark completed.",
                fields={"case_count": len(report.cases)},
            )
        ],
        commands=build_bgen_command_records(
            arguments=arguments,
            output_directory=output_directory,
            run_id=run.run_id,
            report=report,
        ),
        input_files=[
            tooling_artifact_format.build_input_file_record(path=arguments.bgen, kind="bgen"),
            *(
                [tooling_artifact_format.build_input_file_record(path=arguments.sample, kind="sample")]
                if arguments.sample is not None
                else []
            ),
        ],
        summary_markdown=(
            (arguments.markdown_summary_path.read_text(encoding="utf-8"))
            if arguments.markdown_summary_path is not None and arguments.markdown_summary_path.is_file()
            else None
        ),
        hydra_config=hydra_config,
        tool_payload=benchmark_arguments_payload(arguments),
        notes=["Legacy JSON and Markdown summary paths are preserved when configured."],
    )


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Build benchmark parameters from a composed Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return BenchmarkArguments(
        bgen=Path(str(tool_values["bgen"])),
        sample=tooling_hydra_arguments.path_or_none(tool_values.get("sample")),
        chunk_size=int(tool_values["chunk_size"]),
        chunk_sizes=tooling_hydra_arguments.comma_join(tool_values["chunk_sizes"]),
        variant_limit=int(tool_values["variant_limit"]),
        repeat_count=int(tool_values["repeat_count"]),
        path_modes=tooling_hydra_arguments.comma_join(tool_values["path_modes"]),
        sample_selection_mode=str(tool_values["sample_selection_mode"]),
        sample_selection_modes=tooling_hydra_arguments.comma_join(tool_values["sample_selection_modes"]),
        decode_tile_variant_count=tooling_hydra_arguments.integer_or_none(tool_values.get("decode_tile_variant_count")),
        decode_tile_variant_counts=tooling_hydra_arguments.comma_join(tool_values["decode_tile_variant_counts"]),
        rayon_thread_count=tooling_hydra_arguments.integer_or_none(tool_values.get("rayon_thread_count")),
        rayon_thread_counts=tooling_hydra_arguments.comma_join(tool_values["rayon_thread_counts"]),
        trusted_no_missing_diploid=bool(tool_values["trusted_no_missing_diploid"]),
        trusted_no_missing_diploid_modes=tooling_hydra_arguments.comma_join(
            tool_values["trusted_no_missing_diploid_modes"]
        ),
        emit_case_json=bool(tool_values["emit_case_json"]),
        json_summary_path=tooling_hydra_arguments.path_or_none(tool_values.get("json_summary_path")),
        markdown_summary_path=tooling_hydra_arguments.path_or_none(tool_values.get("markdown_summary_path")),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Compose the BGEN reader config and return resolved parameters."""
    config = tooling_configuration.compose_config(config_name="benchmark_bgen_reader", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: BenchmarkArguments, hydra_config: omegaconf.DictConfig | None = None) -> None:
    """Run the benchmark with resolved parameters."""
    if arguments.emit_case_json:
        print(tooling_reports.to_json_text(build_case_report(arguments)).strip())
        return
    report = build_sweep_report(arguments)
    report_json = tooling_reports.to_json_text(report).strip()
    if arguments.json_summary_path is not None:
        arguments.json_summary_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_summary_path.write_text(report_json + "\n", encoding="utf-8")
    if arguments.markdown_summary_path is not None:
        write_text_report(arguments.markdown_summary_path, report)
    write_standard_bgen_artifacts(arguments=arguments, report=report, hydra_config=hydra_config)
    print(report_json)


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_bgen_reader")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the benchmark CLI through Hydra."""
    run_tool(build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run the benchmark CLI."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
