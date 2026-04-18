#!/usr/bin/env python3
"""Capture detailed cProfile and JAX traces for REGENIE step 2 linear runs."""

from __future__ import annotations

import argparse
import collections
import contextlib
import cProfile
import dataclasses
import gzip
import io
import json
import pstats
import shutil
import time
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
import polars as pl

from g import api, models, types
from g import engine as engine_module
from g.io import genotype_processing, reader, regenie, source
from g.io import output as output_module

if typing.TYPE_CHECKING:
    import collections.abc


DEFAULT_BGEN_PATH = Path("data/1kg_chr22_full.bgen")
DEFAULT_SAMPLE_PATH = Path("data/1kg_chr22_full.sample")
DEFAULT_PHENOTYPE_PATH = Path("data/pheno_cont.txt")
DEFAULT_COVARIATE_PATH = Path("data/covariates.txt")
DEFAULT_PREDICTION_LIST_PATH = Path("data/baselines/regenie_step1_qt_pred.list")
DEFAULT_OUTPUT_DIRECTORY = Path("data/profiles/regenie2_linear")
DEFAULT_REPORT_NAME = "regenie2_linear_profile"


@dataclass(frozen=True)
class ProfileEventSummary:
    """Aggregated duration summary for one Perfetto event name."""

    name: str
    event_count: int
    total_duration_microseconds: float
    mean_duration_microseconds: float


@dataclass(frozen=True)
class StageTimingSummary:
    """Aggregated duration summary for one instrumented stage."""

    name: str
    event_count: int
    total_seconds: float
    mean_milliseconds: float
    percent_of_wall_time: float


@dataclass
class StageTimingAccumulator:
    """Mutable accumulator for one stage timing stream."""

    event_count: int = 0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class Regenie2DetailedProfileSummary:
    """Structured summary for one REGENIE step 2 profiling run."""

    backend: str
    device: str
    bgen_path: str
    sample_path: str | None
    phenotype_path: str
    phenotype_name: str
    covariate_path: str | None
    covariate_names: list[str]
    prediction_list_path: str
    chunk_size: int
    variant_limit: int | None
    prefetch_chunks: int
    warmup_pass_count: int
    total_variants: int
    chunk_count: int
    wall_time_seconds: float
    variants_per_second: float
    output_run_directory: str
    final_parquet_path: str | None
    trace_enabled: bool
    memory_profile_enabled: bool
    cprofile_path: str
    cprofile_raw_path: str
    summary_path: str
    trace_dir: str | None
    perfetto_trace_path: str | None
    memory_profile_path: str | None
    stage_timing_summaries: list[StageTimingSummary]
    perfetto_event_summaries: list[ProfileEventSummary]


@dataclass(frozen=True)
class TimingInstrumentationHandle:
    """Installed timing hooks and their shared accumulator state."""

    stage_timing_accumulators: dict[str, StageTimingAccumulator]
    restore_callbacks: tuple[collections.abc.Callable[[], None], ...]


@dataclass(frozen=True)
class PredictionSourceProxy:
    """Proxy that times chromosome-level LOCO prediction loading."""

    wrapped_prediction_source: regenie.Step1PredictionSource
    stage_timing_accumulators: dict[str, StageTimingAccumulator]

    def get_chromosome_predictions(
        self,
        chromosome: str,
        sample_family_identifiers: np.ndarray,
        sample_individual_identifiers: np.ndarray,
    ) -> jax.Array:
        """Return one chromosome's aligned LOCO predictions with timing."""
        start_time = time.perf_counter()
        chromosome_predictions = self.wrapped_prediction_source.get_chromosome_predictions(
            chromosome=chromosome,
            sample_family_identifiers=sample_family_identifiers,
            sample_individual_identifiers=sample_individual_identifiers,
        )
        block_until_ready(chromosome_predictions)
        record_stage_duration(
            self.stage_timing_accumulators,
            "load_loco_predictions",
            time.perf_counter() - start_time,
        )
        return chromosome_predictions


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for REGENIE step 2 profiling."""
    argument_parser = argparse.ArgumentParser(
        description="Profile REGENIE step 2 linear association with cProfile and JAX tracing."
    )
    argument_parser.add_argument("--bgen", type=Path, default=DEFAULT_BGEN_PATH, help="BGEN input path.")
    argument_parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH, help="BGEN sample-file path.")
    argument_parser.add_argument("--pheno", type=Path, default=DEFAULT_PHENOTYPE_PATH, help="Phenotype table path.")
    argument_parser.add_argument(
        "--pheno-name",
        default="phenotype_continuous",
        help="Phenotype column name to analyze.",
    )
    argument_parser.add_argument("--covar", type=Path, default=DEFAULT_COVARIATE_PATH, help="Covariate table path.")
    argument_parser.add_argument(
        "--covar-names",
        default="age,sex",
        help="Comma-separated covariate names.",
    )
    argument_parser.add_argument(
        "--pred",
        type=Path,
        default=DEFAULT_PREDICTION_LIST_PATH,
        help="REGENIE step 1 prediction list path.",
    )
    argument_parser.add_argument(
        "--device",
        type=types.Device,
        choices=list(types.Device),
        default=types.Device.GPU,
        help="JAX execution device.",
    )
    argument_parser.add_argument("--chunk-size", type=int, default=1024, help="Variants per chunk.")
    argument_parser.add_argument("--variant-limit", type=int, help="Optional variant cap.")
    argument_parser.add_argument("--prefetch-chunks", type=int, default=0, help="Prefetched genotype chunks.")
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where profiling artifacts are saved.",
    )
    argument_parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help="Base name for saved profiling artifacts.",
    )
    argument_parser.add_argument(
        "--enable-jax-trace",
        action="store_true",
        help="Capture a JAX Perfetto trace during the profiled run.",
    )
    argument_parser.add_argument(
        "--enable-memory-profile",
        action="store_true",
        help="Capture a JAX device memory profile after the profiled run.",
    )
    argument_parser.add_argument(
        "--cprofile-sort",
        default="cumulative",
        choices=("cumulative", "time", "calls", "name"),
        help="Sort key for cProfile text reports.",
    )
    argument_parser.add_argument(
        "--json-summary-path",
        type=Path,
        help="Optional JSON summary output path. Defaults to <output-dir>/<report-name>_summary.json.",
    )
    argument_parser.add_argument(
        "--warmup-pass-count",
        type=int,
        default=1,
        help="Number of unprofiled warmup runs before timed profiling.",
    )
    argument_parser.add_argument(
        "--transfer-guard-device-to-host",
        choices=("allow", "log", "log_explicit", "disallow", "disallow_explicit"),
        help="Optional JAX device-to-host transfer guard level during the profiled run.",
    )
    return argument_parser


def parse_covariate_names(raw_covariate_names: str) -> tuple[str, ...] | None:
    """Parse a comma-separated covariate name list."""
    covariate_names = tuple(name.strip() for name in raw_covariate_names.split(",") if name.strip())
    return covariate_names or None


def record_stage_duration(
    stage_timing_accumulators: dict[str, StageTimingAccumulator],
    stage_name: str,
    duration_seconds: float,
) -> None:
    """Record one elapsed duration for an instrumented stage."""
    stage_timing_accumulator = stage_timing_accumulators.setdefault(stage_name, StageTimingAccumulator())
    stage_timing_accumulator.event_count += 1
    stage_timing_accumulator.total_seconds += duration_seconds


def block_until_ready(value: typing.Any) -> None:
    """Synchronize any nested JAX arrays contained in a profiling value."""
    if isinstance(value, jax.Array):
        value.block_until_ready()
        return
    if isinstance(value, dict):
        for nested_value in value.values():
            block_until_ready(nested_value)
        return
    if isinstance(value, tuple | list):
        for nested_value in value:
            block_until_ready(nested_value)
        return
    if dataclasses.is_dataclass(value):
        for dataclass_field in dataclasses.fields(value):
            block_until_ready(getattr(value, dataclass_field.name))


def build_stage_timing_summaries(
    stage_timing_accumulators: dict[str, StageTimingAccumulator],
    wall_time_seconds: float,
) -> list[StageTimingSummary]:
    """Convert mutable timing accumulators into sorted summaries."""
    sorted_stage_items = sorted(
        stage_timing_accumulators.items(),
        key=lambda item: item[1].total_seconds,
        reverse=True,
    )
    return [
        StageTimingSummary(
            name=stage_name,
            event_count=stage_timing_accumulator.event_count,
            total_seconds=stage_timing_accumulator.total_seconds,
            mean_milliseconds=(stage_timing_accumulator.total_seconds / stage_timing_accumulator.event_count) * 1000.0,
            percent_of_wall_time=(stage_timing_accumulator.total_seconds / wall_time_seconds) * 100.0,
        )
        for stage_name, stage_timing_accumulator in sorted_stage_items
        if stage_timing_accumulator.event_count > 0
    ]


def generate_cprofile_report(
    profiler: cProfile.Profile,
    output_path: Path,
    sort_key: str,
    *,
    limit: int = 200,
) -> None:
    """Generate a detailed text cProfile report."""
    report_stream = io.StringIO()
    profile_stats = pstats.Stats(profiler, stream=report_stream)
    profile_stats.strip_dirs()
    profile_stats.sort_stats(sort_key)
    profile_stats.print_stats(limit)
    report_stream.write("\n")
    report_stream.write("=" * 80 + "\n")
    report_stream.write("CALLER STATISTICS\n")
    report_stream.write("=" * 80 + "\n")
    profile_stats.print_callers(50)
    report_stream.write("\n")
    report_stream.write("=" * 80 + "\n")
    report_stream.write("CALLEE STATISTICS\n")
    report_stream.write("=" * 80 + "\n")
    profile_stats.print_callees(50)
    output_path.write_text(report_stream.getvalue())


def find_perfetto_trace_path(trace_directory: Path) -> Path | None:
    """Locate the Perfetto trace produced by a JAX trace capture."""
    trace_candidates = sorted(trace_directory.rglob("perfetto_trace.json.gz"))
    if not trace_candidates:
        return None
    return trace_candidates[-1]


def summarize_perfetto_trace(
    perfetto_trace_path: Path,
    *,
    limit: int = 50,
) -> list[ProfileEventSummary]:
    """Aggregate named Perfetto events into a compact summary."""
    with gzip.open(perfetto_trace_path, mode="rt", encoding="utf-8") as file_handle:
        trace_payload = json.load(file_handle)
    event_totals: dict[str, tuple[int, float]] = {}
    for trace_event in trace_payload.get("traceEvents", []):
        event_name = trace_event.get("name")
        event_duration = trace_event.get("dur")
        if not isinstance(event_name, str) or not event_name:
            continue
        if not isinstance(event_duration, int | float):
            continue
        event_count, total_duration_microseconds = event_totals.get(event_name, (0, 0.0))
        event_totals[event_name] = (event_count + 1, total_duration_microseconds + float(event_duration))
    ranked_events = sorted(event_totals.items(), key=lambda item: item[1][1], reverse=True)[:limit]
    return [
        ProfileEventSummary(
            name=event_name,
            event_count=event_count,
            total_duration_microseconds=total_duration_microseconds,
            mean_duration_microseconds=total_duration_microseconds / event_count,
        )
        for event_name, (event_count, total_duration_microseconds) in ranked_events
    ]


def format_summary_report(
    profile_summary: Regenie2DetailedProfileSummary,
    profiler: cProfile.Profile,
    sort_key: str,
) -> str:
    """Format a plain-text execution summary report."""
    profile_stream = io.StringIO()
    summary_profile_stats = pstats.Stats(profiler, stream=profile_stream)
    summary_profile_stats.strip_dirs()
    summary_profile_stats.sort_stats(sort_key)
    summary_profile_stats.print_stats(100)
    report_lines = [
        "=" * 80,
        "REGENIE STEP 2 LINEAR DETAILED PROFILE",
        "=" * 80,
        "",
        "EXECUTION SUMMARY",
        "-" * 80,
        f"  Backend:              {profile_summary.backend}",
        f"  Device:               {profile_summary.device}",
        f"  Chunk Size:           {profile_summary.chunk_size}",
        f"  Variant Limit:        {profile_summary.variant_limit}",
        f"  Prefetch Chunks:      {profile_summary.prefetch_chunks}",
        f"  Warmup Passes:        {profile_summary.warmup_pass_count}",
        f"  Total Variants:       {profile_summary.total_variants}",
        f"  Total Chunks:         {profile_summary.chunk_count}",
        f"  Wall Clock Time:      {profile_summary.wall_time_seconds:.3f} seconds",
        f"  Variants / Second:    {profile_summary.variants_per_second:.1f}",
        "",
        "TOP STAGE TIMINGS",
        "-" * 80,
    ]
    for stage_timing_summary in profile_summary.stage_timing_summaries[:20]:
        report_lines.append(
            "  "
            f"{stage_timing_summary.name:<32} "
            f"count={stage_timing_summary.event_count:<5d} "
            f"total={stage_timing_summary.total_seconds:>8.3f}s "
            f"mean={stage_timing_summary.mean_milliseconds:>8.3f}ms "
            f"wall={stage_timing_summary.percent_of_wall_time:>6.2f}%"
        )
    if profile_summary.perfetto_event_summaries:
        report_lines.extend(["", "TOP PERFETTO EVENTS", "-" * 80])
        for perfetto_event_summary in profile_summary.perfetto_event_summaries[:20]:
            report_lines.append(
                "  "
                f"{perfetto_event_summary.name:<40} "
                f"count={perfetto_event_summary.event_count:<5d} "
                f"total={perfetto_event_summary.total_duration_microseconds / 1000.0:>8.3f}ms"
            )
    report_lines.extend(["", "TOP cPROFILE FUNCTIONS", "-" * 80, profile_stream.getvalue(), "=" * 80])
    return "\n".join(report_lines)


def resolve_report_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    """Resolve all output paths for one profiling run."""
    output_directory = arguments.output_dir
    report_name = arguments.report_name
    return {
        "output_directory": output_directory,
        "output_run_directory": output_directory / f"{report_name}_output.run",
        "trace_directory": output_directory / f"{report_name}_jax_trace",
        "cprofile_path": output_directory / f"{report_name}_cprofile.txt",
        "cprofile_raw_path": output_directory / f"{report_name}_cprofile.prof",
        "summary_path": output_directory / f"{report_name}_summary.txt",
        "memory_profile_path": output_directory / f"{report_name}_memory.prof",
        "json_summary_path": arguments.json_summary_path or (output_directory / f"{report_name}_summary.json"),
    }


def remove_existing_outputs(report_paths: dict[str, Path]) -> None:
    """Delete stale report-specific outputs before profiling."""
    report_paths["output_directory"].mkdir(parents=True, exist_ok=True)
    for directory_key in ("output_run_directory", "trace_directory"):
        directory_path = report_paths[directory_key]
        if directory_path.exists():
            shutil.rmtree(directory_path)
    for file_key in (
        "cprofile_path",
        "cprofile_raw_path",
        "summary_path",
        "memory_profile_path",
        "json_summary_path",
    ):
        file_path = report_paths[file_key]
        if file_path.exists():
            file_path.unlink()


def install_timing_instrumentation() -> TimingInstrumentationHandle:
    """Install stage-timing hooks around the REGENIE step 2 execution path."""
    stage_timing_accumulators: dict[str, StageTimingAccumulator] = {}
    restore_callbacks: list[collections.abc.Callable[[], None]] = []

    def wrap_module_function(
        module: typing.Any,
        attribute_name: str,
        wrapper_builder: collections.abc.Callable[[typing.Any], typing.Any],
    ) -> None:
        original_function = getattr(module, attribute_name)
        wrapped_function = wrapper_builder(original_function)
        setattr(module, attribute_name, wrapped_function)

        def restore() -> None:
            setattr(module, attribute_name, original_function)

        restore_callbacks.append(restore)

    def build_timed_function(stage_name: str) -> collections.abc.Callable[[typing.Any], typing.Any]:
        def wrapper_builder(original_function: typing.Any) -> typing.Any:
            def timed_function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
                start_time = time.perf_counter()
                result = original_function(*args, **kwargs)
                block_until_ready(result)
                record_stage_duration(stage_timing_accumulators, stage_name, time.perf_counter() - start_time)
                return result

            return timed_function

        return wrapper_builder

    wrap_module_function(
        engine_module,
        "load_aligned_sample_data_from_source",
        build_timed_function("load_aligned_sample_data"),
    )
    wrap_module_function(
        engine_module,
        "prepare_regenie2_linear_state",
        build_timed_function("prepare_state"),
    )
    wrap_module_function(
        engine_module,
        "compute_regenie2_linear_chunk",
        build_timed_function("compute_regenie2_linear_chunk"),
    )
    wrap_module_function(
        engine_module,
        "build_chunk_payload",
        build_timed_function("build_chunk_payload"),
    )
    wrap_module_function(
        engine_module,
        "split_linear_genotype_chunk_by_chromosome",
        build_timed_function("split_chunk_by_chromosome"),
    )
    wrap_module_function(
        api,
        "prepare_output_run",
        build_timed_function("prepare_output_run"),
    )
    wrap_module_function(
        api,
        "persist_chunked_results",
        build_timed_function("persist_chunked_results_total"),
    )
    wrap_module_function(
        api,
        "finalize_chunks_to_parquet",
        build_timed_function("finalize_chunks_to_parquet"),
    )
    wrap_module_function(
        output_module,
        "write_chunk_to_disk",
        build_timed_function("write_chunk_to_disk"),
    )

    def prediction_source_wrapper_builder(
        original_function: collections.abc.Callable[..., regenie.Step1PredictionSource],
    ) -> collections.abc.Callable[..., PredictionSourceProxy]:
        def timed_load_prediction_source(*args: typing.Any, **kwargs: typing.Any) -> PredictionSourceProxy:
            start_time = time.perf_counter()
            prediction_source = original_function(*args, **kwargs)
            record_stage_duration(stage_timing_accumulators, "load_prediction_source", time.perf_counter() - start_time)
            return PredictionSourceProxy(
                wrapped_prediction_source=prediction_source,
                stage_timing_accumulators=stage_timing_accumulators,
            )

        return timed_load_prediction_source

    wrap_module_function(engine_module, "load_prediction_source", prediction_source_wrapper_builder)

    original_iter_linear_genotype_chunks_from_source = engine_module.iter_linear_genotype_chunks_from_source

    def profiled_iter_linear_genotype_chunks_from_source(
        genotype_source_config: source.GenotypeSourceConfig,
        sample_indices: np.ndarray,
        expected_individual_identifiers: np.ndarray,
        chunk_size: int,
        variant_limit: int | None = None,
        *,
        prefetch_chunks: int = 0,
        genotype_reader: reader.GenotypeReader | None = None,
    ) -> collections.abc.Iterator[models.LinearGenotypeChunk]:
        if genotype_reader is None or genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
            yield from original_iter_linear_genotype_chunks_from_source(
                genotype_source_config=genotype_source_config,
                sample_indices=sample_indices,
                expected_individual_identifiers=expected_individual_identifiers,
                chunk_size=chunk_size,
                variant_limit=variant_limit,
                prefetch_chunks=prefetch_chunks,
                genotype_reader=genotype_reader,
            )
            return

        source.validate_genotype_source_config(genotype_source_config)
        total_variant_count = reader.resolve_total_variant_count(genotype_reader.variant_count, variant_limit)
        if total_variant_count == 0:
            return

        sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
        reader.validate_sample_order(
            observed_individual_identifiers=genotype_reader.samples,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=expected_individual_identifiers,
            source_name="BGEN",
        )

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)

            metadata_start_time = time.perf_counter()
            variant_table_arrays = genotype_reader.get_variant_table_arrays(variant_start, variant_stop)
            record_stage_duration(
                stage_timing_accumulators,
                "get_variant_table_arrays",
                time.perf_counter() - metadata_start_time,
            )

            host_read_start_time = time.perf_counter()
            genotype_matrix_host = genotype_reader.read(
                index=(sample_index_array, slice(variant_start, variant_stop)),
                dtype=np.float32,
                order=types.ArrayMemoryOrder.C_CONTIGUOUS,
            )
            record_stage_duration(
                stage_timing_accumulators,
                "bgen_read_host",
                time.perf_counter() - host_read_start_time,
            )

            device_put_start_time = time.perf_counter()
            genotype_matrix_device = jax.device_put(genotype_matrix_host)
            block_until_ready(genotype_matrix_device)
            record_stage_duration(
                stage_timing_accumulators,
                "device_put_genotypes",
                time.perf_counter() - device_put_start_time,
            )

            preprocess_start_time = time.perf_counter()
            preprocessed_genotype_arrays = genotype_processing.preprocess_genotype_matrix_arrays(genotype_matrix_device)
            block_until_ready(preprocessed_genotype_arrays)
            record_stage_duration(
                stage_timing_accumulators,
                "preprocess_genotypes",
                time.perf_counter() - preprocess_start_time,
            )

            build_chunk_start_time = time.perf_counter()
            linear_genotype_chunk = models.LinearGenotypeChunk(
                genotypes=preprocessed_genotype_arrays.genotypes,
                metadata=reader.build_variant_metadata(variant_table_arrays, variant_start, variant_stop),
                allele_one_frequency=preprocessed_genotype_arrays.allele_one_frequency,
                observation_count=preprocessed_genotype_arrays.observation_count,
            )
            record_stage_duration(
                stage_timing_accumulators,
                "build_linear_chunk",
                time.perf_counter() - build_chunk_start_time,
            )
            yield linear_genotype_chunk

    setattr(
        engine_module,
        "iter_linear_genotype_chunks_from_source",
        profiled_iter_linear_genotype_chunks_from_source,
    )

    def restore_iter_linear() -> None:
        setattr(
            engine_module,
            "iter_linear_genotype_chunks_from_source",
            original_iter_linear_genotype_chunks_from_source,
        )

    restore_callbacks.append(restore_iter_linear)

    return TimingInstrumentationHandle(
        stage_timing_accumulators=stage_timing_accumulators,
        restore_callbacks=tuple(restore_callbacks),
    )


def restore_timing_instrumentation(instrumentation_handle: TimingInstrumentationHandle) -> None:
    """Restore all monkey-patched profiling hooks."""
    for restore_callback in reversed(instrumentation_handle.restore_callbacks):
        restore_callback()


def run_profiled_regenie2_linear(
    arguments: argparse.Namespace,
    report_paths: dict[str, Path],
) -> tuple[float, cProfile.Profile, list[StageTimingSummary], api.RunArtifacts]:
    """Execute one profiled REGENIE step 2 run and return timing artifacts."""
    instrumentation_handle = install_timing_instrumentation()
    profiler = cProfile.Profile()
    covariate_names = parse_covariate_names(arguments.covar_names)
    transfer_guard_context = (
        jax.transfer_guard(arguments.transfer_guard_device_to_host)
        if arguments.transfer_guard_device_to_host is not None
        else contextlib.nullcontext()
    )
    trace_context = (
        jax.profiler.trace(report_paths["trace_directory"], create_perfetto_trace=True)
        if arguments.enable_jax_trace
        else contextlib.nullcontext()
    )

    try:
        start_time = time.perf_counter()
        with transfer_guard_context, trace_context:
            profiler.enable()
            run_artifacts = api.regenie2_linear(
                bgen=arguments.bgen,
                sample=arguments.sample,
                pheno=arguments.pheno,
                pheno_name=arguments.pheno_name,
                out=report_paths["output_run_directory"],
                covar=arguments.covar,
                covar_names=covariate_names,
                pred=arguments.pred,
                compute=api.ComputeConfig(
                    chunk_size=arguments.chunk_size,
                    device=arguments.device,
                    variant_limit=arguments.variant_limit,
                    prefetch_chunks=arguments.prefetch_chunks,
                    output_mode=types.OutputMode.ARROW_CHUNKS,
                    output_run_directory=report_paths["output_run_directory"],
                    finalize_parquet=True,
                ),
            )
            profiler.disable()
        wall_time_seconds = time.perf_counter() - start_time
    finally:
        restore_timing_instrumentation(instrumentation_handle)

    stage_timing_summaries = build_stage_timing_summaries(
        instrumentation_handle.stage_timing_accumulators,
        wall_time_seconds,
    )
    return wall_time_seconds, profiler, stage_timing_summaries, run_artifacts


def run_warmup_passes(arguments: argparse.Namespace) -> None:
    """Run unprofiled warmup passes before the measured profile."""
    covariate_names = parse_covariate_names(arguments.covar_names)
    for warmup_pass_index in range(max(0, arguments.warmup_pass_count)):
        warmup_output_root = arguments.output_dir / f"{arguments.report_name}_warmup_{warmup_pass_index + 1}.run"
        if warmup_output_root.exists():
            shutil.rmtree(warmup_output_root)
        warmup_start_time = time.perf_counter()
        api.regenie2_linear(
            bgen=arguments.bgen,
            sample=arguments.sample,
            pheno=arguments.pheno,
            pheno_name=arguments.pheno_name,
            out=warmup_output_root,
            covar=arguments.covar,
            covar_names=covariate_names,
            pred=arguments.pred,
            compute=api.ComputeConfig(
                chunk_size=arguments.chunk_size,
                device=arguments.device,
                variant_limit=arguments.variant_limit,
                prefetch_chunks=arguments.prefetch_chunks,
                output_mode=types.OutputMode.ARROW_CHUNKS,
                output_run_directory=warmup_output_root,
                finalize_parquet=True,
            ),
        )
        print(
            f"Warmup pass {warmup_pass_index + 1}/{arguments.warmup_pass_count} complete "
            f"in {time.perf_counter() - warmup_start_time:.3f}s"
        )


def resolve_variant_count(final_parquet_path: Path | None) -> int:
    """Resolve the number of output variants from the finalized Parquet file."""
    if final_parquet_path is None or not final_parquet_path.exists():
        return 0
    count_frame = typing.cast("pl.DataFrame", pl.scan_parquet(final_parquet_path).select(pl.len()).collect())
    return int(count_frame.to_numpy()[0, 0])


def main() -> None:
    """Profile one REGENIE step 2 linear run and save reusable artifacts."""
    arguments = build_argument_parser().parse_args()
    report_paths = resolve_report_paths(arguments)
    remove_existing_outputs(report_paths)

    print("Profiling REGENIE step 2 linear association")
    print(f"Output directory: {report_paths['output_directory']}")
    print(f"Device: {arguments.device}")
    print(f"Chunk size: {arguments.chunk_size}")
    print("-" * 80)

    from g.jax_setup import configure_jax_device
    configure_jax_device(arguments.device)
    run_warmup_passes(arguments)

    wall_time_seconds, profiler, stage_timing_summaries, run_artifacts = run_profiled_regenie2_linear(
        arguments,
        report_paths,
    )
    total_variants = resolve_variant_count(run_artifacts.final_parquet)
    chunk_count = next(
        (
            stage_timing_summary.event_count
            for stage_timing_summary in stage_timing_summaries
            if stage_timing_summary.name == "compute_regenie2_linear_chunk"
        ),
        0,
    )
    variants_per_second = total_variants / wall_time_seconds if wall_time_seconds > 0.0 else 0.0

    generate_cprofile_report(profiler, report_paths["cprofile_path"], arguments.cprofile_sort)
    profiler.dump_stats(str(report_paths["cprofile_raw_path"]))

    memory_profile_path: Path | None = None
    if arguments.enable_memory_profile:
        memory_profile_path = report_paths["memory_profile_path"]
        jax.profiler.save_device_memory_profile(memory_profile_path)

    perfetto_trace_path = (
        find_perfetto_trace_path(report_paths["trace_directory"]) if arguments.enable_jax_trace else None
    )
    perfetto_event_summaries = summarize_perfetto_trace(perfetto_trace_path) if perfetto_trace_path is not None else []

    json_summary = Regenie2DetailedProfileSummary(
        backend=jax.default_backend(),
        device=str(arguments.device),
        bgen_path=str(arguments.bgen),
        sample_path=str(arguments.sample) if arguments.sample is not None else None,
        phenotype_path=str(arguments.pheno),
        phenotype_name=arguments.pheno_name,
        covariate_path=str(arguments.covar) if arguments.covar is not None else None,
        covariate_names=list(parse_covariate_names(arguments.covar_names) or []),
        prediction_list_path=str(arguments.pred),
        chunk_size=arguments.chunk_size,
        variant_limit=arguments.variant_limit,
        prefetch_chunks=arguments.prefetch_chunks,
        warmup_pass_count=arguments.warmup_pass_count,
        total_variants=total_variants,
        chunk_count=chunk_count,
        wall_time_seconds=wall_time_seconds,
        variants_per_second=variants_per_second,
        output_run_directory=str(run_artifacts.output_run_directory),
        final_parquet_path=str(run_artifacts.final_parquet) if run_artifacts.final_parquet is not None else None,
        trace_enabled=arguments.enable_jax_trace,
        memory_profile_enabled=arguments.enable_memory_profile,
        cprofile_path=str(report_paths["cprofile_path"]),
        cprofile_raw_path=str(report_paths["cprofile_raw_path"]),
        summary_path=str(report_paths["summary_path"]),
        trace_dir=str(report_paths["trace_directory"]) if arguments.enable_jax_trace else None,
        perfetto_trace_path=str(perfetto_trace_path) if perfetto_trace_path is not None else None,
        memory_profile_path=str(memory_profile_path) if memory_profile_path is not None else None,
        stage_timing_summaries=stage_timing_summaries,
        perfetto_event_summaries=perfetto_event_summaries,
    )
    report_paths["json_summary_path"].write_text(json.dumps(asdict(json_summary), indent=2))

    summary_report = format_summary_report(json_summary, profiler, arguments.cprofile_sort)
    report_paths["summary_path"].write_text(summary_report)

    print(f"Variants processed: {total_variants}")
    print(f"Chunks processed:   {chunk_count}")
    print(f"Wall clock time:    {wall_time_seconds:.3f} seconds")
    print(f"Throughput:         {variants_per_second:.1f} variants/second")
    print("Saved profiling artifacts:")
    print(f"  {report_paths['summary_path']}")
    print(f"  {report_paths['json_summary_path']}")
    print(f"  {report_paths['cprofile_path']}")
    print(f"  {report_paths['cprofile_raw_path']}")
    if arguments.enable_jax_trace:
        print(f"  {report_paths['trace_directory']}")
    if memory_profile_path is not None:
        print(f"  {memory_profile_path}")


if __name__ == "__main__":
    main()
