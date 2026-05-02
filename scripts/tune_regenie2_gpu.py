#!/usr/bin/env python3
"""Tune GPU REGENIE step 2 and BGEN reader knobs on the current machine."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import textwrap
import typing
from pathlib import Path

from g import api, types


def load_script_module(module_name: str, relative_path: str) -> typing.Any:
    """Load a sibling benchmark script as a module."""
    module_path = Path(__file__).resolve().parent / relative_path
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        message = f"Unable to load script module at {module_path}."
        raise ModuleNotFoundError(message)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


baseline_benchmark = load_script_module("tuning_baseline_benchmark", "benchmark.py")
benchmark_bgen_reader = load_script_module("tuning_bgen_reader_benchmark", "benchmark_bgen_reader.py")
benchmark_regenie_comparison = load_script_module(
    "tuning_regenie_comparison_benchmark",
    "benchmark_regenie_comparison.py",
)


DEFAULT_OUTPUT_DIRECTORY = Path("data/benchmarks/regenie2_gpu_tuning")
DEFAULT_BGEN_PRE_SWEEP_CHUNK_SIZE = 8192
DEFAULT_BGEN_PATH_MODE = benchmark_bgen_reader.BenchmarkPathMode.READ_FLOAT32_INTO_PREPARED


class TraitSelection(enum.StrEnum):
    """Selectable tuning scope."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"
    BOTH = "both"


@dataclasses.dataclass(frozen=True)
class BgenCandidate:
    """One low-level BGEN candidate configuration."""

    decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    benchmark_chunk_size: int


@dataclasses.dataclass(frozen=True)
class BgenCandidateSummary:
    """Measured BGEN reader summary for one candidate."""

    candidate: BgenCandidate
    median_seconds: float
    mean_seconds: float
    repeat_count: int


@dataclasses.dataclass(frozen=True)
class Step2TuningCandidate:
    """One end-to-end GPU REGENIE step 2 candidate."""

    trait_type: types.RegenieTraitType
    chunk_size: int
    prefetch_chunks: int
    arrow_payload_batch_size: int
    output_writer_thread_count: int
    output_writer_queue_depth: int
    bgen_decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    firth_batch_size: int | None


@dataclasses.dataclass(frozen=True)
class Step2TrialResult:
    """One fresh-process trial result."""

    wall_time_seconds: float
    output_path: str
    output_row_count: int


@dataclasses.dataclass(frozen=True)
class Step2CandidateSummary:
    """Aggregate summary for one end-to-end candidate."""

    candidate: Step2TuningCandidate
    warmup_count: int
    measured_trial_count: int
    median_wall_time_seconds: float
    mean_wall_time_seconds: float
    min_wall_time_seconds: float
    max_wall_time_seconds: float
    mean_rows_per_second: float
    trial_results: tuple[Step2TrialResult, ...]


@dataclasses.dataclass(frozen=True)
class ModeTuningSummary:
    """Final tuning summary for one trait mode."""

    trait_type: str
    regenie_baseline_seconds: float
    bgen_candidates: tuple[BgenCandidateSummary, ...]
    compute_stage_candidates: tuple[Step2CandidateSummary, ...]
    writer_stage_candidates: tuple[Step2CandidateSummary, ...]
    finalist_candidates: tuple[Step2CandidateSummary, ...]
    winner: Step2CandidateSummary
    slowdown_vs_regenie: float


@dataclasses.dataclass(frozen=True)
class TuningReport:
    """Top-level tuning report."""

    output_directory: str
    quantitative_summary: ModeTuningSummary | None
    binary_summary: ModeTuningSummary | None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the tuner CLI."""
    argument_parser = argparse.ArgumentParser(description="Tune GPU REGENIE step 2 and BGEN reader knobs.")
    argument_parser.add_argument(
        "--trait-selection",
        choices=tuple(trait_selection.value for trait_selection in TraitSelection),
        default=TraitSelection.BOTH.value,
        help="Which GPU REGENIE trait mode to tune.",
    )
    argument_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    argument_parser.add_argument(
        "--variant-limit",
        type=int,
        help="Optional variant cap for faster exploratory sweeps.",
    )
    argument_parser.add_argument("--warmup-trials", type=int, default=1)
    argument_parser.add_argument("--trials", type=int, default=3)
    argument_parser.add_argument("--finalist-extra-trials", type=int, default=5)
    argument_parser.add_argument("--top-bgen-candidates", type=int, default=3)
    argument_parser.add_argument("--top-compute-candidates", type=int, default=3)
    argument_parser.add_argument("--top-finalists", type=int, default=3)
    argument_parser.add_argument("--chunk-sizes", default="2048,4096,8192,16384")
    argument_parser.add_argument("--prefetch-chunks", default="0,1,2")
    argument_parser.add_argument("--arrow-payload-batch-sizes", default="1,2,4,8")
    argument_parser.add_argument("--output-writer-thread-counts", default="1,2,4,8")
    argument_parser.add_argument("--writer-queue-depth-multipliers", default="1,2")
    argument_parser.add_argument("--firth-batch-sizes", default="32,64,128")
    argument_parser.add_argument("--bgen-benchmark-chunk-size", type=int, default=DEFAULT_BGEN_PRE_SWEEP_CHUNK_SIZE)
    argument_parser.add_argument("--bgen-decode-tile-variant-counts", default="32,64,128,256")
    argument_parser.add_argument("--rayon-thread-counts", default="1,2,4,8")
    return argument_parser


def parse_required_int_list(raw_values: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list."""
    parsed_values = benchmark_bgen_reader.parse_optional_int_list(raw_values)
    integer_values = tuple(int(parsed_value) for parsed_value in parsed_values if parsed_value is not None)
    if not integer_values:
        message = "At least one integer value is required."
        raise ValueError(message)
    return integer_values


def build_queue_depth_values(
    output_writer_thread_count: int,
    queue_depth_multipliers: tuple[int, ...],
) -> tuple[int, ...]:
    """Build queue depths from a writer-thread count and one or more multipliers."""
    queue_depth_values = {
        max(1, output_writer_thread_count * queue_depth_multiplier)
        for queue_depth_multiplier in queue_depth_multipliers
    }
    return tuple(sorted(queue_depth_values))


def build_bgen_sweep_candidates(arguments: argparse.Namespace) -> tuple[BgenCandidate, ...]:
    """Build the initial low-level BGEN benchmark grid."""
    decode_tile_variant_counts = benchmark_bgen_reader.parse_optional_int_list(
        arguments.bgen_decode_tile_variant_counts
    )
    rayon_thread_counts = benchmark_bgen_reader.parse_optional_int_list(arguments.rayon_thread_counts)
    candidates: list[BgenCandidate] = []
    for decode_tile_variant_count in decode_tile_variant_counts:
        for rayon_thread_count in rayon_thread_counts:
            candidates.append(
                BgenCandidate(
                    decode_tile_variant_count=typing.cast("int | None", decode_tile_variant_count),
                    rayon_thread_count=typing.cast("int | None", rayon_thread_count),
                    benchmark_chunk_size=arguments.bgen_benchmark_chunk_size,
                )
            )
    return tuple(candidates)


def build_compute_stage_candidates(
    *,
    trait_type: types.RegenieTraitType,
    chunk_sizes: tuple[int, ...],
    prefetch_chunk_values: tuple[int, ...],
    bgen_candidates: tuple[BgenCandidateSummary, ...],
    firth_batch_sizes: tuple[int, ...],
) -> tuple[Step2TuningCandidate, ...]:
    """Build the first-stage end-to-end candidate grid."""
    candidates: list[Step2TuningCandidate] = []
    for bgen_candidate_summary in bgen_candidates:
        for chunk_size in chunk_sizes:
            for prefetch_chunks in prefetch_chunk_values:
                if trait_type == types.RegenieTraitType.BINARY:
                    for firth_batch_size in firth_batch_sizes:
                        candidates.append(
                            Step2TuningCandidate(
                                trait_type=trait_type,
                                chunk_size=chunk_size,
                                prefetch_chunks=prefetch_chunks,
                                arrow_payload_batch_size=api.DEFAULT_ARROW_PAYLOAD_BATCH_SIZE,
                                output_writer_thread_count=api.output.DEFAULT_WRITER_THREAD_COUNT,
                                output_writer_queue_depth=api.DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH,
                                bgen_decode_tile_variant_count=bgen_candidate_summary.candidate.decode_tile_variant_count,
                                rayon_thread_count=bgen_candidate_summary.candidate.rayon_thread_count,
                                firth_batch_size=firth_batch_size,
                            )
                        )
                    continue
                candidates.append(
                    Step2TuningCandidate(
                        trait_type=trait_type,
                        chunk_size=chunk_size,
                        prefetch_chunks=prefetch_chunks,
                        arrow_payload_batch_size=api.DEFAULT_ARROW_PAYLOAD_BATCH_SIZE,
                        output_writer_thread_count=api.output.DEFAULT_WRITER_THREAD_COUNT,
                        output_writer_queue_depth=api.DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH,
                        bgen_decode_tile_variant_count=bgen_candidate_summary.candidate.decode_tile_variant_count,
                        rayon_thread_count=bgen_candidate_summary.candidate.rayon_thread_count,
                        firth_batch_size=None,
                    )
                )
    return tuple(candidates)


def build_writer_stage_candidates(
    *,
    compute_stage_candidates: tuple[Step2CandidateSummary, ...],
    arrow_payload_batch_sizes: tuple[int, ...],
    output_writer_thread_counts: tuple[int, ...],
    writer_queue_depth_multipliers: tuple[int, ...],
) -> tuple[Step2TuningCandidate, ...]:
    """Expand the top compute candidates across writer-side knobs."""
    candidates: list[Step2TuningCandidate] = []
    for compute_stage_candidate in compute_stage_candidates:
        for arrow_payload_batch_size in arrow_payload_batch_sizes:
            for output_writer_thread_count in output_writer_thread_counts:
                for output_writer_queue_depth in build_queue_depth_values(
                    output_writer_thread_count,
                    writer_queue_depth_multipliers,
                ):
                    candidates.append(
                        dataclasses.replace(
                            compute_stage_candidate.candidate,
                            arrow_payload_batch_size=arrow_payload_batch_size,
                            output_writer_thread_count=output_writer_thread_count,
                            output_writer_queue_depth=output_writer_queue_depth,
                        )
                    )
    return tuple(candidates)


def build_step2_trial_environment(
    candidate: Step2TuningCandidate,
) -> dict[str, str]:
    """Build the environment for one fresh-process end-to-end trial."""
    environment = dict(os.environ)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".50")
    if candidate.bgen_decode_tile_variant_count is None:
        environment.pop("G_BGEN_DECODE_TILE_VARIANT_COUNT", None)
    else:
        environment["G_BGEN_DECODE_TILE_VARIANT_COUNT"] = str(candidate.bgen_decode_tile_variant_count)
    if candidate.rayon_thread_count is None:
        environment.pop("RAYON_NUM_THREADS", None)
    else:
        environment["RAYON_NUM_THREADS"] = str(candidate.rayon_thread_count)
    if candidate.firth_batch_size is None:
        environment.pop("G_REGENIE2_BINARY_FIRTH_BATCH_SIZE", None)
    else:
        environment["G_REGENIE2_BINARY_FIRTH_BATCH_SIZE"] = str(candidate.firth_batch_size)
    return environment


def build_step2_child_command(
    *,
    baseline_paths: baseline_benchmark.BaselinePaths,
    candidate: Step2TuningCandidate,
    output_path: Path,
    variant_limit: int | None,
) -> list[str]:
    """Build one fresh-process child command for a GPU REGENIE step 2 trial."""
    phenotype_path = baseline_paths.continuous_phenotype_path
    phenotype_name = "phenotype_continuous"
    prediction_list_path = baseline_paths.regenie_qt_prediction_list_path
    if candidate.trait_type == types.RegenieTraitType.BINARY:
        phenotype_path = baseline_paths.binary_phenotype_path
        phenotype_name = "phenotype_binary"
        prediction_list_path = baseline_paths.regenie_prediction_list_path
    trait_type_literal = candidate.trait_type.value
    variant_limit_expression = "None" if variant_limit is None else str(variant_limit)
    binary_config_expression = "None"
    if candidate.trait_type == types.RegenieTraitType.BINARY:
        binary_config_expression = (
            "api.Regenie2BinaryConfig(correction=types.RegenieBinaryCorrection.FIRTH_APPROXIMATE)"
        )
    child_code = textwrap.dedent(
        """
        import json
        import time

        import polars as pl

        from g import api, types

        start_time = time.perf_counter()
        artifacts = api.regenie2(
            bgen={bgen_path!r},
            sample={sample_path!r},
            pheno={phenotype_path!r},
            pheno_name={phenotype_name!r},
            out={output_path!r},
            covar={covariate_path!r},
            covar_names="age,sex",
            pred={prediction_path!r},
            trait_type=types.RegenieTraitType({trait_type!r}),
            compute=api.ComputeConfig(
                device=types.Device.GPU,
                chunk_size={chunk_size},
                variant_limit={variant_limit_expression},
                prefetch_chunks={prefetch_chunks},
                finalize_parquet=True,
                arrow_payload_batch_size={arrow_payload_batch_size},
                output_writer_thread_count={output_writer_thread_count},
                output_writer_queue_depth={output_writer_queue_depth},
            ),
            binary={binary_config_expression},
        )
        wall_time_seconds = time.perf_counter() - start_time
        output_row_count = pl.scan_parquet(artifacts.final_parquet).select(pl.len()).collect().item()
        print(
            json.dumps(
                {{
                    "wall_time_seconds": wall_time_seconds,
                    "output_path": str(artifacts.final_parquet),
                    "output_row_count": output_row_count,
                }}
            )
        )
        """
    ).format(
        bgen_path=str(baseline_paths.bgen_path),
        sample_path=str(baseline_paths.sample_path),
        phenotype_path=str(phenotype_path),
        phenotype_name=phenotype_name,
        output_path=str(output_path),
        covariate_path=str(baseline_paths.covariate_path),
        prediction_path=str(prediction_list_path),
        trait_type=trait_type_literal,
        chunk_size=candidate.chunk_size,
        variant_limit_expression=variant_limit_expression,
        prefetch_chunks=candidate.prefetch_chunks,
        arrow_payload_batch_size=candidate.arrow_payload_batch_size,
        output_writer_thread_count=candidate.output_writer_thread_count,
        output_writer_queue_depth=candidate.output_writer_queue_depth,
        binary_config_expression=binary_config_expression,
    )
    return [sys.executable, "-c", child_code]


def build_candidate_slug(candidate: Step2TuningCandidate) -> str:
    """Build a deterministic slug for one tuning candidate."""
    candidate_parts = [
        candidate.trait_type.value,
        f"chunk{candidate.chunk_size}",
        f"prefetch{candidate.prefetch_chunks}",
        f"arrow{candidate.arrow_payload_batch_size}",
        f"writer{candidate.output_writer_thread_count}",
        f"queue{candidate.output_writer_queue_depth}",
        f"tile{resolve_optional_label(candidate.bgen_decode_tile_variant_count)}",
        f"rayon{candidate.rayon_thread_count if candidate.rayon_thread_count is not None else 'default'}",
    ]
    if candidate.firth_batch_size is not None:
        candidate_parts.append(f"firth{candidate.firth_batch_size}")
    return "_".join(candidate_parts)


def resolve_optional_label(value: int | None) -> str:
    """Resolve one optional integer into a deterministic string label."""
    if value is None:
        return "default"
    return str(value)


def run_step2_trial(
    *,
    trial_index: int,
    baseline_paths: baseline_benchmark.BaselinePaths,
    candidate: Step2TuningCandidate,
    output_directory: Path,
    variant_limit: int | None,
) -> Step2TrialResult:
    """Run one fresh-process end-to-end trial."""
    output_path = output_directory / f"{build_candidate_slug(candidate)}_trial{trial_index:02d}"
    command_arguments = build_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_path=output_path,
        variant_limit=variant_limit,
    )
    completed_process = subprocess.run(
        command_arguments,
        check=True,
        capture_output=True,
        text=True,
        env=build_step2_trial_environment(candidate),
    )
    result_payload = json.loads(completed_process.stdout.strip().splitlines()[-1])
    return Step2TrialResult(
        wall_time_seconds=float(result_payload["wall_time_seconds"]),
        output_path=str(result_payload["output_path"]),
        output_row_count=int(result_payload["output_row_count"]),
    )


def summarize_step2_candidate(
    *,
    candidate: Step2TuningCandidate,
    warmup_count: int,
    trial_results: tuple[Step2TrialResult, ...],
) -> Step2CandidateSummary:
    """Summarize one end-to-end candidate."""
    wall_time_values = [trial_result.wall_time_seconds for trial_result in trial_results]
    row_rate_values = [trial_result.output_row_count / trial_result.wall_time_seconds for trial_result in trial_results]
    return Step2CandidateSummary(
        candidate=candidate,
        warmup_count=warmup_count,
        measured_trial_count=len(trial_results),
        median_wall_time_seconds=statistics.median(wall_time_values),
        mean_wall_time_seconds=statistics.fmean(wall_time_values),
        min_wall_time_seconds=min(wall_time_values),
        max_wall_time_seconds=max(wall_time_values),
        mean_rows_per_second=statistics.fmean(row_rate_values),
        trial_results=trial_results,
    )


def sort_step2_candidate_summaries(
    candidate_summaries: tuple[Step2CandidateSummary, ...],
) -> tuple[Step2CandidateSummary, ...]:
    """Sort candidate summaries by median then mean wall time."""
    return tuple(
        sorted(
            candidate_summaries,
            key=lambda candidate_summary: (
                candidate_summary.median_wall_time_seconds,
                candidate_summary.mean_wall_time_seconds,
            ),
        )
    )


def evaluate_step2_candidates(
    *,
    baseline_paths: baseline_benchmark.BaselinePaths,
    candidates: tuple[Step2TuningCandidate, ...],
    output_directory: Path,
    warmup_trials: int,
    measured_trials: int,
    variant_limit: int | None,
) -> tuple[Step2CandidateSummary, ...]:
    """Evaluate one or more end-to-end candidates sequentially."""
    candidate_summaries: list[Step2CandidateSummary] = []
    for candidate in candidates:
        for warmup_index in range(warmup_trials):
            _ = run_step2_trial(
                trial_index=-(warmup_index + 1),
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory,
                variant_limit=variant_limit,
            )
        trial_results = tuple(
            run_step2_trial(
                trial_index=trial_index,
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory,
                variant_limit=variant_limit,
            )
            for trial_index in range(measured_trials)
        )
        candidate_summaries.append(
            summarize_step2_candidate(
                candidate=candidate,
                warmup_count=warmup_trials,
                trial_results=trial_results,
            )
        )
    return sort_step2_candidate_summaries(tuple(candidate_summaries))


def summarize_bgen_candidate(
    case_report: benchmark_bgen_reader.BenchmarkCaseReport,
) -> BgenCandidateSummary:
    """Summarize one BGEN reader benchmark case."""
    matching_path_results = [
        path_result for path_result in case_report.path_results if path_result.path_mode == DEFAULT_BGEN_PATH_MODE.value
    ]
    if len(matching_path_results) != 1:
        message = "Expected exactly one production reusable-buffer BGEN path result."
        raise ValueError(message)
    path_result = matching_path_results[0]
    return BgenCandidateSummary(
        candidate=BgenCandidate(
            decode_tile_variant_count=case_report.decode_tile_variant_count,
            rayon_thread_count=case_report.rayon_thread_count,
            benchmark_chunk_size=case_report.chunk_size,
        ),
        median_seconds=statistics.median(path_result.durations_seconds),
        mean_seconds=path_result.mean_seconds,
        repeat_count=case_report.repeat_count,
    )


def run_bgen_pre_sweep(
    arguments: argparse.Namespace,
    baseline_paths: baseline_benchmark.BaselinePaths,
) -> tuple[BgenCandidateSummary, ...]:
    """Run the reusable-buffer BGEN pre-sweep and rank candidates by median wall time."""
    parser = benchmark_bgen_reader.build_argument_parser()
    candidate_summaries: list[BgenCandidateSummary] = []
    for candidate in build_bgen_sweep_candidates(arguments):
        candidate_argument_list = [
            "--bgen",
            str(baseline_paths.bgen_path),
            "--sample",
            str(baseline_paths.sample_path),
            "--chunk-size",
            str(candidate.benchmark_chunk_size),
            "--variant-limit",
            str(arguments.variant_limit or 16384),
            "--repeat-count",
            str(arguments.trials),
            "--path-modes",
            DEFAULT_BGEN_PATH_MODE.value,
        ]
        if candidate.decode_tile_variant_count is not None:
            candidate_argument_list.extend(
                ["--decode-tile-variant-count", str(candidate.decode_tile_variant_count)]
            )
        if candidate.rayon_thread_count is not None:
            candidate_argument_list.extend(["--rayon-thread-count", str(candidate.rayon_thread_count)])
        benchmark_arguments = parser.parse_args(
            candidate_argument_list
        )
        candidate_summaries.append(
            summarize_bgen_candidate(benchmark_bgen_reader.run_case_subprocess(
                benchmark_arguments,
                candidate.benchmark_chunk_size,
                candidate.decode_tile_variant_count,
                candidate.rayon_thread_count,
                trusted_no_missing_diploid=False,
            ))
        )
    return tuple(
        sorted(
            candidate_summaries,
            key=lambda candidate_summary: (candidate_summary.median_seconds, candidate_summary.mean_seconds),
        )
    )


def run_regenie_baseline_step2(
    *,
    trait_type: types.RegenieTraitType,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_directory: Path,
) -> float:
    """Run the matching original REGENIE step 2 baseline once and return wall time."""
    output_directory.mkdir(parents=True, exist_ok=True)
    log_directory = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    program_specs = benchmark_regenie_comparison.build_regenie_program_specs(
        "regenie",
        baseline_paths,
        only_quantitative_step2=trait_type == types.RegenieTraitType.QUANTITATIVE,
        only_binary_step2=trait_type == types.RegenieTraitType.BINARY,
    )
    program_name, trait_type_name, step, command_arguments, output_prefix = program_specs[0]
    result = benchmark_regenie_comparison.run_regenie_program(
        program_name=program_name,
        trait_type=trait_type_name,
        step=step,
        command_arguments=command_arguments,
        output_prefix=output_prefix,
        log_directory=log_directory,
    )
    if result.wall_time_seconds is None:
        message = f"Original REGENIE {trait_type.value} step 2 run did not produce a wall-clock duration."
        raise ValueError(message)
    return result.wall_time_seconds


def tune_trait_mode(
    *,
    trait_type: types.RegenieTraitType,
    arguments: argparse.Namespace,
    baseline_paths: baseline_benchmark.BaselinePaths,
    bgen_candidate_summaries: tuple[BgenCandidateSummary, ...],
) -> ModeTuningSummary:
    """Tune one GPU REGENIE step 2 trait mode."""
    mode_output_directory = arguments.output_dir / trait_type.value
    compute_stage_candidates = build_compute_stage_candidates(
        trait_type=trait_type,
        chunk_sizes=parse_required_int_list(arguments.chunk_sizes),
        prefetch_chunk_values=parse_required_int_list(arguments.prefetch_chunks),
        bgen_candidates=bgen_candidate_summaries[: arguments.top_bgen_candidates],
        firth_batch_sizes=parse_required_int_list(arguments.firth_batch_sizes),
    )
    compute_stage_candidate_summaries = evaluate_step2_candidates(
        baseline_paths=baseline_paths,
        candidates=compute_stage_candidates,
        output_directory=mode_output_directory / "compute_stage",
        warmup_trials=arguments.warmup_trials,
        measured_trials=arguments.trials,
        variant_limit=arguments.variant_limit,
    )
    writer_stage_candidates = build_writer_stage_candidates(
        compute_stage_candidates=compute_stage_candidate_summaries[: arguments.top_compute_candidates],
        arrow_payload_batch_sizes=parse_required_int_list(arguments.arrow_payload_batch_sizes),
        output_writer_thread_counts=parse_required_int_list(arguments.output_writer_thread_counts),
        writer_queue_depth_multipliers=parse_required_int_list(arguments.writer_queue_depth_multipliers),
    )
    writer_stage_candidate_summaries = evaluate_step2_candidates(
        baseline_paths=baseline_paths,
        candidates=writer_stage_candidates,
        output_directory=mode_output_directory / "writer_stage",
        warmup_trials=arguments.warmup_trials,
        measured_trials=arguments.trials,
        variant_limit=arguments.variant_limit,
    )
    finalist_candidates = tuple(
        candidate_summary.candidate
        for candidate_summary in writer_stage_candidate_summaries[: arguments.top_finalists]
    )
    finalist_candidate_summaries = evaluate_step2_candidates(
        baseline_paths=baseline_paths,
        candidates=finalist_candidates,
        output_directory=mode_output_directory / "finalists",
        warmup_trials=arguments.warmup_trials,
        measured_trials=arguments.trials + arguments.finalist_extra_trials,
        variant_limit=arguments.variant_limit,
    )
    winner = finalist_candidate_summaries[0]
    regenie_baseline_seconds = run_regenie_baseline_step2(
        trait_type=trait_type,
        baseline_paths=baseline_paths,
        output_directory=mode_output_directory / "regenie_baseline",
    )
    return ModeTuningSummary(
        trait_type=trait_type.value,
        regenie_baseline_seconds=regenie_baseline_seconds,
        bgen_candidates=bgen_candidate_summaries,
        compute_stage_candidates=compute_stage_candidate_summaries,
        writer_stage_candidates=writer_stage_candidate_summaries,
        finalist_candidates=finalist_candidate_summaries,
        winner=winner,
        slowdown_vs_regenie=winner.median_wall_time_seconds / regenie_baseline_seconds,
    )


def write_text_summary(
    report_path: Path,
    tuning_report: TuningReport,
) -> None:
    """Write a compact human-readable tuning summary."""
    lines = ["REGENIE GPU Tuning Summary", ""]
    for mode_summary in (tuning_report.quantitative_summary, tuning_report.binary_summary):
        if mode_summary is None:
            continue
        winner = mode_summary.winner
        lines.extend(
            [
                f"{mode_summary.trait_type.upper()}",
                f"  REGENIE baseline: {mode_summary.regenie_baseline_seconds:.6f}s",
                f"  Winner median:    {winner.median_wall_time_seconds:.6f}s",
                f"  Slowdown:         {mode_summary.slowdown_vs_regenie:.4f}x",
                f"  Candidate:        {dataclasses.asdict(winner.candidate)}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full sequential tuning workflow."""
    arguments = build_argument_parser().parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_paths = baseline_benchmark.build_baseline_paths()
    bgen_candidate_summaries = run_bgen_pre_sweep(arguments, baseline_paths)

    quantitative_summary: ModeTuningSummary | None = None
    binary_summary: ModeTuningSummary | None = None
    trait_selection = TraitSelection(arguments.trait_selection)
    if trait_selection in {TraitSelection.QUANTITATIVE, TraitSelection.BOTH}:
        quantitative_summary = tune_trait_mode(
            trait_type=types.RegenieTraitType.QUANTITATIVE,
            arguments=arguments,
            baseline_paths=baseline_paths,
            bgen_candidate_summaries=bgen_candidate_summaries,
        )
    if trait_selection in {TraitSelection.BINARY, TraitSelection.BOTH}:
        binary_summary = tune_trait_mode(
            trait_type=types.RegenieTraitType.BINARY,
            arguments=arguments,
            baseline_paths=baseline_paths,
            bgen_candidate_summaries=bgen_candidate_summaries,
        )

    tuning_report = TuningReport(
        output_directory=str(arguments.output_dir),
        quantitative_summary=quantitative_summary,
        binary_summary=binary_summary,
    )
    json_report_path = arguments.output_dir / "tuning_report.json"
    json_report_path.write_text(json.dumps(dataclasses.asdict(tuning_report), indent=2) + "\n", encoding="utf-8")
    text_report_path = arguments.output_dir / "tuning_summary.txt"
    write_text_summary(text_report_path, tuning_report)
    print(json.dumps(dataclasses.asdict(tuning_report), indent=2))


if __name__ == "__main__":
    main()
