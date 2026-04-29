#!/usr/bin/env python3
"""Search for an effective full-chromosome chunk size for GWAS runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from g.engine import iter_linear_output_frames, iter_logistic_output_frames

DEFAULT_LINEAR_PHENOTYPE_PATH = Path("data/pheno_cont.txt")
DEFAULT_LOGISTIC_PHENOTYPE_PATH = Path("data/pheno_bin.txt")
DEFAULT_COVARIATE_PATH = Path("data/covariates.txt")
DEFAULT_BED_PREFIX = Path("data/1kg_chr22_full")
DEFAULT_MINIMUM_CHUNK_SIZE = 512
DEFAULT_MAXIMUM_CHUNK_SIZE = 8192
DEFAULT_GROWTH_FACTOR = 2.0
DEFAULT_REPEAT_COUNT = 3
DEFAULT_STOPPING_SLOWDOWN_FRACTION = 0.05
DEFAULT_REFINEMENT_IMPROVEMENT_FLOOR = 0.03
CHUNK_SIZE_ALIGNMENT = 256
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SingleRunSummary:
    """Measured output for one full-chromosome run."""

    glm_mode: str
    chunk_size: int
    wall_time_seconds: float
    total_variants: int
    chunk_count: int
    variants_per_second: float
    checksum: float
    firth_variant_count: int
    standard_variant_count: int


@dataclass(frozen=True)
class ChunkSizeEvaluation:
    """Aggregated cold and warmed measurements for one chunk size."""

    chunk_size: int
    cold_run: SingleRunSummary
    warmed_runs: list[SingleRunSummary]
    median_wall_time_seconds: float
    median_variants_per_second: float
    best_wall_time_seconds: float
    worst_wall_time_seconds: float


@dataclass(frozen=True)
class ChunkSearchReport:
    """Search report for one model's full-chromosome chunk-size tuning."""

    glm_mode: str
    minimum_chunk_size: int
    maximum_chunk_size: int
    growth_factor: float
    repeat_count: int
    stopping_slowdown_fraction: float
    refinement_improvement_floor: float
    coarse_chunk_sizes: list[int]
    refined_chunk_sizes: list[int]
    evaluations: list[ChunkSizeEvaluation]
    recommended_chunk_size: int
    recommended_median_wall_time_seconds: float
    recommended_median_variants_per_second: float


def round_chunk_size(candidate_chunk_size: float) -> int:
    """Round a candidate chunk size to the repository alignment."""
    aligned_chunk_size = int(round(candidate_chunk_size / CHUNK_SIZE_ALIGNMENT) * CHUNK_SIZE_ALIGNMENT)
    return max(CHUNK_SIZE_ALIGNMENT, aligned_chunk_size)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for chunk-size search."""
    parser = argparse.ArgumentParser(description="Search full-chromosome chunk sizes for GWAS runs.")
    parser.add_argument("--bfile", type=Path, default=DEFAULT_BED_PREFIX, help="PLINK dataset prefix.")
    parser.add_argument("--covar", type=Path, default=DEFAULT_COVARIATE_PATH, help="Covariate table path.")
    parser.add_argument("--covar-names", default="age,sex", help="Comma-separated covariate names.")
    parser.add_argument("--glm", required=True, choices=("linear", "logistic"), help="Association model.")
    parser.add_argument(
        "--min-chunk-size", type=int, default=DEFAULT_MINIMUM_CHUNK_SIZE, help="Smallest candidate chunk size."
    )
    parser.add_argument(
        "--max-chunk-size", type=int, default=DEFAULT_MAXIMUM_CHUNK_SIZE, help="Largest coarse-sweep chunk size."
    )
    parser.add_argument(
        "--growth-factor", type=float, default=DEFAULT_GROWTH_FACTOR, help="Geometric growth factor for coarse sweep."
    )
    parser.add_argument(
        "--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT, help="Number of warmed full-run repetitions."
    )
    parser.add_argument(
        "--stopping-slowdown-fraction",
        type=float,
        default=DEFAULT_STOPPING_SLOWDOWN_FRACTION,
        help="Stop coarse growth after two slower sizes beyond this fraction.",
    )
    parser.add_argument(
        "--refinement-improvement-floor",
        type=float,
        default=DEFAULT_REFINEMENT_IMPROVEMENT_FLOOR,
        help="Minimum improvement fraction required to keep refining.",
    )
    parser.add_argument("--output-path", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--single-run", action="store_true", help="Internal mode: run exactly one full benchmark and print JSON."
    )
    parser.add_argument("--chunk-size", type=int, help="Chunk size for --single-run mode.")
    parser.add_argument("--pheno", type=Path, help="Optional phenotype path override.")
    parser.add_argument("--pheno-name", help="Optional phenotype column override.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for debugging.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum logistic IRLS iterations.")
    parser.add_argument("--tolerance", type=float, default=1.0e-8, help="Logistic convergence tolerance.")
    return parser


def parse_covariate_names(raw_covariate_names: str) -> tuple[str, ...] | None:
    """Parse a comma-separated covariate name list."""
    covariate_names = tuple(name.strip() for name in raw_covariate_names.split(",") if name.strip())
    return covariate_names or None


def resolve_phenotype_path(glm_mode: str, phenotype_path: Path | None) -> Path:
    """Resolve the phenotype table path for the requested model."""
    if phenotype_path is not None:
        return phenotype_path
    if glm_mode == "linear":
        return DEFAULT_LINEAR_PHENOTYPE_PATH
    return DEFAULT_LOGISTIC_PHENOTYPE_PATH


def resolve_phenotype_name(glm_mode: str, phenotype_name: str | None) -> str:
    """Resolve the phenotype column name for the requested model."""
    if phenotype_name is not None:
        return phenotype_name
    if glm_mode == "linear":
        return "phenotype_continuous"
    return "phenotype_binary"


def run_single_full_chromosome_benchmark(arguments: argparse.Namespace) -> SingleRunSummary:
    """Run one full-chromosome benchmark in the current process."""
    if arguments.chunk_size is None:
        raise ValueError("--chunk-size is required in --single-run mode.")

    bed_prefix = arguments.bfile
    covariate_names = parse_covariate_names(arguments.covar_names)
    phenotype_path = resolve_phenotype_path(arguments.glm, arguments.pheno)
    phenotype_name = resolve_phenotype_name(arguments.glm, arguments.pheno_name)

    total_variants = 0
    chunk_count = 0
    checksum = 0.0
    firth_variant_count = 0
    standard_variant_count = 0

    start_time = time.perf_counter()
    if arguments.glm == "linear":
        for linear_chunk_accumulator in iter_linear_output_frames(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
        ):
            total_variants += len(linear_chunk_accumulator.metadata.variant_identifiers)
            chunk_count += 1
            checksum += float(np.asarray(linear_chunk_accumulator.linear_result.beta).sum())
            checksum += float(np.asarray(linear_chunk_accumulator.linear_result.p_value).sum())
    else:
        for logistic_chunk_accumulator in iter_logistic_output_frames(
            bed_prefix=bed_prefix,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=arguments.chunk_size,
            variant_limit=arguments.variant_limit,
            max_iterations=arguments.max_iterations,
            tolerance=arguments.tolerance,
        ):
            total_variants += len(logistic_chunk_accumulator.metadata.variant_identifiers)
            chunk_count += 1
            checksum += float(np.asarray(logistic_chunk_accumulator.logistic_result.beta).sum())
            checksum += float(np.asarray(logistic_chunk_accumulator.logistic_result.p_value).sum())
            method_code = np.asarray(logistic_chunk_accumulator.logistic_result.method_code)
            firth_variant_count += int(np.count_nonzero(method_code == 1))
            standard_variant_count += int(np.count_nonzero(method_code == 0))
    wall_time_seconds = time.perf_counter() - start_time

    return SingleRunSummary(
        glm_mode=arguments.glm,
        chunk_size=arguments.chunk_size,
        wall_time_seconds=wall_time_seconds,
        total_variants=total_variants,
        chunk_count=chunk_count,
        variants_per_second=total_variants / wall_time_seconds,
        checksum=checksum,
        firth_variant_count=firth_variant_count,
        standard_variant_count=standard_variant_count,
    )


def run_subprocess_benchmark(arguments: argparse.Namespace, chunk_size: int) -> SingleRunSummary:
    """Run one fresh-process full benchmark for a specific chunk size."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        "--glm",
        arguments.glm,
        "--bfile",
        str(arguments.bfile),
        "--covar",
        str(arguments.covar),
        "--covar-names",
        arguments.covar_names,
        "--chunk-size",
        str(chunk_size),
        "--max-iterations",
        str(arguments.max_iterations),
        "--tolerance",
        str(arguments.tolerance),
    ]
    if arguments.pheno is not None:
        command.extend(["--pheno", str(arguments.pheno)])
    if arguments.pheno_name is not None:
        command.extend(["--pheno-name", arguments.pheno_name])
    if arguments.variant_limit is not None:
        command.extend(["--variant-limit", str(arguments.variant_limit)])

    completed_process = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return SingleRunSummary(**json.loads(completed_process.stdout))


def build_coarse_chunk_sizes(minimum_chunk_size: int, maximum_chunk_size: int, growth_factor: float) -> list[int]:
    """Build the geometric coarse-sweep chunk sizes."""
    chunk_sizes: list[int] = []
    current_chunk_size = round_chunk_size(minimum_chunk_size)
    maximum_aligned_chunk_size = round_chunk_size(maximum_chunk_size)
    while current_chunk_size <= maximum_aligned_chunk_size:
        if current_chunk_size not in chunk_sizes:
            chunk_sizes.append(current_chunk_size)
        next_chunk_size = round_chunk_size(current_chunk_size * growth_factor)
        if next_chunk_size <= current_chunk_size:
            next_chunk_size = current_chunk_size + CHUNK_SIZE_ALIGNMENT
        current_chunk_size = next_chunk_size
    if chunk_sizes[-1] != maximum_aligned_chunk_size:
        chunk_sizes.append(maximum_aligned_chunk_size)
    return chunk_sizes


def evaluate_chunk_size(arguments: argparse.Namespace, chunk_size: int) -> ChunkSizeEvaluation:
    """Collect one cold run and warmed runs for a chunk size."""
    cold_run = run_subprocess_benchmark(arguments, chunk_size)
    warmed_runs = [run_subprocess_benchmark(arguments, chunk_size) for _ in range(arguments.repeat_count)]
    warmed_wall_times = [run.wall_time_seconds for run in warmed_runs]
    warmed_throughputs = [run.variants_per_second for run in warmed_runs]
    return ChunkSizeEvaluation(
        chunk_size=chunk_size,
        cold_run=cold_run,
        warmed_runs=warmed_runs,
        median_wall_time_seconds=statistics.median(warmed_wall_times),
        median_variants_per_second=statistics.median(warmed_throughputs),
        best_wall_time_seconds=min(warmed_wall_times),
        worst_wall_time_seconds=max(warmed_wall_times),
    )


def select_best_evaluation(evaluations: list[ChunkSizeEvaluation]) -> ChunkSizeEvaluation:
    """Select the evaluation with the fastest median warmed wall time."""
    return min(evaluations, key=lambda evaluation: evaluation.median_wall_time_seconds)


def should_stop_coarse_growth(
    evaluations: list[ChunkSizeEvaluation],
    stopping_slowdown_fraction: float,
) -> bool:
    """Determine whether coarse growth has slowed enough to stop exploring upward."""
    if len(evaluations) < 3:
        return False
    best_evaluation = select_best_evaluation(evaluations)
    last_two_evaluations = evaluations[-2:]
    return all(
        evaluation.median_wall_time_seconds
        > best_evaluation.median_wall_time_seconds * (1.0 + stopping_slowdown_fraction)
        for evaluation in last_two_evaluations
    )


def build_refinement_chunk_sizes(best_chunk_size: int, evaluated_chunk_sizes: set[int]) -> list[int]:
    """Build local refinement candidates around the current best chunk size."""
    lower_candidate = round_chunk_size(best_chunk_size / math.sqrt(2.0))
    upper_candidate = round_chunk_size(best_chunk_size * math.sqrt(2.0))
    refinement_chunk_sizes = []
    for candidate_chunk_size in [lower_candidate, upper_candidate]:
        if candidate_chunk_size not in evaluated_chunk_sizes and candidate_chunk_size >= CHUNK_SIZE_ALIGNMENT:
            refinement_chunk_sizes.append(candidate_chunk_size)
    return sorted(refinement_chunk_sizes)


def run_search(arguments: argparse.Namespace) -> ChunkSearchReport:
    """Run the coarse sweep and local refinement search."""
    coarse_chunk_sizes = build_coarse_chunk_sizes(
        minimum_chunk_size=arguments.min_chunk_size,
        maximum_chunk_size=arguments.max_chunk_size,
        growth_factor=arguments.growth_factor,
    )
    evaluations: list[ChunkSizeEvaluation] = []
    evaluated_chunk_sizes: set[int] = set()
    for chunk_size in coarse_chunk_sizes:
        evaluation = evaluate_chunk_size(arguments, chunk_size)
        evaluations.append(evaluation)
        evaluated_chunk_sizes.add(chunk_size)
        if should_stop_coarse_growth(evaluations, arguments.stopping_slowdown_fraction):
            break

    refined_chunk_sizes: list[int] = []
    best_evaluation = select_best_evaluation(evaluations)
    refinement_candidates = build_refinement_chunk_sizes(best_evaluation.chunk_size, evaluated_chunk_sizes)
    while refinement_candidates:
        candidate_chunk_size = refinement_candidates.pop(0)
        evaluation = evaluate_chunk_size(arguments, candidate_chunk_size)
        evaluations.append(evaluation)
        evaluated_chunk_sizes.add(candidate_chunk_size)
        refined_chunk_sizes.append(candidate_chunk_size)
        updated_best_evaluation = select_best_evaluation(evaluations)
        improvement_fraction = (
            best_evaluation.median_wall_time_seconds - updated_best_evaluation.median_wall_time_seconds
        ) / best_evaluation.median_wall_time_seconds
        best_evaluation = updated_best_evaluation
        if improvement_fraction < arguments.refinement_improvement_floor:
            break
        refinement_candidates = build_refinement_chunk_sizes(best_evaluation.chunk_size, evaluated_chunk_sizes)

    sorted_evaluations = sorted(evaluations, key=lambda evaluation: evaluation.chunk_size)
    recommended_evaluation = select_best_evaluation(sorted_evaluations)
    return ChunkSearchReport(
        glm_mode=arguments.glm,
        minimum_chunk_size=arguments.min_chunk_size,
        maximum_chunk_size=arguments.max_chunk_size,
        growth_factor=arguments.growth_factor,
        repeat_count=arguments.repeat_count,
        stopping_slowdown_fraction=arguments.stopping_slowdown_fraction,
        refinement_improvement_floor=arguments.refinement_improvement_floor,
        coarse_chunk_sizes=coarse_chunk_sizes,
        refined_chunk_sizes=refined_chunk_sizes,
        evaluations=sorted_evaluations,
        recommended_chunk_size=recommended_evaluation.chunk_size,
        recommended_median_wall_time_seconds=recommended_evaluation.median_wall_time_seconds,
        recommended_median_variants_per_second=recommended_evaluation.median_variants_per_second,
    )


def main() -> None:
    """Run one benchmark or a full chunk-size search."""
    arguments = build_argument_parser().parse_args()
    if arguments.single_run:
        print(json.dumps(asdict(run_single_full_chromosome_benchmark(arguments))))
        return

    search_report = run_search(arguments)
    search_report_json = json.dumps(asdict(search_report), indent=2)
    if arguments.output_path is not None:
        arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_path.write_text(search_report_json)
    print(search_report_json)


if __name__ == "__main__":
    main()
