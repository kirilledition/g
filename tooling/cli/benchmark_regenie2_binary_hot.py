#!/usr/bin/env python3
"""Benchmark the supported native binary REGENIE production lifecycle."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
import typing
from pathlib import Path

import hydra

import tooling.configuration as tooling_configuration
from tooling.benchmark import native_lifecycle
from tooling.common import g_regenie as tooling_g_regenie
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import paths as tooling_paths

if typing.TYPE_CHECKING:
    import omegaconf

REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/profiles")
SUMMARY_SCHEMA_VERSION = 2


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved production benchmark arguments."""

    data_directory: Path
    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    covariate_path: Path
    prediction_list_path: Path
    phenotype_column: str
    covariate_columns: tuple[str, ...]
    output_directory: Path
    device: tooling_g_regenie.RegenieDevice
    chunk_size: int
    firth_batch_size: int
    firth_candidate_capacity: int
    writer_thread_count: int
    p_threshold: float
    expected_variant_count: int | None
    jax_cache_directory: Path
    include_fresh_process: bool
    hot_run_count: int
    diagnostic_run_count: int
    python_executable: str
    summary_path: Path | None


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measurement and evidence for one complete lifecycle."""

    name: str
    role: str
    headline: bool
    telemetry: tooling_g_regenie.RegenieTelemetry
    native: native_lifecycle.NativeRunResult
    output: native_lifecycle.CompletedOutputEvidence
    diagnostics: native_lifecycle.DiagnosticEvidence
    cache_before: native_lifecycle.CacheSnapshot
    cache_after: native_lifecycle.CacheSnapshot
    cache_state: str


def resolve_data_path(data_directory: Path, value: typing.Any) -> Path:
    """Resolve a data path relative to the configured data directory."""
    return tooling_paths.resolve_data_path(data_directory, Path(str(value)))


def default_output_directory() -> Path:
    """Return a timestamped ignored benchmark output directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_PARENT / f"regenie2_binary_hot_{timestamp}_{os.getpid()}"


def build_run_spec(
    arguments: BenchmarkArguments,
    *,
    output_root: Path,
    telemetry: tooling_g_regenie.RegenieTelemetry,
) -> tooling_g_regenie.RegenieRunSpec:
    """Build one current production binary run specification."""
    return tooling_g_regenie.RegenieRunSpec(
        trait_kind=tooling_g_regenie.RegenieTraitKind.BINARY,
        command_prefix=("g", "regenie"),
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=arguments.bgen_path,
            sample_path=arguments.sample_path,
            phenotype_path=arguments.phenotype_path,
            phenotype_columns=(arguments.phenotype_column,),
            covariate_path=arguments.covariate_path,
            covariate_columns=arguments.covariate_columns,
            prediction_list_path=arguments.prediction_list_path,
            output_prefix=output_root,
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=arguments.device,
            bsize=arguments.chunk_size,
            firth_batch_size=arguments.firth_batch_size,
            firth_candidate_capacity=arguments.firth_candidate_capacity,
            jax_cache_dir=arguments.jax_cache_directory,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            output_run_directory=output_root,
            writer_threads=arguments.writer_thread_count,
            resume=False,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(telemetry=telemetry),
        binary=tooling_g_regenie.RegenieBinaryOptions(
            fallback_method=tooling_g_regenie.RegenieBinaryFallback.FIRTH_APPROXIMATE,
            p_threshold=arguments.p_threshold,
            firth_se=False,
        ),
    )


def write_native_config(
    arguments: BenchmarkArguments,
    *,
    output_root: Path,
    telemetry: tooling_g_regenie.RegenieTelemetry,
) -> Path:
    """Write one native CLI config with a distinct production output root."""
    config_path = Path(f"{output_root}.toml")
    return tooling_g_regenie.write_regenie_toml(
        build_run_spec(arguments, output_root=output_root, telemetry=telemetry),
        config_path,
    )


def collect_output_evidence(
    output_root: Path,
    stdout_chunks: typing.Sequence[str],
    expected_variant_count: int | None,
) -> native_lifecycle.CompletedOutputEvidence:
    """Validate direct Parquet parts and collect deterministic evidence."""
    verified_outputs = native_lifecycle.collect_completed_output_evidence(
        stdout_chunks,
        output_root=output_root,
        expected_phenotype_count=1,
        run_label="binary lifecycle",
    )
    output_evidence = verified_outputs.runs[0]
    if expected_variant_count is not None and output_evidence.row_count != expected_variant_count:
        message = f"Expected {expected_variant_count} output rows, observed {output_evidence.row_count}."
        raise RuntimeError(message)
    return output_evidence


def run_trial(
    arguments: BenchmarkArguments,
    *,
    name: str,
    role: str,
    headline: bool,
    telemetry: tooling_g_regenie.RegenieTelemetry,
    fresh_process: bool,
) -> TrialResult:
    """Run and validate one lifecycle."""
    output_root = arguments.output_directory / "runs" / name
    config_path = write_native_config(arguments, output_root=output_root, telemetry=telemetry)
    before = native_lifecycle.snapshot_tree(arguments.jax_cache_directory)
    native = (
        native_lifecycle.run_fresh_process(arguments.python_executable, config_path)
        if fresh_process
        else native_lifecycle.run_same_process(config_path)
    )
    if native.exit_code != 0:
        message = "".join((*native.stderr_chunks, *native.stdout_chunks))
        raise RuntimeError(f"Native CLI failed for {name}: {message}")
    after = native_lifecycle.snapshot_tree(arguments.jax_cache_directory)
    output = collect_output_evidence(output_root, native.stdout_chunks, arguments.expected_variant_count)
    return TrialResult(
        name=name,
        role=role,
        headline=headline,
        telemetry=telemetry,
        native=native,
        output=output,
        diagnostics=native_lifecycle.collect_diagnostic_evidence(
            telemetry=telemetry,
            telemetry_root=output_root,
            run_directories=(Path(output.run_directory),),
        ),
        cache_before=before,
        cache_after=after,
        cache_state=native_lifecycle.cache_state(before, after),
    )


def verify_hot_contract(trials: list[TrialResult]) -> None:
    """Require stable cache and output contracts for all headline trials."""
    headline_trials = [trial for trial in trials if trial.headline]
    if not headline_trials:
        raise RuntimeError("At least one headline hot trial is required.")
    reference = headline_trials[0].output
    for trial in headline_trials:
        if trial.cache_before != trial.cache_after or trial.cache_before.file_count == 0:
            message = f"JAX cache was not populated and unchanged during headline trial {trial.name}."
            raise RuntimeError(message)
        output = trial.output
        if (
            output.row_count != reference.row_count
            or output.schema != reference.schema
            or output.schema_metadata != reference.schema_metadata
            or output.parquet_metadata != reference.parquet_metadata
        ):
            message = f"Output contract differs for headline trial {trial.name}."
            raise RuntimeError(message)


def build_trial_plans(arguments: BenchmarkArguments) -> list[native_lifecycle.TrialPlan]:
    """Build lifecycle plans without mixing incompatible native telemetry policy."""
    plans = [
        native_lifecycle.TrialPlan(
            name="discarded_warm",
            role="discarded_compile_warmup",
            headline=False,
            telemetry=tooling_g_regenie.RegenieTelemetry.OFF,
            fresh_process=False,
        )
    ]
    plans.extend(
        native_lifecycle.TrialPlan(
            name=f"hot_{run_index + 1:02d}",
            role="same_process_hot_production",
            headline=True,
            telemetry=tooling_g_regenie.RegenieTelemetry.OFF,
            fresh_process=False,
        )
        for run_index in range(arguments.hot_run_count)
    )
    if arguments.include_fresh_process:
        plans.append(
            native_lifecycle.TrialPlan(
                name="fresh_process",
                role="fresh_process_diagnostic",
                headline=False,
                telemetry=tooling_g_regenie.RegenieTelemetry.OFF,
                fresh_process=True,
            )
        )
    plans.extend(
        native_lifecycle.TrialPlan(
            name=f"stage_timing_diagnostic_{run_index + 1:02d}",
            role="instrumented_diagnostic",
            headline=False,
            telemetry=tooling_g_regenie.RegenieTelemetry.PROFILE,
            fresh_process=True,
        )
        for run_index in range(arguments.diagnostic_run_count)
    )
    return plans


def run_benchmark(arguments: BenchmarkArguments) -> dict[str, typing.Any]:
    """Run fresh, warm, hot, and isolated diagnostic lifecycles."""
    if arguments.hot_run_count <= 0:
        raise ValueError("hot_run_count must be positive.")
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    arguments.jax_cache_directory.mkdir(parents=True, exist_ok=True)
    initial_cache = native_lifecycle.snapshot_tree(arguments.jax_cache_directory)
    if initial_cache.file_count != 0:
        raise RuntimeError(f"Lifecycle benchmark requires an empty campaign cache: {arguments.jax_cache_directory}")
    environment = native_lifecycle.collect_environment(
        repository_root=REPOSITORY_ROOT,
        input_paths={
            "bgen": arguments.bgen_path,
            "sample": arguments.sample_path,
            "phenotype": arguments.phenotype_path,
            "covariate": arguments.covariate_path,
            "prediction_list": arguments.prediction_list_path,
        },
        configuration=dataclasses.asdict(arguments),
        jax_cache_directory=arguments.jax_cache_directory,
    )
    trials = [
        run_trial(
            arguments,
            name=plan.name,
            role=plan.role,
            headline=plan.headline,
            telemetry=plan.telemetry,
            fresh_process=plan.fresh_process,
        )
        for plan in build_trial_plans(arguments)
    ]
    verify_hot_contract(trials)
    headline_seconds = [trial.native.elapsed_seconds for trial in trials if trial.headline]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "environment": environment,
        "headline": {
            "metric": "same_process_hot_production_elapsed_seconds",
            "telemetry": "off",
            "run_count": len(headline_seconds),
            "elapsed_seconds": headline_seconds,
        },
        "trials": [dataclasses.asdict(trial) for trial in trials],
    }


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Adapt Hydra configuration into the fixed benchmark contract."""
    values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    data_directory = tooling_paths.resolve_repo_relative_path(Path(str(values["data_dir"])), REPOSITORY_ROOT)
    output_directory = tooling_hydra_arguments.path_or_none(values.get("output_dir"))
    if output_directory is not None:
        output_directory = tooling_paths.resolve_repo_relative_path(output_directory, REPOSITORY_ROOT)
    resolved_output_directory = output_directory or default_output_directory()
    configured_cache = tooling_hydra_arguments.path_or_none(values.get("jax_cache_dir"))
    cache_directory = (
        resolved_output_directory / "jax-cache"
        if configured_cache is None
        else tooling_paths.resolve_repo_relative_path(configured_cache, REPOSITORY_ROOT)
    )
    python_executable = values.get("python_executable")
    configured_summary_path = tooling_hydra_arguments.path_or_none(values.get("summary_path"))
    return BenchmarkArguments(
        data_directory=data_directory,
        bgen_path=resolve_data_path(data_directory, values["bgen"]),
        sample_path=resolve_data_path(data_directory, values["sample"]),
        phenotype_path=resolve_data_path(data_directory, values["phenotype_file"]),
        covariate_path=resolve_data_path(data_directory, values["covariate_file"]),
        prediction_list_path=resolve_data_path(data_directory, values["prediction_list"]),
        phenotype_column=str(values["phenotype_column"]),
        covariate_columns=tuple(str(value) for value in values["covariate_columns"]),
        output_directory=resolved_output_directory,
        device=tooling_g_regenie.RegenieDevice(str(values["device"])),
        chunk_size=int(values["chunk_size"]),
        firth_batch_size=int(values["firth_batch_size"]),
        firth_candidate_capacity=int(values["firth_candidate_capacity"]),
        writer_thread_count=int(values["writer_thread_count"]),
        p_threshold=float(values["p_threshold"]),
        expected_variant_count=tooling_hydra_arguments.integer_or_none(values.get("expected_variant_count")),
        jax_cache_directory=cache_directory,
        include_fresh_process=bool(values["include_fresh_process"]),
        hot_run_count=int(values["hot_run_count"]),
        diagnostic_run_count=int(values["diagnostic_run_count"]),
        python_executable=sys.executable if python_executable is None else str(python_executable),
        summary_path=(
            None
            if configured_summary_path is None
            else tooling_paths.resolve_repo_relative_path(configured_summary_path, REPOSITORY_ROOT)
        ),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Compose benchmark configuration and return resolved arguments."""
    config = tooling_configuration.compose_config(config_name="benchmark_regenie2_binary_hot", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: BenchmarkArguments) -> Path:
    """Run the benchmark and write its evidence summary."""
    summary = run_benchmark(arguments)
    summary_path = arguments.summary_path or arguments.output_directory / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote benchmark evidence: {summary_path}")
    return summary_path


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_regenie2_binary_hot")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the benchmark through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the native binary hot benchmark."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
