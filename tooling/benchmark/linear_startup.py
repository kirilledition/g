#!/usr/bin/env python3
"""Benchmark fresh, warm, and hot native quantitative REGENIE lifecycles."""

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
DEFAULT_OUTPUT_PARENT = Path("data/benchmarks")
SINGLE_PHENOTYPE_NAME = "phenotype_continuous"
SUMMARY_SCHEMA_VERSION = 0


@dataclasses.dataclass(frozen=True)
class BenchmarkInputs:
    """Input paths and phenotype columns used by the benchmark."""

    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    phenotype_names: tuple[str, ...]
    covariate_path: Path
    prediction_list_path: Path


@dataclasses.dataclass(frozen=True)
class LinearStartupArguments:
    """Resolved quantitative lifecycle benchmark parameters."""

    device: tooling_g_regenie.RegenieDevice
    chunk_size: int
    cpu_threads: int | None
    output_writer_thread_count: int
    include_fresh_process: bool
    hot_run_count: int
    diagnostic_run_count: int
    multi_phenotype_count: int
    multi_phenotype_sample_mode: tooling_g_regenie.RegenieMultiPhenotypeSampleMode
    expected_variant_count: int | None
    data_dir: Path
    output_dir: Path
    jax_cache_dir: Path
    python_executable: str
    json_summary_path: Path | None


@dataclasses.dataclass(frozen=True)
class OutputEvidence:
    """Direct-Parquet evidence for one lifecycle."""

    runs: tuple[native_lifecycle.CompletedOutputEvidence, ...]
    run_directories: tuple[str, ...]
    parquet_file_count: int
    parquet_total_bytes: int
    parquet_sha256: str
    row_count: int
    schema: str
    schema_metadata: dict[str, str]
    parquet_metadata: tuple[dict[str, str], ...]


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measurement and output evidence for one lifecycle."""

    name: str
    role: str
    headline: bool
    telemetry: tooling_g_regenie.RegenieTelemetry
    native: native_lifecycle.NativeRunResult
    output: OutputEvidence
    diagnostics: native_lifecycle.DiagnosticEvidence
    cache_before: native_lifecycle.CacheSnapshot
    cache_after: native_lifecycle.CacheSnapshot
    cache_state: str


def default_output_directory() -> Path:
    """Return a timestamped ignored benchmark directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_PARENT / f"regenie2_linear_lifecycle_{timestamp}_{os.getpid()}"


def write_cloned_phenotype_table(
    *, source_path: Path, destination_path: Path, phenotype_names: tuple[str, ...]
) -> None:
    """Write a phenotype table containing cloned quantitative traits."""
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    if not source_lines:
        raise ValueError(f"Phenotype file is empty: {source_path}")
    header = source_lines[0].split("\t")
    try:
        family_index = header.index("FID")
        individual_index = header.index("IID")
        phenotype_index = header.index(SINGLE_PHENOTYPE_NAME)
    except ValueError as error:
        message = f"Phenotype file must contain FID, IID, and {SINGLE_PHENOTYPE_NAME}: {source_path}"
        raise ValueError(message) from error
    destination_lines = ["\t".join(("FID", "IID", *phenotype_names))]
    for line_number, source_line in enumerate(source_lines[1:], start=2):
        if not source_line:
            continue
        values = source_line.split("\t")
        if len(values) <= max(family_index, individual_index, phenotype_index):
            raise ValueError(f"Phenotype file line {line_number} has fewer columns than its header.")
        destination_lines.append(
            "\t".join(
                (
                    values[family_index],
                    values[individual_index],
                    *(values[phenotype_index] for _ in phenotype_names),
                )
            )
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("\n".join(destination_lines) + "\n", encoding="utf-8")


def write_cloned_prediction_list(
    *, source_path: Path, destination_path: Path, phenotype_names: tuple[str, ...]
) -> None:
    """Write prediction-list entries for cloned quantitative traits."""
    matching_paths: list[Path] = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Prediction-list row {line_number} must contain a phenotype and LOCO path.")
        if fields[0] == SINGLE_PHENOTYPE_NAME:
            matching_paths.append(Path(fields[1]))
    if len(matching_paths) != 1:
        raise ValueError(f"Prediction list must contain exactly one {SINGLE_PHENOTYPE_NAME!r} row: {source_path}")
    raw_loco_path = matching_paths[0]
    loco_path = raw_loco_path if raw_loco_path.is_absolute() else (source_path.parent / raw_loco_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        "".join(f"{phenotype_name} {loco_path}\n" for phenotype_name in phenotype_names),
        encoding="utf-8",
    )


def prepare_benchmark_inputs(arguments: LinearStartupArguments) -> BenchmarkInputs:
    """Resolve source inputs and generate cloned traits when requested."""
    if arguments.multi_phenotype_count < 1:
        raise ValueError("multi_phenotype_count must be positive.")
    phenotype_path = arguments.data_dir / "pheno_cont.txt"
    prediction_list_path = arguments.data_dir / "baselines/regenie_step1_qt_pred.list"
    phenotype_names = tuple(
        SINGLE_PHENOTYPE_NAME if arguments.multi_phenotype_count == 1 else f"{SINGLE_PHENOTYPE_NAME}_{index + 1}"
        for index in range(arguments.multi_phenotype_count)
    )
    if arguments.multi_phenotype_count > 1:
        generated_directory = arguments.output_dir / "generated_inputs"
        phenotype_path = generated_directory / "phenotypes.tsv"
        prediction_list_path = generated_directory / "predictions.list"
        write_cloned_phenotype_table(
            source_path=arguments.data_dir / "pheno_cont.txt",
            destination_path=phenotype_path,
            phenotype_names=phenotype_names,
        )
        write_cloned_prediction_list(
            source_path=arguments.data_dir / "baselines/regenie_step1_qt_pred.list",
            destination_path=prediction_list_path,
            phenotype_names=phenotype_names,
        )
    return BenchmarkInputs(
        bgen_path=arguments.data_dir / "1kg_chr22_full.bgen",
        sample_path=arguments.data_dir / "1kg_chr22_full.sample",
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        covariate_path=arguments.data_dir / "covariates.txt",
        prediction_list_path=prediction_list_path,
    )


def build_run_spec(
    arguments: LinearStartupArguments,
    benchmark_inputs: BenchmarkInputs,
    *,
    output_root: Path,
    telemetry: tooling_g_regenie.RegenieTelemetry,
) -> tooling_g_regenie.RegenieRunSpec:
    """Build one current production run specification."""
    return tooling_g_regenie.RegenieRunSpec(
        trait_kind=tooling_g_regenie.RegenieTraitKind.QUANTITATIVE,
        command_prefix=("g", "regenie"),
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=benchmark_inputs.bgen_path,
            sample_path=benchmark_inputs.sample_path,
            phenotype_path=benchmark_inputs.phenotype_path,
            phenotype_columns=benchmark_inputs.phenotype_names,
            covariate_path=benchmark_inputs.covariate_path,
            covariate_columns=("age", "sex"),
            prediction_list_path=benchmark_inputs.prediction_list_path,
            output_prefix=output_root,
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=arguments.device,
            bsize=arguments.chunk_size,
            cpu_threads=arguments.cpu_threads,
            multi_phenotype_sample_mode=arguments.multi_phenotype_sample_mode,
            jax_cache_dir=arguments.jax_cache_dir,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            writer_threads=arguments.output_writer_thread_count,
            resume=False,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(telemetry=telemetry),
        binary=None,
    )


def collect_output_evidence(
    output_root: Path,
    *,
    expected_phenotype_count: int,
    expected_variant_count: int | None,
) -> OutputEvidence:
    """Validate direct Parquet parts and collect output evidence."""
    production_root = Path(f"{output_root}.g")
    run_directories = sorted(path for path in production_root.glob("*.run") if path.is_dir())
    if len(run_directories) != expected_phenotype_count:
        raise RuntimeError(
            f"Expected {expected_phenotype_count} phenotype runs below {production_root}, found {len(run_directories)}."
        )
    run_evidence: list[native_lifecycle.CompletedOutputEvidence] = []
    parquet_paths: list[Path] = []
    parquet_metadata: list[dict[str, str]] = []
    row_count = 0
    schema: str | None = None
    schema_metadata: dict[str, str] | None = None
    for run_directory in run_directories:
        completed_output = native_lifecycle.measure_completed_output_run(run_directory)
        if expected_variant_count is not None and completed_output.row_count != expected_variant_count:
            raise RuntimeError(
                f"Expected {expected_variant_count} rows for {run_directory.name}, "
                f"observed {completed_output.row_count}."
            )
        if schema is None:
            schema = completed_output.schema
            schema_metadata = completed_output.schema_metadata
        elif schema != completed_output.schema or schema_metadata != completed_output.schema_metadata:
            raise RuntimeError("Parquet schema changed within one benchmark lifecycle.")
        row_count += completed_output.row_count
        parquet_paths.extend(Path(path) for path in completed_output.parquet_paths)
        parquet_metadata.extend(completed_output.parquet_metadata)
        run_evidence.append(completed_output)
    if schema is None or schema_metadata is None:
        raise RuntimeError("Output schema was not observed.")
    return OutputEvidence(
        runs=tuple(run_evidence),
        run_directories=tuple(str(path) for path in run_directories),
        parquet_file_count=len(parquet_paths),
        parquet_total_bytes=sum(path.stat().st_size for path in parquet_paths),
        parquet_sha256=native_lifecycle.hash_paths(parquet_paths, production_root),
        row_count=row_count,
        schema=schema,
        schema_metadata=schema_metadata,
        parquet_metadata=tuple(parquet_metadata),
    )


def run_trial(
    arguments: LinearStartupArguments,
    benchmark_inputs: BenchmarkInputs,
    *,
    name: str,
    role: str,
    headline: bool,
    telemetry: tooling_g_regenie.RegenieTelemetry,
    fresh_process: bool,
) -> TrialResult:
    """Run and validate one complete lifecycle."""
    output_root = arguments.output_dir / "runs" / name
    run_spec = build_run_spec(arguments, benchmark_inputs, output_root=output_root, telemetry=telemetry)
    config_path = tooling_g_regenie.write_regenie_toml(run_spec, arguments.output_dir / "configs" / f"{name}.toml")
    before = native_lifecycle.snapshot_tree(arguments.jax_cache_dir)
    native = (
        native_lifecycle.run_fresh_process(arguments.python_executable, config_path)
        if fresh_process
        else native_lifecycle.run_same_process(config_path)
    )
    if native.exit_code != 0:
        details = "".join((*native.stderr_chunks, *native.stdout_chunks))
        raise RuntimeError(f"Native CLI failed for {name}: {details}")
    after = native_lifecycle.snapshot_tree(arguments.jax_cache_dir)
    output = collect_output_evidence(
        output_root,
        expected_phenotype_count=len(benchmark_inputs.phenotype_names),
        expected_variant_count=arguments.expected_variant_count,
    )
    return TrialResult(
        name=name,
        role=role,
        headline=headline,
        telemetry=telemetry,
        native=native,
        output=output,
        diagnostics=native_lifecycle.collect_diagnostic_evidence(
            telemetry=telemetry,
            telemetry_root=Path(f"{output_root}.g"),
            run_directories=tuple(Path(path) for path in output.run_directories),
        ),
        cache_before=before,
        cache_after=after,
        cache_state=native_lifecycle.cache_state(before, after),
    )


def verify_hot_outputs(trials: list[TrialResult]) -> None:
    """Require stable cache and output contracts across headline runs."""
    headline_trials = [trial for trial in trials if trial.headline]
    if not headline_trials:
        raise RuntimeError("At least one headline hot run is required.")
    reference = headline_trials[0].output
    for trial in headline_trials:
        if trial.cache_before != trial.cache_after or trial.cache_before.file_count == 0:
            raise RuntimeError(f"JAX cache was not populated and unchanged during headline trial {trial.name}.")
        if (
            trial.output.parquet_sha256 != reference.parquet_sha256
            or trial.output.row_count != reference.row_count
            or trial.output.schema != reference.schema
            or trial.output.schema_metadata != reference.schema_metadata
            or trial.output.parquet_metadata != reference.parquet_metadata
        ):
            raise RuntimeError(f"Output contract differs for headline trial {trial.name}.")


def build_trial_plans(arguments: LinearStartupArguments) -> list[native_lifecycle.TrialPlan]:
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
            name=f"profile_diagnostic_{run_index + 1:02d}",
            role="instrumented_diagnostic",
            headline=False,
            telemetry=tooling_g_regenie.RegenieTelemetry.PROFILE,
            fresh_process=True,
        )
        for run_index in range(arguments.diagnostic_run_count)
    )
    return plans


def run_benchmark(arguments: LinearStartupArguments) -> dict[str, typing.Any]:
    """Run one discarded warmup, hot headlines, and optional diagnostics."""
    if arguments.hot_run_count <= 0:
        raise ValueError("hot_run_count must be positive.")
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    arguments.jax_cache_dir.mkdir(parents=True, exist_ok=True)
    initial_cache = native_lifecycle.snapshot_tree(arguments.jax_cache_dir)
    if initial_cache.file_count != 0:
        raise RuntimeError(f"Lifecycle benchmark requires an empty campaign cache: {arguments.jax_cache_dir}")
    benchmark_inputs = prepare_benchmark_inputs(arguments)
    environment = native_lifecycle.collect_environment(
        repository_root=REPOSITORY_ROOT,
        input_paths={
            "bgen": benchmark_inputs.bgen_path,
            "sample": benchmark_inputs.sample_path,
            "phenotype": benchmark_inputs.phenotype_path,
            "covariate": benchmark_inputs.covariate_path,
            "prediction_list": benchmark_inputs.prediction_list_path,
        },
        configuration=dataclasses.asdict(arguments),
        jax_cache_directory=arguments.jax_cache_dir,
    )
    trials = [
        run_trial(
            arguments,
            benchmark_inputs,
            name=plan.name,
            role=plan.role,
            headline=plan.headline,
            telemetry=plan.telemetry,
            fresh_process=plan.fresh_process,
        )
        for plan in build_trial_plans(arguments)
    ]
    verify_hot_outputs(trials)
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
        "configuration": dataclasses.asdict(arguments),
        "trials": [dataclasses.asdict(trial) for trial in trials],
    }


def build_arguments_from_config(config: omegaconf.DictConfig) -> LinearStartupArguments:
    """Build benchmark arguments from Hydra configuration."""
    values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    data_directory = tooling_paths.resolve_repo_relative_path(Path(str(values["data_dir"])), REPOSITORY_ROOT)
    configured_output = tooling_hydra_arguments.path_or_none(values.get("output_dir"))
    output_directory = (
        default_output_directory()
        if configured_output is None
        else tooling_paths.resolve_repo_relative_path(configured_output, REPOSITORY_ROOT)
    )
    configured_cache = tooling_hydra_arguments.path_or_none(values.get("jax_cache_dir"))
    cache_directory = (
        output_directory / "jax-cache"
        if configured_cache is None
        else tooling_paths.resolve_repo_relative_path(configured_cache, REPOSITORY_ROOT)
    )
    configured_summary = tooling_hydra_arguments.path_or_none(values.get("json_summary_path"))
    python_executable = values.get("python_executable")
    return LinearStartupArguments(
        device=tooling_g_regenie.RegenieDevice(str(values["device"])),
        chunk_size=int(values["chunk_size"]),
        cpu_threads=tooling_hydra_arguments.integer_or_none(values.get("cpu_threads")),
        output_writer_thread_count=int(values["output_writer_thread_count"]),
        include_fresh_process=bool(values["include_fresh_process"]),
        hot_run_count=int(values["hot_run_count"]),
        diagnostic_run_count=int(values["diagnostic_run_count"]),
        multi_phenotype_count=int(values["multi_phenotype_count"]),
        multi_phenotype_sample_mode=tooling_g_regenie.RegenieMultiPhenotypeSampleMode(
            str(values["multi_phenotype_sample_mode"])
        ),
        expected_variant_count=tooling_hydra_arguments.integer_or_none(values.get("expected_variant_count")),
        data_dir=data_directory,
        output_dir=output_directory,
        jax_cache_dir=cache_directory,
        python_executable=sys.executable if python_executable is None else str(python_executable),
        json_summary_path=(
            None
            if configured_summary is None
            else tooling_paths.resolve_repo_relative_path(configured_summary, REPOSITORY_ROOT)
        ),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> LinearStartupArguments:
    """Compose the benchmark configuration and return resolved arguments."""
    config = tooling_configuration.compose_config(config_name="benchmark_linear_startup", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: LinearStartupArguments) -> None:
    """Run the benchmark and write its evidence summary."""
    report = run_benchmark(arguments)
    summary_path = arguments.json_summary_path or arguments.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote benchmark evidence: {summary_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_linear_startup")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the benchmark through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the quantitative lifecycle benchmark."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
