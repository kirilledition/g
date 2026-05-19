"""Public Python API for GWAS execution."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import time
import typing
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

from g import types
from g.interface import config as interface_config
from g.io import output, source

if typing.TYPE_CHECKING:
    from g.compute import regenie2_binary_types

InputConfig = interface_config.InputConfig
TraitConfig = interface_config.TraitConfig
BinaryConfig = interface_config.BinaryConfig
GComputeConfig = interface_config.GComputeConfig
GOutputConfig = interface_config.GOutputConfig
GDiagnosticsConfig = interface_config.GDiagnosticsConfig
RegenieConfig = interface_config.RegenieConfig


@dataclass(frozen=True)
class EngineRunConfig:
    """Internal execution settings derived from the public config."""

    chunk_size: int
    device: types.Device
    staging_depth: int
    output_run_directory: Path
    resume: bool
    resume_mode: types.ResumeMode
    finalize_parquet: bool
    writer_threads: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    alignment_config: GComputeConfig
    binary_kernel_config: regenie2_binary_types.BinaryKernelConfig | None = None
    variant_limit: int | None = None


@dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files."""

    output_run_directory: Path | None = None
    final_parquet: Path | None = None
    final_regenie: Path | None = None
    effective_config: Path | None = None
    phenotype_artifacts: tuple[RunArtifacts, ...] = ()


def load_engine_module() -> typing.Any:
    """Load the JAX-heavy engine module lazily."""
    return importlib.import_module("g.engine")


def load_jax_setup_module() -> typing.Any:
    """Load JAX setup lazily after runtime environment is configured."""
    return importlib.import_module("g.jax_setup")


def configure_jax_runtime(compute_config: GComputeConfig) -> None:
    """Configure JAX lazily."""
    load_jax_setup_module().configure_jax_runtime(
        cache_directory=compute_config.jax_cache_dir,
        matmul_precision=compute_config.jax_matmul_precision,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )


def configure_jax_device(device: types.Device) -> None:
    """Configure JAX lazily."""
    load_jax_setup_module().configure_jax_device(device)


def build_stage_timing_recorder(stage_timing_path: Path | None) -> typing.Any:
    """Build a stage timing recorder lazily."""
    return load_engine_module().build_stage_timing_recorder(stage_timing_path)


def record_stage_duration(stage_timing_recorder: typing.Any, stage_name: str, start_time: float) -> None:
    """Record one stage duration lazily."""
    load_engine_module().record_stage_duration(stage_timing_recorder, stage_name, start_time)


def write_stage_timing_snapshot(stage_timing_recorder: typing.Any, stage_timing_path: Path | None) -> None:
    """Write a stage timing snapshot lazily."""
    load_engine_module().write_stage_timing_snapshot(stage_timing_recorder, stage_timing_path)


def run_regenie2_linear_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the linear native pipeline lazily."""
    return typing.cast("Path | None", load_engine_module().run_regenie2_linear_bgen_pipeline(**kwargs))


def run_regenie2_binary_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the binary native pipeline lazily."""
    return typing.cast("Path | None", load_engine_module().run_regenie2_binary_bgen_pipeline(**kwargs))


def normalize_binary_correction_config(binary_config: BinaryConfig) -> types.BinaryCorrectionPlan:
    """Normalize REGENIE-style binary correction flags into an internal plan."""
    if not (0.0 < binary_config.p_threshold < 1.0):
        message = "pThresh must be in (0, 1)."
        raise ValueError(message)
    if binary_config.spa:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    if binary_config.approx and not binary_config.firth:
        message = "--approx requires --firth."
        raise ValueError(message)
    if binary_config.firth and binary_config.approx:
        return types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=binary_config.p_threshold,
            firth_se=binary_config.firth_se,
        )
    if binary_config.firth:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.SCORE_ONLY,
        p_threshold=binary_config.p_threshold,
        firth_se=False,
    )


def build_binary_kernel_config(compute_config: GComputeConfig) -> regenie2_binary_types.BinaryKernelConfig:
    """Build immutable binary JAX kernel settings from public compute config."""
    binary_types_module = importlib.import_module("g.compute.regenie2_binary_types")
    return binary_types_module.BinaryKernelConfig(
        maximum_null_iterations=compute_config.binary_null_maximum_iterations,
        null_logistic_coefficient_tolerance=compute_config.binary_null_coefficient_tolerance,
        firth_batch_size=compute_config.firth_batch_size,
        firth_candidate_capacity=compute_config.firth_candidate_capacity,
        firth_maximum_iterations=compute_config.firth_maximum_iterations,
        firth_gradient_tolerance=compute_config.firth_gradient_tolerance,
        firth_coefficient_tolerance=compute_config.firth_coefficient_tolerance,
        firth_likelihood_tolerance=compute_config.firth_likelihood_tolerance,
        firth_maximum_step_size=compute_config.firth_maximum_step_size,
        use_block_firth_math=compute_config.use_block_firth_math,
    )


def configure_runtime(compute_config: GComputeConfig, trait_config: TraitConfig) -> None:
    """Apply runtime knobs before engine execution."""
    configure_jax_runtime(compute_config)
    core_module = importlib.import_module("g._core")
    core_module.configure_bgen_decode_tile_variant_count(compute_config.bgen_decode_tile_variant_count)
    if trait_config.threads is not None:
        with contextlib.suppress(RuntimeError):
            core_module.configure_rayon_global_thread_pool(trait_config.threads)


def run_regenie_config(regenie_config: RegenieConfig) -> RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    interface_config.validate_config(regenie_config)
    configure_runtime(regenie_config.g_compute, regenie_config.trait)
    phenotype_artifacts: list[RunArtifacts] = []
    for phenotype_name in regenie_config.input.pheno_columns:
        phenotype_artifacts.append(run_one_phenotype_config(regenie_config, phenotype_name))
    if len(phenotype_artifacts) == 1:
        return phenotype_artifacts[0]
    return RunArtifacts(phenotype_artifacts=tuple(phenotype_artifacts))


def run_one_phenotype_config(regenie_config: RegenieConfig, phenotype_name: str) -> RunArtifacts:
    """Run one phenotype through the existing engine."""
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_root = regenie_config.g_output.output_run_directory or output_prefix.with_name(f"{output_prefix.name}.g")
    binary_kernel_config = (
        build_binary_kernel_config(regenie_config.g_compute)
        if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
        else None
    )
    engine_config = EngineRunConfig(
        chunk_size=regenie_config.trait.bsize,
        device=regenie_config.g_compute.device,
        staging_depth=regenie_config.g_compute.staging_depth,
        variant_limit=regenie_config.g_compute.variant_limit,
        output_run_directory=output_run_root / phenotype_name,
        resume=regenie_config.g_output.resume,
        resume_mode=regenie_config.g_output.resume_mode,
        finalize_parquet=regenie_config.g_output.format in {types.OutputFormat.PARQUET, types.OutputFormat.BOTH}
        or regenie_config.g_output.finalize_parquet,
        writer_threads=regenie_config.g_output.writer_threads,
        writer_queue_depth=regenie_config.g_output.writer_queue_depth,
        chunks_per_arrow_file=regenie_config.g_output.chunks_per_arrow_file,
        arrow_compression=regenie_config.g_output.arrow_compression,
        trusted_no_missing_diploid=regenie_config.g_compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=regenie_config.g_compute.trusted_bgen_validation_mode,
        alignment_config=regenie_config.g_compute,
        binary_kernel_config=binary_kernel_config,
    )
    artifacts = run_existing_engine(
        regenie_config=regenie_config,
        phenotype_name=phenotype_name,
        output_prefix=output_prefix,
        engine_config=engine_config,
    )
    final_regenie_path = None
    effective_config_path = None
    if artifacts.output_run_directory is not None:
        effective_config_path = artifacts.output_run_directory / "effective_config.toml"
        interface_config.write_toml(regenie_config, effective_config_path)
        extend_run_manifest(
            artifacts.output_run_directory,
            regenie_config,
            phenotype_name,
            effective_config_path,
        )
        if regenie_config.g_output.format in {types.OutputFormat.REGENIE, types.OutputFormat.BOTH}:
            final_regenie_path = output_prefix.with_name(f"{output_prefix.name}_{phenotype_name}.regenie")
            output_run_paths = output.OutputRunPaths(
                run_directory=artifacts.output_run_directory,
                chunks_directory=artifacts.output_run_directory / "chunks",
            )
            output.finalize_chunks_to_regenie_text(output_run_paths, final_regenie_path)
    return dataclasses.replace(
        artifacts,
        final_regenie=final_regenie_path,
        effective_config=effective_config_path,
    )


def run_existing_engine(
    *,
    regenie_config: RegenieConfig,
    phenotype_name: str,
    output_prefix: Path,
    engine_config: EngineRunConfig,
) -> RunArtifacts:
    """Run the existing native engine using normalized configuration."""
    api_entry_start_time = time.perf_counter()
    stage_timing_recorder = None
    try:
        device_start_time = time.perf_counter()
        configure_jax_device(engine_config.device)
        stage_timing_recorder = build_stage_timing_recorder(regenie_config.g_diagnostics.stage_timings_json)
        record_stage_duration(stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time)
        genotype_source_config = source.build_bgen_source_config(
            typing.cast("Path", regenie_config.input.bgen),
            regenie_config.input.sample,
        )
        binary_correction_plan = (
            normalize_binary_correction_config(regenie_config.binary)
            if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
            else types.BinaryCorrectionPlan()
        )
        association_mode = (
            types.AssociationMode.REGENIE2_BINARY
            if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
            else types.AssociationMode.REGENIE2_LINEAR
        )
        output_start_time = time.perf_counter()
        prepared_output_run = output.prepare_output_run(
            output_root=engine_config.output_run_directory,
            association_mode=association_mode,
            resume=engine_config.resume,
            resume_mode=engine_config.resume_mode,
        )
        record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        final_parquet_path = dispatch_engine_pipeline(
            regenie_config=regenie_config,
            phenotype_name=phenotype_name,
            genotype_source_config=genotype_source_config,
            engine_config=engine_config,
            output_run_paths=prepared_output_run.output_run_paths,
            committed_chunk_identifiers=set(prepared_output_run.committed_chunk_identifiers),
            binary_correction_plan=binary_correction_plan,
            stage_timing_recorder=stage_timing_recorder,
        )
        return RunArtifacts(
            output_run_directory=prepared_output_run.output_run_paths.run_directory,
            final_parquet=final_parquet_path,
        )
    finally:
        del output_prefix
        if stage_timing_recorder is not None:
            record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            write_stage_timing_snapshot(stage_timing_recorder, regenie_config.g_diagnostics.stage_timings_json)


def dispatch_engine_pipeline(
    *,
    regenie_config: RegenieConfig,
    phenotype_name: str,
    genotype_source_config: source.GenotypeSourceConfig,
    engine_config: EngineRunConfig,
    output_run_paths: output.OutputRunPaths,
    committed_chunk_identifiers: set[int],
    binary_correction_plan: types.BinaryCorrectionPlan,
    stage_timing_recorder: typing.Any,
) -> Path | None:
    """Dispatch one phenotype to the existing linear or binary pipeline."""
    common_arguments = {
        "genotype_source_config": genotype_source_config,
        "phenotype_path": typing.cast("Path", regenie_config.input.pheno_file),
        "phenotype_name": phenotype_name,
        "prediction_list_path": typing.cast("Path", regenie_config.input.pred),
        "covariate_path": regenie_config.input.covar_file,
        "covariate_names": regenie_config.input.covar_columns or None,
        "chunk_size": engine_config.chunk_size,
        "variant_limit": engine_config.variant_limit,
        "staging_depth": engine_config.staging_depth,
        "output_run_paths": output_run_paths,
        "committed_chunk_identifiers": committed_chunk_identifiers,
        "finalize_parquet": engine_config.finalize_parquet,
        "writer_thread_count": engine_config.writer_threads,
        "writer_queue_depth": engine_config.writer_queue_depth,
        "chunks_per_arrow_file": engine_config.chunks_per_arrow_file,
        "arrow_compression": engine_config.arrow_compression,
        "trusted_no_missing_diploid": engine_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": engine_config.trusted_bgen_validation_mode,
        "stage_timing_recorder": stage_timing_recorder,
        "alignment_config": engine_config.alignment_config,
    }
    if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY:
        return run_regenie2_binary_bgen_pipeline(
            **common_arguments,
            correction_plan=binary_correction_plan,
            kernel_config=engine_config.binary_kernel_config,
        )
    return run_regenie2_linear_bgen_pipeline(**common_arguments)


def extend_run_manifest(
    output_run_directory: Path,
    regenie_config: RegenieConfig,
    phenotype_name: str,
    effective_config_path: Path,
) -> None:
    """Add command metadata and input fingerprints to a run manifest."""
    output_run_paths = output.OutputRunPaths(
        run_directory=output_run_directory,
        chunks_directory=output_run_directory / "chunks",
    )
    manifest = output.load_run_manifest(output_run_paths) or {}
    manifest["command"] = {
        "interface": "g regenie",
        "phenotype": phenotype_name,
        "effective_config": str(effective_config_path),
        "output_format": regenie_config.g_output.format.value,
    }
    manifest["runtime"] = {
        "device": regenie_config.g_compute.device.value,
        "staging_depth": regenie_config.g_compute.staging_depth,
        "threads": regenie_config.trait.threads,
        "writer_threads": regenie_config.g_output.writer_threads,
        "writer_queue_depth": regenie_config.g_output.writer_queue_depth,
        "chunks_per_arrow_file": regenie_config.g_output.chunks_per_arrow_file,
        "arrow_compression": regenie_config.g_output.arrow_compression.value,
        "bgen_decode_tile_variant_count": regenie_config.g_compute.bgen_decode_tile_variant_count,
        "trusted_no_missing_diploid": regenie_config.g_compute.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": regenie_config.g_compute.trusted_bgen_validation_mode.value,
    }
    manifest["input_fingerprints"] = build_input_fingerprints(regenie_config)
    output.write_run_manifest(output_run_paths, manifest)


def build_input_fingerprints(regenie_config: RegenieConfig) -> dict[str, typing.Any]:
    """Build lightweight file fingerprints for user inputs."""
    paths = {
        "bgen": regenie_config.input.bgen,
        "sample": regenie_config.input.sample,
        "phenoFile": regenie_config.input.pheno_file,
        "covarFile": regenie_config.input.covar_file,
        "pred": regenie_config.input.pred,
    }
    fingerprints: dict[str, typing.Any] = {}
    for path_name, path_value in paths.items():
        if path_value is None:
            continue
        try:
            path_stat = path_value.stat()
        except FileNotFoundError:
            fingerprints[path_name] = {"path": str(path_value), "missing": True}
            continue
        fingerprints[path_name] = {
            "path": str(path_value.resolve()),
            "size": path_stat.st_size,
            "mtime_ns": path_stat.st_mtime_ns,
        }
    return fingerprints


class RegenieApi:
    """Callable public REGENIE-compatible API."""

    def __call__(self, regenie_config: RegenieConfig) -> RunArtifacts:
        """Run from a normalized config."""
        return run_regenie_config(regenie_config)

    def from_options(self, raw_options: typing.Mapping[str, typing.Any]) -> RunArtifacts:
        """Build a config from Python options and run it."""
        return run_regenie_config(RegenieConfig.from_options(raw_options))


regenie = RegenieApi()
