"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import contextlib
import importlib
import time
import typing
from dataclasses import dataclass

from g import execution_plan, types
from g.interface import config
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files.

    Attributes:
        output_run_directory: Chunked output run directory.
        final_parquet: Finalized Parquet output path.
        effective_config: Written effective TOML config path.
        phenotype_artifacts: Per-phenotype artifacts for multi-phenotype runs.

    """

    output_run_directory: Path | None = None
    final_parquet: Path | None = None
    effective_config: Path | None = None
    phenotype_artifacts: tuple[RunArtifacts, ...] = ()


def load_regenie2_pipeline_module() -> typing.Any:
    """Load the JAX-heavy REGENIE pipeline module lazily."""
    return importlib.import_module("g.engine.regenie2_pipeline")


def load_timing_module() -> typing.Any:
    """Load stage-timing helpers lazily."""
    return importlib.import_module("g.engine.timing")


def load_jax_setup_module() -> typing.Any:
    """Load JAX setup lazily after runtime environment is configured."""
    return importlib.import_module("g.jax_setup")


def configure_jax_platform_before_setup_import(device: types.Device) -> None:
    """Configure JAX platform selection before importing setup helpers."""
    jax_module = importlib.import_module("jax")
    platform_name = "cuda" if device == types.Device.GPU else "cpu"
    jax_module.config.update("jax_platforms", platform_name)


def configure_runtime_before_jax_import(compute_config: config.GComputeConfig) -> None:
    """Configure JAX platform and runtime before compute modules are imported."""
    configure_jax_platform_before_setup_import(compute_config.device)
    load_jax_setup_module().configure_jax_runtime_before_backend_init(
        device=compute_config.device,
        cache_directory=compute_config.jax_cache_dir,
        matmul_precision=compute_config.jax_matmul_precision,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )


def configure_jax_runtime(compute_config: config.GComputeConfig) -> None:
    """Configure JAX lazily."""
    configure_runtime_before_jax_import(compute_config)


def configure_jax_device(device: types.Device) -> None:
    """Configure JAX lazily."""
    load_jax_setup_module().configure_jax_device(device)


def build_stage_timing_recorder(stage_timing_path: Path | None) -> typing.Any:
    """Build a stage timing recorder lazily."""
    return load_timing_module().build_stage_timing_recorder(stage_timing_path)


def record_stage_duration(stage_timing_recorder: typing.Any, stage_name: str, start_time: float) -> None:
    """Record one stage duration lazily."""
    load_timing_module().record_stage_duration(stage_timing_recorder, stage_name, start_time)


def write_stage_timing_snapshot(stage_timing_recorder: typing.Any, stage_timing_path: Path | None) -> None:
    """Write a stage timing snapshot lazily."""
    load_timing_module().write_stage_timing_snapshot(stage_timing_recorder, stage_timing_path)


def run_regenie2_linear_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the linear native pipeline lazily."""
    return typing.cast("Path | None", load_regenie2_pipeline_module().run_regenie2_linear_bgen_pipeline(**kwargs))


def run_regenie2_binary_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the binary native pipeline lazily."""
    return typing.cast("Path | None", load_regenie2_pipeline_module().run_regenie2_binary_bgen_pipeline(**kwargs))


def run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype linear native pipeline lazily."""
    return typing.cast(
        "tuple[Path | None, ...]",
        load_regenie2_pipeline_module().run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs),
    )


def run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype binary native pipeline lazily."""
    return typing.cast(
        "tuple[Path | None, ...]",
        load_regenie2_pipeline_module().run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs),
    )


def configure_runtime(compute_config: config.GComputeConfig, trait_config: config.TraitConfig) -> None:
    """Apply native runtime knobs before engine execution."""
    core_module = importlib.import_module("g._core")
    core_module.configure_bgen_decode_tile_variant_count(compute_config.bgen_decode_tile_variant_count)
    if trait_config.threads is not None:
        with contextlib.suppress(RuntimeError):
            core_module.configure_rayon_global_thread_pool(trait_config.threads)


def regenie(regenie_config: config.RegenieConfig) -> RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    config.validate_config(regenie_config)
    configure_runtime(regenie_config.g_compute, regenie_config.trait)
    return run_validated_regenie_config(regenie_config)


def run_validated_regenie_config(regenie_config: config.RegenieConfig) -> RunArtifacts:
    """Plan, execute, and finalize a validated REGENIE-compatible config."""
    api_entry_start_time = time.perf_counter()
    stage_timing_recorder = None
    try:
        device_start_time = time.perf_counter()
        configure_runtime_before_jax_import(regenie_config.g_compute)
        stage_timing_recorder = build_stage_timing_recorder(regenie_config.g_diagnostics.stage_timings_json)
        record_stage_duration(stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time)
        output_start_time = time.perf_counter()
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        write_execution_plan_start_metadata(regenie_config=regenie_config, plan=plan)
        record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        final_parquet_paths = dispatch_execution_plan(
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
        )
        return finalize_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            final_parquet_paths=final_parquet_paths,
        )
    finally:
        if stage_timing_recorder is not None:
            record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            write_stage_timing_snapshot(stage_timing_recorder, regenie_config.g_diagnostics.stage_timings_json)


def dispatch_execution_plan(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: typing.Any,
) -> tuple[Path | None, ...]:
    """Dispatch an execution plan to the native engine layer."""
    if len(plan.phenotype_run_plans) > 1:
        return dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
        )
    return (
        dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=plan.phenotype_run_plans[0],
            stage_timing_recorder=stage_timing_recorder,
        ),
    )


def build_common_engine_arguments(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: typing.Any,
) -> dict[str, typing.Any]:
    """Build arguments shared by single- and multi-phenotype native wrappers."""
    return {
        "genotype_source_config": plan.genotype_source_config,
        "phenotype_path": plan.phenotype_path,
        "prediction_list_path": plan.prediction_list_path,
        "covariate_path": plan.covariate_path,
        "covariate_names": plan.covariate_names,
        "chunk_size": plan.kernel_config.chunk_size,
        "variant_limit": plan.kernel_config.variant_limit,
        "staging_depth": plan.kernel_config.staging_depth,
        "resume": plan.output_plan.resume,
        "resume_mode": plan.output_plan.resume_mode,
        "finalize_parquet": plan.output_plan.finalize_parquet,
        "writer_thread_count": plan.output_plan.writer_threads,
        "writer_queue_depth": plan.output_plan.writer_queue_depth,
        "chunks_per_arrow_file": plan.output_plan.chunks_per_arrow_file,
        "arrow_compression": plan.output_plan.arrow_compression,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode,
        "stage_timing_recorder": stage_timing_recorder,
        "alignment_config": plan.kernel_config.alignment_config,
    }


def dispatch_one_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    stage_timing_recorder: typing.Any,
) -> Path | None:
    """Dispatch one phenotype to the native linear or binary pipeline."""
    common_arguments = build_common_engine_arguments(
        plan=plan,
        stage_timing_recorder=stage_timing_recorder,
    )
    common_arguments.update(
        {
            "phenotype_name": phenotype_run_plan.phenotype_name,
            "output_run_paths": phenotype_run_plan.output_run_paths,
            "existing_manifest": phenotype_run_plan.existing_manifest,
        }
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        return run_regenie2_binary_bgen_pipeline(
            **common_arguments,
            correction_plan=plan.binary_correction_plan,
            kernel_config=plan.kernel_config.binary_kernel_config,
        )
    return run_regenie2_linear_bgen_pipeline(**common_arguments)


def dispatch_multi_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: typing.Any,
) -> tuple[Path | None, ...]:
    """Dispatch multiple phenotypes to the shared native pipeline."""
    common_arguments = build_common_engine_arguments(
        plan=plan,
        stage_timing_recorder=stage_timing_recorder,
    )
    common_arguments.update(
        {
            "phenotype_names": tuple(
                phenotype_run_plan.phenotype_name for phenotype_run_plan in plan.phenotype_run_plans
            ),
            "output_run_paths_by_phenotype": tuple(
                phenotype_run_plan.output_run_paths for phenotype_run_plan in plan.phenotype_run_plans
            ),
            "existing_manifests_by_phenotype": tuple(
                phenotype_run_plan.existing_manifest for phenotype_run_plan in plan.phenotype_run_plans
            ),
        }
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        return run_regenie2_multi_phenotype_binary_bgen_pipeline(
            **common_arguments,
            correction_plan=plan.binary_correction_plan,
        )
    return run_regenie2_multi_phenotype_linear_bgen_pipeline(**common_arguments)


def write_execution_plan_start_metadata(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
) -> None:
    """Write per-phenotype metadata before native engine execution starts."""
    for phenotype_run_plan in plan.phenotype_run_plans:
        write_run_start_metadata(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plan=phenotype_run_plan,
        )


def write_run_start_metadata(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
) -> None:
    """Write run metadata before native engine execution starts."""
    config.write_toml(regenie_config, phenotype_run_plan.effective_config_path)
    extend_run_manifest(
        plan=plan,
        phenotype_run_plan=phenotype_run_plan,
    )


def finalize_execution_plan(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    final_parquet_paths: tuple[Path | None, ...],
) -> RunArtifacts:
    """Build user-facing artifacts after native execution."""
    phenotype_artifacts = tuple(
        finalize_phenotype_run(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plan=phenotype_run_plan,
            final_parquet_path=final_parquet_path,
        )
        for phenotype_run_plan, final_parquet_path in zip(
            plan.phenotype_run_plans,
            final_parquet_paths,
            strict=True,
        )
    )
    if len(phenotype_artifacts) == 1:
        return phenotype_artifacts[0]
    return RunArtifacts(phenotype_artifacts=phenotype_artifacts)


def finalize_phenotype_run(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    final_parquet_path: Path | None,
) -> RunArtifacts:
    """Build artifacts for one phenotype."""
    del regenie_config, plan
    return RunArtifacts(
        output_run_directory=phenotype_run_plan.output_run_paths.run_directory,
        final_parquet=final_parquet_path,
        effective_config=phenotype_run_plan.effective_config_path,
    )


def extend_run_manifest(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
) -> None:
    """Add command and runtime metadata to a run manifest."""
    manifest = output.load_run_manifest(phenotype_run_plan.output_run_paths) or {}
    manifest["command"] = {
        "interface": "g regenie",
        "phenotype": phenotype_run_plan.phenotype_name,
        "effective_config": str(phenotype_run_plan.effective_config_path),
        "output_format": plan.output_plan.output_format.value,
    }
    manifest["runtime"] = {
        "device": plan.kernel_config.device.value,
        "staging_depth": plan.kernel_config.staging_depth,
        "threads": plan.kernel_config.thread_count,
        "writer_threads": plan.output_plan.writer_threads,
        "writer_queue_depth": plan.output_plan.writer_queue_depth,
        "chunks_per_arrow_file": plan.output_plan.chunks_per_arrow_file,
        "arrow_compression": plan.output_plan.arrow_compression.value,
        "bgen_decode_tile_variant_count": plan.kernel_config.bgen_decode_tile_variant_count,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode.value,
    }
    output.write_run_manifest(phenotype_run_plan.output_run_paths, manifest)
