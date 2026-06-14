"""Run metadata and artifact finalization helpers."""

from __future__ import annotations

import logging
import typing

from g import execution_plan, types
from g.engine import run_events, telemetry
from g.interface import config
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
RunArtifacts = run_events.RunArtifacts


def build_output_initialized_metadata_callback(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    telemetry_session: telemetry.TelemetrySession | None,
) -> typing.Callable[[tuple[str, ...]], None]:
    """Build an idempotent writer for metadata after output compatibility passes."""
    phenotype_run_plans_by_name = {
        phenotype_run_plan.phenotype_name: phenotype_run_plan for phenotype_run_plan in plan.phenotype_run_plans
    }
    written_phenotype_names: set[str] = set()

    def write_initialized_metadata(phenotype_names: tuple[str, ...]) -> None:
        for phenotype_name in phenotype_names:
            if phenotype_name in written_phenotype_names:
                continue
            phenotype_run_plan = phenotype_run_plans_by_name[phenotype_name]
            write_run_start_metadata(
                regenie_config=regenie_config,
                plan=plan,
                phenotype_run_plan=phenotype_run_plan,
                telemetry_session=telemetry_session,
            )
            written_phenotype_names.add(phenotype_name)

    return write_initialized_metadata


def log_writer_finished(
    *,
    telemetry_session: telemetry.TelemetrySession | None,
    association_mode: types.AssociationMode,
    phenotype: str,
    final_output_path: Path | None,
) -> None:
    """Record output writer completion."""
    if telemetry_session is None:
        return
    telemetry_session.log_event(
        "writer_finished",
        level="info",
        association_mode=association_mode.value,
        phenotype=phenotype,
        final_output_path=None if final_output_path is None else str(final_output_path),
    )


def write_run_start_metadata(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    telemetry_session: telemetry.TelemetrySession | None,
) -> None:
    """Write run metadata before native engine execution starts."""
    config.write_toml(regenie_config, phenotype_run_plan.effective_config_path)
    extend_run_manifest(
        plan=plan,
        phenotype_run_plan=phenotype_run_plan,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "effective_config_written",
            level="info",
            association_mode=plan.association_mode.value,
            phenotype=phenotype_run_plan.phenotype_name,
            effective_config=str(phenotype_run_plan.effective_config_path),
            output_run_directory=str(phenotype_run_plan.output_run_paths.run_directory),
        )


def finalize_execution_plan(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    final_output_paths: tuple[Path | None, ...],
) -> RunArtifacts:
    """Build user-facing artifacts after native execution."""
    phenotype_artifacts = tuple(
        finalize_phenotype_run(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plan=phenotype_run_plan,
            final_output_path=final_output_path,
        )
        for phenotype_run_plan, final_output_path in zip(
            plan.phenotype_run_plans,
            final_output_paths,
            strict=True,
        )
    )
    logger.info("Finalized REGENIE run artifacts for %s phenotype(s).", len(phenotype_artifacts))
    if len(phenotype_artifacts) == 1:
        return phenotype_artifacts[0]
    return RunArtifacts(
        output_run_directory=None,
        final_dataset=None,
        final_parquet=None,
        final_regenie=None,
        effective_config=None,
        phenotype_artifacts=phenotype_artifacts,
        phenotype_name=None,
        association_mode=plan.association_mode,
        phenotype_count=len(phenotype_artifacts),
        run_id=None,
    )


def finalize_phenotype_run(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    final_output_path: Path | None,
) -> RunArtifacts:
    """Build artifacts for one phenotype."""
    del regenie_config
    final_dataset = (
        phenotype_run_plan.output_run_paths.chunks_directory
        if plan.output_plan.writer_settings.output_format == types.OutputFormat.PARQUET
        else None
    )
    final_parquet_path = None
    final_regenie_path = None
    if plan.output_plan.writer_settings.output_format == types.OutputFormat.REGENIE:
        final_regenie_path = final_output_path
    else:
        final_parquet_path = final_output_path
    return RunArtifacts(
        output_run_directory=phenotype_run_plan.output_run_paths.run_directory,
        final_dataset=final_dataset,
        final_parquet=final_parquet_path,
        final_regenie=final_regenie_path,
        effective_config=phenotype_run_plan.effective_config_path,
        phenotype_artifacts=(),
        phenotype_name=phenotype_run_plan.phenotype_name,
        association_mode=plan.association_mode,
        phenotype_count=len(plan.phenotype_run_plans),
        run_id=None,
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
        "output_format": plan.output_plan.writer_settings.output_format.value,
    }
    manifest["runtime"] = {
        "device": plan.kernel_config.device.value,
        "staging_depth": plan.kernel_config.staging_depth,
        "threads": plan.kernel_config.thread_count,
        "writer_threads": plan.output_plan.writer_settings.writer_thread_count,
        "writer_queue_depth": plan.output_plan.writer_settings.writer_queue_depth,
        "chunks_per_arrow_file": plan.output_plan.writer_settings.chunks_per_arrow_file,
        "arrow_compression": plan.output_plan.writer_settings.arrow_compression.value,
        "parquet_compression": plan.output_plan.writer_settings.parquet_compression.value,
        "output_statistic_dtype": plan.output_plan.writer_settings.output_statistic_dtype.value,
        "bgen_decode_tile_variant_count": plan.kernel_config.bgen_decode_tile_variant_count,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode.value,
    }
    output.write_run_manifest(phenotype_run_plan.output_run_paths, manifest)
