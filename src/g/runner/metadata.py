"""Run metadata and artifact finalization helpers."""

from __future__ import annotations

import typing

from g import _core, execution_plan, types
from g.interface import config
from g.runner import events

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.runner import outputs

RunArtifacts = events.RunArtifacts


def build_output_initialized_metadata_callback(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plans: tuple[outputs.PreparedPhenotypeRunPlan, ...],
    telemetry_session: events.TelemetrySession | None,
) -> typing.Callable[[tuple[str, ...]], None]:
    """Build an idempotent writer for metadata after output compatibility passes."""
    phenotype_run_plans_by_name = {
        phenotype_run_plan.phenotype_name: phenotype_run_plan for phenotype_run_plan in phenotype_run_plans
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
    telemetry_session: events.TelemetrySession | None,
    association_mode: types.AssociationMode,
    phenotype: str,
    final_output_path: Path | None,
) -> None:
    """Record output writer completion."""
    events.native_run_event_telemetry_policy().record_writer_finished_telemetry_event(
        telemetry_session,
        association_mode.value,
        phenotype,
        None if final_output_path is None else str(final_output_path),
    )


def write_run_start_metadata(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: outputs.PreparedPhenotypeRunPlan,
    telemetry_session: events.TelemetrySession | None,
) -> None:
    """Write run metadata before native engine execution starts."""
    config.write_toml(regenie_config, phenotype_run_plan.effective_config_path)
    extend_run_manifest(
        plan=plan,
        phenotype_run_plan=phenotype_run_plan,
    )
    events.native_run_event_telemetry_policy().record_effective_config_written_telemetry_event(
        telemetry_session,
        plan.association_mode.value,
        phenotype_run_plan.phenotype_name,
        str(phenotype_run_plan.effective_config_path),
        str(phenotype_run_plan.output_run_paths.run_directory),
    )


def finalize_execution_plan(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plans: tuple[outputs.PreparedPhenotypeRunPlan, ...],
    final_output_paths: tuple[Path | None, ...],
) -> RunArtifacts:
    """Build user-facing artifacts after native execution."""
    del regenie_config
    native_metadata_builder = _core.NativeRunMetadataBuilder()
    artifacts = events.run_artifacts_from_native_payload(
        native_metadata_builder.build_execution_run_artifacts_payload(
            plan.association_mode.value,
            len(phenotype_run_plans),
            plan.output_plan.writer_settings.output_format.value,
            tuple(str(phenotype_run_plan.output_run_paths.run_directory) for phenotype_run_plan in phenotype_run_plans),
            tuple(
                str(phenotype_run_plan.output_run_paths.chunks_directory) for phenotype_run_plan in phenotype_run_plans
            ),
            tuple(str(phenotype_run_plan.effective_config_path) for phenotype_run_plan in phenotype_run_plans),
            tuple(phenotype_run_plan.phenotype_name for phenotype_run_plan in phenotype_run_plans),
            tuple(
                None if final_output_path is None else str(final_output_path)
                for final_output_path in final_output_paths
            ),
        )
    )
    events.native_runner_diagnostic_policy().record_runner_metadata_artifacts_finalized_diagnostic_event(
        association_mode=plan.association_mode.value,
        phenotype_count=len(phenotype_run_plans),
    )
    return artifacts


def extend_run_manifest(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: outputs.PreparedPhenotypeRunPlan,
) -> None:
    """Add command and runtime metadata to a run manifest."""
    native_metadata_builder = _core.NativeRunMetadataBuilder()
    native_metadata_builder.extend_run_manifest_metadata(
        str(phenotype_run_plan.output_run_paths.run_directory),
        phenotype_run_plan.phenotype_name,
        str(phenotype_run_plan.effective_config_path),
        plan.output_plan.writer_settings.output_format.value,
        plan.kernel_config.device.value,
        plan.kernel_config.staging_depth,
        plan.kernel_config.native_callback_batch_size,
        plan.kernel_config.thread_count,
        plan.output_plan.writer_settings.writer_thread_count,
        plan.output_plan.writer_settings.writer_queue_depth,
        plan.output_plan.writer_settings.chunks_per_arrow_file,
        plan.output_plan.writer_settings.arrow_compression.value,
        plan.output_plan.writer_settings.parquet_compression.value,
        plan.output_plan.writer_settings.output_statistic_dtype.value,
        plan.kernel_config.bgen_decode_tile_variant_count,
        plan.kernel_config.trusted_no_missing_diploid,
        plan.kernel_config.trusted_bgen_validation_mode.value,
    )
