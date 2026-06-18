"""Run metadata and artifact finalization helpers."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

from g import _core, execution_plan, types
from g.engine import run_events, telemetry
from g.interface import config
from g.io import output

logger = logging.getLogger(__name__)
RunArtifacts = run_events.RunArtifacts


def run_artifacts_from_native_payload(
    artifact_payload: dict[str, object],
    phenotype_artifacts: tuple[RunArtifacts, ...],
) -> RunArtifacts:
    """Adapt a native artifact payload to the public Python dataclass."""
    association_mode_value = typing.cast("str | None", artifact_payload["association_mode"])
    return RunArtifacts(
        output_run_directory=optional_path_from_native_payload(artifact_payload["output_run_directory"]),
        final_dataset=optional_path_from_native_payload(artifact_payload["final_dataset"]),
        final_parquet=optional_path_from_native_payload(artifact_payload["final_parquet"]),
        final_regenie=optional_path_from_native_payload(artifact_payload["final_regenie"]),
        effective_config=optional_path_from_native_payload(artifact_payload["effective_config"]),
        phenotype_artifacts=phenotype_artifacts,
        phenotype_name=typing.cast("str | None", artifact_payload["phenotype_name"]),
        association_mode=None if association_mode_value is None else types.AssociationMode(association_mode_value),
        phenotype_count=typing.cast("int | None", artifact_payload["phenotype_count"]),
        run_id=typing.cast("str | None", artifact_payload["run_id"]),
    )


def optional_path_from_native_payload(path_payload: object) -> Path | None:
    """Adapt an optional native path string to a Python path."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


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
    return run_artifacts_from_native_payload(
        _core.build_multi_run_artifacts_payload(
            plan.association_mode.value,
            len(phenotype_artifacts),
        ),
        phenotype_artifacts,
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
    return run_artifacts_from_native_payload(
        _core.build_phenotype_run_artifacts_payload(
            str(phenotype_run_plan.output_run_paths.run_directory),
            str(phenotype_run_plan.output_run_paths.chunks_directory),
            str(phenotype_run_plan.effective_config_path),
            phenotype_run_plan.phenotype_name,
            plan.association_mode.value,
            len(plan.phenotype_run_plans),
            plan.output_plan.writer_settings.output_format.value,
            None if final_output_path is None else str(final_output_path),
        ),
        (),
    )


def extend_run_manifest(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
) -> None:
    """Add command and runtime metadata to a run manifest."""
    manifest = output.load_run_manifest(phenotype_run_plan.output_run_paths) or {}
    manifest_extension_payload = _core.build_run_manifest_extension_payload(
        phenotype_run_plan.phenotype_name,
        str(phenotype_run_plan.effective_config_path),
        plan.output_plan.writer_settings.output_format.value,
        plan.kernel_config.device.value,
        plan.kernel_config.staging_depth,
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
    manifest["command"] = native_mapping_payload(manifest_extension_payload["command"])
    manifest["runtime"] = native_mapping_payload(manifest_extension_payload["runtime"])
    output.write_run_manifest(phenotype_run_plan.output_run_paths, manifest)
