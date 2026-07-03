"""Runner-local output adapter helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core, execution_plan, types

type OutputWriterSettings = output.OutputWriterSettings


@dataclass(frozen=True)
class PreparedPhenotypeRunPlan:
    """Prepared output state for one phenotype run.

    Attributes:
        phenotype_name: Phenotype column name.
        output_run_paths: Chunked output paths for the phenotype.
        existing_manifest: Existing manifest loaded for resume, if present.
        effective_config_path: Path where the effective TOML config is written.

    """

    phenotype_name: str
    output_run_paths: output.OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None
    effective_config_path: Path


def prepare_execution_plan_outputs(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> tuple[PreparedPhenotypeRunPlan, ...]:
    """Prepare output paths and resume state for a requested execution plan."""
    return tuple(
        prepare_phenotype_run_plan(
            phenotype_run_plan=phenotype_run_plan,
            association_mode=plan.association_mode,
            output_plan=plan.output_plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        for phenotype_run_plan in plan.phenotype_run_plans
    )


def prepare_phenotype_run_plan(
    *,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    association_mode: types.AssociationMode,
    output_plan: execution_plan.OutputPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> PreparedPhenotypeRunPlan:
    """Prepare output paths and resume manifest state for one phenotype."""
    prepared_output_run = output.prepare_output_run(
        output_root=output_plan.output_run_root / phenotype_run_plan.output_directory_name,
        association_mode=association_mode,
        output_format=output_plan.writer_settings.output_format,
        resume=output_plan.resume,
        resume_mode=output_plan.resume_mode,
        runtime_compatibility_token=runtime_compatibility_token,
    )
    return PreparedPhenotypeRunPlan(
        phenotype_name=phenotype_run_plan.phenotype_name,
        output_run_paths=prepared_output_run.output_run_paths,
        existing_manifest=prepared_output_run.existing_manifest,
        effective_config_path=prepared_output_run.output_run_paths.run_directory / "effective_config.toml",
    )


def output_writer_settings_from_plan(writer_plan: execution_plan.OutputWriterPlan) -> OutputWriterSettings:
    """Adapt requested output writer settings to the output adapter dataclass."""
    return output.OutputWriterSettings(
        finalize_parquet=writer_plan.finalize_parquet,
        writer_thread_count=writer_plan.writer_thread_count,
        writer_queue_depth=writer_plan.writer_queue_depth,
        chunks_per_arrow_file=writer_plan.chunks_per_arrow_file,
        arrow_compression=writer_plan.arrow_compression,
        parquet_compression=writer_plan.parquet_compression,
        output_format=writer_plan.output_format,
        output_statistic_dtype=writer_plan.output_statistic_dtype,
    )
