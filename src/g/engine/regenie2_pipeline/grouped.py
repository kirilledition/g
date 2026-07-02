"""Grouped per-phenotype REGENIE step 2 pipeline execution."""

from __future__ import annotations

import time
import typing

import numpy as np
import numpy.typing as npt

import g.engine.callbacks.grouped as callback_grouped
import g.engine.callbacks.shared as callback_shared
from g import _core, types
from g.engine import timing
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import context as pipeline_context
from g.engine.regenie2_pipeline import multi_group, outputs, telemetry_events
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path


def run_regenie2_grouped_per_phenotype_bgen_pipeline(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_names: tuple[str, ...],
    covariate_names: tuple[str, ...] | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Group independently aligned phenotypes and run one BGEN pass per compatible group."""
    _core.record_pipeline_grouped_per_phenotype_started_diagnostic_event(
        association_mode=context.association_mode.value,
        phenotype_count=len(phenotype_names),
        sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE.value,
    )
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    engine = outputs.open_pipeline_bgen_engine(
        context=context,
        pipeline_label="grouped per-phenotype",
        phenotype_name=None,
        phenotype_count=len(phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    grouped_run_inputs = native_dispatch_loaders.load_native_bgen_grouped_run_inputs(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=context.prediction_list_path,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
        planned_compute_groups=context.phenotype_compute_groups,
    )
    timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    _core.record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
        phenotype_count=len(phenotype_names),
        phenotype_group_count=len(grouped_run_inputs),
    )
    telemetry_events.log_sample_alignment_completed(
        context=context,
        sample_count=None,
        covariate_count=None,
        phenotype_name=None,
        phenotype_count=len(phenotype_names),
        phenotype_group_count=len(grouped_run_inputs),
    )
    grouped_sample_counts = tuple(
        int(grouped_run_input.run_input.sample_indices.shape[0])
        for grouped_run_input in grouped_run_inputs
        for _ in grouped_run_input.compute_group.phenotype_names
    )
    grouped_sample_set_fingerprints = tuple(
        grouped_run_input.compute_group.sample_set_fingerprint
        for grouped_run_input in grouped_run_inputs
        for _ in grouped_run_input.compute_group.phenotype_names
    )
    telemetry_events.log_multi_phenotype_sample_summary(
        context=context,
        sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        sample_counts=grouped_sample_counts,
        sample_set_fingerprints=grouped_sample_set_fingerprints,
        phenotype_group_count=len(grouped_run_inputs),
    )
    validate_grouped_per_phenotype_resume_compatibility(
        context=context,
        engine=engine,
        grouped_run_inputs=grouped_run_inputs,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        existing_manifests=existing_manifests,
        resume=resume,
        resume_mode=resume_mode,
    )

    if should_use_union_grouped_bgen_delivery(context=context, grouped_run_inputs=grouped_run_inputs):
        return run_prepared_grouped_per_phenotype_union_bgen_pipeline(
            context=context,
            engine=engine,
            grouped_run_inputs=grouped_run_inputs,
            phenotype_names=phenotype_names,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests=existing_manifests,
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        )

    final_parquet_paths_by_index: list[Path | None] = [None] * len(phenotype_names)
    for grouped_run_input in grouped_run_inputs:
        compute_group = grouped_run_input.compute_group
        group_multi_run_input = grouped_run_input.run_input
        group_final_parquet_paths = multi_group.run_prepared_multi_phenotype_bgen_group(
            context=context,
            engine=engine,
            run_input=group_multi_run_input,
            prediction_source=grouped_run_input.prediction_source,
            compute_group=compute_group,
            output_run_paths_by_phenotype=typing.cast(
                "tuple[output.OutputRunPaths, ...]",
                pipeline_context.select_by_phenotype_indices(
                    output_run_paths_by_phenotype,
                    compute_group.phenotype_indices,
                ),
            ),
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests=typing.cast(
                "tuple[dict[str, typing.Any] | None, ...]",
                pipeline_context.select_by_phenotype_indices(existing_manifests, compute_group.phenotype_indices),
            ),
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            output_sample_mode=output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        )
        for phenotype_index, final_parquet_path in zip(
            compute_group.phenotype_indices,
            group_final_parquet_paths,
            strict=True,
        ):
            final_parquet_paths_by_index[phenotype_index] = final_parquet_path
    return tuple(final_parquet_paths_by_index)


def validate_grouped_per_phenotype_resume_compatibility(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    grouped_run_inputs: tuple[native_dispatch_models.NativeBgenGroupedRunInput, ...],
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
) -> None:
    """Validate all grouped per-phenotype manifests before initializing any group."""
    if not resume:
        return
    selected_output_run_paths: list[output.OutputRunPaths] = []
    selected_existing_manifests: list[dict[str, typing.Any] | None] = []
    selected_current_headers: list[output.RunManifestHeaderInput] = []
    for grouped_run_input in grouped_run_inputs:
        compute_group = grouped_run_input.compute_group
        run_input = grouped_run_input.run_input
        for phenotype_index, phenotype_name in zip(
            compute_group.phenotype_indices,
            compute_group.phenotype_names,
            strict=True,
        ):
            selected_output_run_paths.append(output_run_paths_by_phenotype[phenotype_index])
            selected_existing_manifests.append(existing_manifests[phenotype_index])
            selected_current_headers.append(
                outputs.build_pipeline_manifest_header(
                    context=context,
                    phenotype_name=phenotype_name,
                    covariate_names=tuple(run_input.native_multi_aligned_sample_data.covariate_names),
                    sample_count=int(run_input.sample_indices.shape[0]),
                    variant_count=int(engine.variant_count),
                    multi_phenotype_sample_mode=output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
                    phenotype_compute_group=compute_group,
                )
            )
    outputs.validate_pipeline_resume_compatibility(
        output_run_paths_by_trait=tuple(selected_output_run_paths),
        existing_manifests_by_trait=tuple(selected_existing_manifests),
        current_headers_by_trait=tuple(selected_current_headers),
        resume_mode=resume_mode,
    )


def build_union_sample_indices(
    grouped_run_inputs: tuple[native_dispatch_models.NativeBgenGroupedRunInput, ...],
) -> npt.NDArray[np.int64]:
    """Build an ordered union sample selection for compatible phenotype groups."""
    seen_sample_indices: set[int] = set()
    union_sample_indices: list[int] = []
    for grouped_run_input in grouped_run_inputs:
        for raw_sample_index in grouped_run_input.run_input.sample_indices:
            sample_index = int(raw_sample_index)
            if sample_index in seen_sample_indices:
                continue
            seen_sample_indices.add(sample_index)
            union_sample_indices.append(sample_index)
    return np.asarray(union_sample_indices, dtype=np.int64)


def build_group_sample_position_array(
    *,
    union_sample_indices: npt.NDArray[np.int64],
    group_sample_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.intp]:
    """Map one group's sample order to positions in the union decode buffer."""
    union_position_by_sample_index = {
        int(sample_index): sample_position for sample_position, sample_index in enumerate(union_sample_indices)
    }
    return np.asarray(
        [union_position_by_sample_index[int(sample_index)] for sample_index in group_sample_indices],
        dtype=np.intp,
    )


def should_use_union_grouped_bgen_delivery(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    grouped_run_inputs: tuple[native_dispatch_models.NativeBgenGroupedRunInput, ...],
) -> bool:
    """Return whether grouped per-phenotype delivery should use one union decode pass."""
    if len(grouped_run_inputs) <= 1:
        return False
    if context.uses_packed8_genotypes:
        return False
    if not context.effective_trusted_no_missing_diploid:
        return False
    union_sample_count = int(build_union_sample_indices(grouped_run_inputs).shape[0])
    grouped_sample_count = sum(
        int(grouped_run_input.run_input.sample_indices.shape[0]) for grouped_run_input in grouped_run_inputs
    )
    return union_sample_count < grouped_sample_count


def run_prepared_grouped_per_phenotype_union_bgen_pipeline(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    grouped_run_inputs: tuple[native_dispatch_models.NativeBgenGroupedRunInput, ...],
    phenotype_names: tuple[str, ...],
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Run overlapping per-phenotype groups through one union-sample BGEN delivery."""
    native_schedule_policy().resolve_grouped_union_callback_batch_size(
        native_callback_batch_size=native_callback_batch_size
    )
    union_sample_indices = build_union_sample_indices(grouped_run_inputs)
    grouped_sample_count = sum(
        int(grouped_run_input.run_input.sample_indices.shape[0]) for grouped_run_input in grouped_run_inputs
    )
    union_sample_count = int(union_sample_indices.shape[0])
    _core.record_pipeline_grouped_union_delivery_selected_diagnostic_event(
        grouped_sample_count=grouped_sample_count,
        phenotype_group_count=len(grouped_run_inputs),
        union_sample_count=union_sample_count,
    )
    prepared_deliveries = tuple(
        multi_group.prepare_multi_phenotype_bgen_group_delivery(
            context=context,
            engine=engine,
            run_input=grouped_run_input.run_input,
            prediction_source=grouped_run_input.prediction_source,
            compute_group=grouped_run_input.compute_group,
            output_run_paths_by_phenotype=typing.cast(
                "tuple[output.OutputRunPaths, ...]",
                pipeline_context.select_by_phenotype_indices(
                    output_run_paths_by_phenotype,
                    grouped_run_input.compute_group.phenotype_indices,
                ),
            ),
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests=typing.cast(
                "tuple[dict[str, typing.Any] | None, ...]",
                pipeline_context.select_by_phenotype_indices(
                    existing_manifests,
                    grouped_run_input.compute_group.phenotype_indices,
                ),
            ),
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            output_sample_mode=output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        )
        for grouped_run_input in grouped_run_inputs
    )
    group_fanouts = tuple(
        callback_shared.MultiPhenotypeGroupFanout(
            callback=prepared_delivery.callback,
            sample_position_array=build_group_sample_position_array(
                union_sample_indices=union_sample_indices,
                group_sample_indices=prepared_delivery.run_input.sample_indices,
            ),
        )
        for prepared_delivery in prepared_deliveries
    )
    union_run_input = native_dispatch_models.NativeBgenUnionRunInput(sample_indices=union_sample_indices)
    writer_sessions = tuple(
        writer_session
        for prepared_delivery in prepared_deliveries
        for writer_session in prepared_delivery.writer_sessions
    )
    committed_chunk_identifier_sets = tuple(
        committed_chunk_identifier_set
        for prepared_delivery in prepared_deliveries
        for committed_chunk_identifier_set in prepared_delivery.committed_chunk_identifier_sets
    )
    final_parquet_paths = native_dispatch_delivery.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=union_run_input,
        committed_chunk_identifiers=multi_group.intersect_committed_chunk_identifier_sets(
            committed_chunk_identifier_sets
        ),
        writer_sessions=writer_sessions,
        callback=callback_grouped.GroupedMultiPhenotypeFanoutCallback(group_fanouts),
        stage_timing_recorder=context.stage_timing_recorder,
        writer_finish_thread_count=1,
        variant_major_packed8_probability_pairs=False,
        pipeline_label="Grouped per-phenotype union native BGEN",
    )
    final_parquet_paths_by_index: list[Path | None] = [None] * len(phenotype_names)
    final_path_offset = 0
    for prepared_delivery in prepared_deliveries:
        group_final_paths = final_parquet_paths[
            final_path_offset : final_path_offset + len(prepared_delivery.phenotype_indices)
        ]
        final_path_offset += len(prepared_delivery.phenotype_indices)
        for phenotype_index, final_parquet_path in zip(
            prepared_delivery.phenotype_indices,
            group_final_paths,
            strict=True,
        ):
            final_parquet_paths_by_index[phenotype_index] = final_parquet_path
    return tuple(final_parquet_paths_by_index)


def native_schedule_policy() -> _core.NativeSchedulePolicy:
    """Build the native schedule policy handle."""
    return _core.NativeSchedulePolicy()
