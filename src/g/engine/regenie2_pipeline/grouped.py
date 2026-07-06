"""Grouped per-phenotype REGENIE step 2 pipeline execution."""

from __future__ import annotations

import time
import typing

import g.engine.callbacks.grouped as callback_grouped
import g.engine.callbacks.shared as callback_shared
from g import _core, types
from g.engine import timing as engine_timing
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import context as pipeline_context
from g.engine.regenie2_pipeline import multi_group, outputs
from g.runner import events

if typing.TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


def run_regenie2_grouped_per_phenotype_bgen_pipeline(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_names: tuple[str, ...],
    covariate_names: tuple[str, ...] | None,
    prepared_runs_by_phenotype: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Group independently aligned phenotypes and run one BGEN pass per compatible group."""
    _core.record_pipeline_grouped_per_phenotype_started_diagnostic_event(
        association_mode=context.association_mode.value,
        phenotype_count=len(phenotype_names),
        sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE.value,
    )
    engine = outputs.open_pipeline_bgen_engine(
        context=context,
        pipeline_label="grouped per-phenotype",
        phenotype_name=None,
        phenotype_count=len(phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    sample_path = context.genotype_source_config.sample_path
    native_grouped_aligned_sample_data = engine.align_grouped_sample_data(
        str(sample_path) if sample_path is not None else None,
        str(context.phenotype_path),
        list(phenotype_names),
        str(context.covariate_path) if context.covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        context.is_binary_trait,
        sample_key_mode=native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
    )
    prediction_sources = _core.MultiRegeniePredictionSource.from_native_grouped_aligned_sample_data(
        str(context.prediction_list_path),
        native_grouped_aligned_sample_data,
        sample_key_mode=native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
    )
    if len(native_grouped_aligned_sample_data.groups) != len(prediction_sources):
        message = (
            "Grouped prediction source count does not match grouped aligned sample data count: "
            f"{len(prediction_sources)} prediction source(s), "
            f"{len(native_grouped_aligned_sample_data.groups)} aligned group(s)."
        )
        raise ValueError(message)
    planned_names_by_index: dict[int, str] = {}
    if context.phenotype_compute_groups is not None:
        for planned_compute_group in context.phenotype_compute_groups:
            for phenotype_index, phenotype_name in zip(
                planned_compute_group.phenotype_indices,
                planned_compute_group.phenotype_names,
                strict=True,
            ):
                planned_names_by_index[phenotype_index] = phenotype_name
    grouped_run_inputs_list: list[native_dispatch_models.NativeBgenGroupedRunInput] = []
    for native_group, prediction_source in zip(
        native_grouped_aligned_sample_data.groups,
        prediction_sources,
        strict=True,
    ):
        phenotype_indices = tuple(int(phenotype_index) for phenotype_index in native_group.phenotype_indices)
        run_input = native_dispatch_models.build_native_bgen_multi_run_input(native_group.aligned_sample_data)
        if planned_names_by_index:
            group_phenotype_names = tuple(
                planned_names_by_index[phenotype_index] for phenotype_index in phenotype_indices
            )
        else:
            group_phenotype_names = run_input.phenotype_names
        compute_group = native_dispatch_groups.adapt_native_phenotype_compute_group(
            _core.resolve_per_phenotype_compute_group(
                run_input.native_multi_aligned_sample_data,
                list(phenotype_indices),
                list(group_phenotype_names),
                str(context.prediction_list_path),
                native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
            )
        )
        grouped_run_inputs_list.append(
            native_dispatch_models.NativeBgenGroupedRunInput(
                compute_group=compute_group,
                phenotype_indices=compute_group.phenotype_indices,
                run_input=run_input,
                prediction_source=prediction_source,
            )
        )
    grouped_run_inputs = tuple(grouped_run_inputs_list)
    engine_timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    _core.record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
        phenotype_count=len(phenotype_names),
        phenotype_group_count=len(grouped_run_inputs),
    )
    events.record_sample_alignment_completed_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        None,
        len(phenotype_names),
        None,
        None,
        len(grouped_run_inputs),
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
    _core.record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
        phenotype_count=len(grouped_sample_counts),
        phenotype_group_count=len(grouped_run_inputs),
        sample_counts_differ=len(set(grouped_sample_counts)) > 1,
        sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE.value,
    )
    events.record_multi_phenotype_sample_summary_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        types.MultiPhenotypeSampleMode.PER_PHENOTYPE.value,
        grouped_sample_counts,
        grouped_sample_set_fingerprints,
        len(grouped_run_inputs),
    )
    validate_grouped_per_phenotype_resume_compatibility(
        context=context,
        engine=engine,
        grouped_run_inputs=grouped_run_inputs,
    )

    if should_use_union_grouped_bgen_delivery(context=context, grouped_run_inputs=grouped_run_inputs):
        return run_prepared_grouped_per_phenotype_union_bgen_pipeline(
            context=context,
            engine=engine,
            grouped_run_inputs=grouped_run_inputs,
            phenotype_names=phenotype_names,
            prepared_runs_by_phenotype=prepared_runs_by_phenotype,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
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
            prepared_runs_by_phenotype=typing.cast(
                "tuple[_core.NativeRunLifecyclePhenotypeRun, ...]",
                pipeline_context.select_by_phenotype_indices(
                    prepared_runs_by_phenotype,
                    compute_group.phenotype_indices,
                ),
            ),
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            output_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
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
) -> None:
    """Validate all grouped per-phenotype manifests before initializing any group."""
    if not context.lifecycle_session.output_resume:
        return
    selected_phenotype_names: list[str] = []
    selected_current_headers: list[outputs.RunManifestHeaderInput] = []
    for grouped_run_input in grouped_run_inputs:
        compute_group = grouped_run_input.compute_group
        run_input = grouped_run_input.run_input
        for phenotype_name in compute_group.phenotype_names:
            selected_phenotype_names.append(phenotype_name)
            selected_current_headers.append(
                outputs.build_pipeline_manifest_header(
                    context=context,
                    phenotype_name=phenotype_name,
                    covariate_names=tuple(run_input.native_multi_aligned_sample_data.covariate_names),
                    sample_count=int(run_input.sample_indices.shape[0]),
                    variant_count=int(engine.variant_count),
                    multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
                    phenotype_compute_group=compute_group,
                )
            )
    outputs.validate_pipeline_resume_compatibility(
        context=context,
        phenotype_names=tuple(selected_phenotype_names),
        current_headers_by_trait=tuple(selected_current_headers),
    )


def build_union_sample_indices(
    grouped_run_inputs: tuple[native_dispatch_models.NativeBgenGroupedRunInput, ...],
) -> npt.NDArray[np.int64]:
    """Build an ordered union sample selection for compatible phenotype groups through Rust."""
    return _core.build_union_sample_indices(
        tuple(grouped_run_input.run_input.sample_indices for grouped_run_input in grouped_run_inputs)
    )


def build_validated_grouped_union_sample_indices(
    grouped_run_inputs: tuple[native_dispatch_models.NativeBgenGroupedRunInput, ...],
    native_callback_batch_size: int,
) -> npt.NDArray[np.int64]:
    """Build grouped union sample selection after Rust validates delivery constraints."""
    return _core.build_grouped_union_sample_indices(
        tuple(grouped_run_input.run_input.sample_indices for grouped_run_input in grouped_run_inputs),
        native_callback_batch_size,
    )


def build_group_sample_position_array(
    *,
    union_sample_indices: npt.NDArray[np.int64],
    group_sample_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.intp]:
    """Map one group's sample order to positions in the union decode buffer through Rust."""
    return _core.build_group_sample_position_array(union_sample_indices, group_sample_indices)


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
    prepared_runs_by_phenotype: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Run overlapping per-phenotype groups through one union-sample BGEN delivery."""
    union_sample_indices = build_validated_grouped_union_sample_indices(grouped_run_inputs, native_callback_batch_size)
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
            prepared_runs_by_phenotype=typing.cast(
                "tuple[_core.NativeRunLifecyclePhenotypeRun, ...]",
                pipeline_context.select_by_phenotype_indices(
                    prepared_runs_by_phenotype,
                    grouped_run_input.compute_group.phenotype_indices,
                ),
            ),
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            output_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
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
    final_parquet_paths = native_dispatch_delivery.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=union_run_input,
        committed_chunk_identifiers=outputs.shared_committed_chunk_identifiers_across(
            tuple(prepared_delivery.output_initialization for prepared_delivery in prepared_deliveries)
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
