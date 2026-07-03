"""Callback construction helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

import g.engine.callbacks.binary as callback_binary
import g.engine.callbacks.grouped as callback_grouped
import g.engine.callbacks.linear as callback_linear
import g.engine.callbacks.shared as callback_shared
from g import types
from g.engine.regenie2_pipeline import context as pipeline_context
from g.engine.regenie2_pipeline import inputs

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

type MultiPhenotypeGroupCallbackProtocol = callback_shared.MultiPhenotypeGroupCallbackProtocol
type MultiPhenotypeGroupFanout = callback_shared.MultiPhenotypeGroupFanout


def build_single_trait_callback(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    run_input: inputs.NativeBgenRunInput,
    prediction_source: typing.Any,
    writer_session: typing.Any,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> inputs.BgenDeliveryCallbackProtocol:
    """Build the association-specific single-trait callback."""
    if context.is_binary_trait:
        return callback_binary.BinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_session=writer_session,
            correction_plan=context.correction_plan,
            kernel_config=pipeline_context.require_binary_kernel_config(context.binary_kernel_config),
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
            output_statistic_dtype=context.writer_settings.output_statistic_dtype,
        )
    return callback_linear.LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=context.score_dtype,
        linear_numerical_config=pipeline_context.require_linear_numerical_config(context.linear_numerical_config),
        stage_timing_recorder=context.stage_timing_recorder,
        telemetry_session=context.telemetry_session,
        output_statistic_dtype=context.writer_settings.output_statistic_dtype,
    )


def build_multi_phenotype_group_callback(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    run_input: inputs.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    writer_sessions: tuple[typing.Any, ...],
    committed_chunk_identifier_sets: tuple[set[int], ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> MultiPhenotypeGroupCallbackProtocol:
    """Build the association-specific callback for one compatible phenotype group."""
    if context.is_binary_trait:
        return callback_binary.MultiBinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_sessions,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            correction_plan=context.correction_plan,
            kernel_config=pipeline_context.require_binary_kernel_config(context.binary_kernel_config),
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
            output_statistic_dtype=context.writer_settings.output_statistic_dtype,
        )
    return callback_linear.MultiLinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=committed_chunk_identifier_sets,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=context.score_dtype,
        linear_numerical_config=pipeline_context.require_linear_numerical_config(context.linear_numerical_config),
        stage_timing_recorder=context.stage_timing_recorder,
        telemetry_session=context.telemetry_session,
        output_statistic_dtype=context.writer_settings.output_statistic_dtype,
    )


def build_multi_phenotype_group_fanout(
    *,
    callback: MultiPhenotypeGroupCallbackProtocol,
    sample_position_array: npt.NDArray[np.intp],
) -> MultiPhenotypeGroupFanout:
    """Build one group fanout for union-sample grouped delivery."""
    return callback_shared.MultiPhenotypeGroupFanout(
        callback=callback,
        sample_position_array=sample_position_array,
    )


def build_grouped_multi_phenotype_fanout_callback(
    group_fanouts: tuple[MultiPhenotypeGroupFanout, ...],
) -> inputs.BgenDeliveryCallbackProtocol:
    """Build the union-sample grouped delivery fanout callback."""
    return callback_grouped.GroupedMultiPhenotypeFanoutCallback(group_fanouts)
