"""Callback factory for native-owned REGENIE run orchestration."""

from __future__ import annotations

import dataclasses
import json
import typing

import g.engine.callbacks.binary as callback_binary
import g.engine.callbacks.grouped as callback_grouped
import g.engine.callbacks.linear as callback_linear
import g.engine.callbacks.shared as callback_shared
from g import _core, types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import config as regenie2_linear_config


class NativeRunCallbackFactory:
    """Build Python/JAX callbacks for native-owned run execution."""

    def __init__(
        self,
        regenie_config: _core.RegenieConfig,
        telemetry_session: typing.Any,
        stage_timing_recorder: typing.Any,
    ) -> None:
        """Initialize callback construction state."""
        self.regenie_config = regenie_config
        self.telemetry_session = telemetry_session
        self.stage_timing_recorder = stage_timing_recorder
        self.binary_kernel_config = build_binary_kernel_config(regenie_config.g_compute)
        self.linear_numerical_config = build_linear_numerical_config(regenie_config.g_compute)

    def binary_kernel_config_json(self) -> str | None:
        """Serialize binary kernel config for native output manifests."""
        if self.regenie_config.trait.trait_type != types.RegenieTraitType.BINARY:
            return None
        return json.dumps(
            dataclasses.asdict(self.binary_kernel_config),
            sort_keys=True,
            separators=(",", ":"),
        )

    def build_single_trait_callback(
        self,
        context: _core.NativeRunCallbackContext,
        run_input: typing.Any,
        prediction_source: typing.Any,
        writer_session: typing.Any,
    ) -> typing.Any:
        """Build one single-trait callback."""
        score_dtype = types.FloatingPointDtype(context.score_dtype)
        output_statistic_dtype = types.FloatingPointDtype(context.output_statistic_dtype)
        if context.trait_type == types.RegenieTraitType.BINARY.value:
            return callback_binary.BinaryRegenie2PipelineCallback(
                run_input=run_input,
                prediction_source=prediction_source,
                writer_session=writer_session,
                correction_plan=binary_correction_plan_from_context(context),
                kernel_config=self.binary_kernel_config,
                null_logistic_nonconvergence_policy=self.regenie_config.g_compute.null_logistic_nonconvergence_policy,
                staging_depth=context.staging_depth,
                native_callback_batch_size=context.native_callback_batch_size,
                result_in_flight_limit=context.result_in_flight_limit,
                dosage_buffer_limit=context.dosage_buffer_limit,
                score_dtype=score_dtype,
                stage_timing_recorder=self.stage_timing_recorder,
                telemetry_session=self.telemetry_session,
                output_statistic_dtype=output_statistic_dtype,
            )
        return callback_linear.LinearRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_session=writer_session,
            staging_depth=context.staging_depth,
            native_callback_batch_size=context.native_callback_batch_size,
            result_in_flight_limit=context.result_in_flight_limit,
            dosage_buffer_limit=context.dosage_buffer_limit,
            score_dtype=score_dtype,
            linear_numerical_config=self.linear_numerical_config,
            stage_timing_recorder=self.stage_timing_recorder,
            telemetry_session=self.telemetry_session,
            output_statistic_dtype=output_statistic_dtype,
        )

    def build_multi_trait_callback(
        self,
        context: _core.NativeRunCallbackContext,
        run_input: typing.Any,
        prediction_source: typing.Any,
        writer_sessions: tuple[typing.Any, ...],
        chunk_write_planner: _core.NativeMultiTraitChunkWritePlanner,
    ) -> callback_shared.MultiPhenotypeGroupCallbackProtocol:
        """Build one compatible multi-trait group callback."""
        score_dtype = types.FloatingPointDtype(context.score_dtype)
        output_statistic_dtype = types.FloatingPointDtype(context.output_statistic_dtype)
        if context.trait_type == types.RegenieTraitType.BINARY.value:
            return callback_binary.MultiBinaryRegenie2PipelineCallback(
                run_input=run_input,
                prediction_source=prediction_source,
                writer_sessions=writer_sessions,
                chunk_write_planner=chunk_write_planner,
                correction_plan=binary_correction_plan_from_context(context),
                kernel_config=self.binary_kernel_config,
                null_logistic_nonconvergence_policy=self.regenie_config.g_compute.null_logistic_nonconvergence_policy,
                staging_depth=context.staging_depth,
                native_callback_batch_size=context.native_callback_batch_size,
                result_in_flight_limit=context.result_in_flight_limit,
                dosage_buffer_limit=context.dosage_buffer_limit,
                score_dtype=score_dtype,
                stage_timing_recorder=self.stage_timing_recorder,
                telemetry_session=self.telemetry_session,
                output_statistic_dtype=output_statistic_dtype,
            )
        return callback_linear.MultiLinearRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_sessions,
            chunk_write_planner=chunk_write_planner,
            staging_depth=context.staging_depth,
            native_callback_batch_size=context.native_callback_batch_size,
            result_in_flight_limit=context.result_in_flight_limit,
            dosage_buffer_limit=context.dosage_buffer_limit,
            score_dtype=score_dtype,
            linear_numerical_config=self.linear_numerical_config,
            stage_timing_recorder=self.stage_timing_recorder,
            telemetry_session=self.telemetry_session,
            output_statistic_dtype=output_statistic_dtype,
        )

    def build_grouped_fanout_callback(
        self,
        callbacks: tuple[callback_shared.MultiPhenotypeGroupCallbackProtocol, ...],
        sample_position_arrays: tuple[typing.Any, ...],
    ) -> callback_grouped.GroupedMultiPhenotypeFanoutCallback:
        """Build a fanout callback for grouped union delivery."""
        group_fanouts = tuple(
            callback_shared.MultiPhenotypeGroupFanout(
                callback=callback,
                sample_position_array=sample_position_array,
            )
            for callback, sample_position_array in zip(callbacks, sample_position_arrays, strict=True)
        )
        return callback_grouped.GroupedMultiPhenotypeFanoutCallback(group_fanouts)


def build_binary_kernel_config(
    compute_config: _core.GComputeConfig,
) -> regenie2_binary_config.BinaryKernelConfig:
    """Build binary JAX kernel settings from native compute config."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=compute_config.binary_minimum_probability,
            minimum_variance=compute_config.binary_minimum_variance,
            relative_variance_tolerance=compute_config.binary_relative_variance_tolerance,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=compute_config.binary_null_maximum_iterations,
            coefficient_tolerance=compute_config.binary_null_coefficient_tolerance,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=compute_config.firth_batch_size,
            candidate_capacity=compute_config.firth_candidate_capacity,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=compute_config.firth_maximum_iterations,
            gradient_tolerance=compute_config.firth_gradient_tolerance,
            coefficient_tolerance=compute_config.firth_coefficient_tolerance,
            likelihood_tolerance=compute_config.firth_likelihood_tolerance,
            maximum_step_size=compute_config.firth_maximum_step_size,
            pseudo_maximum_iterations=compute_config.firth_pseudo_maximum_iterations,
            pseudo_inner_maximum_iterations=compute_config.firth_pseudo_inner_maximum_iterations,
            newton_raphson_zero_start_iterations=compute_config.firth_newton_raphson_zero_start_iterations,
            line_search_maximum_attempts=compute_config.firth_line_search_maximum_attempts,
            step_halving_maximum_attempts=compute_config.firth_step_halving_maximum_attempts,
            initial_response_scale=compute_config.firth_initial_response_scale,
            sparse_carrier_dosage_threshold=compute_config.firth_sparse_carrier_dosage_threshold,
            step_halving_scale=compute_config.firth_step_halving_scale,
            use_block_math=compute_config.use_block_firth_math,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=compute_config.null_firth_maximum_iterations,
            gradient_tolerance=compute_config.null_firth_gradient_tolerance,
            maximum_step_size=compute_config.null_firth_maximum_step_size,
            fallback_iteration_multiplier=compute_config.null_firth_fallback_iteration_multiplier,
            fallback_step_divisor=compute_config.null_firth_fallback_step_divisor,
            line_search_maximum_attempts=compute_config.null_firth_line_search_maximum_attempts,
            step_halving_scale=compute_config.null_firth_step_halving_scale,
        ),
    )


def build_linear_numerical_config(
    compute_config: _core.GComputeConfig,
) -> regenie2_linear_config.LinearNumericalConfig:
    """Build linear JAX numerical settings from native compute config."""
    return regenie2_linear_config.LinearNumericalConfig(
        minimum_variance=compute_config.linear_minimum_variance,
        relative_variance_tolerance=compute_config.linear_relative_variance_tolerance,
    )


def binary_correction_plan_from_context(context: _core.NativeRunCallbackContext) -> types.BinaryCorrectionPlan:
    """Build Python binary correction settings from native run context."""
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod(context.correction_method),
        p_threshold=context.correction_p_threshold,
        firth_se=context.correction_firth_se,
    )
