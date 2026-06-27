"""Native REGENIE result materialization and writer helpers."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass

import jax

import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
from g import _core, types

if typing.TYPE_CHECKING:
    from g.engine import timing

cast_statistic_array_for_native_writer = transfers.cast_statistic_array_for_native_writer
cast_statistic_array_for_native_writer_float32 = transfers.cast_statistic_array_for_native_writer_float32
cast_statistic_array_for_native_writer_float64 = transfers.cast_statistic_array_for_native_writer_float64
narrow_public_statistic_array_on_device = transfers.narrow_public_statistic_array_on_device
record_stage_duration_with_optional_chunk = transfers.record_stage_duration_with_optional_chunk
record_transfer_metadata_for_array = transfers.record_transfer_metadata_for_array
select_active_trait_rows_on_device = transfers.select_active_trait_rows_on_device
get_metadata_chromosome = shared.get_metadata_chromosome


@dataclass(frozen=True)
class MaterializedRegenie2NativeChunk:
    """Host-materialized single-trait REGENIE result arrays."""

    beta: object
    standard_error: object
    chi_squared: object
    log10_p_value: object
    extra_code: object | None


@dataclass(frozen=True)
class MaterializedRegenie2MultiNativeChunk:
    """Host-materialized multi-trait REGENIE result arrays and active writers."""

    active_writer_sessions: tuple[typing.Any, ...]
    use_native_multi_writer: bool
    beta: object | None
    standard_error: object | None
    chi_squared: object | None
    log10_p_value: object | None
    extra_code: object | None


def materialize_regenie2_native_chunk_with_optional_timing(
    *,
    metadata: _core.VariantMetadata,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    output_statistic_dtype: types.FloatingPointDtype,
) -> MaterializedRegenie2NativeChunk:
    """Materialize one single-trait REGENIE result chunk on host."""
    materialization_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
    beta_device_array = narrow_public_statistic_array_on_device(beta, output_statistic_dtype)
    standard_error_device_array = narrow_public_statistic_array_on_device(standard_error, output_statistic_dtype)
    chi_squared_device_array = narrow_public_statistic_array_on_device(chi_squared, output_statistic_dtype)
    log10_p_value_device_array = narrow_public_statistic_array_on_device(log10_p_value, output_statistic_dtype)
    if stage_timing_recorder is not None:
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="beta",
            array=beta_device_array,
        )
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="standard_error",
            array=standard_error_device_array,
        )
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="chi_squared",
            array=chi_squared_device_array,
        )
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="log10_p_value",
            array=log10_p_value_device_array,
        )
        if extra_code is not None:
            record_transfer_metadata_for_array(
                stage_timing_recorder=stage_timing_recorder,
                transfer_name="device_to_host_materialization",
                array_role="extra_code",
                array=extra_code,
            )
    host_values = jax.device_get(
        {
            "beta": beta_device_array,
            "standard_error": standard_error_device_array,
            "chi_squared": chi_squared_device_array,
            "log10_p_value": log10_p_value_device_array,
            "extra_code": extra_code,
        }
    )
    if stage_timing_recorder is not None:
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="device_to_host_materialization",
            start_time=materialization_start_time,
            chunk_metadata=metadata,
        )
    return MaterializedRegenie2NativeChunk(
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        chi_squared=host_values["chi_squared"],
        log10_p_value=host_values["log10_p_value"],
        extra_code=host_values["extra_code"],
    )


def write_materialized_regenie2_native_chunk_with_optional_timing(
    *,
    writer_session: typing.Any,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    materialized_chunk: MaterializedRegenie2NativeChunk,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    output_statistic_dtype: types.FloatingPointDtype,
) -> None:
    """Write one already-materialized single-trait REGENIE result chunk."""
    write_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
    write_plan = _core.plan_single_trait_output_write(
        is_native_writer_session=isinstance(writer_session, _core.OutputWriterSession),
        output_statistic_dtype=output_statistic_dtype.value,
    )
    write_chunk_method = getattr(writer_session, write_plan.method_name)
    write_chunk_method(
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=cast_statistic_array_for_native_writer(materialized_chunk.beta, output_statistic_dtype),
        standard_error=cast_statistic_array_for_native_writer(
            materialized_chunk.standard_error,
            output_statistic_dtype,
        ),
        chi_squared=cast_statistic_array_for_native_writer(materialized_chunk.chi_squared, output_statistic_dtype),
        log10_p_value=cast_statistic_array_for_native_writer(
            materialized_chunk.log10_p_value,
            output_statistic_dtype,
        ),
        extra_code=materialized_chunk.extra_code,
    )
    if stage_timing_recorder is not None:
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="output_write",
            start_time=write_start_time,
            chunk_metadata=metadata,
        )
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="single_trait_output_write",
            start_time=write_start_time,
            chunk_metadata=metadata,
        )


def write_regenie2_native_chunk_with_optional_timing(
    *,
    writer_session: typing.Any,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    output_statistic_dtype: types.FloatingPointDtype,
) -> None:
    """Write one native-metadata REGENIE chunk while timing JAX result materialization.

    The native Arrow/Parquet schema stores public result statistics with the
    configured output dtype. Internal arrays are cast immediately before the
    Rust writer call.
    """
    materialized_chunk = materialize_regenie2_native_chunk_with_optional_timing(
        metadata=metadata,
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        stage_timing_recorder=stage_timing_recorder,
        output_statistic_dtype=output_statistic_dtype,
    )
    write_materialized_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=metadata,
        chunk_stats=chunk_stats,
        materialized_chunk=materialized_chunk,
        stage_timing_recorder=stage_timing_recorder,
        output_statistic_dtype=output_statistic_dtype,
    )


def materialize_regenie2_multi_native_chunk_with_optional_timing(
    *,
    writer_sessions: tuple[typing.Any, ...],
    committed_chunk_identifier_sets: tuple[set[int], ...],
    metadata: _core.VariantMetadata,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    output_statistic_dtype: types.FloatingPointDtype,
) -> MaterializedRegenie2MultiNativeChunk:
    """Materialize one multi-trait REGENIE result chunk on host."""
    chunk_identifier = int(metadata.variant_start_index)
    committed_chunk_identifier_batches = tuple(
        tuple(committed_chunk_identifier_set) for committed_chunk_identifier_set in committed_chunk_identifier_sets
    )
    write_plan = _core.plan_multi_trait_chunk_write(
        writer_session_count=len(writer_sessions),
        chunk_identifier=chunk_identifier,
        committed_chunk_identifier_sets=committed_chunk_identifier_batches,
    )
    active_trait_indices = tuple(write_plan.active_trait_indices)
    use_native_multi_writer = all(
        isinstance(writer_session, _core.OutputWriterSession) for writer_session in writer_sessions
    )
    if write_plan.all_traits_committed:
        return MaterializedRegenie2MultiNativeChunk(
            active_writer_sessions=(),
            use_native_multi_writer=use_native_multi_writer,
            beta=None,
            standard_error=None,
            chi_squared=None,
            log10_p_value=None,
            extra_code=None,
        )

    active_writer_sessions = tuple(writer_sessions[trait_index] for trait_index in active_trait_indices)
    total_trait_count = write_plan.total_trait_count
    active_extra_code = None
    if extra_code is not None:
        active_extra_code = select_active_trait_rows_on_device(
            extra_code,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        )

    materialization_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
    beta_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            beta,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        ),
        output_statistic_dtype,
    )
    standard_error_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            standard_error,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        ),
        output_statistic_dtype,
    )
    chi_squared_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            chi_squared,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        ),
        output_statistic_dtype,
    )
    log10_p_value_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            log10_p_value,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        ),
        output_statistic_dtype,
    )
    if stage_timing_recorder is not None:
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="beta",
            array=beta_device_array,
        )
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="standard_error",
            array=standard_error_device_array,
        )
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="chi_squared",
            array=chi_squared_device_array,
        )
        record_transfer_metadata_for_array(
            stage_timing_recorder=stage_timing_recorder,
            transfer_name="device_to_host_materialization",
            array_role="log10_p_value",
            array=log10_p_value_device_array,
        )
        if active_extra_code is not None:
            record_transfer_metadata_for_array(
                stage_timing_recorder=stage_timing_recorder,
                transfer_name="device_to_host_materialization",
                array_role="extra_code",
                array=active_extra_code,
            )
    host_values = jax.device_get(
        {
            "beta": beta_device_array,
            "standard_error": standard_error_device_array,
            "chi_squared": chi_squared_device_array,
            "log10_p_value": log10_p_value_device_array,
            "extra_code": active_extra_code,
        }
    )
    if stage_timing_recorder is not None:
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="device_to_host_materialization",
            start_time=materialization_start_time,
            chunk_metadata=metadata,
        )
    return MaterializedRegenie2MultiNativeChunk(
        active_writer_sessions=active_writer_sessions,
        use_native_multi_writer=use_native_multi_writer,
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        chi_squared=host_values["chi_squared"],
        log10_p_value=host_values["log10_p_value"],
        extra_code=host_values["extra_code"],
    )


def write_materialized_regenie2_multi_native_chunk_with_optional_timing(
    *,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    materialized_chunk: MaterializedRegenie2MultiNativeChunk,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    output_statistic_dtype: types.FloatingPointDtype,
) -> None:
    """Write one already-materialized multi-trait REGENIE result chunk."""
    write_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
    active_writer_sessions = materialized_chunk.active_writer_sessions
    if not active_writer_sessions:
        if stage_timing_recorder is not None:
            record_stage_duration_with_optional_chunk(
                stage_timing_recorder=stage_timing_recorder,
                stage_name="output_write",
                start_time=write_start_time,
                chunk_metadata=metadata,
            )
            record_stage_duration_with_optional_chunk(
                stage_timing_recorder=stage_timing_recorder,
                stage_name="multi_trait_output_write_total",
                start_time=write_start_time,
                chunk_metadata=metadata,
            )
        return
    write_plan = _core.plan_multi_trait_output_write(
        active_trait_count=len(active_writer_sessions),
        all_writer_sessions_native=materialized_chunk.use_native_multi_writer,
        output_statistic_dtype=output_statistic_dtype.value,
    )
    if write_plan.use_native_multi_writer:
        native_writer_sessions = typing.cast("tuple[_core.OutputWriterSession, ...]", active_writer_sessions)
        native_extra_code = typing.cast("typing.Any", materialized_chunk.extra_code)
        beta = materialized_chunk.beta
        standard_error = materialized_chunk.standard_error
        chi_squared = materialized_chunk.chi_squared
        log10_p_value = materialized_chunk.log10_p_value
        active_trait_indices = list(range(write_plan.active_trait_count))
        if write_plan.uses_float64_native_writer:
            _core.write_regenie2_multi_native_chunk_f64(
                writer_sessions=list(native_writer_sessions),
                active_trait_indices=active_trait_indices,
                metadata=metadata,
                chunk_stats=chunk_stats,
                beta=cast_statistic_array_for_native_writer_float64(beta),
                standard_error=cast_statistic_array_for_native_writer_float64(standard_error),
                chi_squared=cast_statistic_array_for_native_writer_float64(chi_squared),
                log10_p_value=cast_statistic_array_for_native_writer_float64(log10_p_value),
                extra_code=native_extra_code,
            )
        else:
            _core.write_regenie2_multi_native_chunk(
                writer_sessions=list(native_writer_sessions),
                active_trait_indices=active_trait_indices,
                metadata=metadata,
                chunk_stats=chunk_stats,
                beta=cast_statistic_array_for_native_writer_float32(beta),
                standard_error=cast_statistic_array_for_native_writer_float32(standard_error),
                chi_squared=cast_statistic_array_for_native_writer_float32(chi_squared),
                log10_p_value=cast_statistic_array_for_native_writer_float32(log10_p_value),
                extra_code=native_extra_code,
            )
        if stage_timing_recorder is not None:
            record_stage_duration_with_optional_chunk(
                stage_timing_recorder=stage_timing_recorder,
                stage_name="output_write",
                start_time=write_start_time,
                chunk_metadata=metadata,
            )
            record_stage_duration_with_optional_chunk(
                stage_timing_recorder=stage_timing_recorder,
                stage_name="multi_trait_output_write_total",
                start_time=write_start_time,
                chunk_metadata=metadata,
            )
        return
    for compact_trait_index, writer_session in enumerate(active_writer_sessions):
        per_trait_write_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
        extra_code_slice = None
        if materialized_chunk.extra_code is not None:
            extra_code = typing.cast("typing.Any", materialized_chunk.extra_code)
            extra_code_slice = extra_code[compact_trait_index]
        beta = typing.cast("typing.Any", materialized_chunk.beta)
        standard_error = typing.cast("typing.Any", materialized_chunk.standard_error)
        chi_squared = typing.cast("typing.Any", materialized_chunk.chi_squared)
        log10_p_value = typing.cast("typing.Any", materialized_chunk.log10_p_value)
        writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(beta[compact_trait_index], output_statistic_dtype),
            standard_error=cast_statistic_array_for_native_writer(
                standard_error[compact_trait_index],
                output_statistic_dtype,
            ),
            chi_squared=cast_statistic_array_for_native_writer(
                chi_squared[compact_trait_index],
                output_statistic_dtype,
            ),
            log10_p_value=cast_statistic_array_for_native_writer(
                log10_p_value[compact_trait_index],
                output_statistic_dtype,
            ),
            extra_code=extra_code_slice,
        )
        if stage_timing_recorder is not None:
            record_stage_duration_with_optional_chunk(
                stage_timing_recorder=stage_timing_recorder,
                stage_name="multi_trait_output_write_per_trait",
                start_time=per_trait_write_start_time,
                chunk_metadata=metadata,
            )
    if stage_timing_recorder is not None:
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="output_write",
            start_time=write_start_time,
            chunk_metadata=metadata,
        )
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="multi_trait_output_write_total",
            start_time=write_start_time,
            chunk_metadata=metadata,
        )


def write_regenie2_multi_native_chunk_with_optional_timing(
    *,
    writer_sessions: tuple[typing.Any, ...],
    committed_chunk_identifier_sets: tuple[set[int], ...],
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    output_statistic_dtype: types.FloatingPointDtype,
) -> None:
    """Materialize one multi-trait result once and write missing per-trait slices."""
    materialized_chunk = materialize_regenie2_multi_native_chunk_with_optional_timing(
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=committed_chunk_identifier_sets,
        metadata=metadata,
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
        stage_timing_recorder=stage_timing_recorder,
        output_statistic_dtype=output_statistic_dtype,
    )
    write_materialized_regenie2_multi_native_chunk_with_optional_timing(
        metadata=metadata,
        chunk_stats=chunk_stats,
        materialized_chunk=materialized_chunk,
        stage_timing_recorder=stage_timing_recorder,
        output_statistic_dtype=output_statistic_dtype,
    )


__all__ = [
    "get_metadata_chromosome",
    "materialize_regenie2_multi_native_chunk_with_optional_timing",
    "materialize_regenie2_native_chunk_with_optional_timing",
    "write_materialized_regenie2_multi_native_chunk_with_optional_timing",
    "write_materialized_regenie2_native_chunk_with_optional_timing",
    "write_regenie2_multi_native_chunk_with_optional_timing",
    "write_regenie2_native_chunk_with_optional_timing",
]
