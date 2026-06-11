"""Native REGENIE result materialization and writer helpers."""

from __future__ import annotations

import time
import typing

import jax

import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
from g import _core, types

if typing.TYPE_CHECKING:
    from g.engine import timing

cast_statistic_array_for_native_writer = transfers.cast_statistic_array_for_native_writer
narrow_public_statistic_array_on_device = transfers.narrow_public_statistic_array_on_device
record_stage_duration_with_optional_chunk = transfers.record_stage_duration_with_optional_chunk
record_transfer_metadata_for_array = transfers.record_transfer_metadata_for_array
select_active_trait_rows_on_device = transfers.select_active_trait_rows_on_device
get_metadata_chromosome = shared.get_metadata_chromosome


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

    write_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
    write_chunk_method_name = "write_regenie2_native_chunk"
    if output_statistic_dtype == types.FloatingPointDtype.FLOAT64 and isinstance(
        writer_session,
        _core.OutputWriterSession,
    ):
        write_chunk_method_name = "write_regenie2_native_chunk_f64"
    write_chunk_method = getattr(writer_session, write_chunk_method_name)
    write_chunk_method(
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=cast_statistic_array_for_native_writer(host_values["beta"], output_statistic_dtype),
        standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"], output_statistic_dtype),
        chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"], output_statistic_dtype),
        log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"], output_statistic_dtype),
        extra_code=host_values["extra_code"],
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
    chunk_identifier = int(metadata.variant_start_index)
    active_trait_indices = tuple(
        trait_index
        for trait_index, _writer_session in enumerate(writer_sessions)
        if chunk_identifier not in committed_chunk_identifier_sets[trait_index]
    )
    if not active_trait_indices:
        if stage_timing_recorder is not None:
            write_start_time = time.perf_counter()
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

    active_writer_sessions = tuple(writer_sessions[trait_index] for trait_index in active_trait_indices)
    total_trait_count = len(writer_sessions)
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

    write_start_time = time.perf_counter() if stage_timing_recorder is not None else 0.0
    if all(isinstance(writer_session, _core.OutputWriterSession) for writer_session in writer_sessions):
        write_multi_chunk = (
            _core.write_regenie2_multi_native_chunk_f64
            if output_statistic_dtype == types.FloatingPointDtype.FLOAT64
            else _core.write_regenie2_multi_native_chunk
        )
        write_multi_chunk(
            writer_sessions=list(active_writer_sessions),
            active_trait_indices=list(range(len(active_writer_sessions))),
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(host_values["beta"], output_statistic_dtype),
            standard_error=cast_statistic_array_for_native_writer(
                host_values["standard_error"], output_statistic_dtype
            ),
            chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"], output_statistic_dtype),
            log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"], output_statistic_dtype),
            extra_code=host_values["extra_code"],
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
        if host_values["extra_code"] is not None:
            extra_code_slice = host_values["extra_code"][compact_trait_index]
        writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(
                host_values["beta"][compact_trait_index], output_statistic_dtype
            ),
            standard_error=cast_statistic_array_for_native_writer(
                host_values["standard_error"][compact_trait_index],
                output_statistic_dtype,
            ),
            chi_squared=cast_statistic_array_for_native_writer(
                host_values["chi_squared"][compact_trait_index],
                output_statistic_dtype,
            ),
            log10_p_value=cast_statistic_array_for_native_writer(
                host_values["log10_p_value"][compact_trait_index],
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


__all__ = [
    "get_metadata_chromosome",
    "write_regenie2_multi_native_chunk_with_optional_timing",
    "write_regenie2_native_chunk_with_optional_timing",
]
