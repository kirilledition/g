"""Native BGEN callback helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import g.engine.callbacks.binary as binary
import g.engine.callbacks.diagnostics as diagnostics
import g.engine.callbacks.grouped as grouped
import g.engine.callbacks.linear as linear
import g.engine.callbacks.runtime as runtime
import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
import g.engine.callbacks.writers as writers

DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = shared.DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS
RESULT_WORKER_JOIN_TIMEOUT_SECONDS = shared.RESULT_WORKER_JOIN_TIMEOUT_SECONDS
GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = shared.GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS
GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS = shared.GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS
WORKER_ABORT_STOP_TIMEOUT_SECONDS = shared.WORKER_ABORT_STOP_TIMEOUT_SECONDS
jax = shared.jax
jnp = shared.jnp
logger = shared.logger
HostGenotypeBuffer = shared.HostGenotypeBuffer
HostOrDeviceFloatArray = shared.HostOrDeviceFloatArray
LinearChunkStatsArrays = shared.LinearChunkStatsArrays
BinaryChunkStatsArrays = shared.BinaryChunkStatsArrays
MultiPhenotypeGroupFanout = shared.MultiPhenotypeGroupFanout
NativeBgenWorkerShutdownError = shared.NativeBgenWorkerShutdownError
PreprocessedDosageChunkWorkItem = shared.PreprocessedDosageChunkWorkItem
PreprocessedVariantMajorDosageChunkWorkItem = shared.PreprocessedVariantMajorDosageChunkWorkItem
PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem = (
    shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
)
Regenie2ResultWriteWorkItem = shared.Regenie2ResultWriteWorkItem
Regenie2MultiResultWriteWorkItem = shared.Regenie2MultiResultWriteWorkItem
NativeBgenRunInputProtocol = shared.NativeBgenRunInputProtocol
NativeBgenMultiRunInputProtocol = shared.NativeBgenMultiRunInputProtocol
RegeniePredictionSourceProtocol = shared.RegeniePredictionSourceProtocol
MultiRegeniePredictionSourceProtocol = shared.MultiRegeniePredictionSourceProtocol
get_metadata_chromosome = shared.get_metadata_chromosome
cast_statistic_array_for_native_writer = transfers.cast_statistic_array_for_native_writer
narrow_public_statistic_array_on_device = transfers.narrow_public_statistic_array_on_device
select_active_trait_rows_on_device = transfers.select_active_trait_rows_on_device
get_chunk_stats_compute_arrays = transfers.get_chunk_stats_compute_arrays
get_linear_chunk_stats_arrays = transfers.get_linear_chunk_stats_arrays
get_binary_chunk_stats_arrays = transfers.get_binary_chunk_stats_arrays
put_compute_array_on_device = transfers.put_compute_array_on_device
put_genotype_matrix_on_device = transfers.put_genotype_matrix_on_device
put_chunk_array_on_device = transfers.put_chunk_array_on_device
block_compute_result_for_timing = transfers.block_compute_result_for_timing
build_chunk_timing_identity = transfers.build_chunk_timing_identity
record_stage_duration_with_optional_chunk = transfers.record_stage_duration_with_optional_chunk
record_transfer_metadata_for_array = transfers.record_transfer_metadata_for_array
build_projected_variant_major_dosage_chunk_stats = transfers.build_projected_variant_major_dosage_chunk_stats
block_until_ready = diagnostics.block_until_ready
enforce_null_logistic_nonconvergence_policy = diagnostics.enforce_null_logistic_nonconvergence_policy
record_binary_chunk_diagnostics = diagnostics.record_binary_chunk_diagnostics
record_binary_chunk_diagnostics_from_count = diagnostics.record_binary_chunk_diagnostics_from_count
write_regenie2_native_chunk_with_optional_timing = writers.write_regenie2_native_chunk_with_optional_timing
write_regenie2_multi_native_chunk_with_optional_timing = writers.write_regenie2_multi_native_chunk_with_optional_timing
require_current_chromosome_state = runtime.require_current_chromosome_state
NativeBgenCallbackRunner = runtime.NativeBgenCallbackRunner
LinearRegenie2PipelineCallback = linear.LinearRegenie2PipelineCallback
MultiLinearRegenie2PipelineCallback = linear.MultiLinearRegenie2PipelineCallback
BinaryRegenie2PipelineCallback = binary.BinaryRegenie2PipelineCallback
MultiBinaryRegenie2PipelineCallback = binary.MultiBinaryRegenie2PipelineCallback
GroupedMultiPhenotypeFanoutCallback = grouped.GroupedMultiPhenotypeFanoutCallback

__all__ = [
    "DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "WORKER_ABORT_STOP_TIMEOUT_SECONDS",
    "BinaryChunkStatsArrays",
    "BinaryRegenie2PipelineCallback",
    "GroupedMultiPhenotypeFanoutCallback",
    "HostGenotypeBuffer",
    "HostOrDeviceFloatArray",
    "LinearChunkStatsArrays",
    "LinearRegenie2PipelineCallback",
    "MultiBinaryRegenie2PipelineCallback",
    "MultiLinearRegenie2PipelineCallback",
    "MultiPhenotypeGroupFanout",
    "MultiRegeniePredictionSourceProtocol",
    "NativeBgenCallbackRunner",
    "NativeBgenMultiRunInputProtocol",
    "NativeBgenRunInputProtocol",
    "NativeBgenWorkerShutdownError",
    "PreprocessedDosageChunkWorkItem",
    "PreprocessedVariantMajorDosageChunkWorkItem",
    "PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem",
    "Regenie2MultiResultWriteWorkItem",
    "Regenie2ResultWriteWorkItem",
    "RegeniePredictionSourceProtocol",
    "block_compute_result_for_timing",
    "block_until_ready",
    "build_chunk_timing_identity",
    "build_projected_variant_major_dosage_chunk_stats",
    "cast_statistic_array_for_native_writer",
    "enforce_null_logistic_nonconvergence_policy",
    "get_binary_chunk_stats_arrays",
    "get_chunk_stats_compute_arrays",
    "get_linear_chunk_stats_arrays",
    "get_metadata_chromosome",
    "jax",
    "jnp",
    "logger",
    "narrow_public_statistic_array_on_device",
    "put_chunk_array_on_device",
    "put_compute_array_on_device",
    "put_genotype_matrix_on_device",
    "record_binary_chunk_diagnostics",
    "record_binary_chunk_diagnostics_from_count",
    "record_stage_duration_with_optional_chunk",
    "record_transfer_metadata_for_array",
    "require_current_chromosome_state",
    "select_active_trait_rows_on_device",
    "write_regenie2_multi_native_chunk_with_optional_timing",
    "write_regenie2_native_chunk_with_optional_timing",
]
