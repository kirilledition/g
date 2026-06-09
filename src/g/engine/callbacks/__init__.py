"""Native BGEN callback helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import sys
import types
import typing

import g.engine.callbacks._legacy as _legacy

from g.engine.callbacks.shared import (
    DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS,
    RESULT_WORKER_JOIN_TIMEOUT_SECONDS,
    GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS,
    GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS,
    WORKER_ABORT_STOP_TIMEOUT_SECONDS,
    jax,
    jnp,
    HostGenotypeBuffer,
    HostOrDeviceFloatArray,
    LinearChunkStatsArrays,
    BinaryChunkStatsArrays,
    MultiPhenotypeGroupFanout,
    NativeBgenWorkerShutdownError,
    PreprocessedDosageChunkWorkItem,
    PreprocessedVariantMajorDosageChunkWorkItem,
    PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem,
    Regenie2ResultWriteWorkItem,
    Regenie2MultiResultWriteWorkItem,
    NativeBgenRunInputProtocol,
    NativeBgenMultiRunInputProtocol,
    RegeniePredictionSourceProtocol,
    MultiRegeniePredictionSourceProtocol,
    logger,
)
from g.engine.callbacks.transfers import (
    cast_statistic_array_for_native_writer,
    narrow_public_statistic_array_on_device,
    select_active_trait_rows_on_device,
    get_chunk_stats_compute_arrays,
    get_linear_chunk_stats_arrays,
    get_binary_chunk_stats_arrays,
    put_compute_array_on_device,
    put_genotype_matrix_on_device,
    put_chunk_array_on_device,
    block_compute_result_for_timing,
    build_chunk_timing_identity,
    record_stage_duration_with_optional_chunk,
    record_transfer_metadata_for_array,
    build_projected_variant_major_dosage_chunk_stats,
)
from g.engine.callbacks.diagnostics import (
    block_until_ready,
    enforce_null_logistic_nonconvergence_policy,
    record_binary_chunk_diagnostics,
)
from g.engine.callbacks.writers import (
    write_regenie2_native_chunk_with_optional_timing,
    write_regenie2_multi_native_chunk_with_optional_timing,
    get_metadata_chromosome,
)
from g.engine.callbacks.runtime import (
    require_current_chromosome_state,
    NativeBgenCallbackRunner,
)
from g.engine.callbacks.linear import (
    LinearRegenie2PipelineCallback,
    MultiLinearRegenie2PipelineCallback,
)
from g.engine.callbacks.binary import (
    BinaryRegenie2PipelineCallback,
    MultiBinaryRegenie2PipelineCallback,
)
from g.engine.callbacks.grouped import (
    GroupedMultiPhenotypeFanoutCallback,
)


__all__ = [
    "DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "WORKER_ABORT_STOP_TIMEOUT_SECONDS",
    "jax",
    "jnp",
    "HostGenotypeBuffer",
    "HostOrDeviceFloatArray",
    "LinearChunkStatsArrays",
    "BinaryChunkStatsArrays",
    "MultiPhenotypeGroupFanout",
    "NativeBgenWorkerShutdownError",
    "PreprocessedDosageChunkWorkItem",
    "PreprocessedVariantMajorDosageChunkWorkItem",
    "PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem",
    "Regenie2ResultWriteWorkItem",
    "Regenie2MultiResultWriteWorkItem",
    "NativeBgenRunInputProtocol",
    "NativeBgenMultiRunInputProtocol",
    "RegeniePredictionSourceProtocol",
    "MultiRegeniePredictionSourceProtocol",
    "logger",
    "cast_statistic_array_for_native_writer",
    "narrow_public_statistic_array_on_device",
    "select_active_trait_rows_on_device",
    "get_chunk_stats_compute_arrays",
    "get_linear_chunk_stats_arrays",
    "get_binary_chunk_stats_arrays",
    "put_compute_array_on_device",
    "put_genotype_matrix_on_device",
    "put_chunk_array_on_device",
    "block_compute_result_for_timing",
    "build_chunk_timing_identity",
    "record_stage_duration_with_optional_chunk",
    "record_transfer_metadata_for_array",
    "build_projected_variant_major_dosage_chunk_stats",
    "block_until_ready",
    "enforce_null_logistic_nonconvergence_policy",
    "record_binary_chunk_diagnostics",
    "write_regenie2_native_chunk_with_optional_timing",
    "write_regenie2_multi_native_chunk_with_optional_timing",
    "get_metadata_chromosome",
    "require_current_chromosome_state",
    "NativeBgenCallbackRunner",
    "LinearRegenie2PipelineCallback",
    "MultiLinearRegenie2PipelineCallback",
    "BinaryRegenie2PipelineCallback",
    "MultiBinaryRegenie2PipelineCallback",
    "GroupedMultiPhenotypeFanoutCallback",
]

_sync_symbol_names = frozenset(__all__)


def _sync_attribute_to_legacy(name: str, value: typing.Any) -> None:
    """Keep root patching effects visible inside legacy implementation objects."""
    if hasattr(_legacy, name):
        setattr(_legacy, name, value)


class _CallbacksFacadeModule(types.ModuleType):
    """Synchronize patched public attributes with legacy implementation internals."""

    def __setattr__(self, name: str, value: typing.Any) -> None:  # type: ignore[override]
        super().__setattr__(name, value)
        if name in _sync_symbol_names:
            _sync_attribute_to_legacy(name, value)


def _install_facet_syncing() -> None:
    """Enable callback-module-level patch synchronization for monkeypatch compatibility."""
    module = sys.modules[__name__]
    if isinstance(module, _CallbacksFacadeModule):
        return
    module.__class__ = _CallbacksFacadeModule
    for key in _sync_symbol_names:
        if key in globals():
            _sync_attribute_to_legacy(key, globals()[key])


_install_facet_syncing()
