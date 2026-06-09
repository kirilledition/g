"""Shared REGENIE callback data contracts and constants."""

from __future__ import annotations

from g.engine.callbacks import _legacy

DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = _legacy.DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS
RESULT_WORKER_JOIN_TIMEOUT_SECONDS = _legacy.RESULT_WORKER_JOIN_TIMEOUT_SECONDS
GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = _legacy.GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS
GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS = _legacy.GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS
WORKER_ABORT_STOP_TIMEOUT_SECONDS = _legacy.WORKER_ABORT_STOP_TIMEOUT_SECONDS
logger = _legacy.logger
jax = _legacy.jax
jnp = _legacy.jnp

HostGenotypeBuffer = _legacy.HostGenotypeBuffer
HostOrDeviceFloatArray = _legacy.HostOrDeviceFloatArray

LinearChunkStatsArrays = _legacy.LinearChunkStatsArrays
BinaryChunkStatsArrays = _legacy.BinaryChunkStatsArrays
MultiPhenotypeGroupFanout = _legacy.MultiPhenotypeGroupFanout
NativeBgenWorkerShutdownError = _legacy.NativeBgenWorkerShutdownError

PreprocessedDosageChunkWorkItem = _legacy.PreprocessedDosageChunkWorkItem
PreprocessedVariantMajorDosageChunkWorkItem = _legacy.PreprocessedVariantMajorDosageChunkWorkItem
PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem = (
    _legacy.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
)
Regenie2ResultWriteWorkItem = _legacy.Regenie2ResultWriteWorkItem
Regenie2MultiResultWriteWorkItem = _legacy.Regenie2MultiResultWriteWorkItem

NativeBgenRunInputProtocol = _legacy.NativeBgenRunInputProtocol
NativeBgenMultiRunInputProtocol = _legacy.NativeBgenMultiRunInputProtocol
RegeniePredictionSourceProtocol = _legacy.RegeniePredictionSourceProtocol
MultiRegeniePredictionSourceProtocol = _legacy.MultiRegeniePredictionSourceProtocol

__all__ = [
    "DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "WORKER_ABORT_STOP_TIMEOUT_SECONDS",
    "logger",
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
]
