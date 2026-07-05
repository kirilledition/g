"""Shared REGENIE callback data contracts and constants."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from g import _core
    from g.compute.regenie2_binary import api as regenie2_binary

type HostGenotypeBuffer = npt.NDArray[np.float32] | npt.NDArray[np.uint8]
type HostOrDeviceFloatArray = jax.Array | npt.NDArray[np.float32]


class MultiPhenotypeGroupCallbackProtocol(typing.Protocol):
    """Callback contract required by grouped union-sample fanout delivery."""

    @property
    def native_callback_batch_size(self) -> int:
        """Return the native callback batch size configured for delivery."""
        ...

    def start(self) -> None:
        """Start callback worker resources."""
        ...

    def finish(self) -> None:
        """Drain callback worker resources."""
        ...

    def abort(self) -> None:
        """Abort callback worker resources."""
        ...

    def acquire_variant_major_dosage_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.float32]:
        """Return a variant-major dosage buffer for native delivery."""
        ...

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: typing.Any,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Consume one preprocessed variant-major dosage chunk."""
        ...


@dataclass(frozen=True)
class LinearChunkStatsArrays:
    """Native statistic arrays needed by linear variant-major compute paths.

    Attributes:
        dosage_sum: Per-variant dosage sums.
        observation_count: Per-variant non-missing observation counts.
        imputed_dosage_square_sum: Per-variant imputed dosage square sums.

    """

    dosage_sum: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    imputed_dosage_square_sum: npt.NDArray[np.float32]


@dataclass(frozen=True)
class BinaryChunkStatsArrays:
    """Native statistic arrays needed by binary variant-major compute paths.

    Attributes:
        dosage_sum: Per-variant dosage sums.
        observation_count: Per-variant non-missing observation counts.
        sparse_candidate_mask: Optional per-variant sparse Firth candidate flags.

    """

    dosage_sum: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    sparse_candidate_mask: npt.NDArray[np.bool_] | None


@dataclass(frozen=True)
class MultiPhenotypeGroupFanout:
    """One compatible phenotype group fed by a union-sample native decode.

    Attributes:
        callback: Existing multi-phenotype callback for this compatible group.
        sample_position_array: Positions of this group's samples within the union decode buffer.

    """

    callback: MultiPhenotypeGroupCallbackProtocol
    sample_position_array: npt.NDArray[np.intp]


class NativeBgenWorkerShutdownError(RuntimeError):
    """Raised when a native callback worker does not stop cleanly."""

    def __init__(self, *, worker_name: str, timeout_seconds: float) -> None:
        """Initialize a worker shutdown error."""
        self.worker_name = worker_name
        self.timeout_seconds = timeout_seconds
        message = f"native pipeline worker {worker_name!r} did not stop within {timeout_seconds:.1f} seconds"
        super().__init__(message)


@dataclass(frozen=True)
class PreprocessedDosageChunkWorkItem:
    """One native-preprocessed dosage chunk staged for asynchronous JAX compute."""

    metadata: typing.Any
    genotype_matrix: npt.NDArray[np.float32]
    chunk_stats: _core.ChunkStats


@dataclass(frozen=True)
class PreprocessedVariantMajorDosageChunkWorkItem:
    """One native-preprocessed variant-major dosage chunk staged for JAX compute."""

    metadata: typing.Any
    genotype_matrix_by_variant: npt.NDArray[np.float32]
    chunk_stats: _core.ChunkStats


@dataclass(frozen=True)
class PreprocessedVariantMajorDosageChunkBatchWorkItem:
    """Variant-major dosage chunks staged together from one native callback."""

    work_items: tuple[PreprocessedVariantMajorDosageChunkWorkItem, ...]


@dataclass(frozen=True)
class PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem:
    """One variant-major packed8 probability-pair chunk staged for JAX compute."""

    metadata: typing.Any
    packed_probability_pairs_by_variant: npt.NDArray[np.uint8]
    chunk_stats: _core.ChunkStats


@dataclass(frozen=True)
class Regenie2ResultWriteWorkItem:
    """One computed REGENIE result awaiting host materialization and output writing."""

    metadata: _core.VariantMetadata
    chunk_stats: _core.ChunkStats
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None
    host_dosage_buffer: HostGenotypeBuffer | None
    release_in_flight_slot: bool
    binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None


@dataclass(frozen=True)
class Regenie2MultiResultWriteWorkItem:
    """One computed multi-trait REGENIE result awaiting materialization and writing."""

    metadata: _core.VariantMetadata
    chunk_stats: _core.ChunkStats
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None
    host_dosage_buffer: HostGenotypeBuffer | None
    release_in_flight_slot: bool
    binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None


class NativeBgenRunInputProtocol(typing.Protocol):
    """Run input fields required by callback compute initialization."""

    @property
    def phenotype_vector(self) -> HostOrDeviceFloatArray:
        """Return the aligned phenotype vector."""
        ...

    @property
    def covariate_matrix(self) -> HostOrDeviceFloatArray:
        """Return the aligned covariate design matrix."""
        ...


class NativeBgenMultiRunInputProtocol(typing.Protocol):
    """Run input fields required by multi-phenotype callbacks."""

    phenotype_names: tuple[str, ...]
    sample_indices: npt.NDArray[np.int64]

    @property
    def phenotype_matrix(self) -> HostOrDeviceFloatArray:
        """Return the aligned trait-major phenotype matrix."""
        ...

    @property
    def covariate_matrix(self) -> HostOrDeviceFloatArray:
        """Return the aligned covariate design matrix."""
        ...


class RegeniePredictionSourceProtocol(typing.Protocol):
    """Native prediction source interface used by the JAX callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return already-aligned LOCO predictions for one chromosome."""
        ...


class MultiRegeniePredictionSourceProtocol(typing.Protocol):
    """Prediction source interface used by multi-phenotype callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return trait-major aligned LOCO predictions for one chromosome."""
        ...
