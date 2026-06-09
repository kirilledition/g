"""Native BGEN callback helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import abc
import contextlib
import logging
import queue
import threading
import time
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.engine import telemetry, timing

DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = 60.0
RESULT_WORKER_JOIN_TIMEOUT_SECONDS = 60.0
GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS = 300.0
GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS = 300.0
WORKER_ABORT_STOP_TIMEOUT_SECONDS = 1.0
logger = logging.getLogger(__name__)
type HostGenotypeBuffer = npt.NDArray[np.float32] | npt.NDArray[np.uint8]
type HostOrDeviceFloatArray = jax.Array | npt.NDArray[np.float32]

if typing.TYPE_CHECKING:
    import collections.abc


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

    callback: object
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


def block_until_ready(value: typing.Any) -> None:
    """Synchronize a JAX value when it supports readiness blocking."""
    block_until_ready_method = getattr(value, "block_until_ready", None)
    if callable(block_until_ready_method):
        block_until_ready_method()


def put_compute_array_on_device(array: HostOrDeviceFloatArray) -> jax.Array:
    """Place an aligned host/JAX input array on the active JAX device."""
    return typing.cast("jax.Array", jax.device_put(array))


def record_transfer_metadata_for_array(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    transfer_name: str,
    array_role: str,
    array: object,
) -> None:
    """Record conservative transfer size metadata when diagnostics are active."""
    if stage_timing_recorder is None:
        return
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return
    try:
        numpy_dtype = np.dtype(dtype)
    except TypeError:
        return
    element_count = 1
    ndim = 0
    for dimension in typing.cast("typing.Iterable[typing.Any]", shape):
        element_count *= int(dimension)
        ndim += 1
    stage_timing_recorder.add_transfer_metadata(
        transfer_name=transfer_name,
        array_role=array_role,
        dtype_name=numpy_dtype.name,
        ndim=ndim,
        byte_count=element_count * int(numpy_dtype.itemsize),
        element_count=element_count,
    )


def enforce_null_logistic_nonconvergence_policy(
    *,
    chromosome: str,
    null_logistic_converged: typing.Any,
    policy: types.NullLogisticNonconvergencePolicy,
    phenotype_names: tuple[str, ...] | None = None,
) -> None:
    """Raise or warn when a binary null-logistic chromosome fit did not converge."""
    convergence_flags = np.asarray(jax.device_get(null_logistic_converged), dtype=np.bool_)
    if convergence_flags.ndim == 0:
        if bool(convergence_flags):
            return
        message = f"Binary null logistic model did not converge for chromosome {chromosome}."
    else:
        failed_trait_indices = tuple(int(index) for index in np.flatnonzero(~convergence_flags))
        if not failed_trait_indices:
            return
        if phenotype_names is None:
            failed_traits = ", ".join(str(index) for index in failed_trait_indices)
        else:
            failed_traits = ", ".join(phenotype_names[index] for index in failed_trait_indices)
        message = f"Binary null logistic model did not converge for chromosome {chromosome}: {failed_traits}."
    if policy == types.NullLogisticNonconvergencePolicy.FAIL:
        raise RuntimeError(message)
    logger.warning("%s Continuing because --null_logistic_nonconvergence_policy=warn.", message)


def record_binary_chunk_diagnostics(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    result: regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult,
) -> None:
    """Record binary candidate and Firth diagnostics for one chunk."""
    if not timing.should_collect_exact_stage_timings(stage_timing_recorder):
        return
    assert stage_timing_recorder is not None
    diagnostics = jax.device_get(regenie2_binary.count_binary_chunk_diagnostics(result))
    stage_timing_recorder.add_binary_chunk_diagnostics(
        {
            "score_test_candidate_count": int(diagnostics.score_test_candidate_count),
            "firth_candidate_count": int(diagnostics.firth_candidate_count),
            "firth_iteration_min": int(diagnostics.firth_iteration_min),
            "firth_iteration_median": float(diagnostics.firth_iteration_median),
            "firth_iteration_max": int(diagnostics.firth_iteration_max),
            "firth_converged_count": int(diagnostics.firth_converged_count),
            "firth_failed_count": int(diagnostics.firth_failed_count),
            "firth_numerical_failure_count": int(diagnostics.firth_numerical_failure_count),
            "firth_max_iteration_failure_count": int(diagnostics.firth_max_iteration_failure_count),
            "firth_invalid_statistic_failure_count": int(diagnostics.firth_invalid_statistic_failure_count),
            "firth_step_halving_failure_count": int(diagnostics.firth_step_halving_failure_count),
            "pseudo_firth_attempt_count": int(diagnostics.pseudo_firth_attempt_count),
            "pseudo_firth_success_count": int(diagnostics.pseudo_firth_success_count),
            "nr_zero_start_attempt_count": int(diagnostics.nr_zero_start_attempt_count),
            "nr_zero_start_success_count": int(diagnostics.nr_zero_start_success_count),
            "nr_warm_start_attempt_count": int(diagnostics.nr_warm_start_attempt_count),
            "nr_warm_start_success_count": int(diagnostics.nr_warm_start_success_count),
            "sparse_correction_count": int(diagnostics.sparse_correction_count),
            "dense_correction_count": int(diagnostics.dense_correction_count),
        }
    )


def put_genotype_matrix_on_device(
    genotype_matrix: jax.Array | HostGenotypeBuffer,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    chunk_metadata: typing.Any | None = None,
    *,
    array_role: str = "genotype_matrix",
) -> jax.Array:
    """Transfer a genotype chunk to the active JAX device with optional timing."""
    start_time = time.perf_counter()
    genotype_device_array = jax.device_put(genotype_matrix)
    if timing.should_collect_exact_stage_timings(stage_timing_recorder):
        block_until_ready(genotype_device_array)
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="host_to_device_transfer",
        start_time=start_time,
        chunk_metadata=chunk_metadata,
    )
    record_transfer_metadata_for_array(
        stage_timing_recorder=stage_timing_recorder,
        transfer_name="host_to_device_transfer",
        array_role=array_role,
        array=genotype_matrix,
    )
    return genotype_device_array


def put_chunk_array_on_device(
    array: jax.Array | npt.NDArray[typing.Any],
    stage_timing_recorder: timing.StageTimingRecorder | None,
    chunk_metadata: typing.Any,
    *,
    array_role: str = "chunk_array",
) -> jax.Array:
    """Transfer one chunk-scoped array to the active JAX device with timing."""
    start_time = time.perf_counter()
    device_array = jax.device_put(array)
    if timing.should_collect_exact_stage_timings(stage_timing_recorder):
        block_until_ready(device_array)
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="host_to_device_transfer",
        start_time=start_time,
        chunk_metadata=chunk_metadata,
    )
    record_transfer_metadata_for_array(
        stage_timing_recorder=stage_timing_recorder,
        transfer_name="host_to_device_transfer",
        array_role=array_role,
        array=array,
    )
    return device_array


def block_compute_result_for_timing(
    *,
    result_ready_value: jax.Array,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    start_time: float,
    chunk_metadata: typing.Any | None = None,
) -> None:
    """Synchronize chunk compute only when detailed stage timings are enabled."""
    if timing.should_collect_exact_stage_timings(stage_timing_recorder):
        block_until_ready(result_ready_value)
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="jax_compute",
        start_time=start_time,
        chunk_metadata=chunk_metadata,
    )


def build_chunk_timing_identity(metadata: typing.Any) -> timing.ChunkTimingIdentity:
    """Build per-chunk timing identity fields from native metadata."""
    variant_start_index = int(metadata.variant_start_index)
    variant_stop_index = int(metadata.variant_stop_index)
    return timing.ChunkTimingIdentity(
        chunk_identifier=variant_start_index,
        chromosome=get_metadata_chromosome(metadata),
        variant_start_index=variant_start_index,
        variant_stop_index=variant_stop_index,
        variant_count=variant_stop_index - variant_start_index,
    )


def record_stage_duration_with_optional_chunk(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
    chunk_metadata: typing.Any | None = None,
) -> None:
    """Record a stage duration globally and optionally against one chunk."""
    if stage_timing_recorder is None:
        return
    if chunk_metadata is None:
        timing.record_stage_duration(stage_timing_recorder, stage_name, start_time)
        return
    timing.record_chunk_stage_duration(
        stage_timing_recorder,
        chunk_identity=build_chunk_timing_identity(chunk_metadata),
        stage_name=stage_name,
        start_time=start_time,
    )


def narrow_public_statistic_array_on_device(array: jax.Array) -> jax.Array:
    """Narrow public result statistics to the native writer dtype before host transfer."""
    return jnp.asarray(array, dtype=jnp.float32)


def select_active_trait_rows_on_device(
    array: jax.Array,
    *,
    active_trait_indices: tuple[int, ...],
    total_trait_count: int,
) -> jax.Array:
    """Return active trait rows without materializing inactive traits on host."""
    if len(active_trait_indices) == total_trait_count and active_trait_indices == tuple(range(total_trait_count)):
        return array
    active_trait_index_array = jnp.asarray(active_trait_indices, dtype=jnp.int32)
    return jnp.take(array, active_trait_index_array, axis=0)


def cast_statistic_array_for_native_writer(array: object) -> npt.NDArray[np.float32]:
    """Cast computed statistics to the public native writer schema dtype."""
    return np.asarray(array, dtype=np.float32)


def get_chunk_stats_compute_arrays(
    chunk_stats: _core.ChunkStats,
    *,
    include_imputed_dosage_square_sum: bool,
    include_sparse_firth_candidate: bool,
) -> typing.Mapping[str, object]:
    """Return compute-needed native stat arrays through the bundled binding when available."""
    compute_arrays_method = getattr(chunk_stats, "compute_arrays", None)
    if callable(compute_arrays_method):
        return typing.cast(
            "typing.Mapping[str, object]",
            compute_arrays_method(
                include_imputed_dosage_square_sum=include_imputed_dosage_square_sum,
                include_sparse_firth_candidate=include_sparse_firth_candidate,
            ),
        )
    compute_arrays: dict[str, object] = {
        "dosage_sum": chunk_stats.dosage_sum,
        "observation_count": chunk_stats.observation_count,
    }
    if include_imputed_dosage_square_sum:
        compute_arrays["imputed_dosage_square_sum"] = chunk_stats.imputed_dosage_square_sum
    if include_sparse_firth_candidate:
        compute_arrays["is_rare_sparse_firth_candidate"] = chunk_stats.is_rare_sparse_firth_candidate
    return compute_arrays


def get_linear_chunk_stats_arrays(chunk_stats: _core.ChunkStats) -> LinearChunkStatsArrays:
    """Return the native stat arrays needed by linear variant-major compute."""
    compute_arrays = get_chunk_stats_compute_arrays(
        chunk_stats,
        include_imputed_dosage_square_sum=True,
        include_sparse_firth_candidate=False,
    )
    return LinearChunkStatsArrays(
        dosage_sum=typing.cast("npt.NDArray[np.float32]", compute_arrays["dosage_sum"]),
        observation_count=typing.cast("npt.NDArray[np.int32]", compute_arrays["observation_count"]),
        imputed_dosage_square_sum=typing.cast(
            "npt.NDArray[np.float32]",
            compute_arrays["imputed_dosage_square_sum"],
        ),
    )


def get_binary_chunk_stats_arrays(
    chunk_stats: _core.ChunkStats,
    *,
    include_sparse_firth_candidate: bool,
) -> BinaryChunkStatsArrays:
    """Return the native stat arrays needed by binary variant-major compute."""
    compute_arrays = get_chunk_stats_compute_arrays(
        chunk_stats,
        include_imputed_dosage_square_sum=False,
        include_sparse_firth_candidate=include_sparse_firth_candidate,
    )
    sparse_candidate_mask: npt.NDArray[np.bool_] | None = None
    if include_sparse_firth_candidate:
        sparse_candidate_mask = typing.cast(
            "npt.NDArray[np.bool_]",
            compute_arrays["is_rare_sparse_firth_candidate"],
        )
    return BinaryChunkStatsArrays(
        dosage_sum=typing.cast("npt.NDArray[np.float32]", compute_arrays["dosage_sum"]),
        observation_count=typing.cast("npt.NDArray[np.int32]", compute_arrays["observation_count"]),
        sparse_candidate_mask=sparse_candidate_mask,
    )


def build_projected_variant_major_dosage_chunk_stats(
    genotype_matrix_by_variant: npt.NDArray[np.float32],
) -> _core.ChunkStats:
    """Build native chunk stats for a projected variant-major dosage buffer."""
    return _core.summarize_variant_major_dosage_chunk_stats(
        np.ascontiguousarray(genotype_matrix_by_variant, dtype=np.float32)
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
) -> None:
    """Write one native-metadata REGENIE chunk while timing JAX result materialization.

    The native Arrow/Parquet schema stores public result statistics as float32.
    Any higher-precision internal arrays are narrowed immediately before the
    Rust writer call.
    """
    materialization_start_time = time.perf_counter()
    beta_device_array = narrow_public_statistic_array_on_device(beta)
    standard_error_device_array = narrow_public_statistic_array_on_device(standard_error)
    chi_squared_device_array = narrow_public_statistic_array_on_device(chi_squared)
    log10_p_value_device_array = narrow_public_statistic_array_on_device(log10_p_value)
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
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="device_to_host_materialization",
        start_time=materialization_start_time,
        chunk_metadata=metadata,
    )

    write_start_time = time.perf_counter()
    writer_session.write_regenie2_native_chunk(
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=cast_statistic_array_for_native_writer(host_values["beta"]),
        standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"]),
        chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"]),
        log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"]),
        extra_code=host_values["extra_code"],
    )
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
) -> None:
    """Materialize one multi-trait result once and write missing per-trait slices."""
    chunk_identifier = int(metadata.variant_start_index)
    active_trait_indices = tuple(
        trait_index
        for trait_index, _writer_session in enumerate(writer_sessions)
        if chunk_identifier not in committed_chunk_identifier_sets[trait_index]
    )
    if not active_trait_indices:
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

    materialization_start_time = time.perf_counter()
    beta_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            beta,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        )
    )
    standard_error_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            standard_error,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        )
    )
    chi_squared_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            chi_squared,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        )
    )
    log10_p_value_device_array = narrow_public_statistic_array_on_device(
        select_active_trait_rows_on_device(
            log10_p_value,
            active_trait_indices=active_trait_indices,
            total_trait_count=total_trait_count,
        )
    )
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
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="device_to_host_materialization",
        start_time=materialization_start_time,
        chunk_metadata=metadata,
    )

    write_start_time = time.perf_counter()
    if all(isinstance(writer_session, _core.OutputWriterSession) for writer_session in writer_sessions):
        _core.write_regenie2_multi_native_chunk(
            writer_sessions=list(active_writer_sessions),
            active_trait_indices=list(range(len(active_writer_sessions))),
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(host_values["beta"]),
            standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"]),
            chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"]),
            log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"]),
            extra_code=host_values["extra_code"],
        )
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
        per_trait_write_start_time = time.perf_counter()
        extra_code_slice = None
        if host_values["extra_code"] is not None:
            extra_code_slice = host_values["extra_code"][compact_trait_index]
        writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=cast_statistic_array_for_native_writer(host_values["beta"][compact_trait_index]),
            standard_error=cast_statistic_array_for_native_writer(host_values["standard_error"][compact_trait_index]),
            chi_squared=cast_statistic_array_for_native_writer(host_values["chi_squared"][compact_trait_index]),
            log10_p_value=cast_statistic_array_for_native_writer(host_values["log10_p_value"][compact_trait_index]),
            extra_code=extra_code_slice,
        )
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=stage_timing_recorder,
            stage_name="multi_trait_output_write_per_trait",
            start_time=per_trait_write_start_time,
            chunk_metadata=metadata,
        )
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


def get_metadata_chromosome(metadata: typing.Any) -> str:
    """Return the first chromosome label from native or Python metadata."""
    chromosome_label = getattr(metadata, "chromosome_label", None)
    if chromosome_label is not None:
        return str(chromosome_label)
    return str(metadata.chromosome[0])


class GroupedMultiPhenotypeFanoutCallback:
    """Fan out one union-sample native decode to compatible phenotype-group callbacks."""

    def __init__(self, group_fanouts: tuple[MultiPhenotypeGroupFanout, ...]) -> None:
        """Initialize fanout callback state."""
        if not group_fanouts:
            message = "At least one phenotype group callback is required for fanout delivery."
            raise ValueError(message)
        self.group_fanouts = group_fanouts

    def start(self) -> None:
        """Start all group callbacks before native chunk delivery."""
        for group_fanout in self.group_fanouts:
            start_method = getattr(group_fanout.callback, "start", None)
            if callable(start_method):
                start_method()

    def finish(self) -> None:
        """Drain all group callbacks after native chunk delivery."""
        first_error: BaseException | None = None
        for group_fanout in self.group_fanouts:
            finish_method = getattr(group_fanout.callback, "finish", None)
            if not callable(finish_method):
                continue
            try:
                finish_method()
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def abort(self) -> None:
        """Abort all group callbacks after a native delivery failure."""
        for group_fanout in self.group_fanouts:
            abort_method = getattr(group_fanout.callback, "abort", None)
            if callable(abort_method):
                with contextlib.suppress(Exception):
                    abort_method()

    def acquire_variant_major_dosage_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.float32]:
        """Return a union-sample host dosage buffer for native decode."""
        return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

    def acquire_variant_major_packed8_probability_pair_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.uint8]:
        """Return a union-sample host packed8 buffer for native decode."""
        return np.empty((variant_count, sample_count, 2), dtype=np.uint8, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Slice one union-sample dosage chunk and forward it to each group callback."""
        del chunk_stats
        variant_count = int(genotype_matrix_by_variant.shape[0])
        for group_fanout in self.group_fanouts:
            group_callback = typing.cast("typing.Any", group_fanout.callback)
            group_sample_count = int(group_fanout.sample_position_array.shape[0])
            group_genotype_matrix = group_callback.acquire_variant_major_dosage_buffer(
                variant_count,
                group_sample_count,
            )
            np.take(
                genotype_matrix_by_variant,
                group_fanout.sample_position_array,
                axis=1,
                out=group_genotype_matrix,
            )
            group_chunk_stats = build_projected_variant_major_dosage_chunk_stats(group_genotype_matrix)
            group_callback.compute_preprocessed_variant_major_dosage_chunk(
                metadata,
                group_genotype_matrix,
                group_chunk_stats,
            )

    def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        self,
        metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Reject packed8 fanout until projected packed statistics are available."""
        del metadata, packed_probability_pairs_by_variant, chunk_stats
        message = "Union grouped packed8 delivery requires projected packed8 chunk statistics."
        raise RuntimeError(message)


def require_current_chromosome_state[ChromosomeStateType](
    chromosome_state: ChromosomeStateType | None,
    *,
    chromosome: str | None,
) -> ChromosomeStateType:
    """Return a prepared chromosome state or fail with an explicit runtime error."""
    if chromosome_state is not None:
        return chromosome_state
    if chromosome is None:
        message = "Chromosome state was not prepared before chunk computation."
    else:
        message = f"Chromosome state for {chromosome!r} was not prepared before chunk computation."
    raise RuntimeError(message)


class NativeBgenCallbackRunner(abc.ABC):
    """Reusable callback lifecycle for native BGEN chunk delivery."""

    def __init__(
        self,
        *,
        worker_name: str,
        staging_depth: int = 1,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize shared native callback state."""
        if staging_depth <= 0:
            message = "staging_depth must be positive."
            raise ValueError(message)
        self.processed_chunk_count = 0
        self.stage_timing_recorder = stage_timing_recorder
        self.telemetry_session = telemetry_session
        self.current_progress_chromosome: str | None = None
        self.dosage_queue_depth = staging_depth
        self.result_queue_depth = staging_depth
        self.result_in_flight_limit = self.result_queue_depth + 1
        self.dosage_buffer_limit = self.dosage_queue_depth + 1
        self.result_in_flight_slot_count = 0
        self.result_in_flight_slot_lock = threading.Lock()
        self.dosage_queue: queue.Queue[
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
            | None
        ] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.result_queue: queue.Queue[Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem | None] = (
            queue.Queue(maxsize=self.result_queue_depth)
        )
        self.result_in_flight_slots = threading.BoundedSemaphore(self.result_in_flight_limit)
        self.free_dosage_buffers: queue.Queue[HostGenotypeBuffer] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_count = 0
        self.dosage_buffer_identifiers: set[int] = set()
        self.worker_error: BaseException | None = None
        self.result_worker_error: BaseException | None = None
        self.worker_thread = threading.Thread(
            target=self.consume_dosage_chunks,
            name=worker_name,
            daemon=True,
        )
        self.result_worker_thread = threading.Thread(
            target=self.consume_result_write_items,
            name=f"{worker_name}-writer",
            daemon=True,
        )
        self.worker_start_lock = threading.Lock()
        self.worker_threads_started = False

    def start(self) -> None:
        """Start asynchronous callback workers after owner setup is complete."""
        with self.worker_start_lock:
            if self.worker_threads_started:
                return
            self.result_worker_thread.start()
            self.worker_thread.start()
            self.worker_threads_started = True

    def worker_threads_have_started(self) -> bool:
        """Return whether callback worker threads have been started."""
        return self.worker_threads_started

    def record_stage_duration(self, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage using this runner's timing recorder."""
        timing.record_stage_duration(self.stage_timing_recorder, stage_name, start_time)

    def record_chunk_stage_duration(self, metadata: typing.Any, stage_name: str, start_time: float) -> None:
        """Record a nested callback stage for a specific native chunk."""
        record_stage_duration_with_optional_chunk(
            stage_timing_recorder=self.stage_timing_recorder,
            stage_name=stage_name,
            start_time=start_time,
            chunk_metadata=metadata,
        )

    def get_stage_duration_recorder(self) -> collections.abc.Callable[[str, float], None] | None:
        """Return an optional nested stage recorder for lower-level compute helpers."""
        if self.stage_timing_recorder is None:
            return None
        return self.record_stage_duration

    def record_queue_operation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        observed_queue: queue.Queue[typing.Any],
        elapsed_seconds: float = 0.0,
        blocked_seconds: float = 0.0,
    ) -> None:
        """Record aggregate queue depth and wait metadata."""
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=queue_name,
            operation_name=operation_name,
            queue_depth=observed_queue.qsize(),
            queue_capacity=observed_queue.maxsize,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    def record_queue_stage_duration(
        self,
        *,
        queue_name: str,
        operation_name: str,
        stage_name: str,
        observed_queue: queue.Queue[typing.Any],
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a queue stage duration plus aggregate pressure metadata."""
        elapsed_seconds = time.perf_counter() - start_time
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_stage_duration(stage_name, elapsed_seconds)
        blocked_seconds = elapsed_seconds if blocked else 0.0
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=queue_name,
            operation_name=operation_name,
            queue_depth=observed_queue.qsize(),
            queue_capacity=observed_queue.maxsize,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    def record_bounded_resource_operation(
        self,
        *,
        resource_name: str,
        operation_name: str,
        current_depth: int,
        capacity: int,
        elapsed_seconds: float = 0.0,
        blocked_seconds: float = 0.0,
    ) -> None:
        """Record aggregate bounded-resource occupancy metadata."""
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            queue_depth=current_depth,
            queue_capacity=capacity,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    def record_bounded_resource_stage_duration(
        self,
        *,
        resource_name: str,
        operation_name: str,
        current_depth: int,
        capacity: int,
        stage_name: str,
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a bounded-resource stage duration plus pressure metadata."""
        elapsed_seconds = time.perf_counter() - start_time
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_stage_duration(stage_name, elapsed_seconds)
        blocked_seconds = elapsed_seconds if blocked else 0.0
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            queue_depth=current_depth,
            queue_capacity=capacity,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    @abc.abstractmethod
    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and write it."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed variant-major chunk and write it."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed packed8 chunk and write it."""
        raise NotImplementedError

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed dosage chunk for JAX association."""
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(
                PreprocessedDosageChunkWorkItem(
                    metadata=metadata,
                    genotype_matrix=genotype_matrix,
                    chunk_stats=chunk_stats,
                )
            )
        finally:
            self.record_chunk_stage_duration(metadata, "native_delivery", native_delivery_start_time)

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed variant-major dosage chunk for JAX association."""
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(
                PreprocessedVariantMajorDosageChunkWorkItem(
                    metadata=metadata,
                    genotype_matrix_by_variant=genotype_matrix_by_variant,
                    chunk_stats=chunk_stats,
                )
            )
        finally:
            self.record_chunk_stage_duration(metadata, "native_delivery", native_delivery_start_time)

    def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        self,
        metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Enqueue one Rust-preprocessed packed8 chunk for JAX association."""
        native_delivery_start_time = time.perf_counter()
        try:
            self.put_dosage_work_item(
                PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem(
                    metadata=metadata,
                    packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                    chunk_stats=chunk_stats,
                )
            )
        finally:
            self.record_chunk_stage_duration(metadata, "native_delivery", native_delivery_start_time)

    def consume_dosage_chunks(self) -> None:
        """Consume queued dosage chunks and run JAX work in order."""
        try:
            while True:
                get_start_time = time.perf_counter()
                work_item = self.dosage_queue.get()
                if work_item is None:
                    return
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="consumer_wait",
                    stage_name="callback_queue_consumer_wait",
                    observed_queue=self.dosage_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                python_callback_start_time = time.perf_counter()
                if isinstance(work_item, PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem):
                    try:
                        self.compute_preprocessed_variant_major_packed8_chunk(
                            variant_metadata=work_item.metadata,
                            packed_probability_pairs_by_variant=work_item.packed_probability_pairs_by_variant,
                            chunk_stats=work_item.chunk_stats,
                        )
                        self.processed_chunk_count += 1
                        self.record_progress(work_item.metadata)
                    finally:
                        self.record_chunk_stage_duration(
                            work_item.metadata, "python_callback", python_callback_start_time
                        )
                    continue
                if isinstance(work_item, PreprocessedVariantMajorDosageChunkWorkItem):
                    try:
                        self.compute_preprocessed_variant_major_chunk(
                            variant_metadata=work_item.metadata,
                            genotype_matrix_by_variant=work_item.genotype_matrix_by_variant,
                            chunk_stats=work_item.chunk_stats,
                        )
                        self.processed_chunk_count += 1
                        self.record_progress(work_item.metadata)
                    finally:
                        self.record_chunk_stage_duration(
                            work_item.metadata, "python_callback", python_callback_start_time
                        )
                    continue
                if isinstance(work_item, PreprocessedDosageChunkWorkItem):
                    try:
                        self.compute_preprocessed_chunk(
                            variant_metadata=work_item.metadata,
                            genotype_matrix=work_item.genotype_matrix,
                            chunk_stats=work_item.chunk_stats,
                        )
                        self.processed_chunk_count += 1
                        self.record_progress(work_item.metadata)
                    finally:
                        self.record_chunk_stage_duration(
                            work_item.metadata, "python_callback", python_callback_start_time
                        )
                    continue
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def record_progress(self, metadata: typing.Any) -> None:
        """Record throttled progress after one chunk is processed."""
        if self.telemetry_session is None:
            return
        chromosome = get_metadata_chromosome(metadata)
        if chromosome != self.current_progress_chromosome:
            if self.current_progress_chromosome is not None:
                self.telemetry_session.log_event(
                    "chromosome_completed",
                    chromosome=self.current_progress_chromosome,
                    processed_chunk_count=self.processed_chunk_count - 1,
                )
            self.current_progress_chromosome = chromosome
            self.telemetry_session.log_event(
                "chromosome_started",
                chromosome=chromosome,
                processed_chunk_count=self.processed_chunk_count,
            )
        variant_start_index = int(metadata.variant_start_index)
        variant_stop_index = int(metadata.variant_stop_index)
        self.telemetry_session.log_progress(
            processed_chunk_count=self.processed_chunk_count,
            chromosome=chromosome,
            chunk_identifier=variant_start_index,
            variant_start_index=variant_start_index,
            variant_stop_index=variant_stop_index,
            variant_count=variant_stop_index - variant_start_index,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed JAX results and write them in order."""
        try:
            while True:
                get_start_time = time.perf_counter()
                work_item = self.result_queue.get()
                if work_item is None:
                    return
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="consumer_wait",
                    stage_name="result_queue_consumer_wait",
                    observed_queue=self.result_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                try:
                    write_regenie2_native_chunk_with_optional_timing(
                        writer_session=typing.cast("typing.Any", self).writer_session,
                        metadata=work_item.metadata,
                        chunk_stats=work_item.chunk_stats,
                        beta=work_item.beta,
                        standard_error=work_item.standard_error,
                        chi_squared=work_item.chi_squared,
                        log10_p_value=work_item.log10_p_value,
                        extra_code=work_item.extra_code,
                        stage_timing_recorder=self.stage_timing_recorder,
                    )
                finally:
                    self.release_result_work_item_buffer(work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def put_dosage_work_item(
        self,
        work_item: (
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
            | None
        ),
    ) -> None:
        """Put work into the bounded worker queue while surfacing worker errors."""
        self.start()
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.dosage_queue.put(work_item, timeout=0.1)
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="put",
                    stage_name="callback_queue_put",
                    observed_queue=self.dosage_queue,
                    start_time=put_start_time,
                    blocked=False,
                )
                return
            except queue.Full:
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="producer_blocking",
                    stage_name="callback_queue_producer_blocking",
                    observed_queue=self.dosage_queue,
                    start_time=put_start_time,
                    blocked=True,
                )
                continue

    def raise_worker_error_if_present(self) -> None:
        """Raise an asynchronous worker failure on the producer thread."""
        if self.worker_error is not None:
            message = f"native pipeline callback worker failed: {self.worker_error}"
            raise RuntimeError(message) from self.worker_error
        if self.result_worker_error is not None:
            message = f"native pipeline result writer worker failed: {self.result_worker_error}"
            raise RuntimeError(message) from self.result_worker_error

    def put_result_write_item(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem | None,
    ) -> None:
        """Put a computed result into the bounded materialization/write queue."""
        self.start()
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.result_queue.put(work_item, timeout=0.1)
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="put",
                    stage_name="result_queue_put",
                    observed_queue=self.result_queue,
                    start_time=put_start_time,
                    blocked=False,
                )
                return
            except queue.Full:
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="producer_blocking",
                    stage_name="result_queue_producer_blocking",
                    observed_queue=self.result_queue,
                    start_time=put_start_time,
                    blocked=True,
                )
                continue

    def acquire_result_in_flight_slot(self) -> None:
        """Reserve capacity for one chunk of pending GPU result work."""
        while True:
            self.raise_worker_error_if_present()
            acquire_start_time = time.perf_counter()
            if self.result_in_flight_slots.acquire(timeout=0.1):
                with self.result_in_flight_slot_lock:
                    self.result_in_flight_slot_count += 1
                    current_depth = self.result_in_flight_slot_count
                self.record_bounded_resource_stage_duration(
                    resource_name="result_in_flight_slots",
                    operation_name="acquire",
                    current_depth=current_depth,
                    capacity=self.result_in_flight_limit,
                    stage_name="result_in_flight_slot_acquire",
                    start_time=acquire_start_time,
                    blocked=False,
                )
                return
            with self.result_in_flight_slot_lock:
                current_depth = self.result_in_flight_slot_count
            self.record_bounded_resource_stage_duration(
                resource_name="result_in_flight_slots",
                operation_name="producer_blocking",
                current_depth=current_depth,
                capacity=self.result_in_flight_limit,
                stage_name="result_in_flight_producer_blocking",
                start_time=acquire_start_time,
                blocked=True,
            )

    def release_result_in_flight_slot(self) -> None:
        """Release capacity for one completed chunk of GPU result work."""
        self.result_in_flight_slots.release()
        with self.result_in_flight_slot_lock:
            if self.result_in_flight_slot_count > 0:
                self.result_in_flight_slot_count -= 1
            current_depth = self.result_in_flight_slot_count
        self.record_bounded_resource_operation(
            resource_name="result_in_flight_slots",
            operation_name="release",
            current_depth=current_depth,
            capacity=self.result_in_flight_limit,
        )

    def finish(self) -> None:
        """Wait until all queued JAX work has been written."""
        self.stop_dosage_worker()
        self.join_dosage_worker(timeout_seconds=GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS)
        self.stop_result_worker()
        self.join_result_worker(timeout_seconds=GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS)
        self.raise_worker_error_if_present()
        if self.telemetry_session is not None and self.current_progress_chromosome is not None:
            self.telemetry_session.log_event(
                "chromosome_completed",
                chromosome=self.current_progress_chromosome,
                processed_chunk_count=self.processed_chunk_count,
            )
            self.current_progress_chromosome = None

    def abort(self) -> None:
        """Stop the worker after an upstream failure."""
        with contextlib.suppress(NativeBgenWorkerShutdownError):
            self.stop_dosage_worker(timeout_seconds=WORKER_ABORT_STOP_TIMEOUT_SECONDS)
        with contextlib.suppress(NativeBgenWorkerShutdownError):
            self.stop_result_worker(timeout_seconds=WORKER_ABORT_STOP_TIMEOUT_SECONDS)

    def stop_dosage_worker(self, timeout_seconds: float | None = None) -> None:
        """Signal the dosage worker to exit after queued dosage chunks drain."""
        effective_timeout_seconds = DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        if not self.worker_threads_have_started():
            return
        if self.worker_error is not None:
            return
        if not self.worker_thread.is_alive():
            return
        stop_deadline = time.monotonic() + effective_timeout_seconds
        while time.monotonic() < stop_deadline:
            if self.worker_error is not None:
                return
            if not self.worker_thread.is_alive():
                return
            current_timeout_seconds = max(0.0, min(0.1, stop_deadline - time.monotonic()))
            try:
                self.dosage_queue.put(None, timeout=current_timeout_seconds)
                return
            except queue.Full:
                continue
        raise NativeBgenWorkerShutdownError(
            worker_name=self.worker_thread.name,
            timeout_seconds=effective_timeout_seconds,
        )

    def join_dosage_worker(self, timeout_seconds: float | None = None) -> None:
        """Join the dosage worker with a bounded shutdown wait."""
        if not self.worker_threads_have_started():
            return
        effective_timeout_seconds = DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        self.worker_thread.join(timeout=effective_timeout_seconds)
        if self.worker_thread.is_alive():
            raise NativeBgenWorkerShutdownError(
                worker_name=self.worker_thread.name,
                timeout_seconds=effective_timeout_seconds,
            )

    def stop_result_worker(self, timeout_seconds: float | None = None) -> None:
        """Signal the result worker to exit after queued results drain."""
        effective_timeout_seconds = RESULT_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        if not self.worker_threads_have_started():
            return
        if self.result_worker_error is not None:
            return
        if not self.result_worker_thread.is_alive():
            return
        stop_deadline = time.monotonic() + effective_timeout_seconds
        while time.monotonic() < stop_deadline:
            if self.result_worker_error is not None:
                return
            if not self.result_worker_thread.is_alive():
                return
            current_timeout_seconds = max(0.0, min(0.1, stop_deadline - time.monotonic()))
            try:
                self.result_queue.put(None, timeout=current_timeout_seconds)
                return
            except queue.Full:
                continue
        raise NativeBgenWorkerShutdownError(
            worker_name=self.result_worker_thread.name,
            timeout_seconds=effective_timeout_seconds,
        )

    def join_result_worker(self, timeout_seconds: float | None = None) -> None:
        """Join the result writer worker with a bounded shutdown wait."""
        if not self.worker_threads_have_started():
            return
        effective_timeout_seconds = RESULT_WORKER_JOIN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        self.result_worker_thread.join(timeout=effective_timeout_seconds)
        if self.result_worker_thread.is_alive():
            raise NativeBgenWorkerShutdownError(
                worker_name=self.result_worker_thread.name,
                timeout_seconds=effective_timeout_seconds,
            )

    def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable host dosage buffer for Rust to fill."""
        expected_shape = (sample_count, variant_count)
        return typing.cast(
            "npt.NDArray[np.float32]",
            self.acquire_dosage_buffer_with_shape(expected_shape, np.float32),
        )

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable host variant-major dosage buffer for Rust to fill."""
        expected_shape = (variant_count, sample_count)
        return typing.cast(
            "npt.NDArray[np.float32]",
            self.acquire_dosage_buffer_with_shape(expected_shape, np.float32),
        )

    def acquire_variant_major_packed8_probability_pair_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.uint8]:
        """Return a reusable host variant-major packed8 probability-pair buffer."""
        expected_shape = (variant_count, sample_count, 2)
        return typing.cast(
            "npt.NDArray[np.uint8]",
            self.acquire_dosage_buffer_with_shape(expected_shape, np.uint8),
        )

    def acquire_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike = np.float32,
    ) -> HostGenotypeBuffer:
        """Return a reusable host dosage buffer with the requested shape."""
        while True:
            self.raise_worker_error_if_present()
            with contextlib.suppress(queue.Empty):
                dosage_buffer = self.free_dosage_buffers.get_nowait()
                if dosage_buffer.shape == expected_shape and dosage_buffer.dtype == dtype:
                    self.record_queue_operation(
                        queue_name="dosage_buffer_pool",
                        operation_name="reuse",
                        observed_queue=self.free_dosage_buffers,
                    )
                    return dosage_buffer
                self.discard_dosage_buffer_slot(dosage_buffer)
                if self.dosage_buffer_count < self.dosage_buffer_limit:
                    return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
                continue
            if self.dosage_buffer_count < self.dosage_buffer_limit:
                return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
            with contextlib.suppress(queue.Empty):
                buffer_wait_start_time = time.perf_counter()
                dosage_buffer = self.free_dosage_buffers.get(timeout=0.1)
                self.record_queue_stage_duration(
                    queue_name="dosage_buffer_pool",
                    operation_name="consumer_wait",
                    stage_name="dosage_buffer_pool_consumer_wait",
                    observed_queue=self.free_dosage_buffers,
                    start_time=buffer_wait_start_time,
                    blocked=True,
                )
                if dosage_buffer.shape == expected_shape and dosage_buffer.dtype == dtype:
                    self.record_queue_operation(
                        queue_name="dosage_buffer_pool",
                        operation_name="reuse",
                        observed_queue=self.free_dosage_buffers,
                    )
                    return dosage_buffer
                self.discard_dosage_buffer_slot(dosage_buffer)
                if self.dosage_buffer_count < self.dosage_buffer_limit:
                    return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)

    def release_dosage_buffer(self, dosage_buffer: HostGenotypeBuffer) -> None:
        """Return a processed host dosage buffer to the reusable pool."""
        if id(dosage_buffer) not in self.dosage_buffer_identifiers:
            return
        try:
            self.free_dosage_buffers.put_nowait(dosage_buffer)
            self.record_queue_operation(
                queue_name="dosage_buffer_pool",
                operation_name="return",
                observed_queue=self.free_dosage_buffers,
            )
        except queue.Full:
            self.record_queue_operation(
                queue_name="dosage_buffer_pool",
                operation_name="return_full",
                observed_queue=self.free_dosage_buffers,
            )
            self.discard_dosage_buffer_slot(dosage_buffer)

    def allocate_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> HostGenotypeBuffer:
        """Allocate and register one host genotype buffer slot."""
        dosage_buffer = typing.cast("HostGenotypeBuffer", np.empty(expected_shape, dtype=dtype, order="C"))
        self.dosage_buffer_count += 1
        self.dosage_buffer_identifiers.add(id(dosage_buffer))
        self.record_queue_operation(
            queue_name="dosage_buffer_pool",
            operation_name="allocate",
            observed_queue=self.free_dosage_buffers,
        )
        return dosage_buffer

    def discard_dosage_buffer_slot(self, dosage_buffer: HostGenotypeBuffer) -> None:
        """Remove one discarded host genotype buffer slot from pool accounting."""
        dosage_buffer_identifier = id(dosage_buffer)
        if dosage_buffer_identifier not in self.dosage_buffer_identifiers:
            return
        self.dosage_buffer_identifiers.remove(dosage_buffer_identifier)
        if self.dosage_buffer_count > 0:
            self.dosage_buffer_count -= 1
        self.record_queue_operation(
            queue_name="dosage_buffer_pool",
            operation_name="discard",
            observed_queue=self.free_dosage_buffers,
        )

    def release_numpy_dosage_buffer(self, dosage_buffer: jax.Array | HostGenotypeBuffer) -> None:
        """Return a NumPy host dosage buffer to the pool after device transfer."""
        if isinstance(dosage_buffer, np.ndarray):
            self.release_dosage_buffer(typing.cast("HostGenotypeBuffer", dosage_buffer))

    def get_releasable_dosage_buffer(
        self,
        dosage_buffer: jax.Array | HostGenotypeBuffer,
    ) -> HostGenotypeBuffer | None:
        """Return a host dosage buffer reference when it belongs to the reusable pool."""
        if isinstance(dosage_buffer, np.ndarray):
            return typing.cast("HostGenotypeBuffer", dosage_buffer)
        return None

    def release_result_work_item_buffer(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Release resources after a dependent JAX result is materialized."""
        if work_item.host_dosage_buffer is not None:
            self.release_dosage_buffer(work_item.host_dosage_buffer)
        if work_item.release_in_flight_slot:
            self.release_result_in_flight_slot()


class LinearRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for quantitative traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        staging_depth: int = 1,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.score_dtype = score_dtype
        self.linear_numerical_config = linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_vector = put_compute_array_on_device(run_input.phenotype_vector)
        self.regenie_state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_linear.Regenie2LinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-linear-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            result = self.compute_linear_result(variant_metadata=variant_metadata, genotype_matrix=genotype_matrix)
            self.put_result_write_item(
                Regenie2ResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=True,
                )
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            result = self.compute_linear_variant_major_result(
                variant_metadata=variant_metadata,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                chunk_stats=chunk_stats,
            )
            self.put_result_write_item(
                Regenie2ResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=True,
                )
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one packed8 chunk and enqueue its result for writing."""
        host_packed_buffer = self.get_releasable_dosage_buffer(packed_probability_pairs_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )

            packed_device_array = put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_observation_count = put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_linear_chunk_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_device_array,
                genotype_dosage_sum=genotype_dosage_sum,
                genotype_observation_count=genotype_observation_count,
                genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_linear_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
            self.release_result_in_flight_slot()
            raise

    def enqueue_linear_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_linear.Regenie2LinearChunkResult,
        host_dosage_buffer: HostGenotypeBuffer | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a linear result for materialization and writing."""
        self.put_result_write_item(
            Regenie2ResultWriteWorkItem(
                metadata=variant_metadata,
                chunk_stats=chunk_stats,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=None,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=release_in_flight_slot,
            )
        )

    def compute_linear_variant_major_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> regenie2_linear.Regenie2LinearChunkResult:
        """Compute quantitative REGENIE step 2 statistics for a variant-major chunk."""
        self.prepare_chromosome_state(variant_metadata)
        chromosome_state = require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = put_genotype_matrix_on_device(
            genotype_matrix_by_variant,
            self.stage_timing_recorder,
            variant_metadata,
        )
        linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
        genotype_dosage_sum = put_chunk_array_on_device(
            linear_chunk_stats_arrays.dosage_sum,
            self.stage_timing_recorder,
            variant_metadata,
        )
        genotype_observation_count = put_chunk_array_on_device(
            linear_chunk_stats_arrays.observation_count,
            self.stage_timing_recorder,
            variant_metadata,
        )
        genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
            linear_chunk_stats_arrays.imputed_dosage_square_sum,
            self.stage_timing_recorder,
            variant_metadata,
        )
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_device_array,
            genotype_dosage_sum=genotype_dosage_sum,
            genotype_observation_count=genotype_observation_count,
            genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
            score_dtype=self.score_dtype,
            linear_minimum_variance=self.linear_numerical_config.minimum_variance,
            linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
        )
        return result

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached linear chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
            self.regenie_state,
            loco_predictions,
            self.score_dtype,
        )
        chromosome_ready_value = getattr(
            self.current_chromosome_state,
            "adjusted_residual",
            self.current_chromosome_state,
        )
        block_until_ready(chromosome_ready_value)
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def compute_linear_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
    ) -> regenie2_linear.Regenie2LinearChunkResult:
        """Compute quantitative REGENIE step 2 statistics for one chunk."""
        self.prepare_chromosome_state(variant_metadata)
        chromosome_state = require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = put_genotype_matrix_on_device(
            genotype_matrix,
            self.stage_timing_recorder,
            variant_metadata,
        )
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_device_array,
            score_dtype=self.score_dtype,
            linear_minimum_variance=self.linear_numerical_config.minimum_variance,
            linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
        )
        return result


class MultiLinearRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback for quantitative multi-phenotype REGENIE step 2."""

    def __init__(
        self,
        run_input: NativeBgenMultiRunInputProtocol,
        prediction_source: MultiRegeniePredictionSourceProtocol,
        writer_sessions: tuple[typing.Any, ...],
        committed_chunk_identifier_sets: tuple[set[int], ...],
        staging_depth: int = 1,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.committed_chunk_identifier_sets = committed_chunk_identifier_sets
        self.score_dtype = score_dtype
        self.linear_numerical_config = linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_matrix = put_compute_array_on_device(run_input.phenotype_matrix)
        self.regenie_state = regenie2_linear.prepare_regenie2_multi_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_matrix=phenotype_matrix,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_linear.Regenie2MultiLinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-multi-linear-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
        try:
            while True:
                get_start_time = time.perf_counter()
                work_item = self.result_queue.get()
                if work_item is None:
                    return
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="consumer_wait",
                    stage_name="result_queue_consumer_wait",
                    observed_queue=self.result_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                multi_work_item = typing.cast("Regenie2MultiResultWriteWorkItem", work_item)
                try:
                    write_regenie2_multi_native_chunk_with_optional_timing(
                        writer_sessions=self.writer_sessions,
                        committed_chunk_identifier_sets=self.committed_chunk_identifier_sets,
                        metadata=multi_work_item.metadata,
                        chunk_stats=multi_work_item.chunk_stats,
                        beta=multi_work_item.beta,
                        standard_error=multi_work_item.standard_error,
                        chi_squared=multi_work_item.chi_squared,
                        log10_p_value=multi_work_item.log10_p_value,
                        extra_code=multi_work_item.extra_code,
                        stage_timing_recorder=self.stage_timing_recorder,
                    )
                finally:
                    self.release_result_work_item_buffer(multi_work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one sample-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
                chromosome_state=chromosome_state,
                genotype_matrix=genotype_device_array,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_observation_count = put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_device_array,
                genotype_dosage_sum=genotype_dosage_sum,
                genotype_observation_count=genotype_observation_count,
                genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one packed8 chunk and enqueue multi-trait results."""
        host_packed_buffer = self.get_releasable_dosage_buffer(packed_probability_pairs_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            packed_device_array = put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_observation_count = put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_multi_linear_chunk_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_device_array,
                genotype_dosage_sum=genotype_dosage_sum,
                genotype_observation_count=genotype_observation_count,
                genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
            self.release_result_in_flight_slot()
            raise

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached multi-linear chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_linear.prepare_regenie2_multi_linear_chromosome_state(
            self.regenie_state,
            loco_predictions,
            self.score_dtype,
        )
        block_until_ready(self.current_chromosome_state.adjusted_residual_matrix)
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def enqueue_multi_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_linear.Regenie2MultiLinearChunkResult,
        host_dosage_buffer: HostGenotypeBuffer | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a multi-linear result for materialization and writing."""
        self.put_result_write_item(
            typing.cast(
                "Regenie2ResultWriteWorkItem",
                Regenie2MultiResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=release_in_flight_slot,
                ),
            )
        )


class BinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for binary traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
        null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
            types.NullLogisticNonconvergencePolicy.FAIL
        ),
        staging_depth: int = 1,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.null_logistic_nonconvergence_policy = null_logistic_nonconvergence_policy
        self.score_dtype = score_dtype
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_vector = put_compute_array_on_device(run_input.phenotype_vector)
        self.regenie_state = regenie2_binary.prepare_regenie2_binary_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary.Regenie2BinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-binary-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            result = self.compute_binary_result(
                variant_metadata=variant_metadata,
                genotype_matrix=genotype_matrix,
                sparse_candidate_mask=sparse_candidate_mask,
            )
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def enqueue_binary_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: (regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult),
        host_dosage_buffer: HostGenotypeBuffer | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a binary result for materialization and writing."""
        self.put_result_write_item(
            Regenie2ResultWriteWorkItem(
                metadata=variant_metadata,
                chunk_stats=chunk_stats,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=result.extra_code,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=release_in_flight_slot,
            )
        )

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached binary chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
            state=self.regenie_state,
            loco_offset=loco_offset,
            correction_plan=self.correction_plan,
            kernel_config=self.kernel_config,
            score_dtype=self.score_dtype,
        )
        chromosome_ready_value = getattr(
            self.current_chromosome_state,
            "score_residual",
            self.current_chromosome_state,
        )
        block_until_ready(chromosome_ready_value)
        enforce_null_logistic_nonconvergence_policy(
            chromosome=chromosome,
            null_logistic_converged=self.current_chromosome_state.null_logistic_converged,
            policy=self.null_logistic_nonconvergence_policy,
        )
        if self.stage_timing_recorder is not None:
            self.stage_timing_recorder.add_null_logistic_diagnostics(
                {
                    "chromosome": chromosome,
                    "iteration_count": int(jax.device_get(self.current_chromosome_state.null_logistic_iteration_count)),
                    "converged": int(jax.device_get(self.current_chromosome_state.null_logistic_converged)),
                    "firth_iteration_count": int(
                        jax.device_get(self.current_chromosome_state.null_firth_iteration_count)
                    ),
                    "firth_convergence_reason_code": int(
                        jax.device_get(self.current_chromosome_state.null_firth_convergence_reason_code)
                    ),
                    "correction_method": self.correction_plan.method.value,
                }
            )
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def compute_binary_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        sparse_candidate_mask: jax.Array | None = None,
    ) -> regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult:
        """Compute binary REGENIE step 2 statistics for one chunk."""
        self.prepare_chromosome_state(variant_metadata)
        chromosome_state = require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = put_genotype_matrix_on_device(
            genotype_matrix,
            self.stage_timing_recorder,
            variant_metadata,
        )
        compute_start_time = time.perf_counter()
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_device_array,
            correction_plan=self.correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=self.kernel_config,
            score_dtype=self.score_dtype,
            stage_duration_recorder=self.get_stage_duration_recorder(),
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
        )
        record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
        return result

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major chunk and enqueue its result for writing."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )

            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_binary_score_test_variant_major_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one packed8 chunk and enqueue its result for writing."""
        host_packed_buffer = self.get_releasable_dosage_buffer(packed_probability_pairs_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )

            packed_device_array = put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_binary_score_test_packed8_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_packed8(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
            self.release_result_in_flight_slot()
            raise


class MultiBinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback for binary multi-phenotype REGENIE step 2."""

    def __init__(
        self,
        run_input: NativeBgenMultiRunInputProtocol,
        prediction_source: MultiRegeniePredictionSourceProtocol,
        writer_sessions: tuple[typing.Any, ...],
        committed_chunk_identifier_sets: tuple[set[int], ...],
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
        null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
            types.NullLogisticNonconvergencePolicy.FAIL
        ),
        staging_depth: int = 1,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.committed_chunk_identifier_sets = committed_chunk_identifier_sets
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.null_logistic_nonconvergence_policy = null_logistic_nonconvergence_policy
        self.score_dtype = score_dtype
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_matrix = put_compute_array_on_device(run_input.phenotype_matrix)
        self.regenie_state = regenie2_binary.prepare_regenie2_multi_binary_state(
            covariate_matrix=covariate_matrix,
            phenotype_matrix=phenotype_matrix,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary.Regenie2MultiBinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-multi-binary-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
        try:
            while True:
                get_start_time = time.perf_counter()
                work_item = self.result_queue.get()
                if work_item is None:
                    return
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="consumer_wait",
                    stage_name="result_queue_consumer_wait",
                    observed_queue=self.result_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                multi_work_item = typing.cast("Regenie2MultiResultWriteWorkItem", work_item)
                try:
                    write_regenie2_multi_native_chunk_with_optional_timing(
                        writer_sessions=self.writer_sessions,
                        committed_chunk_identifier_sets=self.committed_chunk_identifier_sets,
                        metadata=multi_work_item.metadata,
                        chunk_stats=multi_work_item.chunk_stats,
                        beta=multi_work_item.beta,
                        standard_error=multi_work_item.standard_error,
                        chi_squared=multi_work_item.chi_squared,
                        log10_p_value=multi_work_item.log10_p_value,
                        extra_code=multi_work_item.extra_code,
                        stage_timing_recorder=self.stage_timing_recorder,
                    )
                finally:
                    self.release_result_work_item_buffer(multi_work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one sample-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix,
                self.stage_timing_recorder,
                variant_metadata,
            )
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state(
                chromosome_state=chromosome_state,
                genotype_matrix=genotype_device_array,
                correction_plan=self.correction_plan,
                sparse_candidate_mask=sparse_candidate_mask,
                kernel_config=self.kernel_config,
                score_dtype=self.score_dtype,
                stage_duration_recorder=self.get_stage_duration_recorder(),
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one variant-major Rust-preprocessed chunk and enqueue multi-trait results."""
        host_dosage_buffer = self.get_releasable_dosage_buffer(genotype_matrix_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_multi_binary_score_test_variant_major_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one packed8 chunk and enqueue multi-trait binary results."""
        host_packed_buffer = self.get_releasable_dosage_buffer(packed_probability_pairs_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            packed_device_array = put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_multi_binary_score_test_packed8_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
            self.release_result_in_flight_slot()
            raise

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached multi-binary chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_binary.prepare_regenie2_multi_binary_chromosome_state(
            self.regenie_state,
            loco_offset,
            self.correction_plan,
            self.kernel_config,
            self.score_dtype,
        )
        block_until_ready(self.current_chromosome_state.score_residual)
        enforce_null_logistic_nonconvergence_policy(
            chromosome=chromosome,
            null_logistic_converged=self.current_chromosome_state.null_logistic_converged,
            policy=self.null_logistic_nonconvergence_policy,
            phenotype_names=self.run_input.phenotype_names,
        )
        if self.stage_timing_recorder is not None:
            iteration_counts = jax.device_get(self.current_chromosome_state.null_logistic_iteration_count)
            convergence_flags = jax.device_get(self.current_chromosome_state.null_logistic_converged)
            for trait_index, phenotype_name in enumerate(self.run_input.phenotype_names):
                self.stage_timing_recorder.add_null_logistic_diagnostics(
                    {
                        "chromosome": chromosome,
                        "phenotype": phenotype_name,
                        "iteration_count": int(iteration_counts[trait_index]),
                        "converged": int(convergence_flags[trait_index]),
                        "correction_method": self.correction_plan.method.value,
                    }
                )
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def enqueue_multi_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_binary.Regenie2MultiBinaryScoreChunkResult | regenie2_binary.Regenie2MultiBinaryChunkResult,
        host_dosage_buffer: HostGenotypeBuffer | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a multi-binary result for materialization and writing."""
        self.put_result_write_item(
            typing.cast(
                "Regenie2ResultWriteWorkItem",
                Regenie2MultiResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=result.extra_code,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=release_in_flight_slot,
                ),
            )
        )
