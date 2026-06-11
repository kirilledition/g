from __future__ import annotations

import concurrent.futures
import dataclasses
import queue
import threading
import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from g import execution_plan, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.compute.regenie2_linear import result as regenie2_linear_result
from g.compute.regenie2_linear import state as regenie2_linear_state
from g.engine import callbacks, native_dispatch, regenie2_pipeline, shutdown, timing
from g.interface import config as interface_config
from g.io import output, source


def build_default_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build the packaged-default kernel config for tests."""
    return execution_plan.build_binary_kernel_config(interface_config.load_packaged_config().g_compute)


@dataclasses.dataclass(frozen=True)
class PipelineRuntimeOptions:
    """Runtime options that pipeline tests pass explicitly."""

    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    parquet_compression: types.ParquetCompression
    bgen_decode_tile_variant_count: int
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype


def build_default_pipeline_runtime_options() -> PipelineRuntimeOptions:
    """Build default runtime options through the public config boundary."""
    packaged_config = interface_config.load_packaged_config()
    compute_config = packaged_config.g_compute
    output_config = packaged_config.g_output
    return PipelineRuntimeOptions(
        writer_thread_count=output_config.writer_threads,
        writer_queue_depth=output_config.writer_queue_depth,
        chunks_per_arrow_file=output_config.chunks_per_arrow_file,
        parquet_compression=output_config.parquet_compression,
        bgen_decode_tile_variant_count=compute_config.bgen_decode_tile_variant_count,
        score_dtype=compute_config.score_dtype,
        firth_dtype=compute_config.firth_dtype,
    )


def write_test_run_manifest(output_run_paths: output.OutputRunPaths, header: dict[str, object]) -> bytes:
    """Write a minimal run manifest and return its bytes."""
    output.write_run_manifest(output_run_paths, {**header, "committed_chunks": []})
    return output.get_run_manifest_path(output_run_paths).read_bytes()


def test_build_phenotype_compute_groups_distinguishes_sample_modes() -> None:
    per_phenotype_groups = execution_plan.build_phenotype_compute_groups(
        phenotype_names=("trait_a", "trait_b"),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
    )
    complete_case_groups = execution_plan.build_phenotype_compute_groups(
        phenotype_names=("trait_a", "trait_b"),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )
    single_phenotype_groups = execution_plan.build_phenotype_compute_groups(
        phenotype_names=("trait_a",),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )

    assert tuple(group.phenotype_indices for group in per_phenotype_groups) == ((0,), (1,))
    assert tuple(group.group_mode for group in per_phenotype_groups) == (
        types.PhenotypeComputeGroupMode.PER_PHENOTYPE_COMPATIBLE,
        types.PhenotypeComputeGroupMode.PER_PHENOTYPE_COMPATIBLE,
    )
    assert len(complete_case_groups) == 1
    assert complete_case_groups[0].phenotype_indices == (0, 1)
    assert complete_case_groups[0].group_mode == types.PhenotypeComputeGroupMode.COMPLETE_CASE
    assert single_phenotype_groups[0].group_mode == types.PhenotypeComputeGroupMode.SINGLE_PHENOTYPE


class FakePredictionSource:
    instances: typing.ClassVar[list[FakePredictionSource]] = []

    def __init__(
        self,
        prediction_list_path: str | None = None,
        phenotype_name: str | None = None,
        sample_family_identifiers: list[str] | None = None,
        sample_individual_identifiers: list[str] | None = None,
        sample_key_mode: str = "iid",
    ) -> None:
        self.prediction_list_path = prediction_list_path
        self.phenotype_name = phenotype_name
        self.sample_family_identifiers = sample_family_identifiers
        self.sample_individual_identifiers = sample_individual_identifiers
        self.sample_key_mode = sample_key_mode
        self.native_aligned_sample_data: object | None = None
        FakePredictionSource.instances.append(self)

    @staticmethod
    def from_native_aligned_sample_data(
        prediction_list_path: str,
        phenotype_name: str,
        aligned_sample_data: object,
        sample_key_mode: str = "iid",
    ) -> FakePredictionSource:
        prediction_source = FakePredictionSource(
            prediction_list_path,
            phenotype_name,
            sample_key_mode=sample_key_mode,
        )
        prediction_source.native_aligned_sample_data = aligned_sample_data
        return prediction_source

    @staticmethod
    def from_native_multi_aligned_sample_data(
        prediction_list_path: str,
        aligned_sample_data: object,
        sample_key_mode: str = "iid",
    ) -> FakePredictionSource:
        prediction_source = FakePredictionSource(
            prediction_list_path=prediction_list_path,
            sample_key_mode=sample_key_mode,
        )
        prediction_source.native_aligned_sample_data = aligned_sample_data
        return prediction_source

    @staticmethod
    def from_native_grouped_aligned_sample_data(
        prediction_list_path: str,
        grouped_aligned_sample_data: object,
        sample_key_mode: str = "iid",
    ) -> list[FakePredictionSource]:
        return [
            FakePredictionSource.from_native_multi_aligned_sample_data(
                prediction_list_path,
                native_group.aligned_sample_data,
                sample_key_mode=sample_key_mode,
            )
            for native_group in typing.cast("typing.Any", grouped_aligned_sample_data).groups
        ]

    def get_chromosome_predictions(self, chromosome: str) -> np.ndarray:
        del chromosome
        return np.asarray([0.0, 0.0], dtype=np.float32)


class FakeWriterSession:
    def __init__(self) -> None:
        self.finished = False
        self.aborted = False
        self.interrupted_signal_name: str | None = None
        self.native_chunks: list[dict[str, object]] = []

    def write_regenie2_native_chunk(self, **kwargs: object) -> None:
        self.native_chunks.append(kwargs)

    def finish(self) -> str:
        self.finished = True
        return "results/final.parquet"

    def finish_interrupted(self, signal_name: str) -> None:
        self.interrupted_signal_name = signal_name

    def abort(self) -> None:
        self.aborted = True


class RecordingTelemetrySession:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.progress_events: list[dict[str, object]] = []

    def log_event(self, event_name: str, **fields: object) -> None:
        self.events.append((event_name, fields))

    def log_progress(self, **fields: object) -> None:
        self.progress_events.append(fields)


class NoFinalWriterSession:
    def __init__(self) -> None:
        self.finished = False
        self.aborted = False

    def finish(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted = True


def test_require_current_chromosome_state_returns_prepared_state() -> None:
    chromosome_state = object()

    resolved_state = callbacks.require_current_chromosome_state(chromosome_state, chromosome="chr22")

    assert resolved_state is chromosome_state


def test_require_current_chromosome_state_raises_clear_error_when_missing() -> None:
    with pytest.raises(RuntimeError, match="Chromosome state for 'chr22' was not prepared"):
        callbacks.require_current_chromosome_state(None, chromosome="chr22")


def test_cast_statistic_array_for_native_writer_uses_public_float32_schema() -> None:
    precise_values = np.asarray([1.0, 1.0 + 2.0**-30], dtype=np.float64)

    writer_values = callbacks.cast_statistic_array_for_native_writer(precise_values)

    assert writer_values.dtype == np.float32
    np.testing.assert_array_equal(writer_values, precise_values.astype(np.float32))


def test_write_regenie2_native_chunk_downcasts_float64_statistics_before_writing() -> None:
    writer_session = FakeWriterSession()
    precise_values = np.asarray([1.0, 1.0 + 2.0**-30], dtype=np.float64)
    extra_code = np.asarray([0, 3], dtype=np.int32)

    callbacks.write_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=typing.cast("typing.Any", SimpleNamespace()),
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=typing.cast("typing.Any", precise_values),
        standard_error=typing.cast("typing.Any", precise_values + 1.0),
        chi_squared=typing.cast("typing.Any", precise_values + 2.0),
        log10_p_value=typing.cast("typing.Any", precise_values + 3.0),
        extra_code=typing.cast("typing.Any", extra_code),
        stage_timing_recorder=None,
    )

    written_chunk = writer_session.native_chunks[0]
    beta = written_chunk["beta"]
    standard_error = written_chunk["standard_error"]
    chi_squared = written_chunk["chi_squared"]
    log10_p_value = written_chunk["log10_p_value"]
    assert isinstance(beta, np.ndarray)
    assert isinstance(standard_error, np.ndarray)
    assert isinstance(chi_squared, np.ndarray)
    assert isinstance(log10_p_value, np.ndarray)
    assert beta.dtype == np.float32
    assert standard_error.dtype == np.float32
    assert chi_squared.dtype == np.float32
    assert log10_p_value.dtype == np.float32
    np.testing.assert_array_equal(written_chunk["extra_code"], extra_code)


def test_finish_writer_sessions_uses_bounded_concurrent_pool() -> None:
    release_finish = threading.Event()
    started_finishes: queue.Queue[str] = queue.Queue()
    active_lock = threading.Lock()
    active_finish_count = 0
    maximum_active_finish_count = 0

    class BlockingWriterSession:
        def __init__(self, name: str) -> None:
            self.name = name

        def finish(self) -> str:
            nonlocal active_finish_count, maximum_active_finish_count
            with active_lock:
                active_finish_count += 1
                maximum_active_finish_count = max(maximum_active_finish_count, active_finish_count)
            started_finishes.put(self.name)
            release_finish.wait(timeout=5.0)
            with active_lock:
                active_finish_count -= 1
            return f"results/{self.name}.parquet"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        finish_future = executor.submit(
            native_dispatch.finish_writer_sessions,
            writer_sessions=(
                BlockingWriterSession("trait-a"),
                BlockingWriterSession("trait-b"),
                BlockingWriterSession("trait-c"),
            ),
            writer_finish_thread_count=2,
            stage_timing_recorder=None,
        )
        first_started = started_finishes.get(timeout=2.0)
        second_started = started_finishes.get(timeout=2.0)

        assert {first_started, second_started} == {"trait-a", "trait-b"}
        assert not finish_future.done()
        assert maximum_active_finish_count == 2

        release_finish.set()
        final_parquet_paths = finish_future.result(timeout=5.0)

    assert final_parquet_paths == (
        Path("results/trait-a.parquet"),
        Path("results/trait-b.parquet"),
        Path("results/trait-c.parquet"),
    )
    assert maximum_active_finish_count == 2


def test_write_regenie2_native_chunk_records_per_chunk_output_timing() -> None:
    writer_session = FakeWriterSession()
    stage_timing_recorder = timing.StageTimingRecorder()
    metadata = build_native_metadata()

    callbacks.write_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=metadata,
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=None,
        stage_timing_recorder=stage_timing_recorder,
    )

    snapshot = stage_timing_recorder.snapshot()
    chunk_stage_names = tuple(chunk_timing.stage_name for chunk_timing in snapshot.chunk_stage_timings)
    assert chunk_stage_names == (
        "device_to_host_materialization",
        "output_write",
        "single_trait_output_write",
    )
    assert all(
        chunk_timing.chunk_identifier == metadata.variant_start_index for chunk_timing in snapshot.chunk_stage_timings
    )
    assert all(chunk_timing.variant_count == 2 for chunk_timing in snapshot.chunk_stage_timings)


def test_write_regenie2_multi_native_chunk_skips_committed_traits_and_slices_extra_code() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())
    extra_code = jnp.asarray(
        [
            [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
            [types.BinaryExtraCode.TEST_FAIL.value, types.BinaryExtraCode.SCORE.value],
        ],
        dtype=jnp.int32,
    )

    callbacks.write_regenie2_multi_native_chunk_with_optional_timing(
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), {metadata.variant_start_index}),
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[1.1, 1.2], [1.3, 1.4]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[2.1, 2.2], [2.3, 2.4]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[3.1, 3.2], [3.3, 3.4]], dtype=jnp.float32),
        extra_code=extra_code,
        stage_timing_recorder=None,
    )

    assert len(writer_sessions[0].native_chunks) == 1
    assert not writer_sessions[1].native_chunks
    written_chunk = writer_sessions[0].native_chunks[0]
    np.testing.assert_array_equal(written_chunk["extra_code"], np.asarray(extra_code[0]))
    np.testing.assert_array_equal(written_chunk["beta"], np.asarray([0.1, 0.2], dtype=np.float32))


def test_write_regenie2_multi_native_chunk_materializes_only_active_trait_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())
    materialized_shapes: list[tuple[int, ...] | None] = []

    def recording_device_get(value: object) -> object:
        device_values = typing.cast("dict[str, object]", value)
        host_values: dict[str, object] = {}
        for key, device_value in device_values.items():
            if device_value is None:
                materialized_shapes.append(None)
                host_values[key] = None
                continue
            device_array = typing.cast("typing.Any", device_value)
            materialized_shapes.append(tuple(int(dimension) for dimension in device_array.shape))
            host_values[key] = np.asarray(device_array)
        return host_values

    monkeypatch.setattr(callbacks.jax, "device_get", recording_device_get)

    callbacks.write_regenie2_multi_native_chunk_with_optional_timing(
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=({metadata.variant_start_index}, set()),
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[1.1, 1.2], [1.3, 1.4]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[2.1, 2.2], [2.3, 2.4]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[3.1, 3.2], [3.3, 3.4]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
                [types.BinaryExtraCode.TEST_FAIL.value, types.BinaryExtraCode.SCORE.value],
            ],
            dtype=jnp.int32,
        ),
        stage_timing_recorder=None,
    )

    assert materialized_shapes == [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
    assert not writer_sessions[0].native_chunks
    assert len(writer_sessions[1].native_chunks) == 1
    written_chunk = writer_sessions[1].native_chunks[0]
    np.testing.assert_array_equal(written_chunk["beta"], np.asarray([0.3, 0.4], dtype=np.float32))
    np.testing.assert_array_equal(
        written_chunk["extra_code"],
        np.asarray([types.BinaryExtraCode.TEST_FAIL.value, types.BinaryExtraCode.SCORE.value], dtype=np.int32),
    )


def test_write_regenie2_multi_native_chunk_skips_device_get_when_all_traits_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())

    def fail_device_get(value: object) -> object:
        del value
        raise AssertionError("device_get should not run when all trait chunks are committed")

    monkeypatch.setattr(callbacks.jax, "device_get", fail_device_get)

    callbacks.write_regenie2_multi_native_chunk_with_optional_timing(
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(
            {metadata.variant_start_index},
            {metadata.variant_start_index},
        ),
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[1.1, 1.2], [1.3, 1.4]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[2.1, 2.2], [2.3, 2.4]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[3.1, 3.2], [3.3, 3.4]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
                [types.BinaryExtraCode.TEST_FAIL.value, types.BinaryExtraCode.SCORE.value],
            ],
            dtype=jnp.int32,
        ),
        stage_timing_recorder=None,
    )

    assert not writer_sessions[0].native_chunks
    assert not writer_sessions[1].native_chunks


def test_chunk_stats_helpers_use_bundled_compute_arrays_with_path_specific_fields() -> None:
    linear_chunk_stats = BundledChunkStats()
    binary_chunk_stats = BundledChunkStats()

    linear_arrays = callbacks.get_linear_chunk_stats_arrays(typing.cast("typing.Any", linear_chunk_stats))
    binary_arrays = callbacks.get_binary_chunk_stats_arrays(
        typing.cast("typing.Any", binary_chunk_stats),
        include_sparse_firth_candidate=True,
    )

    np.testing.assert_array_equal(linear_arrays.dosage_sum, np.asarray([3.0, 7.0], dtype=np.float32))
    np.testing.assert_array_equal(linear_arrays.observation_count, np.asarray([2, 2], dtype=np.int32))
    np.testing.assert_array_equal(linear_arrays.imputed_dosage_square_sum, np.asarray([5.0, 13.0], dtype=np.float32))
    np.testing.assert_array_equal(binary_arrays.dosage_sum, np.asarray([3.0, 7.0], dtype=np.float32))
    np.testing.assert_array_equal(binary_arrays.observation_count, np.asarray([2, 2], dtype=np.int32))
    np.testing.assert_array_equal(binary_arrays.sparse_candidate_mask, np.asarray([True, False], dtype=np.bool_))
    assert linear_chunk_stats.requests == [
        {
            "include_imputed_dosage_square_sum": True,
            "include_sparse_firth_candidate": False,
        }
    ]
    assert binary_chunk_stats.requests == [
        {
            "include_imputed_dosage_square_sum": False,
            "include_sparse_firth_candidate": True,
        }
    ]


def test_binary_chunk_diagnostics_are_detailed_only_for_exact_timing() -> None:
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.zeros(2, dtype=jnp.float32),
        standard_error=jnp.ones(2, dtype=jnp.float32),
        chi_squared=jnp.zeros(2, dtype=jnp.float32),
        log10_p_value=jnp.zeros(2, dtype=jnp.float32),
        extra_code=jnp.asarray([types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value], dtype=jnp.int32),
        valid_mask=jnp.asarray([True, True]),
    )
    aggregate_recorder = timing.StageTimingRecorder()
    exact_recorder = timing.StageTimingRecorder(exact_stage_timings=True)
    diagnostics = SimpleNamespace(
        score_test_candidate_count=2,
        firth_candidate_count=1,
        firth_iteration_min=1,
        firth_iteration_median=1,
        firth_iteration_max=1,
        firth_converged_count=1,
        firth_failed_count=0,
        firth_numerical_failure_count=0,
        firth_max_iteration_failure_count=0,
        firth_invalid_statistic_failure_count=0,
        firth_step_halving_failure_count=0,
        pseudo_firth_attempt_count=0,
        pseudo_firth_success_count=0,
        nr_zero_start_attempt_count=0,
        nr_zero_start_success_count=0,
        nr_warm_start_attempt_count=0,
        nr_warm_start_success_count=0,
        sparse_correction_count=0,
        dense_correction_count=1,
    )

    with patch("g.compute.regenie2_binary.api.count_binary_chunk_diagnostics", return_value=diagnostics) as mock_count:
        callbacks.record_binary_chunk_diagnostics(stage_timing_recorder=aggregate_recorder, result=result)
        callbacks.record_binary_chunk_diagnostics(stage_timing_recorder=exact_recorder, result=result)

    mock_count.assert_called_once_with(result)
    assert aggregate_recorder.snapshot().binary_chunk_diagnostics == ()
    assert exact_recorder.snapshot().binary_chunk_diagnostics == (
        {
            "score_test_candidate_count": 2,
            "firth_candidate_count": 1,
            "firth_iteration_min": 1,
            "firth_iteration_median": 1.0,
            "firth_iteration_max": 1,
            "firth_converged_count": 1,
            "firth_failed_count": 0,
            "firth_numerical_failure_count": 0,
            "firth_max_iteration_failure_count": 0,
            "firth_invalid_statistic_failure_count": 0,
            "firth_step_halving_failure_count": 0,
            "pseudo_firth_attempt_count": 0,
            "pseudo_firth_success_count": 0,
            "nr_zero_start_attempt_count": 0,
            "nr_zero_start_success_count": 0,
            "nr_warm_start_attempt_count": 0,
            "nr_warm_start_success_count": 0,
            "sparse_correction_count": 0,
            "dense_correction_count": 1,
        },
    )


def test_binary_compute_preprocessed_chunk_defers_diagnostics_until_worker_consumption() -> None:
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
        stage_timing_recorder=timing.StageTimingRecorder(),
    )
    chunk_stats = typing.cast("typing.Any", SparseOnlyChunkStats())
    chromosome_state = build_binary_chromosome_state()
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True]),
    )
    variant_metadata = build_native_metadata()
    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state",
            return_value=result,
        ) as _,
        patch.object(callback, "enqueue_binary_result_for_write") as mock_enqueue,
        patch("g.compute.regenie2_binary.api.count_binary_chunk_diagnostics") as mock_count,
    ):
        callback.compute_preprocessed_chunk(
            variant_metadata=variant_metadata,
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )

    mock_count.assert_not_called()
    mock_enqueue.assert_called_once()
    assert mock_enqueue.call_args.kwargs["binary_chunk_diagnostics"] is None


def test_binary_compute_preprocessed_chunk_collects_diagnostics_only_for_exact_timing() -> None:
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
        stage_timing_recorder=timing.StageTimingRecorder(exact_stage_timings=True),
    )
    chunk_stats = typing.cast("typing.Any", SparseOnlyChunkStats())
    chromosome_state = build_binary_chromosome_state()
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True]),
    )
    variant_metadata = build_native_metadata()
    diagnostics = SimpleNamespace(score_test_candidate_count=2, firth_candidate_count=0, firth_iteration_min=0)
    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state",
            return_value=result,
        ) as _,
        patch.object(callback, "enqueue_binary_result_for_write") as mock_enqueue,
        patch(
            "g.compute.regenie2_binary.api.count_binary_chunk_diagnostics",
            return_value=diagnostics,
        ) as mock_count,
    ):
        callback.compute_preprocessed_chunk(
            variant_metadata=variant_metadata,
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )

    mock_count.assert_called_once_with(result)
    mock_enqueue.assert_called_once()
    assert mock_enqueue.call_args.kwargs["binary_chunk_diagnostics"] is diagnostics


def test_binary_result_worker_records_deferred_diagnostics_from_work_item() -> None:
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )
    callback.result_queue = queue.Queue(maxsize=2)
    diagnostics = typing.cast(
        "regenie2_binary.BinaryChunkDiagnostics",
        SimpleNamespace(score_test_candidate_count=2),
    )
    work_item = callbacks.Regenie2ResultWriteWorkItem(
        metadata=build_native_metadata(),
        chunk_stats=typing.cast("typing.Any", ExplodingChunkStats()),
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=jnp.asarray([types.BinaryExtraCode.SCORE.value], dtype=jnp.int32),
        host_dosage_buffer=None,
        release_in_flight_slot=False,
        binary_chunk_diagnostics=diagnostics,
    )
    callback.result_queue.put_nowait(work_item)
    callback.result_queue.put_nowait(None)
    with (
        patch(
            "g.engine.callbacks.runtime.write_regenie2_native_chunk_with_optional_timing",
        ) as mock_write,
        patch("g.engine.callbacks.runtime.record_binary_chunk_diagnostics_from_count") as mock_record,
    ):
        callback.consume_result_write_items()

    mock_write.assert_called_once()
    mock_record.assert_called_once_with(
        stage_timing_recorder=None,
        diagnostics=diagnostics,
    )


class FakeRunEngine:
    instances: typing.ClassVar[list[FakeRunEngine]] = []

    def __init__(
        self,
        bgen_path: str,
        chunk_size: int,
        variant_limit: int | None = None,
        trusted_no_missing_diploid: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.bgen_path = bgen_path
        self.chunk_size = chunk_size
        self.variant_limit = variant_limit
        self.trusted_no_missing_diploid = trusted_no_missing_diploid
        self.sample_count = 2
        self.variant_count = 10
        self.run_arguments: tuple[np.ndarray, object, list[int] | None] | None = None
        self.run_call_arguments: list[tuple[np.ndarray, object, list[int] | None]] = []
        self.run_method: str | None = None
        self.reset_profile_count = 0
        self.validation_count = 0
        self.trusted_validation_mark_count = 0
        FakeRunEngine.instances.append(self)

    def reset_profile(self) -> None:
        self.reset_profile_count += 1

    def profile_snapshot(self) -> dict[str, int]:
        return {"variant_decode_count": 7}

    def validate_trusted_no_missing_diploid(self) -> None:
        self.validation_count += 1

    def mark_trusted_no_missing_diploid_validated(self) -> None:
        self.trusted_validation_mark_count += 1

    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]:
        selected_variant_count = variant_stop - variant_start
        return (
            ["22"] * selected_variant_count,
            [f"variant{variant_index}" for variant_index in range(variant_start, variant_stop)],
            [variant_index * 100 for variant_index in range(variant_start, variant_stop)],
            ["A"] * selected_variant_count,
            ["G"] * selected_variant_count,
        )

    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        self.run_call_arguments.append(self.run_arguments)
        return 0

    def run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples(
        self,
        aligned_sample_data: object,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        return self.run_bgen_variant_major_dosage_buffered_chunks(
            typing.cast("typing.Any", aligned_sample_data).sample_indices,
            callback,
            committed_chunk_identifiers,
        )

    def run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
        self,
        aligned_sample_data: object,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        return self.run_bgen_variant_major_dosage_buffered_chunks(
            typing.cast("typing.Any", aligned_sample_data).sample_indices,
            callback,
            committed_chunk_identifiers,
        )

    def run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "variant_major_packed8"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        self.run_call_arguments.append(self.run_arguments)
        return 0

    def run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_aligned_samples(
        self,
        aligned_sample_data: object,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        return self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
            typing.cast("typing.Any", aligned_sample_data).sample_indices,
            callback,
            committed_chunk_identifiers,
        )

    def run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples(
        self,
        aligned_sample_data: object,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        return self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
            typing.cast("typing.Any", aligned_sample_data).sample_indices,
            callback,
            committed_chunk_identifiers,
        )


class PartialCommitDeliveringRunEngine(FakeRunEngine):
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        self.run_call_arguments.append(self.run_arguments)
        for chunk_identifier in (0, 64):
            typing.cast("typing.Any", callback).compute_preprocessed_variant_major_dosage_chunk(
                metadata=build_native_metadata_for_chunk(chunk_identifier=chunk_identifier),
                genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
                chunk_stats=typing.cast("typing.Any", LinearNativeSumChunkStats()),
            )
        return 2


def build_native_aligned_sample_data() -> SimpleNamespace:
    return SimpleNamespace(
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        family_identifiers=["family1", "family2"],
        individual_identifiers=["sample1", "sample2"],
        phenotype_name="trait",
        phenotype_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        covariate_names=["intercept", "age"],
        covariate_matrix=np.asarray([[1.0], [1.0]], dtype=np.float32),
        is_binary_trait=False,
    )


def build_native_run_input() -> native_dispatch.NativeBgenRunInput:
    return native_dispatch.NativeBgenRunInput(
        native_aligned_sample_data=typing.cast("typing.Any", build_native_aligned_sample_data()),
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        phenotype_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        covariate_matrix=np.asarray([[1.0], [1.0]], dtype=np.float32),
        is_binary_trait=False,
    )


def test_open_pipeline_bgen_engine_records_selected_backend_telemetry() -> None:
    telemetry_session = RecordingTelemetrySession()
    pipeline_options = build_default_pipeline_runtime_options()
    writer_settings = regenie2_pipeline.build_output_writer_settings(
        finalize_parquet=False,
        writer_thread_count=pipeline_options.writer_thread_count,
        writer_queue_depth=pipeline_options.writer_queue_depth,
        chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
        parquet_compression=pipeline_options.parquet_compression,
        arrow_compression=types.ArrowCompression.ZSTD,
        output_format=types.OutputFormat.PARQUET,
    )
    context = regenie2_pipeline.build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
        phenotype_path=Path("phenotype.tsv"),
        prediction_list_path=Path("pred.list"),
        covariate_path=None,
        chunk_size=32,
        variant_limit=None,
        trusted_no_missing_diploid=False,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
        jax_device=types.Device.GPU,
        jax_matmul_precision=None,
        score_dtype=pipeline_options.score_dtype,
        firth_dtype=pipeline_options.firth_dtype,
        gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        correction_plan=types.BinaryCorrectionPlan(),
        binary_kernel_config=None,
        linear_numerical_config=None,
        writer_settings=writer_settings,
        stage_timing_recorder=None,
        telemetry_session=typing.cast("typing.Any", telemetry_session),
        alignment_config=None,
    )
    engine = FakeRunEngine("study.bgen", chunk_size=32, trusted_no_missing_diploid=True)

    with patch("g.engine.regenie2_pipeline.native_dispatch.build_bgen_run_engine", return_value=engine):
        opened_engine = regenie2_pipeline.open_pipeline_bgen_engine(
            context=context,
            pipeline_label="linear",
            phenotype_name="trait",
        )

    assert opened_engine is engine
    assert telemetry_session.events[0] == (
        "association_backend_selected",
        {
            "association_mode": "regenie2_linear",
            "association_backend_kind": "jax_packed8",
            "device": "gpu",
            "genotype_format": "packed8",
            "phenotype": "trait",
        },
    )
    assert telemetry_session.events[1][0] == "bgen_engine_opened"
    assert telemetry_session.events[1][1]["association_backend_kind"] == "jax_packed8"


def build_native_run_input_with_alignment(
    *,
    phenotype_name: str,
    sample_indices: tuple[int, ...],
    phenotype_values: tuple[float, ...],
    covariate_values: tuple[tuple[float, ...], ...],
) -> native_dispatch.NativeBgenRunInput:
    native_aligned_sample_data = SimpleNamespace(
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        family_identifiers=[f"family{sample_index}" for sample_index in sample_indices],
        individual_identifiers=[f"sample{sample_index}" for sample_index in sample_indices],
        phenotype_name=phenotype_name,
        phenotype_vector=np.asarray(phenotype_values, dtype=np.float32),
        covariate_names=["intercept", "age"],
        covariate_matrix=np.asarray(covariate_values, dtype=np.float32),
        is_binary_trait=False,
    )
    return native_dispatch.NativeBgenRunInput(
        native_aligned_sample_data=typing.cast("typing.Any", native_aligned_sample_data),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        phenotype_vector=np.asarray(phenotype_values, dtype=np.float32),
        covariate_matrix=np.asarray(covariate_values, dtype=np.float32),
        is_binary_trait=False,
    )


def build_grouped_run_input_from_single_trait_inputs(
    *,
    phenotype_indices: tuple[int, ...],
    phenotype_names: tuple[str, ...],
    run_inputs: tuple[native_dispatch.NativeBgenRunInput, ...],
) -> native_dispatch.NativeBgenGroupedRunInput:
    first_run_input = run_inputs[0]
    native_multi_aligned_sample_data = SimpleNamespace(
        phenotype_names=phenotype_names,
        sample_indices=first_run_input.sample_indices,
        family_identifiers=tuple(first_run_input.native_aligned_sample_data.family_identifiers),
        individual_identifiers=tuple(first_run_input.native_aligned_sample_data.individual_identifiers),
        phenotype_matrix=np.stack(
            tuple(np.asarray(run_input.phenotype_vector, dtype=np.float32) for run_input in run_inputs),
            axis=0,
        ),
        covariate_names=tuple(first_run_input.native_aligned_sample_data.covariate_names),
        covariate_matrix=np.asarray(first_run_input.covariate_matrix, dtype=np.float32),
        is_binary_trait=first_run_input.is_binary_trait,
    )
    run_input = native_dispatch.NativeBgenMultiRunInput(
        native_multi_aligned_sample_data=typing.cast("typing.Any", native_multi_aligned_sample_data),
        phenotype_names=phenotype_names,
        sample_indices=np.ascontiguousarray(native_multi_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_matrix=np.asarray(native_multi_aligned_sample_data.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(native_multi_aligned_sample_data.covariate_matrix, dtype=np.float32),
        is_binary_trait=native_multi_aligned_sample_data.is_binary_trait,
    )
    return native_dispatch.NativeBgenGroupedRunInput(
        compute_group=native_dispatch.build_resolved_phenotype_compute_group(
            phenotype_indices=phenotype_indices,
            run_input=run_input,
            prediction_list_path=Path("pred.list"),
            planned_compute_groups=None,
            alignment_config=None,
        ),
        phenotype_indices=phenotype_indices,
        run_input=run_input,
        prediction_source=FakePredictionSource(),
    )


def build_native_multi_run_input() -> native_dispatch.NativeBgenMultiRunInput:
    native_multi_aligned_sample_data = SimpleNamespace(
        phenotype_names=["trait_a", "trait_b"],
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        family_identifiers=["f2", "f1"],
        individual_identifiers=["i2", "i1"],
        phenotype_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        covariate_names=["intercept", "age"],
        covariate_matrix=np.asarray([[1.0], [1.0]], dtype=np.float32),
        is_binary_trait=False,
    )
    return native_dispatch.NativeBgenMultiRunInput(
        native_multi_aligned_sample_data=typing.cast("typing.Any", native_multi_aligned_sample_data),
        phenotype_names=("trait_a", "trait_b"),
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        phenotype_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        covariate_matrix=np.asarray([[1.0], [1.0]], dtype=np.float32),
        is_binary_trait=False,
    )


def test_complete_case_compute_group_resolution_adds_alignment_fingerprints() -> None:
    run_input = build_native_multi_run_input()
    planned_compute_groups = execution_plan.build_phenotype_compute_groups(
        phenotype_names=("trait_a", "trait_b"),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )

    compute_group = native_dispatch.build_resolved_complete_case_phenotype_compute_group(
        run_input=run_input,
        prediction_list_path=Path("pred.list"),
        planned_compute_groups=planned_compute_groups,
        alignment_config=None,
    )

    assert compute_group.group_mode == types.PhenotypeComputeGroupMode.COMPLETE_CASE
    assert compute_group.phenotype_indices == (0, 1)
    assert compute_group.phenotype_names == ("trait_a", "trait_b")
    assert compute_group.sample_set_fingerprint is not None
    assert compute_group.covariate_design_fingerprint is not None
    assert compute_group.prediction_alignment_fingerprint is not None


def build_native_metadata() -> typing.Any:
    return build_native_metadata_for_chunk(chunk_identifier=5)


def build_native_metadata_for_chunk(*, chunk_identifier: int) -> typing.Any:
    return SimpleNamespace(
        variant_start_index=chunk_identifier,
        variant_stop_index=chunk_identifier + 2,
        chromosome=["22", "22"],
        variant_identifiers=[f"variant{chunk_identifier}", f"variant{chunk_identifier + 1}"],
        position=np.asarray([chunk_identifier * 100, (chunk_identifier + 1) * 100], dtype=np.int64),
        allele_one=["A", "C"],
        allele_two=["G", "T"],
    )


def test_get_metadata_chromosome_prefers_scalar_label_without_full_column_access() -> None:
    class ScalarChromosomeMetadata:
        chromosome_label = "22"

        @property
        def chromosome(self) -> list[str]:
            raise AssertionError("chromosome column should not be read when scalar label is available")

    assert callbacks.get_metadata_chromosome(ScalarChromosomeMetadata()) == "22"


def build_binary_chromosome_state(*, converged: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        score_residual=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        null_logistic_iteration_count=jnp.asarray(3, dtype=jnp.int32),
        null_logistic_converged=jnp.asarray(converged, dtype=jnp.bool_),
        null_firth_iteration_count=jnp.asarray(0, dtype=jnp.int32),
        null_firth_convergence_reason_code=jnp.asarray(0, dtype=jnp.int32),
    )


def build_multi_binary_chromosome_state(*, convergence_flags: tuple[bool, ...] = (True, True)) -> SimpleNamespace:
    return SimpleNamespace(
        score_residual=jnp.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32),
        null_logistic_iteration_count=jnp.asarray([3, 3], dtype=jnp.int32),
        null_logistic_converged=jnp.asarray(convergence_flags, dtype=jnp.bool_),
    )


class ExplodingChunkStats:
    @property
    def allele_one_frequency(self) -> np.ndarray:
        message = "Python must not unwrap allele_one_frequency from native chunk stats."
        raise AssertionError(message)

    @property
    def observation_count(self) -> np.ndarray:
        message = "Python must not unwrap observation_count from native chunk stats."
        raise AssertionError(message)


class SparseOnlyChunkStats(ExplodingChunkStats):
    @property
    def dosage_sum(self) -> np.ndarray:
        return np.asarray([3.0, 7.0], dtype=np.float32)

    @property
    def observation_count(self) -> np.ndarray:
        return np.asarray([2, 2], dtype=np.int32)

    @property
    def is_rare_sparse_firth_candidate(self) -> np.ndarray:
        return np.asarray([True, False], dtype=np.bool_)


class ExplodingSparseCandidateChunkStats(ExplodingChunkStats):
    @property
    def dosage_sum(self) -> np.ndarray:
        return np.asarray([3.0, 7.0], dtype=np.float32)

    @property
    def observation_count(self) -> np.ndarray:
        return np.asarray([2, 2], dtype=np.int32)

    @property
    def is_rare_sparse_firth_candidate(self) -> np.ndarray:
        message = "Score-only callbacks must not unwrap or transfer sparse Firth candidate masks."
        raise AssertionError(message)


class LinearNativeSumChunkStats(ExplodingChunkStats):
    @property
    def dosage_sum(self) -> np.ndarray:
        return np.asarray([3.0, 7.0], dtype=np.float32)

    @property
    def observation_count(self) -> np.ndarray:
        return np.asarray([2, 2], dtype=np.int32)

    @property
    def imputed_dosage_square_sum(self) -> np.ndarray:
        return np.asarray([5.0, 13.0], dtype=np.float32)


class BundledChunkStats(ExplodingChunkStats):
    def __init__(self) -> None:
        self.requests: list[dict[str, bool]] = []

    @property
    def dosage_sum(self) -> np.ndarray:
        message = "Python should use compute_arrays instead of dosage_sum."
        raise AssertionError(message)

    @property
    def observation_count(self) -> np.ndarray:
        message = "Python should use compute_arrays instead of observation_count."
        raise AssertionError(message)

    @property
    def imputed_dosage_square_sum(self) -> np.ndarray:
        message = "Python should use compute_arrays instead of imputed_dosage_square_sum."
        raise AssertionError(message)

    @property
    def is_rare_sparse_firth_candidate(self) -> np.ndarray:
        message = "Python should use compute_arrays instead of is_rare_sparse_firth_candidate."
        raise AssertionError(message)

    def compute_arrays(
        self,
        *,
        include_imputed_dosage_square_sum: bool,
        include_sparse_firth_candidate: bool,
    ) -> dict[str, np.ndarray]:
        self.requests.append(
            {
                "include_imputed_dosage_square_sum": include_imputed_dosage_square_sum,
                "include_sparse_firth_candidate": include_sparse_firth_candidate,
            }
        )
        compute_arrays: dict[str, np.ndarray] = {
            "dosage_sum": np.asarray([3.0, 7.0], dtype=np.float32),
            "observation_count": np.asarray([2, 2], dtype=np.int32),
        }
        if include_imputed_dosage_square_sum:
            compute_arrays["imputed_dosage_square_sum"] = np.asarray([5.0, 13.0], dtype=np.float32)
        if include_sparse_firth_candidate:
            compute_arrays["is_rare_sparse_firth_candidate"] = np.asarray([True, False], dtype=np.bool_)
        return compute_arrays


class ManualCallbackRunner(callbacks.NativeBgenCallbackRunner):
    def __init__(self) -> None:
        self.processed_chunk_count = 0
        self.stage_timing_recorder = None
        self.telemetry_session = None
        self.current_progress_chromosome = None
        self.dosage_queue: queue.Queue[
            callbacks.PreprocessedDosageChunkWorkItem | callbacks.PreprocessedVariantMajorDosageChunkWorkItem | None
        ] = queue.Queue()
        self.result_queue: queue.Queue[
            callbacks.Regenie2ResultWriteWorkItem | callbacks.Regenie2MultiResultWriteWorkItem | None
        ] = queue.Queue()
        self.result_in_flight_slots = threading.BoundedSemaphore(2)
        self.free_dosage_buffers: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.dosage_buffer_count = 0
        self.dosage_buffer_identifiers: set[int] = set()
        self.dosage_buffer_limit = 2
        self.worker_error = None
        self.result_worker_error = None
        self.sample_major_metadata: list[object] = []
        self.variant_major_metadata: list[object] = []
        self.packed_metadata: list[object] = []

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix: object,
        chunk_stats: object,
    ) -> None:
        del genotype_matrix, chunk_stats
        self.sample_major_metadata.append(variant_metadata)

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: object,
        genotype_matrix_by_variant: object,
        chunk_stats: object,
    ) -> None:
        del genotype_matrix_by_variant, chunk_stats
        self.variant_major_metadata.append(variant_metadata)

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: object,
        packed_probability_pairs_by_variant: object,
        chunk_stats: object,
    ) -> None:
        del packed_probability_pairs_by_variant, chunk_stats
        self.packed_metadata.append(variant_metadata)


def test_native_callback_runner_records_chromosome_progress_transitions() -> None:
    callback = ManualCallbackRunner()
    telemetry_session = RecordingTelemetrySession()
    callback.telemetry_session = telemetry_session

    callback.processed_chunk_count = 1
    callback.record_progress(build_native_metadata())
    callback.processed_chunk_count = 2
    callback.record_progress(
        SimpleNamespace(
            variant_start_index=7,
            variant_stop_index=9,
            chromosome=["23", "23"],
        )
    )

    assert telemetry_session.events == [
        (
            "chromosome_started",
            {"chromosome": "22", "processed_chunk_count": 1},
        ),
        (
            "chromosome_completed",
            {"chromosome": "22", "processed_chunk_count": 1},
        ),
        (
            "chromosome_started",
            {"chromosome": "23", "processed_chunk_count": 2},
        ),
    ]
    assert telemetry_session.progress_events[0]["variant_count"] == 2
    assert telemetry_session.progress_events[1]["chunk_identifier"] == 7


def test_native_callback_runner_defers_worker_start_until_explicit_start() -> None:
    class ThreadedManualCallbackRunner(callbacks.NativeBgenCallbackRunner):
        def __init__(self) -> None:
            super().__init__(worker_name="threaded-manual-callback")

        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats

        def compute_preprocessed_variant_major_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix_by_variant, chunk_stats

        def compute_preprocessed_variant_major_packed8_chunk(
            self,
            *,
            variant_metadata: object,
            packed_probability_pairs_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    callback = ThreadedManualCallbackRunner()

    assert callback.worker_threads_started is False
    assert not callback.worker_thread.is_alive()
    assert not callback.result_worker_thread.is_alive()

    callback.start()
    try:
        assert callback.worker_threads_started is True
        assert callback.worker_thread.is_alive()
        assert callback.result_worker_thread.is_alive()
    finally:
        callback.finish()

    assert not callback.worker_thread.is_alive()
    assert not callback.result_worker_thread.is_alive()


def test_native_callback_runner_records_native_delivery_timing_for_enqueued_chunk() -> None:
    stage_timing_recorder = timing.StageTimingRecorder()

    class TimedCallbackRunner(callbacks.NativeBgenCallbackRunner):
        def __init__(self) -> None:
            super().__init__(
                worker_name="timed-manual-callback",
                stage_timing_recorder=stage_timing_recorder,
            )
            self.metadata: list[object] = []

        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del genotype_matrix, chunk_stats
            self.metadata.append(variant_metadata)

        def compute_preprocessed_variant_major_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix_by_variant, chunk_stats

        def compute_preprocessed_variant_major_packed8_chunk(
            self,
            *,
            variant_metadata: object,
            packed_probability_pairs_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    callback = TimedCallbackRunner()
    metadata = build_native_metadata()
    try:
        callback.compute_preprocessed_dosage_chunk(
            metadata=metadata,
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        )
        callback.finish()
    finally:
        callback.abort()

    assert callback.metadata == [metadata]
    snapshot = stage_timing_recorder.snapshot()
    assert snapshot.stage_counts["native_delivery"] == 1
    assert snapshot.stage_counts["python_callback"] == 1
    assert {chunk_timing.stage_name for chunk_timing in snapshot.chunk_stage_timings} >= {
        "native_delivery",
        "python_callback",
    }


def test_native_callback_runner_consumes_both_dosage_layouts() -> None:
    callback = ManualCallbackRunner()
    stage_timing_recorder = timing.StageTimingRecorder()
    callback.stage_timing_recorder = stage_timing_recorder
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())

    callback.dosage_queue.put_nowait(
        callbacks.PreprocessedVariantMajorDosageChunkWorkItem(
            metadata=metadata,
            genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
    )
    callback.dosage_queue.put_nowait(
        callbacks.PreprocessedDosageChunkWorkItem(
            metadata=metadata,
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
    )
    callback.dosage_queue.put_nowait(None)

    callback.consume_dosage_chunks()

    assert callback.variant_major_metadata == [metadata]
    assert callback.sample_major_metadata == [metadata]
    assert callback.processed_chunk_count == 2
    assert callback.worker_error is None
    snapshot = stage_timing_recorder.snapshot()
    assert tuple(chunk_timing.stage_name for chunk_timing in snapshot.chunk_stage_timings) == (
        "python_callback",
        "python_callback",
    )
    assert snapshot.stage_counts["python_callback"] == 2


def test_native_callback_runner_records_worker_errors_from_consumer() -> None:
    class FailingCallbackRunner(ManualCallbackRunner):
        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats
            message = "compute failed"
            raise ValueError(message)

    callback = FailingCallbackRunner()
    callback.dosage_queue.put_nowait(
        callbacks.PreprocessedDosageChunkWorkItem(
            metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        )
    )

    callback.consume_dosage_chunks()

    assert isinstance(callback.worker_error, ValueError)


def test_native_callback_runner_reuses_and_replaces_host_dosage_buffers() -> None:
    callback = ManualCallbackRunner()

    first_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    assert first_buffer.shape == (2, 3)
    assert callback.dosage_buffer_count == 1

    callback.release_dosage_buffer(first_buffer)
    reused_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    assert reused_buffer is first_buffer

    mismatched_buffer = callback.acquire_dosage_buffer(sample_count=3, variant_count=2)
    mismatched_buffer_identifier = id(mismatched_buffer)
    callback.release_dosage_buffer(mismatched_buffer)
    replacement_buffer = callback.acquire_variant_major_dosage_buffer(variant_count=2, sample_count=3)
    assert replacement_buffer.shape == (2, 3)
    assert callback.dosage_buffer_count == 2
    assert mismatched_buffer_identifier not in callback.dosage_buffer_identifiers
    assert id(replacement_buffer) in callback.dosage_buffer_identifiers

    limited_callback = ManualCallbackRunner()
    first_limited_buffer = limited_callback.acquire_dosage_buffer_with_shape((1, 1))
    second_limited_buffer = limited_callback.acquire_dosage_buffer_with_shape((1, 2))
    limited_callback.release_dosage_buffer(first_limited_buffer)
    blocked_replacement = limited_callback.acquire_dosage_buffer_with_shape((4, 5))
    assert blocked_replacement.shape == (4, 5)
    assert limited_callback.dosage_buffer_count == limited_callback.dosage_buffer_limit
    assert id(first_limited_buffer) not in limited_callback.dosage_buffer_identifiers
    assert id(second_limited_buffer) in limited_callback.dosage_buffer_identifiers
    assert id(blocked_replacement) in limited_callback.dosage_buffer_identifiers


def test_native_callback_runner_reuses_larger_host_dosage_buffer_as_view() -> None:
    callback = ManualCallbackRunner()

    oversized_buffer = callback.allocate_dosage_buffer_with_shape((4, 5), np.float32)
    callback.release_dosage_buffer(oversized_buffer)
    sliced_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    assert sliced_buffer.shape == (2, 3)
    assert np.shares_memory(sliced_buffer, oversized_buffer)
    assert sliced_buffer.base is oversized_buffer
    assert callback.dosage_buffer_count == 1

    releasable_sliced_buffer = callback.get_releasable_dosage_buffer(sliced_buffer)
    assert releasable_sliced_buffer is not None
    assert releasable_sliced_buffer is oversized_buffer

    callback.release_dosage_buffer(sliced_buffer)
    restored_buffer = callback.acquire_dosage_buffer(sample_count=4, variant_count=5)
    assert restored_buffer is oversized_buffer


def test_native_callback_runner_ignores_unowned_host_dosage_buffers() -> None:
    callback = ManualCallbackRunner()

    callback.release_dosage_buffer(np.empty((2, 2), dtype=np.float32))

    assert callback.dosage_buffer_count == 0
    assert callback.free_dosage_buffers.empty()


def test_native_callback_runner_surfaces_worker_and_writer_errors() -> None:
    callback = ManualCallbackRunner()
    callback.worker_error = ValueError("dosage failed")

    with pytest.raises(RuntimeError, match="callback worker failed"):
        callback.raise_worker_error_if_present()

    callback.worker_error = None
    callback.result_worker_error = ValueError("writer failed")

    with pytest.raises(RuntimeError, match="result writer worker failed"):
        callback.raise_worker_error_if_present()


def test_base_native_callback_runner_compute_methods_are_abstract() -> None:
    class IncompleteCallbackRunner(callbacks.NativeBgenCallbackRunner):
        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats

    with pytest.raises(TypeError, match="abstract"):
        IncompleteCallbackRunner(worker_name="incomplete-callback")


def test_stop_result_worker_returns_when_failed_worker_leaves_full_queue() -> None:
    result_queue: queue.Queue[
        callbacks.Regenie2ResultWriteWorkItem | callbacks.Regenie2MultiResultWriteWorkItem | None
    ] = queue.Queue(maxsize=1)
    result_queue.put_nowait(None)
    stop_event = threading.Event()
    result_worker_thread = threading.Thread(target=stop_event.wait, name="failed-result-worker")
    result_worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.result_queue = result_queue
    callback.result_worker_error = RuntimeError("writer failed")
    callback.result_worker_thread = result_worker_thread
    callback.worker_threads_started = True

    try:
        callback.stop_result_worker()
    finally:
        stop_event.set()
        result_worker_thread.join()

    assert result_queue.full()


def test_stop_dosage_worker_returns_when_failed_worker_leaves_full_queue() -> None:
    dosage_queue: queue.Queue[callbacks.PreprocessedDosageChunkWorkItem | None] = queue.Queue(maxsize=1)
    dosage_queue.put_nowait(None)
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=stop_event.wait, name="failed-dosage-worker")
    worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.dosage_queue = dosage_queue
    callback.worker_error = RuntimeError("dosage failed")
    callback.worker_thread = worker_thread
    callback.worker_threads_started = True

    try:
        callback.stop_dosage_worker()
    finally:
        stop_event.set()
        worker_thread.join()

    assert dosage_queue.full()


def test_stop_result_worker_raises_when_live_worker_leaves_full_queue() -> None:
    result_queue: queue.Queue[
        callbacks.Regenie2ResultWriteWorkItem | callbacks.Regenie2MultiResultWriteWorkItem | None
    ] = queue.Queue(maxsize=1)
    result_queue.put_nowait(None)
    stop_event = threading.Event()
    result_worker_thread = threading.Thread(target=stop_event.wait, name="blocked-result-worker")
    result_worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.result_queue = result_queue
    callback.result_worker_error = None
    callback.result_worker_thread = result_worker_thread
    callback.worker_threads_started = True

    try:
        with (
            patch("g.engine.callbacks.runtime.RESULT_WORKER_JOIN_TIMEOUT_SECONDS", 0.0),
            np.testing.assert_raises_regex(callbacks.NativeBgenWorkerShutdownError, "blocked-result-worker"),
        ):
            callback.stop_result_worker()
    finally:
        stop_event.set()
        result_worker_thread.join()


def test_stop_dosage_worker_raises_when_live_worker_leaves_full_queue() -> None:
    dosage_queue: queue.Queue[callbacks.PreprocessedDosageChunkWorkItem | None] = queue.Queue(maxsize=1)
    dosage_queue.put_nowait(None)
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=stop_event.wait, name="blocked-dosage-worker")
    worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.dosage_queue = dosage_queue
    callback.worker_error = None
    callback.worker_thread = worker_thread
    callback.worker_threads_started = True

    try:
        with (
            patch("g.engine.callbacks.runtime.DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS", 0.0),
            np.testing.assert_raises_regex(callbacks.NativeBgenWorkerShutdownError, "blocked-dosage-worker"),
        ):
            callback.stop_dosage_worker()
    finally:
        stop_event.set()
        worker_thread.join()


def test_join_result_worker_raises_when_worker_does_not_stop() -> None:
    stop_event = threading.Event()
    result_worker_thread = threading.Thread(target=stop_event.wait, name="stuck-result-worker")
    result_worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.result_worker_thread = result_worker_thread
    callback.worker_threads_started = True

    try:
        with (
            patch("g.engine.callbacks.runtime.RESULT_WORKER_JOIN_TIMEOUT_SECONDS", 0.0),
            np.testing.assert_raises_regex(callbacks.NativeBgenWorkerShutdownError, "stuck-result-worker"),
        ):
            callback.join_result_worker()
    finally:
        stop_event.set()
        result_worker_thread.join()


def test_join_dosage_worker_raises_when_worker_does_not_stop() -> None:
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=stop_event.wait, name="stuck-dosage-worker")
    worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.worker_thread = worker_thread
    callback.worker_threads_started = True

    try:
        with (
            patch("g.engine.callbacks.runtime.DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS", 0.0),
            np.testing.assert_raises_regex(callbacks.NativeBgenWorkerShutdownError, "stuck-dosage-worker"),
        ):
            callback.join_dosage_worker()
    finally:
        stop_event.set()
        worker_thread.join()


def test_native_bgen_callback_runner_rejects_nonpositive_staging_depth() -> None:
    class ConcreteCallbackRunner(callbacks.NativeBgenCallbackRunner):
        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats

        def compute_preprocessed_variant_major_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix_by_variant, chunk_stats

        def compute_preprocessed_variant_major_packed8_chunk(
            self,
            *,
            variant_metadata: object,
            packed_probability_pairs_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    with pytest.raises(ValueError, match="staging_depth must be positive"):
        ConcreteCallbackRunner(worker_name="invalid-staging-depth", staging_depth=0)


def test_native_bgen_callback_runner_accepts_explicit_capacity_limits() -> None:
    class ConcreteCallbackRunner(callbacks.NativeBgenCallbackRunner):
        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats

        def compute_preprocessed_variant_major_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix_by_variant, chunk_stats

        def compute_preprocessed_variant_major_packed8_chunk(
            self,
            *,
            variant_metadata: object,
            packed_probability_pairs_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    default_callback = ConcreteCallbackRunner(worker_name="default-capacity", staging_depth=3)
    explicit_callback = ConcreteCallbackRunner(
        worker_name="explicit-capacity",
        staging_depth=3,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )

    assert default_callback.result_in_flight_limit == 4
    assert default_callback.dosage_buffer_limit == 4
    assert explicit_callback.result_in_flight_limit == 7
    assert explicit_callback.dosage_buffer_limit == 8


@pytest.mark.parametrize(
    ("capacity_name", "error_message"),
    [
        ("result_in_flight_limit", "result_in_flight_limit must be positive"),
        ("dosage_buffer_limit", "dosage_buffer_limit must be positive"),
    ],
)
def test_native_bgen_callback_runner_rejects_nonpositive_capacity_limits(
    capacity_name: typing.Literal["result_in_flight_limit", "dosage_buffer_limit"],
    error_message: str,
) -> None:
    class ConcreteCallbackRunner(callbacks.NativeBgenCallbackRunner):
        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats

        def compute_preprocessed_variant_major_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix_by_variant, chunk_stats

        def compute_preprocessed_variant_major_packed8_chunk(
            self,
            *,
            variant_metadata: object,
            packed_probability_pairs_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    with pytest.raises(ValueError, match=error_message):
        if capacity_name == "result_in_flight_limit":
            ConcreteCallbackRunner(worker_name="invalid-capacity", staging_depth=1, result_in_flight_limit=0)
        else:
            ConcreteCallbackRunner(worker_name="invalid-capacity", staging_depth=1, dosage_buffer_limit=0)


def test_linear_callback_passes_native_stats_to_writer_without_python_unwrap() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
    )
    chunk_stats = typing.cast("typing.Any", ExplodingChunkStats())

    with (
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_linear_chromosome_state",
            return_value="chromosome-state",
        ),
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state",
            return_value=result,
        ),
    ):
        callback.compute_preprocessed_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
        callback.finish()

    assert len(writer_session.native_chunks) == 1
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_linear_variant_major_callback_passes_native_sums_to_jitted_compute() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        linear_numerical_config=regenie2_linear_config.LinearNumericalConfig(
            minimum_variance=3.0e-9,
            relative_variance_tolerance=4.0e-6,
        ),
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_linear_state.Regenie2LinearChromosomeState",
        "chromosome-state",
    )
    chunk_stats = typing.cast("typing.Any", LinearNativeSumChunkStats())

    with patch(
        "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state_variant_major",
        return_value=result,
    ) as mock_compute:
        callback.compute_preprocessed_variant_major_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
        callback.finish()

    np.testing.assert_array_equal(np.asarray(mock_compute.call_args.kwargs["genotype_dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(np.asarray(mock_compute.call_args.kwargs["genotype_observation_count"]), [2, 2])
    np.testing.assert_array_equal(
        np.asarray(mock_compute.call_args.kwargs["genotype_imputed_dosage_square_sum"]),
        [5.0, 13.0],
    )
    assert mock_compute.call_args.kwargs["linear_minimum_variance"] == 3.0e-9
    assert mock_compute.call_args.kwargs["linear_relative_variance_tolerance"] == 4.0e-6
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_linear_packed8_callback_passes_native_sums_to_jitted_compute() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_linear_state.Regenie2LinearChromosomeState",
        "chromosome-state",
    )
    chunk_stats = typing.cast("typing.Any", LinearNativeSumChunkStats())
    packed_probability_pairs_by_variant = np.asarray(
        [
            [[255, 0], [0, 0]],
            [[0, 255], [255, 0]],
        ],
        dtype=np.uint8,
    )

    with (
        patch(
            "g.compute.regenie2_linear.api.compute_linear_chunk_packed8_donating_inputs",
            return_value=result,
        ) as mock_packed_compute,
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state_variant_major",
        ) as mock_variant_major_compute,
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state",
        ) as mock_sample_major_compute,
    ):
        callback.compute_preprocessed_variant_major_packed8_probability_pair_chunk(
            metadata=build_native_metadata(),
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            chunk_stats=chunk_stats,
        )
        callback.finish()

    packed_probability_pairs_argument = mock_packed_compute.call_args.kwargs["packed_probability_pairs_by_variant"]
    np.testing.assert_array_equal(np.asarray(packed_probability_pairs_argument), packed_probability_pairs_by_variant)
    np.testing.assert_array_equal(np.asarray(mock_packed_compute.call_args.kwargs["genotype_dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(
        np.asarray(mock_packed_compute.call_args.kwargs["genotype_observation_count"]),
        [2, 2],
    )
    np.testing.assert_array_equal(
        np.asarray(mock_packed_compute.call_args.kwargs["genotype_imputed_dosage_square_sum"]),
        [5.0, 13.0],
    )
    mock_variant_major_compute.assert_not_called()
    mock_sample_major_compute.assert_not_called()
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_linear_callback_does_not_block_chunk_compute_without_timing() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_linear_state.Regenie2LinearChromosomeState",
        "chromosome-state",
    )

    with (
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state",
            return_value=result,
        ),
        patch("g.engine.callbacks.transfers.block_until_ready") as mock_block_until_ready,
    ):
        callback.compute_linear_result(
            variant_metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
        )
        callback.finish()

    mock_block_until_ready.assert_not_called()


def test_linear_callback_records_aggregate_chunk_timing_without_blocking() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    stage_timing_recorder = timing.StageTimingRecorder()
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        stage_timing_recorder=stage_timing_recorder,
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_linear_state.Regenie2LinearChromosomeState",
        "chromosome-state",
    )

    with (
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state",
            return_value=result,
        ),
        patch("g.engine.callbacks.transfers.block_until_ready") as mock_block_until_ready,
    ):
        callback.compute_linear_result(
            variant_metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
        )
        callback.finish()

    mock_block_until_ready.assert_not_called()
    snapshot = stage_timing_recorder.snapshot()
    assert snapshot.stage_counts["host_to_device_transfer"] == 1
    assert snapshot.stage_counts["jax_compute"] == 1


def test_linear_callback_blocks_chunk_compute_with_exact_timing() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=True)
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        stage_timing_recorder=stage_timing_recorder,
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_linear_state.Regenie2LinearChromosomeState",
        "chromosome-state",
    )

    with (
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_linear_chunk_from_chromosome_state",
            return_value=result,
        ),
        patch("g.engine.callbacks.transfers.block_until_ready") as mock_block_until_ready,
    ):
        callback.compute_linear_result(
            variant_metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
        )
        callback.finish()

    assert mock_block_until_ready.call_count == 2
    snapshot = stage_timing_recorder.snapshot()
    assert snapshot.stage_counts["host_to_device_transfer"] == 1
    assert snapshot.stage_counts["jax_compute"] == 1


def test_result_worker_releases_in_flight_slot_after_materialization() -> None:
    writer_session = FakeWriterSession()
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        staging_depth=1,
    )
    host_dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=2)
    host_dosage_buffer.fill(1)
    callback.acquire_result_in_flight_slot()
    callback.acquire_result_in_flight_slot()

    assert callback.result_in_flight_slots.acquire(blocking=False) is False

    callback.put_result_write_item(
        callbacks.Regenie2ResultWriteWorkItem(
            metadata=build_native_metadata(),
            chunk_stats=typing.cast("typing.Any", ExplodingChunkStats()),
            beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
            standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
            chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            extra_code=None,
            host_dosage_buffer=host_dosage_buffer,
            release_in_flight_slot=True,
        )
    )
    callback.finish()

    assert callback.result_in_flight_slots.acquire(blocking=False) is True
    callback.release_result_in_flight_slot()
    callback.release_result_in_flight_slot()
    assert callback.free_dosage_buffers.get_nowait() is host_dosage_buffer
    assert len(writer_session.native_chunks) == 1


def test_binary_callback_passes_native_sparse_mask_without_unwrapping_full_stats() -> None:
    writer_session = FakeWriterSession()
    kernel_config = build_default_binary_kernel_config()
    result = regenie2_binary_result.Regenie2BinaryChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True]),
        firth_iteration_count=jnp.asarray([0, 2], dtype=jnp.int32),
        firth_failure_code=jnp.asarray(
            [types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value], dtype=jnp.int32
        ),
        firth_convergence_reason_code=jnp.asarray(
            [
                regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
            ],
            dtype=jnp.int32,
        ),
        firth_correction_code=jnp.zeros(2, dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros(2, dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros(2, dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros(2, dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros(2, dtype=jnp.int32),
    )
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
        kernel_config=kernel_config,
    )
    chunk_stats = typing.cast("typing.Any", SparseOnlyChunkStats())
    chromosome_state = build_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ) as mock_prepare,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state",
            return_value=result,
        ) as mock_compute,
    ):
        callback.compute_preprocessed_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
        callback.finish()

    sparse_candidate_mask = mock_compute.call_args.kwargs["sparse_candidate_mask"]
    np.testing.assert_array_equal(np.asarray(sparse_candidate_mask), [True, False])
    assert mock_prepare.call_args.kwargs["kernel_config"] is kernel_config
    assert mock_compute.call_args.kwargs["correction_plan"].method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
    assert mock_compute.call_args.kwargs["kernel_config"] is kernel_config
    assert mock_compute.call_args.kwargs["chromosome_state"] is chromosome_state
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_binary_score_only_sample_major_callback_skips_sparse_mask_transfer() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )
    chunk_stats = typing.cast("typing.Any", ExplodingSparseCandidateChunkStats())
    chromosome_state = build_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state",
            return_value=result,
        ) as mock_compute,
    ):
        callback.compute_preprocessed_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
        callback.finish()

    assert mock_compute.call_args.kwargs["sparse_candidate_mask"] is None
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_binary_variant_major_callback_uses_direct_variant_major_firth_compute() -> None:
    writer_session = FakeWriterSession()
    kernel_config = build_default_binary_kernel_config()
    stage_timing_recorder = timing.StageTimingRecorder()
    result = regenie2_binary_result.Regenie2BinaryChunkResult(
        beta=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4, 0.5], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                types.BinaryExtraCode.SCORE.value,
                types.BinaryExtraCode.FIRTH.value,
                types.BinaryExtraCode.FIRTH.value,
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True, True]),
        firth_iteration_count=jnp.asarray([0, 2, 1], dtype=jnp.int32),
        firth_failure_code=jnp.asarray(
            [types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value],
            dtype=jnp.int32,
        ),
        firth_convergence_reason_code=jnp.asarray(
            [
                regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
            ],
            dtype=jnp.int32,
        ),
        firth_correction_code=jnp.zeros(3, dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros(3, dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros(3, dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros(3, dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros(3, dtype=jnp.int32),
    )
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
        kernel_config=kernel_config,
        stage_timing_recorder=stage_timing_recorder,
    )
    variant_major_genotype_matrix = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    variant_metadata = SimpleNamespace(
        variant_start_index=5,
        variant_stop_index=8,
        chromosome=["22", "22", "22"],
        variant_identifiers=["variant5", "variant6", "variant7"],
        position=np.asarray([100, 200, 300], dtype=np.int64),
        allele_one=["A", "C", "G"],
        allele_two=["G", "T", "A"],
    )
    chunk_stats = SimpleNamespace(
        dosage_sum=np.asarray([3.0, 7.0, 11.0], dtype=np.float32),
        observation_count=np.asarray([2, 2, 2], dtype=np.int32),
        is_rare_sparse_firth_candidate=np.asarray([True, False, True], dtype=np.bool_),
    )
    chromosome_state = build_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state_variant_major",
            return_value=result,
        ) as mock_compute,
    ):
        callback.compute_preprocessed_variant_major_dosage_chunk(
            metadata=typing.cast("typing.Any", variant_metadata),
            genotype_matrix_by_variant=variant_major_genotype_matrix,
            chunk_stats=typing.cast("typing.Any", chunk_stats),
        )
        callback.finish()

    genotype_matrix_by_variant = mock_compute.call_args.kwargs["genotype_matrix_by_variant"]
    np.testing.assert_array_equal(np.asarray(genotype_matrix_by_variant), variant_major_genotype_matrix)
    sparse_candidate_mask = mock_compute.call_args.kwargs["sparse_candidate_mask"]
    np.testing.assert_array_equal(np.asarray(sparse_candidate_mask), [True, False, True])
    dosage_sum = mock_compute.call_args.kwargs["dosage_sum"]
    np.testing.assert_array_equal(np.asarray(dosage_sum), [3.0, 7.0, 11.0])
    observation_count = mock_compute.call_args.kwargs["observation_count"]
    np.testing.assert_array_equal(np.asarray(observation_count), [2, 2, 2])
    assert mock_compute.call_args.kwargs["chromosome_state"] is chromosome_state
    assert mock_compute.call_args.kwargs["correction_plan"].method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
    assert mock_compute.call_args.kwargs["kernel_config"] is kernel_config
    stage_duration_recorder = typing.cast(
        "typing.Callable[[str, float], None]",
        mock_compute.call_args.kwargs["stage_duration_recorder"],
    )
    stage_duration_recorder("firth_candidate_dispatch_plan", 0.0)
    assert stage_timing_recorder.snapshot().stage_counts["firth_candidate_dispatch_plan"] == 1
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_binary_score_only_variant_major_callback_uses_jitted_variant_major_score_compute() -> None:
    writer_session = FakeWriterSession()
    kernel_config = build_default_binary_kernel_config()
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4, 0.5], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                types.BinaryExtraCode.SCORE.value,
                types.BinaryExtraCode.SCORE.value,
                types.BinaryExtraCode.SCORE.value,
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True, True]),
    )
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=kernel_config,
    )
    variant_major_genotype_matrix = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    chunk_stats = SimpleNamespace(
        dosage_sum=np.asarray([3.0, 7.0, 11.0], dtype=np.float32),
        observation_count=np.asarray([2, 2, 2], dtype=np.int32),
        is_rare_sparse_firth_candidate=np.asarray([True, False, True], dtype=np.bool_),
    )
    chromosome_state = build_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_binary_score_test_variant_major_donating_inputs",
            return_value=result,
        ) as mock_variant_major_score_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state_variant_major",
        ) as mock_variant_major_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state",
        ) as mock_sample_major_compute,
    ):
        callback.compute_preprocessed_variant_major_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix_by_variant=variant_major_genotype_matrix,
            chunk_stats=typing.cast("typing.Any", chunk_stats),
        )
        callback.finish()

    genotype_matrix_by_variant = mock_variant_major_score_compute.call_args.kwargs["genotype_matrix_by_variant"]
    np.testing.assert_array_equal(np.asarray(genotype_matrix_by_variant), variant_major_genotype_matrix)
    assert mock_variant_major_score_compute.call_args.kwargs["chromosome_state"] is chromosome_state
    assert mock_variant_major_score_compute.call_args.kwargs["kernel_config"] is kernel_config
    dosage_sum = mock_variant_major_score_compute.call_args.kwargs["dosage_sum"]
    np.testing.assert_array_equal(np.asarray(dosage_sum), [3.0, 7.0, 11.0])
    observation_count = mock_variant_major_score_compute.call_args.kwargs["observation_count"]
    np.testing.assert_array_equal(np.asarray(observation_count), [2, 2, 2])
    assert "stage_duration_recorder" not in mock_variant_major_score_compute.call_args.kwargs
    mock_variant_major_compute.assert_not_called()
    mock_sample_major_compute.assert_not_called()
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_binary_score_only_packed8_callback_uses_jitted_packed_score_compute() -> None:
    writer_session = FakeWriterSession()
    kernel_config = build_default_binary_kernel_config()
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4, 0.5], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                types.BinaryExtraCode.SCORE.value,
                types.BinaryExtraCode.SCORE.value,
                types.BinaryExtraCode.SCORE.value,
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True, True]),
    )
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=kernel_config,
    )
    packed_probability_pairs_by_variant = np.asarray(
        [
            [[255, 0], [0, 0]],
            [[0, 255], [255, 0]],
            [[0, 0], [0, 255]],
        ],
        dtype=np.uint8,
    )
    chunk_stats = SimpleNamespace(
        dosage_sum=np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
        observation_count=np.asarray([2, 2, 2], dtype=np.int32),
        is_rare_sparse_firth_candidate=np.asarray([True, False, True], dtype=np.bool_),
    )
    chromosome_state = build_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_binary_score_test_packed8_donating_inputs",
            return_value=result,
        ) as mock_packed_score_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state_packed8",
        ) as mock_packed_chunk_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_binary_score_test_variant_major_donating_inputs",
        ) as mock_variant_major_score_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_binary_chunk_from_chromosome_state",
        ) as mock_sample_major_compute,
    ):
        callback.compute_preprocessed_variant_major_packed8_probability_pair_chunk(
            metadata=build_native_metadata(),
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            chunk_stats=typing.cast("typing.Any", chunk_stats),
        )
        callback.finish()

    packed_probability_pairs_argument = mock_packed_score_compute.call_args.kwargs[
        "packed_probability_pairs_by_variant"
    ]
    np.testing.assert_array_equal(np.asarray(packed_probability_pairs_argument), packed_probability_pairs_by_variant)
    assert mock_packed_score_compute.call_args.kwargs["chromosome_state"] is chromosome_state
    assert mock_packed_score_compute.call_args.kwargs["kernel_config"] is kernel_config
    dosage_sum = mock_packed_score_compute.call_args.kwargs["dosage_sum"]
    np.testing.assert_array_equal(np.asarray(dosage_sum), [2.0, 1.0, 3.0])
    observation_count = mock_packed_score_compute.call_args.kwargs["observation_count"]
    np.testing.assert_array_equal(np.asarray(observation_count), [2, 2, 2])
    assert "stage_duration_recorder" not in mock_packed_score_compute.call_args.kwargs
    mock_packed_chunk_compute.assert_not_called()
    mock_variant_major_score_compute.assert_not_called()
    mock_sample_major_compute.assert_not_called()
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def build_multi_linear_result() -> regenie2_linear_result.Regenie2MultiLinearChunkResult:
    return regenie2_linear_result.Regenie2MultiLinearChunkResult(
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        valid_mask=jnp.asarray([[True, True], [True, True]], dtype=jnp.bool_),
    )


def build_multi_trait_prediction_source() -> typing.Any:
    return SimpleNamespace(
        get_chromosome_predictions=lambda chromosome: np.zeros((2, 2), dtype=np.float32),
    )


def build_packed_probability_pairs_by_variant() -> np.ndarray:
    return np.asarray(
        [
            [[255, 0], [0, 0]],
            [[0, 255], [255, 0]],
        ],
        dtype=np.uint8,
    )


def build_multi_binary_score_result() -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    return regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult(
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([[True, True], [True, True]]),
    )


def build_multi_binary_chunk_result() -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    return regenie2_binary_result.Regenie2MultiBinaryChunkResult(
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([[True, True], [True, True]]),
        firth_iteration_count=jnp.asarray([[0, 2], [0, 2]], dtype=jnp.int32),
        firth_failure_code=jnp.asarray(
            [
                [types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value],
                [types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value],
            ],
            dtype=jnp.int32,
        ),
        firth_convergence_reason_code=jnp.asarray(
            [
                [
                    regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                ],
                [
                    regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                ],
            ],
            dtype=jnp.int32,
        ),
        firth_correction_code=jnp.zeros((2, 2), dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros((2, 2), dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros((2, 2), dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros((2, 2), dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros((2, 2), dtype=jnp.int32),
    )


def test_multi_linear_sample_major_callback_prepares_state_and_writes_traits() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    result = build_multi_linear_result()
    callback = callbacks.MultiLinearRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=build_multi_trait_prediction_source(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
    )

    with patch(
        "g.compute.regenie2_linear.api.compute_regenie2_multi_linear_chunk_from_chromosome_state",
        return_value=result,
    ) as mock_compute:
        callback.compute_preprocessed_chunk(
            variant_metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=typing.cast("typing.Any", ExplodingChunkStats()),
        )
        callback.finish()

    assert callback.current_chromosome == "22"
    assert mock_compute.call_args.kwargs["chromosome_state"] is callback.current_chromosome_state
    assert len(writer_sessions[0].native_chunks) == 1
    assert len(writer_sessions[1].native_chunks) == 1
    np.testing.assert_array_equal(writer_sessions[0].native_chunks[0]["beta"], np.asarray([0.1, 0.2], dtype=np.float32))
    np.testing.assert_array_equal(writer_sessions[1].native_chunks[0]["beta"], np.asarray([0.3, 0.4], dtype=np.float32))


def test_multi_linear_variant_major_callback_passes_native_genotype_summaries() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    result = build_multi_linear_result()
    callback = callbacks.MultiLinearRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=build_multi_trait_prediction_source(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        linear_numerical_config=regenie2_linear_config.LinearNumericalConfig(
            minimum_variance=5.0e-9,
            relative_variance_tolerance=6.0e-6,
        ),
    )
    variant_major_genotype_matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    chunk_stats = typing.cast("typing.Any", LinearNativeSumChunkStats())

    with patch(
        "g.compute.regenie2_linear.api.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major",
        return_value=result,
    ) as mock_compute:
        callback.compute_preprocessed_variant_major_chunk(
            variant_metadata=build_native_metadata(),
            genotype_matrix_by_variant=variant_major_genotype_matrix,
            chunk_stats=chunk_stats,
        )
        callback.finish()

    np.testing.assert_array_equal(
        np.asarray(mock_compute.call_args.kwargs["genotype_matrix_by_variant"]),
        variant_major_genotype_matrix,
    )
    np.testing.assert_array_equal(np.asarray(mock_compute.call_args.kwargs["genotype_dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(np.asarray(mock_compute.call_args.kwargs["genotype_observation_count"]), [2, 2])
    np.testing.assert_array_equal(
        np.asarray(mock_compute.call_args.kwargs["genotype_imputed_dosage_square_sum"]),
        [5.0, 13.0],
    )
    assert mock_compute.call_args.kwargs["linear_minimum_variance"] == 5.0e-9
    assert mock_compute.call_args.kwargs["linear_relative_variance_tolerance"] == 6.0e-6
    assert len(writer_sessions[0].native_chunks) == 1
    assert len(writer_sessions[1].native_chunks) == 1


def test_multi_linear_packed8_callback_uses_multi_packed_compute() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    result = build_multi_linear_result()
    callback = callbacks.MultiLinearRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=build_multi_trait_prediction_source(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_linear_state.Regenie2MultiLinearChromosomeState",
        "chromosome-state",
    )
    chunk_stats = typing.cast("typing.Any", LinearNativeSumChunkStats())
    packed_probability_pairs_by_variant = build_packed_probability_pairs_by_variant()

    with (
        patch(
            "g.compute.regenie2_linear.api.compute_multi_linear_chunk_packed8_donating_inputs",
            return_value=result,
        ) as mock_packed_compute,
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major",
        ) as mock_variant_major_compute,
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_multi_linear_chunk_from_chromosome_state",
        ) as mock_sample_major_compute,
    ):
        callback.compute_preprocessed_variant_major_packed8_probability_pair_chunk(
            metadata=build_native_metadata(),
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            chunk_stats=chunk_stats,
        )
        callback.finish()

    np.testing.assert_array_equal(
        np.asarray(mock_packed_compute.call_args.kwargs["packed_probability_pairs_by_variant"]),
        packed_probability_pairs_by_variant,
    )
    np.testing.assert_array_equal(np.asarray(mock_packed_compute.call_args.kwargs["genotype_dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(
        np.asarray(mock_packed_compute.call_args.kwargs["genotype_observation_count"]), [2, 2]
    )
    np.testing.assert_array_equal(
        np.asarray(mock_packed_compute.call_args.kwargs["genotype_imputed_dosage_square_sum"]),
        [5.0, 13.0],
    )
    mock_variant_major_compute.assert_not_called()
    mock_sample_major_compute.assert_not_called()
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)
    assert writer_sessions[0].native_chunks[0]["chunk_stats"] is chunk_stats


def test_binary_callback_fails_when_null_logistic_does_not_converge() -> None:
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )

    try:
        with (
            patch(
                "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
                return_value=build_binary_chromosome_state(converged=False),
            ),
            pytest.raises(RuntimeError, match="Binary null logistic model did not converge for chromosome 22"),
        ):
            callback.prepare_chromosome_state(build_native_metadata())
    finally:
        callback.finish()


def test_binary_callback_warn_policy_allows_null_logistic_nonconvergence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.WARN,
    )

    try:
        with (
            caplog.at_level("WARNING", logger="g.engine.callbacks"),
            patch(
                "g.compute.regenie2_binary.api.prepare_regenie2_binary_chromosome_state",
                return_value=build_binary_chromosome_state(converged=False),
            ),
        ):
            callback.prepare_chromosome_state(build_native_metadata())
    finally:
        callback.finish()

    assert callback.current_chromosome == "22"
    assert any("--null_logistic_nonconvergence_policy=warn" in record.message for record in caplog.records)


def test_multi_binary_callback_fails_when_any_null_logistic_trait_does_not_converge() -> None:
    callback = callbacks.MultiBinaryRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=(FakeWriterSession(), FakeWriterSession()),
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )

    try:
        with (
            patch(
                "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_chromosome_state",
                return_value=build_multi_binary_chromosome_state(convergence_flags=(True, False)),
            ),
            pytest.raises(RuntimeError, match="chromosome 22: trait_b"),
        ):
            callback.prepare_chromosome_state(build_native_metadata())
    finally:
        callback.finish()


def test_multi_binary_score_only_sample_major_callback_skips_sparse_mask_transfer() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    result = regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult(
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([[True, True], [True, True]]),
    )
    callback = callbacks.MultiBinaryRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )
    chunk_stats = typing.cast("typing.Any", ExplodingSparseCandidateChunkStats())
    chromosome_state = build_multi_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_multi_binary_chunk_from_chromosome_state",
            return_value=result,
        ) as mock_compute,
    ):
        callback.compute_preprocessed_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )
        callback.finish()

    assert mock_compute.call_args.kwargs["sparse_candidate_mask"] is None
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)


def test_multi_binary_score_only_variant_major_callback_uses_donated_score_compute() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    result = regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult(
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value],
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([[True, True], [True, True]]),
    )
    callback = callbacks.MultiBinaryRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )
    variant_major_genotype_matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    chunk_stats = SimpleNamespace(
        dosage_sum=np.asarray([3.0, 7.0], dtype=np.float32),
        observation_count=np.asarray([2, 2], dtype=np.int32),
        is_rare_sparse_firth_candidate=np.asarray([True, False], dtype=np.bool_),
    )
    chromosome_state = build_multi_binary_chromosome_state()

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_chromosome_state",
            return_value=chromosome_state,
        ),
        patch(
            "g.compute.regenie2_binary.api.compute_multi_binary_score_test_variant_major_donating_inputs",
            return_value=result,
        ) as mock_score_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major",
        ) as mock_chunk_compute,
    ):
        callback.compute_preprocessed_variant_major_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix_by_variant=variant_major_genotype_matrix,
            chunk_stats=typing.cast("typing.Any", chunk_stats),
        )
        callback.finish()

    np.testing.assert_array_equal(
        np.asarray(mock_score_compute.call_args.kwargs["genotype_matrix_by_variant"]),
        variant_major_genotype_matrix,
    )
    np.testing.assert_array_equal(np.asarray(mock_score_compute.call_args.kwargs["dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(np.asarray(mock_score_compute.call_args.kwargs["observation_count"]), [2, 2])
    mock_chunk_compute.assert_not_called()
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)


def test_multi_binary_score_only_packed8_callback_uses_packed_score_compute() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    result = build_multi_binary_score_result()
    callback = callbacks.MultiBinaryRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=build_default_binary_kernel_config(),
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_binary_state.Regenie2MultiBinaryChromosomeState",
        "chromosome-state",
    )
    packed_probability_pairs_by_variant = build_packed_probability_pairs_by_variant()
    chunk_stats = typing.cast("typing.Any", ExplodingSparseCandidateChunkStats())

    with (
        patch(
            "g.compute.regenie2_binary.api.compute_multi_binary_score_test_packed8_donating_inputs",
            return_value=result,
        ) as mock_score_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8",
        ) as mock_chunk_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_multi_binary_score_test_variant_major_donating_inputs",
        ) as mock_variant_major_score_compute,
    ):
        callback.compute_preprocessed_variant_major_packed8_probability_pair_chunk(
            metadata=build_native_metadata(),
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            chunk_stats=chunk_stats,
        )
        callback.finish()

    np.testing.assert_array_equal(
        np.asarray(mock_score_compute.call_args.kwargs["packed_probability_pairs_by_variant"]),
        packed_probability_pairs_by_variant,
    )
    np.testing.assert_array_equal(np.asarray(mock_score_compute.call_args.kwargs["dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(np.asarray(mock_score_compute.call_args.kwargs["observation_count"]), [2, 2])
    mock_chunk_compute.assert_not_called()
    mock_variant_major_score_compute.assert_not_called()
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)


def test_multi_binary_variant_major_callback_forwards_non_default_kernel_config() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    stage_timing_recorder = timing.StageTimingRecorder()
    kernel_config = dataclasses.replace(
        build_default_binary_kernel_config(),
        null_logistic=dataclasses.replace(
            build_default_binary_kernel_config().null_logistic,
            maximum_iterations=3,
            coefficient_tolerance=1.0e-12,
        ),
        firth_candidate=dataclasses.replace(
            build_default_binary_kernel_config().firth_candidate,
            batch_size=1,
        ),
        approximate_firth=dataclasses.replace(
            build_default_binary_kernel_config().approximate_firth,
            maximum_iterations=3,
            gradient_tolerance=1.0e-8,
            coefficient_tolerance=1.0e-8,
            likelihood_tolerance=1.0e-8,
            maximum_step_size=1.0,
            pseudo_maximum_iterations=2,
            pseudo_inner_maximum_iterations=2,
            newton_raphson_zero_start_iterations=2,
            line_search_maximum_attempts=2,
            step_halving_maximum_attempts=2,
            use_block_math=True,
        ),
        null_firth=dataclasses.replace(
            build_default_binary_kernel_config().null_firth,
            maximum_iterations=3,
            gradient_tolerance=1.0e-8,
            maximum_step_size=1.0,
            fallback_iteration_multiplier=2,
            fallback_step_divisor=2.0,
            line_search_maximum_attempts=2,
        ),
    )
    result = regenie2_binary_result.Regenie2MultiBinaryChunkResult(
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
                [types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.FIRTH.value],
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([[True, True], [True, True]]),
        firth_iteration_count=jnp.asarray([[0, 2], [0, 2]], dtype=jnp.int32),
        firth_failure_code=jnp.asarray(
            [
                [types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value],
                [types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value],
            ],
            dtype=jnp.int32,
        ),
        firth_convergence_reason_code=jnp.asarray(
            [
                [
                    regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                ],
                [
                    regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                    regenie2_binary_firth_types.FirthConvergenceReason.CONVERGED.value,
                ],
            ],
            dtype=jnp.int32,
        ),
        firth_correction_code=jnp.zeros((2, 2), dtype=jnp.int32),
        firth_sparse_correction_mask=jnp.zeros((2, 2), dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.zeros((2, 2), dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.zeros((2, 2), dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.zeros((2, 2), dtype=jnp.int32),
    )
    chromosome_state = build_multi_binary_chromosome_state()
    callback = callbacks.MultiBinaryRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
        kernel_config=kernel_config,
        stage_timing_recorder=stage_timing_recorder,
    )
    variant_major_genotype_matrix = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    chunk_stats = SimpleNamespace(
        dosage_sum=np.asarray([3.0, 7.0], dtype=np.float32),
        observation_count=np.asarray([2, 2], dtype=np.int32),
        is_rare_sparse_firth_candidate=np.asarray([True, False], dtype=np.bool_),
    )

    with (
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_chromosome_state",
            return_value=chromosome_state,
        ) as mock_prepare,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major",
            return_value=result,
        ) as mock_compute,
    ):
        callback.compute_preprocessed_variant_major_dosage_chunk(
            metadata=build_native_metadata(),
            genotype_matrix_by_variant=variant_major_genotype_matrix,
            chunk_stats=typing.cast("typing.Any", chunk_stats),
        )
        callback.finish()

    assert mock_prepare.call_args.args[3] is kernel_config
    genotype_matrix_by_variant = mock_compute.call_args.kwargs["genotype_matrix_by_variant"]
    np.testing.assert_array_equal(np.asarray(genotype_matrix_by_variant), variant_major_genotype_matrix)
    sparse_candidate_mask = mock_compute.call_args.kwargs["sparse_candidate_mask"]
    np.testing.assert_array_equal(np.asarray(sparse_candidate_mask), [True, False])
    dosage_sum = mock_compute.call_args.kwargs["dosage_sum"]
    np.testing.assert_array_equal(np.asarray(dosage_sum), [3.0, 7.0])
    observation_count = mock_compute.call_args.kwargs["observation_count"]
    np.testing.assert_array_equal(np.asarray(observation_count), [2, 2])
    assert mock_compute.call_args.kwargs["kernel_config"] is kernel_config
    stage_duration_recorder = typing.cast(
        "typing.Callable[[str, float], None]",
        mock_compute.call_args.kwargs["stage_duration_recorder"],
    )
    stage_duration_recorder("firth_candidate_dispatch_plan", 0.0)
    assert stage_timing_recorder.snapshot().stage_counts["firth_candidate_dispatch_plan"] == 1
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)


def test_multi_binary_approximate_firth_packed8_callback_uses_packed_chunk_compute() -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    stage_timing_recorder = timing.StageTimingRecorder()
    kernel_config = dataclasses.replace(
        build_default_binary_kernel_config(),
        firth_candidate=dataclasses.replace(
            build_default_binary_kernel_config().firth_candidate,
            batch_size=1,
        ),
    )
    result = build_multi_binary_chunk_result()
    callback = callbacks.MultiBinaryRegenie2PipelineCallback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
        kernel_config=kernel_config,
        stage_timing_recorder=stage_timing_recorder,
    )
    callback.current_chromosome = "22"
    callback.current_chromosome_state = typing.cast(
        "regenie2_binary_state.Regenie2MultiBinaryChromosomeState",
        "chromosome-state",
    )
    packed_probability_pairs_by_variant = build_packed_probability_pairs_by_variant()
    chunk_stats = SimpleNamespace(
        dosage_sum=np.asarray([3.0, 7.0], dtype=np.float32),
        observation_count=np.asarray([2, 2], dtype=np.int32),
        is_rare_sparse_firth_candidate=np.asarray([True, False], dtype=np.bool_),
    )

    with (
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8",
            return_value=result,
        ) as mock_chunk_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_multi_binary_score_test_packed8_donating_inputs",
        ) as mock_score_compute,
        patch(
            "g.compute.regenie2_binary.api.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major",
        ) as mock_variant_major_compute,
    ):
        callback.compute_preprocessed_variant_major_packed8_probability_pair_chunk(
            metadata=build_native_metadata(),
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            chunk_stats=typing.cast("typing.Any", chunk_stats),
        )
        callback.finish()

    np.testing.assert_array_equal(
        np.asarray(mock_chunk_compute.call_args.kwargs["packed_probability_pairs_by_variant"]),
        packed_probability_pairs_by_variant,
    )
    sparse_candidate_mask = mock_chunk_compute.call_args.kwargs["sparse_candidate_mask"]
    np.testing.assert_array_equal(np.asarray(sparse_candidate_mask), [True, False])
    np.testing.assert_array_equal(np.asarray(mock_chunk_compute.call_args.kwargs["dosage_sum"]), [3.0, 7.0])
    np.testing.assert_array_equal(np.asarray(mock_chunk_compute.call_args.kwargs["observation_count"]), [2, 2])
    assert mock_chunk_compute.call_args.kwargs["kernel_config"] is kernel_config
    assert callable(mock_chunk_compute.call_args.kwargs["stage_duration_recorder"])
    mock_score_compute.assert_not_called()
    mock_variant_major_compute.assert_not_called()
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)


def test_run_linear_bgen_pipeline_invokes_native_engine_and_writer() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()
    preparation_order: list[str] = []

    def record_preflight(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        del kwargs
        preparation_order.append("preflight")
        return SimpleNamespace(sample_count=2, covariate_count=1, chromosome_count=1)

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_run_input", return_value=run_input),
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: preparation_order.append("writer") or writer_session,
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            return_value={"header": "current"},
        ) as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            side_effect=lambda **kwargs: (
                preparation_order.append("manifest")
                or output.InitializedOutputRun(committed_chunk_identifiers=frozenset({64, 0}))
            ),
        ),
        patch(
            "g.engine.regenie2_pipeline.preflight.run_regenie2_preflight",
            side_effect=record_preflight,
        ) as mock_preflight,
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2LinearState", "state"),
        ),
    ):
        final_path = regenie2_pipeline.run_regenie2_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest={"header": "current", "committed_chunks": []},
            resume=True,
            finalize_parquet=True,
            writer_thread_count=2,
            writer_queue_depth=3,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            trusted_no_missing_diploid=True,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            output_initialized_callback=lambda phenotype_names: preparation_order.append("metadata"),
        )

    assert final_path == Path("results/final.parquet")
    assert preparation_order == ["preflight", "manifest", "metadata", "writer"]
    assert writer_session.finished is True
    engine = FakeRunEngine.instances[0]
    assert engine.bgen_path == "study.bgen"
    assert engine.chunk_size == 32
    assert engine.variant_limit == 100
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_buffered"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.LinearRegenie2PipelineCallback)
    assert callback.dosage_queue_depth == 3
    assert callback.dosage_buffer_limit == 4
    assert committed_chunk_identifiers == [0, 64]
    assert mock_preflight.call_args.kwargs["variant_limit"] == 100
    prediction_source = FakePredictionSource.instances[0]
    assert prediction_source.prediction_list_path == "pred.list"
    assert prediction_source.phenotype_name == "trait"
    assert prediction_source.native_aligned_sample_data is run_input.native_aligned_sample_data
    assert prediction_source.sample_key_mode == "iid"
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_DOSAGE


def test_single_trait_preflight_failure_does_not_initialize_output_or_writer(tmp_path: Path) -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()
    output_run_paths = output.OutputRunPaths(tmp_path / "run", tmp_path / "run/chunks")

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
        patch("g.engine.native_dispatch.load_native_bgen_run_input", return_value=run_input),
        patch(
            "g.engine.regenie2_pipeline.preflight.run_regenie2_preflight",
            side_effect=ValueError("invalid preflight"),
        ) as mock_preflight,
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_manifest_header,
        patch("g.engine.regenie2_pipeline.output.initialize_output_run") as mock_initialize_output_run,
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session") as mock_create_writer_session,
        pytest.raises(ValueError, match="invalid preflight"),
    ):
        regenie2_pipeline.run_regenie2_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output_run_paths,
            staging_depth=3,
            existing_manifest=None,
            resume=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
        )

    mock_preflight.assert_called_once()
    mock_manifest_header.assert_not_called()
    mock_initialize_output_run.assert_not_called()
    mock_create_writer_session.assert_not_called()
    assert not output.get_run_manifest_path(output_run_paths).exists()


def test_multi_resume_manifest_mismatch_does_not_partially_initialize_outputs(tmp_path: Path) -> None:
    first_output_run_paths = output.OutputRunPaths(tmp_path / "one.run", tmp_path / "one.run/chunks")
    second_output_run_paths = output.OutputRunPaths(tmp_path / "two.run", tmp_path / "two.run/chunks")
    first_output_run_paths.chunks_directory.mkdir(parents=True)
    second_output_run_paths.chunks_directory.mkdir(parents=True)
    first_header = {"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION, "phenotype_name": "one", "chunk_size": 32}
    second_manifest_header = {
        "schema_version": output.RUN_MANIFEST_SCHEMA_VERSION,
        "phenotype_name": "two",
        "chunk_size": 32,
    }
    second_current_header = {
        "schema_version": output.RUN_MANIFEST_SCHEMA_VERSION,
        "phenotype_name": "two",
        "chunk_size": 64,
    }
    first_manifest_bytes = write_test_run_manifest(first_output_run_paths, first_header)
    second_manifest_bytes = write_test_run_manifest(second_output_run_paths, second_manifest_header)

    with pytest.raises(ValueError, match="chunk_size"):
        regenie2_pipeline.initialize_pipeline_output_runs(
            output_run_paths_by_trait=(first_output_run_paths, second_output_run_paths),
            existing_manifests_by_trait=(
                {**first_header, "committed_chunks": []},
                {**second_manifest_header, "committed_chunks": []},
            ),
            current_headers_by_trait=(first_header, second_current_header),
            resume=True,
            resume_mode=types.ResumeMode.FAST,
        )

    assert output.get_run_manifest_path(first_output_run_paths).read_bytes() == first_manifest_bytes
    assert output.get_run_manifest_path(second_output_run_paths).read_bytes() == second_manifest_bytes


def test_linear_pipeline_invokes_packed8_engine_and_forces_trusted_validation() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_run_input", return_value=run_input),
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session", return_value=writer_session),
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset({64, 0})),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2LinearState", "state"),
        ),
    ):
        mock_manifest_header.return_value = {"header": "current"}
        final_path = regenie2_pipeline.run_regenie2_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest={"header": "current", "committed_chunks": []},
            resume=True,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        )

    assert final_path == Path("results/final.parquet")
    engine = FakeRunEngine.instances[0]
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_packed8"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.LinearRegenie2PipelineCallback)
    assert committed_chunk_identifiers == [0, 64]
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_PACKED8
    assert mock_manifest_header.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.PACKED8
    assert mock_manifest_header.call_args.kwargs["trusted_no_missing_diploid"] is True


class FinishTrackingCallback:
    def __init__(self) -> None:
        self.finished = False
        self.aborted = False

    def finish(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted = True


class GracefulShutdownRunEngine(FakeRunEngine):
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        raise shutdown.GracefulShutdownRequested(shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130))


class HardInterruptRunEngine(FakeRunEngine):
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        raise KeyboardInterrupt


def test_native_dispatch_graceful_shutdown_drains_and_marks_writer_interrupted() -> None:
    engine = GracefulShutdownRunEngine("study.bgen", chunk_size=32)
    callback = FinishTrackingCallback()
    writer_session = FakeWriterSession()

    with pytest.raises(shutdown.GracefulShutdownRequested):
        native_dispatch.run_bgen_engine_with_callback(
            engine=typing.cast("typing.Any", engine),
            run_input=build_native_run_input(),
            committed_chunk_identifiers={0},
            writer_session=writer_session,
            callback=callback,
            stage_timing_recorder=None,
        )

    assert callback.finished is True
    assert callback.aborted is False
    assert writer_session.interrupted_signal_name == "SIGINT"
    assert writer_session.finished is False
    assert writer_session.aborted is False


def test_native_dispatch_hard_interrupt_aborts_callback_and_writer() -> None:
    engine = HardInterruptRunEngine("study.bgen", chunk_size=32)
    callback = FinishTrackingCallback()
    writer_session = FakeWriterSession()

    with pytest.raises(KeyboardInterrupt):
        native_dispatch.run_bgen_engine_with_callback(
            engine=typing.cast("typing.Any", engine),
            run_input=build_native_run_input(),
            committed_chunk_identifiers={0},
            writer_session=writer_session,
            callback=callback,
            stage_timing_recorder=None,
        )

    assert callback.finished is False
    assert callback.aborted is True
    assert writer_session.interrupted_signal_name is None
    assert writer_session.finished is False
    assert writer_session.aborted is True


def test_native_dispatch_records_profile_and_allows_no_final_path() -> None:
    engine = FakeRunEngine("study.bgen", chunk_size=32)
    callback = FinishTrackingCallback()
    writer_session = NoFinalWriterSession()
    stage_timing_recorder = timing.StageTimingRecorder()
    snapshot_calls: list[tuple[timing.StageTimingRecorder | None, Path | None]] = []

    def record_snapshot(
        recorder: timing.StageTimingRecorder | None,
        stage_timing_path: Path | None,
    ) -> None:
        snapshot_calls.append((recorder, stage_timing_path))

    final_path = native_dispatch.run_bgen_engine_with_callback(
        engine=typing.cast("typing.Any", engine),
        run_input=build_native_run_input(),
        committed_chunk_identifiers={2, 1},
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        stage_timing_snapshot_writer=record_snapshot,
    )

    assert final_path is None
    assert callback.finished is True
    assert writer_session.finished is True
    assert writer_session.aborted is False
    assert engine.reset_profile_count == 1
    assert engine.run_arguments is not None
    assert engine.run_arguments[2] == [1, 2]
    assert stage_timing_recorder.snapshot().native_bgen_profile == {"variant_decode_count": 7}
    assert len(snapshot_calls) == 1


def test_multi_dispatch_graceful_shutdown_drains_and_marks_all_writers_interrupted() -> None:
    engine = GracefulShutdownRunEngine("study.bgen", chunk_size=32)
    callback = FinishTrackingCallback()
    writer_sessions = (FakeWriterSession(), FakeWriterSession())

    with pytest.raises(shutdown.GracefulShutdownRequested):
        regenie2_pipeline.run_bgen_engine_with_multi_callback(
            engine=typing.cast("typing.Any", engine),
            run_input=build_native_multi_run_input(),
            committed_chunk_identifiers={0},
            writer_sessions=writer_sessions,
            callback=callback,
            stage_timing_recorder=None,
        )

    assert callback.finished is True
    assert callback.aborted is False
    assert tuple(writer_session.interrupted_signal_name for writer_session in writer_sessions) == ("SIGINT", "SIGINT")
    assert tuple(writer_session.finished for writer_session in writer_sessions) == (False, False)
    assert tuple(writer_session.aborted for writer_session in writer_sessions) == (False, False)


def test_binary_pipeline_invokes_variant_major_engine_for_trusted_bgen() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    kernel_config = build_default_binary_kernel_config()
    pipeline_options = build_default_pipeline_runtime_options()
    preparation_order: list[str] = []

    def record_preflight(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        del kwargs
        preparation_order.append("preflight")
        return SimpleNamespace(sample_count=2, covariate_count=1, chromosome_count=1)

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_run_input", return_value=run_input),
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: preparation_order.append("writer") or writer_session,
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            return_value={"header": "current"},
        ) as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            side_effect=lambda **kwargs: (
                preparation_order.append("manifest")
                or output.InitializedOutputRun(committed_chunk_identifiers=frozenset({64, 0}))
            ),
        ),
        patch(
            "g.engine.regenie2_pipeline.preflight.run_regenie2_preflight",
            side_effect=record_preflight,
        ) as mock_preflight,
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        final_path = regenie2_pipeline.run_regenie2_binary_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest={"header": "current", "committed_chunks": []},
            resume=True,
            trusted_no_missing_diploid=True,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=kernel_config,
        )

    assert final_path == Path("results/final.parquet")
    assert preparation_order == ["preflight", "manifest", "writer"]
    engine = FakeRunEngine.instances[0]
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_buffered"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.BinaryRegenie2PipelineCallback)
    assert callback.kernel_config is kernel_config
    assert committed_chunk_identifiers == [0, 64]
    assert mock_preflight.call_args.kwargs["variant_limit"] == 100
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_DOSAGE


def test_binary_pipeline_invokes_variant_major_engine_for_untrusted_bgen() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
        patch("g.engine.native_dispatch.load_native_bgen_run_input", return_value=run_input),
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session", return_value=writer_session),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header", return_value={"header": "current"}
        ),
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset({64, 0})),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        final_path = regenie2_pipeline.run_regenie2_binary_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest={"header": "current", "committed_chunks": []},
            resume=True,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=build_default_binary_kernel_config(),
        )

    assert final_path == Path("results/final.parquet")
    engine = FakeRunEngine.instances[0]
    assert engine.validation_count == 0
    assert engine.run_method == "variant_major_buffered"
    assert engine.trusted_no_missing_diploid is False


def test_binary_pipeline_invokes_packed8_engine_and_forces_trusted_validation() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_run_input", return_value=run_input),
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session", return_value=writer_session),
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset({64, 0})),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        mock_manifest_header.return_value = {"header": "current"}
        final_path = regenie2_pipeline.run_regenie2_binary_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest={"header": "current", "committed_chunks": []},
            resume=True,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=build_default_binary_kernel_config(),
            gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        )

    assert final_path == Path("results/final.parquet")
    engine = FakeRunEngine.instances[0]
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_packed8"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.BinaryRegenie2PipelineCallback)
    assert committed_chunk_identifiers == [0, 64]
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_PACKED8
    assert mock_manifest_header.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.PACKED8
    assert mock_manifest_header.call_args.kwargs["trusted_no_missing_diploid"] is True


def test_multi_linear_pipeline_opens_engine_once_and_skips_only_shared_committed_chunks() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_input = build_native_multi_run_input()
    preparation_order: list[str] = []
    initialized_chunk_sets = [frozenset({0, 32}), frozenset({32, 64})]
    pipeline_options = build_default_pipeline_runtime_options()

    def record_preflight(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        preparation_order.append("preflight")

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.load_native_bgen_multi_run_input", return_value=run_input),
        patch(
            "g.engine.native_dispatch.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch(
            "g.engine.regenie2_pipeline.run_multi_preflight",
            side_effect=record_preflight,
        ) as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: preparation_order.append("writer") or writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            side_effect=lambda **kwargs: (
                preparation_order.append("manifest")
                or output.InitializedOutputRun(committed_chunk_identifiers=initialized_chunk_sets.pop(0))
            ),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            staging_depth=2,
            existing_manifests_by_phenotype=(
                {"header": "trait_a", "committed_chunks": []},
                {"header": "trait_b", "committed_chunks": []},
            ),
            resume=True,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
            output_initialized_callback=lambda phenotype_names: preparation_order.append("metadata"),
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    assert preparation_order == ["preflight", "manifest", "manifest", "metadata", "writer", "writer"]
    assert len(FakeRunEngine.instances) == 1
    engine = FakeRunEngine.instances[0]
    assert engine.run_method == "variant_major_buffered"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiLinearRegenie2PipelineCallback)
    assert committed_chunk_identifiers == [32]
    assert callback.committed_chunk_identifier_sets == ({0, 32}, {32, 64})
    assert mock_run_multi_preflight.call_args.kwargs["variant_limit"] == 100
    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))


def test_multi_preflight_failure_does_not_initialize_outputs_or_writers(tmp_path: Path) -> None:
    FakeRunEngine.instances.clear()
    run_input = build_native_multi_run_input()
    pipeline_options = build_default_pipeline_runtime_options()
    output_run_paths_by_phenotype = (
        output.OutputRunPaths(tmp_path / "run/a", tmp_path / "run/a/chunks"),
        output.OutputRunPaths(tmp_path / "run/b", tmp_path / "run/b/chunks"),
    )

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.load_native_bgen_multi_run_input", return_value=run_input),
        patch(
            "g.engine.native_dispatch.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch(
            "g.engine.regenie2_pipeline.run_multi_preflight", side_effect=ValueError("invalid multi preflight")
        ) as mock_run_multi_preflight,
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_manifest_header,
        patch("g.engine.regenie2_pipeline.output.initialize_output_run") as mock_initialize_output_run,
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session") as mock_create_writer_session,
        pytest.raises(ValueError, match="invalid multi preflight"),
    ):
        regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=2,
            existing_manifests_by_phenotype=(None, None),
            resume=False,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        )

    mock_run_multi_preflight.assert_called_once()
    mock_manifest_header.assert_not_called()
    mock_initialize_output_run.assert_not_called()
    mock_create_writer_session.assert_not_called()
    assert not output.get_run_manifest_path(output_run_paths_by_phenotype[0]).exists()
    assert not output.get_run_manifest_path(output_run_paths_by_phenotype[1]).exists()


def test_multi_linear_resume_recomputes_partial_chunks_without_duplicate_writes() -> None:
    FakeRunEngine.instances.clear()
    writer_session_for_trait_a = FakeWriterSession()
    writer_session_for_trait_b = FakeWriterSession()
    pending_writer_sessions = [writer_session_for_trait_a, writer_session_for_trait_b]
    run_input = build_native_multi_run_input()
    initialized_chunk_sets = [frozenset({0, 32}), frozenset({32, 64})]
    pipeline_options = build_default_pipeline_runtime_options()
    chromosome_state = SimpleNamespace(adjusted_residual_matrix=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32))

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", PartialCommitDeliveringRunEngine),
        patch("g.engine.native_dispatch.load_native_bgen_multi_run_input", return_value=run_input),
        patch(
            "g.engine.native_dispatch.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.run_multi_preflight"),
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: pending_writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            side_effect=lambda **kwargs: output.InitializedOutputRun(
                committed_chunk_identifiers=initialized_chunk_sets.pop(0)
            ),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_chromosome_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearChromosomeState", chromosome_state),
        ),
        patch(
            "g.compute.regenie2_linear.api.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major",
            return_value=build_multi_linear_result(),
        ) as mock_compute,
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            staging_depth=2,
            existing_manifests_by_phenotype=(
                {"header": "trait_a", "committed_chunks": []},
                {"header": "trait_b", "committed_chunks": []},
            ),
            resume=True,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    engine = FakeRunEngine.instances[0]
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiLinearRegenie2PipelineCallback)
    assert committed_chunk_identifiers == [32]
    assert callback.committed_chunk_identifier_sets == ({0, 32}, {32, 64})
    assert mock_compute.call_count == 2
    assert len(writer_session_for_trait_a.native_chunks) == 1
    assert len(writer_session_for_trait_b.native_chunks) == 1
    trait_a_metadata = typing.cast("typing.Any", writer_session_for_trait_a.native_chunks[0]["metadata"])
    trait_b_metadata = typing.cast("typing.Any", writer_session_for_trait_b.native_chunks[0]["metadata"])
    assert trait_a_metadata.variant_start_index == 64
    assert trait_b_metadata.variant_start_index == 0
    np.testing.assert_array_equal(
        writer_session_for_trait_a.native_chunks[0]["beta"],
        np.asarray([0.1, 0.2], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        writer_session_for_trait_b.native_chunks[0]["beta"],
        np.asarray([0.3, 0.4], dtype=np.float32),
    )
    assert writer_session_for_trait_a.finished is True
    assert writer_session_for_trait_b.finished is True


def test_multi_binary_pipeline_opens_engine_once_and_skips_only_shared_committed_chunks() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_input = build_native_multi_run_input()
    initialized_chunk_sets = [frozenset({0, 32}), frozenset({32, 64})]
    kernel_config = build_default_binary_kernel_config()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.load_native_bgen_multi_run_input", return_value=run_input),
        patch(
            "g.engine.native_dispatch.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            side_effect=lambda **kwargs: output.InitializedOutputRun(
                committed_chunk_identifiers=initialized_chunk_sets.pop(0)
            ),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2MultiBinaryState", "state"),
        ),
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_binary_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            staging_depth=2,
            existing_manifests_by_phenotype=(
                {"header": "trait_a", "committed_chunks": []},
                {"header": "trait_b", "committed_chunks": []},
            ),
            resume=True,
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=kernel_config,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    assert len(FakeRunEngine.instances) == 1
    engine = FakeRunEngine.instances[0]
    assert engine.run_method == "variant_major_buffered"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiBinaryRegenie2PipelineCallback)
    assert callback.kernel_config is kernel_config
    assert committed_chunk_identifiers == [32]
    assert callback.committed_chunk_identifier_sets == ({0, 32}, {32, 64})
    assert mock_run_multi_preflight.call_args.kwargs["variant_limit"] == 100


def test_multi_linear_complete_case_packed8_forces_trusted_delivery_and_manifests() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_input = build_native_multi_run_input()
    planned_compute_groups = execution_plan.build_phenotype_compute_groups(
        phenotype_names=("trait_a", "trait_b"),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.native_dispatch.load_native_bgen_multi_run_input",
            return_value=run_input,
        ) as mock_load_native_multi_run_input,
        patch(
            "g.engine.native_dispatch.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        mock_build_header.side_effect = ({"header": "trait_a"}, {"header": "trait_b"})
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
            phenotype_compute_groups=planned_compute_groups,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    assert mock_load_native_multi_run_input.call_args.kwargs["phenotype_names"] == ("trait_a", "trait_b")
    engine = FakeRunEngine.instances[0]
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_packed8"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiLinearRegenie2PipelineCallback)
    assert committed_chunk_identifiers == []
    assert mock_run_multi_preflight.call_args.kwargs["trusted_no_missing_diploid"] is True
    assert tuple(call.kwargs["gpu_genotype_format"] for call in mock_build_header.call_args_list) == (
        types.GpuGenotypeFormat.PACKED8,
        types.GpuGenotypeFormat.PACKED8,
    )
    assert tuple(call.kwargs["trusted_no_missing_diploid"] for call in mock_build_header.call_args_list) == (
        True,
        True,
    )
    expected_compute_group = native_dispatch.build_resolved_complete_case_phenotype_compute_group(
        run_input=run_input,
        prediction_list_path=Path("pred.list"),
        planned_compute_groups=planned_compute_groups,
        alignment_config=None,
    )
    assert tuple(call.kwargs["multi_phenotype_sample_mode"] for call in mock_build_header.call_args_list) == (
        output.MultiPhenotypeSampleMode.COMPLETE_CASE,
        output.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )
    assert tuple(call.kwargs["sample_set_fingerprint"] for call in mock_build_header.call_args_list) == (
        expected_compute_group.sample_set_fingerprint,
        expected_compute_group.sample_set_fingerprint,
    )
    assert tuple(call.kwargs["covariate_design_fingerprint"] for call in mock_build_header.call_args_list) == (
        expected_compute_group.covariate_design_fingerprint,
        expected_compute_group.covariate_design_fingerprint,
    )
    assert tuple(call.kwargs["prediction_alignment_fingerprint"] for call in mock_build_header.call_args_list) == (
        expected_compute_group.prediction_alignment_fingerprint,
        expected_compute_group.prediction_alignment_fingerprint,
    )


def test_grouped_per_phenotype_pipeline_batches_identical_alignments() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    planned_compute_groups = execution_plan.build_phenotype_compute_groups(
        phenotype_names=("trait_a", "trait_b"),
        multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
    )
    run_inputs = (
        build_native_run_input_with_alignment(
            phenotype_name="trait_a",
            sample_indices=(1, 0),
            phenotype_values=(0.0, 1.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0)),
        ),
        build_native_run_input_with_alignment(
            phenotype_name="trait_b",
            sample_indices=(1, 0),
            phenotype_values=(2.0, 3.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0)),
        ),
    )
    grouped_run_inputs = (
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(0, 1),
            phenotype_names=("trait_a", "trait_b"),
            run_inputs=run_inputs,
        ),
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.load_native_bgen_grouped_run_inputs",
            return_value=grouped_run_inputs,
        ) as mock_load_grouped_run_inputs,
        patch("g.engine.regenie2_pipeline.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ) as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
            phenotype_compute_groups=planned_compute_groups,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    assert mock_load_grouped_run_inputs.call_args.kwargs["planned_compute_groups"] == planned_compute_groups
    assert len(FakeRunEngine.instances) == 1
    engine = FakeRunEngine.instances[0]
    assert len(engine.run_call_arguments) == 1
    sample_indices, callback, committed_chunk_identifiers = engine.run_call_arguments[0]
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiLinearRegenie2PipelineCallback)
    assert callback.run_input.phenotype_names == ("trait_a", "trait_b")
    assert grouped_run_inputs[0].compute_group.phenotype_indices == (0, 1)
    assert grouped_run_inputs[0].compute_group.phenotype_names == ("trait_a", "trait_b")
    assert grouped_run_inputs[0].compute_group.sample_set_fingerprint is not None
    assert grouped_run_inputs[0].compute_group.covariate_design_fingerprint is not None
    assert grouped_run_inputs[0].compute_group.prediction_alignment_fingerprint is not None
    assert committed_chunk_identifiers == []
    assert mock_run_multi_preflight.call_args.kwargs["run_input"].phenotype_names == ("trait_a", "trait_b")
    assert tuple(call.kwargs["multi_phenotype_sample_mode"] for call in mock_build_header.call_args_list) == (
        output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
    )
    assert tuple(call.kwargs["sample_set_fingerprint"] for call in mock_build_header.call_args_list) == (
        grouped_run_inputs[0].compute_group.sample_set_fingerprint,
        grouped_run_inputs[0].compute_group.sample_set_fingerprint,
    )
    assert tuple(call.kwargs["covariate_design_fingerprint"] for call in mock_build_header.call_args_list) == (
        grouped_run_inputs[0].compute_group.covariate_design_fingerprint,
        grouped_run_inputs[0].compute_group.covariate_design_fingerprint,
    )
    assert tuple(call.kwargs["prediction_alignment_fingerprint"] for call in mock_build_header.call_args_list) == (
        grouped_run_inputs[0].compute_group.prediction_alignment_fingerprint,
        grouped_run_inputs[0].compute_group.prediction_alignment_fingerprint,
    )


def test_grouped_per_phenotype_packed8_forces_trusted_delivery_and_manifests() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_inputs = (
        build_native_run_input_with_alignment(
            phenotype_name="trait_a",
            sample_indices=(1, 0),
            phenotype_values=(0.0, 1.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0)),
        ),
        build_native_run_input_with_alignment(
            phenotype_name="trait_b",
            sample_indices=(1, 0),
            phenotype_values=(2.0, 3.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0)),
        ),
    )
    grouped_run_inputs = (
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(0, 1),
            phenotype_names=("trait_a", "trait_b"),
            run_inputs=run_inputs,
        ),
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_grouped_run_inputs", return_value=grouped_run_inputs),
        patch("g.engine.regenie2_pipeline.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        mock_build_header.side_effect = ({"header": "trait_a"}, {"header": "trait_b"})
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
            sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    engine = FakeRunEngine.instances[0]
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_packed8"
    assert len(engine.run_call_arguments) == 1
    sample_indices, callback, committed_chunk_identifiers = engine.run_call_arguments[0]
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiLinearRegenie2PipelineCallback)
    assert committed_chunk_identifiers == []
    assert mock_run_multi_preflight.call_args.kwargs["trusted_no_missing_diploid"] is True
    assert tuple(call.kwargs["gpu_genotype_format"] for call in mock_build_header.call_args_list) == (
        types.GpuGenotypeFormat.PACKED8,
        types.GpuGenotypeFormat.PACKED8,
    )
    assert tuple(call.kwargs["trusted_no_missing_diploid"] for call in mock_build_header.call_args_list) == (
        True,
        True,
    )


def test_grouped_per_phenotype_pipeline_splits_different_alignments() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_inputs = (
        build_native_run_input_with_alignment(
            phenotype_name="trait_a",
            sample_indices=(1, 0),
            phenotype_values=(0.0, 1.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0)),
        ),
        build_native_run_input_with_alignment(
            phenotype_name="trait_b",
            sample_indices=(0, 1),
            phenotype_values=(3.0, 2.0),
            covariate_values=((1.0, 50.0), (1.0, 40.0)),
        ),
    )
    grouped_run_inputs = (
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(0,),
            phenotype_names=("trait_a",),
            run_inputs=(run_inputs[0],),
        ),
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(1,),
            phenotype_names=("trait_b",),
            run_inputs=(run_inputs[1],),
        ),
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch("g.engine.native_dispatch.load_native_bgen_grouped_run_inputs", return_value=grouped_run_inputs),
        patch("g.engine.regenie2_pipeline.run_multi_preflight"),
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    engine = FakeRunEngine.instances[0]
    assert len(engine.run_call_arguments) == 2
    np.testing.assert_array_equal(engine.run_call_arguments[0][0], np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(engine.run_call_arguments[1][0], np.asarray([0, 1], dtype=np.int64))


def test_grouped_per_phenotype_pipeline_uses_union_decode_for_overlapping_alignments() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    telemetry_session = RecordingTelemetrySession()
    run_inputs = (
        build_native_run_input_with_alignment(
            phenotype_name="trait_a",
            sample_indices=(0, 1, 2),
            phenotype_values=(0.0, 1.0, 2.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0), (1.0, 60.0)),
        ),
        build_native_run_input_with_alignment(
            phenotype_name="trait_b",
            sample_indices=(1, 2),
            phenotype_values=(3.0, 4.0),
            covariate_values=((1.0, 50.0), (1.0, 60.0)),
        ),
    )
    grouped_run_inputs = (
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(0,),
            phenotype_names=("trait_a",),
            run_inputs=(run_inputs[0],),
        ),
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(1,),
            phenotype_names=("trait_b",),
            run_inputs=(run_inputs[1],),
        ),
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_grouped_run_inputs", return_value=grouped_run_inputs),
        patch("g.engine.regenie2_pipeline.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ) as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=True,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
            telemetry_session=typing.cast("typing.Any", telemetry_session),
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    engine = FakeRunEngine.instances[0]
    assert engine.validation_count == 1
    assert len(engine.run_call_arguments) == 1
    sample_indices, callback, committed_chunk_identifiers = engine.run_call_arguments[0]
    np.testing.assert_array_equal(sample_indices, np.asarray([0, 1, 2], dtype=np.int64))
    assert isinstance(callback, callbacks.GroupedMultiPhenotypeFanoutCallback)
    np.testing.assert_array_equal(callback.group_fanouts[0].sample_position_array, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(callback.group_fanouts[1].sample_position_array, np.asarray([1, 2]))
    assert committed_chunk_identifiers == []
    assert mock_run_multi_preflight.call_count == 2
    assert tuple(call.kwargs["sample_count"] for call in mock_build_header.call_args_list) == (3, 2)
    assert tuple(call.kwargs["sample_set_fingerprint"] for call in mock_build_header.call_args_list) == (
        grouped_run_inputs[0].compute_group.sample_set_fingerprint,
        grouped_run_inputs[1].compute_group.sample_set_fingerprint,
    )
    assert tuple(call.kwargs["multi_phenotype_sample_mode"] for call in mock_build_header.call_args_list) == (
        output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
    )
    summary_events = [
        fields for event_name, fields in telemetry_session.events if event_name == "multi_phenotype_sample_summary"
    ]
    assert summary_events == [
        {
            "association_mode": "regenie2_linear",
            "multi_phenotype_sample_mode": "per-phenotype",
            "phenotype_count": 2,
            "phenotype_group_count": 2,
            "sample_counts": [3, 2],
            "sample_counts_differ": True,
            "shared_sample_set": False,
        }
    ]


def test_grouped_per_phenotype_pipeline_keeps_multi_pass_when_union_not_cheaper() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_inputs = (
        build_native_run_input_with_alignment(
            phenotype_name="trait_a",
            sample_indices=(0, 1),
            phenotype_values=(0.0, 1.0),
            covariate_values=((1.0, 40.0), (1.0, 50.0)),
        ),
        build_native_run_input_with_alignment(
            phenotype_name="trait_b",
            sample_indices=(2, 3),
            phenotype_values=(3.0, 4.0),
            covariate_values=((1.0, 60.0), (1.0, 70.0)),
        ),
    )
    grouped_run_inputs = (
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(0,),
            phenotype_names=("trait_a",),
            run_inputs=(run_inputs[0],),
        ),
        build_grouped_run_input_from_single_trait_inputs(
            phenotype_indices=(1,),
            phenotype_names=("trait_b",),
            run_inputs=(run_inputs[1],),
        ),
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_grouped_run_inputs", return_value=grouped_run_inputs),
        patch("g.engine.regenie2_pipeline.run_multi_preflight"),
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=True,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    engine = FakeRunEngine.instances[0]
    assert engine.validation_count == 1
    assert len(engine.run_call_arguments) == 2
    np.testing.assert_array_equal(engine.run_call_arguments[0][0], np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(engine.run_call_arguments[1][0], np.asarray([2, 3], dtype=np.int64))


def test_multi_binary_complete_case_packed8_preserves_kernel_config_and_manifests() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_input = build_native_multi_run_input()
    kernel_config = dataclasses.replace(
        build_default_binary_kernel_config(),
        firth_candidate=dataclasses.replace(
            build_default_binary_kernel_config().firth_candidate,
            batch_size=3,
        ),
    )
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.native_dispatch.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch("g.engine.native_dispatch.load_native_bgen_multi_run_input", return_value=run_input),
        patch(
            "g.engine.native_dispatch.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch("g.engine.regenie2_pipeline.output.build_current_run_manifest_header") as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.output.initialize_output_run",
            return_value=output.InitializedOutputRun(committed_chunk_identifiers=frozenset()),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2MultiBinaryState", "state"),
        ),
    ):
        mock_build_header.side_effect = ({"header": "trait_a"}, {"header": "trait_b"})
        final_paths = regenie2_pipeline.run_regenie2_multi_phenotype_binary_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=kernel_config,
            gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        )

    assert final_paths == (Path("results/final.parquet"), Path("results/final.parquet"))
    engine = FakeRunEngine.instances[0]
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_packed8"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, callbacks.MultiBinaryRegenie2PipelineCallback)
    assert callback.kernel_config is kernel_config
    assert committed_chunk_identifiers == []
    assert mock_run_multi_preflight.call_args.kwargs["trusted_no_missing_diploid"] is True
    assert tuple(call.kwargs["gpu_genotype_format"] for call in mock_build_header.call_args_list) == (
        types.GpuGenotypeFormat.PACKED8,
        types.GpuGenotypeFormat.PACKED8,
    )
    assert tuple(call.kwargs["trusted_no_missing_diploid"] for call in mock_build_header.call_args_list) == (
        True,
        True,
    )
    assert tuple(call.kwargs["binary_kernel_config"] for call in mock_build_header.call_args_list) == (
        kernel_config,
        kernel_config,
    )


def test_multi_linear_pipeline_rejects_missing_sample_mode() -> None:
    pipeline_options = build_default_pipeline_runtime_options()

    with pytest.raises(ValueError, match="per-phenotype or complete-case"):
        regenie2_pipeline.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_names=("trait_a", "trait_b"),
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths_by_phenotype=(
                output.OutputRunPaths(Path("run/a"), Path("run/a/chunks")),
                output.OutputRunPaths(Path("run/b"), Path("run/b/chunks")),
            ),
            trusted_no_missing_diploid=False,
            writer_thread_count=pipeline_options.writer_thread_count,
            writer_queue_depth=pipeline_options.writer_queue_depth,
            chunks_per_arrow_file=pipeline_options.chunks_per_arrow_file,
            parquet_compression=pipeline_options.parquet_compression,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
        )


def test_build_bgen_run_engine_rejects_assumed_trusted_validation() -> None:
    FakeRunEngine.instances.clear()

    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine),
        pytest.raises(ValueError, match="assume_validated"),
    ):
        native_dispatch.build_bgen_run_engine(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
            trusted_bgen_validation_mode=types.TrustedBgenValidationMode.ASSUME_VALIDATED,
        )


def test_build_bgen_run_engine_caches_trusted_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRunEngine.instances.clear()
    bgen_path = tmp_path / "study.bgen"
    bgen_path.write_bytes(b"bgen")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine):
        first_engine = native_dispatch.build_bgen_run_engine(
            genotype_source_config=source.build_bgen_source_config(bgen_path),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
        )
        second_engine = native_dispatch.build_bgen_run_engine(
            genotype_source_config=source.build_bgen_source_config(bgen_path),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
        )

    first_fake_engine = typing.cast("FakeRunEngine", first_engine)
    second_fake_engine = typing.cast("FakeRunEngine", second_engine)
    assert first_fake_engine.validation_count == 1
    assert second_fake_engine.validation_count == 0


def test_build_bgen_run_engine_force_validates_trusted_bgen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRunEngine.instances.clear()
    bgen_path = tmp_path / "study.bgen"
    bgen_path.write_bytes(b"bgen")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with patch("g.engine.native_dispatch._core.Regenie2RunEngine", FakeRunEngine):
        engine = native_dispatch.build_bgen_run_engine(
            genotype_source_config=source.build_bgen_source_config(bgen_path),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
            trusted_bgen_validation_mode=types.TrustedBgenValidationMode.FORCE_VALIDATE,
        )

    fake_engine = typing.cast("FakeRunEngine", engine)
    assert fake_engine.validation_count == 1


def test_bgen_source_config_rejects_non_bgen_suffix_before_engine_open() -> None:
    with (
        patch("g.engine.native_dispatch._core.Regenie2RunEngine") as mock_run_engine,
        pytest.raises(ValueError, match=r"Expected a \.bgen source path"),
    ):
        native_dispatch.build_bgen_run_engine(
            genotype_source_config=source.build_bgen_source_config(Path("study.vcf")),
            chunk_size=32,
            variant_limit=100,
        )

    mock_run_engine.assert_not_called()


def test_load_native_bgen_run_input_uses_rust_alignment_for_embedded_samples(tmp_path: Path) -> None:
    native_aligned_sample_data = build_native_aligned_sample_data()
    engine = SimpleNamespace(
        sample_count=2,
        contains_embedded_samples=True,
    )
    genotype_source_config = source.build_bgen_source_config(tmp_path / "study.bgen")

    with (
        patch(
            "g.engine.native_dispatch.load_native_aligned_sample_data",
            return_value=native_aligned_sample_data,
        ) as mock_load_aligned_sample_data,
    ):
        run_input = native_dispatch.load_native_bgen_run_input(
            genotype_source_config=genotype_source_config,
            engine=typing.cast("typing.Any", engine),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            is_binary_trait=True,
        )

    assert run_input.native_aligned_sample_data is native_aligned_sample_data
    np.testing.assert_array_equal(run_input.sample_indices, np.asarray([1, 0], dtype=np.int64))
    mock_load_aligned_sample_data.assert_called_once()
    assert mock_load_aligned_sample_data.call_args.kwargs["engine"] is engine
    assert mock_load_aligned_sample_data.call_args.kwargs["sample_path"] is None


def test_load_native_bgen_run_input_uses_rust_sample_file_alignment() -> None:
    native_aligned_sample_data = build_native_aligned_sample_data()
    engine = SimpleNamespace(
        sample_count=2,
        contains_embedded_samples=False,
    )
    sample_path = Path("study.sample")
    genotype_source_config = source.build_bgen_source_config(Path("study.bgen"), sample_path=sample_path)

    with (
        patch(
            "g.engine.native_dispatch.load_native_aligned_sample_data",
            return_value=native_aligned_sample_data,
        ) as mock_load_aligned_sample_data,
    ):
        run_input = native_dispatch.load_native_bgen_run_input(
            genotype_source_config=genotype_source_config,
            engine=typing.cast("typing.Any", engine),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            is_binary_trait=True,
        )

    assert run_input.native_aligned_sample_data is native_aligned_sample_data
    mock_load_aligned_sample_data.assert_called_once_with(
        engine=engine,
        sample_path=genotype_source_config.resolved_sample_path,
        phenotype_path=Path("phenotype.tsv"),
        phenotype_name="trait",
        covariate_path=Path("covariates.tsv"),
        covariate_names=("age",),
        is_binary_trait=True,
        alignment_config=None,
    )


def test_alignment_config_reaches_native_alignment_and_prediction_source(tmp_path: Path) -> None:
    native_aligned_sample_data = build_native_aligned_sample_data()
    alignment_config = SimpleNamespace(
        sample_key_mode=types.SampleKeyMode.FID_IID,
    )
    engine = SimpleNamespace(
        sample_count=2,
        contains_embedded_samples=True,
    )
    genotype_source_config = source.build_bgen_source_config(tmp_path / "study.bgen")

    with (
        patch(
            "g.engine.native_dispatch.load_native_aligned_sample_data",
            return_value=native_aligned_sample_data,
        ) as mock_load_aligned_sample_data,
        patch("g.engine.native_dispatch._core.RegeniePredictionSource", FakePredictionSource),
    ):
        run_input = native_dispatch.load_native_bgen_run_input(
            genotype_source_config=genotype_source_config,
            engine=typing.cast("typing.Any", engine),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=None,
            covariate_names=None,
            is_binary_trait=False,
            alignment_config=alignment_config,
        )
        prediction_source = native_dispatch.build_regenie_prediction_source(
            prediction_list_path=Path("pred.list"),
            phenotype_name="trait",
            run_input=run_input,
            alignment_config=alignment_config,
        )

    fake_prediction_source = typing.cast("FakePredictionSource", prediction_source)
    assert mock_load_aligned_sample_data.call_args.kwargs["alignment_config"] is alignment_config
    assert fake_prediction_source.sample_key_mode == "fid_iid"
