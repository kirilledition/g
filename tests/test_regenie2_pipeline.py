from __future__ import annotations

import concurrent.futures
import dataclasses
import queue
import threading
import time
import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

import g.engine.callbacks.binary as callback_binary
import g.engine.callbacks.diagnostics as callback_diagnostics
import g.engine.callbacks.grouped as callback_grouped
import g.engine.callbacks.linear as callback_linear
import g.engine.callbacks.runtime as callback_runtime
import g.engine.callbacks.shared as callback_shared
import g.engine.callbacks.transfers as callback_transfers
import g.engine.callbacks.writers as callback_writers
import g.engine.native_dispatch.delivery as native_dispatch_delivery
import g.engine.native_dispatch.engine as native_dispatch_engine
import g.engine.native_dispatch.groups as native_dispatch_groups
import g.engine.native_dispatch.loaders as native_dispatch_loaders
import g.engine.native_dispatch.models as native_dispatch_models
import g.engine.native_dispatch.writers as native_dispatch_writers
import g.engine.regenie2_pipeline.context as pipeline_context
import g.engine.regenie2_pipeline.gpu_format as pipeline_gpu_format
import g.engine.regenie2_pipeline.grouped as pipeline_grouped
import g.engine.regenie2_pipeline.multi_group as pipeline_multi_group
import g.engine.regenie2_pipeline.multi_trait as pipeline_multi_trait
import g.engine.regenie2_pipeline.outputs as pipeline_outputs
import g.engine.regenie2_pipeline.single_trait as pipeline_single_trait
from g import _core, execution_plan, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.compute.regenie2_linear import result as regenie2_linear_result
from g.compute.regenie2_linear import state as regenie2_linear_state
from g.engine import shutdown, timing
from g.interface import config as interface_config
from g.io import output, source


def build_default_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build the packaged-default kernel config for tests."""
    return execution_plan.build_binary_kernel_config(interface_config.load_packaged_config().g_compute)


SCORE_ONLY_PLAN = types.BinaryCorrectionPlan(
    method=types.BinaryFallbackMethod.SCORE_ONLY,
    p_threshold=0.05,
    firth_se=False,
)


def build_test_runtime_compatibility_token() -> _core.NativeRuntimeCompatibilityToken:
    """Build a native runtime compatibility token for pipeline tests."""
    runtime_state = _core.NativeRuntimeState()
    logging_policy_payload = _core.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": None,
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }
    return runtime_state.require_compatible_runtime_policy(logging_policy_payload, None, jax_policy_payload)


@dataclasses.dataclass(frozen=True)
class FakePipelineOutputInitialization:
    committed_chunk_identifier_sequences: tuple[tuple[int, ...], ...]

    @property
    def output_count(self) -> int:
        return len(self.committed_chunk_identifier_sequences)

    def committed_chunk_identifier_sets(self) -> list[list[int]]:
        return [
            list(committed_chunk_identifiers)
            for committed_chunk_identifiers in self.committed_chunk_identifier_sequences
        ]

    def committed_chunk_identifiers(self, output_index: int) -> list[int]:
        return list(self.committed_chunk_identifier_sequences[output_index])


@dataclasses.dataclass(frozen=True)
class FakePipelineOutputPreparationBatch:
    initialization: FakePipelineOutputInitialization
    initialize_callback: typing.Callable[[], None] | None = None

    def initialize(
        self,
        runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    ) -> FakePipelineOutputInitialization:
        del runtime_compatibility_token
        if self.initialize_callback is not None:
            self.initialize_callback()
        return self.initialization


def build_fake_pipeline_output_preparation_batch(
    *committed_chunk_identifier_sequences: typing.Iterable[int],
    initialize_callback: typing.Callable[[], None] | None = None,
) -> FakePipelineOutputPreparationBatch:
    """Build a fake output preparation batch for pipeline integration tests."""
    return FakePipelineOutputPreparationBatch(
        initialization=FakePipelineOutputInitialization(
            committed_chunk_identifier_sequences=tuple(
                tuple(committed_chunk_identifiers)
                for committed_chunk_identifiers in committed_chunk_identifier_sequences
            ),
        ),
        initialize_callback=initialize_callback,
    )


def test_intersect_committed_chunk_identifier_sets_preserves_pipeline_helper_contract() -> None:
    shared_chunk_identifiers = pipeline_multi_group.intersect_committed_chunk_identifier_sets(
        ({0, 32, 64}, {32, 64, 96}, {32, 128})
    )

    assert shared_chunk_identifiers == {32}
    assert pipeline_multi_group.intersect_committed_chunk_identifier_sets(()) == set()


def build_test_genotype_source_config(
    source_path: Path,
    sample_path: Path | None = None,
) -> source.GenotypeSourceConfig:
    """Build genotype source config with an explicit sample path field."""
    return source.GenotypeSourceConfig(source_path=source_path, sample_path=sample_path)


def build_test_output_writer_settings(
    *,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    parquet_compression: types.ParquetCompression,
    arrow_compression: types.ArrowCompression,
    output_format: types.OutputFormat,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> output.OutputWriterSettings:
    """Build writer settings with explicit output statistic dtype."""
    return output.OutputWriterSettings(
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        parquet_compression=parquet_compression,
        arrow_compression=arrow_compression,
        output_format=output_format,
        output_statistic_dtype=output_statistic_dtype,
    )


def build_test_binary_pipeline_callback(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    writer_session: typing.Any,
    correction_plan: types.BinaryCorrectionPlan = SCORE_ONLY_PLAN,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
        types.NullLogisticNonconvergencePolicy.FAIL
    ),
    staging_depth: int = 2,
    native_callback_batch_size: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: typing.Any = None,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> callback_binary.BinaryRegenie2PipelineCallback:
    """Build binary callback tests with explicit production arguments."""
    return callback_binary.BinaryRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=score_dtype,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        output_statistic_dtype=output_statistic_dtype,
    )


def build_test_regenie2_pipeline_context(**keyword_arguments: typing.Any) -> pipeline_context.Regenie2PipelineContext:
    """Build pipeline context with explicit optional test defaults."""
    keyword_arguments.setdefault(
        "phenotype_compute_groups",
        execution_plan.build_phenotype_compute_groups(
            phenotype_names=("trait",),
            multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        ),
    )
    keyword_arguments.setdefault("output_initialized_callback", None)
    keyword_arguments.setdefault(
        "requested_gpu_genotype_format",
        keyword_arguments["gpu_genotype_format"],
    )
    keyword_arguments.setdefault("runtime_compatibility_token", build_test_runtime_compatibility_token())
    return pipeline_context.build_regenie2_pipeline_context(**keyword_arguments)


def open_test_pipeline_bgen_engine(**keyword_arguments: typing.Any) -> typing.Any:
    """Open a pipeline BGEN engine with explicit optional test defaults."""
    keyword_arguments.setdefault("phenotype_count", None)
    return pipeline_outputs.open_pipeline_bgen_engine(**keyword_arguments)


def build_test_linear_pipeline_callback(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    writer_session: typing.Any,
    staging_depth: int = 2,
    native_callback_batch_size: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: typing.Any = None,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> callback_linear.LinearRegenie2PipelineCallback:
    """Build linear callback tests with explicit production arguments."""
    return callback_linear.LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=score_dtype,
        linear_numerical_config=linear_numerical_config,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        output_statistic_dtype=output_statistic_dtype,
    )


def build_test_multi_linear_pipeline_callback(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    writer_sessions: tuple[typing.Any, ...],
    committed_chunk_identifier_sets: tuple[set[int], ...],
    staging_depth: int = 2,
    native_callback_batch_size: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: typing.Any = None,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> callback_linear.MultiLinearRegenie2PipelineCallback:
    """Build multi-trait linear callback tests with explicit production arguments."""
    return callback_linear.MultiLinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=committed_chunk_identifier_sets,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=score_dtype,
        linear_numerical_config=linear_numerical_config,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        output_statistic_dtype=output_statistic_dtype,
    )


def build_test_multi_binary_pipeline_callback(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    writer_sessions: tuple[typing.Any, ...],
    committed_chunk_identifier_sets: tuple[set[int], ...],
    correction_plan: types.BinaryCorrectionPlan = SCORE_ONLY_PLAN,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
        types.NullLogisticNonconvergencePolicy.FAIL
    ),
    staging_depth: int = 2,
    native_callback_batch_size: int = 1,
    result_in_flight_limit: int | None = None,
    dosage_buffer_limit: int | None = None,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: typing.Any = None,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> callback_binary.MultiBinaryRegenie2PipelineCallback:
    """Build multi-trait binary callback tests with explicit production arguments."""
    return callback_binary.MultiBinaryRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=committed_chunk_identifier_sets,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        score_dtype=score_dtype,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        output_statistic_dtype=output_statistic_dtype,
    )


def run_test_bgen_engine_with_callback(**keyword_arguments: typing.Any) -> Path | None:
    """Run native delivery with explicit optional test defaults."""
    keyword_arguments.setdefault("variant_major_packed8_probability_pairs", False)
    keyword_arguments.setdefault("stage_timing_snapshot_writer", timing.write_stage_timing_snapshot)
    return native_dispatch_delivery.run_bgen_engine_with_callback(**keyword_arguments)


def run_test_bgen_engine_with_multi_callback(**keyword_arguments: typing.Any) -> tuple[Path | None, ...]:
    """Run native multi-delivery with explicit optional test defaults."""
    keyword_arguments.setdefault("writer_finish_thread_count", 1)
    keyword_arguments.setdefault("variant_major_packed8_probability_pairs", False)
    return pipeline_multi_group.run_bgen_engine_with_multi_callback(**keyword_arguments)


def _add_single_pipeline_defaults(keyword_arguments: dict[str, typing.Any]) -> None:
    keyword_arguments.setdefault("staging_depth", 2)
    keyword_arguments.setdefault("native_callback_batch_size", 1)
    keyword_arguments.setdefault("result_in_flight_limit", None)
    keyword_arguments.setdefault("dosage_buffer_limit", None)
    keyword_arguments.setdefault("existing_manifest", None)
    keyword_arguments.setdefault("resume", False)
    keyword_arguments.setdefault("resume_mode", types.ResumeMode.FAST)
    keyword_arguments.setdefault("trusted_no_missing_diploid", False)
    keyword_arguments.setdefault("trusted_bgen_validation_mode", types.TrustedBgenValidationMode.CACHE_ON_MISS)
    keyword_arguments.setdefault("jax_device", types.Device.CPU)
    keyword_arguments.setdefault("jax_matmul_precision", None)
    keyword_arguments.setdefault("gpu_genotype_format", types.GpuGenotypeFormat.DOSAGE)
    keyword_arguments.setdefault("stage_timing_recorder", None)
    keyword_arguments.setdefault("telemetry_session", None)
    keyword_arguments.setdefault("alignment_config", None)
    keyword_arguments.setdefault("runtime_compatibility_token", build_test_runtime_compatibility_token())
    keyword_arguments.setdefault("output_initialized_callback", None)


def run_test_regenie2_linear_bgen_pipeline(**keyword_arguments: typing.Any) -> Path | None:
    """Run linear pipeline tests with explicit production arguments."""
    _add_single_pipeline_defaults(keyword_arguments)
    keyword_arguments.setdefault("linear_numerical_config", None)
    return pipeline_single_trait.run_regenie2_linear_bgen_pipeline(**keyword_arguments)


def run_test_regenie2_binary_bgen_pipeline(**keyword_arguments: typing.Any) -> Path | None:
    """Run binary pipeline tests with explicit production arguments."""
    _add_single_pipeline_defaults(keyword_arguments)
    keyword_arguments.setdefault("correction_plan", SCORE_ONLY_PLAN)
    keyword_arguments.setdefault("kernel_config", build_default_binary_kernel_config())
    keyword_arguments.setdefault("null_logistic_nonconvergence_policy", types.NullLogisticNonconvergencePolicy.FAIL)
    return pipeline_single_trait.run_regenie2_binary_bgen_pipeline(**keyword_arguments)


def _add_multi_pipeline_defaults(keyword_arguments: dict[str, typing.Any]) -> None:
    keyword_arguments.setdefault("staging_depth", 2)
    keyword_arguments.setdefault("native_callback_batch_size", 1)
    keyword_arguments.setdefault("result_in_flight_limit", None)
    keyword_arguments.setdefault("dosage_buffer_limit", None)
    keyword_arguments.setdefault("existing_manifests_by_phenotype", None)
    keyword_arguments.setdefault("resume", False)
    keyword_arguments.setdefault("resume_mode", types.ResumeMode.FAST)
    keyword_arguments.setdefault("trusted_no_missing_diploid", False)
    keyword_arguments.setdefault("trusted_bgen_validation_mode", types.TrustedBgenValidationMode.CACHE_ON_MISS)
    keyword_arguments.setdefault("jax_device", types.Device.CPU)
    keyword_arguments.setdefault("jax_matmul_precision", None)
    keyword_arguments.setdefault("gpu_genotype_format", types.GpuGenotypeFormat.DOSAGE)
    keyword_arguments.setdefault("stage_timing_recorder", None)
    keyword_arguments.setdefault("telemetry_session", None)
    keyword_arguments.setdefault("alignment_config", None)
    keyword_arguments.setdefault("sample_mode", None)
    keyword_arguments.setdefault("phenotype_compute_groups", None)
    keyword_arguments.setdefault("runtime_compatibility_token", build_test_runtime_compatibility_token())
    keyword_arguments.setdefault("output_initialized_callback", None)


def run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
    **keyword_arguments: typing.Any,
) -> tuple[Path | None, ...]:
    """Run multi-phenotype linear pipeline tests with explicit production arguments."""
    _add_multi_pipeline_defaults(keyword_arguments)
    keyword_arguments.setdefault("linear_numerical_config", None)
    return pipeline_multi_trait.run_regenie2_multi_phenotype_linear_bgen_pipeline(**keyword_arguments)


def run_test_regenie2_multi_phenotype_binary_bgen_pipeline(
    **keyword_arguments: typing.Any,
) -> tuple[Path | None, ...]:
    """Run multi-phenotype binary pipeline tests with explicit production arguments."""
    _add_multi_pipeline_defaults(keyword_arguments)
    keyword_arguments.setdefault("correction_plan", SCORE_ONLY_PLAN)
    keyword_arguments.setdefault("kernel_config", build_default_binary_kernel_config())
    keyword_arguments.setdefault("null_logistic_nonconvergence_policy", types.NullLogisticNonconvergencePolicy.FAIL)
    return pipeline_multi_trait.run_regenie2_multi_phenotype_binary_bgen_pipeline(**keyword_arguments)


def build_test_bgen_run_engine(**keyword_arguments: typing.Any) -> typing.Any:
    """Build a native run engine with explicit optional test defaults."""
    keyword_arguments.setdefault("trusted_no_missing_diploid", False)
    keyword_arguments.setdefault("trusted_bgen_validation_mode", types.TrustedBgenValidationMode.CACHE_ON_MISS)
    keyword_arguments.setdefault("trusted_bgen_validator", None)
    return native_dispatch_engine.build_bgen_run_engine(**keyword_arguments)


def load_test_native_bgen_run_input(**keyword_arguments: typing.Any) -> native_dispatch_models.NativeBgenRunInput:
    """Load native run input with explicit optional injection points."""
    keyword_arguments.setdefault("alignment_config", None)
    keyword_arguments.setdefault("build_native_bgen_run_input_callable", None)
    keyword_arguments.setdefault("load_aligned_sample_data_callable", None)
    return native_dispatch_loaders.load_native_bgen_run_input(**keyword_arguments)


@dataclasses.dataclass(frozen=True)
class PipelineRuntimeOptions:
    """Runtime options that pipeline tests pass explicitly."""

    writer_settings: output.OutputWriterSettings
    bgen_decode_tile_variant_count: int
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype


def build_default_pipeline_runtime_options() -> PipelineRuntimeOptions:
    """Build default runtime options through the public config boundary."""
    packaged_config = interface_config.load_packaged_config()
    compute_config = packaged_config.g_compute
    output_config = packaged_config.g_output
    return PipelineRuntimeOptions(
        writer_settings=build_test_output_writer_settings(
            finalize_parquet=output_config.finalize_parquet,
            writer_thread_count=output_config.writer_threads,
            writer_queue_depth=output_config.writer_queue_depth,
            chunks_per_arrow_file=output_config.chunks_per_arrow_file,
            arrow_compression=output_config.arrow_compression,
            parquet_compression=output_config.parquet_compression,
            output_format=output_config.format,
            output_statistic_dtype=output_config.output_statistic_dtype,
        ),
        bgen_decode_tile_variant_count=compute_config.bgen_decode_tile_variant_count,
        score_dtype=compute_config.score_dtype,
        firth_dtype=compute_config.firth_dtype,
    )


def write_test_run_manifest(output_run_paths: output.OutputRunPaths, header: typing.Mapping[str, object]) -> bytes:
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


class BufferObservingWriterSession(FakeWriterSession):
    def __init__(
        self,
        callback: callback_runtime.NativeBgenCallbackRunner,
        expected_buffer: callback_shared.HostGenotypeBuffer,
    ) -> None:
        super().__init__()
        self.callback = callback
        self.expected_buffer = expected_buffer
        self.observed_buffer_before_write = False

    def write_regenie2_native_chunk(self, **kwargs: object) -> None:
        observed_buffer_result = self.callback.free_dosage_buffers.get(timeout_seconds=0.0)
        assert observed_buffer_result.has_item is True
        observed_buffer = typing.cast("callback_shared.HostGenotypeBuffer", observed_buffer_result.item)
        assert observed_buffer is self.expected_buffer
        assert self.callback.free_dosage_buffers.put(observed_buffer, timeout_seconds=0.0) is True
        self.observed_buffer_before_write = True
        super().write_regenie2_native_chunk(**kwargs)


class FailingPerfCounterClock:
    """Clock double that fails when default writer paths collect timings."""

    @staticmethod
    def perf_counter() -> float:
        """Fail when no-recorder code attempts wall-time profiling."""
        message = "perf_counter should not be called without a timing recorder"
        raise AssertionError(message)


class RecordingTelemetrySession:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.progress_events: list[dict[str, object]] = []

    def log_event(self, event_name: str, level: str, **fields: object) -> None:
        del level
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

    resolved_state = callback_runtime.require_current_chromosome_state(chromosome_state, chromosome="chr22")

    assert resolved_state is chromosome_state


def test_require_current_chromosome_state_raises_clear_error_when_missing() -> None:
    with pytest.raises(RuntimeError, match="Chromosome state for 'chr22' was not prepared"):
        callback_runtime.require_current_chromosome_state(None, chromosome="chr22")


def test_cast_statistic_array_for_native_writer_uses_public_float32_schema() -> None:
    precise_values = np.asarray([1.0, 1.0 + 2.0**-30], dtype=np.float64)

    writer_values = callback_transfers.cast_statistic_array_for_native_writer(
        precise_values, types.FloatingPointDtype.FLOAT32
    )

    assert writer_values.dtype == np.float32
    np.testing.assert_array_equal(writer_values, precise_values.astype(np.float32))


def test_cast_statistic_array_for_native_writer_preserves_public_float64_schema() -> None:
    precise_values = np.asarray([1.0, 1.0 + 2.0**-30], dtype=np.float64)

    writer_values = callback_transfers.cast_statistic_array_for_native_writer(
        precise_values, types.FloatingPointDtype.FLOAT64
    )

    assert writer_values.dtype == np.float64
    np.testing.assert_array_equal(writer_values, precise_values)


def test_write_regenie2_native_chunk_downcasts_float64_statistics_before_writing() -> None:
    writer_session = FakeWriterSession()
    precise_values = np.asarray([1.0, 1.0 + 2.0**-30], dtype=np.float64)
    extra_code = np.asarray([0, 3], dtype=np.int32)

    callback_writers.write_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=typing.cast("typing.Any", SimpleNamespace()),
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=typing.cast("typing.Any", precise_values),
        standard_error=typing.cast("typing.Any", precise_values + 1.0),
        chi_squared=typing.cast("typing.Any", precise_values + 2.0),
        log10_p_value=typing.cast("typing.Any", precise_values + 3.0),
        extra_code=typing.cast("typing.Any", extra_code),
        stage_timing_recorder=None,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
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


def test_write_regenie2_native_chunk_uses_native_output_write_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    writer_session = FakeWriterSession()
    write_plan_calls: list[dict[str, object]] = []

    def plan_single_trait_output_write(**kwargs: object) -> SimpleNamespace:
        write_plan_calls.append(kwargs)
        return SimpleNamespace(method_name="write_regenie2_native_chunk")

    monkeypatch.setattr(callback_writers._core, "plan_single_trait_output_write", plan_single_trait_output_write)

    callback_writers.write_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=typing.cast("typing.Any", SimpleNamespace()),
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=typing.cast("typing.Any", np.asarray([0.1, 0.2], dtype=np.float64)),
        standard_error=typing.cast("typing.Any", np.asarray([0.3, 0.4], dtype=np.float64)),
        chi_squared=typing.cast("typing.Any", np.asarray([1.0, 2.0], dtype=np.float64)),
        log10_p_value=typing.cast("typing.Any", np.asarray([3.0, 4.0], dtype=np.float64)),
        extra_code=None,
        stage_timing_recorder=None,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT64,
    )

    assert write_plan_calls == [{"is_native_writer_session": False, "output_statistic_dtype": "float64"}]
    assert len(writer_session.native_chunks) == 1


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
            native_dispatch_writers.finish_writer_sessions,
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


def test_resolve_writer_finish_thread_count_uses_native_cleanup_policy() -> None:
    assert native_dispatch_writers.resolve_writer_finish_thread_count(0, 0) == 0
    assert native_dispatch_writers.resolve_writer_finish_thread_count(3, 2) == 2
    assert native_dispatch_writers.resolve_writer_finish_thread_count(3, 5) == 3
    with pytest.raises(ValueError, match="Writer finish thread count must be positive"):
        native_dispatch_writers.resolve_writer_finish_thread_count(1, 0)


def test_plan_writer_finish_execution_uses_native_cleanup_policy() -> None:
    finish_plan = native_dispatch_writers.plan_writer_finish_execution(3, 2)

    assert finish_plan.writer_session_count == 3
    assert finish_plan.thread_count == 2
    assert finish_plan.has_writer_sessions is True
    assert finish_plan.uses_parallel_finish is True


def test_plan_bgen_delivery_cleanup_uses_native_lifecycle_policy() -> None:
    cleanup_plan = native_dispatch_delivery.plan_bgen_delivery_cleanup(
        cleanup_outcome=native_dispatch_delivery.BgenDeliveryCleanupOutcome.INTERRUPTED,
        callback_finished=False,
    )

    assert cleanup_plan.cleanup_actions == [
        "drain_callback",
        "finish_interrupted_writer_sessions",
        "write_stage_timing_snapshot",
    ]
    assert cleanup_plan.drain_callback is True
    assert cleanup_plan.finish_writer_sessions is False
    assert cleanup_plan.finish_interrupted_writer_sessions is True
    assert cleanup_plan.abort_callback is False
    assert cleanup_plan.abort_writer_sessions is False
    assert cleanup_plan.write_stage_timing_snapshot is True


def test_execute_bgen_delivery_cleanup_plan_uses_native_action_order() -> None:
    events: list[str] = []

    class OrderedCallback:
        def finish(self) -> None:
            events.append("callback.finish")

        def abort(self) -> None:
            events.append("callback.abort")

    class OrderedWriterSession:
        def finish(self) -> str:
            events.append("writer.finish")
            return "results/final.parquet"

        def finish_interrupted(self, signal_name: str) -> None:
            events.append(f"writer.finish_interrupted:{signal_name}")

        def abort(self) -> None:
            events.append("writer.abort")

    def record_snapshot(
        recorder: timing.StageTimingRecorder | None,
        stage_timing_path: Path | None,
    ) -> None:
        assert recorder is None
        assert stage_timing_path is None
        events.append("snapshot")

    success_plan = native_dispatch_delivery.plan_bgen_delivery_cleanup(
        cleanup_outcome=native_dispatch_delivery.BgenDeliveryCleanupOutcome.SUCCESS,
        callback_finished=False,
    )
    success_execution = native_dispatch_delivery.execute_bgen_delivery_cleanup_plan(
        cleanup_plan=success_plan,
        callback_finished=False,
        callback=OrderedCallback(),
        writer_sessions=(OrderedWriterSession(),),
        writer_finish_thread_count=1,
        stage_timing_recorder=None,
        stage_timing_snapshot_writer=record_snapshot,
        shutdown_request=None,
    )

    assert events == ["callback.finish", "writer.finish", "snapshot"]
    assert success_execution.callback_finished is True
    assert success_execution.final_parquet_paths == (Path("results/final.parquet"),)

    events.clear()
    shutdown_signal = shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130)
    shutdown_request = shutdown.GracefulShutdownRequested(shutdown_signal)
    interrupted_plan = native_dispatch_delivery.plan_bgen_delivery_cleanup(
        cleanup_outcome=native_dispatch_delivery.BgenDeliveryCleanupOutcome.INTERRUPTED,
        callback_finished=False,
    )
    interrupted_execution = native_dispatch_delivery.execute_bgen_delivery_cleanup_plan(
        cleanup_plan=interrupted_plan,
        callback_finished=False,
        callback=OrderedCallback(),
        writer_sessions=(OrderedWriterSession(),),
        writer_finish_thread_count=1,
        stage_timing_recorder=None,
        stage_timing_snapshot_writer=record_snapshot,
        shutdown_request=shutdown_request,
    )

    assert events == ["callback.finish", "writer.finish_interrupted:SIGINT", "snapshot"]
    assert interrupted_execution.callback_finished is True
    assert interrupted_execution.final_parquet_paths == ()


def test_plan_output_write_methods_use_native_cleanup_policy() -> None:
    single_write_plan = callback_writers._core.plan_single_trait_output_write(
        is_native_writer_session=True,
        output_statistic_dtype="float64",
    )
    assert single_write_plan.method_name == "write_regenie2_native_chunk_f64"
    assert single_write_plan.uses_float64_native_writer is True

    multi_write_plan = callback_writers._core.plan_multi_trait_output_write(
        active_trait_count=2,
        all_writer_sessions_native=True,
        output_statistic_dtype="float64",
    )
    assert multi_write_plan.active_trait_count == 2
    assert multi_write_plan.use_native_multi_writer is True
    assert multi_write_plan.uses_float64_native_writer is True


def test_write_regenie2_native_chunk_records_per_chunk_output_timing() -> None:
    writer_session = FakeWriterSession()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    metadata = build_native_metadata()

    callback_writers.write_regenie2_native_chunk_with_optional_timing(
        writer_session=writer_session,
        metadata=metadata,
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=None,
        stage_timing_recorder=stage_timing_recorder,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
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


def test_write_regenie2_multi_native_chunk_skips_committed_traits_and_slices_extra_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(callback_writers, "time", FailingPerfCounterClock)
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

    callback_writers.write_regenie2_multi_native_chunk_with_optional_timing(
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
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )

    assert len(writer_sessions[0].native_chunks) == 1
    assert not writer_sessions[1].native_chunks
    written_chunk = writer_sessions[0].native_chunks[0]
    np.testing.assert_array_equal(written_chunk["extra_code"], np.asarray(extra_code[0]))
    np.testing.assert_array_equal(written_chunk["beta"], np.asarray([0.1, 0.2], dtype=np.float32))


def test_write_regenie2_multi_native_chunk_uses_native_output_write_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    writer_sessions = (FakeWriterSession(), FakeWriterSession())
    metadata = build_native_metadata()
    write_plan_calls: list[dict[str, object]] = []

    def plan_multi_trait_output_write(**kwargs: object) -> SimpleNamespace:
        write_plan_calls.append(kwargs)
        return SimpleNamespace(
            active_trait_count=2,
            use_native_multi_writer=False,
            uses_float64_native_writer=False,
        )

    monkeypatch.setattr(callback_writers._core, "plan_multi_trait_output_write", plan_multi_trait_output_write)

    callback_writers.write_regenie2_multi_native_chunk_with_optional_timing(
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        metadata=metadata,
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        standard_error=jnp.asarray([[1.1, 1.2], [1.3, 1.4]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[2.1, 2.2], [2.3, 2.4]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[3.1, 3.2], [3.3, 3.4]], dtype=jnp.float32),
        extra_code=None,
        stage_timing_recorder=None,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )

    assert write_plan_calls == [
        {
            "active_trait_count": 2,
            "all_writer_sessions_native": False,
            "output_statistic_dtype": "float32",
        }
    ]
    assert tuple(len(writer_session.native_chunks) for writer_session in writer_sessions) == (1, 1)


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

    monkeypatch.setattr(callback_shared.jax, "device_get", recording_device_get)

    callback_writers.write_regenie2_multi_native_chunk_with_optional_timing(
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
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
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

    monkeypatch.setattr(callback_shared.jax, "device_get", fail_device_get)
    monkeypatch.setattr(callback_writers, "time", FailingPerfCounterClock)

    callback_writers.write_regenie2_multi_native_chunk_with_optional_timing(
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
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )

    assert not writer_sessions[0].native_chunks
    assert not writer_sessions[1].native_chunks


def test_chunk_stats_helpers_use_bundled_compute_arrays_with_path_specific_fields() -> None:
    linear_chunk_stats = BundledChunkStats()
    binary_chunk_stats = BundledChunkStats()

    linear_arrays = callback_transfers.get_linear_chunk_stats_arrays(typing.cast("typing.Any", linear_chunk_stats))
    binary_arrays = callback_transfers.get_binary_chunk_stats_arrays(
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
    aggregate_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    exact_recorder = timing.StageTimingRecorder(exact_stage_timings=True)
    diagnostics = SimpleNamespace(
        score_only_count=1,
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
        callback_diagnostics.record_binary_chunk_diagnostics(stage_timing_recorder=aggregate_recorder, result=result)
        callback_diagnostics.record_binary_chunk_diagnostics(stage_timing_recorder=exact_recorder, result=result)

    mock_count.assert_called_once_with(result)
    assert aggregate_recorder.snapshot().binary_chunk_diagnostics == ()
    assert timing.serialize_binary_chunk_diagnostics(exact_recorder.snapshot().binary_chunk_diagnostics) == (
        {
            "score_only_count": 1,
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


def test_binary_compute_preprocessed_chunk_collects_summary_diagnostics_for_worker_consumption() -> None:
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=SCORE_ONLY_PLAN,
        kernel_config=build_default_binary_kernel_config(),
        stage_timing_recorder=timing.StageTimingRecorder(exact_stage_timings=False),
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
        patch("g.compute.regenie2_binary.api.count_binary_chunk_diagnostics", return_value=object()) as mock_count,
    ):
        callback.compute_preprocessed_chunk(
            variant_metadata=variant_metadata,
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        )

    mock_count.assert_called_once_with(result)
    mock_enqueue.assert_called_once()
    assert mock_enqueue.call_args.kwargs["binary_chunk_diagnostics"] is not None


def test_binary_compute_preprocessed_chunk_collects_diagnostics_with_exact_timing() -> None:
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=SCORE_ONLY_PLAN,
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
    diagnostics = SimpleNamespace(score_only_count=2, score_test_candidate_count=0, firth_candidate_count=0)
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=SCORE_ONLY_PLAN,
        kernel_config=build_default_binary_kernel_config(),
        telemetry_session=typing.cast("typing.Any", RecordingTelemetrySession()),
    )
    diagnostics = typing.cast(
        "regenie2_binary.BinaryChunkDiagnostics",
        SimpleNamespace(score_test_candidate_count=2),
    )
    work_item = callback_shared.Regenie2ResultWriteWorkItem(
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
    assert callback.callback_scheduler_state.acquire_result_queue_slot() is True
    assert callback.result_queue.put(work_item, timeout_seconds=0.0) is True
    assert callback.callback_scheduler_state.acquire_result_queue_slot() is True
    assert callback.result_queue.put(None, timeout_seconds=0.0) is True
    with (
        patch(
            "g.engine.callbacks.runtime.write_materialized_regenie2_native_chunk_with_optional_timing",
        ) as mock_write,
        patch("g.engine.callbacks.runtime.record_binary_chunk_diagnostics_from_count") as mock_record,
        patch(
            "g.engine.callbacks.runtime.binary_chunk_diagnostics_to_summary_counts",
            return_value=regenie2_binary.BinaryCorrectionSummaryCounts(
                chunk_count=1,
                score_only_count=1,
                score_test_candidate_count=2,
                firth_candidate_count=0,
                firth_converged_count=0,
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
                dense_correction_count=0,
            ),
        ) as mock_summary_counts,
    ):
        callback.consume_result_write_items()

    mock_write.assert_called_once()
    mock_record.assert_called_once_with(
        stage_timing_recorder=None,
        diagnostics=diagnostics,
    )
    mock_summary_counts.assert_called_once_with((diagnostics,))
    assert callback.binary_correction_summary.score_only_count == 1
    assert callback.binary_correction_summary.score_test_candidate_count == 2


def test_native_callback_runner_uses_native_binary_summary_plans() -> None:
    callback = ManualCallbackRunner()
    callback.binary_correction_summary = callback_runtime._core.NativeBinaryCorrectionSummary()
    callback.binary_correction_pending_diagnostics = []
    diagnostics = typing.cast(
        "regenie2_binary.BinaryChunkDiagnostics",
        SimpleNamespace(score_test_candidate_count=2),
    )

    callback.record_binary_correction_diagnostics(diagnostics)
    assert callback.binary_correction_summary_chunk_count == 0
    assert callback.binary_correction_pending_diagnostics == []

    callback.telemetry_session = typing.cast("typing.Any", RecordingTelemetrySession())
    callback.record_binary_correction_diagnostics(diagnostics)
    assert callback.binary_correction_summary_chunk_count == 1
    assert callback.binary_correction_pending_diagnostics == [diagnostics]


def test_native_callback_runner_uses_runtime_resource_binary_summary() -> None:
    class ResourceBackedCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
        def __init__(self) -> None:
            super().__init__(
                worker_name="native-resource-binary-summary-runner-test",
                staging_depth=1,
                native_callback_batch_size=1,
                result_in_flight_limit=1,
                dosage_buffer_limit=1,
                stage_timing_recorder=None,
                telemetry_session=typing.cast("typing.Any", RecordingTelemetrySession()),
                output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
            )

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

    callback = ResourceBackedCallbackRunner()
    diagnostics = typing.cast(
        "regenie2_binary.BinaryChunkDiagnostics",
        SimpleNamespace(score_test_candidate_count=2),
    )

    callback.record_binary_null_model_failure_count(2)
    callback.record_binary_correction_diagnostics(diagnostics)

    assert callback.binary_correction_summary.null_model_failure_count == 2
    assert callback.binary_correction_summary_chunk_count == 1
    assert callback.binary_correction_pending_diagnostics == [diagnostics]


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
        callback_batch_size: int = 1,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.callback_batch_size = callback_batch_size
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        self.run_call_arguments.append(self.run_arguments)
        return 0

    def run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples(
        self,
        aligned_sample_data: object,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int:
        return self.run_bgen_variant_major_dosage_buffered_chunks(
            typing.cast("typing.Any", aligned_sample_data).sample_indices,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )

    def run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
        self,
        aligned_sample_data: object,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int:
        return self.run_bgen_variant_major_dosage_buffered_chunks(
            typing.cast("typing.Any", aligned_sample_data).sample_indices,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
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


class IncompatibleTrustedRunEngine(FakeRunEngine):
    def validate_trusted_no_missing_diploid(self) -> None:
        self.validation_count += 1
        message = "packed8 incompatible"
        raise ValueError(message)


class PartialCommitDeliveringRunEngine(FakeRunEngine):
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.callback_batch_size = callback_batch_size
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


def build_native_run_input() -> native_dispatch_models.NativeBgenRunInput:
    return native_dispatch_models.NativeBgenRunInput(
        native_aligned_sample_data=typing.cast("typing.Any", build_native_aligned_sample_data()),
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        phenotype_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        covariate_matrix=np.asarray([[1.0], [1.0]], dtype=np.float32),
        is_binary_trait=False,
    )


def test_open_pipeline_bgen_engine_records_selected_backend_telemetry() -> None:
    telemetry_session = RecordingTelemetrySession()
    pipeline_options = build_default_pipeline_runtime_options()
    writer_settings = build_test_output_writer_settings(
        finalize_parquet=False,
        writer_thread_count=pipeline_options.writer_settings.writer_thread_count,
        writer_queue_depth=pipeline_options.writer_settings.writer_queue_depth,
        chunks_per_arrow_file=pipeline_options.writer_settings.chunks_per_arrow_file,
        parquet_compression=pipeline_options.writer_settings.parquet_compression,
        arrow_compression=types.ArrowCompression.ZSTD,
        output_format=types.OutputFormat.PARQUET,
    )
    context = build_test_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
        correction_plan=SCORE_ONLY_PLAN,
        binary_kernel_config=None,
        linear_numerical_config=None,
        writer_settings=writer_settings,
        stage_timing_recorder=None,
        telemetry_session=typing.cast("typing.Any", telemetry_session),
        alignment_config=None,
    )
    engine = FakeRunEngine("study.bgen", chunk_size=32, trusted_no_missing_diploid=True)

    with patch("g.engine.regenie2_pipeline.outputs.native_dispatch_engine.build_bgen_run_engine", return_value=engine):
        opened_engine = open_test_pipeline_bgen_engine(
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
) -> native_dispatch_models.NativeBgenRunInput:
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
    return native_dispatch_models.NativeBgenRunInput(
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
    run_inputs: tuple[native_dispatch_models.NativeBgenRunInput, ...],
) -> native_dispatch_models.NativeBgenGroupedRunInput:
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
    run_input = native_dispatch_models.NativeBgenMultiRunInput(
        native_multi_aligned_sample_data=typing.cast("typing.Any", native_multi_aligned_sample_data),
        phenotype_names=phenotype_names,
        sample_indices=np.ascontiguousarray(native_multi_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_matrix=np.asarray(native_multi_aligned_sample_data.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(native_multi_aligned_sample_data.covariate_matrix, dtype=np.float32),
        is_binary_trait=native_multi_aligned_sample_data.is_binary_trait,
    )
    return native_dispatch_models.NativeBgenGroupedRunInput(
        compute_group=native_dispatch_groups.build_resolved_phenotype_compute_group(
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


def build_native_multi_run_input() -> native_dispatch_models.NativeBgenMultiRunInput:
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
    return native_dispatch_models.NativeBgenMultiRunInput(
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

    compute_group = native_dispatch_groups.build_resolved_complete_case_phenotype_compute_group(
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

    assert callback_shared.get_metadata_chromosome(ScalarChromosomeMetadata()) == "22"


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


class ManualCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
    def __init__(self) -> None:
        self.progress_state = callback_runtime._core.NativeCallbackProgressState()
        self.stage_timing_recorder = None
        self.telemetry_session = None
        self.callback_scheduler_state = callback_runtime._core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=2,
            dosage_buffer_limit=2,
        )
        self.dosage_queue = callback_runtime._core.NativeCallbackObjectQueue(
            self.callback_scheduler_state.dosage_queue_depth
        )
        self.result_queue = callback_runtime._core.NativeCallbackObjectQueue(
            self.callback_scheduler_state.result_queue_depth
        )
        self.result_in_flight_slot_signal = callback_runtime._core.NativeCallbackWaitSignal()
        self.dosage_buffer_pool_signal = callback_runtime._core.NativeCallbackWaitSignal()
        self.free_dosage_buffers = callback_runtime._core.NativeCallbackObjectQueue(
            self.callback_scheduler_state.dosage_buffer_limit
        )
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


def attach_manual_callback_scheduler_state(callback: typing.Any) -> None:
    callback.callback_scheduler_state = callback_runtime._core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=2,
        dosage_buffer_limit=2,
    )
    callback.dosage_queue = callback_runtime._core.NativeCallbackObjectQueue(
        callback.callback_scheduler_state.dosage_queue_depth
    )
    callback.result_queue = callback_runtime._core.NativeCallbackObjectQueue(
        callback.callback_scheduler_state.result_queue_depth
    )
    callback.result_in_flight_slot_signal = callback_runtime._core.NativeCallbackWaitSignal()
    callback.dosage_buffer_pool_signal = callback_runtime._core.NativeCallbackWaitSignal()
    callback.free_dosage_buffers = callback_runtime._core.NativeCallbackObjectQueue(
        callback.callback_scheduler_state.dosage_buffer_limit
    )


def mark_callback_workers_started(callback: typing.Any) -> None:
    if not hasattr(callback, "callback_scheduler_state"):
        attach_manual_callback_scheduler_state(callback)
    assert callback.callback_scheduler_state.mark_started() is True


def test_native_callback_runner_records_chromosome_progress_transitions() -> None:
    callback = ManualCallbackRunner()
    telemetry_session = RecordingTelemetrySession()
    callback.telemetry_session = telemetry_session

    callback.record_progress(build_native_metadata())
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
    class ThreadedManualCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
        def __init__(self) -> None:
            super().__init__(
                worker_name="threaded-manual-callback",
                staging_depth=2,
                native_callback_batch_size=1,
                result_in_flight_limit=None,
                dosage_buffer_limit=None,
                stage_timing_recorder=None,
                telemetry_session=None,
                output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
            )

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

    assert isinstance(callback.callback_runtime_resources, callback_runtime._core.NativeCallbackRuntimeResources)
    assert callback.callback_scheduler_state is callback.callback_runtime_resources.callback_scheduler_state
    assert callback.progress_state is callback.callback_runtime_resources.progress_state
    assert callback.dosage_queue is callback.callback_runtime_resources.dosage_queue
    assert callback.result_queue is callback.callback_runtime_resources.result_queue
    assert callback.free_dosage_buffers is callback.callback_runtime_resources.free_dosage_buffers
    assert callback.binary_correction_summary is callback.callback_runtime_resources.binary_correction_summary
    assert isinstance(callback.worker_thread, callback_runtime._core.NativeCallbackWorkerThread)
    assert isinstance(callback.result_worker_thread, callback_runtime._core.NativeCallbackWorkerThread)
    assert callback.worker_thread is callback.callback_runtime_resources.worker_thread
    assert callback.result_worker_thread is callback.callback_runtime_resources.result_worker_thread
    assert not hasattr(callback_runtime, "threading")
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


def test_native_callback_worker_thread_starts_and_joins_python_target() -> None:
    started_event = threading.Event()
    release_event = threading.Event()

    def worker_target() -> None:
        started_event.set()
        release_event.wait(timeout=2.0)

    worker_thread = callback_runtime._core.NativeCallbackWorkerThread(
        target=worker_target,
        name="native-callback-worker-test",
    )

    assert worker_thread.name == "native-callback-worker-test"
    assert not worker_thread.is_alive()

    worker_thread.start()
    try:
        assert started_event.wait(timeout=1.0)
        assert worker_thread.is_alive()
    finally:
        release_event.set()
        worker_thread.join(timeout=1.0)

    assert not worker_thread.is_alive()


def test_native_callback_runtime_resources_own_queue_operations() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-queue-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    assert runtime_resources.native_callback_batch_size == 1
    assert runtime_resources.dosage_queue_depth == 1
    assert runtime_resources.result_queue_depth == 1
    assert runtime_resources.result_in_flight_limit == 1
    assert runtime_resources.dosage_buffer_limit == 1
    dosage_item = object()
    result_item = object()

    assert runtime_resources.try_put_dosage_work_item(dosage_item, timeout_seconds=0.0) is True
    assert runtime_resources.dosage_queue_occupied_count == 1
    assert runtime_resources.callback_scheduler_state.dosage_queue_occupied_count == 1
    queued = True
    dosage_put_observation = runtime_resources.plan_dosage_queue_put_observation(queued)
    assert dosage_put_observation.queue_name == "dosage_queue"
    assert dosage_put_observation.operation_name == "put"
    assert dosage_put_observation.blocked is False
    assert dosage_put_observation.should_retry_put is False
    queued = False
    dosage_put_retry_observation = runtime_resources.plan_dosage_queue_put_observation(queued)
    assert dosage_put_retry_observation.operation_name == "producer_blocking"
    assert dosage_put_retry_observation.blocked is True
    assert dosage_put_retry_observation.should_retry_put is True
    dosage_stage_observation = runtime_resources.plan_current_queue_stage_backpressure_observation(
        queue_name="dosage_queue",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert dosage_stage_observation.stage_name == "callback_queue_producer_blocking"
    assert dosage_stage_observation.queue_depth == 1
    assert dosage_stage_observation.queue_capacity == 1
    assert dosage_stage_observation.blocked_seconds == 0.5
    assert runtime_resources.try_put_dosage_work_item(object(), timeout_seconds=0.0) is False
    dosage_result = runtime_resources.get_dosage_work_item()
    assert dosage_result.has_item is True
    assert dosage_result.item is dosage_item
    assert runtime_resources.dosage_queue_occupied_count == 0
    assert runtime_resources.callback_scheduler_state.dosage_queue_occupied_count == 0
    dosage_get_observation = runtime_resources.plan_dosage_queue_get_observation()
    assert dosage_get_observation.queue_name == "dosage_queue"
    assert dosage_get_observation.operation_name == "consumer_wait"
    assert dosage_get_observation.blocked is True
    assert runtime_resources.try_put_dosage_work_item_with_backpressure_timeout(None) is True
    assert runtime_resources.get_dosage_work_item().item is None

    assert runtime_resources.try_put_result_write_item(result_item, timeout_seconds=0.0) is True
    assert runtime_resources.result_queue_occupied_count == 1
    assert runtime_resources.callback_scheduler_state.result_queue_occupied_count == 1
    queued = True
    result_put_observation = runtime_resources.plan_result_queue_put_observation(queued)
    assert result_put_observation.queue_name == "result_queue"
    assert result_put_observation.operation_name == "put"
    assert result_put_observation.blocked is False
    assert result_put_observation.should_retry_put is False
    queued = False
    result_put_retry_observation = runtime_resources.plan_result_queue_put_observation(queued)
    assert result_put_retry_observation.operation_name == "producer_blocking"
    assert result_put_retry_observation.blocked is True
    assert result_put_retry_observation.should_retry_put is True
    result_observation = runtime_resources.plan_current_queue_backpressure_observation(
        queue_name="result_queue",
        operation_name="put",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert result_observation.queue_depth == 1
    assert result_observation.queue_capacity == 1
    assert result_observation.blocked_seconds == 0.0
    assert runtime_resources.try_put_result_write_item(object(), timeout_seconds=0.0) is False
    result = runtime_resources.get_result_write_item()
    assert result.has_item is True
    assert result.item is result_item
    assert runtime_resources.result_queue_occupied_count == 0
    assert runtime_resources.callback_scheduler_state.result_queue_occupied_count == 0
    result_get_observation = runtime_resources.plan_result_queue_get_observation()
    assert result_get_observation.queue_name == "result_queue"
    assert result_get_observation.operation_name == "consumer_wait"
    assert result_get_observation.blocked is True
    assert runtime_resources.try_put_result_write_item_with_backpressure_timeout(None) is True
    assert runtime_resources.get_result_write_item().item is None


def test_native_callback_runtime_resources_own_progress_state() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-progress-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    first_chunk_identity = callback_runtime.build_native_callback_chunk_identity(build_native_metadata())
    second_chunk_identity = callback_runtime.build_native_callback_chunk_identity(
        build_native_metadata_for_chunk(chunk_identifier=2)
    )

    assert runtime_resources.processed_chunk_count == 0
    assert runtime_resources.current_progress_chromosome is None
    first_update = runtime_resources.record_processed_chunk(first_chunk_identity)
    assert first_update.processed_chunk_count == 1
    assert runtime_resources.processed_chunk_count == 1
    assert runtime_resources.current_progress_chromosome == "22"
    runtime_resources.record_processed_chunk(second_chunk_identity)
    assert runtime_resources.processed_chunk_count == 2
    completion = runtime_resources.finish_progress()
    assert completion is not None
    assert completion.chromosome == "22"
    assert completion.processed_chunk_count == 2
    assert runtime_resources.finish_progress() is None

    untimed_runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-progress-untimed-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    untimed_runtime_resources.record_processed_chunk_without_progress()
    assert untimed_runtime_resources.processed_chunk_count == 1
    assert untimed_runtime_resources.current_progress_chromosome is None


def test_native_callback_runtime_resources_own_binary_correction_summary() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-binary-summary-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    assert runtime_resources.binary_correction_chunk_count_with_pending(0) == 0
    has_telemetry_session = False
    has_diagnostics = True
    disabled_record_plan = runtime_resources.plan_binary_correction_diagnostics_record(
        has_telemetry_session,
        has_diagnostics,
    )
    assert disabled_record_plan.should_record is False
    has_telemetry_session = True
    enabled_record_plan = runtime_resources.plan_binary_correction_diagnostics_record(
        has_telemetry_session,
        has_diagnostics,
    )
    assert enabled_record_plan.should_record is True

    runtime_resources.add_binary_null_model_failure_count(2)
    runtime_resources.add_binary_correction_diagnostics_totals(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    )

    assert runtime_resources.binary_correction_chunk_count_with_pending(1) == 2
    pending_diagnostics_count = 0
    emit_plan = runtime_resources.plan_binary_correction_summary_emit(
        has_telemetry_session,
        pending_diagnostics_count,
    )
    assert emit_plan.should_flush_pending_diagnostics is False
    assert emit_plan.should_emit_summary is True
    pending_diagnostics_count = 1
    flush_plan = runtime_resources.plan_binary_correction_summary_emit(
        has_telemetry_session,
        pending_diagnostics_count,
    )
    assert flush_plan.should_flush_pending_diagnostics is True
    assert flush_plan.should_emit_summary is True
    summary_payload = runtime_resources.binary_correction_summary_payload()
    assert summary_payload["chunk_count"] == 1
    assert summary_payload["score_only_count"] == 2
    assert summary_payload["firth_attempted_count"] == 4
    assert summary_payload["dense_correction_count"] == 18
    assert summary_payload["null_model_failure_count"] == 2


def test_native_callback_runtime_resources_own_dispatch_and_drain_plans() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-dispatch-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    dosage_drain_plan = runtime_resources.plan_dosage_work_drain_completion(has_dosage_work_item=False)
    assert dosage_drain_plan.should_stop is True
    dosage_continue_plan = runtime_resources.plan_dosage_work_drain_completion(has_dosage_work_item=True)
    assert dosage_continue_plan.should_stop is False
    dosage_dispatch_plan = runtime_resources.plan_validated_dosage_work_item_dispatch(
        callback_runtime.DosageWorkItemKind.SAMPLE_MAJOR_DOSAGE.value
    )
    assert dosage_dispatch_plan.should_process_sample_major_dosage is True
    with pytest.raises(RuntimeError, match="continued without a work item"):
        runtime_resources.plan_validated_dosage_work_item_dispatch(
            callback_runtime.DosageWorkItemKind.STOP_SIGNAL.value
        )

    result_drain_plan = runtime_resources.plan_result_write_drain_completion(
        has_result_work_item=False,
        flush_binary_correction_diagnostics_on_stop=True,
    )
    assert result_drain_plan.should_stop is True
    assert result_drain_plan.should_flush_binary_correction_diagnostics is True
    result_continue_plan = runtime_resources.plan_result_write_drain_completion(
        has_result_work_item=True,
        flush_binary_correction_diagnostics_on_stop=True,
    )
    assert result_continue_plan.should_stop is False
    assert result_continue_plan.should_flush_binary_correction_diagnostics is False
    result_dispatch_plan = runtime_resources.plan_validated_result_write_item_dispatch(
        callback_runtime.ResultWriteItemKind.SINGLE_RESULT.value,
        callback_runtime.ResultWriteItemKind.SINGLE_RESULT.value,
    )
    assert result_dispatch_plan.should_process_result_write_item is True
    with pytest.raises(RuntimeError, match="expected single_result but received multi_result"):
        runtime_resources.plan_validated_result_write_item_dispatch(
            callback_runtime.ResultWriteItemKind.MULTI_RESULT.value,
            callback_runtime.ResultWriteItemKind.SINGLE_RESULT.value,
        )

    stage_duration_plan = runtime_resources.plan_dosage_work_item_stage_duration(
        callback_runtime.DosageWorkItemKind.VARIANT_MAJOR_DOSAGE_BATCH.value,
        2,
        4.0,
    )
    assert stage_duration_plan.chunk_count == 2
    assert stage_duration_plan.duration_per_chunk == 2.0

    dosage_handoff_plan = runtime_resources.plan_dosage_work_handoff(2)
    assert dosage_handoff_plan.chunk_count == 2
    with pytest.raises(ValueError, match="at least one chunk"):
        runtime_resources.plan_dosage_work_handoff(0)

    variant_major_batch_handoff_plan = runtime_resources.plan_variant_major_dosage_batch_handoff(
        metadata_count=2,
        genotype_matrix_by_variant_count=2,
        chunk_stats_count=2,
    )
    assert variant_major_batch_handoff_plan.chunk_count == 2
    with pytest.raises(ValueError, match="identical lengths"):
        runtime_resources.plan_variant_major_dosage_batch_handoff(
            metadata_count=2,
            genotype_matrix_by_variant_count=1,
            chunk_stats_count=2,
        )


def test_native_callback_runtime_resources_own_result_in_flight_slots() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-result-slot-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    acquire_observation_plan = runtime_resources.acquire_result_in_flight_slot_with_backpressure_timeout()
    assert acquire_observation_plan.resource_name == "result_in_flight_slots"
    assert acquire_observation_plan.operation_name == "acquire"
    assert acquire_observation_plan.blocked is False
    assert acquire_observation_plan.should_retry_acquisition is False
    assert runtime_resources.callback_scheduler_state.result_in_flight_occupied_count == 1

    retry_observation_plan = runtime_resources.acquire_result_in_flight_slot_with_backpressure_timeout()
    assert retry_observation_plan.resource_name == "result_in_flight_slots"
    assert retry_observation_plan.operation_name == "producer_blocking"
    assert retry_observation_plan.blocked is True
    assert retry_observation_plan.should_retry_acquisition is True
    assert runtime_resources.callback_scheduler_state.result_in_flight_occupied_count == 1

    release_observation_plan = runtime_resources.release_result_in_flight_slot()
    assert release_observation_plan.resource_name == "result_in_flight_slots"
    assert release_observation_plan.operation_name == "release"
    assert release_observation_plan.blocked is False
    assert runtime_resources.callback_scheduler_state.result_in_flight_occupied_count == 0
    with pytest.raises(RuntimeError, match="no occupied slot"):
        runtime_resources.release_result_in_flight_slot()


def test_native_callback_runtime_resources_own_dosage_buffer_lifecycle() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-dosage-buffer-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    dosage_buffer = np.empty((2, 2), dtype=np.float32)

    assert runtime_resources.free_dosage_buffer_count == 0
    assert runtime_resources.register_dosage_buffer(id(dosage_buffer)) == 0
    assert runtime_resources.dosage_buffer_allocated_count == 1
    assert id(dosage_buffer) in runtime_resources.dosage_buffer_identifiers
    assert runtime_resources.callback_scheduler_state.dosage_buffer_allocated_count == 1
    return_attempt_plan = runtime_resources.plan_dosage_buffer_return_attempt(id(dosage_buffer))
    assert return_attempt_plan.should_return is True
    assert runtime_resources.plan_dosage_buffer_return_attempt(id(object())).should_return is False
    with pytest.raises(RuntimeError, match="no available slot"):
        runtime_resources.register_dosage_buffer(id(np.empty((2, 2), dtype=np.float32)))

    assert runtime_resources.return_dosage_buffer(id(dosage_buffer), dosage_buffer) == 1
    assert runtime_resources.free_dosage_buffers.occupied_count == 1
    assert runtime_resources.free_dosage_buffer_count == 1
    with pytest.raises(RuntimeError, match="no slot for returned buffer"):
        runtime_resources.return_dosage_buffer(id(dosage_buffer), dosage_buffer)
    free_buffer_result = runtime_resources.free_dosage_buffers.get(timeout_seconds=0.0)
    assert free_buffer_result.has_item is True
    assert free_buffer_result.item is dosage_buffer
    assert runtime_resources.free_dosage_buffer_count == 0

    reuse_observation_plan = runtime_resources.plan_dosage_buffer_pool_reuse_observation()
    assert reuse_observation_plan.operation_name == "reuse"
    assert reuse_observation_plan.blocked is False
    wait_observation_plan = runtime_resources.plan_dosage_buffer_pool_consumer_wait_observation()
    assert wait_observation_plan.operation_name == "consumer_wait"
    assert wait_observation_plan.blocked is True
    exact_reuse_plan = runtime_resources.plan_dosage_buffer_reuse((2, 2), (2, 2))
    assert exact_reuse_plan is not None
    assert exact_reuse_plan.requires_slice is False
    assert exact_reuse_plan.slice_dimensions == [2, 2]
    sliced_reuse_plan = runtime_resources.plan_dosage_buffer_reuse((4, 5), (2, 3))
    assert sliced_reuse_plan is not None
    assert sliced_reuse_plan.requires_slice is True
    assert sliced_reuse_plan.slice_dimensions == [2, 3]
    assert runtime_resources.plan_dosage_buffer_reuse((2, 2), (3, 2)) is None
    pool_observation = runtime_resources.plan_dosage_buffer_pool_backpressure_observation(
        operation_name="return",
        free_buffer_count=1,
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert pool_observation.queue_name == "dosage_buffer_pool"
    assert pool_observation.queue_depth == 1
    assert pool_observation.queue_capacity == 1
    assert pool_observation.blocked_seconds == 0.0
    pool_stage_observation = runtime_resources.plan_dosage_buffer_pool_stage_backpressure_observation(
        operation_name="consumer_wait",
        free_buffer_count=0,
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert pool_stage_observation.stage_name == "dosage_buffer_pool_consumer_wait"
    assert pool_stage_observation.queue_depth == 0
    assert pool_stage_observation.queue_capacity == 1
    assert pool_stage_observation.blocked_seconds == 0.5

    assert runtime_resources.discard_dosage_buffer(id(dosage_buffer)) == 0
    assert runtime_resources.dosage_buffer_allocated_count == 0
    assert runtime_resources.callback_scheduler_state.dosage_buffer_allocated_count == 0
    assert runtime_resources.discard_dosage_buffer(id(dosage_buffer)) is None


def test_native_callback_runtime_resources_own_result_work_item_resource_cleanup() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-result-cleanup-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    dosage_buffer = np.empty((2, 2), dtype=np.float32)

    assert runtime_resources.register_dosage_buffer(id(dosage_buffer)) == 0
    runtime_resources.acquire_result_in_flight_slot_with_backpressure_timeout()

    pre_write_result = runtime_resources.release_result_work_item_pre_write_resources(
        id(dosage_buffer),
        dosage_buffer,
    )
    assert pre_write_result.released_host_buffer is True
    assert pre_write_result.free_buffer_count == 1
    assert pre_write_result.released_result_in_flight_slot is False
    free_buffer_result = runtime_resources.free_dosage_buffers.get(timeout_seconds=0.0)
    assert free_buffer_result.has_item is True
    assert free_buffer_result.item is dosage_buffer

    final_result = runtime_resources.release_result_work_item_final_resources(
        id(dosage_buffer),
        dosage_buffer,
        has_released_host_dosage_buffer=True,
        release_in_flight_slot=True,
    )
    assert final_result.released_host_buffer is False
    assert final_result.free_buffer_count is None
    assert final_result.released_result_in_flight_slot is True
    assert final_result.result_in_flight_resource_name == "result_in_flight_slots"
    assert final_result.result_in_flight_operation_name == "release"
    assert final_result.result_in_flight_blocked is False
    assert runtime_resources.callback_scheduler_state.result_in_flight_occupied_count == 0


def test_native_callback_runtime_resources_own_dosage_buffer_acquisition() -> None:
    def worker_target() -> None:
        return None

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-dosage-buffer-acquire-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    dosage_buffer = np.empty((2, 2), dtype=np.float32)

    allocate_result = runtime_resources.acquire_dosage_buffer_with_backpressure_timeout()
    assert allocate_result.should_allocate is True
    assert allocate_result.dosage_buffer is None
    assert allocate_result.free_buffer_count == 0
    assert allocate_result.waited is False

    assert runtime_resources.register_dosage_buffer(id(dosage_buffer)) == 0
    assert runtime_resources.return_dosage_buffer(id(dosage_buffer), dosage_buffer) == 1

    reuse_result = runtime_resources.acquire_dosage_buffer_with_backpressure_timeout()
    assert reuse_result.should_allocate is False
    assert reuse_result.dosage_buffer is dosage_buffer
    assert reuse_result.free_buffer_count == 0
    assert reuse_result.waited is False

    acquisition_started = threading.Event()

    def acquire_after_pool_capacity_is_full() -> _core.NativeDosageBufferAcquireResult:
        acquisition_started.set()
        return runtime_resources.acquire_dosage_buffer_with_backpressure_timeout()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(acquire_after_pool_capacity_is_full)
        assert acquisition_started.wait(timeout=1.0)
        time.sleep(0.02)
        assert not future.done()

        assert runtime_resources.return_dosage_buffer(id(dosage_buffer), dosage_buffer) == 1
        wait_result = future.result(timeout=2.0)

    assert wait_result.should_allocate is False
    assert wait_result.dosage_buffer is None
    assert wait_result.free_buffer_count == 1
    assert wait_result.waited is True

    reused_after_wait_result = runtime_resources.acquire_dosage_buffer_with_backpressure_timeout()
    assert reused_after_wait_result.dosage_buffer is dosage_buffer


def test_native_callback_runtime_resources_own_worker_stop_and_join() -> None:
    runtime_resources_holder: list[typing.Any] = []

    def dosage_worker_target() -> None:
        runtime_resources_holder[0].get_dosage_work_item()

    def result_worker_target() -> None:
        runtime_resources_holder[0].get_result_write_item()

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-worker-shutdown-test",
        dosage_worker_target=dosage_worker_target,
        result_worker_target=result_worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    runtime_resources_holder.append(runtime_resources)

    start_plan = runtime_resources.start_workers()
    assert start_plan.has_start_error is False
    assert runtime_resources.stop_dosage_worker(1.0) is None
    assert runtime_resources.stop_result_worker(1.0) is None
    assert runtime_resources.join_dosage_worker(1.0) is None
    assert runtime_resources.join_result_worker(1.0) is None
    assert not runtime_resources.worker_thread.is_alive()
    assert not runtime_resources.result_worker_thread.is_alive()


def test_native_callback_runtime_resources_own_worker_finish_and_abort_lifecycle() -> None:
    finish_runtime_resources_holder: list[typing.Any] = []

    def finish_dosage_worker_target() -> None:
        finish_runtime_resources_holder[0].get_dosage_work_item()

    def finish_result_worker_target() -> None:
        finish_runtime_resources_holder[0].get_result_write_item()

    finish_runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-worker-finish-test",
        dosage_worker_target=finish_dosage_worker_target,
        result_worker_target=finish_result_worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    finish_runtime_resources_holder.append(finish_runtime_resources)

    start_plan = finish_runtime_resources.start_workers()
    assert start_plan.has_start_error is False
    finish_result = finish_runtime_resources.finish_worker_lifecycle()

    assert finish_result.has_shutdown_timeout is False
    assert finish_result.shutdown_worker_name is None
    assert finish_result.shutdown_timeout_seconds is None
    assert finish_result.raise_worker_error is True
    assert finish_result.complete_progress is True
    assert finish_result.emit_binary_correction_summary is True
    assert finish_runtime_resources.plan_worker_error_raise().should_raise is False
    dosage_error_update = finish_runtime_resources.update_dosage_worker_error("dosage failed")
    assert dosage_error_update.had_error is False
    assert dosage_error_update.has_error is True
    assert dosage_error_update.error_message == "native pipeline callback worker failed: dosage failed"
    dosage_error_raise_plan = finish_runtime_resources.plan_worker_error_raise()
    assert dosage_error_raise_plan.should_raise is True
    assert dosage_error_raise_plan.raise_dosage_worker_error is True
    assert dosage_error_raise_plan.error_message == "native pipeline callback worker failed: dosage failed"
    dosage_error_clear = finish_runtime_resources.update_dosage_worker_error(None)
    assert dosage_error_clear.had_error is True
    assert dosage_error_clear.has_error is False
    result_error_update = finish_runtime_resources.update_result_worker_error("writer failed")
    assert result_error_update.had_error is False
    assert result_error_update.has_error is True
    assert result_error_update.error_message == "native pipeline result writer worker failed: writer failed"
    result_error_clear = finish_runtime_resources.update_result_worker_error(None)
    assert result_error_clear.had_error is True
    assert result_error_clear.has_error is False
    assert not finish_runtime_resources.worker_thread.is_alive()
    assert not finish_runtime_resources.result_worker_thread.is_alive()

    abort_runtime_resources_holder: list[typing.Any] = []

    def abort_dosage_worker_target() -> None:
        abort_runtime_resources_holder[0].get_dosage_work_item()

    def abort_result_worker_target() -> None:
        abort_runtime_resources_holder[0].get_result_write_item()

    abort_runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-worker-abort-test",
        dosage_worker_target=abort_dosage_worker_target,
        result_worker_target=abort_result_worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )
    abort_runtime_resources_holder.append(abort_runtime_resources)

    start_plan = abort_runtime_resources.start_workers()
    assert start_plan.has_start_error is False
    abort_plan = abort_runtime_resources.abort_worker_lifecycle()

    assert abort_plan.stop_dosage_worker is True
    assert abort_plan.stop_result_worker is True
    assert abort_runtime_resources.join_dosage_worker(1.0) is None
    assert abort_runtime_resources.join_result_worker(1.0) is None
    assert not abort_runtime_resources.worker_thread.is_alive()
    assert not abort_runtime_resources.result_worker_thread.is_alive()


def test_native_callback_runtime_resources_report_worker_shutdown_timeouts() -> None:
    release_event = threading.Event()

    def worker_target() -> None:
        release_event.wait(timeout=2.0)

    runtime_resources = callback_runtime._core.NativeCallbackRuntimeResources(
        worker_name="native-resource-worker-timeout-test",
        dosage_worker_target=worker_target,
        result_worker_target=worker_target,
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    start_plan = runtime_resources.start_workers()
    assert start_plan.has_start_error is False
    try:
        assert runtime_resources.stop_dosage_worker(0.0) == 0.0
        assert runtime_resources.stop_result_worker(0.0) == 0.0
        assert runtime_resources.join_dosage_worker(0.0) == 0.0
        assert runtime_resources.join_result_worker(0.0) == 0.0
    finally:
        release_event.set()
        runtime_resources.join_dosage_worker(1.0)
        runtime_resources.join_result_worker(1.0)


def test_native_callback_runner_uses_native_worker_start_plan() -> None:
    class RecordingWorkerThread:
        def __init__(self, *, worker_name: str, start_events: list[str]) -> None:
            self.worker_name = worker_name
            self.start_events = start_events

        def start(self) -> None:
            self.start_events.append(self.worker_name)

    start_events: list[str] = []
    callback = object.__new__(ManualCallbackRunner)
    scheduler_state = CallbackStartAttemptSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    callback_for_start = typing.cast("typing.Any", callback)
    callback_for_start.result_worker_thread = RecordingWorkerThread(
        worker_name="result",
        start_events=start_events,
    )
    callback_for_start.worker_thread = RecordingWorkerThread(
        worker_name="dosage",
        start_events=start_events,
    )

    with patch(
        "g.engine.callbacks.runtime._core.plan_callback_worker_start",
        side_effect=AssertionError("runner should use scheduler worker start planner"),
    ):
        callback.start()
        callback.start()

    assert start_events == ["result", "dosage"]
    assert scheduler_state.start_attempt_count == 2


def test_native_callback_runner_records_native_delivery_timing_for_enqueued_chunk() -> None:
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)

    class TimedCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
        def __init__(self) -> None:
            super().__init__(
                worker_name="timed-manual-callback",
                staging_depth=2,
                native_callback_batch_size=1,
                result_in_flight_limit=None,
                dosage_buffer_limit=None,
                stage_timing_recorder=stage_timing_recorder,
                telemetry_session=None,
                output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
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


def test_native_callback_runner_batches_variant_major_dosage_queue_handoff() -> None:
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)

    class BatchedCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
        def __init__(self) -> None:
            super().__init__(
                worker_name="batched-manual-callback",
                staging_depth=2,
                native_callback_batch_size=2,
                result_in_flight_limit=None,
                dosage_buffer_limit=2,
                stage_timing_recorder=stage_timing_recorder,
                telemetry_session=None,
                output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
            )
            self.metadata: list[object] = []

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
            del genotype_matrix_by_variant, chunk_stats
            self.metadata.append(variant_metadata)

        def compute_preprocessed_variant_major_packed8_chunk(
            self,
            *,
            variant_metadata: object,
            packed_probability_pairs_by_variant: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, packed_probability_pairs_by_variant, chunk_stats

    first_metadata = build_native_metadata_for_chunk(chunk_identifier=0)
    second_metadata = build_native_metadata_for_chunk(chunk_identifier=2)
    callback = BatchedCallbackRunner()
    try:
        with patch(
            "g.engine.callbacks.runtime._core.plan_callback_queue_stage_backpressure_observation",
            side_effect=AssertionError("runner should use scheduler queue stage backpressure planner"),
        ):
            callback.compute_preprocessed_variant_major_dosage_chunk_batch(
                metadata_batch=(first_metadata, second_metadata),
                genotype_matrix_by_variant_batch=(
                    np.ones((2, 2), dtype=np.float32),
                    np.full((2, 2), 2.0, dtype=np.float32),
                ),
                chunk_stats_batch=(
                    typing.cast("typing.Any", SimpleNamespace()),
                    typing.cast("typing.Any", SimpleNamespace()),
                ),
            )
            callback.finish()
    finally:
        callback.abort()

    assert callback.metadata == [first_metadata, second_metadata]
    assert callback.processed_chunk_count == 2
    snapshot = stage_timing_recorder.snapshot()
    assert snapshot.stage_counts["callback_queue_put"] == 1
    assert snapshot.stage_counts["native_delivery"] == 2
    assert snapshot.stage_counts["python_callback"] == 2


def test_native_callback_runner_uses_scheduler_dosage_work_stage_duration_plan() -> None:
    callback = ManualCallbackRunner()
    callback.stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    scheduler_state = DosageWorkItemStageDurationSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    first_metadata = build_native_metadata_for_chunk(chunk_identifier=0)
    second_metadata = build_native_metadata_for_chunk(chunk_identifier=2)
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())
    batch_work_item = callback_shared.PreprocessedVariantMajorDosageChunkBatchWorkItem(
        work_items=(
            callback_shared.PreprocessedVariantMajorDosageChunkWorkItem(
                metadata=first_metadata,
                genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
                chunk_stats=chunk_stats,
            ),
            callback_shared.PreprocessedVariantMajorDosageChunkWorkItem(
                metadata=second_metadata,
                genotype_matrix_by_variant=np.full((2, 2), 2.0, dtype=np.float32),
                chunk_stats=chunk_stats,
            ),
        )
    )

    callback.record_work_item_stage_elapsed_duration(batch_work_item, "python_callback", 4.0)

    assert scheduler_state.dosage_work_item_kind == "variant_major_dosage_batch"
    assert scheduler_state.chunk_count == 2
    assert scheduler_state.elapsed_seconds == 4.0
    snapshot = callback.stage_timing_recorder.snapshot()
    assert snapshot.stage_counts["python_callback"] == 2
    assert tuple(chunk_timing.duration_seconds for chunk_timing in snapshot.chunk_stage_timings) == (2.0, 2.0)
    assert tuple(chunk_timing.chunk_identifier for chunk_timing in snapshot.chunk_stage_timings) == (0, 2)


def test_native_callback_runner_uses_scheduler_variant_major_batch_handoff_plan() -> None:
    callback = ManualCallbackRunner()
    mark_callback_workers_started(callback)
    metadata = build_native_metadata()

    with patch(
        "g.engine.callbacks.runtime._core.plan_variant_major_dosage_batch_handoff",
        side_effect=AssertionError("runner should use scheduler variant-major batch handoff planner"),
    ):
        callback.compute_preprocessed_variant_major_dosage_chunk_batch(
            metadata_batch=(metadata,),
            genotype_matrix_by_variant_batch=(np.ones((2, 2), dtype=np.float32),),
            chunk_stats_batch=(typing.cast("typing.Any", SimpleNamespace()),),
        )

    queued_work_item = callback.get_dosage_work_item()
    assert queued_work_item is not None
    assert isinstance(queued_work_item, callback_shared.PreprocessedVariantMajorDosageChunkBatchWorkItem)
    assert len(queued_work_item.work_items) == 1


def test_native_callback_runner_uses_scheduler_dosage_work_handoff_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = DosageWorkHandoffSchedulerProbe()
    queued_work_items: list[typing.Any] = []

    def put_dosage_work_item_probe(work_item: typing.Any) -> None:
        queued_work_items.append(work_item)

    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    typing.cast("typing.Any", callback).put_dosage_work_item = put_dosage_work_item_probe

    first_metadata = build_native_metadata_for_chunk(chunk_identifier=0)
    second_metadata = build_native_metadata_for_chunk(chunk_identifier=2)
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())
    callback.compute_preprocessed_dosage_chunk(
        metadata=first_metadata,
        genotype_matrix=np.ones((2, 2), dtype=np.float32),
        chunk_stats=chunk_stats,
    )
    callback.compute_preprocessed_variant_major_dosage_chunk(
        metadata=first_metadata,
        genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
        chunk_stats=chunk_stats,
    )
    callback.compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        metadata=first_metadata,
        packed_probability_pairs_by_variant=np.ones((2, 2, 2), dtype=np.uint8),
        chunk_stats=chunk_stats,
    )
    callback.compute_preprocessed_variant_major_dosage_chunk_batch(
        metadata_batch=(first_metadata, second_metadata),
        genotype_matrix_by_variant_batch=(
            np.ones((2, 2), dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ),
        chunk_stats_batch=(chunk_stats, chunk_stats),
    )

    assert scheduler_state.handoff_chunk_counts == [1, 1, 1, 2]
    assert scheduler_state.variant_major_batch_counts == [(2, 2, 2)]
    assert len(queued_work_items) == 4
    assert isinstance(queued_work_items[-1], callback_shared.PreprocessedVariantMajorDosageChunkBatchWorkItem)


def test_native_callback_runner_rejects_invalid_variant_major_batch_handoffs() -> None:
    callback = ManualCallbackRunner()
    metadata = build_native_metadata()

    with pytest.raises(ValueError, match="identical lengths"):
        callback.compute_preprocessed_variant_major_dosage_chunk_batch(
            metadata_batch=(metadata,),
            genotype_matrix_by_variant_batch=(),
            chunk_stats_batch=(typing.cast("typing.Any", SimpleNamespace()),),
        )
    with pytest.raises(ValueError, match="at least one chunk"):
        callback.compute_preprocessed_variant_major_dosage_chunk_batch(
            metadata_batch=(),
            genotype_matrix_by_variant_batch=(),
            chunk_stats_batch=(),
        )


def test_native_callback_runner_consumes_both_dosage_layouts() -> None:
    callback = ManualCallbackRunner()
    callback.callback_scheduler_state = callback_runtime._core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=1,
        result_in_flight_limit=2,
        dosage_buffer_limit=2,
    )
    callback.dosage_queue = callback_runtime._core.NativeCallbackObjectQueue(
        callback.callback_scheduler_state.dosage_queue_depth
    )
    callback.result_queue = callback_runtime._core.NativeCallbackObjectQueue(
        callback.callback_scheduler_state.result_queue_depth
    )
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    callback.stage_timing_recorder = stage_timing_recorder
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())

    assert callback.try_put_dosage_work_item(
        callback_shared.PreprocessedVariantMajorDosageChunkWorkItem(
            metadata=metadata,
            genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        ),
        timeout_seconds=0.0,
    )
    assert callback.try_put_dosage_work_item(
        callback_shared.PreprocessedDosageChunkWorkItem(
            metadata=metadata,
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=chunk_stats,
        ),
        timeout_seconds=0.0,
    )
    assert callback.try_put_dosage_work_item(None, timeout_seconds=0.0)

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


def test_native_dosage_delivery_forwards_callback_batch_size() -> None:
    engine = FakeRunEngine("study.bgen", chunk_size=32)
    callback = SimpleNamespace(native_callback_batch_size=2)
    run_input = SimpleNamespace(sample_indices=np.asarray([0, 1], dtype=np.int64))

    processed_chunk_count = native_dispatch_delivery.run_variant_major_dosage_delivery(
        engine=typing.cast("typing.Any", engine),
        run_input=typing.cast("typing.Any", run_input),
        callback=callback,
        committed_chunk_identifier_list=[],
    )

    assert processed_chunk_count == 0
    assert engine.callback_batch_size == 2


def test_native_dosage_delivery_defaults_callback_batch_size_in_native_policy() -> None:
    engine = FakeRunEngine("study.bgen", chunk_size=32)
    callback = SimpleNamespace()
    run_input = SimpleNamespace(sample_indices=np.asarray([0, 1], dtype=np.int64))

    processed_chunk_count = native_dispatch_delivery.run_variant_major_dosage_delivery(
        engine=typing.cast("typing.Any", engine),
        run_input=typing.cast("typing.Any", run_input),
        callback=callback,
        committed_chunk_identifier_list=[],
    )

    assert processed_chunk_count == 0
    assert engine.callback_batch_size == 1


def test_plan_bgen_delivery_invocation_uses_native_selection_policy() -> None:
    run_input = SimpleNamespace(
        sample_indices=np.asarray([0, 1], dtype=np.int64),
        native_multi_aligned_sample_data=SimpleNamespace(sample_indices=np.asarray([2, 3], dtype=np.int64)),
        native_aligned_sample_data=SimpleNamespace(sample_indices=np.asarray([4, 5], dtype=np.int64)),
    )
    callback = SimpleNamespace(native_callback_batch_size=2)

    dosage_plan = native_dispatch_delivery.plan_bgen_delivery_invocation(
        callback,
        typing.cast("typing.Any", run_input),
        variant_major_packed8_probability_pairs=False,
    )

    assert (
        dosage_plan.delivery_method == native_dispatch_delivery.BgenDeliveryMethod.DOSAGE_NATIVE_MULTI_ALIGNED_SAMPLES
    )
    assert dosage_plan.callback_batch_size == 2

    packed8_plan = native_dispatch_delivery.plan_bgen_delivery_invocation(
        SimpleNamespace(native_callback_batch_size=1),
        typing.cast("typing.Any", run_input),
        variant_major_packed8_probability_pairs=True,
    )
    assert (
        packed8_plan.delivery_method == native_dispatch_delivery.BgenDeliveryMethod.PACKED8_NATIVE_MULTI_ALIGNED_SAMPLES
    )
    assert packed8_plan.callback_batch_size == 1


def test_native_dosage_delivery_prefers_native_multi_alignment() -> None:
    engine = FakeRunEngine("study.bgen", chunk_size=32)
    callback = SimpleNamespace()
    run_input = SimpleNamespace(
        sample_indices=np.asarray([0, 1], dtype=np.int64),
        native_multi_aligned_sample_data=SimpleNamespace(sample_indices=np.asarray([2, 3], dtype=np.int64)),
        native_aligned_sample_data=SimpleNamespace(sample_indices=np.asarray([4, 5], dtype=np.int64)),
    )

    processed_chunk_count = native_dispatch_delivery.run_variant_major_dosage_delivery(
        engine=typing.cast("typing.Any", engine),
        run_input=typing.cast("typing.Any", run_input),
        callback=callback,
        committed_chunk_identifier_list=[],
    )

    assert processed_chunk_count == 0
    assert engine.run_arguments is not None
    np.testing.assert_array_equal(engine.run_arguments[0], np.asarray([2, 3], dtype=np.int64))


def test_native_packed8_delivery_rejects_callback_batch_size_above_one() -> None:
    engine = FakeRunEngine("study.bgen", chunk_size=32)
    callback = SimpleNamespace(native_callback_batch_size=2)
    run_input = SimpleNamespace(sample_indices=np.asarray([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="packed8 BGEN delivery"):
        native_dispatch_delivery.run_variant_major_packed8_delivery(
            engine=typing.cast("typing.Any", engine),
            run_input=typing.cast("typing.Any", run_input),
            callback=callback,
            committed_chunk_identifier_list=[],
        )


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
    assert callback.try_put_dosage_work_item(
        callback_shared.PreprocessedDosageChunkWorkItem(
            metadata=build_native_metadata(),
            genotype_matrix=np.ones((2, 2), dtype=np.float32),
            chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        ),
        timeout_seconds=0.0,
    )

    callback.consume_dosage_chunks()

    assert isinstance(callback.worker_error, ValueError)
    assert callback.callback_scheduler_state.has_dosage_worker_error is True
    assert (
        callback.callback_scheduler_state.dosage_worker_error_message
        == "native pipeline callback worker failed: compute failed"
    )


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
    first_limited_buffer = limited_callback.acquire_dosage_buffer_with_shape((1, 1), np.float32)
    second_limited_buffer = limited_callback.acquire_dosage_buffer_with_shape((1, 2), np.float32)
    limited_callback.release_dosage_buffer(first_limited_buffer)
    blocked_replacement = limited_callback.acquire_dosage_buffer_with_shape((4, 5), np.float32)
    assert blocked_replacement.shape == (4, 5)
    assert limited_callback.dosage_buffer_count == limited_callback.dosage_buffer_limit
    assert id(first_limited_buffer) not in limited_callback.dosage_buffer_identifiers
    assert id(second_limited_buffer) in limited_callback.dosage_buffer_identifiers
    assert id(blocked_replacement) in limited_callback.dosage_buffer_identifiers


def test_native_callback_runner_uses_native_dosage_buffer_pool_accounting() -> None:
    callback = ManualCallbackRunner()

    first_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    second_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=4)
    assert callback.callback_scheduler_state.dosage_buffer_allocated_count == 2
    assert callback.callback_scheduler_state.has_available_dosage_buffer_slot() is False
    assert callback.callback_scheduler_state.owns_dosage_buffer(id(first_buffer)) is True
    assert callback.callback_scheduler_state.owns_dosage_buffer(id(second_buffer)) is True

    callback.discard_dosage_buffer_slot(first_buffer)

    assert callback.callback_scheduler_state.dosage_buffer_allocated_count == 1
    assert callback.callback_scheduler_state.has_available_dosage_buffer_slot() is True
    assert callback.callback_scheduler_state.owns_dosage_buffer(id(first_buffer)) is False


def test_native_callback_runner_waits_on_native_dosage_buffer_pool_release() -> None:
    callback = ManualCallbackRunner()
    first_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    second_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=4)
    acquisition_started = threading.Event()

    def acquire_buffer_after_pool_capacity_is_full() -> callback_shared.HostGenotypeBuffer:
        acquisition_started.set()
        return callback.acquire_dosage_buffer(sample_count=2, variant_count=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(acquire_buffer_after_pool_capacity_is_full)
        assert acquisition_started.wait(timeout=1.0)
        time.sleep(0.2)
        assert not future.done()

        callback.release_dosage_buffer(first_buffer)
        acquired_buffer = future.result(timeout=2.0)

    assert acquired_buffer is first_buffer
    assert callback.callback_scheduler_state.dosage_buffer_allocated_count == 2
    assert callback.free_dosage_buffer_count == 0

    callback.release_dosage_buffer(acquired_buffer)
    callback.release_dosage_buffer(second_buffer)
    assert callback.free_dosage_buffer_count == 2


def test_native_callback_runner_records_native_dosage_buffer_operation_observations() -> None:
    callback = ManualCallbackRunner()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    callback.stage_timing_recorder = stage_timing_recorder

    with patch(
        "g.engine.callbacks.runtime._core.plan_callback_queue_backpressure_observation",
        side_effect=AssertionError("runner should use scheduler queue backpressure planner"),
    ):
        dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
        callback.release_dosage_buffer(dosage_buffer)
        reused_dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
        callback.discard_dosage_buffer_slot(reused_dosage_buffer)

    queue_backpressure_by_operation = {
        queue_backpressure.operation_name: queue_backpressure
        for queue_backpressure in stage_timing_recorder.snapshot().queue_backpressure
    }

    assert set(queue_backpressure_by_operation) == {"allocate", "return", "reuse", "discard"}
    for queue_backpressure in queue_backpressure_by_operation.values():
        assert queue_backpressure.queue_name == "dosage_buffer_pool"
        assert queue_backpressure.observation_count == 1
        assert queue_backpressure.total_blocked_seconds == 0.0


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


def test_native_callback_runner_uses_scheduler_dosage_buffer_reuse_plan() -> None:
    callback = ManualCallbackRunner()
    oversized_buffer = callback.allocate_dosage_buffer_with_shape((4, 5), np.float32)
    callback.release_dosage_buffer(oversized_buffer)

    with patch(
        "g.engine.callbacks.runtime._core.plan_dosage_buffer_reuse",
        side_effect=AssertionError("runner should use scheduler dosage buffer reuse planner"),
    ):
        sliced_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)

    assert sliced_buffer.shape == (2, 3)
    assert np.shares_memory(sliced_buffer, oversized_buffer)


def test_native_callback_runner_ignores_unowned_host_dosage_buffers() -> None:
    callback = ManualCallbackRunner()

    callback.release_dosage_buffer(np.empty((2, 2), dtype=np.float32))

    assert callback.dosage_buffer_count == 0
    assert callback.free_dosage_buffer_count == 0


def test_native_callback_runner_surfaces_worker_and_writer_errors() -> None:
    callback = ManualCallbackRunner()
    callback.worker_error = ValueError("dosage failed")

    assert callback.callback_scheduler_state.has_dosage_worker_error is True
    with pytest.raises(RuntimeError, match="callback worker failed") as dosage_error:
        callback.raise_worker_error_if_present()
    assert str(dosage_error.value) == "native pipeline callback worker failed: dosage failed"

    callback.worker_error = None
    assert callback.callback_scheduler_state.has_dosage_worker_error is False
    callback.result_worker_error = ValueError("writer failed")

    assert callback.callback_scheduler_state.has_result_worker_error is True
    with pytest.raises(RuntimeError, match="result writer worker failed") as writer_error:
        callback.raise_worker_error_if_present()
    assert str(writer_error.value) == "native pipeline result writer worker failed: writer failed"


def test_base_native_callback_runner_compute_methods_are_abstract() -> None:
    class IncompleteCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
        def compute_preprocessed_chunk(
            self,
            *,
            variant_metadata: object,
            genotype_matrix: object,
            chunk_stats: object,
        ) -> None:
            del variant_metadata, genotype_matrix, chunk_stats

    with pytest.raises(TypeError, match="abstract"):
        IncompleteCallbackRunner(
            worker_name="incomplete-callback",
            staging_depth=2,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            stage_timing_recorder=None,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )


class RecordingShutdownCallbackRunner(ManualCallbackRunner):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_calls: list[str] = []
        self.shutdown_timeouts: dict[str, float | None] = {}

    def stop_dosage_worker(self, timeout_seconds: float | None) -> None:
        self.shutdown_calls.append("stop_dosage")
        self.shutdown_timeouts["stop_dosage"] = timeout_seconds

    def join_dosage_worker(self, timeout_seconds: float | None) -> None:
        self.shutdown_calls.append("join_dosage")
        self.shutdown_timeouts["join_dosage"] = timeout_seconds

    def stop_result_worker(self, timeout_seconds: float | None) -> None:
        self.shutdown_calls.append("stop_result")
        self.shutdown_timeouts["stop_result"] = timeout_seconds

    def join_result_worker(self, timeout_seconds: float | None) -> None:
        self.shutdown_calls.append("join_result")
        self.shutdown_timeouts["join_result"] = timeout_seconds

    def raise_worker_error_if_present(self) -> None:
        self.shutdown_calls.append("raise_worker_error")

    def complete_progress(self) -> None:
        self.shutdown_calls.append("complete_progress")

    def emit_binary_correction_summary(self) -> None:
        self.shutdown_calls.append("emit_binary_correction_summary")


class FailingStopShutdownCallbackRunner(RecordingShutdownCallbackRunner):
    def stop_dosage_worker(self, timeout_seconds: float | None) -> None:
        super().stop_dosage_worker(timeout_seconds)
        raise callback_shared.NativeBgenWorkerShutdownError(
            worker_name="dosage-worker",
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 0.0,
        )

    def stop_result_worker(self, timeout_seconds: float | None) -> None:
        super().stop_result_worker(timeout_seconds)
        raise callback_shared.NativeBgenWorkerShutdownError(
            worker_name="result-worker",
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 0.0,
        )


def test_callback_runtime_does_not_export_legacy_worker_timeout_constants() -> None:
    legacy_constant_names = (
        "DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
        "RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
        "GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
        "GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
        "WORKER_ABORT_STOP_TIMEOUT_SECONDS",
    )

    for legacy_constant_name in legacy_constant_names:
        assert not hasattr(callback_runtime, legacy_constant_name)
        assert not hasattr(callback_shared, legacy_constant_name)


@dataclasses.dataclass(frozen=True)
class CallbackFinishPlanProbe:
    finish_actions: list[str]
    dosage_stop_timeout_seconds: float
    dosage_join_timeout_seconds: float
    result_stop_timeout_seconds: float
    result_join_timeout_seconds: float

    @property
    def stop_dosage_worker(self) -> bool:
        return "stop_dosage_worker" in self.finish_actions

    @property
    def join_dosage_worker(self) -> bool:
        return "join_dosage_worker" in self.finish_actions

    @property
    def stop_result_worker(self) -> bool:
        return "stop_result_worker" in self.finish_actions

    @property
    def join_result_worker(self) -> bool:
        return "join_result_worker" in self.finish_actions

    @property
    def raise_worker_error(self) -> bool:
        return "raise_worker_error" in self.finish_actions

    @property
    def complete_progress(self) -> bool:
        return "complete_progress" in self.finish_actions

    @property
    def emit_binary_correction_summary(self) -> bool:
        return "emit_binary_correction_summary" in self.finish_actions


@dataclasses.dataclass(frozen=True)
class CallbackAbortPlanProbe:
    abort_actions: list[str]
    dosage_stop_timeout_seconds: float
    result_stop_timeout_seconds: float

    @property
    def stop_dosage_worker(self) -> bool:
        return "stop_dosage_worker" in self.abort_actions

    @property
    def stop_result_worker(self) -> bool:
        return "stop_result_worker" in self.abort_actions


@dataclasses.dataclass(frozen=True)
class CallbackStartAttemptPlanProbe:
    start_actions: list[str]
    should_start: bool
    has_marked_started: bool
    has_start_error: bool
    error_message: str | None

    @property
    def start_result_worker(self) -> bool:
        return "start_result_worker" in self.start_actions

    @property
    def start_dosage_worker(self) -> bool:
        return "start_dosage_worker" in self.start_actions


@dataclasses.dataclass(frozen=True)
class DosageWorkHandoffPlanProbe:
    chunk_count: int


@dataclasses.dataclass(frozen=True)
class VariantMajorDosageBatchHandoffPlanProbe:
    chunk_count: int


@dataclasses.dataclass(frozen=True)
class ResultWriteHandoffPlanProbe:
    should_enqueue: bool
    has_result_work_item: bool
    is_stop_signal: bool


@dataclasses.dataclass(frozen=True)
class WorkerErrorRaisePlanProbe:
    should_raise: bool
    raise_dosage_worker_error: bool
    raise_result_worker_error: bool
    error_message: str | None


@dataclasses.dataclass(frozen=True)
class WorkerErrorUpdatePlanProbe:
    had_error: bool
    has_error: bool
    error_message: str | None


@dataclasses.dataclass(frozen=True)
class CallbackQueuePutAttemptPlanProbe:
    should_put: bool
    should_wait: bool
    wait_timeout_seconds: float
    queue_depth: int
    queue_capacity: int


@dataclasses.dataclass(frozen=True)
class CallbackQueuePutObservationPlanProbe:
    queue_name: str
    operation_name: str
    blocked: bool
    should_retry_put: bool


@dataclasses.dataclass(frozen=True)
class CallbackQueueGetAttemptPlanProbe:
    should_get: bool
    should_wait: bool
    has_release_error: bool
    wait_timeout_seconds: float
    queue_depth: int
    queue_capacity: int


@dataclasses.dataclass(frozen=True)
class CallbackQueueGetObservationPlanProbe:
    queue_name: str
    operation_name: str
    blocked: bool


@dataclasses.dataclass(frozen=True)
class CallbackQueueBackpressureObservationProbe:
    queue_name: str
    operation_name: str
    queue_depth: int
    queue_capacity: int
    elapsed_seconds: float
    blocked_seconds: float


@dataclasses.dataclass(frozen=True)
class CallbackQueueStageBackpressureObservationProbe:
    queue_name: str
    operation_name: str
    stage_name: str
    queue_depth: int
    queue_capacity: int
    elapsed_seconds: float
    blocked_seconds: float


@dataclasses.dataclass(frozen=True)
class ResultInFlightAcquireAttemptPlanProbe:
    should_acquire: bool
    should_wait: bool
    wait_timeout_seconds: float
    occupied_count: int
    slot_limit: int


@dataclasses.dataclass(frozen=True)
class ResultInFlightAcquireObservationPlanProbe:
    resource_name: str
    operation_name: str
    blocked: bool
    should_retry_acquisition: bool


@dataclasses.dataclass(frozen=True)
class ResultInFlightReleaseAttemptPlanProbe:
    should_release: bool
    has_release_error: bool
    occupied_count: int
    slot_limit: int


@dataclasses.dataclass(frozen=True)
class ResultInFlightReleaseObservationPlanProbe:
    resource_name: str
    operation_name: str
    blocked: bool


@dataclasses.dataclass(frozen=True)
class ResultWriteItemResourceReleasePlanProbe:
    should_release_host_buffer: bool
    should_release_result_in_flight_slot: bool


@dataclasses.dataclass(frozen=True)
class ResultWriteDrainCompletionPlanProbe:
    should_stop: bool
    should_flush_binary_correction_diagnostics: bool


@dataclasses.dataclass(frozen=True)
class ResultWriteItemDispatchPlanProbe:
    result_work_item_kind: str
    expected_result_work_item_kind: str
    should_process_result_write_item: bool
    should_process_multi_result_write_item: bool
    has_dispatch_error: bool
    error_message: str | None


@dataclasses.dataclass(frozen=True)
class DosageWorkDrainCompletionPlanProbe:
    should_stop: bool


@dataclasses.dataclass(frozen=True)
class DosageWorkItemDispatchPlanProbe:
    dosage_work_item_kind: str
    should_process_sample_major_dosage: bool
    should_process_variant_major_dosage: bool
    should_process_variant_major_dosage_batch: bool
    should_process_variant_major_packed8_probability_pair: bool
    has_dispatch_error: bool
    error_message: str | None


@dataclasses.dataclass(frozen=True)
class DosageWorkItemStageDurationPlanProbe:
    chunk_count: int
    duration_per_chunk: float


@dataclasses.dataclass(frozen=True)
class DosageBufferAcquireAttemptPlanProbe:
    should_take_free_buffer: bool
    should_allocate: bool
    should_wait: bool
    wait_timeout_seconds: float
    free_buffer_count: int
    allocated_count: int
    buffer_limit: int


@dataclasses.dataclass(frozen=True)
class DosageBufferRegisterAttemptPlanProbe:
    should_register: bool
    has_registration_error: bool
    allocated_count: int
    buffer_limit: int


@dataclasses.dataclass(frozen=True)
class DosageBufferReturnAttemptPlanProbe:
    should_return: bool
    allocated_count: int
    buffer_limit: int


@dataclasses.dataclass(frozen=True)
class DosageBufferDiscardAttemptPlanProbe:
    should_discard: bool
    allocated_count: int
    buffer_limit: int


@dataclasses.dataclass(frozen=True)
class DosageBufferPoolObservationPlanProbe:
    operation_name: str
    blocked: bool


@dataclasses.dataclass(frozen=True)
class DosageBufferReusePlanProbe:
    requires_slice: bool
    slice_dimensions: list[int]


@dataclasses.dataclass
class CallbackSchedulerShutdownPlanProbe:
    finish_called: bool = False
    abort_called: bool = False

    def plan_worker_finish(self) -> CallbackFinishPlanProbe:
        self.finish_called = True
        return CallbackFinishPlanProbe(
            finish_actions=[
                "stop_dosage_worker",
                "join_dosage_worker",
                "stop_result_worker",
                "join_result_worker",
                "raise_worker_error",
                "complete_progress",
                "emit_binary_correction_summary",
            ],
            dosage_stop_timeout_seconds=2.0,
            dosage_join_timeout_seconds=3.0,
            result_stop_timeout_seconds=4.0,
            result_join_timeout_seconds=5.0,
        )

    def plan_worker_abort(self) -> CallbackAbortPlanProbe:
        self.abort_called = True
        return CallbackAbortPlanProbe(
            abort_actions=["stop_dosage_worker", "stop_result_worker"],
            dosage_stop_timeout_seconds=0.25,
            result_stop_timeout_seconds=0.5,
        )


@dataclasses.dataclass
class CallbackStartAttemptSchedulerProbe:
    start_attempt_count: int = 0

    def plan_worker_start_attempt(self) -> CallbackStartAttemptPlanProbe:
        self.start_attempt_count += 1
        if self.start_attempt_count > 1:
            return CallbackStartAttemptPlanProbe(
                start_actions=[],
                should_start=False,
                has_marked_started=False,
                has_start_error=False,
                error_message=None,
            )
        return CallbackStartAttemptPlanProbe(
            start_actions=["start_result_worker", "start_dosage_worker"],
            should_start=True,
            has_marked_started=True,
            has_start_error=False,
            error_message=None,
        )


@dataclasses.dataclass
class DosageWorkHandoffSchedulerProbe:
    handoff_chunk_counts: list[int] = dataclasses.field(default_factory=list)
    variant_major_batch_counts: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)

    def plan_dosage_work_handoff(self, chunk_count: int) -> DosageWorkHandoffPlanProbe:
        self.handoff_chunk_counts.append(chunk_count)
        return DosageWorkHandoffPlanProbe(chunk_count=chunk_count)

    def plan_variant_major_dosage_batch_handoff(
        self,
        metadata_count: int,
        genotype_matrix_by_variant_count: int,
        chunk_stats_count: int,
    ) -> VariantMajorDosageBatchHandoffPlanProbe:
        self.variant_major_batch_counts.append((metadata_count, genotype_matrix_by_variant_count, chunk_stats_count))
        return VariantMajorDosageBatchHandoffPlanProbe(chunk_count=metadata_count)


@dataclasses.dataclass
class CallbackQueueAttemptSchedulerProbe:
    dosage_put_wait_timeout_seconds: float | None = None
    dosage_put_backpressure_called: bool = False
    dosage_put_observation_queued_values: list[bool] = dataclasses.field(default_factory=list)
    dosage_get_called: bool = False
    result_put_wait_timeout_seconds: float | None = None
    result_put_backpressure_called: bool = False
    result_put_observation_queued_values: list[bool] = dataclasses.field(default_factory=list)
    result_get_called: bool = False
    result_handoff_has_work_items: list[bool] = dataclasses.field(default_factory=list)
    stage_observation_names: list[str] = dataclasses.field(default_factory=list)

    def plan_dosage_queue_put_attempt(self, wait_timeout_seconds: float) -> CallbackQueuePutAttemptPlanProbe:
        self.dosage_put_wait_timeout_seconds = wait_timeout_seconds
        return CallbackQueuePutAttemptPlanProbe(
            should_put=True,
            should_wait=False,
            wait_timeout_seconds=0.0,
            queue_depth=1,
            queue_capacity=1,
        )

    def plan_dosage_queue_put_backpressure_attempt(self) -> CallbackQueuePutAttemptPlanProbe:
        self.dosage_put_backpressure_called = True
        return CallbackQueuePutAttemptPlanProbe(
            should_put=True,
            should_wait=False,
            wait_timeout_seconds=0.0,
            queue_depth=1,
            queue_capacity=1,
        )

    def plan_dosage_queue_put_observation(self, *, queued: bool) -> CallbackQueuePutObservationPlanProbe:
        self.dosage_put_observation_queued_values.append(queued)
        return CallbackQueuePutObservationPlanProbe(
            queue_name="dosage_queue",
            operation_name="put" if queued else "producer_blocking",
            blocked=not queued,
            should_retry_put=not queued,
        )

    def plan_dosage_queue_get_attempt(self, *, has_queued_item: bool) -> CallbackQueueGetAttemptPlanProbe:
        self.dosage_get_called = True
        assert has_queued_item is True
        return CallbackQueueGetAttemptPlanProbe(
            should_get=True,
            should_wait=False,
            has_release_error=False,
            wait_timeout_seconds=0.0,
            queue_depth=0,
            queue_capacity=1,
        )

    def plan_result_queue_put_attempt(self, wait_timeout_seconds: float) -> CallbackQueuePutAttemptPlanProbe:
        self.result_put_wait_timeout_seconds = wait_timeout_seconds
        return CallbackQueuePutAttemptPlanProbe(
            should_put=True,
            should_wait=False,
            wait_timeout_seconds=0.0,
            queue_depth=1,
            queue_capacity=1,
        )

    def plan_result_queue_put_backpressure_attempt(self) -> CallbackQueuePutAttemptPlanProbe:
        self.result_put_backpressure_called = True
        return CallbackQueuePutAttemptPlanProbe(
            should_put=True,
            should_wait=False,
            wait_timeout_seconds=0.0,
            queue_depth=1,
            queue_capacity=1,
        )

    def plan_result_queue_put_observation(self, *, queued: bool) -> CallbackQueuePutObservationPlanProbe:
        self.result_put_observation_queued_values.append(queued)
        return CallbackQueuePutObservationPlanProbe(
            queue_name="result_queue",
            operation_name="put" if queued else "producer_blocking",
            blocked=not queued,
            should_retry_put=not queued,
        )

    def plan_result_write_handoff(self, *, has_result_work_item: bool) -> ResultWriteHandoffPlanProbe:
        self.result_handoff_has_work_items.append(has_result_work_item)
        return ResultWriteHandoffPlanProbe(
            should_enqueue=True,
            has_result_work_item=has_result_work_item,
            is_stop_signal=not has_result_work_item,
        )

    def plan_current_queue_stage_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueStageBackpressureObservationProbe:
        stage_name = {
            ("dosage_queue", "put"): "callback_queue_put",
            ("dosage_queue", "producer_blocking"): "callback_queue_producer_blocking",
            ("result_queue", "put"): "result_queue_put",
            ("result_queue", "producer_blocking"): "result_queue_producer_blocking",
        }[(queue_name, operation_name)]
        self.stage_observation_names.append(stage_name)
        return CallbackQueueStageBackpressureObservationProbe(
            queue_name=queue_name,
            operation_name=operation_name,
            stage_name=stage_name,
            queue_depth=1,
            queue_capacity=1,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_result_queue_get_attempt(self, *, has_queued_item: bool) -> CallbackQueueGetAttemptPlanProbe:
        self.result_get_called = True
        assert has_queued_item is True
        return CallbackQueueGetAttemptPlanProbe(
            should_get=True,
            should_wait=False,
            has_release_error=False,
            wait_timeout_seconds=0.0,
            queue_depth=0,
            queue_capacity=1,
        )


@dataclasses.dataclass
class CallbackConsumerObservationSchedulerProbe:
    get_observation_names: list[str] = dataclasses.field(default_factory=list)
    stage_observation_names: list[str] = dataclasses.field(default_factory=list)

    def plan_dosage_work_drain_completion(
        self,
        *,
        has_dosage_work_item: bool,
    ) -> DosageWorkDrainCompletionPlanProbe:
        return DosageWorkDrainCompletionPlanProbe(should_stop=not has_dosage_work_item)

    def plan_dosage_work_item_dispatch(
        self,
        *,
        dosage_work_item_kind: str,
    ) -> DosageWorkItemDispatchPlanProbe:
        return DosageWorkItemDispatchPlanProbe(
            dosage_work_item_kind=dosage_work_item_kind,
            should_process_sample_major_dosage=dosage_work_item_kind == "sample_major_dosage",
            should_process_variant_major_dosage=False,
            should_process_variant_major_dosage_batch=False,
            should_process_variant_major_packed8_probability_pair=False,
            has_dispatch_error=False,
            error_message=None,
        )

    def plan_dosage_work_item_stage_duration(
        self,
        *,
        dosage_work_item_kind: str,
        chunk_count: int,
        elapsed_seconds: float,
    ) -> DosageWorkItemStageDurationPlanProbe:
        del dosage_work_item_kind
        return DosageWorkItemStageDurationPlanProbe(
            chunk_count=chunk_count,
            duration_per_chunk=elapsed_seconds / chunk_count,
        )

    def plan_dosage_queue_get_observation(self) -> CallbackQueueGetObservationPlanProbe:
        self.get_observation_names.append("dosage_queue")
        return CallbackQueueGetObservationPlanProbe(
            queue_name="dosage_queue",
            operation_name="consumer_wait",
            blocked=True,
        )

    def plan_result_write_drain_completion(
        self,
        *,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> ResultWriteDrainCompletionPlanProbe:
        del flush_binary_correction_diagnostics_on_stop
        return ResultWriteDrainCompletionPlanProbe(
            should_stop=not has_result_work_item,
            should_flush_binary_correction_diagnostics=False,
        )

    def plan_result_write_item_dispatch(
        self,
        *,
        result_work_item_kind: str,
        expected_result_work_item_kind: str,
    ) -> ResultWriteItemDispatchPlanProbe:
        return ResultWriteItemDispatchPlanProbe(
            result_work_item_kind=result_work_item_kind,
            expected_result_work_item_kind=expected_result_work_item_kind,
            should_process_result_write_item=result_work_item_kind == "single_result",
            should_process_multi_result_write_item=False,
            has_dispatch_error=False,
            error_message=None,
        )

    def plan_result_queue_get_observation(self) -> CallbackQueueGetObservationPlanProbe:
        self.get_observation_names.append("result_queue")
        return CallbackQueueGetObservationPlanProbe(
            queue_name="result_queue",
            operation_name="consumer_wait",
            blocked=True,
        )

    def plan_current_queue_stage_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueStageBackpressureObservationProbe:
        stage_name = {
            ("dosage_queue", "consumer_wait"): "callback_queue_consumer_wait",
            ("result_queue", "consumer_wait"): "result_queue_consumer_wait",
        }[(queue_name, operation_name)]
        self.stage_observation_names.append(stage_name)
        return CallbackQueueStageBackpressureObservationProbe(
            queue_name=queue_name,
            operation_name=operation_name,
            stage_name=stage_name,
            queue_depth=1,
            queue_capacity=1,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )


@dataclasses.dataclass
class WorkerErrorRaiseSchedulerProbe:
    error_raise_called: bool = False

    def plan_worker_error_raise(self) -> WorkerErrorRaisePlanProbe:
        self.error_raise_called = True
        return WorkerErrorRaisePlanProbe(
            should_raise=True,
            raise_dosage_worker_error=True,
            raise_result_worker_error=False,
            error_message="planned dosage worker failure",
        )


@dataclasses.dataclass
class WorkerErrorUpdateSchedulerProbe:
    dosage_error_message: str | None = None
    result_error_message: str | None = None

    def update_dosage_worker_error(self, error_message: str | None) -> WorkerErrorUpdatePlanProbe:
        had_error = self.dosage_error_message is not None
        self.dosage_error_message = error_message
        return WorkerErrorUpdatePlanProbe(
            had_error=had_error,
            has_error=self.dosage_error_message is not None,
            error_message=self.dosage_error_message,
        )

    def update_result_worker_error(self, error_message: str | None) -> WorkerErrorUpdatePlanProbe:
        had_error = self.result_error_message is not None
        self.result_error_message = error_message
        return WorkerErrorUpdatePlanProbe(
            had_error=had_error,
            has_error=self.result_error_message is not None,
            error_message=self.result_error_message,
        )


@dataclasses.dataclass
class CallbackObservationSchedulerProbe:
    current_queue_observation_called: bool = False
    current_queue_stage_observation_called: bool = False
    dosage_buffer_pool_observation_called: bool = False
    dosage_buffer_pool_stage_observation_called: bool = False
    observed_resource_name: str | None = None
    observed_operation_name: str | None = None
    observed_free_buffer_count: int | None = None

    def plan_current_queue_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueBackpressureObservationProbe:
        self.current_queue_observation_called = True
        self.observed_resource_name = queue_name
        self.observed_operation_name = operation_name
        return CallbackQueueBackpressureObservationProbe(
            queue_name=queue_name,
            operation_name=operation_name,
            queue_depth=1,
            queue_capacity=2,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_current_queue_stage_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueStageBackpressureObservationProbe:
        self.current_queue_stage_observation_called = True
        self.observed_resource_name = queue_name
        self.observed_operation_name = operation_name
        return CallbackQueueStageBackpressureObservationProbe(
            queue_name=queue_name,
            operation_name=operation_name,
            stage_name="callback_queue_put",
            queue_depth=1,
            queue_capacity=2,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_dosage_buffer_pool_backpressure_observation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueBackpressureObservationProbe:
        self.dosage_buffer_pool_observation_called = True
        self.observed_operation_name = operation_name
        self.observed_free_buffer_count = free_buffer_count
        return CallbackQueueBackpressureObservationProbe(
            queue_name="dosage_buffer_pool",
            operation_name=operation_name,
            queue_depth=free_buffer_count,
            queue_capacity=3,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_dosage_buffer_pool_stage_backpressure_observation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueStageBackpressureObservationProbe:
        self.dosage_buffer_pool_stage_observation_called = True
        self.observed_operation_name = operation_name
        self.observed_free_buffer_count = free_buffer_count
        return CallbackQueueStageBackpressureObservationProbe(
            queue_name="dosage_buffer_pool",
            operation_name=operation_name,
            stage_name="dosage_buffer_pool_consumer_wait",
            queue_depth=free_buffer_count,
            queue_capacity=3,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )


@dataclasses.dataclass
class ResultInFlightAttemptSchedulerProbe:
    result_in_flight_limit: int = 1
    dosage_worker_error_message: str | None = None
    result_worker_error_message: str | None = None
    acquire_wait_timeout_seconds: float | None = None
    acquire_backpressure_called: bool = False
    acquire_observation_called: bool = False
    release_observation_called: bool = False
    operation_observation_called: bool = False
    stage_observation_called: bool = False
    release_called: bool = False
    observed_resource_name: str | None = None
    observed_operation_name: str | None = None

    @property
    def backpressure_poll_timeout_seconds(self) -> float:
        message = "Runtime should use the native result-slot backpressure attempt plan."
        raise AssertionError(message)

    def plan_worker_error_raise(self) -> WorkerErrorRaisePlanProbe:
        return WorkerErrorRaisePlanProbe(
            should_raise=False,
            raise_dosage_worker_error=False,
            raise_result_worker_error=False,
            error_message=None,
        )

    def plan_result_in_flight_slot_acquire_attempt(
        self,
        wait_timeout_seconds: float,
    ) -> ResultInFlightAcquireAttemptPlanProbe:
        self.acquire_wait_timeout_seconds = wait_timeout_seconds
        return ResultInFlightAcquireAttemptPlanProbe(
            should_acquire=True,
            should_wait=False,
            wait_timeout_seconds=0.0,
            occupied_count=1,
            slot_limit=1,
        )

    def plan_result_in_flight_slot_acquire_backpressure_attempt(self) -> ResultInFlightAcquireAttemptPlanProbe:
        self.acquire_backpressure_called = True
        self.acquire_wait_timeout_seconds = 0.1
        return ResultInFlightAcquireAttemptPlanProbe(
            should_acquire=True,
            should_wait=False,
            wait_timeout_seconds=0.0,
            occupied_count=1,
            slot_limit=1,
        )

    def plan_result_in_flight_slot_acquire_observation(
        self,
        acquire_attempt_plan: ResultInFlightAcquireAttemptPlanProbe,
    ) -> ResultInFlightAcquireObservationPlanProbe:
        self.acquire_observation_called = True
        assert acquire_attempt_plan.should_acquire is True
        return ResultInFlightAcquireObservationPlanProbe(
            resource_name="result_in_flight_slots",
            operation_name="acquire",
            blocked=False,
            should_retry_acquisition=False,
        )

    def plan_result_in_flight_slot_release_observation(self) -> ResultInFlightReleaseObservationPlanProbe:
        self.release_observation_called = True
        return ResultInFlightReleaseObservationPlanProbe(
            resource_name="result_in_flight_slots",
            operation_name="release",
            blocked=False,
        )

    def plan_current_queue_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueBackpressureObservationProbe:
        self.operation_observation_called = True
        self.observed_resource_name = queue_name
        self.observed_operation_name = operation_name
        return CallbackQueueBackpressureObservationProbe(
            queue_name=queue_name,
            operation_name=operation_name,
            queue_depth=1,
            queue_capacity=1,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_current_queue_stage_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueStageBackpressureObservationProbe:
        self.stage_observation_called = True
        self.observed_resource_name = queue_name
        self.observed_operation_name = operation_name
        return CallbackQueueStageBackpressureObservationProbe(
            queue_name=queue_name,
            operation_name=operation_name,
            stage_name="result_in_flight_slot_acquire",
            queue_depth=1,
            queue_capacity=1,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_result_in_flight_slot_release_attempt(self) -> ResultInFlightReleaseAttemptPlanProbe:
        self.release_called = True
        return ResultInFlightReleaseAttemptPlanProbe(
            should_release=True,
            has_release_error=False,
            occupied_count=0,
            slot_limit=1,
        )


@dataclasses.dataclass
class ResultWriteItemResourceReleaseSchedulerProbe:
    pre_write_has_host_dosage_buffer: bool | None = None
    final_has_host_dosage_buffer: bool | None = None
    final_has_released_host_dosage_buffer: bool | None = None
    final_release_in_flight_slot: bool | None = None
    returned_buffer_identifier: int | None = None
    release_called: bool = False

    def plan_result_write_item_pre_write_resource_release(
        self,
        *,
        has_host_dosage_buffer: bool,
    ) -> ResultWriteItemResourceReleasePlanProbe:
        self.pre_write_has_host_dosage_buffer = has_host_dosage_buffer
        return ResultWriteItemResourceReleasePlanProbe(
            should_release_host_buffer=has_host_dosage_buffer,
            should_release_result_in_flight_slot=False,
        )

    def plan_result_write_item_final_resource_release(
        self,
        *,
        has_host_dosage_buffer: bool,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> ResultWriteItemResourceReleasePlanProbe:
        self.final_has_host_dosage_buffer = has_host_dosage_buffer
        self.final_has_released_host_dosage_buffer = has_released_host_dosage_buffer
        self.final_release_in_flight_slot = release_in_flight_slot
        return ResultWriteItemResourceReleasePlanProbe(
            should_release_host_buffer=has_host_dosage_buffer and not has_released_host_dosage_buffer,
            should_release_result_in_flight_slot=release_in_flight_slot,
        )

    def plan_dosage_buffer_return_attempt(self, buffer_identifier: int) -> DosageBufferReturnAttemptPlanProbe:
        self.returned_buffer_identifier = buffer_identifier
        return DosageBufferReturnAttemptPlanProbe(
            should_return=True,
            allocated_count=1,
            buffer_limit=1,
        )

    def plan_result_in_flight_slot_release_attempt(self) -> ResultInFlightReleaseAttemptPlanProbe:
        self.release_called = True
        return ResultInFlightReleaseAttemptPlanProbe(
            should_release=True,
            has_release_error=False,
            occupied_count=0,
            slot_limit=1,
        )


@dataclasses.dataclass
class ResultWriteDrainCompletionSchedulerProbe:
    has_result_work_item: bool | None = None
    flush_binary_correction_diagnostics_on_stop: bool | None = None

    def plan_result_write_drain_completion(
        self,
        *,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> ResultWriteDrainCompletionPlanProbe:
        self.has_result_work_item = has_result_work_item
        self.flush_binary_correction_diagnostics_on_stop = flush_binary_correction_diagnostics_on_stop
        return ResultWriteDrainCompletionPlanProbe(
            should_stop=not has_result_work_item,
            should_flush_binary_correction_diagnostics=(
                not has_result_work_item and flush_binary_correction_diagnostics_on_stop
            ),
        )


@dataclasses.dataclass
class ResultWriteItemDispatchSchedulerProbe:
    result_work_item_kind: str | None = None
    expected_result_work_item_kind: str | None = None

    def plan_result_write_item_dispatch(
        self,
        *,
        result_work_item_kind: str,
        expected_result_work_item_kind: str,
    ) -> ResultWriteItemDispatchPlanProbe:
        self.result_work_item_kind = result_work_item_kind
        self.expected_result_work_item_kind = expected_result_work_item_kind
        return ResultWriteItemDispatchPlanProbe(
            result_work_item_kind=result_work_item_kind,
            expected_result_work_item_kind=expected_result_work_item_kind,
            should_process_result_write_item=result_work_item_kind == "single_result",
            should_process_multi_result_write_item=result_work_item_kind == "multi_result",
            has_dispatch_error=False,
            error_message=None,
        )


@dataclasses.dataclass
class DosageWorkDrainCompletionSchedulerProbe:
    has_dosage_work_item: bool | None = None

    def plan_dosage_work_drain_completion(
        self,
        *,
        has_dosage_work_item: bool,
    ) -> DosageWorkDrainCompletionPlanProbe:
        self.has_dosage_work_item = has_dosage_work_item
        return DosageWorkDrainCompletionPlanProbe(should_stop=not has_dosage_work_item)


@dataclasses.dataclass
class DosageWorkItemDispatchSchedulerProbe:
    dosage_work_item_kind: str | None = None

    def plan_dosage_work_item_dispatch(
        self,
        *,
        dosage_work_item_kind: str,
    ) -> DosageWorkItemDispatchPlanProbe:
        self.dosage_work_item_kind = dosage_work_item_kind
        return DosageWorkItemDispatchPlanProbe(
            dosage_work_item_kind=dosage_work_item_kind,
            should_process_sample_major_dosage=dosage_work_item_kind == "sample_major_dosage",
            should_process_variant_major_dosage=dosage_work_item_kind == "variant_major_dosage",
            should_process_variant_major_dosage_batch=dosage_work_item_kind == "variant_major_dosage_batch",
            should_process_variant_major_packed8_probability_pair=(
                dosage_work_item_kind == "variant_major_packed8_probability_pair"
            ),
            has_dispatch_error=False,
            error_message=None,
        )


@dataclasses.dataclass
class DosageWorkItemStageDurationSchedulerProbe:
    dosage_work_item_kind: str | None = None
    chunk_count: int | None = None
    elapsed_seconds: float | None = None

    def plan_dosage_work_item_stage_duration(
        self,
        *,
        dosage_work_item_kind: str,
        chunk_count: int,
        elapsed_seconds: float,
    ) -> DosageWorkItemStageDurationPlanProbe:
        self.dosage_work_item_kind = dosage_work_item_kind
        self.chunk_count = chunk_count
        self.elapsed_seconds = elapsed_seconds
        return DosageWorkItemStageDurationPlanProbe(
            chunk_count=chunk_count,
            duration_per_chunk=elapsed_seconds / chunk_count,
        )


@dataclasses.dataclass
class DosageBufferAttemptSchedulerProbe:
    dosage_buffer_limit: int = 2
    dosage_worker_error_message: str | None = None
    result_worker_error_message: str | None = None
    acquire_free_buffer_counts: list[int] = dataclasses.field(default_factory=list)
    acquire_wait_timeout_seconds: float | None = None
    acquire_backpressure_called: bool = False
    registered_buffer_identifier: int | None = None
    returned_buffer_identifier: int | None = None
    discarded_buffer_identifier: int | None = None
    reuse_buffered_shape: tuple[int, ...] | None = None
    reuse_expected_shape: tuple[int, ...] | None = None
    pool_observation_names: list[str] = dataclasses.field(default_factory=list)
    pool_backpressure_names: list[str] = dataclasses.field(default_factory=list)
    pool_stage_names: list[str] = dataclasses.field(default_factory=list)

    @property
    def backpressure_poll_timeout_seconds(self) -> float:
        message = "Runtime should use the native dosage-buffer backpressure attempt plan."
        raise AssertionError(message)

    def plan_worker_error_raise(self) -> WorkerErrorRaisePlanProbe:
        return WorkerErrorRaisePlanProbe(
            should_raise=False,
            raise_dosage_worker_error=False,
            raise_result_worker_error=False,
            error_message=None,
        )

    def plan_dosage_buffer_acquire_attempt(
        self,
        free_buffer_count: int,
        wait_timeout_seconds: float,
    ) -> DosageBufferAcquireAttemptPlanProbe:
        self.acquire_free_buffer_counts.append(free_buffer_count)
        self.acquire_wait_timeout_seconds = wait_timeout_seconds
        return DosageBufferAcquireAttemptPlanProbe(
            should_take_free_buffer=free_buffer_count > 0,
            should_allocate=free_buffer_count == 0,
            should_wait=False,
            wait_timeout_seconds=0.0,
            free_buffer_count=free_buffer_count,
            allocated_count=0,
            buffer_limit=self.dosage_buffer_limit,
        )

    def plan_dosage_buffer_acquire_backpressure_attempt(
        self,
        free_buffer_count: int,
    ) -> DosageBufferAcquireAttemptPlanProbe:
        self.acquire_backpressure_called = True
        self.acquire_free_buffer_counts.append(free_buffer_count)
        self.acquire_wait_timeout_seconds = 0.1
        return DosageBufferAcquireAttemptPlanProbe(
            should_take_free_buffer=free_buffer_count > 0,
            should_allocate=free_buffer_count == 0,
            should_wait=False,
            wait_timeout_seconds=0.0,
            free_buffer_count=free_buffer_count,
            allocated_count=0,
            buffer_limit=self.dosage_buffer_limit,
        )

    def plan_dosage_buffer_register_attempt(self, buffer_identifier: int) -> DosageBufferRegisterAttemptPlanProbe:
        self.registered_buffer_identifier = buffer_identifier
        return DosageBufferRegisterAttemptPlanProbe(
            should_register=True,
            has_registration_error=False,
            allocated_count=1,
            buffer_limit=self.dosage_buffer_limit,
        )

    def plan_dosage_buffer_return_attempt(self, buffer_identifier: int) -> DosageBufferReturnAttemptPlanProbe:
        self.returned_buffer_identifier = buffer_identifier
        return DosageBufferReturnAttemptPlanProbe(
            should_return=True,
            allocated_count=1,
            buffer_limit=self.dosage_buffer_limit,
        )

    def plan_dosage_buffer_discard_attempt(self, buffer_identifier: int) -> DosageBufferDiscardAttemptPlanProbe:
        self.discarded_buffer_identifier = buffer_identifier
        return DosageBufferDiscardAttemptPlanProbe(
            should_discard=True,
            allocated_count=0,
            buffer_limit=self.dosage_buffer_limit,
        )

    def plan_dosage_buffer_pool_reuse_observation(self) -> DosageBufferPoolObservationPlanProbe:
        self.pool_observation_names.append("reuse")
        return DosageBufferPoolObservationPlanProbe(operation_name="reuse", blocked=False)

    def plan_dosage_buffer_pool_return_observation(self) -> DosageBufferPoolObservationPlanProbe:
        self.pool_observation_names.append("return")
        return DosageBufferPoolObservationPlanProbe(operation_name="return", blocked=False)

    def plan_dosage_buffer_pool_allocate_observation(self) -> DosageBufferPoolObservationPlanProbe:
        self.pool_observation_names.append("allocate")
        return DosageBufferPoolObservationPlanProbe(operation_name="allocate", blocked=False)

    def plan_dosage_buffer_pool_discard_observation(self) -> DosageBufferPoolObservationPlanProbe:
        self.pool_observation_names.append("discard")
        return DosageBufferPoolObservationPlanProbe(operation_name="discard", blocked=False)

    def plan_dosage_buffer_pool_consumer_wait_observation(self) -> DosageBufferPoolObservationPlanProbe:
        self.pool_observation_names.append("consumer_wait")
        return DosageBufferPoolObservationPlanProbe(operation_name="consumer_wait", blocked=True)

    def plan_dosage_buffer_pool_backpressure_observation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueBackpressureObservationProbe:
        self.pool_backpressure_names.append(operation_name)
        return CallbackQueueBackpressureObservationProbe(
            queue_name="dosage_buffer_pool",
            operation_name=operation_name,
            queue_depth=free_buffer_count,
            queue_capacity=self.dosage_buffer_limit,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_dosage_buffer_pool_stage_backpressure_observation(
        self,
        *,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> CallbackQueueStageBackpressureObservationProbe:
        self.pool_stage_names.append(operation_name)
        return CallbackQueueStageBackpressureObservationProbe(
            queue_name="dosage_buffer_pool",
            operation_name=operation_name,
            stage_name="dosage_buffer_pool_consumer_wait",
            queue_depth=free_buffer_count,
            queue_capacity=self.dosage_buffer_limit,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=elapsed_seconds if blocked else 0.0,
        )

    def plan_dosage_buffer_reuse(
        self,
        buffered_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ) -> DosageBufferReusePlanProbe | None:
        self.reuse_buffered_shape = buffered_shape
        self.reuse_expected_shape = expected_shape
        if buffered_shape != expected_shape:
            return None
        return DosageBufferReusePlanProbe(
            requires_slice=False,
            slice_dimensions=list(expected_shape),
        )


def test_native_callback_runner_uses_scheduler_finish_shutdown_plan() -> None:
    callback = RecordingShutdownCallbackRunner()
    scheduler_state = CallbackSchedulerShutdownPlanProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    callback.finish()

    assert scheduler_state.finish_called is True
    assert callback.shutdown_calls == [
        "stop_dosage",
        "join_dosage",
        "stop_result",
        "join_result",
        "raise_worker_error",
        "complete_progress",
        "emit_binary_correction_summary",
    ]
    assert callback.shutdown_timeouts == {
        "stop_dosage": 2.0,
        "join_dosage": 3.0,
        "stop_result": 4.0,
        "join_result": 5.0,
    }


def test_native_callback_runner_uses_scheduler_abort_shutdown_plan() -> None:
    callback = FailingStopShutdownCallbackRunner()
    scheduler_state = CallbackSchedulerShutdownPlanProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    callback.abort()

    assert scheduler_state.abort_called is True
    assert callback.shutdown_calls == ["stop_dosage", "stop_result"]
    assert callback.shutdown_timeouts == {
        "stop_dosage": 0.25,
        "stop_result": 0.5,
    }


def test_native_callback_runner_uses_scheduler_queue_attempt_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = CallbackQueueAttemptSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    assert callback.try_put_dosage_work_item(None, timeout_seconds=0.5) is True
    assert callback.get_dosage_work_item() is None
    assert callback.try_put_result_write_item(None, timeout_seconds=0.5) is True
    assert callback.get_result_write_item() is None

    dosage_put_wait_timeout_seconds = scheduler_state.dosage_put_wait_timeout_seconds
    assert dosage_put_wait_timeout_seconds is not None
    assert 0.0 < dosage_put_wait_timeout_seconds <= 0.5
    assert scheduler_state.dosage_get_called is True
    result_put_wait_timeout_seconds = scheduler_state.result_put_wait_timeout_seconds
    assert result_put_wait_timeout_seconds is not None
    assert 0.0 < result_put_wait_timeout_seconds <= 0.5
    assert scheduler_state.result_get_called is True


def test_native_callback_runner_uses_scheduler_queue_backpressure_attempt_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = CallbackQueueAttemptSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    assert callback.try_put_dosage_work_item_with_backpressure_timeout(None) is True
    assert callback.get_dosage_work_item() is None
    assert callback.try_put_result_write_item_with_backpressure_timeout(None) is True
    assert callback.get_result_write_item() is None

    assert scheduler_state.dosage_put_backpressure_called is True
    assert scheduler_state.result_put_backpressure_called is True


def test_native_callback_runner_uses_scheduler_queue_put_observation_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = CallbackQueueAttemptSchedulerProbe()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    callback_for_observation = typing.cast("typing.Any", callback)
    callback_for_observation.callback_scheduler_state = scheduler_state
    callback.stage_timing_recorder = stage_timing_recorder
    callback_for_observation.start = lambda: None
    callback_for_observation.raise_worker_error_if_present = lambda: None

    def try_put_dosage_work_item_success(work_item: object) -> bool:
        del work_item
        return True

    def try_put_result_write_item_success(work_item: object) -> bool:
        del work_item
        return True

    callback_for_observation.try_put_dosage_work_item_with_backpressure_timeout = try_put_dosage_work_item_success
    callback_for_observation.try_put_result_write_item_with_backpressure_timeout = try_put_result_write_item_success

    callback.put_dosage_work_item(None)
    callback.put_result_write_item(None)

    assert scheduler_state.dosage_put_observation_queued_values == [True]
    assert scheduler_state.result_put_observation_queued_values == [True]
    assert scheduler_state.stage_observation_names == ["callback_queue_put", "result_queue_put"]


def test_native_callback_object_queue_preserves_fifo_capacity_and_sentinel_payloads() -> None:
    callback_queue = callback_runtime._core.NativeCallbackObjectQueue(2)
    first_item = object()

    assert callback_queue.capacity == 2
    assert callback_queue.occupied_count == 0
    assert callback_queue.has_available_slot is True
    assert callback_queue.has_queued_item is False
    assert callback_queue.wait_for_queued_item(timeout_seconds=0.0) is False
    assert callback_queue.put(first_item, timeout_seconds=0.0) is True
    assert callback_queue.has_queued_item is True
    assert callback_queue.put(None, timeout_seconds=0.0) is True
    assert callback_queue.has_available_slot is False
    assert callback_queue.wait_for_available_slot(timeout_seconds=0.0) is False
    assert callback_queue.put(object(), timeout_seconds=0.0) is False
    assert callback_queue.occupied_count == 2

    first_result = callback_queue.get(timeout_seconds=0.0)
    assert first_result.has_item is True
    assert first_result.item is first_item
    assert callback_queue.wait_for_available_slot(timeout_seconds=0.0) is True
    second_result = callback_queue.get(timeout_seconds=0.0)
    assert second_result.has_item is True
    assert second_result.item is None
    empty_result = callback_queue.get(timeout_seconds=0.0)
    assert empty_result.has_item is False
    assert empty_result.item is None
    assert callback_queue.occupied_count == 0


def test_native_callback_object_queue_waits_for_producer_without_gil_deadlock() -> None:
    callback_queue = callback_runtime._core.NativeCallbackObjectQueue(1)
    produced_item = object()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callback_queue.get, 2.0)
        time.sleep(0.1)
        assert not future.done()
        assert callback_queue.put(produced_item, timeout_seconds=0.0) is True
        get_result = future.result(timeout=2.0)

    assert get_result.has_item is True
    assert get_result.item is produced_item


def test_native_callback_wait_signal_tracks_generation_without_gil_deadlock() -> None:
    wait_signal = callback_runtime._core.NativeCallbackWaitSignal()
    observed_generation = wait_signal.generation

    assert wait_signal.wait_for_change(observed_generation, timeout_seconds=0.0) is False

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait_signal.wait_for_change, observed_generation, 2.0)
        time.sleep(0.1)
        assert not future.done()
        notified_generation = wait_signal.notify_waiters()
        assert notified_generation != observed_generation
        assert future.result(timeout=2.0) is True

    assert wait_signal.generation == notified_generation
    assert wait_signal.wait_for_change(observed_generation, timeout_seconds=0.0) is True


def test_native_callback_runner_uses_scheduler_queue_get_observation_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = CallbackConsumerObservationSchedulerProbe()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    callback_for_observation = typing.cast("typing.Any", callback)
    callback_for_observation.callback_scheduler_state = scheduler_state
    callback.stage_timing_recorder = stage_timing_recorder
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())
    dosage_work_item = callback_shared.PreprocessedDosageChunkWorkItem(
        metadata=metadata,
        genotype_matrix=np.ones((2, 2), dtype=np.float32),
        chunk_stats=chunk_stats,
    )
    result_work_item = callback_shared.Regenie2ResultWriteWorkItem(
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=None,
        host_dosage_buffer=None,
        release_in_flight_slot=False,
        binary_chunk_diagnostics=None,
    )
    dosage_work_items = iter((dosage_work_item, None))
    result_work_items = iter((result_work_item, None))
    processed_dosage_items: list[object] = []
    processed_result_items: list[object] = []

    def get_dosage_work_item_from_probe() -> object:
        return next(dosage_work_items)

    def get_result_write_item_from_probe() -> object:
        return next(result_work_items)

    def process_dosage_work_item_probe(work_item: object, dispatch_plan: object) -> None:
        del dispatch_plan
        processed_dosage_items.append(work_item)

    def process_result_write_item_probe(work_item: object) -> None:
        processed_result_items.append(work_item)

    callback_for_observation.get_dosage_work_item = get_dosage_work_item_from_probe
    callback_for_observation.get_result_write_item = get_result_write_item_from_probe
    callback_for_observation.process_dosage_work_item_with_dispatch_plan = process_dosage_work_item_probe
    callback_for_observation.process_result_write_item = process_result_write_item_probe

    callback.consume_dosage_chunks()
    callback.consume_result_write_items()

    assert processed_dosage_items == [dosage_work_item]
    assert processed_result_items == [result_work_item]
    assert scheduler_state.get_observation_names == ["dosage_queue", "result_queue"]
    assert scheduler_state.stage_observation_names == [
        "callback_queue_consumer_wait",
        "result_queue_consumer_wait",
    ]


def test_native_callback_runner_uses_scheduler_result_write_handoff_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = CallbackQueueAttemptSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    result_work_item = typing.cast("typing.Any", SimpleNamespace())

    assert callback.try_put_result_write_item(result_work_item, timeout_seconds=0.5) is True
    assert callback.get_result_write_item() is result_work_item
    assert callback.try_put_result_write_item_with_backpressure_timeout(None) is True
    assert callback.get_result_write_item() is None

    assert scheduler_state.result_handoff_has_work_items == [True, False]


def test_native_callback_runner_uses_scheduler_resource_observation_plans() -> None:
    callback = ManualCallbackRunner()
    callback.stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    scheduler_state = CallbackObservationSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    callback.record_bounded_resource_operation(
        resource_name="result_in_flight_slots",
        operation_name="release",
        elapsed_seconds=0.25,
        blocked=False,
    )
    callback.record_bounded_resource_stage_duration(
        resource_name="dosage_queue",
        operation_name="put",
        start_time=time.perf_counter(),
        blocked=False,
    )
    callback.record_dosage_buffer_pool_operation(
        operation_name="return",
        free_buffer_count=2,
        elapsed_seconds=0.25,
        blocked=False,
    )
    callback.record_dosage_buffer_pool_stage_duration(
        operation_name="consumer_wait",
        free_buffer_count=1,
        start_time=time.perf_counter(),
        blocked=True,
    )

    assert scheduler_state.current_queue_observation_called is True
    assert scheduler_state.current_queue_stage_observation_called is True
    assert scheduler_state.dosage_buffer_pool_observation_called is True
    assert scheduler_state.dosage_buffer_pool_stage_observation_called is True
    assert scheduler_state.observed_free_buffer_count == 1


def test_native_callback_runner_uses_scheduler_worker_error_raise_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = WorkerErrorRaiseSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    callback.worker_error_cause = ValueError("dosage cause")

    with pytest.raises(RuntimeError, match="planned dosage worker failure") as error_info:
        callback.raise_worker_error_if_present()

    assert scheduler_state.error_raise_called is True
    assert error_info.value.__cause__ is callback.worker_error_cause


def test_native_callback_runner_uses_scheduler_worker_error_update_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = WorkerErrorUpdateSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    dosage_error = ValueError("dosage failed")
    result_error = RuntimeError("writer failed")
    callback.worker_error = dosage_error
    callback.result_worker_error = result_error

    assert callback.worker_error is dosage_error
    assert callback.result_worker_error is result_error
    assert scheduler_state.dosage_error_message == "dosage failed"
    assert scheduler_state.result_error_message == "writer failed"

    callback.worker_error = None

    assert callback.worker_error is None
    assert scheduler_state.dosage_error_message is None


def test_native_callback_runner_uses_scheduler_result_in_flight_attempt_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = ResultInFlightAttemptSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    callback.acquire_result_in_flight_slot()
    callback.release_result_in_flight_slot()

    acquire_wait_timeout_seconds = scheduler_state.acquire_wait_timeout_seconds
    assert acquire_wait_timeout_seconds is not None
    assert acquire_wait_timeout_seconds == 0.1
    assert scheduler_state.acquire_backpressure_called is True
    assert scheduler_state.release_called is True


def test_native_callback_runner_uses_scheduler_result_in_flight_acquire_observation_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = ResultInFlightAttemptSchedulerProbe()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    callback.stage_timing_recorder = stage_timing_recorder

    callback.acquire_result_in_flight_slot()

    assert scheduler_state.acquire_observation_called is True
    assert scheduler_state.stage_observation_called is True
    assert scheduler_state.observed_resource_name == "result_in_flight_slots"
    assert scheduler_state.observed_operation_name == "acquire"
    assert stage_timing_recorder.snapshot().stage_counts["result_in_flight_slot_acquire"] == 1


def test_native_callback_runner_uses_scheduler_result_in_flight_release_observation_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = ResultInFlightAttemptSchedulerProbe()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    callback.stage_timing_recorder = stage_timing_recorder

    callback.release_result_in_flight_slot()

    assert scheduler_state.release_called is True
    assert scheduler_state.release_observation_called is True
    assert scheduler_state.operation_observation_called is True
    assert scheduler_state.observed_resource_name == "result_in_flight_slots"
    assert scheduler_state.observed_operation_name == "release"


def test_native_callback_runner_uses_scheduler_dosage_buffer_pool_observation_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = DosageBufferAttemptSchedulerProbe()
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    callback.stage_timing_recorder = stage_timing_recorder

    dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    callback.release_dosage_buffer(dosage_buffer)
    reused_dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=3)
    callback.discard_dosage_buffer_slot(reused_dosage_buffer)
    callback.record_dosage_buffer_pool_consumer_wait_stage_duration(
        free_buffer_count=0,
        start_time=time.perf_counter(),
    )

    assert scheduler_state.pool_observation_names == [
        "allocate",
        "return",
        "reuse",
        "discard",
        "consumer_wait",
    ]
    assert scheduler_state.pool_backpressure_names == ["allocate", "return", "reuse", "discard"]
    assert scheduler_state.pool_stage_names == ["consumer_wait"]


def test_native_callback_runner_uses_scheduler_result_write_item_resource_release_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = ResultWriteItemResourceReleaseSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    host_dosage_buffer = typing.cast(
        "callback_shared.HostGenotypeBuffer",
        np.empty((2, 2), dtype=np.float32),
    )

    callback.release_result_work_item_buffer(
        callback_shared.Regenie2ResultWriteWorkItem(
            metadata=build_native_metadata(),
            chunk_stats=typing.cast("typing.Any", ExplodingChunkStats()),
            beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
            standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
            chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            extra_code=None,
            host_dosage_buffer=host_dosage_buffer,
            release_in_flight_slot=True,
            binary_chunk_diagnostics=None,
        )
    )

    assert scheduler_state.pre_write_has_host_dosage_buffer is True
    assert scheduler_state.final_has_host_dosage_buffer is True
    assert scheduler_state.final_has_released_host_dosage_buffer is True
    assert scheduler_state.final_release_in_flight_slot is True
    assert scheduler_state.returned_buffer_identifier == id(host_dosage_buffer)
    assert scheduler_state.release_called is True
    free_buffer_result = callback.free_dosage_buffers.get(timeout_seconds=0.0)
    assert free_buffer_result.has_item is True
    assert free_buffer_result.item is host_dosage_buffer


def test_native_callback_runner_uses_scheduler_result_write_drain_completion_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = ResultWriteDrainCompletionSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    flushed = False

    def flush_binary_correction_diagnostics_probe() -> None:
        nonlocal flushed
        flushed = True

    typing.cast("typing.Any", callback).flush_binary_correction_diagnostics = flush_binary_correction_diagnostics_probe

    drain_completion_plan = callback.plan_result_write_drain_completion(
        None,
        flush_binary_correction_diagnostics_on_stop=True,
    )
    should_stop = callback.apply_result_write_drain_completion_plan(drain_completion_plan)

    assert scheduler_state.has_result_work_item is False
    assert scheduler_state.flush_binary_correction_diagnostics_on_stop is True
    assert should_stop is True
    assert flushed is True


def test_native_callback_runner_uses_scheduler_result_write_item_dispatch_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = ResultWriteItemDispatchSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    result_work_item = callback_shared.Regenie2ResultWriteWorkItem(
        metadata=build_native_metadata(),
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=None,
        host_dosage_buffer=None,
        release_in_flight_slot=False,
        binary_chunk_diagnostics=None,
    )

    dispatch_plan = callback.plan_result_write_item_dispatch(
        result_work_item,
        expected_result_work_item_kind=callback_runtime.ResultWriteItemKind.SINGLE_RESULT,
    )

    assert scheduler_state.result_work_item_kind == "single_result"
    assert scheduler_state.expected_result_work_item_kind == "single_result"
    assert dispatch_plan.should_process_result_write_item is True
    callback.apply_result_write_item_dispatch_plan(dispatch_plan)

    multi_result_work_item = callback_shared.Regenie2MultiResultWriteWorkItem(
        metadata=build_native_metadata(),
        chunk_stats=typing.cast("typing.Any", SimpleNamespace()),
        beta=jnp.asarray([[0.1, 0.2]], dtype=jnp.float32),
        standard_error=jnp.asarray([[0.3, 0.4]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[3.0, 4.0]], dtype=jnp.float32),
        extra_code=None,
        host_dosage_buffer=None,
        release_in_flight_slot=False,
        binary_chunk_diagnostics=None,
    )

    dispatch_plan = callback.plan_result_write_item_dispatch(
        multi_result_work_item,
        expected_result_work_item_kind=callback_runtime.ResultWriteItemKind.MULTI_RESULT,
    )

    assert scheduler_state.result_work_item_kind == "multi_result"
    assert scheduler_state.expected_result_work_item_kind == "multi_result"
    assert dispatch_plan.should_process_multi_result_write_item is True

    error_plan = ResultWriteItemDispatchPlanProbe(
        result_work_item_kind="stop_signal",
        expected_result_work_item_kind="single_result",
        should_process_result_write_item=False,
        should_process_multi_result_write_item=False,
        has_dispatch_error=True,
        error_message="planned dispatch failure",
    )
    with pytest.raises(RuntimeError, match="planned dispatch failure"):
        callback.apply_result_write_item_dispatch_plan(typing.cast("typing.Any", error_plan))


def test_native_callback_runner_uses_scheduler_dosage_work_drain_completion_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = DosageWorkDrainCompletionSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    drain_completion_plan = callback.plan_dosage_work_drain_completion(None)
    should_stop = callback.apply_dosage_work_drain_completion_plan(drain_completion_plan)

    assert scheduler_state.has_dosage_work_item is False
    assert should_stop is True


def test_native_callback_runner_uses_scheduler_dosage_work_item_dispatch_plan() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = DosageWorkItemDispatchSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state
    metadata = build_native_metadata()
    chunk_stats = typing.cast("typing.Any", SimpleNamespace())

    sample_major_work_item = callback_shared.PreprocessedDosageChunkWorkItem(
        metadata=metadata,
        genotype_matrix=np.ones((2, 2), dtype=np.float32),
        chunk_stats=chunk_stats,
    )
    dispatch_plan = callback.plan_dosage_work_item_dispatch(sample_major_work_item)

    assert scheduler_state.dosage_work_item_kind == "sample_major_dosage"
    assert dispatch_plan.should_process_sample_major_dosage is True
    callback.apply_dosage_work_item_dispatch_plan(dispatch_plan)

    variant_major_work_item = callback_shared.PreprocessedVariantMajorDosageChunkWorkItem(
        metadata=metadata,
        genotype_matrix_by_variant=np.ones((2, 2), dtype=np.float32),
        chunk_stats=chunk_stats,
    )
    dispatch_plan = callback.plan_dosage_work_item_dispatch(variant_major_work_item)

    assert scheduler_state.dosage_work_item_kind == "variant_major_dosage"
    assert dispatch_plan.should_process_variant_major_dosage is True

    batch_work_item = callback_shared.PreprocessedVariantMajorDosageChunkBatchWorkItem(
        work_items=(variant_major_work_item,)
    )
    dispatch_plan = callback.plan_dosage_work_item_dispatch(batch_work_item)

    assert scheduler_state.dosage_work_item_kind == "variant_major_dosage_batch"
    assert dispatch_plan.should_process_variant_major_dosage_batch is True

    packed8_work_item = callback_shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem(
        metadata=metadata,
        packed_probability_pairs_by_variant=np.ones((2, 2, 2), dtype=np.uint8),
        chunk_stats=chunk_stats,
    )
    dispatch_plan = callback.plan_dosage_work_item_dispatch(packed8_work_item)

    assert scheduler_state.dosage_work_item_kind == "variant_major_packed8_probability_pair"
    assert dispatch_plan.should_process_variant_major_packed8_probability_pair is True

    error_plan = DosageWorkItemDispatchPlanProbe(
        dosage_work_item_kind="stop_signal",
        should_process_sample_major_dosage=False,
        should_process_variant_major_dosage=False,
        should_process_variant_major_dosage_batch=False,
        should_process_variant_major_packed8_probability_pair=False,
        has_dispatch_error=True,
        error_message="planned dosage dispatch failure",
    )
    with pytest.raises(RuntimeError, match="planned dosage dispatch failure"):
        callback.apply_dosage_work_item_dispatch_plan(typing.cast("typing.Any", error_plan))


def test_native_callback_runner_uses_scheduler_dosage_buffer_attempt_plans() -> None:
    callback = ManualCallbackRunner()
    scheduler_state = DosageBufferAttemptSchedulerProbe()
    typing.cast("typing.Any", callback).callback_scheduler_state = scheduler_state

    dosage_buffer = callback.acquire_dosage_buffer_with_shape((2, 3), np.float32)
    callback.release_dosage_buffer(dosage_buffer)
    reused_dosage_buffer = callback.acquire_dosage_buffer_with_shape((2, 3), np.float32)
    callback.discard_dosage_buffer_slot(reused_dosage_buffer)

    assert scheduler_state.acquire_free_buffer_counts == [0, 1]
    assert scheduler_state.acquire_wait_timeout_seconds == 0.1
    assert scheduler_state.acquire_backpressure_called is True
    assert scheduler_state.registered_buffer_identifier == id(dosage_buffer)
    assert scheduler_state.returned_buffer_identifier == id(dosage_buffer)
    assert scheduler_state.reuse_buffered_shape == (2, 3)
    assert scheduler_state.reuse_expected_shape == (2, 3)
    assert reused_dosage_buffer is dosage_buffer
    assert scheduler_state.discarded_buffer_identifier == id(dosage_buffer)


def test_stop_result_worker_returns_when_failed_worker_leaves_full_queue() -> None:
    stop_event = threading.Event()
    result_worker_thread = threading.Thread(target=stop_event.wait, name="failed-result-worker")
    result_worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    attach_manual_callback_scheduler_state(callback)
    assert callback.callback_scheduler_state.acquire_result_queue_slot() is True
    assert callback.result_queue.put(None, timeout_seconds=0.0) is True
    callback.result_worker_error = RuntimeError("writer failed")
    callback.result_worker_thread = result_worker_thread
    mark_callback_workers_started(callback)

    try:
        callback.stop_result_worker(timeout_seconds=None)
    finally:
        stop_event.set()
        result_worker_thread.join()

    assert callback.callback_scheduler_state.has_available_result_queue_slot() is False


def test_stop_dosage_worker_returns_when_failed_worker_leaves_full_queue() -> None:
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=stop_event.wait, name="failed-dosage-worker")
    worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    attach_manual_callback_scheduler_state(callback)
    assert callback.callback_scheduler_state.acquire_dosage_queue_slot() is True
    assert callback.dosage_queue.put(None, timeout_seconds=0.0) is True
    callback.worker_error = RuntimeError("dosage failed")
    callback.worker_thread = worker_thread
    mark_callback_workers_started(callback)

    try:
        callback.stop_dosage_worker(timeout_seconds=None)
    finally:
        stop_event.set()
        worker_thread.join()

    assert callback.callback_scheduler_state.has_available_dosage_queue_slot() is False


def test_stop_result_worker_raises_when_live_worker_leaves_full_queue() -> None:
    stop_event = threading.Event()
    result_worker_thread = threading.Thread(target=stop_event.wait, name="blocked-result-worker")
    result_worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    attach_manual_callback_scheduler_state(callback)
    assert callback.callback_scheduler_state.acquire_result_queue_slot() is True
    assert callback.result_queue.put(None, timeout_seconds=0.0) is True
    callback.result_worker_error = None
    callback.result_worker_thread = result_worker_thread
    mark_callback_workers_started(callback)

    try:
        with np.testing.assert_raises_regex(callback_shared.NativeBgenWorkerShutdownError, "blocked-result-worker"):
            callback.stop_result_worker(timeout_seconds=0.0)
    finally:
        stop_event.set()
        result_worker_thread.join()


def test_native_callback_runner_waits_on_native_result_queue_release() -> None:
    callback = ManualCallbackRunner()
    assert callback.callback_scheduler_state.acquire_result_queue_slot() is True
    assert callback.result_queue.put(None, timeout_seconds=0.0) is True

    enqueue_started = threading.Event()

    def enqueue_after_result_queue_is_full() -> bool:
        enqueue_started.set()
        return callback.try_put_result_write_item(None, timeout_seconds=2.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(enqueue_after_result_queue_is_full)
        assert enqueue_started.wait(timeout=1.0)
        time.sleep(0.2)
        assert not future.done()

        assert callback.get_result_write_item() is None
        assert future.result(timeout=2.0) is True

    assert callback.callback_scheduler_state.result_queue_occupied_count == 1
    assert callback.get_result_write_item() is None
    assert callback.callback_scheduler_state.result_queue_occupied_count == 0


def test_native_callback_runner_waits_on_native_dosage_queue_release() -> None:
    callback = ManualCallbackRunner()
    assert callback.callback_scheduler_state.acquire_dosage_queue_slot() is True
    assert callback.dosage_queue.put(None, timeout_seconds=0.0) is True

    enqueue_started = threading.Event()

    def enqueue_after_dosage_queue_is_full() -> bool:
        enqueue_started.set()
        return callback.try_put_dosage_work_item(None, timeout_seconds=2.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(enqueue_after_dosage_queue_is_full)
        assert enqueue_started.wait(timeout=1.0)
        time.sleep(0.2)
        assert not future.done()

        assert callback.get_dosage_work_item() is None
        assert future.result(timeout=2.0) is True

    assert callback.callback_scheduler_state.dosage_queue_occupied_count == 1
    assert callback.get_dosage_work_item() is None
    assert callback.callback_scheduler_state.dosage_queue_occupied_count == 0


def test_stop_dosage_worker_raises_when_live_worker_leaves_full_queue() -> None:
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=stop_event.wait, name="blocked-dosage-worker")
    worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    attach_manual_callback_scheduler_state(callback)
    assert callback.callback_scheduler_state.acquire_dosage_queue_slot() is True
    assert callback.dosage_queue.put(None, timeout_seconds=0.0) is True
    callback.worker_error = None
    callback.worker_thread = worker_thread
    mark_callback_workers_started(callback)

    try:
        with np.testing.assert_raises_regex(callback_shared.NativeBgenWorkerShutdownError, "blocked-dosage-worker"):
            callback.stop_dosage_worker(timeout_seconds=0.0)
    finally:
        stop_event.set()
        worker_thread.join()


def test_join_result_worker_raises_when_worker_does_not_stop() -> None:
    stop_event = threading.Event()
    result_worker_thread = threading.Thread(target=stop_event.wait, name="stuck-result-worker")
    result_worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.result_worker_thread = result_worker_thread
    mark_callback_workers_started(callback)

    try:
        with np.testing.assert_raises_regex(callback_shared.NativeBgenWorkerShutdownError, "stuck-result-worker"):
            callback.join_result_worker(timeout_seconds=0.0)
    finally:
        stop_event.set()
        result_worker_thread.join()


def test_join_dosage_worker_raises_when_worker_does_not_stop() -> None:
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=stop_event.wait, name="stuck-dosage-worker")
    worker_thread.start()
    callback = object.__new__(ManualCallbackRunner)
    callback.worker_thread = worker_thread
    mark_callback_workers_started(callback)

    try:
        with np.testing.assert_raises_regex(callback_shared.NativeBgenWorkerShutdownError, "stuck-dosage-worker"):
            callback.join_dosage_worker(timeout_seconds=0.0)
    finally:
        stop_event.set()
        worker_thread.join()


def test_native_bgen_callback_runner_rejects_nonpositive_staging_depth() -> None:
    class ConcreteCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
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
        ConcreteCallbackRunner(
            worker_name="invalid-staging-depth",
            staging_depth=0,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            stage_timing_recorder=None,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )


def test_native_bgen_callback_runner_accepts_explicit_capacity_limits() -> None:
    class ConcreteCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
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

    default_callback = ConcreteCallbackRunner(
        worker_name="default-capacity",
        staging_depth=3,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
        stage_timing_recorder=None,
        telemetry_session=None,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )
    explicit_callback = ConcreteCallbackRunner(
        worker_name="explicit-capacity",
        staging_depth=3,
        native_callback_batch_size=1,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
        stage_timing_recorder=None,
        telemetry_session=None,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )

    assert default_callback.result_in_flight_limit == 4
    assert default_callback.dosage_buffer_limit == 4
    assert explicit_callback.result_in_flight_limit == 7
    assert explicit_callback.dosage_buffer_limit == 8


def test_native_bgen_callback_runner_uses_native_runtime_resources() -> None:
    class ConcreteCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
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

    resolved_scheduler_state = SimpleNamespace(
        native_callback_batch_size=5,
        dosage_queue_depth=11,
        result_queue_depth=12,
        result_in_flight_limit=13,
        dosage_buffer_limit=14,
    )
    resolved_dosage_queue = SimpleNamespace()
    resolved_result_queue = SimpleNamespace()
    resolved_free_dosage_buffers = SimpleNamespace()
    resolved_runtime_resources = SimpleNamespace(
        callback_scheduler_state=resolved_scheduler_state,
        progress_state=SimpleNamespace(),
        result_in_flight_slot_signal=SimpleNamespace(),
        dosage_buffer_pool_signal=SimpleNamespace(),
        dosage_queue=resolved_dosage_queue,
        result_queue=resolved_result_queue,
        free_dosage_buffers=resolved_free_dosage_buffers,
        binary_correction_summary=SimpleNamespace(),
        worker_thread=SimpleNamespace(),
        result_worker_thread=SimpleNamespace(),
        has_started=False,
        native_callback_batch_size=5,
        dosage_queue_depth=11,
        result_queue_depth=12,
        result_in_flight_limit=13,
        dosage_buffer_limit=14,
        dosage_queue_occupied_count=0,
        result_queue_occupied_count=0,
        result_in_flight_occupied_count=0,
        dosage_buffer_allocated_count=0,
        dosage_buffer_identifiers=[],
        processed_chunk_count=0,
        current_progress_chromosome=None,
    )
    with patch(
        "g.engine.callbacks.runtime._core.NativeCallbackRuntimeResources",
        return_value=resolved_runtime_resources,
    ) as mock_runtime_resources:
        callback = ConcreteCallbackRunner(
            worker_name="native-policy",
            staging_depth=3,
            native_callback_batch_size=5,
            result_in_flight_limit=7,
            dosage_buffer_limit=8,
            stage_timing_recorder=None,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )

    mock_runtime_resources.assert_called_once()
    runtime_resource_arguments = mock_runtime_resources.call_args.kwargs
    assert runtime_resource_arguments["worker_name"] == "native-policy"
    assert callable(runtime_resource_arguments["dosage_worker_target"])
    assert callable(runtime_resource_arguments["result_worker_target"])
    assert runtime_resource_arguments["staging_depth"] == 3
    assert runtime_resource_arguments["native_callback_batch_size"] == 5
    assert runtime_resource_arguments["result_in_flight_limit"] == 7
    assert runtime_resource_arguments["dosage_buffer_limit"] == 8
    assert callback.callback_runtime_resources is resolved_runtime_resources
    assert callback.callback_scheduler_state is resolved_scheduler_state
    assert callback.dosage_queue_depth == 11
    assert callback.result_queue_depth == 12
    assert callback.result_in_flight_limit == 13
    assert callback.dosage_buffer_limit == 14
    assert callback.dosage_queue is resolved_dosage_queue
    assert callback.result_queue is resolved_result_queue
    assert callback.free_dosage_buffers is resolved_free_dosage_buffers
    assert not hasattr(callback.dosage_queue, "maxsize")
    assert not hasattr(callback.result_queue, "maxsize")
    assert not hasattr(callback.free_dosage_buffers, "maxsize")


def test_native_bgen_callback_runner_rejects_batch_size_above_dosage_buffer_limit() -> None:
    class ConcreteCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
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

    with pytest.raises(ValueError, match="effective dosage_buffer_limit"):
        ConcreteCallbackRunner(
            worker_name="oversized-batch",
            staging_depth=1,
            native_callback_batch_size=3,
            result_in_flight_limit=None,
            dosage_buffer_limit=2,
            stage_timing_recorder=None,
            telemetry_session=None,
            output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        )


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
    class ConcreteCallbackRunner(callback_runtime.NativeBgenCallbackRunner):
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
            ConcreteCallbackRunner(
                worker_name="invalid-capacity",
                staging_depth=1,
                native_callback_batch_size=1,
                result_in_flight_limit=0,
                dosage_buffer_limit=None,
                stage_timing_recorder=None,
                telemetry_session=None,
                output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
            )
        else:
            ConcreteCallbackRunner(
                worker_name="invalid-capacity",
                staging_depth=1,
                native_callback_batch_size=1,
                result_in_flight_limit=None,
                dosage_buffer_limit=0,
                stage_timing_recorder=None,
                telemetry_session=None,
                output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
            )


def test_linear_callback_passes_native_stats_to_writer_without_python_unwrap() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_result.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = build_test_linear_pipeline_callback(
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
    callback = build_test_linear_pipeline_callback(
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
    callback = build_test_linear_pipeline_callback(
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
    callback = build_test_linear_pipeline_callback(
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
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    callback = build_test_linear_pipeline_callback(
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
    callback = build_test_linear_pipeline_callback(
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
    callback = build_test_linear_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        staging_depth=1,
    )
    host_dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=2)
    host_dosage_buffer.fill(1)
    callback.acquire_result_in_flight_slot()
    callback.acquire_result_in_flight_slot()

    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 2
    assert callback.callback_scheduler_state.has_available_result_in_flight_slot() is False
    assert not hasattr(callback, "result_in_flight_slots")

    callback.put_result_write_item(
        callback_shared.Regenie2ResultWriteWorkItem(
            metadata=build_native_metadata(),
            chunk_stats=typing.cast("typing.Any", ExplodingChunkStats()),
            beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
            standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
            chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            extra_code=None,
            host_dosage_buffer=host_dosage_buffer,
            release_in_flight_slot=True,
            binary_chunk_diagnostics=None,
        )
    )
    callback.finish()

    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 1
    callback.release_result_in_flight_slot()
    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 0
    free_buffer_result = callback.free_dosage_buffers.get(timeout_seconds=0.0)
    assert free_buffer_result.has_item is True
    assert free_buffer_result.item is host_dosage_buffer
    assert len(writer_session.native_chunks) == 1


def test_native_callback_runner_uses_native_result_in_flight_slot_accounting() -> None:
    callback = ManualCallbackRunner()

    callback.acquire_result_in_flight_slot()
    callback.acquire_result_in_flight_slot()

    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 2
    assert callback.callback_scheduler_state.has_available_result_in_flight_slot() is False
    assert callback.result_in_flight_slot_count == 2

    callback.release_result_in_flight_slot()

    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 1
    assert callback.callback_scheduler_state.has_available_result_in_flight_slot() is True

    callback.release_result_in_flight_slot()

    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 0


def test_native_callback_runner_waits_on_native_result_in_flight_slot_release() -> None:
    callback = ManualCallbackRunner()
    callback.acquire_result_in_flight_slot()
    callback.acquire_result_in_flight_slot()

    acquisition_started = threading.Event()

    def acquire_slot_after_capacity_is_full() -> None:
        acquisition_started.set()
        callback.acquire_result_in_flight_slot()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(acquire_slot_after_capacity_is_full)
        assert acquisition_started.wait(timeout=1.0)
        time.sleep(0.2)
        assert not future.done()

        callback.release_result_in_flight_slot()
        future.result(timeout=2.0)

    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 2
    callback.release_result_in_flight_slot()
    callback.release_result_in_flight_slot()
    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 0


def test_result_worker_releases_host_dosage_buffer_before_output_write() -> None:
    writer_session = FakeWriterSession()
    callback = build_test_linear_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        staging_depth=1,
    )
    host_dosage_buffer = callback.acquire_dosage_buffer(sample_count=2, variant_count=2)
    host_dosage_buffer.fill(1)
    observing_writer_session = BufferObservingWriterSession(callback, host_dosage_buffer)
    callback.writer_session = observing_writer_session
    callback.acquire_result_in_flight_slot()

    callback.process_result_write_item(
        callback_shared.Regenie2ResultWriteWorkItem(
            metadata=build_native_metadata(),
            chunk_stats=typing.cast("typing.Any", ExplodingChunkStats()),
            beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
            standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
            chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            extra_code=None,
            host_dosage_buffer=host_dosage_buffer,
            release_in_flight_slot=True,
            binary_chunk_diagnostics=None,
        )
    )

    assert observing_writer_session.observed_buffer_before_write is True
    free_buffer_result = callback.free_dosage_buffers.get(timeout_seconds=0.0)
    assert free_buffer_result.has_item is True
    assert free_buffer_result.item is host_dosage_buffer
    assert callback.callback_scheduler_state.result_in_flight_occupied_count == 0


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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=0.05,
            firth_se=False,
        ),
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=SCORE_ONLY_PLAN,
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
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=0.05,
            firth_se=False,
        ),
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_multi_linear_pipeline_callback(
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
    callback = build_test_multi_linear_pipeline_callback(
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
    callback = build_test_multi_linear_pipeline_callback(
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_binary_pipeline_callback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=FakeWriterSession(),
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_multi_binary_pipeline_callback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=(FakeWriterSession(), FakeWriterSession()),
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_multi_binary_pipeline_callback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_multi_binary_pipeline_callback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=SCORE_ONLY_PLAN,
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
    callback = build_test_multi_binary_pipeline_callback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=SCORE_ONLY_PLAN,
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
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
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
    callback = build_test_multi_binary_pipeline_callback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=0.05,
            firth_se=False,
        ),
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
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    kernel_config = dataclasses.replace(
        build_default_binary_kernel_config(),
        firth_candidate=dataclasses.replace(
            build_default_binary_kernel_config().firth_candidate,
            batch_size=1,
        ),
    )
    result = build_multi_binary_chunk_result()
    callback = build_test_multi_binary_pipeline_callback(
        run_input=build_native_multi_run_input(),
        prediction_source=FakePredictionSource(),
        writer_sessions=writer_sessions,
        committed_chunk_identifier_sets=(set(), set()),
        correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=0.05,
            firth_se=False,
        ),
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: preparation_order.append("writer") or writer_session,
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            return_value={"header": "current"},
        ) as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            side_effect=lambda *args, **kwargs: (
                preparation_order.append("manifest") or build_fake_pipeline_output_preparation_batch((64, 0))
            ),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.preflight.run_regenie2_preflight",
            side_effect=record_preflight,
        ) as mock_preflight,
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2LinearState", "state"),
        ),
    ):
        final_path = run_test_regenie2_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=build_test_output_writer_settings(
                finalize_parquet=True,
                writer_thread_count=2,
                writer_queue_depth=3,
                chunks_per_arrow_file=pipeline_options.writer_settings.chunks_per_arrow_file,
                arrow_compression=pipeline_options.writer_settings.arrow_compression,
                parquet_compression=pipeline_options.writer_settings.parquet_compression,
                output_format=types.OutputFormat.PARQUET,
                output_statistic_dtype=pipeline_options.writer_settings.output_statistic_dtype,
            ),
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
    assert isinstance(callback, callback_linear.LinearRegenie2PipelineCallback)
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.preflight.run_regenie2_preflight",
            side_effect=ValueError("invalid preflight"),
        ) as mock_preflight,
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch"
        ) as mock_build_preparation_batch,
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session") as mock_create_writer_session,
        pytest.raises(ValueError, match="invalid preflight"),
    ):
        run_test_regenie2_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
        )

    mock_preflight.assert_called_once()
    mock_manifest_header.assert_not_called()
    mock_build_preparation_batch.assert_not_called()
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
        pipeline_outputs.initialize_pipeline_output_runs(
            output_run_paths_by_trait=(first_output_run_paths, second_output_run_paths),
            existing_manifests_by_trait=(
                {**first_header, "committed_chunks": []},
                {**second_manifest_header, "committed_chunks": []},
            ),
            current_headers_by_trait=(first_header, second_current_header),
            resume=True,
            resume_mode=types.ResumeMode.FAST,
            runtime_compatibility_token=build_test_runtime_compatibility_token(),
        )

    assert output.get_run_manifest_path(first_output_run_paths).read_bytes() == first_manifest_bytes
    assert output.get_run_manifest_path(second_output_run_paths).read_bytes() == second_manifest_bytes


def test_pipeline_output_initialization_returns_native_handle(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(tmp_path / "one.run", tmp_path / "one.run/chunks")
    output_run_paths.chunks_directory.mkdir(parents=True)
    current_header = {"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION, "phenotype_name": "one", "chunk_size": 32}
    native_preparation_batch = pipeline_outputs.build_pipeline_output_preparation_batch(
        output_run_paths_by_trait=(output_run_paths,),
        existing_manifests_by_trait=(None,),
        current_headers_by_trait=(current_header,),
        resume=False,
        resume_mode=types.ResumeMode.FAST,
    )

    initialized_outputs = pipeline_outputs.initialize_pipeline_output_runs(
        output_run_paths_by_trait=(output_run_paths,),
        existing_manifests_by_trait=(None,),
        current_headers_by_trait=(current_header,),
        resume=False,
        resume_mode=types.ResumeMode.FAST,
        runtime_compatibility_token=build_test_runtime_compatibility_token(),
    )

    assert isinstance(native_preparation_batch, _core.NativePipelineOutputPreparationBatch)
    assert native_preparation_batch.output_count == 1
    assert native_preparation_batch.resume is False
    assert isinstance(initialized_outputs, pipeline_outputs.InitializedPipelineOutputRuns)
    assert isinstance(initialized_outputs.native_initialization, _core.NativePipelineOutputInitialization)
    assert initialized_outputs.native_initialization.output_count == 1
    assert initialized_outputs.committed_chunk_identifier_sets == (set(),)
    assert initialized_outputs.committed_chunk_identifiers(0) == set()


def test_linear_pipeline_invokes_packed8_engine_and_forces_trusted_validation() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session", return_value=writer_session),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((64, 0)),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2LinearState", "state"),
        ),
    ):
        mock_manifest_header.return_value = {"header": "current"}
        final_path = run_test_regenie2_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_linear.LinearRegenie2PipelineCallback)
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
        callback_batch_size: int = 1,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.callback_batch_size = callback_batch_size
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        raise shutdown.GracefulShutdownRequested(shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130))


class HardInterruptRunEngine(FakeRunEngine):
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.callback_batch_size = callback_batch_size
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        raise KeyboardInterrupt


def test_native_dispatch_graceful_shutdown_drains_and_marks_writer_interrupted() -> None:
    engine = GracefulShutdownRunEngine("study.bgen", chunk_size=32)
    callback = FinishTrackingCallback()
    writer_session = FakeWriterSession()

    with pytest.raises(shutdown.GracefulShutdownRequested):
        run_test_bgen_engine_with_callback(
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
        run_test_bgen_engine_with_callback(
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
    stage_timing_recorder = timing.StageTimingRecorder(exact_stage_timings=False)
    snapshot_calls: list[tuple[timing.StageTimingRecorder | None, Path | None]] = []

    def record_snapshot(
        recorder: timing.StageTimingRecorder | None,
        stage_timing_path: Path | None,
    ) -> None:
        snapshot_calls.append((recorder, stage_timing_path))

    final_path = run_test_bgen_engine_with_callback(
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
        run_test_bgen_engine_with_multi_callback(
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: preparation_order.append("writer") or writer_session,
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            return_value={"header": "current"},
        ) as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            side_effect=lambda *args, **kwargs: (
                preparation_order.append("manifest") or build_fake_pipeline_output_preparation_batch((64, 0))
            ),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.preflight.run_regenie2_preflight",
            side_effect=record_preflight,
        ) as mock_preflight,
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        final_path = run_test_regenie2_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_binary.BinaryRegenie2PipelineCallback)
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session", return_value=writer_session),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            return_value={"header": "current"},
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((64, 0)),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        final_path = run_test_regenie2_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session", return_value=writer_session),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((64, 0)),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        mock_manifest_header.return_value = {"header": "current"}
        final_path = run_test_regenie2_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_binary.BinaryRegenie2PipelineCallback)
    assert committed_chunk_identifiers == [0, 64]
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_PACKED8
    assert mock_manifest_header.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.PACKED8
    assert mock_manifest_header.call_args.kwargs["trusted_no_missing_diploid"] is True


def test_binary_gpu_auto_uses_packed8_when_trusted_validation_succeeds() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    telemetry_session = RecordingTelemetrySession()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session", return_value=writer_session),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((64, 0)),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        mock_manifest_header.return_value = {"header": "current"}
        final_path = run_test_regenie2_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest=None,
            resume=False,
            trusted_no_missing_diploid=False,
            writer_settings=pipeline_options.writer_settings,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=build_default_binary_kernel_config(),
            gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
            jax_device=types.Device.GPU,
            telemetry_session=typing.cast("typing.Any", telemetry_session),
        )

    assert final_path == Path("results/final.parquet")
    assert len(FakeRunEngine.instances) == 1
    engine = FakeRunEngine.instances[0]
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_packed8"
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_PACKED8
    assert mock_manifest_header.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.PACKED8
    assert mock_manifest_header.call_args.kwargs["trusted_no_missing_diploid"] is True
    assert telemetry_session.events[0] == (
        "gpu_genotype_format_resolved",
        {
            "requested_gpu_genotype_format": "auto",
            "resolved_gpu_genotype_format": "packed8",
            "resolution_reason": "trusted_validation_passed",
        },
    )


def test_binary_gpu_auto_falls_back_to_dosage_when_trusted_validation_fails() -> None:
    IncompatibleTrustedRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    telemetry_session = RecordingTelemetrySession()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", IncompatibleTrustedRunEngine),
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.single_trait.native_dispatch_loaders.load_native_bgen_run_input",
            return_value=run_input,
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session", return_value=writer_session),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((64, 0)),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2BinaryState", "state"),
        ),
    ):
        mock_manifest_header.return_value = {"header": "current"}
        final_path = run_test_regenie2_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest=None,
            resume=False,
            trusted_no_missing_diploid=False,
            writer_settings=pipeline_options.writer_settings,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=build_default_binary_kernel_config(),
            gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
            jax_device=types.Device.GPU,
            telemetry_session=typing.cast("typing.Any", telemetry_session),
        )

    assert final_path == Path("results/final.parquet")
    assert len(IncompatibleTrustedRunEngine.instances) == 2
    failed_engine, fallback_engine = IncompatibleTrustedRunEngine.instances
    assert failed_engine.trusted_no_missing_diploid is True
    assert failed_engine.validation_count == 1
    assert fallback_engine.trusted_no_missing_diploid is False
    assert fallback_engine.validation_count == 0
    assert fallback_engine.run_method == "variant_major_buffered"
    assert mock_manifest_header.call_args.kwargs["association_backend_kind"] == types.AssociationBackendKind.JAX_DOSAGE
    assert mock_manifest_header.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.DOSAGE
    assert mock_manifest_header.call_args.kwargs["trusted_no_missing_diploid"] is False
    assert telemetry_session.events[0] == (
        "gpu_genotype_format_resolved",
        {
            "requested_gpu_genotype_format": "auto",
            "resolved_gpu_genotype_format": "dosage",
            "resolution_reason": "trusted_validation_failed",
            "fallback_error": "packed8 incompatible",
        },
    )


def test_binary_explicit_packed8_still_fails_when_trusted_validation_fails() -> None:
    IncompatibleTrustedRunEngine.instances.clear()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", IncompatibleTrustedRunEngine),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch"
        ) as mock_build_preparation_batch,
        pytest.raises(ValueError, match="packed8 incompatible"),
    ):
        run_test_regenie2_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            prediction_list_path=Path("pred.list"),
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            chunk_size=32,
            variant_limit=100,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            staging_depth=3,
            existing_manifest=None,
            resume=False,
            trusted_no_missing_diploid=False,
            writer_settings=pipeline_options.writer_settings,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            kernel_config=build_default_binary_kernel_config(),
            gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
            jax_device=types.Device.GPU,
        )

    assert len(IncompatibleTrustedRunEngine.instances) == 1
    assert IncompatibleTrustedRunEngine.instances[0].validation_count == 1
    mock_build_preparation_batch.assert_not_called()


@pytest.mark.parametrize(
    ("manifest_gpu_genotype_format", "expected_gpu_genotype_format"),
    [
        ("dosage", types.GpuGenotypeFormat.DOSAGE),
        ("packed8", types.GpuGenotypeFormat.PACKED8),
    ],
)
def test_binary_auto_resume_uses_existing_manifest_genotype_format(
    manifest_gpu_genotype_format: str,
    expected_gpu_genotype_format: types.GpuGenotypeFormat,
) -> None:
    telemetry_session = RecordingTelemetrySession()

    resolution = pipeline_gpu_format.resolve_single_trait_binary_gpu_genotype_format(
        requested_gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        existing_manifest={"gpu_genotype_format": manifest_gpu_genotype_format, "committed_chunks": []},
        resume=True,
        jax_device=types.Device.GPU,
        genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
        chunk_size=32,
        variant_limit=100,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        stage_timing_recorder=None,
        telemetry_session=typing.cast("typing.Any", telemetry_session),
    )

    assert resolution.resolved_gpu_genotype_format == expected_gpu_genotype_format
    assert resolution.prepared_engine is None
    assert telemetry_session.events == [
        (
            "gpu_genotype_format_resolved",
            {
                "requested_gpu_genotype_format": "auto",
                "resolved_gpu_genotype_format": manifest_gpu_genotype_format,
                "resolution_reason": "resume_manifest",
            },
        )
    ]


def test_binary_auto_resume_uses_legacy_association_backend_genotype_format() -> None:
    telemetry_session = RecordingTelemetrySession()

    resolution = pipeline_gpu_format.resolve_single_trait_binary_gpu_genotype_format(
        requested_gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        existing_manifest={
            "association_backend": {"genotype_format": "packed8"},
            "committed_chunks": [],
        },
        resume=True,
        jax_device=types.Device.GPU,
        genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
        chunk_size=32,
        variant_limit=100,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        stage_timing_recorder=None,
        telemetry_session=typing.cast("typing.Any", telemetry_session),
    )

    assert resolution.resolved_gpu_genotype_format == types.GpuGenotypeFormat.PACKED8
    assert resolution.prepared_engine is None
    assert telemetry_session.events == [
        (
            "gpu_genotype_format_resolved",
            {
                "requested_gpu_genotype_format": "auto",
                "resolved_gpu_genotype_format": "packed8",
                "resolution_reason": "resume_manifest",
            },
        )
    ]


def test_multi_linear_pipeline_opens_engine_once_and_skips_only_shared_committed_chunks() -> None:
    FakeRunEngine.instances.clear()
    writer_sessions = [FakeWriterSession(), FakeWriterSession()]
    run_input = build_native_multi_run_input()
    preparation_order: list[str] = []
    pipeline_options = build_default_pipeline_runtime_options()

    def record_preflight(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        preparation_order.append("preflight")

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.load_native_bgen_multi_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_group.run_multi_preflight",
            side_effect=record_preflight,
        ) as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: preparation_order.append("writer") or writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            side_effect=lambda *args, **kwargs: (
                preparation_order.extend(("manifest", "manifest"))
                or build_fake_pipeline_output_preparation_batch((0, 32), (32, 64))
            ),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_linear.MultiLinearRegenie2PipelineCallback)
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.load_native_bgen_multi_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_group.run_multi_preflight",
            side_effect=ValueError("invalid multi preflight"),
        ) as mock_run_multi_preflight,
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_manifest_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch"
        ) as mock_build_preparation_batch,
        patch("g.engine.regenie2_pipeline.outputs.output.create_output_writer_session") as mock_create_writer_session,
        pytest.raises(ValueError, match="invalid multi preflight"),
    ):
        run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
            sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        )

    mock_run_multi_preflight.assert_called_once()
    mock_manifest_header.assert_not_called()
    mock_build_preparation_batch.assert_not_called()
    mock_create_writer_session.assert_not_called()
    assert not output.get_run_manifest_path(output_run_paths_by_phenotype[0]).exists()
    assert not output.get_run_manifest_path(output_run_paths_by_phenotype[1]).exists()


def test_multi_linear_resume_recomputes_partial_chunks_without_duplicate_writes() -> None:
    FakeRunEngine.instances.clear()
    writer_session_for_trait_a = FakeWriterSession()
    writer_session_for_trait_b = FakeWriterSession()
    pending_writer_sessions = [writer_session_for_trait_a, writer_session_for_trait_b]
    run_input = build_native_multi_run_input()
    pipeline_options = build_default_pipeline_runtime_options()
    chromosome_state = SimpleNamespace(adjusted_residual_matrix=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32))

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", PartialCommitDeliveringRunEngine),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.load_native_bgen_multi_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight"),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: pending_writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((0, 32), (32, 64)),
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
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_linear.MultiLinearRegenie2PipelineCallback)
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
    kernel_config = build_default_binary_kernel_config()
    pipeline_options = build_default_pipeline_runtime_options()

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.load_native_bgen_multi_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((0, 32), (32, 64)),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2MultiBinaryState", "state"),
        ),
    ):
        final_paths = run_test_regenie2_multi_phenotype_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_binary.MultiBinaryRegenie2PipelineCallback)
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.load_native_bgen_multi_run_input",
            return_value=run_input,
        ) as mock_load_native_multi_run_input,
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        mock_build_header.side_effect = ({"header": "trait_a"}, {"header": "trait_b"})
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_linear.MultiLinearRegenie2PipelineCallback)
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
    expected_compute_group = native_dispatch_groups.build_resolved_complete_case_phenotype_compute_group(
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.regenie2_pipeline.grouped.native_dispatch_loaders.load_native_bgen_grouped_run_inputs",
            return_value=grouped_run_inputs,
        ) as mock_load_grouped_run_inputs,
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ) as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_linear.MultiLinearRegenie2PipelineCallback)
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


def test_grouped_union_delivery_uses_native_callback_batch_size_policy() -> None:
    with (
        patch(
            "g.engine.regenie2_pipeline.grouped._core.resolve_grouped_union_callback_batch_size",
            side_effect=ValueError("native grouped union policy"),
        ) as mock_resolver,
        pytest.raises(ValueError, match="native grouped union policy"),
    ):
        pipeline_grouped.run_prepared_grouped_per_phenotype_union_bgen_pipeline(
            context=typing.cast("typing.Any", object()),
            engine=typing.cast("typing.Any", object()),
            grouped_run_inputs=(),
            phenotype_names=(),
            output_run_paths_by_phenotype=(),
            staging_depth=1,
            native_callback_batch_size=2,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            existing_manifests=(),
            resume=False,
            resume_mode=types.ResumeMode.FAST,
            null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        )

    mock_resolver.assert_called_once_with(native_callback_batch_size=2)


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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.grouped.native_dispatch_loaders.load_native_bgen_grouped_run_inputs",
            return_value=grouped_run_inputs,
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        mock_build_header.side_effect = ({"header": "trait_a"}, {"header": "trait_b"})
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_linear.MultiLinearRegenie2PipelineCallback)
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.regenie2_pipeline.grouped.native_dispatch_loaders.load_native_bgen_grouped_run_inputs",
            return_value=grouped_run_inputs,
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight"),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.grouped.native_dispatch_loaders.load_native_bgen_grouped_run_inputs",
            return_value=grouped_run_inputs,
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ) as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_grouped.GroupedMultiPhenotypeFanoutCallback)
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.native_dispatch.loaders._core.MultiRegeniePredictionSource", FakePredictionSource),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.grouped.native_dispatch_loaders.load_native_bgen_grouped_run_inputs",
            return_value=grouped_run_inputs,
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight"),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header",
            side_effect=({"header": "trait_a"}, {"header": "trait_b"}),
        ),
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_linear.api.prepare_regenie2_multi_linear_state",
            return_value=typing.cast("regenie2_linear_state.Regenie2MultiLinearState", "state"),
        ),
    ):
        final_paths = run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        patch(
            "g.engine.native_dispatch.engine.trusted_validation.validate_trusted_bgen_with_cache",
            side_effect=lambda *, engine, bgen_path, validation_mode: engine.validate_trusted_no_missing_diploid(),
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.load_native_bgen_multi_run_input",
            return_value=run_input,
        ),
        patch(
            "g.engine.regenie2_pipeline.multi_trait.native_dispatch_loaders.build_multi_regenie_prediction_source",
            return_value=FakePredictionSource(),
        ),
        patch("g.engine.regenie2_pipeline.multi_group.run_multi_preflight") as mock_run_multi_preflight,
        patch(
            "g.engine.regenie2_pipeline.outputs.output.create_output_writer_session",
            side_effect=lambda *args, **kwargs: writer_sessions.pop(0),
        ),
        patch("g.engine.regenie2_pipeline.outputs.output.build_current_run_manifest_header") as mock_build_header,
        patch(
            "g.engine.regenie2_pipeline.outputs.build_pipeline_output_preparation_batch",
            return_value=build_fake_pipeline_output_preparation_batch((), ()),
        ),
        patch(
            "g.compute.regenie2_binary.api.prepare_regenie2_multi_binary_state",
            return_value=typing.cast("regenie2_binary_state.Regenie2MultiBinaryState", "state"),
        ),
    ):
        mock_build_header.side_effect = ({"header": "trait_a"}, {"header": "trait_b"})
        final_paths = run_test_regenie2_multi_phenotype_binary_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
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
    assert isinstance(callback, callback_binary.MultiBinaryRegenie2PipelineCallback)
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
        run_test_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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
            writer_settings=pipeline_options.writer_settings,
            bgen_decode_tile_variant_count=pipeline_options.bgen_decode_tile_variant_count,
            score_dtype=pipeline_options.score_dtype,
            firth_dtype=pipeline_options.firth_dtype,
        )


def test_build_bgen_run_engine_rejects_assumed_trusted_validation() -> None:
    FakeRunEngine.instances.clear()

    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine),
        pytest.raises(ValueError, match="assume_validated"),
    ):
        build_test_bgen_run_engine(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.bgen")),
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

    with patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine):
        first_engine = build_test_bgen_run_engine(
            genotype_source_config=build_test_genotype_source_config(source_path=bgen_path),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
        )
        second_engine = build_test_bgen_run_engine(
            genotype_source_config=build_test_genotype_source_config(source_path=bgen_path),
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

    with patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine", FakeRunEngine):
        engine = build_test_bgen_run_engine(
            genotype_source_config=build_test_genotype_source_config(source_path=bgen_path),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
            trusted_bgen_validation_mode=types.TrustedBgenValidationMode.FORCE_VALIDATE,
        )

    fake_engine = typing.cast("FakeRunEngine", engine)
    assert fake_engine.validation_count == 1


def test_bgen_source_config_rejects_non_bgen_suffix_before_engine_open() -> None:
    with (
        patch("g.engine.native_dispatch.engine._core.Regenie2RunEngine") as mock_run_engine,
        pytest.raises(ValueError, match=r"Expected a \.bgen source path"),
    ):
        build_test_bgen_run_engine(
            genotype_source_config=build_test_genotype_source_config(source_path=Path("study.vcf")),
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
    genotype_source_config = build_test_genotype_source_config(source_path=tmp_path / "study.bgen")

    with (
        patch(
            "g.engine.native_dispatch.loaders.load_native_aligned_sample_data",
            return_value=native_aligned_sample_data,
        ) as mock_load_aligned_sample_data,
    ):
        run_input = load_test_native_bgen_run_input(
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
    genotype_source_config = build_test_genotype_source_config(source_path=Path("study.bgen"), sample_path=sample_path)

    with (
        patch(
            "g.engine.native_dispatch.loaders.load_native_aligned_sample_data",
            return_value=native_aligned_sample_data,
        ) as mock_load_aligned_sample_data,
    ):
        run_input = load_test_native_bgen_run_input(
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
        sample_path=genotype_source_config.sample_path,
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
    genotype_source_config = build_test_genotype_source_config(source_path=tmp_path / "study.bgen")

    with (
        patch(
            "g.engine.native_dispatch.loaders.load_native_aligned_sample_data",
            return_value=native_aligned_sample_data,
        ) as mock_load_aligned_sample_data,
        patch("g.engine.native_dispatch.loaders._core.RegeniePredictionSource", FakePredictionSource),
    ):
        run_input = load_test_native_bgen_run_input(
            genotype_source_config=genotype_source_config,
            engine=typing.cast("typing.Any", engine),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=None,
            covariate_names=None,
            is_binary_trait=False,
            alignment_config=alignment_config,
        )
        prediction_source = native_dispatch_loaders.build_regenie_prediction_source(
            prediction_list_path=Path("pred.list"),
            phenotype_name="trait",
            run_input=run_input,
            alignment_config=alignment_config,
        )

    fake_prediction_source = typing.cast("FakePredictionSource", prediction_source)
    assert mock_load_aligned_sample_data.call_args.kwargs["alignment_config"] is alignment_config
    assert fake_prediction_source.sample_key_mode == "fid_iid"
