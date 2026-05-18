from __future__ import annotations

import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from g import types
from g.compute import regenie2_binary_types, regenie2_linear, regenie2_linear_types
from g.engine import regenie2_pipeline
from g.io import output, source

if typing.TYPE_CHECKING:
    import pytest


class FakePredictionSource:
    instances: typing.ClassVar[list[FakePredictionSource]] = []

    def __init__(
        self,
        prediction_list_path: str | None = None,
        phenotype_name: str | None = None,
        sample_family_identifiers: list[str] | None = None,
        sample_individual_identifiers: list[str] | None = None,
    ) -> None:
        self.prediction_list_path = prediction_list_path
        self.phenotype_name = phenotype_name
        self.sample_family_identifiers = sample_family_identifiers
        self.sample_individual_identifiers = sample_individual_identifiers
        self.native_aligned_sample_data: object | None = None
        FakePredictionSource.instances.append(self)

    @staticmethod
    def from_native_aligned_sample_data(
        prediction_list_path: str,
        phenotype_name: str,
        aligned_sample_data: object,
    ) -> FakePredictionSource:
        prediction_source = FakePredictionSource(prediction_list_path, phenotype_name)
        prediction_source.native_aligned_sample_data = aligned_sample_data
        return prediction_source

    def get_chromosome_predictions(self, chromosome: str) -> np.ndarray:
        del chromosome
        return np.asarray([0.0, 0.0], dtype=np.float32)


class FakeWriterSession:
    def __init__(self) -> None:
        self.finished = False
        self.aborted = False
        self.native_chunks: list[dict[str, object]] = []

    def write_regenie2_native_chunk(self, **kwargs: object) -> None:
        self.native_chunks.append(kwargs)

    def finish(self) -> str:
        self.finished = True
        return "results/final.parquet"

    def abort(self) -> None:
        self.aborted = True


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
        self.variant_count = 10
        self.run_arguments: tuple[np.ndarray, object, list[int] | None] | None = None
        self.run_method: str | None = None
        self.validation_count = 0
        FakeRunEngine.instances.append(self)

    def validate_trusted_no_missing_diploid(self) -> None:
        self.validation_count += 1

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

    def run_bgen_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "buffered"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        return 0

    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: np.ndarray,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int:
        self.run_method = "variant_major_buffered"
        self.run_arguments = (sample_indices, callback, committed_chunk_identifiers)
        return 0


def build_native_aligned_sample_data() -> SimpleNamespace:
    return SimpleNamespace(
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        phenotype_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        covariate_matrix=np.asarray([[1.0], [1.0]], dtype=np.float32),
        is_binary_trait=False,
    )


def build_native_run_input() -> regenie2_pipeline.NativeBgenRunInput:
    return regenie2_pipeline.NativeBgenRunInput(
        native_aligned_sample_data=typing.cast("typing.Any", build_native_aligned_sample_data()),
        sample_indices=np.asarray([1, 0], dtype=np.int64),
        phenotype_vector=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        covariate_matrix=jnp.asarray([[1.0], [1.0]], dtype=jnp.float32),
        is_binary_trait=False,
    )


def build_native_metadata() -> typing.Any:
    return SimpleNamespace(
        variant_start_index=5,
        variant_stop_index=7,
        chromosome=["22", "22"],
        variant_identifiers=["variant5", "variant6"],
        position=np.asarray([100, 200], dtype=np.int64),
        allele_one=["A", "C"],
        allele_two=["G", "T"],
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
    def is_sparse_candidate(self) -> np.ndarray:
        return np.asarray([True, False], dtype=np.bool_)


def test_linear_callback_passes_native_stats_to_writer_without_python_unwrap() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_linear_types.Regenie2LinearChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True]),
    )
    callback = regenie2_pipeline.LinearRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
    )
    chunk_stats = typing.cast("typing.Any", ExplodingChunkStats())

    with (
        patch(
            "g.engine.regenie2_pipeline.regenie2_linear.prepare_regenie2_linear_chromosome_state",
            return_value="chromosome-state",
        ),
        patch(
            "g.engine.regenie2_pipeline.regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state",
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


def test_binary_callback_passes_native_sparse_mask_without_unwrapping_full_stats() -> None:
    writer_session = FakeWriterSession()
    result = regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        standard_error=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        chi_squared=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        log10_p_value=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        extra_code=jnp.asarray([0, 1], dtype=jnp.int32),
        valid_mask=jnp.asarray([True, True]),
        firth_iteration_count=jnp.asarray([0, 2], dtype=jnp.int32),
        firth_failure_code=jnp.asarray([0, 0], dtype=jnp.int32),
    )
    callback = regenie2_pipeline.BinaryRegenie2PipelineCallback(
        run_input=build_native_run_input(),
        prediction_source=FakePredictionSource(),
        writer_session=writer_session,
        correction_plan=types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
    )
    chunk_stats = typing.cast("typing.Any", SparseOnlyChunkStats())

    with (
        patch(
            "g.engine.regenie2_pipeline.regenie2_binary.prepare_regenie2_binary_chromosome_state",
            return_value="chromosome-state",
        ),
        patch(
            "g.engine.regenie2_pipeline.regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state",
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
    assert mock_compute.call_args.kwargs["correction_plan"].method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
    assert writer_session.native_chunks[0]["chunk_stats"] is chunk_stats


def test_run_linear_bgen_pipeline_invokes_native_engine_and_writer() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()

    with (
        patch("g.engine.regenie2_pipeline._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.regenie2_pipeline._core.RegeniePredictionSource", FakePredictionSource),
        patch("g.engine.regenie2_pipeline.load_native_bgen_run_input", return_value=run_input),
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session", return_value=writer_session),
        patch.object(
            regenie2_linear,
            "prepare_regenie2_linear_state",
            return_value=typing.cast("regenie2_linear_types.Regenie2LinearState", "state"),
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
            prefetch_chunks=3,
            committed_chunk_identifiers={64, 0},
            finalize_parquet=True,
            writer_thread_count=2,
            writer_queue_depth=3,
            trusted_no_missing_diploid=True,
        )

    assert final_path == Path("results/final.parquet")
    assert writer_session.finished is True
    engine = FakeRunEngine.instances[0]
    assert engine.bgen_path == "study.bgen"
    assert engine.chunk_size == 32
    assert engine.variant_limit == 100
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 1
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, regenie2_pipeline.LinearRegenie2PipelineCallback)
    assert callback.dosage_queue_depth == 3
    assert callback.dosage_buffer_limit == 4
    assert committed_chunk_identifiers == [0, 64]
    prediction_source = FakePredictionSource.instances[0]
    assert prediction_source.prediction_list_path == "pred.list"
    assert prediction_source.phenotype_name == "trait"
    assert prediction_source.native_aligned_sample_data is run_input.native_aligned_sample_data


def test_binary_pipeline_invokes_variant_major_engine_for_trusted_bgen() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()

    with (
        patch("g.engine.regenie2_pipeline._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.regenie2_pipeline._core.RegeniePredictionSource", FakePredictionSource),
        patch("g.engine.regenie2_pipeline.load_native_bgen_run_input", return_value=run_input),
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session", return_value=writer_session),
        patch(
            "g.engine.regenie2_pipeline.regenie2_binary.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_types.Regenie2BinaryState", "state"),
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
            prefetch_chunks=3,
            committed_chunk_identifiers={64, 0},
            trusted_no_missing_diploid=True,
        )

    assert final_path == Path("results/final.parquet")
    engine = FakeRunEngine.instances[0]
    assert engine.validation_count == 1
    assert engine.run_method == "variant_major_buffered"
    assert engine.run_arguments is not None
    sample_indices, callback, committed_chunk_identifiers = engine.run_arguments
    np.testing.assert_array_equal(sample_indices, np.asarray([1, 0], dtype=np.int64))
    assert isinstance(callback, regenie2_pipeline.BinaryRegenie2PipelineCallback)
    assert committed_chunk_identifiers == [0, 64]


def test_binary_pipeline_uses_sample_major_engine_for_untrusted_bgen() -> None:
    FakeRunEngine.instances.clear()
    FakePredictionSource.instances.clear()
    writer_session = FakeWriterSession()
    run_input = build_native_run_input()

    with (
        patch("g.engine.regenie2_pipeline._core.Regenie2RunEngine", FakeRunEngine),
        patch("g.engine.regenie2_pipeline._core.RegeniePredictionSource", FakePredictionSource),
        patch("g.engine.regenie2_pipeline.load_native_bgen_run_input", return_value=run_input),
        patch("g.engine.regenie2_pipeline.output.create_output_writer_session", return_value=writer_session),
        patch(
            "g.engine.regenie2_pipeline.regenie2_binary.prepare_regenie2_binary_state",
            return_value=typing.cast("regenie2_binary_types.Regenie2BinaryState", "state"),
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
            prefetch_chunks=3,
            committed_chunk_identifiers={64, 0},
            trusted_no_missing_diploid=False,
        )

    assert final_path == Path("results/final.parquet")
    engine = FakeRunEngine.instances[0]
    assert engine.validation_count == 0
    assert engine.run_method == "buffered"


def test_build_bgen_run_engine_skips_trusted_validation_when_marked_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRunEngine.instances.clear()
    monkeypatch.setenv(
        regenie2_pipeline.ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE,
        "1",
    )

    with patch("g.engine.regenie2_pipeline._core.Regenie2RunEngine", FakeRunEngine):
        engine = regenie2_pipeline.build_bgen_run_engine(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            chunk_size=32,
            variant_limit=100,
            trusted_no_missing_diploid=True,
        )

    assert isinstance(engine, FakeRunEngine)
    assert engine.trusted_no_missing_diploid is True
    assert engine.validation_count == 0


def test_load_native_bgen_run_input_rejects_non_bgen_source_suffix() -> None:
    with np.testing.assert_raises_regex(ValueError, r"Expected a \.bgen source path"):
        regenie2_pipeline.load_native_bgen_run_input(
            genotype_source_config=source.GenotypeSourceConfig(source_path=Path("study.vcf")),
            engine=typing.cast("typing.Any", object()),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=None,
            covariate_names=None,
            is_binary_trait=False,
        )


def test_load_native_bgen_run_input_uses_rust_alignment_for_embedded_samples() -> None:
    native_aligned_sample_data = build_native_aligned_sample_data()
    engine = SimpleNamespace(
        sample_count=2,
        contains_embedded_samples=True,
        sample_identifiers=lambda: ["sample1", "sample2"],
    )

    with (
        patch("g.engine.regenie2_pipeline.source.resolve_bgen_sample_path", return_value=None),
        patch(
            "g.engine.regenie2_pipeline.load_native_aligned_sample_data_from_individual_identifier_table",
            return_value=native_aligned_sample_data,
        ) as mock_load_from_sample_table,
    ):
        run_input = regenie2_pipeline.load_native_bgen_run_input(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            engine=typing.cast("typing.Any", engine),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            is_binary_trait=True,
        )

    assert run_input.native_aligned_sample_data is native_aligned_sample_data
    np.testing.assert_array_equal(run_input.sample_indices, np.asarray([1, 0], dtype=np.int64))
    mock_load_from_sample_table.assert_called_once()


def test_load_native_bgen_run_input_uses_rust_sample_file_alignment() -> None:
    native_aligned_sample_data = build_native_aligned_sample_data()
    engine = SimpleNamespace(
        sample_count=2,
        contains_embedded_samples=False,
    )
    sample_path = Path("study.sample")

    with (
        patch("g.engine.regenie2_pipeline.source.resolve_bgen_sample_path", return_value=sample_path),
        patch(
            "g.engine.regenie2_pipeline.load_native_aligned_sample_data_from_sample_file",
            return_value=native_aligned_sample_data,
        ) as mock_rust_sample_file_load,
    ):
        run_input = regenie2_pipeline.load_native_bgen_run_input(
            genotype_source_config=source.build_bgen_source_config(Path("study.bgen")),
            engine=typing.cast("typing.Any", engine),
            phenotype_path=Path("phenotype.tsv"),
            phenotype_name="trait",
            covariate_path=Path("covariates.tsv"),
            covariate_names=("age",),
            is_binary_trait=True,
        )

    assert run_input.native_aligned_sample_data is native_aligned_sample_data
    mock_rust_sample_file_load.assert_called_once_with(
        sample_path=sample_path,
        expected_sample_count=2,
        phenotype_path=Path("phenotype.tsv"),
        phenotype_name="trait",
        covariate_path=Path("covariates.tsv"),
        covariate_names=("age",),
        is_binary_trait=True,
    )
