from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g import execution_plan, types
from g.engine import warm_cache
from g.interface import config as interface_config
from g.io import source

if typing.TYPE_CHECKING:
    from g import _core
    from g.compute.regenie2_binary import config as regenie2_binary_config


def build_default_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build the packaged-default kernel config for tests."""
    return execution_plan.build_binary_kernel_config(interface_config.load_packaged_config().g_compute)


@dataclasses.dataclass(frozen=True)
class FakeEngine:
    variant_count: int
    chromosomes: tuple[str, ...] = ("22",)
    boundary_indices: tuple[int, ...] | None = None

    def chromosome_boundary_indices(self) -> list[int]:
        if self.boundary_indices is not None:
            return list(self.boundary_indices)
        return [0, self.variant_count]

    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]:
        selected_variant_count = variant_stop - variant_start
        return (
            list(self.chromosomes[:selected_variant_count]),
            ["variant"] * selected_variant_count,
            [1] * selected_variant_count,
            ["A"] * selected_variant_count,
            ["G"] * selected_variant_count,
        )


def cast_fake_engine(engine: FakeEngine) -> _core.Regenie2RunEngine:
    """Cast a Python fake to the native engine protocol needed by cache planning."""
    return typing.cast("_core.Regenie2RunEngine", engine)


@dataclasses.dataclass(frozen=True)
class FakeRunInput:
    sample_indices: np.ndarray
    phenotype_vector: jax.Array
    covariate_matrix: jax.Array
    native_aligned_sample_data: object


@dataclasses.dataclass(frozen=True)
class FakePredictionSource:
    sample_count: int

    def get_chromosome_predictions(self, chromosome: str) -> np.ndarray:
        assert chromosome == "22"
        return np.zeros(self.sample_count, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class FakeReadyValue:
    shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FakeChunkResult:
    log10_p_value: FakeReadyValue


def build_fake_run_input(*, is_binary_trait: bool) -> FakeRunInput:
    sample_count = 6
    phenotype_values = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float32)
    if not is_binary_trait:
        phenotype_values = np.linspace(-1.0, 1.0, sample_count, dtype=np.float32)
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float32),
            np.linspace(-0.5, 0.5, sample_count, dtype=np.float32),
        ]
    )
    return FakeRunInput(
        sample_indices=np.arange(sample_count, dtype=np.int64),
        phenotype_vector=jnp.asarray(phenotype_values, dtype=jnp.float32),
        covariate_matrix=jnp.asarray(covariate_matrix, dtype=jnp.float32),
        native_aligned_sample_data=object(),
    )


def install_native_dispatch_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: FakeEngine,
    run_input: FakeRunInput,
) -> None:
    def fake_build_bgen_run_engine(**keyword_arguments: object) -> FakeEngine:
        assert keyword_arguments["chunk_size"] == 50
        return engine

    def fake_load_native_bgen_run_input(**keyword_arguments: object) -> FakeRunInput:
        assert keyword_arguments["engine"] is engine
        return run_input

    def fake_build_regenie_prediction_source(**keyword_arguments: object) -> FakePredictionSource:
        assert keyword_arguments["run_input"] is run_input
        return FakePredictionSource(sample_count=int(run_input.sample_indices.shape[0]))

    monkeypatch.setattr(warm_cache.native_dispatch_engine, "build_bgen_run_engine", fake_build_bgen_run_engine)
    monkeypatch.setattr(
        warm_cache.native_dispatch_loaders,
        "load_native_bgen_run_input",
        fake_load_native_bgen_run_input,
    )
    monkeypatch.setattr(
        warm_cache.native_dispatch_loaders,
        "build_regenie_prediction_source",
        fake_build_regenie_prediction_source,
    )


def test_build_warm_cache_shapes_plans_all_unique_production_chunks() -> None:
    shapes = warm_cache.build_warm_cache_shapes(
        engine=cast_fake_engine(FakeEngine(variant_count=155, boundary_indices=(0, 45, 120, 155))),
        chunk_size=50,
        variant_limit=None,
        sample_count=2504,
    )

    assert shapes == (
        warm_cache.WarmCacheShape(sample_count=2504, variant_count=45),
        warm_cache.WarmCacheShape(sample_count=2504, variant_count=5),
        warm_cache.WarmCacheShape(sample_count=2504, variant_count=50),
        warm_cache.WarmCacheShape(sample_count=2504, variant_count=20),
        warm_cache.WarmCacheShape(sample_count=2504, variant_count=30),
    )


def test_build_warm_cache_shapes_honors_variant_limit() -> None:
    shapes = warm_cache.build_warm_cache_shapes(
        engine=cast_fake_engine(FakeEngine(variant_count=105)),
        chunk_size=50,
        variant_limit=40,
        sample_count=8,
    )

    assert shapes == (warm_cache.WarmCacheShape(sample_count=8, variant_count=40),)


def test_build_synthetic_genotype_matrix_uses_trait_specific_patterns() -> None:
    phenotype_vector = jnp.asarray([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32)

    binary_matrix = warm_cache.build_synthetic_genotype_matrix(
        phenotype_vector=phenotype_vector,
        variant_count=3,
        is_binary_trait=True,
    )
    quantitative_matrix = warm_cache.build_synthetic_genotype_matrix(
        phenotype_vector=phenotype_vector,
        variant_count=2,
        is_binary_trait=False,
    )

    np.testing.assert_array_equal(np.asarray(binary_matrix[:, 0]), np.asarray([0.0, 2.0, 2.0, 0.0]))
    assert binary_matrix.shape == (4, 3)
    assert quantitative_matrix.shape == (4, 2)
    np.testing.assert_allclose(np.asarray(quantitative_matrix.mean(axis=0)), np.zeros(2), atol=1e-6)


def test_encode_variant_major_dosage_to_packed8_probability_pairs() -> None:
    genotype_matrix_by_variant = jnp.asarray(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    packed_probability_pairs = warm_cache.encode_variant_major_dosage_to_packed8_probability_pairs(
        genotype_matrix_by_variant
    )

    np.testing.assert_array_equal(
        np.asarray(packed_probability_pairs),
        np.asarray(
            [
                [[255, 0], [0, 255], [0, 0]],
                [[0, 0], [255, 0], [0, 255]],
            ],
            dtype=np.uint8,
        ),
    )


def test_first_engine_chromosome_rejects_empty_metadata() -> None:
    with pytest.raises(ValueError, match="empty BGEN dataset"):
        warm_cache.first_engine_chromosome(cast_fake_engine(FakeEngine(variant_count=0, chromosomes=())))


def test_warm_regenie2_linear_bgen_cache_executes_full_and_tail_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine(variant_count=105)
    run_input = build_fake_run_input(is_binary_trait=False)
    install_native_dispatch_fakes(monkeypatch, engine=engine, run_input=run_input)
    observed_shapes: list[tuple[int, int]] = []
    ready_values: list[FakeReadyValue] = []
    observed_score_dtypes: list[types.FloatingPointDtype] = []

    monkeypatch.setattr(warm_cache.regenie2_linear, "prepare_regenie2_linear_state", lambda **_: object())
    monkeypatch.setattr(
        warm_cache.regenie2_linear,
        "prepare_regenie2_linear_chromosome_state",
        lambda *_, **__: object(),
    )

    def fake_compute_linear_chunk(
        *,
        chromosome_state: object,
        genotype_matrix_by_variant: jax.Array,
        genotype_dosage_sum: jax.Array,
        genotype_observation_count: jax.Array,
        genotype_imputed_dosage_square_sum: jax.Array,
        score_dtype: types.FloatingPointDtype,
        linear_minimum_variance: float,
        linear_relative_variance_tolerance: float,
    ) -> FakeChunkResult:
        del chromosome_state
        observed_score_dtypes.append(score_dtype)
        assert (
            linear_minimum_variance
            == warm_cache.regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG.minimum_variance
        )
        assert (
            linear_relative_variance_tolerance
            == warm_cache.regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG.relative_variance_tolerance
        )
        observed_shapes.append(typing.cast("tuple[int, int]", genotype_matrix_by_variant.shape))
        expected_column = np.asarray([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(genotype_matrix_by_variant)[0], expected_column)
        np.testing.assert_allclose(np.asarray(genotype_dosage_sum), np.zeros(genotype_matrix_by_variant.shape[0]))
        np.testing.assert_array_equal(
            np.asarray(genotype_observation_count),
            np.full(genotype_matrix_by_variant.shape[0], 6, dtype=np.int32),
        )
        np.testing.assert_allclose(
            np.asarray(genotype_imputed_dosage_square_sum),
            np.full(genotype_matrix_by_variant.shape[0], 4.0, dtype=np.float32),
        )
        return FakeChunkResult(log10_p_value=FakeReadyValue(shape=(genotype_matrix_by_variant.shape[0],)))

    def fake_block_until_ready(value: FakeReadyValue) -> None:
        ready_values.append(value)

    monkeypatch.setattr(
        warm_cache.regenie2_linear,
        "compute_regenie2_linear_chunk_from_chromosome_state_variant_major",
        fake_compute_linear_chunk,
    )
    monkeypatch.setattr(warm_cache.callback_diagnostics, "block_until_ready", fake_block_until_ready)

    report = warm_cache.warm_regenie2_linear_bgen_cache(
        genotype_source_config=source.GenotypeSourceConfig(Path("input.bgen"), Path("input.sample")),
        phenotype_path=Path("phenotypes.tsv"),
        phenotype_name="trait",
        prediction_list_path=Path("predictions.list"),
        covariate_path=None,
        covariate_names=None,
        chunk_size=50,
        variant_limit=None,
        trusted_no_missing_diploid=False,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        alignment_config=None,
        gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        score_dtype=types.FloatingPointDtype.FLOAT64,
    )

    assert observed_shapes == [(50, 6), (5, 6)]
    assert observed_score_dtypes == [types.FloatingPointDtype.FLOAT64, types.FloatingPointDtype.FLOAT64]
    assert tuple(value.shape for value in ready_values) == ((50,), (5,))
    assert report.warmed_shapes == (
        warm_cache.WarmCacheShape(sample_count=6, variant_count=50),
        warm_cache.WarmCacheShape(sample_count=6, variant_count=5),
    )
    assert report.warmed_signatures == (
        warm_cache.WarmCacheSignature(
            shape=warm_cache.WarmCacheShape(sample_count=6, variant_count=50),
            association_mode=types.AssociationMode.REGENIE2_LINEAR,
            genotype_format=types.GpuGenotypeFormat.DOSAGE,
            genotype_path=warm_cache.WarmCacheGenotypePath.LINEAR_DOSAGE,
            trait_count=1,
            score_dtype=types.FloatingPointDtype.FLOAT64,
            correction_method=None,
            correction_p_threshold=None,
            correction_firth_se=None,
            firth_candidate_batch_size=None,
            firth_candidate_capacity=None,
        ),
        warm_cache.WarmCacheSignature(
            shape=warm_cache.WarmCacheShape(sample_count=6, variant_count=5),
            association_mode=types.AssociationMode.REGENIE2_LINEAR,
            genotype_format=types.GpuGenotypeFormat.DOSAGE,
            genotype_path=warm_cache.WarmCacheGenotypePath.LINEAR_DOSAGE,
            trait_count=1,
            score_dtype=types.FloatingPointDtype.FLOAT64,
            correction_method=None,
            correction_p_threshold=None,
            correction_firth_se=None,
            firth_candidate_batch_size=None,
            firth_candidate_capacity=None,
        ),
    )


def test_warm_regenie2_binary_bgen_cache_executes_with_resolved_kernel_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine(variant_count=55)
    run_input = build_fake_run_input(is_binary_trait=True)
    install_native_dispatch_fakes(monkeypatch, engine=engine, run_input=run_input)
    correction_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=0.01,
        firth_se=True,
    )
    default_kernel_config = build_default_binary_kernel_config()
    kernel_config = dataclasses.replace(
        default_kernel_config,
        firth_candidate=dataclasses.replace(
            default_kernel_config.firth_candidate,
            batch_size=4,
        ),
    )
    observed_shapes: list[tuple[int, int]] = []
    observed_kernel_configs: list[regenie2_binary_config.BinaryKernelConfig] = []
    observed_score_dtypes: list[types.FloatingPointDtype] = []

    monkeypatch.setattr(warm_cache.regenie2_binary, "prepare_regenie2_binary_state", lambda **_: object())
    monkeypatch.setattr(
        warm_cache.regenie2_binary,
        "prepare_regenie2_binary_chromosome_state",
        lambda **keyword_arguments: keyword_arguments,
    )

    def fake_compute_binary_chunk(
        *,
        chromosome_state: object,
        genotype_matrix_by_variant: jax.Array,
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
        sparse_candidate_mask: jax.Array | None,
        stage_duration_recorder: typing.Callable[[str, float], None] | None,
        dosage_sum: jax.Array,
        observation_count: jax.Array,
        score_dtype: types.FloatingPointDtype,
    ) -> FakeChunkResult:
        del chromosome_state, correction_plan
        assert sparse_candidate_mask is None
        assert stage_duration_recorder is None
        observed_shapes.append(typing.cast("tuple[int, int]", genotype_matrix_by_variant.shape))
        observed_kernel_configs.append(kernel_config)
        observed_score_dtypes.append(score_dtype)
        np.testing.assert_array_equal(np.asarray(genotype_matrix_by_variant)[0], np.asarray([0, 2, 0, 2, 0, 2]))
        np.testing.assert_array_equal(np.asarray(dosage_sum), np.full(genotype_matrix_by_variant.shape[0], 6.0))
        np.testing.assert_array_equal(
            np.asarray(observation_count),
            np.full(genotype_matrix_by_variant.shape[0], 6, dtype=np.int32),
        )
        return FakeChunkResult(log10_p_value=FakeReadyValue(shape=(genotype_matrix_by_variant.shape[0],)))

    monkeypatch.setattr(
        warm_cache.regenie2_binary,
        "compute_regenie2_binary_chunk_from_chromosome_state_variant_major",
        fake_compute_binary_chunk,
    )
    monkeypatch.setattr(warm_cache.callback_diagnostics, "block_until_ready", lambda _: None)

    report = warm_cache.warm_regenie2_binary_bgen_cache(
        genotype_source_config=source.GenotypeSourceConfig(Path("input.bgen"), Path("input.sample")),
        phenotype_path=Path("phenotypes.tsv"),
        phenotype_name="trait",
        prediction_list_path=Path("predictions.list"),
        covariate_path=None,
        covariate_names=None,
        chunk_size=50,
        variant_limit=None,
        correction_plan=correction_plan,
        trusted_no_missing_diploid=False,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        alignment_config=None,
        kernel_config=kernel_config,
        gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
        score_dtype=types.FloatingPointDtype.FLOAT64,
    )

    assert observed_shapes == [(50, 6), (5, 6)]
    assert observed_kernel_configs == [kernel_config, kernel_config]
    assert observed_score_dtypes == [types.FloatingPointDtype.FLOAT64, types.FloatingPointDtype.FLOAT64]
    assert report.warmed_shapes == (
        warm_cache.WarmCacheShape(sample_count=6, variant_count=50),
        warm_cache.WarmCacheShape(sample_count=6, variant_count=5),
    )
    assert report.warmed_signatures == (
        warm_cache.WarmCacheSignature(
            shape=warm_cache.WarmCacheShape(sample_count=6, variant_count=50),
            association_mode=types.AssociationMode.REGENIE2_BINARY,
            genotype_format=types.GpuGenotypeFormat.DOSAGE,
            genotype_path=warm_cache.WarmCacheGenotypePath.BINARY_DOSAGE_CORRECTION,
            trait_count=1,
            score_dtype=types.FloatingPointDtype.FLOAT64,
            correction_method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            correction_p_threshold=0.01,
            correction_firth_se=True,
            firth_candidate_batch_size=4,
            firth_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
        ),
        warm_cache.WarmCacheSignature(
            shape=warm_cache.WarmCacheShape(sample_count=6, variant_count=5),
            association_mode=types.AssociationMode.REGENIE2_BINARY,
            genotype_format=types.GpuGenotypeFormat.DOSAGE,
            genotype_path=warm_cache.WarmCacheGenotypePath.BINARY_DOSAGE_CORRECTION,
            trait_count=1,
            score_dtype=types.FloatingPointDtype.FLOAT64,
            correction_method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            correction_p_threshold=0.01,
            correction_firth_se=True,
            firth_candidate_batch_size=4,
            firth_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
        ),
    )


def test_warm_regenie2_binary_packed8_cache_executes_donating_score_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine(variant_count=50)
    run_input = build_fake_run_input(is_binary_trait=True)
    install_native_dispatch_fakes(monkeypatch, engine=engine, run_input=run_input)
    correction_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.SCORE_ONLY,
        p_threshold=0.05,
        firth_se=False,
    )
    kernel_config = build_default_binary_kernel_config()
    observed_packed_shapes: list[tuple[int, int, int]] = []

    monkeypatch.setattr(warm_cache.regenie2_binary, "prepare_regenie2_binary_state", lambda **_: object())
    monkeypatch.setattr(
        warm_cache.regenie2_binary,
        "prepare_regenie2_binary_chromosome_state",
        lambda **keyword_arguments: keyword_arguments,
    )

    def fake_packed_score_chunk(
        *,
        chromosome_state: object,
        packed_probability_pairs_by_variant: jax.Array,
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
        dosage_sum: jax.Array,
        observation_count: jax.Array,
        score_dtype: types.FloatingPointDtype,
    ) -> FakeChunkResult:
        del chromosome_state, correction_plan, kernel_config, score_dtype
        observed_packed_shapes.append(typing.cast("tuple[int, int, int]", packed_probability_pairs_by_variant.shape))
        np.testing.assert_array_equal(
            np.asarray(packed_probability_pairs_by_variant)[0],
            np.asarray(
                [[255, 0], [0, 255], [0, 0], [255, 0], [0, 255], [0, 0]],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(np.asarray(dosage_sum), np.full(50, 6.0))
        np.testing.assert_array_equal(np.asarray(observation_count), np.full(50, 6, dtype=np.int32))
        return FakeChunkResult(log10_p_value=FakeReadyValue(shape=(packed_probability_pairs_by_variant.shape[0],)))

    def forbidden_variant_major_chunk(**_: object) -> FakeChunkResult:
        message = "packed8 score warming should use the packed8 donating entrypoint."
        raise AssertionError(message)

    monkeypatch.setattr(
        warm_cache.regenie2_binary,
        "compute_binary_score_test_packed8_donating_inputs",
        fake_packed_score_chunk,
    )
    monkeypatch.setattr(
        warm_cache.regenie2_binary,
        "compute_regenie2_binary_chunk_from_chromosome_state_variant_major",
        forbidden_variant_major_chunk,
    )
    monkeypatch.setattr(warm_cache.callback_diagnostics, "block_until_ready", lambda _: None)

    report = warm_cache.warm_regenie2_binary_bgen_cache(
        genotype_source_config=source.GenotypeSourceConfig(Path("input.bgen"), Path("input.sample")),
        phenotype_path=Path("phenotypes.tsv"),
        phenotype_name="trait",
        prediction_list_path=Path("predictions.list"),
        covariate_path=None,
        covariate_names=None,
        chunk_size=50,
        variant_limit=None,
        correction_plan=correction_plan,
        trusted_no_missing_diploid=False,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        alignment_config=None,
        kernel_config=kernel_config,
        gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        score_dtype=types.FloatingPointDtype.FLOAT32,
    )

    assert observed_packed_shapes == [(50, 6, 2)]
    assert report.warmed_shapes == (warm_cache.WarmCacheShape(sample_count=6, variant_count=50),)
    assert report.warmed_signatures == (
        warm_cache.WarmCacheSignature(
            shape=warm_cache.WarmCacheShape(sample_count=6, variant_count=50),
            association_mode=types.AssociationMode.REGENIE2_BINARY,
            genotype_format=types.GpuGenotypeFormat.PACKED8,
            genotype_path=warm_cache.WarmCacheGenotypePath.BINARY_PACKED8_SCORE,
            trait_count=1,
            score_dtype=types.FloatingPointDtype.FLOAT32,
            correction_method=types.BinaryFallbackMethod.SCORE_ONLY,
            correction_p_threshold=0.05,
            correction_firth_se=False,
            firth_candidate_batch_size=None,
            firth_candidate_capacity=None,
        ),
    )
