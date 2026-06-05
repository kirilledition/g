from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from g.engine import preflight


class FakeEngine:
    def __init__(self, chromosome_values: list[str] | None = None) -> None:
        resolved_chromosome_values = ["1"] if chromosome_values is None else chromosome_values
        self.variant_count = len(resolved_chromosome_values)
        self.chromosome_values = resolved_chromosome_values

    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]:
        selected_chromosomes = self.chromosome_values[variant_start:variant_stop]
        return (
            selected_chromosomes,
            [f"variant{variant_index}" for variant_index in range(variant_start, variant_stop)],
            list(range(variant_start, variant_stop)),
            ["A"] * len(selected_chromosomes),
            ["G"] * len(selected_chromosomes),
        )


class FakePredictionSource:
    def __init__(self, predictions_by_chromosome: dict[str, np.ndarray]) -> None:
        self.predictions_by_chromosome = predictions_by_chromosome

    def get_chromosome_predictions(self, chromosome: str) -> np.ndarray:
        return self.predictions_by_chromosome[chromosome]


def build_run_input(
    *,
    phenotype_vector: np.ndarray | None = None,
    covariate_matrix: np.ndarray | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        phenotype_vector=jnp.asarray(
            phenotype_vector if phenotype_vector is not None else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        ),
        covariate_matrix=jnp.asarray(
            covariate_matrix
            if covariate_matrix is not None
            else np.asarray(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                ],
                dtype=np.float32,
            )
        ),
    )


def test_preflight_accepts_valid_binary_inputs() -> None:
    report = preflight.run_regenie2_preflight(
        run_input=build_run_input(),
        prediction_source=FakePredictionSource({"1": np.asarray([0.1, 0.2, 0.3], dtype=np.float32)}),
        engine=FakeEngine(["1", "1"]),
        variant_limit=None,
        is_binary_trait=True,
        trusted_no_missing_diploid=False,
    )

    assert report.sample_count == 3
    assert report.covariate_count == 2
    assert report.chromosome_count == 1


def test_preflight_rejects_non_finite_predictions() -> None:
    with pytest.raises(ValueError, match="Prediction values for chromosome 2 contains non-finite values"):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(),
            prediction_source=FakePredictionSource({"1": np.zeros(3), "2": np.asarray([0.0, np.nan, 0.0])}),
            engine=FakeEngine(["1", "2"]),
            variant_limit=None,
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


def test_preflight_rejects_rank_deficient_covariates() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(
                covariate_matrix=np.asarray(
                    [
                        [1.0, 2.0],
                        [1.0, 2.0],
                        [1.0, 2.0],
                    ],
                    dtype=np.float32,
                )
            ),
            prediction_source=FakePredictionSource({"1": np.zeros(3)}),
            engine=FakeEngine(["1"]),
            variant_limit=None,
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


def test_preflight_rejects_binary_trait_without_cases() -> None:
    with pytest.raises(ValueError, match="at least one case and one control"):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(phenotype_vector=np.asarray([0.0, 0.0, 0.0], dtype=np.float32)),
            prediction_source=FakePredictionSource({"1": np.zeros(3)}),
            engine=FakeEngine(["1"]),
            variant_limit=None,
            is_binary_trait=True,
            trusted_no_missing_diploid=False,
        )


def test_preflight_variant_limit_ignores_later_chromosomes() -> None:
    report = preflight.run_regenie2_preflight(
        run_input=build_run_input(),
        prediction_source=FakePredictionSource({"1": np.zeros(3)}),
        engine=FakeEngine(["1", "1", "2"]),
        variant_limit=2,
        is_binary_trait=False,
        trusted_no_missing_diploid=False,
    )

    assert report.chromosome_count == 1


def test_preflight_without_variant_limit_requires_all_chromosomes() -> None:
    with pytest.raises(KeyError):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(),
            prediction_source=FakePredictionSource({"1": np.zeros(3)}),
            engine=FakeEngine(["1", "1", "2"]),
            variant_limit=None,
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


def test_preflight_rejects_prediction_sample_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Prediction sample count for chromosome 1 is 2, expected 3"):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(),
            prediction_source=FakePredictionSource({"1": np.zeros(2, dtype=np.float32)}),
            engine=FakeEngine(["1"]),
            variant_limit=None,
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


@pytest.mark.parametrize(
    ("covariate_matrix", "message"),
    [
        (np.asarray([1.0, 2.0, 3.0], dtype=np.float32), "Covariate matrix must be two-dimensional"),
        (
            np.asarray([[1.0], [1.0]], dtype=np.float32),
            "Covariate matrix sample count does not match phenotype sample count",
        ),
        (
            np.eye(3, dtype=np.float32),
            "Sample count must exceed the number of covariate degrees of freedom",
        ),
    ],
)
def test_preflight_rejects_invalid_covariate_shapes(covariate_matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(covariate_matrix=covariate_matrix),
            prediction_source=FakePredictionSource({"1": np.zeros(3, dtype=np.float32)}),
            engine=FakeEngine(["1"]),
            variant_limit=None,
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


def test_preflight_rejects_non_binary_values_for_binary_trait() -> None:
    with pytest.raises(ValueError, match="Binary phenotype must be coded as 0/1"):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(phenotype_vector=np.asarray([0.0, 0.5, 1.0], dtype=np.float32)),
            prediction_source=FakePredictionSource({"1": np.zeros(3, dtype=np.float32)}),
            engine=FakeEngine(["1"]),
            variant_limit=None,
            is_binary_trait=True,
            trusted_no_missing_diploid=False,
        )


@pytest.mark.parametrize(
    ("engine", "variant_limit", "message"),
    [
        (FakeEngine([]), None, "BGEN input contains no variants"),
        (FakeEngine(["1"]), 0, "BGEN scan contains no variants"),
    ],
)
def test_preflight_rejects_empty_variant_scans(
    engine: FakeEngine,
    variant_limit: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(),
            prediction_source=FakePredictionSource({"1": np.zeros(3, dtype=np.float32)}),
            engine=engine,
            variant_limit=variant_limit,
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


def test_preflight_records_low_degrees_of_freedom_and_trusted_path_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="g.engine.preflight"):
        report = preflight.run_regenie2_preflight(
            run_input=build_run_input(),
            prediction_source=FakePredictionSource({"1": np.zeros(3, dtype=np.float32)}),
            engine=FakeEngine(["1"]),
            variant_limit=None,
            is_binary_trait=False,
            trusted_no_missing_diploid=True,
        )

    assert report.warning_messages == (
        "REGENIE step 2 is running with fewer than 10 residual degrees of freedom.",
        "Trusted no-missing diploid BGEN path is enabled after compatibility validation.",
    )
    assert len(caplog.records) == 2
