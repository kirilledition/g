from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from g.engine import preflight


class FakeEngine:
    def __init__(self, chromosome_values: list[str] | None = None) -> None:
        self.variant_count = len(chromosome_values or ["1"])
        self.chromosome_values = chromosome_values or ["1"]

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
            is_binary_trait=False,
            trusted_no_missing_diploid=False,
        )


def test_preflight_rejects_binary_trait_without_cases() -> None:
    with pytest.raises(ValueError, match="at least one case and one control"):
        preflight.run_regenie2_preflight(
            run_input=build_run_input(phenotype_vector=np.asarray([0.0, 0.0, 0.0], dtype=np.float32)),
            prediction_source=FakePredictionSource({"1": np.zeros(3)}),
            engine=FakeEngine(["1"]),
            is_binary_trait=True,
            trusted_no_missing_diploid=False,
        )
