"""Correctness tests for the covariate-only Firth null solver."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import tests.numerical
from g.compute.regenie2_binary.firth import null as regenie2_binary_firth_null

PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE = 50.0e-6
PRODUCTION_NULL_FIRTH_MAXIMUM_STEP_SIZE = 25.0


@dataclass(frozen=True)
class NullFirthFixture:
    """Full-rank covariate-only Firth inputs."""

    covariate_matrix: npt.NDArray[np.float64]
    phenotype_vector: npt.NDArray[np.float64]
    loco_offset: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]


@dataclass(frozen=True)
class NullFirthComponentReference:
    """Independent NumPy null-Firth components."""

    information_cholesky_factor: npt.NDArray[np.float64]
    deviance: float
    modified_score: npt.NDArray[np.float64]


def build_null_firth_fixture() -> NullFirthFixture:
    """Build a non-separated fixture with a nonzero offset and start."""
    return NullFirthFixture(
        covariate_matrix=np.asarray(
            [
                [1.0, -1.5],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
                [1.0, 1.5],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        phenotype_vector=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        loco_offset=np.asarray([0.04, -0.03, 0.01, 0.00, -0.02, 0.03, -0.01, 0.02], dtype=np.float64),
        coefficients=np.asarray([0.15, -0.08], dtype=np.float64),
    )


def compute_null_firth_component_reference(fixture: NullFirthFixture) -> NullFirthComponentReference:
    """Evaluate the Jeffreys-adjusted null score with NumPy."""
    linear_predictor = fixture.covariate_matrix @ fixture.coefficients + fixture.loco_offset
    probability = np.reciprocal(1.0 + np.exp(-linear_predictor))
    weight = probability * (1.0 - probability)
    information = (fixture.covariate_matrix.T * weight) @ fixture.covariate_matrix
    cholesky_factor = np.linalg.cholesky(information)
    negative_log_likelihood = -np.where(
        fixture.phenotype_vector > 0.5,
        np.log(probability),
        np.log1p(-probability),
    )
    deviance = 2.0 * np.sum(negative_log_likelihood) - np.linalg.slogdet(information).logabsdet
    projected_covariates = np.linalg.solve(information, fixture.covariate_matrix.T).T
    leverage = weight * np.sum(projected_covariates * fixture.covariate_matrix, axis=1)
    modified_score = fixture.covariate_matrix.T @ (
        fixture.phenotype_vector - probability + leverage * (0.5 - probability)
    )
    return NullFirthComponentReference(
        information_cholesky_factor=cholesky_factor,
        deviance=float(deviance),
        modified_score=modified_score,
    )


def test_null_firth_components_match_independent_numpy_formula() -> None:
    """Validate information, penalized deviance, leverage, and modified score."""
    fixture = build_null_firth_fixture()
    reference = compute_null_firth_component_reference(fixture)

    observed = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=jnp.asarray(fixture.coefficients),
    )

    tests.numerical.assert_absolute_difference_less_than(
        observed.information_cholesky_factor,
        reference.information_cholesky_factor,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(observed.deviance, reference.deviance, 1.0e-12)
    tests.numerical.assert_absolute_difference_less_than(observed.modified_score, reference.modified_score, 1.0e-12)
    assert bool(np.asarray(observed.valid))


def test_null_firth_line_search_accepts_a_deviance_decreasing_newton_step() -> None:
    """Accept the first full or halved Newton step that improves the objective."""
    fixture = build_null_firth_fixture()
    current_components = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=jnp.asarray(fixture.coefficients),
    )
    information = np.asarray(current_components.information_cholesky_factor)
    information = information @ information.T
    coefficient_step = np.linalg.solve(information, np.asarray(current_components.modified_score))

    observed = regenie2_binary_firth_null.run_null_firth_line_search(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        current_coefficients=jnp.asarray(fixture.coefficients),
        current_deviance=current_components.deviance,
        coefficient_step=jnp.asarray(coefficient_step),
        maximum_attempts=8,
        step_halving_scale=0.5,
    )

    assert bool(np.asarray(observed.accepted))
    assert bool(np.asarray(observed.valid))
    assert float(np.asarray(observed.deviance)) < float(np.asarray(current_components.deviance))


def test_null_firth_zero_attempt_line_search_retains_trusted_state() -> None:
    """Leave coefficients and deviance untouched when no attempt is authorized."""
    fixture = build_null_firth_fixture()
    current_components = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=jnp.asarray(fixture.coefficients),
    )

    observed = regenie2_binary_firth_null.run_null_firth_line_search(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        current_coefficients=jnp.asarray(fixture.coefficients),
        current_deviance=current_components.deviance,
        coefficient_step=jnp.asarray([1.0, -1.0]),
        maximum_attempts=0,
        step_halving_scale=0.5,
    )

    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, fixture.coefficients, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(observed.deviance, current_components.deviance, 1.0e-15)
    assert not bool(np.asarray(observed.accepted))
    assert bool(np.asarray(observed.valid))


def test_null_firth_single_attempt_converges_to_a_small_modified_score() -> None:
    """Converge on a regular fixture under the production numerical policy."""
    fixture = build_null_firth_fixture()
    observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        initial_coefficients=jnp.zeros((fixture.covariate_matrix.shape[1],), dtype=jnp.float64),
        maximum_iterations=100,
        maximum_step_size=PRODUCTION_NULL_FIRTH_MAXIMUM_STEP_SIZE,
        tolerance=PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE,
        line_search_maximum_attempts=25,
        line_search_step_halving_scale=0.5,
        check_score_increase=True,
    )
    terminal_components = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=observed.coefficients,
    )

    assert bool(np.asarray(observed.converged))
    assert bool(np.asarray(terminal_components.valid))
    assert (
        float(np.max(np.abs(np.asarray(terminal_components.modified_score)))) < PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_log_likelihood,
        -0.5 * np.asarray(terminal_components.deviance),
        1.0e-12,
    )


def test_null_firth_zero_iteration_budget_returns_failure_without_moving_start() -> None:
    """Make exhaustion explicit and never present an untrusted likelihood."""
    fixture = build_null_firth_fixture()
    observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        initial_coefficients=jnp.asarray(fixture.coefficients),
        maximum_iterations=0,
        maximum_step_size=5.0,
        tolerance=1.0e-8,
        line_search_maximum_attempts=20,
        line_search_step_halving_scale=0.5,
        check_score_increase=True,
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.isnan(np.asarray(observed.penalized_log_likelihood)))
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, fixture.coefficients, 1.0e-15)
