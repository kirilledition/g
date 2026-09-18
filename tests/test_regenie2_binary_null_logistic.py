"""Analytical and independent regressions for safeguarded null logistic fits."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import null_logistic as regenie2_binary_null_logistic

type FloatingType = type[np.float32] | type[np.float64]


def build_null_logistic_config() -> regenie2_binary_config.BinaryScoreConfig:
    """Use the production null iteration, clipping, and convergence defaults."""
    return regenie2_binary_config.BinaryScoreConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-6,
            minimum_variance=1.0e-8,
            relative_variance_tolerance=1.0e-7,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=50,
            coefficient_tolerance=1.0e-6,
        ),
    )


def find_intercept_reference(
    phenotype_vector: npt.NDArray[np.float64],
    loco_offset: npt.NDArray[np.float64],
) -> float:
    """Find the monotone intercept-score root by independent bisection."""
    lower_intercept = -float(np.max(loco_offset)) - 100.0
    upper_intercept = -float(np.min(loco_offset)) + 100.0
    for _iteration in range(100):
        midpoint = 0.5 * (lower_intercept + upper_intercept)
        probability = np.exp(-np.logaddexp(0.0, -(midpoint + loco_offset)))
        if float(np.sum(phenotype_vector - probability)) > 0.0:
            lower_intercept = midpoint
        else:
            upper_intercept = midpoint
    return 0.5 * (lower_intercept + upper_intercept)


@pytest.mark.parametrize("floating_type", [np.float32, np.float64])
@pytest.mark.parametrize("offset", [-1.0e8, -30.0, -3.0, 0.0, 3.0, 30.0, 1.0e8])
@pytest.mark.parametrize("case_count", [10, 1])
def test_constant_loco_matches_analytical_null_intercept(
    floating_type: FloatingType,
    offset: float,
    case_count: int,
) -> None:
    """Constant offsets preserve balanced and unbalanced cohort probabilities."""
    sample_count = 20
    phenotype = np.zeros(sample_count, dtype=np.float64)
    phenotype[:case_count] = 1.0
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((sample_count, 1), dtype=floating_type),
        phenotype_vector=jnp.asarray(phenotype, dtype=floating_type),
        loco_offset=jnp.full((sample_count,), offset, dtype=floating_type),
        kernel_config=build_null_logistic_config(),
    )
    expected_intercept = np.log(case_count / (sample_count - case_count)) - offset

    assert bool(np.asarray(observed.converged))
    assert not bool(np.asarray(observed.failed))
    assert int(np.asarray(observed.iteration_count)) <= 2
    assert observed.coefficients.dtype == jnp.float64
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, [expected_intercept], 3.0e-6)


@pytest.mark.parametrize("floating_type", [np.float32, np.float64])
@pytest.mark.parametrize("offset_shift", [-30.0, 0.0, 30.0])
def test_nonconstant_loco_shift_preserves_the_independent_score_root(
    floating_type: FloatingType,
    offset_shift: float,
) -> None:
    """Centering protects convergence and fitted probabilities after LOCO shifts."""
    phenotype = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    original_offsets = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0, 3.0], dtype=np.float64)
    shifted_offsets = original_offsets + offset_shift
    reference_intercept = find_intercept_reference(phenotype, original_offsets)
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((phenotype.size, 1), dtype=floating_type),
        phenotype_vector=jnp.asarray(phenotype, dtype=floating_type),
        loco_offset=jnp.asarray(shifted_offsets, dtype=floating_type),
        kernel_config=build_null_logistic_config(),
    )
    observed_predictor = np.asarray(observed.coefficients, dtype=np.float64)[0] + shifted_offsets

    assert bool(np.asarray(observed.converged))
    assert not bool(np.asarray(observed.failed))
    tests.numerical.assert_absolute_difference_less_than(
        observed_predictor,
        reference_intercept + original_offsets,
        3.0e-6,
    )


@pytest.mark.parametrize("floating_type", [np.float32, np.float64])
def test_covariate_null_matches_two_group_analytical_solution(floating_type: FloatingType) -> None:
    """Match intercept and slope from two independently estimable binomial groups."""
    covariates = np.column_stack((np.ones(40), np.repeat([-1.0, 1.0], 20)))
    phenotype = np.zeros(40, dtype=np.float64)
    phenotype[:4] = 1.0
    phenotype[20:32] = 1.0
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.asarray(covariates, dtype=floating_type),
        phenotype_vector=jnp.asarray(phenotype, dtype=floating_type),
        loco_offset=jnp.full((40,), 3.0, dtype=floating_type),
        kernel_config=build_null_logistic_config(),
    )
    first_log_odds = np.log(4.0 / 16.0)
    second_log_odds = np.log(12.0 / 8.0)
    expected = np.asarray([0.5 * (first_log_odds + second_log_odds) - 3.0, 0.5 * (second_log_odds - first_log_odds)])

    assert bool(np.asarray(observed.converged))
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, expected, 2.0e-6)


@pytest.mark.parametrize("floating_type", [np.float32, np.float64])
def test_saturated_nonconstant_offsets_converge_without_newton_overshoot(floating_type: FloatingType) -> None:
    """A near-zero initial Hessian must not send the intercept thousands of units away."""
    phenotype = np.concatenate((np.ones(10), np.zeros(10)))
    offsets = np.asarray([10.0] * 19 + [-190.0], dtype=np.float64)
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((20, 1), dtype=floating_type),
        phenotype_vector=jnp.asarray(phenotype, dtype=floating_type),
        loco_offset=jnp.asarray(offsets, dtype=floating_type),
        kernel_config=build_null_logistic_config(),
    )

    assert bool(np.asarray(observed.converged))
    assert not bool(np.asarray(observed.failed))
    tests.numerical.assert_absolute_difference_less_than(
        observed.coefficients,
        [find_intercept_reference(phenotype, offsets)],
        3.0e-6,
    )


@pytest.mark.parametrize("maximum_iterations", [0, 1])
def test_iteration_exhaustion_remains_an_explicit_failure(maximum_iterations: int) -> None:
    """Neither initialization nor a bounded first step bypasses the iteration budget."""
    kernel_config = build_null_logistic_config()
    limited_config = dataclasses.replace(
        kernel_config,
        null_logistic=dataclasses.replace(kernel_config.null_logistic, maximum_iterations=maximum_iterations),
    )
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((20, 1), dtype=jnp.float64),
        phenotype_vector=jnp.asarray([1.0] * 10 + [0.0] * 10, dtype=jnp.float64),
        loco_offset=jnp.asarray([10.0] * 19 + [-190.0], dtype=jnp.float64),
        kernel_config=limited_config,
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.asarray(observed.failed))
    assert int(np.asarray(observed.iteration_count)) == maximum_iterations
    assert np.all(np.isfinite(np.asarray(observed.coefficients)))


@pytest.mark.parametrize("floating_type", [np.float32, np.float64])
def test_complete_separation_does_not_report_convergence(floating_type: FloatingType) -> None:
    """Clipped probabilities cannot turn an infinite separated estimate into success."""
    covariates = jnp.asarray([[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]], dtype=floating_type)
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=covariates,
        phenotype_vector=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=floating_type),
        loco_offset=jnp.zeros((4,), dtype=floating_type),
        kernel_config=build_null_logistic_config(),
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.asarray(observed.failed))
    assert np.all(np.isfinite(np.asarray(observed.coefficients)))


@pytest.mark.parametrize("invalid_offset", [float("nan"), float("inf")])
def test_nonfinite_input_fails_before_updates(invalid_offset: float) -> None:
    """Invalid inputs never expose a converged fit or propagate invalid coefficients."""
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((4, 1), dtype=jnp.float64),
        phenotype_vector=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float64),
        loco_offset=jnp.asarray([0.0, invalid_offset, 0.0, 0.0], dtype=jnp.float64),
        kernel_config=build_null_logistic_config(),
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.asarray(observed.failed))
    assert int(np.asarray(observed.iteration_count)) == 0
    assert np.all(np.isfinite(np.asarray(observed.coefficients)))


@pytest.mark.parametrize("phenotype_value", [0.0, 1.0])
def test_single_class_cohort_fails_before_updates(phenotype_value: float) -> None:
    """A cohort with no finite intercept maximum cannot report convergence."""
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((4, 1), dtype=jnp.float64),
        phenotype_vector=jnp.full((4,), phenotype_value, dtype=jnp.float64),
        loco_offset=jnp.full((4,), 3.0, dtype=jnp.float64),
        kernel_config=build_null_logistic_config(),
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.asarray(observed.failed))
    assert int(np.asarray(observed.iteration_count)) == 0
    assert np.all(np.isfinite(np.asarray(observed.coefficients)))


def test_line_search_exhaustion_does_not_report_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rejected search must retain its trusted state and report failure."""
    monkeypatch.setattr(regenie2_binary_null_logistic, "NULL_LOGISTIC_LINE_SEARCH_MAXIMUM_ATTEMPTS", 0)
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((4, 1), dtype=jnp.float64),
        phenotype_vector=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float64),
        loco_offset=jnp.asarray([-2.0, -1.0, 1.0, 3.0], dtype=jnp.float64),
        kernel_config=build_null_logistic_config(),
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.asarray(observed.failed))
    assert int(np.asarray(observed.iteration_count)) == 1
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, [-0.25], 1.0e-12)


def test_jitted_null_fit_accepts_shifted_balanced_initial_optimum() -> None:
    """Keep the complete fit and diagnostics usable across the compiled boundary."""
    fit_null = jax.jit(regenie2_binary_null_logistic.fit_null_logistic_coefficients, static_argnames="kernel_config")
    observed = fit_null(
        covariate_matrix=jnp.ones((20, 1), dtype=jnp.float32),
        phenotype_vector=jnp.asarray([0.0, 1.0] * 10, dtype=jnp.float32),
        loco_offset=jnp.full((20,), 3.0, dtype=jnp.float32),
        kernel_config=build_null_logistic_config(),
    )

    assert bool(np.asarray(observed.converged))
    assert not bool(np.asarray(observed.failed))
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, [-3.0], 1.0e-12)
