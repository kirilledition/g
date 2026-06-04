from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g import types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


def build_full_model_fixture() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    covariate_matrix = jnp.asarray(
        [
            [1.0, -1.0],
            [1.0, -0.2],
            [1.0, 0.4],
            [1.0, 1.2],
            [1.0, 1.8],
            [1.0, 2.5],
        ],
        dtype=jnp.float32,
    )
    phenotype_vector = jnp.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    genotype_vector = jnp.asarray([0.0, 1.0, 0.0, 2.0, 1.0, 2.0], dtype=jnp.float32)
    loco_offset = jnp.zeros_like(phenotype_vector)
    return covariate_matrix, phenotype_vector, genotype_vector, loco_offset


def build_kernel_config(
    *,
    maximum_iterations: int = 25,
    use_block_math: bool = False,
) -> regenie2_binary_config.BinaryKernelConfig:
    return dataclasses.replace(
        regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
        approximate_firth=dataclasses.replace(
            regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG.approximate_firth,
            maximum_iterations=maximum_iterations,
            use_block_math=use_block_math,
        ),
    )


def initialize_fixture_coefficients(
    *,
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> jax.Array:
    return regenie2_binary_firth_full_model.initialize_full_model_coefficients_without_mask(
        covariate_matrix,
        genotype_vector[None, :],
        phenotype_vector,
        kernel_config,
    )[0]


def test_full_model_information_matrix_stacks_blocks() -> None:
    information_matrix = regenie2_binary_firth_full_model.build_full_model_information_matrix(
        covariate_information_matrix=jnp.asarray([[2.0, 0.5], [0.5, 1.5]], dtype=jnp.float32),
        cross_information_vector=jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        genotype_information=jnp.asarray(3.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(information_matrix),
        np.asarray(
            [
                [2.0, 0.5, 0.25],
                [0.5, 1.5, -0.75],
                [0.25, -0.75, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_weighted_information_components_match_probability_path() -> None:
    covariate_matrix, _phenotype_vector, genotype_vector, _loco_offset = build_full_model_fixture()
    probability_vector = jnp.asarray([0.15, 0.25, 0.55, 0.45, 0.75, 0.85], dtype=jnp.float32)
    kernel_config = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG
    weight_vector = probability_vector * (1.0 - probability_vector)

    probability_components = regenie2_binary_firth_full_model.compute_information_components(
        covariate_matrix,
        genotype_vector,
        probability_vector,
        kernel_config,
    )
    weighted_components = regenie2_binary_firth_full_model.compute_weighted_full_model_information_components(
        covariate_matrix,
        genotype_vector,
        weight_vector,
    )

    np.testing.assert_allclose(
        np.asarray(weighted_components.information_matrix),
        np.asarray(probability_components.information_matrix),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_full_model_score_components_match_manual_blocks() -> None:
    covariate_matrix, _phenotype_vector, genotype_vector, _loco_offset = build_full_model_fixture()
    score_weight_vector = jnp.asarray([-0.1, 0.25, 0.5, -0.2, 0.3, -0.05], dtype=jnp.float32)

    score_components = regenie2_binary_firth_full_model.compute_full_model_score_components(
        covariate_matrix,
        genotype_vector,
        score_weight_vector,
    )

    np.testing.assert_allclose(
        np.asarray(score_components.covariate_score),
        np.asarray(covariate_matrix.T @ score_weight_vector),
    )
    np.testing.assert_allclose(
        np.asarray(score_components.genotype_score),
        np.asarray(jnp.dot(genotype_vector, score_weight_vector)),
    )


def test_penalized_log_likelihood_rejects_invalid_cholesky_factor() -> None:
    phenotype_vector = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    probability_vector = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    information_cholesky_factor = jnp.asarray([[1.0, 0.0], [0.0, jnp.nan]], dtype=jnp.float32)

    likelihood = regenie2_binary_firth_full_model.compute_firth_penalized_log_likelihood_from_cholesky(
        probability_vector,
        phenotype_vector,
        information_cholesky_factor,
        regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    )

    assert np.isneginf(np.asarray(likelihood))


@pytest.mark.parametrize("use_block_math", [False, True])
def test_full_model_firth_solver_converges_for_tiny_fixture(use_block_math: bool) -> None:  # noqa: FBT001
    covariate_matrix, phenotype_vector, genotype_vector, loco_offset = build_full_model_fixture()
    kernel_config = build_kernel_config(use_block_math=use_block_math)
    initial_coefficients = initialize_fixture_coefficients(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        kernel_config=kernel_config,
    )

    result = regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth=jnp.zeros((), dtype=jnp.bool_),
        null_penalized_log_likelihood=jnp.asarray(-10.0, dtype=jnp.float32),
        kernel_config=kernel_config,
    )

    assert bool(np.asarray(result.valid_mask))
    assert bool(np.asarray(result.converged_mask))
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.NONE.value
    assert int(np.asarray(result.correction_code)) == types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value
    assert int(np.asarray(result.iteration_count)) > 0
    assert np.isfinite(np.asarray(result.beta))
    assert np.isfinite(np.asarray(result.standard_error))


def test_full_model_firth_solver_reports_skipped_variant_without_iterations() -> None:
    covariate_matrix, phenotype_vector, genotype_vector, loco_offset = build_full_model_fixture()
    kernel_config = build_kernel_config()
    initial_coefficients = initialize_fixture_coefficients(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        kernel_config=kernel_config,
    )

    result = regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth=jnp.ones((), dtype=jnp.bool_),
        null_penalized_log_likelihood=jnp.asarray(-10.0, dtype=jnp.float32),
        kernel_config=kernel_config,
    )

    assert not bool(np.asarray(result.valid_mask))
    assert not bool(np.asarray(result.converged_mask))
    assert int(np.asarray(result.iteration_count)) == 0
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.NONE.value
    assert int(np.asarray(result.correction_code)) == types.FirthCorrectionCode.NONE.value
    assert np.isnan(np.asarray(result.beta))


def test_full_model_firth_solver_reports_null_failure() -> None:
    covariate_matrix, phenotype_vector, genotype_vector, loco_offset = build_full_model_fixture()
    kernel_config = build_kernel_config()
    initial_coefficients = initialize_fixture_coefficients(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        kernel_config=kernel_config,
    )

    result = regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth=jnp.zeros((), dtype=jnp.bool_),
        null_penalized_log_likelihood=jnp.asarray(jnp.nan, dtype=jnp.float32),
        kernel_config=kernel_config,
    )

    assert not bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.iteration_count)) == 0
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.NUMERICAL.value
    assert (
        int(np.asarray(result.convergence_reason_code))
        == regenie2_binary_firth_types.FirthConvergenceReason.NULL_FAILURE.value
    )


def test_full_model_firth_solver_reports_max_iteration_failure() -> None:
    covariate_matrix, phenotype_vector, genotype_vector, loco_offset = build_full_model_fixture()
    kernel_config = build_kernel_config(maximum_iterations=1)
    initial_coefficients = initialize_fixture_coefficients(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        kernel_config=kernel_config,
    )

    result = regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth=jnp.zeros((), dtype=jnp.bool_),
        null_penalized_log_likelihood=jnp.asarray(-10.0, dtype=jnp.float32),
        kernel_config=kernel_config,
    )

    assert not bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.iteration_count)) == 1
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.MAX_ITERATIONS.value
    assert (
        int(np.asarray(result.convergence_reason_code))
        == regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value
    )


def test_initializer_produces_finite_clipped_probability_vector() -> None:
    covariate_matrix, phenotype_vector, genotype_vector, _loco_offset = build_full_model_fixture()
    kernel_config = build_kernel_config()

    initial_coefficients = initialize_fixture_coefficients(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        kernel_config=kernel_config,
    )
    probability_vector = regenie2_binary_logistic.compute_clipped_logistic_probability(
        covariate_matrix @ initial_coefficients[:-1] + genotype_vector * initial_coefficients[-1],
        kernel_config,
    )

    assert initial_coefficients.shape == (covariate_matrix.shape[1] + 1,)
    assert np.isfinite(np.asarray(initial_coefficients)).all()
    assert ((np.asarray(probability_vector) > 0.0) & (np.asarray(probability_vector) < 1.0)).all()
