from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from g.compute.logistic import (
    BINARY_CASE_THRESHOLD,
    FIRTH_BATCH_SIZE,
    INITIAL_RESPONSE_SCALE,
    LogisticErrorCode,
    LogisticMethod,
    build_firth_padded_index_batches,
    compute_covariate_only_probability_matrix,
    compute_firth_association_chunk_with_mask,
    compute_firth_pre_dispatch_mask,
    compute_firth_pre_dispatch_mask_without_mask,
    compute_logistic_association_chunk,
    compute_logistic_association_chunk_with_mask,
    compute_probability_matrix,
    compute_standard_logistic_association_chunk_with_mask,
    compute_standard_logistic_association_chunk_without_mask,
    fit_covariate_only_logistic_regression,
    fit_masked_covariate_only_logistic_regression,
    fit_single_variant_firth_logistic_regression,
    initialize_full_model_coefficients,
    initialize_full_model_coefficients_without_mask,
    initialize_mixed_firth_batch_coefficients,
    prepare_logistic_chunk_precomputation,
    prepare_no_missing_logistic_constants,
)


def build_logistic_fixture() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build a small non-separated logistic regression fixture."""
    covariate_matrix = jnp.array(
        [
            [1.0, -1.0],
            [1.0, -0.2],
            [1.0, 0.4],
            [1.0, 1.2],
            [1.0, 1.8],
            [1.0, 2.5],
        ]
    )
    phenotype_vector = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    genotype_matrix = jnp.array(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    observation_mask = jnp.ones((genotype_matrix.shape[1], genotype_matrix.shape[0]), dtype=bool)
    return covariate_matrix, phenotype_vector, genotype_matrix, observation_mask


def test_compute_probability_matrix_matches_manual_sigmoid() -> None:
    """Ensure full-model logistic probabilities use covariates and genotype coefficients."""
    covariate_matrix = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    genotype_matrix_by_variant = jnp.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    coefficients = jnp.array([[0.0, 1.0, 0.5], [1.0, -1.0, -0.25]])

    probability_matrix = compute_probability_matrix(covariate_matrix, genotype_matrix_by_variant, coefficients)

    expected_probability_matrix = 1.0 / (
        1.0
        + np.exp(
            -np.array(
                [
                    [0.0, 1.5, 3.0],
                    [0.5, -0.25, -1.0],
                ]
            )
        )
    )
    np.testing.assert_allclose(probability_matrix, expected_probability_matrix, atol=1e-6)


def test_initialize_full_model_coefficients_without_mask_matches_all_true_mask() -> None:
    """Ensure the no-missing initializer matches the masked path on identical data."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    genotype_matrix_by_variant = genotype_matrix.T
    no_missing_constants = prepare_no_missing_logistic_constants(covariate_matrix, phenotype_vector)
    logistic_chunk_precomputation = prepare_logistic_chunk_precomputation(covariate_matrix)

    masked_coefficients = initialize_full_model_coefficients(
        covariate_matrix,
        genotype_matrix_by_variant,
        phenotype_vector,
        observation_mask,
        logistic_chunk_precomputation,
    )
    no_missing_coefficients = initialize_full_model_coefficients_without_mask(
        covariate_matrix,
        genotype_matrix_by_variant,
        no_missing_constants,
    )

    np.testing.assert_allclose(no_missing_coefficients, masked_coefficients, atol=1e-6)


def test_covariate_only_fit_matches_masked_all_true_path() -> None:
    """Ensure the scalar covariate-only fit matches the batched masked implementation."""
    covariate_matrix, phenotype_vector, _, _ = build_logistic_fixture()
    initial_coefficients = jnp.linalg.solve(
        covariate_matrix.T @ covariate_matrix,
        covariate_matrix.T @ (INITIAL_RESPONSE_SCALE * (phenotype_vector - BINARY_CASE_THRESHOLD)),
    )
    observation_mask = jnp.ones((1, phenotype_vector.shape[0]), dtype=bool)

    unmasked_coefficients = fit_covariate_only_logistic_regression(
        covariate_matrix,
        phenotype_vector,
        max_iterations=25,
        tolerance=1.0e-8,
    )
    masked_coefficients = fit_masked_covariate_only_logistic_regression(
        covariate_matrix,
        phenotype_vector,
        observation_mask,
        initial_coefficients,
        max_iterations=25,
        tolerance=1.0e-8,
    )[0]

    np.testing.assert_allclose(unmasked_coefficients, masked_coefficients, atol=1e-6)


def test_firth_pre_dispatch_masks_match_between_masked_and_unmasked_paths() -> None:
    """Ensure separation heuristics agree when observation masks are all true."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    genotype_matrix_by_variant = genotype_matrix.T
    no_missing_constants = prepare_no_missing_logistic_constants(covariate_matrix, phenotype_vector)

    masked_pre_dispatch_mask = compute_firth_pre_dispatch_mask(
        phenotype_vector,
        genotype_matrix_by_variant,
        observation_mask,
    )
    unmasked_pre_dispatch_mask = compute_firth_pre_dispatch_mask_without_mask(
        genotype_matrix_by_variant,
        no_missing_constants,
    )

    np.testing.assert_array_equal(masked_pre_dispatch_mask, unmasked_pre_dispatch_mask)


def test_standard_logistic_with_mask_matches_without_mask_for_complete_data() -> None:
    """Ensure masked and unmasked standard-logistic kernels agree on complete data."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    no_missing_constants = prepare_no_missing_logistic_constants(covariate_matrix, phenotype_vector)

    masked_evaluation = compute_standard_logistic_association_chunk_with_mask(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        observation_mask,
        max_iterations=25,
        tolerance=1.0e-8,
    )
    unmasked_evaluation = compute_standard_logistic_association_chunk_without_mask(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        no_missing_constants,
        max_iterations=25,
        tolerance=1.0e-8,
    )

    np.testing.assert_allclose(
        masked_evaluation.logistic_result.beta, unmasked_evaluation.logistic_result.beta, atol=1e-5
    )
    np.testing.assert_allclose(
        masked_evaluation.logistic_result.standard_error,
        unmasked_evaluation.logistic_result.standard_error,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        masked_evaluation.logistic_result.converged_mask,
        unmasked_evaluation.logistic_result.converged_mask,
    )


def test_fit_single_variant_firth_logistic_regression_respects_skip_flag() -> None:
    """Ensure skipped Firth fits return dummy values without consuming iterations."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    genotype_vector = genotype_matrix[:, 0]
    initial_coefficients = jnp.array([0.1, -0.1, 0.05])
    skip_firth = jnp.asarray(np.array(1, dtype=np.bool_))

    result = fit_single_variant_firth_logistic_regression(
        covariate_matrix,
        phenotype_vector,
        genotype_vector,
        observation_mask[0],
        initial_coefficients,
        skip_firth,
        max_iterations=25,
        tolerance=1.0e-8,
    )

    assert bool(result.converged_mask)
    assert not bool(result.valid_mask)
    assert int(result.error_code) == LogisticErrorCode.NONE
    assert int(result.iteration_count) == 0
    assert np.isnan(float(result.beta))


def test_build_firth_padded_index_batches_preserves_active_indices_and_padding() -> None:
    """Ensure Firth batch padding preserves active ordering and fixed batch size."""
    fallback_indices = np.arange(FIRTH_BATCH_SIZE + 2, dtype=np.int64)

    firth_index_batches = build_firth_padded_index_batches(fallback_indices)

    assert len(firth_index_batches) == 2
    np.testing.assert_array_equal(
        firth_index_batches[0].active_index_array, np.arange(FIRTH_BATCH_SIZE, dtype=np.int64)
    )
    assert firth_index_batches[0].padded_index_array.shape == (FIRTH_BATCH_SIZE,)
    np.testing.assert_array_equal(
        firth_index_batches[1].active_index_array, np.array([FIRTH_BATCH_SIZE, FIRTH_BATCH_SIZE + 1])
    )
    np.testing.assert_array_equal(
        firth_index_batches[1].padded_index_array[:2],
        np.array([FIRTH_BATCH_SIZE, FIRTH_BATCH_SIZE + 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        firth_index_batches[1].padded_index_array[2:],
        np.full((FIRTH_BATCH_SIZE - 2,), FIRTH_BATCH_SIZE, dtype=np.int32),
    )


def test_initialize_mixed_firth_batch_coefficients_preserves_standard_rows_when_no_heuristics() -> None:
    """Ensure mixed initialization keeps standard coefficients for non-heuristic rows."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    standard_coefficients = jnp.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    firth_index_batch = build_firth_padded_index_batches(np.array([0, 1], dtype=np.int64))[0]
    logistic_chunk_precomputation = prepare_logistic_chunk_precomputation(covariate_matrix)

    batch_coefficients = initialize_mixed_firth_batch_coefficients(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        observation_mask,
        logistic_chunk_precomputation,
        firth_index_batch,
        np.array([False, False]),
        standard_coefficients,
    )

    np.testing.assert_allclose(batch_coefficients[:2], np.array(standard_coefficients), atol=0.0)


def test_compute_firth_association_chunk_with_mask_marks_firth_method() -> None:
    """Ensure explicit Firth chunk execution labels every variant with the Firth method code."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    initial_coefficients = jnp.tile(jnp.array([[0.0, 0.0, 0.0]]), (genotype_matrix.shape[1], 1))

    logistic_result = compute_firth_association_chunk_with_mask(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        observation_mask,
        initial_coefficients,
        max_iterations=5,
        tolerance=1.0e-6,
    )

    np.testing.assert_array_equal(
        logistic_result.method_code,
        np.full((genotype_matrix.shape[1],), LogisticMethod.FIRTH, dtype=np.int32),
    )
    assert logistic_result.beta.shape == (genotype_matrix.shape[1],)


def test_top_level_logistic_chunk_functions_match_on_complete_data() -> None:
    """Ensure the masked and unmasked top-level logistic paths agree on complete data."""
    covariate_matrix, phenotype_vector, genotype_matrix, observation_mask = build_logistic_fixture()
    no_missing_constants = prepare_no_missing_logistic_constants(covariate_matrix, phenotype_vector)

    unmasked_result = compute_logistic_association_chunk(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        max_iterations=25,
        tolerance=1.0e-8,
        no_missing_constants=no_missing_constants,
    )
    masked_result = compute_logistic_association_chunk_with_mask(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        observation_mask,
        max_iterations=25,
        tolerance=1.0e-8,
    )

    np.testing.assert_allclose(masked_result.beta, unmasked_result.beta, atol=1e-4)
    np.testing.assert_allclose(masked_result.p_value, unmasked_result.p_value, atol=1e-5)
    np.testing.assert_array_equal(masked_result.method_code, unmasked_result.method_code)


def test_covariate_only_probability_matrix_stays_clipped_at_extremes() -> None:
    """Ensure extreme coefficients still produce finite, clipped probabilities."""
    covariate_matrix = jnp.array([[1.0], [1.0], [1.0]])
    coefficients = jnp.array([[1000.0], [-1000.0]])

    probability_matrix = compute_covariate_only_probability_matrix(covariate_matrix, coefficients)

    assert np.isfinite(np.asarray(probability_matrix)).all()
    assert float(probability_matrix[0, 0]) < 1.0
    assert float(probability_matrix[1, 0]) > 0.0
