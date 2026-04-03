import jax.numpy as jnp
import numpy as np

from g.compute.logistic import (
    InformationComponents,
    assemble_full_information_matrix,
    compute_covariate_information_matrix,
    compute_covariate_only_probability_matrix,
    compute_covariate_score,
    compute_firth_penalized_log_likelihood,
    compute_information_components,
    compute_log_likelihood,
    initialize_full_model_coefficients,
    prepare_logistic_chunk_precomputation,
    prepare_no_missing_logistic_constants,
)


def test_assemble_full_information_matrix_basic():
    # Setup
    batch_size = 2
    num_covariates = 3

    # M x P x P
    covariate_information_matrix = jnp.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ]
    )

    # M x P
    cross_information_vector = jnp.array([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]])

    # M
    genotype_information = jnp.array([25.0, 26.0])

    # Execute
    result = assemble_full_information_matrix(
        covariate_information_matrix, cross_information_vector, genotype_information
    )

    # Assert
    assert result.shape == (batch_size, num_covariates + 1, num_covariates + 1)

    # Check first batch element
    expected_0 = jnp.array(
        [[1.0, 2.0, 3.0, 19.0], [4.0, 5.0, 6.0, 20.0], [7.0, 8.0, 9.0, 21.0], [19.0, 20.0, 21.0, 25.0]]
    )
    np.testing.assert_allclose(result[0], expected_0)

    # Check second batch element
    expected_1 = jnp.array(
        [[10.0, 11.0, 12.0, 22.0], [13.0, 14.0, 15.0, 23.0], [16.0, 17.0, 18.0, 24.0], [22.0, 23.0, 24.0, 26.0]]
    )
    np.testing.assert_allclose(result[1], expected_1)


def test_assemble_full_information_matrix_no_covariates():
    # Setup: P = 0
    batch_size = 2
    num_covariates = 0

    covariate_information_matrix = jnp.zeros((batch_size, num_covariates, num_covariates))
    cross_information_vector = jnp.zeros((batch_size, num_covariates))
    genotype_information = jnp.array([1.0, 2.0])

    # Execute
    result = assemble_full_information_matrix(
        covariate_information_matrix, cross_information_vector, genotype_information
    )

    # Assert
    assert result.shape == (batch_size, 1, 1)

    expected_0 = jnp.array([[1.0]])
    np.testing.assert_allclose(result[0], expected_0)

    expected_1 = jnp.array([[2.0]])
    np.testing.assert_allclose(result[1], expected_1)


def test_assemble_full_information_matrix_batch_size_one():
    # Setup: M = 1
    batch_size = 1
    num_covariates = 2

    covariate_information_matrix = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    cross_information_vector = jnp.array([[5.0, 6.0]])
    genotype_information = jnp.array([7.0])

    # Execute
    result = assemble_full_information_matrix(
        covariate_information_matrix, cross_information_vector, genotype_information
    )

    # Assert
    assert result.shape == (batch_size, num_covariates + 1, num_covariates + 1)

    expected_0 = jnp.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [5.0, 6.0, 7.0]])
    np.testing.assert_allclose(result[0], expected_0)


def test_compute_covariate_information_matrix() -> None:
    """Test computation of covariate information matrix."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    effective_weights = jnp.array(
        [
            [0.1, 0.2, 0.15],
            [0.3, 0.1, 0.4],
        ]
    )

    result = compute_covariate_information_matrix(covariate_matrix, effective_weights)

    assert result.shape == (2, 2, 2)


def test_compute_covariate_score() -> None:
    """Test computation of covariate score."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    residual_matrix = jnp.array(
        [
            [0.1, 0.2, 0.15],
            [0.3, 0.1, 0.4],
        ]
    )

    result = compute_covariate_score(covariate_matrix, residual_matrix)

    assert result.shape == (2, 2)


def test_prepare_no_missing_logistic_constants() -> None:
    """Test preparation of no-missing logistic constants."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5, 25.0],
            [1.0, -0.5, 30.0],
            [1.0, 2.0, 35.0],
            [1.0, 0.0, 40.0],
        ]
    )
    phenotype_vector = jnp.array([0.0, 1.0, 0.0, 1.0])

    constants = prepare_no_missing_logistic_constants(covariate_matrix, phenotype_vector)

    assert constants.case_sample_count == 2
    assert constants.control_sample_count == 2
    assert constants.case_mask.shape == (4,)
    assert constants.control_mask.shape == (4,)


def test_compute_log_likelihood_basic() -> None:
    """Test basic log-likelihood computation."""
    probability_matrix = jnp.array([[0.5, 0.8, 0.2], [0.1, 0.9, 0.5]])
    phenotype_matrix = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    log_likelihood = compute_log_likelihood(probability_matrix, phenotype_matrix)

    assert log_likelihood.shape == (2,)

    expected_row_0 = np.log(0.5) + np.log(0.8) + np.log(0.8)
    expected_row_1 = np.log(0.1) + np.log(0.1) + np.log(0.5)

    np.testing.assert_allclose(log_likelihood[0], expected_row_0, rtol=1e-5)
    np.testing.assert_allclose(log_likelihood[1], expected_row_1, rtol=1e-5)


def test_compute_log_likelihood_extreme_values() -> None:
    """Test log-likelihood with extreme probability values."""
    probability_matrix = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    phenotype_matrix = jnp.array([[1.0, 1.0], [0.0, 0.0]])

    log_likelihood = compute_log_likelihood(probability_matrix, phenotype_matrix)

    assert log_likelihood.shape == (2,)
    assert jnp.isfinite(log_likelihood).all()


def test_compute_covariate_only_probability_matrix_basic() -> None:
    """Test covariate-only probability matrix computation."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )

    coefficients = jnp.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
        ]
    )

    result = compute_covariate_only_probability_matrix(covariate_matrix, coefficients)

    assert result.shape == (2, 3)

    np.testing.assert_allclose(result[0], jnp.array([0.5, 0.5, 0.5]), rtol=1e-5)


def test_compute_information_components() -> None:
    """Test computation of information components for logistic regression."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    genotype_vector = jnp.array([0.0, 1.0, 2.0])
    probability_vector = jnp.array([0.3, 0.7, 0.5])
    observation_mask = jnp.array([True, True, True])

    result = compute_information_components(covariate_matrix, genotype_vector, probability_vector, observation_mask)

    assert isinstance(result, InformationComponents)
    assert result.covariate_information_matrix.shape == (2, 2)
    assert result.cross_information_vector.shape == (2,)
    assert result.genotype_information.shape == ()


def test_compute_firth_penalized_log_likelihood() -> None:
    """Test computation of Firth penalized log-likelihood."""
    probability_vector = jnp.array([0.3, 0.7, 0.5, 0.9])
    phenotype_vector = jnp.array([0.0, 1.0, 0.0, 1.0])
    observation_mask = jnp.array([True, True, True, True])
    information_matrix = jnp.eye(2) * 2.0

    result = compute_firth_penalized_log_likelihood(
        probability_vector, phenotype_vector, observation_mask, information_matrix
    )

    assert jnp.isscalar(result) or result.shape == ()
    assert jnp.isfinite(result)


def test_initialize_full_model_coefficients() -> None:
    """Test initialization of full model coefficients."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    genotype_matrix_by_variant = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
        ]
    )
    phenotype_vector = jnp.array([0.0, 1.0, 0.0])
    observation_mask = jnp.array(
        [
            [True, True, True],
            [True, True, True],
        ]
    )
    logistic_chunk_precomputation = prepare_logistic_chunk_precomputation(covariate_matrix)

    result = initialize_full_model_coefficients(
        covariate_matrix,
        genotype_matrix_by_variant,
        phenotype_vector,
        observation_mask,
        logistic_chunk_precomputation,
    )

    assert result.shape == (2, 3)
