import jax.numpy as jnp
import numpy as np

from g.compute.logistic import assemble_full_information_matrix


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
