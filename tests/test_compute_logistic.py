import jax.numpy as jnp
import numpy as np
import pytest

from g.compute.logistic import assemble_full_information_matrix

def test_assemble_full_information_matrix_basic():
    # Setup
    batch_size = 2
    num_covariates = 3

    # M x P x P
    covariate_information_matrix = jnp.array([
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]
    ])

    # M x P
    cross_information_vector = jnp.array([
        [19., 20., 21.],
        [22., 23., 24.]
    ])

    # M
    genotype_information = jnp.array([25., 26.])

    # Execute
    result = assemble_full_information_matrix(
        covariate_information_matrix,
        cross_information_vector,
        genotype_information
    )

    # Assert
    assert result.shape == (batch_size, num_covariates + 1, num_covariates + 1)

    # Check first batch element
    expected_0 = jnp.array([
        [1., 2., 3., 19.],
        [4., 5., 6., 20.],
        [7., 8., 9., 21.],
        [19., 20., 21., 25.]
    ])
    np.testing.assert_allclose(result[0], expected_0)

    # Check second batch element
    expected_1 = jnp.array([
        [10., 11., 12., 22.],
        [13., 14., 15., 23.],
        [16., 17., 18., 24.],
        [22., 23., 24., 26.]
    ])
    np.testing.assert_allclose(result[1], expected_1)

def test_assemble_full_information_matrix_no_covariates():
    # Setup: P = 0
    batch_size = 2
    num_covariates = 0

    covariate_information_matrix = jnp.zeros((batch_size, num_covariates, num_covariates))
    cross_information_vector = jnp.zeros((batch_size, num_covariates))
    genotype_information = jnp.array([1., 2.])

    # Execute
    result = assemble_full_information_matrix(
        covariate_information_matrix,
        cross_information_vector,
        genotype_information
    )

    # Assert
    assert result.shape == (batch_size, 1, 1)

    expected_0 = jnp.array([[1.]])
    np.testing.assert_allclose(result[0], expected_0)

    expected_1 = jnp.array([[2.]])
    np.testing.assert_allclose(result[1], expected_1)

def test_assemble_full_information_matrix_batch_size_one():
    # Setup: M = 1
    batch_size = 1
    num_covariates = 2

    covariate_information_matrix = jnp.array([
        [[1., 2.], [3., 4.]]
    ])
    cross_information_vector = jnp.array([
        [5., 6.]
    ])
    genotype_information = jnp.array([7.])

    # Execute
    result = assemble_full_information_matrix(
        covariate_information_matrix,
        cross_information_vector,
        genotype_information
    )

    # Assert
    assert result.shape == (batch_size, num_covariates + 1, num_covariates + 1)

    expected_0 = jnp.array([
        [1., 2., 5.],
        [3., 4., 6.],
        [5., 6., 7.]
    ])
    np.testing.assert_allclose(result[0], expected_0)
