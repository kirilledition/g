import jax.numpy as jnp
import numpy as np

from g.compute.logistic import (
    MINIMUM_PROBABILITY,
    compute_covariate_only_probability_matrix,
    compute_log_likelihood,
)


def test_compute_log_likelihood() -> None:
    """Ensure logistic log-likelihood behaves correctly with normal and extreme inputs."""
    probability_matrix = jnp.array([[0.5, 0.8, 0.2], [0.0, 1.0, 0.5]])
    phenotype_matrix = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    log_likelihood = compute_log_likelihood(probability_matrix, phenotype_matrix)

    expected_row_0 = np.log(0.5) + np.log(0.8) + np.log(0.8)
    expected_row_1 = np.log(MINIMUM_PROBABILITY) + np.log(MINIMUM_PROBABILITY) + np.log(0.5)

    expected_log_likelihood = jnp.array([expected_row_0, expected_row_1])

    np.testing.assert_allclose(log_likelihood, expected_log_likelihood, rtol=1e-5)


def test_compute_covariate_only_probability_matrix() -> None:
    """Test batched covariate-only logistic probabilities computation."""
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
            [10.0, 20.0],
        ]
    )

    result = compute_covariate_only_probability_matrix(covariate_matrix, coefficients)

    assert result.shape == (2, 3)

    expected_prob_0 = jnp.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(result[0], expected_prob_0, rtol=1e-5)

    expected_prob_1 = jnp.array(
        [
            1.0 - MINIMUM_PROBABILITY,
            0.5,
            1.0 - MINIMUM_PROBABILITY,
        ]
    )
    np.testing.assert_allclose(result[1], expected_prob_1, rtol=1e-5)

    coefficients_neg = jnp.array(
        [
            [-100.0, -100.0],
        ]
    )
    result_neg = compute_covariate_only_probability_matrix(covariate_matrix, coefficients_neg)

    expected_prob_neg = jnp.array(
        [
            MINIMUM_PROBABILITY,
            MINIMUM_PROBABILITY,
            MINIMUM_PROBABILITY,
        ]
    )
    np.testing.assert_allclose(result_neg[0], expected_prob_neg, rtol=1e-5)
