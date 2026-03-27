import jax.numpy as jnp
import numpy as np

from g.compute.logistic import MINIMUM_PROBABILITY, compute_log_likelihood


def test_compute_log_likelihood() -> None:
    """Ensure logistic log-likelihood behaves correctly with normal and extreme inputs."""
    probability_matrix = jnp.array([[0.5, 0.8, 0.2], [0.0, 1.0, 0.5]])
    phenotype_matrix = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    log_likelihood = compute_log_likelihood(probability_matrix, phenotype_matrix)

    expected_row_0 = np.log(0.5) + np.log(0.8) + np.log(0.8)
    expected_row_1 = np.log(MINIMUM_PROBABILITY) + np.log(MINIMUM_PROBABILITY) + np.log(0.5)

    expected_log_likelihood = jnp.array([expected_row_0, expected_row_1])

    np.testing.assert_allclose(log_likelihood, expected_log_likelihood, rtol=1e-5)
