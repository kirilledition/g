"""Shared binary logistic probability and deviance helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary import config as regenie2_binary_config


def compute_regenie_logistic_probability(linear_predictor: jax.Array) -> jax.Array:
    """Compute probabilities with REGENIE's glm-style endpoint clipping."""
    epsilon = jnp.asarray(
        regenie2_binary_config.REGENIE_NUMERICAL_EPSILON_MULTIPLIER * jnp.finfo(linear_predictor.dtype).eps,
        dtype=linear_predictor.dtype,
    )
    lower_probability = epsilon / (1.0 + epsilon)
    upper_probability = jnp.reciprocal(1.0 + epsilon)
    return jnp.where(
        linear_predictor > regenie2_binary_config.REGENIE_LOGISTIC_MAXIMUM_ETA,
        upper_probability,
        jnp.where(
            linear_predictor < regenie2_binary_config.REGENIE_LOGISTIC_MINIMUM_ETA,
            lower_probability,
            jax.nn.sigmoid(linear_predictor),
        ),
    )


def compute_clipped_logistic_probability(
    linear_predictor: jax.Array,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> jax.Array:
    """Compute sigmoid probabilities clipped by configured binary floors."""
    probability = jax.nn.sigmoid(linear_predictor)
    return jnp.clip(
        probability,
        kernel_config.numerical.minimum_probability,
        1.0 - kernel_config.numerical.minimum_probability,
    )


def compute_logistic_deviance(
    phenotype_vector: jax.Array,
    probability_vector: jax.Array,
    active_sample_mask: jax.Array,
) -> jax.Array:
    """Compute Bernoulli deviance from REGENIE-clipped probabilities."""
    negative_log_likelihood = -jnp.where(
        phenotype_vector > regenie2_binary_config.BINARY_CASE_THRESHOLD,
        jnp.log(probability_vector),
        jnp.log1p(-probability_vector),
    )
    return 2.0 * jnp.sum(jnp.where(active_sample_mask, negative_log_likelihood, 0.0))
