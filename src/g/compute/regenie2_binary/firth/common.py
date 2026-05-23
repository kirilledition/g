"""Shared Firth utilities for REGENIE step 2 binary kernels."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config

FIRTH_PENALTY_LOG_DETERMINANT_MULTIPLIER = 0.5


def compute_firth_penalized_log_likelihood_from_cholesky(
    probability_vector: jax.Array,
    phenotype_vector: jax.Array,
    information_cholesky_factor: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> jax.Array:
    """Compute Firth-penalized log-likelihood from a Cholesky factor."""
    clipped_probability = jnp.clip(
        probability_vector,
        kernel_config.minimum_probability,
        1.0 - kernel_config.minimum_probability,
    )
    true_class_probability = jnp.where(phenotype_vector == 1.0, clipped_probability, 1.0 - clipped_probability)
    log_likelihood = jnp.sum(jnp.log(true_class_probability))
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(information_cholesky_factor)))
    cholesky_valid = jnp.all(jnp.isfinite(information_cholesky_factor))
    penalty_term = jnp.where(
        cholesky_valid,
        FIRTH_PENALTY_LOG_DETERMINANT_MULTIPLIER * log_determinant,
        -jnp.inf,
    )
    return log_likelihood + penalty_term


def map_firth_reason_code_to_failure_code(reason_code: jax.Array) -> jax.Array:
    """Map internal Firth termination reasons to public failure labels."""
    return jnp.where(
        reason_code == regenie2_binary_firth_types.FirthConvergenceReason.MAX_ITERATIONS.value,
        types.FirthFailureCode.MAX_ITERATIONS.value,
        jnp.where(
            reason_code == regenie2_binary_firth_types.FirthConvergenceReason.INVALID_STATISTIC.value,
            types.FirthFailureCode.INVALID_STATISTIC.value,
            jnp.where(
                reason_code == regenie2_binary_firth_types.FirthConvergenceReason.NEGATIVE_LRT.value,
                types.FirthFailureCode.INVALID_STATISTIC.value,
                jnp.where(
                    (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.STEP_HALVING_EXHAUSTED.value)
                    | (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.STEP_SIZE_INCREASE.value),
                    types.FirthFailureCode.STEP_HALVING.value,
                    jnp.where(
                        (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.NUMERICAL_FAILURE.value)
                        | (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.NULL_FAILURE.value)
                        | (reason_code == regenie2_binary_firth_types.FirthConvergenceReason.PROBABILITY_FAILURE.value),
                        types.FirthFailureCode.NUMERICAL.value,
                        types.FirthFailureCode.NONE.value,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
