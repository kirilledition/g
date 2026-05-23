"""Firth candidate batching helpers for REGENIE step 2 binary tests."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary.firth import full_model as regenie2_binary_firth_full_model
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config


def compute_firth_variantwise(
    covariate_matrix: jax.Array,
    null_logistic_coefficients: jax.Array,
    null_firth_offset: jax.Array,
    phenotype_vector: jax.Array,
    genotype_matrix_by_variant: jax.Array,
    raw_genotype_matrix_by_variant: jax.Array,
    loco_offset: jax.Array,
    initial_coefficients: jax.Array,
    skip_firth_mask: jax.Array,
    sparse_correction_mask: jax.Array,
    null_penalized_log_likelihood: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_firth_types.FirthVariantResult:
    """Compute device-side Firth fits for a padded set of candidate lanes."""
    del null_logistic_coefficients

    scalar_offset_vector = jnp.asarray(null_firth_offset, dtype=jnp.float64)
    scalar_phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)

    def fit_variant(
        genotype_vector: jax.Array,
        raw_genotype_vector: jax.Array,
        variant_initial_coefficients: jax.Array,
        skip_firth: jax.Array,
        sparse_correction: jax.Array,
    ) -> regenie2_binary_firth_types.FirthVariantResult:
        if not kernel_config.approximate_firth.use_block_math:
            return regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
                phenotype_vector=scalar_phenotype_vector,
                genotype_vector=jnp.asarray(genotype_vector, dtype=jnp.float64),
                offset_vector=scalar_offset_vector,
                carrier_sample_mask=raw_genotype_vector
                > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
                sparse_correction=sparse_correction,
                warm_start_beta=jnp.asarray(0.0, dtype=jnp.float64),
                skip_firth=skip_firth,
                null_failed=~jnp.isfinite(null_penalized_log_likelihood),
                kernel_config=kernel_config,
            )
        return regenie2_binary_firth_full_model.fit_single_variant_firth_logistic_regression(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            loco_offset=loco_offset,
            initial_coefficients=variant_initial_coefficients,
            skip_firth=skip_firth,
            null_penalized_log_likelihood=null_penalized_log_likelihood,
            kernel_config=kernel_config,
        )

    return jax.vmap(fit_variant, in_axes=(0, 0, 0, 0, 0))(
        genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
        initial_coefficients,
        skip_firth_mask,
        sparse_correction_mask,
    )
