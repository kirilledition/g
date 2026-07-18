"""CUDA approximate-Firth component evaluation."""

from __future__ import annotations

import jax
import numpy as np

from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

FIRTH_COMPONENTS_FFI_TARGET = "g.firth.components.v1"


def compute_scalar_firth_components(
    *,
    phenotype_vector: jax.Array,
    genotype_vector: jax.Array,
    offset_vector: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    beta: jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.ScalarFirthComponents:
    """Compute one lane through the vectorizable CUDA FFI."""
    output_shape = beta.shape
    foreign_outputs = jax.ffi.ffi_call(
        FIRTH_COMPONENTS_FFI_TARGET,
        (
            jax.ShapeDtypeStruct(output_shape, np.float64),
            jax.ShapeDtypeStruct(output_shape, np.float64),
            jax.ShapeDtypeStruct(output_shape, np.float64),
            jax.ShapeDtypeStruct(output_shape, np.float64),
            jax.ShapeDtypeStruct(output_shape, np.bool_),
        ),
        vmap_method="broadcast_all",
    )(
        phenotype_vector,
        genotype_vector,
        offset_vector,
        active_sample_mask,
        non_active_deviance,
        beta,
        minimum_variance,
    )
    genotype_information, score_adjustment, penalized_deviance, score, valid = foreign_outputs
    return regenie2_binary_firth_types.ScalarFirthComponents(
        genotype_information=genotype_information,
        score_adjustment=score_adjustment,
        penalized_deviance=penalized_deviance,
        score=score,
        valid=valid,
    )
