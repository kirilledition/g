from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np

from g.compute.regenie2_binary.firth import cuda_components as regenie2_binary_firth_cuda_components
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx

if typing.TYPE_CHECKING:
    import pytest


def test_cuda_component_abstract_evaluation_preserves_multi_axis_batch_prefix() -> None:
    """Keep typed-XLA batching compatible with nested lane vectorization."""
    sample_operand = jax.ShapeDtypeStruct((2, 3, 5), np.float64)
    mask_operand = jax.ShapeDtypeStruct((2, 3, 5), np.bool_)
    lane_operand = jax.ShapeDtypeStruct((2, 3), np.float64)

    components = jax.eval_shape(
        lambda phenotype, genotype, offset, active_mask, non_active_deviance, beta, minimum_variance: (
            regenie2_binary_firth_cuda_components.compute_scalar_firth_components(
                phenotype_vector=phenotype,
                genotype_vector=genotype,
                offset_vector=offset,
                active_sample_mask=active_mask,
                non_active_deviance=non_active_deviance,
                beta=beta,
                minimum_variance=minimum_variance,
            )
        ),
        sample_operand,
        sample_operand,
        sample_operand,
        mask_operand,
        lane_operand,
        lane_operand,
        lane_operand,
    )

    assert components.genotype_information.shape == (2, 3)
    assert components.score_adjustment.shape == (2, 3)
    assert components.penalized_deviance.shape == (2, 3)
    assert components.score.shape == (2, 3)
    assert components.valid.shape == (2, 3)


def test_disabled_cuda_components_retain_jax_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the fallback independent from CUDA target registration."""

    def reject_cuda_call(**_operands: jax.Array) -> typing.NoReturn:
        raise AssertionError("disabled CUDA components must not invoke the FFI wrapper")

    monkeypatch.setattr(
        regenie2_binary_firth_cuda_components,
        "compute_scalar_firth_components",
        reject_cuda_call,
    )
    components = regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=jnp.asarray([0.0, 1.0, 1.0]),
        genotype_vector=jnp.asarray([0.25, 1.5, 0.75]),
        offset_vector=jnp.asarray([-0.2, 0.1, 0.3]),
        active_sample_mask=jnp.asarray([True, True, False]),
        non_active_deviance=jnp.asarray(0.7),
        beta=jnp.asarray(0.4),
        minimum_variance=jnp.asarray(1.0e-8),
        use_cuda_components=False,
    )

    np.testing.assert_allclose(components.genotype_information, 0.51443997, rtol=1.0e-6)
    np.testing.assert_allclose(components.score_adjustment, -0.24444907, rtol=1.0e-6)
    np.testing.assert_allclose(components.penalized_deviance, 3.45984183, rtol=1.0e-6)
    np.testing.assert_allclose(components.score, 0.13451406, rtol=1.0e-6)
    assert bool(components.valid)
