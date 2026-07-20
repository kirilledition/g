"""CPU-safe contracts around the optional raw-CUDA Firth component path."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np

import tests.numerical
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


def test_disabled_cuda_components_retain_independent_jax_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU correctness independent from CUDA target registration."""

    def reject_cuda_call(**operands: jax.Array) -> typing.NoReturn:
        del operands
        raise AssertionError("disabled CUDA components must not invoke the FFI wrapper")

    monkeypatch.setattr(
        regenie2_binary_firth_cuda_components,
        "compute_scalar_firth_components",
        reject_cuda_call,
    )
    phenotype = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    genotype = np.asarray([0.25, 1.5, 0.75], dtype=np.float64)
    offset = np.asarray([-0.2, 0.1, 0.3], dtype=np.float64)
    active_mask = np.asarray([True, True, False])
    beta = 0.4
    non_active_deviance = 0.7

    components = regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=jnp.asarray(phenotype),
        genotype_vector=jnp.asarray(genotype),
        offset_vector=jnp.asarray(offset),
        active_sample_mask=jnp.asarray(active_mask),
        non_active_deviance=jnp.asarray(non_active_deviance),
        beta=jnp.asarray(beta),
        minimum_variance=jnp.asarray(1.0e-8),
        use_cuda_components=False,
    )

    probability = np.reciprocal(1.0 + np.exp(-(offset + genotype * beta)))
    weight = probability * (1.0 - probability)
    information_diagonal = genotype**2 * np.where(active_mask, weight, 0.0)
    genotype_information = float(np.sum(information_diagonal))
    score_adjustment = float(
        np.sum(np.where(active_mask, genotype * information_diagonal * (0.5 - probability), 0.0)) / genotype_information
    )
    negative_log_likelihood = -np.where(phenotype > 0.5, np.log(probability), np.log1p(-probability))
    penalized_deviance = (
        non_active_deviance
        + 2.0 * np.sum(np.where(active_mask, negative_log_likelihood, 0.0))
        - np.log(genotype_information)
    )
    score = float(np.sum(np.where(active_mask, genotype * (phenotype - probability), 0.0)) + score_adjustment)

    tests.numerical.assert_absolute_difference_less_than(
        components.genotype_information,
        genotype_information,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(components.score_adjustment, score_adjustment, 1.0e-12)
    tests.numerical.assert_absolute_difference_less_than(
        components.penalized_deviance,
        penalized_deviance,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(components.score, score, 1.0e-12)
    assert bool(np.asarray(components.valid))
