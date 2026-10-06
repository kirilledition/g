"""Regressions for correction-only JIT specialization."""

from __future__ import annotations

import dataclasses
import typing

import jax
import numpy as np

import tests.test_regenie2_binary_pipeline as firth_pipeline
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary.firth import scalar_approx

if typing.TYPE_CHECKING:
    import pytest

    from g.compute.regenie2_binary import result as regenie2_binary_result
    from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


def test_null_fit_policy_does_not_recompile_chunk_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse a correction executable while retaining its real solver settings."""
    prepared = firth_pipeline.build_prepared_firth_pipeline()
    kernel_config = dataclasses.replace(
        prepared.kernel_config,
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(candidate_capacity=5, batch_size=3),
    )
    original_parameters_builder = scalar_approx.build_scalar_approximate_firth_solver_parameters
    observed_chunk_policies: list[regenie2_binary_config.BinaryChunkConfig] = []

    def record_traced_policy(
        traced_config: regenie2_binary_config.BinaryKernelConfig | regenie2_binary_config.BinaryChunkConfig,
    ) -> regenie2_binary_firth_types.ScalarApproximateFirthSolverParameters:
        assert isinstance(traced_config, regenie2_binary_config.BinaryChunkConfig)
        observed_chunk_policies.append(traced_config)
        return original_parameters_builder(traced_config)

    monkeypatch.setattr(scalar_approx, "build_scalar_approximate_firth_solver_parameters", record_traced_policy)

    def compute_chunk(
        selected_config: regenie2_binary_config.BinaryKernelConfig,
    ) -> regenie2_binary_result.CorrectedMultiBinaryScoreChunkResult:
        return jax.block_until_ready(
            firth_pipeline.run_production_firth_pipeline(
                prepared=prepared,
                firth_se=False,
                p_threshold=1.0,
                kernel_config=selected_config,
                chromosome_state=prepared.chromosome_state,
            )
        )

    expected = compute_chunk(kernel_config)
    first_trace_count = len(observed_chunk_policies)
    assert first_trace_count > 0
    unrelated_null_config = dataclasses.replace(
        kernel_config,
        null_logistic=dataclasses.replace(
            kernel_config.null_logistic,
            maximum_iterations=kernel_config.null_logistic.maximum_iterations + 1,
            coefficient_tolerance=kernel_config.null_logistic.coefficient_tolerance / 2.0,
        ),
        null_firth=dataclasses.replace(
            kernel_config.null_firth,
            maximum_iterations=kernel_config.null_firth.maximum_iterations + 1,
            maximum_step_size=kernel_config.null_firth.maximum_step_size / 2.0,
        ),
    )
    observed = compute_chunk(unrelated_null_config)
    assert len(observed_chunk_policies) == first_trace_count
    for expected_values, observed_values in zip(jax.tree.leaves(expected), jax.tree.leaves(observed), strict=True):
        np.testing.assert_array_equal(np.asarray(observed_values), np.asarray(expected_values))

    compute_chunk(
        dataclasses.replace(
            kernel_config,
            approximate_firth=dataclasses.replace(
                kernel_config.approximate_firth,
                maximum_step_size=kernel_config.approximate_firth.maximum_step_size / 2.0,
            ),
        )
    )
    assert len(observed_chunk_policies) > first_trace_count
