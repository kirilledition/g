from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import execution_plan
from g.compute.regenie2_binary.firth import null as regenie2_binary_firth_null
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.interface import config as interface_config

if typing.TYPE_CHECKING:
    import pytest

    from g.compute.regenie2_binary import config as regenie2_binary_config


@dataclass(frozen=True)
class StubbedNullFirthRun:
    """Observed result from a stubbed null Firth fallback run.

    Attributes:
        result: Selected null Firth fit result.
        runtime_attempts: Fallback attempt indices observed at runtime.

    """

    result: regenie2_binary_firth_types.NullFirthFitResult
    runtime_attempts: list[int]


def build_default_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build the packaged-default kernel config for tests."""
    return execution_plan.build_binary_kernel_config(interface_config.load_packaged_config().g_compute)


def run_stubbed_covariate_only_null_firth(
    monkeypatch: pytest.MonkeyPatch,
    successful_attempt: int,
) -> StubbedNullFirthRun:
    """Run the null Firth selector with observable runtime attempt callbacks."""
    traced_attempts: list[int] = []
    runtime_attempts: list[int] = []

    def record_runtime_attempt(attempt_index_array: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        attempt_index = int(np.asarray(attempt_index_array))
        runtime_attempts.append(attempt_index)
        return np.asarray(attempt_index, dtype=np.int32)

    def fit_once(
        *,
        covariate_matrix: jax.Array,
        phenotype_vector: jax.Array,
        loco_offset: jax.Array,
        initial_coefficients: jax.Array,
        maximum_iterations: int,
        maximum_step_size: float,
        tolerance: float,
        line_search_maximum_attempts: int,
        line_search_step_halving_scale: float,
        check_score_increase: bool,
    ) -> regenie2_binary_firth_types.NullFirthFitResult:
        del (
            covariate_matrix,
            phenotype_vector,
            loco_offset,
            maximum_iterations,
            maximum_step_size,
            tolerance,
            line_search_maximum_attempts,
            line_search_step_halving_scale,
            check_score_increase,
        )
        attempt_index = len(traced_attempts) + 1
        traced_attempts.append(attempt_index)
        observed_attempt_index = jax.experimental.io_callback(
            record_runtime_attempt,
            jax.ShapeDtypeStruct((), jnp.int32),
            jnp.asarray(attempt_index, dtype=jnp.int32),
            ordered=True,
        )
        attempt_value = observed_attempt_index.astype(initial_coefficients.dtype)
        return regenie2_binary_firth_types.NullFirthFitResult(
            coefficients=jnp.full(initial_coefficients.shape, attempt_value, dtype=initial_coefficients.dtype),
            penalized_log_likelihood=observed_attempt_index.astype(jnp.float64) + jnp.asarray(0.25, dtype=jnp.float64),
            iteration_count=observed_attempt_index,
            convergence_reason_code=observed_attempt_index,
            converged=observed_attempt_index == jnp.asarray(successful_attempt, dtype=jnp.int32),
        )

    monkeypatch.setattr(regenie2_binary_firth_null, "fit_covariate_only_firth_null_model_once", fit_once)
    jax.clear_caches()
    covariate_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    phenotype_vector = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    loco_offset = jnp.zeros(phenotype_vector.shape, dtype=jnp.float32)
    initial_coefficients = jnp.asarray([0.125, 0.25], dtype=jnp.float32)
    kernel_config = build_default_binary_kernel_config()

    @jax.jit
    def fit_model() -> regenie2_binary_firth_types.NullFirthFitResult:
        return regenie2_binary_firth_null.fit_covariate_only_firth_null_model(
            covariate_matrix,
            phenotype_vector,
            loco_offset,
            initial_coefficients,
            kernel_config,
        )

    result = fit_model()
    jax.block_until_ready(result.coefficients)
    return StubbedNullFirthRun(result=result, runtime_attempts=runtime_attempts)


def test_covariate_only_null_firth_skips_fallbacks_after_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_stubbed_covariate_only_null_firth(monkeypatch, successful_attempt=1)

    assert run.runtime_attempts == [1]
    np.testing.assert_allclose(np.asarray(run.result.coefficients), np.asarray([1.0, 1.0]))
    assert float(np.asarray(run.result.penalized_log_likelihood)) == 1.25
    assert int(np.asarray(run.result.iteration_count)) == 1
    assert bool(np.asarray(run.result.converged))


def test_covariate_only_null_firth_runs_second_attempt_only_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_stubbed_covariate_only_null_firth(monkeypatch, successful_attempt=2)

    assert run.runtime_attempts == [1, 2]
    np.testing.assert_allclose(np.asarray(run.result.coefficients), np.asarray([2.0, 2.0]))
    assert float(np.asarray(run.result.penalized_log_likelihood)) == 2.25
    assert int(np.asarray(run.result.iteration_count)) == 2
    assert bool(np.asarray(run.result.converged))


def test_covariate_only_null_firth_uses_fourth_attempt_when_all_earlier_attempts_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_stubbed_covariate_only_null_firth(monkeypatch, successful_attempt=0)

    assert run.runtime_attempts == [1, 2, 3, 4]
    np.testing.assert_allclose(np.asarray(run.result.coefficients), np.asarray([4.0, 4.0]))
    assert np.isnan(np.asarray(run.result.penalized_log_likelihood))
    assert int(np.asarray(run.result.iteration_count)) == 4
    assert not bool(np.asarray(run.result.converged))
