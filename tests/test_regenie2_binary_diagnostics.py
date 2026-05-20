from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from g import types
from g.compute import regenie2_binary_diagnostics, regenie2_binary_types


def build_binary_chunk_result(
    *,
    extra_code: jax.Array,
    firth_iteration_count: jax.Array,
    firth_failure_code: jax.Array,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    variant_count = extra_code.shape[0]
    zeros = jnp.zeros(variant_count, dtype=jnp.float32)
    return regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=zeros,
        standard_error=zeros,
        chi_squared=zeros,
        log10_p_value=zeros,
        extra_code=extra_code,
        valid_mask=jnp.ones(variant_count, dtype=jnp.bool_),
        firth_iteration_count=firth_iteration_count,
        firth_failure_code=firth_failure_code,
        firth_convergence_reason_code=jnp.zeros(variant_count, dtype=jnp.int32),
    )


def test_binary_chunk_diagnostics_report_zeroes_without_firth_candidates() -> None:
    result = build_binary_chunk_result(
        extra_code=jnp.asarray([types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value]),
        firth_iteration_count=jnp.asarray([0, 0], dtype=jnp.int32),
        firth_failure_code=jnp.asarray([types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value]),
    )

    diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics(result)

    assert int(diagnostics.score_test_candidate_count) == 0
    assert int(diagnostics.firth_candidate_count) == 0
    assert int(diagnostics.firth_iteration_min) == 0
    assert int(diagnostics.firth_iteration_median) == 0
    assert int(diagnostics.firth_iteration_max) == 0
    assert int(diagnostics.firth_converged_count) == 0
    assert int(diagnostics.firth_failed_count) == 0


def test_binary_chunk_diagnostics_count_all_failure_categories() -> None:
    result = build_binary_chunk_result(
        extra_code=jnp.asarray(
            [
                types.BinaryExtraCode.FIRTH.value,
                types.BinaryExtraCode.TEST_FAIL.value,
                types.BinaryExtraCode.TEST_FAIL.value,
                types.BinaryExtraCode.TEST_FAIL.value,
                types.BinaryExtraCode.TEST_FAIL.value,
                types.BinaryExtraCode.SPA.value,
            ],
            dtype=jnp.int32,
        ),
        firth_iteration_count=jnp.asarray([5, 4, 7, 2, 9, 0], dtype=jnp.int32),
        firth_failure_code=jnp.asarray(
            [
                types.FirthFailureCode.NONE.value,
                types.FirthFailureCode.NUMERICAL.value,
                types.FirthFailureCode.MAX_ITERATIONS.value,
                types.FirthFailureCode.INVALID_STATISTIC.value,
                types.FirthFailureCode.STEP_HALVING.value,
                types.FirthFailureCode.NONE.value,
            ],
            dtype=jnp.int32,
        ),
    )

    diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics(result)

    assert int(diagnostics.score_test_candidate_count) == 6
    assert int(diagnostics.firth_candidate_count) == 5
    assert int(diagnostics.firth_iteration_min) == 2
    assert int(diagnostics.firth_iteration_median) == 5
    assert int(diagnostics.firth_iteration_max) == 9
    assert int(diagnostics.firth_converged_count) == 1
    assert int(diagnostics.firth_failed_count) == 4
    np.testing.assert_array_equal(
        np.asarray(
            [
                diagnostics.firth_numerical_failure_count,
                diagnostics.firth_max_iteration_failure_count,
                diagnostics.firth_invalid_statistic_failure_count,
                diagnostics.firth_step_halving_failure_count,
            ]
        ),
        np.asarray([1, 1, 1, 1]),
    )
