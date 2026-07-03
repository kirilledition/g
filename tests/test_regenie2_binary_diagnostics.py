from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from g import types
from g.compute.regenie2_binary import diagnostics as regenie2_binary_diagnostics
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.engine.callbacks import diagnostics as callback_diagnostics


def build_binary_chunk_result(
    *,
    extra_code: jax.Array,
    firth_iteration_count: jax.Array,
    firth_failure_code: jax.Array,
    firth_correction_code: jax.Array | None = None,
    firth_sparse_correction_mask: jax.Array | None = None,
    pseudo_firth_iteration_count: jax.Array | None = None,
    nr_zero_start_iteration_count: jax.Array | None = None,
    nr_warm_start_iteration_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2BinaryChunkResult:
    variant_count = extra_code.shape[0]
    zeros = jnp.zeros(variant_count, dtype=jnp.float32)
    zero_integers = jnp.zeros(variant_count, dtype=jnp.int32)
    return regenie2_binary_result.Regenie2BinaryChunkResult(
        beta=zeros,
        standard_error=zeros,
        chi_squared=zeros,
        log10_p_value=zeros,
        extra_code=extra_code,
        valid_mask=jnp.ones(variant_count, dtype=jnp.bool_),
        firth_iteration_count=firth_iteration_count,
        firth_failure_code=firth_failure_code,
        firth_convergence_reason_code=jnp.zeros(variant_count, dtype=jnp.int32),
        firth_correction_code=zero_integers if firth_correction_code is None else firth_correction_code,
        firth_sparse_correction_mask=(
            jnp.zeros(variant_count, dtype=jnp.bool_)
            if firth_sparse_correction_mask is None
            else firth_sparse_correction_mask
        ),
        pseudo_firth_iteration_count=(
            zero_integers if pseudo_firth_iteration_count is None else pseudo_firth_iteration_count
        ),
        nr_zero_start_iteration_count=(
            zero_integers if nr_zero_start_iteration_count is None else nr_zero_start_iteration_count
        ),
        nr_warm_start_iteration_count=(
            zero_integers if nr_warm_start_iteration_count is None else nr_warm_start_iteration_count
        ),
    )


def test_binary_chunk_diagnostics_report_zeroes_without_firth_candidates() -> None:
    result = build_binary_chunk_result(
        extra_code=jnp.asarray([types.BinaryExtraCode.SCORE.value, types.BinaryExtraCode.SCORE.value]),
        firth_iteration_count=jnp.asarray([0, 0], dtype=jnp.int32),
        firth_failure_code=jnp.asarray([types.FirthFailureCode.NONE.value, types.FirthFailureCode.NONE.value]),
    )

    diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics(result)

    assert int(diagnostics.score_only_count) == 2
    assert int(diagnostics.score_test_candidate_count) == 0
    assert int(diagnostics.firth_candidate_count) == 0
    assert int(diagnostics.firth_iteration_min) == 0
    assert int(diagnostics.firth_iteration_median) == 0
    assert int(diagnostics.firth_iteration_max) == 0
    assert int(diagnostics.firth_converged_count) == 0
    assert int(diagnostics.firth_failed_count) == 0
    assert int(diagnostics.pseudo_firth_attempt_count) == 0
    assert int(diagnostics.nr_zero_start_attempt_count) == 0
    assert int(diagnostics.nr_warm_start_attempt_count) == 0


def test_binary_chunk_diagnostics_accept_score_result_without_firth_arrays() -> None:
    result = regenie2_binary_result.Regenie2BinaryScoreChunkResult(
        beta=jnp.zeros(3, dtype=jnp.float32),
        standard_error=jnp.ones(3, dtype=jnp.float32),
        chi_squared=jnp.zeros(3, dtype=jnp.float32),
        log10_p_value=jnp.zeros(3, dtype=jnp.float32),
        extra_code=jnp.asarray(
            [
                types.BinaryExtraCode.SCORE.value,
                types.BinaryExtraCode.FIRTH.value,
                types.BinaryExtraCode.TEST_FAIL.value,
            ],
            dtype=jnp.int32,
        ),
        valid_mask=jnp.asarray([True, True, False], dtype=jnp.bool_),
    )

    diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics(result)

    assert int(diagnostics.score_only_count) == 1
    assert int(diagnostics.score_test_candidate_count) == 2
    assert int(diagnostics.firth_candidate_count) == 0
    assert int(diagnostics.firth_converged_count) == 0
    assert int(diagnostics.firth_failed_count) == 0
    assert int(diagnostics.pseudo_firth_attempt_count) == 0
    assert int(diagnostics.nr_zero_start_attempt_count) == 0
    assert int(diagnostics.nr_warm_start_attempt_count) == 0


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
        firth_correction_code=jnp.asarray(
            [
                types.FirthCorrectionCode.PSEUDO_FIRTH.value,
                types.FirthCorrectionCode.NONE.value,
                types.FirthCorrectionCode.NONE.value,
                types.FirthCorrectionCode.NONE.value,
                types.FirthCorrectionCode.NONE.value,
                types.FirthCorrectionCode.NEWTON_RAPHSON_ZERO_START.value,
            ],
            dtype=jnp.int32,
        ),
        firth_sparse_correction_mask=jnp.asarray([False, True, False, False, True, False], dtype=jnp.bool_),
        pseudo_firth_iteration_count=jnp.asarray([3, 2, 2, 1, 4, 0], dtype=jnp.int32),
        nr_zero_start_iteration_count=jnp.asarray([0, 0, 3, 0, 0, 2], dtype=jnp.int32),
        nr_warm_start_iteration_count=jnp.asarray([0, 1, 0, 0, 2, 0], dtype=jnp.int32),
    )

    diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics(result)

    assert int(diagnostics.score_only_count) == 0
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
    assert int(diagnostics.pseudo_firth_attempt_count) == 5
    assert int(diagnostics.pseudo_firth_success_count) == 1
    assert int(diagnostics.nr_zero_start_attempt_count) == 2
    assert int(diagnostics.nr_zero_start_success_count) == 1
    assert int(diagnostics.nr_warm_start_attempt_count) == 2
    assert int(diagnostics.nr_warm_start_success_count) == 0
    assert int(diagnostics.sparse_correction_count) == 2
    assert int(diagnostics.dense_correction_count) == 3

    diagnostics_mapping = callback_diagnostics.binary_chunk_diagnostics_to_mapping(diagnostics)
    assert diagnostics_mapping["score_only_count"] == 0
    assert diagnostics_mapping["firth_candidate_count"] == 5
    assert diagnostics_mapping["firth_failed_count"] == 4
