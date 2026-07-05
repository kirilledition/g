"""Public binary REGENIE step 2 compute API."""

from __future__ import annotations

import functools
import time

import jax

from g import types as g_types
from g.compute.common import genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import diagnostics as regenie2_binary_diagnostics
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.variant_major_correction import dispatch as regenie2_binary_variant_major_dispatch
from g.compute.regenie2_binary.variant_major_correction import public as regenie2_binary_variant_major_correction

BinaryChunkDiagnostics = regenie2_binary_diagnostics.BinaryChunkDiagnostics
BinaryCorrectionSummaryCounts = regenie2_binary_diagnostics.BinaryCorrectionSummaryCounts
Regenie2BinaryState = regenie2_binary_state.Regenie2BinaryState
Regenie2BinaryChromosomeState = regenie2_binary_state.Regenie2BinaryChromosomeState
Regenie2MultiBinaryState = regenie2_binary_state.Regenie2MultiBinaryState
Regenie2MultiBinaryChromosomeState = regenie2_binary_state.Regenie2MultiBinaryChromosomeState
Regenie2BinaryScoreChunkResult = regenie2_binary_result.Regenie2BinaryScoreChunkResult
Regenie2BinaryChunkResult = regenie2_binary_result.Regenie2BinaryChunkResult
Regenie2MultiBinaryScoreChunkResult = regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult
Regenie2MultiBinaryChunkResult = regenie2_binary_result.Regenie2MultiBinaryChunkResult
StageDurationRecorder = regenie2_binary_variant_major_correction.StageDurationRecorder
count_binary_chunk_diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics


def prepare_regenie2_binary_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_state.Regenie2BinaryState:
    """Prepare reusable binary step 2 state."""
    return regenie2_binary_state.build_binary_state(covariate_matrix, phenotype_vector, score_dtype)


def prepare_regenie2_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_state.Regenie2MultiBinaryState:
    """Prepare reusable multi-trait binary step 2 state."""
    return regenie2_binary_state.build_multi_binary_state(covariate_matrix, phenotype_matrix, score_dtype)


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "score_dtype"))
def prepare_regenie2_binary_chromosome_state(
    state: regenie2_binary_state.Regenie2BinaryState,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_state.Regenie2BinaryChromosomeState:
    """Prepare chromosome-specific null logistic state reused across chunks."""
    return regenie2_binary_state.build_binary_chromosome_state(
        state,
        loco_offset,
        correction_plan,
        kernel_config,
        score_dtype,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "score_dtype"))
def prepare_regenie2_multi_binary_chromosome_state(
    state: regenie2_binary_state.Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_state.Regenie2MultiBinaryChromosomeState:
    """Prepare chromosome-specific null logistic state for all requested binary traits."""
    return regenie2_binary_state.build_multi_binary_chromosome_state(
        state,
        loco_offset_matrix,
        correction_plan,
        kernel_config,
        score_dtype,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "score_dtype"))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Compute the uncorrected score-test result for one variant-major binary chunk."""
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
    donate_argnames=("genotype_matrix_by_variant", "dosage_sum", "observation_count"),
)
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major_donating_inputs(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Compute score-only binary statistics while donating one-shot chunk inputs."""
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
    donate_argnames=("genotype_matrix_by_variant", "dosage_sum", "observation_count"),
)
def compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_variant_major_donating_inputs(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Compute multi-trait score-only binary statistics while donating one-shot chunk inputs."""
    return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


compute_binary_score_test_variant_major_donating_inputs = (
    compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major_donating_inputs
)
compute_multi_binary_score_test_variant_major_donating_inputs = (
    compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_variant_major_donating_inputs
)


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
)
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_packed8(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Decode packed8 probabilities on device and compute binary score statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
    donate_argnames=("packed_probability_pairs_by_variant", "dosage_sum", "observation_count"),
)
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Decode packed8 probabilities on device and compute score-only binary statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


compute_binary_score_test_packed8_donating_inputs = (
    compute_regenie2_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs
)


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
)
def compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_packed8(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Decode packed8 probabilities on device and compute multi-trait score statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
    donate_argnames=("packed_probability_pairs_by_variant", "dosage_sum", "observation_count"),
)
def compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult:
    """Decode packed8 probabilities on device and compute multi-trait score-only binary statistics."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


compute_multi_binary_score_test_packed8_donating_inputs = (
    compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs
)


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "score_dtype",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
        "overflow_candidate_capacity",
    ),
)
def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major_no_overflow(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    overflow_candidate_capacity: int,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute a multi-trait binary chunk in one executable when overflow is impossible."""
    score_test_result = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    correction_module = regenie2_binary_variant_major_dispatch
    return correction_module.apply_device_candidate_corrections_multi_firth_variant_major_with_device_dispatch(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        tiny_candidate_capacity=tiny_candidate_capacity,
        small_candidate_capacity=small_candidate_capacity,
        bounded_candidate_capacity=bounded_candidate_capacity,
        overflow_candidate_capacity=overflow_candidate_capacity,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        kernel_config=kernel_config,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "correction_plan",
        "kernel_config",
        "score_dtype",
        "tiny_candidate_capacity",
        "small_candidate_capacity",
        "bounded_candidate_capacity",
        "overflow_candidate_capacity",
    ),
)
def compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8_no_overflow(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    tiny_candidate_capacity: int,
    small_candidate_capacity: int,
    bounded_candidate_capacity: int,
    overflow_candidate_capacity: int,
) -> regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute a packed8 multi-trait binary chunk in one executable when overflow is impossible."""
    genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    score_test_result = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    correction_module = regenie2_binary_variant_major_dispatch
    return correction_module.apply_device_candidate_corrections_multi_firth_packed8_with_device_dispatch(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        tiny_candidate_capacity=tiny_candidate_capacity,
        small_candidate_capacity=small_candidate_capacity,
        bounded_candidate_capacity=bounded_candidate_capacity,
        overflow_candidate_capacity=overflow_candidate_capacity,
        sparse_candidate_mask=sparse_candidate_mask,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
        kernel_config=kernel_config,
    )


def build_multi_binary_capacity_plan_for_chunk(
    *,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    variant_count: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_candidate_planning.FirthCandidateCapacityPlan:
    """Build the static multi-trait Firth capacity plan for one chunk shape."""
    return regenie2_binary_candidate_planning.build_multi_firth_candidate_capacity_plan(
        trait_count=chromosome_state.phenotype_matrix.shape[0],
        variant_count=variant_count,
        preferred_candidate_capacity=kernel_config.firth_candidate.candidate_capacity,
    )


def can_use_multi_binary_no_overflow_entrypoint(
    capacity_plan: regenie2_binary_candidate_planning.FirthCandidateCapacityPlan,
) -> bool:
    """Return whether bounded capacity covers every flattened trait-variant lane."""
    return capacity_plan.bounded_candidate_capacity == capacity_plan.overflow_candidate_capacity


def record_firth_candidate_dispatch_plan_duration(
    stage_duration_recorder: StageDurationRecorder | None,
    start_time: float,
) -> None:
    """Record candidate-dispatch planning time when profiling is enabled."""
    if stage_duration_recorder is not None:
        stage_duration_recorder("firth_candidate_dispatch_plan", start_time)


def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    stage_duration_recorder: StageDurationRecorder | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary REGENIE step 2 association using one genotype chunk."""
    return compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix, score_dtype),
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        score_dtype=score_dtype,
        stage_duration_recorder=stage_duration_recorder,
        dosage_sum=None,
        observation_count=None,
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    stage_duration_recorder: StageDurationRecorder | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes.

    Multi-binary score-only and approximate Firth execution share one batched
    score result per chunk. Approximate Firth correction then runs only the
    selected flattened trait-variant candidate lanes.

    """
    if correction_plan.method != g_types.BinaryFallbackMethod.SCORE_ONLY:
        capacity_plan_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
        capacity_plan = build_multi_binary_capacity_plan_for_chunk(
            chromosome_state=chromosome_state,
            variant_count=genotype_matrix_by_variant.shape[0],
            kernel_config=kernel_config,
        )
        if can_use_multi_binary_no_overflow_entrypoint(capacity_plan):
            record_firth_candidate_dispatch_plan_duration(stage_duration_recorder, capacity_plan_start_time)
            return compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major_no_overflow(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                correction_plan=correction_plan,
                sparse_candidate_mask=sparse_candidate_mask,
                kernel_config=kernel_config,
                score_dtype=score_dtype,
                dosage_sum=dosage_sum,
                observation_count=observation_count,
                tiny_candidate_capacity=capacity_plan.tiny_candidate_capacity,
                small_candidate_capacity=capacity_plan.small_candidate_capacity,
                bounded_candidate_capacity=capacity_plan.bounded_candidate_capacity,
                overflow_candidate_capacity=capacity_plan.overflow_candidate_capacity,
            )
    score_test_result = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_multi_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        stage_duration_recorder=stage_duration_recorder,
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    stage_duration_recorder: StageDurationRecorder | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from packed8 BGEN probability pairs."""
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            score_dtype=score_dtype,
        )
    capacity_plan_start_time = time.perf_counter() if stage_duration_recorder is not None else 0.0
    capacity_plan = build_multi_binary_capacity_plan_for_chunk(
        chromosome_state=chromosome_state,
        variant_count=packed_probability_pairs_by_variant.shape[0],
        kernel_config=kernel_config,
    )
    if can_use_multi_binary_no_overflow_entrypoint(capacity_plan):
        record_firth_candidate_dispatch_plan_duration(stage_duration_recorder, capacity_plan_start_time)
        return compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8_no_overflow(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
            score_dtype=score_dtype,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            tiny_candidate_capacity=capacity_plan.tiny_candidate_capacity,
            small_candidate_capacity=capacity_plan.small_candidate_capacity,
            bounded_candidate_capacity=capacity_plan.bounded_candidate_capacity,
            overflow_candidate_capacity=capacity_plan.overflow_candidate_capacity,
        )
    score_test_result = compute_regenie2_multi_binary_score_test_chunk_from_chromosome_state_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_multi_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        stage_duration_recorder=stage_duration_recorder,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


def compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    stage_duration_recorder: StageDurationRecorder | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute binary association from a variant-major chunk."""
    score_test_result = compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        stage_duration_recorder=stage_duration_recorder,
    )


def compute_regenie2_binary_chunk_from_chromosome_state_packed8(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    stage_duration_recorder: StageDurationRecorder | None,
    dosage_sum: jax.Array | None,
    observation_count: jax.Array | None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute binary association from packed8 BGEN probability pairs.

    Score-test and candidate correction work remains on the JAX device. The
    packed probability rows are decoded to dosage on device before reusing the
    canonical variant-major score and approximate-Firth kernels.
    """
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return compute_regenie2_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            score_dtype=score_dtype,
        )
    score_test_result = compute_regenie2_binary_score_test_chunk_from_chromosome_state_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_packed8(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        stage_duration_recorder=stage_duration_recorder,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
        score_dtype=score_dtype,
    )


def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    sparse_candidate_mask: jax.Array | None,
    score_dtype: g_types.FloatingPointDtype,
    stage_duration_recorder: StageDurationRecorder | None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association using cached null state."""
    return compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix, score_dtype),
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        score_dtype=score_dtype,
        stage_duration_recorder=stage_duration_recorder,
        dosage_sum=None,
        observation_count=None,
    )
