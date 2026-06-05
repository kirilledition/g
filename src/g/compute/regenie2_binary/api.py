"""Public binary REGENIE step 2 compute API."""

from __future__ import annotations

import functools
import typing

import jax

from g import types as g_types
from g.compute.common import genotype
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import diagnostics as regenie2_binary_diagnostics
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary import variant_major_correction as regenie2_binary_variant_major_correction

BinaryChunkDiagnostics = regenie2_binary_diagnostics.BinaryChunkDiagnostics
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
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_state.Regenie2BinaryState:
    """Prepare reusable binary step 2 state."""
    return regenie2_binary_state.build_binary_state(covariate_matrix, phenotype_vector, score_dtype)


def prepare_regenie2_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_state.Regenie2MultiBinaryState:
    """Prepare reusable multi-trait binary step 2 state."""
    return regenie2_binary_state.build_multi_binary_state(covariate_matrix, phenotype_matrix, score_dtype)


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "score_dtype"))
def prepare_regenie2_binary_chromosome_state(
    state: regenie2_binary_state.Regenie2BinaryState,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
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
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
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
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Compute the uncorrected score-test result for one binary chunk."""
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix, score_dtype),
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        score_dtype=score_dtype,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config", "score_dtype"))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
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
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
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
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
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
    static_argnames=("score_dtype",),
    donate_argnames=("packed_probability_pairs_by_variant",),
)
def decode_packed8_probability_pairs_to_variant_major_dosage_donating_input(
    packed_probability_pairs_by_variant: jax.Array,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> jax.Array:
    """Decode packed8 probability pairs while donating the packed input buffer."""
    return genotype.decode_packed8_probability_pairs_to_variant_major_dosage(
        packed_probability_pairs_by_variant,
        score_dtype,
    )


@functools.partial(
    jax.jit,
    static_argnames=("correction_plan", "kernel_config", "score_dtype"),
    donate_argnames=("packed_probability_pairs_by_variant", "dosage_sum", "observation_count"),
)
def compute_regenie2_binary_score_test_chunk_from_chromosome_state_packed8_donating_inputs(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
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


def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
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
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes.

    Multi-binary score-only execution is true batched GPU compute. Multi-binary approximate Firth currently shares
    the genotype transfer, then applies correction one trait at a time in Python and should be benchmarked separately.

    """
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
            score_dtype=score_dtype,
        )

    def compute_one_trait(trait_index: int) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        single_chromosome_state = regenie2_binary_state.build_single_binary_chromosome_state_from_multi(
            chromosome_state,
            trait_index,
        )
        result = compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
            chromosome_state=single_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
            score_dtype=score_dtype,
            stage_duration_recorder=stage_duration_recorder,
            dosage_sum=dosage_sum,
            observation_count=observation_count,
        )
        return typing.cast("regenie2_binary_result.Regenie2BinaryChunkResult", result)

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    # Approximate Firth is not yet a fully batched multi-trait correction workload:
    # it reuses the variant-major genotype chunk, then dispatches each trait through
    # the single-trait correction path before stacking the results.
    return regenie2_binary_result.stack_binary_chunk_results(
        [compute_one_trait(trait_index) for trait_index in range(trait_count)]
    )


def compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
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
        stage_duration_recorder=stage_duration_recorder,
    )


def compute_regenie2_binary_chunk_from_chromosome_state_packed8(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    packed_probability_pairs_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
    dosage_sum: jax.Array | None = None,
    observation_count: jax.Array | None = None,
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
    genotype_matrix_by_variant = decode_packed8_probability_pairs_to_variant_major_dosage_donating_input(
        packed_probability_pairs_by_variant,
        score_dtype,
    )
    return compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        score_dtype=score_dtype,
        stage_duration_recorder=stage_duration_recorder,
        dosage_sum=dosage_sum,
        observation_count=observation_count,
    )


def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
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
    )


def compute_regenie2_binary_chunk(
    state: regenie2_binary_state.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
    stage_duration_recorder: StageDurationRecorder | None = None,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = prepare_regenie2_binary_chromosome_state(
        state,
        loco_offset,
        correction_plan,
        kernel_config,
        score_dtype,
    )
    return compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
        score_dtype=score_dtype,
        stage_duration_recorder=stage_duration_recorder,
    )
