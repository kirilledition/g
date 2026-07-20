"""Production-path correctness tests for approximate-Firth dispatch."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
from g import jax_backend, types
from g.compute.common import result as association_result
from g.compute.regenie2_binary import api as regenie2_binary_api
from g.compute.regenie2_binary import candidates as regenie2_binary_candidates
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.compute.regenie2_binary.firth.batch import prepare as regenie2_binary_firth_prepare
from g.compute.regenie2_binary.variant_major_correction import fixed_capacity

# The production solver intentionally stops at a 2.5e-4 adjusted-score
# tolerance. These exclusive bounds are approximately twice the measured
# maxima against the independent NumPy oracle (7.85e-4, 5.70e-4, 4.14e-7, and
# 1.23e-7), preserving that policy while detecting statistic-specific drift.
FIRTH_BETA_ABSOLUTE_TOLERANCE = 1.6e-3
FIRTH_STANDARD_ERROR_ABSOLUTE_TOLERANCE = 1.2e-3
FIRTH_CHI_SQUARED_ABSOLUTE_TOLERANCE = 9.0e-7
FIRTH_LOG10_P_VALUE_ABSOLUTE_TOLERANCE = 3.0e-7
# The API computes the LRT standard error before float32 result materialization;
# recomputing from materialized beta and chi-square differs by at most one
# observed float32 ulp (1.20e-7), so retain two ulps of headroom.
FIRTH_SE_TRANSFORMATION_ABSOLUTE_TOLERANCE = 2.5e-7


@dataclass(frozen=True)
class FirthPipelineFixture:
    """Small binary inputs containing one dense and one sparse candidate."""

    covariate_matrix: npt.NDArray[np.float64]
    phenotype_matrix: npt.NDArray[np.float64]
    loco_offset_matrix: npt.NDArray[np.float64]
    genotype_matrix_by_variant: npt.NDArray[np.float64]
    sparse_candidate_mask: npt.NDArray[np.bool_]
    native_genotype_mean: npt.NDArray[np.float64] | None


@dataclass(frozen=True)
class IndependentNullLogisticState:
    """Independent null-logistic quantities required by Firth preparation."""

    coefficients: npt.NDArray[np.float64]
    square_root_weight: npt.NDArray[np.float64]
    weighted_genotype_projection_matrix: npt.NDArray[np.float64]


@dataclass(frozen=True)
class IndependentNullFirthComponents:
    """Independent covariate-only Firth objective and Newton operands."""

    information_matrix: npt.NDArray[np.float64]
    deviance: float
    modified_score: npt.NDArray[np.float64]


@dataclass(frozen=True)
class IndependentFirthTraitState:
    """Independently prepared state consumed by one scalar Firth oracle."""

    square_root_weight: npt.NDArray[np.float64]
    weighted_genotype_projection_matrix: npt.NDArray[np.float64]
    null_firth_offset: npt.NDArray[np.float64]


@dataclass(frozen=True)
class PreparedFirthPipeline:
    """Production state and policy shared by pipeline tests."""

    fixture: FirthPipelineFixture
    kernel_config: regenie2_binary_config.BinaryKernelConfig
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState
    independent_trait_states: tuple[IndependentFirthTraitState, ...]


@dataclass(frozen=True)
class SuccessfulFirthPipeline:
    """Successful production results under both standard-error policies."""

    prepared: PreparedFirthPipeline
    information_standard_error_result: regenie2_binary_result.CorrectedMultiBinaryScoreChunkResult
    likelihood_ratio_standard_error_result: regenie2_binary_result.CorrectedMultiBinaryScoreChunkResult


@dataclass(frozen=True)
class ScalarFirthOracleComponents:
    """Independent adjusted-score quantities at one candidate coefficient."""

    genotype_information: float
    penalized_deviance: float
    score: float


@dataclass(frozen=True)
class FirthReferenceResult:
    """Independent scalar approximate-Firth association result."""

    beta: float
    standard_error: float
    chi_squared: float
    log10_p_value: float


def build_firth_pipeline_fixture() -> FirthPipelineFixture:
    """Build well-conditioned inputs with a carrier-only second variant."""
    return FirthPipelineFixture(
        covariate_matrix=np.asarray(
            [
                [1.0, -1.5],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
                [1.0, 1.5],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        phenotype_matrix=np.asarray(
            [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        loco_offset_matrix=np.asarray(
            [[0.04, -0.03, 0.01, 0.0, -0.02, 0.03, -0.01, 0.02]],
            dtype=np.float64,
        ),
        genotype_matrix_by_variant=np.asarray(
            [
                [0.2, 1.4, 0.3, 1.6, 0.4, 1.5, 1.3, 0.7],
                [0.0, 1.0, 0.0, 1.5, 0.0, 0.8, 0.0, 1.2],
            ],
            dtype=np.float64,
        ),
        sparse_candidate_mask=np.asarray([False, True], dtype=np.bool_),
        native_genotype_mean=None,
    )


def build_packed_firth_pipeline_fixture() -> FirthPipelineFixture:
    """Build exact packed8 inputs spanning natural and native-statistic flips."""
    fixture = build_firth_pipeline_fixture()
    genotype_matrix_by_variant = np.asarray(
        [
            [2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    return dataclasses.replace(
        fixture,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        native_genotype_mean=np.asarray(
            [
                np.mean(genotype_matrix_by_variant[0]),
                float(np.nextafter(np.float32(1.0), np.float32(2.0))),
            ],
            dtype=np.float64,
        ),
    )


def build_reordered_multi_trait_fixture() -> FirthPipelineFixture:
    """Build two traits whose candidate lanes require production bucketing."""
    fixture = build_firth_pipeline_fixture()
    return dataclasses.replace(
        fixture,
        phenotype_matrix=np.asarray(
            [
                fixture.phenotype_matrix[0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        loco_offset_matrix=np.asarray(
            [
                fixture.loco_offset_matrix[0],
                [-0.02, 0.01, 0.03, -0.04, 0.05, -0.01, 0.00, 0.02],
            ],
            dtype=np.float64,
        ),
        genotype_matrix_by_variant=np.asarray(
            [
                fixture.genotype_matrix_by_variant[0],
                fixture.genotype_matrix_by_variant[1],
                [1.8, 1.6, 0.9, 1.7, 1.1, 0.4, 1.5, 0.8],
            ],
            dtype=np.float64,
        ),
        sparse_candidate_mask=np.asarray([False, True, False], dtype=np.bool_),
        native_genotype_mean=None,
    )


def build_over_capacity_sparse_fixture() -> FirthPipelineFixture:
    """Build a legitimate sparse lane with 65 carriers, above compact capacity."""
    sample_count = 72
    carrier_start_index = 7
    sample_indices = np.arange(sample_count, dtype=np.float64)
    phenotype_vector = np.zeros(sample_count, dtype=np.float64)
    phenotype_vector[:carrier_start_index] = 1.0
    phenotype_vector[carrier_start_index:] = (np.arange(sample_count - carrier_start_index, dtype=np.int64) % 2).astype(
        np.float64
    )
    genotype_vector = np.zeros(sample_count, dtype=np.float64)
    genotype_vector[carrier_start_index:] = 0.75
    return FirthPipelineFixture(
        covariate_matrix=np.column_stack(
            [
                np.ones(sample_count, dtype=np.float64),
                np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
            ]
        ),
        phenotype_matrix=phenotype_vector[None, :],
        loco_offset_matrix=(0.02 * np.sin(sample_indices))[None, :],
        genotype_matrix_by_variant=genotype_vector[None, :],
        sparse_candidate_mask=np.asarray([True], dtype=np.bool_),
        native_genotype_mean=None,
    )


def encode_integer_dosages_as_packed8(
    genotype_matrix_by_variant: npt.NDArray[np.float64],
) -> npt.NDArray[np.uint8]:
    """Encode exact zero, one, and two dosages as packed probability pairs."""
    if not bool(np.all(np.isin(genotype_matrix_by_variant, np.asarray([0.0, 1.0, 2.0])))):
        raise ValueError("packed8 test dosages must be zero, one, or two")
    homozygous_reference_probability = np.where(genotype_matrix_by_variant == 0.0, 255, 0).astype(np.uint8)
    heterozygous_probability = np.where(genotype_matrix_by_variant == 1.0, 255, 0).astype(np.uint8)
    return np.stack(
        [homozygous_reference_probability, heterozygous_probability],
        axis=2,
    )


def build_binary_kernel_config(
    *,
    candidate_capacity: int,
    batch_size: int,
) -> regenie2_binary_config.BinaryKernelConfig:
    """Build a bounded CPU policy with the production approximate-Firth phases."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-7,
            minimum_variance=1.0e-10,
            relative_variance_tolerance=1.0e-7,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=100,
            coefficient_tolerance=1.0e-6,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=batch_size,
            candidate_capacity=candidate_capacity,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=100,
            gradient_tolerance=2.5e-4,
            maximum_step_size=5.0,
            pseudo_maximum_iterations=50,
            pseudo_inner_maximum_iterations=25,
            line_search_maximum_attempts=25,
            sparse_carrier_dosage_threshold=0.5,
            use_cuda_components=False,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=100,
            gradient_tolerance=50.0e-6,
            maximum_step_size=25.0,
            fallback_iteration_multiplier=2,
            fallback_step_divisor=5.0,
            line_search_maximum_attempts=25,
            step_halving_scale=0.5,
        ),
    )


def build_prepared_firth_pipeline() -> PreparedFirthPipeline:
    """Prepare the current production chromosome state for the fixture."""
    fixture = build_firth_pipeline_fixture()
    kernel_config = build_binary_kernel_config(candidate_capacity=2, batch_size=2)
    return prepare_firth_pipeline(fixture=fixture, kernel_config=kernel_config)


def prepare_firth_pipeline(
    *,
    fixture: FirthPipelineFixture,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> PreparedFirthPipeline:
    """Prepare production and independent states for one fixture."""
    shared_state = regenie2_binary_state.build_multi_binary_state(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_matrix=jnp.asarray(fixture.phenotype_matrix),
    )
    chromosome_state = regenie2_binary_state.build_multi_binary_firth_chromosome_state(
        state=shared_state,
        loco_offset_matrix=jnp.asarray(fixture.loco_offset_matrix),
        kernel_config=kernel_config,
    )
    return PreparedFirthPipeline(
        fixture=fixture,
        kernel_config=kernel_config,
        chromosome_state=chromosome_state,
        independent_trait_states=tuple(
            build_independent_firth_trait_state(
                fixture=fixture,
                trait_index=trait_index,
                kernel_config=kernel_config,
            )
            for trait_index in range(fixture.phenotype_matrix.shape[0])
        ),
    )


def run_production_firth_pipeline(
    *,
    prepared: PreparedFirthPipeline,
    firth_se: bool,
    p_threshold: float,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
) -> regenie2_binary_result.CorrectedMultiBinaryScoreChunkResult:
    """Run the current public dosage API with one explicit selection threshold."""
    return regenie2_binary_api.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(prepared.fixture.genotype_matrix_by_variant),
        correction_plan=types.BinaryCorrectionPlan(p_threshold=p_threshold, firth_se=firth_se),
        kernel_config=kernel_config,
        sparse_candidate_mask=jnp.asarray(prepared.fixture.sparse_candidate_mask),
        native_genotype_mean=(
            None
            if prepared.fixture.native_genotype_mean is None
            else jnp.asarray(prepared.fixture.native_genotype_mean)
        ),
    )


@pytest.fixture(scope="module")
def successful_firth_pipeline() -> SuccessfulFirthPipeline:
    """Run the two static standard-error policies once per test process."""
    prepared = build_prepared_firth_pipeline()
    return SuccessfulFirthPipeline(
        prepared=prepared,
        information_standard_error_result=run_production_firth_pipeline(
            prepared=prepared,
            firth_se=False,
            p_threshold=1.0,
            kernel_config=prepared.kernel_config,
            chromosome_state=prepared.chromosome_state,
        ),
        likelihood_ratio_standard_error_result=run_production_firth_pipeline(
            prepared=prepared,
            firth_se=True,
            p_threshold=1.0,
            kernel_config=prepared.kernel_config,
            chromosome_state=prepared.chromosome_state,
        ),
    )


def compute_regenie_probability(linear_predictor: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate REGENIE's endpoint-clipped logistic probability in NumPy."""
    epsilon = regenie2_binary_config.REGENIE_NUMERICAL_EPSILON_MULTIPLIER * np.finfo(np.float64).eps
    ordinary_probability = np.reciprocal(1.0 + np.exp(-np.clip(linear_predictor, -30.0, 30.0)))
    return np.where(
        linear_predictor < regenie2_binary_config.REGENIE_LOGISTIC_MINIMUM_ETA,
        epsilon / (1.0 + epsilon),
        np.where(
            linear_predictor > regenie2_binary_config.REGENIE_LOGISTIC_MAXIMUM_ETA,
            1.0 / (1.0 + epsilon),
            ordinary_probability,
        ),
    )


def compute_independent_null_logistic_state(
    *,
    covariate_matrix: npt.NDArray[np.float64],
    phenotype_vector: npt.NDArray[np.float64],
    loco_offset: npt.NDArray[np.float64],
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> IndependentNullLogisticState:
    """Prepare null-logistic weights and projections without production state."""
    coefficient_count = covariate_matrix.shape[1]
    coefficients = np.zeros(coefficient_count, dtype=np.float64)
    minimum_probability = kernel_config.numerical.minimum_probability
    minimum_variance = kernel_config.numerical.minimum_variance
    identity_matrix = np.eye(coefficient_count, dtype=np.float64)
    for _iteration_index in range(kernel_config.null_logistic.maximum_iterations):
        linear_predictor = covariate_matrix @ coefficients + loco_offset
        probability = np.reciprocal(1.0 + np.exp(-linear_predictor))
        fitted_probability = np.clip(probability, minimum_probability, 1.0 - minimum_probability)
        weight = np.maximum(fitted_probability * (1.0 - fitted_probability), minimum_variance)
        score = covariate_matrix.T @ (phenotype_vector - fitted_probability)
        information = (covariate_matrix.T * weight) @ covariate_matrix
        coefficient_delta = np.linalg.solve(information + identity_matrix * minimum_variance, score)
        coefficients += coefficient_delta
        if float(np.max(np.abs(coefficient_delta))) <= 1.0e-12:
            break
    else:
        raise AssertionError("Independent null-logistic oracle did not converge.")

    linear_predictor = covariate_matrix @ coefficients + loco_offset
    probability = np.clip(
        np.reciprocal(1.0 + np.exp(-linear_predictor)),
        minimum_probability,
        1.0 - minimum_probability,
    )
    weight = np.maximum(probability * (1.0 - probability), minimum_variance)
    square_root_weight = np.sqrt(weight)
    weighted_covariate_matrix = square_root_weight[:, None] * covariate_matrix
    information = weighted_covariate_matrix.T @ weighted_covariate_matrix
    cholesky_factor = np.linalg.cholesky(information + identity_matrix * minimum_variance)
    return IndependentNullLogisticState(
        coefficients=coefficients,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=np.linalg.solve(
            cholesky_factor,
            weighted_covariate_matrix.T,
        ),
    )


def compute_independent_null_firth_components(
    *,
    covariate_matrix: npt.NDArray[np.float64],
    phenotype_vector: npt.NDArray[np.float64],
    loco_offset: npt.NDArray[np.float64],
    coefficients: npt.NDArray[np.float64],
) -> IndependentNullFirthComponents:
    """Evaluate the covariate-only Firth objective independently in NumPy."""
    linear_predictor = covariate_matrix @ coefficients + loco_offset
    probability = compute_regenie_probability(linear_predictor)
    weight = probability * (1.0 - probability)
    information = (covariate_matrix.T * weight) @ covariate_matrix
    log_determinant = np.linalg.slogdet(information)
    if float(log_determinant.sign) <= 0.0:
        raise AssertionError("Independent null-Firth information was not positive definite.")
    projected_covariate_matrix = np.linalg.solve(information, covariate_matrix.T).T
    leverage = weight * np.sum(projected_covariate_matrix * covariate_matrix, axis=1)
    modified_score = covariate_matrix.T @ (phenotype_vector - probability + leverage * (0.5 - probability))
    return IndependentNullFirthComponents(
        information_matrix=information,
        deviance=(
            compute_masked_logistic_deviance(
                phenotype_vector,
                probability,
                np.ones_like(phenotype_vector, dtype=np.bool_),
            )
            - float(log_determinant.logabsdet)
        ),
        modified_score=modified_score,
    )


def fit_independent_null_firth_offset(
    *,
    covariate_matrix: npt.NDArray[np.float64],
    phenotype_vector: npt.NDArray[np.float64],
    loco_offset: npt.NDArray[np.float64],
    initial_coefficients: npt.NDArray[np.float64],
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> npt.NDArray[np.float64]:
    """Fit the covariate-only null Firth predictor with NumPy Newton steps."""
    coefficients = initial_coefficients.copy()
    maximum_iterations = (
        kernel_config.null_firth.maximum_iterations * kernel_config.null_firth.fallback_iteration_multiplier
    )
    for _iteration_index in range(maximum_iterations):
        components = compute_independent_null_firth_components(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            loco_offset=loco_offset,
            coefficients=coefficients,
        )
        if float(np.max(np.abs(components.modified_score))) < kernel_config.null_firth.gradient_tolerance:
            return covariate_matrix @ coefficients + loco_offset

        coefficient_step = np.linalg.solve(components.information_matrix, components.modified_score)
        step_scale = max(
            float(np.max(np.abs(coefficient_step))) / kernel_config.null_firth.maximum_step_size,
            1.0,
        )
        candidate_step = coefficient_step / step_scale
        accepted = False
        for _attempt_index in range(kernel_config.null_firth.line_search_maximum_attempts):
            candidate_coefficients = coefficients + candidate_step
            candidate_components = compute_independent_null_firth_components(
                covariate_matrix=covariate_matrix,
                phenotype_vector=phenotype_vector,
                loco_offset=loco_offset,
                coefficients=candidate_coefficients,
            )
            if candidate_components.deviance < components.deviance:
                coefficients = candidate_coefficients
                accepted = True
                break
            candidate_step *= kernel_config.null_firth.step_halving_scale
        if not accepted:
            raise AssertionError("Independent null-Firth oracle line search failed.")
    raise AssertionError("Independent null-Firth oracle did not converge.")


def build_independent_firth_trait_state(
    *,
    fixture: FirthPipelineFixture,
    trait_index: int,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> IndependentFirthTraitState:
    """Build every prepared Firth operand directly from the raw fixture."""
    phenotype_vector = fixture.phenotype_matrix[trait_index]
    loco_offset = fixture.loco_offset_matrix[trait_index]
    null_logistic_state = compute_independent_null_logistic_state(
        covariate_matrix=fixture.covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        kernel_config=kernel_config,
    )
    null_firth_offset = fit_independent_null_firth_offset(
        covariate_matrix=fixture.covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset,
        initial_coefficients=null_logistic_state.coefficients,
        kernel_config=kernel_config,
    )
    return IndependentFirthTraitState(
        square_root_weight=null_logistic_state.square_root_weight,
        weighted_genotype_projection_matrix=null_logistic_state.weighted_genotype_projection_matrix,
        null_firth_offset=null_firth_offset,
    )


def compute_masked_logistic_deviance(
    phenotype_vector: npt.NDArray[np.float64],
    probability_vector: npt.NDArray[np.float64],
    active_sample_mask: npt.NDArray[np.bool_],
) -> float:
    """Compute a masked Bernoulli deviance independently in NumPy."""
    negative_log_likelihood = -np.where(
        phenotype_vector > 0.5,
        np.log(probability_vector),
        np.log1p(-probability_vector),
    )
    return 2.0 * float(np.sum(np.where(active_sample_mask, negative_log_likelihood, 0.0)))


def compute_scalar_firth_oracle_components(
    *,
    phenotype_vector: npt.NDArray[np.float64],
    genotype_vector: npt.NDArray[np.float64],
    offset_vector: npt.NDArray[np.float64],
    active_sample_mask: npt.NDArray[np.bool_],
    non_active_deviance: float,
    beta: float,
) -> ScalarFirthOracleComponents:
    """Evaluate the adjusted-score objective independently in NumPy."""
    probability_vector = compute_regenie_probability(offset_vector + genotype_vector * beta)
    weight_vector = probability_vector * (1.0 - probability_vector)
    active_weight_vector = np.where(active_sample_mask, weight_vector, 0.0)
    information_diagonal = genotype_vector**2 * active_weight_vector
    genotype_information = float(np.sum(information_diagonal))
    score_adjustment = float(
        np.sum(
            np.where(
                active_sample_mask,
                genotype_vector * information_diagonal * (0.5 - probability_vector),
                0.0,
            )
        )
        / genotype_information
    )
    score = float(
        np.sum(
            np.where(
                active_sample_mask,
                genotype_vector * (phenotype_vector - probability_vector),
                0.0,
            )
        )
        + score_adjustment
    )
    return ScalarFirthOracleComponents(
        genotype_information=genotype_information,
        penalized_deviance=(
            non_active_deviance
            + compute_masked_logistic_deviance(phenotype_vector, probability_vector, active_sample_mask)
            - math.log(genotype_information)
        ),
        score=score,
    )


def find_scalar_firth_adjusted_score_root(
    *,
    phenotype_vector: npt.NDArray[np.float64],
    genotype_vector: npt.NDArray[np.float64],
    offset_vector: npt.NDArray[np.float64],
    active_sample_mask: npt.NDArray[np.bool_],
    non_active_deviance: float,
) -> float:
    """Find the scalar adjusted-score root with an independent bisection."""

    def compute_score(beta: float) -> float:
        return compute_scalar_firth_oracle_components(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=beta,
        ).score

    lower_beta = -1.0
    upper_beta = 1.0
    lower_score = compute_score(lower_beta)
    upper_score = compute_score(upper_beta)
    for _expansion_index in range(8):
        if lower_score * upper_score <= 0.0:
            break
        lower_beta *= 2.0
        upper_beta *= 2.0
        lower_score = compute_score(lower_beta)
        upper_score = compute_score(upper_beta)
    else:
        raise AssertionError("Independent Firth oracle could not bracket the adjusted-score root.")

    for _iteration_index in range(100):
        midpoint_beta = (lower_beta + upper_beta) / 2.0
        midpoint_score = compute_score(midpoint_beta)
        if abs(midpoint_score) < 1.0e-13 or upper_beta - lower_beta < 1.0e-13:
            return midpoint_beta
        if lower_score * midpoint_score <= 0.0:
            upper_beta = midpoint_beta
        else:
            lower_beta = midpoint_beta
            lower_score = midpoint_score
    return (lower_beta + upper_beta) / 2.0


def compute_firth_reference(
    *,
    prepared: PreparedFirthPipeline,
    trait_index: int,
    variant_index: int,
    sparse_correction: bool,
) -> FirthReferenceResult:
    """Solve one candidate from independently prepared NumPy state."""
    raw_genotype = prepared.fixture.genotype_matrix_by_variant[variant_index]
    genotype_mean = (
        float(np.mean(raw_genotype))
        if prepared.fixture.native_genotype_mean is None
        else float(prepared.fixture.native_genotype_mean[variant_index])
    )
    genotype_flipped = genotype_mean > 1.0
    coded_genotype = 2.0 - raw_genotype if genotype_flipped else raw_genotype
    independent_trait_state = prepared.independent_trait_states[trait_index]
    square_root_weight = independent_trait_state.square_root_weight
    weighted_projection_matrix = independent_trait_state.weighted_genotype_projection_matrix
    weighted_genotype = coded_genotype * square_root_weight
    projection_coordinates = weighted_projection_matrix @ weighted_genotype
    residualized_genotype = (
        weighted_genotype - projection_coordinates @ weighted_projection_matrix
    ) / square_root_weight
    carrier_sample_mask = coded_genotype > prepared.kernel_config.approximate_firth.sparse_carrier_dosage_threshold
    active_sample_mask = carrier_sample_mask if sparse_correction else np.ones_like(carrier_sample_mask)
    phenotype_vector = prepared.fixture.phenotype_matrix[trait_index]
    offset_vector = independent_trait_state.null_firth_offset
    null_probability = compute_regenie_probability(offset_vector)
    full_null_deviance = compute_masked_logistic_deviance(
        phenotype_vector,
        null_probability,
        np.ones_like(active_sample_mask),
    )
    active_null_deviance = compute_masked_logistic_deviance(
        phenotype_vector,
        null_probability,
        active_sample_mask,
    )
    non_active_deviance = full_null_deviance - active_null_deviance if sparse_correction else 0.0
    beta = find_scalar_firth_adjusted_score_root(
        phenotype_vector=phenotype_vector,
        genotype_vector=residualized_genotype,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
    )
    null_components = compute_scalar_firth_oracle_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=residualized_genotype,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=0.0,
    )
    terminal_components = compute_scalar_firth_oracle_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=residualized_genotype,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=beta,
    )
    null_penalized_deviance = full_null_deviance - math.log(null_components.genotype_information)
    chi_squared = max(null_penalized_deviance - terminal_components.penalized_deviance, 0.0)
    output_beta = -beta if genotype_flipped else beta
    return FirthReferenceResult(
        beta=output_beta,
        standard_error=math.sqrt(1.0 / terminal_components.genotype_information),
        chi_squared=chi_squared,
        log10_p_value=-math.log10(math.erfc(math.sqrt(chi_squared / 2.0))),
    )


def assert_firth_association_matches_references(
    association: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    references: list[list[FirthReferenceResult]],
) -> None:
    """Compare every reported statistic with a trait-major oracle matrix."""
    tests.numerical.assert_absolute_difference_less_than(
        association.beta,
        np.asarray([[reference.beta for reference in trait_references] for trait_references in references]),
        FIRTH_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.standard_error,
        np.asarray([[reference.standard_error for reference in trait_references] for trait_references in references]),
        FIRTH_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.chi_squared,
        np.asarray([[reference.chi_squared for reference in trait_references] for trait_references in references]),
        FIRTH_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.log10_p_value,
        np.asarray([[reference.log10_p_value for reference in trait_references] for trait_references in references]),
        FIRTH_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )


def test_production_approximate_firth_matches_independent_dense_and_sparse_oracles(
    successful_firth_pipeline: SuccessfulFirthPipeline,
) -> None:
    """Cover score dispatch, mixed sparse routing, solving, and result merge."""
    prepared = successful_firth_pipeline.prepared
    observed = successful_firth_pipeline.information_standard_error_result
    references = [
        compute_firth_reference(prepared=prepared, trait_index=0, variant_index=0, sparse_correction=False),
        compute_firth_reference(prepared=prepared, trait_index=0, variant_index=1, sparse_correction=True),
    ]
    dense_reference_for_sparse_variant = compute_firth_reference(
        prepared=prepared,
        trait_index=0,
        variant_index=1,
        sparse_correction=False,
    )
    association = observed.association

    assert int(np.asarray(observed.firth_candidate_count)) == 2
    assert observed.firth_candidate_capacity == 2
    np.testing.assert_array_equal(
        np.asarray(association.correction_code),
        np.full((1, 2), types.BinaryCorrectionCode.FIRTH_SUCCESS.value, dtype=np.uint8),
    )
    assert abs(references[1].beta - dense_reference_for_sparse_variant.beta) > 1.0e-2
    tests.numerical.assert_absolute_difference_less_than(
        association.beta,
        np.asarray([[reference.beta for reference in references]]),
        FIRTH_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.standard_error,
        np.asarray([[reference.standard_error for reference in references]]),
        FIRTH_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.chi_squared,
        np.asarray([[reference.chi_squared for reference in references]]),
        FIRTH_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.log10_p_value,
        np.asarray([[reference.log10_p_value for reference in references]]),
        FIRTH_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )


def test_packed8_pipeline_uses_native_mean_and_restores_flipped_alleles() -> None:
    """Cover packed delivery and both native and dosage-derived allele flips."""
    fixture = build_packed_firth_pipeline_fixture()
    kernel_config = build_binary_kernel_config(candidate_capacity=2, batch_size=2)
    prepared = prepare_firth_pipeline(fixture=fixture, kernel_config=kernel_config)
    packed_probability_pairs = encode_integer_dosages_as_packed8(fixture.genotype_matrix_by_variant)
    native_genotype_mean = fixture.native_genotype_mean
    if native_genotype_mean is None:
        raise AssertionError("Packed Firth fixture requires native genotype means.")
    dosage_result = run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=kernel_config,
        chromosome_state=prepared.chromosome_state,
    )
    packed_result = regenie2_binary_api.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
        chromosome_state=prepared.chromosome_state,
        packed_probability_pairs_by_variant=jnp.asarray(packed_probability_pairs),
        correction_plan=types.BinaryCorrectionPlan(p_threshold=1.0, firth_se=False),
        kernel_config=kernel_config,
        sparse_candidate_mask=jnp.asarray(fixture.sparse_candidate_mask),
        native_genotype_mean=jnp.asarray(native_genotype_mean),
    )
    references = [
        [
            compute_firth_reference(prepared=prepared, trait_index=0, variant_index=0, sparse_correction=False),
            compute_firth_reference(prepared=prepared, trait_index=0, variant_index=1, sparse_correction=True),
        ]
    ]
    no_native_mean_prepared = dataclasses.replace(
        prepared,
        fixture=dataclasses.replace(fixture, native_genotype_mean=None),
    )
    no_native_mean_reference = compute_firth_reference(
        prepared=no_native_mean_prepared,
        trait_index=0,
        variant_index=1,
        sparse_correction=True,
    )

    assert float(np.mean(fixture.genotype_matrix_by_variant[0])) > 1.0
    assert float(np.mean(fixture.genotype_matrix_by_variant[1])) == 1.0
    assert float(native_genotype_mean[1]) > 1.0
    assert abs(references[0][1].beta - no_native_mean_reference.beta) > 1.0e-2
    for observed in [dosage_result, packed_result]:
        assert int(np.asarray(observed.firth_candidate_count)) == 2
        assert observed.firth_candidate_capacity == 2
        np.testing.assert_array_equal(
            np.asarray(observed.association.correction_code),
            np.full((1, 2), types.BinaryCorrectionCode.FIRTH_SUCCESS.value, dtype=np.uint8),
        )
        assert_firth_association_matches_references(observed.association, references)


def test_multi_trait_pipeline_preserves_reordered_candidate_associations() -> None:
    """Keep every lane field aligned through bucketing, solving, and scatter."""
    fixture = build_reordered_multi_trait_fixture()
    kernel_config = build_binary_kernel_config(candidate_capacity=3, batch_size=2)
    prepared = prepare_firth_pipeline(fixture=fixture, kernel_config=kernel_config)
    observed = run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=kernel_config,
        chromosome_state=prepared.chromosome_state,
    )
    references = [
        [
            compute_firth_reference(
                prepared=prepared,
                trait_index=trait_index,
                variant_index=variant_index,
                sparse_correction=bool(fixture.sparse_candidate_mask[variant_index]),
            )
            for variant_index in range(fixture.genotype_matrix_by_variant.shape[0])
        ]
        for trait_index in range(fixture.phenotype_matrix.shape[0])
    ]

    assert observed.firth_candidate_capacity > kernel_config.firth_candidate.batch_size
    assert int(np.asarray(observed.firth_candidate_count)) == 6
    assert observed.firth_candidate_capacity == 6
    np.testing.assert_array_equal(
        np.asarray(observed.association.correction_code),
        np.full((2, 3), types.BinaryCorrectionCode.FIRTH_SUCCESS.value, dtype=np.uint8),
    )
    assert_firth_association_matches_references(observed.association, references)


def test_zero_candidate_pipeline_retains_score_results() -> None:
    """Leave score rows untouched when no lane crosses the Firth threshold."""
    prepared = build_prepared_firth_pipeline()
    observed = run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0e-300,
        kernel_config=prepared.kernel_config,
        chromosome_state=prepared.chromosome_state,
    )
    score_result = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=prepared.chromosome_state.score_state,
        genotype_matrix_by_variant=jnp.asarray(prepared.fixture.genotype_matrix_by_variant),
        firth_candidate_p_threshold=None,
        minimum_variance=prepared.kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=prepared.kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )

    assert int(np.asarray(observed.firth_candidate_count)) == 0
    tests.numerical.assert_absolute_difference_less_than(observed.association.beta, score_result.beta, 1.0e-12)
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.standard_error,
        score_result.standard_error,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.chi_squared,
        score_result.chi_squared,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.log10_p_value,
        score_result.log10_p_value,
        1.0e-12,
    )
    np.testing.assert_array_equal(observed.association.correction_code, score_result.correction_code)


def test_sparse_lane_above_compact_capacity_uses_carrier_only_dense_storage() -> None:
    """Retain sparse semantics when 65 carriers cannot enter the compact stream."""
    fixture = build_over_capacity_sparse_fixture()
    kernel_config = build_binary_kernel_config(candidate_capacity=1, batch_size=1)
    prepared = prepare_firth_pipeline(fixture=fixture, kernel_config=kernel_config)
    observed = run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=kernel_config,
        chromosome_state=prepared.chromosome_state,
    )
    sparse_reference = compute_firth_reference(
        prepared=prepared,
        trait_index=0,
        variant_index=0,
        sparse_correction=True,
    )
    dense_reference = compute_firth_reference(
        prepared=prepared,
        trait_index=0,
        variant_index=0,
        sparse_correction=False,
    )
    carrier_count = int(
        np.sum(fixture.genotype_matrix_by_variant[0] > kernel_config.approximate_firth.sparse_carrier_dosage_threshold)
    )

    assert carrier_count == 65
    assert int(np.asarray(observed.firth_candidate_count)) == 1
    assert observed.firth_candidate_capacity == 1
    assert abs(sparse_reference.beta - dense_reference.beta) > 1.0e-3
    np.testing.assert_array_equal(
        np.asarray(observed.association.correction_code),
        np.asarray([[types.BinaryCorrectionCode.FIRTH_SUCCESS.value]], dtype=np.uint8),
    )
    assert_firth_association_matches_references(observed.association, [[sparse_reference]])


def test_firth_se_uses_likelihood_ratio_standard_error(
    successful_firth_pipeline: SuccessfulFirthPipeline,
) -> None:
    """Change only standard error to absolute beta divided by square-root LRT."""
    information_result = successful_firth_pipeline.information_standard_error_result.association
    likelihood_ratio_result = successful_firth_pipeline.likelihood_ratio_standard_error_result.association
    expected_standard_error = np.abs(np.asarray(information_result.beta)) / np.sqrt(
        np.asarray(information_result.chi_squared)
    )

    tests.numerical.assert_absolute_difference_less_than(
        likelihood_ratio_result.standard_error,
        expected_standard_error,
        FIRTH_SE_TRANSFORMATION_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        likelihood_ratio_result.beta,
        information_result.beta,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        likelihood_ratio_result.chi_squared,
        information_result.chi_squared,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        likelihood_ratio_result.log10_p_value,
        information_result.log10_p_value,
        1.0e-12,
    )
    assert bool(
        np.any(
            np.abs(np.asarray(information_result.standard_error) - expected_standard_error)
            > FIRTH_STANDARD_ERROR_ABSOLUTE_TOLERANCE
        )
    )
    np.testing.assert_array_equal(likelihood_ratio_result.correction_code, information_result.correction_code)


def test_null_firth_failure_propagates_through_production_pipeline(
    successful_firth_pipeline: SuccessfulFirthPipeline,
) -> None:
    """Turn a failed null fit into explicit failed correction rows and NaNs."""
    prepared = successful_firth_pipeline.prepared
    failed_chromosome_state = dataclasses.replace(
        prepared.chromosome_state,
        null_firth_penalized_log_likelihood=jnp.full_like(
            prepared.chromosome_state.null_firth_penalized_log_likelihood,
            jnp.nan,
        ),
    )
    observed = run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=prepared.kernel_config,
        chromosome_state=failed_chromosome_state,
    )

    assert int(np.asarray(observed.firth_candidate_count)) == 2
    np.testing.assert_array_equal(
        np.asarray(observed.association.correction_code),
        np.full((1, 2), types.BinaryCorrectionCode.FIRTH_FAILED.value, dtype=np.uint8),
    )
    assert bool(np.all(np.isnan(np.asarray(observed.association.beta))))
    assert bool(np.all(np.isnan(np.asarray(observed.association.standard_error))))
    assert bool(np.all(np.isnan(np.asarray(observed.association.chi_squared))))
    assert bool(np.all(np.isnan(np.asarray(observed.association.log10_p_value))))


def test_fixed_capacity_selection_and_merge_preserve_flat_candidate_order() -> None:
    """Select trait-major lanes and merge active, flipped, and failed results."""
    genotype_matrix_by_variant = jnp.asarray(
        [[0.0, 0.5], [1.0, 1.5], [2.0, 0.0]],
        dtype=jnp.float32,
    )
    candidate_mask = jnp.asarray(
        [[False, True, True], [True, False, False]],
        dtype=jnp.bool_,
    )
    selected_rows = regenie2_binary_firth_prepare.select_multi_firth_candidate_rows(
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        candidate_mask=candidate_mask,
        candidate_capacity=4,
        firth_batch_size=2,
    )
    lanes = regenie2_binary_candidates.FirthCandidateLaneInputs(
        flat_trait_indices=selected_rows.flat_trait_indices,
        flat_variant_indices=selected_rows.flat_variant_indices,
        flat_active_mask=selected_rows.flat_active_mask,
        phenotype_matrix=jnp.zeros((2, 2), dtype=jnp.float32),
    )
    score_result = association_result.AssociationResult(
        beta=jnp.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=jnp.float32),
        standard_error=jnp.asarray([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]], dtype=jnp.float32),
        chi_squared=jnp.asarray([[50.0, 51.0, 52.0], [60.0, 61.0, 62.0]], dtype=jnp.float32),
        log10_p_value=jnp.asarray([[70.0, 71.0, 72.0], [80.0, 81.0, 82.0]], dtype=jnp.float32),
        correction_code=jnp.asarray(
            [
                [
                    types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                    types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                    types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                ],
                [
                    types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                    types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                    types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                ],
            ],
            dtype=jnp.uint8,
        ),
    )
    firth_result = regenie2_binary_firth_types.FirthVariantResult(
        beta=jnp.asarray([1.0, 2.0, 3.0, 999.0], dtype=jnp.float64),
        standard_error=jnp.asarray([0.1, 0.2, 0.3, 999.0], dtype=jnp.float64),
        chi_squared=jnp.asarray([1.0, 4.0, 9.0, 999.0], dtype=jnp.float64),
        log10_p_value=jnp.asarray([0.2, 0.7, 1.2, 999.0], dtype=jnp.float64),
        valid_mask=jnp.asarray([True, True, False, True]),
    )

    observed = fixed_capacity.merge_fixed_capacity_firth_result(
        result=score_result,
        firth_result=firth_result,
        lanes=lanes,
        genotype_flip_mask=jnp.asarray([False, True, False, False]),
        candidate_capacity=4,
        firth_se=False,
    )

    np.testing.assert_array_equal(np.asarray(selected_rows.flat_trait_indices), np.asarray([0, 0, 1, 0]))
    np.testing.assert_array_equal(np.asarray(selected_rows.flat_variant_indices), np.asarray([1, 2, 0, 0]))
    np.testing.assert_array_equal(np.asarray(selected_rows.flat_active_mask), np.asarray([True, True, True, False]))
    tests.numerical.assert_absolute_difference_less_than(
        selected_rows.genotype_matrix_by_variant,
        np.asarray([[1.0, 1.5], [2.0, 0.0], [0.0, 0.5], [0.0, 0.5]]),
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.beta,
        np.asarray([[10.0, 1.0, -2.0], [np.nan, 21.0, 22.0]]),
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.standard_error,
        np.asarray([[30.0, 0.1, 0.2], [np.nan, 41.0, 42.0]]),
        1.0e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.correction_code),
        np.asarray(
            [
                [
                    types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                    types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                    types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                ],
                [
                    types.BinaryCorrectionCode.FIRTH_FAILED.value,
                    types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                    types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                ],
            ],
            dtype=np.uint8,
        ),
    )


def test_host_materialization_rejects_production_fixed_capacity_overflow(
    successful_firth_pipeline: SuccessfulFirthPipeline,
) -> None:
    """Reject a device result whose selected score candidates exceed capacity."""
    prepared = successful_firth_pipeline.prepared
    overflow_config = dataclasses.replace(
        prepared.kernel_config,
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=1,
            candidate_capacity=1,
        ),
    )
    overflow_result = run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=overflow_config,
        chromosome_state=prepared.chromosome_state,
    )
    device_batch: jax_backend.DeviceAssociationBatch = jax_backend.AssociationBatch(
        association=overflow_result.association,
        raw_packed8_statistics=None,
        firth_candidate_count=overflow_result.firth_candidate_count,
        firth_candidate_capacity=overflow_result.firth_candidate_capacity,
    )

    assert int(np.asarray(overflow_result.firth_candidate_count)) == 2
    assert overflow_result.firth_candidate_capacity == 1
    with pytest.raises(ValueError, match=r"candidate count 2 exceeded.*capacity of 1"):
        jax_backend.JaxBackendBase().materialize_batch(
            device_result=device_batch,
            active_trait_indices=None,
            logical_variant_count=2,
        )
