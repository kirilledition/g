"""Default binary REGENIE step 2 kernel policy."""

from __future__ import annotations

from dataclasses import dataclass

BINARY_CASE_THRESHOLD = 0.5
REGENIE_LOGISTIC_MINIMUM_ETA = -30.0
REGENIE_LOGISTIC_MAXIMUM_ETA = 30.0
REGENIE_NUMERICAL_EPSILON_MULTIPLIER = 10.0


@dataclass(frozen=True)
class BinaryNumericalConfig:
    """Shared binary numerical floors and tolerances.

    Attributes:
        minimum_probability: Logistic probability clipping floor.
        minimum_variance: Bernoulli and information-matrix variance floor.
        relative_variance_tolerance: Relative score-test variance floor multiplier.

    """

    minimum_probability: float
    minimum_variance: float
    relative_variance_tolerance: float


@dataclass(frozen=True)
class BinaryNullLogisticConfig:
    """Null logistic IRLS policy.

    Attributes:
        maximum_iterations: Maximum IRLS iterations for the null logistic model.
        coefficient_tolerance: Coefficient convergence tolerance for the null logistic model.

    """

    maximum_iterations: int
    coefficient_tolerance: float


@dataclass(frozen=True)
class FirthCandidateConfig:
    """Device Firth candidate batching policy.

    Attributes:
        batch_size: Fixed batch size for device-resident Firth fallback lanes.
        candidate_capacity: Preferred fixed candidate capacity before falling back to full chunk capacity.

    """

    batch_size: int
    candidate_capacity: int


@dataclass(frozen=True)
class ApproximateFirthConfig:
    """Approximate Firth solver policy.

    Attributes:
        maximum_iterations: Maximum Firth solver iterations.
        gradient_tolerance: Firth adjusted-score convergence tolerance.
        coefficient_tolerance: Firth coefficient-step convergence tolerance.
        likelihood_tolerance: Firth penalized-likelihood convergence tolerance.
        maximum_step_size: Maximum absolute Firth coefficient update before step scaling.
        pseudo_maximum_iterations: Maximum approximate pseudo-Firth outer iterations.
        pseudo_inner_maximum_iterations: Maximum pseudo-response logistic inner iterations.
        newton_raphson_zero_start_iterations: Maximum zero-start scalar Newton-Raphson iterations.
        line_search_maximum_attempts: Maximum scalar/full-model Firth line-search attempts.
        step_halving_maximum_attempts: Maximum full-model Firth step-halving attempts.
        initial_response_scale: Pseudo-response scale for block full-model Firth initialization.
        sparse_carrier_dosage_threshold: Raw dosage threshold for sparse carrier-only Firth samples.
        step_halving_scale: Full-model Firth step multiplier after each rejected backtracking attempt.
        use_block_math: Whether to use the experimental block-matrix Firth path.

    """

    maximum_iterations: int
    gradient_tolerance: float
    coefficient_tolerance: float
    likelihood_tolerance: float
    maximum_step_size: float
    pseudo_maximum_iterations: int
    pseudo_inner_maximum_iterations: int
    newton_raphson_zero_start_iterations: int
    line_search_maximum_attempts: int
    step_halving_maximum_attempts: int
    initial_response_scale: float
    sparse_carrier_dosage_threshold: float
    step_halving_scale: float
    use_block_math: bool


@dataclass(frozen=True)
class NullFirthConfig:
    """Covariate-only null Firth solver policy.

    Attributes:
        maximum_iterations: Maximum covariate-only null Firth iterations.
        gradient_tolerance: Covariate-only null Firth adjusted-score tolerance.
        maximum_step_size: Covariate-only null Firth maximum coefficient step size.
        fallback_iteration_multiplier: Multiplier for null Firth fallback retry iterations.
        fallback_step_divisor: Divisor for null Firth fallback retry step size.
        line_search_maximum_attempts: Maximum covariate-only null Firth line-search attempts.
        step_halving_scale: Null Firth step multiplier after each rejected line-search attempt.

    """

    maximum_iterations: int
    gradient_tolerance: float
    maximum_step_size: float
    fallback_iteration_multiplier: int
    fallback_step_divisor: float
    line_search_maximum_attempts: int
    step_halving_scale: float


@dataclass(frozen=True)
class BinaryScoreConfig:
    """Static settings required by binary score kernels.

    Attributes:
        numerical: Shared binary numerical floors and tolerances.
        null_logistic: Null logistic IRLS policy.

    """

    numerical: BinaryNumericalConfig
    null_logistic: BinaryNullLogisticConfig


@dataclass(frozen=True)
class BinaryKernelConfig(BinaryScoreConfig):
    """Static binary-kernel settings that affect traced JAX programs.

    Attributes:
        firth_candidate: Device Firth candidate batching policy.
        approximate_firth: Approximate Firth solver policy.
        null_firth: Covariate-only null Firth solver policy.

    """

    firth_candidate: FirthCandidateConfig
    approximate_firth: ApproximateFirthConfig
    null_firth: NullFirthConfig
