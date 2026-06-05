"""Default binary REGENIE step 2 kernel policy."""

from __future__ import annotations

from dataclasses import dataclass

MINIMUM_PROBABILITY = 1.0e-6
MINIMUM_VARIANCE = 1.0e-8
RELATIVE_VARIANCE_TOLERANCE = 1.0e-6
DEFAULT_MAXIMUM_NULL_ITERATIONS = 50
NULL_LOGISTIC_COEFFICIENT_TOLERANCE = 1.0e-6
BINARY_CASE_THRESHOLD = 0.5
FIRTH_GRADIENT_TOLERANCE = 2.5e-4
FIRTH_COEFFICIENT_TOLERANCE = 2.5e-4
FIRTH_LIKELIHOOD_TOLERANCE = 2.5e-4
FIRTH_MAXIMUM_STEP_SIZE = 5.0
FIRTH_MAXIMUM_ITERATIONS = 250
FIRTH_PSEUDO_MAXIMUM_ITERATIONS = 50
FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS = 25
FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS = 100
FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS = 12
FIRTH_INITIAL_RESPONSE_SCALE = 4.863891244002886
FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD = 1.0e-4
FIRTH_STEP_HALVING_SCALE = 0.5
NULL_FIRTH_MAXIMUM_ITERATIONS = 1000
NULL_FIRTH_GRADIENT_TOLERANCE = 50.0e-6
NULL_FIRTH_MAXIMUM_STEP_SIZE = 25.0
NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER = 5
NULL_FIRTH_FALLBACK_STEP_DIVISOR = 5.0
NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
NULL_FIRTH_STEP_HALVING_SCALE = 0.5
REGENIE_LOGISTIC_MINIMUM_ETA = -30.0
REGENIE_LOGISTIC_MAXIMUM_ETA = 30.0
REGENIE_NUMERICAL_EPSILON = 10.0 * 2.220446049250313e-16
DEFAULT_FIRTH_BATCH_SIZE = 64
DEFAULT_FIRTH_CANDIDATE_CAPACITY = 1024


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

    def __post_init__(self) -> None:
        """Validate positive numerical settings."""
        if self.minimum_probability <= 0.0:
            message = "Minimum probability must be positive."
            raise ValueError(message)
        if self.minimum_probability >= 0.5:
            message = "Minimum probability must be less than 0.5."
            raise ValueError(message)
        if self.minimum_variance <= 0.0:
            message = "Minimum variance must be positive."
            raise ValueError(message)
        if self.relative_variance_tolerance <= 0.0:
            message = "Relative variance tolerance must be positive."
            raise ValueError(message)


@dataclass(frozen=True)
class BinaryNullLogisticConfig:
    """Null logistic IRLS policy.

    Attributes:
        maximum_iterations: Maximum IRLS iterations for the null logistic model.
        coefficient_tolerance: Coefficient convergence tolerance for the null logistic model.

    """

    maximum_iterations: int
    coefficient_tolerance: float

    def __post_init__(self) -> None:
        """Validate null-logistic settings."""
        if self.maximum_iterations <= 0:
            message = "Maximum null iterations must be positive."
            raise ValueError(message)
        if self.coefficient_tolerance <= 0.0:
            message = "Null logistic coefficient tolerance must be positive."
            raise ValueError(message)


@dataclass(frozen=True)
class FirthCandidateConfig:
    """Device Firth candidate batching policy.

    Attributes:
        batch_size: Fixed batch size for device-resident Firth fallback lanes.
        candidate_capacity: Preferred fixed candidate capacity before falling back to full chunk capacity.

    """

    batch_size: int
    candidate_capacity: int

    def __post_init__(self) -> None:
        """Validate candidate batching settings."""
        if self.batch_size <= 0:
            message = "Firth batch size must be positive."
            raise ValueError(message)
        if self.candidate_capacity <= 0:
            message = "Firth candidate capacity must be positive."
            raise ValueError(message)


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
    pseudo_maximum_iterations: int = FIRTH_PSEUDO_MAXIMUM_ITERATIONS
    pseudo_inner_maximum_iterations: int = FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS
    newton_raphson_zero_start_iterations: int = FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS
    line_search_maximum_attempts: int = FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS
    step_halving_maximum_attempts: int = FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS
    initial_response_scale: float = FIRTH_INITIAL_RESPONSE_SCALE
    sparse_carrier_dosage_threshold: float = FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD
    step_halving_scale: float = FIRTH_STEP_HALVING_SCALE
    use_block_math: bool = False

    def __post_init__(self) -> None:
        """Validate approximate Firth settings."""
        if self.maximum_iterations <= 0:
            message = "Firth maximum iterations must be positive."
            raise ValueError(message)
        if self.gradient_tolerance <= 0.0:
            message = "Firth gradient tolerance must be positive."
            raise ValueError(message)
        if self.coefficient_tolerance <= 0.0:
            message = "Firth coefficient tolerance must be positive."
            raise ValueError(message)
        if self.likelihood_tolerance <= 0.0:
            message = "Firth likelihood tolerance must be positive."
            raise ValueError(message)
        if self.maximum_step_size <= 0.0:
            message = "Firth maximum step size must be positive."
            raise ValueError(message)
        if self.pseudo_maximum_iterations <= 0:
            message = "Firth pseudo maximum iterations must be positive."
            raise ValueError(message)
        if self.pseudo_inner_maximum_iterations <= 0:
            message = "Firth pseudo inner maximum iterations must be positive."
            raise ValueError(message)
        if self.newton_raphson_zero_start_iterations <= 0:
            message = "Firth zero-start Newton-Raphson iterations must be positive."
            raise ValueError(message)
        if self.line_search_maximum_attempts <= 0:
            message = "Firth line-search maximum attempts must be positive."
            raise ValueError(message)
        if self.step_halving_maximum_attempts <= 0:
            message = "Firth step-halving maximum attempts must be positive."
            raise ValueError(message)
        if self.initial_response_scale <= 0.0:
            message = "Firth initial response scale must be positive."
            raise ValueError(message)
        if self.sparse_carrier_dosage_threshold <= 0.0:
            message = "Firth sparse carrier dosage threshold must be positive."
            raise ValueError(message)
        if self.step_halving_scale <= 0.0:
            message = "Firth step-halving scale must be positive."
            raise ValueError(message)


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

    maximum_iterations: int = NULL_FIRTH_MAXIMUM_ITERATIONS
    gradient_tolerance: float = NULL_FIRTH_GRADIENT_TOLERANCE
    maximum_step_size: float = NULL_FIRTH_MAXIMUM_STEP_SIZE
    fallback_iteration_multiplier: int = NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER
    fallback_step_divisor: float = NULL_FIRTH_FALLBACK_STEP_DIVISOR
    line_search_maximum_attempts: int = NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS
    step_halving_scale: float = NULL_FIRTH_STEP_HALVING_SCALE

    def __post_init__(self) -> None:
        """Validate null Firth settings."""
        if self.maximum_iterations <= 0:
            message = "Null Firth maximum iterations must be positive."
            raise ValueError(message)
        if self.gradient_tolerance <= 0.0:
            message = "Null Firth gradient tolerance must be positive."
            raise ValueError(message)
        if self.maximum_step_size <= 0.0:
            message = "Null Firth maximum step size must be positive."
            raise ValueError(message)
        if self.fallback_iteration_multiplier <= 0:
            message = "Null Firth fallback iteration multiplier must be positive."
            raise ValueError(message)
        if self.fallback_step_divisor <= 0.0:
            message = "Null Firth fallback step divisor must be positive."
            raise ValueError(message)
        if self.line_search_maximum_attempts <= 0:
            message = "Null Firth line-search maximum attempts must be positive."
            raise ValueError(message)
        if self.step_halving_scale <= 0.0:
            message = "Null Firth step-halving scale must be positive."
            raise ValueError(message)


@dataclass(frozen=True)
class BinaryKernelConfig:
    """Static binary-kernel settings that affect traced JAX programs.

    Attributes:
        numerical: Shared binary numerical floors and tolerances.
        null_logistic: Null logistic IRLS policy.
        firth_candidate: Device Firth candidate batching policy.
        approximate_firth: Approximate Firth solver policy.
        null_firth: Covariate-only null Firth solver policy.

    """

    numerical: BinaryNumericalConfig
    null_logistic: BinaryNullLogisticConfig
    firth_candidate: FirthCandidateConfig
    approximate_firth: ApproximateFirthConfig
    null_firth: NullFirthConfig


DEFAULT_BINARY_KERNEL_CONFIG = BinaryKernelConfig(
    numerical=BinaryNumericalConfig(
        minimum_probability=MINIMUM_PROBABILITY,
        minimum_variance=MINIMUM_VARIANCE,
        relative_variance_tolerance=RELATIVE_VARIANCE_TOLERANCE,
    ),
    null_logistic=BinaryNullLogisticConfig(
        maximum_iterations=DEFAULT_MAXIMUM_NULL_ITERATIONS,
        coefficient_tolerance=NULL_LOGISTIC_COEFFICIENT_TOLERANCE,
    ),
    firth_candidate=FirthCandidateConfig(
        batch_size=DEFAULT_FIRTH_BATCH_SIZE,
        candidate_capacity=DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    ),
    approximate_firth=ApproximateFirthConfig(
        maximum_iterations=FIRTH_MAXIMUM_ITERATIONS,
        gradient_tolerance=FIRTH_GRADIENT_TOLERANCE,
        coefficient_tolerance=FIRTH_COEFFICIENT_TOLERANCE,
        likelihood_tolerance=FIRTH_LIKELIHOOD_TOLERANCE,
        maximum_step_size=FIRTH_MAXIMUM_STEP_SIZE,
        pseudo_maximum_iterations=FIRTH_PSEUDO_MAXIMUM_ITERATIONS,
        pseudo_inner_maximum_iterations=FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS,
        newton_raphson_zero_start_iterations=FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS,
        line_search_maximum_attempts=FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
        step_halving_maximum_attempts=FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS,
        initial_response_scale=FIRTH_INITIAL_RESPONSE_SCALE,
        sparse_carrier_dosage_threshold=FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD,
        step_halving_scale=FIRTH_STEP_HALVING_SCALE,
        use_block_math=False,
    ),
    null_firth=NullFirthConfig(
        maximum_iterations=NULL_FIRTH_MAXIMUM_ITERATIONS,
        gradient_tolerance=NULL_FIRTH_GRADIENT_TOLERANCE,
        maximum_step_size=NULL_FIRTH_MAXIMUM_STEP_SIZE,
        fallback_iteration_multiplier=NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER,
        fallback_step_divisor=NULL_FIRTH_FALLBACK_STEP_DIVISOR,
        line_search_maximum_attempts=NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
        step_halving_scale=NULL_FIRTH_STEP_HALVING_SCALE,
    ),
)
