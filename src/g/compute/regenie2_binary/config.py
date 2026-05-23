"""Default binary REGENIE step 2 kernel policy."""

from __future__ import annotations

from dataclasses import dataclass

from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning

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


@dataclass(frozen=True)
class BinaryKernelConfig:
    """Static binary-kernel settings that affect traced JAX programs.

    Attributes:
        maximum_null_iterations: Maximum IRLS iterations for the null logistic model.
        null_logistic_coefficient_tolerance: Coefficient convergence tolerance for the null logistic model.
        minimum_probability: Logistic probability clipping floor.
        minimum_variance: Bernoulli and information-matrix variance floor.
        relative_variance_tolerance: Relative score-test variance floor multiplier.
        firth_batch_size: Fixed batch size for device-resident Firth fallback lanes.
        firth_candidate_capacity: Preferred fixed candidate capacity before falling back to full chunk capacity.
        firth_maximum_iterations: Maximum Firth solver iterations.
        firth_gradient_tolerance: Firth adjusted-score convergence tolerance.
        firth_coefficient_tolerance: Firth coefficient-step convergence tolerance.
        firth_likelihood_tolerance: Firth penalized-likelihood convergence tolerance.
        firth_maximum_step_size: Maximum absolute Firth coefficient update before step scaling.
        firth_pseudo_maximum_iterations: Maximum approximate pseudo-Firth outer iterations.
        firth_pseudo_inner_maximum_iterations: Maximum pseudo-response logistic inner iterations.
        firth_newton_raphson_zero_start_iterations: Maximum zero-start scalar Newton-Raphson iterations.
        firth_line_search_maximum_attempts: Maximum scalar/full-model Firth line-search attempts.
        firth_step_halving_maximum_attempts: Maximum full-model Firth step-halving attempts.
        firth_initial_response_scale: Pseudo-response scale for block full-model Firth initialization.
        firth_sparse_carrier_dosage_threshold: Raw dosage threshold for sparse carrier-only Firth samples.
        firth_step_halving_scale: Full-model Firth step multiplier after each rejected backtracking attempt.
        null_firth_maximum_iterations: Maximum covariate-only null Firth iterations.
        null_firth_gradient_tolerance: Covariate-only null Firth adjusted-score tolerance.
        null_firth_maximum_step_size: Covariate-only null Firth maximum coefficient step size.
        null_firth_fallback_iteration_multiplier: Multiplier for null Firth fallback retry iterations.
        null_firth_fallback_step_divisor: Divisor for null Firth fallback retry step size.
        null_firth_line_search_maximum_attempts: Maximum covariate-only null Firth line-search attempts.
        null_firth_step_halving_scale: Null Firth step multiplier after each rejected line-search attempt.
        use_block_firth_math: Whether to use the experimental block-matrix Firth path.

    """

    maximum_null_iterations: int
    null_logistic_coefficient_tolerance: float
    minimum_probability: float
    minimum_variance: float
    relative_variance_tolerance: float
    firth_batch_size: int
    firth_candidate_capacity: int
    firth_maximum_iterations: int
    firth_gradient_tolerance: float
    firth_coefficient_tolerance: float
    firth_likelihood_tolerance: float
    firth_maximum_step_size: float
    firth_pseudo_maximum_iterations: int = FIRTH_PSEUDO_MAXIMUM_ITERATIONS
    firth_pseudo_inner_maximum_iterations: int = FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS
    firth_newton_raphson_zero_start_iterations: int = FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS
    firth_line_search_maximum_attempts: int = FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS
    firth_step_halving_maximum_attempts: int = FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS
    firth_initial_response_scale: float = FIRTH_INITIAL_RESPONSE_SCALE
    firth_sparse_carrier_dosage_threshold: float = FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD
    firth_step_halving_scale: float = FIRTH_STEP_HALVING_SCALE
    null_firth_maximum_iterations: int = NULL_FIRTH_MAXIMUM_ITERATIONS
    null_firth_gradient_tolerance: float = NULL_FIRTH_GRADIENT_TOLERANCE
    null_firth_maximum_step_size: float = NULL_FIRTH_MAXIMUM_STEP_SIZE
    null_firth_fallback_iteration_multiplier: int = NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER
    null_firth_fallback_step_divisor: float = NULL_FIRTH_FALLBACK_STEP_DIVISOR
    null_firth_line_search_maximum_attempts: int = NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS
    null_firth_step_halving_scale: float = NULL_FIRTH_STEP_HALVING_SCALE
    use_block_firth_math: bool = False

    def __post_init__(self) -> None:
        """Validate positive static kernel settings."""
        if self.maximum_null_iterations <= 0:
            message = "Maximum null iterations must be positive."
            raise ValueError(message)
        if self.null_logistic_coefficient_tolerance <= 0.0:
            message = "Null logistic coefficient tolerance must be positive."
            raise ValueError(message)
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
        if self.firth_batch_size <= 0:
            message = "Firth batch size must be positive."
            raise ValueError(message)
        if self.firth_candidate_capacity <= 0:
            message = "Firth candidate capacity must be positive."
            raise ValueError(message)
        if self.firth_maximum_iterations <= 0:
            message = "Firth maximum iterations must be positive."
            raise ValueError(message)
        if self.firth_gradient_tolerance <= 0.0:
            message = "Firth gradient tolerance must be positive."
            raise ValueError(message)
        if self.firth_coefficient_tolerance <= 0.0:
            message = "Firth coefficient tolerance must be positive."
            raise ValueError(message)
        if self.firth_likelihood_tolerance <= 0.0:
            message = "Firth likelihood tolerance must be positive."
            raise ValueError(message)
        if self.firth_maximum_step_size <= 0.0:
            message = "Firth maximum step size must be positive."
            raise ValueError(message)
        if self.firth_pseudo_maximum_iterations <= 0:
            message = "Firth pseudo maximum iterations must be positive."
            raise ValueError(message)
        if self.firth_pseudo_inner_maximum_iterations <= 0:
            message = "Firth pseudo inner maximum iterations must be positive."
            raise ValueError(message)
        if self.firth_newton_raphson_zero_start_iterations <= 0:
            message = "Firth zero-start Newton-Raphson iterations must be positive."
            raise ValueError(message)
        if self.firth_line_search_maximum_attempts <= 0:
            message = "Firth line-search maximum attempts must be positive."
            raise ValueError(message)
        if self.firth_step_halving_maximum_attempts <= 0:
            message = "Firth step-halving maximum attempts must be positive."
            raise ValueError(message)
        if self.firth_initial_response_scale <= 0.0:
            message = "Firth initial response scale must be positive."
            raise ValueError(message)
        if self.firth_sparse_carrier_dosage_threshold <= 0.0:
            message = "Firth sparse carrier dosage threshold must be positive."
            raise ValueError(message)
        if self.firth_step_halving_scale <= 0.0:
            message = "Firth step-halving scale must be positive."
            raise ValueError(message)
        if self.null_firth_maximum_iterations <= 0:
            message = "Null Firth maximum iterations must be positive."
            raise ValueError(message)
        if self.null_firth_gradient_tolerance <= 0.0:
            message = "Null Firth gradient tolerance must be positive."
            raise ValueError(message)
        if self.null_firth_maximum_step_size <= 0.0:
            message = "Null Firth maximum step size must be positive."
            raise ValueError(message)
        if self.null_firth_fallback_iteration_multiplier <= 0:
            message = "Null Firth fallback iteration multiplier must be positive."
            raise ValueError(message)
        if self.null_firth_fallback_step_divisor <= 0.0:
            message = "Null Firth fallback step divisor must be positive."
            raise ValueError(message)
        if self.null_firth_line_search_maximum_attempts <= 0:
            message = "Null Firth line-search maximum attempts must be positive."
            raise ValueError(message)
        if self.null_firth_step_halving_scale <= 0.0:
            message = "Null Firth step-halving scale must be positive."
            raise ValueError(message)


DEFAULT_BINARY_KERNEL_CONFIG = BinaryKernelConfig(
    maximum_null_iterations=DEFAULT_MAXIMUM_NULL_ITERATIONS,
    null_logistic_coefficient_tolerance=NULL_LOGISTIC_COEFFICIENT_TOLERANCE,
    minimum_probability=MINIMUM_PROBABILITY,
    minimum_variance=MINIMUM_VARIANCE,
    relative_variance_tolerance=RELATIVE_VARIANCE_TOLERANCE,
    firth_batch_size=regenie2_binary_candidate_planning.DEFAULT_FIRTH_BATCH_SIZE,
    firth_candidate_capacity=regenie2_binary_candidate_planning.DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    firth_maximum_iterations=FIRTH_MAXIMUM_ITERATIONS,
    firth_gradient_tolerance=FIRTH_GRADIENT_TOLERANCE,
    firth_coefficient_tolerance=FIRTH_COEFFICIENT_TOLERANCE,
    firth_likelihood_tolerance=FIRTH_LIKELIHOOD_TOLERANCE,
    firth_maximum_step_size=FIRTH_MAXIMUM_STEP_SIZE,
    firth_pseudo_maximum_iterations=FIRTH_PSEUDO_MAXIMUM_ITERATIONS,
    firth_pseudo_inner_maximum_iterations=FIRTH_PSEUDO_INNER_MAXIMUM_ITERATIONS,
    firth_newton_raphson_zero_start_iterations=FIRTH_NEWTON_RAPHSON_ZERO_START_ITERATIONS,
    firth_line_search_maximum_attempts=FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
    firth_step_halving_maximum_attempts=FIRTH_STEP_HALVING_MAXIMUM_ATTEMPTS,
    firth_initial_response_scale=FIRTH_INITIAL_RESPONSE_SCALE,
    firth_sparse_carrier_dosage_threshold=FIRTH_SPARSE_CARRIER_DOSAGE_THRESHOLD,
    firth_step_halving_scale=FIRTH_STEP_HALVING_SCALE,
    null_firth_maximum_iterations=NULL_FIRTH_MAXIMUM_ITERATIONS,
    null_firth_gradient_tolerance=NULL_FIRTH_GRADIENT_TOLERANCE,
    null_firth_maximum_step_size=NULL_FIRTH_MAXIMUM_STEP_SIZE,
    null_firth_fallback_iteration_multiplier=NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER,
    null_firth_fallback_step_divisor=NULL_FIRTH_FALLBACK_STEP_DIVISOR,
    null_firth_line_search_maximum_attempts=NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
    null_firth_step_halving_scale=NULL_FIRTH_STEP_HALVING_SCALE,
    use_block_firth_math=False,
)
