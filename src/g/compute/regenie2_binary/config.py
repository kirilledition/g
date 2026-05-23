"""Default binary REGENIE step 2 kernel policy."""

from __future__ import annotations

from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning
from g.compute.regenie2_binary import types as regenie2_binary_types

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
NULL_FIRTH_MAXIMUM_ITERATIONS = 1000
NULL_FIRTH_GRADIENT_TOLERANCE = 50.0e-6
NULL_FIRTH_MAXIMUM_STEP_SIZE = 25.0
NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER = 5
NULL_FIRTH_FALLBACK_STEP_DIVISOR = 5.0
NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS = 25
REGENIE_LOGISTIC_MINIMUM_ETA = -30.0
REGENIE_LOGISTIC_MAXIMUM_ETA = 30.0
REGENIE_NUMERICAL_EPSILON = 10.0 * 2.220446049250313e-16

DEFAULT_BINARY_KERNEL_CONFIG = regenie2_binary_types.BinaryKernelConfig(
    maximum_null_iterations=DEFAULT_MAXIMUM_NULL_ITERATIONS,
    null_logistic_coefficient_tolerance=NULL_LOGISTIC_COEFFICIENT_TOLERANCE,
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
    null_firth_maximum_iterations=NULL_FIRTH_MAXIMUM_ITERATIONS,
    null_firth_gradient_tolerance=NULL_FIRTH_GRADIENT_TOLERANCE,
    null_firth_maximum_step_size=NULL_FIRTH_MAXIMUM_STEP_SIZE,
    null_firth_fallback_iteration_multiplier=NULL_FIRTH_FALLBACK_ITERATION_MULTIPLIER,
    null_firth_fallback_step_divisor=NULL_FIRTH_FALLBACK_STEP_DIVISOR,
    null_firth_line_search_maximum_attempts=NULL_FIRTH_LINE_SEARCH_MAXIMUM_ATTEMPTS,
    use_block_firth_math=False,
)
