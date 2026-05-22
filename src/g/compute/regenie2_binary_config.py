"""Default binary REGENIE step 2 kernel policy."""

from __future__ import annotations

from g.compute import regenie2_binary_candidate_planning, regenie2_binary_types

MINIMUM_PROBABILITY = 1.0e-6
DEFAULT_MAXIMUM_NULL_ITERATIONS = 50
NULL_LOGISTIC_COEFFICIENT_TOLERANCE = 1.0e-6
BINARY_CASE_THRESHOLD = 0.5
FIRTH_GRADIENT_TOLERANCE = 2.5e-4
FIRTH_COEFFICIENT_TOLERANCE = 2.5e-4
FIRTH_LIKELIHOOD_TOLERANCE = 2.5e-4
FIRTH_MAXIMUM_STEP_SIZE = 5.0
FIRTH_MAXIMUM_ITERATIONS = 250
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
    use_block_firth_math=False,
)
