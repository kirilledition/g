"""Shared strict numerical assertions for correctness tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def assert_absolute_difference_less_than(
    actual: npt.ArrayLike,
    reference: npt.ArrayLike,
    tolerance: float,
) -> None:
    """Assert matching nonfinite masks and strict absolute finite tolerance.

    Args:
        actual: Values produced by the implementation under test.
        reference: Values produced by an independent correctness oracle.
        tolerance: Exclusive upper bound for each absolute difference.

    Raises:
        ValueError: If the tolerance is not positive and finite.
        AssertionError: If shapes, nonfinite masks, or finite values differ.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    actual_values = np.asarray(actual, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if actual_values.shape != reference_values.shape:
        raise AssertionError(f"shape mismatch: actual {actual_values.shape}, reference {reference_values.shape}")

    np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(reference_values))
    np.testing.assert_array_equal(np.isposinf(actual_values), np.isposinf(reference_values))
    np.testing.assert_array_equal(np.isneginf(actual_values), np.isneginf(reference_values))

    finite_mask = np.isfinite(actual_values)
    finite_differences = np.abs(actual_values[finite_mask] - reference_values[finite_mask])
    if finite_differences.size == 0:
        return
    maximum_difference = float(np.max(finite_differences))
    if not bool(np.all(finite_differences < tolerance)):
        raise AssertionError(f"maximum absolute difference {maximum_difference} is not less than {tolerance}")
