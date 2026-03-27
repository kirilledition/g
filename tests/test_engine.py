import numpy as np

from g.compute.logistic import (
    LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL,
    LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL,
    LOGISTIC_ERROR_UNFINISHED,
    LOGISTIC_METHOD_FIRTH,
)
from g.engine import format_logistic_error_codes, format_logistic_method_codes


def test_format_logistic_method_codes() -> None:
    """Test that logistic method codes are correctly mapped to FIRTH flags."""
    method_codes = np.array([LOGISTIC_METHOD_FIRTH, 999, 0, LOGISTIC_METHOD_FIRTH])
    expected = np.array(["Y", "N", "N", "Y"])

    result = format_logistic_method_codes(method_codes)

    np.testing.assert_array_equal(result, expected)


def test_format_logistic_error_codes() -> None:
    """Test that logistic error codes are correctly mapped to PLINK-style labels."""
    error_codes = np.array(
        [
            LOGISTIC_ERROR_FIRTH_CONVERGE_FAIL,
            LOGISTIC_ERROR_LOGISTIC_CONVERGE_FAIL,
            LOGISTIC_ERROR_UNFINISHED,
            0,  # No error / unknown
            999,  # Unknown
        ]
    )

    expected = np.array(
        [
            "FIRTH_CONVERGE_FAIL",
            "LOGISTIC_CONVERGE_FAIL",
            "UNFINISHED",
            ".",
            ".",
        ]
    )

    result = format_logistic_error_codes(error_codes)

    np.testing.assert_array_equal(result, expected)
