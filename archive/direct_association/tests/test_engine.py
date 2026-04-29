import numpy as np
from g.compute.logistic import (
    LogisticErrorCode,
    LogisticMethod,
)

from g.engine import format_logistic_error_codes, format_logistic_method_codes


def test_format_logistic_error_codes():
    """Test that logistic error codes are correctly formatted."""
    input_codes = np.array(
        [
            LogisticErrorCode.FIRTH_CONVERGE_FAIL,
            LogisticErrorCode.LOGISTIC_CONVERGE_FAIL,
            LogisticErrorCode.UNFINISHED,
            LogisticErrorCode.NONE,
            999,  # Some unknown code
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

    result = format_logistic_error_codes(input_codes)

    np.testing.assert_array_equal(result, expected)


def test_format_logistic_method_codes() -> None:
    """Test that logistic method codes are correctly mapped to FIRTH flags."""
    method_codes = np.array([LogisticMethod.FIRTH, 999, LogisticMethod.STANDARD, LogisticMethod.FIRTH])
    expected = np.array(["Y", "N", "N", "Y"])

    result = format_logistic_method_codes(method_codes)

    np.testing.assert_array_equal(result, expected)
