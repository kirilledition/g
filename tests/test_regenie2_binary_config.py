from __future__ import annotations

import dataclasses
import typing

import pytest

from g import execution_plan
from g.interface import config as interface_config

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import config as regenie2_binary_config


def build_default_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build the packaged-default kernel config for tests."""
    return execution_plan.build_binary_kernel_config(interface_config.GComputeConfig())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"minimum_probability": 0.0}, "Minimum probability must be positive"),
        ({"minimum_probability": 0.5}, "Minimum probability must be less than 0.5"),
        ({"minimum_variance": 0.0}, "Minimum variance must be positive"),
        ({"relative_variance_tolerance": 0.0}, "Relative variance tolerance must be positive"),
    ],
)
def test_binary_numerical_config_rejects_invalid_values(
    overrides: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(build_default_binary_kernel_config().numerical, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"maximum_iterations": 0}, "Maximum null iterations must be positive"),
        ({"coefficient_tolerance": 0.0}, "Null logistic coefficient tolerance must be positive"),
    ],
)
def test_binary_null_logistic_config_rejects_invalid_values(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(build_default_binary_kernel_config().null_logistic, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 0}, "Firth batch size must be positive"),
        ({"candidate_capacity": 0}, "Firth candidate capacity must be positive"),
    ],
)
def test_firth_candidate_config_rejects_invalid_values(
    overrides: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(build_default_binary_kernel_config().firth_candidate, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"maximum_iterations": 0}, "Firth maximum iterations must be positive"),
        ({"gradient_tolerance": 0.0}, "Firth gradient tolerance must be positive"),
        ({"coefficient_tolerance": 0.0}, "Firth coefficient tolerance must be positive"),
        ({"likelihood_tolerance": 0.0}, "Firth likelihood tolerance must be positive"),
        ({"maximum_step_size": 0.0}, "Firth maximum step size must be positive"),
        ({"pseudo_maximum_iterations": 0}, "Firth pseudo maximum iterations must be positive"),
        ({"pseudo_inner_maximum_iterations": 0}, "Firth pseudo inner maximum iterations must be positive"),
        ({"newton_raphson_zero_start_iterations": 0}, "Firth zero-start Newton-Raphson iterations must be positive"),
        ({"line_search_maximum_attempts": 0}, "Firth line-search maximum attempts must be positive"),
        ({"step_halving_maximum_attempts": 0}, "Firth step-halving maximum attempts must be positive"),
        ({"initial_response_scale": 0.0}, "Firth initial response scale must be positive"),
        ({"sparse_carrier_dosage_threshold": 0.0}, "Firth sparse carrier dosage threshold must be positive"),
        ({"step_halving_scale": 0.0}, "Firth step-halving scale must be positive"),
    ],
)
def test_approximate_firth_config_rejects_invalid_values(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(build_default_binary_kernel_config().approximate_firth, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"maximum_iterations": 0}, "Null Firth maximum iterations must be positive"),
        ({"gradient_tolerance": 0.0}, "Null Firth gradient tolerance must be positive"),
        ({"maximum_step_size": 0.0}, "Null Firth maximum step size must be positive"),
        ({"fallback_iteration_multiplier": 0}, "Null Firth fallback iteration multiplier must be positive"),
        ({"fallback_step_divisor": 0.0}, "Null Firth fallback step divisor must be positive"),
        ({"line_search_maximum_attempts": 0}, "Null Firth line-search maximum attempts must be positive"),
        ({"step_halving_scale": 0.0}, "Null Firth step-halving scale must be positive"),
    ],
)
def test_null_firth_config_rejects_invalid_values(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(build_default_binary_kernel_config().null_firth, **overrides)
