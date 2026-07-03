"""Compute configuration helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import config as regenie2_linear_config

type BinaryKernelConfig = regenie2_binary_config.BinaryKernelConfig
type LinearNumericalConfig = regenie2_linear_config.LinearNumericalConfig


def require_binary_kernel_config(
    kernel_config: BinaryKernelConfig | None,
) -> BinaryKernelConfig:
    """Return the binary kernel config or fail at an internal boundary."""
    if kernel_config is None:
        message = "Binary kernel config is required for binary association."
        raise ValueError(message)
    return kernel_config


def require_linear_numerical_config(
    linear_numerical_config: LinearNumericalConfig | None,
) -> LinearNumericalConfig:
    """Return linear numerical settings, using package defaults for direct pipeline calls."""
    return linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
