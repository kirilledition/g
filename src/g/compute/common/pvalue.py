"""P-value conversion helpers shared by REGENIE compute kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats


def chi_squared_to_log10_p_value(chi_squared: jax.Array) -> jax.Array:
    """Convert chi-squared statistics to negative log10 p-values.

    Uses the exact relationship ``chi2(df=1) = Z^2`` so the survival function
    can be evaluated through the normal tail in log-space. This stays finite
    for the large statistics that would underflow through ``chi2.logsf``.

    Args:
        chi_squared: Chi-squared statistics (1 df).

    Returns:
        Negative log10 p-values (-log10(p)).

    """
    chi_squared_array = jnp.asarray(chi_squared)
    compute_dtype = jnp.result_type(chi_squared_array, jnp.float32)
    safe_chi_squared = jnp.maximum(chi_squared_array.astype(compute_dtype), jnp.asarray(0.0, dtype=compute_dtype))
    log_p_value = jnp.log(2.0) + jax.scipy.stats.norm.logsf(jnp.sqrt(safe_chi_squared))
    return -log_p_value / jnp.log(10.0)
