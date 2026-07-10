"""Binary result constructors for REGENIE step 2 compute."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryScoreChunkResult:
    """Trait-major score-test outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        correction_code: Integer values from `types.BinaryCorrectionCode` with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    correction_code: jax.Array
