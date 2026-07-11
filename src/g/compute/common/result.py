"""Shared association result container."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AssociationResult[StatisticArray, CorrectionArray]:
    """Trait-major association statistics.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        correction_code: Optional correction codes with shape ``traits x variants``.

    """

    beta: StatisticArray
    standard_error: StatisticArray
    chi_squared: StatisticArray
    log10_p_value: StatisticArray
    correction_code: CorrectionArray
