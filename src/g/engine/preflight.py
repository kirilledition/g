"""Preflight validation for REGENIE step 2 execution."""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreflightReport:
    """Summary of preflight validation.

    Attributes:
        sample_count: Number of samples entering association testing.
        covariate_count: Number of covariate columns in the null model.
        chromosome_count: Number of BGEN chromosomes requiring LOCO predictions.
        warning_messages: Non-fatal warnings emitted during validation.

    """

    sample_count: int
    covariate_count: int
    chromosome_count: int
    warning_messages: tuple[str, ...]


def run_regenie2_preflight(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    engine: typing.Any,
    variant_limit: int | None,
    is_binary_trait: bool,
    trusted_no_missing_diploid: bool,
) -> PreflightReport:
    """Validate aligned REGENIE step 2 inputs before chunk execution.

    Raises:
        ValueError: If input data are incompatible with REGENIE step 2 execution.

    """
    phenotype_vector = np.asarray(run_input.phenotype_vector)
    covariate_matrix = np.asarray(run_input.covariate_matrix)
    sample_count = int(phenotype_vector.shape[0])
    covariate_count = int(covariate_matrix.shape[1]) if covariate_matrix.ndim == 2 else 0
    validate_finite_array("Phenotype", phenotype_vector)
    validate_finite_array("Covariate matrix", covariate_matrix)
    validate_covariate_matrix(covariate_matrix, sample_count)
    if is_binary_trait:
        validate_binary_phenotype(phenotype_vector)

    required_chromosomes = collect_required_chromosomes(engine, variant_limit)
    for chromosome in required_chromosomes:
        prediction_values = np.asarray(prediction_source.get_chromosome_predictions(chromosome))
        if prediction_values.shape[0] != sample_count:
            message = (
                f"Prediction sample count for chromosome {chromosome} is {prediction_values.shape[0]}, "
                f"expected {sample_count}."
            )
            raise ValueError(message)
        validate_finite_array(f"Prediction values for chromosome {chromosome}", prediction_values)

    warning_messages = build_preflight_warnings(
        sample_count=sample_count,
        covariate_count=covariate_count,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    for warning_message in warning_messages:
        logger.warning("%s", warning_message)
    return PreflightReport(
        sample_count=sample_count,
        covariate_count=covariate_count,
        chromosome_count=len(required_chromosomes),
        warning_messages=warning_messages,
    )


def validate_finite_array(label: str, values: np.ndarray) -> None:
    """Validate that an array contains only finite values."""
    if np.isfinite(values).all():
        return
    message = f"{label} contains non-finite values."
    raise ValueError(message)


def validate_covariate_matrix(covariate_matrix: np.ndarray, sample_count: int) -> None:
    """Validate covariate shape, rank, and model degrees of freedom."""
    if covariate_matrix.ndim != 2:
        message = "Covariate matrix must be two-dimensional."
        raise ValueError(message)
    if covariate_matrix.shape[0] != sample_count:
        message = "Covariate matrix sample count does not match phenotype sample count."
        raise ValueError(message)
    covariate_count = int(covariate_matrix.shape[1])
    if sample_count <= covariate_count:
        message = "Sample count must exceed the number of covariate degrees of freedom."
        raise ValueError(message)
    rank = int(np.linalg.matrix_rank(covariate_matrix))
    if rank < covariate_count:
        message = "Covariate matrix is rank deficient."
        raise ValueError(message)


def validate_binary_phenotype(phenotype_vector: np.ndarray) -> None:
    """Validate binary phenotype coding and case/control counts."""
    unique_values = {float(value) for value in np.unique(phenotype_vector)}
    if not unique_values.issubset({0.0, 1.0}):
        message = "Binary phenotype must be coded as 0/1 after alignment."
        raise ValueError(message)
    control_count = int(np.count_nonzero(phenotype_vector == 0.0))
    case_count = int(np.count_nonzero(phenotype_vector == 1.0))
    if control_count == 0 or case_count == 0:
        message = "Binary phenotype must contain at least one case and one control."
        raise ValueError(message)


def collect_required_chromosomes(engine: typing.Any, variant_limit: int | None) -> tuple[str, ...]:
    """Collect chromosome labels represented in the native BGEN engine."""
    variant_count = int(engine.variant_count)
    if variant_count <= 0:
        message = "BGEN input contains no variants."
        raise ValueError(message)
    scanned_variant_count = variant_count if variant_limit is None else min(variant_count, variant_limit)
    if scanned_variant_count <= 0:
        message = "BGEN scan contains no variants."
        raise ValueError(message)
    chromosome_values, _, _, _, _ = engine.variant_metadata_slice(0, scanned_variant_count)
    required_chromosomes: list[str] = []
    seen_chromosomes: set[str] = set()
    for chromosome_value in chromosome_values:
        chromosome = str(chromosome_value)
        if chromosome in seen_chromosomes:
            continue
        seen_chromosomes.add(chromosome)
        required_chromosomes.append(chromosome)
    return tuple(required_chromosomes)


def build_preflight_warnings(
    *,
    sample_count: int,
    covariate_count: int,
    trusted_no_missing_diploid: bool,
) -> tuple[str, ...]:
    """Build non-fatal preflight warnings."""
    warning_messages: list[str] = []
    residual_degrees_of_freedom = sample_count - covariate_count
    if residual_degrees_of_freedom < 10:
        warning_messages.append("REGENIE step 2 is running with fewer than 10 residual degrees of freedom.")
    if trusted_no_missing_diploid:
        warning_messages.append("Trusted no-missing diploid BGEN path is enabled after compatibility validation.")
    return tuple(warning_messages)
