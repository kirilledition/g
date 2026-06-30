"""Preflight validation for REGENIE step 2 execution."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

import g
from g.engine import run_events


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


@dataclass(frozen=True)
class SingleTraitPreflightShape:
    """Native-owned shape payload for single-trait preflight.

    Attributes:
        sample_count: Number of samples entering association testing.
        covariate_count: Number of covariate columns in the null model.

    """

    sample_count: int
    covariate_count: int


def emit_preflight_warning_diagnostic_event_payload(payload: typing.Mapping[str, object]) -> None:
    """Emit one native preflight warning diagnostic payload through native tracing."""
    g._core.emit_diagnostic_event_fields(
        str(payload["level"]),
        str(payload["event_name"]),
        str(payload["message"]),
        typing.cast("typing.Mapping[str, object]", payload["fields"]),
    )


def emit_preflight_warnings(
    *,
    preflight_scope: str,
    preflight_report: PreflightReport,
    trusted_no_missing_diploid: bool,
) -> None:
    """Emit all non-fatal preflight warnings through native tracing."""
    for warning_index, warning_message in enumerate(preflight_report.warning_messages):
        emit_preflight_warning_diagnostic_event_payload(
            run_events.build_preflight_warning_diagnostic_payload(
                message=warning_message,
                chromosome_count=preflight_report.chromosome_count,
                covariate_count=preflight_report.covariate_count,
                preflight_scope=preflight_scope,
                sample_count=preflight_report.sample_count,
                trusted_no_missing_diploid=trusted_no_missing_diploid,
                warning_index=warning_index,
            )
        )


@dataclass(frozen=True)
class MultiTraitPreflightShape:
    """Native-owned shape payload for multi-trait preflight.

    Attributes:
        trait_count: Number of traits represented in the phenotype matrix.
        sample_count: Number of samples entering association testing.
        covariate_count: Number of covariate columns in the null model.

    """

    trait_count: int
    sample_count: int
    covariate_count: int


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
    validate_finite_array("Phenotype", phenotype_vector)
    validate_finite_array("Covariate matrix", covariate_matrix)
    preflight_shape = resolve_single_trait_preflight_shape(phenotype_vector, covariate_matrix)
    sample_count = preflight_shape.sample_count
    covariate_count = preflight_shape.covariate_count
    validate_covariate_matrix_rank(covariate_matrix, covariate_count)
    if is_binary_trait:
        validate_binary_phenotype(phenotype_vector)

    required_chromosomes = collect_required_chromosomes(engine, variant_limit)
    for chromosome in required_chromosomes:
        prediction_values = np.asarray(prediction_source.get_chromosome_predictions(chromosome))
        g._core.validate_single_prediction_preflight_shape(
            chromosome,
            array_shape_counts(prediction_values),
            sample_count,
        )
        validate_finite_array(f"Prediction values for chromosome {chromosome}", prediction_values)

    preflight_report = build_preflight_report(
        sample_count=sample_count,
        covariate_count=covariate_count,
        chromosome_count=len(required_chromosomes),
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    emit_preflight_warnings(
        preflight_scope="single_trait",
        preflight_report=preflight_report,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    return preflight_report


def run_regenie2_multi_preflight(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    engine: typing.Any,
    variant_limit: int | None,
    is_binary_trait: bool,
    trusted_no_missing_diploid: bool,
) -> PreflightReport:
    """Validate shared and trait-major inputs before multi-trait chunk execution.

    Raises:
        ValueError: If input data are incompatible with REGENIE step 2 execution.

    """
    phenotype_matrix = np.asarray(run_input.phenotype_matrix)
    covariate_matrix = np.asarray(run_input.covariate_matrix)
    preflight_shape = resolve_multi_trait_preflight_shape(phenotype_matrix, covariate_matrix)
    trait_count = preflight_shape.trait_count
    sample_count = preflight_shape.sample_count
    covariate_count = preflight_shape.covariate_count
    validate_finite_array("Phenotype matrix", phenotype_matrix)
    validate_finite_array("Covariate matrix", covariate_matrix)
    validate_covariate_matrix_rank(covariate_matrix, covariate_count)
    if is_binary_trait:
        for trait_index in range(trait_count):
            validate_binary_phenotype(phenotype_matrix[trait_index])

    required_chromosomes = collect_required_chromosomes(engine, variant_limit)
    for chromosome in required_chromosomes:
        prediction_matrix = np.asarray(prediction_source.get_chromosome_predictions(chromosome))
        g._core.validate_multi_prediction_preflight_shape(
            chromosome,
            array_shape_counts(prediction_matrix),
            trait_count,
            sample_count,
        )
        validate_finite_array(f"Prediction matrix for chromosome {chromosome}", prediction_matrix)

    preflight_report = build_preflight_report(
        sample_count=sample_count,
        covariate_count=covariate_count,
        chromosome_count=len(required_chromosomes),
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    emit_preflight_warnings(
        preflight_scope="multi_trait",
        preflight_report=preflight_report,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    return preflight_report


def validate_finite_array(label: str, values: np.ndarray) -> None:
    """Validate that an array contains only finite values."""
    g._core.validate_finite_array(label, bool(np.isfinite(values).all()))


def validate_covariate_matrix_rank(covariate_matrix: np.ndarray, covariate_count: int) -> None:
    """Validate covariate matrix rank after native shape checks."""
    rank = int(np.linalg.matrix_rank(covariate_matrix))
    g._core.validate_covariate_matrix_rank(rank, covariate_count)


def validate_binary_phenotype(phenotype_vector: np.ndarray) -> None:
    """Validate binary phenotype coding and case/control counts."""
    unique_values = {float(value) for value in np.unique(phenotype_vector)}
    g._core.validate_binary_phenotype_coding(unique_values.issubset({0.0, 1.0}))
    control_count = int(np.count_nonzero(phenotype_vector == 0.0))
    case_count = int(np.count_nonzero(phenotype_vector == 1.0))
    g._core.validate_binary_phenotype_case_control_counts(case_count, control_count)


def resolve_single_trait_preflight_shape(
    phenotype_vector: np.ndarray,
    covariate_matrix: np.ndarray,
) -> SingleTraitPreflightShape:
    """Validate single-trait shape policy through the native engine crate."""
    payload = typing.cast(
        "dict[str, object]",
        g._core.validate_single_trait_preflight_shape_payload(
            shape_count(phenotype_vector.shape, 0),
            int(covariate_matrix.ndim),
            shape_count(covariate_matrix.shape, 0),
            shape_count(covariate_matrix.shape, 1),
        ),
    )
    return SingleTraitPreflightShape(
        sample_count=typing.cast("int", payload["sample_count"]),
        covariate_count=typing.cast("int", payload["covariate_count"]),
    )


def resolve_multi_trait_preflight_shape(
    phenotype_matrix: np.ndarray,
    covariate_matrix: np.ndarray,
) -> MultiTraitPreflightShape:
    """Validate multi-trait shape policy through the native engine crate."""
    payload = typing.cast(
        "dict[str, object]",
        g._core.validate_multi_trait_preflight_shape_payload(
            int(phenotype_matrix.ndim),
            shape_count(phenotype_matrix.shape, 0),
            shape_count(phenotype_matrix.shape, 1),
            int(covariate_matrix.ndim),
            shape_count(covariate_matrix.shape, 0),
            shape_count(covariate_matrix.shape, 1),
        ),
    )
    return MultiTraitPreflightShape(
        trait_count=typing.cast("int", payload["trait_count"]),
        sample_count=typing.cast("int", payload["sample_count"]),
        covariate_count=typing.cast("int", payload["covariate_count"]),
    )


def shape_count(array_shape: tuple[int, ...], dimension_index: int) -> int:
    """Return a shape dimension count or zero when the dimension is absent."""
    if dimension_index >= len(array_shape):
        return 0
    return int(array_shape[dimension_index])


def array_shape_counts(values: np.ndarray) -> tuple[int, ...]:
    """Return array shape counts as plain Python integers."""
    return tuple(int(dimension_count) for dimension_count in values.shape)


def collect_required_chromosomes(engine: typing.Any, variant_limit: int | None) -> tuple[str, ...]:
    """Collect chromosome labels represented in the native BGEN engine."""
    variant_count = int(engine.variant_count)
    scanned_variant_count = int(g._core.resolve_preflight_variant_count(variant_count, variant_limit))
    native_required_chromosomes = getattr(engine, "required_chromosomes", None)
    if callable(native_required_chromosomes):
        return tuple(str(chromosome) for chromosome in native_required_chromosomes(variant_limit))
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


def build_preflight_report(
    *,
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
    trusted_no_missing_diploid: bool,
) -> PreflightReport:
    """Build the native-owned preflight report payload."""
    payload = typing.cast(
        "dict[str, object]",
        g._core.build_preflight_report_payload(
            sample_count,
            covariate_count,
            chromosome_count,
            trusted_no_missing_diploid,
        ),
    )
    return PreflightReport(
        sample_count=typing.cast("int", payload["sample_count"]),
        covariate_count=typing.cast("int", payload["covariate_count"]),
        chromosome_count=typing.cast("int", payload["chromosome_count"]),
        warning_messages=typing.cast("tuple[str, ...]", payload["warning_messages"]),
    )
