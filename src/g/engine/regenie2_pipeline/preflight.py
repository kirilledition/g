"""Preflight validation for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from g import _core


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


def emit_preflight_warnings(
    *,
    preflight_scope: str,
    preflight_report: PreflightReport,
    trusted_no_missing_diploid: bool,
) -> None:
    """Emit all non-fatal preflight warnings through native tracing."""
    native_preflight_diagnostic_policy = native_output_preflight_diagnostic_policy()
    for warning_index, warning_message in enumerate(preflight_report.warning_messages):
        native_preflight_diagnostic_policy.record_preflight_warning_diagnostic_event(
            message=warning_message,
            chromosome_count=preflight_report.chromosome_count,
            covariate_count=preflight_report.covariate_count,
            preflight_scope=preflight_scope,
            sample_count=preflight_report.sample_count,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
            warning_index=warning_index,
        )


def native_output_preflight_diagnostic_policy() -> _core.NativeOutputPreflightDiagnosticPolicy:
    """Build the native output/preflight diagnostic policy handle."""
    return _core.NativeOutputPreflightDiagnosticPolicy()


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


class BgenPreflightEngineProtocol(typing.Protocol):
    """Native BGEN engine contract required by preflight validation."""

    variant_count: int

    def required_chromosomes(self, variant_limit: int | None) -> list[str]:
        """Return chromosome labels represented in the preflight scan window."""
        ...


class SingleTraitPreflightRunInputProtocol(typing.Protocol):
    """Single-trait input arrays required by preflight validation."""

    @property
    def phenotype_vector(self) -> object:
        """Return the aligned phenotype vector."""
        ...

    @property
    def covariate_matrix(self) -> object:
        """Return the aligned covariate design matrix."""
        ...


class MultiTraitPreflightRunInputProtocol(typing.Protocol):
    """Multi-trait input arrays required by preflight validation."""

    @property
    def phenotype_matrix(self) -> object:
        """Return the aligned trait-major phenotype matrix."""
        ...

    @property
    def covariate_matrix(self) -> object:
        """Return the aligned covariate design matrix."""
        ...


class PreflightPredictionSourceProtocol(typing.Protocol):
    """Prediction source contract required by preflight validation."""

    def get_chromosome_predictions(self, chromosome: str) -> object:
        """Return LOCO predictions for one chromosome."""
        ...


def run_regenie2_preflight(
    *,
    run_input: SingleTraitPreflightRunInputProtocol,
    prediction_source: PreflightPredictionSourceProtocol,
    engine: BgenPreflightEngineProtocol,
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
        native_preflight_validator().validate_single_prediction_preflight_shape(
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
    run_input: MultiTraitPreflightRunInputProtocol,
    prediction_source: PreflightPredictionSourceProtocol,
    engine: BgenPreflightEngineProtocol,
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
        native_preflight_validator().validate_multi_prediction_preflight_shape(
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
    native_preflight_validator().validate_finite_array_values(label, values)


def validate_covariate_matrix_rank(covariate_matrix: np.ndarray, covariate_count: int) -> None:
    """Validate covariate matrix rank after native shape checks."""
    native_preflight_validator().validate_covariate_matrix_rank_array(covariate_matrix, covariate_count)


def validate_binary_phenotype(phenotype_vector: np.ndarray) -> None:
    """Validate binary phenotype coding and case/control counts."""
    native_preflight_validator().validate_binary_phenotype_array(phenotype_vector)


def resolve_single_trait_preflight_shape(
    phenotype_vector: np.ndarray,
    covariate_matrix: np.ndarray,
) -> SingleTraitPreflightShape:
    """Validate single-trait shape policy through the native engine crate."""
    payload = native_preflight_validator().validate_single_trait_preflight_shape_payload(
        shape_count(phenotype_vector.shape, 0),
        int(covariate_matrix.ndim),
        shape_count(covariate_matrix.shape, 0),
        shape_count(covariate_matrix.shape, 1),
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
    payload = native_preflight_validator().validate_multi_trait_preflight_shape_payload(
        int(phenotype_matrix.ndim),
        shape_count(phenotype_matrix.shape, 0),
        shape_count(phenotype_matrix.shape, 1),
        int(covariate_matrix.ndim),
        shape_count(covariate_matrix.shape, 0),
        shape_count(covariate_matrix.shape, 1),
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


def collect_required_chromosomes(engine: BgenPreflightEngineProtocol, variant_limit: int | None) -> tuple[str, ...]:
    """Collect chromosome labels represented in the native BGEN engine."""
    variant_count = int(engine.variant_count)
    native_preflight_validator().resolve_preflight_variant_count(variant_count, variant_limit)
    return tuple(str(chromosome) for chromosome in engine.required_chromosomes(variant_limit))


def build_preflight_report(
    *,
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
    trusted_no_missing_diploid: bool,
) -> PreflightReport:
    """Build the native-owned preflight report payload."""
    payload = native_preflight_validator().build_preflight_report_payload(
        sample_count,
        covariate_count,
        chromosome_count,
        trusted_no_missing_diploid,
    )
    return PreflightReport(
        sample_count=typing.cast("int", payload["sample_count"]),
        covariate_count=typing.cast("int", payload["covariate_count"]),
        chromosome_count=typing.cast("int", payload["chromosome_count"]),
        warning_messages=typing.cast("tuple[str, ...]", payload["warning_messages"]),
    )


def native_preflight_validator() -> _core.NativePreflightValidator:
    """Build the native preflight validator handle."""
    return _core.NativePreflightValidator()
