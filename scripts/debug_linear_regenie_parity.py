#!/usr/bin/env python3
"""Emit per-variant quantitative REGENIE parity diagnostics from `g` internals."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core
from g.compute import regenie2_linear, regenie2_linear_types
from g.engine import native_dispatch


@dataclass(frozen=True)
class VariantSelector:
    """Selected variants for debug capture.

    Attributes:
        variant_identifiers: Variant IDs requested on the command line.
        variant_indices: Zero-based global variant indices requested on the command line.

    """

    variant_identifiers: frozenset[str]
    variant_indices: frozenset[int]

    def matches(self, *, variant_identifier: str, variant_index: int) -> bool:
        """Return whether a variant should be captured."""
        return variant_identifier in self.variant_identifiers or variant_index in self.variant_indices


@dataclass(frozen=True)
class LinearDebugArrays:
    """Quantitative score internals for selected variants."""

    allele_count: npt.NDArray[np.float64]
    normalization_offset: npt.NDArray[np.float64]
    normalized_genotype_sum_squares: npt.NDArray[np.float64]
    projection_sum_squares: npt.NDArray[np.float64]
    genotype_residual_sum_squares: npt.NDArray[np.float64]
    covariance_with_phenotype: npt.NDArray[np.float64]
    null_mean_squared_error: float
    adjusted_residual_sum_squares: float


@dataclass(frozen=True)
class VariantDebugRecord:
    """Serializable quantitative diagnostics for one selected variant."""

    variant_index: int
    chromosome: str
    position: int
    variant_identifier: str
    allele_zero: str
    allele_one: str
    allele_count: float
    allele_one_frequency: float
    minor_allele_count: float
    info_score: float
    observation_count: int
    sparse_candidate: bool
    normalization_offset: float
    normalized_genotype_sum_squares: float
    projection_sum_squares: float
    genotype_residual_sum_squares: float
    covariance_with_phenotype: float
    null_mean_squared_error: float
    adjusted_residual_sum_squares: float
    adjusted_residual: dict[str, float]
    adjusted_residual_projection: dict[str, float]
    beta: float | None
    standard_error: float | None
    chi_squared: float | None
    log10_p_value: float | None
    valid: bool


@dataclass(frozen=True)
class NumericDifference:
    """One numeric difference between `g` and REGENIE debug records."""

    path: str
    g_value: float
    reference_value: float
    absolute_error: float


@dataclass(frozen=True)
class VariantDebugComparison:
    """Comparison for one variant against optional REGENIE debug JSON."""

    variant_identifier: str
    g_record: dict[str, typing.Any]
    reference_record: dict[str, typing.Any] | None
    differences: list[NumericDifference]
    missing_reference: bool


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the per-variant debug harness."""
    parser = argparse.ArgumentParser(description="Capture quantitative REGENIE step 2 parity diagnostics.")
    parser.add_argument("--bgen", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--pheno-file", type=Path, required=True)
    parser.add_argument("--pheno-col", required=True)
    parser.add_argument("--covar-file", type=Path, required=True)
    parser.add_argument("--covar-col-list", required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--variant-id", action="append", default=[])
    parser.add_argument("--variant-index", action="append", type=int, default=[])
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--variant-limit", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--regenie-debug-jsonl", type=Path)
    parser.add_argument("--trusted-no-missing-diploid", action="store_true")
    return parser


def finite_float_or_none(value: typing.Any) -> float | None:
    """Convert finite numeric values to float and non-finite values to null."""
    float_value = float(value)
    if math.isfinite(float_value):
        return float_value
    return None


def summarize_array(values: jax.Array | npt.NDArray[np.floating[typing.Any]]) -> dict[str, float]:
    """Build a compact numeric summary for a vector."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "sum": float(np.sum(array)),
        "l2_norm": float(np.linalg.norm(array)),
    }


def parse_covar_column_list(raw_column_list: str) -> tuple[str, ...]:
    """Parse comma-separated covariate names."""
    return tuple(column_name.strip() for column_name in raw_column_list.split(",") if column_name.strip())


def build_selector(arguments: argparse.Namespace) -> VariantSelector:
    """Build selected-variant matcher from CLI arguments."""
    variant_identifiers = frozenset(typing.cast("list[str]", arguments.variant_id))
    variant_indices = frozenset(typing.cast("list[int]", arguments.variant_index))
    if not variant_identifiers and not variant_indices:
        message = "Provide at least one --variant-id or --variant-index."
        raise ValueError(message)
    return VariantSelector(variant_identifiers=variant_identifiers, variant_indices=variant_indices)


def compute_linear_debug_arrays(
    chromosome_state: regenie2_linear_types.Regenie2LinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> LinearDebugArrays:
    """Compute quantitative score-test internal arrays for selected variants."""
    raw_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant,
        dtype=regenie2_linear.LINEAR_COMPUTE_DTYPE,
    )
    genotype_mean = jnp.mean(raw_genotype_matrix_by_variant, axis=1)
    normalization_offset = jnp.where(genotype_mean > 1.0, 2.0, 0.0)
    normalized_genotype_matrix_by_variant = raw_genotype_matrix_by_variant - normalization_offset[:, None]
    whitened_covariate_transpose = chromosome_state.stacked_score_matrix[:-1]
    covariate_projection_coordinates = whitened_covariate_transpose @ normalized_genotype_matrix_by_variant.T
    raw_covariance_with_phenotype = chromosome_state.stacked_score_matrix[-1] @ normalized_genotype_matrix_by_variant.T
    covariance_with_phenotype = raw_covariance_with_phenotype - (
        chromosome_state.adjusted_residual_projection_coordinates @ covariate_projection_coordinates
    )
    normalized_genotype_sum_squares = jnp.einsum(
        "ij,ij->i",
        normalized_genotype_matrix_by_variant,
        normalized_genotype_matrix_by_variant,
    )
    projection_sum_squares = jnp.einsum(
        "ij,ij->j",
        covariate_projection_coordinates,
        covariate_projection_coordinates,
    )
    genotype_residual_sum_squares = jnp.maximum(normalized_genotype_sum_squares - projection_sum_squares, 0.0)
    null_mean_squared_error = chromosome_state.adjusted_residual_sum_squares / chromosome_state.degrees_of_freedom
    host_values = jax.device_get(
        {
            "allele_count": jnp.sum(raw_genotype_matrix_by_variant, axis=1).astype(jnp.float64),
            "normalization_offset": normalization_offset.astype(jnp.float64),
            "normalized_genotype_sum_squares": normalized_genotype_sum_squares.astype(jnp.float64),
            "projection_sum_squares": projection_sum_squares.astype(jnp.float64),
            "genotype_residual_sum_squares": genotype_residual_sum_squares.astype(jnp.float64),
            "covariance_with_phenotype": covariance_with_phenotype.astype(jnp.float64),
            "null_mean_squared_error": null_mean_squared_error.astype(jnp.float64),
            "adjusted_residual_sum_squares": chromosome_state.adjusted_residual_sum_squares.astype(jnp.float64),
        }
    )
    return LinearDebugArrays(
        allele_count=np.asarray(host_values["allele_count"], dtype=np.float64),
        normalization_offset=np.asarray(host_values["normalization_offset"], dtype=np.float64),
        normalized_genotype_sum_squares=np.asarray(host_values["normalized_genotype_sum_squares"], dtype=np.float64),
        projection_sum_squares=np.asarray(host_values["projection_sum_squares"], dtype=np.float64),
        genotype_residual_sum_squares=np.asarray(host_values["genotype_residual_sum_squares"], dtype=np.float64),
        covariance_with_phenotype=np.asarray(host_values["covariance_with_phenotype"], dtype=np.float64),
        null_mean_squared_error=float(host_values["null_mean_squared_error"]),
        adjusted_residual_sum_squares=float(host_values["adjusted_residual_sum_squares"]),
    )


def build_debug_records_for_chunk(
    *,
    chromosome_state: regenie2_linear_types.Regenie2LinearChromosomeState,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    genotype_matrix_by_variant: npt.NDArray[np.float32],
    selected_offsets: list[int],
) -> list[VariantDebugRecord]:
    """Build quantitative debug records for selected offsets from one native BGEN chunk."""
    selected_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant[selected_offsets, :],
        dtype=regenie2_linear.LINEAR_COMPUTE_DTYPE,
    )
    result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=selected_genotype_matrix_by_variant,
        genotype_sum_squares=jnp.asarray(chunk_stats.imputed_dosage_square_sum[selected_offsets]),
    )
    host_result = jax.device_get(result)
    debug_arrays = compute_linear_debug_arrays(chromosome_state, selected_genotype_matrix_by_variant)
    adjusted_residual = jax.device_get(chromosome_state.adjusted_residual)
    adjusted_residual_projection = jax.device_get(chromosome_state.adjusted_residual_projection_coordinates)
    records: list[VariantDebugRecord] = []
    for result_offset, chunk_offset in enumerate(selected_offsets):
        variant_index = int(metadata.variant_start_index + chunk_offset)
        records.append(
            VariantDebugRecord(
                variant_index=variant_index,
                chromosome=str(metadata.chromosome[chunk_offset]),
                position=int(metadata.position[chunk_offset]),
                variant_identifier=str(metadata.variant_identifiers[chunk_offset]),
                allele_zero=str(metadata.allele_two[chunk_offset]),
                allele_one=str(metadata.allele_one[chunk_offset]),
                allele_count=float(debug_arrays.allele_count[result_offset]),
                allele_one_frequency=float(chunk_stats.allele_one_frequency[chunk_offset]),
                minor_allele_count=float(chunk_stats.minor_allele_count[chunk_offset]),
                info_score=float(chunk_stats.info_score[chunk_offset]),
                observation_count=int(chunk_stats.observation_count[chunk_offset]),
                sparse_candidate=bool(chunk_stats.is_sparse_candidate[chunk_offset]),
                normalization_offset=float(debug_arrays.normalization_offset[result_offset]),
                normalized_genotype_sum_squares=float(debug_arrays.normalized_genotype_sum_squares[result_offset]),
                projection_sum_squares=float(debug_arrays.projection_sum_squares[result_offset]),
                genotype_residual_sum_squares=float(debug_arrays.genotype_residual_sum_squares[result_offset]),
                covariance_with_phenotype=float(debug_arrays.covariance_with_phenotype[result_offset]),
                null_mean_squared_error=debug_arrays.null_mean_squared_error,
                adjusted_residual_sum_squares=debug_arrays.adjusted_residual_sum_squares,
                adjusted_residual=summarize_array(adjusted_residual),
                adjusted_residual_projection=summarize_array(adjusted_residual_projection),
                beta=finite_float_or_none(host_result.beta[result_offset]),
                standard_error=finite_float_or_none(host_result.standard_error[result_offset]),
                chi_squared=finite_float_or_none(host_result.chi_squared[result_offset]),
                log10_p_value=finite_float_or_none(host_result.log10_p_value[result_offset]),
                valid=bool(host_result.valid_mask[result_offset]),
            )
        )
    return records


class LinearVariantDebugCaptureCallback:
    """Native BGEN callback that captures selected quantitative debug records."""

    def __init__(
        self,
        *,
        run_input: native_dispatch.NativeBgenRunInput,
        prediction_source: _core.RegeniePredictionSource,
        selector: VariantSelector,
    ) -> None:
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.selector = selector
        self.regenie_state = regenie2_linear.prepare_regenie2_linear_state(
            self.run_input.covariate_matrix,
            self.run_input.phenotype_vector,
        )
        self.chromosome_states: dict[str, regenie2_linear_types.Regenie2LinearChromosomeState] = {}
        self.records: list[VariantDebugRecord] = []
        self.free_buffers: list[npt.NDArray[np.float32]] = []

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> npt.NDArray[np.float32]:
        """Return a reusable C-contiguous variant-major dosage buffer."""
        expected_shape = (variant_count, sample_count)
        while self.free_buffers:
            candidate_buffer = self.free_buffers.pop()
            if candidate_buffer.shape == expected_shape:
                return candidate_buffer
        return np.empty(expected_shape, dtype=np.float32, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Capture selected variants from one native BGEN chunk."""
        selected_offsets_by_chromosome: dict[str, list[int]] = {}
        for chunk_offset, variant_identifier in enumerate(metadata.variant_identifiers):
            variant_index = int(metadata.variant_start_index + chunk_offset)
            if self.selector.matches(variant_identifier=str(variant_identifier), variant_index=variant_index):
                chromosome = str(metadata.chromosome[chunk_offset])
                selected_offsets_by_chromosome.setdefault(chromosome, []).append(chunk_offset)
        for chromosome, selected_offsets in selected_offsets_by_chromosome.items():
            chromosome_state = self.prepare_chromosome_state(chromosome)
            self.records.extend(
                build_debug_records_for_chunk(
                    chromosome_state=chromosome_state,
                    metadata=metadata,
                    chunk_stats=chunk_stats,
                    genotype_matrix_by_variant=genotype_matrix_by_variant,
                    selected_offsets=selected_offsets,
                )
            )
        self.free_buffers.append(genotype_matrix_by_variant)

    def prepare_chromosome_state(self, chromosome: str) -> regenie2_linear_types.Regenie2LinearChromosomeState:
        """Build or reuse the quantitative chromosome state for one chromosome."""
        if chromosome not in self.chromosome_states:
            loco_predictions = jnp.asarray(
                self.prediction_source.get_chromosome_predictions(chromosome),
                dtype=regenie2_linear.LINEAR_COMPUTE_DTYPE,
            )
            self.chromosome_states[chromosome] = regenie2_linear.prepare_regenie2_linear_chromosome_state(
                self.regenie_state,
                loco_predictions,
            )
        return self.chromosome_states[chromosome]


def load_reference_debug_records(reference_path: Path | None) -> dict[str, dict[str, typing.Any]]:
    """Load optional REGENIE debug JSONL keyed by variant identifier."""
    if reference_path is None:
        return {}
    records: dict[str, dict[str, typing.Any]] = {}
    for line in reference_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        variant_identifier = str(record.get("variant_identifier", record.get("variant_id", record.get("id", ""))))
        if not variant_identifier:
            message = f"Reference debug record lacks variant identifier: {line}"
            raise ValueError(message)
        records[variant_identifier] = typing.cast("dict[str, typing.Any]", record)
    return records


def collect_numeric_differences(
    *,
    g_value: typing.Any,
    reference_value: typing.Any,
    path: str,
    tolerance: float,
) -> list[NumericDifference]:
    """Recursively collect common numeric fields that differ beyond tolerance."""
    if isinstance(g_value, dict) and isinstance(reference_value, dict):
        differences: list[NumericDifference] = []
        for key in sorted(set(g_value) & set(reference_value)):
            differences.extend(
                collect_numeric_differences(
                    g_value=g_value[key],
                    reference_value=reference_value[key],
                    path=f"{path}.{key}" if path else str(key),
                    tolerance=tolerance,
                )
            )
        return differences
    if isinstance(g_value, int | float) and isinstance(reference_value, int | float):
        g_float = float(g_value)
        reference_float = float(reference_value)
        if math.isfinite(g_float) and math.isfinite(reference_float):
            absolute_error = abs(g_float - reference_float)
            if absolute_error > tolerance:
                return [
                    NumericDifference(
                        path=path,
                        g_value=g_float,
                        reference_value=reference_float,
                        absolute_error=absolute_error,
                    )
                ]
    return []


def build_comparisons(
    *,
    records: list[VariantDebugRecord],
    reference_records: dict[str, dict[str, typing.Any]],
    tolerance: float = 1.0e-8,
) -> list[VariantDebugComparison]:
    """Compare captured `g` records with optional reference records."""
    comparisons: list[VariantDebugComparison] = []
    for record in records:
        g_record = dataclasses.asdict(record)
        reference_record = reference_records.get(record.variant_identifier)
        differences = (
            []
            if reference_record is None
            else collect_numeric_differences(
                g_value=g_record,
                reference_value=reference_record,
                path="",
                tolerance=tolerance,
            )
        )
        comparisons.append(
            VariantDebugComparison(
                variant_identifier=record.variant_identifier,
                g_record=g_record,
                reference_record=reference_record,
                differences=differences,
                missing_reference=reference_record is None,
            )
        )
    return comparisons


def count_missing_selections(*, records: list[VariantDebugRecord], selector: VariantSelector) -> int:
    """Count requested IDs and indices that were not found in captured records."""
    matched_identifiers = {record.variant_identifier for record in records}
    matched_indices = {record.variant_index for record in records}
    missing_identifiers = selector.variant_identifiers - matched_identifiers
    missing_indices = selector.variant_indices - matched_indices
    return len(missing_identifiers) + len(missing_indices)


def capture_g_records(arguments: argparse.Namespace, selector: VariantSelector) -> list[VariantDebugRecord]:
    """Run native BGEN streaming and capture selected `g` quantitative debug records."""
    engine = _core.Regenie2RunEngine(
        str(arguments.bgen),
        int(arguments.chunk_size),
        typing.cast("int | None", arguments.variant_limit),
        bool(arguments.trusted_no_missing_diploid),
    )
    run_input = native_dispatch.build_native_bgen_run_input(
        engine.align_sample_data(
            sample_path=str(arguments.sample),
            phenotype_path=str(arguments.pheno_file),
            phenotype_name=str(arguments.pheno_col),
            covariate_path=str(arguments.covar_file),
            covariate_names=list(parse_covar_column_list(str(arguments.covar_col_list))),
            is_binary_trait=False,
        )
    )
    prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(arguments.pred),
        str(arguments.pheno_col),
        run_input.native_aligned_sample_data,
    )
    callback = LinearVariantDebugCaptureCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        selector=selector,
    )
    engine.run_bgen_variant_major_dosage_buffered_chunks(run_input.sample_indices, callback)
    return sorted(callback.records, key=lambda record: record.variant_index)


def main() -> None:
    """Run the debug harness."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    selector = build_selector(arguments)
    records = capture_g_records(arguments, selector)
    reference_records = load_reference_debug_records(arguments.regenie_debug_jsonl)
    comparisons = build_comparisons(records=records, reference_records=reference_records)
    output_payload = {
        "records": [dataclasses.asdict(record) for record in records],
        "comparisons": [
            {
                "variant_identifier": comparison.variant_identifier,
                "missing_reference": comparison.missing_reference,
                "differences": [dataclasses.asdict(difference) for difference in comparison.differences],
                "reference_record": comparison.reference_record,
            }
            for comparison in comparisons
        ],
        "missing_selection_count": count_missing_selections(records=records, selector=selector),
        "linear_compute_dtype": str(regenie2_linear.LINEAR_COMPUTE_DTYPE),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(f"{json.dumps(output_payload, indent=2)}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
