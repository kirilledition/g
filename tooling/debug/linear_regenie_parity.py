#!/usr/bin/env python3
"""Emit per-variant quantitative REGENIE parity diagnostics from `g` internals."""

from __future__ import annotations

import dataclasses
import json
import math
import typing
from dataclasses import dataclass

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import state as regenie2_linear_state
from g.engine import native_dispatch
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    from pathlib import Path

    import omegaconf


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


@dataclass(frozen=True)
class LinearDebugArguments:
    """Resolved parameters for quantitative per-variant debug capture.

    Attributes:
        bgen: Input BGEN path.
        sample: BGEN sample path.
        pheno_file: Phenotype table path.
        pheno_col: Quantitative phenotype column.
        covar_file: Covariate table path.
        covar_col_list: Comma-separated covariate columns.
        pred: REGENIE prediction list path.
        variant_ids: Requested variant identifiers.
        variant_indices: Requested zero-based variant indices.
        chunk_size: BGEN chunk size.
        variant_limit: Optional variant cap.
        output_json: Output JSON path.
        regenie_debug_jsonl: Optional reference debug JSONL path.
        trusted_no_missing_diploid: Whether to use the trusted BGEN path.

    """

    bgen: Path
    sample: Path
    pheno_file: Path
    pheno_col: str
    covar_file: Path
    covar_col_list: str
    pred: Path
    variant_ids: tuple[str, ...]
    variant_indices: tuple[int, ...]
    chunk_size: int
    variant_limit: int | None
    output_json: Path
    regenie_debug_jsonl: Path | None
    trusted_no_missing_diploid: bool


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


def build_selector(arguments: LinearDebugArguments) -> VariantSelector:
    """Build selected-variant matcher from CLI arguments."""
    variant_identifiers = frozenset(arguments.variant_ids)
    variant_indices = frozenset(arguments.variant_indices)
    if not variant_identifiers and not variant_indices:
        message = "Provide at least one tool.variant_ids entry or tool.variant_indices entry."
        raise ValueError(message)
    return VariantSelector(variant_identifiers=variant_identifiers, variant_indices=variant_indices)


def compute_linear_debug_arrays(
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    genotype_matrix_by_variant: jax.Array,
) -> LinearDebugArrays:
    """Compute quantitative score-test internal arrays for selected variants."""
    raw_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant,
        dtype=jnp.float32,
    )
    genotype_mean = jnp.mean(raw_genotype_matrix_by_variant, axis=1)
    normalization_offset = jnp.where(genotype_mean > 1.0, 2.0, 0.0)
    normalized_genotype_matrix_by_variant = raw_genotype_matrix_by_variant - normalization_offset[:, None]
    whitened_covariate_transpose = chromosome_state.whitened_covariate_transpose
    covariate_projection_coordinates = whitened_covariate_transpose @ normalized_genotype_matrix_by_variant.T
    raw_covariance_with_phenotype = chromosome_state.adjusted_residual @ normalized_genotype_matrix_by_variant.T
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
    chromosome_state: regenie2_linear_state.Regenie2LinearChromosomeState,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    genotype_matrix_by_variant: npt.NDArray[np.float32],
    selected_offsets: list[int],
) -> list[VariantDebugRecord]:
    """Build quantitative debug records for selected offsets from one native BGEN chunk."""
    selected_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant[selected_offsets, :],
        dtype=jnp.float32,
    )
    result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=selected_genotype_matrix_by_variant,
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
        """Initialize the callback state for selected quantitative variant capture."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.selector = selector
        self.regenie_state = regenie2_linear.prepare_regenie2_linear_state(
            jnp.asarray(self.run_input.covariate_matrix, dtype=jnp.float32),
            jnp.asarray(self.run_input.phenotype_vector, dtype=jnp.float32),
        )
        self.chromosome_states: dict[str, regenie2_linear_state.Regenie2LinearChromosomeState] = {}
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

    def prepare_chromosome_state(self, chromosome: str) -> regenie2_linear_state.Regenie2LinearChromosomeState:
        """Build or reuse the quantitative chromosome state for one chromosome."""
        if chromosome not in self.chromosome_states:
            loco_predictions = jnp.asarray(
                self.prediction_source.get_chromosome_predictions(chromosome),
                dtype=jnp.float32,
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


def capture_g_records(arguments: LinearDebugArguments, selector: VariantSelector) -> list[VariantDebugRecord]:
    """Run native BGEN streaming and capture selected `g` quantitative debug records."""
    engine = _core.Regenie2RunEngine(
        str(arguments.bgen),
        int(arguments.chunk_size),
        arguments.variant_limit,
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


def run_tool(arguments: LinearDebugArguments) -> None:
    """Run the debug harness."""
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
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(f"{json.dumps(output_payload, indent=2)}\n", encoding="utf-8")


def required_path(tool_values: dict[str, typing.Any], key: str) -> Path:
    """Return a required path from a Hydra tool config."""
    path = tooling_hydra_arguments.path_or_none(tool_values[key])
    if path is None:
        message = f"tool.{key} is required."
        raise ValueError(message)
    return path


def build_arguments_from_config(config: omegaconf.DictConfig) -> LinearDebugArguments:
    """Resolve quantitative debug parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return LinearDebugArguments(
        bgen=required_path(tool_values, "bgen"),
        sample=required_path(tool_values, "sample"),
        pheno_file=required_path(tool_values, "pheno_file"),
        pheno_col=str(tool_values["pheno_col"]),
        covar_file=required_path(tool_values, "covar_file"),
        covar_col_list=str(tool_values["covar_col_list"]),
        pred=required_path(tool_values, "pred"),
        variant_ids=tuple(str(value) for value in typing.cast("list[typing.Any]", tool_values["variant_ids"])),
        variant_indices=tuple(int(value) for value in typing.cast("list[typing.Any]", tool_values["variant_indices"])),
        chunk_size=int(tool_values["chunk_size"]),
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values["variant_limit"]),
        output_json=required_path(tool_values, "output_json"),
        regenie_debug_jsonl=tooling_hydra_arguments.path_or_none(tool_values["regenie_debug_jsonl"]),
        trusted_no_missing_diploid=tooling_hydra_arguments.boolean_value(tool_values["trusted_no_missing_diploid"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="debug_linear_regenie_parity")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run quantitative per-variant debug capture from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run quantitative per-variant debug capture from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
