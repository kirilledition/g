#!/usr/bin/env python3
"""Emit per-variant binary REGENIE parity diagnostics from `g` internals."""

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

from g import _core, execution_plan, types
from g.compute.common import genotype
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.engine import native_dispatch
from g.interface import config as interface_config


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
class ScoreDebugArrays:
    """Score-test internals for a selected genotype matrix.

    Attributes:
        score: Score numerator for each selected variant.
        variance: Projected score-test variance for each selected variant.
        allele_count: Raw tested-allele count for each selected variant.
        flipped_allele_count: REGENIE-coded allele count for each selected variant.
        flip_mask: Whether REGENIE minor-allele coding was used.
        carrier_count: Count of samples with non-zero REGENIE-coded dosage.

    """

    score: npt.NDArray[np.float64]
    variance: npt.NDArray[np.float64]
    allele_count: npt.NDArray[np.float64]
    flipped_allele_count: npt.NDArray[np.float64]
    flip_mask: npt.NDArray[np.bool_]
    carrier_count: npt.NDArray[np.int64]


@dataclass(frozen=True)
class VariantDebugRecord:
    """Serializable diagnostics for one selected variant.

    Attributes:
        variant_index: Zero-based global variant index.
        chromosome: Chromosome label.
        position: Genomic position.
        variant_identifier: Variant ID.
        allele_zero: Output allele 0.
        allele_one: Output allele 1.
        allele_count: Raw allele-one dosage sum.
        flipped_allele_count: REGENIE-coded allele-one dosage sum.
        flip_mask: Whether REGENIE flipped genotype coding before testing.
        minor_allele_count: Native minor allele count summary.
        sparse_candidate: Native sparse candidate flag.
        rare_sparse_firth_candidate: Native rare-sparse Firth candidate flag.
        carrier_count: Count of carriers after REGENIE coding.
        score: Score numerator.
        score_variance: Score-test variance after covariate projection.
        score_beta: Score-test beta.
        score_standard_error: Score-test standard error.
        score_chi_squared: Score-test chi-squared statistic.
        score_log10_p_value: Score-test negative log10 p-value.
        score_extra_code: Internal score-test extra code.
        null_logistic_offset: Summary of the ordinary null logistic offset.
        null_firth_offset: Summary of the approximate-Firth null offset.
        null_logistic_iteration_count: Ordinary null logistic iteration count.
        null_logistic_converged: Whether the ordinary null logistic fit converged.
        null_firth_iteration_count: Null Firth iteration count.
        null_firth_convergence_reason: Null Firth convergence reason.
        firth_correction_branch: Scalar Firth correction branch.
        firth_iteration_count: Total Firth iteration count.
        pseudo_firth_iteration_count: Pseudo-Firth iteration count.
        nr_zero_start_iteration_count: Zero-start Newton-Raphson iteration count.
        nr_warm_start_iteration_count: Warm-start Newton-Raphson iteration count.
        final_beta: Final beta after optional Firth correction.
        final_standard_error: Final standard error.
        final_chi_squared: Final chi-squared or LRT statistic.
        final_log10_p_value: Final negative log10 p-value.
        final_extra_code: Internal final extra code.
        final_valid: Whether the final row has valid statistics.
        firth_failure_code: Public Firth failure code.
        firth_convergence_reason: Internal Firth convergence reason.

    """

    variant_index: int
    chromosome: str
    position: int
    variant_identifier: str
    allele_zero: str
    allele_one: str
    allele_count: float
    flipped_allele_count: float
    flip_mask: bool
    minor_allele_count: float
    sparse_candidate: bool
    rare_sparse_firth_candidate: bool
    carrier_count: int
    score: float
    score_variance: float
    score_beta: float | None
    score_standard_error: float | None
    score_chi_squared: float | None
    score_log10_p_value: float | None
    score_extra_code: str
    null_logistic_offset: dict[str, float]
    null_firth_offset: dict[str, float]
    null_logistic_iteration_count: int
    null_logistic_converged: bool
    null_firth_iteration_count: int
    null_firth_convergence_reason: str
    firth_correction_branch: str
    firth_iteration_count: int
    pseudo_firth_iteration_count: int
    nr_zero_start_iteration_count: int
    nr_warm_start_iteration_count: int
    final_beta: float | None
    final_standard_error: float | None
    final_chi_squared: float | None
    final_log10_p_value: float | None
    final_extra_code: str
    final_valid: bool
    firth_failure_code: str
    firth_convergence_reason: str


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
    parser = argparse.ArgumentParser(description="Capture binary REGENIE step 2 parity diagnostics for variants.")
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
    parser.add_argument("--p-threshold", type=float, default=0.05)
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


def enum_name(enum_type: type[typing.Any], value: typing.Any) -> str:
    """Render an integer enum value as a stable lowercase name."""
    try:
        enum_value = enum_type(int(value))
    except ValueError:
        return f"unknown_{int(value)}"
    name = typing.cast("str", enum_value.name)
    return name.lower()


def extra_code_name(value: object) -> str:
    """Render an internal binary extra code."""
    return enum_name(types.BinaryExtraCode, value)


def firth_failure_code_name(value: object) -> str:
    """Render a Firth failure code."""
    return enum_name(types.FirthFailureCode, value)


def firth_correction_code_name(value: object) -> str:
    """Render a Firth correction branch code."""
    return enum_name(types.FirthCorrectionCode, value)


def firth_convergence_reason_name(value: object) -> str:
    """Render an internal Firth convergence reason code."""
    return enum_name(regenie2_binary_firth_types.FirthConvergenceReason, value)


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


def compute_score_debug_arrays(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> ScoreDebugArrays:
    """Compute score-test internal arrays for selected variants."""
    raw_genotype_matrix_by_variant = jnp.asarray(genotype_matrix_by_variant, dtype=jnp.float32)
    genotype_flip_result = genotype.build_regenie_flipped_genotypes(raw_genotype_matrix_by_variant)
    genotype_matrix_by_variant_float32 = genotype_flip_result.genotype_matrix_by_variant
    weighted_genotype_matrix_by_variant = (
        genotype_matrix_by_variant_float32 * chromosome_state.square_root_weight[None, :]
    )
    projection_coordinates = (
        weighted_genotype_matrix_by_variant @ chromosome_state.weighted_genotype_projection_matrix.T
    )
    weighted_genotype_sum_squares = jnp.einsum(
        "ij,ij->i",
        weighted_genotype_matrix_by_variant,
        weighted_genotype_matrix_by_variant,
    )
    projection_sum_squares = jnp.einsum("ij,ij->i", projection_coordinates, projection_coordinates)
    variance = jnp.maximum(weighted_genotype_sum_squares - projection_sum_squares, 0.0)
    score = genotype_matrix_by_variant_float32 @ chromosome_state.score_residual
    allele_count = jnp.sum(raw_genotype_matrix_by_variant, axis=1)
    flipped_allele_count = jnp.sum(genotype_matrix_by_variant_float32, axis=1)
    carrier_count = jnp.sum(
        genotype_matrix_by_variant_float32 > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
        axis=1,
    )
    host_values = jax.device_get(
        {
            "score": score.astype(jnp.float64),
            "variance": variance.astype(jnp.float64),
            "allele_count": allele_count.astype(jnp.float64),
            "flipped_allele_count": flipped_allele_count.astype(jnp.float64),
            "flip_mask": genotype_flip_result.flip_mask,
            "carrier_count": carrier_count.astype(jnp.int64),
        }
    )
    return ScoreDebugArrays(
        score=np.asarray(host_values["score"], dtype=np.float64),
        variance=np.asarray(host_values["variance"], dtype=np.float64),
        allele_count=np.asarray(host_values["allele_count"], dtype=np.float64),
        flipped_allele_count=np.asarray(host_values["flipped_allele_count"], dtype=np.float64),
        flip_mask=np.asarray(host_values["flip_mask"], dtype=np.bool_),
        carrier_count=np.asarray(host_values["carrier_count"], dtype=np.int64),
    )


def build_debug_records_for_chunk(
    *,
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    metadata: _core.VariantMetadata,
    chunk_stats: _core.ChunkStats,
    genotype_matrix_by_variant: npt.NDArray[np.float32],
    selected_offsets: list[int],
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> list[VariantDebugRecord]:
    """Build debug records for selected offsets from one native BGEN chunk."""
    selected_genotype_matrix_by_variant = jnp.asarray(
        genotype_matrix_by_variant[selected_offsets, :],
        dtype=jnp.float32,
    )
    score_result = regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state,
        selected_genotype_matrix_by_variant.T,
        correction_plan,
        kernel_config,
    )
    final_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=selected_genotype_matrix_by_variant.T,
        correction_plan=correction_plan,
        sparse_candidate_mask=jnp.asarray(chunk_stats.is_rare_sparse_firth_candidate[selected_offsets]),
        kernel_config=kernel_config,
    )
    score_debug_arrays = compute_score_debug_arrays(
        chromosome_state, selected_genotype_matrix_by_variant, kernel_config
    )
    score_host = jax.device_get(score_result)
    final_host = jax.device_get(final_result)
    null_logistic_offset = chromosome_state.covariate_matrix @ chromosome_state.null_logistic_coefficients + (
        chromosome_state.loco_offset
    )
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
                allele_count=float(score_debug_arrays.allele_count[result_offset]),
                flipped_allele_count=float(score_debug_arrays.flipped_allele_count[result_offset]),
                flip_mask=bool(score_debug_arrays.flip_mask[result_offset]),
                minor_allele_count=float(chunk_stats.minor_allele_count[chunk_offset]),
                sparse_candidate=bool(chunk_stats.is_sparse_candidate[chunk_offset]),
                rare_sparse_firth_candidate=bool(chunk_stats.is_rare_sparse_firth_candidate[chunk_offset]),
                carrier_count=int(score_debug_arrays.carrier_count[result_offset]),
                score=float(score_debug_arrays.score[result_offset]),
                score_variance=float(score_debug_arrays.variance[result_offset]),
                score_beta=finite_float_or_none(score_host.beta[result_offset]),
                score_standard_error=finite_float_or_none(score_host.standard_error[result_offset]),
                score_chi_squared=finite_float_or_none(score_host.chi_squared[result_offset]),
                score_log10_p_value=finite_float_or_none(score_host.log10_p_value[result_offset]),
                score_extra_code=extra_code_name(score_host.extra_code[result_offset]),
                null_logistic_offset=summarize_array(null_logistic_offset),
                null_firth_offset=summarize_array(chromosome_state.null_firth_offset),
                null_logistic_iteration_count=int(chromosome_state.null_logistic_iteration_count),
                null_logistic_converged=bool(chromosome_state.null_logistic_converged),
                null_firth_iteration_count=int(chromosome_state.null_firth_iteration_count),
                null_firth_convergence_reason=firth_convergence_reason_name(
                    chromosome_state.null_firth_convergence_reason_code
                ),
                firth_correction_branch=firth_correction_code_name(final_host.firth_correction_code[result_offset]),
                firth_iteration_count=int(final_host.firth_iteration_count[result_offset]),
                pseudo_firth_iteration_count=int(final_host.pseudo_firth_iteration_count[result_offset]),
                nr_zero_start_iteration_count=int(final_host.nr_zero_start_iteration_count[result_offset]),
                nr_warm_start_iteration_count=int(final_host.nr_warm_start_iteration_count[result_offset]),
                final_beta=finite_float_or_none(final_host.beta[result_offset]),
                final_standard_error=finite_float_or_none(final_host.standard_error[result_offset]),
                final_chi_squared=finite_float_or_none(final_host.chi_squared[result_offset]),
                final_log10_p_value=finite_float_or_none(final_host.log10_p_value[result_offset]),
                final_extra_code=extra_code_name(final_host.extra_code[result_offset]),
                final_valid=bool(final_host.valid_mask[result_offset]),
                firth_failure_code=firth_failure_code_name(final_host.firth_failure_code[result_offset]),
                firth_convergence_reason=firth_convergence_reason_name(
                    final_host.firth_convergence_reason_code[result_offset]
                ),
            )
        )
    return records


class BinaryVariantDebugCaptureCallback:
    """Native BGEN callback that captures selected binary debug records."""

    def __init__(
        self,
        *,
        run_input: native_dispatch.NativeBgenRunInput,
        prediction_source: _core.RegeniePredictionSource,
        selector: VariantSelector,
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
    ) -> None:
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.selector = selector
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.chromosome_states: dict[str, regenie2_binary_state.Regenie2BinaryChromosomeState] = {}
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
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                )
            )
        self.free_buffers.append(genotype_matrix_by_variant)

    def prepare_chromosome_state(self, chromosome: str) -> regenie2_binary_state.Regenie2BinaryChromosomeState:
        """Build or reuse the binary chromosome state for one chromosome."""
        if chromosome not in self.chromosome_states:
            state = regenie2_binary.prepare_regenie2_binary_state(
                self.run_input.covariate_matrix,
                self.run_input.phenotype_vector,
            )
            loco_offset = jnp.asarray(self.prediction_source.get_chromosome_predictions(chromosome), dtype=jnp.float32)
            self.chromosome_states[chromosome] = regenie2_binary.prepare_regenie2_binary_chromosome_state(
                state,
                loco_offset,
                self.correction_plan,
                self.kernel_config,
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
    """Run native BGEN streaming and capture selected `g` debug records."""
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
            is_binary_trait=True,
        )
    )
    prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(arguments.pred),
        str(arguments.pheno_col),
        run_input.native_aligned_sample_data,
    )
    correction_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=float(arguments.p_threshold),
        firth_se=False,
    )
    kernel_config = execution_plan.build_binary_kernel_config(interface_config.GComputeConfig())
    callback = BinaryVariantDebugCaptureCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        selector=selector,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
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
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(f"{json.dumps(output_payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(f"Wrote binary parity diagnostics: {arguments.output_json}")


if __name__ == "__main__":
    main()
