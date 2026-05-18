"""Compare production and experimental binary Firth compute paths."""

from __future__ import annotations

import argparse
import json
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from g import types
from g.compute import regenie2_binary, regenie2_binary_types


@dataclass(frozen=True)
class BinaryParityInputs:
    """Inputs for one binary path parity comparison.

    Attributes:
        covariate_matrix: Null-model covariate matrix.
        phenotype_vector: Binary phenotype vector.
        genotype_matrix: Sample-major genotype matrix.
        loco_offset: Per-sample LOCO offset.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    genotype_matrix: jax.Array
    loco_offset: jax.Array


@dataclass(frozen=True)
class BinaryPathMetrics:
    """Summary metrics for one binary compute path.

    Attributes:
        score_test_count: Number of variants left as score-test rows.
        firth_candidate_count: Number of variants selected for Firth fallback.
        firth_converged_count: Number of Firth candidate rows without a failure code.
        firth_failure_count: Number of Firth candidate rows with a failure code.
        extra_counts: Histogram of output EXTRA codes.

    """

    score_test_count: int
    firth_candidate_count: int
    firth_converged_count: int
    firth_failure_count: int
    extra_counts: dict[str, int]


@dataclass(frozen=True)
class BinaryPathComparison:
    """Comparison between production and experimental binary paths.

    Attributes:
        passed: Whether all parity checks passed.
        production_metrics: Metrics from the production sample-major path.
        experimental_metrics: Metrics from the experimental variant-major path.
        maximum_absolute_deltas: Maximum absolute differences by numeric column.
        mismatch_messages: Human-readable mismatch descriptions.

    """

    passed: bool
    production_metrics: BinaryPathMetrics
    experimental_metrics: BinaryPathMetrics
    maximum_absolute_deltas: dict[str, float]
    mismatch_messages: tuple[str, ...]


@dataclass(frozen=True)
class NumericColumnComparison:
    """Comparison result for one numeric output column.

    Attributes:
        maximum_absolute_delta: Largest absolute difference in the column.
        mismatch_message: Optional mismatch message.

    """

    maximum_absolute_delta: float
    mismatch_message: str | None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Compare binary Firth production and experimental compute paths.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        help=(
            "Optional NPZ containing covariate_matrix, phenotype_vector, genotype_matrix, "
            "and optional loco_offset arrays. Omit for the built-in small fixture."
        ),
    )
    parser.add_argument("--p-threshold", type=float, default=0.05, help="Firth fallback p-value threshold.")
    parser.add_argument("--firth-se", action="store_true", help="Use LRT-derived Firth standard errors.")
    parser.add_argument("--rtol", type=float, default=1.0e-5, help="Relative tolerance for numeric comparisons.")
    parser.add_argument("--atol", type=float, default=1.0e-5, help="Absolute tolerance for numeric comparisons.")
    parser.add_argument("--output-json", type=Path, help="Optional path for a JSON comparison summary.")
    return parser


def build_synthetic_inputs() -> BinaryParityInputs:
    """Build the default small parity fixture."""
    covariate_matrix = jnp.asarray(
        [
            [1.0, 20.0],
            [1.0, 25.0],
            [1.0, 30.0],
            [1.0, 35.0],
            [1.0, 40.0],
            [1.0, 45.0],
            [1.0, 50.0],
            [1.0, 55.0],
        ],
        dtype=jnp.float32,
    )
    phenotype_vector = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    genotype_matrix = jnp.asarray(
        [
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 25.0],
            [0.0, 1.0, 30.0],
            [0.0, 1.0, 35.0],
            [2.0, 1.0, 40.0],
            [2.0, 1.0, 45.0],
            [2.0, 2.0, 50.0],
            [2.0, 2.0, 55.0],
        ],
        dtype=jnp.float32,
    )
    return BinaryParityInputs(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_matrix=genotype_matrix,
        loco_offset=jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32),
    )


def load_npz_inputs(input_npz_path: Path) -> BinaryParityInputs:
    """Load parity inputs from an NPZ file."""
    input_arrays = np.load(input_npz_path)
    phenotype_vector = jnp.asarray(input_arrays["phenotype_vector"], dtype=jnp.float32)
    loco_offset = jnp.asarray(
        input_arrays["loco_offset"] if "loco_offset" in input_arrays else np.zeros(phenotype_vector.shape[0]),
        dtype=jnp.float32,
    )
    return BinaryParityInputs(
        covariate_matrix=jnp.asarray(input_arrays["covariate_matrix"], dtype=jnp.float32),
        phenotype_vector=phenotype_vector,
        genotype_matrix=jnp.asarray(input_arrays["genotype_matrix"], dtype=jnp.float32),
        loco_offset=loco_offset,
    )


def prepare_chromosome_state(inputs: BinaryParityInputs) -> regenie2_binary_types.Regenie2BinaryChromosomeState:
    """Prepare the binary chromosome state shared by both paths."""
    regenie_state = regenie2_binary.prepare_regenie2_binary_state(
        covariate_matrix=inputs.covariate_matrix,
        phenotype_vector=inputs.phenotype_vector,
    )
    return regenie2_binary.prepare_regenie2_binary_chromosome_state(regenie_state, inputs.loco_offset)


def compute_path_metrics(
    *,
    score_test_result: regenie2_binary_types.Regenie2BinaryChunkResult,
    corrected_result: regenie2_binary_types.Regenie2BinaryChunkResult,
) -> BinaryPathMetrics:
    """Compute parity metrics for one path."""
    score_extra_code = np.asarray(score_test_result.extra_code)
    corrected_extra_code = np.asarray(corrected_result.extra_code)
    firth_candidate_mask = score_extra_code == regenie2_binary.EXTRA_CODE_FIRTH
    firth_failure_code = np.asarray(corrected_result.firth_failure_code)
    unique_extra_codes, extra_code_counts = np.unique(corrected_extra_code, return_counts=True)
    return BinaryPathMetrics(
        score_test_count=int(np.count_nonzero(score_extra_code == regenie2_binary.EXTRA_CODE_SCORE)),
        firth_candidate_count=int(np.count_nonzero(firth_candidate_mask)),
        firth_converged_count=int(
            np.count_nonzero(firth_candidate_mask & (firth_failure_code == regenie2_binary.FIRTH_FAILURE_NONE))
        ),
        firth_failure_count=int(
            np.count_nonzero(firth_candidate_mask & (firth_failure_code != regenie2_binary.FIRTH_FAILURE_NONE))
        ),
        extra_counts={
            str(int(extra_code)): int(extra_count)
            for extra_code, extra_count in zip(unique_extra_codes, extra_code_counts, strict=True)
        },
    )


def compare_numeric_column(
    *,
    column_name: str,
    production_values: jax.Array,
    experimental_values: jax.Array,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NumericColumnComparison:
    """Compare one numeric result column."""
    production_array = np.asarray(production_values)
    experimental_array = np.asarray(experimental_values)
    maximum_absolute_delta = float(np.nanmax(np.abs(production_array - experimental_array)))
    if np.allclose(
        production_array,
        experimental_array,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        equal_nan=True,
    ):
        return NumericColumnComparison(maximum_absolute_delta=maximum_absolute_delta, mismatch_message=None)
    return NumericColumnComparison(
        maximum_absolute_delta=maximum_absolute_delta,
        mismatch_message=f"{column_name} differs beyond tolerance.",
    )


def compare_binary_paths(
    *,
    inputs: BinaryParityInputs,
    correction_plan: types.BinaryCorrectionPlan,
    relative_tolerance: float = 1.0e-5,
    absolute_tolerance: float = 1.0e-5,
) -> BinaryPathComparison:
    """Compare production sample-major and experimental variant-major binary paths."""
    chromosome_state = prepare_chromosome_state(inputs)
    production_score_test_result = regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state,
        inputs.genotype_matrix,
        correction_plan,
    )
    production_corrected_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state,
        inputs.genotype_matrix,
        correction_plan,
    )
    genotype_matrix_by_variant = jnp.transpose(inputs.genotype_matrix)
    experimental_score_test_result = (
        regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(
            chromosome_state,
            genotype_matrix_by_variant,
            correction_plan,
        )
    )
    experimental_corrected_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state,
        genotype_matrix_by_variant,
        correction_plan,
    )
    production_metrics = compute_path_metrics(
        score_test_result=production_score_test_result,
        corrected_result=production_corrected_result,
    )
    experimental_metrics = compute_path_metrics(
        score_test_result=experimental_score_test_result,
        corrected_result=experimental_corrected_result,
    )
    mismatch_messages: list[str] = []
    if production_metrics != experimental_metrics:
        mismatch_messages.append("Score-test/Firth candidate metrics differ.")
    if not np.array_equal(
        np.asarray(production_corrected_result.extra_code), np.asarray(experimental_corrected_result.extra_code)
    ):
        mismatch_messages.append("EXTRA codes differ.")
    maximum_absolute_deltas: dict[str, float] = {}
    for column_name, production_values, experimental_values in [
        ("beta", production_corrected_result.beta, experimental_corrected_result.beta),
        ("standard_error", production_corrected_result.standard_error, experimental_corrected_result.standard_error),
        ("chi_squared", production_corrected_result.chi_squared, experimental_corrected_result.chi_squared),
        ("log10_p_value", production_corrected_result.log10_p_value, experimental_corrected_result.log10_p_value),
    ]:
        numeric_column_comparison = compare_numeric_column(
            column_name=column_name,
            production_values=production_values,
            experimental_values=experimental_values,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        maximum_absolute_deltas[column_name] = numeric_column_comparison.maximum_absolute_delta
        if numeric_column_comparison.mismatch_message is not None:
            mismatch_messages.append(numeric_column_comparison.mismatch_message)
    return BinaryPathComparison(
        passed=not mismatch_messages,
        production_metrics=production_metrics,
        experimental_metrics=experimental_metrics,
        maximum_absolute_deltas=maximum_absolute_deltas,
        mismatch_messages=tuple(mismatch_messages),
    )


def comparison_to_json_dict(comparison: BinaryPathComparison) -> dict[str, typing.Any]:
    """Convert a comparison result into JSON-serializable values."""
    return {
        "passed": comparison.passed,
        "production_metrics": comparison.production_metrics.__dict__,
        "experimental_metrics": comparison.experimental_metrics.__dict__,
        "maximum_absolute_deltas": comparison.maximum_absolute_deltas,
        "mismatch_messages": list(comparison.mismatch_messages),
    }


def main() -> None:
    """Run the parity harness."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    inputs = load_npz_inputs(arguments.input_npz) if arguments.input_npz is not None else build_synthetic_inputs()
    correction_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=float(arguments.p_threshold),
        firth_se=bool(arguments.firth_se),
    )
    comparison = compare_binary_paths(
        inputs=inputs,
        correction_plan=correction_plan,
        relative_tolerance=float(arguments.rtol),
        absolute_tolerance=float(arguments.atol),
    )
    payload = comparison_to_json_dict(comparison)
    rendered_payload = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output_json is not None:
        arguments.output_json.write_text(rendered_payload + "\n", encoding="utf-8")
    print(rendered_payload)
    if not comparison.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
