from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from g.compute.linear import (
    compute_linear_association_chunk,
    compute_linear_association_chunk_with_pallas,
    prepare_linear_association_state,
)

if TYPE_CHECKING:
    from g.models import LinearAssociationChunkResult


class SyntheticInputs(NamedTuple):
    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    genotype_matrix: jax.Array


class CompiledCallable(Protocol):
    def __call__(
        self,
        linear_association_state: object,
        genotype_matrix: jax.Array,
    ) -> LinearAssociationChunkResult: ...


class LoweredCallable(Protocol):
    def compile(self) -> CompiledCallable: ...


class LowerableCallable(Protocol):
    def lower(
        self,
        linear_association_state: object,
        genotype_matrix: jax.Array,
    ) -> LoweredCallable: ...


class CompiledTiming(NamedTuple):
    compiled_callable: CompiledCallable
    compile_seconds: float


class ExecutionTiming(NamedTuple):
    result: LinearAssociationChunkResult
    mean_execution_seconds: float
    minimum_execution_seconds: float


@dataclass
class TimingSummary:
    compile_seconds: float
    mean_execution_seconds: float
    minimum_execution_seconds: float


@dataclass
class BenchmarkSummary:
    device_kind: str
    sample_count: int
    covariate_count: int
    variant_count: int
    repeat_count: int
    baseline: TimingSummary
    pallas: TimingSummary
    execution_speedup: float
    valid_variant_count: int
    baseline_valid_variant_count: int
    pallas_valid_variant_count: int
    max_beta_difference: float
    max_standard_error_difference: float
    max_test_statistic_difference: float
    max_p_value_difference: float
    valid_mask_equal: bool


def compute_masked_maximum_difference(
    left_values: jax.Array,
    right_values: jax.Array,
    valid_mask: jax.Array,
) -> float:
    if not bool(jnp.any(valid_mask)):
        return 0.0
    difference = jnp.abs(left_values - right_values)
    masked_difference = jnp.where(valid_mask, difference, 0.0)
    return float(jnp.max(masked_difference))


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--sample-count", type=int, default=4096)
    argument_parser.add_argument("--covariate-count", type=int, default=3)
    argument_parser.add_argument("--variant-count", type=int, default=4096)
    argument_parser.add_argument("--repeat-count", type=int, default=20)
    argument_parser.add_argument("--seed", type=int, default=0)
    return argument_parser.parse_args()


def block_result_until_ready(result: object) -> None:
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        result,
    )


def build_synthetic_inputs(
    sample_count: int,
    covariate_count: int,
    variant_count: int,
    seed: int,
) -> SyntheticInputs:
    key = jax.random.key(seed)
    covariate_key, phenotype_key, genotype_key = jax.random.split(key, 3)
    intercept_column = jnp.ones((sample_count, 1), dtype=jnp.float32)
    additional_covariates = jax.random.normal(
        covariate_key,
        (sample_count, covariate_count - 1),
        dtype=jnp.float32,
    )
    covariate_matrix = jnp.concatenate([intercept_column, additional_covariates], axis=1)
    phenotype_vector = jax.random.normal(phenotype_key, (sample_count,), dtype=jnp.float32)
    genotype_matrix = jax.random.randint(
        genotype_key,
        (sample_count, variant_count),
        minval=0,
        maxval=3,
        dtype=jnp.int32,
    ).astype(jnp.float32)
    return SyntheticInputs(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_matrix=genotype_matrix,
    )


def compile_callable(
    function: LowerableCallable,
    linear_association_state: object,
    genotype_matrix: jax.Array,
) -> CompiledTiming:
    compile_start_time = time.perf_counter()
    compiled_function = function.lower(linear_association_state, genotype_matrix).compile()
    compile_seconds = time.perf_counter() - compile_start_time
    return CompiledTiming(compiled_callable=compiled_function, compile_seconds=compile_seconds)


def measure_execution(
    compiled_function: CompiledCallable,
    linear_association_state: object,
    genotype_matrix: jax.Array,
    repeat_count: int,
) -> ExecutionTiming:
    execution_times: list[float] = []
    result = compiled_function(linear_association_state, genotype_matrix)
    block_result_until_ready(result)
    for _ in range(repeat_count):
        execution_start_time = time.perf_counter()
        result = compiled_function(linear_association_state, genotype_matrix)
        block_result_until_ready(result)
        execution_times.append(time.perf_counter() - execution_start_time)
    execution_time_array = np.asarray(execution_times, dtype=np.float64)
    return ExecutionTiming(
        result=result,
        mean_execution_seconds=float(np.mean(execution_time_array)),
        minimum_execution_seconds=float(np.min(execution_time_array)),
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.covariate_count < 1:
        message = "covariate_count must be at least 1."
        raise ValueError(message)

    synthetic_inputs = build_synthetic_inputs(
        sample_count=arguments.sample_count,
        covariate_count=arguments.covariate_count,
        variant_count=arguments.variant_count,
        seed=arguments.seed,
    )
    linear_association_state = prepare_linear_association_state(
        synthetic_inputs.covariate_matrix,
        synthetic_inputs.phenotype_vector,
    )

    baseline_compiled_timing = compile_callable(
        compute_linear_association_chunk,
        linear_association_state,
        synthetic_inputs.genotype_matrix,
    )
    pallas_compiled_timing = compile_callable(
        compute_linear_association_chunk_with_pallas,
        linear_association_state,
        synthetic_inputs.genotype_matrix,
    )

    baseline_execution_timing = measure_execution(
        baseline_compiled_timing.compiled_callable,
        linear_association_state,
        synthetic_inputs.genotype_matrix,
        arguments.repeat_count,
    )
    pallas_execution_timing = measure_execution(
        pallas_compiled_timing.compiled_callable,
        linear_association_state,
        synthetic_inputs.genotype_matrix,
        arguments.repeat_count,
    )
    baseline_result = baseline_execution_timing.result
    pallas_result = pallas_execution_timing.result
    shared_valid_mask = baseline_result.valid_mask & pallas_result.valid_mask

    benchmark_summary = BenchmarkSummary(
        device_kind=jax.devices()[0].device_kind,
        sample_count=arguments.sample_count,
        covariate_count=arguments.covariate_count,
        variant_count=arguments.variant_count,
        repeat_count=arguments.repeat_count,
        baseline=TimingSummary(
            compile_seconds=baseline_compiled_timing.compile_seconds,
            mean_execution_seconds=baseline_execution_timing.mean_execution_seconds,
            minimum_execution_seconds=baseline_execution_timing.minimum_execution_seconds,
        ),
        pallas=TimingSummary(
            compile_seconds=pallas_compiled_timing.compile_seconds,
            mean_execution_seconds=pallas_execution_timing.mean_execution_seconds,
            minimum_execution_seconds=pallas_execution_timing.minimum_execution_seconds,
        ),
        execution_speedup=(
            baseline_execution_timing.mean_execution_seconds / pallas_execution_timing.mean_execution_seconds
        ),
        valid_variant_count=int(jnp.sum(shared_valid_mask, dtype=jnp.int32)),
        baseline_valid_variant_count=int(jnp.sum(baseline_result.valid_mask, dtype=jnp.int32)),
        pallas_valid_variant_count=int(jnp.sum(pallas_result.valid_mask, dtype=jnp.int32)),
        max_beta_difference=compute_masked_maximum_difference(
            baseline_result.beta,
            pallas_result.beta,
            shared_valid_mask,
        ),
        max_standard_error_difference=compute_masked_maximum_difference(
            baseline_result.standard_error,
            pallas_result.standard_error,
            shared_valid_mask,
        ),
        max_test_statistic_difference=compute_masked_maximum_difference(
            baseline_result.test_statistic,
            pallas_result.test_statistic,
            shared_valid_mask,
        ),
        max_p_value_difference=compute_masked_maximum_difference(
            baseline_result.p_value,
            pallas_result.p_value,
            shared_valid_mask,
        ),
        valid_mask_equal=bool(jnp.all(baseline_result.valid_mask == pallas_result.valid_mask)),
    )
    print(json.dumps(asdict(benchmark_summary), indent=2))


if __name__ == "__main__":
    main()
