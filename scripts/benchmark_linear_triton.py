from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from g.compute.linear_triton import (  # type: ignore
    TritonLinearAssociationState,
    compute_linear_association_statistics_with_triton,
    prepare_triton_linear_association_state,
)

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state

torch: Any = importlib.import_module("torch")

if TYPE_CHECKING:
    from typing import Protocol

    from g.compute.linear_triton import TritonLinearAssociationStatistics  # type: ignore

    from g.models import LinearAssociationChunkResult

    class CompiledCallable(Protocol):
        def __call__(
            self,
            linear_association_state: object,
            genotype_matrix: jax.Array,
        ) -> LinearAssociationChunkResult: ...


@dataclass(frozen=True)
class CompiledTiming:
    compiled_callable: CompiledCallable
    compile_seconds: float


@dataclass(frozen=True)
class TritonTiming:
    triton_state: TritonLinearAssociationState
    compile_seconds: float


@dataclass(frozen=True)
class JaxTiming:
    compiled_callable: CompiledCallable
    linear_association_state: object
    compile_seconds: float


@dataclass(frozen=True)
class TimingSummary:
    compile_seconds: float
    mean_execution_seconds: float
    minimum_execution_seconds: float


@dataclass(frozen=True)
class BenchmarkSummary:
    device_kind: str
    sample_count: int
    covariate_count: int
    variant_count: int
    repeat_count: int
    baseline: TimingSummary
    triton: TimingSummary
    execution_speedup: float
    max_beta_difference: float
    max_standard_error_difference: float
    max_test_statistic_difference: float
    valid_mask_equal: bool


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--sample-count", type=int, default=4096)
    argument_parser.add_argument("--covariate-count", type=int, default=3)
    argument_parser.add_argument("--variant-count", type=int, default=4096)
    argument_parser.add_argument("--repeat-count", type=int, default=20)
    argument_parser.add_argument("--seed", type=int, default=0)
    return argument_parser.parse_args()


def build_synthetic_inputs(
    sample_count: int,
    covariate_count: int,
    variant_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    intercept_column = np.ones((sample_count, 1), dtype=np.float32)
    additional_covariates = random_generator.normal(
        size=(sample_count, covariate_count - 1),
    ).astype(np.float32)
    covariate_matrix = np.concatenate([intercept_column, additional_covariates], axis=1)
    phenotype_vector = random_generator.normal(size=(sample_count,)).astype(np.float32)
    genotype_matrix = random_generator.integers(
        low=0,
        high=3,
        size=(sample_count, variant_count),
        dtype=np.int32,
    ).astype(np.float32)
    return covariate_matrix, phenotype_vector, genotype_matrix


def block_jax_result_until_ready(result: object) -> None:
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        result,
    )


def compile_jax_baseline(
    covariate_matrix: np.ndarray,
    phenotype_vector: np.ndarray,
    genotype_matrix: np.ndarray,
) -> JaxTiming:
    linear_association_state = prepare_linear_association_state(
        jnp.asarray(covariate_matrix),
        jnp.asarray(phenotype_vector),
    )
    genotype_matrix_jax = jnp.asarray(genotype_matrix)
    compile_start_time = time.perf_counter()
    compiled_function = compute_linear_association_chunk.lower(
        linear_association_state,
        genotype_matrix_jax,
    ).compile()
    compile_seconds = time.perf_counter() - compile_start_time
    return JaxTiming(
        compiled_callable=compiled_function,  # type: ignore[arg-type]
        linear_association_state=linear_association_state,
        compile_seconds=compile_seconds,
    )


def benchmark_jax_baseline(
    compiled_function: CompiledCallable,
    linear_association_state: object,
    genotype_matrix: np.ndarray,
    repeat_count: int,
) -> tuple[LinearAssociationChunkResult, float, float]:
    execution_times: list[float] = []
    genotype_matrix_jax = jnp.asarray(genotype_matrix)
    result = compiled_function(linear_association_state, genotype_matrix_jax)
    block_jax_result_until_ready(result)
    for _ in range(repeat_count):
        execution_start_time = time.perf_counter()
        result = compiled_function(linear_association_state, genotype_matrix_jax)
        block_jax_result_until_ready(result)
        execution_times.append(time.perf_counter() - execution_start_time)
    execution_time_array = np.asarray(execution_times, dtype=np.float64)
    return result, float(np.mean(execution_time_array)), float(np.min(execution_time_array))


def compile_triton_experiment(
    covariate_matrix: np.ndarray,
    phenotype_vector: np.ndarray,
    genotype_matrix: np.ndarray,
    device: torch.device,
) -> TritonTiming:
    triton_linear_association_state = prepare_triton_linear_association_state(
        covariate_matrix,
        phenotype_vector,
        device,
    )
    compile_start_time = time.perf_counter()
    warmup_result = compute_linear_association_statistics_with_triton(
        triton_linear_association_state,
        genotype_matrix,
    )
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - compile_start_time
    del warmup_result
    return TritonTiming(
        triton_state=triton_linear_association_state,
        compile_seconds=compile_seconds,
    )


def benchmark_triton_experiment(
    triton_linear_association_state: TritonLinearAssociationState,
    genotype_matrix: np.ndarray,
    repeat_count: int,
    device: torch.device,
) -> tuple[TritonLinearAssociationStatistics, float, float]:
    execution_times: list[float] = []
    result = compute_linear_association_statistics_with_triton(
        triton_linear_association_state,
        genotype_matrix,
    )
    torch.cuda.synchronize(device)
    for _ in range(repeat_count):
        execution_start_time = time.perf_counter()
        result = compute_linear_association_statistics_with_triton(
            triton_linear_association_state,
            genotype_matrix,
        )
        torch.cuda.synchronize(device)
        execution_times.append(time.perf_counter() - execution_start_time)
    execution_time_array = np.asarray(execution_times, dtype=np.float64)
    return result, float(np.mean(execution_time_array)), float(np.min(execution_time_array))


def main() -> None:
    arguments = parse_arguments()
    if arguments.covariate_count < 1:
        message = "covariate_count must be at least 1."
        raise ValueError(message)
    if not torch.cuda.is_available():
        message = "CUDA is required for the Triton benchmark."
        raise RuntimeError(message)

    covariate_matrix, phenotype_vector, genotype_matrix = build_synthetic_inputs(
        sample_count=arguments.sample_count,
        covariate_count=arguments.covariate_count,
        variant_count=arguments.variant_count,
        seed=arguments.seed,
    )
    device = torch.device("cuda")
    jax_timing = compile_jax_baseline(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
    )
    torch.cuda.empty_cache()
    triton_timing = compile_triton_experiment(
        covariate_matrix,
        phenotype_vector,
        genotype_matrix,
        device,
    )
    baseline_result, baseline_mean_execution_seconds, baseline_minimum_execution_seconds = benchmark_jax_baseline(
        jax_timing.compiled_callable,
        jax_timing.linear_association_state,
        genotype_matrix,
        arguments.repeat_count,
    )
    triton_result, triton_mean_execution_seconds, triton_minimum_execution_seconds = benchmark_triton_experiment(
        triton_timing.triton_state,
        genotype_matrix,
        arguments.repeat_count,
        device,
    )

    baseline_beta = np.asarray(jax.device_get(baseline_result.beta))
    baseline_standard_error = np.asarray(jax.device_get(baseline_result.standard_error))
    baseline_test_statistic = np.asarray(jax.device_get(baseline_result.test_statistic))
    baseline_valid_mask = np.asarray(jax.device_get(baseline_result.valid_mask))
    triton_beta = triton_result.beta.detach().cpu().numpy()
    triton_standard_error = triton_result.standard_error.detach().cpu().numpy()
    triton_test_statistic = triton_result.test_statistic.detach().cpu().numpy()
    triton_valid_mask = triton_result.valid_mask.detach().cpu().numpy()
    shared_valid_mask = baseline_valid_mask & triton_valid_mask

    benchmark_summary = BenchmarkSummary(
        device_kind=torch.cuda.get_device_name(device),
        sample_count=arguments.sample_count,
        covariate_count=arguments.covariate_count,
        variant_count=arguments.variant_count,
        repeat_count=arguments.repeat_count,
        baseline=TimingSummary(
            compile_seconds=jax_timing.compile_seconds,
            mean_execution_seconds=baseline_mean_execution_seconds,
            minimum_execution_seconds=baseline_minimum_execution_seconds,
        ),
        triton=TimingSummary(
            compile_seconds=triton_timing.compile_seconds,
            mean_execution_seconds=triton_mean_execution_seconds,
            minimum_execution_seconds=triton_minimum_execution_seconds,
        ),
        execution_speedup=baseline_mean_execution_seconds / triton_mean_execution_seconds,
        max_beta_difference=float(np.max(np.abs(baseline_beta[shared_valid_mask] - triton_beta[shared_valid_mask]))),
        max_standard_error_difference=float(
            np.max(np.abs(baseline_standard_error[shared_valid_mask] - triton_standard_error[shared_valid_mask]))
        ),
        max_test_statistic_difference=float(
            np.max(np.abs(baseline_test_statistic[shared_valid_mask] - triton_test_statistic[shared_valid_mask]))
        ),
        valid_mask_equal=bool(np.array_equal(baseline_valid_mask, triton_valid_mask)),
    )
    print(json.dumps(asdict(benchmark_summary), indent=2))


if __name__ == "__main__":
    main()
