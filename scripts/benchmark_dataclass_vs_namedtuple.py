"""Benchmark JAX dataclasses vs NamedTuples for pure array containers."""

import time
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


# NamedTuple version (old approach)
class StateNamedTuple(NamedTuple):
    """State with only JAX arrays - NamedTuple version."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array


# Dataclass version (new approach)
@jax.tree_util.register_dataclass
@dataclass
class StateDataclass:
    """State with only JAX arrays - dataclass version."""

    coefficients: jax.Array
    converged_mask: jax.Array
    iteration_count: jax.Array
    previous_log_likelihood: jax.Array


def create_test_state(variant_count: int, coefficient_count: int, use_dataclass: bool = True):
    """Create a test state with random data."""
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    coefficients = jax.random.normal(key1, (variant_count, coefficient_count))
    converged_mask = jnp.zeros((variant_count,), dtype=bool)
    iteration_count = jnp.zeros((variant_count,), dtype=jnp.int32)
    previous_log_likelihood = jax.random.normal(key3, (variant_count,))

    if use_dataclass:
        return StateDataclass(
            coefficients=coefficients,
            converged_mask=converged_mask,
            iteration_count=iteration_count,
            previous_log_likelihood=previous_log_likelihood,
        )
    return StateNamedTuple(
        coefficients=coefficients,
        converged_mask=converged_mask,
        iteration_count=iteration_count,
        previous_log_likelihood=previous_log_likelihood,
    )


@jax.jit
def update_state_namedtuple(state: StateNamedTuple) -> StateNamedTuple:
    """Simple update function for NamedTuple state."""
    return StateNamedTuple(
        coefficients=state.coefficients + 0.01,
        converged_mask=state.converged_mask | (state.iteration_count > 5),
        iteration_count=state.iteration_count + 1,
        previous_log_likelihood=state.previous_log_likelihood * 0.99,
    )


@jax.jit
def update_state_dataclass(state: StateDataclass) -> StateDataclass:
    """Simple update function for dataclass state."""
    return StateDataclass(
        coefficients=state.coefficients + 0.01,
        converged_mask=state.converged_mask | (state.iteration_count > 5),
        iteration_count=state.iteration_count + 1,
        previous_log_likelihood=state.previous_log_likelihood * 0.99,
    )


def condition_namedtuple(state: StateNamedTuple) -> jax.Array:
    """Loop condition for NamedTuple."""
    return jnp.max(state.iteration_count) < 10


def condition_dataclass(state: StateDataclass) -> jax.Array:
    """Loop condition for dataclass."""
    return jnp.max(state.iteration_count) < 10


def benchmark_jit(variant_count: int, coefficient_count: int, iterations: int = 100) -> tuple[float, float]:
    """Benchmark JIT compilation and execution."""
    state_nt = create_test_state(variant_count, coefficient_count, use_dataclass=False)
    state_dc = create_test_state(variant_count, coefficient_count, use_dataclass=True)

    # Warmup JIT
    _ = update_state_namedtuple(state_nt)
    _ = update_state_dataclass(state_dc)

    # Benchmark NamedTuple
    start = time.perf_counter()
    for _ in range(iterations):
        state_nt = update_state_namedtuple(state_nt)
    # Access one of the array fields to trigger sync
    _ = state_nt.coefficients.block_until_ready()
    namedtuple_time = time.perf_counter() - start

    # Benchmark Dataclass
    start = time.perf_counter()
    for _ in range(iterations):
        state_dc = update_state_dataclass(state_dc)
    _ = state_dc.coefficients.block_until_ready()
    dataclass_time = time.perf_counter() - start

    return namedtuple_time, dataclass_time


def benchmark_while_loop(variant_count: int, coefficient_count: int, iterations: int = 100) -> tuple[float, float]:
    """Benchmark while_loop performance."""

    @jax.jit
    def loop_namedtuple(initial_state: StateNamedTuple) -> StateNamedTuple:
        def body_fn(state):
            return StateNamedTuple(
                coefficients=state.coefficients + 0.01,
                converged_mask=state.converged_mask | (state.iteration_count > 5),
                iteration_count=state.iteration_count + 1,
                previous_log_likelihood=state.previous_log_likelihood * 0.99,
            )

        return jax.lax.while_loop(condition_namedtuple, body_fn, initial_state)

    @jax.jit
    def loop_dataclass(initial_state: StateDataclass) -> StateDataclass:
        def body_fn(state):
            return StateDataclass(
                coefficients=state.coefficients + 0.01,
                converged_mask=state.converged_mask | (state.iteration_count > 5),
                iteration_count=state.iteration_count + 1,
                previous_log_likelihood=state.previous_log_likelihood * 0.99,
            )

        return jax.lax.while_loop(condition_dataclass, body_fn, initial_state)

    state_nt = create_test_state(variant_count, coefficient_count, use_dataclass=False)
    state_dc = create_test_state(variant_count, coefficient_count, use_dataclass=True)

    # Warmup
    _ = loop_namedtuple(state_nt)
    _ = loop_dataclass(state_dc)

    # Benchmark NamedTuple
    start = time.perf_counter()
    for _ in range(iterations):
        state_nt = loop_namedtuple(state_nt)
    _ = state_nt.coefficients.block_until_ready()
    namedtuple_time = time.perf_counter() - start

    # Benchmark Dataclass
    start = time.perf_counter()
    for _ in range(iterations):
        state_dc = loop_dataclass(state_dc)
    _ = state_dc.coefficients.block_until_ready()
    dataclass_time = time.perf_counter() - start

    return namedtuple_time, dataclass_time


def benchmark_tree_operations(variant_count: int, coefficient_count: int, iterations: int = 100) -> tuple[float, float]:
    """Benchmark tree_map and tree_flatten operations."""
    state_nt = create_test_state(variant_count, coefficient_count, use_dataclass=False)
    state_dc = create_test_state(variant_count, coefficient_count, use_dataclass=True)

    # Benchmark NamedTuple tree operations
    start = time.perf_counter()
    for _ in range(iterations):
        leaves, treedef = jax.tree.flatten(state_nt)
        new_leaves = [x + 1.0 for x in leaves]
        state_nt = jax.tree.unflatten(treedef, new_leaves)
    _ = state_nt.coefficients.block_until_ready()
    namedtuple_time = time.perf_counter() - start

    # Benchmark Dataclass tree operations
    start = time.perf_counter()
    for _ in range(iterations):
        leaves, treedef = jax.tree.flatten(state_dc)
        new_leaves = [x + 1.0 for x in leaves]
        state_dc = jax.tree.unflatten(treedef, new_leaves)
    _ = state_dc.coefficients.block_until_ready()
    dataclass_time = time.perf_counter() - start

    return namedtuple_time, dataclass_time


def main():
    """Run all benchmarks."""
    print("JAX Dataclass vs NamedTuple Performance Benchmark")
    print("=" * 60)
    print()

    sizes = [
        (100, 10),  # Small
        (1000, 20),  # Medium
        (5000, 50),  # Large
    ]

    for variant_count, coefficient_count in sizes:
        print(f"Size: {variant_count} variants x {coefficient_count} coefficients")
        print("-" * 60)

        # JIT benchmark
        nt_time, dc_time = benchmark_jit(variant_count, coefficient_count, iterations=100)
        speedup = nt_time / dc_time if dc_time > 0 else float("inf")
        print("  JIT Update (100 iters):")
        print(f"    NamedTuple: {nt_time:.4f}s")
        print(f"    Dataclass:  {dc_time:.4f}s")
        print(f"    Speedup:    {speedup:.2f}x")
        print()

        # While loop benchmark
        nt_time, dc_time = benchmark_while_loop(variant_count, coefficient_count, iterations=100)
        speedup = nt_time / dc_time if dc_time > 0 else float("inf")
        print("  While Loop (100 iters):")
        print(f"    NamedTuple: {nt_time:.4f}s")
        print(f"    Dataclass:  {dc_time:.4f}s")
        print(f"    Speedup:    {speedup:.2f}x")
        print()

        # Tree operations benchmark
        nt_time, dc_time = benchmark_tree_operations(variant_count, coefficient_count, iterations=100)
        speedup = nt_time / dc_time if dc_time > 0 else float("inf")
        print("  Tree Operations (100 iters):")
        print(f"    NamedTuple: {nt_time:.4f}s")
        print(f"    Dataclass:  {dc_time:.4f}s")
        print(f"    Speedup:    {speedup:.2f}x")
        print()

    print("=" * 60)
    print("\nNote: Speedup > 1.0 means dataclass is faster")
    print("      Speedup < 1.0 means NamedTuple is faster")


if __name__ == "__main__":
    main()
