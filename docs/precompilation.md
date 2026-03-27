# JAX Precompilation Plan

## Goal

Reduce or eliminate expensive JAX compilation on rented GPU machines by letting users pre-populate the persistent compilation cache on a cheaper but compatible machine, then transfer that cache to the production environment.

This document is a future implementation plan. It does not describe fully implemented behavior yet.

## Product Goal

Provide a workflow that lets users:

1. Prepare a JAX compilation cache ahead of time.
2. Copy that cache to the target machine.
3. Run the real GWAS job with minimal first-run compilation overhead.

## Important Constraint

JAX compilation artifacts are not universally portable.

Cache reuse should be treated as supported only when the source and target environments are compatible, including at least:

- same `jax` version
- same `jaxlib` version
- same backend type (`gpu`)
- same float32 runtime configuration
- same effective shapes and static arguments
- same or highly similar GPU architecture
- compatible CUDA/XLA/runtime stack

If compatibility does not hold, JAX may ignore the cache and compile again.

## Existing Building Blocks

The repository already has the core primitive needed for this feature:

- persistent compilation cache configuration in `src/g/jax_setup.py`
- stable bucket sizes for no-missing Firth fallback in `src/g/compute/logistic.py`

This means a cache-warming workflow can be built without changing the overall execution model.

## Proposed User-Facing Commands

### `g compile-cache`

Create or refresh the JAX persistent compilation cache for a specified set of workloads.

Suggested options:

- `--device gpu`
- `--numeric-mode float32`
- `--chunk-sizes 128,256,512,1024`
- `--models linear,logistic`
- `--include-missing-path`
- `--cache-dir <path>`
- `--output-metadata <path>`

Behavior:

- configure JAX to use the requested cache directory
- run warmup calls for hot jitted kernels with synthetic arrays
- force compilation for all expected shape families
- write metadata describing the cache environment

### `g cache-info`

Inspect an existing cache directory and print the metadata needed to determine whether it is likely reusable on the current machine.

### `g run --require-cache-hit`

Optional strict runtime mode that warns or fails when an analysis triggers significant recompilation on the target machine.

## Warmup Coverage Plan

The warmup phase should compile the kernels that dominate current runtime.

### Linear path

Warm the chunk-level linear association path for representative chunk sizes.

Suggested shapes:

- phenotype length equal to representative sample count
- chunk widths: `128`, `256`, `512`, `1024`

### Logistic standard path

Warm both:

- no-missing logistic chunk path
- missing-data logistic chunk path

Suggested chunk widths:

- `128`, `256`, `512`, `1024`

### Logistic Firth fallback path

Warm all supported Firth bucket sizes in the no-missing fallback scheduler:

- `1`, `2`, `4`, `8`, `16`, `32`, `64`

Warm both cases:

- no heuristic initialization lanes
- mixed heuristic / standard initialization lanes

If the masked fallback path remains important, warm representative masked batches too.

## Implementation Strategy

### 1. Add a dedicated warmup module

Create a module responsible for building synthetic input arrays with the same dtypes and representative shapes as production code.

Suggested file:

- `src/g/precompile.py`

Responsibilities:

- create representative covariate, phenotype, genotype, and mask arrays
- call the hot jitted functions in a deterministic order
- cover all supported chunk sizes and Firth bucket sizes

### 2. Warm by calling real jitted entry points

Prefer calling the same public or near-public jitted kernels used by the real program rather than inventing parallel fake wrappers.

This keeps warmup aligned with the actual compiled signatures.

Current candidates:

- linear chunk compute path in `src/g/compute/linear.py`
- `compute_standard_logistic_association_chunk_without_mask`
- `compute_standard_logistic_association_chunk_with_mask`
- `compute_firth_association_chunk_with_mask`
- `compute_logistic_association_chunk`

### 3. Store cache compatibility metadata

Write a metadata file alongside the cache directory, for example `cache-metadata.json`, containing:

- `jax` version
- `jaxlib` version
- backend
- GPU device kind
- float dtype configuration
- matmul precision
- supported chunk sizes warmed
- supported Firth bucket sizes warmed
- optional timestamp and git revision

This should be checked by `g cache-info` and optionally by the runtime path.

### 4. Add a transfer-friendly cache layout

Allow users to choose a self-contained cache directory, for example:

- `/path/to/g-jax-cache/`

The directory should contain:

- JAX cache files
- metadata file
- optional warmup manifest

This makes it easy to archive and move with `tar` or object storage.

### 5. Add runtime diagnostics

At runtime, print a concise summary such as:

- cache directory used
- metadata compatibility status
- whether the current device appears compatible
- whether new compilations were still triggered

## Recommended Warmup Policy

For production users on expensive GPU machines:

- warm on a cheaper machine with the same GPU family if possible
- use the same container, Nix shell, or software image as production
- use the same float32 runtime configuration and chunk sizes intended for production

This should be documented as a compatibility requirement, not a soft suggestion.

## Validation Plan

### Functional validation

- generated cache is accepted on a second compatible machine
- first production run avoids most compilation stalls
- outputs remain unchanged relative to normal execution

### Performance validation

Measure:

- cold run without precompiled cache
- warm run with transferred cache
- cache-hit warm run on the same machine

Primary success metrics:

- lower startup latency
- lower time spent in `compile_or_get_cached`
- reduced end-to-end wall time for the first production run

## Risks

- cache portability may be narrower than users expect
- warming too many shapes can make cache generation slow
- missing a key shape family may still cause runtime compilation
- driver or XLA differences can invalidate reuse unexpectedly

## Recommended First Milestone

Implement a minimal but useful version:

1. `g compile-cache` for GPU `float32`
2. warm logistic standard path for chunk sizes `256` and `512`
3. warm all no-missing Firth bucket sizes `1,2,4,8,16,32,64`
4. emit compatibility metadata
5. document how to copy `JAX_COMPILATION_CACHE_DIR` between machines

That milestone should already help users avoid paying the largest compilation cost on rented hardware.
