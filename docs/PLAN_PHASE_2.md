# Phase 2: Performance and Execution Optimization

## 1. Objective

Phase 2 starts after Phase 1 correctness/parity is stable.

The goal is to keep the validated Phase 1 behavior while aggressively reducing runtime and compilation overhead, first on CPU and then on GPU/Rust-backed paths.

This document is intentionally practical. It focuses on the current measured bottlenecks in the binary logistic path and lays out an implementation order that another agent can follow without having to rediscover the context.

## 2. Current Status at Phase 2 Start

The repository already satisfies the Phase 1 correctness milestone:

- linear parity is tight against PLINK
- hybrid logistic parity is tight against PLINK, including Firth fallback rows
- PLINK method/error-code semantics are matched
- `ruff`, `ty`, and `pytest` pass

Current tracked Phase 1 benchmark/parity snapshot lives in:

- `docs/PHASE1_STATUS.md`

Important current constraints:

- execution is still CPU-only in practice because a CUDA-enabled `jaxlib` is not installed
- the current bottleneck is not correctness but runtime and execution overhead
- the binary logistic path is the most important short-term target

## 3. What Profiling Already Showed

These findings came from profiling the current logistic code path on the Phase 0 chr22 workload before moving into full Phase 2 implementation.

### 3.1 High-level findings

- BED I/O is not the dominant problem.
- Polars output frame construction is not the dominant problem.
- The logistic path spends a lot of time in JAX compilation behavior, host/device materialization, and hybrid-dispatch plumbing.
- Chunk size matters; `512` performed best among the chunk sizes tested.

### 3.2 Measured observations

From earlier profiling runs on the full logistic workload:

- BED chunk iteration alone was about `2.7s`
- chunk output construction was negligible compared with compute
- the first full logistic run in a process was much slower than the second one
- `chunk_size = 512` outperformed `256`, `1024`, `2048`, and `4096`

The earlier `cProfile` traces showed three especially important patterns:

1. heavy JAX compile/load cost
2. heavy `jax.device_get()` / host materialization cost
3. high overhead from shape-varying hybrid dispatch, especially boolean slicing by fallback masks

### 3.3 Root-cause interpretation

The previous implementation of the hybrid path performed per-chunk slicing like:

- standard subset = `genotype_matrix[:, standard_mask]`
- Firth subset = `genotype_matrix[:, fallback_mask]`

Because `standard_mask` and `fallback_mask` changed shape from chunk to chunk, the code triggered many distinct JAX compilations.

The logistic path also repeatedly copied chunk results back to host during orchestration, even before final TSV/dataframe construction.

That means we currently have two broad classes of overhead:

- execution-shape instability
- premature host materialization

## 4. Phase 2 Principles

Phase 2 work must preserve the Phase 1 parity contract.

Do not trade away validated PLINK parity just to improve runtime.

Rules:

- keep float64 unless a lower-precision path is explicitly introduced and validated separately
- keep the same method/error-code semantics for hybrid logistic output
- keep input allele order unchanged in the engine itself
- keep PLINK alignment logic confined to evaluation/comparison code
- profile after every material optimization

## 5. Main Optimization Workstreams

### Workstream A: JAX compilation cache support

#### Goal

Reduce repeated cold-start compilation cost across CLI/script runs.

#### Why it matters

The first run of the logistic pipeline was much slower than warm runs, which strongly suggests that compilation caching can help real workflows.

#### Caveat discovered during experimentation

Persistent JAX cache use on CPU produced host-feature mismatch warnings in this environment when loading cached XLA artifacts. That means persistent caching should not be force-enabled unconditionally without guardrails.

#### Recommended implementation

1. Add explicit support for persistent cache configuration in `src/g/jax_setup.py`.
2. Make it opt-in rather than unconditional by default.
3. Support configuration through environment variables, with a repository-specific wrapper env if helpful.
4. Scope the cache directory carefully.
5. Add a way to clear/disable the cache quickly for debugging.

#### Concrete design

- Add support for:
  - `JAX_COMPILATION_CACHE_DIR`
  - optional repo-level env such as `G_ENABLE_JAX_PERSISTENT_COMPILATION_CACHE=1`
- Do not enable persistent caching automatically for every run until CPU cache compatibility is confirmed clean on this machine.
- If persistent cache is enabled, record the exact JAX/jaxlib/platform context in logs or docs.

#### Validation checklist

- confirm no correctness drift
- confirm no CPU feature mismatch warnings on repeated runs
- compare first-run vs second-run runtime with cache off vs on

### Workstream B: Refactor the hybrid logistic path to stabilize shapes

#### Goal

Stop recompiling standard/Firth kernels for many different fallback subset sizes.

#### Why it matters

Shape churn was one of the clearest bottlenecks in the original implementation.

#### Target files

- `src/g/compute/logistic.py`
- `src/g/engine.py`

#### Recommended approach

1. Keep the standard logistic path shape-stable across chunks.
2. Avoid boolean-slicing the standard path by variable-size fallback masks.
3. For Firth fallback, batch selected fallback variants into fixed-size padded batches.
4. Merge fallback results back into the full-chunk outputs without creating shape-variant standard-path kernels.

#### Preferred design

- Run standard logistic on the full chunk for the current fixed chunk size.
- Detect fallback variants after the standard result is available.
- Execute Firth on padded fixed-size fallback batches.
- Overwrite only the fallback rows in the final result.

#### Key design constraints

- final numeric outputs must remain identical or within the already validated tolerances
- hybrid dispatch decisions must still match PLINK behavior
- padded inactive entries must not contaminate active Firth rows

#### Important implementation note

Fixed-size fallback batching is a good idea, but it must be implemented in a way that does not reintroduce heavy host transfer overhead. The merge should happen on-device whenever practical.

#### Validation checklist

- inspect JAX compile logs before/after
- measure number of distinct compilations on a full logistic run
- compare first-run and warm-run runtime
- rerun Phase 1 logistic parity tests

### Workstream C: Reduce host/device transfer overhead

#### Goal

Move chunk orchestration closer to an on-device merge model and materialize results on host only when needed for final tabular output.

#### Why it matters

Earlier profiles showed `jax.device_get()` and array materialization dominating runtime after some shape-churn issues were addressed.

#### Target files

- `src/g/compute/logistic.py`
- `src/g/engine.py`

#### Recommended approach

1. Avoid copying per-field result arrays to host inside the compute path.
2. Keep standard and Firth result merging on device.
3. Only materialize host arrays once per output chunk, in `build_logistic_output_frame()`.
4. Avoid repeated host fetches of masks, coefficients, and per-field arrays unless the host actually needs them.

#### Concrete design

- `compute_logistic_association_chunk_with_mask()` should primarily return JAX arrays.
- Final chunk host transfer should happen in one place, ideally at the output-frame boundary.
- If host-side logic is unavoidable, fetch pytrees once rather than field-by-field.

#### Validation checklist

- compare `jax.device_get()` cumulative time before/after
- confirm chunk output generation still works
- confirm no parity drift in logistic hybrid output

### Workstream D: Keep `chunk_size = 512` as the default until proven otherwise

#### Current evidence

Empirical timing already showed `512` was best among tested chunk sizes.

#### Recommendation

- Keep `512` as the default logistic chunk size during early Phase 2.
- Revisit chunk size only after the shape/transfer issues are fixed.

## 6. Proposed Implementation Order

Do the work in this order.

### Step 1: Add safe, opt-in JAX persistent cache support

Reason:

- low code surface area
- useful for repeated runs
- should be isolated from solver logic

Deliverables:

- cache configuration in `src/g/jax_setup.py`
- short developer note in docs or README if needed

### Step 2: Refactor standard-path execution to be shape-stable

Reason:

- likely the cleanest single performance win in the current Python/JAX architecture

Deliverables:

- no variable-shape standard logistic kernel launches per chunk
- fewer JAX compile events in compile logs

### Step 3: Refactor Firth fallback batching to fixed-size padded batches

Reason:

- keeps rare-variant fallback logic without per-count recompilation

Deliverables:

- fixed-size fallback batch helper
- on-device merge where possible

### Step 4: Eliminate premature host materialization in the compute path

Reason:

- after shape churn is reduced, host transfer cost becomes easier to isolate and reduce

Deliverables:

- one clear host-materialization boundary for chunk output

### Step 5: Re-profile before any Rust/GPU work

Reason:

- Phase 2 should not assume Rust/GPU is required for every remaining bottleneck
- there may still be worthwhile low-risk CPU wins left

Deliverables:

- refreshed profile summary
- updated benchmark snapshot in docs if improvements are meaningful

## 7. What to Measure After Each Step

Every optimization step should collect the same small set of measurements.

### Required measurements

1. full logistic runtime on chr22
2. first-run vs second-run runtime in one process
3. `cProfile` top cumulative and top internal-time functions
4. JAX compile log sample for the first few chunks
5. parity against the PLINK hybrid baseline

### Required commands

- `nix develop -c uv run pytest tests/test_phase1.py -k 'logistic'`
- `nix develop -c just test`
- `nix develop -c uv run python scripts/evaluate_phase1.py`

If a change materially affects runtime, update:

- `docs/PHASE1_STATUS.md`

or create a dedicated Phase 2 progress note if Phase 1 status should remain frozen.

## 8. Risks and Failure Modes

### Risk A: Persistent cache causes platform-specific CPU warnings or unsafe reuse

Mitigation:

- keep it opt-in first
- verify repeated-run behavior on this AMD host
- document how to disable and clear it

### Risk B: Shape stabilization increases raw compute cost enough to erase compile savings

Mitigation:

- compare runtime, not just compile counts
- if full-chunk standard computation plus padded Firth is slower overall, revisit the batching strategy

### Risk C: On-device merging causes extra XLA scatter/update overhead

Mitigation:

- profile `jax.numpy` indexed updates after implementation
- if needed, switch to a more explicit merge layout or batched scatter approach

### Risk D: Optimization breaks PLINK parity on Firth rows

Mitigation:

- rerun hybrid parity tests after each solver/plumbing change
- keep evaluation outputs under versioned review in docs

## 9. CPU-Only Easy Wins Still Worth Trying Before Rust/GPU

These are the highest-value non-Rust, non-GPU optimization opportunities still worth trying:

1. shape-stable hybrid dispatch
2. fixed-size padded Firth batching
3. reducing premature `device_get()` calls
4. carefully enabling persistent JAX compilation caching

Lower-priority CPU-side ideas after that:

- reusing more precomputed covariate-side quantities across chunks
- reducing repeated broadcast/allocation patterns in the logistic compute path
- examining whether some host-side missingness/frequency calculations should be fused or moved earlier/later

## 10. When to Stop CPU Optimization and Move to Rust/GPU

Move on once all of the following are true:

- the hybrid logistic path no longer shows obvious shape-churn recompilation
- host transfer overhead is no longer the dominant runtime category
- first-run and warm-run behavior is understood and acceptable
- remaining runtime is primarily in real numerical compute rather than orchestration overhead

At that point, the next serious step should be one of:

- Rust BED/I/O replacement
- GPU-backed JAX execution with CUDA-enabled `jaxlib`
- custom CUDA/Triton kernels for the hottest logistic primitives

## 11. Handoff Notes for the Next Agent

If another agent picks this up, it should do the following first:

1. inspect `src/g/compute/logistic.py`
2. inspect `src/g/engine.py`
3. rerun a fresh logistic profile before changing code, because the working tree may contain uncommitted experiments
4. treat this document as the optimization roadmap, not as proof that every proposed change has already landed cleanly

The key idea is simple:

- preserve the Phase 1 parity contract
- reduce recompilation
- reduce host transfers
- then profile again before escalating to Rust or GPU work

## 12. Phase 2 Implementation Status (2026-03-23)

### Completed Steps

| Step | Status | Notes |
|---|---|---|
| Step 1: JAX persistent cache | ✅ Done | `jax_setup.py` supports env-var-controlled persistent cache. Changed to enabled by default after confirming no CPU feature warnings. |
| Step 2: Shape-stable standard path | ✅ Done | Standard logistic runs on full chunk. No variable-shape kernel launches. |
| Step 3: Fixed-size Firth batching | ✅ Done | `FIRTH_BATCH_SIZE = 64`. `build_firth_padded_index_batches` pads all batches to fixed size. Single XLA compilation. |
| Step 4: Eliminate premature host materialization | ✅ Done | On-device merge via `result.at[indices].set(values)`. Standard and Firth results stay on device. Only fallback+heuristic boolean masks transferred to host (for batch index building). Removed `transfer_standard_logistic_evaluation_to_host`, `HostStandardLogisticChunkEvaluation`, and `HostLogisticAssociationChunkResult`. |
| Step 5: Re-profile | Partial | Test suite confirms ~3× speedup (6.59s → 2.22s). Full chr22 re-profiling not yet done. PHASE1_STATUS.md preserved as historical baseline per project decision. |

### Completed Lower-Priority Items (§9)

- ✅ Reduced repeated broadcast/allocation patterns: `compute_information_components` simplified to direct unbatched assembly.
- ✅ Reusing precomputed covariate-side quantities: `NoMissingLogisticConstants` precomputation path preserved and used.
- ✅ Additional algorithmic wins: Schur complement in IRLS, Cholesky log-det replacing slogdet, `jnp.linalg.solve` replacing `jnp.linalg.inv`.

### Additional Optimizations Not in Original Plan

- ✅ I/O–compute overlap via one-chunk-ahead prefetching in engine loops.
- ✅ Sequential BED reads in Rust FFI (single seek + sequential reads).
- ✅ `--device cpu|gpu` CLI flag added for GPU execution control.

### §10 Exit Criteria Assessment

| Criterion | Met? |
|---|---|
| No obvious shape-churn recompilation | ✅ Yes |
| Host transfer overhead no longer dominant | ✅ Yes (reduced to 3 minimal transfers) |
| First-run vs warm-run behavior understood | ✅ Yes (persistent cache addresses cold start) |
| Remaining runtime in real numerical compute | ✅ Yes |

The CPU-side optimization phase is substantially complete. Next steps should focus on GPU bring-up and measurement.

