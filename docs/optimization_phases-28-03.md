## Logistic optimization phases - 2026-03-28 (updated)

Goal: maximize warmed-state GPU throughput for `src/g/compute/logistic.py` (primary) and `src/g/compute/linear.py` (secondary) while preserving PLINK-equivalent correctness.

### Constraints

- Preserve numerical parity with current logistic/Firth paths and PLINK baselines.
- Optimize for warmed steady-state throughput first; compilation time is acceptable.
- Avoid host-device synchronization in hot loops unless profiling proves it is harmless.
- Keep benchmark/profiling outputs machine-readable for before/after comparisons.
- Run `just check` and `just test` after each self-sufficient optimization step.

## Benchmark environment (current recommendation)

### Throughput-first default

Use this first on the local RTX 4080 SUPER:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.95
```

Notes:

- Keep Triton GEMM enabled by default (do not set `--xla_gpu_enable_triton_gemm=false`).
- Recent local probes showed a small warmed-state advantage with Triton enabled.

### Compatibility fallback (only if Triton/autotune becomes unstable)

```bash
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.95
```

Use this only when the throughput-first preset fails or is numerically unstable in your environment.

## Completed optimizations

### Already implemented and kept

- Profiling/benchmarking improvements:
  - JSON outputs for benchmark scripts.
  - Structured summary export in `scripts/profile_full_chr22_detailed.py`.
  - Perfetto trace summarization and optional transfer-guard profiling.
- Logistic standard path improvements:
  - Removed redundant hot-path dtype coercions when inputs are already device `float32`.
  - Switched heavy batched information solves to batched Cholesky + triangular solve.
  - Reduced loop-control overhead by replacing per-iteration `jnp.max(iteration_count)` checks with scalar global counters in logistic IRLS loop state.
  - Precomputed flattened per-sample covariate outer products once per chunk and rewrote hot IRLS Fisher assembly as batched matmul.
  - Replaced hot-path covariate-score and cross-information `einsum` forms with direct matmul layouts.
- Firth path improvements:
  - Replaced explicit inverse and generic solves with Cholesky-based solves in the single-variant Firth solver.
- Linear path improvement:
  - Replaced `cho_solve` wrapper usage with explicit triangular solves in SPD systems.

### Investigated but not adopted

- Device-only padded Firth fallback plan: rejected for steady-state throughput (inactive padded lanes still paid heavy compute cost).
- Transient grouped/packed merge experiments: rejected (regressed merge hotspot vs simple per-field scatter).
- Full device-resident no-missing planner path: mixed result (better first-run/profiled latency, worse warmed fallback-heavy throughput).

## Ideas that do not make sense to prioritize

- Blanket host-side dtype-cast cleanup as a standalone project.
- `jax.pmap` for Firth on a single-GPU workstation.
- Generic masking-removal rewrites without profiler evidence.
- Factorization reuse across IRLS iterations where weights change each iteration.
- More standard-IRLS covariate-information rewrites without fresh kernel evidence.
- Linear-kernel algebra rewrites unless full-chromosome profiling shows linear is materially on the critical path.

## Active backlog (warmed-throughput focused)

### P0 - Refresh warmed-state profiling with compilation noise isolated

Why:

- The codebase changed materially and the remaining bottleneck moved.
- The current detailed profile still contains substantial lowering/cache-miss noise, which obscures steady-state GPU cost.

What to measure:

- warmed throughput for `chunk_size=512` and `chunk_size=1024`
- kernel launch count and dominant kernels in standard masked/no-missing IRLS
- fallback-heavy behavior at `variant_limit=2048`, `chunk_size=256`
- warmed full-chromosome logistic throughput after one explicit warmup pass
- relative time split between standard path, Firth compute, fallback planning, and host transfers

Success criterion:

- clear attribution of steady-state time split between Firth arithmetic, fallback orchestration, and any remaining standard-path kernels.

### P0 - Optimize no-missing Firth fallback compute

Why:

- Fresh profiling shows the dominant remaining chunk-level hotspot is now `compute_batched_firth_updates_without_mask(...)` and the vmapped `fit_single_variant_firth_logistic_regression(...)` path, not standard IRLS assembly.

Candidate directions:

- Rewrite leverage / projected-design work inside Firth to minimize repeated solves and transient matrices.
- Benchmark whether a bucket-specialized vmapped Firth kernel reduces overhead versus the current generic path.
- Inspect whether genotype-last scalar-stat extraction can avoid rebuilding full intermediate matrices late in the solver.

Constraint:

- Keep Firth convergence semantics and PLINK-equivalent output parity unchanged.

### P1 - Reduce fallback orchestration overhead only where profiles still justify it

Why:

- Fresh profile still shows planning/orchestration cost (`nonzero`, batch planning, Python-side batch loop), but it is now secondary to Firth arithmetic.

Plan:

- Keep warmed-throughput path as production default.
- Focus only on changes that reduce host transfers or batch-loop overhead without reintroducing padded inactive-lane compute.
- Prefer changes that preserve the current good fallback-heavy warmed throughput characteristics.

### P1 - Add a warmed-only profiling harness

Why:

- The existing detailed profile is useful, but first-run tracing/lowering work still dominates too much of the summary to guide steady-state kernel work precisely.

Plan:

- Warm once, then profile a second pass over representative chunk sizes and one full-chromosome run.
- Emit machine-readable summaries separating compile/lower time from warmed execution time.

### P2 - Revisit fallback merge layout only if merge reappears as top hotspot

Why:

- previous transient packed/grouped variants regressed.

Constraint:

- only pursue native packed pipeline representation if profiles show merge dominates again.

### P2 - Linear path: treat as opportunistic

Why:

- linear kernel is already near algebraic minimum and currently smaller in end-to-end impact.

Action:

- only invest further if fresh profile shows linear compute is material vs I/O/formatting.

## Fresh findings from the latest implementation pass

- The standard logistic no-missing IRLS matrix-assembly rewrite delivered a small warmed-state improvement, but not a large one:
  - `chunk_size=1024` compute-only improved slightly.
  - fallback-heavy `chunk_size=256` improved modestly.
  - `chunk_size=512` was effectively flat, and some compute+format surfaces regressed slightly.
- Fresh detailed profiling shows the remaining dominant chunk-level cost is Firth fallback compute and its orchestration, not the standard no-missing IRLS Fisher assembly.
- Because of that shift, more standard-path algebra rewrites should be deprioritized until fresh `nsys`/`ncu` evidence says otherwise.

## Recommended next implementation order

1. Add warmed-only profiling so steady-state kernel costs are isolated cleanly.
2. Optimize no-missing Firth fallback compute.
3. Recheck fallback orchestration only if it still consumes a meaningful share after the Firth rewrite.
4. Leave `src/g/compute/linear.py` alone unless a later full profile elevates it.

## Suggested command sequence

Run inside `nix develop` with the throughput-first env preset:

```bash
just benchmark-jax
just benchmark-logistic-loop
just benchmark-logistic-fallback
just profile-logistic-detailed
just check
just test
```

For iteration speed, use Python benchmark scripts with smaller `--variant-limit`, then rerun full sweeps before final comparison.

## Recommended profiling commands

```bash
# Timeline profiling to identify dominant kernels and launch overhead
nsys profile -o gwas_timeline python scripts/benchmark_logistic_fallback.py
nsys stats gwas_timeline.nsys-rep --report cuda_gpu_kern_sum

# Kernel-level metrics for candidate hot kernels
ncu --kernel-name "jit_.*" --metrics sm__warps_active.avg,sm__warps_eligible.avg,gpu__cycles_active.avg --set full -o gwas_kernels python scripts/benchmark_logistic_loop.py
```

Interpretation hints:

- low warps active + high cycles active -> latency/occupancy issue
- high sectors/request -> poor coalescing or excess temporary traffic
- high launch count with low per-kernel work -> fusion/control overhead candidate

## Immediate next steps

1. Re-baseline with throughput-first env and archive JSON outputs.
2. Confirm top warmed-state hotspot with `nsys` before touching kernels.
3. Only implement kernel algebra rewrites that win in warmed throughput and keep parity tests green.
