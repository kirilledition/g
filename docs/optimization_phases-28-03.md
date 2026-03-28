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

## Active backlog (warmed-throughput focused)

### P0 - Refresh kernel-level profiling on current code

Why:

- The codebase changed materially; old hotspot assumptions are stale.

What to measure:

- warmed throughput for `chunk_size=512` and `chunk_size=1024`
- kernel launch count and dominant kernels in standard masked/no-missing IRLS
- fallback-heavy behavior at `variant_limit=2048`, `chunk_size=256`

Success criterion:

- clear attribution of time split between standard IRLS arithmetic, loop/control overhead, and fallback orchestration.

### P0 - Rework standard IRLS covariate information assembly only if profiling confirms matrix-assembly pressure

Why:

- `compute_covariate_information_matrix(...)` still materializes per-sample crossproducts each iteration.

Candidate directions (A/B under `nsys`/`ncu`):

- current `einsum + tensordot`
- weighted-matmul layouts that avoid large transient tensors
- alternate `dot_general`/`matmul` formulations for cross-information terms

Constraint:

- keep Schur-complement update and convergence semantics unchanged.

### P1 - Optimize fallback orchestration behind explicit mode split

Why:

- warm-throughput and first-run latency optimizations pull in opposite directions.

Plan:

- keep warmed-throughput path as production default
- optionally retain a low-orchestration mode for profiling and one-shot latency runs

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
