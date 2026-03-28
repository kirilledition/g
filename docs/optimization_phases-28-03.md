## Logistic optimization phases - 2026-03-29 (updated)

Goal: maximize warmed-state GPU throughput for `src/g/compute/logistic.py` (primary) and `src/g/compute/linear.py` (secondary) while preserving PLINK-equivalent correctness.

### Constraints

- Preserve numerical parity with current logistic/Firth paths and PLINK baselines.
- Optimize for warmed steady-state throughput first; compilation time is acceptable.
- Avoid host-device synchronization in hot loops unless profiling proves it is harmless.
- Keep benchmark/profiling outputs machine-readable for before/after comparisons.
- Run `just check` and `just test` after each self-sufficient optimization step.

## Benchmark environment (current recommendation)

### Throughput-first default (benchmarks)

Use this first on the local RTX 4080 SUPER:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.95
```

Notes:

- Keep Triton GEMM enabled by default (do not set `--xla_gpu_enable_triton_gemm=false`).
- Recent local probes showed a small warmed-state advantage with Triton enabled.

### Stability fallback (profiling and low-memory runs)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.50
```

Use this when profiler or benchmark runs fail with cuBLAS/cuSOLVER handle allocation errors.

## Completed optimizations

### Already implemented and kept

- Profiling/benchmarking improvements:
  - JSON outputs for benchmark scripts.
  - Structured summary export in `scripts/profile_full_chr22_detailed.py`.
  - Perfetto trace summarization and optional transfer-guard profiling.
  - Warmed-only profile support via `--warmup-pass-count` in `scripts/profile_full_chr22_detailed.py`.
- Logistic standard path improvements:
  - Removed redundant hot-path dtype coercions when inputs are already device `float32`.
  - Switched heavy batched information solves to batched Cholesky + triangular solve.
  - Reduced loop-control overhead by replacing per-iteration `jnp.max(iteration_count)` checks with scalar global counters in logistic IRLS loop state.
  - Precomputed flattened per-sample covariate outer products once per chunk and rewrote hot IRLS Fisher assembly as batched matmul.
  - Replaced hot-path covariate-score and cross-information `einsum` forms with direct matmul layouts.
- Firth path improvements:
  - Replaced explicit inverse and generic solves with Cholesky-based solves in the single-variant Firth solver.
  - Reduced Firth leverage/solve overhead by reusing factorization and avoiding redundant solve dispatch work.
- Linear path improvement:
  - Replaced `cho_solve` wrapper usage with explicit triangular solves in SPD systems.

### Investigated but not adopted

- Device-only padded Firth fallback plan: rejected for steady-state throughput (inactive padded lanes still paid heavy compute cost).
- Transient grouped/packed merge experiments: rejected (regressed merge hotspot vs simple per-field scatter).
- Full device-resident no-missing planner path: mixed result (better first-run/profiled latency, worse warmed fallback-heavy throughput).
- Full no-missing specialized Firth kernel split (mask-free variant stack): reverted after instability/regression in fallback-heavy sweeps.

## Ideas that do not make sense to prioritize now

- Blanket host-side dtype-cast cleanup as a standalone project.
- `jax.pmap` for Firth on a single-GPU workstation.
- Generic masking-removal rewrites without profiler evidence.
- Factorization reuse across IRLS iterations where weights change each iteration.
- More standard-IRLS covariate-information rewrites without fresh kernel evidence.
- Linear-kernel algebra rewrites unless full-chromosome profiling shows linear is materially on the critical path.
- Any optimization that increases GPU handle churn or adds many new tiny solver invocations per iteration.

## Active backlog (warmed-throughput focused)

### P0 - Reduce fallback orchestration and merge overhead

Why:

- Fresh warmed profiles still show `compute_batched_firth_updates_without_mask(...)` as the dominant chunk-level hotspot.
- `merge_firth_result_once(...)` and scatter-heavy merge operations remain a meaningful secondary cost.

Plan:

- Fuse fallback result materialization to reduce per-field concatenate/scatter traffic.
- Reduce host round-trips in fallback planning (`device_get` of batch counts and masks) where possible.
- Keep fixed-bucket strategy only if inactive-lane compute does not increase.

Success criterion:

- Improved fallback-heavy benchmark (`variant_limit=2048`, `chunk_size=256`) with unchanged checksums.

### P0 - Harden warmed benchmarking/profiling execution presets

Why:

- Throughput-first memory settings can intermittently fail with cuBLAS/cuSOLVER handle allocation errors on long profiling runs.

Plan:

- Keep throughput-first settings for benchmark comparisons.
- Add an explicit low-memory profiling preset to avoid run failures.
- Record environment preset in every benchmark/profile JSON artifact.

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
- Firth leverage/factorization work improved fallback-heavy throughput materially relative to older baseline runs.
- Full no-missing Firth kernel specialization attempt was reverted due instability/regression signals in fallback-heavy runs.
- Because of that, further Firth changes should be incremental and benchmark-gated, not large-path rewrites.

## Recommended next implementation order

1. Optimize fallback merge/orchestration in `compute_batched_firth_updates_without_mask(...)`.
2. Add benchmark/profile preset metadata and stability guardrails.
3. Re-profile and only then consider another small Firth arithmetic optimization.
4. Leave `src/g/compute/linear.py` unchanged unless future full-run profile elevates it.

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

1. Re-baseline fallback-heavy and loop-sweep benchmarks under throughput-first env.
2. Run warmed detailed profile with low-memory preset if handle allocation failures appear.
3. Prioritize fallback merge/orchestration reductions with checksum parity gates.
