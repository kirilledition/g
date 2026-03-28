## Logistic optimization phases - 2026-03-28

Goal: optimize `src/g/compute/logistic.py` for GPU-first GWAS execution while preserving PLINK-equivalent results.

### Constraints

- Preserve numerical parity with the existing logistic/Firth paths and PLINK baselines.
- Prefer GPU-friendly execution even if the implementation becomes more specialized.
- Avoid unnecessary host-device transfers.
- Export profiler outputs in machine-readable form for later analysis.
- Run lint/tests and create a git commit after each major phase.

### Baseline capture

1. Record runtime environment and visible JAX devices.
2. Run benchmark scripts to capture current throughput and fallback behavior.
3. Capture detailed profiling artifacts with:
   - plain-text cProfile summary
   - raw cProfile `.prof`
   - JAX Perfetto trace directory and `perfetto_trace.json.gz`
   - JAX memory profile
4. Store baseline numbers in a JSON report for before/after comparison.

### Phase 1 - profiling and benchmark export

Deliverables:

- A durable profiling script that exports summary metrics as JSON.
- Structured timing for logistic standard path, missing-data path, and Firth fallback path.
- Transfer-guard-aware profiling mode to surface host/device synchronization.

Hypothesis:

- Current profiling already captures traces, but it does not yet emit enough structured summaries for repeatable analysis and regression tracking.

### Phase 2 - remove host/device orchestration bottlenecks

Deliverables:

- Reduce Python-side orchestration in logistic fallback handling.
- Minimize `jax.device_get(...)` on fallback masks and counts.
- Shift more batch planning/selection/merging onto the device when practical.

Hypothesis:

- The masked path still performs host-side fallback planning and batch-by-batch orchestration. That likely serializes execution and introduces synchronization overhead that is especially costly on accelerators.

### Phase 3 - optimize standard logistic IRLS kernels

Deliverables:

- Benchmark-informed updates to the standard IRLS linear algebra.
- Reuse specialized solves where numerically safe.
- Reduce redundant matrix assembly and intermediate traffic.

Hypothesis:

- The standard no-missing and masked IRLS loops dominate chunk runtime. Small-matrix solves and repeated per-iteration tensor assembly may still leave GPU performance on the table.

### Phase 4 - optimize Firth fallback kernels

Deliverables:

- Remove explicit inverse operations where a solve or factorization reuse is sufficient.
- Reuse information-system factorizations inside each Firth iteration when possible.
- Keep the implementation numerically stable for separation-heavy variants.

Hypothesis:

- Firth is a smaller share of total runtime on clean chunks, but it becomes dominant on fallback-heavy chunks and is currently expensive due to repeated solves and a final matrix inverse.

### Phase 5 - final validation and comparison

Deliverables:

- Re-run the same baseline benchmarks.
- Compare before/after throughput and fallback timings.
- Confirm parity tests and targeted coverage tests pass.
- Summarize optimized paths, remaining bottlenecks, and next candidates.

### Suggested command sequence

Run inside `nix develop`:

```bash
just benchmark-jax
just benchmark-logistic-loop
just benchmark-logistic-fallback
just profile-logistic-detailed
just check
just test
```

If iteration speed matters during development, use the Python scripts directly with a smaller `--variant-limit`, then rerun the full benchmark set at the end.
