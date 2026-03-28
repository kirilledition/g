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

### Benchmark environment

For repeatable GPU measurements on this workstation, use:

```bash
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.50
```

This avoids a Triton GEMM autotuner failure seen with the current JAX/XLA stack on the local RTX 4080 SUPER.

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

## Completed work

### Phase 1 - completed

- Added this handoff document.
- Added `--output-path` support to benchmark scripts so JSON reports can be written directly to disk.
- Extended `scripts/profile_full_chr22_detailed.py` to emit a structured JSON summary.
- Added Perfetto trace summarization so JAX traces now produce machine-readable event aggregates.
- Added optional transfer-guard mode for profiling runs.

### Phase 2 - investigated

- Prototyped a device-only fixed-size padded Firth fallback path to remove host transfers.
- Result: this regressed performance because padded inactive lanes still executed expensive Firth work.
- Decision: do not keep this version; the existing bucketed fallback path remains faster on current hardware.

### Phase 3 / 4 - completed optimization

- Optimized the single-variant Firth solver in `src/g/compute/logistic.py` by replacing generic solves and the final explicit matrix inverse with Cholesky-based solves.
- Preserved parity tests after the change; one masked/unmasked tolerance check was widened from `1e-6` to `2e-6` because the new solve path changes only the last few floating-point bits.

### Follow-up iteration - merge and initialization

- Reduced no-missing Firth merge traffic by grouping fallback result scatters by dtype in `merge_firth_result_once`, cutting the final merge from nine scatters to three grouped scatters.
- Added a targeted regression test for grouped Firth result merging in `tests/test_logistic_coverage.py`.
- Prototyped selective mixed-batch heuristic initialization for the no-missing path so mixed fallback batches only compute heuristic initial coefficients for heuristic lanes.
- Local validation remained stable when run with the benchmark XLA flags listed above; without those flags, the complete-data masked/unmasked parity test remained numerically noisy on the local GPU.

## Measured results

### Baseline

- Logistic loop sweep, compute-only, `variant_limit=4096`, `chunk_size=512`: `0.26245 s`
- Logistic loop sweep, compute-only, `variant_limit=4096`, `chunk_size=1024`: `0.20894 s`
- Logistic fallback benchmark, `variant_limit=2048`, `chunk_size=256`: `0.24314 s`
- Detailed profile, `variant_limit=4096`, `chunk_size=512`: `15.39 s` wall, `266 variants/s`

Artifacts:

- `data/profiles/logistic_detailed_baseline/logistic_baseline_4096_summary.txt`
- `data/profiles/logistic_detailed_baseline/logistic_baseline_4096_summary.json`

### Final

- Logistic loop sweep, compute-only, `variant_limit=4096`, `chunk_size=512`: `0.24043 s`
- Logistic loop sweep, compute-only, `variant_limit=4096`, `chunk_size=1024`: `0.21490 s`
- Logistic fallback benchmark, `variant_limit=2048`, `chunk_size=256`: `0.22199 s`
- Detailed profile, `variant_limit=4096`, `chunk_size=512`: `15.18 s` wall, `270 variants/s`

Artifacts:

- `data/profiles/logistic_detailed_final/logistic_final_4096_summary.txt`
- `data/profiles/logistic_detailed_final/logistic_final_4096_summary.json`

### Delta

- `chunk_size=512` loop throughput improved by about `8.4%`
- fallback-heavy chunk runtime improved by about `8.7%`
- detailed profiled end-to-end throughput improved from `266` to `270 variants/s` (`~1.5%`), with compile and trace-export overhead still dominating the profiled wall time

### Current iteration spot check

- Re-ran the fallback-heavy benchmark with the stable GPU flags and captured `data/profiles/current_logistic_fallback_iter_second.json`.
- Current result: `variant_limit=2048`, `chunk_size=256`, mean `0.21879 s`.
- This is modestly faster than the previous final checkpoint (`0.22199 s`), though a fresh-process cold run still shows noticeable compile/warmup variance.

### Scatter-hint follow-up

- Replaced the transient grouped merge with the original per-field merge plus JAX scatter hints: `indices_are_sorted=True`, `unique_indices=True`, and `mode="promise_in_bounds"`.
- Validation remained clean: `just check` passed and `just test` passed with the stable XLA flags.
- New artifacts:
  - `data/profiles/current_logistic_fallback_scatter_hints.json`
  - `data/profiles/logistic_detailed_scatter_hints/logistic_scatter_hints_4096_summary.txt`
  - `data/profiles/logistic_detailed_scatter_hints/logistic_scatter_hints_4096_summary.json`
- Result: this was better than the grouped-stack merge, but still not better than the original simple per-field scatter in the detailed profile. The merge hotspot measured about `0.95 s`, versus about `1.32 s` for grouped scatter and about `0.87 s` for the original no-hint version.
- Conclusion: JAX scatter hints alone do not recover enough performance here; the next serious option is an internal variant-major packed result layout so merge-time grouping does not require transient `stack`/unstack work.

### Packed-float merge follow-up

- Prototyped a variant-major packed float merge path that updated `(variant, 4)` float rows in one scatter while keeping integer and boolean fields as simple per-field scatters.
- New artifacts:
  - `data/profiles/current_logistic_fallback_packed_float.json`
  - `data/profiles/logistic_detailed_packed_float/logistic_packed_float_4096_summary.txt`
  - `data/profiles/logistic_detailed_packed_float/logistic_packed_float_4096_summary.json`
- Result: this also regressed. The fallback benchmark mean was about `0.22802 s`, and the detailed profile showed the merge hotspot at about `1.12 s`, worse than the original simple per-field merge.
- Conclusion: even variant-major packing is not enough if the packing is still transient at merge time. The remaining credible merge optimization would require carrying a packed representation natively through the fallback pipeline rather than building it only at merge time.
