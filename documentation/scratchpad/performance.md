# Performance Notes

> Internal scratchpad. Not user docs. May stale.

## General Rules

- Profile before custom kernels.
- Stop if microbenchmarks win but hot app runs do not.
- Prod timing must not force JAX sync by default.
- Exact stage timings/trace mode diagnostic; can perturb perf.
- GPU via SLURM on `landau`; broad CPU via configured CPU SLURM.

## 2026-07-14 Binary GPU Crate Wave

Target: one binary trait with approximate Firth, full 1KG chromosome 22,
packed8 BGEN input, and a V100 on `landau`. The clean deep-profile artifacts are
under `data/profiles/crate_wave_deep_firth512_clean_20260714`.

Profile findings:

- The 20.021-second runner comprises 9.891 seconds of JAX runtime setup,
  0.187 seconds of native preparation, and 9.660 seconds of native execution.
- First-chunk tracing and executable loading dominate latency: the first result
  takes 8.187 seconds, while the remaining 25 chunks finish in 1.661 seconds.
- Nsight recorded 26 packed8 host-to-device copies of 82,051,072 bytes. They
  total 199.519 milliseconds, but none overlap GPU kernels.
- Warm batches arrive every 56.662 milliseconds. Transfer through the last GPU
  kernel takes 28.784 milliseconds, leaving the device idle for 27.878
  milliseconds while it waits for the next Rust-decoded batch.
- Output work is mostly overlapped. Its terminal flush is about 135
  milliseconds, so it is secondary to BGEN delivery and first-use JAX work.

Accepted:

- Reusable per-Rayon-worker `libdeflate` decoding replaces `zlib-rs` for UKB's
  Layout-2 zlib blocks. The 16,384-variant full-sample packed8 benchmark improves
  9.86% on `landau` and 3.28% on `cantor`. Full chr22 validation compared all
  418,943 variants, 2,098,066,544 probability bytes, and derived statistics
  exactly.
- The paired full-GPU comparison remains wall-time neutral (+0.16%, p=0.88), as
  expected when JAX startup and filesystem variance dominate a sub-second Rust
  improvement. Retain the change based on the isolated hot-path result, not an
  unsupported end-to-end claim.

Rejected:

- Forced JAX transfer lookahead does not improve the current decode-bound
  pipeline. Revisit only when decoded batches arrive faster than the measured
  28.784-millisecond GPU stage.
- AVX-512 packed8 copy/statistics is 1.20% slower than AVX2 on the Xeon Gold
  5220, despite halving loop iterations.
- Eight chunks per Parquet file reduces terminal flush time but raises total
  Parquet work 12.8%; paired full-run wall time is unchanged. Keep 16.
- Skipping validated packed8 header parsing changes decode by only -0.24%
  (p=0.70), so retain the defensive checks.

Next targets:

- Reduce or avoid cold Firth executable deserialization/tracing before the first
  result.
- Continue zlib/packed8 decode work until CPU delivery approaches the GPU stage;
  only then reconsider one bounded asynchronous device-input lookahead.
- Profile the two dominant Firth reduction kernels before considering Pallas or
  custom CUDA fusion.

## 2026-07-14 Binary GPU Scalar/Allocation Wave

The target and machine match the crate wave above. Production CLI artifacts are
under `data/benchmarks/next_wave_*_20260714` and
`data/benchmarks/next_wave_{prod_baseline,scalar_candidate,pool_candidate}_*`.
The obsolete binary-hot wrapper was not repaired because it still imports the
removed `g.api`; comparisons use the production CLI and its native progress
timestamps instead.

Accepted:

- Scalar approximate Firth now reduces the leverage adjustment directly and
  carries only scalar information, deviance, adjustment, and score state. It no
  longer carries a 512-by-2,504 probability matrix and information diagonal
  through Newton iterations. Cold wall time improved from 58.75 to 47.19
  seconds, and cold first-chunk time improved from 44.055 to 33.866 seconds.
  Across three non-cold runs the median association window improved from 9.711
  to 9.532 seconds. Full chr22 Parquet tables remained exactly equal.
- Packed8 decode now reuses at most three session-owned host buffers. A private
  immutable NumPy base object retains each allocation through JAX's asynchronous
  transfer, and `device_put(..., may_alias=False)` makes the copy contract
  explicit. The same-node Criterion median for a 16,384-by-2,504 batch improved
  from 59.188 to 40.991 milliseconds (30.7%). The syscall trace contains only
  three 82,055,168-byte mappings, all unmapped at process teardown, instead of
  allocation churn in the decode loop.
- Against three non-contended production baselines, the final three-run median
  wall time improved from 21.61 to 20.88 seconds (3.38%), the association window
  from 9.711 to 9.405 seconds (3.15%), and the post-first-chunk 25-batch window
  from 1.571 to 1.321 seconds (15.9%). Median system CPU time fell from 3.82 to
  2.53 seconds. The final output remains exactly equal across all 418,943 rows.

Next targets:

- Split the device-side 64/1,024/16,384 Firth mega-dispatch into separately
  cached fixed-capacity executables selected by one host-visible candidate
  count. Preserve overflow and candidate-ordering semantics, and reject the
  change if the scalar synchronization regresses the warm 25-batch window.
- Re-profile BGEN delivery after pooling before considering checksum elision.
  A validated raw-deflate path is only worthwhile if Adler-32 remains visible
  after allocation churn is gone.

## Output Performance

Historical profiling: output cost dominated by Rust Arrow writer + optional Parquet finalization, not Python/JAX handoff.

Signals:

- Arrow write + Parquet finalization largest stages.
- Larger chunks helped measured multi-trait.
- Fewer intermediate Arrow files helped measured multi-trait.
- Intermediate Arrow `zstd` not clear speed win.

Next:

- Refresh evidence before changing defaults.
- Re-test direct Parquet: current Arrow-then-Parquet writes Arrow chunks, reopens, reads, writes Parquet.
- Re-test wide multi-trait layout if per-trait metadata/file work dominates.
- Use `tooling.cli.benchmark_output_stages` for current measurements.

## Rust Build Profiles

Profiles:

```text
dev       fast PyO3 iteration
release   expensive maximum-performance builds and benchmarks
```

Rules: use `dev` for quick `maturin develop`; use `release` for benchmark evidence and final builds; keep profiler symbols; measure clean+incremental build; test app runtime, not compile-only.

Knobs:

- ThinLTO versus FatLTO.
- `codegen-units` 1, 4, 8, 16.
- `opt-level=2` versus `opt-level=3`.
- `target-cpu=native` on server-local builds.
- Mold is the default Linux linker; `sccache` remains environment policy.
- PGO only for serious release-candidate evidence.

Per profile measure clean build, incremental rebuild, shared-object size, import smoke, BGEN throughput, binary-hot GPU smoke, profiler symbols. Use `just bench-rust-build-profiles`.
