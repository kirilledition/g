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
