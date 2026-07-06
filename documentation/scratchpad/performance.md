# Performance Notes

> Internal scratchpad. Not user docs. May stale.

## General Rules

- Profile before custom kernels.
- Stop if microbenchmarks win but hot app runs do not.
- Prod timing must not force JAX sync by default.
- Exact stage timings/trace mode diagnostic; can perturb perf.
- GPU via SLURM on `landau`; broad CPU via configured CPU SLURM.

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
dev-fast   fast PyO3 iteration
dev-opt    mildly optimized local checks
perf       routine benchmark profile
perf-max   expensive final benchmark profile
perf-o2    opt-level=2 comparison profile
```

Rules: no fat LTO for normal iteration; `dev-fast` for quick `maturin develop`; `perf`/`perf-max` only for evidence; keep profiler symbols; measure clean+incremental build; test app runtime, not compile-only.

Knobs:

- ThinLTO versus FatLTO.
- `codegen-units` 1, 4, 8, 16.
- `opt-level=2` versus `opt-level=3`.
- `target-cpu=native` on server-local builds.
- Linker and `sccache` policy through environment, not hard-coded repo logic.
- PGO only for serious release-candidate evidence.

Per profile measure clean build, incremental rebuild, shared-object size, import smoke, BGEN throughput, binary-hot GPU smoke, profiler symbols. Use `just bench-rust-build-profiles`.
