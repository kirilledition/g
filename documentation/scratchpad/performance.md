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

## 2026-07-14 Binary GPU Firth Dispatch Wave

The target and machine are unchanged. Artifacts are under
`data/benchmarks/next_wave_dispatch_{count,legacycache}_*`. The accepted design
keeps the established score executable, computes the candidate count in one
small JIT, materializes only that scalar, and invokes one separately cached
fixed-capacity correction executable. The public compute API no longer builds
or forwards the four capacity tiers, and the correction kernel receives only
the `firth_se` flag it uses rather than the entire correction plan.

Accepted:

- Removing the nested 64/1,024/16,384/overflow `lax.cond` mega-executable cuts
  cold fixed-capacity correction compilation from 36.225 to 11.525 seconds.
  With an empty cache, cold wall time improves from 47.19 to 34.65 seconds and
  the association window from 35.482 to 23.044 seconds.
- Across three warm runs, median association time improves from 9.405 to 5.153
  seconds (45.2%) and median wall time from 20.88 to 16.42 seconds (21.4%). The
  per-chunk host synchronization is therefore decisively outweighed by the
  smaller correction executable on this workload.
- Repeated runs with the same compiled executables are bitwise identical across
  all 418,943 rows. Reusing the baseline score executable also produces a
  bitwise-identical table with the new dispatcher, including correction method
  and status. Separately compiled empty-cache score executables show only the
  expected GPU reduction variation and preserve all correction decisions.

Rejected:

- Folding the count reduction into a new retained-score wrapper was no faster
  than the isolated count kernel and enlarged `api.py`. Keeping the score JIT
  unchanged is smaller, cleaner, and permits existing score cache reuse.

Next targets:

- Re-profile the accepted pipeline. Prioritize Rust BGEN delivery if the GPU is
  still starved; only pursue raw-deflate/checksum work when the profile shows a
  measurable checksum share after buffer pooling.
- Inspect the now-isolated 1,024-candidate correction executable before writing
  custom GPU code. Require kernel-level evidence for any Pallas or CUDA path.

## 2026-07-14 Binary GPU Newton Reuse Wave

The target and machine are unchanged. The clean dispatch profile is under
`data/profiles/next_wave_deep_dispatch_clean_20260714`; production comparisons
are under `data/benchmarks/newton_{reuse,control,legacycache}_*`. All production
runs use 512-row Firth batches, 16,384-variant input chunks, eight host threads,
and four output writers.

Accepted:

- Newton step-halving now carries the accepted candidate's penalized deviance,
  genotype information, and score back to the outer iteration. The outer loop
  no longer recomputes the same full-sample Firth components immediately after
  line search. Failed or exhausted searches retain the already-known current
  scalars and follow the existing failure path.
- In seven candidate and seven immediately subsequent old-code control runs,
  median association time improved from 5.144 to 5.036 seconds (2.11%) and
  median native execution from 4.944 to 4.852 seconds (1.87%). Mean improvements
  were 1.83% and 1.78%, respectively.
- In the shared-cache comparison, every one of the 17,938 Firth-corrected rows
  remained bitwise equal, including correction method and status. Independently
  compiled score executables retain the previously documented GPU reduction
  variation.

Rejected:

- A five-result variadic Firth reduction is neutral end to end: 5.113 seconds
  median versus 5.122 seconds for the established implementation. GPU XLA was
  already combining the reductions, so the additional reducer and contribution
  helper were removed.
- Raw DEFLATE delivery that omitted Adler-32 after packed8 cache attestation is
  neutral when both candidate and baseline are compiled natively on `landau`:
  5.125 versus 5.122 seconds median association time. It also weakened detection
  of silent content changes, so all 100-plus experimental lines were removed.
- A 256-row Firth batch is 2.23% slower than 512 rows in the five-run median.
  Keep 512 for this target.
- The eight-thread packed8 Criterion median is 22.308 milliseconds with 32-row
  Rayon tiles. A 16-row tile is indistinguishable at 22.265 milliseconds; 64 and
  128 rows regress to 22.591 and 22.902 milliseconds. Keep the smaller existing
  code and the 32-row constant.

Next targets:

- Measure the marginal cost of packed8 sparse statistics before moving any
  integer summaries to the GPU. Do not add a public benchmark-only reader API.
- Evaluate pseudo-Firth component-state reuse only if it removes a guaranteed
  full-sample evaluation without introducing array state or batched conditional
  work.
- Re-profile the accepted Newton executable before considering a custom GPU
  kernel; the current profile lacks hardware counters because NCU access is
  disabled on `landau`.

## 2026-07-14 Binary GPU Pseudo-State Reuse Wave

The post-Newton profile is under
`data/profiles/next_wave_deep_newton_20260714`; production comparisons are under
`data/benchmarks/pseudo_{reuse,sharedcache}_*`.

Profile findings:

- Newton line-search reuse reduced native execution from 4.932 to 4.821 seconds
  in the deep-profile headline and from 4.895 to 4.793 seconds in the finalist.
  Nsight kernel calls fell from 19,196 to 17,303, total kernel time from 315.837
  to 288.871 milliseconds, and the GPU kernel span from 4.843 to 4.640 seconds.
- Warm correction work now has a 17.954-millisecond median, while correction
  starts remain 51.208 milliseconds apart. The 32.894-millisecond post-kernel
  delivery gap makes the full path host-delivery-bound even though solver work
  remains worth reducing for cold latency and future reader improvements.

Accepted:

- `ScalarPseudoFirthState` now carries the existing five-scalar
  `ScalarFirthComponents` value. It consumes the initial components directly,
  refreshes them once at each returned beta, preserves them on convergence, and
  uses the final state value for output. This removes the guaranteed initial
  duplicate and the normal converged-final duplicate without redefining the
  component fields or carrying sample-sized arrays.
- Across seven runs, median association time improves from 5.036 to 4.913
  seconds (2.44%) and median native execution from 4.852 to 4.728 seconds
  (2.56%). Fresh-cache cold wall time improves from 33.08 to 31.95 seconds and
  the cold association window from 21.763 to 20.536 seconds.
- Reusing the pre-change score cache produces a bitwise-identical table across
  all 418,943 rows, including every score, corrected statistic, correction
  method, and correction status.

Deferred:

- Disabling packed8 sparse-candidate counts improves the isolated eight-thread
  reader median from 22.308 to 21.678 milliseconds (2.82%). The absolute saving
  is only about 0.63 milliseconds per chunk, roughly a 0.3% association-time
  ceiling before adding replacement GPU reductions. Keep the current exact host
  mask until a lean correction-side derivation demonstrates an end-to-end win.

Next targets:

- Deep-profile the pseudo-state executable and verify that initial/final
  component kernels disappear before attempting further solver state changes.
- Treat H2D transfer and decoded-batch delivery as the primary warm-path target;
  they now dominate the gap between correction launches.

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
