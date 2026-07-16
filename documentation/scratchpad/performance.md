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

## 2026-07-14 Binary GPU Reducer Debloat Wave

The pseudo-state deep profile is under
`data/profiles/next_wave_deep_pseudo_20260714`; the paired reducer comparison is
under `data/benchmarks/clamp_{control,candidate}_*`.

Profile findings:

- Pseudo-state reuse reduced finalist native execution from 4.793 to 4.661
  seconds (2.76%) and the headline native execution from 4.821 to 4.738 seconds
  (1.74%). Excluding first use, correction mean fell from 18.107 to 17.722
  milliseconds and its median from 17.954 to 17.802 milliseconds.
- The compiled full-component reducers remain the compute hotspot. Their 1,869
  dense launches process 512 by 2,504 samples and use 162 registers per thread.
  The preceding probability materialization accounts for another 2,083 launches
  and 97.169 milliseconds.
- Full packed8 H2D copies remain serialized with compute. The 26 copies move
  2.133 GB and take about 216 milliseconds; the warm buffers are already pinned
  and reuse their registration, so custom pinned allocators are not a target.

Accepted:

- Logistic deviance now consumes the probabilities already clipped by
  `compute_regenie_logistic_probability` instead of applying the same bounds a
  second time. The pseudo-logistic inner solver also omits its zero-weight
  reduction: endpoint clipping makes zero weight impossible, while the existing
  score and information finiteness checks still reject NaNs.
- Against the immediate five-run control, median native execution improves from
  4.767 to 4.661 seconds (2.21%) and median association time from 4.940 to 4.846
  seconds (1.92%). Mean improvements are 1.57% and 1.40%, respectively.
- All 418,943 correction methods and statuses remain identical. Independently
  lowered GPU executables differ only by normal float32 reassociation, with a
  maximum absolute statistic delta of 1.91e-6.

Deferred:

- A custom Pallas or CUDA map-reduce could remove the separate probability
  materialization, but fusing it into a 162-register reducer risks spills and
  adds a second kernel implementation. Do not add that architecture without
  hardware-counter evidence and a decisive end-to-end result.
- A bounded session-owned pool for Rayon worker libdeflate contexts regressed
  the same-node packed8 Criterion median from 40.921 to 41.647 milliseconds
  (1.78%). The lock and guard machinery was removed; retain the smaller
  per-iterator-state scratch allocation.

Next targets:

- Prototype decoded-batch pre-transfer on the existing delivery thread so H2D
  for the next batch can overlap current compute; do not add another worker or
  custom CUDA memory layer.

## 2026-07-14 Binary GPU Pre-Transfer Wave

The bounded production confirmation runs are under
`data/benchmarks/pretransfer_bounded_confirm_*`; the exact accepted build was
profiled under `data/profiles/next_wave_deep_pretransfer_20260714`. The target,
512-row Firth batches, 16,384-variant input chunks, and eight host threads are
unchanged.

Accepted:

- The delivery thread now validates and splits the next decoded batch, asks the
  backend to initiate its device transfer, and places the opaque transferred
  input in the existing one-slot compute queue. The current batch computes
  concurrently. This reuses the two established workers and bounded channels;
  it adds no worker, staging queue, custom allocator, or CUDA implementation.
  Queue capacity is checked before transfer, so backpressure retains the
  decoded host batch and bounds genotype device residency to the active batch
  plus one transferred successor.
- The Rust/Python boundary now has one native-buffer conversion and transfer
  path plus one generic compute call. Kernel selection remains in the concrete
  JAX backend rather than being duplicated as three mode-specific binding
  helpers.
- Against the preceding seven-run reducer checkpoint, the seven-run
  confirmation median improves from 4.669 to 4.362 seconds in native execution
  (6.56%) and from 4.849 to 4.565 seconds in association time (5.86%). Native
  execution ranges from 4.297 to 4.437 seconds. One association measurement is
  4.889 seconds because input preparation, not native execution, paused for an
  additional 395 milliseconds.
- Nsight observes full packed8 transfer overlapping kernels for 14 of 26
  batches, with 31.439 milliseconds of aggregate overlap. The 26 full transfers
  move 2.133 GB in 207.949 milliseconds. Except for startup and three bounded
  delivery waits, a kernel is queued within 0.02 milliseconds after each
  transfer completes.
- Output is bitwise identical to the reducer checkpoint across all 418,943
  rows, including metadata, statistics, correction methods, and correction
  statuses.

Next targets:

- The accepted standard Nsight profile records 17,801 kernel launches and
  useful relative kernel counts, but it disables XLA command buffers while
  profiling. Its 6.252-second native execution is not representative of the
  4.36--4.46-second production path, so do not optimize the apparent
  4.051-second launch span. Re-profile solver-control experiments with command
  buffers explicitly enabled during profiling.
- Revisit the full-component reducer only with hardware-counter evidence that a
  custom implementation avoids spilling its 164-register state.

## 2026-07-14 Binary GPU Scalar Initialization Reuse Wave

Production comparisons are under
`data/benchmarks/scalar_initial_reuse_clean_*`; the target, 512-row Firth
batches, 16,384-variant input chunks, eight host threads, and the shared JAX
cache are unchanged.

Accepted:

- Each active scalar lane now computes its beta-zero `ScalarFirthComponents`
  once. Pseudo-Firth consumes that value directly, the likelihood-ratio null
  deviance uses its genotype information, and the lazy Newton-Raphson fallback
  reuses the same value. This removes a guaranteed probability/information
  evaluation and another full-component evaluation whenever fallback runs.
- Production only ever supplied a zero warm start. The unreachable nonzero
  warm-start cascade, its three forwarding helpers, and the unused
  `firth_newton_raphson_zero_start_iterations` configuration field are removed
  through the Rust plan, binding, and JAX policy. The wave removes 110 net
  production and benchmark lines rather than adding a second solver path.
- Against the preceding seven-run bounded pre-transfer checkpoint, seven clean
  runs improve median native execution from 4.362 to 4.245 seconds (2.69%) and
  median association completion from 4.565 to 4.436 seconds (2.82%). Candidate
  native execution ranges from 4.139 to 4.465 seconds.
- Output remains bitwise identical across all 418,943 rows and every column,
  including all 17,938 corrected statistics, methods, and statuses.

Compatibility:

- Pre-release configurations must delete
  `firth_newton_raphson_zero_start_iterations`; it never selected a production
  execution path. Its removal changes the serialized execution-plan schema and
  hash; the run-manifest schema is therefore version 15, and output directories
  created by an older binary run must not be resumed with this build.

Measured follow-up:

- Six, seven, and eight Rayon workers are indistinguishable after pre-transfer.
  Their five-run native medians are 4.100, 4.134, and 4.122 seconds; mean and
  median rankings disagree, and association medians differ by less than 0.5%.
  Keep the explicit eight-worker target and do not add an automatic host-core
  reservation policy.
- The full-component validity predicate no longer reduces
  `isfinite(probability_vector)` across every row. Validated predictors produce
  finite clipped probabilities, while active-row failures already make the
  deviance, information, or score invalid. The deleted fifth reducer uniquely
  rejected only a NaN confined to masked-out sparse or padding rows.
- Against a fresh seven-run control, the reduced predicate improves median
  native execution from 4.135 to 4.107 seconds (0.68%) and median association
  completion from 4.339 to 4.304 seconds (0.81%). Mean gains are only 0.30% and
  0.32%, so treat the timing effect as below confidence; the deletion is kept
  for guaranteed reducer and code debloat, not as a claimed performance win.
  Output remains bitwise identical across all 418,943 rows.

Command-buffer-aware profile:

- The accepted executable is captured under
  `data/profiles/final_command_buffers_20260714`, with
  `--xla_enable_command_buffers_during_profiling=true`. Command-buffer-aware
  Nsight profiling still perturbs the whole run severely: native execution
  expands to 50.621 seconds, so neither its gaps nor absolute kernel timings
  represent production. The cited hot reducers have no recorded graph ID.
- A separate six-pair production control is indistinguishable with the default
  command-buffer policy and `--xla_gpu_enable_command_buffer=`: native medians
  are 4.408 and 4.405 seconds. Do not target command-buffer policy itself.
- Structural comparison with the pre-initialization-reuse trace is useful. The
  dominant `input_reduce_fusion_27` count falls from 1,700 to 1,653 and
  `loop_select_fusion_20` from 2,023 to 1,976. Their aggregate device time in
  the perturbed trace is 205.977 milliseconds. The hot reducer's registers per
  thread fall from 164 to 160; the secondary reducer falls from 166 to 160.
  This verifies that initialization reuse removes 47 hot
  probability/reduction pairs and that reducer debloat lowers compiled state.
- The trace contains 1,085 CUDA-graph launches and 18,645 recorded kernels,
  including 568 profiler redzone kernels and 110 profiler delay kernels. Do not
  compare that raw total with the op-by-op trace. The 26 full H2D copies still
  move 2.133 GB in 209.670 milliseconds.

Rejected:

- Blocking two pseudo-logistic iterations per while-body is bitwise exact but
  speculatively evaluates a discarded tail iteration for completed lanes. Six
  valid hot runs average 4.236 seconds versus 4.113 seconds for the accepted
  single-step loop, a 2.99% regression. The prototype is removed completely.

Next targets:

- The remaining 1,653 hot scalar reductions are genuine solver iterations.
  Reduce them only through a numerically reviewed solver change, or eliminate
  map/reduce work inside the hot fused reducer with a benchmarked GPU kernel.
  Do not add a second implementation without a decisive production win and
  memory-traffic or hardware-counter evidence.
- Treat the roughly nine-second separate-process JAX runtime setup and the
  first executable load as the largest end-to-end architectural ceiling.
  Evaluate a resident execution lifecycle before further sub-percent scalar
  expression tuning.

## 2026-07-14 Binary GPU Architecture And Crate Wave

The production cleanup comparison is under
`data/benchmarks/block_cleanup_paired_isolated_*`. The target remains one
binary trait with approximate Firth, full 1KG chromosome 22, 512-row Firth
batches, 16,384-variant chunks, eight host threads, and a V100 on `landau`.

Accepted:

- The unused experimental block-Firth implementation, its separate line search,
  block-only containers, and six block-only configuration fields are removed.
  Seven order-alternated isolated pairs are bitwise identical across all
  418,943 rows and columns. Median native execution is neutral-to-better at
  4.238 versus 4.202 seconds (-0.83%, paired p=0.12); retain the roughly
  1,000-line deletion for architecture and bloat reduction, not a speed claim.
- BGEN index construction reuses a chromosome dictionary code while adjacent
  raw chromosome bytes are equal. Open/index improves from 96.389 to 93.341
  milliseconds (3.16%, p<0.01), with a longer confirmation improving 5.28%.
  Packed8 16,384-variant decode is neutral at 26.462 versus 26.469 milliseconds.
- LOCO sample indexing retains one raw header and byte bounds instead of two
  owned strings per sample. The 100,000-sample core improves from 16.079 to
  2.166 milliseconds (7.42x) and removes 200,000 heap-string allocations.
  At 500,000 samples, direct indexed row reading improves a 2.50 MB row from
  28.746 to 25.119 milliseconds and a 6.00 MB row from 66.086 to 57.849
  milliseconds while eliminating one row-sized buffer.
- The Parquet writer uses Data Page V2 delta fallbacks, 16,384-row internal
  batches, and `BYTE_STREAM_SPLIT` for all six `Float32` columns. The exact
  418,943-row benchmark improves score-only writing from 180.48 to 155.23
  milliseconds and Firth-success writing from 182.41 to 159.45 milliseconds;
  output shrinks from 11,512,914 to 10,471,971 bytes. The physical policy is
  fingerprinted in manifest schema 16; logical output schema remains 3.
- The final order-alternated `main`-versus-wave GPU gate uses seven warm pairs
  in one `landau` allocation. Native execution is neutral-to-better: paired
  geometric mean is -0.34%, paired median is -0.06%, and paired t-test
  p=0.49. The writer paired median improves 7.64%; observed Parquet size falls
  from 10,792,165 to 10,768,924 bytes. The paired datasets are bitwise
  identical in every column across all 418,943 rows.

Rejected:

- A Pallas/Triton scalar-Firth reduction cannot run on the V100: installed JAX
  requires compute capability 8.0 or newer, while `landau` is 7.0. Do not add an
  unmeasured CUDA extension or an Ampere-only production branch.
- Skylos Rust dead-code entries 3502--3529 are cross-module, Serde, PyO3, or
  shutdown/telemetry uses missed by the analyzer. Every symbol has a live
  production reference; none is removed.
- AHash interning, statistics-buffer reuse, lexical parsing, and unsafe
  preflight-SVD prototypes either regress, are statistically neutral, or save
  too little to justify their complexity. All prototypes are removed.

## 2026-07-14 Capacity And BGEN Follow-up Wave

The target remains one binary trait with approximate Firth, full 1KG
chromosome 22, 512-row Firth batches, 16,384-variant chunks, eight host
threads, and a V100 on `landau`. CPU microbenchmarks use paired builds on the
same allocated node. All accepted BGEN changes preserve exact packed bytes,
summary fields, and decoded values on actual chromosome-22 input.

Accepted:

- Candidate capacity selection now rounds the observed correction count to a
  bounded power of two, keeps the two-batch floor, and retains the established
  overflow path. Removing the capacity-tier plan deletes 74 lines while adding
  43. Fifteen paired production runs are statistically neutral at -0.27%
  median, so retain this as an architecture and executable-cache simplification,
  not a speed claim.
- Packed8 raw dosage uses one AVX2 `maddubs` operation with exact `[2, 1]`
  weights instead of five vector operations. Four paired Criterion runs improve
  16,384-variant decoding from 42.041 to 40.836 milliseconds (2.87%) and
  32,768-variant decoding from 81.963 to 80.168 milliseconds (2.19%). The hot
  function shrinks by 16 bytes and introduces no spills.
- Sparse zero and homozygous-alternate counts accumulate in 16-bit AVX2 lanes
  and reduce once per bounded 4,096-vector window. This removes two serial
  `movemask`/`popcnt` chains per vector without overflow or hot-loop spills. A
  decompression-free benchmark over actual 1KG probability bytes improves
  3.82% across four of four pairs. Seven full chromosome-22 GPU pairs improve
  native execution from 4.238956 to 4.164570 seconds (1.755%, paired
  p=0.00034), with all Parquet parts byte-identical. An apparent 33% whole-reader
  result is discarded because function growth shifted downstream `libdeflate`
  addresses; it is not attributed to the SIMD change.
- Zlib decompression writes into reserved spare vector capacity and publishes
  the length only after `libdeflate` reports exact success and complete input
  consumption. This removes the output-sized zero fill without weakening error
  handling. Actual 1KG reads are neutral within 0.34%; synthetic UK Biobank
  blocks improve 14.82% at 1.5 MB and 16.43% at 4.5 MB.
- BGEN variant records retain format-native 32-bit compressed and uncompressed
  lengths beside a 64-bit file offset. Each record shrinks from 24 to 16 bytes.
  Opening chromosome 10 reduces median RSS by 11.36 MiB and chromosome-22
  open/index time improves from 86.876 to 85.456 milliseconds, while dosage
  and packed8 throughput remain unchanged.
- A fresh all-file Skylos scan finds one unreferenced JAX helper that only
  performs Cholesky decomposition before calling the existing factor solver;
  it is removed. Other reported production symbols are stale or have verified
  Rust trait, macro, PyO3 dynamic, telemetry, cache, scheduler, or required JAX
  callback consumers. No hot JAX function is split solely to satisfy structural
  heuristics.

Rejected:

- Packed8 non-temporal stores improve an isolated decode loop by roughly 28%,
  but 14 valid full GPU pairs regress 0.39% in the median, with a 95% interval
  from -0.86% to +1.65%. The 126-line unsafe implementation is removed.
- Dynamic and static dense-Firth sentinel specializations are neutral. The
  dynamic form measures -0.93% with a confidence interval crossing zero; the
  static form wins only five of 15 pairs and regresses 0.20% in the paired
  median. Both prototypes are removed.
- A private typed CUDA Firth reduction is numerically close on a
  512-by-2,504 V100 smoke case, but takes 0.38--0.44 milliseconds versus 0.11
  milliseconds for XLA and adds more than 600 lines. The prototype is removed.

Final deep-profile evidence:

- A fresh `landau` campaign at commit `478f83e2` completed the headline run,
  JAX trace and device-memory capture, cProfile, py-spy, Scalene, Memray, and
  Nsight Systems. The artifact bundle is
  `data/profiles/next_wave_final_deep_20260715`. NVIDIA Compute profiling was
  unavailable because the node denies hardware performance counters, and
  Linux `perf` was unavailable with `perf_event_paranoid=4`; both are recorded
  as skipped rather than successful measurements.
- The one-shot headline records 4.068 seconds in native execution, 0.176
  seconds in native preparation, 8.728 seconds in JAX runtime configuration,
  and 13.241 seconds in the runner. Native execution is 0.37% below the
  pre-wave one-shot profile, but this unpaired comparison is not used as the
  speed claim; the seven-pair GPU gate above isolates the causal 1.755% gain.
- Nsight records a 3.732-second kernel-launch span. Its dominant kernels remain
  `input_reduce_fusion_27` at 122.500 milliseconds across 1,836 launches and
  `loop_select_fusion_20` at 101.065 milliseconds across 2,159 launches.
  Host-to-device transfer remains 1.989 GiB across 110 copies and consumes
  211.849 milliseconds. The final run executes more solver kernels than the
  baseline, so kernel totals are not interpreted as an optimization regression.
- cProfile attributes most process time outside native execution to imports,
  backend initialization, plugin discovery, and shared-library loading. A
  resident multi-run lifecycle is explicitly outside this wave's scope. The
  next in-scope opportunities are therefore the dominant Firth reduction and
  selection fusions and host delivery, each requiring a numerically bounded
  implementation and the same paired GPU gate before acceptance.

## 2026-07-16 Native GPU BGEN Delivery Wave

The target remains one binary trait with approximate Firth, full 1KG
chromosome 22, 512-row Firth batches, 16,384-variant chunks, eight host
threads, and a V100 on `landau`. The comparison used fresh shared JAX caches,
warmed each process once, and alternated main and candidate order across seven
pairs in one allocation.

Accepted:

- Zlib-compressed packed8 BGEN groups remain compressed on the host and are
  decompressed directly into device buffers with the official nvCOMP raw
  DEFLATE API. Rust prepares immutable borrowed slab, metadata, selection, and
  raw-statistic arrays once per transfer group; the Python binding only
  registers the FFI target and manages the lifetime of those arrays.
- The CUDA finalizer validates the decoded row structure, gathers selected
  samples, and emits exact integer raw statistics plus the rounded `Float32`
  genotype mean expected by the existing JAX path. Full and tail-batch FFI
  probes match canonical host decoding for every probability byte, raw sum,
  raw squared sum, zero count, homozygous-alternate count, status, padding
  byte, and genotype mean. An independent 16,384-by-2,504 dosage probe has no
  mismatches.
- The order-alternated hot gate improves the median from 1.178368 to 0.958154
  seconds, an 18.69% reduction. The paired geometric speedup is 1.2202x, or an
  18.04% paired-time reduction. Candidate wins all seven pairs; the two-sided
  all-wins sign test is p=0.015625. Candidate output is stable across all seven
  runs.
- A controlled same-state GPU diagnostic proves producer equivalence. Canonical
  host decoding, nvCOMP FFI output, and a distinct-pointer device copy have
  bitwise-identical compute inputs and identical StableHLO. Decoded dosage,
  every score field and correction code, and full score-plus-Firth beta,
  standard error, chi square, log10 p-value, and correction code are bitwise
  identical. The compressed producer and its device layout therefore introduce
  no numerical change.
- Cross-process current-main and candidate output have identical schema, row
  order, non-floating columns, allele frequency, information score, and
  correction status. Of 418,943 variants, all floating results are bitwise
  equal except two approximate-Firth rows. Their maximum absolute differences
  are 0.00376 for beta, 0.00269 for standard error, 0.16172 for chi square, and
  0.03918 for log10 p-value. Independent candidate processes also produce
  widespread but much smaller floating-point bit differences despite exact
  warm-versus-hot output within a process. Cross-process output is therefore
  a bounded numerical check, not a bitwise causal oracle; the controlled
  same-state result is the delivery-path acceptance proof.
- Exact integer sparse classification corrects one threshold-boundary variant,
  `rs552477617`: its exact minor count is 50 alleles, whereas the historical
  `Float32` calculation rounds just below 50. It remains on the score path, so
  this semantic correction does not alter its output.

Next:

- Profile the integrated commit with Nsight Systems and the standard deep
  campaign. Attribute time among host-to-device delivery, nvCOMP kernels, the
  packed8 finalizer, association/Firth kernels, device-to-host transfer, and
  output before changing the next hot path.
- If finalization remains material, merge its structural and checksum scans,
  parallelize the Adler combination, and reduce per-variant status output to a
  first-failure result. Each change must pass the same exact decode probes and
  paired hot GPU gate.

## 2026-07-16 Packed8 CUDA Finalizer Wave

The target remains one binary trait with approximate Firth, full 1KG
chromosome 22, 512-row Firth batches, 16,384-variant chunks, eight host
threads, and a V100 on `landau`.

Baseline evidence:

- The integrated nvCOMP delivery commit `ca8a5bda` was profiled in SLURM job
  45281. The complete bundle is
  `data/profiles/nvcomp_delivery_ca8a5bda68b_20260716`; headline, JAX,
  cProfile, py-spy, Scalene, Memray, and Nsight Systems passes completed.
  Nsight Compute and Linux `perf` remain unavailable because the node denies
  their hardware counters.
- The one-shot profile records 3.873 seconds of native execution. Across 26
  batches, nvCOMP inflate consumes 136.485 milliseconds and the packed8
  finalizer consumes 40.975 milliseconds, or 1.576 milliseconds per batch.
  The finalizer alone is 8.7% of recorded GPU kernel time.
- A fresh low-threshold `skylos --all` scan completed in job 45285 and reported
  318 findings. Manual review found no dead production symbol: reported Rust
  facade imports, PyO3 and Serde callbacks, trait imports, JAX scan arguments,
  and the dynamically loaded nvCOMP package all have live consumers. Function
  length and clone-group suggestions would split cohesive JIT graphs or merge
  semantically distinct state. No code is changed merely to satisfy those
  heuristics.

Accepted:

- Identity selection now validates the packed8 row, accumulates Adler-32,
  emits probability pairs, and computes exact integer statistics in one
  coalesced source pass. Direct weighted-byte accumulation removes two runtime
  64-bit division helpers per thread. Other selection modes retain the indexed
  gather pass and the FFI ABI is unchanged.
- Eight 256-element shared-memory reduction arrays and an eleven-barrier tree
  are replaced by warp-shuffle reductions through eight warp partials. CUDA
  reports 40 registers, 360 bytes of static shared memory, no stack frame or
  spills, two block barriers, and no integer divide or remainder in the
  finalizer. The host rejects source counts above 126,789,562, where the exact
  unreduced Adler weighted sum no longer fits its 64-bit accumulator.
- Production-path parity in job 45291 is bit-exact for FFI inputs, integer
  summaries, sparse classification, decoded dosage, every score field, and
  every full approximate-Firth field. The rebuilt direct probe in job 45293
  additionally passes full and tail batches, contiguous and nonmonotonic
  indexed selections, out-of-range index handling, Adler corruption, and the
  descriptor neutral-output gate.
- The 14-run order-alternated hot gate in job 45294 improves the median from
  0.978769 to 0.947684 seconds, a 3.18% reduction. The paired geometric time
  reduction is 3.49%; the candidate wins all seven pairs, giving a two-sided
  all-wins sign-test p-value of 0.015625.
- A CUDA-profiler-API-bounded Nsight capture in job 45296 confirms causality.
  The finalizer falls from 1.575946 to 0.449548 milliseconds per batch, a
  71.47% reduction. Its 26-batch total falls from 40.975 to 11.688
  milliseconds, saving 29.286 milliseconds and explaining about 94% of the
  31.086-millisecond median end-to-end gain. nvCOMP inflate is now the largest
  delivery kernel at 139.486 milliseconds in that trace.

Deferred:

- Returning the sparse predicate directly from CUDA would remove two small
  count outputs and one post-FFI fusion, but the current profile bounds that
  opportunity below one millisecond per run. Do not enlarge this accepted
  causal patch for it.
- The next delivery experiment should measure aligned decompressed-row or slab
  geometry against nvCOMP inflate. The next compute experiment should target
  the dominant Firth `input_reduce_fusion_27` and `loop_select_fusion_20`
  kernels with exact numerical and paired hot gates.

## 2026-07-16 Trait-Shared Firth Inputs Wave

The target remains one binary trait with approximate Firth on full 1KG
chromosome 22. The latest Nsight trace shows 1,720 dominant sigmoid/reduction
pairs consuming separately materialized 512-by-2,504 phenotype and null-offset
operands. Those duplicated lane inputs account for about 32.8 GiB of logical
reads across the hot run.

Accepted:

- Candidate preparation now retains phenotype and null-Firth-offset arrays in
  trait-major form. Lane ordering moves only lane indices and lane-specific
  payload; dense batches select the trait row inside the lane map, and compact
  sparse batches gather only their existing 64 carrier slots.
- Scalar Firth arithmetic and reduction order are unchanged. Both production
  chr22 Parquet partitions are byte-identical to the baseline, including all
  418,943 rows and every approximate-Firth result.
- An exact-shape affected-kernel probe is bitwise identical and 11.5--12.5%
  faster. In ten order-alternated production processes with five hot trials per
  process, the paired geometric hot-time reduction is 3.53%; the bootstrap 95%
  interval is 1.06--6.74%, and the candidate wins four of five process pairs.
- The refactor removes three more lines than it adds, does not change the public
  Python or Rust API, and leaves the scalar solver implementation untouched.

Next:

- Remove the per-chunk four-byte Firth candidate-count device-to-host
  synchronization with a configurable hard capacity and explicit overflow.
  The current trace contains 26 such copies and about 35.8 milliseconds of
  following launch gaps; never silently truncate a trait exceeding capacity.

## 2026-07-16 Static Firth Capacity Wave

The target remains one binary trait with approximate Firth on full 1KG
chromosome 22. A focused current-main trace contains 26 four-byte candidate
count transfers followed by 43.917 milliseconds of aggregate launch gaps. The
count was materialized midway through every chunk to select a correction
executable.

Accepted:

- Candidate count remains on the device until ordinary whole-batch result
  materialization. One donating JIT now runs the established masked correction
  at an aggregate static capacity of
  `min(firth_candidate_capacity, static_compute_variant_count) * trait_count`.
  The packaged per-trait scaling value is 1,024, which covers this workload's
  observed maximum of 959 candidates.
- The materializer compares the device count with the static capacity before
  parsing or writing results. A forced one-lane overflow reports the observed
  count, capacity, and configuration key, and produces no Parquet output.
  Candidate rows are never silently truncated.
- Full production chr22 output remains bitwise identical across all 418,943
  rows, including every score field, corrected field, correction label, and
  status.
- Across ten paired hot trials in forward and reverse order, the candidate wins
  eight pairs. The conservative median paired reduction is 5.18%; the paired
  geometric reduction is 11.81%, with a deterministic bootstrap 95% interval
  of 3.45--21.11%. The upper tail reflects main's host-barrier sensitivity
  during node-latency spikes rather than additional GPU arithmetic savings.
- A final focused Nsight trace reduces the gaps following the relevant
  four-byte transfers from 43.917 to 16.695 milliseconds, a 62.0% reduction.
  The remaining count metadata transfers occur with final result
  materialization rather than between score and correction launches.
- The change deletes the dynamic capacity selector and is net 21 lines smaller
  while retaining one correction implementation and one overflow contract.

Rejected:

- A device `lax.cond` around correction is end-to-end neutral and causes
  last-bit differences by changing compilation and reduction scheduling. The
  unconditional masked correction preserves the command-buffer-friendly path
  and full-run bitwise output while retaining identity semantics for
  zero-candidate batches.

Next:

- Reduce redundant pseudo-Firth scalar score, information, adjustment, and
  deviance evaluation. Carry already-computed terminal scalars, calculate
  penalized deviance only where convergence or the final likelihood-ratio test
  needs it, and retain the exact full-output and paired hot gates.

## 2026-07-17 Batched Lazy Firth Fallback Wave

The target remains one binary trait with approximate Firth on full 1KG
chromosome 22. The scalar solver appeared to select Newton-Raphson lazily with
a lane-level `lax.cond`, but that condition was inside `jax.vmap`. JAX's batching
rule lowered the condition to a select and executed both pseudo-Firth and
Newton-Raphson for every active batch before discarding one result per lane.

Accepted:

- Scalar initialization and pseudo-Firth remain lane-vectorized. The fallback
  mask is now reduced to one scalar outside every lane `vmap`; a batch enters
  vectorized Newton-Raphson only when at least one active pseudo-Firth lane
  fails. A compact terminal result selects the winning solver quantities, then
  computes the p-value once. Padded and null-failed lanes retain the existing
  empty-result semantics.
- The refactor removes the obsolete lane-dispatch container and wrappers. The
  shared initial state retains only values needed by both solvers, and the
  implementation is net 48 source lines larger while replacing an ineffective
  control-flow architecture.
- Normal production output is bitwise identical to current main across all
  418,943 rows and every field. A second full-chromosome run with
  `firth_pseudo_maximum_iterations = 1` forces Newton-Raphson and is also
  bitwise identical, proving that the retained fallback path selects and
  finalizes correctly.
- Across ten paired hot processes in forward and reverse order, with five hot
  trials per process, the candidate wins every pair. Median paired time falls
  26.66%; the paired geometric reduction is 26.99%, with a deterministic
  bootstrap 95% interval of 26.18--27.80%.
- Focused Nsight confirms that production pseudo-Firth succeeds without
  entering a Newton batch. The 1,752-launch, 160-register Newton reduction is
  absent, and the corresponding full-sample probability loop falls from 2,075
  launches to 323. The four-kernel Firth pool falls from 237.904 to 38.663
  milliseconds, an 83.75% reduction. Across the complete trace, kernel calls
  fall from 19,231 to 8,176 and aggregate kernel time from 448.125 to 227.976
  milliseconds. Nsight reports the known non-fatal `NumTpcs` importer error,
  but the CUDA activity tables are complete.
- The inner pseudo-logistic proposal now evaluates only its transient
  predictor, sigmoid, and per-sample score and information products in
  `float32`. Products widen before `float64` reductions; scalar state, outer
  objective components, convergence, corrected statistics, and lazy Newton
  fallback remain `float64`. Manifest schema 17 fingerprints this fixed policy
  so resume cannot mix it with prior all-`float64` inner chunks.
- The focused 512-by-2,504 pseudo-solver median falls from 1.5472 to 1.1118
  milliseconds, a 28.14% reduction. Across ten position-balanced full hot
  process pairs, the candidate wins nine. Median paired time falls 3.82%, with
  a deterministic bootstrap 95% interval of 2.49--4.72%; the paired geometric
  reduction is 2.62%.
- All 418,943 correction methods and statuses remain identical. On 17,938
  corrected rows, maximum public-result changes are 4.77e-7 beta, 2.38e-7
  standard error, 1.91e-6 chi square, and 4.77e-7 log10 p-value. Synthetic
  separated, sparse, one-to-128-carrier, and eta-to-44 lanes preserve every
  validity and fallback decision. Actual chromosome-22 terminal eta stays
  between -6.356 and 3.546.

Rejected:

- Reusing terminal pseudo-logistic information and score reductions removed
  10.503 milliseconds from the affected profiled kernels, but did not clear the
  full-run gate. Across twenty paired processes it wins twelve, with a 0.07%
  median reduction and a -0.14% geometric change; the bootstrap 95% interval is
  -0.78--0.48%. The patch is not integrated.
- Raising the Firth batch size from 512 to 1,024 preserves bitwise output but
  loses all five paired processes. Median hot time regresses 10.29%, and the
  paired geometric regression is 7.92%. Keep the 512-lane batch.
- Representing an ordinary dense Firth batch with a static all-sample mask
  removes the carrier and sparse-mask operands from the specialized StableHLO,
  along with 16 selects, one reduction, and two logarithms. Production output
  remains bitwise identical across all 418,943 rows and every field. Focused
  Nsight confirms that the two dominant Firth reductions fall from 22.901 to
  16.522 milliseconds, saving 6.379 milliseconds, while their register counts
  fall from 40 and 160 to 32 and 84.
- The kernel saving does not clear the full-application hot gate. In the
  predeclared 30 paired processes, the candidate wins 20, with a 1.129839%
  median reduction (95% bootstrap interval -0.343213--2.253859%) and a
  1.473956% geometric reduction (-0.828735--3.913967%). Including all five
  unplanned intermediate pairs prevents cherry-picking but remains ambiguous:
  22 of 35 wins, 0.675660% median reduction
  (-0.688781--2.233241%), and 1.205939% geometric reduction
  (-0.789758--3.348061%). The final 30-process sequence ran in one allocation
  reserving the sole GPU, eight CPUs, and 64 GiB; a 35-day resident non-GPU
  allocation made Slurm node exclusivity impossible. The source patch is not
  integrated.
- Arrow dictionary inputs do not preserve a cheaper metadata path through the
  current Parquet writer. Parquet resolves the keys back to strings, then
  hashes and interns them into its own dictionary. Six full-output processes
  regress score-only writing 7.89% and Firth-success writing 14.68%, with no
  file-size change. Keep the existing `Utf8View` columns.
- Zstd level 1 remains both the fastest and smallest tested direct-Parquet
  codec. Across six position-balanced full-output processes per codec, Snappy
  is 8.57% slower and 19.94% larger; raw LZ4 is 8.40% slower and 21.65% larger.
  The temporary codec dependencies and source changes are not retained.
- Padding each 7,522-byte decompressed row to a 128-byte pitch does not improve
  nvCOMP delivery. The 7,552-byte candidate wins only three of eight direct-FFI
  pairs; its median and geometric time changes are regressions of 0.176% and
  0.094%, with both confidence intervals crossing zero. Smaller 32- and
  64-byte pitches have the same 7,552-byte geometry, while 256 bytes would add
  still more transfer. Keep the logical row stride.
- Custom host pinning cannot remove the measured transfer path. Nsight labels
  all 68 host-to-device copies as pinned-to-device transfers already: 118.58
  MiB takes 14.511 milliseconds. PJRT stages the current Rust owners through
  its own pinned buffers, and bypassing that lifecycle would complicate buffer
  reuse for an absolute CUDA-activity ceiling of 3.55%. Keep the pooled host
  buffers and PJRT-owned staging.
- Score-test beta is not a valid approximate-Firth warm start. It is in output
  allele orientation and comes from the ordinary-logistic null, whereas the
  correction solves the Firth-null objective in minor-allele orientation.
  REGENIE's normal approximate-Firth path starts at zero, and this repository
  already observed convergence failures from score-beta initialization. A
  correct nonzero start would also require a second component evaluation and
  could change convergence and fallback labels, so no source experiment is
  retained.
- Borrowing one contiguous mmap span per chunk and scattering raw members on
  the GPU loses the cost model before implementation. The existing host pack
  takes about 0.447 milliseconds per full chunk, while shape-stable borrowed
  spans would increase chromosome transfer from 119.261 to 139.297 MB and add
  a new scatter kernel plus 4.587 MB of device staging per resident batch.
  Seventy-five percent of source members are unaligned and cannot be passed to
  nvCOMP directly. Keep the pooled host pack.
- Parallel raw-DEFLATE packing does not improve the source-neutral production
  path. Caching all member descriptors cuts a 16,384-variant pack about 66.7%
  and improves open plus layout plus all packs about 2.05%, but retains 12
  bytes per variant, binds the layout to one source, and moves validation ahead
  of the producer/GPU pipeline. Descriptor-free variants that parallelize copy
  alone or validation plus copy are flat to 13% slower on the full production
  chunk and regress smaller chunks. Keep the one-pass pooled packer.
- Increasing the transferred-batch queue from one to two preserves byte-exact
  Parquet output but does not convert the profiled 6.8-millisecond stall ceiling
  into a stable full-run win. The five position-balanced process pairs change
  by +0.821%, +2.500%, -2.922%, -7.987%, and +4.813%; the paired geometric
  result is a 0.456% regression. Keep one queued successor instead of retaining
  another 78.641 MiB device genotype batch.
- Aligned 16-bit probability-pair I/O is slower than the finalizer's two byte
  operations. The candidate keeps 40 registers, has no spills, and exactly
  preserves full, tail, selection, invalid-index, descriptor, Adler, statistic,
  status, and mean outputs. Nevertheless it loses all five direct-FFI process
  pairs: median and geometric time regress 0.727% and 0.759%, with both
  bootstrap intervals wholly negative. Uniform mode branches inside the sample
  loop outweigh the reduced load/store instruction count, so keep byte I/O.
- Further Firth lane bucketing has too little divergence to recover its own
  routing cost. The production trace contains 17,938 active lanes, with no
  null-fit failures or Newton fallbacks. The current dense and compact streams
  consume 299 batch-maximum pseudo-Firth outer iterations; even perfect oracle
  ordering saves only four, a 1.338% solver-loop ceiling. The best cheap
  carrier- or minor-allele-count split saves three, while score statistics have
  only about 0.02 correlation with iteration count. The opportunity is confined
  to two tail batches, so no sorting or scatter path is retained.
- A fused native pseudo-Firth solver does not have a credible implementation
  gate on the V100. The dense path's entire ceiling is 38.663 milliseconds, but
  the previously removed bare CUDA reducer took 0.38--0.44 milliseconds where
  XLA took 0.11. Applying the measured lane divergence puts that primitive at
  a 34--48-millisecond lower bound before adding solver control flow, exponent,
  and logarithm work. A faithful 600--900-line second solver would need to beat
  the old reducer by at least 34% while doing more work merely to reach a useful
  application gate, so no CUDA prototype is added.
- Hoisting the three invariant `float64`-to-`float32` conversions out of the
  inner loop materializes and carries three 512-by-2,504 arrays. The focused
  solver gain falls from 28.14% to 10.3%, so conversions remain fusion-local
  inside the loop body.

Next:

- Re-profile the accepted mixed-precision path before further Firth expression
  tuning. Rank the remaining nvCOMP delivery, Rust index/open, output tail, and
  association kernels from the refreshed trace.

## 2026-07-16 BGEN Allele Interning Wave

The full chromosome-22 index performs 837,886 allele interning calls. Of
these, 803,358 (95.88%) are exact one-byte ASCII alleles, predominantly
`A`, `C`, `G`, and `T`; the previous path repeated lossy UTF-8 conversion and
hash-table lookup for every call.

Accepted:

- `StringDictionaryBuilder` keeps a fixed 128-entry optional-code cache. A
  first encounter still follows the existing conversion, lookup, insertion,
  and dictionary-order path before filling the cache. Empty, multibyte, and
  non-ASCII inputs retain the old path and error behavior. The cache allocates
  nothing and adds about one KiB to the transient index builder.
- Sixteen order-alternated Criterion processes on an exclusive `godel` core
  reduce median full chr22 open/index time from 87.018 to 66.876 milliseconds.
  The paired geometric reduction is 23.52%, all eight pairs win, and the
  bootstrap 95% interval is 23.11--23.99%.
- The production extension emits byte-identical chr22 Parquet partitions. In
  ten order-alternated `landau` processes with five hot trials each, median hot
  time improves 1.66% and the paired geometric reduction is 1.23%; the
  bootstrap 95% interval is 0.22--1.99%, with four of five process-pair wins.

## 2026-07-17 BGEN Index Traversal Wave

Rejected:

- All 418,943 resolved chromosome-22 identifiers are ASCII. Bypassing lossy
  UTF-8 conversion for that strict subset wins all eight exclusive-core pairs,
  reducing median open/index time from 66.193 to 65.521 milliseconds. The
  paired geometric reduction is 1.102%, with a 0.279--2.124% bootstrap
  interval.
- Prefetching the next record header after its checked offset wins a separate
  eight of eight pairs, reducing median open/index time from 65.435 to 63.011
  milliseconds. Its paired geometric reduction is 3.782%, with a
  2.892--4.544% interval. The two local changes combine to about a 4.8% index
  stage improvement without allocating or retaining more metadata.
- A full structural audit remains exact across every variant: metadata digest
  `14b55b774ec075c971a081f0adfdbede0790490c0ffd5e92066ca7d2a1f14dd0`,
  packed-record digest
  `2a752984ea66d9f83b624c9876aa50b414ef855a0945340ae6cb4159e9f3d038`,
  one chromosome boundary, and 26 production chunks. All 60 application-gate
  Parquet outputs are byte-identical.
- The predeclared application endpoint does not clear zero after 30
  position-balanced pairs. The candidate wins 19 pairs and reduces the median
  paired time 1.391%, but severe bidirectional node outliers leave the
  geometric reduction at 0.0606% with a -4.203--3.416% bootstrap interval
  (`p=0.9025`). Per the fixed gate, the source patch is not retained despite
  the isolated index-stage win.

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
