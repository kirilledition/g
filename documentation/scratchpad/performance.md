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
  Parquet work 12.8%; paired full-run wall time is unchanged. This result was
  superseded by the 2026-07-18 direct-Parquet gate below.
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

## 2026-07-17 Compact Firth Outer-Unroll Wave

Rejected:

- Grouping four pseudo-Firth outer iterations only for compact batches keeps
  the dense factor-one StableHLO byte-identical and preserves exact dense and
  compact result digests. Across five position-balanced focused blocks, compact
  solver reductions are 1.8321%, 2.8190%, 0.5410%, 0.6076%, and -0.1942%.
  The 0.6076% median does not clear zero under bootstrap resampling.
- The candidate enlarges compact compiled HLO from 98,453 to 308,917 bytes
  (+213.8%), cold compact compilation from 1.060 to 1.842 seconds (+73.9%),
  and the measured cache from 260,311 to 332,411 bytes (+27.7%). The small,
  uncertain hot benefit does not justify that executable bloat, so no full
  application gate was run and the source patch was removed. Artifacts are
  under `data/profiles/firth_compact_outer_unroll_focused_45491_*` and
  `data/profiles/firth_compact_outer_unroll_static_20260717`.

## 2026-07-17 Tiny-DEFLATE CUDA Spike

Rejected:

- The full chromosome-22 corpus contains 418,943 zlib members whose raw
  DEFLATE payloads are single final dynamic-Huffman blocks with 7,522-byte
  outputs. A bounded custom CUDA decoder prototype preserved exact probability
  pairs, dosage sums, status masking, long-code behavior, and corruption,
  truncation, Adler-32, and short-output contracts.
- Warp-local synchronization and literal-run copying removed the prototype's
  block barriers, but did not remove its serial lane-zero Huffman and token
  decoder. Across 20 position-balanced direct-FFI processes, nvCOMP has a
  6.223-millisecond median and the custom kernel 7.496 milliseconds. The custom
  path is 20.46% slower, and all five balanced quartet ratios regress by
  19.8--21.0%.
- The experiment fails both the 10% direct-performance floor and the stronger
  20% threshold required to justify a second format-restricted decoder. The
  roughly 500-line CUDA implementation, generated PTX, and integration changes
  were deleted without a full application gate. The retained characterization
  is `data/profiles/tiny_deflate_characterization_45493.json`.

## 2026-07-17 Current-Main Deep Profile

Commit `ca25b20b` was profiled on the full chromosome-22, one-trait binary
approximate-Firth GPU path. The uninstrumented headline run spent 3.493 seconds
in native execution. Rust output work accounted for 329.5 milliseconds,
including 243.4 milliseconds writing Parquet batches and 120.9 milliseconds in
terminal output completion. The complete retained bundle is
`data/profiles/main_ca25b20b_chr22_binary_gpu_full_20260717`.

The cProfile, py-spy, JAX, Memray, and Scalene captures completed. Linux
`perf` was blocked by the node's `perf_event_paranoid=4` policy, and Nsight
Compute was skipped. Nsight Systems 2026.1.3 deadlocked while finalizing both
the full-process capture and a CUDA capture-range retry; the latter left an
incomplete `qdstrm` that cannot be imported. Do not treat the absence of a
current Nsight timeline as evidence that GPU overlap or kernels are optimal.

### pprof-rs CPU sampling

`pprof-rs` was added temporarily behind a feature gate and then removed. The
first 1 kHz capture using its default libgcc/backtrace unwinder segfaulted on
`landau` despite the recommended blocklist. The successful configuration used
the framehop unwinder at 199 Hz and primed framehop's lazy loaded-object map in
the discarded warm lifecycle.

Five fully compiled, same-process hot captures lasted 0.653, 0.632, 0.629,
0.626, and 0.640 seconds and produced 326 product CPU samples after excluding
the fixed framehop warm-up samples. The mutually exclusive sample attribution
was:

- output and Parquet: 141 samples (43.25%);
- BGEN index parsing and raw-DEFLATE packing: 67 (20.55%);
- JAX/CUDA host dispatch: 55 (16.87%);
- result materialization: 34 (10.43%);
- other Rust orchestration and input: 19 (5.83%); and
- other Python/native work: 10 (3.07%).

Output was the largest category in every capture. Its samples split into 63 in
string dictionary/min-max work, 43 in Zstd, 18 in numeric pages, 13 in record
batch construction, and four elsewhere. BGEN split into 51 index-parser, 14
raw-DEFLATE pack/validation, and two other samples. The most useful inclusive
frames were the output chunk writer (138), Parquet byte-array encoder (78),
BGEN index parser (51), JAX `PjitFunction::Call` (43), Zstd block compression
(41), PJRT execute (39), GPU thunk host setup (36), and Parquet float encoder
(27). Full stacks and per-capture artifacts are under
`data/profiles/pprof_main_ca25b20b_20260717`; its `analysis.md` is the canonical
breakdown.

These are active process-CPU samples, not wall-time percentages. `ITIMER_PROF`
omits blocked waits and syscalls, output workers overlap the reader and GPU,
and CUDA kernels are invisible. The process-global `SIGPROF` handler, observed
unwinder crash, framehop priming requirement, and large optional dependency
graph make pprof-rs unsuitable as a shipped runtime dependency. Keep it as an
isolated diagnostic only.

### Profile-directed rejected changes

- Replacing the complete-part output jobs with long-lived streaming file
  actors regresses the writer benchmark by 129.3% when all chunks are ready and
  the paced terminal finish by 45.2%. File-level parallelism is more valuable
  than avoiding the existing per-part setup, so the rewrite is discarded.
- Aligning each packed raw-DEFLATE member to 32 bytes is neutral in 40 direct
  FFI pairs: the geometric change is +0.024%, with a -0.95% to +6.21%
  interval. It increases the transfer slab by 5.46%, so the compact pooled pack
  remains.
- The broad product Skylos scan is contaminated by ignored tool environments,
  build outputs, reference data, and an incomplete Cargo workspace graph. All
  32 high-confidence production dead-code findings and all 122 Rust unused
  imports are false positives after manual call-graph review. Three fields in
  the JAX Firth chromosome state are genuinely unused; they are gated
  separately because changing a pytree can alter JIT boundaries even when the
  downstream executable prunes the leaves.
- Replacing the checked fixed-size slice conversion in BGEN scalar reads with
  checked unaligned loads wins seven of eight exclusive-core blocks and shrinks
  the index parser machine code 11.1%. One sustained candidate excursion leaves
  the all-block geometric reduction at 0.286%, however, with a -4.43% to +4.79%
  paired interval. The predeclared gate does not clear zero, so the localized
  unsafe implementation is not retained.
- Of 43 pprof Zstd samples, 41 are inside the bundled Zstd 1.5.7 compressor and
  runtime BMI2 dispatch is active. Contexts are already reused per Parquet
  column, level 1 is the measured optimum, and adding Zstd worker threads would
  oversubscribe the parallel output pool. Parquet 59.1 still allocates and
  copies a temporary compression buffer per page and eagerly constructs an
  unused decoder; fixing that could remove about 10.75 MB of copies and 300
  allocations per chromosome-22 output. Carrying a complete Parquet fork for
  this likely sub-percent ceiling would be application bloat, so defer it until
  an upstream direct-buffer/lazy-decoder fix is released.
- Disabling Parquet statistics for all six string columns demonstrates a
  positive but spike-sensitive full-application ceiling. It is not a valid
  production policy because chromosome statistics support dataset pruning and
  correction method/status statistics support audit queries. The query-safe
  subset disables only `ID`, `ALLELE0`, and `ALLELE1` statistics. It wins all
  three direct-writer blocks, with a 2.711% geometric reduction and a
  2.427--3.187% block-bootstrap interval, while reducing output by only 754
  bytes. Exact schema, values, encodings, custom metadata, offset indexes, and
  the retained chromosome/correction statistics remain equal.
- The subset does not clear the whole-application gate. After the fixed
  extension to 30 adjacent process pairs, the candidate wins 18 pairs and has
  a 0.244% median reduction, but its 2.053% geometric direction has a
  -2.312--7.213% bootstrap interval. Six pairs move by at least 10%, and the
  leave-one-pair-out range reaches -0.011%. The production diff is therefore
  rejected despite the robust writer-only result. Evidence is in
  `data/profiles/parquet_string_stats_subset_gpu_abba_45556_analysis.json`.

### Accepted dead-state cleanup

- `Regenie2MultiBinaryFirthChromosomeState` no longer exports the unused
  covariate matrix, null-logistic coefficients, or LOCO offset. Its pytree has
  11 rather than 14 leaves, the state-builder HLO is 153 bytes smaller, and the
  downstream correction executable HLO is byte-identical.
- Across 20 adjacent process pairs and 240 successful trials, the hot geometric
  change is +0.278%, with a -5.26% to +4.61% bootstrap interval. This is a
  structural cleanup, not a claimed performance win.
- Narrowing the state-builder output changes its deterministic floating-point
  code generation. The 99th-percentile absolute changes are at most
  `3.576e-7` for `BETA`, `1.192e-7` for `SE`, `8.345e-6` for `CHISQ`, and
  `2.444e-6` for `LOG10P`; the respective maxima are `5.206e-4`, `1.836e-4`,
  `7.951e-4`, and `2.218e-4`. No correction choice, `p < 0.05`
  classification, or `p < 5e-8` classification changes. This is within the
  accepted bounded numerical tolerance. Evidence is in
  `data/profiles/firth_state_dead_leaves_hot_abba_45528_analysis.json`.

### Accepted full-batch materialization fast path

- The pprof captures place 34 of 326 product CPU samples in result
  materialization. Twenty-five of the 26 chromosome-22 chunks already have the
  requested trait set and full logical variant width. For those chunks,
  `materialize_batch` now passes the existing, correctly typed association and
  packed8-statistics pytrees directly to `jax.device_get` instead of creating
  eight full-width JAX slices and replacement containers. The partial tail and
  active-trait selection retain the existing slicing and dtype-normalization
  path.
- The focused full-batch path falls from 194.4 microseconds to 0.900
  microseconds, a 216-fold reduction. The sole partial tail remains neutral.
  A 24-case semantic matrix covers full and partial widths, all/subset/empty
  traits, correction codes present and absent, packed8 statistics present and
  absent, and Firth count/capacity errors; baseline and candidate artifacts are
  byte-identical.
- A first full-application gate used separate JAX persistent caches. Although
  it showed a 3.120% reduction with 20 of 20 pair wins, independently compiled
  artifacts under the same nine cache keys differed and caused tightly bounded
  numerical drift. That gate is diagnostic only, not acceptance evidence.
- The causal gate prepopulates one cache from baseline, proves nine of nine
  candidate cache hits with zero misses, and preserves the exact 1,097,162-byte
  cache tree before and after all timings. Across 20 adjacent pairs and ten
  ABBA blocks, median hot time falls from 0.656299 to 0.634945 seconds. The
  geometric reduction is 3.2438%; all 20 pairs win, with pair-bootstrap and
  block-bootstrap 95% intervals of 2.7689--3.7379% and 2.7490--3.8012%.
  Pair and block leave-one-out ranges remain strictly positive.
- Both Parquet parts are byte-identical across all 200 hot outputs; normalized
  manifests are equal. Evidence is in
  `data/profiles/materialization_fastpath_shared_cache_gpu_abba_45562_analysis.json`
  and the adjacent `cache_final_proof.json`.

## 2026-07-18 Profile-Led Current-Main Wave

Commit `94ec9ea9` was re-profiled on Landau with the one-trait, full
chromosome-22 binary approximate-Firth GPU workload, 16,384-variant
application batches, Firth batch size 512, candidate capacity 1,024, and eight
output writers. The current-main pprof-rs campaign attributes 49.44% of 3,317
unblocked active-CPU samples to JAX/CUDA host work, 28.70% to output, and
12.33% to BGEN indexing and packing. `SynchronizeStream` accounts for 1,118
inclusive samples. This is active CPU attribution rather than wall time, but it
makes the device timeline the primary compute lead. Evidence is in
`data/profiles/pprof_main_94ec9ea9_20260717/analysis.md`.

The successful serial Nsight Systems capture contains 7,104 kernels totaling
216.309 milliseconds on one CUDA stream. nvCOMP inflate accounts for 142.258
milliseconds across 26 launches and `finalize_packed8` for 11.819 milliseconds.
Host-to-device copies total 118.579 MiB and 13.777 milliseconds; device-to-host
copies total 14.785 MiB and 5.209 milliseconds. The trace confirms serialized
delivery, but not a useful immediate overlap window: steady-state compressed
inputs usually become ready only 0.011--0.063 milliseconds before the preceding
compute ends, or after it has ended. An auxiliary CUDA stream would therefore
add ownership and synchronization machinery without enough work to hide.
Meaningful overlap first requires earlier compressed-input lookahead; the
existing two-batch queue experiment already regressed. Artifacts are under
`data/profiles/current_main_94ec9ea9_targeted_nsys_serial_20260718`.

### Chunk geometry

The serial current-main sweep retains 16,384 variants and capacity 1,024. Five
hot-run medians were 0.671510 seconds for 16,384/1,024, 0.800758 seconds for
32,768/2,048 (+19.25%), and 0.816802 seconds for 65,536/4,096 (+21.64%). The
attempted 8,192/512 configuration is not a timing result: warm execution
correctly rejected it because 531 aggregate Firth candidates exceeded capacity
512. Capacity 640 still pads the fixed correction shape to 1,024, so it does
not reduce correction work. Evidence is in
`data/profiles/chunk_geometry_current_serial_20260717` and
`data/profiles/chunk_geometry_small_current_20260718`.

### Retained isolated candidates

- The Firth dispatcher unrolls the outer `lax.scan` only when its fixed batch
  count is exactly two. Dense and compact StableHLO shrink by 1.11% and 1.36%,
  respectively. Synchronized GPU calls improve by 0.835% for 400 active
  candidates and 0.422% for 900. In the isolated 20-pair application ABBA gate,
  median hot time falls from 0.647222 to 0.643793 seconds and the paired
  geometric reduction is 1.2814%. The pair-bootstrap interval narrowly crosses
  zero (-0.0337% to 3.1354%), while the ten-block interval is 0.0011% to
  3.0500%. All 418,943 rows and every output value are bit-for-bit equal.
  Evidence is `data/profiles/profile_wave_firth_only_abba_45609_analysis.json`.
- Fresh raw-DEFLATE storage remains length zero while real members are copied,
  initializes only alignment gaps and the unused tail, and publishes its fixed
  length after every byte is initialized. Pooled fixed-shape storage retains
  its already initialized bytes. The isolated 16,384-variant fresh-storage
  midpoint falls from 781.74 to 653.82 microseconds, a significant 16.36%
  reduction; the pooled path is neutral. An adversarial review found no
  uninitialized read or error-path `Vec` invariant violation. The retained
  Criterion case is
  `bgen_gpu_host_delivery_full_samples/raw_deflate_pack_fresh_storage`.
- All 418,943 members in the chromosome-22 fixture use zlib header `0x789c`.
  Packing now fully validates the first header and every changed header while
  caching repeated valid headers in batch-local register state. The focused
  pooled 16,384-member pack improves by 2.691%, with a 2.303--3.195% interval.
  The first-header and every changed-header path retains all RFC 1950 checks
  and per-variant error context. Full and tail GPU probability, statistic,
  status, and genotype mean outputs are exact. Evidence is under
  `data/profiles/nvcomp_framing_candidate_20260718`.

The raw fresh-storage whole-application gate is deliberately inconclusive, not
positive evidence. A transient slowdown affected neighboring implementations;
the candidate wins 9 of 20 pairs, its pair-bootstrap interval spans -11.8604%
to +2.1902%, and its block leave-one-out range spans -4.5382% to +0.9002%.
Outputs remain exact. The focused allocation benchmark supplies the causal
evidence, while the final combined gate retains veto power. Evidence is
`data/profiles/profile_wave_raw_only_abba_45612_analysis.json`.

### Rejected candidates

- Correction-column-only dictionary disabling is order-sensitive and adds
  23,486 bytes (+0.224%) to the chromosome output. Disabling every string
  dictionary is neutral after its first noisy block and adds 21,451 bytes
  (+0.205%). Both policies were removed.
- Owner-backed ID and position Arrow buffers would avoid 8,378,860 copied bytes
  per chromosome-22 output, but the direct confirmation loses both
  position-balanced pairs for an approximately 0.46% geometric regression.
  The safe and unchecked candidates were removed despite a favorable paced
  direction.
- Retaining Adler-32 trailers in wire order improves focused CPU packing by
  0.839%, but requires one byte permutation in the serialized GPU finalizer.
  The fresh-build whole-application gate is neutral with a -0.0586% geometric
  direction, a -1.0416% to +0.9287% pair-bootstrap interval, and a -0.3962% to
  +0.2952% block-bootstrap interval. Moving overlapped host work onto the
  critical CUDA stream is not justified; the CUDA, PTX, and checksum-contract
  changes were removed. Job 45633 nevertheless confirmed exact selection,
  injected-checksum-corruption, and descriptor-gate behavior for the discarded
  artifact.
- Larger chunks, an immediate auxiliary CUDA stream, custom raw-DEFLATE CUDA
  decoding, output dictionary changes, and owner-backed metadata therefore do
  not remain in the production diff.

Output experiments are retained under
`data/profiles/output_dictionary_wave_20260718`. The final combined controlled
gate is recorded below after its fresh native build and exact-output comparison.

### Final lean gate

The accepted candidate contains only the two-trip Firth unroll, fresh-slab
initialization, and repeated-zlib-header validation cache. It restores the
existing CUDA source and embedded PTX byte-for-byte. In Slurm job 45635, median
hot time moves from 0.646895 to 0.646747 seconds and the paired geometric
direction is +0.1120%; 9 of 20 pairs favor the candidate. Symmetric early
latency spikes make this a non-inferiority result rather than an application
speedup claim: the pair-bootstrap interval is -2.6839% to +2.6749%, the
block-bootstrap interval is -2.6786% to +2.6503%, and pair/block leave-one-out
ranges are -0.7909% to +1.1224% and -0.8065% to +1.2022%. All 418,943 rows,
all categorical and metadata columns, and every floating-point result are
bit-for-bit equal. The retained components are accepted from their causal
focused measurements with no measured aggregate regression. Evidence is
`data/profiles/profile_wave_lean_abba_45635_analysis.json`; fresh-build
provenance is under `data/profiles/profile_wave_lean_build_20260718`.

## 2026-07-18 Frozen-Baseline Measurement Reset

The optimization baseline is native commit `891b0a82`, with measurement-only
tooling at `97568f31`. The release extension hash is
`0401514dc83d2c84caddfb8ed368599ade2b94eee1cadc57385e742e36c2b899`.
Ten independent Landau processes each discarded one same-process warm
lifecycle and measured five telemetry-off hot lifecycles. All 50 measurements
produced 418,943 rows, the same two Parquet parts, output hash
`c475ff0e698ffb3d72c6478214b984d696dc71a7beacc56553578cefe4742daf`,
and a byte-identical JAX cache tree. Median hot time is 0.644293 seconds, mean
is 0.647941 seconds, and the population standard deviation is 0.016487
seconds. Position medians range from 0.635637 to 0.650670 seconds without a
monotonic lifecycle trend. Evidence is under
`data/profiles/baseline_891b0a82`.

The focused deep campaign is retained at
`data/profiles/profile_chr22_baseline_97568f31`. Its exact Rust timing reports
326.923 milliseconds of aggregate writer work, including 301.556 milliseconds
inside Parquet file writing, and 126.682 milliseconds in terminal output
completion. The instrumented Memray lifecycle attributes 600.6 MiB to the
compute worker stack, 532.0 MiB to compressed transfer, 264.0 MiB to native
packed8 decode buffers, 117.8 MiB to BGEN open/index state, 29.3 MiB to output
workers, and 9.1 MiB to one packed raw-DEFLATE slab. These figures are
instrumented allocation attribution, not hot-path peak-memory claims.

Profiler limits are explicit. The JAX trace reached its one-million-event cap
during Python/JAX tracing and retained only 21 device events. Nsight Systems
completed the application but hung during capture finalization, Nsight Compute
was denied GPU counters, and Linux `perf` was denied by
`perf_event_paranoid=4`. Py-spy and Scalene primarily sampled startup and cache
deserialization. They cannot override the uninstrumented headline. The prior
serial Nsight capture remains applicable to the unchanged CUDA decoder and
finalizer binaries, while changed JAX control flow requires a new bounded
device capture before another compute optimization is accepted.

A fresh local Skylos 4.29 scan is retained at
`data/profiles/skylos_97568f31/skylos.json`. In this release, `--all` changed
meaning and enables networked security, dependency, secret, and AI checks; the
local whole-product dead-code equivalent is `skylos .`. All 156 reported
production findings were manually adjudicated:

- all 27 functions have direct Rust/Python call sites, trait-dispatch call
  sites, or PyO3 registration; `prepare_compressed_transfer_selection` alone
  has two direct JAX backend calls;
- all three backend classes are constructed dynamically by exact-name
  `getattr` calls in `src/binding/engine.rs`;
- both reported `carry` parameters are required callback positions for JAX
  control flow and are explicitly discarded inside the callback;
- the 124 imports comprise 53 crate/module facade re-exports, 42 direct
  type/value consumers, 19 traits required for method or macro resolution,
  nine generated/cfg/submodule consumers, and one benchmark consumer.

No Skylos finding is dead production code, so this scan causes no product
change. The earlier `--all` attempt was stopped after its changed semantics
were confirmed and is not evidence.

### Ranked opportunities

| Rank | Lane and evidence | Recoverable ceiling | Risk | Cost constraints | Next causal gate |
| ---: | --- | --- | --- | --- | --- |
| 1 | JAX/Firth critical stream: the prior serial device trace has 216.309 ms of kernels; after 142.258 ms of nvCOMP inflate and 11.819 ms of finalization, at most 62.232 ms remains in association/correction kernels. Pprof has 1,118 inclusive `SynchronizeStream` samples. | At most 62.2 ms per chromosome before overlap; actual exposed wall time is lower. | High numerical and control-flow risk. | HLO/cache growth is a veto without a positive application gate; no new dependency or parallel solver. | Already-compiled synchronized executables near 400 and 900 active candidates, plus capacity edges; record HLO, kernels, and cache bytes. |
| 2 | Output terminal drain: 126.682 ms is directly exposed at finish; 326.923 ms of writer work overlaps compute. Pprof assigns 43.29% of non-sync active CPU to output, and Memray attributes 29.3 MiB to its workers. | At most 126.7 ms exposed; ready-all gains are not a wall-time prediction. | Medium schema and ownership risk. | Preserve Zstd level 1, statistics, schema, metadata, and direct Parquet parts. Avoid a Parquet fork for a sub-percent ceiling. | Score/Firth ready-all and paced-finish at 1/4/8 writers, with file bytes and exact Parquet oracle. |
| 3 | BGEN index and packing: pprof assigns 18.60% of non-sync active CPU, split 233 index and 136 packing samples. Memray attributes 117.8 MiB to open/index state and 9.1 MiB to a packed slab. | Wall ceiling is unknown because reading overlaps device/output work; focused CPU throughput is the causal metric. | Low to medium corruption and unsafe-tail risk. | Keep reader-owned pooling and localized initialized-length invariants; no allocator replacement. | Open/index byte throughput; full/tail fresh/pooled pack; sequential/random offsets; corruption and truncation oracle. |
| 4 | CUDA delivery: the unchanged serial trace has 142.258 ms of nvCOMP inflate, 13.777 ms of H2D copies, and 11.819 ms of finalization. Prior readiness analysis found no immediate overlap window. | The measured device work is large, but the tested custom decoder and immediate auxiliary stream both regress; no credible untried ceiling is assigned. | High CUDA ABI, synchronization, and memory risk. | No second decoder, larger chunk, or auxiliary stream until a new timeline disproves earlier results. | Persisted-slab CUDA-event boundary benchmark separating registration, transfer, inflate, finalization, and synchronization. |
| 5 | Remaining materialization: pprof assigns 171 samples and Memray 2.77 MiB after the accepted full-batch fast path. | Small; the common 25-of-26 full batches already avoid JAX slicing. | Low if the partial-tail oracle is exact. | No alternate container representation without measured allocation evidence. | Full and tail materialization replay with all/subset traits and correction/status capacity errors. |

The first benchmark-foundation increment adds the rank-two and rank-three
boundary shapes. It does not itself support a speed claim.

The second foundation increment adds the rank-one compiled-executable gate.
On native baseline `891b0a82` from foundation parent `4e5d73f4`, 30 synchronized
Landau calls have medians of 1.701 milliseconds at 400 active candidates,
2.898 milliseconds at 900, and 2.792 milliseconds at the 1,024-candidate
capacity edge. Every case reports the expected number of valid lanes and a
stable full-result digest. The 900-candidate trace contains 187 device events
with 2.301 milliseconds of aggregate event duration; the largest groups are
the input reductions and loop-select fusions. StableHLO is 108,600 bytes, the
compiled executable text is 711,322 bytes, and compiled temporary memory is
42,419,576 bytes. The warmed 42-file, 284,872-byte persistent-cache tree is
byte-identical before compilation, after compilation, and after all measured
calls. Evidence is under
`data/profiles/firth_compute_baseline_4e5d73f4`. This establishes a causal
comparison gate and does not itself support a speed claim.

The first rank-one experiment attempted to bypass per-lane null-probability
and active-null-deviance initialization when an entire Firth batch was dense.
The predicted ceiling was part of the 1.46 milliseconds attributed to the
three largest input-reduction groups in the 900-candidate trace; the complexity
budget was one batch-level conditional and no new algorithm. A
baseline/candidate/candidate/baseline block measured 60 synchronized calls per
implementation and active-count shape. At 400 candidates the paired geometric
direction is -0.049% with a -1.517% to +1.497% interval. At 900 it is -2.261%
(-3.414% to -1.059%), and at 1,024 it is -1.304% (-3.027% to -0.039%). All
result hashes and valid masks are exact, but StableHLO grows from 108,600 to
120,032 bytes (+10.5%), executable text grows from 711,148 to 785,475 bytes
(+10.5%), temporary device memory grows from 42,419,576 to 52,620,712 bytes
(+24.0%), and the trace grows from 187 to 201 device events. The experiment is
rejected and its production change was removed. Evidence is under
`data/profiles/firth_dense_init_focused_{b1,c1,c2,b2}`.

The second rank-one experiment stacked the information, score-adjustment, and
score vectors into one same-axis reduction, with a predicted ceiling in the
three dominant input-reduction groups and a one-expression complexity budget.
One adjacent 30-call baseline/candidate campaign was sufficient for rejection:
the candidate lost every pair. Paired geometric directions are -46.558% at
400 candidates (-48.773% to -44.705%), -45.803% at 900 (-47.601% to
-44.195%), and -48.667% at 1,024 (-50.166% to -47.222%). Result hashes and
valid masks remain exact, but XLA materializes the stack: StableHLO grows 2.4%,
executable text grows 1.4%, device events increase from 187 to 189, and
aggregate 900-candidate trace duration rises from 2.296 to 3.531 milliseconds.
The experiment was stopped without selective extension and its production
change was removed. Evidence is under
`data/profiles/firth_fused_reductions_{b1,c1}`.

The rank-two part-geometry experiment reduces the internal grouping from 16 to
eight chunks per direct Parquet part. This exposes four chromosome-22 write
tasks to the production eight-writer pool instead of two. In two repeated
30-sample focused campaigns, the eight-writer ready-all median improves from
193.477 to 164.027 milliseconds for score-only output (+15.23%) and from
196.173 to 166.397 milliseconds for Firth output (+15.02%). The GPU-paced
finish median improves from 130.650 to 89.126 milliseconds (+31.51%). All 60
pairs favor the candidate in every focused case. The synthetic benchmark file
size rises 0.061%, below the one-percent veto.

Fresh baseline and candidate wheels have native hashes `dd584a3b` and
`8e5e4aa0`. The complete Landau gate contains ten ABBA blocks, 40 processes,
and five hot lifecycles per process. Median hot
time falls from 0.646805 to 0.609182 seconds. The 20-process-pair geometric
reduction is 5.606%; all 20 pairs favor the candidate, the pair-bootstrap
interval is 4.888% to 6.234%, and the block-bootstrap interval is 4.833% to
6.241%. Pair leave-one-out reductions range from 5.502% to 5.841%, and block
leave-one-out reductions range from 5.435% to 5.900%. The lifecycle-paired
direction is +5.840%, with 98 of 100 lifecycle pairs favoring the candidate.
Every process retains a byte-identical JAX cache tree.

All 418,943 production rows, schema metadata, and all 14 columns are bit-for-bit
equal. Production output changes from two parts to four while total Parquet
bytes fall 0.012%. This fresh direct-Parquet and whole-application evidence
supersedes the July 14 rejection, so eight chunks per part is accepted.
Artifacts, wheels, raw Criterion samples, and statistics are under
`data/profiles/output_part_geometry_3535f2be`.

## 2026-07-18 Raw CUDA Firth Component Experiment

The experiment tested whether a private raw CUDA/XLA FFI kernel could improve
the approximate-Firth compute path without replacing its established pseudo-
Firth and Newton control flow. The kernel fuses probability clipping,
information, score-adjustment numerator, score, and active deviance into one
256-thread block per candidate lane with `f64` accumulation. The predicted
ceiling was the dominant Firth input-reduction work identified by the focused
JAX trace. The complexity budget was one kernel and one internal FFI target;
no dependency, alternate solver, or public API was added.

The synchronized focused gate used five ABBA blocks and 140 paired calls per
shape. Paired geometric reductions are 20.47% at 400 active candidates, 20.44%
at 900, and 20.15% at the 1,024-candidate capacity edge. Pair-bootstrap
intervals are 19.11% to 21.82%, 19.68% to 21.17%, and 19.49% to 20.78%; all
three block-bootstrap intervals and leave-one-block-out ranges are positive.
The candidate StableHLO is 87,142 bytes versus 108,600 (-19.8%), executable
text is approximately 594,157 versus 711,398 bytes (-16.5%), and temporary
device memory is 32,167,800 versus 42,419,576 bytes (-24.2%). The compiled
`sm_70` kernel uses 37 registers, no stack or spills, one barrier, and 256 bytes
of shared memory. The 900-candidate trace attributes 0.535 milliseconds to its
eight raw-kernel launches within 1.648 milliseconds of aggregate device events,
versus 2.301 milliseconds for the baseline executable.

The complete Landau gate used ten ABBA blocks, 40 processes, and five hot
lifecycles per process against one populated persistent cache. Median hot time
falls from 0.606564 to 0.600297 seconds. The 20-process-pair geometric reduction
is 1.043%; 13 of 20 process pairs favor the candidate, the pair-bootstrap
interval is 0.190% to 1.924%, and the block-bootstrap interval is 0.242% to
1.978%. Pair leave-one-out reductions range from 0.793% to 1.211%, and block
leave-one-out reductions range from 0.669% to 1.266%. The lifecycle-paired
direction is +1.043%, with 59 of 100 lifecycle pairs favoring the candidate.
Every measured cache tree remains byte-identical.

All focused valid masks, correction decisions, and `p < 0.05`/`p < 5e-8`
classifications are identical. Maximum absolute differences are `3.33e-16`
for beta, `2.78e-17` for standard error, `9.09e-13` for chi-square, and
`2.97e-11` for log10p. Under the shared production cache, all four Parquet
parts and all 418,943 rows are byte-for-byte equal across every baseline and
candidate lifecycle; their aggregate hash is
`a2912b8a260f572d4355d50d151e28c55bc3152bbb63c8b1cda88e23adf159b1`.

One baseline process in block seven stopped on hot lifecycle three with a
transient packed8 descriptor status `0x00000800`. Its incomplete directory is
retained, and the same prespecified baseline position succeeded when rerun;
no completed process was removed. This failure occurred without the Firth FFI
registration or use flags and is not attributed to the candidate.

The speed hypothesis is accepted, but the diagnostic implementation is not a
merge candidate. It embeds a 40,800-byte V100 `sm_70` cubin in
`g-genotype-cuda`, couples association compute to the genotype-delivery crate,
and lacks a multi-architecture artifact policy. Productionization should put
the kernel under compute ownership, provide an explicit architecture-compatible
module and pure-JAX fallback, restore the packed8 source/artifact provenance,
and rerun this complete gate before enabling it. Focused evidence is under
`data/profiles/raw_cuda_firth_focus_analysis.json`; whole-application evidence
is under `data/profiles/raw_cuda_firth_app_abba_analysis.json` and
`data/profiles/raw_cuda_firth_app_abba`.

## 2026-07-19 Raw CUDA Firth Productionization

The accepted production implementation moves the component kernel and typed-
XLA FFI handler into the independent `g-compute-cuda` crate. It embeds
reproducible compute-70 PTX instead of a V100 cubin, checks the maintained CUDA
source and PTX SHA-256 values during every Linux build, loads the module lazily
per XLA context, and unloads partial module construction through RAII. Binary
Firth GPU backend construction selects it only after an independent Linux,
CUDA-driver-12.2, and compute-capability-7.0 check. CPU, unsupported-device,
and registration-failure paths retain the JAX reduction without an environment
or public configuration switch. Packed8/nvCOMP initialization remains
independent, and the unchanged OpenXLA FFI headers now have one repository-
root vendor owner.

The final release artifacts are frozen at native SHA-256
`bdcdeaf59f72833aea0e5afba74597068b417497043734d4a7ef14d0629fafec`
for baseline commit `ed921664` and
`4c236ca1d455f731090dc5b22e067303c5d2ae7674e645e7ff6478c95f4526e9`
for the candidate. The maintained CUDA source hash is
`4a823918e8b198ef8079cf54e159467c0942ee3d59c99924558d413f7c43585c`;
the NVRTC 12.2.140 PTX hash is
`a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f`.
CUDA 12.9 `ptxas` reports 38 registers, one barrier, 256 bytes of shared
memory, and no stack or spills for `sm_70`.

The final Landau gate used ten ABBA blocks, 40 processes, and five hot
lifecycles per process against the shared populated cache. Median hot time
falls from 0.607211 to 0.597798 seconds. The 20-process-pair geometric
reduction is 2.641%; 18 of 20 process pairs favor the candidate, the pair-
bootstrap interval is 0.866% to 5.470%, and the block-bootstrap interval is
1.017% to 5.225%. Pair leave-one-out reductions range from 1.344% to 2.875%,
and block leave-one-out reductions range from 1.415% to 2.934%. The lifecycle-
paired direction is +2.641%, with 67 of 100 lifecycle pairs favoring the
candidate. Every headline and discarded-warm cache tree remains byte-
identical.

All 418,943 rows, four Parquet parts, schema, metadata, correction decisions,
and output bytes are identical in every completed baseline and candidate
lifecycle. The aggregate Parquet hash is
`fc38c0661b496bf81db9186f10f4b293a61012242f21942248dab532dba4233c`.
A candidate discarded-warm run at block four, position three stopped before
headline measurement with the known packed8 descriptor status `0x00000800`.
Its incomplete directory is retained, and the exact prespecified position
succeeded on retry; no completed measurement was removed or repeated.

The release ELF text grows from 7,717,944 to 7,812,876 bytes (+1.230%), and
the debuginfo-bearing extension file grows 3.837%. The independently positive
pair and block intervals and robust leave-one-out direction clear the code-
growth whole-application veto. The production candidate is accepted. Final
evidence is under
`data/profiles/raw_cuda_firth_production_final_abba_analysis.json` and
`data/profiles/raw_cuda_firth_production_final_abba`.

## 2026-07-20 Production Hygiene And CUDA Driver Ownership

The frozen baseline is `d80447fa1725305d81ebfa2203601382b13bc8a7`; the
candidate is `3cd8f6b6` on `refactor/production-hygiene`. The Cargo and uv lock
hashes are `50a2a94bd901b5ddab5cf86bba946c404c976e676765d98c2abf5da0dbfc49a9`
and `5e5bb29ff46eb578847412fb92950a14dffa44aedacf95b20f65ed3de43dec4b`.
The cleanup gives the duplicated CUDA Driver ABI and loader one private owner
in `native/cuda-driver/cuda_driver.h`, while crate-local adapters preserve the
existing status, XLA error, and module-unload contracts. No Cargo dependency or
public re-export was added. The maintained Firth and packed8 CUDA sources remain
unchanged at hashes `1d15fd1aad609023c849942478764c8d2c67a74ff5acd0909652f2dfa180fce0`
and `673df9629dcb5fec1fc9d688f16349eba7d75bb8a942724f7bcdcd0a0c5dbf1d`;
their PTX remains unchanged at `a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f`
and `a4b7b84171b6a78e6677a5fe1ba84fa6b4fd5a307eef198a5573fb83381ed088`.
The candidate extension is 50,370,872 bytes versus 50,465,240 bytes for the
baseline (-0.187%), a diagnostic size result rather than a causal claim.

One Clippy-driven packed8 API experiment is rejected. Replacing its flat hot
tile boundary with a by-value request aggregate changed the Hilbert Criterion
median from an opening baseline of 7.4892 milliseconds
`[7.4813, 7.5013]` to candidate repeats of 7.6586
`[7.6467, 7.6739]` and 7.6867 `[7.6727, 7.7039]`; the closing baseline was
7.5093 `[7.4985, 7.5210]`. The roughly 2.31% regression caused the wrapper to
be removed. The flat boundary and one locally documented
`clippy::too_many_arguments` allowance remain; the revised candidate is neutral
at 7.5171 `[7.5083, 7.5279]` against the adjacent closing baseline.

The dosage tile request struct is retained as an architecture and lint cleanup,
not a speed optimization. Its opening baseline was 35.784 milliseconds
`[35.475, 36.151]`; candidate repeats were 35.432 `[34.976, 35.810]` and
35.549 `[35.254, 35.867]`; the closing baseline was 36.107
`[35.944, 36.275]`; and the final candidate was 35.681
`[35.469, 35.902]`. Focused commands used the `g-genotype` `bgen_read` bench
with the `bgen_variant_major_{packed8,dosage}_full_samples/16384` filters on an
exclusive 40-CPU Hilbert allocation. Preserved final Criterion data is under
`data/profiles/production_hygiene_20260720/cpu`; the alternating campaign
values above were captured before Criterion overwrote its same-name result.

The bounded whole-application smoke used serial Landau jobs 45798, 45802,
45803, and 45805 in BCCB order against one shared populated JAX cache. The
cache remained byte-identical at nine files, 1,135,968 bytes, and SHA-256
`b5193d5d97fa1b7492eae31d2058735cea75eea50dd576b17b5d0d8aa6229500`
during every discarded warm and hot lifecycle. Hot times were 0.709150 seconds
for B1, 0.617234 for C1, 0.587186 for C2, and 0.609534 for B2. The two pair
directions are +12.96% and +3.67%, and the geometric point direction is +8.43%,
but two one-lifecycle pairs are only a non-regression smoke and do not support
an application speed claim. The baseline and candidate native hashes are
`4d2f06edfc9312fbccf44ac9df6ec9e41672af0d140c893eb1523d886238df68`
and `17c872bc09c72153933536cb8f06e69c60fdeb7c6919e59b67c52c6bdd975409`.

All runs produced 418,943 rows in four parts. The primary numerical oracle is
the matching upstream REGENIE v4.1 approximate-Firth output, SHA-256
`0b9dc124525b6fec63e1b0d3f446263c05f690862235bd84f51b1b3c77b6ed72`.
Variant order, coordinates, identifiers, alleles, and sample counts match.
Against the existing external binary tolerances, standard error, chi-square,
and log10p pass all rows: their maximum absolute differences are 0.000518,
0.002375, and 0.000658. Beta has four score-path rows outside `1e-3`, with a
maximum difference of 0.001475; this is a pre-existing upstream parity gap, not
a candidate change. Both `p < 0.05` and `p < 5e-8` classifications match, all
candidate corrections succeed, and the 17,938 corrected-row count matches the
upstream log.

The old-`g` comparison is retained only as a secondary regression oracle.
`abs(baseline - candidate)` must be less than `5e-7` for beta, `2.5e-7` for
standard error, `2e-6` for chi-square, and `5e-7` for log10p. Its observed
maximum is zero for every field; correction method/status and both significance
classifications are identical. Parquet hashes are diagnostic only. The BGEN,
sample, covariate, phenotype, and prediction-list input hashes were
`bd3f1a8095ca7738878d997910744d33887f087a4127fca3c098cc195b5c1c21`,
`3e165ce47ed36cc3e6d4a8ae422053544f7e64a1b0d7536bd7f4ba70fae20e80`,
`5fa448aa3e8ec96825c62560f29212d7c7c67a98903cf341199499cdc69604e5`,
`1d63186de717b2f9b723fa4949fca59fc87fcf6afa7428f52860dbf2eabebc24`,
and `9f1c44ba25aadb8d1397d3ad39f214f1ae1295cd00c52b0d60dff38451982f26`.
Full summaries and the saved numeric oracles are under
`data/profiles/production_hygiene_20260720`.

Production Clippy with `-D warnings`, Ruff format/check, ty, cargo-machete,
Skylos, CUDA format/lint, CUDA crate checks, relocatable link/ODR inspection,
and documentation build all pass. The `g-genotype` lib-test target has 224
stale test API errors on both baseline and candidate and is deferred with the
requested test/tooling phase; it is not used as candidate evidence.

## 2026-07-20 Stable Minor-Allele Score Reduction

The binary score path previously decoded flipped variants by subtracting the
major-coded dosage from two, then multiplied and reduced the complemented
matrix. For ultra-rare variants this formed a large nearly constant float32
term and cancelled it again in the score statistic. The resulting error was
the cause of the remaining upstream REGENIE score-path beta failures. The
accepted implementation computes the minor-coded value directly. Packed8
delivery uses the exact byte numerators `2 * p0 + p1` for flipped variants and
`510 - 2 * p0 - p1` otherwise. The decoded path reduces 256-sample tiles and
never materializes the complement. The chromosome state now stores the score
right-hand matrix and full Bernoulli weight explicitly instead of maintaining
complement-sum correction terms.

The focused CPU gate ran on an exclusive 40-core Hilbert allocation with fixed
affinity, NUMA interleave, eight warmups, and 80 position-balanced blocks (160
measurements per implementation). The production decoded score median falls
from 43.918273 to 39.144199 milliseconds. Its paired geometric reduction is
15.558%; 77 of 80 blocks favor the candidate and the block-bootstrap 95%
interval is 13.586% to 17.461%. Temporary executable memory falls from
164,626,432 to 26,738,808 bytes, while StableHLO grows from 16,788 to 20,448
bytes. Full materialization (-26.81%), a fused `vmap` reduction (-17.43%), and
128- and 512-sample tiles (-18.27% and -19.91%) were rejected. A first
256-sample campaign was ambiguous, so only the positive fixed-affinity repeat
supports acceptance.

The exact production packed8 GPU gate ran serially on Landau with eight
warmups and the same 80-block, 160-measurement geometry. Median synchronized
time falls from 1.643646 to 1.635858 milliseconds. The paired geometric
reduction is 0.667%; 55 of 80 blocks favor the candidate and the block-
bootstrap interval is 0.249% to 1.077%. StableHLO falls from 18,046 to 16,013
bytes and temporary memory falls from 332,727,080 to 332,415,784 bytes. An
earlier direct-numerator campaign remained ambiguous after extension: its
point was +0.054%, with a -1.039% to +0.750% block interval. A packed
1,252-sample tiled reduction was rejected at -7.070%, with a -7.574% to
-6.560% interval, despite halving temporary memory. A full materialization
variant was also rejected after its repeat showed a stable 0.428% regression.

The whole-application gate used ten ABBA blocks, 40 processes, five hot
lifecycles per process, the production 16,384-variant/512-Firth/1,024-capacity/
eight-writer configuration, and one populated JAX cache. Baseline and
candidate medians are 0.615521 and 0.615272 seconds. The 20-process-pair point
estimate is +0.0117%, with 11 wins, a -1.461% to +1.274% pair-bootstrap
interval, and a -1.567% to +1.261% block-bootstrap interval. Pair leave-one-out
values range from -0.216% to +0.525%; block leave-one-out values range from
-0.297% to +0.637%. Fifty-five of 100 lifecycle pairs favor the candidate.
Every discarded-warm and headline cache tree is identical, and all measured
runs use native SHA-256
`8265e3ad6f5a59cf607941ec67c78f5529b56e82cf1f9caca9a8d43e129b378c`.
The application evidence is neutral, so this change is retained as focused-
only and does not support a whole-application speed claim.

The first attempted gate was stopped before an ABBA summary because native
DEBUG logging polluted the run. After applying the production logging setup,
the clean run contains no `DEBUG` or `jax._src` records. After 39 of 40
measurements, the shared baseline worktree advanced to an unrelated tooling
commit and the final unmeasured position refused the now-changed cache policy.
That empty attempt is retained; only the missing position was run from an
immutable detached worktree at the frozen `6ad69f40` baseline. All 40 measured
summaries report that exact commit. The production diff and four source-file
hashes were identical before and after the campaign.

Final correctness uses upstream REGENIE v4.1, not old `g`, as the primary
oracle. Both full chromosome 22 binary workflows pass 418,943 rows with exact
keys, sample counts, schema, correction aggregates, and significance
classifications. Score-only maximum absolute differences are
`8.72401046753124e-6` beta, `9.53237915035654e-6` standard error,
`5.738945007305318e-5` chi-square, and `1.5566101074115934e-5` log10p; all are
strictly below their configured tolerances. Approximate-Firth maxima are
`8.72401046753124e-6`, `9.53237915035654e-6`,
`8.694763183569876e-5`, and `1.5566101074115934e-5`, respectively. All 17,938
expected Firth corrections succeed and score-only reports no corrections.

Focused reports, rejected experiments, the clean application analysis, run
log, and upstream qualification reports are ignored under
`results/score-flip-stability` and
`results/parity/score-flip-stability/final-qualification`. Benchmark scripts
and generated evidence are not committed.

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
- Use `cargo bench --package g-output --bench writer` for current measurements.

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
