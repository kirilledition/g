# Compute Optimization Opportunities

Review date: 2026-06-07.

Scope: compute and compute-adjacent data movement in the active GWAS engine
paths. This is a findings document only. It does not propose changing numerical
semantics without follow-up validation.

## Executive Summary

The current hot compute paths are already mostly expressed as JAX programs:
score kernels are entered through jitted API wrappers, candidate correction uses
device-side `lax.cond`, and Firth solver loops use `vmap`, `scan`, and
`while_loop` instead of Python loops. The biggest remaining optimization
opportunities are not missing `jit` decorators in the core loops. They are:

1. Default chunk timing/diagnostics force device synchronization and likely
   reduce overlap between native decode, host-to-device transfer, JAX compute,
   and writing.
2. Packed8 delivery saves host-to-device bandwidth, but JAX then decodes into a
   full float dosage matrix before score and Firth work.
3. Binary and linear score kernels reread the same genotype operand through
   several matrix/reduction expressions that may be better stacked or fused.
4. Firth candidate preparation has fixed-capacity gather/sort shapes; for small
   candidate counts, this can allocate and move far more candidate rows than are
   active.
5. Sparse Firth lanes still execute dense sample-length reductions in JAX.
   Sparse metadata changes the correction mask, not the arithmetic complexity.

## Ranked Findings

| Rank | Opportunity | Ease | Risk | Expected value |
| --- | --- | --- | --- | --- |
| 1 | Make default timing/diagnostics non-synchronizing unless exact profiling is requested | Medium | Low to medium | High, because it can restore asynchronous overlap |
| 2 | Cast public float32 output statistics on device before `device_get` | Medium | Low | Medium to high for Firth/float64 result paths |
| 3 | Extend warm-cache coverage to production variant-major and packed8 donating entry points | Easy to medium | Low | Medium for startup latency and first-chunk stability |
| 4 | Stack/fuse score-kernel matrix products that share the genotype operand | Medium | Medium | Medium to high on GPU score paths |
| 5 | Add tiered Firth candidate capacities below the current bounded capacity | Medium | Medium | Medium for low-candidate chunks |
| 6 | Avoid per-chunk cloning/device_put of small native stats arrays where possible | Medium | Low | Low to medium |
| 7 | Make binary diagnostics cheaper or optional at chunk granularity | Easy to medium | Low | Medium when diagnostics are enabled |
| 8 | Represent sparse Firth lanes with compact carrier lists or a sparse custom kernel | Hard | Medium to high | Potentially high for rare sparse Firth |
| 9 | Fuse packed8 decode, normalization/flipping, score reductions, and candidate gathers with custom kernels | Hard | High | Highest ceiling, especially packed8 GPU runs |
| 10 | Consider custom CUDA/Pallas kernels for Firth lane reductions | Hard | High | High ceiling for approximate Firth, but most validation-heavy |

## Implementation Results

Status on 2026-06-07: all non-custom-kernel opportunities from this review have
been implemented and validated. Custom packed8/Firth kernels remain deliberately
deferred for separate design review.

Implemented changes:

- exact stage timing is now opt-in for blocking synchronization; production
  callbacks keep aggregate timing records without forcing per-chunk
  `block_until_ready` or diagnostic `device_get`;
- public result statistics are narrowed to float32 on device before host
  materialization;
- warm-cache coverage now uses production variant-major and packed8 donating
  entry points;
- linear score and binary score paths stack shared genotype matrix products;
- Firth candidate dispatch now has monotonic tiny/small/bounded/overflow
  capacities and skips sorting for tiny/small tiers;
- approximate sparse Firth can run compact fixed-size carrier lanes for rare
  sparse candidates, with guards that keep dense-only runs on the prior dense
  path;
- native chunk stats needed for compute are bundled through one PyO3 call.

Focused validation:

- `uv run pytest tests/test_regenie2_linear.py tests/test_regenie2_binary.py tests/test_timing.py tests/test_telemetry.py tests/test_warm_cache.py tests/test_regenie2_pipeline.py -q`:
  170 passed, 1 skipped before the final worker-timeout follow-up.
- `uv run pytest tests/test_regenie2_binary.py -q`: 56 passed, 1 skipped after
  compact sparse Firth guard changes.
- `uv run pytest tests/test_regenie_comparison_scripts.py -q -k 'binary_hot'`:
  7 passed after benchmark timing-mode support.
- `uv run pytest tests/test_regenie2_pipeline.py -q -k 'worker or timing or binary_chunk_diagnostics_are_detailed_only_for_exact_timing'`:
  14 passed after graceful worker join timeout extension.
- `uv run ty check` and `uv run ruff check` passed for each touched scope, and
  `cargo fmt --check` plus `. scripts/server_env.sh && cargo test --test rust_python_bindings`
  passed after the timing/output changes.

GPU hot benchmark result, measured on `landau` with one binary trait, 50k
variants, chr22/chr10, variant-major/packed8 storage, default/high Firth
fallback density, no final Parquet materialization:

| Dataset | Case | Baseline exact | Optimized exact | Optimized production | Production speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| chr22 | variant/default | 0.865716s | 0.865059s | 0.750624s | 1.153x |
| chr22 | variant/high-Firth | 1.940100s | 1.436061s | 1.309201s | 1.482x |
| chr22 | packed8/default | 0.769198s | 0.760673s | 0.651605s | 1.180x |
| chr22 | packed8/high-Firth | 1.867004s | 1.367826s | 1.236620s | 1.510x |
| chr10 | variant/default | 1.346740s | 1.372751s | 1.174464s | 1.147x |
| chr10 | variant/high-Firth | 2.422376s | 1.878304s | 1.706274s | 1.420x |
| chr10 | packed8/default | 1.268596s | 1.221303s | 1.062988s | 1.193x |
| chr10 | packed8/high-Firth | 2.308150s | 1.787682s | 1.617899s | 1.427x |

Summary JSON paths:

- `data/profiles/compute_opt_baseline_chr22_20260607/regenie2_binary_hot_summary.json`
- `data/profiles/compute_opt_baseline_chr10_20260607/regenie2_binary_hot_summary.json`
- `data/profiles/compute_opt_optimized_exact_chr22_20260607/regenie2_binary_hot_summary.json`
- `data/profiles/compute_opt_optimized_exact_chr10_20260607/regenie2_binary_hot_summary.json`
- `data/profiles/compute_opt_optimized_off_chr22_20260607/regenie2_binary_hot_summary.json`
- `data/profiles/compute_opt_optimized_off_chr10_retry_20260607/regenie2_binary_hot_summary.json`

An initial chr10 production-throughput repeat exposed a cold no-exact shutdown
timeout while the callback worker was still compiling/draining queued JAX work.
Normal graceful callback joins now use a longer timeout; abort paths keep their
short timeout.

## Findings

### 1. Default Timing And Diagnostics Synchronize Every Chunk

The callback helpers are written as if synchronization is conditional on
`stage_timing_recorder is not None`: host-to-device transfer blocks at
[callbacks.py:200](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:200),
compute blocks at
[callbacks.py:214](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:214),
and binary diagnostics call `jax.device_get` at
[callbacks.py:176](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:176).
However, pipeline context construction turns `None` into a recorder by default
at
[regenie2_pipeline.py:197](/mnt/beegfs/kirill/Projects/g/src/g/engine/regenie2_pipeline.py:197).

That means normal runs appear to get exact per-stage timing at the cost of
forced chunk synchronization. This can reduce or remove overlap between native
decode, `device_put`, JAX compute, and writer work. It also makes performance
benchmarks reflect instrumentation overhead unless the benchmark explicitly
models this behavior.

Potential direction: split "record stage durations" from "block for exact stage
durations". Keep passive wall-clock stage records available, but make exact
blocking an explicit profiling mode. For binary diagnostics, either attach the
diagnostic payload to the main materialization point or sample/aggregate less
often.

Validation: compare end-to-end chr22 and chr10 runs with exact timing on and
off; also compare stage timing totals against a profiler trace so we know what
visibility is lost.

### 2. Public Output Narrowing Happens After Device-To-Host Transfer

The native writer stores public statistics as float32, but current write helpers
materialize JAX arrays first and then cast on the host:
[callbacks.py:243](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:243),
[callbacks.py:258](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:258),
[callbacks.py:283](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:283),
and [callbacks.py:307](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:307).

For float32 score-only paths this is mostly a no-op. For approximate Firth
paths, Firth result arrays are float64 internally and the current sequence can
double device-to-host bandwidth for `beta`, `standard_error`, `chi_squared`, and
`log10_p_value`, then spend host CPU/memory bandwidth narrowing them.

Potential direction: introduce a narrow-to-public-output JAX helper close to the
writer boundary. It should return float32 statistic arrays plus int32 extra code
without changing internal result dataclasses or numerical checks. This preserves
the public output schema while moving the bandwidth reduction to the device.

Validation: exact output parity at float32 file precision, dtype assertions in
writer tests, and end-to-end timing of Firth-heavy chunks.

### 3. Warm Cache Does Not Cover Production Packed8/Variant-Major Entrypoints

The warm-cache helpers build sample-major synthetic genotype matrices and call
`compute_regenie2_*_chunk_from_chromosome_state`:
[warm_cache.py:133](/mnt/beegfs/kirill/Projects/g/src/g/engine/warm_cache.py:133)
and [warm_cache.py:204](/mnt/beegfs/kirill/Projects/g/src/g/engine/warm_cache.py:204).
Production BGEN GPU callbacks commonly call variant-major donating or packed8
donating entry points with separate static/donate signatures:
[callbacks.py:1562](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1562),
[callbacks.py:1628](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1628),
[callbacks.py:1812](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1812),
and [callbacks.py:1881](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1881).

This can leave first production chunks paying compilation or autotune cost even
after a warm-cache call. Packed8 in particular has a distinct decode-plus-score
JIT path at
[api.py:201](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/api.py:201)
and [api.py:236](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/api.py:236).

Potential direction: warm the same entrypoint family selected by
`gpu_genotype_format`, association mode, score-only versus Firth correction, and
single versus multi-trait mode. Use synthetic packed8 probability pairs for the
packed8 path.

Validation: inspect compile logs or persistent cache hits, then verify first
real chunk latency with and without warming.

### 4. Linear Score Reads The Same Normalized Genotype Operand In Multiple GEMMs

The linear score path normalizes/flips the variant-major genotype matrix, then
uses it for genotype sum-of-squares, covariate projection, and phenotype
covariance:
[score.py:52](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_linear/score.py:52),
[score.py:67](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_linear/score.py:67),
[score.py:80](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_linear/score.py:80),
and [score.py:81](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_linear/score.py:81).

When native stats are present, sum-of-squares avoids a full matrix square, but
the two GEMMs still read the normalized genotype operand separately. A stacked
left-hand matrix combining `whitened_covariate_transpose` and
`adjusted_residual_matrix` could compute both products in one larger GEMM, then
split the result. This may increase BLAS efficiency and reduce memory traffic.

Potential direction: prototype a stacked-GEMM variant and compare HLO/profiler
output against the current two-GEMM version. The shape balance matters:
covariate count is usually small, trait count can vary, and the current version
may already be good when cuBLAS sees favorable dimensions.

Validation: bitwise/close parity tests for linear score, plus GPU benchmark on
single-trait and multi-trait runs.

### 5. Binary Score Uses Several Genotype-Operand Reductions

The multi-binary score path builds several products from the same raw genotype
matrix: projection coordinates, weighted genotype sum-of-squares, weighted
genotype sum, and score:
[score.py:113](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/score.py:113),
[score.py:123](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/score.py:123),
[score.py:128](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/score.py:128),
and [score.py:146](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/score.py:146).

Some of these can be re-expressed as fewer matrix products. For example,
`score_residual`, `bernoulli_weight`, and flattened projection rows can be
stacked into one right-hand matrix so one `G @ RHS.T` produces raw scores,
weighted sums, and projection coordinates. Weighted sum-of-squares still needs
`G^2`, unless a fused kernel computes it while streaming genotypes.

Potential direction: prototype a stacked binary score kernel and compare XLA
HLO, memory traffic, and cuBLAS calls. This is especially relevant in
multi-trait runs where the right-hand side is wide enough for BLAS to be
efficient.

Validation: score-only parity, Firth-candidate mask parity, and benchmark cases
with one trait, moderate trait counts, and packed8 input.

### 6. Fixed-Capacity Firth Candidate Prep Can Over-Allocate For Sparse Candidate Counts

Device-side candidate planning uses `jnp.nonzero(..., size=candidate_capacity)`,
pads to a multiple of batch size, and creates active positions:
[candidates.py:189](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/candidates.py:189)
through [candidates.py:207](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/candidates.py:207).
Candidate input preparation then gathers padded genotype rows and per-lane
state:
[batch.py:176](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/batch.py:176),
[batch.py:258](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/batch.py:258),
and [batch.py:288](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/batch.py:288).

The current bounded/overflow design avoids the old host candidate-count sync,
but if `fallback_count` is tiny it still prepares the bounded fixed shape. With
the default `firth-candidate-capacity = 2048`, one candidate can still mean
gathering and sorting thousands of padded lanes.

Potential direction: add tiered device dispatch, for example zero, tiny, bounded,
overflow. A tiny capacity such as 64 or 128 could reduce preparation memory for
low-candidate chunks while preserving static shapes. The cost is more compiled
branches and more configuration/tuning surface.

Validation: candidate-count distribution from real runs, compile-cache impact,
and hot Firth benchmarks with candidate counts near 1, 64, 512, 2048, and
overflow.

### 7. Candidate Lane Sorting Has A Real Cost

The Firth prep sorts active lanes by heuristic expected runtime:
[candidates.py:235](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/candidates.py:235),
[candidates.py:237](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/candidates.py:237),
[candidates.py:267](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/candidates.py:267),
and [candidates.py:269](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/candidates.py:269).

This is a reasonable tradeoff if it reduces batch tail latency, but it is also a
full fixed-capacity sort. For very low candidate counts, the sort may dominate
preparation. For high candidate counts, the grouping likely helps the scanned
batch loop avoid one long-running lane keeping a batch active.

Potential direction: only sort when `fallback_count` crosses a threshold, or
measure whether a two-bucket stable partition is enough. This should not be
changed without profiling, because it may worsen tail latency.

Validation: compare Firth iteration histograms and batch runtimes with sorting,
no sorting, and bucket-only grouping.

### 8. Sparse Firth Still Scans Dense Sample Vectors

The scalar approximate-Firth path has a sparse correction mask, but component
computation still performs sample-length vector math and reductions with
`jnp.where(active_sample_mask, ..., 0.0)`:
[scalar_approx.py:67](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/scalar_approx.py:67),
[scalar_approx.py:82](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/scalar_approx.py:82),
[scalar_approx.py:132](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/scalar_approx.py:132),
and [scalar_approx.py:435](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/scalar_approx.py:435).

This means rare sparse Firth candidates do not get an arithmetic reduction
proportional to carrier count; they still scan every sample. Native chunk stats
already identify rare sparse candidates, and callbacks pass that mask to the
device:
[callbacks.py:1585](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1585)
and [callbacks.py:1891](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1891).

Potential direction: explore carrier-index lists for rare sparse candidates, or
a custom kernel that works directly on packed/dense genotype and skips inactive
samples. This is a larger design change because candidate lanes have variable
carrier counts and numerical parity with REGENIE must be preserved.

Validation: start with a read-only benchmark that reports carrier-count
distribution, sparse candidate count, and time spent in scalar Firth reductions.

### 9. Chunk Stats Are Cloned Into Python Arrays Per Access

Rust exposes `ChunkStats` properties by cloning native vectors into NumPy arrays:
[mod.rs:73](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:73),
[mod.rs:78](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:78),
[mod.rs:93](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:93),
and [mod.rs:128](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:128).
Callbacks then immediately `device_put` several of those arrays:
[callbacks.py:1566](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1566),
[callbacks.py:1567](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1567),
[callbacks.py:1816](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1816),
and [callbacks.py:1817](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:1817).

These arrays are small compared with the genotype payload, but the overhead is
per chunk and gets worse as more stats are consumed. The writer still benefits
from holding the native `ChunkStats`, so this is mainly a compute-side staging
issue.

Potential direction: expose a method that returns the compute-needed stats in
one Python object, cache NumPy views for a chunk, or pass a single stats pytree
to JAX. The goal is fewer property calls, fewer clones, and fewer small
`device_put`s.

Validation: microbenchmark per-chunk callback overhead with genotype compute
mocked out, then end-to-end benchmark with small chunks where Python overhead is
more visible.

### 10. Binary Diagnostics Sort Whole Chunks

Binary diagnostic counting computes a median by sorting a full chunk-sized
iteration-count vector:
[diagnostics.py:83](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/diagnostics.py:83).
The host sync happens immediately after compute through
[callbacks.py:176](/mnt/beegfs/kirill/Projects/g/src/g/engine/callbacks.py:176).

This is useful for detailed Firth diagnosis, but expensive for routine timing.
It can also make a low-candidate chunk pay a full-chunk sort just to record a
median.

Potential direction: make median diagnostics opt-in, compute only count/min/max
by default, or compute median from compacted active positions. If diagnostics
are retained, materialize them with output arrays rather than introducing a
separate synchronization point.

Validation: compare diagnostics payloads before/after and measure chunk runtime
with diagnostics on/off.

### 11. Packed8 Decode Is A Major Fusion Candidate

Packed8 delivery writes trusted probability pairs into reusable host buffers:
[mod.rs:907](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:907)
through [mod.rs:975](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:975).
Rust also computes chunk stats while writing packed8 rows:
[reader.rs:487](/mnt/beegfs/kirill/Projects/g/src/genotype/bgen/reader.rs:487)
through [reader.rs:512](/mnt/beegfs/kirill/Projects/g/src/genotype/bgen/reader.rs:512).
The JAX side then converts the entire packed buffer to the score dtype and
returns a dense dosage matrix:
[genotype.py:64](/mnt/beegfs/kirill/Projects/g/src/g/compute/common/genotype.py:64)
through [genotype.py:72](/mnt/beegfs/kirill/Projects/g/src/g/compute/common/genotype.py:72).

This is the cleanest custom-kernel opportunity. A fused kernel could read packed
bytes, decode dosage in registers, apply flip/normalization, and accumulate
score/projection/sum-square outputs without storing the full decoded float
matrix in global memory. For Firth candidates, the same kernel family could
gather only selected candidate rows.

Potential direction: start with a custom score-only packed8 kernel for the
binary score path or linear path, not full Firth. That isolates correctness and
lets us compare against JAX output before layering in candidate correction.

Validation: exact/close parity against current JAX for packed8 score-only,
candidate-mask parity for binary, and Nsight memory-throughput comparison.

### 12. Firth Lane Reductions Are Custom-CUDA Candidates, Not Simple JIT Gaps

Fixed-size Firth batching is already JAX control flow:
[batch.py:526](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/batch.py:526)
and [batch.py:596](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/batch.py:596).
The per-lane full model uses `while_loop`, Cholesky solves, and repeated
sample-length reductions:
[full_model.py:286](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/full_model.py:286),
[full_model.py:301](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/full_model.py:301),
[full_model.py:343](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/full_model.py:343),
and [full_model.py:440](/mnt/beegfs/kirill/Projects/g/src/g/compute/regenie2_binary/firth/full_model.py:440).

The next large speedup is probably not another `jit`; it is reducing global
memory traffic and launch overhead in the repeated lane reductions. A custom
kernel could assign one candidate lane or lane group to a block, use shared or
warp reductions for score/information terms, and avoid materializing temporary
vectors such as probability, weight, leverage, and adjusted response when only
their reductions are needed.

This is high risk. It touches numerical convergence, double precision behavior,
branch divergence, and diagnostics. It should follow only after profiler traces
show Firth lane reductions dominate and after the score-only packed8/custom
kernel path proves the integration approach.

## Notes On What Already Looks Reasonable

- The core score and Firth compute paths are mostly not blocked by Python loops.
  The hot Firth loops use JAX `while_loop`, `scan`, and `vmap`.
- Device-side Firth candidate dispatch has already removed the old host
  candidate-count synchronization. Current dispatch uses device `lax.cond` and
  static bounded/overflow capacities.
- Native BGEN variant-major reads already use reusable callback-owned buffers
  and Rayon tile parallelism:
  [mod.rs:851](/mnt/beegfs/kirill/Projects/g/src/python/mod.rs:851)
  and [reader.rs:550](/mnt/beegfs/kirill/Projects/g/src/genotype/bgen/reader.rs:550).
- Packed8 host delivery is justified when the input satisfies trusted
  no-missing diploid constraints. The concern is the later dense JAX decode, not
  the native packed delivery itself.

## Suggested Review Order

1. Decide whether default runs should block for exact stage timings. This is
   non-mathematical and likely the safest high-value change.
2. Prototype device-side public-output narrowing. This is local to the writer
   boundary and should be easy to validate.
3. Extend warm-cache coverage to actual production entrypoints.
4. Prototype score-kernel stacking for linear and binary score-only paths.
5. Add low-candidate Firth capacity tiering if candidate-count distributions
   support it.
6. Defer custom kernels until profiler traces identify whether packed8 decode,
   score reductions, or Firth lane reductions dominate on the target datasets.
