# Custom Kernel Optimization Plan

Last reviewed: 2026-06-07. This is a planning document only. It does not
authorize replacing the current JAX paths without a profiling result, parity
tests, and a runtime fallback.

## Current Recommendation

Custom kernels are still viable, but the post-refactor code changed the order
of attack:

1. Start with a Pallas prototype for packed8 score reductions only if profiler
   traces show packed8 decode plus score reductions are memory-traffic limited.
2. Treat Firth kernels as high-Firth workload specialists, not as a default-run
   speedup. Forced high-Firth chr10 timings still show JAX compute dominating,
   but the compact sparse lane work already removed the easiest all-sample
   sparse waste.
3. Keep CUDA FFI as the second integration path. Pallas is the safer first
   route because it stays inside JAX tracing and shape specialization; FFI is
   better only if the kernel needs lower-level CUDA libraries, explicit warp
   primitives, or packaging control that Pallas cannot provide.
4. Do not pursue GPU-side BGEN decompression. The native reader already
   delivers variant-major dosage or packed8 probability pairs into reusable
   host buffers. The custom-kernel boundary should start after host-to-device
   transfer, where JAX currently decodes or reduces the device arrays.

## Documentation Check

Context7 was queried for current JAX documentation on 2026-06-07. Relevant
upstream docs:

- JAX Pallas quickstart: `https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.md`
- Pallas block/grid design: `https://github.com/jax-ml/jax/blob/main/docs/pallas/design/design.md`
- Pallas GPU reference: `https://github.com/jax-ml/jax/blob/main/docs/pallas/gpu/reference.md`
- JAX FFI guide: `https://github.com/jax-ml/jax/blob/main/docs/ffi.md`

The docs confirm that Pallas is the JAX-native custom-kernel surface for GPU
and TPU code through `pallas_call`, `BlockSpec`, and explicit grid programs.
The FFI guide is the current lower-level route for C++/CUDA implementations and
GPU functions receive a CUDA stream. The repo currently depends on
`jax[cuda12]>=0.10.1` in `pyproject.toml`.

Context7 also returned JAX-Triton as a related library, but the repo does not
currently depend on it. Treat it as a later option only if Pallas and FFI both
prove awkward for the chosen kernel.

## Current Compute Surfaces

Packed8 decode is still the cleanest kernel boundary. The current decoder
casts the whole `uint8` probability-pair tensor to the score dtype and returns a
dense variant-major dosage matrix in `src/g/compute/common/genotype.py:50`.
Packed8 binary entrypoints decode first and then call the canonical
variant-major score paths in `src/g/compute/regenie2_binary/api.py:206` and
`src/g/compute/regenie2_binary/api.py:241`. Non-score-only packed8 Firth runs
decode once and then reuse variant-major correction in
`src/g/compute/regenie2_binary/api.py:401`.

Binary score paths are already heavily vectorized. The main multi-trait binary
score path builds one stacked right-hand matrix and computes genotype products
with a single matrix multiply in `src/g/compute/regenie2_binary/score.py:121`.
It still materializes the decoded genotype matrix and its square before
weighted sum-square reductions in `src/g/compute/regenie2_binary/score.py:116`
and `src/g/compute/regenie2_binary/score.py:151`.

Linear score paths are also already stacked. They normalize the variant-major
dosage, optionally use native sums for dosage sum-square calculation, and run
one stacked projection product in `src/g/compute/regenie2_linear/score.py:52`
through `src/g/compute/regenie2_linear/score.py:85`.

Firth candidate planning no longer has the old host candidate-count sync. It
uses device-side fixed-capacity `nonzero` planning in
`src/g/compute/regenie2_binary/candidates.py:200`, then gathers candidate
genotypes in `src/g/compute/regenie2_binary/firth/batch.py:193`. This is much
better than the old host-dispatched shape choice, but it can still gather and
residualize fixed padded lanes.

Compact sparse Firth already avoids the old full-sample arithmetic for rare
sparse lanes when carrier count is at most 64. Carrier index construction uses
`nonzero` in `src/g/compute/regenie2_binary/firth/batch.py:600`, and the
single-trait and multi-trait compact paths gather fixed carrier slots around
`src/g/compute/regenie2_binary/firth/batch.py:901` and
`src/g/compute/regenie2_binary/firth/batch.py:1063`.

Dense approximate Firth remains reduction-heavy. The scalar approximate Firth
component calculation repeatedly builds probability, weight, information,
deviance, leverage, adjusted response, and score vectors in
`src/g/compute/regenie2_binary/firth/scalar_approx.py:56`. The solver dispatch
runs pseudo-Firth and Newton-Raphson fallback attempts through
`src/g/compute/regenie2_binary/firth/scalar_approx.py:681`.

Full-model Firth is still a high-risk custom-kernel target. Its per-lane
information and adjusted-weight components repeatedly combine small matrix
solves with sample-length reductions in
`src/g/compute/regenie2_binary/firth/full_model.py:103` and
`src/g/compute/regenie2_binary/firth/full_model.py:181`.

The callback boundary is already clean. `src/g/engine/callbacks.py:238` does
one `device_put` per genotype buffer, and the packed8 binary callback passes
the device buffer plus native statistic arrays into packed8 entrypoints at
`src/g/engine/callbacks.py:1767`.

## Existing Timing Evidence

These are stage-timing or benchmark artifacts already present under
`data/profiles`. They are not a substitute for Nsight or HLO inspection, but
they constrain the priorities:

- `compute_opt_optimized_exact_chr10_20260607`, single-trait packed8 default:
  `jax_compute` was about 0.109s for 50k variants, while host-to-device was
  about 0.120s and native engine delivery was about 0.277s.
- `compute_opt_optimized_exact_chr10_20260607`, single-trait packed8 high
  Firth: `jax_compute` was about 0.685s, with roughly 49k total Firth
  candidates across four chunks.
- `firth_large_batch_chr10_ext_20260607`, two-trait packed8 high Firth:
  `jax_compute` was about 2.831s, native engine delivery about 1.945s, and
  host-to-device about 0.109s.
- `firth_large_batch_chr10_ext_20260607` swept batch sizes and candidate
  capacities up to 16,384. Hot two-trait packed8 high-Firth cases clustered
  around 4.1s to 4.35s, so larger candidate capacity by itself does not appear
  to expose a new optimum.

The default packed8 score path is now fast enough that a custom score kernel is
mainly an integration proof and memory-traffic experiment. High-Firth cases are
the stronger speedup candidate, but only after profiler traces show whether the
solver reductions, carrier gathering, or native decode/delivery dominate.

## Ranked Opportunities

### 1. Packed8 Score Reduction Fusion

Status: still viable, first implementation candidate.

Proposed scope: Pallas kernel family that reads packed8 probability pairs,
decodes dosage in registers, applies high-frequency allele flipping or
normalization from native dosage stats, and accumulates score-test components
without writing a dense decoded dosage matrix to global memory.

Start with binary score-only, one trait, one chunk shape. The output should be
the raw components needed to reconstruct the current `Regenie2BinaryScoreChunkResult`:
projection coordinates, weighted genotype sums, weighted genotype sum squares,
score, and flip mask. Keep p-value and extra-code construction in JAX at first.

Then extend to multi-binary and linear score-only. Multi-trait support should
be a wide right-hand-side reduction rather than one kernel launch per trait.

Why this remains useful:

- It removes the dense decoded dosage allocation from packed8 score-only runs.
- It tests Pallas integration against a narrow deterministic surface.
- It can fall back to the current JAX packed8 entrypoints without changing the
  native reader or writer.

Why this is not automatically high impact:

- Existing default packed8 exact timing shows score JAX compute is already a
  small part of wall time.
- Current score paths already use stacked matrix products that are likely
  cuBLAS-backed for the dense-dosage route.

Gate before implementation: Nsight Systems/Compute or HLO evidence that packed8
decode plus score reductions write/read enough dense dosage traffic to matter
for chr10 and chr22 production runs.

### 2. Packed8 Candidate Gather And Separation Fusion

Status: viable after rank 1, not first.

Proposed scope: for non-score-only binary packed8 runs, avoid decoding the full
chunk solely to gather Firth candidate rows. A kernel could combine packed8
decode, candidate mask or pre-dispatch separation checks, flip handling, and
candidate-row gathering into fixed candidate buffers.

Why it may matter:

- High-Firth runs still decode packed8 to dense variant-major dosage before
  candidate correction.
- Candidate preparation gathers padded lanes after score extra-code selection.

Why it should wait:

- Candidate masks depend on the current score result and correction policy.
- The rank 1 score kernel gives the same packed8 decoding machinery with fewer
  numerical branches.

Gate before implementation: profile a high-Firth packed8 run to separate score
decode time from candidate preparation and solver time. If solver reductions
dominate, do not start here.

### 3. Compact Sparse Carrier Fusion

Status: still viable, but the old "sparse Firth custom kernel" note needs
updating.

The current code already creates compact carrier lanes for sparse approximate
Firth when carrier count is at most 64. A custom kernel should not be framed as
"skip inactive samples"; that has largely been done. The remaining opportunity
is to fuse carrier mask construction, `nonzero`, carrier gathers, and the
small fixed-slot scalar Firth component reductions.

Useful kernel surfaces:

- Build carrier counts and fixed carrier-slot indices directly from dense or
  packed8 genotype rows.
- Gather genotype, phenotype, and offset carrier slots in one pass.
- Evaluate compact scalar Firth components over 64 slots with one lane or lane
  group per program.

Why it may matter:

- High-Firth diagnostics show many candidates are sparse correction lanes.
- The current compact path still uses several `take`, `take_along_axis`, and
  scatter steps before running the fixed-slot solver.

Risk:

- The current compact path is correct and simple JAX. A custom kernel must
  preserve exact active-slot masking, null-failure behavior, diagnostics, and
  scalar float64 calculations.

Gate before implementation: profiler evidence that compact carrier
construction/gather or compact scalar Firth components are a meaningful fraction
of high-Firth JAX compute.

### 4. Dense Approximate Firth Component Reductions

Status: viable only for high-Firth workloads.

Proposed scope: custom kernels for the repeated scalar approximate-Firth
component reductions over all samples. These kernels would compute probability,
weight, genotype information, penalized deviance, score, and validity flags
without materializing intermediate sample-length vectors.

Why it may matter:

- The scalar path repeats the same vector operations inside nested while loops.
- Two-trait high-Firth timing shows JAX compute can dominate the measured hot
  run.

Why it is risky:

- The solver uses float64, multiple fallback attempts, convergence diagnostics,
  and branch-heavy line search behavior.
- Pallas support for the exact reduction structure must be proven before any
  production switch.

Gate before implementation: use profiling to confirm this path dominates over
candidate prep and compact sparse handling, then implement a standalone
component-kernel parity test before integrating into the solver loop.

### 5. Candidate Compaction And Grouping Kernel

Status: lower priority after the refactor.

The current tiered tiny/small/bounded/overflow dispatch reduced the worst
fixed-capacity waste. Larger capacity sweeps did not reveal a better hot
setting. A custom compaction kernel could still combine candidate counting,
active position generation, heuristic grouping, and maybe sparse carrier flags,
but it is unlikely to beat the higher-ranked opportunities unless traces show
`nonzero`, grouping, or scatter dominates low-candidate chunks.

Gate before implementation: HLO or profiler evidence that candidate planning is
larger than solver time for realistic candidate counts.

### 6. Full-Model Firth Block-Math Kernels

Status: possible but not a near-term target.

The full-model Firth path has small dense solves and repeated reductions. It is
more numerically complex than scalar approximate Firth and less central to the
current tuned default. Custom kernels here should wait until scalar approximate
Firth work is exhausted and block-math is proven to matter for production
workloads.

Gate before implementation: a benchmark where `use_block_math` is the chosen
production path and profiler traces show repeated full-model information
component reductions dominate.

### 7. GPU BGEN Decode Or Direct Native-To-Device Delivery

Status: not recommended now.

The current native reader performs decompression, validation, selected-sample
alignment, and packed8/dosage host-buffer filling. Moving this to GPU would
cross the Rust/Python/JAX ownership boundary and require solving decompression
and IO scheduling problems, not just CUDA math. Packed8 already reduces
host-to-device bytes; focus GPU kernel work on device-side decode and
reductions.

## Validation Plan

Every custom-kernel branch must keep the existing JAX implementation as a
runtime fallback and must add parity tests before benchmark claims.

Packed8 score parity:

- Compare packed8 custom score components against current decoded JAX results
  for binary one-trait, multi-binary two-trait, linear one-trait, and
  multi-linear cases.
- Cover flipped variants, non-flipped variants, all 0/1/2 dosage examples, and
  tail chunk shapes.
- Compare binary `extra_code`, Firth candidate mask, beta, standard error,
  chi-squared, log10 p-value, and valid mask.
- Compare linear beta, standard error, chi-squared, log10 p-value, and valid
  mask.

Firth parity:

- Preserve existing packed8 versus variant-major Firth parity tests.
- Add direct scalar component parity for dense and compact sparse lanes before
  replacing any solver body.
- Cover pseudo-Firth success, Newton-Raphson warm-start fallback, null failure,
  sparse carrier-only correction, empty candidate batches, tiny/small/bounded
  capacity paths, and overflow paths.
- Compare diagnostics counts and per-lane correction/failure codes, not just
  public statistics.

Performance validation:

- Measure chr22 and chr10 50k slices, then full chromosome runs if the 50k
  slices improve.
- Use both `--stage-timing-mode off` for production throughput and
  `--stage-timing-mode exact` for synchronized stage attribution.
- Use Nsight or XLA profiler traces for accepted kernels. Stage timings alone
  are not enough to prove a custom kernel is better.
- Report compile time separately from hot same-process runtime.
- Benchmark packed8 and variant-major paths; a custom packed8 path must not
  regress the canonical variant-major path.

## Suggested Profiling Commands

Use `srun` on a GPU node for these workloads. The exact SLURM options depend on
the current cluster queue, but the benchmark surface should stay the same.

Score/default packed8 gate:

```bash
uv run python scripts/benchmark_regenie2_binary_hot.py \
  --bgen 1kg_chr10_full.bgen \
  --sample 1kg_chr10_full.sample \
  --prediction-list baselines_chr10/regenie_step1_pred.list \
  --storage-modes packed8 \
  --fallback-density-scenarios default \
  --stage-timing-mode exact \
  --variant-limit 50000 \
  --no-include-cold-process \
  --no-include-finalized-hot
```

High-Firth packed8 gate:

```bash
uv run python scripts/benchmark_regenie2_binary_hot.py \
  --bgen 1kg_chr10_full.bgen \
  --sample 1kg_chr10_full.sample \
  --phenotype-file profiles/firth_tuning_inputs/pheno_bin_two_traits.txt \
  --prediction-list profiles/firth_tuning_inputs/regenie_step1_chr10_two_traits_pred.list \
  --phenotype-columns phenotype_binary,phenotype_binary_flip \
  --binary-trait-counts 2 \
  --storage-modes packed8 \
  --fallback-density-scenarios high \
  --firth-batch-sizes 1024 \
  --firth-candidate-capacities 2048 \
  --stage-timing-mode exact \
  --variant-limit 50000 \
  --no-include-cold-process \
  --no-include-finalized-hot
```

Production-throughput confirmation:

```bash
uv run python scripts/benchmark_regenie2_binary_hot.py \
  --storage-modes variant_major,packed8 \
  --fallback-density-scenarios default,high \
  --stage-timing-mode off \
  --variant-limit 50000 \
  --no-include-cold-process \
  --no-include-finalized-hot
```

## Implementation Plan When Approved

1. Add an experimental module for custom kernel wrappers, guarded by a static
   kernel mode that defaults to the current JAX implementation.
2. Implement packed8 one-trait binary score component parity with Pallas.
3. Add focused tests that compare the custom component output to the current
   decoded JAX score path on small CPU/GPU-compatible fixtures. If Pallas is
   GPU-only for the chosen primitive, skip CPU execution cleanly and keep JAX
   fallback tests mandatory.
4. Run chr10 and chr22 score-only packed8 stage-timing and production
   benchmarks.
5. Extend only if there is a measured speedup after compile warmup. The next
   extension should be multi-binary score if the first kernel is faster, or
   compact sparse Firth if high-Firth traces point there instead.
6. Consider CUDA FFI only after a Pallas prototype proves the math and shows
   that Pallas code generation or supported primitives are the limiting factor.

## Stop Criteria

Stop custom-kernel work if any of the following holds:

- Packed8 score fusion improves microbenchmarks but does not improve chr10 or
  chr22 production hot runs after warmup.
- Numeric parity requires tolerances wider than the current JAX versus
  variant-major/packed8 tolerances.
- The fallback path or packaging story makes normal CPU and non-CUDA test runs
  fragile.
- Nsight traces show native BGEN delivery, Python orchestration, or output
  writing dominate the target workload instead of JAX kernel memory traffic.
