# Firth Optimization Plan

Date: 2026-06-06
Base worktree: `cleanup/review-easy-nonconfig`

This note expands the Firth-related findings in `docs/codebase-review-findings.md`.
It focuses on three performance items:

- removing the Firth candidate-count host synchronization, addressed in
  `opt/firth-device-capacity-dispatch`;
- batching multi-binary approximate Firth instead of dispatching one trait at a
  time, addressed in `opt/multi-binary-firth-batching`;
- extending the binary benchmark harness so Firth batching and capacity choices
  are measured across trait counts, addressed in `bench/multi-binary-firth-hot`.

SLURM GPU validation has passed for the completed batching work: the default
smoke benchmark, default full binary-hot benchmark, a targeted two-trait
variant-major plus packed8 smoke, and two-trait 50k-variant tuning sweeps on
chr10 and chr22 all ran on `landau`. A larger 1/2/4/8-trait matrix remains
optional follow-up for deeper tuning.

## Current State

Single-trait approximate Firth is device-side after static chunk-capacity bounds
are built from the chunk shape. `apply_device_candidate_corrections_firth_variant_major`
records a non-blocking `firth_candidate_dispatch_plan` timing stage, then calls a
jitted dispatcher. The dispatcher builds `candidate_mask` and `fallback_count` on
device and uses `jax.lax.cond` to return empty Firth diagnostics for zero
candidates, use bounded capacity for normal chunks, or use full chunk capacity
for overflow chunks.

The old host sync existed because `build_device_firth_batch_plan` uses
`jnp.nonzero(..., size=candidate_capacity)`, fixed padding, reshapes, and a fixed
scan length. Those shapes require a Python/static `candidate_capacity`.

Multi-binary score-only execution is genuinely trait-batched. Multi-binary
approximate Firth is now also batched over flattened trait-variant candidate
lanes. The non-score multi path computes one batched multi-score result per
chunk, expands it with multi-shaped Firth diagnostics, applies device-side
zero/bounded/overflow dispatch over selected lanes, and scatters corrected
statistics back to trait-major `[traits, variants]` arrays.

## Recommended Priority

1. Done: add preparatory correctness tests for the current Firth behavior.
2. Done in `opt/firth-device-capacity-dispatch`: remove the single-trait host
   sync with device-side bounded/overflow dispatch.
3. Done in `opt/multi-binary-firth-batching`: batch multi-binary approximate
   Firth over flattened trait-variant candidate lanes and reuse one batched
   multi-score result.
4. Done in `bench/multi-binary-firth-hot`: extend benchmark coverage for
   multi-binary Firth trait counts, candidate capacities, and Firth batch sizes.
5. Done for the default full benchmark, targeted two-trait smoke, bounded
   chr10/chr22 two-trait 50k tuning sweeps, and larger chr10/chr22 two-trait
   50k batch/capacity sweeps. The larger sweeps support the current defaults
   `firth-batch-size = 1024` and `firth-candidate-capacity = 2048`. Run larger
   1/2/4/8-trait GPU sweeps only if future tuning decisions need more data.

The completed single-trait dispatcher is the primitive the multi-binary batching
work reuses conceptually. Multi-binary batching did not reintroduce per-trait
host counts.

## Task 1: Preparatory Tests

Add CPU-safe tests before optimizing:

- zero-candidate approximate Firth returns score results expanded with empty
  Firth diagnostics and does not require a usable chromosome state;
- explicit bounded-capacity and overflow-capacity correction results match;
- overflow corrects every forced candidate and does not truncate the candidate
  vector through `jnp.nonzero(size=...)`;
- multi-binary approximate Firth equals stacked single-trait results for
  different per-trait candidate masks, not just identical duplicated traits;
- sparse candidate masks stay variant-only and do not create new Firth
  candidates;
- null Firth/logistic failure in one trait does not poison other traits;
- packed8 approximate Firth multi-binary output matches decoded variant-major
  dosage output.

Current coverage strengthens the zero-candidate diagnostic contract and adds
variant-major multi-binary approximate Firth parity whose traits have different
score-stage Firth candidate masks. It also covers sparse masks, single-binary
packed8 approximate Firth parity, multi-binary packed8 approximate Firth parity,
failed Firth candidate labeling, one multi-score dispatch per corrected chunk,
and per-trait null Firth/logistic failure isolation.

Expected files:

- `tests/test_regenie2_binary.py`
- `tests/test_regenie2_pipeline.py`

Suggested checks:

```bash
uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q
uv run ty check src/g tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py
```

## Task 2: Remove Candidate Count Host Sync

Status: implemented in `opt/firth-device-capacity-dispatch`.

Recommended implementation: replace host capacity selection with a jitted
device-side bounded/overflow branch.

Shape contract:

- compute `candidate_mask` and `fallback_count` on device;
- return expanded empty Firth diagnostics when `fallback_count == 0`;
- use bounded capacity when `fallback_count <= bounded_candidate_capacity`;
- use full chunk capacity when `fallback_count > bounded_candidate_capacity`;
- both branches return the same `Regenie2BinaryChunkResult` shape.

Implementation status:

- the fixed-capacity body is in `apply_firth_variant_major_fixed_capacity_corrections`;
- `apply_device_candidate_corrections_firth_variant_major_with_device_dispatch`
  takes static `bounded_candidate_capacity` and `overflow_candidate_capacity`;
- `jax.lax.cond` handles zero, bounded, and overflow branches;
- `count_firth_candidates_on_host` and host capacity selection helpers were
  removed from the runtime path;
- `firth_candidate_count_host_sync` was replaced with the non-blocking
  `firth_candidate_dispatch_plan` timing label.

Expected files:

- `src/g/compute/regenie2_binary/variant_major_correction.py`
- `src/g/compute/regenie2_binary/candidates.py`
- `tests/test_regenie2_binary.py`
- `tests/test_regenie2_pipeline.py`

Lower-risk alternative: always use full-chunk capacity. This is simpler and
correct, but likely wastes memory and preparation work on sparse fallback chunks.

Higher-risk alternative: bounded-first correction with a later overflow repair
pass. This can be fast when overflows are rare, but it complicates result
contracts and can easily produce partial corrected output if mishandled.

Remaining correctness and performance risks:

- silent candidate truncation if bounded capacity is used when
  `fallback_count > bounded_candidate_capacity`;
- branch output PyTree or dtype mismatches inside `jax.lax.cond`;
- larger compilation cost because bounded and overflow branches are in one
  executable.

Acceptance checks:

- `rg count_firth_candidates_on_host src/g tests` finds no runtime use and no
  active test expectation for host dispatch;
- `rg firth_candidate_count_host_sync src/g tests` finds no active timing
  expectation;
- zero-candidate, bounded-capacity, and overflow-capacity paths return matching
  result PyTrees and dtypes;
- stage-timing tests expect `firth_candidate_dispatch_plan`;
- `uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q`
  and `uv run ty check src/g tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py`
  pass.

## Task 3: Batch Multi-Binary Approximate Firth And Reuse One Score Result

Status: implemented in `opt/multi-binary-firth-batching`.

Implemented direction: flatten trait-variant candidates into one lane axis.

The corrected multi-binary path was replaced rather than wrapped: it computes
the multi-trait score result once and applies correction only to selected
trait-variant lanes.

New candidate shape contract:

- score result arrays stay trait-major: `[traits, variants]`;
- candidate mask is `[traits, variants]`;
- flattened lane id is `trait_index * variant_count + variant_index`;
- candidate genotype lanes are `[lanes, samples]`;
- lane-specific phenotype, offset, null-state, and coefficient inputs are
  gathered by trait index;
- corrected output scatters back to `[traits, variants]`.

Implementation status:

- added a multi-shaped empty Firth diagnostics helper for
  `Regenie2MultiBinaryScoreChunkResult`;
- added `apply_device_candidate_corrections_multi_firth_variant_major`;
- changed the multi-binary approximate Firth path to compute one batched multi
  score result first, expand it with multi-shaped empty Firth diagnostics, and
  call the multi correction path;
- added candidate-planning helpers that preserve `trait_index`, `variant_index`,
  and flattened lane id through sorting/grouping;
- added Firth batch helpers that handle lane-specific phenotype vectors,
  null offsets, null coefficients, covariate weights, and initial coefficients;
- added a multi-result merge helper that scatters by `(trait_index, variant_index)`
  and preserves current beta sign restoration, `firth_se`, invalid candidate
  handling, and diagnostic columns;
- removed production reliance on `compute_one_trait(...)` and
  `stack_binary_chunk_results(...)` for multi-binary approximate Firth.

Expected files:

- `src/g/compute/regenie2_binary/api.py`
- `src/g/compute/regenie2_binary/variant_major_correction.py`
- `src/g/compute/regenie2_binary/candidates.py`
- `src/g/compute/regenie2_binary/firth/batch.py`
- `src/g/compute/regenie2_binary/correction.py`
- `src/g/compute/regenie2_binary/result.py`
- `tests/test_regenie2_binary.py`
- `tests/test_regenie2_pipeline.py`

Capacity contract:

- preserve current per-trait capacity semantics without config changes by using
  `preferred_candidate_capacity * trait_count`, capped at
  `trait_count * variant_count`;
- overflow capacity is `trait_count * variant_count`;
- if Task 2 has landed, reuse the device-side bounded/overflow dispatcher.

Remaining performance risks:

- full overflow capacity can become `traits * variants`, so larger trait-count
  sweeps may still be useful before deeper tuning work;
- both bounded and overflow branches are part of one jitted dispatcher, so peak
  memory and compile behavior should be watched in larger GPU runs.
- current two-trait chr10/chr22 50k sweeps support the defaults
  `firth-batch-size = 1024` and `firth-candidate-capacity = 2048`: batch 1024
  was fastest on both chromosomes, batch 2048 regressed, and capacities above
  2048 did not improve hot runtime enough to justify the larger fixed shapes.

Acceptance checks:

- done: multi-binary approximate Firth equals stacked single-trait results for
  different per-trait candidate masks;
- done: the multi-Firth corrected path invokes `compute_multi_binary_score_test_chunk_variant_major`
  once per chunk and does not run one single-trait score kernel per trait;
- done: packed8 multi-binary approximate Firth matches decoded variant-major dosage;
- done: sparse candidate masks remain variant-only modifiers and never create new
  Firth candidates;
- done: null Firth or null logistic failure in one trait does not poison other traits.

Checks run after rebasing onto `main`:

```bash
uv run pytest tests/test_regenie2_binary.py -q -k 'multi_trait_approximate_firth_uses_one_multi_score_dispatch or multi_trait_approximate_firth_variant_major_handles_distinct_candidate_masks or packed8_multi_trait_approximate_firth_matches_variant_major_dosage or multi_trait_sparse_candidate_mask_does_not_create_firth_candidates or multi_trait_null_firth_failure_does_not_poison_other_traits or multi_trait_null_logistic_failure_does_not_poison_other_traits'
uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q
uv run ty check src/g tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py
uv run ruff check src/g/compute/regenie2_binary tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py --output-format=concise
git diff --check
```

Results: 6 focused tests passed; 118 broader tests passed with 1 skipped; type,
lint, and whitespace checks passed. The only warning was the known JAX packed8
buffer-donation warning.

## Parallelization

Task 1 prep coverage, Task 2 device-side dispatch, Task 3 multi-binary batching,
benchmark harness extension, and focused GPU validation have landed on `main`.
Remaining Firth work is only follow-up tuning suggested by future larger
benchmark sweeps.

- no runtime path may call `count_firth_candidates_on_host`;
- capacity selection is device-side bounded/overflow dispatch;
- multi-binary capacity is based on flattened trait-variant lanes;
- single-trait and multi-trait correction paths must share the same overflow
  semantics and failure labels;
- sparse masks are modifiers for already-selected Firth lanes, not candidate
  selectors.

## Benchmark Plan

Do not run GPU workloads on the head node. Start with a smoke run on the GPU
node:

```bash
just slurm-gpu-just benchmark-regenie2-binary-hot-gpu-smoke
```

Smoke status: passed on `landau` with summary at
`data/profiles/regenie2_binary_hot_20260606T182522Z/regenie2_binary_hot_summary.json`.
The hot no-final timing was 0.38s for the 1,000-variant smoke slice; stage
timings included `firth_candidate_dispatch_plan` and no host-sync timing.

Then run the full binary hot benchmark:

```bash
just slurm-benchmark-regenie2-binary-hot-gpu
```

Full default status: passed on `landau` with summary at
`data/profiles/regenie2_binary_hot_20260606T183223Z/regenie2_binary_hot_summary.json`.
Headline timings were 54.39s cold finalized, 3.88s hot no-final, and 3.93s
hot finalized.

Targeted multi-trait status: a two-trait, high-fallback smoke over variant-major
and packed8 storage passed on `landau` with summary at
`data/profiles/regenie2_binary_hot_20260606T184618Z/regenie2_binary_hot_summary.json`.
The summary aggregated 2,000 output rows per two-trait case, and stage timings
included `firth_candidate_dispatch_plan` with no host-sync timing.

The benchmark workload now supports the multi-binary batching cases needed
for deeper tuning sweeps:

- 1, 2, 4, and 8 binary traits;
- variant-major dosage and packed8 inputs;
- low fallback density and high fallback density;
- multiple `--g-firth-batch-size` and `--g-firth-candidate-capacity` values;
- alternate `--bgen`, `--sample`, and expected variant count inputs, so chr10
  and chr22 can be tuned with the same harness;
- stage timing JSON enabled.

Two-trait tuning evidence from 2026-06-07:

| Input | Variants | Storage | Batch | Capacity | Hot seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| chr10 | 50,000 | variant-major | 32 | 1024 | 10.31 |
| chr10 | 50,000 | variant-major | 64 | 1024 | 7.24 |
| chr10 | 50,000 | packed8 | 32 | 1024 | 10.14 |
| chr10 | 50,000 | packed8 | 64 | 1024 | 7.15 |
| chr22 | 50,000 | variant-major | 32 | 1024 | 9.97 |
| chr22 | 50,000 | variant-major | 64 | 1024 | 6.79 |
| chr22 | 50,000 | packed8 | 32 | 1024 | 9.87 |
| chr22 | 50,000 | packed8 | 64 | 1024 | 6.69 |

The same matrices also compared capacity 512. Capacity 512 was close in hot
runtime but gives less headroom under high fallback density, so the existing
capacity 1024 default remains the better conservative default.

Primary metrics:

- total wall time;
- `jax_compute`;
- confirm there is no remaining Firth candidate host-sync timing;
- total duration of `firth_candidate_dispatch_plan` after Task 2 lands;
- binary chunk diagnostics: score candidates, Firth candidates, failures,
  correction branches;
- corrected-path score dispatch count if available from logs or a temporary
  benchmark-only counter;
- peak memory if available from the job environment.

Implemented benchmark extension:

- added multi-binary trait-count sweeps over supplied phenotype columns;
- exposed Firth candidate capacity and batch size sweeps;
- included selected trait count, storage mode, fallback density, Firth batch
  size, candidate capacity, stage timing path, and binary chunk summary in the
  JSON report.
