# Firth Optimization Plan

Date: 2026-06-06
Base worktree: `cleanup/review-easy-nonconfig`

This note expands the Firth-related findings in `docs/codebase-review-findings.md`.
It focuses on two performance items:

- removing the Firth candidate-count host synchronization;
- batching multi-binary approximate Firth instead of dispatching one trait at a time.

No GPU benchmark was run while writing this plan.

## Current State

Single-trait approximate Firth is device-side after a static candidate capacity is
chosen. `apply_device_candidate_corrections_firth_variant_major` builds a device
`candidate_mask`, then calls `count_firth_candidates_on_host`, which performs a
device sum followed by `jax.device_get`. The host count chooses either the
preferred bounded capacity or full-chunk overflow capacity, and it also skips the
Firth correction path when there are zero candidates.

The host sync exists because `build_device_firth_batch_plan` uses
`jnp.nonzero(..., size=candidate_capacity)`, fixed padding, reshapes, and a fixed
scan length. Those shapes require a Python/static `candidate_capacity`.

Multi-binary score-only execution is genuinely trait-batched. Multi-binary
approximate Firth is not: the non-score path in `regenie2_binary/api.py` slices a
single-trait chromosome state, recomputes that trait's score result, applies the
single-trait correction path, and stacks the Python list of per-trait results.
Packed8 multi-binary approximate Firth decodes the packed chunk once, then uses
the same per-trait correction path.

## Recommended Priority

1. Add preparatory correctness tests for the current Firth behavior.
2. Remove the single-trait host sync with device-side bounded/overflow dispatch.
3. Batch multi-binary approximate Firth over flattened trait-variant candidate lanes.
4. Run GPU benchmarks and decide whether the old per-trait path should remain as
   a temporary fallback.

The host-sync work should go first. It is smaller, has a clear correctness
contract, and gives the multi-binary batching work a dispatch primitive to reuse.
Multi-binary batching can proceed independently if needed, but it must not
reintroduce per-trait host counts.

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

Current prep coverage now strengthens the zero-candidate diagnostic contract and
adds a variant-major multi-binary approximate Firth parity test whose traits
have different score-stage Firth candidate masks. Existing tests already cover
bounded/overflow equivalence, sparse masks, single-binary packed8 approximate
Firth parity, and failed Firth candidate labeling.

Remaining gap: a lightweight passing null-Firth failure-isolation regression was
not added. Local inspection found that vectorized multi-trait null-Firth
preparation can still couple a failing trait with another trait in some small
fixtures, so this should be handled as a production correctness fix before
locking the desired isolation behavior in a passing test.

Expected files:

- `tests/test_regenie2_binary.py`
- `tests/test_regenie2_pipeline.py`

Suggested checks:

```bash
uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q
uv run ty check src/g tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py
```

## Task 2: Remove Candidate Count Host Sync

Recommended implementation: replace host capacity selection with a jitted
device-side bounded/overflow branch.

Shape contract:

- compute `candidate_mask` and `fallback_count` on device;
- return expanded empty Firth diagnostics when `fallback_count == 0`;
- use bounded capacity when `fallback_count <= bounded_candidate_capacity`;
- use full chunk capacity when `fallback_count > bounded_candidate_capacity`;
- both branches return the same `Regenie2BinaryChunkResult` shape.

Implementation outline:

- move the current fixed-capacity body into a non-public helper that can be
  called from branch functions;
- add a jitted dispatcher that takes static `bounded_candidate_capacity` and
  `overflow_candidate_capacity`;
- use `jax.lax.cond` for zero, bounded, and overflow branches;
- remove `count_firth_candidates_on_host` from the runtime path;
- remove or rename the `firth_candidate_count_host_sync` timing label, because
  there should no longer be a host-blocking count stage.

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

Correctness risks:

- silent candidate truncation if bounded capacity is used when
  `fallback_count > bounded_candidate_capacity`;
- branch output PyTree or dtype mismatches inside `jax.lax.cond`;
- larger compilation cost because bounded and overflow branches are in one
  executable;
- telemetry tests expecting the old host-sync label.

## Task 3: Batch Multi-Binary Approximate Firth

Recommended implementation: flatten trait-variant candidates into one lane axis.

New candidate shape contract:

- score result arrays stay trait-major: `[traits, variants]`;
- candidate mask is `[traits, variants]`;
- flattened lane id is `trait_index * variant_count + variant_index`;
- candidate genotype lanes are `[lanes, samples]`;
- lane-specific phenotype, offset, null-state, and coefficient inputs are
  gathered by trait index;
- corrected output scatters back to `[traits, variants]`.

Implementation outline:

- add `apply_device_candidate_corrections_multi_firth_variant_major`;
- change the multi-binary approximate Firth path to compute one batched multi
  score result first, expand it with multi-shaped empty Firth diagnostics, and
  call the multi correction path;
- add candidate-planning helpers that preserve `trait_index`, `variant_index`,
  and flattened lane id through sorting/grouping;
- add Firth batch helpers that handle lane-specific phenotype vectors,
  null offsets, null coefficients, covariate weights, and initial coefficients;
- add a multi-result merge helper that scatters by `(trait_index, variant_index)`
  and preserves current beta sign restoration, `firth_se`, invalid candidate
  handling, and diagnostic columns.

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

Correctness risks:

- candidate eligibility is per trait and per variant; do not OR candidate masks
  across traits;
- the sparse mask is still variant-only and must not create new Firth
  candidates;
- null failures are per trait and must not affect other traits;
- sorted candidate lanes must keep trait and variant indices attached;
- failed Firth lanes must become `TEST_FAIL` with NaN statistics and populated
  diagnostics;
- packed8 approximate Firth must continue to match decoded dosage semantics,
  including native `dosage_sum` and `observation_count` use in the score phase;
- full overflow capacity can become `traits * variants`, so memory must be
  measured before removing the old path entirely.

## Parallelization

Task 1 can be done by one agent immediately. Task 2 and Task 3 can be done by
separate agents only after agreeing on the dispatch and capacity contract:

- no runtime path may call `count_firth_candidates_on_host`;
- capacity selection is device-side bounded/overflow dispatch;
- multi-binary capacity is based on flattened trait-variant lanes;
- single-trait and multi-trait correction paths must share the same overflow
  semantics and failure labels;
- sparse masks are modifiers for already-selected Firth lanes, not candidate
  selectors.

If done in parallel, assign disjoint ownership:

- host-sync agent owns `variant_major_correction.py`, `candidates.py`, and
  host-sync tests;
- multi-binary agent owns the multi entry point and new multi helpers, but
  depends on the host-sync agent's dispatch API before final integration.

## Benchmark Plan

Do not run GPU workloads on the head node. Start with a smoke run on the GPU
node:

```bash
just slurm-gpu-just benchmark-regenie2-binary-hot-gpu-smoke
```

Then run the full binary hot benchmark:

```bash
just slurm-benchmark-regenie2-binary-hot-gpu
```

For multi-binary batching, extend the benchmark workload before making a final
performance call:

- 2, 4, and 8 binary traits;
- variant-major dosage and packed8 inputs;
- low fallback density and high fallback density;
- multiple `--g-firth-batch-size` values;
- stage timing JSON enabled.

Primary metrics:

- total wall time;
- `jax_compute`;
- number and total duration of any remaining Firth candidate host-sync timing;
- binary chunk diagnostics: score candidates, Firth candidates, failures,
  correction branches;
- peak memory if available from the job environment.

Recommended benchmark extension:

- add `--pheno-col-list` / multi-trait support to `scripts/benchmark_regenie2_binary_hot.py`;
- expose `--firth-candidate-capacity` so bounded/overflow behavior can be
  profiled directly.
