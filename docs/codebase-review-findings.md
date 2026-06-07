# Codebase Review Findings

Date: 2026-06-06
Reviewed base: `7ec2d10f` (`main` before cleanup integration)

This is a current-state review of active application code after the config default cleanup,
output, packed8, BGEN orchestration, REGENIE2 lifecycle, easy non-config cleanup, and
orchestrated review cleanup work. It replaces the older review notes; items that were fixed
are summarized near the end instead of kept as historical work logs.

## Checks Run

- `uv run ruff check src/g tests --output-format=concise`: passed.
- `uv run ty check src/g`: passed.
- `uv run pytest tests/test_interface.py tests/test_api.py tests/test_io_output.py tests/test_regenie2_pipeline.py tests/test_regenie2_binary.py tests/test_regenie2_binary_config.py tests/test_jax_setup.py tests/test_warm_cache.py -q`: 320 passed, 1 skipped.
- `cargo test --lib output::`: 29 passed.
- `cargo test --lib genotype::preprocess`: 6 passed.
- `cargo test --lib genotype::bgen::trusted`: 5 passed.
- `cargo test --lib genotype::bgen::decode`: 7 passed.
- `cargo test --lib genotype::bgen::simd`: 7 passed.
- `cargo fmt --check`: passed after row-major BGEN buffer cleanup.
- `. scripts/server_env.sh && cargo test --lib genotype::bgen`: 25 passed after
  row-major BGEN buffer cleanup.
- `uv run pytest tests/test_regenie2_binary.py -q`: 44 passed, 1 skipped after
  preparatory Firth correctness tests.
- `uv run ty check src/g tests/test_regenie2_binary.py`: passed after preparatory
  Firth correctness tests.
- `uv run ruff check tests/test_regenie2_binary.py --output-format=concise`: passed
  after preparatory Firth correctness tests.
- `. scripts/server_env.sh && cargo test --lib output::`: 31 passed after
  Python-owned Arrow buffer cleanup.
- `. scripts/server_env.sh && cargo test --lib python::output`: 2 passed after
  Python-owned Arrow buffer cleanup.
- `uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q`:
  111 passed, 1 skipped after device-side Firth capacity dispatch.
- `uv run ty check src/g tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py`:
  passed after device-side Firth capacity dispatch.
- `uv run ruff check src/g/compute/regenie2_binary tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py --output-format=concise`:
  passed after device-side Firth capacity dispatch.
- `git diff --check`: passed after device-side Firth capacity dispatch.
- `uv run pytest tests/test_regenie_comparison_scripts.py -q -k 'binary_hot'`: 4 passed
  after binary hot benchmark sweep expansion.
- `uv run ruff check scripts/benchmark_regenie2_binary_hot.py tests/test_regenie_comparison_scripts.py --output-format=concise`:
  passed after binary hot benchmark sweep expansion.
- `uv run ty check scripts/benchmark_regenie2_binary_hot.py tests/test_regenie_comparison_scripts.py`:
  passed after binary hot benchmark sweep expansion.
- `cargo fmt --check`: passed after BGEN decode output boundary cleanup.
- `. scripts/server_env.sh && cargo test --lib genotype::bgen`: 27 passed after
  BGEN decode output boundary cleanup.
- `git diff --check`: passed after BGEN decode output boundary cleanup.
- `uv run pytest tests/test_regenie2_binary.py -q -k 'multi_trait_approximate_firth_uses_one_multi_score_dispatch or multi_trait_approximate_firth_variant_major_handles_distinct_candidate_masks or packed8_multi_trait_approximate_firth_matches_variant_major_dosage or multi_trait_sparse_candidate_mask_does_not_create_firth_candidates or multi_trait_null_firth_failure_does_not_poison_other_traits or multi_trait_null_logistic_failure_does_not_poison_other_traits'`:
  6 passed after multi-binary approximate Firth batching.
- `uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q`:
  118 passed, 1 skipped after multi-binary approximate Firth batching.
- `uv run ty check src/g tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py`:
  passed after multi-binary approximate Firth batching.
- `uv run ruff check src/g/compute/regenie2_binary tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py --output-format=concise`:
  passed after multi-binary approximate Firth batching.
- `git diff --check`: passed after multi-binary approximate Firth batching.
- `uv run pytest tests/test_regenie_comparison_scripts.py -q -k 'binary_hot'`: 4 passed
  after the binary hot benchmark telemetry fix.
- `uv run pytest tests/test_regenie_comparison_scripts.py -q -k 'binary_hot'`: 5 passed
  after multi-trait benchmark output metric aggregation.
- `uv run ruff check scripts/benchmark_regenie2_binary_hot.py tests/test_regenie_comparison_scripts.py --output-format=concise`:
  passed after the binary hot benchmark telemetry and metric aggregation fixes.
- `uv run ty check scripts/benchmark_regenie2_binary_hot.py tests/test_regenie_comparison_scripts.py`:
  passed after the binary hot benchmark telemetry and metric aggregation fixes.
- `XDG_RUNTIME_DIR=/tmp TMPDIR=/tmp just slurm-gpu-just benchmark-regenie2-binary-hot-gpu-smoke`:
  passed on `landau`; summary path
  `data/profiles/regenie2_binary_hot_20260606T182522Z/regenie2_binary_hot_summary.json`.
- `XDG_RUNTIME_DIR=/tmp TMPDIR=/tmp just slurm-benchmark-regenie2-binary-hot-gpu`:
  passed on `landau`; summary path
  `data/profiles/regenie2_binary_hot_20260606T183223Z/regenie2_binary_hot_summary.json`.
- Targeted two-trait GPU smoke with variant-major plus packed8, high fallback
  density, `--firth-batch-sizes 32`, and `--firth-candidate-capacities 128`:
  passed on `landau`; summary path
  `data/profiles/regenie2_binary_hot_20260606T184618Z/regenie2_binary_hot_summary.json`.
- `git diff --check`: passed after approved archive removal and patched
  REGENIE reference relocation.
- `uv run ruff check src/g tests --output-format=concise`: passed after
  approved archive removal and patched REGENIE reference relocation.
- `uv run ty check src/g`: passed after approved archive removal and patched
  REGENIE reference relocation.

I also fetched `origin` and confirmed the reviewed base matched `origin/main` before the
cleanup integration. The GPU smoke, default full binary-hot benchmark, and targeted
two-trait GPU smoke ran through SLURM on `landau`; I did not run the full test suite on the
head node.

## Severity Guide

- P1: user-visible correctness risk or misleading public behavior.
- P2: performance, maintainability, or safety risk that can become a bug as code changes.
- P3: cleanup, hygiene, or low-probability edge case.

## Findings

### P2. Float64 compute/output knobs are internal-only

Status: policy clarified in `cleanup/review-easy-nonconfig`. The current contract is
internal-compute-only dtype knobs with fixed float32 public result statistics. Config
comments, option help, and architecture docs now say `score-dtype` and `firth-dtype`
control internal JAX compute precision; Arrow/Parquet `BETA`, `SE`, `CHISQ`, and `LOG10P`
remain float32 under the current writer schema.

The new config advertises score and Firth dtypes in `src/g/config.default.toml:51` and
`src/g/config.default.toml:55`, and manifests record `score_dtype` and `firth_dtype` in
`src/g/io/output.py:305` and `src/g/io/output.py:306`. However, the native-aligned sample
views still force phenotypes and covariates to float32 in `src/g/engine/native_dispatch.py:107`,
`src/g/engine/native_dispatch.py:108`, `src/g/engine/native_dispatch.py:123`, and
`src/g/engine/native_dispatch.py:124`.

The output schema also always narrows public result statistics to float32:
`src/g/io/output.py:28`, `src/g/io/output.py:193`, and
`src/g/engine/callbacks.py:219`. Single- and multi-trait writes then apply the float32 cast
at `src/g/engine/callbacks.py:258`, `src/g/engine/callbacks.py:259`,
`src/g/engine/callbacks.py:260`, `src/g/engine/callbacks.py:261`,
`src/g/engine/callbacks.py:307`, `src/g/engine/callbacks.py:308`,
`src/g/engine/callbacks.py:309`, and `src/g/engine/callbacks.py:310`.

Why this matters: this is now explicit rather than misleading. Users who need float64
result files will need a separate output schema feature, not just `--g-score-dtype=float64`.

Suggested direction: defer end-to-end float64 or a separate output dtype until there is a
clear user requirement. That future work would need output schema/version, writer, manifest,
resume, and native dispatch changes.

### P2. Pipeline lifecycle code is duplicated across too many entry points

Status: addressed in `refactor/regenie2-lifecycle-runner`.

The pipeline now builds a shared typed lifecycle context and writer settings, then reuses
helpers for manifest initialization, writer creation, preflight, telemetry, callback drain,
engine delivery, interrupted finish, abort, and finalization. Single linear/binary,
complete-case multi-phenotype, and grouped per-phenotype paths share this lifecycle while
keeping association-specific compute in the callback classes.

The refactor preserves the current multi-phenotype resume contract: native engine delivery
skips only chunks committed by every phenotype, and each writer/callback still receives its
own committed chunk set.

Remaining risk: public entry points still forward a large set of CLI/API options by design,
and callback worker threads still start in callback constructors. Splitting callback
construction from thread startup remains a separate lifecycle cleanup.

### P2. Multi-binary approximate Firth batching

Status: addressed in `opt/multi-binary-firth-batching`.

The multi-binary approximate Firth path now computes one batched multi-trait
score result per chunk, builds a trait-major candidate mask, flattens selected
trait-variant lanes, and applies Firth correction through a device-side
zero/bounded/overflow dispatcher. Candidate lane identity keeps both
`trait_index` and `variant_index` through grouping so corrected statistics and
diagnostics scatter back to `[traits, variants]`.

The implementation preserves the single-trait correction path and uses
multi-specific helpers for capacity planning, sparse-mask handling, lane
residualization, lane-specific null state, and multi-result merge. Packed8
multi-binary approximate Firth still decodes to variant-major dosage before
entering the same multi correction path.

Focused GPU validation passed on `landau`: the default full binary-hot benchmark
and a targeted two-trait variant-major plus packed8 smoke both completed with
`firth_candidate_dispatch_plan` timings and no host-sync timing. Remaining
performance risk is limited to larger 1/2/4/8-trait tuning sweeps if future
optimization decisions need more data. Full overflow capacity is
`trait_count * variant_count`, and both bounded and overflow branches are part
of the jitted dispatcher.

### P2. Firth candidate capacity selection syncs to host

Status: addressed in `opt/firth-device-capacity-dispatch`.

`apply_device_candidate_corrections_firth_variant_major` now builds only static
chunk-capacity bounds on the host. Runtime `candidate_mask` and `fallback_count`
stay on device, and the jitted dispatcher uses `jax.lax.cond` to choose the
zero-candidate, bounded-capacity, or full-chunk overflow branch. The obsolete
`count_firth_candidates_on_host` helper and `firth_candidate_count_host_sync`
timing label were removed; the replacement `firth_candidate_dispatch_plan`
stage records only non-blocking host planning.

Why this matters: approximate Firth chunks no longer force the host to observe
the candidate count before launching correction.

Focused GPU validation covered the compiled dispatcher for the default
single-trait path and a targeted two-trait multi-binary path. Larger sweeps can
still measure peak memory and compile behavior before deeper tuning work.

Detailed follow-up plan: `docs/firth-optimization-plan.md`.

### P2. Multi-phenotype resume recomputes partially committed chunks

Status: current behavior is covered by multi-linear and multi-binary complete-case
pipeline tests in `tests/test_regenie2_pipeline.py`. The tests lock the
resume-fast-but-not-minimal contract: native delivery skips only chunks committed by every
phenotype, while the callback keeps each phenotype's committed set for duplicate-write
suppression.

The multi-phenotype path reads a committed chunk set for each phenotype at
`src/g/engine/regenie2_pipeline.py:1064` to `src/g/engine/regenie2_pipeline.py:1066`, but
engine delivery skips only chunks committed by every phenotype:
`src/g/engine/regenie2_pipeline.py:1134` to `src/g/engine/regenie2_pipeline.py:1142`.
The writer then filters per-trait writes using `active_trait_indices` in
`src/g/engine/callbacks.py:296` to `src/g/engine/callbacks.py:304`.

This preserves correctness, because already committed trait outputs are skipped, but it can
recompute chunks whose output is needed by only one phenotype. The waste grows with trait
count and interrupted long runs.

Suggested direction: for now document this as resume-fast-but-not-minimal. A later
optimization could group phenotypes by remaining chunk sets or deliver a sparse
trait/chunk schedule to the compute layer.

### P2. Callback worker startup lifecycle is now explicit

Status: addressed in `cleanup/review-easy-nonconfig`. `NativeBgenCallbackRunner.__init__`
now constructs queues and worker thread objects without starting them. `start()` owns
thread startup, and native dispatch calls it immediately before Rust engine delivery.
`finish()` and shutdown helpers tolerate callbacks that were constructed but never started.

Focused coverage lives in `tests/test_callback_lifecycle.py` and
`tests/test_regenie2_pipeline.py`. Remaining risk is ordinary lifecycle regression risk if
new callback owners bypass the native dispatch helpers; future callback types should either
use the same dispatch path or explicitly call `start()` before queuing work.

### P2. Native unsafe pointer writes are spread across several wrappers

Status: addressed. The first bounded Rust patch landed in
`cleanup/review-easy-nonconfig`: trusted no-missing variant-major dosage and packed8 paths
now use a local `VariantMajorOutputMatrix` helper to validate non-null output pointers,
centralize row-offset overflow checks, and reconstitute typed mutable row slices in one
place. The row-major BGEN preprocessing path now uses a local `RowMajorDosageBuffer`
helper in `src/genotype/bgen/reader.rs` to validate null/alignment boundaries and
centralize typed slice reconstruction before preprocessing. The Python-owned Arrow output
path now uses `PythonOwnedArrowValues` in `src/python/output.rs` to centralize typed slice
pointer validation, byte-length bookkeeping, and the retained Python-owner lifetime
contract. The hot decode loops still write through ordinary slices and do not add
per-element abstraction overhead.

The lower-level BGEN decode paths now share audited output helpers in
`src/genotype/bgen/decode.rs`: variant-major rows go through
`VariantMajorOutputMatrix`, row-major contiguous spans go through
`RowMajorOutputMatrix::row_range_mut`, and row-major column writes go through
`RowMajorOutputColumnMut`. Trusted decode paths import the same variant-major
helper instead of carrying a duplicate local definition.

This is expected at the Python/Rust zero-copy boundary, and the surrounding code has useful
shape checks. Remaining unsafe is the expected zero-copy contract: caller-owned allocation
lifetime, exclusivity, and disjoint Rayon ownership. SIMD internals remain separate
performance-specific unsafe code.

### P2. BGEN decode orchestration functions are too large to audit comfortably

Status: addressed in `refactor/bgen-reader-orchestration-helpers`. The public variant-major
dosage and packed8 reader methods now delegate selected count resolution, output-value
validation, empty-chunk stats, trusted packed8 precondition checks, profiling setup, stats
buffer allocation/reduction, and Rayon tile execution to named helpers in
`src/genotype/bgen/reader.rs`. The low-level generic, trusted, SIMD, and packed8 decode
kernels were not redesigned.

The variant-major dosage path starts at `src/genotype/bgen/reader.rs:304`; the packed8
probability-pair path starts at `src/genotype/bgen/reader.rs:445`. Both coordinate bounds
validation, row sizing, profiling, Rayon chunking, per-thread profiles, decode dispatch,
preprocessing, and stats reduction in long functions. The explicit
`#[allow(clippy::too_many_lines)]` markers are accurate.

Why this matters: this code is performance-critical and unsafe-adjacent. Long orchestration
functions make it hard to review future SIMD, packed format, or validation changes.

Suggested direction: split each path into validation, output-view construction,
parallel-decode planning, decode execution, and stats reduction helpers. Avoid abstracting
the hot inner loop until benchmarks show it is safe.

### P2. INFO score denominator contract remains a statistical decision

Status: characterized and documented in the easy cleanup branch. A Rust unit test locks the
current behavior, and the implementation now states that INFO is defined on observed
genotype calls. Missing calls are still mean-imputed for downstream dosage sums, but not for
the expected Hardy-Weinberg variance denominator.

The shared preprocessing builder rejects selected sample counts that do not fit i32 at
`src/genotype/preprocess.rs:326` to `src/genotype/preprocess.rs:330`. For variants with
missing calls, it imputes missing dosage square mass using the selected sample count at
`src/genotype/preprocess.rs:354` to `src/genotype/preprocess.rs:356`, but computes expected
variance using observed call count at `src/genotype/preprocess.rs:363`.

Suggested direction: only change the formula after comparing against the intended
REGENIE/BGEN INFO definition. The code now makes the current behavior explicit, but the
statistical policy remains a product/science decision.

### P3. The repository was dominated by tracked archive files

Status: addressed. Non-destructive search and documentation hygiene landed
first, the old archive was preserved on dedicated Git refs, and the approved
archive removal is now complete. Active `main` no longer tracks
`archive/**`. The retained patched REGENIE source moved to
`reference/regenie-patched` because it is an external parity reference rather
than historical `g` application code.

Current tracked file counts:

- Total tracked files: 862.
- Tracked files under `archive/**`: 0.
- Tracked files under `reference/regenie-patched/**`: 671.

Why this matters: code search, reviews, and agent context gathering all have to filter a
large historical tree that is not part of the active app. This has already produced noisy
static searches.

Resolution: preservation is done on branch
`preserve-direct-association-g-code-20260607` and tag
`archive-direct-association-g-code-20260607`. The approved
`archive/direct_association/src`, `archive/direct_association/tests`,
`archive/direct_association/scripts`, and upstream BGEN `release/` snapshot
now have no tracked files on active `main`. The patched REGENIE source remains
available at `reference/regenie-patched`. No archive history rewrite was
performed.

The removed `release/` snapshot was the upstream C++ BGEN reference
implementation, not active GWAS-engine code. It accounted for 17,551 tracked
files and about 190 MB on disk; 17,038 of those files were vendored
`boost_1_86_0`, with additional build artifacts and example BGEN data. Active
repo code did not reference it. The retained patched REGENIE source expects a
BGEN library through `BGEN_PATH` when built directly, but it did not point at
the sibling archive checkout and its Dockerfiles download BGEN themselves.

Recommended staged plan:

1. Done: initially add a root `.ignore` entry for `/archive/` and an
   `archive/README.md` that says the archive is historical, not active app
   code. These were removed after `archive/**` itself was removed from active
   `main`.
2. Done: preserve the archive on dedicated Git refs:
   `preserve-direct-association-g-code-20260607` and
   `archive-direct-association-g-code-20260607`.
3. Done: remove the approved archived GWAS-engine code scope from active
   `main` with a normal commit while keeping the patched REGENIE tree intact.
   This preserves history without rewriting every clone.
4. Done: remove the unrelated upstream BGEN `release/` snapshot from active
   `main` after confirming the app does not use it and the retained REGENIE
   tree does not directly reference that checkout.
5. Done: move patched REGENIE from `archive/direct_association/regenie` to
   `reference/regenie-patched` and document its external BGEN dependency.
6. Avoid destructive `git filter-repo` history rewrites unless clone size becomes a real
   problem and all branch/worktree users coordinate.

## Test And Benchmark Gaps

- Packed8 multi-phenotype dispatch has interface and pipeline coverage, and a
  targeted two-trait packed8 GPU smoke passed. I did not run an exhaustive
  packed8 parity workload across the full trait/storage/fallback matrix.
- Firth tests now cover zero-candidate diagnostic preservation, distinct
  per-trait multi-binary approximate Firth candidate masks, packed8
  multi-binary approximate Firth parity, sparse-mask non-expansion, one
  multi-score dispatch per corrected chunk, and per-trait null Firth/logistic
  failure isolation.
- The binary hot benchmark harness now supports multi-binary trait-count,
  Firth batch-size, Firth candidate-capacity, storage-mode, and fallback-density
  sweeps. The SLURM GPU smoke, default full binary-hot benchmark, and targeted
  two-trait variant-major plus packed8 smoke passed on `landau`.
- Multi-phenotype resume has focused multi-linear and multi-binary partial-commit coverage
  for the current resume-fast-but-not-minimal behavior; future resume-minimization work
  should add optimization-specific tests.
- Config-default cleanup has good focused coverage, but future config changes should keep
  the historical planning docs out of the active architecture path.

## Closed From Previous Review

These older findings appear resolved in the current code and should not be carried as open
work:

- Python explicit `None` option handling was fixed.
- Repeated Rayon configuration warnings were removed.
- Runtime callback `assert` checks were replaced with explicit runtime handling.
- Packed8 multi-phenotype dispatch is now wired through interface, runner, pipeline, and
  callback code.
- Old Python grouped-alignment helpers and legacy config helper surfaces were removed.
- Runtime unsupported correction branches were consolidated behind validation helpers.
- Exact Firth without `--approx` is now rejected during config validation at
  `src/g/interface/config.py:644` to `src/g/interface/config.py:646`.
- Buffer pool accounting drift was fixed.
- TOML array serialization was fixed.
- Stale manual unknown-option behavior was fixed.
- The scaffold `hello_from_bin` native binding, Python stub, and tests were removed.
- Trusted BGEN selected-sample counts now use explicit checked conversion before filling
  i32 statistics fields instead of saturating to `i32::MAX`.
- The INFO score denominator behavior is documented and covered by a named Rust regression
  test. The formula itself was not changed.
- The config conversion fallback finding no longer applies. Typed TOML conversion now lives
  in `src/g/interface/config_layers.py` and raises `ValueError` around msgspec validation
  failures instead of surfacing an implementation placeholder.
- Completed config rewrite planning docs now have historical headers, and the active
  architecture doc says runtime subsystems should receive resolved `RegenieConfig` or
  `ExecutionPlan` values rather than packaged default views.
- Multi-phenotype resume has targeted multi-linear and multi-binary pipeline regressions
  for partial per-phenotype committed chunk sets.
- Dtype docs now state the current contract: score/Firth dtype options control internal
  compute precision, while public association statistics remain float32.
- Callback worker startup now has an explicit lifecycle and focused tests.
- Non-destructive archive hygiene first kept `/archive/` out of default
  `rg`/`fd` searches and documented the tree as historical reference material;
  those temporary markers were removed when the archive tree was removed from
  active `main`.
- The row-major BGEN preprocessing boundary now uses `RowMajorDosageBuffer` to centralize
  null/alignment validation and typed mutable slice reconstruction.
- The Python-owned Arrow output boundary now uses `PythonOwnedArrowValues` to centralize
  pointer validation, buffer sizing, and the retained-owner safety contract.
- Lower-level BGEN decode output rows, row ranges, and row-major columns now go through
  shared helper types that centralize pointer validation, row-offset checks, and typed
  slice or column reconstruction.
- Preparatory Firth correctness tests now cover zero-candidate diagnostics and distinct
  per-trait multi-binary approximate Firth candidate masks.
- Single-trait approximate Firth now uses device-side zero, bounded, and overflow
  capacity dispatch instead of a host candidate-count synchronization.
- Multi-binary approximate Firth now batches flattened trait-variant candidate
  lanes while reusing one batched multi-score result per chunk.
- The binary hot benchmark harness now expands reproducible multi-binary
  approximate-Firth sweeps and records the per-case configuration in JSON output.
- The binary hot benchmark harness now disables run telemetry for same-process
  synthetic timing trials so process-global logging can be reused while
  per-trial stage timing JSON is still written.
- The binary hot benchmark harness now aggregates output row, INFO, chunk-count,
  and byte metrics across multi-trait `RunArtifacts`.
- Approved archive removal now leaves no tracked files under `archive/**` on
  active `main`; patched REGENIE moved to `reference/regenie-patched`.

## Suggested Implementation Order

1. Keep public output statistics float32 unless a future user requirement explicitly asks
   for float64 result files.
2. Run a larger 1/2/4/8-trait GPU benchmark matrix only if deeper Firth tuning
   decisions need more data.
