# Codebase Review Findings

Date: 2026-06-06
Reviewed base: `7ec2d10f` (`main` before cleanup integration)

This is a current-state review of active application code after the config default cleanup,
output, packed8, BGEN orchestration, REGENIE2 lifecycle, and easy non-config cleanup work.
It replaces the older review notes; items that were fixed are summarized near the end
instead of kept as historical work logs.

## Checks Run

- `uv run ruff check src/g tests --output-format=concise`: passed.
- `uv run ty check src/g`: passed.
- `uv run pytest tests/test_interface.py tests/test_api.py tests/test_io_output.py tests/test_regenie2_pipeline.py tests/test_regenie2_binary.py tests/test_regenie2_binary_config.py tests/test_jax_setup.py tests/test_warm_cache.py -q`: 320 passed, 1 skipped.
- `cargo test --lib output::`: 29 passed.
- `cargo test --lib genotype::preprocess`: 6 passed.
- `cargo test --lib genotype::bgen::trusted`: 5 passed.
- `cargo test --lib genotype::bgen::decode`: 7 passed.
- `cargo test --lib genotype::bgen::simd`: 7 passed.

I also fetched `origin` and confirmed the reviewed base matched `origin/main` before the
cleanup integration. I did not run GPU benchmarks or the full test suite on the head node.

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

### P2. Multi-binary approximate Firth is not batched

Multi-binary score-only execution is batched, but approximate Firth still falls back to
one-trait-at-a-time Python dispatch. The code documents this at
`src/g/compute/regenie2_binary/api.py:306` to `src/g/compute/regenie2_binary/api.py:307`,
then builds `compute_one_trait` at `src/g/compute/regenie2_binary/api.py:321` and stacks a
Python list at `src/g/compute/regenie2_binary/api.py:343`. Packed8 non-score multi-binary
first decodes to dosage and then calls the same per-trait correction path at
`src/g/compute/regenie2_binary/api.py:370` to `src/g/compute/regenie2_binary/api.py:374`.

Why this matters: packed8 and multi-phenotype speedups mostly help score-only execution.
When approximate Firth is enabled, the pipeline can still pay per-trait launch and Python
overhead on the fallback subset.

Suggested direction: keep this as a known performance limitation until score-only packed8
and output are stable. Then batch candidate extraction and correction over trait dimension,
or split the fallback workload into a clearly separate optimized kernel path.

Detailed implementation plan: `docs/firth-optimization-plan.md`.

### P2. Firth candidate capacity selection syncs to host

`apply_device_candidate_corrections_firth_variant_major` builds `candidate_mask` on device
at `src/g/compute/regenie2_binary/variant_major_correction.py:95`, then calls
`count_firth_candidates_on_host` at `src/g/compute/regenie2_binary/variant_major_correction.py:97`.
The timing label explicitly records `firth_candidate_count_host_sync` at
`src/g/compute/regenie2_binary/variant_major_correction.py:99`.

Why this matters: every chunk with approximate Firth enabled can introduce a host/device
synchronization point before correction. That is often more expensive than the count itself.

Suggested direction: use a fixed bounded capacity, a JAX-side count with a compiled branch,
or an overflow strategy that avoids forcing the host to observe every chunk before launching
the correction.

Detailed implementation plan: `docs/firth-optimization-plan.md`.

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

### P2. Callback worker threads start during construction

`NativeBgenCallbackRunner.__init__` creates worker threads at
`src/g/engine/callbacks.py:397` and `src/g/engine/callbacks.py:402`, then starts them at
`src/g/engine/callbacks.py:407` and `src/g/engine/callbacks.py:408`. The abstract compute
hooks are declared below at `src/g/engine/callbacks.py:420`,
`src/g/engine/callbacks.py:431`, and `src/g/engine/callbacks.py:442`.

The ABC prevents direct instantiation of incomplete subclasses, so this is not an immediate
abstract-method bug. It is still a lifecycle smell: thread startup is coupled to object
construction before the owner has a chance to finish external setup, register it, or enter a
managed context.

Suggested direction: add an explicit `start()` method or context manager and have pipeline
runners start callbacks after writer sessions and telemetry are fully prepared.

### P2. Native unsafe pointer writes are spread across several wrappers

Status: first bounded Rust patch landed in `cleanup/review-easy-nonconfig`. The trusted
no-missing variant-major dosage and packed8 paths now use a local `VariantMajorOutputMatrix`
helper to validate non-null output pointers, centralize row-offset overflow checks, and
reconstitute typed mutable row slices in one place. The hot decode loops still write through
ordinary slices and do not add per-element abstraction overhead.

The BGEN reader validates buffer lengths and then reconstitutes Python-owned memory with
raw pointers, for example `src/genotype/bgen/reader.rs:298` to
`src/genotype/bgen/reader.rs:300`. Trusted decode workers write variant rows through raw
pointer-derived slices at `src/genotype/bgen/trusted.rs:336`,
`src/genotype/bgen/trusted.rs:480`, `src/genotype/bgen/trusted.rs:503`, and
`src/genotype/bgen/trusted.rs:536`. Output uses a Python-owned Arrow buffer wrapper at
`src/python/output.rs:348`.

This is expected at the Python/Rust zero-copy boundary, and the surrounding code has useful
shape checks. The risk is that the safety contract is implicit and duplicated: buffer
lifetime, alignment, exclusivity, and row partitioning assumptions are spread across
multiple call sites.

Suggested direction: continue centralizing the unsafe boundary behind small helper types.
Next candidates are the generic BGEN row-major preprocessing slice in
`src/genotype/bgen/reader.rs`, lower-level generic decode output slices, and the Python-owned
Arrow buffer wrapper in `src/python/output.rs`.

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

### P3. The repository is still dominated by tracked archive files

Current tracked file counts:

- Total tracked files: 18,428.
- Tracked files under `archive/**`: 18,240.

Why this matters: code search, reviews, and agent context gathering all have to filter a
large historical tree that is not part of the active app. This has already produced noisy
static searches.

Suggested direction: move archive history out of the active repository, convert it to an
external branch/tag/artifact, or add very explicit tooling/docs so reviewers do not scan it
by default.

Recommended staged plan:

1. Add a root `.ignore` entry for `/archive/` and an `archive/README.md` that says the
   archive is historical, not active app code. This is non-destructive and removes default
   `rg`/`fd` noise.
2. Preserve the archive on a dedicated branch/tag before removal, for example with
   `git subtree split --prefix=archive/direct_association -b archive/direct-association`
   plus a dated archive tag.
3. Remove `archive/direct_association` from active `main` with a normal commit, leaving the
   README/index. This preserves history without rewriting every clone.
4. Avoid destructive `git filter-repo` history rewrites unless clone size becomes a real
   problem and all branch/worktree users coordinate.

## Test And Benchmark Gaps

- Packed8 multi-phenotype dispatch has interface and pipeline coverage, but I did not run a
  GPU benchmark or full parity workload in this review.
- Approximate Firth multi-trait performance should be benchmarked separately from score-only
  packed8 because it still uses the per-trait correction path.
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

## Suggested Implementation Order

1. Split callback start from construction.
2. Continue centralizing Rust unsafe buffer boundary helpers.
3. Apply non-destructive archive search/documentation hygiene, then decide whether to move
   archive snapshots to a dedicated branch/tag.
4. Benchmark and then redesign multi-binary approximate Firth batching.
5. Revisit public output dtype only if users need float64 result files.
