# Codebase Review Findings

Date: 2026-06-06
Reviewed commit: `01c8db0a` (`main`)

This is a current-state review of active application code after the config default cleanup,
output, and packed8 work. It replaces the older review notes; items that were fixed are
summarized near the end instead of kept as historical work logs.

## Checks Run

- `uv run ruff check src/g tests --output-format=concise`: passed.
- `uv run ty check src/g`: passed.
- `uv run pytest tests/test_interface.py tests/test_api.py tests/test_io_output.py tests/test_regenie2_pipeline.py tests/test_regenie2_binary.py tests/test_regenie2_binary_config.py tests/test_jax_setup.py tests/test_warm_cache.py -q`: 320 passed, 1 skipped.
- `cargo test --lib output::`: 29 passed.
- `cargo test --lib genotype::preprocess`: 6 passed.
- `cargo test --lib genotype::bgen::trusted`: 5 passed.
- `cargo test --lib genotype::bgen::decode`: 7 passed.
- `cargo test --lib genotype::bgen::simd`: 7 passed.

I also fetched `origin` and confirmed `HEAD` matches `origin/main`. I did not run GPU
benchmarks or the full test suite on the head node.

## Severity Guide

- P1: user-visible correctness risk or misleading public behavior.
- P2: performance, maintainability, or safety risk that can become a bug as code changes.
- P3: cleanup, hygiene, or low-probability edge case.

## Findings

### P1. Float64 compute/output knobs are not end-to-end

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

Why this matters: users can choose float64 score/Firth internals, but inputs and outputs do
not preserve a full float64 contract. This is fine as a deliberate performance policy, but
the current naming and manifest fields imply more precision control than the pipeline
actually provides.

Suggested direction: decide the contract explicitly. Either rename/document these as
internal compute dtype knobs with float32 output, or add an output/result dtype option and
plumb selected input dtypes where parity requires them.

### P2. Pipeline lifecycle code is duplicated across too many entry points

The main pipeline file now contains separate single linear, single binary, shared
multi-phenotype, grouped per-phenotype, prepared-group, and multi-callback runners:

- `src/g/engine/regenie2_pipeline.py:32`
- `src/g/engine/regenie2_pipeline.py:237`
- `src/g/engine/regenie2_pipeline.py:630`
- `src/g/engine/regenie2_pipeline.py:821`
- `src/g/engine/regenie2_pipeline.py:965`
- `src/g/engine/regenie2_pipeline.py:1169`

These paths repeat output initialization, writer creation, preflight, telemetry, callback
construction, engine delivery, shutdown, and manifest handling. The recent packed8 and
config-default cleanup moved more options into explicit runtime arguments, which is cleaner
but increases the amount of state that has to stay synchronized across every path.

Why this matters: this is the biggest architecture risk in the Python engine layer. New
flags such as output format, packed8, dtype policy, resume mode, and timing can easily be
wired into one path but missed in another.

Suggested direction: extract a typed run context and shared lifecycle runner. Keep
association-specific parts behind small strategy objects or callbacks, but make output,
telemetry, resume, and engine-delivery policy one implementation.

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

### P2. Multi-phenotype resume recomputes partially committed chunks

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

Suggested direction: centralize the unsafe boundary behind small helper types, document each
invariant once, and keep the `unsafe` blocks as small as possible.

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

### P2. INFO score denominator contract remains undecided

The shared preprocessing builder now rejects selected sample counts that do not fit i32 at
`src/genotype/preprocess.rs:326` to `src/genotype/preprocess.rs:330`. For variants with
missing calls, it imputes missing dosage square mass using the selected sample count at
`src/genotype/preprocess.rs:354` to `src/genotype/preprocess.rs:356`, but computes expected
variance using observed call count at `src/genotype/preprocess.rs:360`.

The behavior is characterized by tests and may be intentional, but the statistical contract
is not obvious from the implementation.

Suggested direction: compare against the intended REGENIE/BGEN INFO definition and write a
short comment plus a named test case describing the selected-sample versus observed-sample
choice.

### P3. Some trusted sample-count conversions still saturate

Most shared preprocessing now errors on sample counts outside i32 range, but a few trusted
decode paths still saturate to `i32::MAX` or rely on an `expect`:

- `src/genotype/bgen/trusted.rs:383`
- `src/genotype/bgen/trusted.rs:473`
- `src/genotype/bgen/simd.rs:421` to `src/genotype/bgen/simd.rs:422`

This is probably unreachable in practical memory limits, but it is inconsistent with the
new explicit error behavior.

Suggested direction: replace these with shared checked conversion helpers. This is a small
non-destructive cleanup.

### P3. Config conversion fallback still raises `NotImplementedError`

The public config path now rejects exact Firth before execution, which closes the old
public-option placeholder. One small config cleanup remains: the msgspec conversion hook
only handles `Path` and raises `NotImplementedError` for any future custom type at
`src/g/interface/config.py:398` to `src/g/interface/config.py:402`.

This is not currently user-visible because the active runtime dataclasses only need the
`Path` hook. It is still a poor failure mode for future config fields: users would see an
implementation placeholder instead of a typed validation error.

Suggested direction: raise `TypeError` or `ValueError` with the target type in the message,
and add a tiny unit test around the fallback.

### P3. Demo native binding is still exported

The production extension still exposes a demo function: `hello_from_bin` is defined at
`src/python/mod.rs:1193`, registered at `src/python/mod.rs:1340`, and declared in
`src/g/_core.pyi:237`.

This is harmless but looks like scaffold code in the public native API.

Suggested direction: remove the binding and its tests unless it is intentionally serving as
a smoke-test API.

### P3. Some architecture docs are stale after the config cleanup

The completed config rewrite left old planning documents that still describe the former
state as current. For example, `docs/01.rewrite_configuration.md:1` says `config.py` still
defines many `DEFAULT_*` values, which is no longer true after `src/g/interface/defaults.py`
was added. `docs/msgspec.md` has similar historical rewrite-plan language.
`docs/configuration_cli_architecture.md:491` still allows `PACKAGED_*` views at subsystem
boundaries, while the active code moved away from that pattern.

Why this matters: these docs are now easy to misread as active architecture guidance. That
can cause future workers to reintroduce defaults or plan work that already landed.

Suggested direction: either delete/archive completed rewrite plans or add a clear
"historical plan, already implemented" header. Update the architecture doc to say resolved
runtime values should be passed explicitly rather than exposed as subsystem `PACKAGED_*`
constants.

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

## Test And Benchmark Gaps

- Packed8 multi-phenotype dispatch has interface and pipeline coverage, but I did not run a
  GPU benchmark or full parity workload in this review.
- Approximate Firth multi-trait performance should be benchmarked separately from score-only
  packed8 because it still uses the per-trait correction path.
- Resume tests should include multi-phenotype partial-commit cases where each phenotype has
  a different committed chunk set.
- Config-default cleanup has good focused coverage, but the old rewrite-plan docs still need
  documentation hygiene.

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

## Suggested Implementation Order

1. Remove or intentionally rename `hello_from_bin`.
2. Replace the config conversion fallback `NotImplementedError` with a typed validation
   error.
3. Replace residual trusted-path sample-count saturation with checked helpers.
4. Mark completed config rewrite docs as historical or remove them.
5. Document and test the INFO score denominator contract.
6. Add a targeted multi-phenotype resume test for partial committed chunk sets.
7. Split callback start from construction.
8. Refactor pipeline lifecycle duplication around a shared typed run context.
9. Centralize Rust unsafe buffer boundary helpers.
10. Benchmark and then redesign multi-binary approximate Firth batching.
11. Decide the public dtype/output precision contract.
