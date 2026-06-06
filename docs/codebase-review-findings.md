# Codebase Review Findings

Date: 2026-06-06
Commit reviewed: `a3616917`

Scope: active application code on `main`, with `archive/` treated only as a repository-hygiene concern. This was a static/source review plus lightweight checks, not a full GPU parity or benchmark pass.

Checks run:

- `uv run ruff check src/g --output-format=concise` - passed.
- `uv run ty check src/g` - passed.
- `cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic` - passed.
- A small Python config repro confirmed one bug described below.

Severity guide:

- P1: correctness, hangs, or misleading public behavior.
- P2: significant maintainability/performance risk.
- P3: cleanup, hygiene, or future-proofing.

## Branch Progress

Worktree: `/mnt/beegfs/kirill/Projects/g-worktrees/codebase-review-cleanups`
Branch: `codebase-review-cleanups`

Completed in this branch:

- Fixed the Python config `None` explicit-option bug.
- Added TOML array serialization for list/tuple values.
- Removed the stale manual `g-trusted-no-missing-diploid` unknown-option exception.
- Removed legacy config helpers that were superseded by the TOML layer system.
- Replaced runtime callback chromosome-state assertions with explicit invariant checks.
- Added warning-only Rayon thread pool configuration tracking.
- Documented and tested the current packed8 single-phenotype-only dispatch contract.
- Replaced Rust preprocessing sample-count saturation with an explicit range error.
- Hardened native host genotype buffer pool accounting across shape/dtype replacement.
- Removed old Python-owned grouped-alignment helpers from the production pipeline module.

Verification in this branch:

- `uv run pytest tests/test_interface.py -q` - 58 passed.
- `uv run ruff format --check src/g/interface/config.py src/g/interface/config_layers.py tests/test_interface.py` - passed.
- `uv run ruff check src/g/interface/config.py src/g/interface/config_layers.py tests/test_interface.py` - passed.
- `uv run ty check src/g/interface` - passed.
- `uv run pytest tests/test_regenie2_pipeline.py -q` - 54 passed after callback invariant cleanup.
- `uv run ruff check src/g/engine/callbacks.py tests/test_regenie2_pipeline.py` - passed.
- `uv run ty check src/g/engine/callbacks.py` - passed.
- `uv run pytest tests/test_api.py -q` - 27 passed after Rayon thread tracking cleanup.
- `uv run ruff check src/g/runner.py tests/test_api.py` - passed.
- `uv run ty check src/g/runner.py` - passed.
- `uv run pytest tests/test_interface.py tests/test_api.py -q` - 86 passed after packed8 contract cleanup.
- `uv run ruff check src/g/runner.py tests/test_interface.py` - passed.
- `uv run ty check src/g/runner.py src/g/interface` - passed.
- `rustfmt --check src/genotype/preprocess.rs src/genotype/bgen/reader.rs` - passed after Rust sample-count saturation cleanup.
- `env LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --lib genotype::preprocess` - 5 passed after Rust sample-count saturation cleanup.
- `uv run pytest tests/test_regenie2_pipeline.py -q` - 55 passed after buffer pool accounting cleanup.
- `uv run ruff check src/g/engine/callbacks.py tests/test_regenie2_pipeline.py` - passed after buffer pool accounting cleanup.
- `uv run ty check src/g/engine/callbacks.py` - passed after buffer pool accounting cleanup.
- `uv run pytest tests/test_regenie2_pipeline.py -q` - 55 passed after grouped-alignment helper cleanup.
- `uv run ruff format --check src/g/engine/regenie2_pipeline.py tests/test_regenie2_pipeline.py` - passed after grouped-alignment helper cleanup.
- `uv run ruff check src/g/engine/regenie2_pipeline.py tests/test_regenie2_pipeline.py` - passed after grouped-alignment helper cleanup.
- `uv run ty check src/g/engine/regenie2_pipeline.py` - passed after grouped-alignment helper cleanup.

Implementation learnings:

- `option_dictionary_to_toml_config_layer()` was already filtering `None` out of emitted TOML; only `explicit_options` needed the same filter.
- The snake-case Python alias path is normalized before explicit tracking, so `firth_se=None` now correctly maps to omitted `firth-se`.
- `g-trusted-no-missing-diploid` is already in the option registry, so it does not need a special case in unknown-option validation.
- Normal `RegenieConfig.to_toml()` currently emits comma-delimited column-list strings, but `format_toml_value()` is a general helper and now handles list/tuple values safely.
- The suspected legacy config helpers were genuinely production-dead. The only active references were test assertions that preserved the old helper surface.
- Callback chromosome state is guaranteed by control flow after `prepare_chromosome_state()`, but an explicit helper preserves the invariant under optimized Python and gives clearer failures if the lifecycle is broken.
- Rayon thread configuration is process-global. Warning-only handling preserves current repeated-run permissiveness while making ignored incompatible requests visible.
- The packed8/multi-phenotype contract is currently enforced entirely by config validation. Multi dispatch intentionally stays dosage-only until packed8 multi-trait execution exists end-to-end.
- The native stats schema is still `i32` for count-like fields. Failing at the shared stats builder is the narrowest fix until the schema moves to wider counters.
- Host genotype buffers need ownership tracking, not just queue membership, because result work items can carry any NumPy array-shaped value through the same release path in tests and fallback paths.
- The active grouped per-phenotype path already uses native grouped alignment; the removed Python-owned grouping code was only a test fixture builder.

## Suggested Work Order

1. Done: Fix the Python config `None` explicit-option bug.
2. Decide and enforce the dtype contract for float64 input, compute, and output.
3. Harden native callback shutdown, worker errors, and runtime invariants.
4. Collapse duplicated single/multi pipeline lifecycle code.
5. Remove or move stale production helpers that are now test-only.
6. Optimize multi-binary approximate Firth and resume behavior after correctness/lifecycle work.
7. Deal with repository hygiene around the tracked archive.

## Easy Fix Ranking

This branch will take the remaining review findings in order of implementation ease and non-destructiveness. Completed config cleanups are excluded from this ranking.

1. Done: Runtime callback `assert` invariants.
   Easy, highly non-destructive: replace optimized-away assertions with explicit `RuntimeError` checks.
2. Done: Rayon thread configuration feedback.
   Easy, non-destructive if warning-only: stop silently ignoring incompatible repeated thread settings.
3. Done: Packed8 multi-dispatch contract.
   Easy, highly non-destructive: make the current config rejection explicit in tests/docs so future packed8 work does not miss the multi path.
4. Done: Rust sample-count saturation.
   Easy-medium, non-destructive for realistic data: replace silent `i32::MAX` saturation with explicit error behavior.
5. Done: Buffer pool accounting drift.
   Medium, mostly local but concurrency-sensitive.
6. Done: Old Python grouped-alignment helpers.
   Medium, production-dead but test refactoring is needed.
7. Callback runner abstract contract.
   Medium, lifecycle-sensitive.
8. Unsupported exact Firth and SPA compute branches.
   Medium, changes lower-level behavior and tests currently assert failures.
9. Native callback bounded shutdown.
   Medium, important but concurrency-sensitive.
10. INFO score missingness contract test.
    Medium, safe as characterization-only; formula changes need a statistical decision.

## Findings

### P1. Python API treats `None` options as explicit user options

Status: done in `codebase-review-cleanups`. `explicit_options` now records only non-`None` normalized options, with regression coverage for binary-only Python options set to `None`.

Original finding: `config_layers.option_dictionary_to_toml_config_layer()` skipped `None` values while building the TOML layer, but recorded every normalized key in `explicit_options`:

- `src/g/interface/config_layers.py:257-273`

Later quantitative validation treats option names in `explicit_options` as user-provided binary-only flags:

- `src/g/interface/config.py:1049-1064`

Repro:

```bash
uv run python -c 'from g.interface import config
try:
    config.from_options({"qt": True, "bgen": "x.bgen", "phenoFile": "p", "pred": "r", "out": "o", "firth": None})
except Exception as error:
    print(type(error).__name__)
    print(error)'
```

Observed:

```text
ValueError
--firth can only be used with --bt; omit binary-only options when using --qt.
```

Impact: Python callers that pass optional keyword values with `None` can get false validation failures. This is especially likely in wrappers that forward a dictionary of optional arguments.

Recommendation: record only non-`None` values in `explicit_options`, plus explicitly handled trait flags when actually supplied. Add a regression for `firth=None`, `approx=None`, `firth-se=None`, `spa=None`, and `pThresh=None`.

### P1. Float64 compute knobs are not end-to-end

JAX x64 is globally enabled:

- `src/g/runtime_policy.py:5`
- `src/g/jax_setup.py:98`

The public config also exposes float64 policy:

- `src/g/config.default.toml:51-55`
- `src/g/io/output.py:271-304`

But native-aligned phenotype and covariate arrays are materialized as float32 before kernel dtype policy applies:

- `src/g/engine/native_dispatch.py:100-124`
- `src/g/engine/regenie2_pipeline.py:1007-1022`

The writer also intentionally narrows statistics to the public float32 schema:

- `src/g/engine/callbacks.py:216-218`
- `tests/test_regenie2_pipeline.py:138-147`

Impact: `score_dtype=float64` and `firth_dtype=float64` do not mean full end-to-end float64. Kernels can run in float64, but phenotype/covariate inputs have already lost precision at the native boundary. Output precision is also float32 by schema. This is misleading for parity/debugging workflows.

Recommendation: make a product decision:

- If the runtime is float32 I/O plus optional float64 internal accumulation, document that explicitly and consider renaming the knobs.
- If float64 parity is a real contract, carry phenotype, covariates, and prediction values to JAX in the requested dtype, and add tests at the native dispatch boundary.
- Decide whether result output should remain float32-only or gain an optional float64 output schema/version.

### P1. Rayon thread configuration can be silently ignored

Status: done in `codebase-review-cleanups`. The runner now tracks the configured Rayon thread count, skips identical repeats, and logs warnings for incompatible repeats or native configuration failures.

Original finding: `runner.configure_runtime()` suppressed `RuntimeError` from `configure_rayon_global_thread_pool()`:

- `src/g/runner.py:264-271`

The Rust side uses Rayon global thread pool initialization:

- `src/python/mod.rs:1311-1318`

Tests already show repeated global configuration raises `RuntimeError`:

- `tests/rust_python_bindings.rs:581-584`

Impact: a process that runs multiple jobs with different `--threads` values may silently continue using the first configured Rayon thread count. The user sees no warning even though the requested runtime setting was not applied.

Recommendation: track the configured thread count in Python and log or raise on incompatible subsequent values. At minimum, do not suppress the error silently.

### P1. Native callback finish can hang indefinitely

`NativeBgenCallbackRunner.finish()` joins the dosage worker with no timeout:

- `src/g/engine/callbacks.py:636-641`

The result worker has bounded stop/join logic:

- `src/g/engine/callbacks.py:658-684`

`abort()` uses non-blocking sentinel enqueue and suppresses `queue.Full`:

- `src/g/engine/callbacks.py:651-657`

Impact: a stuck JAX call, native callback, or full queue can turn into an indefinite process hang. Abort may also fail to deliver a sentinel if queues are full.

Recommendation: use a bounded join for the dosage worker too, make abort robust against full queues, and propagate `NativeBgenWorkerShutdownError` with enough context to diagnose the stalled stage/chunk.

### P1. Runtime invariants rely on `assert`

Status: done in `codebase-review-cleanups`. Runtime assertions were replaced with `require_current_chromosome_state()`, which returns the prepared state or raises `RuntimeError` with chromosome context.

Original finding: callback implementations used `assert self.current_chromosome_state is not None` in runtime paths:

- `src/g/engine/callbacks.py:876`
- `src/g/engine/callbacks.py:942`
- `src/g/engine/callbacks.py:993`
- `src/g/engine/callbacks.py:1082`
- `src/g/engine/callbacks.py:1120`
- `src/g/engine/callbacks.py:1346`
- `src/g/engine/callbacks.py:1379`
- `src/g/engine/callbacks.py:1442`
- `src/g/engine/callbacks.py:1574`
- `src/g/engine/callbacks.py:1621`

Impact: Python `assert` statements can be stripped with optimization flags. If that happens, a missing chromosome state becomes a later, less clear failure.

Recommendation: replace these with explicit invariant helpers that raise `RuntimeError` with the current chromosome and chunk metadata.

### P2. Pipeline lifecycle code is duplicated and diverging

The single linear and single binary pipelines repeat the same structure:

- `src/g/engine/regenie2_pipeline.py:27-225`
- `src/g/engine/regenie2_pipeline.py:228-436`

The multi path reimplements a similar callback/writer lifecycle instead of reusing `native_dispatch.run_bgen_engine_with_callback()`:

- `src/g/engine/native_dispatch.py:422-497`
- `src/g/engine/regenie2_pipeline.py:1243-1310`

Impact: lifecycle behavior, shutdown, timing, telemetry, packed8 support, and resume handling must be kept in sync across multiple long functions. This increases the chance of feature drift and makes new pipeline modes expensive.

Recommendation: introduce a shared pipeline runner that owns engine open, alignment, preflight, writer creation, callback drain, interrupted finish, abort, timing snapshots, and telemetry. Specialize only the small parts: state construction, callback type, correction plan, and writer shape.

### P2. Packed8 support is single-path only by validation and not threaded through multi dispatch

Status: done in `codebase-review-cleanups`. The canonical `phenoColList` multi-phenotype packed8 path is covered by validation tests, and multi dispatch documents that it intentionally remains dosage-only while config rejects packed8 multi-trait runs.

Single-phenotype dispatch passes `gpu_genotype_format`:

- `src/g/runner.py:494-516`

Multi-phenotype dispatch does not:

- `src/g/runner.py:526-564`

Config currently rejects `packed8` with multiple phenotypes:

- `src/g/interface/config.py:998-1004`

Impact: this is not a current runtime bug because validation blocks the unsupported combination. It is a fragile architecture boundary: future multi-packed8 work can silently miss a required dispatch argument.

Recommendation: either keep the validation and add a comment/test that multi dispatch intentionally omits packed8, or pass `gpu_genotype_format` through the multi runner and fail closer to the unsupported implementation.

### P2. Old Python grouped-alignment implementation remains in production module

Status: done in `codebase-review-cleanups`. The old Python-owned grouped alignment dataclasses/builders were removed from `regenie2_pipeline.py`; the only remaining need was moved into `tests/test_regenie2_pipeline.py` as local fixture construction.

Original finding: the active grouped per-phenotype path used native grouped alignment:

- `src/g/engine/regenie2_pipeline.py:831-925`
- `src/g/engine/native_dispatch.py:275-307`

But older Python-owned grouped helpers remained in the production pipeline module:

- `src/g/engine/regenie2_pipeline.py:459-500`
- `src/g/engine/regenie2_pipeline.py:965-1027`

Search results show these are not used by production dispatch. `build_grouped_native_bgen_multi_run_input()` is only used by tests.

Impact: dead or test-only production code increases review burden and can hide assumptions about sample/covariate grouping. The fingerprint helper also hashes sample indices and float32 covariates only, not full sample identity metadata.

Recommendation: move these helpers into tests if they are fixture builders, or delete them and update tests to exercise the native grouped path directly.

### P2. Legacy config helpers look superseded by the new layer system

Status: done in `codebase-review-cleanups`. These helpers were removed from production config, and tests now cover only the still-supported normalizer helpers.

Original finding: the following helpers were defined in `src/g/interface/config.py` but were not used by production code:

- `load_default_trait_option()` and `load_default_g_output_option()` at `src/g/interface/config.py:22-29`
- `merge_option_dictionaries()` at `src/g/interface/config.py:373-391`
- `from_option_layers()` at `src/g/interface/config.py:403-417`
- `from_normalized_options()` at `src/g/interface/config.py:420-430`
- `floating_point_dtype_or_default()` at `src/g/interface/config.py:794-802`
- `resolve_configured_trait_type()` at `src/g/interface/config.py:805-813`
- `resolve_exclusive_columns()` at `src/g/interface/config.py:877-892`

Some are referenced by tests only.

Impact: these functions preserve old configuration architecture concepts after the layer/TOML rewrite. They make it harder to tell which API is authoritative.

Recommendation: delete unused helpers or move compatibility-only functions behind a clearly named internal module. Keep tests focused on public config entry points.

### P2. Public-ish unsupported binary correction paths still contain `NotImplementedError`

Exact Firth and SPA are rejected by config:

- `src/g/interface/config.py:1024-1032`

But compute modules still contain branches that raise `NotImplementedError`:

- `src/g/compute/regenie2_binary/correction.py:31-36`
- `src/g/compute/regenie2_binary/variant_major_correction.py:133-140`

Impact: these branches are effectively unreachable from valid config, but they still look like partially implemented public execution modes. Tests encode the failures, which can make unfinished behavior look like supported behavior.

Recommendation: either remove the unreachable branches from runtime compute and keep unsupported-mode validation at config boundaries, or move exact Firth/SPA into explicit future-work adapters that are not reachable from normal execution.

### P2. Multi-binary approximate Firth is not actually batched across traits

The score-only multi-binary path is batched. The approximate Firth path loops over traits in Python:

- `src/g/compute/regenie2_binary/api.py:258-310`

The code comments already acknowledge this:

- `src/g/compute/regenie2_binary/api.py:269-273`
- `src/g/compute/regenie2_binary/api.py:304-309`

Impact: multi-binary with Firth shares genotype transfer, but then pays per-trait Python dispatch and single-trait correction costs. This can be a major performance gap for many binary phenotypes.

Recommendation: keep the current behavior as a correctness baseline, but make a dedicated batched Firth implementation plan. Treat this as one of the highest-impact performance refactors after lifecycle hardening.

### P2. Firth candidate counting forces a host synchronization

Approximate Firth candidate selection counts candidates on the host:

- `src/g/compute/regenie2_binary/variant_major_correction.py:85-120`

The stage is timed as `firth_candidate_count_host_sync`:

- `src/g/compute/regenie2_binary/variant_major_correction.py:96-99`

Impact: every chunk that reaches approximate Firth selection can force device/host synchronization before correction. That may be acceptable for sparse candidates, but it should be measured and bounded.

Recommendation: benchmark this stage on realistic chunks. If it is visible, keep candidate capacity fixed per chunk or use device-side compaction/counting to avoid host sync in the hot path.

### P2. Multi-phenotype resume skips only chunks committed by every phenotype

The multi path computes the intersection of per-phenotype committed chunk sets:

- `src/g/engine/regenie2_pipeline.py:1200-1217`

Impact: if phenotype A has chunk 100 committed and phenotype B does not, chunk 100 is decoded/computed again for the whole group. The writer can skip per-trait committed slices, but genotype decode and compute are still repeated.

Recommendation: decide whether this conservative behavior is acceptable. If resume performance matters, split resumed multi runs by missing chunk sets or allow the callback to compute only traits that still need a chunk.

### P2. Callback runner abstract contract is implicit and threads start in the constructor

`NativeBgenCallbackRunner` defines methods that raise `NotImplementedError` but is not an `abc.ABC`:

- `src/g/engine/callbacks.py:341-429`

The worker dispatches to those virtual methods:

- `src/g/engine/callbacks.py:476-512`

Subclasses start the base worker threads during `super().__init__()`.

Impact: this makes subclass lifecycle fragile and harder to unit-test. A partially initialized subclass can theoretically receive work if external code gets a reference early, and missing overrides are caught only at runtime.

Recommendation: make this an explicit ABC/protocol or split it into a concrete worker object that receives callable handlers after subclass initialization is complete.

### P2. Buffer pool accounting can drift across shape or dtype changes

Status: done in `codebase-review-cleanups`. The native callback runner now registers buffers allocated by the pool, ignores unowned arrays on release, and removes discarded owned buffers from accounting before allocating shape/dtype replacements.

Original finding: `acquire_dosage_buffer_with_shape()` replaced a free buffer when shape or dtype differed, but did not account for the discarded buffer:

- `src/g/engine/callbacks.py:714-734`

Impact: mixed buffer shapes or switching dosage/packed8 paths can allocate replacements while the pool count still represents old buffers. Current chunk shapes are mostly stable, so this is likely not a hot leak today, but it is fragile for future mixed modes.

Recommendation: keep separate pools by `(shape, dtype)` or decrement/count discarded buffers explicitly.

### P2. Unsafe native pointer writes should stay behind audited wrappers

Rust BGEN reading writes into Python-owned NumPy buffers through raw pointer addresses:

- `src/genotype/bgen/reader.rs:237-303`
- `src/genotype/bgen/reader.rs:306-380`
- `src/python/mod.rs:860-890`
- `src/python/mod.rs:911-961`

The current Python engine path validates shapes and contiguity before taking the pointer, which is good. The unsafe boundary is still large and spread across multiple functions.

Impact: future callers or refactors can accidentally bypass a required shape/contiguity/lifetime check. A mistake here is memory unsafety, not just a Python exception.

Recommendation: centralize the "validated writable NumPy buffer to raw pointer" pattern into one Rust helper and keep direct address-taking APIs out of public Python surfaces.

### P2. Large Rust decode functions are difficult to reason about

Variant-major decode logic has long, deeply nested iterator chains and unsafe/tile-specific branches:

- `src/genotype/bgen/reader.rs:306-443`
- `src/genotype/bgen/reader.rs:446-571`

Impact: this is high-performance code, but the structure makes it hard to audit correctness around sample selection, tile offsets, stats arrays, and trusted/non-trusted paths.

Recommendation: keep the hot loops, but refactor the surrounding orchestration into named structs/functions for tile context, output views, stats views, and trusted vs generic decode. The goal is to make invariants obvious without changing the low-level kernel.

### P2. INFO score denominator needs an explicit statistical contract

Rust preprocessing computes an imputed dosage square sum using missing-count imputation:

- `src/genotype/preprocess.rs:348-352`

But INFO score uses an expected variance denominator based on observed count:

- `src/genotype/preprocess.rs:353-358`

Impact: this may be intentional, but the contract is not obvious. If the intended INFO metric is after mean-imputation over all selected samples, the denominator may need selected sample count instead of observed count.

Recommendation: verify against REGENIE or the desired statistical definition with missing genotypes. Add a focused test with known missingness where the expected INFO differs between observed-only and imputed-denominator definitions.

### P2. Rust sample-count conversion silently saturates in one preprocessing path

Status: done in `codebase-review-cleanups`. The shared Rust stats builder now rejects selected sample counts that exceed the existing `i32` statistics representation, and BGEN preprocessing paths surface that as a range error.

Original finding: missing count used `i32::try_from(selected_sample_count).unwrap_or(i32::MAX)`:

- `src/genotype/preprocess.rs:350`

Impact: absurdly large sample counts are not realistic today, but silent saturation is the wrong failure mode in statistical code. If limits are exceeded, output statistics become wrong instead of failing loudly.

Recommendation: return an error on sample counts that exceed the supported statistics representation, or move counts to a wider type consistently.

### P3. TOML serialization only handles scalars

Status: done in `codebase-review-cleanups`. `format_toml_value()` now serializes list/tuple values as TOML arrays, with direct regression coverage.

Original finding: `format_toml_value()` formatted booleans, numbers, paths, and then stringified everything else:

- `src/g/interface/config.py:1277-1287`

But the TOML schema accepts string lists:

- `src/g/interface/toml_schema.py:10-11`

Impact: if a list reaches `dumps_toml()`, it becomes a quoted Python representation rather than a TOML array. Current call paths mostly pre-join repeated/list options, but the serializer is easy to misuse.

Recommendation: either support TOML arrays in `format_toml_value()` or make the function private and assert it only accepts scalar values.

### P3. Unknown-option validation carries a likely stale manual entry

Status: done in `codebase-review-cleanups`. The manual `g-trusted-no-missing-diploid` exception was removed because the option is already registry-backed.

Original finding: `validate_unknown_options()` manually included `"g-trusted-no-missing-diploid"` in addition to registry-supported options:

- `src/g/interface/config.py:847-856`

Impact: this looks like migration residue from the config rewrite. It may be harmless, but it weakens confidence that the registry is the single source of truth.

Recommendation: remove the manual exception if the option registry already owns it, or add a comment explaining why it is not represented there.

### P3. Repository is dominated by tracked archive files

Tracked file counts:

- `git ls-files | wc -l` -> `18427`
- `git ls-files 'archive/**' | wc -l` -> `18240`

`pyproject.toml` excludes the archive from Ruff:

- `pyproject.toml:53-56`

`.gitignore` ignores only selected generated archive build artifacts:

- `.gitignore:13-16`

Impact: the repository is much harder to review, search, and reason about. Most tracked files are not active application code. Tooling avoids them by exclusion, which means tracked code can rot without checks.

Recommendation: move historical/vendor archive content out of the main repository history if possible, or isolate it as a submodule/artifact with explicit ownership. If it must remain, document that it is not active code and keep all active-code searches/checks scoped away from it.

## Positive Notes

- The active Python package passes Ruff and Ty.
- The Rust workspace passes strict Clippy pedantic checks.
- Preflight already validates finite arrays, covariate shape, covariate rank, residual degrees of freedom, binary coding, prediction lengths, and required chromosomes in `src/g/engine/preflight.py:32-80` and `src/g/engine/preflight.py:91-156`.
- Runtime AVX2 selection now matches the desired policy: use AVX2 when available and scalar otherwise, with no user-facing SIMD switch in `src/genotype/preprocess.rs:184-193`.
