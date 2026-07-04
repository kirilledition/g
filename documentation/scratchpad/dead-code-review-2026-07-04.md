# Dead Code Review - 2026-07-04

## Scope

Audited production code only: `src/`, `crates/`, and root manifests/configuration. Tests, docs, and tooling were used only as evidence for whether production symbols are referenced. The current dirty worktree was reviewed as-is. No code was removed by this task.

Baseline branch was `main` tracking `origin/main`. The worktree already had dirty Rust/Python/docs/test changes before this audit, including deletion of `src/python/association_backend.rs`, PyO3 export/stub cleanup in `src/g/_core.pyi`, and logging/runtime-state changes.

Production entrypoints considered: Python package imports under `src/g`, PyO3 registrations from `src/python/*`, Rust workspace crates under `crates/*`, root `Cargo.toml`, and dynamic imports from `src/g/runner/runtime.py`.

## Executive Summary

Safest removal found: root `Cargo.toml` depends on `serde` for package `g`, but root `src/` has no direct `serde` use. `cargo machete` flags this dependency, and workspace crates already declare their own `serde` dependencies. TOML config still uses `serde` through `crates/interface`; remove only the root package dependency, not `[workspace.dependencies] serde` or `crates/interface`'s dependency.

Strong dead-code candidates: `src/g/engine/timing.py` contains unreferenced key/serializer/summary helpers; `src/g/runtime_paths.py` and `src/g/jax_runtime/state.py` appear test-only or public-helper-only; several `src/g/io/output.py` wrappers are used by tests but not production callers.

Current dirty cleanup is mostly valid: association backend PyO3 exports, top-level genotype chunk planning, top-level sample alignment wrappers, and top-level logging functions have been removed from stubs/registration. Remaining issue: `cargo clippy` now fails on `NativeRuntimeState.shutdown_logging_runtime(&self)` because `self` is unused.

Main caveat: Python static analysis over-reports because this project uses PyO3, dynamic imports, callback protocols, and public compute entrypoints. `vulture` findings at 60% confidence are evidence, not proof.

## Findings

| Area | Symbol/path | Category | Confidence | Evidence | Recommended action |
| --- | --- | --- | --- | --- | --- |
| Root manifest | `Cargo.toml:89` package `g` dependency `serde = { workspace = true }` | Safe remove now | High | `cargo machete --with-metadata --skip-target-dir .` reports `g -- Cargo.toml -- serde`. `rg` finds `serde` usage in `crates/*`, not root `src/`. TOML config reads/writes use `serde` in `crates/interface/src/toml.rs` and `crates/interface/src/partial.rs`, and `crates/interface/Cargo.toml` already declares `serde = { workspace = true }`. | Remove root package dependency only. Keep `[workspace.dependencies] serde` and crate-level deps, especially `crates/interface`. |
| PyO3 runtime state | `src/python/runtime_state.rs:360` `shutdown_logging_runtime(&self)` | Likely dead receiver | High for receiver, medium for API shape | `cargo clippy -j 30 --workspace --all-targets -- -D warnings -W clippy::pedantic` fails with `unused self argument`. Method only forwards to `logging::shutdown_logging()`. | If PyO3 instance method shape is not required, make this an associated function. If `atexit`/Python API needs bound instance method, add targeted `#[expect(clippy::unused_self, reason = "...")]` with a focused test. |
| PyO3 logging | `src/python/logging.rs:613` `NativeTelemetrySession` | Unused public export / stale PyO3 facade | Medium-high | `_core.pyi` removed `NativeTelemetrySession`; `src/python/logging.rs:1115` no longer registers it; `tests/test_core.py:237` asserts absence. Struct is still internal state of `NativeTelemetryRunSession`. | Keep internal session writer, but convert PyO3-facing impl to private Rust methods where possible. Delete PyO3-only methods after telemetry close/session tests pass. |
| PyO3 association adapter | deleted `src/python/association_backend.rs`; removed from `src/python/mod.rs` | Safe cleanup already in dirty worktree | High | `mod association_backend` and registration removed; `_core.pyi` removed `NativePythonAssociationBackend` family; no current PyO3 registration remains. | Keep deletion. Run core stub parity check before merging cleanup branch. |
| PyO3 top-level wrappers | `_core.pyi` removed `ChunkSpec`, `plan_genotype_chunks`, top-level `align_*`, top-level logging functions | Safe cleanup already in dirty worktree | High | Dirty diff shows stubs removed; tests assert removed top-level exports absent; production now uses `Regenie2RunEngine`/native policies. | Keep cleanup. Do not remove underlying Rust crate types that are still used by engine/native policy code. |
| Python output wrappers | `src/g/io/output.py:175`, `:186`, `:197`, `:480`, `:586` | Stale test/docs-only code | Medium | `vulture` flags several output helpers. `rg` shows production callers mostly absent; tests and architecture checks call them heavily. Some wrappers forward to `native_output_lifecycle_policy()`. | Decide whether these remain public Python helpers. If not, migrate tests to the native lifecycle policy or runner output path first, then remove wrappers. |
| Runtime path helper | `src/g/runtime_paths.py:10` `default_local_cache_directory` | Stale test-only helper | Medium | Static import graph has zero production incoming refs. `rg` shows only `tests/test_jax_runtime.py` and stub/native policy references. | Remove module or document as public helper. If removing, update test to call `NativeRuntimeState.default_local_cache_directory_value`. |
| JAX runtime state helper | `src/g/jax_runtime/state.py:11` `describe_jax_runtime_policy` | Stale test/docs-only code | Medium-low | `vulture` flags it; no production caller found. | Remove if not documented public API. Otherwise add a production caller or keep with explicit public-helper rationale. |
| Timing diagnostics | `src/g/engine/timing.py:137`, `:151`, `:704`, `:738`, `:756`, `:775`, `:793` | Safe remove now / stale helper | High for no references, medium for public API | `rg` finds only definitions for `QueueBackpressureKey`, `TransferMetadataKey`, `serialize_chunk_stage_timings`, `serialize_queue_backpressure`, `serialize_transfer_metadata`, `build_chunk_stage_summary`, `build_binary_chunk_summary`. | Remove or move behind documented diagnostics API. If kept, add tests or production caller; current code has no static users. |
| Dynamic pipeline modules | `src/g/jax_runtime/setup.py`, `src/g/engine/regenie2_pipeline/single_trait.py`, `src/g/engine/regenie2_pipeline/multi_trait.py` | Keep despite low static usage | High | AST import graph reports zero incoming refs, but `src/g/runner/runtime.py:188`, `:373`, `:379`, `:385`, `:391` imports them through `importlib.import_module`. | Keep. Static import graph false positive. |
| Callback/runtime classes | `src/g/engine/callbacks/runtime.py` and callback-related `_core` classes | Keep despite low static usage | Medium | `vulture` flags many methods; PyO3 registration/reference audit shows these are callback protocol surfaces and runtime objects, often exercised dynamically. | Do not remove from vulture alone. Require focused callback integration test before any cleanup. |
| Compute kernels | `src/g/compute/*` API helpers, Firth helpers, enum members in `src/g/types.py` | Keep despite vulture findings | Medium | `vulture` reports 60% confidence only; these are public compute surfaces, dynamic dispatch targets, or enum values. | Treat as risky suspects. Remove only after caller graph plus focused numerical regression test. |

## PyO3 Export Audit

Read-only parser compared `src/python/*` registrations, `src/g/_core.pyi`, and `_core` references from production Python.

- Registered PyO3 symbols: 154.
- Stub symbols: 155.
- Registered symbols missing from stub: none.
- Stub symbols missing from registration: `ChunkStatsComputeArrays` only; this is a `TypedDict` helper, not a PyO3 export.
- Many registered classes have no direct `src/g` `_core` reference because they are return types, callback plans, or nested/native runtime objects. This is not enough by itself to remove them.

Notable result: current dirty PyO3 stub/registration alignment is good after removing association backend, top-level logging, top-level genotype planning, and top-level sample alignment wrappers.

## Python Audit

`ruff` and `ty` both passed on `src`. `vulture src/g` exited with findings at 60% confidence. Most useful clusters were:

- `src/g/engine/timing.py`: unreferenced timing key/serializer/summary helpers.
- `src/g/io/output.py`: wrappers referenced heavily by tests but little/no production code.
- `src/g/runtime_paths.py` and `src/g/jax_runtime/state.py`: helper modules with no production incoming refs.
- `src/g/engine/callbacks/runtime.py`, `src/g/compute/*`, and `src/g/types.py`: likely false positives or public/dynamic surfaces.

Static Python import graph found six modules with zero incoming static imports:

- `g.compute` package marker.
- `g.engine.regenie2_pipeline.multi_trait`.
- `g.engine.regenie2_pipeline.single_trait`.
- `g.jax_runtime.setup`.
- `g.jax_runtime.state`.
- `g.runtime_paths`.

`single_trait`, `multi_trait`, and `jax_runtime.setup` are imported dynamically by `src/g/runner/runtime.py`, so keep them.

## Rust Audit

`cargo machete` reports one unused dependency: root package `g` depends on `serde`. Manual search confirms root `src/` has no direct `serde` use; workspace crates keep their own `serde` dependencies. This does not mean `serde` is unused globally: TOML config decoding/encoding still uses it in `crates/interface`.

`cargo clippy` on the head node with `-j 30` and a 64 GiB virtual-memory limit failed on one issue:

```text
error: unused self argument
src/python/runtime_state.rs:360:33
```

This is from the dirty-worktree addition of `NativeRuntimeState.shutdown_logging_runtime(&self)`.

`cargo-udeps` was not available. Installing `cargo-udeps v0.1.61` failed because OpenSSL pkg-config metadata was unavailable (`openssl.pc` not found). No `cargo udeps` audit was run.

`cargo llvm-cov` was available, but full coverage was not run. It would be supporting evidence only and not proof of dead code; the clippy/machete/vulture results were enough for this report.

## Commands Run

Repository root was `/mnt/beegfs/kirill/Projects/g`. Broad Rust commands were run on the head node with `-j 30` and `ulimit -v` where noted, per user instruction.

```bash
git -C /mnt/beegfs/kirill/Projects/g status --short --branch
git -C /mnt/beegfs/kirill/Projects/g log --oneline -8
just --list
uv --project /mnt/beegfs/kirill/Projects/g run ruff check /mnt/beegfs/kirill/Projects/g/src
uv --project /mnt/beegfs/kirill/Projects/g run ty check /mnt/beegfs/kirill/Projects/g/src
uv tool install vulture
vulture /mnt/beegfs/kirill/Projects/g/src/g
cargo --version
rustc --version
cargo metadata --manifest-path /mnt/beegfs/kirill/Projects/g/Cargo.toml --no-deps --format-version 1
cargo install cargo-machete --locked
bash -lc 'ulimit -v 33554432; cargo machete --with-metadata --skip-target-dir /mnt/beegfs/kirill/Projects/g'
bash -lc 'ulimit -v 67108864; cd /mnt/beegfs/kirill/Projects/g && cargo clippy -j 30 --workspace --all-targets -- -D warnings -W clippy::pedantic'
bash -lc 'ulimit -v 67108864; cargo install cargo-udeps --locked -j 30'
cargo udeps --version
cargo llvm-cov --version
```

Additional read-only scripts were run to parse PyO3 registrations/stubs/references and Python AST imports. They did not write files.

Validation commands run after writing this report:

```bash
git diff --check
test -f documentation/scratchpad/dead-code-review-2026-07-04.md
rg -n "<temporary-path-pattern>" documentation/scratchpad/dead-code-review-2026-07-04.md
```

## Tool Versions

- `cargo 1.96.0 (30a34c682 2026-05-25)`
- `rustc 1.96.0 (ac68faa20 2026-05-25)`
- `vulture 2.16`
- `cargo-machete 0.9.2`
- `cargo-llvm-cov 0.8.7`
- `cargo-udeps` not installed; install attempt failed before executable became available.

## Limitations

No code removal was performed. No `docs-build` was run, per task instructions. `cargo-udeps` could not be installed because the local environment lacked OpenSSL pkg-config metadata. `cargo llvm-cov` was not run because it is supporting evidence only and would be broad/heavy for this audit.

Existing dirty worktree changes were not reverted or normalized. Findings that rely on current dirty diffs should be rechecked after those changes settle.
