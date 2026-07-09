I reviewed the uploaded audit and the current GitHub state. I did not run the repo locally, so this is a static/code-architecture cleanup plan based on the audit output plus the latest files I fetched.

The good news: the latest update already fixed one of the big structural issues. The workspace no longer contains `crates/cli`; the members are now root `g`, `engine`, `genotype`, `input`, `interface`, `output`, `plan`, and `runtime`.  The native CLI frontend has moved into `g-interface` as `native_cli`, and it now supports a Python bridge through `G_NATIVE_CLI_PYTHON_BRIDGE`.   That was the right direction.

The bad news: the audit confirms your intuition. The codebase is still carrying a lot of structural bloat. The issue is not mostly dead code; `vulture` found almost nothing important. The issue is wrapper chains, wide public facades, PyO3 surface explosion, and Python orchestration that still exists because the Rust engine boundary is not finished.

## Audit summary

From the uploaded audit:

```text
Total LOC:                ~77,537
Python functions:         578
Python trivial wrappers:  137
Rust functions detected:  3,496
Rust trivial candidates:  1,784
Public re-export blocks:  62
Vulture findings:         one unused test variable
cargo-machete:            no unused dependencies found
cargo-udeps:              not installed / unavailable
```

Largest files by LOC are concentrated in exactly the places we expected:

```text
src/binding/callback_runtime_resources.rs       3254
src/binding/schedule.rs                         2004
crates/engine/src/schedule.rs                  1930
src/binding/run_events.rs                       1748
crates/genotype/src/bgen/decode/mod.rs         1707
src/g/compute/regenie2_binary/firth/*.py       1000+
src/binding/config/mod.rs                       1186
src/binding/run_engine.rs                       1083
src/g/engine/callbacks/runtime.py              1044
```

Important interpretation: **the codebase is not full of unused code. It is full of used glue that should not exist at that layer.** That means cleanup should be responsibility migration and facade shrinking, not random deletion.

---

# Immediate priority: make the baseline green

The uploaded `cargo_clippy.txt` shows compile/test failures. Some may have been fixed by the latest push, but at least one issue still appears present in current GitHub: `crates/input/src/sample/tests.rs` imports `align_multi_sample_data_from_sample_file`, `align_sample_data_from_sample_file`, and `validate_sample_file_header` from `super`, but the current `sample/mod.rs` exposes `align_sample_data`, `align_multi_sample_data`, and `align_grouped_sample_data`, not those removed helper names.

**PR 0 should be only test/build stabilization.**

Agent task:

```text
1. Run cargo test --workspace --all-features.
2. Run cargo clippy --workspace --all-targets --all-features.
3. Fix stale tests after module split.
4. Do not refactor architecture in this PR.
```

For the input tests specifically, either update tests to use `AlignmentInputs` / `MultiAlignmentInputs` directly, or restore the removed helpers as `#[cfg(test)]` helpers inside `sample/tests.rs`. Do not reintroduce them as production public functions.

Acceptance criteria:

```text
cargo test --workspace --all-features passes
cargo clippy --workspace --all-targets --all-features passes
pytest architecture tests pass
```

---

# Main cleanup strategy

Use four labels for every cleanup item:

```text
DELETE       unused or obsolete compatibility path
INLINE       wrapper has one caller and no semantic value
PRIVATIZE    keep internally but remove from public facade/PyO3
MIGRATE      useful logic, but belongs in Rust instead of Python
```

Do not delete code only because it is one-caller. Some one-caller functions are valuable boundaries: JAX compiled-shape wrappers, unsafe isolation, error-context wrappers, and import-boundary functions.

---

# Phase 1 — Shrink crate public facades

This is the highest leverage. The recent commit cleaned internal module layout, but several `api.rs` files still export too much.

## 1A. Shrink `g-engine::api.rs`

Current `g-engine::api.rs` still exports callback diagnostics, callback progress, `BoundedCallbackQueue`, callback summary, coordinator internals, effects, output manifest helpers, preflight helpers, preparation helpers, and a huge list of schedule internals.

Target production facade:

```rust
pub use crate::backend::{
    AssociationBackend,
    AssociationBatchResult,
    BackendError,
    GenotypeBatchView,
    PredictionView,
    PreparedGroupInput,
};

pub use crate::coordinator::{
    EngineCoordinator,
    EngineRunInput,
    EngineRunReport,
    EngineGroupRunInput,
    EngineGroupRunReport,
};

pub use crate::error::{EngineError, EngineResult};
pub use crate::phase::RunPhase;

// Temporary until PyO3 is rewritten around NativeRunEngineSession.
pub use crate::pipeline::Regenie2RunEngineCore;
```

Move these out of production public API:

```text
CallbackQueue*
CallbackSchedulerState
CallbackWorker*Plan
DosageBuffer*Plan
ResultInFlight*Plan
ResultWrite*Plan
BgenDeliveryCleanup* internals
GpuGenotypeFormatResolutionPlan
preflight scalar validators
output manifest helper
injected/fake failure types
```

Where to move them:

```text
pub(crate)                 if only engine internals use them
#[cfg(test)]               if only unit tests use them
feature = "test-support"   if integration tests need them
debug facade               if PyO3 diagnostics still need them temporarily
```

Acceptance criteria:

```text
g-engine api.rs loses at least 50% of public re-exports.
Production crates do not import schedule internals from g-engine.
PyO3 schedule/callback internals are either removed or explicitly debug/test-only.
```

## 1B. Shrink `g-runtime::api.rs`

`g-runtime::api.rs` is still the biggest public-surface offender. It exports CLI runtime, JAX runtime internals, logging sink functions, Rayon helpers, a huge `run_events` surface, run metadata builders, runtime paths, runtime policy, runtime state, shutdown, telemetry policy/session/writer internals, timing internals, and trusted BGEN validation cache helpers.

Target production facade:

```rust
pub use crate::error::{RuntimeError, RuntimeResult, RuntimeCompatibilityError};

pub use crate::runtime_state::{
    ProcessRuntimeState,
    RuntimePolicyPayload,
    RunRuntime,
    RuntimeCompatibilityToken,
    JaxRuntimePolicyPayload,
};

pub use crate::jax_runtime::{
    JaxRuntimeSetupSession,
    JaxRuntimeSetupPayload,
    JaxRuntimeSetupSideEffectPlan,
    JaxRuntimeDiagnosticEventPayload,
    resolve_jax_runtime_setup,
    plan_jax_runtime_setup_side_effects,
};

pub use crate::logging_sink::{
    LoggingSinkConfig,
    initialize_logging_sinks,
    shutdown_logging_sinks,
};

pub use crate::shutdown::{
    ShutdownHandlerSession,
    ShutdownSignalPayload,
};

pub use crate::telemetry_session::{
    TelemetryRunSessionState,
    TelemetryEventEnvelope,
};

pub use crate::timing::{
    StageTimingRecorder,
    StageTimingSnapshotPayload,
};
```

Move event-builder explosion out of the main facade. Ideally make a submodule:

```rust
pub mod events {
    pub use crate::run_events::{
        RunCompletedEventPayload,
        RunFailedEventPayload,
        RunInterruptedEventPayload,
        build_run_completed_event_from_artifacts,
        build_run_failed_event_payload,
        build_run_interrupted_event_payload,
    };
}
```

Then eventually make event names and line renderers private to runtime.

Acceptance criteria:

```text
g-runtime main api.rs no longer re-exports every diagnostic builder.
Code that needs event internals imports a deliberate runtime::events/debug path.
Python-facing code does not call dozens of individual runtime event builders.
```

## 1C. Shrink `g-output::api.rs`

`g-output::api.rs` still exposes many manifest/resume helpers: current/prepared manifest JSON construction, hash helpers, manifest file loading/writing, prepared-plan reconstruction, committed chunk scanning, strict repair, and resume validation.

Target production facade:

```rust
pub use crate::chunk::{NativeChunkHandle, NativeChunkStats, VariantMetadataColumns};
pub use crate::error::{OutputError, OutputResult};
pub use crate::session::OutputWriterSession;
pub use crate::writer::OutputFileFormat;

pub use crate::manifest::{
    OutputResumeMode,
    OutputRunPaths,
    PreparedOutputRun,
    InitializedOutputRun,
    prepare_output_run,
    initialize_output_run,
    validate_run_manifest_compatibility,
};

pub use crate::finalization::finalize_output_run_chunks;
```

Move to `pub(crate)` / `admin` / `test_support`:

```text
build_manifest_json_sha256
build_file_content_sha256
build_prepared_run_plan_from_current_header_json
build_prepared_run_plan_json_from_current_header_json
load_run_manifest_json
write_run_manifest_json
repair_strict_manifest_chunk_commits
scan_committed_chunk_identifiers
validate_strict_manifest_chunks
```

Acceptance criteria:

```text
output API says “prepare/write/finalize output,” not “free-form manifest JSON toolkit.”
```

## 1D. Shrink `g-genotype::api.rs`

Current `g-genotype::api.rs` exports BGEN internals, common contracts, planner functions, preprocess functions, and `BgenGenotypeSource`.

Target production facade:

```rust
pub use crate::source::BgenGenotypeSource;
pub use crate::common::{
    ChunkSpec,
    ChunkStats,
    GenotypeReaderCore,
    VariantMetadataColumns,
};
pub use crate::error::{GenotypeError, GenotypeResult};
pub use crate::bgen::{BgenError, ReaderProfileSnapshot};
```

Move these behind `pub(crate)` or `test_support` unless external crates truly require them:

```text
plan_chromosome_homogeneous_chunks
resolve_total_variant_count
build_chunk_stats_from_summaries
build_empty_chunk_stats
increment_dosage_summary_counts
preprocess_row_major_dosage_matrix
summarize_variant_major_dosage_matrix
set_bgen_decode_tile_variant_count
set_bgen_row_major_direct_write_enabled
CompressionType if not part of public contract
```

`BgenGenotypeSource` is a useful facade, but it is currently mostly pass-throughs into `BgenReaderCore`. It also exposes raw pointer-address buffer APIs.  Keep those temporarily for PyO3/hot path, but mark them as FFI/native-buffer APIs, not the generic genotype source contract.

Acceptance criteria:

```text
Engine can use genotype source through source-level methods.
Preprocess/planner internals are not broad public API.
Raw pointer APIs are quarantined as FFI/buffer-specific.
```

---

# Phase 2 — Split production API from debug/test API

The audit shows many functions are useful for tests and diagnostics, but should not be public production API.

Add this convention to every crate:

```rust
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

For PyO3:

```text
_core.debug.engine
_core.debug.runtime
_core.debug.output
```

or feature-gated registration.

Move these categories out of production:

```text
fake backends
fake effects
injected failures
schedule plan internals
callback queue internals
callback runtime resource internals
manifest repair/scanning helpers used only in tests
low-level event payload builders
```

Current `src/binding/mod.rs` still registers callback summary, progress, queue, runtime resources, diagnostics, schedule, run engine, lifecycle, and preflight all into the root `_core` module.  That should not be production surface forever.

Acceptance criteria:

```text
Production PyO3 surface exposes high-level handles only.
Debug/test APIs require _core.debug.* or test-support feature.
Architecture checker rejects new production imports of debug internals.
```

---

# Phase 3 — Finish `_core` submodules or keep root tiny

The latest code added the Rust-owned CLI driver, but `_core` registration is still mostly flat. `src/binding/mod.rs` defines many modules and registers them into the same root module; it also registers CLI driver exports directly into the runtime domain/root.

Target:

```text
_core.cli
_core.config
_core.runtime
_core.telemetry
_core.engine
_core.genotype
_core.input
_core.output
_core.debug
```

Short-term root compatibility aliases are okay, but new Python code should use submodules.

Immediate cleanup:

```text
_core.run_cli_with_python_backend -> _core.cli.run_with_python_backend
_core.NativeCliRunContext         -> _core.cli.NativeRunContext
_core.NativeCliRunResult          -> _core.cli.NativeRunResult
```

Do not use `.functions` modules.

Acceptance criteria:

```text
import g._core.cli works
import g._core.engine works
root _core compatibility aliases are documented as temporary
no production code imports _core.debug
```

---

# Phase 4 — Delete Python orchestration modules after Rust owner exists

Your Python CLI is now much better. `src/g/runner/cli.py` is only a small adapter around `_core.run_cli_with_python_backend`, prints stdout/stderr chunks, and invokes the temporary Python backend callback.  Good.

Now target the remaining Python orchestration.

From audit, biggest Python structural-bloat files:

```text
src/g/engine/callbacks/runtime.py               1044 lines, 82 functions
src/g/engine/callbacks/binary.py                 829 lines
src/g/engine/callbacks/linear.py                 679 lines
src/g/engine/regenie2_pipeline/grouped.py        large orchestration functions
src/g/engine/regenie2_pipeline/multi_trait.py    168-line dispatch function
src/g/engine/regenie2_pipeline/single_trait.py   123-line pipeline function
src/g/io.py                                      manifest-header adapter
src/g/execution_plan.py                          Python plan mirror
src/g/runner/execution.py                        run lifecycle / dispatch
```

Do this as ownership migration, not deletion.

## 4A. Move output/manifest preparation to Rust

Delete or shrink:

```text
src/g/io.py::build_current_run_manifest_header
src/g/engine/regenie2_pipeline/outputs.py::build_pipeline_manifest_header
src/g/engine/regenie2_pipeline/outputs.py::initialize_pipeline_output_runs
src/g/engine/regenie2_pipeline/outputs.py::create_pipeline_writer_sessions
```

Replace with:

```rust
NativeRunEngineSession::prepare_output_bundle(...)
```

Python should not build manifest headers manually.

## 4B. Move pipeline orchestration to Rust

Delete or shrink:

```text
run_single_trait_bgen_pipeline
run_regenie2_multi_phenotype_bgen_pipeline
run_regenie2_grouped_per_phenotype_bgen_pipeline
run_prepared_grouped_per_phenotype_union_bgen_pipeline
```

Target:

```rust
NativeRunEngineSession::run_with_backend(...)
```

Python backend only implements:

```python
prepare_group(...)
prepare_chromosome(...)
compute_batch(...)
```

## 4C. Move callback queue/runtime ownership to Rust

`callbacks/runtime.py` is the biggest Python bloat target. It owns worker loops, queue dispatch, buffer pool calls, result slots, result materialization/write routing, diagnostics flushing, and progress emission.

Target:

```text
Rust owns:
  queue workers
  buffer pool
  result in-flight slots
  progress
  result writer
  cleanup

Python owns:
  JAX state
  JAX compute_batch
```

Acceptance criteria:

```text
NativeBgenCallbackRunner no longer has consume_dosage_chunks loop.
Python no longer processes ResultWriteWorkItem queues.
Python no longer calls writer_session directly for every chunk.
```

---

# Phase 5 — Simplify Rust PyO3 files

Audit shows `src/binding` is the largest Rust bloat area:

```text
src/binding/callback_runtime_resources.rs   3254 lines
src/binding/schedule.rs                     2004 lines
src/binding/run_events.rs                   1748 lines
src/binding/config/mod.rs                   1186 lines
src/binding/run_engine.rs                   1083 lines
src/binding/logging.rs                       853 lines
src/binding/output.rs                        834 lines
```

Most of these are PyO3 wrappers around too-broad internal APIs.

Cleanup order:

## 5A. `src/binding/schedule.rs`

After `g-engine::api.rs` stops exporting scheduler internals, move this to debug/test or delete large parts.

Production Python should not call:

```text
plan_dosage_work_handoff
plan_result_write_item_dispatch
plan_callback_worker_finish
plan_dosage_buffer_reuse
...
```

## 5B. `src/binding/callback_runtime_resources.rs`

Do not polish this file; replace it. It exists because Python owns callback queues/resources. After Rust owns callback runtime, this file should either disappear or become debug-only.

## 5C. `src/binding/run_events.rs`

Move many event payload builder bindings behind `_core.runtime.events` or `_core.debug.runtime.events`. The long-term runtime should emit events internally, not expose every builder to Python.

## 5D. `src/binding/run_engine.rs`

Once `NativeRunEngineSession` exists, this should become a high-level engine binding, not a BGEN/input/output orchestration binding.

Acceptance criteria:

```text
No src/binding file over 1000 lines, except generated/config stubs if unavoidable.
PyO3 files adapt handles; they do not implement business orchestration.
```

---

# Phase 6 — Rust internal wrapper cleanup

The audit found many Rust trivial candidates. Many are harmless getters, especially PyO3 properties, but several categories need cleanup.

## 6A. Pass-through source wrappers

`BgenGenotypeSource` currently wraps many methods from `BgenReaderCore`. Some are fine because they make a clean source facade; others are duplicated API. Keep the source facade, but remove direct public exposure of the underlying reader.

Decision rule:

```text
If BgenGenotypeSource method is the stable genotype-source API: keep.
If it only exists to expose BgenReaderCore internals: make pub(crate) or move to FFI adapter.
```

## 6B. Schedule aggregation wrappers

`crates/engine/src/schedule.rs` now aggregates many submodules. Internally okay, externally too broad. After public facade shrink, keep `schedule.rs` as `pub(crate)` policy hub.

## 6C. Test-only long functions

Many large Rust functions are tests. Do not optimize for line count there until production bloat is reduced. But fix stale tests and split large integration-style tests into scenario helpers.

---

# Phase 7 — Compute-layer cleanup

Do this after orchestration cleanup. The JAX compute code is large but more legitimate; it is real math.

Audit shows large compute files:

```text
firth/full_model.py
firth/batch/compute.py
firth/scalar_approx.py
variant_major_correction/dispatch.py
binary/api.py
binary/state.py
linear/score.py
```

Do **not** rewrite or move these just to reduce LOC.

Clean only clear wrapper bloat:

```text
single-trait wrappers that only call multi-trait wrappers
tiny wrappers around same JAX function with no compiled-shape reason
duplicate packed8/dosage entrypoints if backend planner no longer uses both
```

Keep wrappers that correspond to distinct JIT shapes, donation semantics, or benchmarked compiled paths.

Acceptance criteria:

```text
No statistical output changes.
No extra JAX recompilation.
No performance regression.
```

---

# Phase 8 — Enforce bloat prevention

Add these checks to CI / architecture checker.

## Rust public API checks

```text
No new pub use from crates/*/src/api.rs without PUBLIC_API.md update.
No public fake/test types outside test_support.
No broad pub use crate::schedule::{...} blocks in production facade.
No Result<T, String> in production crate APIs.
```

## Python bloat checks

```text
No new production Python wrapper around exactly one _core call unless allowlisted.
No production imports from g.engine.callbacks.runtime after Rust callback runtime lands.
No production imports from g.engine.regenie2_pipeline after Rust engine session lands.
No production imports from _core.debug.
No function over 100 lines without allowlist.
No Python file over 600 lines without allowlist.
```

## PyO3 checks

```text
No production registration of callback_queue/callback_runtime_resources/schedule/preflight internals.
Root _core compatibility aliases must shrink over time.
New PyO3 symbols must be registered in a domain submodule.
```

---

# Concrete agent PR list

Give agents this sequence.

## PR 0 — Restore green baseline

```text
Fix stale Rust tests after module split.
Run cargo test/clippy.
No architecture refactor.
```

## PR 1 — Shrink `g-engine` public facade

```text
Move scheduler/callback/preflight internals out of api.rs.
Add test_support/debug path where needed.
Update PyO3 imports.
```

## PR 2 — Shrink `g-runtime` public facade

```text
Move run-event builder explosion out of main api.rs.
Create runtime::events/debug namespaces.
Keep high-level runtime/telemetry/timing handles public.
```

## PR 3 — Shrink `g-output` and `g-genotype` facades

```text
Hide manifest/resume internals.
Hide genotype preprocess/planner internals.
Keep stable source/writer APIs.
```

## PR 4 — `_core` domain submodules

```text
_core.cli, _core.config, _core.runtime, _core.engine, _core.output, _core.debug.
Keep temporary root aliases.
Move production Python call sites to submodules.
```

## PR 5 — Native output preparation

```text
Move Python manifest/header/output-run/writer preparation into Rust.
Delete src/g/io.py production dependency.
```

## PR 6 — Native engine session skeleton

```text
Add NativeRunEngineSession.
Rust owns prepared run/lifecycle handles.
Python still supplies execution backend temporarily.
```

## PR 7 — Native pipeline orchestration

```text
Move single/multi/grouped pipeline orchestration to Rust.
Python only supplies JAX backend.
Delete or shrink regenie2_pipeline modules.
```

## PR 8 — Native callback runtime

```text
Rust owns queues, buffer pool, result slots, writer handoff.
Delete/shrink callbacks/runtime.py and PyO3 callback_runtime_resources.
```

## PR 9 — Python dead-wrapper pass

```text
Use python_function_audit.csv.
Delete/in-line trivial wrappers not needed after PRs 5–8.
```

## PR 10 — Rust wrapper pass

```text
Use rust_function_audit.csv.
Inline private trivial wrappers.
Move test-only helpers to test_support.
No public API churn unless already planned.
```

## PR 11 — Compute wrapper rationalization

```text
Only after benchmarks.
Remove redundant JAX wrappers with no distinct compiled-shape/benchmark value.
```

## PR 12 — Add bloat CI

```text
Public facade budget.
Python wrapper detector.
PyO3 production-surface detector.
Large-file/function allowlist.
```

---

# Suggested targets

Reasonable near-term targets:

```text
Reduce total LOC by 10k–15k without touching JAX math.
Reduce production PyO3 surface by 40–60%.
Reduce g-engine api.rs public exports by 60%+.
Remove execution_plan.py from production dispatch.
Delete most of engine/regenie2_pipeline after Rust engine session lands.
Delete or debug-gate callback_runtime_resources.rs after Rust callback runtime lands.
```

Longer-term target:

```text
Python contains:
  public API
  JAX setup
  JAX backend
  JAX kernels

Rust contains:
  config
  CLI
  runtime
  run lifecycle
  input
  genotype
  output
  scheduling
  manifests
  telemetry
  cleanup
```

---

# Bottom line

The audit says the codebase is not mainly dead-code bloated. It is **abstraction-bloated**.

The right cleanup order is:

```text
1. make tests green;
2. shrink Rust public facades;
3. split production vs debug/test APIs;
4. move remaining Python orchestration to Rust;
5. delete wrapper chains made obsolete by that migration;
6. only then rationalize JAX compute wrappers.
```

The latest CLI work is a good example of the direction: `runner/cli.py` is now tiny and Rust owns the lifecycle driver.  Repeat that pattern for execution planning, output preparation, pipeline orchestration, and callback runtime.
