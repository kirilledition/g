# Crates Architecture Implementation Plan

> Internal scratchpad. Agent execution plan. Not user docs.
> Caveman-compressed: terse checklist, all source-plan signal kept.

This plan merges current public API boundary rules, crate `PUBLIC_API.md` files,
and internal architecture notes into one implementation sequence.

Primary sources:

- `documentation/development/rust-crate-boundaries.md`
- `documentation/scratchpad/architecture.md`
- `documentation/scratchpad/crates_internal_archintechture.md`
- `crates/*/PUBLIC_API.md`

## Mission

Make Rust crates boring and agent-safe:

- one small crate-root facade per crate;
- private implementation modules by default;
- typed public request/result/error contracts;
- no permanent legacy wrappers after callers move;
- no cross-crate JSON in hot paths;
- batch-oriented compute and I/O boundaries;
- root `g` PyO3 code as glue only.

Do not broaden scope into algorithm changes, tests, or tooling. During this
plan, ignore `tests/`, test-support rewrites, benchmark tooling, dev tooling,
and Justfile/tooling cleanup. Tests and tooling are separate follow-up work.

## Non-Goals

- Do not create new crates such as `g-common` or `g-utils`.
- Do not stabilize old API names for compatibility. App unreleased.
- Do not preserve Python fallback paths once Rust owns the contract.
- Do not add wrappers that only rename one function and have one caller.
- Do not expose fake/test-only types from production facades.
- Do not add data-file scans to config parsing.
- Do not move JAX kernels out of Python in this pass.
- Do not edit tests to satisfy this plan.
- Do not edit `tooling/`, benchmark scripts, or Justfile recipes for this plan.

## Global Rules

Each crate root should look like this unless the crate has a documented reason:

```rust
mod api;
mod error;
mod internal_module;

pub use api::*;
pub use error::{CrateError, CrateResult};

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

Avoid this:

```rust
pub mod parser;
pub mod scheduler;
pub mod writer;
```

Public module trees are accidental APIs unless the crate `PUBLIC_API.md`
explicitly says that module namespace is public.

Dependency direction:

```text
g-plan
g-interface -> g-plan
g-cli -> g-interface
g-genotype
g-input
g-output -> g-plan
g-runtime
g-engine -> g-plan, g-genotype, g-input, g-output, g-runtime
root g -> internal crates
```

Only `g-engine` coordinates domains. Root `g` adapts Rust to Python.

## Slice Protocol

For each implementation slice:

1. Read crate `PUBLIC_API.md`.
2. Read crate root `lib.rs` and target module.
3. Run `rg` for public callers before moving anything.
4. Move one boundary at a time.
5. Update callers to use the facade.
6. Delete obsolete wrapper/fallback in the same slice.
7. Update `PUBLIC_API.md` if facade items changed.
8. Run focused compile/architecture checks.
9. Stop if behavior change is required but not described in this plan.

No "prepare but leave old path" slices. Move caller, delete old path, validate.

## Standard Validation

Use focused production/codegen checks. Ruff format may run directly; no
format-check step needed.

```bash
CARGO_BUILD_JOBS=30 RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo check -p <crate> --lib
CARGO_BUILD_JOBS=30 RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo clippy -p <crate> --lib -- -D warnings -W clippy::pedantic
just check-core-stub
just check-rust-architecture
just check-python-architecture
uv run ty check src
uv run ruff format src
uv run ruff check src
```

Do not work on tests or tooling while executing this plan. If a validation
command reports failures only in `tests/`, `tooling/`, benchmark scripts, or
Justfile recipes, record the failure and continue with architecture cleanup.
Only fix production crate/root-PyO3 code touched by the current slice.

Fix `cargo check` errors and clippy pedantic warnings for production code in
scope. For this plan, run Rust compile/lint on head node with 30 cores and mold
linker as shown above. Do not run heavy tests, benchmarks, or tooling suites.
Use SLURM only if a later non-plan task needs heavy CPU/GPU execution.

## Phase 0 - Inventory

Goal: know actual public surface before editing.

Run:

```bash
rg "pub mod|pub use" crates/*/src/lib.rs
rg "pub enum .*Error|pub struct .*Error|type .*Result|Result<[^>]+,\s*String>|anyhow::Result" crates src/binding
rg "serde_json::Value|PyDict|PyAny|PyObject" crates src/binding
rg "map_err\(.*PyValueError|PyValueError::new_err|PyRuntimeError::new_err" src/binding
rg "g_genotype::bgen::BgenReaderCore|g_genotype::planner" crates/engine src/binding
rg "test_support|Fake|Mock" crates/*/src
```

Build a scratch table while working:

```text
crate
current public modules
current crate-root pub use
documented PUBLIC_API items
public errors and locations
Result<T, String> exports
PyO3 callers
delete candidates
```

Done when:

- every public module/export has a keep/remove decision;
- every public error has a target home;
- every PyO3 orchestration path has an owning Rust facade target.

## Phase 1 - Public Facade Boundaries

Goal: each crate exposes one deliberate facade.

Steps:

1. Compare `crates/<crate>/PUBLIC_API.md` to `crates/<crate>/src/lib.rs`.
2. Replace accidental `pub mod` exports with private `mod` plus crate-root
   `pub use` for documented items.
3. If downstream callers use implementation modules, add or use a facade item
   and update those callers.
4. Gate fake/test helpers behind:

```rust
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

5. Remove public wrappers that only forward to another public item.
6. Update `PUBLIC_API.md` immediately.

Done when:

- no production crate root exposes implementation module trees;
- `PUBLIC_API.md` matches crate-root facade;
- root `g` does not bind low-level Rust helper chains when one Rust owner can
  call another directly.

Useful guard:

```bash
rg "pub mod" crates/*/src/lib.rs
```

Every result must be deliberate and documented.

## Phase 2 - Error Boundaries

Goal: public APIs return crate-owned typed errors; implementation errors stay
near implementation and convert upward.

Standard:

```text
src/error.rs
  public crate error type
  public Result alias
  From conversions from lower-level public/domain errors
```

Use:

```rust
pub use error::{CrateError, CrateResult};
```

Avoid public:

```rust
Result<T, String>
anyhow::Result<T>
```

Layered pattern:

```text
src/error.rs              crate boundary
src/<domain>/error.rs     domain/internal error, public only if justified
```

Do not make one giant error enum containing every implementation failure.

Per crate targets:

- `g-interface`: move `ConfigError` from `lib.rs` to `src/error.rs`; expose
  `ConfigError` and `ConfigResult`.
- `g-plan`: expose `PlanError`; keep `HostPolicyError` and
  `PreparedPlanError` public only if callers need distinction.
- `g-genotype`: move `GenotypeError` out of `common.rs` to `src/error.rs`;
  expose `BgenError` only if BGEN remains public contract.
- `g-input`: replace public `Result<..., String>` with `InputError` and
  `InputResult`.
- `g-output`: rename or wrap `OutputWriterError` as `OutputError`; keep
  `OutputWriterError` as temporary alias only if needed.
- `g-runtime`: expose `RuntimeError` and `RuntimeResult`; move
  `RuntimeCompatibilityError` out of `runtime_state.rs`.
- `g-engine`: move root `EngineError` from `coordinator.rs` to `error.rs`;
  wrap backend/effect/schedule errors with phase/operation context.
- root `g`: add/keep `src/binding/errors.rs`; PyO3 converts Rust errors there.

Done when:

- no production crate public API returns `Result<T, String>`;
- no library crate public API returns `anyhow::Result`;
- PyO3 files do not invent ad hoc error conversion policies.

Guards:

```bash
rg "Result<[^>]+,\s*String>" crates
rg "anyhow::Result" crates/*/src
rg "map_err\(.*PyValueError|PyValueError::new_err|PyRuntimeError::new_err" src/binding
```

## Phase 3 - Low-Risk Crates

Clean small/contract crates before heavy domains.

### `g-cli`

Owns: native CLI frontend shell around `g-interface`.

Target files:

```text
crates/cli/src/
  main.rs
  lib.rs
  exit.rs
  signals.rs
```

Steps:

1. Keep public API to `NativeCliOutcome` and `dispatch_native_cli`.
2. Ensure signal handling internals are private.
3. Ensure CLI dispatch stays parse-only until native execution is deliberately
   implemented.
4. Do not add engine/runtime orchestration here.

Done when:

- `g-cli` depends only on `g-interface` plus process/signal crates;
- native binary entrypoint is the only downstream production user.

### `g-plan`

Owns: stable serializable run/config/prepared-plan DTOs.

Target shape:

```text
crates/plan/src/
  lib.rs
  api.rs
  error.rs
  enums.rs
  request/
  prepared/
  host_policy/
```

Steps:

1. Move public request/prepared/host-policy exports behind crate facade.
2. Split `request.rs` only when it is clearly reducing schema-bucket size.
3. Keep DTO construction deterministic and allocation-visible.
4. Keep operational code out: no BGEN, sample parsing, output sessions, JAX,
   telemetry writers, PyO3.
5. Add `PlanError` boundary as Phase 2 describes.

Done when:

- `g-plan` is boring contract code;
- no hot-path parsing, I/O, or JSON round trips live here.

### `g-interface`

Owns: CLI/TOML/Python option normalization and config-to-plan compilation.

Target shape:

```text
crates/interface/src/
  lib.rs
  api.rs
  error.rs
  config/
  options/
  toml/
  cli/
  compile/
```

Steps:

1. Move `ConfigError` to `error.rs`.
2. Keep defaults in `crates/interface/src/config.default.toml`.
3. Keep parse-time work pure: no filesystem/data scans.
4. Split `GComputeConfigData` into narrower config subdomains:
   compute runtime, native decode, binary numerics, linear numerics, JAX
   runtime, alignment.
5. Split config-to-plan compilation by domain:
   input, trait request, compute, correction, output, runtime, phenotype.
6. Remove dummy linear/binary mode leakage where possible. Quantitative runs
   should not carry binary-only Firth defaults.
7. Ensure `src/g/interface/config.py` normalizes Python input and calls PyO3
   only; no second option table.

Done when:

- downstream crates receive clean `g-plan` values, not giant config blobs;
- `g-interface` exposes config metadata/parse/validate/compile only.

## Phase 4 - `g-input`

Owns: sample, phenotype, covariate, prediction, phenotype-group alignment.

Target shape:

```text
crates/input/src/
  lib.rs
  api.rs
  error.rs

  sample/
    mod.rs
    keys.rs
    oxford_sample.rs
    identity.rs

  table/
    mod.rs
    reader.rs
    columns.rs
    missing.rs

  phenotype/
    mod.rs
    single.rs
    multi.rs
    binary_validation.rs

  covariate/
    mod.rs
    read.rs
    design.rs
    rank.rs

  alignment/
    mod.rs
    single.rs
    complete_case.rs
    per_phenotype.rs
    grouped.rs
    fingerprints.rs

  prediction/
    mod.rs
    list.rs
    loco.rs
    cache.rs
    source.rs

  grouping/
    mod.rs
    compute_group.rs
    union_samples.rs
```

Steps:

1. Add `InputError` and `InputResult`.
2. Replace public alignment `Result<..., String>`.
3. Split `sample/mod.rs` by domain without changing behavior.
4. Move union-sample and group-position planning into `g-input`.
5. Keep output row/matrix layout explicit and stable.
6. Keep BGEN decoding, output writing, JAX device handling, scheduler state,
   callback queues, and PyO3 classes out.

Done when:

- public API exposes aligned DTOs, sample-key mode, prediction sources/errors;
- downstream no longer cares about column names, missing tokens, FID/IID parse,
  or LOCO ordering;
- root PyO3 does not implement alignment decisions.

## Phase 5 - `g-genotype`

Owns: genotype source contracts, BGEN reader, chunk planning, preprocessing.

Target shape:

```text
crates/genotype/src/
  lib.rs
  api.rs
  error.rs

  source/
  bgen/
  decode/
  preprocess/
  buffer/
```

Steps:

1. Move `GenotypeError` to `error.rs`.
2. Decide if `BgenError` remains public. If yes, re-export error only, not the
   whole `bgen` module.
3. Add/strengthen `GenotypeSource` facade.
4. Move raw pointer APIs into `buffer::raw_pointer`.
5. Use typed internal buffers for normal crate APIs.
6. Split `ChunkStats` internally into allele, observation, dosage, and sparse
   candidate stats if it reduces coupling.
7. Keep SIMD/unsafe isolated in `preprocess/simd.rs` and buffer adapters.
8. Keep sample/phenotype alignment, output writers, runtime/JAX policy, engine
   scheduling, callback queues, and PyO3 classes out.

Done when:

- `g-engine` uses genotype facade, not `g_genotype::bgen::BgenReaderCore`;
- public hot APIs are chunk/batch-oriented;
- no cross-crate raw pointer contract except explicit low-level adapter.

## Phase 6 - `g-output`

Owns: output run prep, manifest compatibility, resume, chunk writing,
finalization.

Target shape:

```text
crates/output/src/
  lib.rs
  api.rs
  error.rs

  chunk/
    mod.rs
    handle.rs
    stats.rs
    metadata.rs
    arrays.rs

  session/
    mod.rs
    writer_session.rs
    coordinator.rs
    worker_pool.rs
    jobs.rs
    completion.rs

  manifest/
    mod.rs
    schema.rs
    fingerprint.rs
    prepare.rs
    validate.rs
    commit.rs
    metadata.rs
    contract.rs

  resume/
  writer/
  finalization/
  timing/
```

Steps:

1. Create/rename public crate error to `OutputError`.
2. Keep `OutputWriterError` alias only as short-term bridge if unavoidable.
3. Make `write_regenie2_native_chunk_handle_arrays(...)` or its successor the
   canonical internal write path.
4. Move slice-to-Arrow conversion to `chunk::arrays`.
5. Move `OutputStageTimingAccumulator` to `timing/accumulator.rs`.
6. Move writer-pool logic to `session/worker_pool.rs`.
7. Move manifest-to-prepared-plan reconstruction to `manifest/contract.rs`.
8. Remove old overloads once callers move.
9. Keep runtime telemetry sinks, engine scheduler queues, BGEN internals,
   sample alignment internals, and PyO3 classes out.

Done when:

- engine/Python say "write this chunk result";
- only output owns manifest JSON, Arrow field construction, resume/finalize;
- hot path writes chunk batches through handles/array views, not JSON.

## Phase 7 - `g-runtime`

Owns: runtime policy, logging, telemetry, timing, shutdown, Rayon/JAX policy,
trusted-validation cache.

Target shape:

```text
crates/runtime/src/
  lib.rs
  api.rs
  error.rs

  state/
  policy/
  services/
  telemetry/
  diagnostics/
  timing/
  paths.rs
```

Steps:

1. Add `RuntimeError` and `RuntimeResult`.
2. Move `RuntimeCompatibilityError` out of `runtime_state.rs`.
3. Split `runtime_state.rs` into process state, compatibility, token, policy,
   and service operations.
4. Split `run_events.rs` into typed event families.
5. Replace exported event-name constants with typed events where feasible:

```rust
pub enum TelemetryEvent {
    RunStarted(RunStartedFields),
    RunCompleted(RunCompletedFields),
    ExecutionPlanPrepared(ExecutionPlanPreparedFields),
    BgenEngineOpened(BgenEngineOpenedFields),
    SampleAlignmentCompleted(SampleAlignmentCompletedFields),
    WriterFinished(WriterFinishedFields),
    Diagnostic(DiagnosticEvent),
}
```

6. Keep string names private serialization details.
7. Keep event construction outside inner loops.
8. Keep phenotype/sample logic, BGEN internals, output writer internals, engine
   scheduler state, and PyO3 classes out.

Done when:

- runtime owns process side effects and observability only;
- core domains emit typed events/observers, not telemetry string registries.

## Phase 8 - `g-engine`

Owns: orchestration across genotype/input/output/runtime, scheduling, preflight,
backend seam, cleanup.

Target shape:

```text
crates/engine/src/
  lib.rs
  api.rs
  error.rs

  request/
  coordinator/
  backend/
  scheduler/
  delivery/
  effects/
  preflight/
  test_support/
```

Steps:

1. Move `EngineError` to `error.rs`; keep phase/operation context.
2. Split `schedule.rs` by domain:

```text
scheduler/chunk_batch.rs
scheduler/callback_queue.rs
scheduler/result_slots.rs
scheduler/dosage_buffer_pool.rs
scheduler/worker_lifecycle.rs
scheduler/backpressure.rs
delivery/bgen.rs
delivery/cleanup.rs
delivery/genotype_format.rs
```

3. Keep pure planning functions pure and grouped by domain.
4. Replace string action lists with enums. Serialize at PyO3/telemetry edge
   only.
5. Make `g-engine` depend on `g-genotype` facade, not BGEN internals.
6. Replace broad `EngineRunEffects` with smaller traits or an
   `EngineServices` struct:

```rust
pub struct EngineServices<I, O, T, P> {
    input: I,
    output: O,
    telemetry: T,
    preflight: P,
}
```

7. Make backend contracts real and batch-oriented:

```rust
pub struct GenotypeBatchView<'a> {
    metadata: VariantMetadataView<'a>,
    stats: ChunkStatsView<'a>,
    genotype: GenotypeMatrixView<'a>,
}

pub struct AssociationBatchResult<'a> {
    beta: ResultArrayView<'a>,
    standard_error: ResultArrayView<'a>,
    chi_squared: ResultArrayView<'a>,
    log10_p_value: ResultArrayView<'a>,
    extra_code: Option<ResultArrayView<'a>>,
}
```

8. Keep fake engines under `test_support`.
9. Keep direct JAX device-transfer logic and PyO3 classes out.

Done when:

- `g-engine` coordinates workflow, not low-level policies;
- scheduling/delivery/output cleanup are split and independently readable;
- public compute boundaries are chunk/batch-oriented.

## Phase 9 - Root PyO3 Crate and Python

Owns: Python facade only.

Target shape:

```text
src/binding/
  mod.rs
  error.rs

  config/
  runtime/
  engine/
  input/
  output/
  diagnostics/
```

Steps:

1. Add/keep central Rust-to-Python error conversion:

```rust
pub fn engine_error_to_pyerr(error: g_engine::EngineError) -> PyErr;
pub fn genotype_error_to_pyerr(error: g_genotype::GenotypeError) -> PyErr;
pub fn input_error_to_pyerr(error: g_input::InputError) -> PyErr;
pub fn output_error_to_pyerr(error: g_output::OutputError) -> PyErr;
pub fn runtime_error_to_pyerr(error: g_runtime::RuntimeError) -> PyErr;
```

2. Split `src/binding/mod.rs` long flat registration into domain modules only
   if it reduces registration complexity.
3. Shrink `src/binding/run_engine.rs`: it should wrap Rust handles and call Rust
   facades.
4. Remove PyO3 logic that opens BGEN, chooses sample-source behavior, builds
   compute groups, reconstructs sample indices, prepares manifests, schedules
   queues, or finalizes outputs.
5. Keep GIL attach/detach, NumPy/JAX buffer bridge, and Python class wrappers
   here.
6. Remove Python wrappers that only rename one native call and have one caller.

Done when:

- PyO3 is adapter layer only;
- run behavior lives in Rust domain owners or Python JAX kernels, not bindings;
- `_core.pyi` and production `_core` references match real registered classes.

## Phase 10 - Guards

Add or update architecture guards after cleanup, not before they would fail.

Useful guard checks:

```bash
rg "pub mod" crates/*/src/lib.rs
rg "Result<[^>]+,\s*String>" crates
rg "anyhow::Result" crates/*/src
rg "serde_json::Value|PyDict" crates/engine crates/genotype crates/input crates/output
rg "g_genotype::bgen::BgenReaderCore|g_genotype::planner" crates/engine src/binding
rg "PyValueError::new_err|PyRuntimeError::new_err" src/binding
rg "Fake|Mock" crates/*/src/lib.rs crates/*/src/api.rs
```

Expected final rules:

- public facade items documented in `PUBLIC_API.md`;
- no implementation `pub mod` from crate roots unless documented;
- no production `Result<T, String>` in crate public APIs;
- no `anyhow::Result` in library public APIs;
- no cross-crate JSON for hot compute paths;
- no public test scaffolding outside `test_support`;
- PyO3 error conversion centralized;
- root `g` bindings do not own domain policy.

## Recommended Execution Order

Use this order to reduce dependency churn:

1. Phase 0 inventory.
2. Phase 1 facade boundaries for `g-cli`, `g-plan`, `g-interface`.
3. Phase 2 error boundaries for `g-cli`, `g-plan`, `g-interface`.
4. Phase 3 low-risk crate cleanup.
5. Phase 4 `g-input`, because typed alignment errors unblock PyO3 cleanup.
6. Phase 5 `g-genotype`, because engine must move to genotype facade.
7. Phase 6 `g-output`, because engine write/finalize calls depend on it.
8. Phase 7 `g-runtime`, because engine telemetry/error reporting depends on it.
9. Phase 8 `g-engine`, after domain facades are ready.
10. Phase 9 root PyO3 cleanup.
11. Phase 10 guards.

If a later phase needs a facade from an earlier crate, add the smallest facade
first and continue. Do not tunnel through implementation modules.

## Stop Conditions

Stop and report if:

- a public API item has unclear owner after reading `PUBLIC_API.md`;
- deleting a wrapper would require changing user-facing CLI/config behavior;
- compile fixes require changing statistical behavior;
- a module split needs broad behavior rewrite instead of mechanical moves;
- a guard would block known transitional code outside current slice.

Report with:

```text
blocked slice
file/function
why ownership unclear
smallest proposed decision
commands already run
```

## Final Acceptance

Architecture cleanup complete when:

- every crate has a small documented facade;
- errors have crate-level boundaries;
- internal module names can change without cross-crate churn;
- root PyO3 is glue only;
- `g-engine` coordinates via facades and typed services;
- hot paths remain batch-oriented;
- architecture guards pass;
- docs and `PUBLIC_API.md` match implemented surfaces.

## Source Coverage Check

This plan keeps all durable signal from source plans:

- public API: one crate-root facade, `PUBLIC_API.md`, no accidental `pub mod`,
  no public fake/test support;
- ownership: `g-plan`, `g-interface`, `g-cli`, `g-genotype`, `g-input`,
  `g-output`, `g-runtime`, `g-engine`, root `g`;
- internals: split `g-engine::schedule`, `g-output::session`,
  `g-input::sample`, `g-runtime::run_events`; refine `g-genotype` facade;
  narrow `GComputeConfigData`;
- errors: crate `src/error.rs`, typed crate results, no public
  `Result<T, String>`, no library `anyhow::Result`, central PyO3 conversion;
- PyO3: adapter only; no run behavior, sample policy, manifests, scheduling,
  finalization;
- execution: phases, guards, stop conditions, final acceptance;
- scope: no tests/tooling work in this plan.
