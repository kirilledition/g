# Architecture Cleanup

| Status | Applies to | Owner |
| --- | --- | --- |
| Active consolidation plan | Native Rust workspace, PyO3 bindings, and Python orchestration | Development maintainers |

This page is the canonical cleanup and migration plan for `g`. It supersedes
the older scratchpad cleanup plans and the standalone Rust migration plan. Keep
new architectural policy and remaining cleanup steps here instead of creating
another parallel plan.

## Direction

The root Python package remains `g`, and the Maturin/PyO3 native module remains
`g._core`. The root Rust crate is a composition and PyO3 adapter crate. Internal
domain crates must stay Python-free.

Ownership is:

```text
crates/*    = Rust domain and application logic
src/binding = PyO3 adaptation for g._core
src/g       = public Python API, JAX setup, and JAX kernels
```

Target crate responsibilities are:

| Crate | Responsibility |
| --- | --- |
| `g-plan` | Immutable requested and prepared run contracts. |
| `g-interface` | Clap, TOML, defaults, overlays, validation, config-to-plan conversion, and native CLI frontend. |
| `g-genotype` | BGEN mmap/index/decode, genotype chunk planning, preprocessing, and genotype benchmarks. |
| `g-input` | Sample, phenotype, covariate, prediction-list, and LOCO alignment. |
| `g-output` | Output paths, Arrow/Parquet/REGENIE writing, manifests, resume, and finalization. |
| `g-runtime` | Logging, tracing, telemetry, timing, runtime policy, Rayon policy, and shutdown. |
| `g-engine` | Application state machine, preflight, batching, queues, backend trait, and cleanup. |

Do not introduce generic dumping-ground crates such as `g-utils`, `g-common`,
or `g-types`.

## Invariants

- Preserve sample order, phenotype and covariate masks, LOCO prediction
  alignment, allele orientation, output row order, correction selection,
  correction status, and fresh/resumed equivalence unless a separate
  science-change issue explicitly approves the difference.
- Keep `g --help`, `g regenie --help`, CLI exit codes, TOML merge/default
  behavior, `effective_config.toml`, PyO3 export names, and `_core.pyi` stable
  unless an interface-change issue approves the change.
- Keep crate dependencies acyclic and phase-aware.
- Do not keep permanent production Python fallbacks for Rust-owned behavior.
  Temporary dual implementations are allowed only for equivalence tests and must
  have a removal task.

## Policies

### Public Rust API

Each crate has exactly one public Rust facade: `api.rs`.

- Crate roots should declare private implementation modules and `pub use api::*`.
- `api.rs` contains public re-exports and type aliases only; it should not hold
  business logic.
- `PUBLIC_API.md` must describe every intentional public export group.
- `debug.rs` must not be a second public API. Do not expose `pub mod debug` from
  crate roots. If a transitional debug or compatibility export is still needed,
  either make it private, move the behavior behind a real production API in
  `api.rs`, or expose a clearly named temporary compatibility item from `api.rs`
  with a removal task.
- Test-only or benchmark-only internals should live under `#[cfg(test)]`,
  `test_support`, benches, or private modules rather than a public `debug`
  namespace.

### Binding Layer

`src/binding` is PyO3 adaptation only. If a function can be written using only
Rust crate types, it belongs in a crate. If it needs Python, PyO3, `PyAny`,
`PyErr`, `PyModule`, NumPy/Python buffers, or direct Python callback invocation,
it may live in `src/binding`.

Allowed in `src/binding`:

- `#[pyclass]` wrappers and `#[pymethods]` accessors.
- `#[pyfunction]` wrappers.
- Python module and submodule registration.
- Python callback invocation and callback-object extraction.
- NumPy/Python buffer adapters.
- Python-to-Rust data extraction and Rust-to-Python object construction.
- `PyErr` conversion.
- Temporary compatibility aliases.

Forbidden in `src/binding`:

- Scheduling policy.
- Callback worker lifecycle policy.
- BGEN delivery cleanup policy.
- Manifest/header construction policy.
- Output resume or repair planning.
- Preflight validation policy beyond converting Python arrays and calling crate
  validation.
- Run-event or diagnostic payload construction policy.
- Telemetry rendering policy.
- Genotype preprocessing policy.
- Sample alignment policy.
- Domain-level config validation.

Production exports should live under domain submodules:

```text
g._core.cli
g._core.config
g._core.runtime
g._core.telemetry
g._core.engine
g._core.genotype
g._core.input
g._core.output
```

`g._core.debug` is only for debug, compatibility, and migration internals. Root
`g._core` symbols are compatibility aliases only; new Python callers must use a
domain submodule.

### Python Ownership

Python keeps public user-facing APIs, JAX runtime setup boundaries, and JAX
kernels. Python should not own Rust-domain orchestration indefinitely.

Target Python-owned areas:

```text
src/g/api.py
src/g/runner/cli.py
src/g/runner/runtime.py
src/g/jax_runtime/
src/g/compute/
```

Migration targets to delete or demote after Rust ownership lands:

```text
src/g/execution_plan.py
src/g/engine/regenie2_pipeline/
src/g/engine/native_dispatch/
src/g/engine/callbacks/runtime.py
```

### Module Boundaries

`mod.rs` is a module boundary, not an implementation dumping ground.

- `lib.rs`: crate root declarations and facade re-export.
- `api.rs`: public facade, no heavy logic.
- `mod.rs`: child module declarations, local re-exports, and tiny glue only.
- named implementation files: real parsing, validation, orchestration,
  scheduling, I/O, and algorithms.

As a guideline, `mod.rs` should usually be under 100 lines. Files over 200 lines
or functions over 30 lines need a concrete reason or should be split.

### Integer Boundaries

The project does not use one integer type everywhere.

- In-memory Rust indexing, chunk lengths, matrix dimensions, queue sizes, and
  slice offsets use `usize`.
- Persistent schemas do not use `usize`.
- Python, JSON, TOML, manifest, Arrow, and Parquet numeric fields use
  fixed-width integers.
- Output statistic count columns use the output schema type. Current native
  chunk statistics keep signed `i32` count columns for compatibility.
- Genomic positions use `i64` unless a file format requires otherwise.
- File byte offsets use `u64` or parser-local checked `usize` after validating
  mapped buffer bounds.
- Large stored index arrays may use `u32` only after validation and benchmark
  evidence.
- Raw pointer addresses use `usize` only behind explicit wrappers such as
  `OutputBufferAddress` and `OutputValueCount`.
- Narrowing or sign-changing conversions use `TryFrom` or a named boundary
  helper.
- Unchecked integer `as` casts require an audited allowlist entry.

### Wrapper Policy

Remove wrapper chains that do not express a real boundary. Keep wrappers when
they represent:

- a public compatibility contract;
- an unsafe or FFI boundary;
- a JAX compiled-shape or donation boundary;
- a typed invariant;
- error-context conversion;
- a true facade over a subsystem.

Do not add one-line wrapper functions only to make a move look architectural.

## Current State

Already completed:

- Cargo workspace and target crates exist.
- `src/binding` is domain-organized.
- `_core.<domain>` submodules exist.
- `_core.debug` owns debug callback/schedule/preflight bindings.
- CLI uses `_core.cli.run_with_python_backend`.
- Binding layer policy, integer policy, integer audit, raw pointer wrappers,
  checked conversion helpers, and architecture checks exist.
- Main `g-engine`, `g-runtime`, and `g-output` facades are much narrower than
  the original audit state.
- Major CLI, runtime, telemetry, timing, output writer, and output preparation
  policies have moved into Rust.
- `g-genotype` exposes BGEN, buffer wrapper, preprocessing, and temporary tuning
  items only through `api.rs`; its public `debug`, `ffi`, and `internal`
  module facades have been removed.
- `g-engine`, `g-runtime`, and `g-output` expose public Rust APIs only through
  `api.rs`; their public `debug`, `events`, `trusted_validation`, and `admin`
  module facades have been removed.
- Phase 10 callback scheduler/resource ownership is mostly complete; Python
  still retains transitional JAX/backend wiring and some side-effect adapters.

Still active:

- Production Python still depends on `execution_plan.py`,
  `engine/regenie2_pipeline/`, and `engine/native_dispatch/`.
- Large PyO3 modules remain, especially `engine/run_engine.rs`,
  `telemetry/run_events.rs`, `output/mod.rs`, and `config/mod.rs`.
- `crates/input/src/sample/mod.rs` still contains real alignment logic.

## Remaining Roadmap

### 1. Binding Adapter Collapse

Continue shrinking PyO3 modules so they adapt rather than decide.

- `src/binding/engine/run_engine.rs`: move remaining scheduling, cleanup, and
  dispatch policy into `g-engine`.
- `src/binding/telemetry/run_events.rs`: expose typed handles or recorders, not
  every event builder.
- `src/binding/output/mod.rs`: keep writer/session/chunk handle adapters only.
- `src/binding/config/mod.rs`: keep PyO3 config classes, but move validation and
  planning policy to `g-interface`.

### 2. Native Engine Ownership

Move the remaining production orchestration out of Python.

- `NativeRunEngineSession` should own lifecycle, input/output preparation,
  dispatch orchestration, cleanup, telemetry, and finalization.
- Python should supply JAX backend setup and compute callbacks only.
- Remove production dependence on `src/g/execution_plan.py`,
  `src/g/engine/regenie2_pipeline/`, and `src/g/engine/native_dispatch/` after
  the native session owns equivalent behavior.

### 3. Callback Runtime Finalization

Finish removing transitional Python callback runtime ownership.

- Rust owns queue capacity, result slots, dosage buffers, worker lifecycle,
  writer handoff, progress, cleanup, and scheduling decisions.
- Python retains only local callback item storage/wakeups until the Rust-owned
  session can remove them.
- Delete or demote `src/g/engine/callbacks/runtime.py` and matching PyO3 debug
  resources when no production path uses them.

### 4. Module Boundary Cleanup

Move real logic out of `mod.rs` files.

- Start with `crates/input/src/sample/mod.rs`.
- Keep `mod.rs` as declarations and re-exports.
- Add or extend architecture checks for large `mod.rs`, logic-heavy `api.rs`,
  and allowlisted exceptions.

### 5. Integer Follow-Up

Do not restart a broad integer migration.

- Add missing output count overflow validation where production sites still
  produce signed count columns.
- Keep unaudited casts out of production.
- Introduce compact `u32` buffers only after benchmark evidence.

### 6. Enforcement

Extend existing architecture checks rather than adding isolated scripts.

- `api.rs` is the only public crate facade.
- No public `debug` modules in internal crate roots.
- Public export changes require `PUBLIC_API.md` updates.
- Production Python must not import `_core.debug`.
- Root `_core` compatibility aliases must be allowlisted and shrink over time.
- After native ownership lands, production Python imports from
  `regenie2_pipeline`, `native_dispatch`, and callback runtime are rejected.
- Integer casts remain checked or allowlisted.

## Superseded Work

Do not carry these forward as active tasks:

- Initial Cargo workspace extraction.
- `src/binding` rename and domain-folder reorganization.
- `_core` domain submodule creation.
- Binding policy creation.
- Debug binding relocation under `_core.debug`.
- CLI submodule migration.
- Integer policy, audit, helper, and raw-pointer setup.
- Initial public facade shrink for `g-engine`, `g-runtime`, and `g-output`.
- Output manifest/preparation migration already moved into Rust.
- Historical baseline failures unless current checks reproduce them.

## Validation

Documentation consolidation must run:

```bash
just docs-build
git diff --check
```

Code cleanup phases should run the narrowest relevant checks plus `just check`
before integration. Correctness-boundary phases need parity coverage,
manifest/schema snapshots, fresh-versus-resumed equivalence, and malformed input
tests. Hot-path phases need Criterion and representative CPU/GPU benchmarks on
the appropriate SLURM nodes.

## Stop Conditions

Pause rather than force a phase through when a dependency cycle appears, a leaf
crate needs PyO3, sample or prediction alignment changes unexpectedly, resume
mutates outputs before validation, parity changes cannot be attributed, or a
performance regression cannot be measured and explained.
