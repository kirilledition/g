# `g` Python-to-Rust and Multi-Crate Migration Plan

**Repository:** `kirilledition/g`<br>
**Baseline reviewed:** `main` at `25ec7701b8b4a9339d424d8d08b1bc4527ddb2c8`<br>
**Purpose:** Move application ownership from Python to Rust while introducing a Cargo workspace, preserving statistical behavior, public interfaces, reproducibility, and performance.

---

## 1. Executive direction

The migration should pursue this end state:

```text
Rust owns the application
├── CLI and TOML
├── config validation
├── requested and resolved execution plans
├── BGEN input and genotype preprocessing
├── sample/phenotype/covariate alignment
├── prediction-list and LOCO prediction handling
├── preflight validation
├── manifest, resume, output, and finalization
├── run state machine
├── queues, batching, buffers, and backpressure
├── logging, telemetry, timing, and shutdown
└── association-backend interface
    └── Python/JAX backend
        ├── prepare statistical state
        ├── prepare chromosome state
        └── compute association batches
```

Python should ultimately own only:

```text
- the public Python convenience API;
- JAX runtime setup that must happen in Python;
- JAX quantitative and binary kernels;
- a thin adapter implementing the Rust association-backend contract;
- optional Python-only tooling and benchmarks.
```

The migration must not optimize for Rust line count. It must optimize for:

1. one authoritative implementation per contract;
2. Rust ownership of mutable lifecycle state;
3. coarse and typed Rust/Python boundaries;
4. statistical parity;
5. safe resume semantics;
6. measured performance.

---

## 2. Non-negotiable invariants

These rules apply to every phase.

### 2.1 Statistical invariants

For behavior-neutral architecture changes:

- input sample order must remain identical;
- phenotype and covariate inclusion masks must remain identical;
- LOCO prediction alignment must remain identical;
- allele orientation and genotype flipping must remain identical;
- output row order must remain identical;
- correction candidate selection must remain identical;
- correction method and failure status must remain identical;
- public output values must be bitwise identical when practical, otherwise within existing documented tolerances;
- fresh and resumed runs must produce equivalent final results.

Any intentional statistical change must be a separate issue and PR.

### 2.2 Interface invariants

Unless a dedicated interface change is approved:

- `g --help` behavior stays stable;
- `g regenie --help` stays stable;
- CLI exit codes stay stable;
- unsupported options continue to fail clearly;
- TOML merge/default behavior stays stable;
- `effective_config.toml` stays semantically equivalent;
- the Python `g.api` surface remains stable;
- PyO3 export names remain stable;
- `_core.pyi` remains synchronized;
- output and manifest schema versions change only when the persisted contract changes.

### 2.3 Architecture invariants

- No internal domain crate may depend on PyO3 or NumPy.
- The root `g` package remains the PyO3/Maturin composition crate during migration.
- JAX modules must not be imported before runtime policy is configured.
- Core crates must not depend on the root binding crate.
- Crate dependency direction must remain acyclic.
- Do not solve dependency cycles with a generic `g-utils`, `g-common`, or `g-types` dumping ground.
- Do not keep permanent production Python fallbacks for Rust-owned behavior.

### 2.4 Delivery invariants

- Extracting a crate and changing behavior must be separate changes.
- Optimizing a path and moving it across a crate boundary must normally be separate changes.
- Each PR must be independently revertible.
- Each migration phase must remove or clearly deprecate the old production owner.
- Temporary dual implementations are allowed only for equivalence tests and must have a removal issue.

---

## 3. Target Cargo workspace

Keep the existing root package:

```text
package name: g
library target: _core
crate types: cdylib + rlib
purpose: Maturin/PyO3 composition and Python/JAX adapter
```

Introduce this workspace gradually:

```text
g/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs
│   └── python/
├── crates/
│   ├── plan/
│   │   └── Cargo.toml          # package g-plan
│   ├── interface/
│   │   └── Cargo.toml          # package g-interface
│   ├── genotype/
│   │   └── Cargo.toml          # package g-genotype
│   ├── input/
│   │   └── Cargo.toml          # package g-input
│   ├── output/
│   │   └── Cargo.toml          # package g-output
│   ├── runtime/
│   │   └── Cargo.toml          # package g-runtime
│   └── engine/
│       └── Cargo.toml          # package g-engine
└── optional later:
    └── crates/cli/
        └── Cargo.toml          # package g-cli, binary name g
```

Directories may use short names. Cargo package names should use the `g-*` prefix.

### 3.1 Intended responsibilities

#### `g-plan`

Pure immutable contracts:

- requested run configuration after frontend normalization;
- prepared/resolved execution plan;
- association mode;
- backend kind;
- requested and resolved genotype format;
- phenotype compute groups;
- output writer plan;
- runtime policy values;
- correction plan;
- dtype and numerical policy values;
- stable serialization of plan-affecting state.

It should have very few dependencies.

#### `g-interface`

- Clap parser;
- TOML parsing;
- packaged defaults;
- config overlay;
- aliases;
- validation;
- effective-config serialization;
- conversion from resolved config into `g_plan::RunRequest`.

It should not read BGEN data, open output writers, initialize JAX, or run analysis.

#### `g-genotype`

- BGEN mmap/index/metadata;
- BGEN decoding;
- genotype chunk planning;
- genotype preprocessing;
- chunk statistics;
- trusted structural validation mechanisms;
- genotype-specific benchmarks.

It should not know about phenotypes, output manifests, telemetry sessions, PyO3, or JAX.

#### `g-input`

- Oxford sample parsing;
- phenotype and covariate parsing;
- sample-key validation;
- sample alignment;
- complete-case and per-phenotype grouping;
- prediction-list parsing;
- LOCO file loading;
- LOCO sample alignment;
- prediction caches;
- sample/covariate/prediction fingerprints.

It should not own BGEN delivery, output writing, JAX, or process lifecycle.

#### `g-output`

- output paths;
- Arrow/Parquet/REGENIE schemas;
- output sessions;
- chunk commit records;
- manifests;
- resume validation and strict repair;
- atomic manifest writes;
- finalization.

It should not know about Python or JAX. It should consume prepared plan/input identities rather than discover analysis semantics itself.

#### `g-runtime`

- logging initialization;
- tracing;
- telemetry session;
- stage timing;
- profiling summaries;
- process-global runtime compatibility;
- Rayon policy;
- graceful shutdown;
- run IDs and lifecycle event sinks.

It should not perform association computation or input parsing.

#### `g-engine`

- application state machine;
- preflight;
- requested-plan to prepared-plan resolution;
- coordination of genotype, input, output, and runtime crates;
- batching and backpressure;
- committed-chunk skipping;
- buffer ownership;
- association-backend trait;
- error propagation and cleanup.

It should not depend on `g-interface`; it should consume `g_plan::RunRequest` or `PreparedRunPlan`.

#### Root `g` crate

- PyO3 classes and functions;
- conversion between Rust types and Python objects;
- NumPy/JAX buffer adapters;
- Python association-backend adapter;
- module registration.

No application policy should remain here.

---

## 4. Target dependency graph

```text
                         g-plan
                      /    |    \
                     /     |     \
          g-interface   g-input   g-output
                          |          |
                       g-genotype    |
                           \         /
                            \       /
                            g-engine
                               |
                           g-runtime
                               |
                root g / _core PyO3 composition
```

More precisely:

```text
g-plan       -> serde, thiserror
g-interface  -> g-plan, clap, toml, serde
g-genotype   -> memmap2, flate2, rayon, low-level decode dependencies
g-input      -> g-plan, csv, sha2
g-output     -> g-plan, arrow, parquet, serde_json, sha2
g-runtime    -> g-plan, tracing, chrono, signal-hook, rayon
g-engine     -> g-plan, g-genotype, g-input, g-output, g-runtime, crossbeam-channel
root g       -> all internal crates, pyo3, numpy
```

Preferred refinements:

- `g-engine` should not depend on `g-interface`.
- `g-output` should not depend on `g-input`; the prepared plan should carry input identities.
- `g-input` should avoid depending on `g-genotype` where simple identifier/value views suffice.
- `g-runtime` should not depend on `g-engine`.
- Only the root crate may depend on PyO3 and NumPy.

---

## 5. Two plan levels are required

Current behavior includes choices that can only be resolved after inputs are inspected. For example, `gpu_genotype_format=auto` can become packed8 or dosage after eligibility and trusted validation.

Do not force these choices into one static config-derived object.

Use two levels:

```rust
pub struct RunRequest {
    // User intent after defaults/TOML/CLI/Python normalization.
    // May contain Device::Gpu and GpuGenotypeFormat::Auto.
}

pub struct PreparedRunPlan {
    // Fully resolved after opening inputs and preflight.
    // Contains concrete backend, concrete genotype format,
    // aligned sample identities, prediction identities,
    // variant count, sample count, and output compatibility state.
}
```

The manifest must be built from `PreparedRunPlan`, not directly from `RunRequest`.

---

# Migration phases

## Phase 0 — Establish the migration baseline

### Objective

Create a reproducible correctness and performance baseline before moving files or ownership.

### Actions

1. Record the baseline commit.
2. Add or update `documentation/development/rust-migration.md`.
3. Create a migration ledger containing:
   - current Python owner;
   - current Rust helper;
   - target crate;
   - target owner;
   - old path removal status;
   - tests;
   - benchmarks.
4. Capture public interface snapshots:
   - `g --help`;
   - `g regenie --help`;
   - representative validation errors;
   - Python API construction examples;
   - effective config output;
   - run manifest output.
5. Generate golden tiny-fixture artifacts for:
   - quantitative single phenotype;
   - binary score-only;
   - binary approximate Firth, marked experimental;
   - multi-phenotype per-phenotype mode;
   - multi-phenotype complete-case mode;
   - Arrow, Parquet, and REGENIE output;
   - interrupted/resumed run.
6. Record output schemas, row counts, sample counts, plan hashes, and correction summaries.
7. Capture compilation metrics:
   - clean workspace build;
   - incremental rebuild after touching Rust binding code;
   - Maturin development build;
   - wheel size;
   - `_core` binary size.

### Required test baseline

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
cargo test --workspace
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests scripts tooling
uv run pytest tests/ -m "not phase0_data and not phase1_parity"
just check-core-stub
just check-internal-defaults
just check-internal-init-exports
just docs-build
```

Also run existing parity lanes and relevant SLURM tests.

### Required benchmark baseline

Store machine metadata and JSON summaries for:

```bash
just benchmark-rust
just benchmark-bgen-reader
just benchmark-callback-overhead
just benchmark-callback-overhead-gpu
just benchmark-output-stages-gpu
just benchmark-regenie2-linear-fresh-gpu
just benchmark-regenie2-binary-hot-gpu
```

Use the current chr10/chr22 matrices for correctness and representative end-to-end timings.

### Exit criteria

- Golden outputs are stored or reproducibly generated.
- Benchmark summaries are archived.
- Every later phase can compare against a named baseline.
- No migration code has begun.

---

## Phase 1 — Add workspace infrastructure only

### Objective

Introduce workspace mechanics without moving production code.

### Actions

1. Add to the root `Cargo.toml`:

```toml
[workspace]
members = ["."]
default-members = ["."]
resolver = "3"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.96.0"
publish = false
```

2. Move suitable shared dependency declarations into `[workspace.dependencies]`.
3. Add workspace lints and have each future crate opt in with:

```toml
[lints]
workspace = true
```

4. Keep all `[profile.*]` sections at the workspace root.
5. Keep the root package name `g` and library target `_core`.
6. Update Maturin/uv cache keys to cover:
   - `crates/**/*.rs`;
   - `crates/**/Cargo.toml`;
   - root and crate-local embedded config/resources.
7. Audit CI path filters for `crates/**`.
8. Add dependency-graph validation tooling that can fail CI when:
   - an internal crate depends on PyO3/NumPy;
   - a forbidden internal edge appears;
   - a dependency cycle is introduced.
9. Add non-mutating Just recipes:
   - `rust-format-check`;
   - `rust-lint-check`;
   - `rust-check`;
   - `workspace-check`.

### Tests

Run all baseline tests. Build both:

```bash
cargo build --workspace --all-targets
uv run maturin develop
uv run python -c "import g; import g._core"
uv run g --help
```

### Benchmarks

Only build/incremental-build metrics are required. Runtime behavior should be unchanged.

### Exit criteria

- Root package still builds through Maturin.
- Python import and CLI smoke pass.
- Workspace commands pass.
- No production module has moved.

---

## Phase 2 — Extract `g-genotype`

### Objective

Move the clearest low-level leaf domain first.

### Actions

1. Create `crates/genotype/Cargo.toml` with package name `g-genotype`.
2. Move `src/genotype/` to `crates/genotype/src/`.
3. Move BGEN-related benches to the crate's `benches/`.
4. Keep `src/pipeline/` in the root temporarily; it is orchestration, not genotype I/O.
5. Replace root imports with `g_genotype`.
6. Expose a narrow crate facade.
7. Do not convert every `pub(crate)` to `pub`. Create constructors, views, and explicit methods.
8. Remove PyO3-specific assumptions from genotype types.
9. Preserve all decode buffer layouts and statistics.

### Tests

- All existing Rust genotype tests.
- BGEN parsing corruption tests.
- chunk planning tests;
- missingness and imputation statistics;
- packed8 and dosage paths;
- trusted-validation behavior;
- root PyO3 tests that expose genotype data.

### Benchmarks

Mandatory:

```bash
cargo bench -p g-genotype
just benchmark-bgen-reader
```

Compare:

- decode throughput;
- preprocessing throughput;
- allocations;
- mmap/index startup;
- clean and incremental compile times.

### Exit criteria

- `g-genotype` has no PyO3, output, interface, or runtime dependency.
- Runtime output is unchanged.
- No statistically significant BGEN throughput regression is unexplained.

---

## Phase 3 — Extract `g-output`

### Objective

Move output persistence and resume into an isolated native crate.

### Actions

1. Create `g-output`.
2. Move `src/output/`.
3. Keep schema, manifest, resume, writer, session, and finalization together.
4. Move output-specific error types into the crate.
5. Update root PyO3 output adapters to call `g_output`.
6. Preserve manifest and schema serialization exactly.
7. Keep all atomicity guarantees.
8. Add crate-level fault-injection seams where useful.

### Tests

Mandatory coverage:

- fresh output initialization;
- non-empty directory rejection;
- fast resume;
- strict resume;
- missing chunk files;
- corrupted metadata;
- conflicting chunk commits;
- Arrow/Parquet/REGENIE finalization;
- output dtype variants;
- manifest schema compatibility;
- incompatible resume leaves previous artifacts unchanged;
- multi-phenotype all-or-nothing compatibility preflight.

Use byte-for-byte manifest snapshots where the schema is intentionally unchanged.

### Benchmarks

```bash
just benchmark-output-stages-gpu
```

Also add Rust-level writer benchmarks for:

- enqueue throughput;
- grouping multiple chunks per output file;
- Arrow compression;
- Parquet writing/finalization;
- manifest write frequency.

### Exit criteria

- `g-output` has no PyO3, JAX, interface, or engine dependency.
- All resume and finalization behavior is preserved.
- Output throughput and finalization time remain within the agreed regression budget.

---

## Phase 4 — Extract `g-interface`

### Objective

Isolate the already Rust-owned frontend.

### Actions

1. Create `g-interface`.
2. Move:
   - CLI parser;
   - option metadata;
   - config domain;
   - defaults;
   - partial/resolved config;
   - overlay;
   - TOML;
   - validation;
   - run validation.
3. Move `config.default.toml` into the crate.
4. Keep Rust option metadata canonical.
5. Root PyO3 config classes become adapters around `g_interface`.
6. Preserve the public config and help surface.

### Tests

- CLI help snapshots;
- invalid option and invalid value tests;
- default/TOML/CLI precedence;
- explicit negative booleans;
- unknown key rejection;
- Python `from_options` aliases;
- deterministic TOML serialization;
- effective-config roundtrip;
- unsupported behavior errors.

### Benchmarks

Only startup-sensitive measurements are needed:

- `g --help`;
- `g regenie --help`;
- config parsing;
- Python import;
- Maturin build.

### Exit criteria

- `g-interface` depends on neither input data nor output/runtime/engine crates.
- CLI and config golden snapshots are unchanged.
- Python API configuration tests pass.

---

## Phase 5 — Extract `g-input`

### Objective

Create one owner for non-genotype scientific inputs.

### Actions

1. Create `g-input`.
2. Move native sample alignment code.
3. Move prediction-list and LOCO prediction code.
4. Move:
   - tabular parsing;
   - sample-key validation;
   - binary phenotype recoding;
   - complete-case/per-phenotype grouping;
   - prediction caches;
   - sample/covariate/prediction fingerprints;
   - LOCO input fingerprints.
5. Keep BGEN embedded sample retrieval in `g-genotype`; pass identifier views into `g-input`.
6. Ensure fingerprint caching is run-scoped and path-canonicalized.

### Tests

- malformed selected columns;
- true missing values;
- duplicate IID and FID/IID;
- embedded and sample-file identifiers;
- sample order;
- binary recoding;
- complete-case versus per-phenotype semantics;
- prediction-list parsing;
- LOCO sample alignment;
- chromosome normalization;
- prediction matrix caching;
- LOCO content change invalidates resume identity;
- shared LOCO files are hashed/loaded once.

### Benchmarks

Add or retain benchmarks for:

- large phenotype/covariate parsing;
- alignment across many samples;
- many-phenotype grouping;
- LOCO load and alignment;
- repeated phenotype headers with fingerprint cache;
- startup for 1, 10, 100, and 1,000 phenotypes.

### Exit criteria

- `g-input` is Python-free.
- Sample and prediction parity is exact.
- Multi-phenotype startup does not scale through repeated file hashing unnecessarily.

---

## Phase 6 — Introduce `g-plan`

### Objective

Create canonical Rust execution contracts and remove Python planning authority.

### Actions

1. Create `g-plan`.
2. Define stable value types.
3. Introduce:
   - `RunRequest`;
   - `PreparedRunPlan`;
   - `PhenotypeRunPlan`;
   - `PhenotypeComputeGroup`;
   - `AssociationBackendPlan`;
   - `OutputWriterPlan`;
   - `RuntimePlan`;
   - `CorrectionPlan`;
   - input identity/fingerprint structures.
4. Make `g-interface` compile resolved config into `RunRequest`.
5. Make dynamic input/backend resolution produce `PreparedRunPlan`.
6. Move existing host policy and backend-planning logic into Rust.
7. Preserve the distinction between requested `gpu_genotype_format=auto` and resolved `dosage`/`packed8`.
8. Build manifests from `PreparedRunPlan`.
9. Temporarily adapt the Rust plan into old Python dataclasses for tests/API compatibility.
10. Delete Python planning logic once equivalence tests are complete.

### Tests

Create a plan matrix covering:

- quantitative and binary;
- CPU and GPU;
- requested auto/dosage/packed8;
- eligible and ineligible packed8 paths;
- score-only and approximate Firth;
- single phenotype;
- grouped per-phenotype;
- complete-case;
- all output formats;
- all resume modes;
- float32 and float64 output statistics.

For a temporary transition period, compare normalized old Python and new Rust plans field by field.

### Benchmarks

Plan construction is startup-only. Use a microbenchmark, but prioritize correctness. Measure multi-phenotype plan construction and manifest preparation.

### Exit criteria

- Python no longer determines backend, grouping, or writer plan.
- `PreparedRunPlan` is the manifest authority.
- Temporary dual-plan code is removed from production.

---

## Phase 7 — Extract and strengthen `g-runtime`

### Objective

Create the Rust runtime foundation before moving the full engine state machine.

### Actions

1. Create `g-runtime`.
2. Move native:
   - logging;
   - telemetry policy;
   - timing state;
   - shutdown metadata;
   - process runtime policy;
   - run metadata;
   - profiling summaries.
3. Replace payload-building helpers with typed Rust objects.
4. Introduce RAII handles:
   - `RunRuntime`;
   - `TelemetrySession`;
   - `StageTimingRecorder`;
   - `ShutdownController`.
5. Initially preserve Python-owned side effects through adapters.
6. Add explicit runtime compatibility checks before output preparation.
7. Move remaining side-effect ownership later after `g-engine` is stable.

### Tests

- compatible repeated runs;
- incompatible logging policy;
- incompatible Rayon count;
- incompatible JAX policy;
- logging initialization failure;
- telemetry writer failure;
- clean close;
- close after another error;
- first and second signal behavior;
- timing snapshot consistency.

### Benchmarks

- startup/initialization;
- event throughput;
- timing recorder disabled fast path;
- timing recorder enabled path;
- signal-controller overhead;
- logging perturbation.

### Exit criteria

- Runtime state is represented by Rust types.
- Python holds handles, not independent mutable copies.
- No output mutation occurs before runtime compatibility checks.

---

## Phase 8 — Create `g-engine` with a fake backend

### Objective

Build the native coordinator without immediately coupling it to JAX.

### Actions

1. Create `g-engine`.
2. Move `src/pipeline/` into it.
3. Define the run state machine:

```rust
enum RunPhase {
    Planned,
    InputsOpened,
    InputsAligned,
    PreflightValidated,
    OutputsInitialized,
    Running,
    Draining,
    Finalizing,
    Completed,
    Interrupted,
    Failed,
}
```

4. Define an association backend trait:

```rust
pub trait AssociationBackend {
    type GroupState;
    type ChromosomeState;

    fn prepare_group(
        &mut self,
        input: &PreparedGroupInput,
    ) -> Result<Self::GroupState, BackendError>;

    fn prepare_chromosome(
        &mut self,
        group: &Self::GroupState,
        chromosome: &str,
        predictions: PredictionView<'_>,
    ) -> Result<Self::ChromosomeState, BackendError>;

    fn compute_batch(
        &mut self,
        chromosome: &Self::ChromosomeState,
        batch: GenotypeBatchView<'_>,
    ) -> Result<AssociationBatchResult, BackendError>;
}
```

5. Implement a deterministic Rust fake backend for tests.
6. Have the fake backend produce simple known statistics and configurable failures.
7. Test the coordinator independently of Python, JAX, and GPU availability.

### Tests

Fault injection at every phase:

- input open failure;
- alignment failure;
- preflight failure;
- output compatibility failure;
- writer construction failure;
- backend prepare failure;
- batch failure;
- writer failure;
- interruption;
- finalization failure;
- telemetry failure.

Verify cleanup, manifest state, and writer abortion.

### Benchmarks

Only coordinator overhead with fake backend. It should be negligible relative to I/O and compute.

### Exit criteria

- A complete tiny run can execute using only Rust crates and the fake backend.
- The engine is testable without Python.
- All state transitions are explicit and covered.

---

## Phase 9 — Move preflight and run preparation into `g-engine`

### Objective

Remove Python ownership of input opening, alignment, prepared-plan construction, manifest preparation, and writer initialization.

### Actions

Move from Python pipeline/native-dispatch modules:

- BGEN engine open;
- required chromosome resolution;
- sample alignment;
- prediction loading;
- preflight;
- trusted validation decisions;
- concrete GPU genotype format resolution;
- prepared plan construction;
- manifest-header construction;
- all-manifest compatibility preflight;
- output run initialization;
- writer-session construction.

The root/Python side should receive a prepared native run handle rather than raw paths, manifests, arrays, and policies.

### Tests

- single and multi-phenotype preparation;
- auto packed8 resolution;
- trusted validation modes;
- prediction identity;
- all manifests validated before any mutation;
- resume behavior;
- prepared-plan snapshots;
- comparison against the previous Python pipeline preparation path.

### Benchmarks

- cold startup;
- BGEN open/index;
- alignment;
- LOCO loading;
- manifest/fingerprint preparation;
- many-phenotype preparation;
- fresh process startup benchmark.

### Exit criteria

- Python no longer opens scientific input files.
- Python no longer initializes output runs.
- Python no longer constructs manifest dictionaries.
- Python receives a ready compute request from Rust.

---

## Phase 10 — Move callback scheduling, queues, and buffer ownership into Rust

### Objective

Remove Python as the chunk-level scheduler.

### Actions

1. Move bounded staging queues into `g-engine`.
2. Move result queues and writer dispatch into Rust.
3. Move:
   - worker startup and join;
   - backpressure;
   - buffer pools;
   - chunk batching;
   - committed-chunk skipping;
   - result draining;
   - failure propagation;
   - writer completion/abort;
   - summary accumulation.
4. Implement a PyO3-backed `AssociationBackend`.
5. Use coarse backend calls—one call per batch, never per variant.
6. Pass typed objects or buffer views, not nested dictionaries.
7. Release the GIL during Rust I/O, queue waits, decode, and output writing.
8. Keep device conversion and JAX invocation in Python.
9. Preserve callback batch-size semantics.

### Current implementation notes

- Callback producer backpressure attempts now use native scheduler methods that
  select the default poll timeout internally for dosage queues, result queues,
  result in-flight slots, and dosage-buffer acquisition.
- Callback queue, result-slot, and dosage-buffer pool timing observations now
  use scheduler-owned native planners for occupancy capacity and observation
  validation.
- Binary correction summary telemetry now uses native summary plans for
  diagnostics retention, pending flush decisions, pending chunk counts, and
  final emission gating.
- Result work-item cleanup now uses native release plans to select pre-write
  host buffer release and final in-flight slot cleanup decisions.
- Result write drain completion now uses a native stop/diagnostic-flush plan
  when callback result consumers observe the queue sentinel.
- Dosage work drain completion now uses a native stop plan when callback
  dosage consumers observe the queue sentinel.
- Callback worker error updates now use native scheduler plans for record/clear
  state transitions and formatted error messages.
- Dosage work handoffs now use native scheduler plans for non-empty chunk
  counts before callback queueing.
- Result write handoffs now use native scheduler plans to classify result items
  and writer stop signals before callback queueing.
- Result write item dispatch now uses native scheduler plans to validate
  single-result versus multi-result consumer paths after queue drain checks.
- Dosage work item dispatch now uses native scheduler plans to select
  sample-major, variant-major, variant-major batch, and packed8 processing
  paths after queue drain checks.
- Dosage work stage timing attribution now uses native scheduler plans to split
  batch-level elapsed durations across queued chunk work items.
- Result in-flight slot acquire timing now uses a native scheduler observation
  plan to select acquire versus producer-blocking accounting.
- Callback worker start, finish, and abort dispatch now consume native
  lifecycle selectors instead of matching action strings in Python.
- Callback queue producer put timing now uses native scheduler observation
  plans to select put versus producer-blocking accounting.
- Callback queue consumer wait timing now uses native scheduler observation
  plans for dosage and result queue consumers.
- Result in-flight slot release timing now uses a native scheduler observation
  plan for release accounting.
- Dosage-buffer pool operation timing now uses native scheduler observation
  plans for reuse, return, allocation, discard, and consumer wait accounting.
- A PyO3 native bounded callback object queue primitive now owns FIFO storage,
  capacity checks, and blocking put/get waits for future queue-runtime swaps.
- Callback dosage/result queue storage and wait predicates now use the native
  object queue while the native scheduler remains the occupancy authority.
- Result in-flight slot and dosage-buffer pool waits now use native
  generation-counted wait signals instead of Python condition variables.
- Free host dosage-buffer storage now uses the native callback object queue,
  leaving Python to inspect buffer shape and dtype before reuse or discard.
- Callback worker thread handles now use a native PyO3 wrapper for thread
  construction, start, liveness checks, and joins while Python still supplies
  coarse dosage/result worker targets.
- Callback runtime resources now expose production worker names and liveness
  reads for public status and shutdown-timeout reporting.
- Callback runtime resources now use one native PyO3 owner to construct and
  retain scheduler state, progress state, queues, wait signals, binary summary
  state, worker handles, and the worker-start lock for production runners.
- Production callback runners now resolve scheduler state from the native
  runtime-resource owner, leaving direct scheduler assignment only for manual
  fallback runners.
- Production callback runners now resolve progress state from the native
  runtime-resource owner, leaving direct progress-state assignment only for
  manual fallback runners.
- Production callback runners now resolve binary correction summary state from
  the native runtime-resource owner, leaving direct summary assignment only for
  manual fallback runners.
- Production callback runners now resolve callback queues, free-buffer storage,
  and wait signals from the native runtime-resource owner, leaving direct handle
  assignment only for manual fallback runners.
- Production callback runners now resolve callback worker thread handles from
  the native runtime-resource owner, leaving direct handle assignment only for
  manual fallback runners.
- Callback runtime resources now own production dosage/result queue put,
  backpressure, and get loops, including scheduler slot rollback on native
  storage inconsistencies.
- Callback runtime resources now own production current-queue/resource
  backpressure observation planning while Python keeps timing emission.
- Callback runtime resources now own production dosage-buffer pool observation
  planning while Python keeps free-buffer count calculation and timing emission.
- Callback runtime resources now own production dosage/result queue put/get
  observation planning while Python keeps queue timing emission.
- Callback runtime resources now own production timed dosage/result queue
  producer put attempts and observation selection in one native call while
  Python keeps timing measurement and emission.
- Callback runtime resources now own production timed dosage/result queue
  consumer get attempts, drain decisions, and observation selection in one
  native call while Python keeps timing measurement and emission.
- Callback runtime resources now own production untimed dosage/result queue
  consumer get attempts and drain decisions in one native call without building
  timing observation plans.
- Callback runtime resources now own production callback limit, queue/resource
  occupancy, and free-buffer reads while Python keeps public runner property
  accessors.
- Callback runtime resources now own production progress state reads, per-chunk
  records, and completion while Python keeps telemetry emission.
- Callback runtime resources now own production binary correction summary
  counters, retention/emit plans, and payload reads while Python keeps pending
  diagnostics materialization and telemetry emission.
- Callback runtime resources now own production dosage/result drain planning
  and validated work-item dispatch planning, leaving Python to classify work
  item dataclasses and perform JAX/write side effects.
- Callback runtime resources now own production dosage work handoff planning,
  including variant-major batch handoff validation, while Python keeps the
  temporary callback work-item dataclasses.
- Callback runtime resources now own production dosage work stage-duration
  attribution planning while Python keeps timing emission.
- Callback runtime resources now own production worker-error scheduler state
  updates while Python keeps the original exception objects for chaining.
- Callback runtime resources now own production worker stop and join loops,
  including sentinel enqueue retries and native thread joins, while Python keeps
  the public shutdown exception type.
- Callback runtime resources now own production result in-flight slot acquire
  attempts, native wait-signal waits, releases, and release notifications while
  Python keeps worker-error chaining and timing emission.
- Callback runtime resources now own production untimed result in-flight slot
  acquire attempts without building timing observation plans.
- Callback runtime resources now own production dosage-buffer registration,
  return eligibility planning, free-buffer returns, free-queue storage
  failures, discard accounting, and buffer-pool wakeups while Python keeps
  NumPy allocation and shape/dtype reuse checks.
- Callback runtime resources now own production timing-enabled dosage-buffer
  register, return, and discard observation selection in the same native call
  as the buffer-pool state mutation.
- Callback runtime resources now own production dosage-buffer acquisition
  attempts, free-buffer pops, wait-signal waits, and wait result accounting
  while Python keeps NumPy allocation and shape/dtype reuse checks.
- Callback runtime resources now own production dosage-buffer reuse shape
  planning while Python keeps dtype checks and NumPy view slicing.
- Callback runtime resources now own production worker finish and abort
  lifecycle execution, including stop/join sequencing and worker-error raise
  planning, while Python keeps public shutdown exceptions, progress completion,
  summary emission, and exception chaining.
- Callback runtime resources now own production result work-item resource
  cleanup, including host-buffer returns and result in-flight slot releases,
  while Python keeps materialization, writes, and timing emission.
- Callback runtime resources now own production result cleanup buffer-pool
  return observation selection in the same native call as host-buffer cleanup.
- Native telemetry run sessions now own production run completed, interrupted,
  and failed lifecycle event emission names, levels, and field construction.
- CLI runtime-initialization failures now route run-failed telemetry through
  the native run session instead of manually emitting the event payload.
- Native telemetry run sessions now own production run-started and
  execution-plan-prepared lifecycle event names, levels, and field construction.
- Native telemetry run sessions now own production effective-config-written and
  writer-finished lifecycle event names, levels, and field construction.
- Native telemetry run sessions now own production preflight-completed event
  names, levels, and single-/multi-phenotype field construction.
- Native telemetry run sessions now own production sample-alignment-completed
  and prediction-source-loaded event names, levels, and optional fields.
- Native telemetry run sessions now own production multi-phenotype sample
  summary event names, levels, and derived sample-set fields.
- Native telemetry run sessions now own production GPU genotype-format
  resolution event names, levels, and optional fallback fields.
- Native telemetry run sessions now own production association-backend-selected
  and BGEN-engine-opened event names, levels, and optional phenotype fields.
- Native telemetry run sessions now emit native callback progress event objects
  directly for chromosome start/completion telemetry.

### Tests

- queue backpressure;
- staging depth;
- native callback batch size;
- result in-flight limits;
- dosage buffer limits;
- committed-chunk skip behavior;
- buffer reuse;
- Python exception propagation;
- worker panic/error propagation;
- interrupt during decode;
- interrupt during compute;
- interrupt during writer drain;
- single and multi-trait behavior;
- dosage and packed8 paths.

### Benchmarks

Mandatory before and after:

```bash
just benchmark-callback-overhead
just benchmark-callback-overhead-gpu
just benchmark-regenie2-binary-hot-gpu
just benchmark-output-stages-gpu
```

Profile:

- Python calls per chunk;
- queue wait;
- GIL wait;
- H2D and D2H time;
- GPU idle time;
- memory use;
- batch-size/staging-depth grid;
- throughput by chunk size.

### Exit criteria

- Rust owns all queues and workers.
- Python is called only for coarse JAX operations.
- No material hot-path regression is unexplained.
- Memory remains bounded under configured limits.

---

## Phase 11 — Move remaining runtime side effects into Rust

### Objective

Make `g-runtime` the actual owner, not just the type/payload owner.

### Actions

Move to Rust:

- telemetry file creation/writes;
- lifecycle event emission;
- logging sink ownership;
- timing/profile file writes;
- process-global runtime state;
- signal handler ownership for the CLI path;
- first/second signal state;
- graceful drain coordination;
- final flush and close.

Python/JAX should emit typed diagnostic events through a native handle.

For the Python API, define signal semantics explicitly. Do not silently override host-application signal handlers unless the API contract allows it.

### Tests

- CLI first/second signal;
- Python API signal policy;
- logging failures;
- telemetry failures;
- shutdown during every run phase;
- repeated runs;
- final flush;
- no duplicate run-failed events.

### Benchmarks

- disabled telemetry fast path;
- profile/trace mode;
- logging perturbation;
- graceful shutdown latency.

### Exit criteria

- Runtime lifecycle is Rust-owned.
- Python no longer owns telemetry or shutdown session state.
- CLI shutdown is deterministic and tested.

---

## Phase 12 — Collapse the root crate into a PyO3/JAX adapter

### Objective

Remove application policy from the root crate.

### Actions

The root should approach:

```text
src/
├── lib.rs
└── python/
    ├── module.rs
    ├── config.rs
    ├── plan.rs
    ├── engine.rs
    ├── backend.rs
    ├── arrays.rs
    └── errors.rs
```

1. Remove root copies of domain modules.
2. Re-export or adapt only intentional public PyO3 objects.
3. Keep `_core` export names stable.
4. Replace large registration modules with focused submodule registration if useful.
5. Ensure all Rust logic is in internal crates.
6. Delete temporary Python fallback implementations.

### Tests

- `_core.pyi` sync;
- Python API;
- CLI bridge;
- object lifetime and ownership;
- exceptions;
- NumPy buffer layouts;
- no duplicate exported names;
- wheel build/import.

### Benchmarks

- Maturin build;
- import time;
- wheel size;
- `_core` size;
- Python/Rust call count;
- end-to-end smoke.

### Exit criteria

- Root crate contains only binding/composition code.
- Internal crates compile without Python.
- No production fallback path exists.

---

## Phase 13 — Add a native CLI only after the Rust coordinator is complete

### Objective

Make Rust the process owner without prematurely embedding Python around a Python-owned pipeline.

### Actions

1. Create `g-cli` only when `g-engine` can run the full lifecycle.
2. Binary name remains `g`.
3. Use `g-interface` for CLI/TOML.
4. Use `g-engine` for execution.
5. Load or embed the Python/JAX backend through a deliberate adapter.
6. Preserve the Python package for JAX kernels and Python API users.
7. Decide packaging only after a prototype proves:
   - Python environment discovery;
   - JAX import and device setup;
   - signal behavior;
   - wheel/binary installation;
   - cluster execution.

A transitional option is to retain the tiny Python console entry while it calls one coarse native `run` function.

### Tests

- native and Python-bridge CLI help equivalence;
- exit codes;
- runtime failures;
- signals;
- config/TOML behavior;
- installation;
- CPU/GPU environment discovery.

### Benchmarks

- CLI startup;
- help latency;
- cold run startup;
- process memory;
- packaging size.

### Exit criteria

- Rust owns the CLI process lifecycle.
- The Python CLI shim can be removed or retained solely as compatibility glue.

---

## Phase 14 — Remove obsolete Python orchestration

### Objective

Complete the host migration and prevent regression.

### Candidate Python packages to delete or radically reduce

```text
g/runner/
g/engine/native_dispatch/
g/engine/regenie2_pipeline/
g/engine/callbacks/runtime.py
g/engine/callbacks/writers.py
most of g/engine/telemetry.py
most of g/engine/timing.py
most of g/engine/shutdown.py
most of g/io/output.py
most of g/execution_plan.py
```

Keep:

```text
g/api.py
g/interface/config.py       # tiny adapter
g/jax_runtime/
g/compute/
thin JAX backend adapter
```

### Guardrails

Add checks that fail when:

- production Python writes manifests;
- production Python owns native queues;
- internal crates add PyO3;
- Python reconstructs canonical plans;
- Python fallback implementations are reintroduced;
- JAX kernels read CLI/config/files directly.

### Exit criteria

- Rust owns all non-kernel production behavior.
- Python is a compute backend and public convenience layer.
- Documentation reflects the new architecture.

---

## Phase 15 — Optional independent Rust compute backends

This is a separate program, not part of the host migration.

Recommended order:

1. linear CPU reference backend;
2. binary score-only CPU reference backend;
3. optimized Rust CPU backend;
4. accelerator backend;
5. approximate Firth last.

The first Rust compute implementations should be correctness oracles, not immediate replacements for JAX.

Do not port approximate Firth merely to increase Rust percentage.

---

# 6. Test strategy by migration risk

## Tier A — Every PR

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
cargo test -p <affected-crate>
uv run ruff format --check <changed-python-paths>
uv run ruff check <changed-python-paths>
uv run ty check <changed-python-paths-and-tests>
targeted pytest
just check-core-stub
git diff --check
```

## Tier B — Every crate extraction

Also run:

```bash
cargo build --workspace --all-targets
cargo test --workspace
uv run pytest tests/ -m "not phase0_data and not phase1_parity"
just check-internal-defaults
just check-internal-init-exports
uv run maturin develop
uv run g --help
```

## Tier C — Every correctness-boundary migration

Required for changes to input, plan, output, resume, correction status, or scheduler:

- full CPU product suite;
- parity harness;
- tiny end-to-end CLI fixture;
- fresh versus resumed equality;
- multi-phenotype equivalence;
- manifest snapshots;
- output schema snapshots;
- malformed/corrupted input tests;
- fault injection.

## Tier D — Every hot-path migration

Required for genotype decode, queues, batching, transfers, or writer changes:

- Criterion benchmark for affected Rust code;
- callback-overhead benchmark;
- representative CPU benchmark;
- representative GPU benchmark;
- memory/backpressure measurement;
- stage-timing comparison.

## Tier E — Major milestones

Run after Phases 5, 9, 10, 12, and 13:

- full test suite;
- Rust and Python coverage;
- docs build;
- chr10/chr22 matrix;
- REGENIE parity suite;
- cold and hot benchmarks;
- deep profiling when a major performance boundary changed.

---

# 7. Benchmark policy

## 7.1 Benchmark before optimizing

Never mix a migration with an unmeasured optimization.

For every performance-sensitive phase:

1. benchmark the current parent commit;
2. make the structural change;
3. benchmark the new commit on the same node/environment;
4. compare stage-level and headline results;
5. only then optimize.

## 7.2 Record environment

Every summary should include:

- commit SHA;
- CPU model;
- GPU model and driver;
- JAX/JAXLIB version;
- Rust version;
- Python version;
- thread counts;
- chunk size;
- staging depth;
- callback batch size;
- output settings;
- dataset identity.

## 7.3 Recommended initial regression policy

For behavior-neutral structural changes:

- exact correctness is mandatory;
- investigate any reproducible hot end-to-end regression above 2%;
- investigate memory growth above 5%;
- treat cold JAX timing as noisy and use repeated trials plus stage attribution;
- permit a once-per-run helper slowdown only when its absolute cost is negligible and architecture materially improves;
- never accept a regression by citing a single noisy run.

## 7.4 Compile-time benchmarks

Track:

- clean `cargo build --workspace --all-targets`;
- clean Maturin build;
- incremental rebuild after touching:
  - PyO3 adapter;
  - genotype crate;
  - output crate;
  - engine crate;
- test-link time;
- final binary/wheel size.

The multi-crate design should improve dependency isolation even if full fat-LTO builds do not become faster.

---

# 8. Coverage strategy

The current project has 90% Python and Rust line-coverage gates. Preserve them throughout migration.

As Python code is deleted:

- Python coverage denominator should shrink naturally;
- do not add exclusions merely because code is transitional.

For Rust:

- keep `cargo llvm-cov --workspace --all-targets`;
- add crate-specific coverage reports during extraction;
- prioritize branch coverage around:
  - resume;
  - input validation;
  - state transitions;
  - shutdown;
  - error cleanup.

Coverage is not sufficient for numerical correctness; parity tests remain mandatory.

---

# 9. Pull request and issue structure

Create one epic for the complete transition, then one issue per phase and smaller implementation issues within each phase.

Preferred PR sequence for a domain:

```text
PR 1: Mechanical crate extraction
PR 2: Typed API cleanup and equivalence tests
PR 3: Move Python ownership into the crate
PR 4: Delete old Python production path
PR 5: Performance optimization, only if measurements justify it
```

Do not combine all five.

Every issue should include:

- current owner;
- target owner;
- files involved;
- forbidden behavior changes;
- required tests;
- required benchmarks;
- acceptance criteria;
- rollback plan;
- follow-up deletion task if dual paths are temporarily required.

---

# 10. Dependency and API enforcement

Add a repository check that reads `cargo metadata` and enforces:

```text
Only root g may depend on pyo3 or numpy.

g-plan may not depend on another workspace crate.
g-interface may depend only on g-plan.
g-genotype may not depend on interface, input, output, runtime, engine, or root.
g-input may not depend on interface, output, runtime, engine, or root.
g-output may not depend on interface, input, runtime, engine, or root.
g-runtime may not depend on interface, genotype, input, output, engine, or root.
g-engine may depend on all internal domain crates, but not root.
root g may depend on all internal crates.
```

Adjust only when a concrete domain requirement justifies it.

Add an import-policy check for Python:

```text
g.compute must not import CLI, output, or file parsers.
g.jax_runtime must not import runner orchestration.
Production Python must not write run manifests after Phase 10.
Production Python must not create native worker queues after Phase 10.
```

---

# 11. Specific repository updates needed when the workspace is introduced

1. Update `pyproject.toml` uv cache keys for `crates/**`.
2. Audit Maturin workspace-root behavior.
3. Change Rust benchmark recipes from generic `cargo bench` to:
   - `cargo bench --workspace`, or
   - explicit `cargo bench -p g-genotype`, etc.
4. Keep performance profiles in root `Cargo.toml`.
5. Ensure docs and CI watch `crates/**`.
6. Move Criterion benches to owning crates.
7. Update coverage source mapping if required.
8. Ensure `_core.pyi` checker still sees all root registrations.
9. Add `cargo tree`/`cargo metadata` architecture checks.
10. Keep one Cargo.lock at the workspace root.

---

# 12. Stop conditions

Pause a phase rather than forcing it through when:

- a dependency cycle appears;
- a leaf crate needs PyO3;
- public APIs require exposing large internal structs directly;
- Rust/Python call count increases on a hot path;
- output or manifest identity changes unexpectedly;
- sample or prediction alignment differs;
- resume mutates output before validation;
- parity results change;
- a performance regression cannot be attributed;
- the old and new paths cannot be compared deterministically;
- tests require broad skips to pass.

A stop is a signal that the ownership boundary is wrong or premature.

---

# 13. Definition of complete

The host migration is complete when:

- the root crate contains only PyO3/JAX adapter code;
- internal crates contain no PyO3 or NumPy;
- Rust owns execution plans and resolved plans;
- Rust opens all scientific inputs;
- Rust owns sample and prediction alignment;
- Rust owns preflight;
- Rust owns manifests and resume;
- Rust owns output sessions and finalization;
- Rust owns queues, buffers, batching, and worker lifecycle;
- Rust owns telemetry, timing, logging, and shutdown;
- `g-engine` is fully testable using a fake backend without Python;
- production Python owns only JAX kernels, JAX-required setup, and public wrappers;
- no production Rust/Python duplicate implementations remain;
- all supported workflows pass parity gates;
- no material performance regression remains unexplained.

---

# 14. Instruction block for the implementation agent

Use the following as the controlling instruction:

> Migrate `g` to a Cargo workspace while moving application ownership from Python to Rust. Introduce the workspace before substantial new Rust migration, but extract and migrate incrementally. Preserve the existing root `g` package and `_core` Maturin/PyO3 library as the composition crate. Create domain crates only when their ownership boundary is concrete. Extract existing Rust leaf domains first, then migrate each Python subsystem directly into its target crate. Do not build a monolithic Rust replacement and split it later. Do not perform the complete crate split in one PR. Keep JAX statistical kernels in Python during the host migration. Use coarse typed PyO3 calls. Maintain one authoritative implementation per contract. Every phase must include the prescribed correctness tests, parity checks, and benchmarks, and old production paths must be removed after equivalence is established.
