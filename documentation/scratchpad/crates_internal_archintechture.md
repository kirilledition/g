Yes: crate split good; internal architecture still transitional. Some crates have solid boundaries; others are policy/adapter warehouses with many concerns in same file/public module.

Main pattern:

```text
Good high-level crate ownership
+ too many internal public exports
+ several oversized “central” files
+ too many stringly typed policies
+ PyO3 binding files doing business orchestration
```

Direction: not more crates. Need cleaner internals per crate: small public facade, private modules, typed domain contracts, batch hot paths, Rust/Python glue adapts only.

---

# Overall judgment

Crate map is right: `g-plan` contracts, `g-interface` config-to-plan, `g-genotype` BGEN/read/preprocess, `g-input` sample/prediction alignment, `g-output` manifests/writers, `g-runtime` process state, `g-engine` orchestration.

Weak internals:

* `g-engine::schedule`: callback queues, result slots, buffer pools, GPU format, delivery cleanup, writer finish plans, string actions in one file.
* `g-output::session`: chunk metadata, Arrow arrays, writer pool/jobs, timings, writes, finish/abort/finalization in one module.
* `g-input::sample`: sample parse, phenotype/covariate parse, single/multi/grouped alignment, compute groups, validation, hashing in one module.
* `g-runtime::run_events`: event names/messages/payloads/diagnostics/serialization in one file.
* `src/binding/run_engine.rs`: PyO3 binding opens BGEN, chooses sample source, builds alignment inputs, calls `g-input`, converts results, exposes engine methods.

Conclusion: high-level crates good; refactor large procedural modules into domain submodules with small facades.

---

# Crate-by-crate internal architecture review

## `g-plan`

### What looks good

Minimal deps; canonical contract layer. `request.rs` defines serializable request contracts: association mode, trait type, device, BGEN validation, sample key mode, genotype format, dtype, output format, compression, `RunRequest`, `InputRequest`, `TraitRequest`, `ComputeRequest`, `CorrectionPlan`, `OutputWriterPlan`, `RuntimePlan`, phenotype run plans, compute groups.

`prepared.rs` canonicalizes resolved compute/output/input identity into `PreparedRunPlan`; rejects unresolved `GpuGenotypeFormat::Auto` before prepared backend plan.

### What is weak internally

`request.rs` becoming schema bucket. String-enum macro convenient but pulls every closed set into one file.

`PreparedRunPlan` mixes durable identity, backend planning, compute, correction, phenotype group, output writer. Acceptable top-level contract; subdomains should become explicit.

### Ideal internal architecture

```text
crates/plan/src/
  lib.rs
  enums.rs              shared closed-set values
  request/
    mod.rs
    input.rs
    compute.rs
    output.rs
    runtime.rs
    phenotype.rs
  prepared/
    mod.rs
    identity.rs
    backend.rs
    compute.rs
    output.rs
  host_policy/
    mod.rs
    association_mode.rs
    correction.rs
    phenotype_groups.rs
  error.rs
```

### Guiding principle

`g-plan` = stable serializable contracts + pure planning. No BGEN open, Arrow write, JAX call, telemetry emit, PyO3.

---

## `g-interface`

### What looks good

Clean crate root: hides `cli`, `defaults`, `domain`, `options`, `overlay`, `partial`, `resolved`, `run_validation`, `toml`, `validation`; exposes selected config entrypoints.

`plan_request.rs` has clear job: compile resolved `RegenieConfigData` to `g_plan::RunRequest`.

### What is weak internally

`resolved.rs` too large. `GComputeConfigData` mixes device, callback staging, BGEN trust, sample alignment, Firth, binary null, linear numerics, tile size, genotype format, dtype, JAX cache/precision, persistent cache, XLA autotune, transfer guard.

`plan_request.rs` leaks modes: quantitative traits carry dummy score-only correction plan with `p_threshold: 0.05` and `firth_se: false`.

### Ideal internal architecture

```text
crates/interface/src/
  lib.rs
  config/
    mod.rs
    resolved.rs
    partial.rs
    overlay.rs
    defaults.rs
    validation.rs
  options/
    mod.rs
    metadata.rs
    parse.rs
  toml/
    mod.rs
    load.rs
    dump.rs
  cli/
    mod.rs
    dispatch.rs
  compile/
    mod.rs
    input.rs
    trait_request.rs
    compute.rs
    correction.rs
    output.rs
    runtime.rs
    phenotype.rs
  error.rs
```

### Concrete changes

Split `GComputeConfigData`:

```rust
pub struct ComputeRuntimeConfigData {
    device: DeviceValue,
    staging_depth: NonZeroU32,
    native_callback_batch_size: NonZeroU32,
    result_in_flight_limit: Option<NonZeroU32>,
    dosage_buffer_limit: Option<NonZeroU32>,
}

pub struct NativeDecodeConfigData {
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: TrustedBgenValidationModeValue,
    bgen_decode_tile_variant_count: NonZeroU32,
    gpu_genotype_format: GpuGenotypeFormatValue,
}

pub struct BinaryNumericalConfigData { ... }
pub struct LinearNumericalConfigData { ... }
pub struct JaxRuntimeConfigData { ... }
pub struct AlignmentConfigData { ... }
```

Compile each separately into `g-plan`.

### Guiding principle

`g-interface` is only place user-facing config shape may be messy. After `g-plan`, downstream sees clean domain objects.

---

## `g-cli`

### What looks good

Good dependency shape: only `g-interface` + `signal-hook`.

### Ideal internal architecture

Keep small:

```text
crates/cli/src/
  main.rs
  lib.rs
  exit.rs
  signals.rs
```

### Guiding principle

`g-cli` never becomes application layer. Parse process details, call `g-interface`, exit.

---

## `g-genotype`

### What looks good

BGEN submodule split is good: `decode`, `error`, `format`, `index`, `metadata`, `profile`, `reader`, `sample_selection`, `simd`, `trusted`.

`BgenReaderCore` coherent: path, mmap, sample/variant count, embedded samples, compression, trust state, records, chromosome boundaries, sample selection, profiling.

Preprocess has scalar/SIMD boundary: `summarize_variant_major_row_simd_or_scalar` checks AVX2 then calls unsafe AVX2 only when available.

### What is weak internally

`common.rs` exposes large `ChunkStats` and `GenotypeReaderCore` with raw pointer-address output buffers. Raw pointers belong near PyO3/Numpy, not general crate contract.

`reader.rs` does open/index parse, sample IDs/selection, metadata slicing, trusted validation, row/variant-major decode, packed8, profiling, raw pointer validation.

`preprocess.rs` does shape validation, stats, missing impute, sparse classify, scalar/SIMD, `ChunkStats` construction.

### Ideal internal architecture

```text
crates/genotype/src/
  lib.rs
  api.rs
  error.rs

  source/
    mod.rs                 public-ish GenotypeSource facade
    chunk.rs
    stats.rs
    metadata.rs
    sample_selection.rs

  bgen/
    mod.rs
    reader.rs              thin BgenSource facade
    index.rs
    metadata.rs
    format.rs
    sample_block.rs
    trusted.rs
    profile.rs

  decode/
    mod.rs
    tile.rs
    row_major.rs
    variant_major.rs
    packed8.rs
    zlib.rs

  preprocess/
    mod.rs
    stats.rs
    impute.rs
    sparse.rs
    scalar.rs
    simd.rs

  buffer/
    mod.rs
    raw_pointer.rs         unsafe quarantined here
    owned.rs
```

### Concrete changes

1. Move raw pointer APIs to `buffer::raw_pointer`.
2. Use typed internal buffers:

```rust
pub struct VariantMajorDosageBuffer<'a> {
    values: &'a mut [f32],
    variant_count: usize,
    sample_count: usize,
}
```

3. Split `ChunkStats`:

```rust
VariantAlleleStats
VariantObservationStats
VariantDosageStats
SparseCandidateStats
```

Public result may aggregate; internals stop passing giant stat object everywhere.

### Guiding principle

`g-genotype` hides unsafe/SIMD/file-format complexity behind typed batch APIs. Other crates should not know scalar/AVX2/packed8/tiled/mmap/cache details.

---

## `g-input`

### What looks good

Right domain: sample alignment + prediction loading. `regenie/mod.rs` handles prediction list parsing, LOCO cache, alignment cache, single/multi prediction sources, chromosome prediction matrix cache.

`sample/mod.rs` exposes needed ops: single phenotype alignment, complete-case multi alignment, grouped per-phenotype alignment.

### What is weak internally

`sample/mod.rs` everything file: sample-file parse, tabular CSV, sample keys, phenotype/covariate tables, single/multi/grouped alignment, sample-file adapters, compute-group resolution.

Public functions return `Result<..., String>` in places.

### Ideal internal architecture

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

### Concrete changes

Replace public `String` errors:

```rust
#[derive(thiserror::Error, Debug)]
pub enum InputError {
    #[error("sample alignment failed: {0}")]
    Alignment(String),
    #[error("phenotype table error: {0}")]
    Phenotype(String),
    #[error("covariate table error: {0}")]
    Covariate(String),
    #[error("prediction error: {0}")]
    Prediction(#[from] PredictionError),
}
```

Move union-sample and group-position planning here. Python currently does alignment-domain work; make native.

### Guiding principle

`g-input` converts messy files to clean aligned matrices + sample identities. Downstream should not care about column names, missing tokens, FID/IID parsing, LOCO ordering.

---

## `g-output`

### What looks good

Crate root hides `finalization`, `manifest`, `resume`, `schema`, `session`, `writer`; selected re-exports only.

`NativeChunkHandle` is good boundary: metadata, stats, chunk id, lazy writer arrays.

### What is weak internally

`session.rs` too broad: stats, metadata, handles, Arrow arrays, config, timings, coordinator jobs, writer jobs/tasks/pool, completion, session methods.

Write path has too many overloads: raw metadata/stats/slices, handle+slices, handle+Arrow arrays. Collapse to one preferred path.

`manifest.rs` too broad: run paths, prepared/initialized runs, resume mode, chunk commits, file fingerprints/cache, current manifest input/contract, JAX/output writer manifests, reconstruction into `g_plan::PreparedRunPlanInput`.

### Ideal internal architecture

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

  resume/
    mod.rs
    scan.rs
    strict.rs
    repair.rs

  writer/
    mod.rs
    arrow.rs
    regenie.rs
    batch.rs
    schema.rs

  finalization/
    mod.rs
    parquet.rs
    dataset.rs

  timing/
    mod.rs
    accumulator.rs
    snapshot.rs
```

### Concrete changes

1. Make `write_regenie2_native_chunk_handle_arrays(...)` canonical internal write path.
2. Move slice-to-Arrow conversion to `chunk::arrays`.
3. Move `OutputStageTimingAccumulator` to `timing/accumulator.rs`.
4. Move worker pool to `session/worker_pool.rs`.
5. Move manifest-to-prepared-plan reconstruction to `manifest/contract.rs`.

### Guiding principle

`g-output` owns durability/schema. Engine/Python say “write this chunk result,” not build manifest JSON or understand Arrow fields.

---

## `g-runtime`

### What looks good

`ProcessRuntimeState` good concept: process-global logging, Rayon threads, JAX policy, compatibility checks, `RuntimeCompatibilityToken`.

### What is weak internally

`runtime_state.rs` combines policy payloads, process state, run handles, compatibility token/errors, Rayon config plans, JAX lifecycle/policy, direct Rayon configuration.

`run_events.rs` is global string registry: lifecycle, native CLI, runner, pipeline, native dispatch, runtime, preflight, engine dispatch.

### Ideal internal architecture

```text
crates/runtime/src/
  lib.rs
  api.rs
  error.rs

  state/
    mod.rs
    process_state.rs
    compatibility.rs
    token.rs

  policy/
    mod.rs
    logging.rs
    jax.rs
    rayon.rs

  services/
    logging.rs
    rayon.rs
    jax_setup.rs
    shutdown.rs

  telemetry/
    mod.rs
    session.rs
    writer.rs
    throttle.rs
    event.rs
    payload.rs

  diagnostics/
    mod.rs
    event.rs
    fields.rs
    render.rs

  timing/
    mod.rs
    stage.rs
    profile.rs
    output.rs

  paths.rs
```

### Concrete changes

Replace many exported event-name constants with typed events:

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

String names private serialization detail.

### Guiding principle

`g-runtime` owns process-global side effects + observability. It should not be dumping ground for every event string.

---

## `g-engine`

### What looks good

Best conceptual potential. Coordinator has phases and `EngineRunEffects` boundary. `EngineCoordinator` records phase history, handles injected failure/interruption, calls effects at phase transitions, prepares group/chromosome backend state, computes batches, writes, drains, finalizes, returns report.

Backend trait seam good: `prepare_group`, `prepare_chromosome`, `compute_batch`.

### What is weak internally

`backend.rs` scaffold: `PreparedGroupInput`, `PredictionView`, `GenotypeBatchView`, `AssociationBatchResult` are placeholder-like, not real high-throughput contracts.

`effects.rs` has one broad `EngineRunEffects` trait for telemetry, input open/alignment, preflight, output compatibility, writer construction/write/drain/finalize/abort. Useful for tests, too coarse for final design.

`pipeline.rs` uses `g_genotype::bgen::BgenReaderCore` and `g_genotype::planner`; engine depends on genotype internals, opens BGEN, plans chunks, resolves chromosomes, performs trusted validation.

`schedule.rs` biggest smell: callback queue limits, dosage reuse, chunk batching, GPU format resolution, output plans, queue observations, result slots, result write/dosage dispatch, buffer pool, worker lifecycle, scheduler state, delivery cleanup.

### Ideal internal architecture

```text
crates/engine/src/
  lib.rs
  api.rs
  error.rs

  request/
    mod.rs
    workload.rs
    linear.rs
    binary.rs
    single_trait.rs
    multi_trait.rs

  coordinator/
    mod.rs
    phases.rs
    run.rs
    errors.rs

  backend/
    mod.rs
    trait.rs
    batch.rs
    prepared_state.rs
    result.rs

  scheduler/
    mod.rs
    chunk_batch.rs
    callback_queue.rs
    result_slots.rs
    dosage_buffer_pool.rs
    worker_lifecycle.rs
    backpressure.rs

  delivery/
    mod.rs
    bgen.rs
    cleanup.rs
    genotype_format.rs

  effects/
    mod.rs
    input.rs
    output.rs
    telemetry.rs
    preflight.rs

  preflight/
    mod.rs
    phenotype.rs
    prediction.rs
    genotype.rs
    covariate.rs

  test_support/
    fake_backend.rs
    fake_effects.rs
```

### Concrete changes

1. Split `schedule.rs` by domain immediately.
2. Make `g-engine` depend on `g-genotype` facade, not `g_genotype::bgen::BgenReaderCore`.
3. Replace broad `EngineRunEffects` with smaller traits or concrete services:

```rust
trait InputEffects { ... }
trait OutputEffects { ... }
trait TelemetryEffects { ... }
trait PreflightEffects { ... }
```

or better, pass one `EngineServices` struct:

```rust
pub struct EngineServices<I, O, T, P> {
    input: I,
    output: O,
    telemetry: T,
    preflight: P,
}
```

4. Make backend contracts real and batch-oriented:

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

### Guiding principle

`g-engine` owns workflow, not every low-level policy. It coordinates input/genotype/backend/output/runtime/telemetry through typed services.

---

## Root PyO3 crate / `src/binding`

### What looks good

Root PyO3 module intentionally only `_core` registration.

### What is weak internally

`src/binding/mod.rs` registers long flat module list: callback diagnostics/progress/queue/runtime resources/summary, config, genotype, host policy, JAX runtime, JSON bridge, logging, output, prediction sources, preflight, profile, run engine, run events, run lifecycle, runtime, runtime state, sample alignment, schedule, shutdown, telemetry policy, timing.

`run_engine.rs` does domain orchestration: imports `g_engine`, `g_genotype`, `g_input`, `g_runtime`; opens BGEN; chooses embedded samples/sample file; builds `AlignmentInputs`/`MultiAlignmentInputs`; calls sample alignment.

### Ideal internal architecture

```text
src/binding/
  mod.rs

  config/
    mod.rs
    classes.rs
    functions.rs

  runtime/
    mod.rs
    state.rs
    jax.rs
    telemetry.rs
    logging.rs

  engine/
    mod.rs
    session.rs
    requests.rs
    backend_bridge.rs

  input/
    mod.rs
    aligned_data.rs
    prediction.rs

  output/
    mod.rs
    writer_session.rs
    chunk.rs

  diagnostics/
    mod.rs
    timing.rs
    events.rs

  error.rs
```

### Guiding principle

PyO3 = adapter, not application layer. Convert errors, wrap handles, manage GIL. Do not choose run behavior, reconstruct sample indices, build manifests, schedule queues, finalize outputs.

---

# Cross-cutting architecture principles

## 1. Public facade, private internals

Every crate target:

```rust
mod api;
mod error;
mod internal_a;
mod internal_b;

pub use api::{PublicRequest, PublicResult, PublicService};
pub use error::CrateError;
```

Avoid public module trees unless deliberate API namespace.

---

## 2. One module, one reason to change

If one file changes for BGEN decode, callback queues, writer finalization, and GPU format policy, it is too broad.

Current split targets:

```text
g-engine/src/schedule.rs
g-output/src/session.rs
g-input/src/sample/mod.rs
g-runtime/src/run_events.rs
```

---

## 3. Functional core, imperative shell

Keep pure planners; group by domain.

Example target:

```text
scheduler/chunk_batch.rs
scheduler/callback_queue.rs
scheduler/result_slots.rs
scheduler/dosage_buffers.rs
delivery/gpu_format.rs
delivery/cleanup.rs
output/writer_finish.rs
```

Each remains testable/pure.

---

## 4. Typed policies, not string action lists

Avoid string action/method/queue/result names. Prefer enums:

```rust
enum CallbackWorkerAction {
    StartDosageWorker,
    StartResultWorker,
}

enum DeliveryCleanupAction {
    DrainCallback,
    FinishWriterSessions,
    AbortCallback,
    AbortWriterSessions,
    WriteStageTimingSnapshot,
}
```

Serialize to strings only at PyO3/telemetry edge.

---

## 5. Batch-oriented hot paths

Hot APIs operate on batches/chunks:

```rust
read_chunk_into(...)
compute_batch(...)
write_chunk(...)
```

No per-variant/per-sample trait calls. Dynamic dispatch ok at batch boundaries only.

---

## 6. Unsafe quarantine

Raw pointers/SIMD isolated:

```text
genotype/buffer/raw_pointer.rs
genotype/preprocess/simd.rs
```

Rest uses typed views/safe slices.

---

## 7. No domain logic in PyO3

PyO3 must not decide:

```text
which sample alignment path to use
how to build compute groups
how to prepare manifests
how to schedule callback queues
how to finalize outputs
```

Call Rust facade that decides.

---

## 8. Observability should be sidecar-like

Telemetry/timing should not spread as string constants/JSON builders. Core crates emit typed events or typed observers.

Target:

```rust
trait EngineObserver {
    fn on_phase_started(&mut self, phase: RunPhase);
    fn on_batch_written(&mut self, report: BatchWriteReport);
}
```

Serialization belongs in `g-runtime`.

---

## 9. Mode-specific contracts

Avoid mixed linear/binary structs requiring dummy values. Quantitative runs should not carry binary Firth config; binary runs should not carry linear-only numeric config unless needed.

---

## 10. Test support is not public API

Fake backends/effects useful, but live under:

```rust
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

Production facade should not export fake types.

---

# Ideal crate architecture template

For most crates:

```text
src/
  lib.rs              small facade
  api.rs              public request/result/service types
  error.rs            crate error type

  model/              internal domain models
  planner/            pure decisions
  service/            side-effecting operations
  adapter/            external format / PyO3 / file boundary
  tests/ or test_support/
```

For high-performance crates:

```text
src/
  api.rs
  error.rs
  buffer/
  decode/
  simd/
  scalar/
  stats/
  profile/
```

For orchestration crates:

```text
src/
  api.rs
  coordinator/
  scheduler/
  backend/
  effects/
  reports/
```

---

# Priority order for internal cleanup

1. **Split `g-engine::schedule`** into scheduler/delivery/output policy modules.
2. **Split `g-output::session`** into chunk handle, writer pool, coordinator, timing, session.
3. **Split `g-input::sample`** into sample keys, tabular reader, phenotype, covariate, alignment, grouping.
4. **Simplify `src/binding/run_engine.rs`** so PyO3 calls Rust engine/input facade.
5. **Split `g-runtime::run_events`** into typed event families/private serialization.
6. **Refine `g-genotype` facade** so raw pointer APIs are not general cross-crate contract.
7. **Break up `g-interface::GComputeConfigData`** into narrower config subdomains.
8. **Keep `g-plan` boring**; avoid universal types crate.

---

# Bottom line

High-level crate map good. Internal architecture uneven:

```text
Best internal shape today:
  g-plan, g-interface top-level structure, g-genotype/bgen submodules, g-output crate root

Most in need of internal cleanup:
  g-engine::schedule
  g-output::session
  g-input::sample
  g-runtime::run_events
  src/binding/run_engine.rs
```

Guiding philosophy:

```text
Small public facade.
Private implementation modules.
Typed contracts.
Pure planners separated from side-effect services.
Batch-oriented hot paths.
Unsafe/SIMD quarantined.
PyO3 as glue only.
Observability as typed sidecar.
No dummy mode fields.
No public test scaffolding.
```

Keep performance work; make codebase safer for agents/future optimization.

# Errors

Yes. Each crate needs clear error boundary, usually top-level `src/error.rs` or `src/error/mod.rs`.

Nuance:

```text
crate public API returns crate-owned errors;
internal modules may have private/domain-specific errors;
conversion to the crate error happens at module boundaries;
conversion to Python errors happens only in src/binding.
```

Current scattering mirrors architecture issue: `g-engine` has `EngineError` in `coordinator.rs`, `BackendError` in `backend.rs`, `EngineEffectError` in `effects.rs`; `g-genotype` has `GenotypeError` in `common.rs` plus BGEN `BgenError`; `g-input` exposes `Result<..., String>`.

# Recommended rule

Each crate:

```text
src/error.rs
  public crate error type
  public Result alias
  From conversions from lower-level public/domain errors
```

`lib.rs` re-exports crate error:

```rust
mod error;

pub use error::{GenotypeError, GenotypeResult};
```

or:

```rust
pub use error::GenotypeError;

pub type GenotypeResult<T> = std::result::Result<T, GenotypeError>;
```

Public API mostly returns crate-level error:

```rust
pub fn open_bgen_source(options: BgenOpenOptions) -> GenotypeResult<GenotypeSource>;
```

not:

```rust
pub fn open_bgen_source(...) -> Result<_, BgenError>;
pub fn open_bgen_source(...) -> Result<_, String>;
pub fn open_bgen_source(...) -> Result<_, anyhow::Error>;
```

# But do not create a giant God `error.rs`

Avoid:

```rust
// bad long-term
pub enum EngineError {
    BgenOpen,
    BgenDecode,
    SampleFileParse,
    PhenotypeParse,
    CovariateRank,
    PredictionLoad,
    CallbackQueuePut,
    CallbackQueueGet,
    WriterFinish,
    ManifestCompatibility,
    TelemetryJson,
    JaxSetup,
    ...
}
```

Layered pattern:

```text
src/error.rs
  public facade error

src/bgen/error.rs
  BGEN-specific error, public only if BGEN is public

src/decode/error.rs
  private/internal decode error

src/manifest/error.rs
  manifest-specific error

src/session/error.rs
  writer-session-specific error
```

Top-level wraps meaningful domain errors:

```rust
// crates/output/src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("invalid output request: {0}")]
    InvalidInput(String),

    #[error("manifest error: {0}")]
    Manifest(#[from] ManifestError),

    #[error("writer error: {0}")]
    Writer(#[from] WriterError),

    #[error("finalization error: {0}")]
    Finalization(#[from] FinalizationError),

    #[error("runtime error: {0}")]
    Runtime(String),
}

pub type OutputResult<T> = std::result::Result<T, OutputError>;
```

If suberrors are not public API, convert to message:

```rust
impl From<manifest::ManifestError> for OutputError {
    fn from(error: manifest::ManifestError) -> Self {
        Self::Manifest(error.to_string())
    }
}
```

# Three levels of errors

## 1. Public crate error

Seen by other crates/PyO3:

```rust
ConfigError
PlanError
GenotypeError
InputError
OutputError
RuntimeError
EngineError
```

Live in `src/error.rs`.

## 2. Public domain errors, only when justified

Allowed only if caller can branch/use detail:

```rust
BgenError
BackendError
TelemetryError
ManifestCompatibilityError
PreparedPlanError
```

## 3. Private implementation errors

Most errors local/private:

```rust
decode::DecodeError
sample::SampleTableError
writer::ArrowBatchError
scheduler::QueuePlanError
```

Convert upward.

# Crate-by-crate recommendation

## `g-interface`

Move `ConfigError` from `lib.rs` to `src/error.rs`.

Target:

```text
crates/interface/src/
  lib.rs
  error.rs
  cli/
  config/
  compile/
  toml/
```

```rust
pub use error::{ConfigError, ConfigResult};
```

`ConfigError` remains only public error for config parsing, TOML, options, defaults, overlay, run validation.

## `g-plan`

Use `PlanError` top-level; keep subdomain errors when useful:

```rust
pub enum PlanError {
    HostPolicy(HostPolicyError),
    Prepared(PreparedPlanError),
    InvalidRequest(String),
}
```

`HostPolicyError` and `PreparedPlanError` may stay public if callers distinguish them; define/re-export through `error.rs`.

Target:

```text
crates/plan/src/
  error.rs
  request/
  prepared/
  host_policy/
```

Pure narrow functions may return specific errors:

```rust
fn build_prepared_run_plan(...) -> Result<PreparedRunPlan, PreparedPlanError>
```

Cross-domain functions return `PlanError`.

## `g-genotype`

Create:

```text
crates/genotype/src/error.rs
```

Move `GenotypeError` out of `common.rs`; `common.rs` should own shared structs or become `api.rs`/`types.rs`.

Target:

```rust
#[derive(Debug, thiserror::Error)]
pub enum GenotypeError {
    #[error("invalid genotype input: {0}")]
    InvalidInput(String),

    #[error("BGEN reader error: {0}")]
    Bgen(#[from] BgenError),

    #[error("genotype decode error: {0}")]
    Decode(String),

    #[error("sample selection error: {0}")]
    SampleSelection(String),
}

pub type GenotypeResult<T> = std::result::Result<T, GenotypeError>;
```

Expose `BgenError` only if BGEN is public contract:

```rust
pub use bgen::BgenError;
```

Do not expose entire `bgen` module.

## `g-input`

Biggest error cleanup. Replace public `Result<..., String>` in `align_sample_data`, `align_multi_sample_data`, `align_grouped_sample_data`.

Target:

```rust
#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error("invalid input request: {0}")]
    InvalidInput(String),

    #[error("sample file error: {0}")]
    SampleFile(String),

    #[error("phenotype table error: {0}")]
    PhenotypeTable(String),

    #[error("covariate table error: {0}")]
    CovariateTable(String),

    #[error("sample alignment error: {0}")]
    Alignment(String),

    #[error("prediction source error: {0}")]
    Prediction(#[from] PredictionError),
}

pub type InputResult<T> = std::result::Result<T, InputError>;
```

Then:

```rust
pub fn align_sample_data(inputs: AlignmentInputs) -> InputResult<AlignedSampleData>;
pub fn align_multi_sample_data(inputs: MultiAlignmentInputs) -> InputResult<MultiAlignedSampleData>;
pub fn align_grouped_sample_data(inputs: &MultiAlignmentInputs) -> InputResult<GroupedAlignedSampleData>;
```

Improves PyO3 conversion and testability without string matching.

## `g-output`

`OutputWriterError` increasingly inaccurate; crate owns manifests/resume/finalization/chunk handles too. Prefer public `OutputError`, keep alias short-term if needed.

Target:

```rust
pub enum OutputError {
    InvalidInput(String),
    Manifest(ManifestError),
    Resume(ResumeError),
    Writer(WriterError),
    Finalization(FinalizationError),
    Runtime(String),
}
```

Short-term:

```rust
pub type OutputWriterError = OutputError;
```

Long-term:

```text
OutputError        public crate error
WriterError        internal writer implementation error
ManifestError      public only if manifest API is public
ResumeError        public only if resume API is public
```

## `g-runtime`

Use:

```rust
RuntimeError
RuntimeResult<T>
```

Domain errors:

```rust
LoggingError
TelemetryError
JaxRuntimeError
RayonRuntimeError
ShutdownError
RuntimeCompatibilityError
```

Move `RuntimeCompatibilityError` from `runtime_state.rs` to:

```text
runtime/src/state/error.rs
```

or:

```text
runtime/src/error.rs
```

Public API should not require knowing compatibility errors live in `runtime_state.rs`.

## `g-engine`

Root `EngineError` should live in `error.rs`, not `coordinator.rs`.

Today:

```text
coordinator.rs -> EngineError
backend.rs     -> BackendError
effects.rs     -> EngineEffectError
schedule.rs    -> ScheduleError
```

Target:

```rust
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("backend failed during {phase}: {source}")]
    Backend {
        phase: RunPhase,
        #[source]
        source: BackendError,
    },

    #[error("engine side effect {operation} failed during {phase}: {source}")]
    Effect {
        phase: RunPhase,
        operation: EngineEffectOperation,
        #[source]
        source: EngineEffectError,
    },

    #[error("scheduler failed: {0}")]
    Schedule(#[from] ScheduleError),

    #[error("run interrupted during {phase}")]
    Interrupted { phase: RunPhase },

    #[error("engine failed during {phase}: {message}")]
    Coordinator { phase: RunPhase, message: String },
}
```

Keep `BackendError` public if backend trait public. Keep `EngineEffectError` private/semi-private depending on `EngineRunEffects`.

## Root PyO3 crate

`src/binding` defines conversion, not business errors.

Target:

```rust
// src/binding/errors.rs

pub fn engine_error_to_pyerr(error: g_engine::EngineError) -> PyErr;
pub fn genotype_error_to_pyerr(error: g_genotype::GenotypeError) -> PyErr;
pub fn input_error_to_pyerr(error: g_input::InputError) -> PyErr;
pub fn output_error_to_pyerr(error: g_output::OutputError) -> PyErr;
pub fn runtime_error_to_pyerr(error: g_runtime::RuntimeError) -> PyErr;
```

No per-file ad hoc `map_err(PyValueError::new_err)`.

# What the structure should look like

Simple crate:

```text
crates/interface/src/
  lib.rs
  error.rs
  ...
```

Complex crate:

```text
crates/output/src/
  lib.rs
  error.rs

  manifest/
    mod.rs
    error.rs

  writer/
    mod.rs
    error.rs

  resume/
    mod.rs
    error.rs

  finalization/
    mod.rs
    error.rs
```

`src/error.rs` = crate boundary. Submodule `error.rs` = domain internals.

# Error design principles

## 1. Libraries should not return `String`

Use `String` inside error variant if needed; never expose `Result<T, String>` from prod crate APIs.

Bad:

```rust
pub fn align_sample_data(...) -> Result<AlignedSampleData, String>;
```

Good:

```rust
pub fn align_sample_data(...) -> Result<AlignedSampleData, InputError>;
```

## 2. Do not use `anyhow` in library crate public APIs

`anyhow` ok for binaries/tools/tests.

Good for `g-cli`:

```rust
fn main() -> anyhow::Result<()>
```

Not ideal for `g-genotype` public API:

```rust
pub fn open_bgen(...) -> anyhow::Result<BgenSource>
```

## 3. Use `thiserror`

Repo already uses `thiserror`; keep it.

```rust
#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error("phenotype table `{path}` is missing required column `{column}`")]
    MissingPhenotypeColumn { path: PathBuf, column: String },
}
```

## 4. Preserve source errors where they matter

Use `#[source]`/`#[from]` where source type matters.

```rust
#[error("failed to open BGEN file `{path}`")]
OpenBgen {
    path: PathBuf,
    #[source]
    source: std::io::Error,
}
```

## 5. Add context at the boundary

Low-level:

```rust
DecodeError::UnexpectedEndOfVariantBlock
```

Boundary:

```rust
GenotypeError::Decode {
    chromosome: String,
    variant_start: usize,
    variant_stop: usize,
    source: DecodeError,
}
```

Add path/chunk/phase where caller knows it.

## 6. Do not put telemetry wording into errors

Errors explain failure; telemetry/rendering logs it.

Bad:

```rust
EngineError::NativeDispatchDeliveryFailedDiagnosticEvent(...)
```

Good:

```rust
EngineError::Delivery(DeliveryError)
```

Telemetry turns error into event.

## 7. Error types are part of API

Public enum variant = public API. For future flexibility:

```rust
#[non_exhaustive]
pub enum GenotypeError { ... }
```

# Migration plan

## Phase 1 — Inventory

Run:

```bash
rg "struct .*Error|enum .*Error|type .*Result|Result<.*String|thiserror|anyhow|map_err" crates src/binding
```

Create table:

```text
crate
error type
location
public/private
returned by public API?
used by PyO3?
should move?
```

## Phase 2 — Add `error.rs` without behavior changes

For each crate, create `src/error.rs` and move/re-export existing errors.

Example first `g-engine` step:

```rust
// crates/engine/src/error.rs
pub use crate::backend::BackendError;
pub use crate::coordinator::EngineError;
pub use crate::effects::EngineEffectError;
pub use crate::schedule::ScheduleError;
```

No behavior change.

## Phase 3 — Move definitions

Move definitions:

```text
coordinator.rs -> error.rs
backend.rs     -> backend/error.rs or error.rs
effects.rs     -> effects/error.rs or error.rs
```

Keep temporary re-exports if needed.

## Phase 4 — Replace `Result<T, String>`

Start with `g-input`.

```rust
Result<T, String>
```

becomes:

```rust
InputResult<T>
```

Small, high-value cleanup.

## Phase 5 — Normalize PyO3 conversions

All PyO3 bindings call `src/binding/errors.rs` helpers:

```rust
.map_err(convert_input_error)
.map_err(convert_output_error)
.map_err(convert_engine_error)
```

No random string-to-`PyValueError` in binding methods.

## Phase 6 — Enforce

Add CI checks:

```bash
rg "Result<[^>]+,\s*String>" crates && exit 1
rg "anyhow::Result" crates/*/src && exit 1
```

Exceptions for binaries/tools/tests.

Convention check:

```text
public error types must be declared in:
  src/error.rs
  src/*/error.rs
```

# My recommended standard

Use this in `documentation/development/rust-error-policy.md`:

```markdown
# Rust error policy

Each crate owns a public crate error in `src/error.rs`.

Public crate APIs should return the crate error or a documented public
domain error.

Implementation modules may define private domain errors, but they must
convert to the crate error at the crate boundary.

Production library crates must not expose `Result<T, String>` or
`anyhow::Error`.

PyO3 bindings must convert Rust errors to Python errors only in
`src/binding/errors.rs`.

Error messages should include actionable context, but telemetry event names
and user-facing rendering belong to `g-runtime`, not error variants.
```

# Bottom line

Yes, standardize on `error.rs` per crate.

Real goal:

```text
clear crate error boundary
typed domain errors
no Result<T, String>
no random public errors in operational files
central PyO3 conversion
```

Highest-value first moves:

```text
1. Move ConfigError, EngineError, GenotypeError, RuntimeCompatibilityError into crate error modules.
2. Replace g-input Result<..., String> with InputError.
3. Rename or wrap OutputWriterError as OutputError.
4. Put PyO3 error conversion behind src/binding/errors.rs.
5. Add CI checks to stop new scattered public errors.
```
