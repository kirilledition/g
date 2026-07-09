# Rust Pipeline Migration

> Internal scratchpad. Post-architecture implementation plan. Not user docs.
> Caveman-compressed: pipeline ownership only.

This plan runs after `documentation/scratchpad/crates_architecture_implementation_plan.md`.
Assume crate facades, errors, internal module splits, PyO3 surface cleanup, and
architecture guards are already handled there.

## Scope

Move production run ownership from Python orchestration to Rust engine. Keep
Python/JAX as compute backend.

Do not work on:

- tests;
- test support;
- tooling;
- benchmark scripts;
- CI guard scripts;
- crate facade/error/module architecture already covered by
  `crates_architecture_implementation_plan.md`.

If validation reports only tests/tooling failures, record and continue. Fix only
production crate/root-PyO3 code touched by current slice.

## Goal

Current shape:

```text
Python owns run:
  CLI/API -> Python runner -> Python execution plan -> Python pipelines
  -> Rust BGEN/input/output helpers
  -> Python callback runtime
  -> JAX kernels
  -> Python result materialization
  -> Rust writer
```

Target shape:

```text
Rust owns run:
  CLI/API -> Rust config/runtime/lifecycle/engine
  -> Rust BGEN/input/prediction/preflight/output/scheduling/writer
  -> Python JAX backend at coarse batch boundaries
  -> Rust writer/finalizer/artifacts
```

Python keeps:

```text
public convenience API
JAX import/setup boundary
JAX kernels
JAX backend methods:
  prepare_group
  prepare_chromosome
  compute_batch
```

Rust owns:

```text
run lifecycle
execution planning
BGEN open/decode/chunking
sample/phenotype/covariate alignment
prediction loading
preflight
output manifests/resume
writer sessions
delivery cleanup
callback queues
host buffer ownership
result in-flight slots
progress
telemetry/timing orchestration
final artifacts
```

## Rules

1. Keep JAX statistical kernels in Python unless separate approved Rust/CUDA
   replacement exists.
2. Reduce Python boundary crossings. No per-variant or tiny-chunk Python calls.
3. No Python orchestration fallback requirement. App unreleased; delete old path
   when Rust path owns contract.
4. Preserve statistical behavior: no formula/Firth/dtype changes unless plan
   explicitly says so.
5. No new public Python surface for Rust internals. Expose high-level handles
   only.
6. Run Rust/JAX boundary in coarse batches: group, chromosome, batch.
7. Ruff may format directly; no format-check step needed.

Focused validation:

```bash
CARGO_BUILD_JOBS=30 RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo check -p <crate> --lib
CARGO_BUILD_JOBS=30 RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo clippy -p <crate> --lib -- -D warnings -W clippy::pedantic
uv run ty check src
uv run ruff format src
uv run ruff check src
```

Run Rust compile/lint on head node with 30 cores and mold linker. Fix
`cargo check` errors and clippy pedantic warnings in production code touched by
current slice. Do not run tests, benchmark suites, tooling checks, or CI guard
scripts for this plan.

## Remove From Old Pipeline Plan

Covered by architecture plan; do not repeat here:

- crate facade shrink and `PUBLIC_API.md` sync;
- crate `src/error.rs` migration;
- internal splits such as `g-engine::schedule`, `g-output::session`,
  `g-input::sample`, `g-runtime::run_events`;
- PyO3 public surface shrink as a standalone architecture task;
- bloat-prevention tooling/CI scripts;
- test/debug API gating;
- benchmark harnesses and baseline docs;
- Python orchestration fallback env vars.

## Phase 0 - Production Inventory

Goal: identify production run-ownership code still in Python.

Inspect only production paths:

```text
src/g/runner/
src/g/execution_plan.py
src/g/engine/regenie2_pipeline/
src/g/engine/native_dispatch/
src/g/engine/callbacks/
src/binding/
crates/engine/
crates/input/
crates/output/
crates/genotype/
crates/runtime/
```

Classify Python code:

```text
keep:
  public API
  JAX setup
  JAX kernels
  JAX backend compute

migrate to Rust:
  runner lifecycle
  execution planning
  pipeline orchestration
  delivery cleanup
  output preparation
  callback queues/resources
  result writer lifecycle

delete after Rust owner exists:
  one-call wrappers
  Python dataclass plan mirrors
  Python enum mirrors of Rust scheduler actions
  Python delivery cleanup executors
```

Done when each production Python orchestration module has owner: keep, migrate,
or delete.

## Phase 1 - Native Run Session

Goal: create Rust object that owns run state.

Target:

```rust
pub struct NativeRunEngineSession {
    lifecycle_session: NativeRunLifecycleSession,
    runtime_token: RuntimeCompatibilityToken,
}
```

Expose one high-level Python handle:

```python
_core.NativeRunEngineSession
```

Initial methods:

```python
session.run_request()
session.prepared_phenotype_runs()
session.finalize_success(...)
session.abort(...)
```

Tasks:

1. Add `crates/engine/src/run_session.rs` or equivalent.
2. Add PyO3 wrapper under `src/binding/engine/` or current engine domain.
3. Register only high-level session handle.
4. Do not call JAX yet.
5. Delete old Python state wrappers once session owns same data.

Done when production can construct native session from config/runtime token and
read prepared phenotype runs without Python plan mirror.

## Phase 2 - Native Execution Plan Ownership

Goal: stop production dependence on `src/g/execution_plan.py`.

Current:

```text
NativeRunRequest -> Python RegenieExecutionPlan -> PipelineCommonRequest -> pipeline call
```

Target:

```text
NativeRunEngineSession owns NativeRunRequest and PreparedRunPlan directly
```

Add native accessors only if production needs them:

```rust
fn association_mode(&self) -> AssociationMode;
fn phenotype_count(&self) -> usize;
fn output_run_root(&self) -> PathBuf;
fn chunk_size(&self) -> u32;
fn variant_limit(&self) -> Option<u32>;
```

Tasks:

1. Move production data pulled from `execution_plan.RegenieExecutionPlan` into
   native session.
2. Update runner/pipeline entry to use native session values.
3. Delete production use of `build_regenie_execution_plan_from_run_request(...)`.
4. Delete or reduce:

```text
execution_plan.KernelConfig
execution_plan.OutputWriterPlan
execution_plan.OutputPlan
execution_plan.PhenotypeRunPlan
execution_plan.PhenotypeComputeGroup
execution_plan.RegenieExecutionPlan
build_regenie_execution_plan_from_run_request
build_kernel_config_from_run_request
```

Done when production no longer requires Python `RegenieExecutionPlan`.

## Phase 3 - Rust Runner Lifecycle

Goal: replace most of `runner/execution.py` with JAX setup + Rust call.

Target Python:

```python
def regenie(config, *, run_telemetry_session, close_telemetry_session_on_exit, initialize_logging_on_entry):
    runtime.configure_runtime_before_jax_import(config.g_compute, telemetry_session=run_telemetry_session)
    backend = g.jax_backend.JaxAssociationBackend.from_config(config)
    native_artifacts = g._core.run_regenie_with_backend(config, backend)
    return events.run_artifacts_from_native_artifacts(native_artifacts)
```

Add Rust entry:

```rust
pub fn run_regenie_with_backend(
    config: RegenieConfigData,
    backend: PyAssociationBackend,
) -> Result<NativeRunArtifacts, EngineError>;
```

Move into Rust:

```text
validate config for run
record run started/failed/interrupted/completed
attach run metadata
runtime policy application
lifecycle session creation
success finalization
failure/interruption cleanup
final timing context
```

Leave in Python:

```text
JAX import ordering
JAX runtime setup before backend init
backend object construction
small artifact adapter if public API needs it
```

Done when `runner/execution.py` is small shell and Rust owns run lifecycle.

## Phase 4 - Rust Output Preparation

Goal: Python stops building manifest headers/output runs/writer sessions.

Move into `g-output` + `g-engine`:

```text
manifest header construction
resume compatibility validation
output run initialization
writer session creation
committed chunk identifier resolution
writer session batch creation
```

Target:

```rust
pub struct PreparedOutputBundle {
    pub initialized_runs: Vec<InitializedOutputRun>,
    pub writer_sessions: Vec<OutputWriterSession>,
    pub committed_chunk_sets: Vec<CommittedChunkSet>,
}

impl NativeRunEngineSession {
    pub fn prepare_outputs_for_group(
        &mut self,
        group: &AlignedComputeGroup,
        backend_plan: &AssociationBackendPlan,
    ) -> Result<PreparedOutputBundle, EngineError>;
}
```

PyO3 exposes handle only if production Python must pass it:

```python
_core.PreparedOutputBundle
```

Delete/migrate Python functions:

```text
build_pipeline_manifest_header
initialize_pipeline_output_runs
validate_pipeline_resume_compatibility
create_pipeline_writer_sessions
committed_chunk_identifiers
shared_committed_chunk_identifiers_across
```

Done when Python no longer builds manifest JSON/header payloads or writer
session sets.

## Phase 5 - Rust Delivery Cleanup

Goal: delete Python execution of native cleanup plans.

Create Rust runner:

```rust
pub struct BgenDeliveryRunner;
```

Target:

```rust
pub fn run_bgen_delivery(
    engine: &mut Regenie2RunEngineCore,
    run_input: BgenDeliveryInput,
    writer_sessions: WriterSessionSet,
    callback_or_backend: DeliveryBackend,
    options: BgenDeliveryOptions,
) -> Result<BgenDeliveryReport, EngineError>;
```

Rust owns:

```text
start callback/backend
run BGEN delivery
success cleanup
interrupted cleanup
failure cleanup
writer finish/abort
callback drain/abort
stage timing snapshot
final path collection
diagnostic event emission
```

Delete/migrate Python functions:

```text
BgenDeliveryCleanupAction
BgenDeliveryCleanupExecution
finish_writer_sessions
execute_bgen_delivery_cleanup_plan
run_bgen_engine_with_writer_sessions
run_bgen_engine_with_callback
```

Done when no Python loop executes cleanup actions; Rust returns final delivery
report/paths.

## Phase 6 - Rust Input, Prediction, Preflight

Goal: Python stops loading aligned inputs, prediction sources, preflight.

Create:

```rust
pub struct PreparedTraitInput {
    pub aligned_input: AlignedInputHandle,
    pub prediction_source: PredictionSourceHandle,
    pub preflight_report: PreflightReport,
    pub compute_group: ResolvedPhenotypeComputeGroup,
}
```

Add:

```rust
impl NativeRunEngineSession {
    pub fn prepare_single_trait_input(...) -> Result<PreparedTraitInput, EngineError>;
    pub fn prepare_multi_trait_input(...) -> Result<PreparedTraitInput, EngineError>;
    pub fn prepare_grouped_trait_inputs(...) -> Result<Vec<PreparedTraitInput>, EngineError>;
}
```

Rust owns:

```text
sample file vs embedded sample source decision
sample alignment
covariate alignment
phenotype loading
prediction source loading
preflight shape validation
case/control checks
covariate rank validation
compute group resolution
sample alignment telemetry
prediction loaded telemetry
preflight telemetry
```

Delete/migrate Python:

```text
load_single_trait_run_input
build_single_trait_prediction_source
run_single_trait_preflight
multi-trait complete-case input loading block
grouped per-phenotype input construction loop
```

Done when Python receives prepared native input handles and does not call
alignment/preflight helpers directly.

## Phase 7 - Rust Grouped Delivery Planner

Goal: grouped union/separate delivery policy belongs to Rust.

Target:

```rust
pub enum GroupedDeliveryStrategy {
    SeparateGroupPasses,
    UnionSampleDelivery {
        union_sample_indices: SampleIndexBuffer,
        group_position_maps: Vec<GroupPositionMap>,
    },
}

pub fn plan_grouped_per_phenotype_delivery(
    groups: &[PreparedTraitInput],
    options: GroupedDeliveryOptions,
) -> Result<GroupedDeliveryStrategy, EngineError>;
```

Move into Rust:

```text
should_use_union_grouped_bgen_delivery
union sample count vs grouped sample count comparison
packed8 exclusion
trusted-no-missing requirement
native callback batch-size validation
fanout group construction
mapping final output paths back to phenotype order
```

Delete/migrate Python:

```text
build_union_sample_indices
build_validated_grouped_union_sample_indices
build_group_sample_position_array
should_use_union_grouped_bgen_delivery
run_prepared_grouped_per_phenotype_union_bgen_pipeline
```

Done when grouped union decision and fanout plan are native.

## Phase 8 - Real Rust Backend Seam

Goal: Rust engine calls backend trait at coarse boundaries; Python callback
classes stop being engine abstraction.

Production trait:

```rust
pub trait AssociationBackend {
    type GroupState;
    type ChromosomeState;

    fn backend_kind(&self) -> AssociationBackendKind;

    fn prepare_group(
        &mut self,
        input: BackendGroupInput<'_>,
    ) -> Result<Self::GroupState, BackendError>;

    fn prepare_chromosome(
        &mut self,
        group_state: &mut Self::GroupState,
        input: BackendChromosomeInput<'_>,
    ) -> Result<Self::ChromosomeState, BackendError>;

    fn compute_batch(
        &mut self,
        chromosome_state: &mut Self::ChromosomeState,
        batch: GenotypeBatchView<'_>,
    ) -> Result<AssociationBatchResult, BackendError>;
}
```

Python JAX backend:

```python
class JaxAssociationBackend:
    def prepare_group(self, group_input): ...
    def prepare_chromosome(self, chromosome_input): ...
    def compute_batch(self, genotype_batch): ...
```

PyO3 bridge:

```rust
pub struct PyJaxAssociationBackend {
    py_object: Py<PyAny>,
}
```

Rust calls Python only for:

```text
prepare_group
prepare_chromosome
compute_batch
```

No per-variant Python calls. Existing Python callbacks can survive only as
temporary adapters into backend methods; delete once replaced.

Done when Rust engine can run through Python JAX backend seam.

## Phase 9 - Rust Callback Scheduler

Goal: delete Python-owned `NativeBgenCallbackRunner` worker runtime.

Move into Rust:

```text
bounded queues
worker lifecycle
dosage work handoff
result write handoff
result in-flight slot acquire/release
host dosage buffer pool
progress throttling
binary diagnostics summary accumulation
worker error propagation
shutdown timeout handling
```

Rust scheduler flow:

```text
decode batch
call backend.compute_batch(...)
write result
release resources
record progress
```

Python should not have:

```text
consume_dosage_chunks
consume_result_write_items
put_dosage_work_item
put_result_write_item
acquire_result_in_flight_slot
release_result_in_flight_slot
acquire_dosage_buffer
release_dosage_buffer
stop/join worker
```

Transition:

```text
Step 9A:
  Rust queue -> Python compute_chunk adapter -> Rust result writer

Step 9B:
  Rust batch scheduler -> Python compute_batch -> Rust writer
```

Done when Python callback runtime no longer owns dosage/result loops, workers,
buffers, slots, or writer handoff.

## Phase 10 - Rust Result Writer Handoff

Goal: Python computes; Rust validates/materializes/writes/releases.

Python backend returns:

```python
AssociationBatchResult(
    beta=...,
    standard_error=...,
    chi_squared=...,
    log10_p_value=...,
    extra_code=...,
    diagnostics=...,
)
```

Rust does:

```rust
writer.write_chunk(chunk_handle, result)?;
release_buffers();
release_result_slot();
record_diagnostics();
```

Define PyO3-compatible result:

```rust
pub struct NativeAssociationBatchResult {
    beta: PyArrayLike,
    standard_error: PyArrayLike,
    chi_squared: PyArrayLike,
    log10_p_value: PyArrayLike,
    extra_code: Option<PyArrayLike>,
    diagnostics: Option<BinaryDiagnostics>,
}
```

Tasks:

1. Rust validates shape/dtype once per batch.
2. Rust narrows/casts output statistic dtype where feasible.
3. Rust calls `g-output` writer directly.
4. Rust releases buffers/result slots and records diagnostics.

Done when Python no longer calls writer sessions or
`_core.write_regenie2_native_chunk_with_output_dtype`.

## Phase 11 - Delete Python Orchestration

Goal: remove obsolete production Python pipeline/delivery/callback modules after
Rust path owns contract.

Delete or shrink:

```text
src/g/execution_plan.py
src/g/engine/dispatch_requests.py
src/g/engine/regenie2_pipeline/single_trait.py
src/g/engine/regenie2_pipeline/multi_trait.py
src/g/engine/regenie2_pipeline/grouped.py
src/g/engine/regenie2_pipeline/multi_group.py
src/g/engine/native_dispatch/delivery.py
src/g/engine/native_dispatch/loaders.py
src/g/engine/native_dispatch/groups.py
src/g/engine/callbacks/runtime.py
```

Keep:

```text
src/g/compute/*
src/g/jax_runtime/*
src/g/jax_backend/*
```

Target Python tree:

```text
src/g/
  api.py
  cli.py
  interface/config.py

  runner/
    runtime.py
    events.py

  jax_backend/
    backend.py
    linear.py
    binary.py
    transfer.py
    materialize.py

  compute/
    regenie2_linear/
    regenie2_binary/
```

Done when Python mainly contains API, JAX setup, JAX backend, kernels.

## Implementation Order

1. Production inventory.
2. `NativeRunEngineSession`.
3. Native execution-plan ownership.
4. Rust lifecycle ownership.
5. Rust output preparation.
6. Rust delivery cleanup.
7. Rust input/prediction/preflight.
8. Rust grouped delivery planner.
9. Real backend trait + PyJAX bridge.
10. Rust callback scheduler.
11. Rust result writer handoff.
12. Delete Python orchestration modules.

## Stop Conditions

Stop and report if:

- change would alter statistical formula, Firth behavior, or dtype;
- JAX import ordering becomes ambiguous;
- Rust needs per-variant Python calls;
- ownership belongs to architecture plan, not pipeline migration;
- change requires tests/tooling/benchmark work.

Report:

```text
phase
file/function
blocked reason
smallest owner decision needed
commands run
```

## Final Result

```text
Rust:
  owns application run
  owns lifecycle/planning/input/output/delivery/cleanup/writer handoff
  owns callback queues/resources/progress

Python:
  public API convenience
  JAX setup boundary
  JAX backend implementation
  JAX kernels

PyO3:
  high-level handles only
```

Concrete success:

```text
runner/execution.py tiny or gone
execution_plan.py gone from production
engine/regenie2_pipeline gone from production
engine/native_dispatch/delivery.py gone from production
callbacks/runtime.py no longer owns queues/workers/writers
Rust engine owns lifecycle, planning, input, output, delivery, cleanup, writer handoff
JAX compute unchanged
No per-variant Python calls
No writer JSON hot path
```
