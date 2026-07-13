# Native GWAS Architecture Cleanup Plan

| Status | Applies to | Owner |
| --- | --- | --- |
| Historical implementation ledger | Superseded production architecture plan | Development maintainers |

This file preserves the decisions and acceptance language used during the
earlier cleanup. It is not the current production contract: later work removed
the grouped-union delivery, public raw-buffer wrappers, process-global BGEN
decode state, and intermediary re-exports described below. See
[Architecture](../development/architecture.md),
[Rust Crate Boundaries](../development/rust-crate-boundaries.md), and each
crate's `PUBLIC_API.md` for active ownership and dependency rules.

## Execution Ledger

- [x] Canonical typed `RunPlan`, numeric policy values, and native CLI removal.
- [x] Production-dead scalar Firth/result paths and unused diagnostics removal.
- [x] Canonical genotype/output chunk data and bounded run-scoped writer workers.
- [x] Bounded association scheduler, active-trait propagation, and reusable genotype buffers.
- [x] Initial Python/JAX wrapper and dead-code reduction.
- [x] One trait-major aligned phenotype group and unified prediction ownership.
- [x] Output lifecycle ownership in `g-output`.
- [x] Consuming `RunEngine` preparation/execution in `g-engine`.
- [x] Typed execution reports and runtime-owned telemetry transport.
- [x] Thin binding, module flattening, numeric audit, and final dead-code sweep.
- [x] Parquet-only result contract with chunked `parts/` as the completed dataset.
- [x] Coarse `g-engine::execute_coordinated_run` ownership above prepared execution.
- [x] Canonical manifest schema 14 with no JSON DTO crossing domain crates.
- [x] Output schema 3 with dense Parquet columns and explicit correction method/status.
- [x] Cached Parquet schemas/dictionaries, shared writer configuration, and single-open strict resume.

## Decisions

- All pre-release APIs, configuration, manifests, telemetry, and output schemas may break.
- Scientific formulas remain unchanged except for explicit numeric-policy fixes.
- Rust host arrays, JAX score arithmetic, and public statistics use `float32`.
- Approximate-Firth arithmetic is always `float64`; `firth_dtype` is removed.
- The nonfunctional standalone Rust CLI binary is deleted. The installed `g` console remains the thin Python launcher into `_core`.
- Tests, benchmarks, and tooling are not modified or run during this pass.
- Result output is only `parts/part_*.parquet`; no output-format selector,
  text writer, IPC writer, or consolidated file is retained.

## Target Dependency Graph

```text
g-plan --> g-interface / g-input / g-output / g-engine / g-runner
g-genotype-contracts --> g-genotype / g-output / g-engine
g-runtime --> g-engine / g-runner
g-engine --> g-runner

_core --> runner + the domain types adapted at the PyO3/NumPy boundary
```

Keep the existing eight domain crates. Reassign misplaced responsibilities instead of adding, merging, or splitting crates by line count.

## Implementation Sequence

1. Build one complete typed `g-plan::RunPlan` covering input, analysis/kernel, compute, output, runtime, diagnostics, and phenotype policy. Remove duplicate enum conversions, the unused prepared-plan graph, fixed/unsupported options (`step`, `firth_dtype`, SPA, exact Firth, unsafe assume-validated mode), `is_validated`, and the native CLI subprocess bridge.
2. Normalize all input modes to a trait-major `AlignedPhenotypeGroup`; a single phenotype is a one-row group. Store sample identities once, retain group sample indices, unify prediction sources, and return cached chromosome matrices through shared ownership.
3. Delete the production-dead row-major BGEN pipeline and confirmed dead scalar/single-trait Firth paths. Narrow final genotype statistics to production-consumed fields.
4. Make `g-output` consume canonical genotype metadata/statistics. Move owned statistic matrices into Parquet's Arrow arrays once, slice trait rows without copying, and share immutable metadata arrays.
5. Replace the global unbounded output pool and per-phenotype coordinators with a run-scoped `OutputManager` using fixed workers, bounded queues, owned joins, and deterministic finish/interruption/abort.
6. Add `g-engine::RunEngine::prepare` and consuming `PreparedRun::execute_with_progress<B: AssociationBackend, H: RunHooks>`. Normalize single, complete-case, per-phenotype, and grouped-union execution into `Vec<PreparedAssociationGroup>` and move all BGEN delivery, scheduling, resume, output, and lifecycle logic out of `src/binding`.
7. Bound every association stage, carry active trait indices through completed batches, reuse genotype buffers by capacity, and share grouped-union metadata rather than cloning it.
8. Reduce `g-runtime` to generic logging, telemetry transport, timing, process/JAX policy, and shutdown. Move GWAS events/artifacts to engine and manifest extension logic to output. Replace event-forwarder functions with a small typed event model.
9. Reduce `src/binding` to `_core` registration, CLI result adaptation, JAX configuration/device calls, four backend method calls, NumPy conversion, opaque Python state, and one native-error-to-Python boundary.
10. Remove remaining one-line forwarders and one-field wrappers unless they enforce a validated invariant, unsafe ownership boundary, PyO3/JIT contract, or RAII behavior. Run a manually verified production reachability sweep after the main migration.

## Implemented Architecture

- `g-engine::RunEngine::open` owns one shared `RunPlan`; consuming `prepare`
  resolves BGEN/input/resume/output state and consuming
  `PreparedRun::execute_with_progress` owns delivery plus terminal output
  policy.
- `g-output::OutputManager` owns run directories, manifests, bounded Parquet
  writer workers, immutable shared resume sets, interruption, abort, and
  output completion.
- Output completion is `Vec<CompletedOutputRun>`, not parallel artifact vectors.
- `g-runtime::TelemetryRunSession` owns telemetry state, serialization, writer,
  counters, and close. PyO3 supplies only the current Python thread name.
- Dead progress throttling/configuration and pre-migration pipeline, preflight,
  alignment, and BGEN event families are removed. The telemetry session stores
  only its run ID and optional owned writer.
- Stage timing stores only production-recorded stage totals/counts. Empty BGEN,
  null-logistic, queue, transfer, chunk-timing, and derived-metric schemas are
  removed, and the former timing submodule folder is flattened.
- `g-plan` carries request-derived policy only. `g-output` owns fixed strict
  resume, Parquet grouping/compression, queue capacity, and statistic width;
  runtime crates own their fixed execution policy.
- The binding creates the Python-backed four-method JAX adapter and makes one
  coarse `g-engine::execute_coordinated_run` call. It has no input, BGEN,
  resume, writer, artifact-construction, or execution-report policy.
- Production-dead reader profiling, GPU string planners, single-trait preflight
  payloads, row-major decoding, scalar result paths, and output-only Firth
  diagnostic arrays are removed from production reachability.
- Backend chromosome preparation accepts the canonical prediction matrix view
  directly, and output completion returns `Vec<CompletedOutputRun>` directly.
- Alternate result writers, format selection, final output paths, and the
  consolidation module are deleted. Parquet part commits carry the metadata
  required for strict resume.
- Output schema 3 removes legacy `TEST`/`EXTRA` columns, uses non-null Arrow
  fields, and represents invalid statistics as `NaN` with explicit score/Firth
  success or failure labels.
- Internal correction codes are contiguous method/outcome states; no legacy
  output code or text label travels through JAX, PyO3, engine, or output.
- Manifest schema 14 stores immutable compatibility data once in
  `execution_plan`. Engine passes typed phenotype-specific inputs and typed
  LOCO fingerprints; JSON is created only inside `g-output` at persistence.
  Every compute group carries a required fingerprint of its aligned
  trait-major phenotype values so resume cannot combine chunks computed from
  different phenotype matrices.
- Parquet schemas and dictionary values are process-cached, writer jobs share
  immutable configuration, and strict resume propagates directory errors while
  opening each part once.
- The output resume/timing, engine preparation, and binding engine one-child
  module directories are flattened. Null-solver iteration/reason diagnostics
  that never affected native policy no longer leave JAX.

## Remaining Stabilization

- Update deliberately deferred tests and benchmark/tooling configurations to
  the Parquet-only config and artifact names after the production API settles.
- Run the full CPU/GPU correctness and performance matrix only after those
  callers are migrated.
- Keep reviewing `src/binding/cli.rs` terminal/error adaptation. Code that must
  retain a concrete `PyErr`, attach to Python, or configure the Python process
  belongs there; any newly discovered crate-only orchestration belongs in
  `g-engine` or `g-runtime`.

## Numeric Contracts

- Configuration thresholds and tolerances are validated finite `f64` values.
- Host phenotype, covariate, LOCO, dosage, and summary buffers remain `f32`.
- Score-test operands and output statistics use `float32`.
- Firth solver operands are always `f64`; corrected values narrow once into float32 score results.
- Epsilon and convergence operands derive from the active JAX dtype. Step-halving scales are in `(0, 1)`, probability thresholds in `(0, 1)`, and sparse dosage thresholds in `(0, 2]`.
- Rust memory indices and shapes use `usize`; JAX indices, loop counts, and count arrays use checked `i32`; persisted counts/byte sizes use fixed-width integers. Correctness paths do not use saturation or multi-stage sign-changing conversions.
- [x] Enforce the integer contract end to end: deny unsafe Rust integer casts; keep raw pointers behind exposed-provenance wrappers; serialize telemetry counters as fixed width; retain unsigned host configuration but validate and expose JAX loop/capacity controls as `i32`; reject out-of-domain sample, trait, chunk, flattened-lane, and padded-batch sizes before backend dispatch; and force index-producing JAX operations to `int32` under x64 mode.

## Acceptance

- Root `_core` uses domain crates only to implement the private PyO3/NumPy
  adapter; it owns no domain orchestration or policy.
- No unbounded association/output channels or global output worker pool remain.
- No string/JSON execution DTO travels between Rust domain crates; JSON exists only at persistence and telemetry serialization edges.
- Completed chunks do not clone genotype metadata/statistics or materialized result matrices before output.
- Single- and multi-trait runs use the same Rust preparation and delivery model.
- Binding contains no BGEN, alignment, resume, writer, telemetry-lifecycle, or execution-policy implementation.
- Architecture, crate-boundary, binding, integer, float, configuration, and output documentation reflects the new contracts.
- Validation is library/static only: formatting check, `cargo check -j 30 --workspace --lib`, Clippy, Ruff, Python type checking, docs build, dependency scan, and `git diff --check`, using mold on the head node. Tests, benches, all-target builds, and tooling changes are deferred.
