# Architecture Cleanup

| Status | Applies to | Owner |
| --- | --- | --- |
| Production cleanup implemented; stabilization deferred | `src/` and `crates/` as of 2026-07-10 | Development maintainers |

This page records the implemented cleanup and the decisions that replaced the
earlier migration plan. Tests, benchmark tooling, Hydra configuration, and
Justfile cleanup remain a separate stabilization project.

## Result

The production application now has one ownership model:

```text
Rust owns:
  CLI/config validation and run planning
  BGEN and tabular input
  sample/prediction alignment and preflight
  bounded scheduling, workers, host buffers, and output order
  runtime, telemetry, timing, interruption, cleanup, and Parquet persistence

Python owns:
  console forwarding
  one four-operation JAX backend
  JAX kernel state and statistical computation
```

The root PyO3 module exposes `g._core.cli` plus read-only exact-source build
provenance attributes. There are no legacy aliases, backend exchange classes,
callback APIs, writer APIs, runtime APIs, or config object graphs registered
for Python.

## Implemented Changes

### Native backend and scheduler

- Replaced the synthetic coordinator/effect scaffold with the production
  `AssociationBackend` contract.
- Added typed group, chromosome, genotype, null-diagnostic, materialization,
  and host-result contracts.
- Added the bounded `AssociationBatchPipeline` with separate compute and
  materialization workers, a bounded completed-result queue, explicit
  close/drain/join, panic/error propagation, and cancellation-aware abort.
- Kept variant metadata, output identity, result validation, and writing in
  Rust.
- `g-genotype` owns decoded batch, buffer, and compute-statistics contracts.
  `g-engine` owns BGEN decode orchestration, chromosome validation,
  scheduling, and completed-result writing. Backend-bound genotype, phenotype,
  covariate, and single-use LOCO buffers transfer their allocation directly
  into NumPy. Repeated noncontiguous chromosome blocks retain a counted
  prediction clone fallback.

### Python JAX island

- Replaced the single/multi/grouped callback hierarchy with mode-specialized
  linear, binary-score, and binary-Firth backend classes.
- Limited the backend to `prepare_group`, `prepare_chromosome`, `compute_batch`,
  and `materialize_batch`.
- Reused kernel state dataclasses directly instead of adding one-field wrapper
  state types.
- Split binary score policy and chromosome state from approximate-Firth state.
  Score-only construction receives only numerical and null-logistic policy and
  retains only score-kernel operands. The Firth state composes that compact
  score state with its null-Firth offsets, likelihoods, and deviance arrays.
- Removed all registered backend config and matrix exchange PyClasses. The
  private binding passes NumPy arrays directly and selects mode/correction once
  during backend construction.
- Performs one batched `jax.device_get` per materialized result.
- Removed output-only PyClasses and dead PyClass getters. Python now returns one
  ordinary typed host-result dataclass that Rust consumes directly.
- Deleted Python worker, writer, transfer, timing, lifecycle, telemetry, and
  runtime wrappers.

### Native host path

- Native Rust dispatch now covers single-trait, complete-case multi-trait, and
  per-phenotype modes through the same direct delivery implementation.
- `RunEngine::prepare` owns BGEN opening, input alignment, preflight, resume,
  manifest headers, and writer initialization. Consuming
  `PreparedRun::execute_with_progress` owns delivery and every terminal output
  path.
- Output sessions are plain Rust values; Python never writes output.
- Native CLI validation/help precedes JAX import and backend construction.
- SIGINT uses Python pending-signal checks. SIGTERM uses a native first-signal
  request flag and second-signal default action.
- Configured stage timing and profile outputs are written by the Rust recorder
  on every terminal path without masking a primary run failure.
- `g-output` consumes canonical `g-genotype-contracts` metadata/statistics and
  constructs each chunk's Arrow metadata set once for its trait writers.
- Output workers are run-scoped and bounded; the global pool and per-phenotype
  coordinator are deleted.
- Completed outputs use one `CompletedOutputRun` per phenotype rather than five
  parallel vectors. Resume commit sets and the run plan are shared immutably.
- Telemetry state, serialization, writer counters, and close lifecycle moved
  from the binding into `g-runtime::TelemetryRunSession`.

### Removed surface

- Deleted callback schedulers, queues, progress/summary wrappers, callback
  resource bundles, and the unused coordinator.
- Deleted the Python `g.engine` tree, Python runner lifecycle/runtime modules,
  and the separate `jax_runtime.py` wrapper.
- Deleted unregistered PyO3 config, input, genotype, output, runtime, and
  telemetry adapter graphs.
- Deleted inert `native_callback_batch_size` and `dosage_buffer_limit` config
  fields; neither influenced the new scheduler.
- Deleted the standalone Rust CLI subprocess bridge, dead prepared-plan graph,
  row-major production BGEN path, scalar Firth path, unused binary batch
  diagnostics, and unused shutdown-controller hierarchy.
- Removed `step`, `firth_dtype`, `is_validated`, `assume_validated`, SPA, and
  exact-Firth configuration states.
- Removed the output-format choice entirely and removed duplicate resume-mode,
  statistic-dtype, telemetry-mode, and backend-plan types; the remaining
  canonical definitions are in `g-plan`.
- Removed alternate result writers plus the derived-file consolidation path.
  `parts/part_*.parquet` is the sole result contract and requires no post-run
  materialization.
- Removed always-disabled BGEN reader profiling and output-only per-variant
  Firth diagnostic arrays.
- Flattened module directories whose private builders or data models had only
  one consumer.
- Canonical TOML accepts snake_case only. The CLI accepts only `--config` and
  the supported REGENIE Step 2 flags.

## Original Roadmap Accounting

| Original phase | Production result |
| --- | --- |
| Inventory, facades, and errors | Every domain crate exports one documented `api.rs` facade. Dead umbrella errors and convenience constructors are deleted. Public production APIs use crate-owned typed errors; no public `Result<T, String>` or library `anyhow::Result` remains. |
| Plan and interface | Configuration compiles to one typed `RunPlan`. Duplicate enum mirrors, prepared-plan DTOs, Python option normalization, and compatibility aliases are deleted. Numeric controls use validated finite `f64` newtypes. |
| Input and genotype | Alignment workflows return `InputResult` and moved out of `sample/mod.rs`. LOCO files are structurally indexed once per canonical path; identical loader-only headers share one identifier index and alignment recipe, indexed metadata plus row digests protect deferred reads, and only post-resume chromosome blocks are parsed and assembled lazily into final trait-major matrices. BGEN variant IDs use one UTF-8 arena and repeated chromosome/allele text uses compact dictionary codes. The production decoder is split into matrix, probability, and variant-major modules; the row-major production path is deleted. Owned decoding initializes reserved typed-vector capacity and publishes it only after complete success; no raw address/count contract crosses a crate boundary. |
| Output | Canonical `g-genotype-contracts` DTOs flow directly into `NativeChunkHandle`; `g-output` does not depend on the BGEN implementation crate. A run-scoped bounded worker pool is shared by Parquet writer sessions; the global pool, coordinator, duplicate DTOs, row-copy write plan, alternate writers, and derived-file consolidation are deleted. Manifest and resume counts cross checked signed `i64` boundaries. |
| Runtime | Duplicate facades, callback-era diagnostics, event-specific payload builders, JAX policy, packed8-validation cache policy, and public event constants are deleted. Runtime owns generic logging/telemetry/timing/shutdown infrastructure. |
| Engine | The backend is batch-oriented and Python-free. `RunEngine`/`PreparedRun` own preparation, delivery, packed8 negotiation, and writer completion; the genotype crate owns compatibility validation. Scheduler helpers stay internal and the bounded pipeline retains ownership of queues, joins, first-error capture, drain, and abort. |
| PyO3 and Python | The input, output, lifecycle, conversion, and JSON adapter trees are deleted. Telemetry lifecycle is runtime-owned. Python contains only console forwarding, the four-operation backend, and JAX kernels. |
| Dependency and integer audit | Cargo dependency scanning reports no unused dependencies. Production engine/binding code has no unchecked integer `as` casts or bare tuple result mirrors. |

Architecture guard source changes remain part of tooling stabilization because
tooling was explicitly excluded from this production pass. Equivalent direct
facade, error, import, cast, dead-code, dependency, and export scans pass on the
production tree.

## Binding Reduction

The root crate depends directly on `g-runner`, `g-engine`, and canonical
`g-plan` contracts. It imports owner-defined `g-genotype`, `g-input`, and
`g-output` payload types only at the private `AssociationBackend`/NumPy
boundary; it does not call their services. Adapter-specific settings,
`g-interface`, and `g-runtime` remain behind `g-runner`, which owns dispatch,
process policy, timing, terminal rendering, and the coordinated engine call.
`g-engine` owns preparation, decode orchestration, scheduling, and result
delivery. `g-runner` owns terminal output policy. Binding code retains only Python
attachment, opaque JAX objects, NumPy conversion, Python thread labels, and
original `PyErr` adaptation. The binding implements the runner's Python host
callbacks; no lifecycle is assembled in `src/binding/cli.rs`.

## Preserved Contracts

- Statistical formulas, correction selection, sample masks,
  LOCO alignment, allele orientation, row order, and output schemas.
- Fresh/resumed equivalence and manifest compatibility for the Parquet dataset
  contract.
- Supported REGENIE Step 2 option spellings.
- Quantitative, binary score-only, approximate-Firth, single, complete-case,
  per-phenotype, dosage, and packed8 production paths.

Python/PyO3 internals, output-only diagnostics, manifest `firth_dtype`,
camelCase TOML aliases, callback-era tuning knobs, and unreleased helper APIs
were intentionally not preserved. The active dtype contract is documented in
[Floating-Point Policy](floating-point-policy.md).

## Stabilization Work

After the production API settles:

1. Delete or migrate stale tests to the CLI-only `_core` API.
2. Update benchmark and profiling tooling to the new native host path.
3. Run the full CPU/GPU correctness matrix and capture new performance
   baselines.
4. Remove stale ignored local build/import artifacts from developer checkouts as
   needed; they are not source or package contents.

Do not add production compatibility exports to make stale tests or tooling pass.

## Current Validation

Production changes should run directly on the development host with the
configured mold linker and 30 Cargo jobs:

```bash
cargo fmt --all --check
cargo check -j 30 --workspace --lib
cargo clippy -j 30 --workspace --lib --no-deps -- -D warnings
uv run --no-sync ruff format --check src/g
uv run --no-sync ruff check src/g
uv run --no-sync ty check src/g
cargo machete
just docs-build
git diff --check
```

Tests, benches, and all-target compilation are intentionally not part of this
validation pass. They still reference removed unreleased APIs and must be
updated during stabilization rather than forcing compatibility exports back
into production.

GPU association runs and large CPU scans still require an appropriate compute
node. Development compilation and static checks do not require SLURM.
