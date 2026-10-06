# Architecture Cleanup

| Status | Applies to | Owner |
| --- | --- | --- |
| Maintained production and tooling cleanup | `src/`, `crates/`, and `tooling/` as of 2026-10-06 | Development maintainers |

This page records the implemented cleanup and the decisions that replaced the
earlier migration plan. Tests and Hydra-based benchmark/profiling tooling use
the current native host. Architecture checks, mathematical regressions,
required-fixture parity, and performance qualification remain maintained gates.

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
  mode-specific batch-oriented JAX backends
  JAX kernel state and statistical computation
```

The root PyO3 module exposes only `g._core.cli`. There are no legacy aliases,
backend exchange classes, callback APIs, writer APIs, runtime APIs, or config
object graphs registered for Python.

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
- Limited the backend to group/chromosome preparation, host or compressed
  transfer, shared-source preparation/selection, computation, and materialization.
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
| PyO3 and Python | The input, output, lifecycle, conversion, and JSON adapter trees are deleted. Telemetry lifecycle is runtime-owned. Python contains only console forwarding, batch-oriented backends, and JAX kernels. |
| Dependency and integer audit | Cargo dependency scanning reports no unused dependencies. Production engine/binding code has no unchecked integer `as` casts or bare tuple result mirrors. |

Maintained architecture guards cover canonical crate facades, imports, casts,
exports, and Python ownership boundaries. Tests and tooling are included in
the supported validation surface rather than requiring compatibility exports.

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

## Ongoing Maintenance

Keep tests and tooling aligned with the CLI-only `_core` API. Remove private
helpers only after tracing configured, native, platform-specific, and test
entrypoints. Static reachability checks complement reviewed symbol analysis;
neither proves that every branch is necessary. Do not add production
compatibility exports to make stale tests or tooling pass.

The profiling CLI delegates to modules that own artifact handling, commands,
trials, profiler execution, and reporting. Python backend transport and
materialization likewise have private owners, while the mode-specific classes
remain the stable Rust import boundary. Native variant metadata uses shared
immutable ownership across tiled groups, including its lazy Arrow-array cache.

## Current Validation

Run production qualification on allocated compute nodes. On Gauss, use Slurm
for compilation, full tests, and CPU profiling; use Landau for GPU work. Cargo
uses the allocated CPU count, and node-specific targets isolate native builds.
Nix development environments are preferred where available; the maintained
server environment provides the toolchain on servers without Nix.

```bash
just check
just test
cargo test --workspace
cargo machete
just docs-build
git diff --check
```

Changes to scientific kernels or native delivery also require appropriate GPU
tests and `just test-parity-required` with local fixtures. Qualify performance
against frozen baseline and candidate sources/binaries, record excluded warm
runs separately, and audit persisted results. A warmed lifecycle benchmark
does not measure pure compilation time. Keep observer runs separate from
headline timing.
