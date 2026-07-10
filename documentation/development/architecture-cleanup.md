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
  runtime, telemetry, timing, interruption, cleanup, and finalization

Python owns:
  console forwarding
  one four-operation JAX backend
  JAX kernel state and statistical computation
```

The root PyO3 module exposes only `g._core.cli` and `g._core.engine`. There are
no legacy aliases, callback APIs, writer APIs, runtime APIs, or config object
graphs registered for Python.

## Implemented Changes

### Native backend and scheduler

- Replaced the synthetic coordinator/effect scaffold with the production
  `AssociationBackend` contract.
- Added typed group, chromosome, genotype, null-diagnostic, materialization,
  and host-result contracts.
- Added the bounded `AssociationBatchPipeline` with separate compute and
  materialization workers, deterministic drain, panic/error propagation, and
  abort handling.
- Kept variant metadata, output identity, result validation, and writing in
  Rust.

### Python JAX island

- Replaced the single/multi/grouped callback hierarchy with
  `JaxAssociationBackend`.
- Limited the backend to `prepare_group`, `prepare_chromosome`, `compute_batch`,
  and `materialize_batch`.
- Reused kernel state dataclasses directly instead of adding one-field wrapper
  state types.
- Performs one batched `jax.device_get` per materialized result.
- Deleted Python worker, writer, transfer, timing, lifecycle, telemetry, and
  runtime wrappers.

### Native host path

- Native Rust dispatch now covers single-trait, complete-case multi-trait,
  per-phenotype, and grouped-union modes.
- Grouped-union delivery decodes the union once and projects native group
  columns.
- Output sessions are plain Rust values; Python never writes or finalizes
  output.
- Native CLI validation/help precedes JAX import and backend construction.
- SIGINT uses Python pending-signal checks. SIGTERM uses a native first-signal
  request flag and second-signal default action.
- Configured stage timing and profile outputs are written by the Rust recorder
  on every terminal path without masking a primary run failure.

### Removed surface

- Deleted callback schedulers, queues, progress/summary wrappers, callback
  resource bundles, and the unused coordinator.
- Deleted the Python `g.engine` tree, Python runner lifecycle/runtime modules,
  and the separate `jax_runtime.py` wrapper.
- Deleted unregistered PyO3 config, input, genotype, output, runtime, and
  telemetry adapter graphs.
- Deleted inert `native_callback_batch_size` and `dosage_buffer_limit` config
  fields; neither influenced the new scheduler.
- Canonical TOML accepts snake_case only. The CLI accepts only `--config` and
  the supported REGENIE Step 2 flags.

## Original Roadmap Accounting

| Original phase | Production result |
| --- | --- |
| Inventory, facades, and errors | Every domain crate exports one documented `api.rs` facade. Dead umbrella errors and convenience constructors are deleted. Public production APIs use crate-owned typed errors; no public `Result<T, String>` or library `anyhow::Result` remains. |
| Plan and interface | Configuration has one native parser/default/overlay path. Python option normalization and compatibility aliases are deleted. |
| Input and genotype | Alignment workflows return `InputResult` and moved out of `sample/mod.rs`. The former 1,700-line BGEN `decode/mod.rs` is split into named matrix, probability, row-major, and variant-major modules with narrow BGEN-internal visibility. Raw caller-owned buffers validate address, alignment, counts, and offsets before unsafe writes. |
| Output | `session.rs` is a module boundary over writer-session, coordinator, worker-pool, and validation modules. Manifest, batch, finalizer, and resume counts cross checked signed `i64` boundaries, with explicit aggregate overflow errors. Dead reconstruction, write-plan, finalizer, and resume wrappers are deleted. |
| Runtime | Duplicate facades, callback-era diagnostics, and public event-name constants are deleted. Telemetry envelopes, fields, lifecycle payloads, counters, and close summaries serialize directly from typed Rust values while retaining their wire shape. |
| Engine | The backend is batch-oriented and Python-free. Scheduler helper types stay internal; the forwarding schedule trampoline is deleted. The bounded pipeline retains ownership of queues, worker joins, first-error capture, drain, and abort. |
| PyO3 and Python | CLI and output `mod.rs` files contain registration only. The unregistered input adapter tree and JSON bridge are deleted. Telemetry is a plain Rust session rather than a `PyClass`/`PyAny`/`PyDict` round trip. Python contains only console forwarding, the four-operation backend, and JAX kernels. |
| Dependency and integer audit | Cargo dependency scanning reports no unused dependencies. Production engine/binding code has no unchecked integer `as` casts or bare tuple result mirrors. |

Architecture guard source changes remain part of tooling stabilization because
tooling was explicitly excluded from this production pass. Equivalent direct
facade, error, import, cast, dead-code, dependency, and export scans pass on the
production tree.

## Deliberate Plan Change

The old target placed the entire run coordinator in `g-engine::RunEngine`.
Instead, `g-engine` owns the Python-free contract and performance-sensitive
scheduler, while the root Rust extension owns host orchestration that directly
coordinates opaque Python handles, `PyErr`, and terminal interruption flushing.

This avoids both prohibited alternatives: adding PyO3 to a domain crate or
creating a second generic adapter/error hierarchy that mirrors the actual host
session. The coordinator is Rust, not Python, and it has no parallel legacy
path.

## Preserved Contracts

- Statistical formulas, correction selection, dtype behavior, sample masks,
  LOCO alignment, allele orientation, row order, and output schemas.
- Fresh/resumed equivalence, manifest compatibility, output paths, and writer
  finalization behavior.
- Supported REGENIE Step 2 option spellings.
- Quantitative, binary score-only, approximate-Firth, single, complete-case,
  per-phenotype, grouped-union, dosage, packed8, Arrow, Parquet, and REGENIE
  production paths.

Python/PyO3 internals, camelCase TOML aliases, callback-era tuning knobs, and
unreleased helper APIs were intentionally not preserved.

## Stabilization Work

After the production API settles:

1. Delete or migrate stale tests to the two-submodule `_core` API.
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
