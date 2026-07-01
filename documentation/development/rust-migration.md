# Rust Migration

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; active migration contract | main branch as of 2026-06-30 Cargo workspace migration | Development maintainers |

This page is the durable migration contract for moving `g` from a single Rust
package embedded in a Python-owned application into a Cargo workspace where Rust
owns host orchestration and Python owns public convenience APIs plus JAX compute
kernels.

Baseline commit: `25ec7701b8b4a9339d424d8d08b1bc4527ddb2c8`.

## Direction

The end state keeps the root package named `g` and the Maturin/PyO3 library
target named `_core`. The root crate becomes a composition and adapter crate.
Internal domain crates must remain Python-free unless they are the root binding
crate.

Target crates:

| Crate | Responsibility |
| --- | --- |
| `g-plan` | Immutable requested and prepared run contracts. |
| `g-interface` | Clap, TOML, defaults, overlays, validation, and config-to-plan conversion. |
| `g-genotype` | BGEN mmap/index/decode, genotype chunk planning, preprocessing, and genotype benchmarks. |
| `g-input` | Sample, phenotype, covariate, prediction-list, and LOCO alignment. |
| `g-output` | Output paths, Arrow/Parquet/REGENIE writing, manifests, resume, and finalization. |
| `g-runtime` | Logging, tracing, telemetry, timing, runtime policy, Rayon policy, and shutdown. |
| `g-engine` | Application state machine, preflight, batching, queues, backend trait, and cleanup. |

No internal crate may depend on `pyo3` or `numpy`. The root `g` crate may depend
on all internal crates and remains the only native Python binding crate.

## Non-Negotiable Invariants

- Preserve sample order, phenotype and covariate masks, LOCO prediction
  alignment, allele orientation, output row order, correction selection,
  correction status, and fresh/resumed equivalence unless a separate
  science-change issue explicitly approves the difference.
- Keep `g --help`, `g regenie --help`, CLI exit codes, TOML merge/default
  behavior, `effective_config.toml`, the Python `g.api` surface, PyO3 export
  names, and `_core.pyi` stable unless an interface-change issue approves the
  change.
- Keep crate dependencies acyclic and phase-aware. Do not introduce generic
  dumping-ground crates such as `g-utils`, `g-common`, or `g-types`.
- Do not keep permanent production Python fallbacks for Rust-owned behavior.
  Temporary dual implementations are allowed only for equivalence tests and must
  have a removal task.

## Migration Ledger

| Area | Current Python owner | Current Rust helper | Target crate | Target owner | Removal status | Primary tests and benchmarks |
| --- | --- | --- | --- | --- | --- | --- |
| CLI/TOML/config frontend | `src/g/cli.py`, `src/g/interface/config.py` | `crates/interface/src/` | `g-interface` | Rust | Extracted as native crate; Python remains adapter | `tests/test_interface.py`, CLI help/package smoke |
| Execution planning | `src/g/execution_plan.py`, `src/g/engine/backend_planner.py` | `crates/plan/src/`, `crates/interface/src/plan_request.rs`, native metadata helpers | `g-plan`, then `g-engine` | Rust | Started: `g-interface` compiles resolved config into `g-plan::RunRequest`; manifest headers now serialize through a Rust-built `g-plan::PreparedRunPlan` consumed by `g-output`, with backend validation, prepared-plan assembly, and backend-kind derivation owned by `g-plan`; Python still supplies other transitional header fields from legacy dataclasses until dynamic preparation moves to Rust | `tests/test_backend_planner.py`, `tests/test_regenie2_pipeline.py`, `tests/test_io_output.py`, `cargo test -p g-plan`, `cargo test -p g-interface`, `cargo test -p g-output` |
| BGEN and genotype preprocessing | Python native-dispatch wrappers | `crates/genotype/src/` | `g-genotype` | Rust | Extracted as leaf crate | Rust genotype tests, `just benchmark-bgen-reader`, `cargo bench -p g-genotype` |
| Sample and phenotype alignment | `src/g/io/source.py`, pipeline wrappers | `crates/input/src/sample/` | `g-input` | Rust | Extracted as native crate; Python remains adapter | `tests/test_io_sample.py`, `tests/test_tabular.py`, `cargo test -p g-input` |
| Prediction and LOCO input | `src/g/engine/native_dispatch/loaders.py` | `crates/input/src/regenie/` | `g-input` | Rust | Extracted as native crate; Python remains adapter | pipeline, parity, and LOCO alignment tests |
| Output, manifest, and resume | `src/g/io/output.py`, callbacks/writers wrappers | `crates/output/src/` | `g-output` | Rust | Extracted as native crate | `tests/test_io_output.py`, `cargo bench -p g-output`, `just benchmark-output-stages-gpu` |
| Runtime, telemetry, timing, shutdown | `src/g/runner/`, `src/g/engine/telemetry.py`, `src/g/engine/timing.py`, `src/g/engine/shutdown.py` | `crates/runtime/src/`, root PyO3 adapters | `g-runtime` | Rust | Started: pure runtime policy without Python fallback, aggregate runtime policy handles, run runtime handles, telemetry path/session policy handle, telemetry run session handle, telemetry run ID generation, telemetry session cap/counter/envelope/progress-throttle/close-metadata state, telemetry close planning and close-metadata payloads, current telemetry event metadata and payload construction, telemetry event/progress emission gating, telemetry event/close-event/JSON-line emission, run artifact metadata attachment, execution artifact tree construction, completed/interrupted/failed run lifecycle event construction/payload/rendering policy without Python fallback, shutdown metadata/controller state, shutdown handler lifecycle planning, and repeated-signal exception planning without Python fallback, timing state, stage timing recorder handle, exact-stage timing policy, timing recorder creation/write gating, transfer metadata shape/byte expansion, stage-timing/profile JSON payloads and file writes, run metadata, default local runtime cache path construction, JAX runtime policy payload construction, JAX runtime setup session handle, JAX runtime setup/config-update/GPU-validation/diagnostic event payloads, diagnostic record planning, setup lifecycle planning, setup side-effect planning, setup validation completion, and NVIDIA driver visibility probing, runtime compatibility tokens guarding output preparation, Rayon global thread-pool initialization/configuration planning/failure formatting, and logging/Rayon/JAX process state handle extracted; Python still owns other side effects and JAX setup side-effect execution | `tests/test_telemetry.py`, `tests/test_timing.py`, `tests/test_jax_runtime.py`, shutdown tests, `cargo test -p g-runtime` |
| Pipeline scheduling and queues | `src/g/engine/native_dispatch/`, `src/g/engine/callbacks/` | `crates/engine/src/` | `g-engine` | Rust | Started: crate scaffolded with explicit run phases, backend trait, deterministic fake backend, deterministic fake side effects, single-batch coordinator, native side-effect hook contract for input/preflight/output/telemetry/finalization operations with abort cleanup on initialized-output failures, fake output lifecycle/manifest state and abort counting for Rust-only coordinator tests, injected failures for every entered phase, backend trait-method failures, interruption tests, a coordinator-overhead benchmark, the BGEN-backed `Regenie2RunEngineCore`/chunk planning core, native required-chromosome resolution, native preflight report/warning/scan-count helpers, native preflight shape/count, binary class-count, and prediction-shape policy payloads, native all-manifest resume compatibility preflight and output-run initialization orchestration using `g-output`, native pipeline output preparation batch handle, native pipeline output initialization result handle, native committed-chunk intersection for multi-output resume scheduling, native callback batch-size delivery policy, native grouped-union callback batch-size policy, native callback queue-limit policy, native callback queue stage/operation observation policy and backpressure payload construction, native callback queue put/get attempt planning, native variant-major dosage batch handoff planning, native result in-flight slot accounting and acquire/release attempt planning, native dosage-buffer pool accounting and acquire/register/return/discard attempt planning, native dosage-buffer reuse shape planning, native writer-finish thread cleanup and execution planning, native BGEN delivery invocation and cleanup lifecycle action planning/execution order, native output write method planning, native effective trusted BGEN mode resolution, native callback worker lifecycle start state, start action planning, and start-attempt lifecycle marking, native callback worker shutdown timeout, stop/join-planning, finish/abort action planning, worker error raise planning, stop-loop poll policy, and failure message formatting, native callback worker stop-attempt decision policy, native callback worker backpressure poll-timeout policy, native BGEN delivery method selection, native GPU genotype-format auto/resume/validation-result policy, native null-logistic nonconvergence policy, native binary correction summary accumulation state, native callback progress/chunk identity state and progress telemetry payload planning, native array-finiteness/covariate-rank/binary-coding preflight policy, and native multi-trait committed-chunk write selection; production queues, writer side-effect calls, telemetry side-effect writes, NumPy/JAX numeric scans, and PyO3/JAX backend wiring remain transitional | `tests/test_callback_lifecycle.py`, callback overhead benchmarks, `tests/test_preflight.py`, `tests/test_regenie2_pipeline.py`, `cargo test -p g-engine`, `cargo bench -p g-engine --bench coordinator` |
| JAX kernels | `src/g/compute/`, `src/g/jax_runtime/` | PyO3 array adapters | Root adapter plus Python modules | Python | Kept | kernel tests, parity harness, GPU benchmarks |

Phase 10 queue migration also has a native callback scheduler state handle that
consolidates queue limits, worker-start state, result in-flight accounting, and
dosage-buffer pool accounting and reuse planning, worker failure state, worker
finish/abort planning, worker stop/join planning, chunk batch handoff planning,
queue observation planning, bounded-resource backpressure observation payload
construction, worker start action planning, and backpressure timeout policy for
production callback runners. Callback worker start attempts now mark the native
lifecycle state through native scheduler plans. Callback worker finish and abort sequencing now
comes from native action lists alongside native timeout policy. Callback modules
no longer resolve worker shutdown timeout constants at import time; production
shutdown timeout policy comes from native scheduler plans. Result in-flight
capacity is no longer mirrored by a Python
bounded semaphore; the native scheduler state is the capacity owner while the
Python transition layer only waits on slot-release notifications. Result
in-flight acquire/release decisions now come from native attempt plans. Free
host dosage buffers are no longer held in a bounded Python queue; the native
scheduler state owns buffer capacity and the Python transition layer keeps only
an available-buffer free list. Dosage-buffer acquisition, allocation
registration, return, and discard decisions now come from native attempt plans.
Worker error propagation now uses a native raise plan, with Python selecting
only the local exception cause to chain.
Result write queue occupancy is now tracked by native scheduler state instead
of a bounded Python queue, with Python retaining only transitional item storage
and wakeups. Dosage queue occupancy is now
tracked by native scheduler state instead of a bounded Python queue, with Python
retaining only transitional item storage and wakeups. Dosage and result queue
put/get wait decisions now come from native attempt plans. Multi-trait linear
and binary result consumers now always use the native runtime-resource
get/drain loop instead of a production Python fallback loop. The base callback
runner's dosage and single-result consumers now also require native
runtime-resource get/drain helpers, leaving manual scheduler consumer loops in
test fixtures only. Base dosage/result drain and dispatch planner helpers now
also require native runtime resources, with manual scheduler planner helpers
confined to test fixtures. Base dosage handoff and variant-major batch handoff
planner helpers now follow the same native-resource-only rule. The base
callback runner no longer branches on manual versus native runtime-resource
ownership; manual scheduler lifecycle, queue, result-slot, dosage-buffer,
progress, summary, handoff, and cleanup paths are test-fixture behavior only.

## Phase Order

1. Establish baseline records and promote this migration contract.
2. Add Cargo workspace infrastructure without moving production modules.
3. Extract `g-genotype`.
4. Extract `g-output`.
5. Extract `g-interface`.
6. Extract `g-input`.
7. Introduce `g-plan` and make manifests derive from `PreparedRunPlan`.
8. Extract and strengthen `g-runtime`.
9. Create `g-engine` with a deterministic fake backend.
10. Move preparation, scheduling, queues, and runtime side effects into Rust.
11. Collapse the root crate into a PyO3/JAX adapter.
12. Add a native CLI only after the Rust coordinator owns the full lifecycle.
13. Remove obsolete Python orchestration.

Each phase should be independently revertible. Mechanical extraction,
ownership transfer, fallback removal, and performance optimization should be
separate changes unless the change is too small to split safely.

## Required Checks

Every crate extraction must run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
cargo build --workspace --all-targets
cargo test --workspace
uv run pytest tests/ -m "not phase0_data and not phase1_parity"
just check-core-stub
just check-internal-defaults
just check-internal-init-exports
just check-rust-architecture
uv run maturin develop
uv run python -c "import g; import g._core"
uv run g --help
```

Correctness-boundary phases also need parity harness coverage, manifest/schema
snapshots, fresh-versus-resumed equivalence, and malformed input tests. Hot-path
phases need Criterion benchmarks plus representative CPU/GPU benchmarks on the
appropriate SLURM nodes.

## Stop Conditions

Pause rather than force a phase through when a dependency cycle appears, a leaf
crate needs PyO3, sample or prediction alignment changes unexpectedly, resume
mutates outputs before validation, parity changes cannot be attributed, or a
performance regression cannot be measured and explained.
