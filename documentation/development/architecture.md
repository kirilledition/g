# Architecture

| Status | Applies to | Owner |
| --- | --- | --- |
| Active production architecture | Production code as of 2026-07-19 | Development maintainers |

`g` is a Rust host application with a Python/JAX numerical backend. Rust owns
configuration, planning, input, scheduling, lifecycle, telemetry, shutdown,
resume, and output. Python owns JAX arrays and statistical kernels.

## Production Flow

```text
g.cli (argument/output bootstrap only)
        |
g._core.cli.run(arguments)
        |
g-runner::run_cli
        |
g-interface -> g-plan -> g-engine::execute_coordinated_run
        |                       |
        |                       +-> g-genotype + g-input
        |                       +-> g-genotype-contracts
        |                       +-> g-runtime + g-output
        v
PreparedRun::execute_with_progress<AssociationBackend, RunHooks> + bounded pipeline
        |
four-operation PyO3 adapter
        |
g.jax_backend -> JAX kernels
```

The backend operations are `prepare_group`, `prepare_chromosome`,
`compute_batch`, and `materialize_batch`. The first three may retain opaque JAX
state. Materialization performs one batched device-to-host transfer and returns
typed arrays to Rust.

## Ownership

| Area | Owner |
| --- | --- |
| CLI parsing, TOML, defaults, validation | `g-interface` |
| Immutable run contracts and host policy | `g-plan` |
| Shared variant metadata and output-facing genotype columns | `g-genotype-contracts` |
| BGEN owned-snapshot/positioned source I/O, index, immutable read sessions, delivery planning, decode/preprocessing, and owned decoded or compressed batches/buffers | `g-genotype` |
| Optional capability-gated packed8 raw-DEFLATE delivery and typed-XLA FFI handlers | `g-genotype-cuda` |
| Optional capability-gated CUDA association kernels and typed-XLA FFI handlers | `g-compute-cuda` |
| Private header-only CUDA driver ABI, loading, and device/context validation | `native/cuda-driver` |
| Sample, phenotype, covariate, and prediction alignment | `g-input` |
| BGEN delivery orchestration, result writing, backend trait, bounded compute/materialize pipeline | `g-engine` |
| Logging/telemetry transport, timing, Rayon state, SIGTERM | `g-runtime` |
| CLI lifecycle, process/JAX policy, crate orchestration, terminal rendering | `g-runner` |
| Parquet writers, manifests, and resume | `g-output` |
| PyO3 object construction, opaque Python state, NumPy conversion, PyErr adaptation, JAX process calls | root Rust extension under `src/binding` |
| Device state, compressed-device decode invocation, and association mathematics | `src/g/jax_backend.py`, `src/g/compute/`, and capability-gated kernels in `g-compute-cuda` |

`g-runner` is the root dependency through which the binding reaches CLI,
planning, runtime, and execution services. It owns CLI dispatch, resolves the
application plan into concrete logging, telemetry, and timing resources,
process-global runtime compatibility, terminal rendering, and JAX setup before
the coarse engine invocation. `g-runtime` consumes only that generic resolved
session policy; it does not import or interpret `g-plan`. `RunEngine` owns the shared plan
and output manager; `PreparedRun` owns aligned groups, BGEN state, resume state,
bounded scheduling, move-only backend buffers, delivery, interruption, abort,
and output completion. Input retains shared
LOCO row indexes and compact alignment recipes, then materializes only the next
post-resume chromosome. Group, genotype, and single-use LOCO matrices move
directly into NumPy ownership. The binding
retains only calls that attach to Python, hold opaque
JAX objects, preserve a concrete `PyErr`, or label telemetry with the current
Python thread. It imports canonical genotype, input, and output payload types
only to convert the private `AssociationBackend` boundary to and from NumPy;
all services remain behind `g-engine` and `g-runner`.

Engine resolves the prediction list once into an input-owned path catalog.
Input indexing/alignment and output-manifest fingerprinting borrow that same
catalog, so group preparation does not reparse the list or duplicate its path
strings.

Genotype, engine, and output import one canonical set of data-plane column
contracts from `g-genotype-contracts`. Genotype retains dictionary-coded
metadata through a shared store and output builds Arrow metadata lazily, so the
crate boundary adds no row copies, eager arrays, or adapter DTOs.

The backend advertises a genotype-delivery capability to `g-engine`.
`g-genotype` chooses either an owned host-decoded batch or a canonical
compressed packed8 batch; `g-engine` schedules that typed payload without
knowing Python or CUDA details. The root backend only lends the crate-owned
slab and metadata to NumPy and calls the private JAX transfer method. The CUDA
FFI handler and all raw-DEFLATE validation/finalization live in
`g-genotype-cuda`; binding code performs only lazy library loading and target
registration required by PyO3/JAX.

Binary approximate-Firth GPU runs independently probe `g-compute-cuda` and
register its private component-reduction target once. Compatible runs use the
embedded PTX through typed XLA FFI; an unavailable driver, unsupported device,
or registration failure selects the mathematically equivalent JAX reduction.
This optional compute capability is not coupled to packed8 delivery or nvCOMP.
The two CUDA crates have independent Rust APIs and PTX/kernel ownership. They
source-include only the neutral repository-private `native/cuda-driver` support
header so CUDA ABI declarations, dynamic loading, and device/context checks
have one maintained implementation without a linked support library.

Runner holds process runtime state across logging-compatibility validation,
native session construction, subscriber installation, and topology recording.
An incompatible repeated run therefore cannot open a log/telemetry file or
start an asynchronous writer before it is rejected.

`g-runner::run_cli` is the coarse native coordinator used by the binding. It
owns the call to `g-engine::execute_coordinated_run`, which in turn owns
preparation, execution, delivery-report handling, completed-artifact
construction, and writer-completion telemetry. The binding does not
orchestrate domain crate calls.

## Python Surface

Production Python outside the kernels is limited to:

```text
src/g/cli.py          console bootstrap and output forwarding
src/g/jax_backend.py  typed four-operation JAX backend
src/g/compute/        JAX kernel state and mathematics
```

Python does not parse files, align samples, schedule workers, write results,
manage manifests/resume, select cleanup policy, or own telemetry lifecycle.

## Invariants

- Domain crates remain free of PyO3.
- The JAX boundary is batch-oriented; there are no per-variant Python calls.
- Rust owns all bounded queues, worker joins, host buffers, and output order;
  backend-bound genotype allocations move into NumPy without a second matrix.
- Planning enums have one definition in `g-plan`. Only domain owners interpret
  them; infrastructure crates such as `g-runtime` receive projected generic
  policy.
- Resume commit sets and the immutable `RunPlan` use shared ownership rather
  than per-group clones.
- CLI validation and help complete before importing JAX-heavy modules.
- Binary score-only execution does not construct Firth policy, fit a null-Firth
  model, compute full-null deviance, or retain Firth-only chromosome arrays.
  Approximate-Firth state composes the compact score state when correction is
  enabled.
- First SIGINT/SIGTERM preserves resumable output; a second SIGTERM uses the
  operating system default action.
- Unsupported REGENIE options are rejected, not silently adapted.
- Rust host scientific buffers, association scores, and persisted public
  statistics are `f32`. Approximate-Firth state, reductions, objective
  components, and validation are `f64`; its inner pseudo-logistic elementwise
  proposal is `f32`.
- Production exports are limited to `g._core.cli`.

Detailed contracts:

- [Architecture Cleanup](architecture-cleanup.md)
- [Binding Layer Policy](binding-layer-policy.md)
- [Rust Crate Boundaries](rust-crate-boundaries.md)
- [Configuration Frontend](configuration-frontend.md)
- [Native I/O](native-io.md)
- [Compute Kernels](compute-kernels.md)
- [Floating-Point Policy](floating-point-policy.md)
