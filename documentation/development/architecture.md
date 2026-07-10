# Architecture

| Status | Applies to | Owner |
| --- | --- | --- |
| Active production architecture | Production code as of 2026-07-10 | Development maintainers |

`g` is a Rust host application with a Python/JAX numerical backend. Rust owns
configuration, planning, input, scheduling, lifecycle, telemetry, shutdown,
resume, and output. Python owns JAX arrays and statistical kernels.

## Production Flow

```text
g.cli (argument/output bootstrap only)
        |
g._core.cli.run(arguments)
        |
g-interface -> g-plan -> g-engine::RunEngine::prepare
        |                       |
        |                       +-> g-genotype + g-input
        |                       +-> g-runtime + g-output
        v
PreparedRun::execute<AssociationBackend> + bounded pipeline
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
| BGEN mmap/index/decode and genotype preprocessing | `g-genotype` |
| Sample, phenotype, covariate, and prediction alignment | `g-input` |
| Host buffers, BGEN delivery primitives, result writing, backend trait, bounded compute/materialize pipeline | `g-engine` |
| Logging, telemetry, timing, process policy, SIGTERM | `g-runtime` |
| Parquet writers, manifests, and resume | `g-output` |
| PyO3 object construction, opaque Python state, NumPy conversion, PyErr adaptation | root Rust extension under `src/binding` |
| Device state and association mathematics | `src/g/jax_backend.py`, `src/g/compute/` |

`g-engine` is the only root dependency through which the binding reaches
genotype, input, and output services. `RunEngine` owns the shared plan and
output manager; `PreparedRun` owns aligned groups, BGEN state, resume state,
bounded scheduling, reusable host buffers, delivery, interruption, abort, and
output completion. The binding retains only calls that attach to Python, hold opaque
JAX objects, preserve a concrete `PyErr`, or label telemetry with the current
Python thread.

`g-engine::execute_coordinated_run` is the coarse native coordinator used by
the binding. It owns preparation, execution, delivery-report handling,
completed-artifact construction, and writer-completion telemetry. The binding
does not orchestrate a chain of low-level crate functions.

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
- Rust owns all bounded queues, worker joins, host buffers, and output order.
- Planning enums have one definition in `g-plan`; runtime/output crates consume
  those types directly.
- Resume commit sets and the immutable `RunPlan` use shared ownership rather
  than per-group clones.
- CLI validation and help complete before importing JAX-heavy modules.
- First SIGINT/SIGTERM preserves resumable output; a second SIGTERM uses the
  operating system default action.
- Unsupported REGENIE options are rejected, not silently adapted.
- Rust host scientific buffers are `f32`; `score_dtype` controls JAX arithmetic
  and result width after device transfer. Firth arithmetic is always `f64`.
- Production exports are limited to `g._core.cli` and `g._core.engine`.

Detailed contracts:

- [Architecture Cleanup](architecture-cleanup.md)
- [Binding Layer Policy](binding-layer-policy.md)
- [Rust Crate Boundaries](rust-crate-boundaries.md)
- [Configuration Frontend](configuration-frontend.md)
- [Native I/O](native-io.md)
- [Compute Kernels](compute-kernels.md)
- [Floating-Point Policy](floating-point-policy.md)
