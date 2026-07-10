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
g-interface -> g-plan -> native Rust host session
        |                       |
        |                       +-> g-genotype + g-input
        |                       +-> g-runtime + g-output
        v
g-engine::AssociationBatchPipeline<AssociationBackend>
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
| Backend trait, bounded compute/materialize pipeline | `g-engine` |
| Logging, telemetry, timing, process policy, SIGTERM | `g-runtime` |
| Writers, manifests, resume, finalization | `g-output` |
| Native host coordination and PyErr adaptation | root Rust extension under `src/binding` |
| Device state and association mathematics | `src/g/jax_backend.py`, `src/g/compute/` |

The host coordinator remains in the root Rust extension because it directly
coordinates opaque Python backend handles and `PyErr` terminal behavior. Moving
it into `g-engine` would require PyO3 in a domain crate or a second generic
adapter/error hierarchy. `g-engine` instead owns the performance-sensitive,
Python-free scheduler and backend contract.

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
- CLI validation and help complete before importing JAX-heavy modules.
- First SIGINT/SIGTERM preserves resumable output; a second SIGTERM uses the
  operating system default action.
- Unsupported REGENIE options are rejected, not silently adapted.
- Production exports are limited to `g._core.cli` and `g._core.engine`.

Detailed contracts:

- [Architecture Cleanup](architecture-cleanup.md)
- [Binding Layer Policy](binding-layer-policy.md)
- [Rust Crate Boundaries](rust-crate-boundaries.md)
- [Configuration Frontend](configuration-frontend.md)
- [Native I/O](native-io.md)
- [Compute Kernels](compute-kernels.md)
