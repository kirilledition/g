# Binding Layer Policy

| Status | Applies to | Owner |
| --- | --- | --- |
| Active contract | Native `_core` code under `src/binding` | Development maintainers |

## Purpose

The root Rust extension is the native host for the Python/JAX boundary. Domain
logic belongs in workspace crates. Boundary code owns only the coordination
that necessarily touches opaque Python handles, NumPy objects, `PyErr`, or the
Python CLI return object.

```text
crates/*    domain contracts, algorithms, scheduling, I/O, runtime, output
src/binding native host coordination and PyO3/NumPy adaptation
src/g       console bootstrap, JAX backend, JAX kernels
```

## Allowed

- The high-level native CLI entrypoint and terminal result.
- Lazy construction of the Python JAX backend after native validation/setup.
- One call into the coarse `g-engine::execute_coordinated_run` coordinator.
- The four backend method invocations and typed NumPy exchange classes.
- Retention of opaque Python JAX group, chromosome, and device-result handles.
- Interruption checks and terminal adaptation that must retain a concrete
  `PyErr`.
- Supplying the current Python thread name to the runtime telemetry session.
- Checked conversion between Python/NumPy values and crate-owned types.

## Forbidden

- A second scheduler, queue protocol, or callback worker hierarchy.
- Python-owned BGEN delivery, input alignment, output, resume, or cleanup.
- Binding-owned telemetry state, serialization, writer, counter, or close
  lifecycle.
- JSON or dictionaries between Rust domain crates.
- Public wrappers for crate APIs that production Python does not consume.
- Root aliases, migration adapters, deprecated names, or test/tooling exports.
- Per-variant Python calls.
- Direct dependencies on `g-genotype`, `g-input`, or `g-output`; those services
  are reached through `g-engine`.

Preparation, execution, output completion, and run-artifact construction are
coordinated in `g-engine` and are Python-free. The root host supplies one
`AssociationBackend` implementation that calls:

```text
prepare_group
prepare_chromosome
compute_batch
materialize_batch
```

## Namespace Policy

The complete production namespace is:

```text
g._core.cli       run, NativeCliRunResult
g._core.engine    JAX backend config and typed exchange classes
```

Every registered item must appear in `src/g/_core.pyi`, and every stub item must
be registered. Unregistered Rust structs are implementation details, not a
compatibility surface.

## Placement Test

Move code to a domain crate whenever it can use crate-owned Rust types and
errors. Opaque Python state and `PyErr` are generic backend/error parameters,
not reasons to keep BGEN, output, buffer, numeric, or scheduling policy in the
binding. The same rule applies to telemetry lifecycle: only Python thread-name
lookup belongs here. Prefer deletion or a direct re-export over a forwarding
adapter.
