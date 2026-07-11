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
g-runner    CLI/run lifecycle coordination across domain crates
src/binding PyO3/NumPy adaptation and Python host callbacks
src/g       console bootstrap, JAX backend, JAX kernels
```

## Allowed

- The registered CLI entrypoint and its typed terminal result.
- Lazy construction of the Python JAX backend after `g-runner` completed
  native validation and runtime setup.
- The four backend method invocations with direct typed NumPy arguments.
- Retention of opaque Python JAX group, chromosome, and device-result handles.
- Python signal checks, JAX configuration/device observation, and conversion
  of a concrete `PyErr` into runner-host callbacks.
- Supplying the current Python thread name to `g-runner` for telemetry labels.
- Checked conversion between Python/NumPy values and crate-owned types.

## Forbidden

- A second scheduler, queue protocol, or callback worker hierarchy.
- Python-owned BGEN delivery, input alignment, output, resume, or cleanup.
- Binding-owned telemetry state, serialization, writer, counter, or close
  lifecycle.
- Binding-owned CLI dispatch, process-global policy checks, stage timing,
  terminal rendering, or calls that sequence `g-interface`, `g-plan`,
  `g-runtime`, and `g-engine`.
- JSON or dictionaries between Rust domain crates.
- Public wrappers for crate APIs that production Python does not consume.
- Root aliases, migration adapters, deprecated names, or test/tooling exports.
- Per-variant Python calls.
- Direct dependencies on `g-genotype`, `g-input`, or `g-output`; those services
  are reached through `g-engine`.

CLI dispatch, process policy, timing, terminal rendering, and coordinated
engine execution are owned by `g-runner` and are Python-free. The root host
supplies one `AssociationBackend` implementation that calls:

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
```

Every registered item must appear in `src/g/_core.pyi`, and every stub item
must be registered. The JAX backend bridge is private and does not create a
Python extension namespace or exchange-object compatibility surface.

## Placement Test

Move code to a domain crate whenever it can use crate-owned Rust types and
errors. Opaque Python state and `PyErr` are generic backend/error parameters,
not reasons to keep BGEN, output, buffer, numeric, or scheduling policy in the
binding. The same rule applies to telemetry lifecycle: only Python thread-name
lookup belongs here. Prefer deletion or a direct re-export over a forwarding
adapter.
