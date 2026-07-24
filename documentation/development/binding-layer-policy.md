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
- The five backend lifecycle stages with direct typed NumPy arguments,
  including exactly one of the decoded or compressed transfer methods.
- Retention of opaque Python JAX group, chromosome, and device-result handles.
- Python signal checks, JAX configuration/device observation, and conversion
  of a concrete `PyErr` into runner-host callbacks.
- GPU-only loading of the official nvCOMP Python wheel and private typed-XLA
  FFI target registration around the capability-checked crate handler.
- GPU binary-Firth registration of the private `g-compute-cuda` typed-XLA FFI
  target after its independent driver and device capability check.
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
- Calls into `g-genotype`, `g-input`, or `g-output` services. Type-only
  dependencies on their canonical `AssociationBackend` payloads are allowed
  for NumPy conversion; the binding must not orchestrate those crates or
  redefine or re-export their types.

CLI dispatch, process policy, timing, terminal rendering, and coordinated
engine execution are owned by `g-runner` and are Python-free. The root host
supplies one `AssociationBackend` implementation that calls:

```text
prepare_group
prepare_chromosome
transfer_batch | transfer_compressed_batch
compute_batch
materialize_batch
```

Backend construction receives the canonical `g-plan::Device` separately from
the mode-specific kernel plan. CPU and host-delivered GPU runs never import or
initialize nvCOMP. The first compressed packed8 group registers the
process-global private target once.

Only GPU binary-Firth backend construction probes the optional CUDA component
target. The binding passes the resulting static capability into the typed JAX
configuration; it does not expose an environment or user configuration knob.
Only the six engine-owned recoverable capability classes select JAX. CUDA
driver-operation and unknown native failures are fatal, as are capsule
construction, JAX import, and FFI registration failures. A final selection is
cached per exact JAX device and cannot change in flight. This policy does not
affect packed8 target registration.

Any process-global lock or once cell reached while Python is attached must use
PyO3's attached-thread synchronization helpers. A Python-running initializer
uses `OnceLockExt::get_or_init_py_attached`, and short map access uses
`MutexExt::lock_py_attached`; no ordinary Rust guard may be held while importing
Python modules, loading nvCOMP, probing CUDA, or registering an FFI target.

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
lookup belongs here. Prefer deletion or a direct owner-type import over a
forwarding adapter.
