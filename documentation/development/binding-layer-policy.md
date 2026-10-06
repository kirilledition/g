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

## Private Module Ownership

The private `engine` adapter implements the backend lifecycle. Its
`ffi_registration` module owns runtime-version checks and process-lifetime
CUDA target registration; `array_conversion` owns checked NumPy conversion
and immutable native-backed array owners. These modules do not register a
Python namespace or add wrappers around domain-crate services.

## Allowed

- The registered CLI entrypoint and its typed terminal result.
- Lazy construction of the Python JAX backend after `g-runner` completed
  native validation and runtime setup.
- Coarse backend lifecycle stages with direct typed NumPy arguments,
  including decoded/compressed transfer or shared-source preparation and selection.
- Retention of opaque Python JAX group, chromosome, shared-source, and
  device-result handles.
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
transfer_batch | transfer_compressed_batch | (prepare_shared_source, select_shared_source)
compute_batch
materialize_batch
```

Backend construction receives the canonical `g-plan::Device` separately from
the mode-specific kernel plan. CPU and host-delivered GPU runs never import or
initialize nvCOMP. The first compressed packed8 group registers the
process-global private target once.

Compressed GPU linear backends also advertise immutable source sharing. The
engine owns eligibility, memory admission, two-group scheduling, resume masks,
and all cleanup ordering. The binding only adapts one owned compressed payload
to immutable NumPy views, retains the Python source handle, and converts each
group's selection to an ordinary transferred batch. Group-specific statistics
are private because compute kernels may donate them. Source release attaches
Python after the engine has drained every consumer; it does not introduce a
second scheduler or Python-owned input lifetime policy.

Only GPU binary-Firth backend construction probes the optional CUDA component
target. The binding passes the resulting static capability into the typed JAX
configuration; it does not expose an environment or user configuration knob.
Capability or registration failure selects the JAX implementation and does
not affect packed8 target registration.

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

## Materialized Result Ownership

Materialized association columns are copied once from NumPy into Rust-owned
vectors before the asynchronous output pipeline accepts them. Output then
moves those vectors into Arrow buffers without copying the statistic values
again. A NumPy read-only flag does not establish allocation ownership: with
the supported JAX 0.11.0 CPU runtime, `device_get` can return a NumPy view whose
base is a `PyCapsule`, whose `owndata` flag is false, and whose allocation is
shared with the JAX array and later calls to `device_get`.

Keeping such a Python reference alive cannot transfer the allocation to a
Rust vector or establish exclusive storage ownership. Any future removal of
this copy must specify the allocation owner, asynchronous writer lifetime,
and interaction with JAX buffer donation; it must also preserve trait-major
ordering for strided and Fortran-contiguous NumPy arrays. Conversion uses
the row-major contiguous fast path and logical iteration for other layouts.
Current conversion tests check independent result ownership after the source
storage changes, along with row-major, Fortran-contiguous, and strided ordering.

An allocation-and-copy microbenchmark on Leibniz on 2026-10-06 covered 26 full
chunks of 16,384 variants, or 425,984 variant rows. Each chunk copied four
float32 association columns and uint8 correction codes with shape
`(trait_count, 16384)`, plus two uint64 and one uint32 packed8 statistic columns
with shape `(16384,)`. The raw genotype statistics are copied once per variant,
independent of trait count. All copied columns remain owned simultaneously until the end of each chunk.
Median times over 21 runs were 8.31 ms for one trait (15.76 MB), 83.18 ms for
16 traits (124.39 MB), and 250.33 ms for 64 traits
(471.99 MB). These measure the isolated ownership copy, not an end-to-end
application speedup; the probe uses a full final chunk and includes correction
codes even when an analysis would omit them. Larger trait counts make allocation
and buffer reuse worth investigating, but these measurements do not establish
an end-to-end benefit from sharing JAX allocations. The explicit ownership
boundary remains until a safe shared-buffer contract is qualified.
