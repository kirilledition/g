# Public API

## This crate owns

Process runtime infrastructure: logging sinks, telemetry session/writer
lifecycle, stage timing, SIGTERM state, Rayon configuration, and compatibility
checks for native process-global settings.

## Public types

The resolved `NativeRunSessionPolicy`, opaque `NativeRunSession`,
`ProcessRuntimeState`, `TelemetryRunSession`, `StageTimingRecorder`, and errors
that appear in their public signatures. The session policy is the single
logging/telemetry policy representation shared by compatibility checks,
subscriber setup, and run-owned writer construction. Runtime has no dependency
on application planning types.

## Public functions

Open and finish the native run session from resolved paths and sink flags,
atomically validate and record logging compatibility around writer/subscriber
initialization, enforce/configure native process state, emit generic structured
diagnostics, observe SIGTERM, and serialize/flush telemetry through the
runtime-owned session. Event meaning, telemetry-mode interpretation, path
defaults, and terminal rendering stay with the runner or engine that owns them.

The installed subscriber is process-global; stderr, plain-file, and structured
telemetry workers are run-owned and reached through private dynamic routes so
every run can drain its queues without leaving the subscriber bound to a dead
writer. `NativeRunSession` checks compatibility before creating resources and
records the installed topology only after successful subscriber initialization;
the runner holds its process-state lock across that constructor.

## This crate must not expose

Phenotype/sample domain logic, BGEN reader internals, output writer internals,
engine scheduler state, JAX policy, terminal output, trusted-BGEN cache policy,
PyO3 classes, `g-plan` types, output-layout defaults, callback-era execution
APIs, event-specific payload DTOs, or diagnostic event-name/message constants.

## Performance constraints

Disabled telemetry/timing creates no run ID or shared session allocation.
Enabled session clones share state, and event envelopes borrow stable string
fields through immediate serialization. Stage timing owns one key and aggregate
per stage; borrowed serialization views preserve the separate totals/counts
JSON contract without duplicate maps. Do not add data-matrix copies, genotype
parsing, or JAX host/device transfers here.

## Allowed downstream users

`g-engine` and `g-runner` only.
