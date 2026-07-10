# Public API

## This crate owns

Process runtime policy, logging sinks, owned telemetry session/writer lifecycle,
typed event helpers, JAX runtime setup policy, shutdown, timing, and
trusted-validation cache policy.

## Public types

Typed runtime compatibility and subsystem errors, runtime state/policy DTOs,
`TelemetryRunSession`, telemetry path errors, logging config, JAX setup reports,
shutdown signal/session/install/restore plans and errors, the stage timing recorder, and active typed lifecycle/diagnostic payloads.
Telemetry mode is re-exported from canonical `g-plan`.

## Public functions

Build/apply runtime policy plans, initialize/shutdown logging, build telemetry and diagnostic payloads, configure Rayon policy,
resolve runtime paths, write timing artifacts, plan trusted-validation cache
access, and serialize active run diagnostics. Telemetry events are
serialized and flushed by the runtime-owned session.

## This crate must not expose

Phenotype/sample domain logic, BGEN reader internals, output writer internals,
engine scheduler state, PyO3 classes, public debug/event/trusted-validation
submodules, callback-era execution APIs, duplicate event facades, or diagnostic
event-name/message constants as public API.

## Performance constraints

Keep event construction outside inner loops. Do not add data-matrix copies, genotype parsing, or JAX host/device transfers here.

## Allowed downstream users

`g-engine` and root PyO3 facade.
