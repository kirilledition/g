# Public API

## This crate owns

Process runtime policy, logging sinks, telemetry/session/event helpers, JAX runtime setup policy, shutdown, timing, and trusted-validation cache policy.

## Public types

`RuntimeError`/`RuntimeResult`, runtime state/policy/session DTOs, telemetry mode, envelopes/plans/writers/path errors, logging config, JAX setup reports,
shutdown signal/session/install/restore plans and errors, timing recorders, trusted-validation cache DTOs, and typed telemetry/diagnostic event payloads and kinds.

## Public functions

Build/apply runtime policy plans, initialize/shutdown logging, build telemetry and diagnostic payloads, configure Rayon policy, resolve runtime paths, and write timing/validation artifacts.

## This crate must not expose

Phenotype/sample domain logic, BGEN reader internals, output writer internals, engine scheduler state, PyO3 classes, or diagnostic event-name/message constants as public API.

## Performance constraints

Keep event construction outside inner loops. Do not add data-matrix copies, genotype parsing, or JAX host/device transfers here.

## Allowed downstream users

`g-engine` and root PyO3 facade.
