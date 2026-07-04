# Architecture Cleanup Follow-Up - 2026-07-04

## Changes

- Replaced runner-to-pipeline `**kwargs` forwarding with typed
  `PipelineCommonRequest`, `SingleTraitPipelineRequest`, and
  `MultiTraitPipelineRequest` objects.
- Kept lazy JAX/runtime import boundary in `g.runner.runtime`; wrappers now accept
  one request object instead of arbitrary keyword dictionaries.
- Replaced `NativeHostPlanningPolicy.plan_association_backend_payload()` with typed
  `NativeAssociationBackendPlan` getters.
- Added module-level native schedule functions for stateless pipeline/native-dispatch
  schedule calls; production Python no longer constructs `NativeSchedulePolicy`.
- Added coarse native callback queue/resource operation outcomes so Rust owns
  callback queue, buffer, backpressure, dispatch, and resource-release decisions
  at the Python boundary.
- Removed production access to raw native callback queue, worker-thread, scheduler,
  and signal resources; Python callback code now consumes coarse outcome objects.
- Replaced internal dict payload bridges with typed PyO3 objects for runtime policy,
  JAX setup reports/events, preflight shapes/reports, and output fingerprints.

## Boundary Metrics

Baseline from `architecture-review-2026-07-04.md`:

- PyO3 surface: 139 classes, 15 functions.
- `src/python/schedule.rs`: 50 registered classes.
- Main concern: Python saw many tiny `Native*Plan` and dict payload objects.

Current local counts after this follow-up pass:

- `src/g/_core.pyi`: 151 classes.
- `Native*Policy` classes in stub: 22.
- `Native*Plan` classes in stub: 47.
- `payload` mentions in stub: 26.
- `runner/runtime.py` `**kwargs` wrappers: 0.
- `regenie2_pipeline/backend.py` backend payload dict reads: 0.
- `regenie2_pipeline/schedule.py` `NativeSchedulePolicy` constructions: 0.
- `src/g` production `NativeSchedulePolicy()` / `native_schedule_policy()`
  constructions: 0.
- `engine/callbacks/runtime.py` local dispatch/drain planning wrappers: 0.
- `src/g` production references to raw callback scheduler, queue, worker-thread,
  and signal internals outside `_core.pyi`: 0.
- `runner/runtime.py`, `g.jax_runtime`, and `regenie2_pipeline/preflight.py`
  internal payload adapters: 0.
- Remaining payload mentions in `g.io.output`: 23 matches, all manifest JSON or
  existing-run manifest normalization paths.

## Still Open

- Telemetry, shutdown, run-event, callback-summary, config, and manifest JSON
  payload bridges remain dict-based by design or by follow-up scope.
- `NativeSchedulePolicy` remains exported for compatibility and tests, but no
  production Python path constructs it.
- Legacy fine-grained callback runtime PyO3 methods remain exported for now, but
  production callback paths no longer call them.
- `g-runtime` event surface remains broad.

## Validation

- `cargo fmt --check`: passed.
- `cargo check -j 30 --workspace --all-targets`: passed.
- `uv run --no-sync ruff check src/g/engine/callbacks/runtime.py
  src/g/engine/callbacks/writers.py src/g/engine/native_dispatch/delivery.py
  src/g/engine/native_dispatch/writers.py src/g/_core.pyi`: passed.
- `uv run --no-sync ty check src`: passed.
- `just check-core-stub`: passed.
- `XDG_RUNTIME_DIR=/tmp TMPDIR=/tmp just dev-install-gpu-dependencies`: passed.
- `XDG_RUNTIME_DIR=/tmp TMPDIR=/tmp just slurm-gpu-just matrix-chr22-smoke`:
  passed 6/6 runs.

Smoke output: `data/benchmarks/regenie2_chr22_matrix_smoke_20260704T120242Z`.

| Run | Status | Rows |
| --- | --- | ---: |
| binary_cpu | success | 1000 |
| binary_gpu | success | 1000 |
| binary_gpu_cached | success | 1000 |
| linear_cpu | success | 1000 |
| linear_gpu | success | 1000 |
| linear_gpu_cached | success | 1000 |
