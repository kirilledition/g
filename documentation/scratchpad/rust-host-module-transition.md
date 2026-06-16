# Rust Host-Module Transition Scratchpad

## Goal

Move low-risk host-side Python helpers behind Rust/PyO3 exports while preserving
the current Python public API and test behavior.

## Candidate Order

1. Config normalization.
2. Output manifest/header construction.
3. Sample/group fingerprints.
4. Run events and telemetry shaping.

## Verification

- Config: `uv run pytest tests/test_interface.py tests/test_api.py tests/test_regenie_binary_correction_contract.py -q`
- Output: `uv run pytest tests/test_io_output.py tests/test_regenie2_pipeline.py -q`
- Groups: `uv run pytest tests/test_regenie2_pipeline.py tests/test_tabular.py -q`
- Telemetry: `uv run pytest tests/test_telemetry.py tests/test_cli_bridge.py tests/test_api.py tests/test_callback_lifecycle.py -q`
- Shared native checks: `just check-core-stub`, `cargo test --workspace`
- Final: `uv run --no-sync ty check src tests scripts tooling`
- Final: targeted `ruff check` on changed Python files.

## Notes

- Base: local `main` at `2ceb3b1c`.
- Keep Python APIs stable.
- Keep JAX kernels, callbacks, and GPU backend behavior Python-owned.
- User-facing documentation changes are only needed if behavior changes.

## Results

- Config normalization now runs through native Rust/PyO3 from `RegenieConfig.from_options`.
- Current-run manifest header construction now uses native JSON when `_core` is available.
- Sample/group fingerprints and compute-group resolution now use native helpers for real native aligned-data objects, with Python fallback for tests and adapters using fake objects.
- Run-event rendering and telemetry payload shaping now use native helpers while keeping Python dataclass APIs.
- Host-policy follow-up moved backend planning, binary correction normalization,
  config-time phenotype compute-group planning and IDs, phenotype output slug
  construction, association-mode resolution, and pure JAX runtime setup payload
  resolution into Rust/PyO3 while preserving Python dataclass and enum APIs.
- Timing recorder follow-up moved mutable stage timing state and aggregate
  queue/transfer/diagnostic bookkeeping into Rust/PyO3 while preserving the
  Python `StageTimingSnapshot` dataclasses and JSON writer payloads.

## Verification Run

- `uv run pytest tests/test_interface.py tests/test_api.py tests/test_regenie_binary_correction_contract.py -q` passed: 154 tests.
- `uv run pytest tests/test_io_output.py tests/test_regenie2_pipeline.py -q` passed: 205 tests.
- `uv run pytest tests/test_regenie2_pipeline.py tests/test_tabular.py -q` passed: 110 tests.
- `uv run pytest tests/test_telemetry.py tests/test_cli_bridge.py tests/test_api.py tests/test_callback_lifecycle.py -q` passed: 74 tests.
- `just check-core-stub` passed.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run --no-sync ruff check src/g/interface/config.py src/g/io/output.py src/g/engine/native_dispatch/groups.py src/g/engine/run_events.py src/g/engine/telemetry.py src/g/_core.pyi` passed.

## Host-Policy Follow-Up Verification Run

- `uv run pytest tests/test_backend_planner.py tests/test_regenie_binary_correction_contract.py tests/test_jax_runtime.py -q` passed: 31 tests.
- `uv run pytest tests/test_regenie2_pipeline.py tests/test_api.py -q` passed: 135 tests.
- `just check-core-stub` passed.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run ruff check src/g/engine/backend_planner.py src/g/execution_plan.py src/g/jax_runtime/resolution.py` passed.

## Timing Recorder Follow-Up Verification Run

- `uv run pytest tests/test_timing.py -q` passed: 10 tests.
- `uv run pytest tests/test_timing.py tests/test_callback_lifecycle.py tests/test_regenie2_pipeline.py -q` passed: 111 tests.
- `just check-core-stub` passed.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run ruff check src/g/engine/timing.py src/g/_core.pyi` passed.
