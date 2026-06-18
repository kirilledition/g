# Rust Host-Module Transition Scratchpad

## Thread Summary

- Worktree: `/mnt/beegfs/kirill/Projects/g-worktrees/rust-host-module-transition`.
- Branch: `feature/rust-host-module-transition`, based on local `main` at
  `2ceb3b1c`.
- Committed follow-ups already on the branch:
  - `e26a84bd runtime: move host policy helpers to Rust`
  - `6d9af58f runtime: move timing recorder state to Rust`
- Current checkpoint: moved deterministic run artifact payloads and
  run-manifest command/runtime payload shaping from Python into Rust/PyO3,
  while keeping Python-owned TOML writes, manifest I/O, telemetry side effects,
  and the public `RunArtifacts` adapter stable.
- Current checkpoint also moved trusted BGEN validation cache
  fingerprinting, cache path naming, and cache payload construction into
  Rust/PyO3 while keeping validation policy, engine calls, cache existence
  checks, and atomic cache writes in Python.
- Verification for the current checkpoint passed: focused pytest,
  `just check-core-stub`, `cargo test --workspace`, full `ty`, targeted `ruff`,
  and `just check-internal-defaults`.
- Full chr22 binary GPU performance check on `landau` did not show a meaningful
  slowdown from the current follow-up: cold finalized improved from 134.756s to
  130.790s; hot same-process no-final changed from 4.043s to 4.093s (+50ms,
  +1.24%), with metadata-adjacent stages flat or faster and output/count
  correctness unchanged.
- Trusted-validation follow-up performance check did not show an app-level hot
  regression: the moved helper microbenchmark improved fingerprint construction
  from 480.375us to 375.881us per call and cache payload construction from
  424.095us to 219.359us per call; full chr22 GPU hot no-final changed from
  4.093140s to 4.070816s (-22ms, -0.55%). Cold finalized was slower in the
  single app run, 130.790133s to 138.862896s (+8.073s, +6.17%), but the delta
  was in JAX/native delivery and queue stages rather than the trusted-validation
  startup path.
- Current state before this checkpoint commit: code changes were intentionally
  left uncommitted until requested.

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
- Run metadata follow-up moved deterministic run artifact payload construction
  and run-manifest command/runtime extension payload shaping into Rust/PyO3
  while keeping Python-owned TOML writes, manifest reads/writes, telemetry
  side effects, and the public `RunArtifacts` dataclass adapter.
- Trusted-validation follow-up moved stable BGEN validation fingerprint hashing,
  validation cache filename construction, and validation cache payload shaping
  into Rust/PyO3 while preserving Python-owned validation policy and filesystem
  side effects.

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

## Run Metadata Follow-Up Verification Run

- `uv run pytest tests/test_api.py::test_extend_run_manifest_adds_command_metadata -q` passed: 1 test.
- `just check-core-stub` passed.
- `uv run pytest tests/test_api.py tests/test_regenie2_pipeline.py tests/test_telemetry.py -q` passed: 150 tests.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed: 122 tests.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run --no-sync ruff check src/g/runner/metadata.py src/g/_core.pyi` passed.
- `just check-internal-defaults` passed.

## Run Metadata Performance Check

- Compared baseline `6d9af58f` against the run-metadata rewrite with
  `benchmark_regenie2_binary_hot` on `landau`, full chr22 binary GPU, default
  variant-major case, cold finalized plus hot no-final modes.
- Summaries:
  - Baseline: `/mnt/beegfs/kirill/Projects/g/data/benchmarks/rust-host-module-transition/metadata-payload-20260616T080709Z/baseline/regenie2_binary_hot_summary.json`
  - Current: `/mnt/beegfs/kirill/Projects/g/data/benchmarks/rust-host-module-transition/metadata-payload-20260616T080709Z/current/regenie2_binary_hot_summary.json`
- Headline timings:
  - Cold finalized: baseline 134.756314s, current 130.790133s, delta -3.966181s (-2.94%).
  - Hot same-process no-final: baseline 4.042954s, current 4.093140s, delta +0.050186s (+1.24%).
- Output/count checks matched: 418,943 output rows, 26 chunks, 17,938 score-test/Firth candidates, and identical correction outcome counts.
- Metadata-adjacent hot stages did not regress: output run preparation -0.000007s,
  output writer preparation -0.000856s, single-trait output write -0.001273s,
  writer finish/finalization -0.017574s. The observed +0.050s hot headline
  delta came from compute/queue noise, not the moved metadata payload path.

## Trusted Validation Follow-Up Verification Run

- `uv run pytest tests/test_regenie2_pipeline.py::test_build_bgen_run_engine_caches_trusted_validation -q` passed: 1 test.
- Native fingerprint parity check matched the old Python `json.dumps(..., sort_keys=True, separators=(",", ":"))` SHA-256 recipe on full chr22 BGEN.
- `uv run pytest tests/test_regenie2_pipeline.py::test_build_bgen_run_engine_rejects_assumed_trusted_validation tests/test_regenie2_pipeline.py::test_build_bgen_run_engine_caches_trusted_validation tests/test_regenie2_pipeline.py::test_build_bgen_run_engine_force_validates_trusted_bgen -q` passed: 3 tests.
- `just check-core-stub` passed.
- `uv run pytest tests/test_regenie2_pipeline.py tests/test_api.py -q` passed: 135 tests.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed: 122 tests.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run --no-sync ruff check src/g/engine/trusted_validation.py src/g/_core.pyi` passed.
- `just check-internal-defaults` passed.

## Trusted Validation Follow-Up Performance Check

- Focused helper microbenchmark on full chr22 BGEN path:
  - Old Python fingerprint recipe: median 480.375us per call.
  - Native PyO3 fingerprint helper: median 375.881us per call, delta -104.495us (-21.75%).
  - Old Python cache payload recipe: median 424.095us per call.
  - Native PyO3 cache payload helper: median 219.359us per call, delta -204.736us (-48.28%).
- Full chr22 binary GPU comparison used previous metadata-only summary as
  baseline and current trusted-validation rewrite summary as the new run:
  - Baseline: `/mnt/beegfs/kirill/Projects/g/data/benchmarks/rust-host-module-transition/metadata-payload-20260616T080709Z/current/regenie2_binary_hot_summary.json`
  - Current: `/mnt/beegfs/kirill/Projects/g/data/benchmarks/rust-host-module-transition/trusted-validation-20260618T051455Z/current/regenie2_binary_hot_summary.json`
- Headline timings:
  - Cold finalized: baseline 130.790133s, current 138.862896s, delta +8.072763s (+6.17%).
  - Hot same-process no-final: baseline 4.093140s, current 4.070816s, delta -0.022324s (-0.55%).
  - Current run also measured hot same-process finalized: 3.988363s.
- Startup/open stages around the moved path did not explain the cold delta:
  - Cold `bgen_engine_open_index_setup`: +0.006766s.
  - Cold `preflight_validation`: +0.003384s.
  - Hot `bgen_engine_open_index_setup`: -0.002337s.
  - Hot `preflight_validation`: +0.000609s.
- The cold app-level slowdown came from unrelated noisy stages: cold `jax_compute`
  +6.846s, `native_engine_delivery` +4.093s, `callback_drain` +3.483s, and
  `result_queue_consumer_wait` +7.516s.
- Output/count checks matched on common modes: 418,943 rows, 418,943 non-null
  INFO values, and identical finalized Parquet size of 20,648,112 bytes.
