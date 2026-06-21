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
- Current uncommitted follow-up: moved output manifest file content hashing,
  file fingerprint payload/mapping construction, and execution-plan SHA-256
  text hashing into Rust/PyO3 while keeping Python dataclass adapters and
  execution-plan normalization stable.
- Additional uncommitted follow-up: moved telemetry timestamp formatting,
  telemetry output-root/path resolution, stream-file conflict policy, path
  equivalence checks, and empty writer-counter payload shaping into Rust/PyO3
  while keeping Python `TelemetryPaths` and `TelemetrySession` ownership stable.
  Timestamp formatting uses a direct minimal-feature `chrono` dependency rather
  than hand-written UTC calendar conversion.
- Current uncommitted follow-up: moved binary correction summary counter state
  and summary payload construction from the Python callback runtime into
  Rust/PyO3 while keeping JAX diagnostic counting, Python callback workers,
  and telemetry emission in Python.
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
- Manifest fingerprint follow-up moved output file content SHA-256, manifest
  file-fingerprint payload/mapping construction, and execution-plan JSON text
  SHA-256 into Rust/PyO3 while preserving Python public helpers and dataclasses.
- Telemetry path-policy follow-up moved deterministic telemetry timestamp,
  path resolution, stream-file conflict checks, path equivalence checks, and
  empty writer-counter payload construction into Rust/PyO3 while preserving
  Python telemetry session and dataclass APIs. The timestamp renderer delegates
  UTC/RFC3339 formatting to `chrono`.
- Callback summary follow-up moved binary correction summary accumulation and
  summary payload construction into a native `NativeBinaryCorrectionSummary`
  object while preserving Python-owned JAX diagnostics and telemetry emission.

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

## Manifest Fingerprint Follow-Up Verification Run

- `uv run pytest tests/test_io_output.py::test_current_run_manifest_hashes_small_control_files tests/test_io_output.py::test_bgen_content_change_with_preserved_metadata_keeps_metadata_only_fingerprint tests/test_io_output.py::test_output_manifest_helpers_cover_empty_paths_and_invalid_json tests/test_io_output.py::test_initialize_output_run_rejects_execution_plan_hash_mismatch tests/test_io_output.py::test_initialize_output_run_rejects_execution_plan_hash_only_mismatch -q` passed: 24 tests.
- `just check-core-stub` passed.
- `uv run pytest tests/test_io_output.py tests/test_regenie2_pipeline.py -q` passed: 205 tests.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed: 122 tests.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run --no-sync ruff check src/g/io/output.py src/g/_core.pyi` passed.
- `just check-internal-defaults` passed.

## Manifest Fingerprint Follow-Up Performance Check

- Focused helper microbench on full chr22 inputs:
  - BGEN metadata-only fingerprint: old Python 446.503us, native 312.176us,
    delta -134.328us (-30.08%).
  - Sample content fingerprint: old Python 2746.854us, native 2268.845us,
    delta -478.010us (-17.40%).
  - Sample standalone content SHA-256: old Python 2002.093us, native 2036.509us,
    delta +34.416us (+1.72%).
  - Fingerprint mapping only: old Python 0.224us, native 1.087us, delta
    +0.863us; PyO3 overhead dominates this tiny dict construction.
  - Execution-plan JSON SHA-256 only: old Python 4.999us, native 12.215us,
    delta +7.216us; PyO3 overhead dominates this tiny hash.
  - Public execution-plan hash helper: old Python 78.363us, native 82.917us,
    delta +4.554us (+5.81%).
- Production-style current-run manifest wrapper on chr22 control files:
  - `build_current_run_manifest_header`: old Python 11073.813us, native-backed
    9923.092us, delta -1150.720us (-10.39%).
  - Interleaved `build_current_run_manifest_header` plus
    `current_run_manifest_header_to_mapping`: old Python 21261.386us,
    native-backed 20587.033us, delta -674.353us (-3.17%).
- Standalone full 118 MB BGEN content SHA-256 was slower with native `sha2`:
  old Python/OpenSSL 646.055ms, native 1240.668ms, delta +594.613ms (+92.04%).
  This is not the production chr22 current-manifest path because BGEN manifest
  identity is metadata-only; content hashes are used for the small control files.

## Telemetry Path-Policy Follow-Up Verification Run

- `uv run pytest tests/test_telemetry.py::test_resolve_telemetry_paths_defaults_to_output_run_logs tests/test_telemetry.py::test_trace_telemetry_paths_default_profile_summary_without_exact_stage_timings tests/test_telemetry.py::test_explicit_stage_timings_path_enables_exact_stage_output tests/test_telemetry.py::test_telemetry_stream_uses_log_file_or_trace_file_alias tests/test_telemetry.py::test_telemetry_stream_rejects_different_log_and_trace_files tests/test_telemetry.py::test_log_file_replaces_default_telemetry_stream -q` passed: 6 tests.
- `uv run pytest tests/test_telemetry.py tests/test_cli_bridge.py tests/test_api.py tests/test_callback_lifecycle.py -q` passed: 74 tests.
- `just check-core-stub` passed.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed: 122 tests.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run --no-sync ruff check src/g/engine/telemetry.py src/g/_core.pyi` passed.
- `just check-internal-defaults` passed.
- After replacing manual UTC timestamp logic with `chrono`, the follow-up
  re-ran focused telemetry tests, full telemetry-related pytest, `just
  check-core-stub`, `cargo test --workspace`, `ty`, targeted `ruff`, timestamp
  smoke checks, and `git diff --check`; all passed.

## Telemetry Path-Policy Follow-Up Performance Check

- Focused public-helper microbench compared the old Python implementation to
  the current Rust/PyO3-backed Python helpers in the same process.
- Results:
  - `format_timestamp`: old Python 1.935us, native-backed 0.336us, delta
    -1.599us (-82.63%).
  - `resolve_output_run_root`: old Python 1.670us, native-backed 1.648us,
    delta -0.022us (-1.30%).
  - `resolve_telemetry_paths` default trace case: old Python 6.269us,
    native-backed 6.461us, delta +0.192us (+3.06%).
  - `resolve_telemetry_stream_file` default case: old Python 1.051us,
    native-backed 1.636us, delta +0.585us (+55.67%).
  - `paths_refer_to_same_file`: old Python 1822.756us, native-backed
    1783.295us, delta -39.461us (-2.16%).
  - `build_empty_writer_counters`: old Python 0.310us, native-backed 1.257us,
    delta +0.947us (+305.89%).
- Interpretation: this slice is performance-neutral at application scale.
  Timestamp formatting and path equivalence improved, but PyO3/dict adapter
  overhead makes the smallest helper calls slightly slower. These helpers run
  at startup/teardown/path-policy frequency, not per variant or per result row.

## Callback Summary Follow-Up Verification Run

- Moved callback binary correction summary counters and summary payload shaping
  to Rust/PyO3. Python still decides when diagnostics are materialized, still
  counts diagnostics through JAX, and still emits telemetry events.
- `uv run pytest tests/test_callback_lifecycle.py::test_native_callback_runner_emits_binary_correction_summary tests/test_callback_lifecycle.py::test_binary_correction_summary_skips_materialization_without_telemetry tests/test_regenie2_pipeline.py::test_binary_result_worker_records_deferred_diagnostics_from_work_item -q` passed: 3 tests.
- `uv run pytest tests/test_callback_lifecycle.py tests/test_regenie2_pipeline.py tests/test_regenie2_binary_diagnostics.py -q` passed: 109 tests.
- `just check-core-stub` passed.
- `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed: 122 tests.
- `uv run --no-sync ty check src tests scripts tooling` passed.
- `uv run --no-sync ruff check src/g/engine/callbacks/runtime.py src/g/_core.pyi` passed.
- `just check-internal-defaults` passed.
- Follow-up deferred materialization optimization added a pending diagnostics
  buffer, batches JAX diagnostics materialization at result-worker drain or
  summary emit, and uses native `add_diagnostics_totals` to cross PyO3 once for
  the aggregate counters.
- After that follow-up:
  - `uv run pytest tests/test_callback_lifecycle.py::test_native_callback_runner_emits_binary_correction_summary tests/test_callback_lifecycle.py::test_binary_correction_summary_skips_materialization_without_telemetry tests/test_regenie2_pipeline.py::test_binary_result_worker_records_deferred_diagnostics_from_work_item -q` passed: 3 tests.
  - `uv run pytest tests/test_callback_lifecycle.py tests/test_regenie2_pipeline.py tests/test_regenie2_binary_diagnostics.py -q` passed: 109 tests.
  - `uv run --no-sync ruff check src/g/compute/regenie2_binary/diagnostics.py src/g/compute/regenie2_binary/api.py src/g/engine/callbacks/runtime.py src/g/engine/callbacks/binary.py tests/test_callback_lifecycle.py tests/test_regenie2_pipeline.py` passed.
  - `uv run --no-sync ruff format --check src/g/compute/regenie2_binary/diagnostics.py src/g/compute/regenie2_binary/api.py src/g/engine/callbacks/runtime.py src/g/engine/callbacks/binary.py tests/test_callback_lifecycle.py tests/test_regenie2_pipeline.py` passed.
  - `just check-core-stub` passed.
  - `LD_LIBRARY_PATH=/home/kirill/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/lib cargo test --workspace` passed: 122 tests.
  - `uv run --no-sync ty check src tests scripts tooling` passed.
  - `just check-internal-defaults` passed.

## Callback Summary Follow-Up Performance Check

- Focused public-path microbench compared the old Python counter/dataclass
  logic against the optimized Rust/PyO3-backed `NativeBinaryCorrectionSummary`.
  The native path now passes typed counters directly and returns the native
  summary payload without an extra Python `dict(...)` copy.
- Results:
  - One chunk, prebuilt diagnostics mapping: old Python 3.002us,
    native-backed 3.262us, delta +0.261us (+8.68%).
  - Twenty-six chunks, prebuilt diagnostics mapping: old Python 35.162us,
    native-backed 26.757us, delta -8.405us (-23.90%).
  - Twenty-six chunks including the shared JAX `device_get` diagnostics
    materialization: old Python 4218.427us, native-backed 4191.104us, delta
    -27.323us (-0.65%).
- Interpretation: the optimized native path is still slightly slower for a
  single chunk, but it is faster for the realistic chr22-style 26-chunk summary
  path. Including the shared JAX diagnostics materialization, the measured path
  improved by 0.027ms.
- Deferred aggregate follow-up microbench:
  - Old per-chunk mapping plus native count add for twenty-six chunks:
    4511.691us.
  - New deferred aggregate materialization plus one native totals add for
    twenty-six chunks: 4212.903us.
  - Delta: -298.788us (-6.62%) for this callback-summary slice.
- Interpretation after the follow-up: this removes per-chunk summary
  synchronization from the result processing path and makes the chr22-style
  summary path faster in the CPU/JAX scalar microbench. The app-level effect is
  still expected to be small because this path runs once per result chunk, not
  per variant.
