# Rust Migration

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; active migration contract | main branch as of 2026-06-30 Cargo workspace migration | Development maintainers |

This page is the durable migration contract for moving `g` from a single Rust
package embedded in a Python-owned application into a Cargo workspace where Rust
owns host orchestration and Python owns public convenience APIs plus JAX compute
kernels.

Baseline commit: `25ec7701b8b4a9339d424d8d08b1bc4527ddb2c8`.

## Direction

The end state keeps the root package named `g` and the Maturin/PyO3 library
target named `_core`. The root crate becomes a composition and adapter crate.
Internal domain crates must remain Python-free unless they are the root binding
crate.

Target crates:

| Crate | Responsibility |
| --- | --- |
| `g-plan` | Immutable requested and prepared run contracts. |
| `g-cli` | Native CLI frontend binary and eventual process-owner entrypoint. |
| `g-interface` | Clap, TOML, defaults, overlays, validation, and config-to-plan conversion. |
| `g-genotype` | BGEN mmap/index/decode, genotype chunk planning, preprocessing, and genotype benchmarks. |
| `g-input` | Sample, phenotype, covariate, prediction-list, and LOCO alignment. |
| `g-output` | Output paths, Arrow/Parquet/REGENIE writing, manifests, resume, and finalization. |
| `g-runtime` | Logging, tracing, telemetry, timing, runtime policy, Rayon policy, and shutdown. |
| `g-engine` | Application state machine, preflight, batching, queues, backend trait, and cleanup. |

No internal crate may depend on `pyo3` or `numpy`. The root `g` crate may depend
on all internal crates and remains the only native Python binding crate.
`check_rust_architecture` enforces this split and also rejects root-crate public
Rust re-exports of internal domain crates, a public root `python` adapter
module, public root PyO3 registration, and root PyO3 adapter calls back into
legacy Python telemetry fallback methods.

## Non-Negotiable Invariants

- Preserve sample order, phenotype and covariate masks, LOCO prediction
  alignment, allele orientation, output row order, correction selection,
  correction status, and fresh/resumed equivalence unless a separate
  science-change issue explicitly approves the difference.
- Keep `g --help`, `g regenie --help`, CLI exit codes, TOML merge/default
  behavior, `effective_config.toml`, the Python `g.api` surface, PyO3 export
  names, and `_core.pyi` stable unless an interface-change issue approves the
  change.
- Keep crate dependencies acyclic and phase-aware. Do not introduce generic
  dumping-ground crates such as `g-utils`, `g-common`, or `g-types`.
- Do not keep permanent production Python fallbacks for Rust-owned behavior.
  Temporary dual implementations are allowed only for equivalence tests and must
  have a removal task.

## Migration Ledger

| Area | Current Python owner | Current Rust helper | Target crate | Target owner | Removal status | Primary tests and benchmarks |
| --- | --- | --- | --- | --- | --- | --- |
| CLI/TOML/config frontend | `src/g/cli.py`, `src/g/interface/config.py` | `crates/interface/src/`, `crates/cli/src/` | `g-cli`, `g-interface` | Rust | Native frontend prototype added: Cargo binary `g` owns help, parse errors, `--config` TOML, CLI-over-TOML overrides, and run-config validation through `g-interface`; validated run execution crosses an explicit `NativeExecutionAdapter` seam whose default adapter preserves the temporary unsupported-execution error, whose boundary converts backend panics into deterministic runtime-failure outcomes, whose execution context carries SIGINT/SIGTERM shutdown requests, and whose subprocess adapter can delegate to the Python/JAX backend; the Python console script is now compatibility glue that calls the coarse native PyO3 CLI runner and uses a sentinel-protected legacy backend path to avoid recursive dispatch until the embedded Python/JAX or direct `g-engine` backend boundary is implemented | `cargo test -p g-cli`, `cargo bench -p g-cli --bench frontend`, `tests/test_interface.py`, CLI help/package smoke |
| Execution planning | `src/g/execution_plan.py`, `src/g/engine/backend_planner.py` | `crates/plan/src/`, `crates/interface/src/plan_request.rs`, native metadata helpers | `g-plan`, then `g-engine` | Rust | Started: `g-interface` compiles resolved config into `g-plan::RunRequest`; Python consumes the native run-request payload through a PyO3 value wrapper instead of parsing JSON; host planning policy for association mode, backend selection, binary correction normalization, phenotype grouping, and output directory names now enters through `NativeHostPlanningPolicy`; manifest headers now serialize through a Rust-built `g-plan::PreparedRunPlan` consumed by `g-output`, with backend validation, prepared-plan assembly, and backend-kind derivation owned by `g-plan`; Python still supplies other transitional header fields from legacy dataclasses until dynamic preparation moves to Rust | `tests/test_backend_planner.py`, `tests/test_regenie2_pipeline.py`, `tests/test_io_output.py`, `cargo test -p g-plan`, `cargo test -p g-interface`, `cargo test -p g-output` |
| BGEN and genotype preprocessing | Python native-dispatch wrappers | `crates/genotype/src/` | `g-genotype` | Rust | Extracted as leaf crate | Rust genotype tests, `just benchmark-bgen-reader`, `cargo bench -p g-genotype` |
| Sample and phenotype alignment | `src/g/io/source.py`, pipeline wrappers | `crates/input/src/sample/` | `g-input` | Rust | Extracted as native crate; Python remains adapter | `tests/test_io_sample.py`, `tests/test_tabular.py`, `cargo test -p g-input` |
| Prediction and LOCO input | `src/g/engine/native_dispatch/loaders.py` | `crates/input/src/regenie/` | `g-input` | Rust | Extracted as native crate; Python remains adapter | pipeline, parity, and LOCO alignment tests |
| Output, manifest, and resume | `src/g/io/output.py`, callbacks/writers wrappers | `crates/output/src/` | `g-output` | Rust | Extracted as native crate; Python output adapter now passes manifest/header/preparation values through PyO3 value wrappers instead of serializing those contracts itself | `tests/test_io_output.py`, `cargo bench -p g-output`, `just benchmark-output-stages-gpu` |
| Runtime, telemetry, timing, shutdown | `src/g/runner/`, `src/g/engine/telemetry.py`, `src/g/engine/timing.py`, `src/g/engine/shutdown.py` | `crates/runtime/src/`, root PyO3 adapters | `g-runtime` | Rust | Started: pure runtime policy without Python fallback, aggregate runtime policy handles, run runtime handles, telemetry path/session policy handle, telemetry run session handle, telemetry run ID generation, telemetry session cap/counter/envelope/progress-throttle/close-metadata state, telemetry file writer creation/shared-stream reuse/event-cap enforcement/counter snapshots/close-flush lifecycle/final counter metadata, logging sink subscriber setup/filter validation/guard ownership/shutdown, telemetry close planning/helper dispatch and close-metadata payloads, current telemetry event metadata and payload construction, telemetry event/progress emission gating, telemetry event/close-event/JSON-line emission, run artifact metadata attachment, execution artifact tree construction, completed/interrupted/failed run lifecycle event construction/payload/rendering policy without Python fallback, CLI run lifecycle state, run-failed duplicate-suppression planning, run-failed telemetry emission, runner lifecycle/artifact, pipeline, and callback progress/binary-summary telemetry dispatch, and telemetry close-failure planning, shutdown metadata/controller state/default signal resolution, shutdown handler lifecycle planning/session previous-handler storage, and repeated-signal exception planning without Python fallback, timing state, stage timing recorder handle, exact-stage timing policy, final timing output context, final timing, run-event, and JAX runtime diagnostic field JSON serialization, timing recorder creation/write gating, transfer metadata shape/byte expansion, stage-timing/profile JSON payloads and file writes, runtime knob configuration execution, JAX persistent-cache directory creation, config-update execution, GPU-validation execution, default NVIDIA driver probe path policy, and setup-session completion recording, trusted BGEN validation cache metadata/default path policy/cache-hit planning/unsafe-mode policy/cache-backed engine execution/JSON serialization/atomic writes, run metadata, default local runtime cache path construction, JAX runtime policy payload construction, JAX runtime setup session handle, JAX runtime setup/config-update/GPU-validation/diagnostic event payloads, diagnostic record planning and event recording, setup lifecycle planning, setup side-effect planning, setup validation completion, and NVIDIA driver visibility probing, runtime compatibility tokens guarding output preparation, Rayon global thread-pool initialization/configuration planning/failure formatting, and logging/Rayon/JAX process state plus seeded process-runtime state handles extracted; Python still owns other side effects while triggering the native JAX setup boundary | `tests/test_telemetry.py`, `tests/test_timing.py`, `tests/test_jax_runtime.py`, shutdown tests, `cargo test -p g-runtime` |
| Pipeline scheduling and queues | `src/g/engine/native_dispatch/`, `src/g/engine/callbacks/` | `crates/engine/src/` | `g-engine` | Rust | Started: crate scaffolded with explicit run phases, backend trait, deterministic fake backend, deterministic fake side effects, single-batch coordinator, native side-effect hook contract for input/preflight/output/telemetry/finalization operations with abort cleanup on initialized-output failures, fake output lifecycle/manifest state and abort counting for Rust-only coordinator tests, injected failures for every entered phase, backend trait-method failures, interruption tests, a coordinator-overhead benchmark, the BGEN-backed `Regenie2RunEngineCore`/chunk planning core, native required-chromosome resolution, native preflight report/warning/scan-count helpers and Python-facing variant-count validation, native preflight shape/count, binary class-count, and prediction-shape policy payloads, native all-manifest resume compatibility preflight and output-run initialization orchestration using `g-output`, native pipeline output preparation batch handle, native pipeline output initialization result handle, native committed-chunk intersection for multi-output resume scheduling, native callback batch-size delivery policy, native grouped-union callback batch-size policy, native callback queue-limit policy, native callback queue stage/operation observation policy and backpressure payload construction, native callback queue put/get attempt planning, native variant-major dosage batch handoff planning, native result in-flight slot accounting and acquire/release attempt planning, native dosage-buffer pool accounting and acquire/register/return/discard attempt planning, native dosage-buffer reuse shape planning, native writer-finish thread cleanup and execution planning, native BGEN delivery invocation and cleanup lifecycle action planning/execution order, native output write method planning, native effective trusted BGEN mode resolution, native callback worker lifecycle start state, start action planning, and start-attempt lifecycle marking, native callback worker shutdown timeout, stop/join-planning, finish/abort action planning, worker error raise planning, stop-loop poll policy, and failure message formatting, native callback worker stop-attempt decision policy, native callback worker backpressure poll-timeout policy, native BGEN delivery method selection, native GPU genotype-format auto/resume/validation-result policy, native null-logistic nonconvergence policy, native binary correction summary accumulation state, typed binary diagnostic-result normalization, callback-owned binary diagnostic host materialization, native callback progress/chunk identity state and progress telemetry payload planning, native array-finiteness/covariate-rank/binary-coding preflight policy, typed `NativePreflightValidator` PyO3 handle for finite-array, SVD-backed covariate-rank, binary phenotype coding/class-count, shape, variant-count, report, and prediction-shape validation, native PyO3 null-logistic convergence-array flatten/count planning, native null-logistic failure-count reuse and timing-diagnostic row construction, native multi-trait committed-chunk write selection, and native PyO3 output-writer finish/interrupted/abort lifecycle calls for real native writer sessions; production queues, test/fake writer side-effect fallbacks, telemetry side-effect writes, JAX host materialization for callback diagnostic values, other JAX numeric scans, and PyO3/JAX backend wiring remain transitional | `tests/test_callback_lifecycle.py`, callback overhead benchmarks, `tests/test_preflight.py`, `tests/test_regenie2_pipeline.py`, `cargo test -p g-engine`, `cargo bench -p g-engine --bench coordinator` |
| JAX kernels | `src/g/compute/`, `src/g/jax_runtime/` | PyO3 array adapters | Root adapter plus Python modules | Python | Kept | kernel tests, parity harness, GPU benchmarks |

Telemetry close dispatch is native-only: `telemetry.close_telemetry_session`
now closes enabled native telemetry sessions, no-ops disabled native sessions,
and no longer falls back to Python `close_with_event` objects.
JAX runtime diagnostic telemetry dispatch also resolves the native telemetry
session handle directly instead of calling Python fallback logging methods.
JAX runtime diagnostic event recording is the single root PyO3 boundary for
that path and returns the typed native record plan; standalone diagnostic
plan/log helper exports are no longer exposed.
Native CLI stdout/stderr and rendered line diagnostics now expose only native
recording functions at the root PyO3 boundary; direct payload builders are kept
inside `g-runtime` and are no longer Python exports.
Native runtime-knob diagnostics follow the same recorder-only root PyO3 export
pattern.
Runner metadata artifact-finalization diagnostics also expose only the native
recorder at the root PyO3 boundary.
Preflight warning and legacy output-run resume committed-chunk diagnostics also
expose only native recorder functions at the root PyO3 boundary.
Pipeline output lifecycle diagnostics for BGEN engine open/reuse, resume chunk
counts, and writer-session creation follow the same recorder-only root PyO3
export pattern.
GPU genotype-format resolution, callback null-logistic nonconvergence warning,
and multi-phenotype sample-summary diagnostics now also expose only native
recorder functions at the root PyO3 boundary.
Pipeline phase diagnostics for complete-case multi-trait, grouped
per-phenotype, multi-group preflight, and single-trait execution now follow the
same recorder-only root PyO3 export pattern.
Native-dispatch BGEN construction, trusted-validation, callback-drain,
delivery, pipeline-finished, and writer lifecycle diagnostics now also expose
only native recorder functions at the root PyO3 boundary.
Runner lifecycle, execution-plan, and dispatch diagnostics now also expose only
native recorder functions at the root PyO3 boundary.
Callback progress telemetry dispatch follows the same native-handle rule for
progress events and progress records.
Binary correction summary telemetry dispatch also uses the native telemetry
handle directly instead of Python fallback methods.
CLI run-failed telemetry dispatch resolves the native telemetry handle directly
and preserves its existing suppress-telemetry-errors policy.
The detached CLI run-failed plan/emission and telemetry close-failure helper
exports are no longer root PyO3 objects; production enters those policies
through the `NativeCliRunLifecycleState` handle.
The Rust architecture checker guards this telemetry boundary by rejecting root
PyO3 adapter calls to the old Python fallback method names.
The Python architecture checker also guards production-side runtime diagnostics:
direct native diagnostic payload builders are limited to compatibility adapters,
raw diagnostic emitters are rejected, and old Python telemetry fallback method
calls cannot reappear in production modules.
The root PyO3 module no longer exports raw diagnostic emitters; diagnostics
must flow through typed native recorder helpers.
The root PyO3 module also no longer exports raw telemetry event or writer
counter payload builders; telemetry payloads are built through native telemetry
session handles or typed native dispatch helpers.
The raw telemetry session policy payload helper has also been removed; direct
native tests use `NativeTelemetrySessionPolicy`.
Unused telemetry utility helpers for timestamp formatting, stream-file
resolution, path comparison, and explicit run-ID generation are no longer root
PyO3 exports.
Detached telemetry output-run root and telemetry path payload helpers are also
no longer root PyO3 functions; production resolves those paths through
`NativeTelemetrySessionPolicy`.
Standalone shutdown signal/default-list/second-signal helper exports have also
been removed; Python enters shutdown policy through `NativeShutdownController`.
It also rejects direct production event emission through `TelemetrySession`
compatibility wrappers or `native_session_handle`/`native_telemetry_session`
handles outside the telemetry adapter; production callers must use typed native
PyO3 dispatch helpers.
The real Python `TelemetrySession` no longer exposes those old fallback methods
or the typed `log_*` compatibility dispatch wrappers; focused telemetry tests
call the native telemetry session handle directly, and the Python architecture
checker rejects reintroduced production definitions of those methods.
Production JAX setup now validates GPU availability through the native
setup-session default-probe method; the old Python explicit-path validation
wrapper has been removed.
Unused direct payload-builder exports for manifest fingerprint mappings,
standalone manifest file fingerprints, prediction LOCO fingerprints,
current-run manifest headers, raw prepared-plan/header construction,
standalone file-content hashing, lower-level run artifacts, run-manifest
metadata extensions, and trusted BGEN validation cache entries have also been
removed from the root PyO3 module.
Run lifecycle telemetry-field builders are no longer Python-visible root
exports; native telemetry dispatch builds those fields inside the logging
adapter before emitting events.
Trusted BGEN validation cache default-directory resolution now enters through
the `Regenie2RunEngine` cache-validation method, so Python no longer calls a
standalone root helper and passes the cache directory back into the engine.
The explicit cache-directory engine method has also been removed from the
Python-visible `_core` surface.
The old top-level runtime-knob functions for BGEN tile sizing and Rayon thread
pool setup have also been removed; production setup goes through
`NativeRuntimeState`.
Standalone `require_gpu_device()` validation now also builds a native setup
session and uses the native default-probe method.
The detached default NVIDIA probe-path and driver-visibility root exports were
removed; Python convenience checks now enter through `NativeJaxRuntimeSetupSession`.
The detached JAX runtime policy payload builder was also removed from the root
module; payload construction now goes through `NativeRuntimeState`.
Detached logging policy payload, runtime policy handle, and seeded process
runtime-state builders were also removed; those builder calls now go through
`NativeRuntimeState`, as does concise logging policy formatting.
Default local JAX cache-directory resolution now comes from `g-runtime`; the
Python runtime-path adapter no longer reads the platform temporary directory or
current user name itself, and enters the policy through `NativeRuntimeState`.
The injected default local cache-directory builder is no longer a root PyO3
export; deterministic construction remains covered inside `g-runtime`, and
Python keeps only the production default-path adapter.
Production process-runtime JAX setup sessions now resolve default cache
directories inside `g-runtime`; the explicit Python cache-directory resolver
and Python setup-payload helper have been removed.
The Python architecture checker still rejects reintroduced production calls to
an explicit JAX cache-directory resolver outside the compatibility adapter.
It also rejects raw native setup-payload and setup-session construction from
production Python; setup sessions must come from native runtime state.
JAX backend initialization now takes only the native setup session from the
caller, so the production setup path cannot fall back to Python-side
setup-session construction or duplicate requested policy arguments. Production
JAX setup-resolution/config-update/side-effect, GPU-validation, and
validation-completion payloads now come through `NativeRuntimeState` and
`NativeJaxRuntimeSetupSession` methods, and the root PyO3 module no longer
exports detached setup helper payload functions for those paths.
The setup path also reads typed native setup-session properties instead of
unpacking side-effect-plan dictionaries, and the architecture checker rejects
reintroduced production calls to those dict payload helpers.
It also rejects direct production calls to `jax.config.update` and
`jax.devices`, keeping JAX setup side effects behind native setup sessions.
The runner import rule also rejects direct `jax`/`jaxlib` imports so JAX-facing
modules stay behind runtime setup.
Prediction-input LOCO manifest fingerprints now route through the native
manifest fingerprint cache handle, which composes `g-input` LOCO path
resolution with `g-output` file fingerprinting. Python adapts the native
payload for the transitional manifest-header mapping; the old Python-facing
raw LOCO path resolver and detached LOCO fingerprint function are no longer
exported from `_core`.
Preflight finite-array checks, SVD-backed covariate-rank validation, binary
phenotype coding/case-control scans, shape validation, prediction-shape
validation, variant-count validation, and report construction now enter through
the typed `NativePreflightValidator` handle. That handle executes the root PyO3
adapter over NumPy buffers before calling `g-engine` policy helpers. Python
preflight no longer uses `np.isfinite`, `np.linalg.matrix_rank`, `np.unique`,
or `np.count_nonzero` for those production checks, and the Python architecture
checker guards against reintroducing them in `g.engine.preflight`.
Schedule policy helpers for GPU genotype-format resolution, effective trusted
BGEN mode, delivery callback batch sizes, grouped-union validation,
committed-chunk intersection, writer-finish planning, BGEN delivery
invocation/cleanup, and output-write method selection now enter through the
typed `NativeSchedulePolicy` handle instead of detached root `_core` functions.
Callback helper entry points for null-logistic nonconvergence array planning,
callback chunk identity construction, callback progress telemetry dispatch, and
binary correction summary telemetry dispatch now enter through typed callback
policy handles instead of detached root `_core` functions.
Run lifecycle, writer, preflight, alignment, backend, BGEN-open, sample-summary,
GPU-format, and prediction-source telemetry dispatch helpers now enter through
the typed `NativeRunEventTelemetryPolicy` handle instead of detached root
`_core` functions.
Run-event lifecycle payload construction and terminal-line rendering now enter
through the typed `NativeRunEventPayloadPolicy` handle instead of detached root
`_core` functions.
Native CLI stdout/stderr, completion, interruption, failure, and runtime-knob
diagnostic recording now enters through the typed `NativeCliDiagnosticPolicy`
handle instead of detached root `_core` functions.
Runner lifecycle, execution-plan, engine-dispatch, and metadata-finalized
diagnostic recording now enters through the typed
`NativeRunnerDiagnosticPolicy` handle instead of detached root `_core`
functions.
Preflight/output, pipeline/callback-warning, and native-dispatch diagnostic
recording now enter through typed `NativeOutputPreflightDiagnosticPolicy`,
`NativePipelineDiagnosticPolicy`, and `NativeDispatchDiagnosticPolicy` handles
instead of detached root `_core` functions.
JAX runtime diagnostic recording and final timing output context/diagnostic
policy now enter through typed `NativeJaxRuntimeDiagnosticPolicy` and
`NativeFinalTimingOutputPolicy` handles instead of detached root `_core`
functions.
Pipeline output-preparation batch construction now enters through
`NativePipelineOutputPreparationPolicy` instead of a detached root `_core`
function.
Output lifecycle helpers for run-path resolution, manifest I/O, resume
validation/repair, initialization, and final chunk compaction now enter through
`NativeOutputLifecyclePolicy` instead of detached root `_core` functions.
Process-global runtime-state access now enters through
`NativeRuntimeState.global_process_runtime_state()`, and telemetry close
dispatch enters through `NativeTelemetryClosePolicy` instead of detached root
`_core` functions.
Output writer finish/interrupted-finish/abort lifecycle now uses
`OutputWriterSession` methods directly instead of detached root `_core`
functions.
Multi-trait output chunk writes now enter through `NativeOutputChunkWritePolicy`
instead of detached root `_core` functions.
Preflight chromosome collection now requires the native engine
`required_chromosomes` API directly; the old Python metadata-slice collection
fallback has been removed.
Callback chunk metadata now uses the native scalar `chromosome_label` contract
directly; Python no longer falls back to reading a full chromosome column from
metadata objects.
Callback readiness blocking now uses JAX's direct `block_until_ready` boundary,
and single-trait linear/binary callbacks require typed chromosome-state
readiness arrays (`adjusted_residual` and `score_residual`) instead of probing
prepared state objects for optional attributes.
Callback transfer helpers now require arrays with direct `shape`/`dtype`
metadata and native chunk stats with the `compute_arrays` contract; Python no
longer skips transfer metadata for missing attributes or falls back to per-field
chunk-stat property reads when native bundled arrays are absent.
Single-trait callback output writes now branch on the native write plan's typed
float64-native-writer flag instead of resolving writer methods from a
method-name string at runtime.
Callback null-logistic nonconvergence planning now has a PyO3 bool-array entry
point that owns scalar detection, flattening, total-fit counts, and
nonconverged counts before calling `g-engine` policy helpers. The callback
diagnostic path now materializes the JAX chromosome diagnostic values in one
host-transfer request and passes bool/int arrays into native timing-recorder
methods for scalar and multi-trait null-logistic rows. Python no longer builds
those timing dictionaries or rescans convergence counts with NumPy; the Python
architecture checker guards against reintroducing those reductions in
`g.engine.callbacks.diagnostics`, and it keeps production `jax.device_get`
host materialization isolated to callback diagnostic and writer adapters. It
also rejects readiness `getattr` probes in callback diagnostics and
single-trait chromosome-state preparation, plus optional transfer/chunk-stat
probing in `g.engine.callbacks.transfers` and callback writer method-name
probing in `g.engine.callbacks.writers`.
Binary chunk diagnostics now normalize score-only result dataclasses through
typed empty-Firth expansion helpers before reading diagnostic fields directly,
and the architecture checker rejects optional `getattr` result-field probes in
`g.compute.regenie2_binary.diagnostics`.
The same compute diagnostics module now stays device-side only: binary
diagnostic mapping and aggregate summary host materialization live in
`g.engine.callbacks.diagnostics`, and a source guard rejects `jax.device_get`
in the compute diagnostics module.
Run-scoped manifest file fingerprint caching now lives in `g-output` behind a
native PyO3 cache handle; control-file and prediction-input LOCO fingerprints
share that handle, and Python no longer resolves paths, stats files, or
maintains cache keys for manifest input fingerprints.
Standalone manifest file-fingerprint payload construction also routes through
that cache handle instead of a detached root PyO3 function.
Standalone file-content hashing and raw prepared-plan/prepared-header JSON
construction helpers have also been removed from the root PyO3 surface;
production output code keeps only the current-header based prepared-plan
adapters.
Current-run manifest header construction also routes through that native cache
handle; Python now passes the scalar header policy to the handle, adapts the
native prepared-header mapping, and no longer builds the production
manifest-header dataclass itself. The temporary Python manifest-header
dataclasses and sub-builders have been removed; the output adapter now passes
native manifest-header mappings through, and the old many-argument PyO3
manifest-header export and detached JSON-input root function have been removed.
Manifest writes, execution-plan hashing, prepared-plan construction, strict
resume validation/repair, and pipeline output-preparation batch construction
now use value-based PyO3 wrappers. The output adapter no longer calls
`json.dumps` for those manifest/preparation contracts before entering Rust.
Run manifest loading and prepared-output existing-manifest loading also return
native PyO3 value payloads, so production output code no longer parses those
native JSON strings with Python `json.loads`.
The root `_core` surface no longer exports the old output JSON-string helpers
for manifest load/write, manifest checksum, compatibility, initialization,
strict-resume validation/repair, or prepared-plan/header construction; Python
callers use the value-based output adapter exports instead.
JSON-string manifest cache methods and `NativePreparedOutputRun.existing_manifest_json`
are also no longer Python-visible; callers use the payload methods.
The resolved-config run-request JSON helper has also been removed from the
root `_core` surface; execution planning uses the payload export.
Pipeline output-preparation JSON batch constructors and direct JSON batch
initializers are likewise no longer exported; callers use the value-based
batch factory and the returned native handle.
Timing diagnostic snapshot serialization no longer reflects over dataclass
fields with production `getattr`; the Python architecture checker guards that
typed mapping boundary.
The Rust architecture checker also rejects re-exporting the removed
run-request JSON helper and pipeline output-preparation JSON batch initializer
functions, plus removed JSON compatibility class methods/getters.
Run-start manifest command/runtime metadata extension now goes through the
`NativeRunMetadataBuilder` handle before the native `g-output` manifest upsert;
Python no longer loads, mutates, serializes, and rewrites run manifests for
that metadata.
Real `OutputWriterSession` finish, interrupted flush, and abort cleanup now
route through root PyO3 module functions before entering `g-output`. The
native-dispatch writer adapter keeps the direct Python method fallback only for
fake and transitional test writer sessions, and the Python architecture checker
rejects direct calls to those native writer lifecycle helpers outside that
adapter.

Phase 10 callback-runner fallback removal is complete on this branch:
production scheduling, queue/resource ownership, worker lifecycle, result-slot,
and dosage-buffer paths no longer use manual Python fallback ownership.
Remaining Python side effects are tracked as Phase 11/12 adapter work.
Native BGEN delivery cleanup no longer carries a Python timing snapshot writer
callback; final timing snapshots and profile summaries are written once through
the runner's native final-timing boundary after dispatch.
Native BGEN delivery invocation also requires typed run-input alignment handles
and the callback runner's explicit `native_callback_batch_size`; production
delivery code no longer probes those contracts through optional Python
attribute fallbacks.
Native-dispatch callback start, drain, and abort helpers call the typed
callback lifecycle methods directly instead of probing for optional hooks.
Grouped union-sample fanout uses the same typed callback delivery contract and
calls child callback lifecycle methods directly instead of probing for optional
hooks.
For interactive `maturin develop -j 30 --profile dev-fast --uv` builds on
gauss, the 2026-07-02 warm incremental comparison after touching the root PyO3
crate favored wild over mold: mold took `24.43s` wall time (`18.08s` Cargo
target time), while wild `0.9.0` via the GCC driver `-B` linker path took
`21.17s` wall time (`14.46s` Cargo target time). Use the local
`/mnt/beegfs/kirill/Projects/g/.tools/bin/wild` binary for subsequent
interactive development builds when available.
Phase 13 GPU environment discovery has been exercised on `landau`: after
installing the locked `jax[cuda12]==0.10.1` extra into the shared development
environment, `check_native_cli_frontend` passed through `just slurm-gpu-run`
with `tool.expected_jax_device=gpu` and no `JAX_PLATFORMS` override, reporting
Python `3.14.3`, JAX `0.10.1`, and a visible GPU JAX device.
Phase 13 wheel installation has also been smoke-tested: a `dev-fast` wheel
built in `0:41.53` wall time (`29.23s` Cargo target time), installed into a
temporary Python 3.14 environment, imported `g` and `g._core`, and rendered
installed `g --help` output through the Python compatibility shim.
The Python architecture checker now guards that CLI boundary: public
`g.cli.run_args` must enter the coarse native PyO3 CLI runner, direct
`dispatch_cli` calls must stay in the sentinel-protected legacy backend, and
the Python console script cannot silently become the production process owner
again.
The root PyO3 timing recorder binding no longer exports direct
stage-timing/profile payload builders, the final timing write-started payload
builder, or per-file writer methods; Python callers use typed snapshots,
native diagnostic recorders, and the combined native final-timing output
writer.
JAX runtime setup diagnostics now come from `NativeJaxRuntimeSetupSession`;
the root PyO3 module no longer exports the direct setup diagnostic payload
builder, and Python no longer adapts diagnostics from a detached setup report.

Phase 10 queue migration also has a native callback scheduler state handle that
consolidates queue limits, worker-start state, result in-flight accounting, and
dosage-buffer pool accounting and reuse planning, worker failure state, worker
finish/abort planning, worker stop/join planning, chunk batch handoff planning,
queue observation planning, bounded-resource backpressure observation payload
construction, worker start action planning, and backpressure timeout policy for
production callback runners. Callback worker start attempts now mark the native
lifecycle state through native scheduler plans. Callback worker finish and abort sequencing now
comes from native action lists alongside native timeout policy. Callback modules
no longer resolve worker shutdown timeout constants at import time; production
shutdown timeout policy comes from native scheduler plans. Result in-flight
capacity is no longer mirrored by a Python
bounded semaphore; the native scheduler state is the capacity owner while the
Python transition layer only waits on slot-release notifications. Result
in-flight acquire/release decisions now come from native attempt plans. Free
host dosage buffers are no longer held in a bounded Python queue; the native
scheduler state owns buffer capacity and the Python transition layer keeps only
an available-buffer free list. Dosage-buffer acquisition, allocation
registration, return, and discard decisions now come from native attempt plans.
Worker error propagation now uses a native raise plan, with Python selecting
only the local exception cause to chain.
Result write queue occupancy is now tracked by native scheduler state instead
of a bounded Python queue, with Python retaining only transitional item storage
and wakeups. Dosage queue occupancy is now
tracked by native scheduler state instead of a bounded Python queue, with Python
retaining only transitional item storage and wakeups. Dosage and result queue
put/get wait decisions now come from native attempt plans. Multi-trait linear
and binary result consumers now always use the native runtime-resource
get/drain loop instead of a production Python fallback loop. The base callback
runner's dosage and single-result consumers now also require native
runtime-resource get/drain helpers, leaving manual scheduler consumer loops in
test fixtures only. Base dosage/result drain and dispatch planner helpers now
also require native runtime resources, with manual scheduler planner helpers
confined to test fixtures. Base dosage handoff and variant-major batch handoff
planner helpers now follow the same native-resource-only rule. The base
callback runner no longer branches on manual versus native runtime-resource
ownership; manual scheduler lifecycle, queue, result-slot, dosage-buffer,
progress, summary, handoff, and cleanup paths are test-fixture behavior only.

## Phase Order

1. Establish baseline records and promote this migration contract.
2. Add Cargo workspace infrastructure without moving production modules.
3. Extract `g-genotype`.
4. Extract `g-output`.
5. Extract `g-interface`.
6. Extract `g-input`.
7. Introduce `g-plan` and make manifests derive from `PreparedRunPlan`.
8. Extract and strengthen `g-runtime`.
9. Create `g-engine` with a deterministic fake backend.
10. Move preparation, scheduling, queues, and runtime side effects into Rust.
11. Collapse the root crate into a PyO3/JAX adapter.
12. Add a native CLI only after the Rust coordinator owns the full lifecycle.
13. Remove obsolete Python orchestration.

Each phase should be independently revertible. Mechanical extraction,
ownership transfer, fallback removal, and performance optimization should be
separate changes unless the change is too small to split safely.

## Required Checks

Every crate extraction must run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
cargo build --workspace --all-targets
cargo test --workspace
uv run pytest tests/ -m "not phase0_data and not phase1_parity"
just check-core-stub
just check-internal-defaults
just check-internal-init-exports
just check-rust-architecture
just check-python-architecture
uv run maturin develop
uv run python -c "import g; import g._core"
uv run g --help
```

Correctness-boundary phases also need parity harness coverage, manifest/schema
snapshots, fresh-versus-resumed equivalence, and malformed input tests. Hot-path
phases need Criterion benchmarks plus representative CPU/GPU benchmarks on the
appropriate SLURM nodes.

## Stop Conditions

Pause rather than force a phase through when a dependency cycle appears, a leaf
crate needs PyO3, sample or prediction alignment changes unexpectedly, resume
mutates outputs before validation, parity changes cannot be attributed, or a
performance regression cannot be measured and explained.
