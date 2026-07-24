# Telemetry

| Status | Applies to | Owner |
| --- | --- | --- |
| Active native contract | telemetry, logging, and timing as of 2026-07-10 | Runtime maintainers |

Rust owns telemetry policy, event selection, the JSONL writer, timing output,
terminal rendering, and session close. `g-runner` interprets diagnostics modes
and output layout, then projects concrete paths and sink flags into a generic
native-session policy. `g-runtime` owns only transport and session state;
`g-runner`/`g-engine` shape the events they own, and `g-runner` renders terminal
output. Python/JAX returns typed numerical diagnostics only.

## Configuration

The production frontend currently exposes one diagnostics setting:

```toml
[diagnostics]
telemetry = "progress" # off, progress, or profile
```

No other diagnostics keys are accepted. Fixed logging and queue behavior belongs
to `g-runner`; unknown fields are rejected during native config validation.

## Modes

| Mode | Behavior |
| --- | --- |
| `off` | No telemetry stream or profile summary. |
| `progress` | Bounded lifecycle events suitable for normal runs. |
| `profile` | Lifecycle events plus aggregate stage timing output. |

Production logging must not force JAX synchronization. Profile runs may
synchronize intentionally and can perturb performance.

## Paths

After output ownership is claimed, the frontend derives:

```text
<out>.g/attempts/<attempt-id>/diagnostics/<owner-claim-id>/events.jsonl
```

Profile mode also writes `profile.summary.json` in that claim-specific
diagnostics directory. Failed pre-activation claims and completed read-only
resumes remove the directory only after timing, telemetry, and logging close;
an activated writable attempt retains it as attempt-bound diagnostics. Output
writers write `output_stage_timings.json` under
`attempts/<attempt>/<phenotype-output-name>/` before terminal authority is
published. The same native run ID is used by the telemetry stream and profile
summary.

The telemetry JSONL envelope and `profile.summary.json` currently carry
`schema_version: 0`. The application is still unreleased, so these persisted
contracts remain at version `0` until the first public release establishes the
initial compatibility baseline.

## Lifecycle

The native CLI owns:

1. path and logging policy resolution;
2. construction of the runtime session from resolved, plan-free policy;
3. frontend diagnostic recording;
4. run, writer, and artifact events;
5. timing/profile serialization;
6. terminal result rendering;
7. telemetry close and writer counters;
8. after successful telemetry and logging close, exactly one deferred output
   action: claimed-state abort, pre-activation rollback, or terminal-produced
   completed-resume cleanup.

Help, parser errors, and validation failures return before JAX/backend
construction. Active run failures produce a concise terminal error and a typed
`run_failed` event. Graceful SIGINT/SIGTERM produces signal metadata and the
signal-derived exit code after resumable writer flushing.

Runner freezes execution into one typed primary outcome: completed,
interrupted, or failed. A backend, output, or interruption outcome is never
replaced by later timing, telemetry-close, logging-close, or signal
observations. Required timing and close failures fail only an otherwise
successful run, and completed artifact paths remain on standard output. A
signal first observed after durable completion is warning-only.
Deferred rollback runs only after timing plus successful telemetry and logging
close. If either asynchronous close fails, output cleanup is skipped and the
exact claim remains fail-closed for explicit fencing. If rollback itself fails,
the primary terminal line and nonzero exit remain first and the rollback error
is appended as secondary stderr.
Completed-resume cleanup authority is neither available nor clonable before a
terminal success or typed terminal error. Its cleanup failure preserves the
fixed primary result and completed artifact paths, changing only an otherwise
successful exit to failure. The runner attempts it once without a blind retry;
the retained exact claim remains fence-recoverable on persistent failure.

The `writer_finished` event reports `parquet_dataset_path` for one phenotype or
`parquet_dataset_paths` for a multi-phenotype run. These required paths point to
the completed `parts/` datasets; there is no optional derived-file path.

`TelemetryRunSession` treats a concrete stream path as enabled and no path as
disabled; it does not know the `off`, `progress`, or `profile` planning enum.
The tracing subscriber is process-global, but its asynchronous writers are
run-owned. Subscriber layers resolve the currently registered stderr, log-file,
and telemetry writers for each record. Run close first removes those routes and
waits for any formatter already holding a writer, then drops each worker guard
to drain and flush the queue. A later compatible run registers fresh workers
without reinstalling or invalidating the global subscriber. Compatibility
therefore covers only subscriber topology and formatting (filter, enabled
layers, source locations, and span events); output paths, queue capacity, and
lossy mode are run-owned and may differ between invocations.

Runner holds process runtime state while `NativeRunSession` checks subscriber
compatibility, constructs writers, installs the subscriber, and records the
topology. The same resolved session policy drives every step, so an
incompatible repeated run has no file-open or worker-start side effects.
Subscriber installation errors are typed failures and leave the runtime's
subscriber state uninitialized rather than recording a subscriber it did not
install.

When both telemetry and timing are disabled, the native session allocates no
run ID or shared telemetry state. Enabled session clones share one run ID and
writer state. Event envelopes borrow the run ID, event name, level, and thread
name through immediate JSON serialization (production levels map to static
uppercase labels); only the timestamp and final JSONL record are owned per
event.

An enabled telemetry session receives progress, execution-plan preparation,
JAX setup, and the once-per-run `association_implementation_selected`
observation directly as typed events. When telemetry is disabled,
execution-plan preparation, JAX setup, and implementation selection use the
tracing diagnostic route instead. No diagnostic is sent through both routes,
and progress bookkeeping is not constructed when telemetry is off. The
implementation-selection event records exact JAX/JAXlib versions; optional
Firth requested/effective/fallback names; raw-CUDA target, API, handler and PTX
identity, and minimum driver/compute-capability thresholds; and the
reason-appropriate observed driver/device fields. Free-text fallback detail is
excluded. The persistent-cache diagnostic always reports `enabled=true`, the
resolved directory,
`min_entry_size_bytes=-1`, and `min_compile_time_seconds=0`. Auxiliary-cache and
transfer-guard diagnostics report their fixed disabled policy. Diagnostic
emission after runtime work starts is best-effort: failures are warned about but
do not replace the primary run outcome. JAX setup is committed as process state
before its completed-setup diagnostics are emitted. Progress and lifecycle
telemetry emission follow the same best-effort rule, including the final
progress update after durable output completion. Observer calls and the warning
path that reports their failure are panic-contained.

The optional CUDA observation fields use this exact presence matrix:

| Selection result | `cuda_driver_version` | `cuda_device_ordinal` | `cuda_compute_capability_major` / `cuda_compute_capability_minor` |
|---|---:|---:|---:|
| Qualified raw CUDA | present | present | present |
| `unsupported_platform`, `cuda_driver_unavailable`, or `required_symbol_unavailable` | absent | absent | absent |
| `cuda_driver_too_old` | present | absent | absent |
| `cuda_device_unavailable` | present | present | absent |
| `unsupported_compute_capability` | present | present | present |

Whenever raw CUDA was requested, the event also carries
`raw_cuda_ffi_target`, `raw_cuda_ffi_api_version`,
`raw_cuda_handler_sha256`, `raw_cuda_ptx_sha256`, `raw_cuda_ptx_isa`,
`raw_cuda_ptx_target`, `raw_cuda_minimum_cuda_driver_version`,
`raw_cuda_minimum_compute_capability_major`, and
`raw_cuda_minimum_compute_capability_minor`.

Progress registers one uniquely owned counter entry per delivery; the joined
phenotype label is payload text, not an identity key. Complete-plan totals are
computed once per run, and writer completion updates lock only their delivery's
counter entry rather than a process-wide group map. The first totals,
registration, initialization, update, final-emission, or observer-panic failure
warns once and atomically disables the reporter. Later progress calls are
no-ops, so progress can neither stop active execution nor hide durable output.

Delivery-report and association-warning counters are observer-only unsigned
64-bit values and are recorded infallibly, including native `usize::MAX` on
64-bit hosts. Phenotype output-count validation uses the exact native count;
conversion to the telemetry counter domain is separate and cannot abort
structural artifact validation. These changes preserve the schema-version `0`
JSON field names and shapes.

## Timing

`g-runtime::StageTimingRecorder` is constructed in profile mode. It stores one
owned key and aggregate per stage, then serializes borrowed totals/counts views
to preserve the public JSON shape without duplicate maps. Engine stages record
through the native host; final timing files are written on success, failure,
and interruption. A timing
write error fails an otherwise successful run but never masks the primary run
or interruption error. Completed artifact paths remain visible when the timing
write is the only failure.

Stage timing records only host stages that have active production producers.
Output-writer timing keeps the durability boundaries separately attributable:

| Stage key | Boundary |
| --- | --- |
| `rust_output_writer_parquet_file_sync` | `sync_all` of the completed temporary Parquet file |
| `rust_output_writer_parquet_file_hash` | full raw-byte SHA-256 and size reread |
| `rust_output_writer_parquet_file_publish` | no-replace final-name publication or existing-final reconciliation |
| `rust_output_writer_parquet_directory_sync` | synchronization of the `parts/` directory |
| `rust_output_writer_receipt_publish` | immutable receipt write, publication, and directory durability |

All five are also included in `rust_output_writer_total`. Hash time remains a
separate reread cost so benchmark evidence can decide whether a streaming
digest is warranted without weakening recovery-time raw-byte verification.

## Queue Counters

The close event reports accepted, written, queue-dropped, and total dropped
event counts. Use these counters when interpreting profile output.

## Event Design

- Emit production events at run, chromosome, chunk, or writer-batch scope.
- Keep identifiers, counts, and host-measured durations structured.
- Do not serialize large arrays or per-variant payloads into telemetry.
- Build payload shape in the runner or engine that owns the event; use
  `g-runtime` only for generic emission/session transport. Binding code only
  adapts `PyErr` and Python runtime observations.
- Keep terminal output derived from the same typed lifecycle facts used by
  telemetry.
