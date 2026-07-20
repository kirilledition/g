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

When telemetry is enabled, the frontend derives:

```text
<out>.g/logs/events.jsonl
```

Profile mode also writes `profile.summary.json` under that directory. Output
writers write `output_stage_timings.json` in each phenotype run directory.
The same native run ID is used by the telemetry stream and profile summary.

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
7. telemetry close and writer counters.

Help, parser errors, and validation failures return before JAX/backend
construction. Active run failures produce a concise terminal error and a typed
`run_failed` event. Graceful SIGINT/SIGTERM produces signal metadata and the
signal-derived exit code after resumable writer flushing.

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

When both telemetry and timing are disabled, the native session allocates no
run ID or shared telemetry state. Enabled session clones share one run ID and
writer state. Event envelopes borrow the run ID, event name, level, and thread
name through immediate JSON serialization (production levels map to static
uppercase labels); only the timestamp and final JSONL record are owned per
event.

An enabled telemetry session receives progress, execution-plan preparation, and
JAX setup diagnostics directly as typed events. When telemetry is disabled,
execution-plan preparation and JAX setup diagnostics use the tracing diagnostic
route instead. No diagnostic is sent through both routes, and progress
bookkeeping is not constructed when telemetry is off. The persistent-cache
diagnostic always reports `enabled=true`, the resolved directory,
`min_entry_size_bytes=-1`, and `min_compile_time_seconds=0`. Auxiliary-cache and
transfer-guard diagnostics report their fixed disabled policy.

Progress registers one uniquely owned counter entry per delivery; the joined
phenotype label is payload text, not an identity key. Complete-plan totals are
computed once per run, and writer completion updates lock only their delivery's
counter entry rather than a process-wide group map.

## Timing

`g-runtime::StageTimingRecorder` is constructed in profile mode. It stores one
owned key and aggregate per stage, then serializes borrowed totals/counts views
to preserve the public JSON shape without duplicate maps. Engine stages record
through the native host; final timing files are written on success, failure,
and interruption. A timing
write error fails an otherwise successful run but never masks the primary run
or interruption error.

Stage timing records only host stages that have active production producers.

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
