# Telemetry

| Status | Applies to | Owner |
| --- | --- | --- |
| Active native contract | telemetry, logging, and timing as of 2026-07-10 | Runtime maintainers |

Rust owns telemetry policy, event payloads, the JSONL writer, timing output,
terminal rendering, and session close. Python/JAX returns typed numerical
diagnostics only; it does not own telemetry lifecycle or event selection.

## Configuration

Diagnostics use canonical TOML fields:

```toml
[diagnostics]
telemetry = "progress" # off, progress, profile, or trace
log_dir = "results/example.g/logs"
log_filter = "g=info"
log_stderr = true
log_queue_size = 4096
log_lossy = false
include_source_location = false
include_span_events = false
```

Optional paths and trace policy:

```toml
[diagnostics]
log_file = "results/example.g/logs/events.jsonl"
stage_timings_json = "results/example.g/logs/stage-timings.json"
profile_summary_json = "results/example.g/logs/profile.summary.json"
trace_file = "results/example.g/logs/events.jsonl"
trace_filter = "g.native.bgen=trace,g.output=debug"
trace_event_cap = 1000000
```

Unknown fields and conflicting stream paths are rejected during native config
validation. The command line does not duplicate this native diagnostics
surface.

## Modes

| Mode | Behavior |
| --- | --- |
| `off` | No telemetry stream. Explicit timing/profile paths still produce their configured files. |
| `progress` | Bounded lifecycle events suitable for normal runs. |
| `profile` | Lifecycle events plus aggregate stage timing output. |
| `trace` | High-volume filtered tracing with an event cap. |

Production logging must not force JAX synchronization. Profile/trace runs may
synchronize intentionally and can perturb performance.

## Paths

When telemetry is enabled and `log_dir` is omitted, native path resolution uses:

```text
<out>.g/logs/events.jsonl
```

Profile and trace modes resolve `profile.summary.json` under that directory
unless an explicit profile path is configured. Stage timing output is written
when `stage_timings_json` is configured. The same native run ID is used by the
telemetry stream and profile summary.

## Lifecycle

The native CLI owns:

1. path and logging policy resolution;
2. telemetry session construction;
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

## Timing

`g-runtime::StageTimingRecorder` is constructed when a stage-timing path
or profile output is requested. Engine stages record through the native host;
final timing files are written on success, failure, and interruption. A timing
write error fails an otherwise successful run but never masks the primary run
or interruption error.

Stage timing records only host stages that have active production producers.

## Trace Cap

Trace mode applies `trace_event_cap` before events are queued. A positive value
limits the event count; `0` disables the cap for intentional deep traces. With
`log_lossy = true`, excess events are dropped and counters report the loss. With
lossless logging, exceeding the cap fails the run.

The close event reports accepted, written, cap-dropped, queue-dropped, and total
dropped counts. Use these counters when
interpreting profiling results.

## Event Design

- Emit production events at run, chromosome, chunk, or writer-batch scope.
- Keep identifiers, counts, and host-measured durations structured.
- Do not serialize large arrays or per-variant payloads into telemetry.
- Build payload shape in `g-runtime`; binding code only adapts `PyErr` or emits
  the native payload.
- Keep terminal output derived from the same typed lifecycle facts used by
  telemetry.
