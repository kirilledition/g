# Telemetry And Logging

`g` has one run-level diagnostics surface. Python builds lifecycle, progress,
and profile payloads, while Rust owns the physical JSONL writer queue used by
both Python telemetry and native tracing.

The production rule is strict:

```text
Production logging must not force JAX synchronization.
Profile logging may block or synchronize intentionally.
Trace logging is diagnostic and can perturb performance.
```

## Current Architecture

The normalized configuration lives in `GDiagnosticsConfig` and is reachable
from CLI, TOML, and Python config options. The main options are:

```text
g-telemetry
g-log-dir
g-stage-timings-json
g-log-filter
g-log-file
g-log-stderr
g-progress-interval-seconds
g-progress-interval-chunks
g-profile-summary-json
g-trace-file
g-trace-filter
g-trace-event-cap
g-log-queue-size
g-log-lossy
g-include-source-location
g-include-span-events
```

By default, telemetry mode is `progress`. If `g-log-dir` is not set and
telemetry is enabled, paths resolve under:

```text
<out>.g/logs/
```

The default stream layout is:

```text
<out>.g/logs/
  events.jsonl            # unified Python telemetry and Rust tracing stream
  profile.summary.json    # Python profile or trace mode, unless explicitly configured
  stage-timings.json      # Python profile or trace mode, unless explicitly configured
```

`g-log-file` configures the unified JSONL stream path. `g-trace-file` remains
accepted as a compatibility alias for the same stream. If both are configured,
they must point at the same path.

## Lifecycle Events

Run lifecycle facts are represented once as structured payloads and then used
for both terminal rendering and the JSONL diagnostics stream. `g regenie`
success and graceful-interruption messages are derived from these typed
lifecycle events.

`run_completed` includes the user-visible output artifacts needed by operators
and diagnostic tools:

```json
{
  "event": "run_completed",
  "association_mode": "regenie2_linear",
  "phenotype_count": 1,
  "output_run_directory": "results/output.g/trait.regenie2_linear.run",
  "final_dataset": "results/output.g/trait.regenie2_linear.run/parts",
  "final_parquet": "results/output.g/trait.regenie2_linear.run/final.parquet",
  "phenotype_artifacts": [
    {
      "phenotype": "trait",
      "output_run_directory": "results/output.g/trait.regenie2_linear.run",
      "final_dataset": "results/output.g/trait.regenie2_linear.run/parts",
      "final_parquet": "results/output.g/trait.regenie2_linear.run/final.parquet",
      "effective_config": "results/output.g/trait.regenie2_linear.run/effective_config.toml"
    }
  ]
}
```

For multi-phenotype runs, `phenotype_artifacts` contains one entry per
phenotype. Graceful shutdown is emitted as `run_failed` with
`failure_kind = "graceful_shutdown"`, signal metadata, the signal-derived exit
code, and `flushed_for_resume = true`.

## Supported Modes

### Off

```toml
[g.diagnostics]
telemetry = "off"
```

No telemetry stream is written. Rust stderr logging still follows `log-stderr`
and `log-filter`.

### Progress

```toml
[g.diagnostics]
telemetry = "progress"
log-filter = "g=info"
log-stderr = true
progress-interval-seconds = 5
progress-interval-chunks = 10
log-lossy = true
```

This is the production default. It writes low-volume lifecycle events and
throttled progress ticks into `events.jsonl`. It must not call `jax.device_get`,
`block_until_ready`, `np.asarray(jax_array)`, or any other operation that
forces device synchronization just to log.

Safe production events include:

- run start, config resolution, and execution-plan preparation
- JAX runtime setup choices, including platform, cache directory, XLA auxiliary
  cache mode, transfer guard, and GPU validation
- preflight completion
- chromosome start and completion
- throttled progress ticks
- output writer completion
- run completion or failure

Never log per sample, per genotype, per probability byte, full phenotype
values, covariate matrices, genotype arrays, or large sample ID lists.

### Profile

```toml
[g.diagnostics]
telemetry = "profile"
log-filter = "g=debug,g.native=debug,g.output=debug"
log-stderr = false
stage-timings-json = "results/run/logs/stage-timings.json"
profile-summary-json = "results/run/logs/profile.summary.json"
log-lossy = true
```

Profile mode is for benchmarks and optimization work. It may intentionally
synchronize JAX work so stage timings are meaningful. For example, stage timing
recorders can block on host-to-device transfer and compute completion.

Profile-mode numbers answer a different question from production throughput:

```text
profile mode: detailed synchronized stage measurements
progress mode: normal production behavior
```

Use both when evaluating performance changes.

### Trace

```toml
[g.diagnostics]
telemetry = "trace"
log-filter = "g=debug"
trace-filter = "g.native.bgen=trace,g.output=debug"
trace-event-cap = 1000000
log-file = "results/run/logs/events.jsonl"
log-stderr = false
log-lossy = true
```

Trace mode is for small runs, targeted chromosomes, or `--g-variant-limit`.
It may emit high-volume native events and can perturb performance. Do not use
it for full production-scale scans unless the goal is to diagnose a specific
runtime problem.

Trace mode has a default `trace-event-cap = 1000000`, enforced by the
Rust-owned JSONL stream before completed event lines are queued for writing.
The cap applies only when `telemetry = "trace"`; progress and profile streams
are not constrained by this trace-only cap.

When `log-lossy = true`, events after the cap are dropped and the native writer
prints one stderr diagnostic that additional trace events are being dropped.
When `log-lossy = false`, the run fails with:

```text
Trace telemetry event cap exceeded at <cap> events for <path>. Increase --g-trace-event-cap or set --g-trace-event-cap 0 to disable the cap for intentional deep traces. Use --g-log-lossy to drop events after the cap instead of failing.
```

Raise the cap for a planned deep trace, or set it to `0` to disable cap
enforcement:

```toml
[g.diagnostics]
telemetry = "trace"
trace-event-cap = 5000000
```

```bash
g regenie --g-telemetry trace --g-trace-event-cap 0
```

Telemetry sessions record native writer counters on close. The final
`telemetry_session_closed` event includes `writer_counters` with accepted,
written, cap-dropped, queue-dropped, and total dropped event counts, the cap
state, lossy mode, and native finish/flush duration. Use these counters when
profiling logging overhead so a faster run caused by dropped events is not
mistaken for real application throughput.

## CLI Examples

Production default with an explicit log directory:

```bash
g regenie \
  --step 2 \
  --qt \
  --g-telemetry progress \
  --g-log-dir results/bmi.g/logs \
  --g-log-filter g=info
```

Benchmark profile mode:

```bash
g regenie \
  --step 2 \
  --bt \
  --g-telemetry profile \
  --g-log-dir results/binary-profile.g/logs \
  --g-log-stderr false
```

Focused trace:

```bash
g regenie \
  --step 2 \
  --bt \
  --g-telemetry trace \
  --g-variant-limit 1000 \
  --g-trace-filter g.native.bgen=trace,g.output=debug \
  --g-trace-event-cap 1000000
```

## Production-Safe Logging

Production events should be bounded by run, chromosome, chunk, or writer batch.
They should contain identifiers, counts, durations measured without device
synchronization, and high-level status.

For binary Firth diagnostics, prefer aggregate chunk summaries:

```json
{
  "event": "firth_chunk_summary",
  "candidate_count": 122,
  "converged_count": 119,
  "failed_count": 3,
  "iteration_min": 4,
  "iteration_median": 11,
  "iteration_max": 41
}
```

Do not emit one event per candidate iteration in production mode.

## Follow-Up Tracking

Open telemetry implementation work is tracked in Linear instead of this page.
Durable telemetry lessons from historical task notes are consolidated in
[Agent Learning](../scratchpad/agent-learning.md).
