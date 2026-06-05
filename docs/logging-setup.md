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
log-file = "results/run/logs/events.jsonl"
log-stderr = false
log-lossy = true
```

Trace mode is for small runs, targeted chromosomes, or `--g-variant-limit`.
It may emit high-volume native events and can perturb performance. Do not use
it for full production-scale scans unless the goal is to diagnose a specific
runtime problem.

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
  --g-trace-filter g.native.bgen=trace,g.output=debug
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

## Remaining Telemetry Roadmap

- Keep file handles or queue-backed writers open for profile and trace streams
- Add explicit timing for Firth candidate-count host synchronization.
- Add bounded event caps for trace mode so accidental high-volume tracing
  fails clearly instead of filling disks.
- Keep profile summary generation aligned with stage timings, native BGEN
  profiles, output finalization timing, and binary diagnostic counters.
