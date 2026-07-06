# Telemetry

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | main branch as of 2026-06-30 telemetry and logging runtime | Runtime maintainers |

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
telemetry
log_dir
stage_timings_json
log_filter
log_file
log_stderr
progress_interval_seconds
progress_interval_chunks
profile_summary_json
trace_file
trace_filter
trace_event_cap
log_queue_size
log_lossy
include_source_location
include_span_events
```

By default, telemetry mode is `progress`. If `log_dir` is not set and
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

`log_file` configures the unified JSONL stream path. `trace_file` remains
accepted as a compatibility alias for the same stream. If both are configured,
they must point at the same path.

Progress and profile telemetry install the unified JSONL tracing layer with
`g-log-filter`, so durable operational diagnostics are present by default.
Trace telemetry uses `g-trace-filter` for targeted high-volume native tracing.

## Lifecycle Events

Run lifecycle facts are represented once as structured payloads and then used
for both terminal rendering and the JSONL diagnostics stream. `g regenie`
success and graceful-interruption messages are derived from these typed
lifecycle events.

Native frontend diagnostics are also included as structured tracing events for
run paths after diagnostics are initialized. `g.cli` prints user-facing native
stdout/stderr text exactly once, and mirrors bounded stdout/stderr previews plus
run-completion/interruption rendering lines through the native tracing bridge:

- `native_cli_stdout`
- `native_cli_stderr`
- `native_cli_completed_line`
- `native_cli_interrupted_line`

Help, parser-error, and validation-error paths that do not produce a run
configuration only print the native stdout/stderr text. They do not import the
Python runner, telemetry, shutdown handlers, or JAX runtime setup modules.
Native stdout/stderr diagnostics include character and byte counts, a bounded
preview, and truncation metadata instead of unbounded full payloads.

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
- association backend selection, including the stable `association_backend_kind`
- JAX runtime setup choices, including platform, cache directory, XLA auxiliary
  cache mode, transfer guard, and GPU validation
- preflight completion
- chromosome start and completion
- throttled progress ticks
- output writer completion
- run completion or failure

Never log per sample, per genotype, per probability byte, full phenotype
values, covariate matrices, genotype arrays, or large sample ID lists.

`association_backend_selected` is emitted before native BGEN delivery opens for
a run or phenotype group. It records `association_mode`,
`association_backend_kind`, requested `device`, and `genotype_format`. Current
backend kinds are `jax_dosage` and `jax_packed8`.

`gpu_genotype_format_resolved` is emitted when `gpu_genotype_format=auto` is
resolved. It records the requested format, resolved concrete format, and reason;
fallback events also include the trusted BGEN validation error.

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

Trace mode is for small runs, targeted chromosomes, or `--variant_limit`.
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
Trace telemetry event cap exceeded at <cap> events for <path>. Increase --trace_event_cap or set --trace_event_cap 0 to disable the cap for intentional deep traces. Use --log_lossy to drop events after the cap instead of failing.
```

Raise the cap for a planned deep trace, or set it to `0` to disable cap
enforcement:

```toml
[g.diagnostics]
telemetry = "trace"
trace-event-cap = 5000000
```

```bash
g regenie --telemetry trace --trace_event_cap 0
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
  --telemetry progress \
  --log_dir results/bmi.g/logs \
  --log_filter g=info
```

Benchmark profile mode:

```bash
g regenie \
  --step 2 \
  --bt \
  --telemetry profile \
  --log_dir results/binary-profile.g/logs \
  --no-log_stderr
```

Focused trace:

```bash
g regenie \
  --step 2 \
  --bt \
  --telemetry trace \
  --variant_limit 1000 \
  --trace_filter g.native.bgen=trace,g.output=debug \
  --trace_event_cap 1000000
```

## Production-Safe Logging

Production events should be bounded by run, chromosome, chunk, or writer batch.
They should contain identifiers, counts, durations measured without device
synchronization, and high-level status.

Binary correction diagnostics are emitted as one run-level
`binary_correction_summary` event. It aggregates score-only rows, score-test
candidates, approximate-Firth attempts, successes, failures, null-model
failures, and branch/failure diagnostics:

```json
{
  "event": "binary_correction_summary",
  "chunk_count": 8,
  "score_only_count": 100000,
  "score_test_candidate_count": 122,
  "firth_attempted_count": 122,
  "firth_success_count": 119,
  "firth_failed_count": 3,
  "firth_numerical_failure_count": 1,
  "firth_max_iteration_failure_count": 1,
  "firth_invalid_statistic_failure_count": 0,
  "firth_step_halving_failure_count": 1,
  "pseudo_firth_attempt_count": 80,
  "pseudo_firth_success_count": 78,
  "nr_zero_start_attempt_count": 44,
  "nr_zero_start_success_count": 39,
  "nr_warm_start_attempt_count": 5,
  "nr_warm_start_success_count": 2,
  "sparse_correction_count": 40,
  "dense_correction_count": 82,
  "null_model_failure_count": 0
}
```

Do not emit one event per candidate or per candidate iteration in production
mode.

## Follow-Up Tracking

Open telemetry implementation work is tracked in Linear instead of this page.
Durable telemetry lessons from historical task notes are consolidated in
[Agent Memory](../scratchpad/memory.md).
