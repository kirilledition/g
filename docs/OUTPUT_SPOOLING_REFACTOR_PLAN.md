# Output Spooling Refactor Plan (Arrow Chunks + Manifest + Optional Parquet Finalization)

## Goal

Implement a resumable, low-interference output pipeline for GWAS compute where:

1. Compute progression is minimally impacted by persistence overhead.
2. Interrupted runs can be resumed deterministically.
3. Output dtypes remain explicit and stable end-to-end.

The recommended architecture is:

- During compute: immutable Arrow IPC chunk files.
- During compute: append-only manifest transactions for chunk commit state.
- After compute: optional Parquet compaction/finalization.

## Why this is needed in `g`

The current engine flow accumulates chunk outputs and builds one final table in-memory at the end of processing (`concatenate_linear_results` and `concatenate_logistic_results` in `src/g/engine.py`). This design can increase host memory pressure and complicates resumability for long runs.

## Proposed output layout

Each execution run receives its own directory:

```text
<output_root>/
  run_<run_identifier>/
    manifest.sqlite
    schema.arrow.json
    chunks/
      000000.arrow
      000001.arrow
      ...
```

### Naming and deterministic chunk identity

Chunk identifiers must be deterministic from genomic bounds and chunking policy, not just process order. Use fixed metadata fields such as:

- chromosome
- start_variant_index
- end_variant_index

Persist these fields in the manifest and as Arrow columns to support deterministic resume and auditability.

## Proposed module additions

### 1) `src/g/output/schema.py`

Defines explicit Arrow schema and schema version utilities.

Suggested responsibilities:

- Build one canonical `pyarrow.Schema` for active-run chunk files.
- Persist schema JSON to `schema.arrow.json` at run start.
- Validate chunk `RecordBatch` objects against the schema before write.

### 2) `src/g/output/manifest.py`

Implements append-only manifest backed by SQLite.

Suggested table model:

- `run_metadata` (run identifier, schema version, creation timestamp, status)
- `chunks` (chunk identifier, genomic bounds, row count, relative path, checksum, committed timestamp, status)

A chunk is considered committed only after:

1. Arrow temp file write completes.
2. File is atomically renamed to final name.
3. Manifest transaction commits chunk row with `status='committed'`.

### 3) `src/g/output/spooler.py`

Creates a dedicated spooler process and message protocol.

Suggested typed messages:

- `StartRunMessage`
- `ChunkPayloadMessage`
- `FinalizeRunMessage`
- `AbortRunMessage`

Spooler responsibilities:

- Accept chunk payload messages from compute.
- Build Arrow `RecordBatch` from explicitly typed arrays.
- Write `chunk_<identifier>.arrow.tmp` then atomic rename.
- Commit manifest row in same logical operation.

### 4) `src/g/output/finalize.py`

Optional post-run compaction.

Responsibilities:

- Scan manifest committed chunks in deterministic order.
- Emit Parquet dataset directory or single Parquet file.
- Apply compression only in finalization phase.

## Engine integration plan

### Current state

`src/g/engine.py` currently yields per-chunk accumulators and later concatenates all outputs into one frame.

### Target state

Refactor iterator loops to publish chunk payloads to spooler during compute.

High-level integration pattern:

1. Start run context and spooler process.
2. For each computed chunk:
   - `jax.device_get(...)` arrays in compute process.
   - build typed `ChunkPayloadMessage` with metadata and array buffers.
   - enqueue payload.
3. Spooler persists Arrow chunk and manifest commit.
4. Compute continues immediately unless bounded queue backpressure is reached.
5. On normal completion, request run finalization state update and exit.

## Backpressure and failure policy

Absolute zero blocking is impossible with finite memory and finite storage throughput. The implementation must define explicit behavior for queue saturation.

Recommended policy:

- Use a large bounded queue for normal non-blocking operation.
- If queue remains full beyond timeout, fail run explicitly with a clear diagnostic.
- Do not silently degrade into unbounded memory growth.

## Resume algorithm

On restart for an existing run directory:

1. Load schema and manifest.
2. Enumerate committed chunk identifiers.
3. Skip committed chunks in compute iteration.
4. Recompute only missing chunks.
5. Ignore orphan `.tmp` files not recorded as committed.

## Staged implementation milestones

### Milestone 1: manifest and schema foundations

- Add schema module and schema serialization.
- Add SQLite manifest implementation with commit semantics.
- Add unit tests for transaction boundaries and recovery behavior.

### Milestone 2: synchronous per-chunk writer (no new process yet)

- Add chunk writer utility used directly from compute loop.
- Persist immutable Arrow files and manifest rows.
- Retain current in-memory output path behind feature flag for regression comparison.

### Milestone 3: spooler process and queue protocol

- Introduce process boundary and message classes.
- Route compute outputs through spooler.
- Add integration tests with forced interruption and resume.

### Milestone 4: optional Parquet finalizer

- Add post-run compaction command.
- Verify dtype parity from Arrow chunks to Parquet artifact.

## Validation and tests

Minimum required tests:

1. Schema fidelity test: float32/int32/bool/string round-trip from chunk payload to Arrow file.
2. Crash simulation test: terminate during active run and verify resume skips committed chunks only.
3. Deterministic resume test: same chunk policy and same inputs produce identical committed chunk set.
4. Finalizer test: compacted Parquet row counts and value parity match manifest + Arrow chunks.

## CLI surface (proposed)

Add optional flags to existing linear/logistic commands:

- `--output-mode in-memory|arrow-chunks` (default `arrow-chunks` after rollout)
- `--output-run-dir <path>`
- `--resume` (resume from existing run directory)
- `--finalize-parquet` (post-run compaction)

## Operational notes

- Prefer local NVMe scratch directories for active chunk writes.
- Avoid network filesystems for active spooling where possible.
- Keep chunk boundaries stable across reruns to preserve deterministic resume behavior.

## Rollout strategy

1. Land modules and tests behind feature flags.
2. Compare performance and correctness against current path.
3. Promote Arrow chunk mode to default when parity and throughput are validated.
4. Keep finalizer optional to avoid compute-path compression overhead.
