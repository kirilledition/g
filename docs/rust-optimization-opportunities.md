# Rust Optimization Opportunities

Review date: 2026-06-07

Scope: native Rust code under `src/`, with emphasis on BGEN decoding,
preprocessing, Python bridge data movement, output writing, and setup paths.

Non-goal: this document does not propose code changes as final decisions. Each
finding should be validated with targeted benchmarks before implementation.

## Executive Summary

The Rust layer already has several good performance foundations:

- BGEN decoding uses per-thread scratch buffers for zlib output and dosage
  tiles.
- The main 8-bit, unphased, all-present variant-major dosage path has an AVX2
  fast path.
- Python output can pass NumPy result buffers into Arrow without copying the
  result columns.
- Output sessions batch chunks and write them asynchronously.

The highest-value opportunities appear to be:

1. Avoid reading and decompressing trusted BGEN blocks once for validation and
   again for decode.
2. Fuse packed 8-bit probability-pair copying with summary/stat computation.
3. Reduce repeated allocation and cloning of chunk statistics and metadata.
4. Stream output record batches instead of building all batches for a grouped
   output file in memory.
5. Avoid full reread/rewrite finalization when Parquet output can be produced
   directly in final form.
6. Broaden SIMD coverage beyond the current all-present contiguous 8-bit dosage
   path.

## Ranked Findings

### 1. Trusted BGEN validation rereads and redecompresses data later decoded

**Where**

- `src/genotype/bgen/reader.rs:296` validates every variant in trusted mode.
- `src/genotype/bgen/trusted.rs:36` reads each probability block during
  validation.
- `src/genotype/bgen/trusted.rs:277` and `src/genotype/bgen/trusted.rs:415`
  read probability blocks again during trusted packed-pair and dosage decoding.

**Observation**

Trusted mode validates that every variant satisfies the no-missing diploid
contract by scanning each probability block. Later, the decode path seeks back
through the same records and reads/decompresses those blocks again.

That means trusted mode pays a duplicate I/O and decompression cost before any
useful chunk output is produced. This is especially visible for zlib-compressed
BGEN and for short analyses where validation is a large fraction of wall time.

**Potential direction**

- Cache validation metadata and, if memory permits, decompressed probability
  blocks for the next decode pass.
- Consider validating lazily while decoding trusted chunks, with the same
  contract checks but without a separate full-file pass.
- If eager validation must remain available, expose it as an explicit mode or
  reuse the validated/decompressed data through a block cache.

**Validation**

Benchmark trusted 8-bit BGEN reads with and without a separate validation pass:

- compressed versus uncompressed BGEN,
- full sample selection versus contiguous subset,
- packed probability-pair output versus dosage output.

### 2. Packed 8-bit trusted output copies bytes and then rescans for stats

**Where**

- `src/genotype/bgen/trusted.rs:350` copies full packed bytes to output.
- `src/genotype/bgen/trusted.rs:357` copies contiguous selected bytes to output.
- `src/genotype/bgen/trusted.rs:245` summarizes copied packed bytes in a second
  scalar pass.
- `src/genotype/bgen/simd.rs:122` has AVX2 infrastructure for 8-bit dosage
  decode, but not for packed probability-pair summary.

**Observation**

The trusted packed probability-pair path has three sample-selection cases:

- identity selection: copy the row, then summarize the copied row,
- contiguous selection: copy the selected byte range, then summarize it,
- non-contiguous selection: copy and summarize in the same scalar loop.

The fastest selection cases read the selected bytes twice. This is the opposite
of the trusted dosage path, where identity and contiguous paths already use the
AVX2 decode/stat routine.

**Potential direction**

- Fuse copy and summary for identity and contiguous selection.
- Add a SIMD summary routine for packed 8-bit probability pairs that computes
  raw `p1 + 2 * p2` dosage sums and square sums while copying bytes.
- Reuse the existing lookup table only for scalar tails or non-contiguous
  selection.

**Validation**

Add a benchmark for `read_preprocessed_variant_major_probability_pairs_u8` that
covers identity, contiguous, and non-contiguous sample selections. Measure both
decode throughput and stat generation time.

### 3. Chunk statistics allocate many vectors and duplicate `dosage_sum`

**Where**

- `src/genotype/bgen/reader.rs:91` allocates eight summary vectors per chunk.
- `src/genotype/bgen/reader.rs:478` creates the variant-major stats buffers for
  each prepared read.
- `src/genotype/preprocess.rs:331` allocates derived per-variant stat vectors.
- `src/genotype/preprocess.rs:391` clones `dosage_sum` so both `dosage_sum` and
  `allele_count` are stored.
- `src/python/mod.rs:67` through `src/python/mod.rs:129` clone stat vectors
  again when exposing them to Python.

**Observation**

The native chunk stats path first allocates mutable decode summary vectors, then
builds derived `ChunkStats` vectors, then Python getters clone those vectors
when NumPy arrays are requested. `dosage_sum` and `allele_count` currently store
the same values as separate vectors.

This is not necessarily a hot-path compute bottleneck, but it can be a memory
bandwidth and allocation bottleneck when chunk sizes are large or many chunks
are in flight.

**Potential direction**

- Reuse `VariantMajorStatsBuffers` across chunks when shape is stable.
- Store `allele_count` as an alias or derived view where the public API allows
  it.
- Consider a representation where Rust-owned stats can be exposed to Python as
  arrays without cloning, similar in spirit to the Python-owned Arrow buffer
  path for result arrays.
- If API compatibility requires separate vectors, delay the clone until a
  consumer actually requests both columns.

**Validation**

Measure allocations and memory bandwidth for a large variant-major run. A good
starting point is `heaptrack`, `valgrind massif`, or allocator-level counters
around `read_preprocessed_variant_major_dosage_f32_into_address_prepared`.

### 4. Metadata and stats are cloned repeatedly across Python and output paths

**Where**

- `src/python/mod.rs:324` through `src/python/mod.rs:347` clone metadata fields
  for Python getters.
- `src/python/output.rs:333` builds native chunk handles from Python objects.
- `src/output/writer.rs:351` through `src/output/writer.rs:365` clone metadata
  and stat vectors into Arrow arrays for each record batch.

**Observation**

Chunk metadata and stats cross several ownership boundaries:

1. Rust creates metadata and stats.
2. Python receives chunk handles and may request NumPy arrays, cloning values.
3. Native output code rebuilds Arrow arrays from cloned metadata and stats.

The result columns have a more efficient path: Python-owned NumPy buffers are
wrapped into Arrow buffers without copying. Metadata and stats do not yet get
similar treatment.

**Potential direction**

- Cache Arrow arrays for immutable chunk metadata and stats inside the native
  chunk handle.
- Separate "Python inspection" accessors from "writer" accessors so the writer
  does not need to rebuild arrays through cloned vectors.
- For string metadata, evaluate Arrow string builders fed directly from existing
  string slices instead of cloning the whole vector first.

**Validation**

Profile output-heavy runs where association computation is cheap relative to
writing. Track clone counts and bytes copied in `build_regenie_step2_record_batch`.

### 5. Output writer builds all record batches for a grouped file before writing

**Where**

- `src/output/writer.rs:159` builds all record batches for a write job.
- `src/output/writer.rs:292` through `src/output/writer.rs:308` collects a
  `Vec<RecordBatch>`.
- `src/output/writer.rs:430` through `src/output/writer.rs:488` writes the
  already-built batches.

**Observation**

For each grouped output file, all `RecordBatch` values are built before any of
them are written. Large `chunks_per_arrow_file` settings increase peak memory
because each batch contains cloned metadata/stat arrays plus result arrays.

**Potential direction**

- Build and write each `RecordBatch` in sequence once schema and output path are
  known.
- Keep only the commit metadata needed for the manifest rather than the full
  batch list.
- Preserve the current cache for constant arrays such as `"ADD"` and all-null
  `extra`, because that is already a useful optimization.

**Validation**

Benchmark peak RSS and write throughput for different `chunks_per_arrow_file`
values. Include Arrow IPC and Parquet output modes.

### 6. Finalization can reread and rewrite all output data

**Where**

- `src/output/finalization.rs:86` lists chunk files for finalization.
- `src/output/finalization.rs:114` through `src/output/finalization.rs:139`
  reads every chunk batch, projects it, and writes a final Parquet file.
- `src/output/finalization.rs:197` through `src/output/finalization.rs:226`
  reads chunk files from Arrow or Parquet.

**Observation**

When final Parquet output is requested, finalization performs a full second pass
over chunk files. For Parquet chunk files, this can become Parquet read followed
by Parquet write. The code supports skipping finalization, but the default final
output path still incurs the extra I/O and serialization pass.

**Potential direction**

- Write directly to the final Parquet layout when possible.
- If chunk files are required for recovery, make finalization optional for
  workflows that can consume chunk manifests.
- Evaluate whether chunk files can be concatenated or row-group-appended without
  rebuilding every `RecordBatch`, subject to schema and metadata constraints.

**Validation**

Measure finalization separately from compute and chunk writing. Include final
file sizes, wall time, and temporary disk usage.

### 7. Row-major BGEN decode uses an intermediate tile and an extra copy pass

**Where**

- `src/genotype/bgen/decode.rs:299` allocates/resizes the per-thread
  row-major dosage tile.
- `src/genotype/bgen/decode.rs:315` decodes variants into that scratch tile.
- `src/genotype/bgen/decode.rs:351` copies selected samples from the scratch
  tile into the caller's row-major output matrix.
- `src/genotype/bgen/reader.rs:724` through `src/genotype/bgen/reader.rs:780`
  uses this row-major tile path.

**Observation**

The row-major path decodes into a variant-major-ish scratch tile and then copies
values into the row-major output matrix by selected sample. This is reasonable
for code clarity and selection handling, but it means an additional memory pass.

The variant-major path writes each decoded row directly to output and computes
summaries in the same pass, so it avoids this transposition-style copy.

**Potential direction**

- Prefer variant-major APIs for high-throughput workflows where possible.
- If row-major remains important, consider a specialized decode path that writes
  directly into row-major sample rows for identity and contiguous selections.
- Consider fusing row-major decode and preprocessing for the common no-missing
  8-bit path.

**Validation**

Benchmark row-major reads separately from variant-major reads for the same BGEN
and selection. Track memory bandwidth and copy volume.

### 8. Generic BGEN probability decoding is scalar and rebuilds bit windows

**Where**

- `src/genotype/bgen/decode.rs:544` and `src/genotype/bgen/decode.rs:999` use
  the generic probability reader for non-specialized paths.
- `src/genotype/bgen/decode.rs:1477` through `src/genotype/bgen/decode.rs:1511`
  implement `PackedProbabilityReader`.
- `src/genotype/bgen/decode.rs:1500` rebuilds a local byte window for every
  probability value.

**Observation**

The generic packed probability reader reconstructs up to an 8-byte window for
each probability, masks it, scales to `f64`, and later casts results to `f32`.
This keeps the code general across bit depths, ploidy, and phased/unphased
layouts, but it is unlikely to be optimal for common bit depths.

**Potential direction**

- Add specialized readers for common bit depths beyond 8-bit, especially where
  probabilities are byte- or nibble-aligned.
- Use `f32` scaling where final output and stats are `f32` and numerical tests
  allow it.
- Maintain a rolling bit buffer instead of rebuilding the byte window for every
  read.

**Validation**

Add fixtures or synthetic benchmarks for non-8-bit BGEN probabilities. Compare
the generic reader to specialized readers before expanding the implementation.

### 9. SIMD coverage is narrow

**Where**

- `src/genotype/bgen/simd.rs:122` dispatches AVX2 for 8-bit identity/contiguous
  all-present dosage decode.
- `src/genotype/bgen/simd.rs:139` dispatches AVX2 for all-samples-present
  checks.
- `src/genotype/bgen/decode.rs:1148` and `src/genotype/bgen/decode.rs:1201`
  use SIMD in the untrusted 8-bit variant-major dosage fast path.
- `src/genotype/bgen/trusted.rs:501` and `src/genotype/bgen/trusted.rs:512`
  use SIMD in the trusted 8-bit dosage fast path.

**Observation**

Current SIMD acceleration is focused on the most important 8-bit unphased
all-present dosage case. Several nearby paths remain scalar:

- non-contiguous sample selection,
- packed probability-pair summary,
- row-major output,
- generic bit-depth decoding,
- missingness/imputation paths,
- non-x86 targets.

The current shape is sensible as an initial optimization, but the same raw
integer summary approach could likely be reused in more places.

**Potential direction**

- Add a packed-pair SIMD summary routine first, because it is close to the
  existing AVX2 dosage path and avoids a known second pass.
- For non-contiguous selections, test whether gather-style approaches or
  selection compaction pay off for realistic selection sizes.
- Consider AVX-512 only if target hardware is known and dispatch complexity is
  justified.
- Consider NEON only if ARM servers or developer laptops are a real target.

**Validation**

Add benchmarks that isolate identity, contiguous, and non-contiguous sample
selection. Record CPU model and enabled target features with each result.

### 10. Profiling counters do work even when global profiling is disabled

**Where**

- `src/genotype/bgen/decode.rs:334` through `src/genotype/bgen/decode.rs:347`
  accumulate row-major local profile fields per variant.
- `src/genotype/bgen/decode.rs:421` through `src/genotype/bgen/decode.rs:434`
  accumulate variant-major local profile fields per variant.
- `src/genotype/bgen/decode.rs:1377` increments variant decode counts
  unconditionally.
- `src/genotype/bgen/decode.rs:1404` and `src/genotype/bgen/decode.rs:1432`
  update byte counters during probability-block reads.

**Observation**

Global profile merging is gated, but local profile snapshots still receive many
counter updates even when profiling is disabled. The overhead is probably small
relative to decompression and decode, but it is paid in tight per-variant loops.

**Potential direction**

- Add a fast path that skips local profile accounting when profiling is disabled.
- Keep timing and byte counters only inside profiling-enabled branches, unless
  the values are used for non-profiling behavior.

**Validation**

Run decode benchmarks with profiling disabled and compare a branch that removes
local profile accounting. Only keep this change if it is measurable.

### 11. Setup paths allocate per record and can reread the same prediction files

**Where**

- `src/sample.rs:659` through `src/sample.rs:680` allocate selected record
  values and sample keys.
- `src/sample.rs:747` through `src/sample.rs:760` clone sample keys for
  phenotype alignment.
- `src/sample.rs:1027` through `src/sample.rs:1051` store grouped alignment
  keys as cloned `Vec<usize>` values.
- `src/regenie.rs:174` through `src/regenie.rs:234` load prediction sources by
  group and phenotype.
- `src/regenie.rs:310` through `src/regenie.rs:356` parse LOCO files with
  per-line allocation.

**Observation**

These are setup paths rather than the main decode/output loops, but they can
matter for many phenotypes, many groups, or repeated runs. Sample alignment
builds temporary vectors and strings per record. Multi-trait LOCO loading can
parse files repeatedly across grouped phenotype entries.

**Potential direction**

- Reuse temporary buffers while reading tabular sample files.
- Intern or borrow sample keys where lifetimes allow it.
- Cache parsed LOCO files by path and phenotype/group key during multi-trait
  loading.

**Validation**

Only prioritize this after measuring setup time on realistic phenotype and LOCO
inputs. The payoff is likely workload-dependent.

## Benchmark Gaps To Fill Before Optimizing

Existing Rust benchmarks cover several BGEN read and preprocessing paths, but
the following cases would make optimization work safer:

- trusted packed probability-pair output,
- identity versus contiguous versus non-contiguous sample selections,
- missingness-heavy BGEN variants,
- non-8-bit packed probabilities,
- output writer memory and throughput for different `chunks_per_arrow_file`
  values,
- finalization time and temporary disk usage,
- Python bridge clone overhead for metadata and stats getters.

Relevant starting points:

- `benches/bgen_read.rs`
- `benches/preprocess.rs`

## Suggested Work Order

1. Add benchmarks for trusted packed-pair output and output writer memory use.
2. Prototype fused copy-plus-summary for trusted packed 8-bit output.
3. Prototype lazy or reused trusted validation.
4. Reduce stats and metadata cloning where benchmarks show allocation pressure.
5. Stream output record batches directly to writers.
6. Revisit broader SIMD work after the packed-pair and output-path changes have
   numbers.

