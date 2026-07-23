# Performance Guide

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; guidance, not a benchmark guarantee | main branch as of 2026-07-01 CPU and GPU Step 2 runs | Public user docs |

Performance depends on genotype format, trait mode, phenotype count, BGEN
decode cost, host-device transfer, JAX compilation, Parquet writing, storage, and
cluster placement. Treat numbers from one machine as local evidence, not a
portable guarantee.

## What To Check First

1. Confirm you are running on the intended node.
2. Confirm JAX sees the intended device:

   ```bash
   uv run python -c "import jax; print(jax.devices())"
   ```

3. Confirm the statistical modes are comparable. Compare score-only to
   score-only, and approximate Firth to approximate Firth with the same
   threshold.
4. Check whether the run is dominated by BGEN decode, host-device transfer,
   JAX compute, or output writing before changing knobs.
5. Use telemetry/profile output from the same command shape you intend to tune.

## Common Bottlenecks

| Symptom | Likely first checks |
| --- | --- |
| GPU visible but little speedup | Single phenotype, small chunks, BGEN decode, host-device transfer, or output may dominate. |
| First run slow but repeated work faster | JAX backend initialization and compilation are likely visible. |
| Approximate Firth much slower than binary score test | Candidate density from `--pThresh` and Firth solver work dominate. |
| Output stage slow | Storage throughput and writer threads. |
| Resume startup slow | Manifest validation and strict chunk reconciliation. |

Binary score-only runs prepare only the null-logistic and score state. They do
not pay the chromosome-level null-Firth fit, full-null deviance, or Firth-state
memory cost; those operations are exclusive to approximate-Firth runs.

## Cold, Warm, And Hot Runs

Do not compare timing modes as if they measured the same thing:

| Mode | Includes |
| --- | --- |
| Cold process | Python process startup, JAX backend initialization, first compilation, BGEN decode, compute, and output. |
| Warm JAX cache | New process startup plus cache lookup/reuse when persistent cache entries match. |
| Hot same process | Already-live Python process, JAX backend, and compiled functions for compatible shapes. |

Use cold-process timing for batch-job wall-clock expectations. Use warm-cache
or hot same-process timing only when that is the workflow being measured.

For several compatible production scans, use the native batch command to make
the hot same-process path explicit:

```bash
uv run --no-sync g batch \
  --config chromosome_21.toml \
  --config chromosome_22.toml
```

Batch mode constructs every frontend config and checks output-root and
process-policy compatibility before starting. Run-owned input, output, and
resume preflight remains per entry. The process reuses only process-global
JAX/CUDA state and compiled executables. Shape changes may compile an additional
executable, so group configs with stable shapes when throughput is the priority.

Native decode buffers submitted to JAX transfer their allocation into NumPy;
there is no full genotype memcpy at the binding boundary. Phenotype,
covariate, and single-use LOCO matrices also transfer their allocations
directly into NumPy. Native input indexes each selected LOCO file once using
byte offsets and line numbers. Files with identical headers share one
loader-side identifier index and one alignment recipe per group; those large
header strings are discarded after source construction. Resume planning
validates only chromosome blocks that still need output. When execution
reaches one of those blocks, input reads, parses, finite-validates, and aligns
just that chromosome into its final trait-major matrix. File metadata snapshots
and raw-row SHA-256 digests reject changes between indexing and deferred
reading. Fully committed chromosomes therefore never allocate or parse
prediction values. A repeated noncontiguous chromosome block alone keeps one
matrix for safe clones until its final planned use.

For each complete JAX input shape, dosage and packed8 delivery keep the variant
dimension fixed across chunks. A short chromosome-final chunk is padded only
after exact BGEN decode to `min(bsize, input variant count)`; metadata,
statistics written to output, and result rows retain the logical variant count.
Chunk grids restart at each chromosome, avoiding the extra boundary fragment
created by a whole-file grid. This trades bounded tail transfer and compute for
reuse of that shape's executable. Different aligned sample counts or trait
counts produce different complete JAX shapes and may compile separate
executables.

The native association scheduler starts one compute thread, one host-result
materialization thread, and one bounded channel set for each direct delivery.
The compute worker prepares, uses, and releases each chromosome's JAX state,
including explicit release after a caught worker failure. At a chromosome
boundary, all results drain and state destruction completes before its
successor is built, avoiding both per-chromosome worker churn and overlapping
chromosome-state device memory. Host-to-device genotype transfer remains on
the delivery thread and can overlap the preceding compute batch. Group-level
device state is created at first use and released after its final chromosome
completes. Fully resumed phenotype groups initialize progress but do not select
BGEN samples, prepare JAX state, or start scheduler workers; every remaining
group uses the same direct delivery path.

## Runtime Knobs

Use the current packaged defaults first. Override only with measurements.

| Option | Main effect |
| --- | --- |
| `--bsize` | Variants per chunk; affects memory, JAX shapes, compilation, and per-chunk overhead. |
| `[compute] device` | JAX execution target, `cpu` or `gpu`. |
| `[compute] cpu_threads` | Native Rayon worker count for Rust-owned work. |
| `[compute] multi_phenotype_sample_mode` | Per-phenotype or shared complete-case sample alignment. |
| `[output] writer_threads` | Output writer worker count. |
| `[compute] firth_batch_size` | Approximate-Firth candidate batch size. |
| `[compute] firth_candidate_capacity` | Per-trait scaling value for the aggregate hard approximate-Firth capacity; larger values increase the static executable shape. |
| `[compute] jax_cache_dir` | Optional location override for the always-enabled persistent JAX compilation cache. |
| `[diagnostics] telemetry` | `off`, `progress`, or `profile`; profile mode can perturb timing. |

The BGEN decode tile and GPU genotype representation are internal runtime
policies, not configuration keys. The current reader uses 32-variant decode
tiles. Linear and binary GPU runs use packed8 delivery when no-missing diploid
compatibility validation passes, including multi-phenotype groups. CPU and
otherwise-supported biallelic diploid Layout-2 inputs that are not
packed8-compatible use dosage delivery. Multiallelic, non-diploid,
or otherwise unsupported input fails instead.

Approximate Firth uses an aggregate hard static JAX capacity of the configured
value, capped by the static compute chunk width, multiplied by the trait count.
The packaged per-trait scaling value is 1,024. A batch with more aggregate
candidates is not truncated: it finishes the normal device-to-host result
synchronization and then fails with an instruction to increase
`[compute] firth_candidate_capacity`.

For compatible zlib-compressed packed8 input, GPU runs transfer aligned raw
DEFLATE members and decode them on the active device with nvCOMP. Selection,
exact integer summaries, and packed probability-pair construction stay on the
device; association kernels consume that representation without first
materializing a host dosage matrix. CPU runs, non-zlib BGEN encodings, and
inputs requiring dosage retain the native host decoder. The nvCOMP library and
private JAX FFI target are initialized lazily only when a compressed packed8
group is prepared, so those fallback paths do not pay its initialization cost.

Current default values are in `crates/interface/src/config.default.toml`.
Older pre-release configurations may contain
`firth_newton_raphson_zero_start_iterations`. Remove that key when upgrading;
the production scalar solver always starts at zero and the separate setting was
unreachable, so it is no longer accepted.
The V100-tuned binary path uses 512-lane Firth batches and a
1,024-candidate-per-trait middle dispatch tier. Larger candidate sets retain
the full-chunk overflow path.
Larger batches can reduce some separate-process cache-load overhead while
increasing device execution time, so compare complete workflows rather than
choosing from compile or kernel timing alone.

Project profiling recipes isolate CPU JAX caches by host and CPU feature
fingerprint under `/tmp/g-jax-cpu-profile-cache` by default. This avoids reusing
CPU AOT artifacts across SLURM nodes when profile outputs are stored on shared
filesystems. GPU profile caches remain node-local under `/tmp/g-jax-profile-cache`
or `/tmp/g-jax-binary-hot-cache` unless `G_PROFILE_GPU_JAX_CACHE_PARENT` is
overridden.

## CPU Runs

CPU runs exercise native BGEN decode, sample alignment, output writing, and JAX
CPU kernels. For large scans on a cluster, submit to a compute node instead of a
login node and set `[compute].cpu_threads` to the scheduler CPU count.

```bash
uv run --no-sync g regenie \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_cpu_regenie2
```

## GPU Runs

GPU runs require a CUDA-capable JAX environment and a GPU allocation. Single
phenotype scans can still be limited by BGEN decode, transfer, and output rather
than accelerator compute.

Useful GPU checks:

```bash
uv run python -c "import jax; print(jax.devices())"
uv run --no-sync g regenie --config gpu.toml ...
```

Multi-phenotype quantitative runs can amortize BGEN decode and process startup
when sample semantics allow it. Use `per-phenotype` for separate-run semantics
and `complete-case` only when the shared complete-case intersection is intended.

As a practical rule of thumb:

- `per-phenotype` maximizes per-trait usable samples and is the conservative
  choice when missingness differs by phenotype.
- `complete-case` tends to reduce repeated alignment work because one shared mask
  is used across traits, but it can reduce sample size for each trait and change
  statistics when missingness differs.

If speed is the only concern, test both modes on a small representative subset
before changing production scripts.

## Parquet Output

Each phenotype run writes a `parts/` Parquet dataset. `writer_threads` is the
public filesystem-throughput control. Queue depth, part grouping, and
compression are internal output policies.

## Measuring

Production mode:

```toml
[diagnostics]
telemetry = "progress"
```

Profile mode:

```toml
[diagnostics]
telemetry = "profile"
```

Development benchmark protocols and repository-specific SLURM recipes live in
[Benchmarking](../development/benchmarking.md) and
[Server Gauss SLURM](../development/server-gauss-slurm.md).
