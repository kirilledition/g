# Performance Guide

| Status | Applies to | Owner |
| --- | --- | --- |
| Public performance guidance, not a benchmark guarantee | CPU and GPU `g regenie` Step 2 runs | Public interface |

Performance depends on genotype format, trait mode, phenotype count, BGEN
decode cost, host-device transfer, JAX compilation, output format, storage, and
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
| Output stage slow | Storage throughput, writer threads, queue depth, output format, compression, and finalization. |
| Resume startup slow | Manifest validation mode and strict chunk reconciliation. |

## Runtime Knobs

Use the current packaged defaults first. Override only with measurements.

| Option | Main effect |
| --- | --- |
| `--bsize` | Variants per chunk; affects memory, JAX shapes, compilation, and per-chunk overhead. |
| `--threads` | Native Rayon thread request for Rust-owned work. |
| `--g-device` | JAX execution target, `cpu` or `gpu`. |
| `--g-staging-depth` | Native callback staging depth. |
| `--g-result-in-flight-limit` | Optional cap for result chunks awaiting materialization. |
| `--g-dosage-buffer-limit` | Optional cap for reusable native dosage decode buffers. |
| `--g-bgen-decode-tile-variant-count` | Native BGEN decode tile size. |
| `--g-gpu-genotype-format` | Host-to-device genotype representation for GPU-compatible paths. |
| `--g-output-format` | Arrow, Parquet, or REGENIE text materialization. |
| `--g-writer-threads` | Output writer worker count. |
| `--g-writer-queue-depth` | Output writer queue depth. |
| `--g-output-chunks-per-arrow-file` | Number of engine chunks grouped into each Arrow/Parquet/text part. |
| `--g-firth-batch-size` | Approximate-Firth candidate batch size. |
| `--g-firth-candidate-capacity` | Candidate capacity for binary fallback staging. |
| `--g-jax-persistent-cache` and `--g-jax-cache-dir` | JAX compilation cache behavior. |
| `--g-telemetry` | Progress, profile, and trace modes. Profile/trace can perturb timing. |

Current default values are in `src/g/config.default.toml`.

## CPU Runs

CPU runs exercise native BGEN decode, sample alignment, output writing, and JAX
CPU kernels. For large scans on a cluster, submit to a compute node instead of a
login node and pass the scheduler CPU count to `--threads`.

```bash
uv run --no-sync g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_cpu_regenie2 \
  --threads "${SLURM_CPUS_PER_TASK:-16}" \
  --g-device cpu
```

## GPU Runs

GPU runs require a CUDA-capable JAX environment and a GPU allocation. Single
phenotype scans can still be limited by BGEN decode, transfer, and output rather
than accelerator compute.

Useful GPU checks:

```bash
uv run python -c "import jax; print(jax.devices())"
uv run --no-sync g regenie --g-device gpu ...
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

## Output Choices

| Format | Use when |
| --- | --- |
| `parquet` | You want dataset parts and efficient downstream analytics. |
| `arrow` | You want Arrow IPC chunks for inspection or intermediate workflows. |
| `regenie` | You need REGENIE Step 2-style text compatibility. |

Finalizing a single `final.parquet` adds work after chunk output. Keep
`--g-finalize-parquet` off when the parts dataset is sufficient.

## Measuring

Production mode:

```bash
--g-telemetry progress
```

Profile mode:

```bash
--g-telemetry profile
--g-log-dir /path/to/logs
```

Trace mode is for small or capped runs:

```bash
--g-telemetry trace
--g-variant-limit 1000
--g-trace-event-cap 1000000
```

Trace can perturb performance and generate high-volume logs. Use it to diagnose
a specific small case, not as a production timing mode.

Development benchmark protocols and repository-specific SLURM recipes live in
[Benchmarking](../development/benchmarking.md) and
[Server Gauss SLURM](../development/server-gauss-slurm.md).
