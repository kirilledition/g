# GPU And Clusters

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-24 GPU and cluster operation | Public user docs |

`g` executes statistical kernels through JAX. Choose the target device in TOML:

```toml
[compute]
device = "gpu" # or "cpu"
```

CPU support is installed by the base runtime dependencies. GPU support requires the
[GPU installation flow](installation.md#gpu-install-from-source) and a scheduler allocation on a GPU
node.

GPU acceleration is workload-dependent. Single-trait runs can be limited by BGEN decode, host-device
transfer, or output writing rather than JAX compute.

## Cold, Warm, And Steady-State Runs

The first process on a node may pay for Python import, JAX backend
initialization, and JAX compilation. A later run with the same shapes and cache
policy can be faster. Treat these as different measurements:

| Run class | What it measures |
| --- | --- |
| Cold process | Startup, JAX initialization, compilation, decode, compute, and output. |
| Warm cache | Reused compilation artifacts when always-enabled persistent-cache entries match. |
| Steady state | Chunk decode, transfer, compute, and writer throughput after startup effects. |

The persistent cache is always enabled and defaults to
`<platform temporary directory>/<user>/g-jax-cache`.
`[compute].jax_cache_dir` overrides that location. Put overrides on local or
fast user-writable storage, and do not share CPU cache artifacts across nodes
with different CPU features.

## When GPU May Not Help

GPU execution is not automatically faster. CPU can match or beat GPU for small
or I/O-bound runs when:

- the scan has one phenotype and few variants;
- BGEN decode or sample alignment dominates;
- host-to-device transfer dominates compute;
- Parquet writing dominates runtime;
- approximate-Firth candidate density is low enough that GPU work is sparse;
- the command repeatedly changes shapes and recompiles.

## What To Check First

Run these checks inside the same environment and allocation where the scan will
run:

```bash
hostname
uv run python -c "import jax; print(jax.devices())"
uv run g regenie --help
```

If JAX does not list the expected GPU, fix the node allocation, driver, CUDA, or
JAX install before tuning `g`.

## Probe the JAX Runtime

Run this check in the same environment and on the same kind of node where the scan will run:

```bash
uv run python -c "import jax; print(jax.devices())"
```

If JAX cannot see the expected accelerator, fix the driver, CUDA, or JAX wheel environment before
measuring `g` performance. On a login node without NVIDIA devices, a CPU-only result can be
expected even when the GPU environment is otherwise valid.

## Generic SLURM GPU Job

Install and sync the checkout before submitting the job. Inside the batch script, use
`uv run --no-sync` so the job uses the already-created `.venv/` instead of trying to change the
environment while the scan is running:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=g-regenie2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd /path/to/g
export UV_CACHE_DIR="${SCRATCH:-$HOME}/.cache/uv"
export UV_LINK_MODE=copy

uv run --no-sync g regenie \
  --config /path/to/gpu.toml \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_gpu_regenie2
```

Adjust `#SBATCH` options for your site's partitions, accounts, GPU resource syntax, and memory
policy.

## Generic SLURM CPU Job

Large CPU scans should also run on a compute node rather than a login node:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=g-regenie2-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd /path/to/g
export UV_CACHE_DIR="${SCRATCH:-$HOME}/.cache/uv"
export UV_LINK_MODE=copy

uv run --no-sync g regenie \
  --config /path/to/cpu.toml \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_cpu_regenie2
```

## Cluster Notes

- Do not run GPU scans or large CPU scans on a login node.
- Keep `UV_CACHE_DIR` and run output on user-writable storage with enough quota.
- Use `UV_LINK_MODE=copy` on shared filesystems where hardlinks or reflinks are unreliable.
- Run `uv sync` before submitting production jobs, then use `uv run --no-sync` inside batch jobs.
- Use upstream `regenie` to create Step 1 predictions before running `g` Step 2.

The gauss development-server recipes and benchmark wrappers are documented in
[Server Gauss SLURM](../development/server-gauss-slurm.md).

## Runtime Knobs

Important runtime knobs include:

| Setting | Purpose |
| --- | --- |
| `--bsize` / `[trait].bsize` | Variants per chunk. |
| `[compute].device` | JAX execution target. |
| `[compute].cpu_threads` | Native Rayon worker count. |
| `[compute].multi_phenotype_sample_mode` | Per-phenotype or shared complete-case sample alignment. |
| `[output].writer_threads` | Output writer worker count. |
| `[compute].firth_batch_size` | Binary approximate-Firth batch size. |
| `[compute].jax_cache_dir` | Override for the always-enabled persistent JAX compilation cache directory. |

Scheduler queue depths, packed8 BGEN compatibility validation, decode tiling,
packed8 selection, and Parquet grouping/compression are internal
genotype/engine/output policies.

Compatible zlib-compressed packed8 BGEN groups use the installed nvCOMP runtime
path automatically. The runtime is loaded only for a group that can use this
path; CPU runs and GPU groups using host dosage or a non-zlib encoding do not
initialize it. A source installation must therefore preserve the locked
`nvidia-libnvcomp-cu12` 5.3 dependency for Linux x86-64 GPU execution. This path
requires an NVIDIA driver exposing CUDA driver API 12.2 or newer and a device
with compute capability 7.0 or newer. An eligible zlib packed8 run fails with a
specific initialization error when those requirements or nvCOMP are missing;
it does not silently replace the requested device decode with host decode.
Descriptor bounds or alignment status `0x00000800` is terminal and is never
retried. The error identifies the first affected variant and includes status,
logical and compute geometry, and slab size. The hot transfer path does not
scan genotype or metadata contents solely to construct failure diagnostics.

The supported production runtime is exactly `jax==0.11.0` with
`jaxlib==0.11.0`, and its GPU install deliberately uses the CUDA 12 extra on
gauss/landau. The node's V100 GPUs expose compute capability 7.0 through an
R535 driver; JAX's CUDA 13 wheels require compute capability 7.5 and an
R580-or-newer driver. Upgrade the cluster hardware and driver before changing
the project to the CUDA 13 extra.

Binary approximate-Firth GPU runs also use a private raw-CUDA component kernel
when the CUDA driver exposes API 12.2 or newer and the device has compute
capability 7.0 or newer. This compute path is independent from nvCOMP and does
not add a user-facing switch. JAX is retained only for a typed recoverable
capability result: unsupported platform, unavailable driver library, unavailable
required symbol, driver too old, unavailable selected device, or unsupported
compute capability. CUDA driver-operation failures, unexpected native
initialization failures, and JAX FFI capsule/import/registration failures stop
the run before output activation.
For the selected device lookup, only CUDA's `INVALID_DEVICE` result means that
the configured ordinal is unavailable and permits the typed JAX fallback.
Invalid arguments, uninitialized or deinitialized driver state, invalid
contexts, deferred errors, and unknown statuses are fatal driver failures.
An unexpected CUDA module or launch failure after that selection is reported
as an execution error; `g` does not change implementations inside a compiled
solver lifecycle. Run provenance records exact JAX/JAXlib versions, requested
and effective component implementations, a typed reason when JAX fallback is
selected, and the raw artifact's FFI target/API plus framed source/ABI handler
SHA-256, PTX SHA-256/ISA/target, and reviewed minimum CUDA driver and
compute-capability thresholds whenever raw CUDA was requested. Free-text detail
and observed
driver/device/compute capability are diagnostic only and are not part of resume
compatibility.

Fair performance comparisons require equivalent statistical modes. Compare score-only to score-only,
and compare approximate Firth only when both tools use approximate Firth with the same fallback
threshold.

For broader tuning and measurement guidance, see [Performance Guide](performance-guide.md).
