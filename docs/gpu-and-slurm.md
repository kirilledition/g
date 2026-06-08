# GPU and SLURM

`g` can execute statistical kernels on CPU or GPU through JAX:

```bash
--g-device cpu
--g-device gpu
```

GPU acceleration is workload-dependent. Single-trait runs can be limited by BGEN decode, host-device transfer, or output writing rather than JAX compute.

## Local GPU Checks

Bootstrap and probe the JAX runtime:

```bash
just bootstrap-gpu
just doctor-jax
```

If JAX cannot see the expected accelerator, fix the environment before measuring `g` performance.

## Server Rules

On the gauss server, do not run heavy computation, large test suites, or GPU workloads on the login node. Use SLURM recipes for GPU work. The default GPU node is configured through `GWAS_ENGINE_GPU_NODE` and defaults to `landau`.
CPU-heavy validation and benchmark wrappers use `GWAS_ENGINE_CPU_NODE`, which
defaults to `cantor`. CPU helpers request one task on one node, use
`--exclusive` by default, and derive `CARGO_BUILD_JOBS` and pytest worker counts
from the allocation.

Useful recipes:

```bash
just slurm-gpu-shell
just slurm-gpu-run 'uv run --no-sync python scripts/probe_jax_runtime.py'
just slurm-gpu-just regenie2-binary-gpu-smoke
just slurm-cpu-check
just slurm-cpu-test
just slurm-cpu-rust-build
just slurm-cpu-just benchmark-bgen-reader
```

## Performance Harness

These commands are the stable entrypoints for optimization baseline evidence:

```bash
just perf-smoke
just perf-compare BASE.json NEW.json
```

`perf-smoke` and `perf-compare` are safe on the login node. `perf-smoke` writes a
small JSON summary under `results/perf/smoke/`.

```bash
just perf-cpu
just perf-gpu
```

`perf-cpu` and `perf-gpu` require SLURM. The CPU wrapper submits the BGEN reader
benchmark through `slurm-cpu-just`; the GPU wrapper reuses
`slurm-benchmark-regenie2-binary-hot-gpu` on `landau`. Both write under the
gitignored `results/perf/` tree by default.

Binary GPU examples:

```bash
just setup-regenie2-binary-gpu-inputs
just verify-regenie2-binary-gpu-inputs
just slurm-regenie2-binary-gpu-smoke
just slurm-regenie2-binary-gpu
```

## Performance Notes

Important runtime knobs include:

| Option | Purpose |
| --- | --- |
| `--bsize` | Variants per chunk. |
| `--g-device` | JAX execution target. |
| `--g-staging-depth` | Native callback staging depth. |
| `--g-trusted-no-missing-diploid` | Enables trusted BGEN fast path after validation policy. |
| `--g-bgen-decode-tile-variant-count` | Native BGEN decode tile size. |
| `--g-writer-threads` | Output writer worker count. |
| `--g-writer-queue-depth` | Output writer queue depth. |
| `--g-firth-batch-size` | Binary approximate-Firth batch size. |
| `--g-jax-persistent-cache` | Enable JAX persistent compilation cache. |

Fair performance comparisons require equivalent statistical modes. Compare score-only to score-only, and compare approximate Firth only when both tools use approximate Firth with the same fallback threshold.
