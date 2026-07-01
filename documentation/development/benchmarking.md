# Benchmarking

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development protocol | main branch as of 2026-06-30 benchmark and profiling workflows | Performance maintainers |

This page defines development benchmark categories and evidence expectations.
Public tuning guidance lives in [Performance Guide](../public/performance-guide.md).

## Benchmark Categories

| Category | Purpose | Typical entry points |
| --- | --- | --- |
| Smoke | Verify a benchmark harness and output schema quickly. | `just perf-smoke`, smoke variants of benchmark recipes. |
| BGEN reader | Isolate native decode, sample selection, trusted paths, and Rayon effects. | `tooling.cli.benchmark_bgen_reader`, `just benchmark-bgen-reader`. |
| Callback overhead | Isolate Python callback queue handoff and optional host-to-device transfer without BGEN decode. | `tooling.cli.benchmark_callback_overhead`, `just benchmark-callback-overhead`. |
| Output stages | Isolate writer threads, queue depth, compression, grouping, and finalization. | `tooling.cli.benchmark_output_stages`, `just benchmark-output-stages-*`. |
| Binary hot path | Measure binary Step 2 score/Firth runtime without full campaign overhead. | `tooling.cli.benchmark_regenie2_binary_hot`. |
| Matrix comparisons | Compare CPU/GPU/cache combinations for standard workloads. | `tooling.cli.run_regenie2_matrix`. |
| Deep profiling | Run multi-tool profiling campaigns with JAX and native evidence. | `tooling.cli.profile_regenie2_deep`. |
| External comparison | Compare `g` with upstream or patched REGENIE under equivalent modes. | `-m tooling.cli.benchmark tool.name=regenie_comparison`. |
| Competitor comparison | Compare `g` against a published competing implementation with explicit model caveats. | `tooling.cli.benchmark_torchgwas_chr22`, `just slurm-gpu-bench-torchgwas-chr22`. |

See [Tooling](tooling.md) and [Justfile Command Reference](justfile.md) for
the current command surface.

## Evidence Requirements

Every benchmark result should record:

- schema version;
- command and full overrides;
- commit SHA and branch;
- host or SLURM node;
- date;
- input dataset and workload size;
- trait mode and correction plan;
- device, dtype, cache, and output settings;
- artifact directory;
- summary metric and confidence signal.

Do not report a speedup without naming the baseline command.

For competitor comparisons, state the semantic and input-format boundary in the
artifact. For example, the TorchGWAS chr22 benchmark is a single-trait
quantitative workflow/runtime comparison: `g` runs REGENIE Step 2 with LOCO
predictions on BGEN input, while TorchGWAS runs covariate-adjusted linear GWAS.
Full TorchGWAS runs use the local PLINK triplet because the pinned TorchGWAS
BGEN path stalls while parsing the intermediate PLINK2 raw table at chr22
scale; bounded smoke runs use a generated NPY subset. TorchGWAS PLINK runs do
not emit a persistent genotype cache, so warm cases are repeated-process
measurements with possible filesystem cache effects rather than explicit
genotype-cache reuse.

Durable JSON artifacts used for comparison or migration decisions must include
`schema_version` and should be written through `tooling.common.reports` so
missing, unknown, or incompatible fields fail early.

## Login Node Policy

Login-node-safe:

- repository inspection;
- dry-run planning commands;
- docs builds;
- `just perf-smoke` when configured for tiny deterministic work;
- JSON/Markdown comparison of existing artifacts.

Use SLURM or another suitable compute node for:

- GPU work;
- heavy CPU workloads;
- native performance builds;
- large tests;
- full benchmark sweeps;
- profiler campaigns.

Server-specific routing lives in [Server Gauss SLURM](server-gauss-slurm.md).

## Artifact Policy

Benchmark and profile outputs belong under ignored paths such as:

```text
data/benchmarks/
data/profiles/
results/perf/
```

Never commit large benchmark artifacts, raw traces, generated local data, or
output datasets. Commit only durable docs, scripts, tests, and small fixtures
intended for source control.

## Interpreting Results

Separate these effects before proposing an optimization:

- first-process Python/JAX startup;
- JAX compilation versus steady-state execution;
- BGEN decode and sample alignment;
- host-device transfer;
- statistical kernel compute;
- output writing/finalization;
- telemetry/profile perturbation;
- storage and scheduler placement.

For startup findings, include a same-process or multi-phenotype measurement
before optimizing import/runtime boundaries.

For native BGEN to JAX callback findings, start with the callback-overhead
microbenchmark before running an end-to-end BGEN profile:

```bash
just benchmark-callback-overhead \
  tool.chunk_count=10000 \
  tool.trials=5 \
  'tool.stage_timing_modes=[off,aggregate]' \
  'tool.workload_modes=[queue_only,host_to_device]'
```

Use SLURM for CPU/GPU evidence:

```bash
just slurm-benchmark-callback-overhead-cpu \
  tool.chunk_count=1000 \
  tool.trials=1 \
  'tool.stage_timing_modes=[off,aggregate]' \
  'tool.workload_modes=[queue_only,host_to_device]'

just slurm-benchmark-callback-overhead-gpu \
  tool.chunk_count=1000 \
  tool.trials=1 \
  'tool.stage_timing_modes=[off,aggregate]' \
  'tool.workload_modes=[queue_only,host_to_device]'
```

`tool.stage_timing_modes=[off]` represents the production default when no stage
timing JSON path or forced recorder is configured. `aggregate` and `exact` are
diagnostic modes; they intentionally perturb the hot path by collecting queue,
stage, transfer, and optional blocking observations.

## Performance Discovery

Broad optimization searches must follow [Performance Discovery Playbook](performance-discovery.md):

1. define the target;
2. record a baseline;
3. isolate a suspected bottleneck;
4. propose one bounded change;
5. name validation;
6. state non-goals;
7. rank findings.

Speculation without a baseline is not implementation-ready.
