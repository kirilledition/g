# Benchmarking

| Status | Applies to | Owner |
| --- | --- | --- |
| Development protocol | Benchmark recipes, profiling campaigns, performance comparisons, and artifact handling | Performance maintainers |

This page defines development benchmark categories and evidence expectations.
Public tuning guidance lives in [Performance Guide](../public/performance-guide.md).

## Benchmark Categories

| Category | Purpose | Typical entry points |
| --- | --- | --- |
| Smoke | Verify a benchmark harness and output schema quickly. | `just perf-smoke`, smoke variants of benchmark recipes. |
| BGEN reader | Isolate native decode, sample selection, trusted paths, and Rayon effects. | `tooling.cli.benchmark_bgen_reader`, `just benchmark-bgen-reader`. |
| Output stages | Isolate writer threads, queue depth, compression, grouping, and finalization. | `tooling.cli.benchmark_output_stages`, `just benchmark-output-stages-*`. |
| Binary hot path | Measure binary Step 2 score/Firth runtime without full campaign overhead. | `tooling.cli.benchmark_regenie2_binary_hot`. |
| Matrix comparisons | Compare CPU/GPU/cache combinations for standard workloads. | `tooling.cli.run_regenie2_matrix`. |
| Deep profiling | Run multi-tool profiling campaigns with JAX and native evidence. | `tooling.cli.profile_regenie2_deep`. |
| External comparison | Compare `g` with upstream or patched REGENIE under equivalent modes. | `-m tooling.cli.benchmark tool.name=regenie_comparison`. |

See [Tooling](tooling.md) and [Justfile Command Reference](justfile.md) for
the current command surface.

## Evidence Requirements

Every benchmark result should record:

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
