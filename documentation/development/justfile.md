# Justfile Command Reference

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 Justfile command surface | Development maintainers |

The repository `Justfile` is a thin, stable entrypoint layer. It names common
workflows and delegates workflow truth to saved Hydra configs under
`tooling/configs/`.

Run `just help` for the full recipe list.

## Policy

```text
Environment variables configure the machine.
Hydra configs configure the workflow.
Justfile recipes select the workflow.
```

Environment variables are appropriate for secrets, scheduler-provided state,
local cache/toolchain roots, and explicit one-off path overrides. They must not
be the primary source of truth for benchmark grids, profiler modes, dataset or
chromosome identity, output format, run semantics, or named machine profiles.

Hydra-backed recipes accept trailing overrides for ad hoc work, but routine
workflow behavior belongs in a saved config. Every migrated `tooling.cli.*`
recipe should use `--config-name`.

## Core Development

Use these on the login node when they are lightweight:

```bash
just doctor
just doctor-server
just dev-bootstrap
just dev-bootstrap-gpu
just dev-install
just dev-install-opt
just dev-install-perf
just dev-install-perf-max
just check-local
just check-artifact-schema data/profiles/example/report.json
just check-rust-architecture
just rust-check
just workspace-check
just test-local
just docs-build
just docs-check
```

Use SLURM for CPU-heavy validation:

```bash
just slurm-cpu-check
just slurm-cpu-test
just slurm-cpu-test-full
just slurm-cpu-rust-build
just slurm-cpu-rust-test
```

## Data

```bash
just data-fetch
just data-simulate
just data-prepare
just data-baseline-binary
just data-baseline-qt
just data-verify-binary-gpu-inputs
```

`GWAS_ENGINE_DATA_DIR` may point at a shared data directory, such as the main
checkout's gitignored `data/` directory from a temporary worktree.

## Matrices

```bash
just matrix-chr10-dry
just matrix-chr10-smoke
just slurm-gpu-matrix-chr10
just matrix-chr22-dry
just matrix-chr22-smoke
just slurm-gpu-matrix-chr22
```

The `matrix-*` workflows replace direct hand-written `g regenie` recipes. Use
smoke configs for bounded validation and full matrix configs for parity or
performance runs.

## Benchmarks

```bash
just bench-bgen-reader
just bench-callback-overhead
just bench-callback-overhead-gpu
just bench-linear-startup-gpu
just bench-linear-startup-gpu-parquet
just bench-binary-hot-gpu
just bench-binary-hot-gpu-smoke
just bench-output-stages-gpu
just bench-torchgwas-chr22
just bench-tensorqtl-chr22
just bench-rust-build-profiles
just slurm-gpu-bench-binary-hot
just slurm-gpu-bench-torchgwas-chr22
just slurm-gpu-bench-tensorqtl-chr22
```

`bench-rust-build-profiles` uses the repo Cargo configuration by default, so
Linux Rust builds enable `target-cpu=native` without per-recipe `RUSTFLAGS`.
Linker and rustc-wrapper choices stay outside the repo and should be supplied
through environment variables when needed.

Historical external baseline comparisons remain available under `legacy-*`:

```bash
just legacy-baselines
just legacy-baselines-full
just legacy-regenie-comparison-cpu
just legacy-regenie-comparison-gpu
just legacy-profile-regenie-comparison-cpu
just legacy-profile-regenie-comparison-gpu
```

Use the TorchGWAS chr22 competitor benchmark through SLURM for real GPU
evidence:

```bash
just slurm-gpu-bench-torchgwas-chr22
just slurm-gpu-bench-torchgwas-chr22 tool.variant_limit=1000
```

The benchmark records cold and warm-style repeated runs for single-trait
quantitative chr22. It is a workflow/runtime comparison, not strict statistical
parity with REGENIE Step 2 LOCO output. `g` reads the chr22 BGEN plus sample
file; full TorchGWAS runs read the existing PLINK `.bed/.bim/.fam` triplet
because the pinned TorchGWAS BGEN path stalls in PLINK2 raw-table parsing at
chr22 scale. TorchGWAS PLINK runs do not emit a persistent genotype cache, so
their warm cases reflect repeated process/filesystem-cache behavior.
Variant-limited smoke runs still use a generated NPY subset.

Use the tensorQTL chr22 competitor benchmark through SLURM for real GPU
evidence:

```bash
just slurm-gpu-bench-tensorqtl-chr22
just slurm-gpu-bench-tensorqtl-chr22 tool.variant_limit=1000
```

The benchmark records cold and warm-style repeated runs for single-trait
quantitative chr22, but the comparison boundary is QTL-shaped: `g` runs REGENIE
Step 2 with LOCO predictions on BGEN input, while tensorQTL runs dense `trans`
nominal linear association on generated phenotype/covariate matrices and local
PLINK `.bed/.bim/.fam` input. Full runs expose the PLINK files through a
BED-only symlink prefix under the benchmark output directory because tensorQTL
auto-selects PGEN when PGEN and BED files share the same prefix, and its PGEN
reader fails on the local chr22 multiallelic records. Do not treat the result as
statistical parity.

## Performance

```bash
just perf-smoke
just perf-cpu
just perf-gpu
just perf-compare results/perf/baseline.json results/perf/new.json
just perf-jax-runtime
just perf-tune-regenie2-gpu
```

`perf-smoke` and `perf-compare` are login-node-safe. `perf-cpu` and `perf-gpu`
submit through SLURM and use saved benchmark configs for output locations.

## Profiling

```bash
just profile-deep-dry
just profile-deep-smoke
just profile-app-full-dry
just profile-app-full-smoke
just profile-app-full
just profile-chr10-binary-gpu-dry
just profile-chr10-binary-gpu-smoke
just profile-chr10-binary-gpu-full
```

The full profile recipes submit GPU work through SLURM. The chr10 binary GPU
full profile uses `profile_chr10_binary_gpu_full.yaml`, which contains the
previous profiler selection and campaign budget settings that used to live in
the Justfile.

## SLURM Substrates

```bash
just slurm-cpu-shell
just slurm-cpu-run 'cargo build --workspace --all-targets'
just slurm-cpu-just check
just slurm-gpu-shell
just slurm-gpu-run 'nvidia-smi'
just slurm-gpu-just bench-binary-hot-gpu-smoke
```

The SLURM wrappers are execution substrates. Workflow choices should still live
in saved configs or named Just recipes.

## Guardrails

`just check-rust-architecture` verifies Cargo workspace dependency boundaries
for the Rust migration. `just check-justfile` verifies that the command surface
stays config-backed and that maintained docs do not reference removed recipe
names. Both are included in `just check` and `just check-local`; `just
rust-check` and `just workspace-check` keep a Rust-focused validation lane for
multicrate migration phases.
