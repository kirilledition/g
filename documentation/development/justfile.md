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
just dev-install-release
just cuda-format
just cuda-format-check
just cuda-lint
just check-local
just check-artifact-schema data/profiles/example/report.json
just check-rust-architecture
just rust-check
just workspace-check
just test-local
just docs-build
just docs-check
```

`server-setup-tools` runs in a minimal `uv --no-project` environment so it can
install Mold before any Linux build attempts to compile the project itself.

`dev-install` uses Cargo's fast `dev` profile. `dev-install-release` uses the
FatLTO, single-codegen-unit `release` profile used by benchmarks and final
builds. Cargo defaults to 30 build jobs through `.cargo/config.toml`;
`CARGO_BUILD_JOBS` or an explicit `--jobs` option can impose a different cap.
SLURM jobs use the scheduler allocation unless `CARGO_BUILD_JOBS` is already
set.

`cuda-format` applies the repository ClangFormat policy to the explicit list of
maintained `.cu`, `.cc`, and `.h` files, including the shared private CUDA
driver support header. `cuda-format-check` checks the same files without
rewriting them. `cuda-lint` runs the pinned ClangTidy analysis of
CUDA host/device code and native C++ through the saved
`debug_check_cuda_native` configuration. These commands exclude generated PTX
and vendored sources. Static analysis runs on Linux x86-64 without a GPU and
does not compile or regenerate PTX. The aggregate `format`, `lint`, `check`,
and `check-local` recipes include the corresponding native checks.

Use SLURM for CPU-heavy validation:

```bash
just slurm-cpu-check
just slurm-cpu-test
just slurm-cpu-rust-build
just slurm-cpu-rust-test
just slurm-gpu-test-parity-required
```

CPU and GPU correctness run in separate processes. The CPU recipe excludes
data-dependent parity; the GPU recipe qualifies all blocking upstream-REGENIE
workflows with required local fixtures.

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
just rust-bench
just bench-torchgwas-chr22
just bench-tensorqtl-chr22
just bench-rust-build-profiles
just slurm-gpu-bench-binary-hot
just slurm-gpu-bench-torchgwas-chr22
just slurm-gpu-bench-tensorqtl-chr22
```

`rust-bench` runs Criterion benchmarks through `cargo bench --workspace` so
workspace-owned benches are discovered from their owning crates. Cargo's
`bench` profile inherits the maximum-performance `release` profile.
`bench-rust-build-profiles` compares the built-in `dev` and `release` profiles.
The repo Cargo configuration defaults to 30 build jobs and, on Linux, enables
`target-cpu=native` and links through Mold using the `cc` compiler driver.
`doctor` verifies the compiler driver can complete a tiny Mold link. SLURM jobs
isolate native Cargo artifacts under `target/slurm/<node>/`; an explicit
`CARGO_TARGET_DIR` still takes precedence.
`RUSTC_WRAPPER` remains an environment-specific choice.

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
just profile-rust-criterion
just profile-app-full-dry
just profile-app-full-smoke
just profile-app-full
just profile-chr10-binary-gpu-dry
just profile-chr10-binary-gpu-smoke
just profile-chr10-binary-gpu-full
just profile-chr22-binary-gpu-dry
just profile-chr22-binary-gpu-full
```

The full profile recipes submit GPU work through SLURM. Native Criterion
profiles use the configured CPU compute node. The focused chr10 and chr22
binary GPU recipes keep their profiler selection and campaign budgets in saved
Hydra configs rather than in the Justfile.

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
for the Rust migration. `just check-python-architecture` verifies Python
package ownership boundaries for compute kernels and JAX runtime helpers.
`just check-justfile` verifies that the command surface stays config-backed and
that maintained docs do not reference removed recipe names. These guardrails
are included in `just check` and `just check-local`; `just rust-check` and
`just workspace-check` keep a Rust-focused validation lane for multicrate
migration phases.
