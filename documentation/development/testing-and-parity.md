# Testing And Parity

| Status | Applies to | Owner |
| --- | --- | --- |
| Exact-head foundation | Main branch as of 2026-07-23 | Correctness maintainers |

This page separates fast mathematical checks from full external REGENIE
comparisons. A comparison with an earlier `g` build is useful for locating a
regression, but it is not the primary correctness oracle: two `g` versions can
share the same defect.

## Test Tiers

| Tier | Command | Execution policy |
| --- | --- | --- |
| External-contract harness | `just test-local-focused` | Login-node safe; uses metadata, temporary Git repositories, synthetic Slurm observations, and tiny in-memory frames only. |
| Active non-data Python suite | `just test-local` | Run on an appropriate CPU allocation when JAX compilation would be material. |
| Optional external parity | `just test-parity` | Full chr22 GPU work when fixtures exist; missing local fixtures skip. |
| Exact-head required-fixture parity | [Trusted scheduler launch](regenie-parity-suite.md#trusted-scheduler-launch) | An extracted, hash-bound bootstrap runs one serialized `landau` allocation; missing fixtures fail. |
| Full repository suite | `just test-full` | GPU allocation only; CPU and parity tests run in separate processes. |

Do not run GPU workloads, heavy compilation, large suites, or benchmark sweeps
on a login node. GitHub-hosted CI runs the active non-data tests and the
login-safe parity harness. The harness rejects malformed typed evidence and a
required workflow whose status/marker disappears, checked-in self-qualification
evidence, a stale source SHA/fingerprint, a disallowed qualification host, or
non-CUDA device evidence. It does not claim to run the protected full chr22
fixture.

## Correctness Oracle

Use an independently generated, version-pinned upstream REGENIE output as the
primary oracle whenever the project claims REGENIE compatibility. The current
goldens use upstream REGENIE v4.1 and are recorded in
`tests/parity/golden_metadata.json` with their commands, row counts, and
SHA-256 digests. Quantitative, binary score-only, and binary approximate Firth
all retain their upstream oracles and required comparison contracts.
Historical reports do not qualify current production source; the checked-in
workflows remain `required` with null evidence until an exact-head bundle is
attached by a trusted external status publisher.

The comparison contract is:

- align all rows by `(CHROM, GENPOS, ID, ALLELE0, ALLELE1)`; `ID` alone is not
  unique in the full chr22 fixture;
- require both inputs and the one-to-one join to contain all 418,943 rows;
- for each finite statistic, require `abs(g_value - regenie_value) < tolerance`;
- require identical `NaN`, positive-infinity, and negative-infinity masks;
- require exact `N` and exact significance decisions at `p < 0.05` and
  `p < 5e-8`;
- verify every BGEN, sample, phenotype, covariate, Step-1 prediction list and
  referenced `.loco` file, REGENIE output, and REGENIE log hash before
  execution and again after the long run and protected reads;
- parse correction and correction-failure counts from the pinned upstream log
  and require the production aggregate to match it.

The hashes identify the external oracle artifacts. They are not a request for
byte-for-byte equality between REGENIE text and `g` Parquet output.

Binary approximate-Firth labels need a narrower statement. Upstream REGENIE's
saved table does not expose a per-row successful-Firth label. The comparison
therefore checks its recorded aggregate correction and failure counts, checks `g`'s
allowed method/status values, and compares every public statistic and
significance decision. It does not claim per-row correction-label parity that
the upstream artifact cannot establish.

## Supported Production Boundary

Full parity invokes only the supported `g._core.cli.run` binding. It requires
exactly one `Parquet dataset saved to <absolute path>` stdout line, resolves
that path inside the requested output root, and reads only its direct Parquet
parts. It does not recursively discover `*.run` directories, restore legacy
Python orchestration, or expect a post-run `final.parquet`.

After the output-transaction change is merged, the parity reader must also
validate the selected path against durable completed
lineage/finalization/manifests and bind the relevant `.g-output` hashes into
qualification evidence. The completion line remains the unambiguous selector;
it is not sufficient transaction authority by itself.

Local missing data are a skip so contributors can run the harness without the
protected fixture. Required scheduled runs set
`G_REGENIE_PARITY_REQUIRE_DATA=1`; every missing BGEN, sample, phenotype,
covariate, prediction, output, or log artifact is then a hard failure.

The trusted scheduler selects a full commit and uses replacement-disabled
`/usr/bin/git` to write that commit's
`tooling/server/exact_parity_bootstrap.sh` blob to a temporary executable. It
computes the bootstrap SHA-256 and launches that exact file on `landau` with
its fixed `/usr/bin/bash` shebang as the direct `srun` command under `env -i`.
Only the required launch home, fixture/report, bootstrap, numeric resource
expectations, and scheduler-selected absolute
`uv`/`just`/direct `cargo`/direct `rustc`/`mold`/Python values, Cargo-cache
snapshot, and Rustup installation enter the launch. The bootstrap replaces the
launch home with a private run home. Inherited loader/compiler overrides,
including every dynamic-loader path override, are rejected.

The bootstrap binds itself and its extracted checkout/Slurm helpers to the
selected commit. The Slurm helper takes two controller snapshots around exact
host-PID/start, local `listpids`, namespace, command-line, and structural
cgroup-v2 observations. Its commit-bound constants require eight CPUs, 64 GiB,
one task, and one GPU, and any GPU field exposed by allocated TRES must agree.
Environment variables are consistency assertions, not the scheduler
authority. The resulting strict canonical schema-0 attestation
records scheduler entitlement as proven and kernel enforcement as unproven on
`abraxas`; strict enforcement mode is unsupported and fails closed. The
bootstrap then creates a unique detached non-local clone without checkout
filters. It neutralizes inherited Git configuration, replacement objects, hooks,
templates, Python paths/environments, uv configuration, and pytest arguments
and plugins. It then re-executes the exact inner recipe with an explicit
allowlist. After the locked sync, both `VIRTUAL_ENV` and `UV_PYTHON` are bound
to the verified private `.venv` before Maturin runs; uv's managed base
interpreter is never an installation target. An unselected `patchelf` on the
fixed build PATH is rejected. The sole live release extension is validated as
a canonical regular ELF before import, loaded only through the exact
nonsubclassed `ExtensionFileLoader` type from that exact path, checked through
its PyO3 identity, and rechecked for unchanged bytes and file identity after
import. The validated specification executes directly without a second finder
resolution, failed validation rolls the module cache back, and live plus
durable evidence must expose exactly one JAX CUDA device. No extension copy
becomes evidence.

Qualification reports live below
`results/parity/qualification/<Slurm job>/<Slurm step>/<run nonce>/`.
The run root contains `slurm_process_attestation.json`; every private schema-0
workflow report is a canonical nonsymlink regular file and binds the
attestation's relative path and SHA-256.
The final node atomically publishes
`qualification_bundle_<exact Git SHA>_<Slurm job>_<Slurm step>_<run nonce>.json`,
whose strict schema version is 0. It covers all three workflows and binds their
report and Parquet-output digests. The sanitized bundle thereby covers the
private attestation binding
through each report digest without publishing the attestation's absolute
paths. Schema versions must be JSON integers, report references must be
canonical relative paths below their workflow directory, and the published
bundle must be a canonical nonsymlink regular file. Reports and bundles record
the bootstrap identity and the path,
version, and SHA-256 of `bash`, `ar`, `as`, `cc`, GCC
`cc1`/`cc1plus`/`collect2`, `cargo`, `c++`, `env`, `git`, `just`, Maturin,
Mold, both selected and private-venv Python interpreters, `ranlib`, `rustc`,
`scontrol`, and `uv`; they also
record the four effective Rust
flag/wrapper overrides as empty strings. An existing path is never replaced.
The payload and dependencies are validated before the final atomic link, then
the linked bundle is validated again and by a separate Python process before
the run succeeds.

Optional `just test-parity` retains all diagnostic comparisons but skips exact
bundle publication. Mutable worktree qualification recipes are nonqualifying;
`just test-parity-required`, `just test-parity-required-exact`, and
`just slurm-gpu-test-parity-required` deliberately refuse publication. This
provenance model trusts the host, scheduler, and selected tool executables; the
selected commit is hash-bound evidence. The scheduler-selected Cargo cache,
Rustup toolchain installation, and Python installation trees are trusted
inputs; Cargo cache content is copied into the private run root before use.
Hardcoded system helpers such as coreutils, `tar`, Python 3 used only
for the isolated nonce, and `git-upload-pack` are part of the trusted host
image even though the twenty primary tools are serialized. It does not claim
that a hostile host, shell, Git implementation, or scheduler can authenticate
itself.

## Coverage Reports

`just coverage-python` and `just coverage-rust` generate reports for the active
test surface. They are not advertised as percentage gates while the product
suite is being rebuilt around the current native boundary. Restore a threshold
only with a measured baseline and tests for real supported contracts; do not
meet a number by adding placeholder product tests.

## Documentation Changes

For documentation-only changes, run:

```bash
just docs-build
```

Run code tests as well when documentation describes behavior changed in code.
