# Python Test Suite Audit

| Status | Applies to | Owner |
| --- | --- | --- |
| Exact-head foundation | Python tests as of 2026-07-23 | Correctness maintainers |

The active Python suite is intentionally a mathematical and external-parity
surface. It does not pretend that deleted product tests still protect the
current native application boundary.

## Active Tests

| Path | Contract |
| --- | --- |
| `tests/numerical.py` | Shared strict `abs(actual - reference) < tolerance` assertion with exact nonfinite masks. |
| `tests/parity/` | Login-safe upstream metadata, source/evidence freshness, schema, composite-key alignment, strict tolerance, and significance-decision checks. |
| `tests/test_regenie2_parity.py` | Three required exact-source full-chromosome workflows through `g._core.cli.run`, each compared with upstream REGENIE v4.1 and bundled as sanitized evidence. |
| `tests/test_regenie2_linear.py` | Quantitative mathematical behavior. |
| `tests/test_regenie2_binary.py` | Binary score and approximate-Firth mathematical behavior. |
| `tests/test_regenie2_binary_firth_null.py` | Binary null-Firth behavior. |
| `tests/test_regenie2_binary_scalar_firth.py` | Scalar approximate-Firth behavior. |

Additional focused compute tests may be added only for supported production
code. Product tests for deleted Python orchestration modules are not revived.

## Plumbing Repairs

- `just test-local-focused` now names an existing login-safe test instead of
  deleted `tests/test_core.py` and `tests/test_io_output.py`.
- GitHub workflows no longer require deleted `tests/test_cli_smoke.py`. The
  package-install job remains the real installed-console-script smoke check.
- Non-data CI still collects the active mathematical suite and the parity
  harness.
- Exact-head publication starts from a scheduler-selected full commit. The
  trusted launcher extracts and hashes that commit's
  `tooling/server/exact_parity_bootstrap.sh`, then invokes it on `landau`
  through system Bash under `env -i`. The bootstrap validates the live Slurm
  job/step/user/node/state, creates a unique detached non-local clone, invokes
  exact required node IDs, and turns absent protected fixtures into failures.
- Qualification evidence binds the job, step, nonce, bootstrap, and the
  paths, versions, and SHA-256 digests of `bash`, `ar`, `as`, `cc`, GCC
  `cc1`/`cc1plus`/`collect2`, `cargo`, `c++`, `env`, `git`, `just`, Maturin,
  Mold, the selected and private-venv Python interpreters, `ranlib`, `rustc`,
  `scontrol`, and `uv`, and records the four effective Rust
  flag/wrapper overrides as empty strings. Its report layout is
  `<base>/<job>/<step>/<nonce>/<workflow>/`; its bundle is named
  `qualification_bundle_<Git SHA>_<job>_<step>_<nonce>.json`.
- The scheduler-selected Cargo-cache snapshot, Rustup installation, and Python
  installation are trusted inputs. Qualification copies Cargo cache content
  into its private run root and uses a private uv environment with Python
  downloads disabled.
- Mutable Justfile parity recipes are diagnostic and nonqualifying.
  `just test-parity-required`, `just test-parity-required-exact`, and
  `just slurm-gpu-test-parity-required` deliberately refuse publication; ordinary
  local parity may skip data and never emits an exact-source bundle.
- Coverage recipes report the measured active surface without claiming an
  unsupported 90% product-coverage gate.

## Known Boundary

CLI validation, output/resume lifecycle, telemetry, orchestration, and tooling
automation are not broadly covered by Python product tests at this foundation
stage. Those contracts should be added later against the supported `g.cli` and
`g._core.cli.run` surfaces, with committed small fixtures where possible. Their
absence must stay visible rather than being hidden behind stale node IDs or
placeholder tests.
