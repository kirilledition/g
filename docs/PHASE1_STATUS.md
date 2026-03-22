# Phase 1 Status

Date: 2026-03-22

## Environment

- CPU: AMD Ryzen 7 9800X3D 8-Core Processor
- Memory: 60.44 GiB
- GPU in benchmark report: none detected by `scripts/benchmark.py`
- PLINK baseline binary: `PLINK v2.0.0-a.6.33LM AVX2 AMD (28 Feb 2026)`
- Regenie baseline binary: `v4.1.gz`

The Nix development shell now installs the AMD-tuned PLINK 2 binary from the official PLINK 2 download page.

## Phase 1 Completion Status

Phase 1 acceptance criteria are currently satisfied:

- Continuous-trait parity is within tolerance.
- Binary-trait hybrid logistic parity is within tolerance, including Firth fallback rows.
- `ruff`, `ty`, and `pytest` pass in the project environment.

## Latest Baseline Benchmark Snapshot

Generated with `nix develop -c uv run python scripts/benchmark.py`.

- PLINK continuous baseline: 0.379 s
- PLINK binary hybrid baseline: 5.061 s
- Regenie step 1 baseline: 188.821 s
- Regenie step 2 baseline: 1.763 s

## Latest Phase 1 Evaluation Snapshot

Generated with `nix develop -c uv run python scripts/evaluate_phase1.py`.

- Linear runtime:
  - PLINK: 0.255 s
  - Phase 1 engine: 5.735 s
  - Phase 1 / PLINK slowdown: 22.459x
- Logistic runtime:
  - PLINK: 5.234 s
  - Phase 1 engine: 85.275 s
  - Phase 1 / PLINK slowdown: 16.292x

## Latest Parity Snapshot

- Linear parity:
  - variants compared: 418943
  - max abs beta diff: 4.9968e-06
  - max abs SE diff: 4.99995e-07
  - max abs t diff: 4.99987e-06
  - max abs p diff: 5.17602e-07
- Logistic hybrid parity:
  - variants compared: 418943
  - non-Firth variants: 407076
  - Firth variants: 11867
  - method mismatches: 0
  - error-code mismatches: 0
  - max abs beta diff: 1.99773e-05
  - max abs SE diff: 1.80500e-05
  - max abs z diff: 1.11230e-05
  - max abs p diff: 4.58049e-06

## Notes

- Switching from the Intel-labeled PLINK binary to the AMD AVX2 build improved benchmark hygiene for this machine.
- The engine remains correctness-first at the end of Phase 1; performance optimization remains a Phase 2 concern.
- The raw machine-local JSON outputs are written to `data/benchmark_report.json` and `data/phase1_evaluation_report.json`, but `data/` is git-ignored, so this document records the current tracked snapshot.
