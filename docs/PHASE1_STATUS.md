# Phase 1 Status

Date: 2026-03-28

## Environment

- CPU: AMD Ryzen 7 9800X3D 8-Core Processor
- Memory: 60.44 GiB
- GPU in benchmark report: `NVIDIA GeForce RTX 4080 SUPER`
- PLINK 1.9 baseline binary: `PLINK v1.9.0-b.7.11 64-bit (19 Aug 2025)`
- PLINK 2 baseline binary: `PLINK v2.0.0-a.6.33LM AVX2 AMD (28 Feb 2026)`
- Regenie baseline binary: `v4.1.gz`

The Nix development shell now installs both the official PLINK 1.9 Linux x86_64 binary and the AMD-tuned PLINK 2 AVX2 binary.

## Phase 1 Completion Status

Phase 1 acceptance criteria are currently satisfied:

- Continuous-trait parity is within tolerance.
- Binary-trait hybrid logistic parity is within tolerance, including Firth fallback rows.
- `ruff`, `ty`, and `pytest` pass in the project environment.

## Latest Baseline Benchmark Snapshot

Generated with `nix develop -c uv run python scripts/benchmark.py`.

- PLINK 1.9 continuous baseline: 15.901 s
- PLINK 1.9 binary baseline: 7.476 s
- PLINK 2 continuous baseline: 0.247 s
- PLINK 2 binary hybrid baseline: 3.153 s
- Regenie step 1 baseline: 176.502 s
- Regenie step 2 baseline: 1.835 s

## Latest Phase 1 Evaluation Snapshot

Generated with `nix develop -c uv run python scripts/evaluate_phase1.py`.

- Linear runtime:
  - PLINK 1.9: 15.905 s
  - PLINK 2: 0.251 s
  - `g` CPU: 7.369 s
  - `g` GPU: 3.814 s
- Logistic runtime:
  - PLINK 1.9: 7.481 s
  - PLINK 2 hybrid: 3.161 s
  - `g` CPU: 152.089 s
  - `g` GPU: 27.844 s

## Latest Parity Snapshot

- Linear parity:
  - variants compared: 418943
  - max abs beta diff vs PLINK 2: 2.87150e-04
  - max abs SE diff vs PLINK 2: 1.02400e-04
  - max abs t diff vs PLINK 2: 3.39700e-04
  - max abs log10(p) diff vs PLINK 2: 3.84265e-03
- Logistic hybrid parity:
  - variants compared: 418943
  - non-Firth variants: 407076
  - Firth variants: 11867
  - method mismatches vs PLINK 2: 19
  - error-code mismatches vs PLINK 2: 2656
  - max abs beta diff vs PLINK 2: 3.25018e-01
  - max abs SE diff vs PLINK 2: 1.74578e-01
  - max abs z diff vs PLINK 2: 2.03318e-01
  - max abs log10(p) diff vs PLINK 2: 1.14253e-01

## Notes

- PLINK 1.9 is now included to avoid conflating PLINK 2 benchmarking with `.bed` to `.pgen` workflow differences.
- The refreshed report shows GPU acceleration helping `g` substantially, especially on logistic regression.
- PLINK 1.9 parity is currently best-effort only; its assoc output does not line up cleanly with the existing PLINK 2 allele/effect conventions, so PLINK 2 remains the authoritative parity target.
- The engine remains correctness-first at the end of Phase 1; performance optimization remains a Phase 2 concern.
- The raw machine-local JSON outputs are written to `data/benchmark_report.json` and `data/phase1_evaluation_report.json`, but `data/` is git-ignored, so this document records the current tracked snapshot.
