# Correctness review and fix plan — 2026-09-18

This review starts from `46a04020`, which fixed the first seven review findings.
The additional work covers accepted-input numerical correctness, input
ambiguity, output isolation and crash recovery, and interruption handling.
Findings below were established by source analysis and targeted reproductions;
the review does not claim exhaustive verification of every input or device.

## Findings and ownership

| ID | Priority | Confirmed problem | Implementation owner | Required regression |
| --- | --- | --- | --- | --- |
| R1 | P1 | Omitting the optional covariate file fails because an empty selection is passed as present. | Engine/input worker | Real engine preparation without covariates, both trait modes and sample modes; retain errors for nonempty selections without a file. |
| R2 | P2 | Raw float32 covariate normal equations make linear results depend on covariate origin; a shift of 2000 changed chi-square by 8.9%, and larger shifts produced NaNs. | Linear worker | Independent reference and shift/scale invariance, rank-deficient designs, explicit chromosome residual projection. |
| R3 | P2 | Undamped zero-start binary Newton fitting diverges for balanced cohorts with constant LOCO offset 3. | Binary worker | Constant and nonconstant offset shifts, unbalanced cohorts, analytical solutions, finite failure and iteration-budget behavior. |
| R4 | P2 | Distinct per-trait output directories can alias one physical directory and overwrite manifests and results. | Output worker | Equal/nested physical run or parts paths, symlink ancestors, rejection before mutation, valid disjoint paths. |
| R5 | P2 | Interrupted tail writes leave visible temporary Parquet files; after regrouped resume, directory readers duplicate rows or fail. | Output staging worker | Hidden staging, interrupted tail regrouping, compatible legacy cleanup, incompatible-resume nonmutation. |
| R6 | P1 | Rank validation rejects valid designs at realistic cohort sizes: 100,000 samples with intercept and ages 40/60 fail. | Engine/input worker with numerical workers | Normalize before rank analysis, use float64 tolerance, preserve true rank rejection and accept large valid designs in both modes. |
| R7 | P2 | Early float32 conversion of exact genotype moments biases INFO and can produce allele frequency greater than one. | Genotype worker | High-frequency rare alleles, reversed allele coding, monomorphic bounds, CPU/CUDA summary conversion. |
| R8 | P2 | Dense decoding accumulates moments sequentially in float32, biasing means, imputation, and native square sums at large sample counts. | Genotype worker | Large dense/missing/subset fixtures against a float64 oracle, packed/dense consistency. |
| R9 | P2 | BGEN chromosome X/chrX fails to match standard human REGENIE LOCO row 23. | Input review/fix worker | Both label directions, duplicate normalized rows, retain existing ploidy restrictions. |
| R10 | P2 | Duplicate selected phenotype, covariate, or identity headers silently select the first column. | Input review/fix worker | Reject ambiguous selected and identity headers while preserving harmless unselected columns. |
| R11 | P2 | Python KeyboardInterrupt loses its identity across the backend boundary, producing exit 1 and ordinary abort instead of exit 130 and interruption recovery. | Runtime worker | Backend preparation/execution and startup interruption, correct classification, queued-output recovery. |
| R12 | P2 | A phenotype exactly in the covariate span can yield finite association statistics from projection roundoff alone. | Linear worker | Scale-aware zero-residual detection with exact-span and genuinely varying phenotype controls. |
| R13 | P1 | A missing path component followed by `..` can hide populated output during inspection, then expose and overwrite it during directory creation. | Output worker | Freeze resolved paths for inspection and writing; populated-output and retargeting regressions. |

## Execution plan

1. Run independent source audits of input/cohort handling, genotype decoding and
   device delivery, execution scheduling, and CLI/runtime boundaries. Convert
   only confirmed findings into implementation tasks.
2. Implement the independent groups above in the existing fix branch, with
   exclusive file ownership. Coordinate shared numerical contracts between
   preflight, Python preparation, and manifest compatibility.
3. Fingerprint every changed result-affecting numerical policy. Reject older
   incompatible runs before modifying their manifests, configuration, parts,
   or temporary files.
4. Have reviewers who did not implement each group inspect its diff and
   regressions. Resolve findings before centralized validation.
5. Run Rust workspace tests, `just check`, `just test`, and `just docs-build`
   on Slurm CPU allocations. Build/install the native extension and run
   relevant CUDA regressions and all three required full-chromosome upstream
   parity workflows on landau.
6. Record final results and remaining limitations here. Preserve the existing
   untracked scratchpad and keep external datasets and generated reports out
   of version control.

## Review boundaries

The scheduler review covered backpressure, cancellation, chromosome barriers,
resumed chunk intersections, active-trait mapping, and padded tails. The
runtime review covered configuration layering, repeated execution, backend
exceptions, and startup. Input review covered selection and identity alignment,
chromosome labels, and table ambiguity. Genotype review covers CPU/CUDA decode
and summary boundaries.

Manual replacement of output parts from another run was considered a possible
hardening task, but is not part of this plan: the documented resume contract
validates execution plans, schema, and commit geometry rather than arbitrary
out-of-band replacement of result data. Concurrent external mutation of
symlinks is also outside the preflight collision guarantee.

This is a correctness qualification, not a full performance benchmark campaign.
The additional float64 preparation and persistent linear state costs are
described in the [performance guide](../public/performance-guide.md).

## Status

R1–R13 are implemented. Independent reviews accepted the input/preflight,
genotype, numerical, output, and interruption changes. Numerical integration
also adds an explicit constant-dosage binary score guard; its regression tests
retain rare variants and exclude invalid constants from Firth selection.

The Rust workspace passed 323 tests. The final output-policy marker and test
lint corrections also passed the affected crate suites. A PyArrow directory
probe confirmed that complete and truncated hidden staging files are excluded
while committed rows remain readable.

One backend-only beta assertion had a tighter bound than the direct kernel it
calls. The QR basis changed float32 projection accumulation: measured error was
`5.852765e-7`, versus `5.856285e-7` against an oracle using exactly rounded input
values. Only that backend bound was aligned with the existing direct-kernel
`1.2e-6` bound. The independent oracle, production arithmetic, other statistic
bounds, and external-parity tolerances were retained.

`just check` passed, and all 294 Python tests passed against the rebuilt release
extension on a Slurm CPU allocation. All 178 targeted CUDA tests also passed.
The CUDA backend probe measured beta error `2.202988e-7` against the unchanged
independent oracle, within even the former backend-specific bound.

All three required upstream REGENIE parity workflows passed on CUDA, each
covering 418,943 chromosome 22 variants. Approximate Firth completed 17,938
corrections with zero failures. The qualification reports are stored locally
under `results/parity/qualification/` and are not version controlled:

| Workflow | Report timestamp (UTC) | Result |
| --- | --- | --- |
| Quantitative LOCO | 2026-09-18 09:58:25 | Passed |
| Binary score-only | 2026-09-18 09:58:41 | Passed |
| Binary approximate Firth | 2026-09-18 09:59:10 | Passed |

Older output directories must be replaced with fresh directories because the
new numerical policies are incompatible with their recorded execution plans.
The implementation and documentation have passed independent final review.

Documentation build command: `just docs-build`. All compilation and test
workloads run on Slurm compute nodes; Nix is unavailable on this server, so
validation uses the documented no-Nix workflow.
