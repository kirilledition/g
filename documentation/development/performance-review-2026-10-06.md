# Architecture cleanup and performance — 2026-10-06

This review compares the implementation after architecture cleanup with main
commit `225756c488a126405b13944338a4b176a3f63007`. Rust remains responsible for
planning, input, scheduling and output; Python/JAX owns numerical state and
kernels. No numerical formula, convergence tolerance or genotype validation
policy changed.

## Retained changes

- Tiled delivery shares one immutable metadata owner across active groups,
  preserving validation and lazy Arrow arrays while retaining separately
  owned genotype statistics.
- Shared-source CUDA decoding returns probability pairs and validation status
  without computing statistics that selection recomputes. Full decoding drops
  an unused mean result and uses a new private FFI target version.
- Compiled group preparation combines conditioning, QR, rank evaluation and
  quantitative residual preparation. Approximate-Firth chunk specialization
  excludes null-model settings that only chromosome preparation consumes.
- Backend transport/materialization, native conversion/registration and the
  deep-profile command have separate owners. Confirmed dead tooling is removed
  and current ownership/module-reachability checks run in maintained checks
  and CI.
- Native result conversion preserves logical ordering for Fortran-contiguous
  arrays. Result buffers still move into independently owned Rust storage
  before asynchronous writers consume them.

## Complete application measurements

The frozen fixture contains 2,504 Oxford sample IDs and 418,943 chromosome 22
variants. Equal-mask workloads select 2,379 samples per phenotype; heterogeneous
workloads select `[2379, 2254, 2129, 2004]` twice. Each process uses a Landau V100,
eight pinned physical host cores, chunk size 16,384 and telemetry off.

Source, native extension and input contents are pinned and checked around each
campaign. Fresh ABBA processes use separate initially empty persistent JAX
caches, then one excluded warm-up and repeated hot scans. Every hot scan
requires populated, unchanged cache contents. Timers cover the complete native
CLI call, including output writes. Configuration generation, input hashing,
output audits and separate resource observers are outside timing. All trials,
including slow observations and failed comparisons, remain in the evidence.

| Workload | Masks | Hot samples per version | Baseline median (s) | Candidate median (s) | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal mask sizes | 1 | 10 | 0.458184 | 0.435616 | 4.93% |
| Equal mask sizes | 8 | 10 | 2.250780 | 2.167729 | 3.69% |
| Equal mask sizes | 32 | 10 | 8.616962 | 8.195678 | 4.89% |
| Heterogeneous mask sizes | 8 | 6 | 2.262753 | 2.115787 | 6.50% |

Medians pool every eligible raw hot sample; they do not average process
medians. These are descriptive local measurements, not statistical significance
claims or portable speed guarantees. All 687 phenotype-output comparisons in
these workloads match exactly, including every numeric bit, null placement,
ordered row/sample coverage, schema, stable footer metadata, logical chunk
commitments and execution-plan fields.

The initial single-trait campaign overlapped CPU validation and documentation
installation on the shared filesystem. It is retained as a confounded
observation. An isolated ABBA repeat with seven hot scans per process gives
quantitative medians of 0.426164 and 0.420013 seconds, and binary medians of
0.545069 and 0.547261 seconds. Both quantitative suites match exactly. The
binary default-compilation comparisons fail the exact gate, so they do not
support an exact-qualified binary speed claim.

## Fresh-compilation variation and controlled replay

Across the two binary campaigns, one candidate process changes low float32 bits
in `BETA`, `SE`, `CHISQ` and `LOG10P`. The isolated campaign's final baseline also
differs from its first baseline. Every process retains identical outputs across
its own hot scans. Null masks, correction methods/statuses, schemas and stable
metadata match across processes.

Inspecting the saved executables finds score cuBLASLt algorithm IDs
`5/1/6/5` in the first campaign and `2/2/6/0` in the isolated campaign.
Chromosome preparation also chooses different vendor GEMM algorithms.
Candidate preparation and correction executables are identical within each
campaign. The affected score and chromosome programs differ only in selected
algorithms and derived fingerprints. This agrees with the previously documented
[compilation variability](performance-review-2026-09-24.md#independent-compilation-variability).

Ten controlled GPU replays cover all five observed score algorithm IDs
`5/1/6/2/0`. All 15 complete-output comparisons pass bit for bit: the five
baseline/candidate pairs and both replay arms against their retained canonical
producer. Each comparison covers all 418,943 rows and their output contracts.

The replays copy only the two structurally unchanged score/chromosome entries
into separate source-compatible private caches. Decoder, new preparation and
correction programs retain their own compiled entries. Required cache hits,
actual algorithm choices and unchanged artifacts are recorded. Some canonical
producers come from candidate processes; selecting their algorithms in baseline
source is an explicitly controlled counterfactual. These seeded replays are
numerical controls, not independent-compilation or headline timing evidence.
The original failed reports remain failed; the production compiler policy
remains unchanged.

## Validation and provenance

Combined validation passed `just check`, `just test` (402 passed, two skipped),
complete workspace Rust tests and `just docs-build` on allocated CPU nodes.
GPU validation passed 30 decoder tests and all three required full-chromosome
external REGENIE parity tests. Independent component/integration reviews and
all pull-request CI checks passed. These qualify the tested local implementation
and do not establish release readiness for every input or device.

Original finalization compared POSIX device numbers across compute nodes. That
host-local value differs between Landau and Shannon for the same shared files.
Independent allocated-node revalidation confirmed all 19 pinned input hashes,
paths, sizes, timestamps and inodes. A reviewed correction retains strict
device identity before/after each local hash and compares global identity across
nodes. It writes separate corrected verdicts; the original reports and complete
harness archive remain intact. Only the multi-mask campaign qualifies under
that provenance correction; both binary-containing single-trait campaigns
retain their numerical failures.

## Component measurements and limits

Metadata setup fell from 202.717 to 101.378 microseconds for 16,384 rows and from
115.743 to 57.905 microseconds for the 9,343-row tail. GPU group preparation
improved from 5.580 to 0.850 milliseconds for 2,504-sample linear preparation and
5.617 to 1.312 milliseconds for binary preparation. These component
measurements do not establish whole-application gains of the same magnitude.

The source-only decoder's full FFI probe improved about 1.2%. Alternative launch
geometries regressed other cohort sizes, so the 256-thread launch stays. A
faithful native owned-copy probe measured 8.312, 83.178 and 250.327 milliseconds
for one, 16 and 64 traits over 425,984 rows. JAX/NumPy ownership does not permit
unqualified transfer of those allocations into asynchronous Rust writers.

Ignored `results/architecture-implementation-20261006/` retains raw campaigns,
timing ranges/drift, exact comparisons, separate resource observations,
source/native fingerprints, independent input revalidation and controlled
replay evidence. Related worker evidence is preserved under
`results/architecture-cleanup-20261006/`, `results/architecture-metadata-20261006/`
and `results/jit-cleanup-20261006/`. The installed baseline extension and original
Python environment stay untouched throughout qualification.
