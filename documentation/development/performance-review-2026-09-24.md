# Profiling and optimization — 2026-09-24

This campaign compares against `e60853d8`, following the previous
[performance review](performance-review-2026-09-23.md). It covers the native
application, CUDA kernels and host dispatch, binary algorithms, input handling,
output construction, and fresh-process startup. Numerical formulas, convergence
tolerances, sample selection, and source validation remain acceptance constraints.

## Decisions

| Change | Decision and measured scope |
| --- | --- |
| Defer runtime initialization for completed resumes | Retain: fresh harness-process median 13.49 → 1.08 s; output validation preserved. |
| Decode only selected packed8 Firth candidates | Retain: 27.25% lower process-lifetime peak live JAX allocation on the large synthetic fixture; reserved pool unchanged. |
| Exclude inactive CUDA solver lanes | Retain: 83–85% less correction time for mixed synthetic batches; ordinary scan speed does not improve consistently. |
| Validate profiler completion and warm JAX traces explicitly | Retain: incomplete runs cannot be reported as successful captures. |
| Cross-group metadata cache, buffered input hashing, scalar JIT, extra zero-candidate conditional | Reject: no robust application benefit or workload regressions. |
| CUDA finalizer block/offset changes and Firth command-buffer registration | Reject: inconsistent shape results or no actual replay/application gain. |

The final ordinary binary hot-scan comparison takes 1.53% more time on this
small-cohort fixture. Retention trades that observed latency cost for lower
memory use and the correction-heavy benefit; it is not a general speedup claim.

## Measurement and ownership

Independent workers own the profiling harness, numerical experiments, CUDA
experiments, native input experiments, metadata reuse, and startup/resume changes.
Separate reviewers inspect lifecycle, cache, and numerical correctness.

The baseline checkout, environment, and native extension remain frozen while
profiling and comparisons run. The candidate uses a separate checkout and real
environment installed from the frozen lock. CPU and GPU builds have distinct
Cargo target directories: `target-cpu=native` makes the cantor and landau artifacts
non-interchangeable. Builds, tests, and benchmarks run in Slurm allocations;
Nix is unavailable, so the documented server environment supplies the tools.

Headline measurements have telemetry disabled. Fresh-process, same-process hot,
instrumented, and isolated-kernel results are recorded separately. Balanced
baseline/candidate/candidate/baseline comparisons record source bytes, native
hashes, configuration, dependencies, inputs, output validation, and run order.
Repeated measurements within one process are not independent process samples.

The principal real-genotype fixture has 418,943 chromosome-22 variants and
2,504 source samples. The multi-mask fixture uses deterministic quantitative
traits, 2,379 selected samples per trait, and synthetic nonzero LOCO predictions.
These workloads establish local performance and regression evidence, not
biobank-scale or scientific model validation.

GPU qualification uses one Tesla V100 PCIe 16 GB on landau, driver 535.274.02,
Python 3.14.7, and JAX/jaxlib 0.11.0. Comparisons keep the existing float32 score
arithmetic and enabled float64 support. The preserved landau native artifacts
have these SHA-256 identities:

- Baseline: `4147c24dc4e537585d0f12de8cc2e02f72d47839e920019d19227c907bb7c8fd`.
- Rejected metadata-cache prototype:
  `63aa36c4fa022699f51bac975c444de2953e7f0634726e1cbfa81a41b9e1a609`.
- Retained candidate:
  `7ab0c70f52be1238920de2f84e57a037fcd476630017442e58caf4d3f1f99bc3`.

## Profiling coverage and reliability

The extensive suite includes quantitative and binary tuning matrices on CPU and
GPU, finalist runs, uninstrumented headlines, stage diagnostics, JAX timelines
and memory snapshots, cProfile, py-spy, Scalene, Memray, Nsight Systems, Nsight
Compute, and Linux perf attempts. Generated plans and reports are retained in
`results/perf-20260924/baseline-suite/`.

The three principal campaigns completed 107 subprocess attempts. Their 18
major-profiler attempts include 12 successes, one partial Nsight Systems capture,
four permission skips, and one invalid native py-spy capture. A separate
nonblocking Python sampler completed successfully with GIL-holder stacks only:
115 samples and 19 sampling errors support qualitative attribution, not reliable
runtime fractions. It does not qualify the failed native-stack capture.

The first binary capture exposed a profiler-reporting bug: interrupted py-spy
returned zero without completing application output. The repaired harness
requires complete manifests, logical variant coverage, and readable Parquet
metadata with matching row counts and schemas before reporting profiler success.
This completion check does not decode every data page. Incomplete output remains
a failed run even when the profiler itself exits successfully.

JAX trace capture now performs a discarded same-process warm-up into a distinct
output root with resume disabled. The capture records its scope and disables
Python event tracing. In the initial real-data validation, trace events fell
from 2,215,911 to 47,105 and compressed JSON from approximately 27 MiB to
966 KiB, retaining all 26 decompressions and 506 Firth launches. This improves
measurement usability; it is not an application speedup. Normal headline and
cold-stage measurement behavior is unchanged.

On landau, Nsight Systems produced a usable timeline and complete application
output but reported metadata/import failures; its status is partial. Nsight
Compute hardware-counter access and Linux perf permissions were denied. Missing
counters cannot support an achieved-occupancy or measured-bandwidth claim.
After-run JAX memory snapshots are not peak device-memory measurements; host RSS, allocator
statistics, and compiled buffer estimates are reported with their own scopes.

The full CPU matrix completed 48 subprocess attempts: 46 successes and two
explicit Linux perf permission skips. It includes 16 tuning runs, eight
finalist repetitions, four finalist stage diagnostics, six headline runs, two
headline diagnostics, six profiler attempts, and six telemetry-mode runs. Both
JAX and cProfile captures completed all 418,943 rows for both trait types.
The isolated cantor baseline native SHA-256 is
`19aa9f4226eef5ec56436c1af3b9cda8f197438ca4dbedd18d3bf1491ab49cf8`.

CPU fresh-process headline medians are 30.81 s for binary and 19.94 s for
quantitative. The selected binary configuration uses 4,096-variant chunks,
four writer workers, four Rayon workers, and Firth batches of 32. The selected
quantitative configuration uses 2,048-variant chunks, four writers, and eight
Rayon workers. These tune the measured fixture; they are not new global defaults.
The profiling harness partitions CPU compilation caches by host and CPU features;
the production CLI honors its explicit cache path without that tooling remap.

The full GPU matrix completed all 46 subprocess runs successfully. Fresh-process
headline medians are 19.816 s for binary and 16.704 s for quantitative. Those
survey configurations and their compilation state differ from the production
hot comparison below; their timings cannot establish a candidate speedup.
Single-trial configuration and telemetry rankings are especially sensitive to
startup variation, so no global defaults are changed from this survey.

## Initial attribution

The single-group binary Nsight capture attributes about 140 ms of 217 ms of
aggregate kernel time to nvCOMP decompression, 11.7 ms to packed8 finalization,
and 11.2 ms to Firth component kernels. These are instrumented kernel totals,
not elapsed application time. The same region contains thousands of host stream
synchronizations and device-to-host API calls, motivating a CUDA capture probe.

Writer diagnostics attribute approximately 407 ms of aggregate worker time to
Parquet output and 21 ms to metadata arrays; worker work overlaps computation.
The terminal writer wait is approximately 52–66 ms. Eliminating metadata work
cannot be valued by adding those overlapping totals to application latency.

Native no-op scheduler measurements show 26 chunks across 32 groups taking
25.8 ms, including approximately 5.3 ms of lifecycle overhead. This does not
justify a scheduler rewrite. Existing group-major execution still decodes each
group separately; a small rolling genotype cache would miss on subsequent full
scans. A change to chunk-major scheduling requires a separate design for active
group state and memory bounds.

The dedicated 32-mask capture validates all 32 manifests and 13,406,176 rows
after each run. Its discarded warm-up takes 38.44 s; the captured diagnostic
takes 9.352 s. Direct trace counts show 832 nvCOMP decompressions totaling
4.277 s, 832 finalizers totaling 411 ms, descriptor kernels totaling 2.32 ms,
and host-to-device copies totaling 501 ms. This confirms the repeated work
without extrapolating a single-mask capture. Sharing descriptors alone has
little kernel-time upside; eliminating repeated decompression would require
the broader scheduling/state design described above.

## Experiments

### Active Firth lanes

The batched solver previously allowed padded or already successful lanes to
prolong shared iteration loops. The candidate intersects component validity
with the existing pseudo-stage active mask and Newton fallback mask. It leaves
retained-lane formulas and convergence policies unchanged.

Masking is limited to the existing CUDA-component path. A separate CPU
application comparison found no benefit from applying it to CPU components;
the CPU path retains its original validity handling.

Ten balanced synthetic correction rounds compare the baseline and four
alternatives. All 40 workload/alternative comparisons preserve all five output
fields bit for bit, including NaN and correction status. The workload has
2,504 samples and a correction capacity of 1,024.

| Active candidates | Baseline correction median | Masked correction median |
| --- | ---: | ---: |
| 0 | 0.490 ms | 0.483 ms |
| 1 | 1.511 ms | 1.632 ms |
| 128 | 30.767 ms | 4.520 ms |
| 1,024 | 33.082 ms | 5.711 ms |

The larger mixed workloads take about 83–85% less correction time. The broad
case contains 948 pseudo-stage successes, 43 eligible Newton fallbacks, and
33 invalid lanes. The single-candidate case instead adds approximately 0.12 ms.
These measurements exclude input, scoring, output, and startup; whole-application
measurements assess that workload dependence independently.

The real chromosome-22 mask-only GPU comparison uses the same baseline native
library in every process, 16,384-variant chunks, Firth batches of 512, eight
Rayon workers, and eight writers. Each ABBA position discards one warm-up and
retains 12 complete hot scans:

| Position | Source | Median GPU scan |
| --- | --- | ---: |
| 1 | Baseline | 0.557654 s |
| 2 | CUDA masking | 0.564533 s |
| 3 | CUDA masking | 0.566617 s |
| 4 | Baseline | 0.573337 s |

Pooled medians are 0.566291 s and 0.565705 s, a 0.10% difference. Baseline
process medians drift by 2.81%, while candidate medians differ by 0.37%.
This comparison establishes no ordinary-scan speedup or repeatable penalty.
Masking is retained for the correction-heavy mixed-batch benefit above.

The CPU application screen used the frozen cantor native library, 4,096-row
chunks, Firth batches of 32, four Rayon workers, and four writers. Each ABBA
position had one discarded warm-up and 12 hot complete scans:

| Position | Source | Median CPU scan |
| --- | --- | ---: |
| 1 | Baseline | 10.771 s |
| 2 | Masking on CPU | 11.188 s |
| 3 | Masking on CPU | 10.887 s |
| 4 | Baseline | 10.746 s |

Pooled medians are 10.752 s and 10.954 s, approximately 1.9% more time with
masking. Baseline arm medians differ by 0.24% and candidate medians by 2.69%.
Both candidate arm medians are above both baseline medians. This establishes
no CPU benefit and motivates the GPU-only scope; it does not establish a
universal 1.9% CPU regression.
Both comparison pairs also match bit for bit across every output column, null
mask, schema, logical chunk commit, and stable metadata. Within each process,
the maintained harness verifies identical output hashes for all 12 hot scans.
After limiting masking to CUDA, the CPU resolver's traced computation matches
the baseline exactly on the mixed-lane abstract workload. Both traces hash to
`0203da36a2681cecbff683d0f20376f59195d6479b12198b516553c01a45c57e`.

An additional outer zero-candidate conditional saves approximately 0.22 ms when
there are no candidates, but adds overhead to populated workloads. It is removed.
Separate scalar-phase JIT boundaries show no clear cold benefit and only small,
drift-sensitive hot differences; that experiment remains outside production.

### Packed candidate decoding

Binary packed8 scoring retains packed input instead of returning a full decoded
dosage matrix for correction. Firth preparation selects candidate rows before
decoding them with the existing float32 formula. An optimization barrier
preserves the former materialized rounding boundary before allele flipping and
residualization. The packed score entrypoint does not donate the retained input.
Unused decoded-result plumbing is removed.

A separate 100,000-sample, 1,024-variant fixture uses a correction capacity of
128 and covers zero candidates and 79 candidates. All result fields match bit
for bit in all four comparison processes. The process-lifetime JAX allocator
live peak falls from 1,295,984,384 to 942,819,584 bytes: 353,164,800 bytes
(336.8 MiB), or 27.25% less. This measurement includes the real-data warm-up
and both larger synthetic cases; it is not an isolated per-case peak. With
preallocation disabled, the allocator pool reservation remains 2,157,969,408
bytes in both arms. It does not establish lower reserved VRAM. The broad-case
hot time is approximately 0.814 s in both arms, so the established benefit here
is lower live memory.

One source-identical process in the packed-only application ABBA selected a
different cuBLAS Lt score-matrix algorithm and changed low float32 bits. The
other candidate process matched both baseline processes exactly. This was
investigated through controlled replay rather than relaxed output tolerances:

- A shared autotune cache preserves all 224 baseline entries and adds only five
  candidate-specific fusion entries. The controlled comparison holds CUDA
  masking fixed and isolates the old full-dosage boundary from candidate-only
  decoding; its old path is not pristine `e60853d8`. Strict replays preserve all
  229 entries, select matrix algorithm 1 in actual compiled HLO, and match every
  output bit across all 418,943 variants.
- Replaying the outlier's original compilation cache reproduces its output and
  confirms algorithm 5. Changing only that observed autotune entry makes both
  dosage paths consume algorithm 5 and match every output bit again.
- The old full-dosage path with algorithm 5 reproduces the original candidate
  outlier exactly; its outputs with algorithms 1 and 5 differ. This establishes vendor-choice
  causality independently of candidate-only decoding. It does not claim an
  unforced pristine-baseline process selected algorithm 5 in this controlled comparison.

Both replays also preserve statuses, null masks, schema, stable metadata, and
logical manifest commits. The diagnostic flags remain outside production;
precision, convergence, and compiler-autotuning policy stay unchanged.
[OpenXLA's determinism documentation](https://openxla.org/xla/determinism) and
[persisted-autotuning documentation](https://openxla.org/xla/persisted_autotuning)
describe the relevant compiler controls. The uncontrolled packed ABBA does not
support a precise packed-only speed claim.

### Input hashing

A larger buffered-read prototype was rejected and reverted. Balanced complete
input-preparation measurements changed from 495.843 to 498.474 ms for 100,000
samples and from 2.546 to 2.561 s for 500,000 samples. Temporal drift and an
unchanged sample-ID control exceed those differences. There is no established
gain or small-regression claim; input production code matches the baseline.

### CUDA kernels and capture

The finalizer screen reproduces baseline PTX byte for byte with the pinned
compiler and compares eight variants. All seven outputs match exactly across
identity, contiguous, indexed, reordered, duplicate, invalid-selection, tail,
and malformed-row cases. Resident-kernel timing excludes allocation, transfer,
nvCOMP decompression, JAX, and the application lifecycle.

| Finalizer shape | Baseline, 256 threads | 128 threads | 512 threads |
| --- | ---: | ---: | ---: |
| 16,384 variants × 2,504 samples, identity | 436.6 µs | 400.3 µs | 576.8 µs |
| Real contiguous selection | 487.1 µs | 495.1 µs | 637.4 µs |
| Real indexed selection | 502.8 µs | 502.8 µs | 665.1 µs |
| 128 variants × 100,000 samples, identity | 254.6 µs | 421.9 µs | 162.6 µs |

Uniform block-size changes regress other shapes. Narrower row-offset arithmetic
at 256 threads improves these cases by only approximately 0.6–4.9%. Even the
128-thread identity improvement saves approximately 0.97 ms over the measured
chromosome, less than 0.45% of aggregate GPU kernel time. A shape policy needs
more dimensions and full-application evidence; no production PTX refresh is
retained from this screen.

A separate Firth foreign-function probe moves module initialization into the
supported initialization stage and advertises command-buffer compatibility.
Kernel PTX and numerical work stay unchanged. On this server it does not produce
native graph replay: both variants execute 9,599 native calls, while the candidate
also receives 1,874 initialization calls. Broad correction changes from 32.768
to 32.825 ms and the partial case from 30.479 to 31.369 ms. The registration
prototype is rejected; maintained CUDA, Rust FFI, and binding code remain unchanged.

### Completed-resume startup

An explicit resume prepares and validates its input/output plan before deciding
whether a compute runtime is needed. When every phenotype is already committed,
it finishes without importing JAX or initializing a device backend. Pending work
initializes the backend once. Existing same-process configuration compatibility,
input fingerprints, output reconciliation, and committed-part validation remain
in force.

Fresh runs retain eager runtime initialization before output preparation, so a
backend setup failure does not leave a newly created output directory. Pending
resume failures preserve existing commits; interruption still performs the
required flush and records the signal. Native subprocess regressions cover
completed, partial, mismatched, corrupt, failed, and interrupted lifecycles.

The timing comparison uses cloned completed outputs, three ABBA blocks, and
six fresh processes per version. Two warm-ups and two instrumented attribution
runs are excluded from the headline distribution. All 16 calls retain output
integrity checks. Measurements use the final retained native build;
the timed process invokes the native CLI through the frozen-extension harness.
Output cloning and post-run inspection are outside the timed interval, and
filesystem caches are not flushed.

| Headline metric | Baseline median | Retained candidate median | Time reduction |
| --- | ---: | ---: | ---: |
| Native CLI call | 10.874318 s | 0.312033 s | 97.13% |
| Complete fresh harness process | 13.491901 s | 1.077847 s | 92.01% |

The six baseline process times span 13.396–13.581 s and the six candidate times
1.062–1.106 s. An independent physical audit rehashes all 64 Parquet files:
every call preserves 418,943 rows, 26 logical commits, four parts, 10,770,556
bytes, schema, and metadata. Manifests differ only in the relocated effective-
config path. Numerical/runtime policy and input provenance remain identical;
the existing BGEN opened-file identity contract is retained. All calls succeed.
All eight candidate calls leave JAX unimported, while all eight baseline calls
import it.

Separate profiles attribute 10.269 s to baseline JAX configuration and 0.252 s
to backend initialization; neither stage occurs in the candidate. Native
preparation takes 0.226 s before and 0.247 s after, so the gain comes from
avoiding unnecessary runtime initialization. This result applies to completed
resumes through this harness; ordinary unfinished scans, partial resumes, and
cold filesystem caches require their own measurements.

### CPU startup deployment

CPU cProfile captures attribute substantial startup work to CUDA plugin discovery,
despite explicit CPU execution. In installed JAX 0.11, discovery precedes platform
filtering and the CUDA plugin loads NVIDIA libraries before checking that filter.
No supported runtime switch to bypass discovery was found. CPU-only JAX
installation is a deployment option, but this project's base dependency includes
CUDA and its CPU dependency group is additive. A separate CPU-only distribution
needs its own installation and startup qualification. The instrumented discovery
time is not an established savings estimate, and no private runtime patch is added.

### Output metadata cache

The native prototype shares lazily constructed Arrow metadata across exact
variant ranges in different sample groups, with conservative admission into a
128 MiB estimated budget. Within-group sharing already exists in the baseline;
there is no separate new within-group benefit to retain.

The real-genotype quantitative comparison has one, eight, or 32 distinct sample
groups. Each ABBA process discards one warm-up and retains five hot scans.
Reported group totals below are medians of the two process medians per version:

| Sample groups | Baseline | Prototype | Candidate time change |
| --- | ---: | ---: | ---: |
| 1 | 0.456023 s | 0.463147 s | +1.56% |
| 8 | 2.362935 s | 2.349283 s | −0.58% |
| 32 | 9.118648 s | 9.075342 s | −0.47% |

The eight-group adjacent comparisons improve by 0.87% and 0.28%, while its
baseline process medians drift by 0.88%. At 32 groups one adjacent comparison
is 0.34% slower and the other 1.28% faster; baseline medians drift by 1.07%.
This does not establish a robust scaling benefit. The one-group control has
the cache disabled. Its first hot scan is slower in every process, consistent
with the index-admission lifecycle; all predeclared observations are retained.

The cache, its extra allocation/indirection, budget machinery, and cache-specific
tests are removed together. Source snapshots and the exact prototype/removal
patches are archived. After all timed arms completed, the prototype allocation
was intentionally stopped; partial resource observations are excluded. Fresh
builds and release checks qualify the smaller retained change separately.

### Combined prototype application check

The rebuilt-native binary prototype comparison uses the same chunk,
batch, worker, and repetition settings as the mask-only GPU comparison:

| Position | Source and native library | Median GPU scan |
| --- | --- | ---: |
| 1 | Baseline | 0.571957 s |
| 2 | Combined prototype, including metadata cache | 0.575626 s |
| 3 | Combined prototype, including metadata cache | 0.568033 s |
| 4 | Baseline | 0.557572 s |

Pooled medians are 0.562752 s and 0.574324 s: the candidate takes 2.06% more
time in this comparison. Baseline process medians differ by 2.52%, and candidate
medians by 1.32%. This does not establish a whole-application throughput gain
or a precise portable regression size. The packed path is retained for its
measured live-memory reduction and masking for correction-heavy workloads;
ordinary small-cohort scan latency is a tradeoff to continue monitoring. This
prototype measurement is not the final rebuilt-native qualification.

### Retained application comparison

After removing the metadata cache, rebuilding the native library, and passing
the full CPU checks, the same binary ABBA protocol gives:

| Position | Source and native library | Median GPU scan |
| --- | --- | ---: |
| 1 | Baseline | 0.559976 s |
| 2 | Retained candidate | 0.572360 s |
| 3 | Retained candidate | 0.560651 s |
| 4 | Baseline | 0.550067 s |

Pooled medians are 0.556690 s and 0.565180 s: 8.49 ms, or 1.53%, more time
for the candidate in this comparison. Baseline process medians differ by
1.77%, and candidate medians by 2.05%; both candidate medians are above both
baseline medians. No general scan-speed improvement is claimed. Retention
accepts this small-cohort latency tradeoff for the measured reduction in live
GPU memory and faster correction-heavy mixed batches. Neither comparison
supports applying an exact percentage to other cohorts or devices.

Both final binary comparison pairs match every output bit across all 418,943
rows, along with all 14 columns, null masks, statuses, schemas, stable metadata,
and logical commits. Each process also verifies identical output hashes across
its 12 hot scans.

### Independent-compilation variability

The earlier mask-only raw comparisons fail exact score-field comparison in
both pairs; the combined-prototype comparison passes its first pair and fails
its second. Those reports retain their original failed status. A separate CPU
audit makes 76 exact comparisons across all 12 ABBA arms, all six pristine
`e60853d8` baseline outputs, and both controlled replay references.

Every candidate output matches an output produced by pristine baseline source.
Source-, native-, and input-identical baseline processes produce three observed
output groups. The mask-only baselines match the controlled algorithm-5 output;
all six candidates and three other baselines match the controlled algorithm-1
output. The remaining combined-prototype baseline matches neither reference;
its cause stays unclassified. Matching outputs alone do not identify which
vendor algorithm an uncontrolled executable selected.

All differences are confined to successful score-test rows in `BETA`, `SE`,
`CHISQ`, and `LOG10P`. Every Firth result, status, null mask, identifier, schema,
footer, and logical manifest comparison is exact. The final retained ABBA has
matching outputs in all four arms. No tolerance is introduced and no claim of
bitwise determinism across arbitrary fresh compilations is made. Source hashes,
input/native identities, and all inspected evidence remain unchanged after
the audit.

## Validation

After removing the metadata prototype, the complete native workspace test
suite passes on hilbert. The final `just check` lane passes CUDA/C++ formatting
and lint, Rust Clippy, Python
formatting/lint/types, binding-stub validation, and architecture/default/Justfile
guardrails. After the fixture correction, `just check` passes again and
`just test` passes 330 CPU tests in 97.83 s;
two native-CUDA mixed-lane cases are deliberately reserved for an initialized
GPU process. Native artifacts and source hashes are recorded before and after
validation. Independent reviews accept both the retained changes and the
metadata rollback without outstanding findings.

The final GPU compute suite passes 195 tests in 764.78 s. After the fixture
correction, all four mixed-lane cases pass in 27.75 s in a separately initialized
GPU process, covering both native CUDA and JAX components at both iteration
budgets. The CPU suite deliberately skips the two native CUDA cases.
Expected donation warnings are unchanged.

The added mixed-lane regression initially failed before reaching the optimized
batch resolver: its unchanged independent scalar reference stalled near the
existing convergence threshold. Failed attempts and a finite CPU/native fixture
screen are preserved. The final hard-call fixture has a known nonzero optimum
`log(11/5)` and identical CPU/native scalar results at both 30- and 60-iteration
budgets. Independent review confirms it preserves pseudo success, forced
pseudo rejection with Newton fallback, padding, failed-null lanes, all validity
assertions, and the original `1e-10` comparisons. No production solver or
convergence tolerance changed.

All three required full-chromosome upstream REGENIE comparisons pass in
84.56 s: quantitative, binary score-only, and binary approximate Firth. Missing
data is configured as an error. The binary approximate-Firth case verifies
17,938 corrections with zero failures across the 418,943 output rows.

All 15 full-output audits for the discarded metadata prototype pass, covering
all five hot repetitions at one, eight, and 32 groups. The aggregate historical
audit launcher exits nonzero because it also includes the earlier score-field
failures described above; the final retained binary gate passes independently.

The validation ledger also records the documentation build and confirms all
252 production source files remain unchanged across the final builds, checks,
and parity runs. The final test fixture is hashed separately. These gates
qualify the retained changes locally; they do not establish readiness for every
research dataset, device, or deployment environment.

## Next optimization targets

1. **Share decompression across sample groups.** The measured 832 decompressions
   provide the strongest remaining multi-group target. Evaluate bounded tiling
   across chunks and groups, including device-state residency, source identity,
   exact sample selection, resume subsets, and output ordering. A small rolling
   cache under the existing group-major schedule would not remove full rescans;
   retaining an entire chromosome would need an explicit memory budget.
2. **Qualify batching by candidate density and cohort size.** The small-batch
   profile launches 4,742 Firth component kernels versus 506 in the larger-batch
   profile. Those runs differ in chunk geometry and compilation state, so they
   do not establish a causal speedup. Further tuning should compare complete
   workflows with memory limits and mixed fallback distributions, preserving
   exact retained-lane outputs.
3. **Reduce repeated process initialization.** Existing native batch execution
   can reuse one initialized process for compatible scans. A separately packaged
   CPU-only environment could avoid CUDA discovery, but requires installation
   and startup qualification. GPU command-buffer work needs a compatible driver
   and proof of actual replay before kernel-launch savings can be claimed.

## Local evidence

Ignored `results/perf-20260924/` contains source manifests, native artifacts,
profiling reports, traces, discarded experiments, and benchmark harnesses.
`EVIDENCE.md` records artifact provenance and invalidated measurement attempts.
Generated datasets and profiler output are not committed.
