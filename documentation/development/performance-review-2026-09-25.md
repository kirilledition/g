# Profiling and optimization — 2026-09-25

This campaign starts from `b3d3311e87dff87ae29f8802831602ad9f4c4d58`,
after the [previous optimization review](performance-review-2026-09-24.md).
It measures the current application again before qualifying further changes.
Earlier campaign improvements are part of this baseline, not new gains.

## Outcome

The retained changes reduce hot quantitative application time by **9.5–10.7%**
when eight to 32 distinct sample masks share compressed GPU input. All 687
multi-mask trait-output comparisons pass exactly. The tradeoff is approximately
75–153 MiB more peak live JAX allocation and a retained allocator pool that grows
from 522 to 1,034 MiB on this fixture. Single-group controls show no comparable
GPU gain; the final CPU quantitative result is 0.83% slower, within observed
drift. These measurements do not establish population-scale performance.

Four optimizations remain in the implementation:

| Area | Retained change | Evidence and limit |
| --- | --- | --- |
| GPU delivery | Decode once for a bounded pair of sample masks; submit both groups before collecting results | 9.5–10.7% less complete hot application time, with the measured memory tradeoff |
| CPU genotype decoding | Replace the dosage lookup table with bit-exact arithmetic | 8.6–18.2% less time in missing-value decode probes; real all-present controls remain mixed |
| Firth lane ordering | Two integer prefix scans and one permutation scatter | 30–42% less time in representative GPU helper calls; no whole binary-scan speed claim |
| Output completion | Publish pending commits and terminal state in one atomic manifest replacement | One fewer replacement when commits remain; modest writer timing differences are drift-sensitive |

The first shared-source schedule is rejected because its eight-mask application
runs regress. Its replacement passes the declared linear output audits and
improves all three multi-mask workloads. The binary global exact audit remains
**failed** because unchanged baseline processes also disagree in successful
SCORE rows. Separate exact comparisons prove every final candidate output
matches the second baseline; no tolerance is relaxed. Cross-process bitwise
binary reproducibility remains an existing limitation, not a solved outcome of
this optimization pass.

## Measurement protocol

The baseline checkout, Python environment, and native extension stay frozen.
Candidate development uses a separate worktree and environment installed from
the frozen lockfile. Final application comparisons and GPU qualification use
frozen source/native overlays with the original immutable Python environment.
Source and native manifests identify the exact artifacts imported by each
measurement. Builds and workloads run in Slurm allocations; the server
environment supplies the toolchain because Nix is unavailable on this server.
Native artifacts compiled with `target-cpu=native` are specific to their CPU node.

The principal fixture contains 418,943 chromosome-22 variants and 2,504 source
samples. The additional multi-mask workload uses deterministic quantitative
traits, distinct sample masks, and nonzero synthetic LOCO predictions. Synthetic
mechanism probes and this small-cohort fixture do not establish biobank-scale
performance or scientific validation of a new analysis model.

The extensive suite covers CPU and GPU configuration matrices, uninstrumented
headlines, stage diagnostics, JAX traces and memory snapshots, cProfile, native
sampling, allocation profilers, and NVIDIA profiling tools. Failed, partial, and
permission-blocked profiler attempts retain their actual status. Instrumented
kernel totals and overlapping writer time are not treated as elapsed application
time. Device allocator reservation is reported separately from live allocation.

Candidate comparisons use balanced baseline/candidate/candidate/baseline order,
with compilation and warm-up excluded from hot measurements. Fresh-process and
hot measurements remain separate. Numerical formulas, convergence thresholds,
sample selection and order, source validation, and resume guarantees are
acceptance constraints. Comparisons retain all declared trials and audit output
fields, correction statuses, null placement, schemas, and logical commits.

## Qualification plan

1. Measure the frozen baseline using the extensive profiling suite and inspect
   attribution before selecting optimizations.
2. Independently qualify exact integer Firth lane compaction, direct CPU dosage
   arithmetic, and coalesced terminal manifest publication.
3. Probe full-source packed8 sharing before changing scheduling. Require exact
   selected statistics, explicit ownership, bounded retained state, compiled
   workspace estimates, and measured memory and timing evidence.
4. Retain measured improvements with acceptable controls; remove unsuccessful
   production prototypes while preserving their experiment evidence.
5. Run the application comparisons, numerical regressions, native tests, strict
   checks, required external REGENIE parity, and documentation build on the final
   retained source.

## Initial attribution

The fresh binary stage diagnostic attributes 11.54 s to JAX runtime
configuration, 4.91 s to native execution, and 0.23 s to native input/output
preparation. The separate cProfile capture contains substantial import, shared
library loading, plugin initialization, tracing, and cached-executable loading
costs. Nested cumulative profile entries overlap and cannot be summed. These
startup costs make a hot-kernel improvement a poor predictor of fresh-process
latency on this small fixture.

The discarded-warm-up binary JAX capture contains 47,802 events and all 26
chunks. Its device events attribute 133.71 ms to nvCOMP decompression, 11.38 ms
to packed8 finalization, 10.84 ms to 486 Firth component launches, and 15.66 ms
to host-to-device transfers. Descriptor kernels total only 0.098 ms. These
instrumented component totals motivate testing shared decompression; they are
not application speedup estimates.

The dedicated eight-mask capture validates 3,351,544 rows and contains 208
decompressions totaling 1.071 s. The 32-mask capture validates 13,406,176 rows
and contains 832 decompressions totaling 4.276 s, 411 ms of finalizers, and
507 ms of host-to-device copies. Their captured lifecycle times are 2.492 and
9.174 s respectively. These instrumented runs independently confirm repeated
decompression across groups; they do not measure the candidate.

Raw experiment artifacts are stored in the ignored local directory
`results/perf-20260925/`. The measured decisions and validation results below distinguish isolated
mechanism probes from complete application comparisons.

## Shared-source mechanism qualification

The source-sharing helper uses the unchanged native decoder once with full-source
selection, retains only packed pairs and source validation statuses, and gathers
each group's original samples with exact integer summaries. It preserves the
existing unsigned arithmetic, float64-to-float32 moment conversion, status
precedence, and padded rows. Source validation remains authoritative even for
bad probabilities outside a group's selected samples.

At 16,384 compute variants and 2,504 source samples, retained source arrays total
82,116,608 bytes. A 2,379-sample indexed selection returns 78,413,872 bytes;
compiled temporary storage is only 528 bytes, because the larger intermediate
integer arrays fuse into reductions. Contiguous and identity selections use
272 bytes of compiled temporary storage. Full and partial chunks, malformed
rows, invalid indices, and every returned field match direct native selection
exactly. Twenty-eight focused tests pass. Two sequential real linear consumers
also produce identical association values; explicit donation deletes their
private moment buffers while leaving source arrays intact and reusable.

Sixteen balanced same-process two-group stage comparisons give medians of
17.911 ms for separate decoding and 10.232 ms for shared decoding, 42.9% less
time. The stage includes upload, decoding, selected statistics, and
synchronization; it excludes engine scheduling, association, and output. All
declared samples, including two larger shared-stage outliers, are retained.
Separate fresh-process stage probes report peak live JAX allocations of
273,370,624 and 273,370,880 bytes respectively, with the same 274,726,912-byte
pool. These probes include compilation and five two-group stages, and do not
measure complete application memory.

This evidence qualifies a bounded scheduling prototype. The integrated V1
results below reject its initial schedule despite this isolated stage gain. The initial scope is compressed GPU linear runs with two eligible
groups, a 128 MiB source-array limit including statuses, and a separate 64 MiB
limit on the combined retained group/chromosome state estimate. These admission
limits do not represent a cap on total application or compiler memory.

### Integrated V1: exact outputs, unsuccessful scheduling

The first integrated schedule preserves output correctness but does not qualify
as a general application optimization. Its isolated two-group decode stage was
42.9% faster, yet both eight-mask application workloads regress. V1 is rejected
as the delivery schedule; its measurements remain part of the record.

The complete linear application comparison uses four fresh processes in
baseline/candidate/candidate/baseline order. Each process discards one complete
warm-up, then retains five hot scans for the equal-size masks or three for the
heterogeneous masks. Each table entry is the median of the two process medians
for that artifact. Timers surround the native CLI lifecycle, including native
preparation, execution, and output completion; earlier Python imports and the
discarded warm-up are excluded. Negative changes indicate lower elapsed time.

| Workload | Baseline s | V1 s | Change |
| --- | ---: | ---: | ---: |
| One mask, ordinary-path control | 0.443353 | 0.433245 | -2.28% |
| Eight distinct masks, each 2,379 samples | 2.316327 | 2.606084 | +12.51% |
| 32 distinct masks, each 2,379 samples | 9.797077 | 9.348713 | -4.58% |
| Eight distinct heterogeneous masks, 2,004–2,379 samples | 2.328107 | 2.384834 | +2.44% |

The equal-size eight-mask candidate arms drift by 17.22%; both adjacent
baseline/candidate comparisons nevertheless regress, by 2.45% and 22.79%.
The heterogeneous comparisons regress by 2.60% and 2.28%. The 32-mask result
has baseline drift of 5.89%, larger than its aggregate 4.58% improvement;
its adjacent improvements are 1.39% and 7.59%. These four-process comparisons
provide no confidence interval or population-level significance claim. The
one-mask control never takes the shared-source path. No hot scan is discarded
as an outlier, and the integrated artifact includes the other candidate changes.

All 18 requested exact-audit reports pass: 54 arm-pair comparisons, 687
trait-output comparisons, and 9,618 whole-column comparisons. Each report
compares the other three arms with the first baseline at the same hot-run
index. Each trait output contains 418,943 rows and 14 columns. Numeric bits at
non-null rows, null masks, schemas, logical commits, audited manifest fields,
and non-commit footer metadata match without numerical tolerance. These are
repeated-output equivalence checks, not independent scientific validation
datasets. Successful qualification checks establish the declared
correctness and provenance gates; they do not override the timing regressions
or make an automatic retention decision.

### Memory measurements and limits

Separate fresh-process observations report the following JAX allocator peaks
across startup, one warm-up, and one hot lifecycle:

| Workload | Baseline peak live MiB | V1 peak live MiB |
| --- | ---: | ---: |
| One equal-size mask | 513.103 | 513.103 |
| Eight equal-size masks | 513.103 | 520.881 |
| 32 equal-size masks | 513.103 | 520.881 |
| Eight heterogeneous masks | 513.103 | 520.842 |

All eight observations retain a 522 MiB allocator pool. Available boundary
snapshots and the post-GC snapshot report zero live bytes. Raw reserved-byte
counters also report zero; they do not negate the retained pool or establish
zero process GPU reservation. Cleanup performs garbage collection without
clearing JAX caches or shutting down the backend. Separate sampled host HWM
observations are 1,143.39 to 1,124.50 MiB for eight equal-size masks, 1,188.45
to 1,155.98 MiB for 32, and 1,252.76 to 1,227.39 MiB for heterogeneous masks.

These observations are separate from the earlier isolated decode-stage peak of
273,370,624 versus 273,370,880 bytes. JAX allocator peaks include warm-up and are
not reset between lifecycles; they exclude native allocations outside that
allocator. Retained allocator pool capacity, post-run live bytes, sampled host
RSS, and total GPU memory are different quantities. The process observations
exclude descendants and the observer, can miss short peaks, and do not establish
a final host high-water mark. The post-run trace memory snapshots establish no
peak-memory result.

The 128 MiB source limit and 64 MiB combined group/chromosome/selection-state
limit bound specified retained array payloads. They exclude selected input and
result sets, compiler workspace, preparation temporaries, and allocator capacity;
they are not total-device-memory admission guarantees.

### Trace evidence for lost overlap

Separate warm diagnostic traces confirm that V1 halves decompression launches:
208 to 104 for eight masks and 832 to 416 for 32 masks. Decompression device time
falls from 1,071.399 to 535.300 ms and from 4,276.440 to 2,141.853 ms,
respectively. Selection adds 77.789 and 313.366 ms of identified GPU interval
union, already included in total busy time.

The saved device work is offset by longer idle intervals. Across the physical
GPU stream envelope, eight-mask idle time rises from 287.873 to 1,331.357 ms;
32-mask idle time rises from 1,291.605 to 4,693.715 ms. Host transfer-dispatch
time overlapping GPU activity falls from 275.017 to 3.151 ms and from 1,106.217
to 12.321 ms. The one-mask control retains nearly unchanged GPU interval
structure. These observations support lost overlap as the next scheduling
hypothesis. They do not measure hardware occupancy, and overlapping host
categories cannot be summed into additive attribution.

Code inspection identifies a relevant barrier: V1 waits for each group's
association, host materialization, and writer acceptance before selecting the
next group. The traces do not directly instrument Rust channel waits, so they
cannot independently prove that this operation causes every observed gap.
The eight- and 32-mask baseline traces were captured earlier; these are single
instrumented diagnostics, not the balanced application timing comparison.
No new CUDA-kernel speedup follows from this diagnosis.

### Revised V2 schedule

The reviewed V2 schedule submits one private selected batch to each existing
pipeline before draining either, then drains in group order before releasing
the shared source or advancing to another chunk. It retains the two-group
limit, per-group state and resume selections, interruption checks between
successful drains, and worker abort/join before source release. It changes no
numerical kernel or solver policy. Static review found no blocker; the passing
deterministic tests exercise overlap and abort ordering with bounded waits.

Relative to V1's serial tile delivery, the payload calculation for one
additional selected input and result is
`2*C*S + 48*C + 16*C*T` payload bytes. At 16,384 compute variants, 2,379 selected
samples, and one trait, this is 79,003,648 bytes (75.34375 MiB). The estimate
allows input and result coexistence. It is neither a prediction nor an upper
bound on the total change in allocator peak, which includes other working
buffers and allocation lifetimes.

V2 passes the complete CPU checks and tests, 55 focused GPU tests, and all three
required external parity cases. A fresh application campaign repeats the same
sealed harness and process-median statistic with the qualified V2 artifact.
All 16 speed processes complete, preserving records for 16 warm-ups excluded
from timing and 72 timed hot lifecycles. No hot outlier is removed.

| Workload | Baseline s | V2 s | Change | Baseline arm drift |
| --- | ---: | ---: | ---: | ---: |
| One mask, ordinary-path control | 0.457019 | 0.453572 | -0.75% | +3.76% |
| Eight equal-size masks | 2.353672 | 2.115795 | -10.11% | +0.12% |
| 32 equal-size masks | 9.084585 | 8.111314 | -10.71% | -1.85% |
| Eight heterogeneous masks | 2.303187 | 2.083550 | -9.54% | -1.33% |

Both V2 process medians are below both baseline process medians for each
multi-mask workload. The one-mask control is neutral within observed drift.
These are complete hot native CLI lifecycles on the declared small-cohort
fixture, not fresh-process startup measurements or population-scale estimates.
The comparison includes the other retained candidate changes, so the table
measures the combined artifact rather than isolating one source edit.

Separate fresh observation processes measure the following memory tradeoff:

| Workload | Baseline peak live MiB | V2 peak live MiB | Baseline pool MiB | V2 pool MiB |
| --- | ---: | ---: | ---: | ---: |
| One equal-size mask | 513.103 | 513.103 | 522 | 522 |
| Eight equal-size masks | 513.103 | 587.787 | 522 | 1,034 |
| 32 equal-size masks | 513.103 | 666.319 | 522 | 1,034 |
| Eight heterogeneous masks | 513.103 | 642.843 | 522 | 1,034 |

The pool columns are both peak and post-GC retained capacity; post-GC live bytes
are zero in every case. These are JAX allocator counters, with the same scope
limitations as V1, not total process GPU memory. The measured multi-mask peak
increases are approximately 75–153 MiB; the retained pool grows by 512 MiB.
Sampled host RSS/HWM observations change from 1,171,176 to 1,157,512 KiB for
eight masks, 1,218,616 to 1,205,876 KiB for 32 masks, and 1,287,508 to
1,278,008 KiB for heterogeneous masks. One observation per artifact and case
does not establish a general host-memory reduction. Peak sampled thread counts
increase from 54 to 61, 57 to 67, and 59 to 61 respectively.

All 18 exact-audit reports pass: 687 trait-output comparisons, including 458
candidate comparisons and 229 baseline repeats, and 9,618 whole-column
comparisons. Values and numeric bits, nulls, schemas, semantic metadata, logical
commits, and 418,943 rows per trait match. Source, native, fixture, cache, GPU
identity, and completion guards pass. Both authoritative summary files mark
qualification checks passed; this report makes the separate retention decision.

V2 is retained for its consistent observed multi-mask gains, exact outputs,
acceptable single-group controls, and explicitly measured memory cost. Evidence
is under `multi-mask-final/qualification-2/{equal,heterogeneous}/`; Landau timing
and resource job 51571 and Cantor audit job 51575 both complete successfully.
The sealed harness is unchanged from V1.

### V2 trace: partial overlap recovery

A separate diagnostic capture confirms 104 and 416 decompressions for eight
and 32 masks, half the original baseline counts. Relative to V1, physical GPU
stream-envelope idle time falls from approximately 1,331 to 941 ms and 4,694 to
3,790 ms. Busy interval unions remain similar at 865 to 854 ms and 3,456 to
3,431 ms. Selection-submission host intervals overlapping GPU idle time fall
from 31.4 to 0.19 ms and 119.8 to 0.87 ms: that submission now mostly overlaps
GPU work.

Transfer dispatch does not regain the original overlap. Its host interval union
overlapping GPU activity is only 2.12 and 12.94 ms in V2, versus 275 and
1,106 ms in the earlier baseline traces. The revised schedule removes part of
the selection/drain serialization; transfer-related waits remain a target for
future bounded-prefetch experiments. Such a change would need fresh lifetime,
cancellation, memory, and output qualification before increasing concurrency.

These are single instrumented captures using the frozen trace fixture. The
V2 capture overlaps CPU exact audits on other nodes, with a shared filesystem;
its lifecycle times are not used as application speed evidence. Timing gains
come from the separate uninstrumented ABBA campaign. Landau job 51577 passes
source/native/cache/output guards and its declared 104/416-call coverage checks.
`final-traces/qualification-2/DIAGNOSIS.md` records the full interval definitions,
raw results, concurrency limitation, and provenance.

### Evidence identity

The application campaign is Landau job 51561, its exact audits are Cantor job
51565, and the separate diagnostic captures are Landau job 51566. Baseline and
V1 use the same original immutable Python environment with explicit frozen
package/native paths. Source, native, fixture, cache, and completion guards pass.

| Artifact | Python-source SHA256 | Native SHA256 |
| --- | --- | --- |
| Baseline `b3d3311e` | `ff34601be0dec9caa69a279ce46021611787bef351b3709a283b80b2c91682ea` | `7ab0c70f52be1238920de2f84e57a037fcd476630017442e58caf4d3f1f99bc3` |
| Integrated V1 | `6e652161c11d0b27f1044c6d83d48759d53921a88fff7a8797ce1840c219e10a` | `0d5c5711883b382ca96a60d37fcdc7ac4b2261a2cfd5f8be02baf628de7c6bd5` |

The Python digest covers the recorded package-source set; the complete V1
snapshot manifest is under `final-gpu-validation/qualification-1/`. Relative
to `results/perf-20260925/`, the supporting records are:

- `multi-mask-final/qualification-1/{equal,heterogeneous}/summary.{json,md}`:
  process medians, every retained timing, resource scopes, and identity guards.
- The matching `audits/audit-suite.json` files and linked comparator reports:
  exact comparisons and audit counts.
- `final-traces/qualification-1/DIAGNOSIS.md`, `interval-*.json`, and per-capture
  `summary.json`: interval definitions, source/driver hashes, and trace coverage.
- `scheduler-v2/{README.md,two-phase-delivery.patch,source-provenance.json}`:
  the bounded proposal, its memory calculation, and exact source delta.

## Stable Firth lane compaction

The candidate replaces three independent compact-index constructions with two
integer prefix scans and one permutation scatter. It retains the exact stable
order of regular active, heuristic active, and inactive lanes. Inactive
heuristic flags do not change membership; subtraction occurs before addition to
keep intermediate indices within the existing int32 bound. Solver arithmetic,
convergence policy, and candidate selection remain unchanged.

Seven focused tests pass on each of CPU and GPU, including exhaustive small
mask combinations and large padded/interleaved cases. The isolated probe checks
15 cases per backend. Representative all-active helper calls take 30–42% less
time on GPU and 39–64% less on the allocated Landau CPU. These timings include
dispatch. At 1,024 lanes, optimized GPU entry operations fall from six fusions
and three copies to three fusions and no copies; compiled temporary storage
falls from 18,480 to 1,808 bytes. The CPU mechanism measurements are not a
whole-application CPU benchmark.

Four fresh binary application processes retain 12 hot scans each after a
discarded warm-up. ABBA arm medians are 0.55444, 0.54700, 0.55854, and 0.62421 s.
The baseline arms drift by 12.6%, so the pooled difference does not establish
an application speedup. Candidate arms do not show a consistent penalty.

The initial paired exact comparisons pass. The stronger 47-comparison audit
against the first baseline fails for every third- and fourth-arm output and
remains recorded as failed. A separate 23-comparison audit establishes two
exact clusters across all 48 retained outputs: first baseline plus first
candidate, and second candidate plus second baseline. Every candidate output
therefore matches an unchanged-baseline output exactly. Within each process,
all hot outputs match.

Only successful score-test rows differ between the clusters: 262,179 beta,
38,778 standard-error, 270,999 chi-squared, and 191,564 log-p values across
26 chunks. Firth rows, correction statuses, null placement, and metadata do not
differ. The compiled score cache key is identical in all four processes, while
serialized executable bytes form the same two clusters. The two baseline
anchors exactly match the earlier controlled choice-1 and choice-5 output
references. No current score HLO/autotuning dump was captured, so this does not
directly identify the algorithm selected in these new processes. This preserves
the previously documented baseline reproducibility limitation without relaxing
numerical tolerances or converting the failed global audit into a pass.

## Terminal manifest publication

The output candidate merges verified worker commits and completed/interrupted
status into one atomic manifest replacement. The previous path made two
replacements when new commits existed. Completed resumes with no new commits
retain one status-only replacement. All commit conflicts are checked before
publication. If terminal publication fails, a commit-only fallback retains the
exact previous status and signal while the operation still reports failure.
This preserves the existing atomic-publication and strict-resume contracts;
it does not introduce stronger power-loss durability guarantees.

The isolated native candidate contains only the genotype and output changes,
with source and executable hashes verified before and after measurement.
On Hilbert, 95 output tests and strict Clippy pass, including real temporary-file
failure followed by strict resume without changing retained Parquet bytes.
The maintained writer benchmark uses 418,943 rows, 26 chunks, eight writer
threads, and the shared filesystem. Each ABBA arm has ten Criterion samples,
a one-second warm-up, and two seconds of requested measurement.

| Measured interval | Baseline arm means | Candidate arm means | Average change |
| --- | --- | --- | --- |
| Complete writer delivery and finish | 181.631, 172.180 ms | 174.700, 166.881 ms | -3.46% (6.115 ms) |
| Paced terminal finish | 106.432, 94.360 ms | 96.078, 93.665 ms | -5.50% (5.524 ms) |

Complete writer time improves in both adjacent comparisons by 6.931 and
5.299 ms. Paced finish has substantial drift: the baseline arms differ by
12.072 ms, and the later adjacent comparison saves only 0.695 ms. These short
measurements and completed application controls support retaining the change,
but do not establish a stable paced-finish or whole-application speedup. The
candidate manifest timer includes status publication while the baseline timer
excluded it; that timer is not used as a direct before/after comparison.

Raw writer measurements, the failure/recovery tests, and independent source
lineage verification are retained under
`results/perf-20260925/output-terminal/cpu-51555/`.

## CPU dosage arithmetic

The CPU decoder replaces its 65,536-entry probability-pair lookup table with
the same signed integer numerator and rounded float32 reciprocal multiplication
used to construct that table. This removes a retained 256 KiB table payload;
it is not an RSS measurement. Missing imputation, selected-sample order, raw
integer moments, validation, and sparse thresholds remain unchanged. Fifty-one
genotype tests and strict Clippy pass, including exact comparisons against the
frozen table for all 32,896 valid probability pairs through present, missing,
identity, contiguous, and indexed paths.

Hilbert job 51555 measures separate baseline/candidate/candidate/baseline
processes with four Rayon threads, 30 Criterion samples per case, a two-second
warm-up, and five seconds of requested measurement. Each table time is the
arithmetic mean of its two process-arm estimates. Negative changes are faster.
The real chromosome-22 input has 2,504 samples; each synthetic uncompressed
input has 16,384 variants and 2,048 samples. The missing fixture marks about
one call in 97 missing in every variant.

| Input | Selection | Chunk variants | Baseline ms | Candidate ms | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Real chromosome 22 | Full | 2,048 | 7.130 | 7.174 | +0.62% |
| Real chromosome 22 | Full | 16,384 | 88.318 | 88.786 | +0.53% |
| Real chromosome 22 | Contiguous half | 2,048 | 6.324 | 6.117 | -3.26% |
| Real chromosome 22 | Contiguous half | 16,384 | 66.433 | 69.319 | +4.34% |
| Real chromosome 22 | Strided half | 2,048 | 8.593 | 8.631 | +0.44% |
| Real chromosome 22 | Strided half | 16,384 | 93.032 | 89.775 | -3.50% |
| Synthetic present | Full | 2,048 | 1.019 | 1.019 | +0.01% |
| Synthetic present | Full | 16,384 | 50.681 | 50.513 | -0.33% |
| Synthetic present | Contiguous half | 2,048 | 0.613 | 0.615 | +0.30% |
| Synthetic present | Contiguous half | 16,384 | 30.215 | 30.255 | +0.13% |
| Synthetic present | Strided half | 2,048 | 3.269 | 3.237 | -0.99% |
| Synthetic present | Strided half | 16,384 | 50.097 | 50.210 | +0.22% |
| Synthetic missing | Full | 2,048 | 9.902 | 8.105 | -18.15% |
| Synthetic missing | Full | 16,384 | 119.276 | 105.612 | -11.46% |
| Synthetic missing | Contiguous half | 2,048 | 8.417 | 6.933 | -17.63% |
| Synthetic missing | Contiguous half | 16,384 | 92.552 | 77.513 | -16.25% |
| Synthetic missing | Strided half | 2,048 | 8.085 | 7.332 | -9.32% |
| Synthetic missing | Strided half | 16,384 | 89.105 | 81.435 | -8.61% |

Both candidate arms beat both baseline arms in all six missing-data cases,
with disjoint within-arm 95% confidence intervals. Missing-case process drift
reaches 7.44%, so these intervals do not establish an across-process confidence
interval. The real input's compatible packed8 validation and completed
418,943-row packed8 GPU scan independently establish that it is all-present.
Its full and contiguous selections therefore exercise unchanged source paths.
The 4.34% contiguous control regression remains unexplained; its synthetic
equivalent is nearly unchanged. Indexed all-present results are mixed. The
change is retained for the measured missing-path benefit and table removal,
with the completed application controls reported below. These measurements
establish no universal
all-present or whole-application speedup.

The baseline is an immutable archive of `b3d3311e`; the native candidate overlays
only the three genotype and seven output files. Separate Cargo targets and
hashed frozen executables prevent cross-workspace artifact reuse. Release
benchmarks were built in job 51548 before a test-only Clippy correction;
the revised manifest proves all 1,195 other files unchanged, and job 51555
validates the corrected tests. Both use the repository's Linux native-CPU
release configuration. Source manifests, executable/input hashes, all 18 cases'
four estimates and confidence intervals, and per-arm drift are preserved in
`results/perf-20260925/genotype/cpu-51555/analysis.json` and `analysis.md`, with
raw Criterion data under `abba/`.


## Completed baseline profiling suite

The refreshed baseline completed **107 principal subprocess attempts and 18 major
profiler attempts**: focused binary GPU 13/8, full CPU matrix 48/6, and full GPU
matrix 46/4. All 46 GPU matrix applications succeeded; the CPU matrix had 46
successes and two permission skips. Final profiler statuses are **13 successful,
one partial, and four skipped**. Same-process warm-ups inside JAX captures and
additional hot/multi-mask lifecycles are separate from the subprocess budget.

The original source, environment, and native library stayed unchanged. A final
byte audit also verifies all 37 files in each frozen CPU/GPU source overlay.
CPU and GPU baseline work used separate nodes concurrently; the later GPU matrix
overlapped prebuilt CPU native benchmarks while builds and tests were paused.
Configuration rankings remain surveys with startup and shared-filesystem
variation, rather than causal comparisons between code changes.

The following headline medians use three fresh processes with populated
persistent caches. The first trial's tooling label `cold_trial_name` does not
establish an empty compilation cache. Inside-CLI time excludes earlier Python
imports. Configuration entries are chunk size / writers / Rayon / Firth batch;
quantitative analysis does not use Firth.

| Workload | Configuration | Whole process | Inside CLI |
| --- | --- | ---: | ---: |
| CPU binary | 4096 / 4 / 4 / 32 | 31.368 s | 26.040 s |
| CPU quantitative | 2048 / 4 / 8 / — | 15.954 s | 11.747 s |
| GPU binary | 4096 / 4 / 4 / 32 | 19.399 s | 15.040 s |
| GPU quantitative | 2048 / 1 / 8 / — | 16.530 s | 12.124 s |

Separate ordinary hot drivers discard one complete warm-up and retain seven
complete native CLI scans in one process. Cache contents stay populated and
unchanged, and every hot output has identical Parquet bytes within its workload.
These are repeated scans within one process, not seven independent process
samples. CPU hot medians are 10.645 s binary and 3.536 s quantitative; GPU hot
medians are 0.567 s binary and 0.461 s quantitative. CPU hot drivers use chunk
4096, writers/Rayon 4, and Firth batch 32. GPU hot drivers use chunk 16384,
writers/Rayon 8, and Firth batch 512. Their different geometry and timing scope
prevent interpreting headline-to-hot ratios as a code optimization speedup.

JAX and cProfile both succeed for all five selected workloads. Matrix traces retain all
103 binary GPU chunks and 205 quantitative GPU chunks. The focused native
py-spy capture completes with 201 samples and zero sampler errors; its
Python-associated thread samples do not establish complete native-only Rust
worker activity. Scalene and Memray also complete the focused application.
Memray header metadata records 545,508 allocations and a tracked peak of
7,255,440,701 bytes, including allocation/mapping reservations; this is not
resident RAM or peak live device memory. Bounded high-watermark reconstructions
exceed their 120-second and 30-second limits, so no new per-stack allocation
attribution is claimed. JAX device-memory artifacts are post-run live-allocation
snapshots, not peak-memory measurements.

Nsight Systems retains a complete application and usable CUDA records, but GPU
metadata import errors leave its final status partial. Initial process-success
log lines do not override that status. Nsight Compute hardware counters are
denied, as is Linux perf on both Landau and Hilbert. No occupancy, register
pressure, or hardware-stall conclusions come from those denied attempts.

The retained evidence continues to prioritize bounded cross-mask decompression
reuse and amortizing initialization for repeated short research jobs. Further
Firth batching/dispatch experiments should use controlled hot workloads: the
matrix's batch-32 trace contains 4,689 component launches totaling 36.42 ms,
versus 486 launches totaling 10.84 ms in the focused batch-512 trace, with
otherwise different geometry. Descriptor kernels remain a small target. Integrated V1 confirms lower decompression work but fails application
retention because of eight-mask regressions. The retained V2 schedule improves
complete hot application time while partially recovering overlap. Its remaining
transfer serialization motivates bounded prefetch as a future experiment;
larger tiles remain outside the qualified scope. Initialization amortization
and broader real-cohort qualification remain higher-level opportunities.
None of these observations promises a population-scale gain from the
2,504-sample fixture.

The consolidated local index is
`results/perf-20260925/baseline-suite/EVIDENCE.md`; `suite-audit.json` retains
statuses, exact commands, manifest coverage, and source-summary hashes.
`final-overlay-audit.json` records the closing source/native byte checks.

## Final single-group application controls

The final CPU and GPU native artifacts repeat fresh-process ABBA controls with
one warm-up and five hot lifecycles per process. All 12 processes complete and
all 60 hot trials remain in the reports. GPU timing runs first, then CPU timing;
no builds, audits, or other campaign workloads overlap these timing windows.
Every process imports a frozen 36-file runtime snapshot and the explicitly
hashed native artifact through the unchanged original Python environment.

This harness reports the **pooled median of ten hot calls per artifact**, unlike
the multi-mask harness's predefined median of two process medians. Calls within
a process are not independent process samples. CPU controls use chunk 4,096,
four writers and Rayon workers, and Firth batch 32; the GPU binary control uses
16,384 / eight / eight / 512. The quantitative control uses the same single
indexed 2,379-sample synthetic trait as the multi-mask fixture.

| Control | Baseline s | Final candidate s | Change | Baseline drift | Candidate drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU binary | 10.584216 | 10.452919 | -1.24% | -0.72% | -0.41% |
| CPU quantitative | 3.221893 | 3.248646 | +0.83% | +1.71% | -1.05% |
| GPU binary | 0.562186 | 0.550373 | -2.10% | +0.97% | -3.72% |

CPU binary shows a modest reduction, with both candidate process medians below
both baseline medians. The CPU quantitative difference is smaller than the
observed baseline drift and its two adjacent comparisons have opposite signs;
no repeatable regression or improvement is established. The GPU binary change
is smaller than candidate drift and does not establish an application speedup.
The earlier V1 CPU artifact showed -1.50% binary and -1.74% quantitative pooled
differences; it is not substituted for these final-artifact results.

All 38 CPU exact comparisons pass. The final GPU binary global audit remains
failed: four first-arm repeats pass, while all 15 outputs from the candidate
arms and second baseline differ from the first baseline. A separate set of
14 exact comparisons establishes that every candidate hot output and every
second-baseline repeat equals the second baseline's first hot output. Thus the
20 outputs form an initial-baseline cluster and a candidate-plus-second-baseline
cluster. The original failed report and its hash remain unchanged.

Only successful SCORE rows differ between the unchanged baseline processes:
249,061 BETA, 32,052 SE, 257,566 CHISQ, and 179,978 LOG10P values. Maximum
absolute differences are approximately 1.64e-6, 1.22e-6, 1.26e-5, and 3.64e-6,
respectively; these diagnostics are not acceptance tolerances. Firth results,
nulls, statuses, schema, logical commits, and semantic metadata match exactly.
Baseline native/source artifacts, input/configuration, runtime, and effective
environment are identical. The score executable cache key matches across all
four arms, but serialized executable bytes differ and do not map one-to-one to
output clusters. No current HLO algorithm dump proves a particular vendor
algorithm choice. This campaign neither fixes nor conceals the pre-existing
cross-process reproducibility limitation.

Full timings, arm medians, private-cache records, import manifests, native
hashes, raw failures, and separate classification reports are under
`results/perf-20260925/single-group-final/{cpu-final-v2,gpu-final-v2}/`.
`v2-qualification-summary.json` indexes their exact hashes and qualified native
identities. All timing and audit allocations are released.

## Combined-source validation

The combined V1 candidate passes `just check`, including formatting, Python types,
Rust Clippy, maintained CUDA/C++ static analysis, and architecture policies.
The complete Rust workspace test run passes. `just test` reports 368 passed and
two GPU-only skips on the CPU allocation. The GPU numerical suite reports
152 passed and two gated native Firth skips; the three required external REGENIE
parity cases all pass. The initially skipped native cases retain their original
status. A separate process initializes the frozen native extension through the
full CLI, then passes both native mixed-lane cases (two passed, nine deselected).

CPU validation runs on Hilbert; GPU validation runs on Landau. The validated
CPU native SHA256 starts `9faef372650d65f0`, and the GPU native SHA256 starts
`0d5c5711883b382c`. Full hashes, build flags, logs, source manifests, and parity
reports are retained under `results/perf-20260925/final-cpu-validation/` and
`results/perf-20260925/final-gpu-validation/qualification-1/`. The baseline native
artifacts and environment remain unchanged. Node-specific optimized binaries
are not interchangeable between the CPU and GPU machines.

The revised V2 scheduler also passes `just check`, the complete Rust workspace
test run, and the CPU suite (368 passed, two GPU-only skips). Its focused GPU
suite passes 55 tests, with 17 tests outside the selected scope; all three
required external parity cases pass again. The numerical Python and CUDA
sources are byte-identical to fully qualified V1. The source delta guard admits
only the five reviewed scheduler/test/API files and the V1 validation-report
addition. A failed preflight that initially rejected that documentation-only
addition is retained; no build or test ran in that attempt.

V2's Hilbert native SHA256 is
`4bd83d615f00f9506a7ba6c571e9f608f68420ffb9f894913652a85df623fae3`;
its Landau native SHA256 is
`d3f10cb00ea08ca8d5fef812ba3b109c96ad06b21611358079ab697c91d7d944`.
Evidence is under `final-cpu-validation-v2/` and
`final-gpu-validation/qualification-2/`. Both final source/runtime guards pass.

The nested GPU qualification snapshots inherit the same three custom Rust
flags twice from parent and snapshot Cargo configuration. The canonical CPU
build and preserved baseline-era fingerprints contain one copy. Repeated lint
allowance and native CPU settings request the same values, but the linker
argument is additive: the GPU candidate passes `-fuse-ld=mold` twice. Thus the
baseline/candidate build argument lists are not literally identical. No
controlled three-versus-six-flag build comparison or binary-equivalence check
was performed, and no effect is attributed to the duplication. Raw fingerprint
evidence and its coverage limitations are recorded in
`final-gpu-validation/build-rustflags-provenance.json`.
