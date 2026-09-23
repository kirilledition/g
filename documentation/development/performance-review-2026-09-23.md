# Performance review and optimization — 2026-09-23

The frozen comparison source is `f7acead8`. This work follows the correctness
review and preserves its statistical policies, per-phenotype sample sets,
source validation, and output/resume contracts. Historical July profiles guide
the experiments but are not performance measurements of this source.

## Work and ownership

| Workstream | Implementation owner | Experiment and acceptance condition |
| --- | --- | --- |
| BGEN index reuse | Genotype worker | Bounded process-local reuse of a verified source index; cold/hit timing, source invalidation, concurrency, and memory bounds. |
| LOCO indexing and fingerprinting | Input worker | Produce the exact whole-file hash during indexing; avoid rereading verified sources without weakening replacement or mutation checks. |
| Repeated phenotype-group preparation | Scheduler worker | Reuse one compressed layout plan only for identical ordered pending chunk geometry; test different missingness masks and resumed subsets. |
| Null-Firth accepted components | Numerical worker | Carry accepted calculations into the next iteration; preserve convergence and failure behavior and measure preparation cost. |
| Quantitative packed8 moments | Linear worker | Determine whether existing moments can replace the shifted-square reduction without changing rounded-dosage semantics. |
| Output completion | Output worker | Benchmark small encoding or scheduling changes against ready-all and paced-finish workloads; retain no unmeasured writer rewrite. |

Independent reviewers inspect retained changes. Cleanup is limited to removing
redundant work and clarifying ownership in the affected paths.

## Measurement protocol

- Use Slurm compute nodes: landau for GPU and an available full CPU allocation
  for CPU builds, tests, and benchmarks. Nix is unavailable on this server.
- Keep the baseline immutable and use separate output directories. Record the
  source revision, native library hash, environment, inputs, and configuration.
- Separate fresh-process, populated-cache, and hot same-process results.
  Production headlines use telemetry off; stage timing is diagnostic.
- Use paired or interleaved baseline/candidate measurements for retained
  optimizations. A kernel or stage gain alone is not an application speedup.
- Keep numerical tolerances fixed. Run upstream quantitative, binary score,
  and approximate-Firth qualification alongside relevant regression tests.
- Measure memory and workload dependence; the 2,504-person chromosome-22
  fixture cannot alone establish performance at biobank scale.
- Store generated inputs, traces, and benchmark reports in ignored directories.

## Scope decisions

Full cross-group genotype reuse is not a local cache change. Current execution
finishes one aligned group before starting another, so a small chunk cache
would miss during the next complete scan. Chunk-major scheduling would need a
new state and memory policy. This campaign instead tests sharing immutable
compressed layout plans, without retaining all groups' device states.

Raw integer first and second moments do not uniquely determine the squared
values after float32 dosage decoding and allele shifting. For example,
numerators `[500, 503, 503]` and `[501, 501, 504]` share integer sum and square
sum but have different rounded shifted-square sums. The original quantitative
shortcut therefore cannot be retained as proposed. A decoder fusion would need
separate CUDA/ABI and numerical qualification and a measured benefit.

## Results

Three changes are retained: bounded BGEN index reuse, single-pass LOCO
fingerprinting, and reuse of one compressed layout plan. Each has independent
code review and targeted correctness coverage. Numerical shortcuts and the
output encoding prototype were excluded for the reasons below.

Validation exposed one cache hazard on the cluster filesystem: an in-place
same-size edit with restored modification time can retain the same change time
within a one-second timestamp interval. Cache admission therefore needs an
observed stability interval and a fresh parse before promotion; simply adding
a sleep to the regression would conceal the stale-index problem.

The 100,000-sample, four-trait synthetic null-Firth workload started from zero
converges for three of four traits in the frozen baseline. Reports retain that outcome and
separate their timings from successful fits. The optimization must preserve
convergence flags as well as numerical results; parameters and tolerances are
unchanged.

Accepted-component reuse in null Firth was rejected despite preliminary speed
gains. For the 100,000-sample, four-trait workload started from fitted logistic
coefficients, the baseline converged for all traits but the candidate failed
one trait. Its coefficient difference was only about `9.74e-9`, yet the reported
likelihood became NaN. Sensitivity of strict line-search comparisons near
floating-point resolution is a likely cause; it was not isolated further. No tolerance
or convergence-policy change was introduced to retain the optimization; the
production numerical implementation remains unchanged.

The output correction-code decoding prototype was removed. Initial writer
timings were mixed, and their executable provenance was not qualified after
discovery of a shared Cargo target-directory collision in the BGEN campaign.
They do not establish a prototype speedup or regression. The patch and raw
measurements are retained as ignored experiment artifacts; the production
writer remains unchanged.

### LOCO input preparation

On cantor, the public input/output API harness ran in baseline/candidate/
candidate/baseline order, with seven timed repetitions per workload in each
position and one unmeasured warm-up. All 84 measurements matched sample counts,
file sizes, and independent source SHA-256 values. The standalone harnesses
resolved identical dependency versions, checksums, and dependency edges.

| Workload | Baseline pooled median | Candidate pooled median | Time reduction |
| --- | ---: | ---: | ---: |
| Synthetic, 100,000 samples, 23 chromosome rows | 1.162 s | 0.768 s | 33.9% |
| Synthetic, 500,000 samples, 23 chromosome rows | 5.623 s | 3.662 s | 34.9% |
| Existing real LOCO file | 34.762 ms | 31.435 ms | 9.6% |

These measurements include alignment/indexing and manifest fingerprinting for
one trait, using a warm filesystem cache. They exclude association computation
and result writing. Scaled baseline drift was below 0.25% across the two
positions; real-file drift was about 11%, so its smaller improvement is less
conclusive. The scaled results support retaining the single-pass fingerprint
change without claiming an equivalent whole-scan improvement.

### Build provenance

Alternating baseline and candidate workspace builds in one Cargo target
directory reused an older BGEN benchmark executable. Those measurements and
the associated memory probe are excluded. Qualified BGEN measurements and
native application builds use isolated target directories. The LOCO harnesses
have different package and executable names; their symbols, embedded source
paths, dependency-file paths, and Cargo fingerprints independently confirm
the intended source for each retained measurement.

### Quantitative moment experiment

A landau GPU probe supplied the existing rounded shifted moments for free,
removing only the association kernel's reduction. Its three shape medians
changed by approximately 0.5%, 0.2%, and 3.6%, with substantial timing
variability. All four output statistics matched exactly in that optimistic
experiment. It excludes the cost of producing those moments in the decoder
and excludes the rest of the application; it is not an implemented speedup.
Together with the compiled integer-moment collision, this does not justify a
CUDA ABI/PTX fusion in this campaign.

### BGEN index reuse

The qualified cantor experiment used separate Cargo targets and
baseline/candidate/candidate/baseline ordering, with 20 Criterion samples per
case. It opened the 418,943-variant, 2,504-sample chromosome-22 source after
filesystem warm-up. The following are averages of the two run estimates, not
cold-disk timings:

| Operation | Time | Compared with baseline indexing |
| --- | ---: | ---: |
| Baseline indexing | 67.962 ms | — |
| Candidate index miss | 70.799 ms | 4.2% more time |
| Candidate ready-cache reopen | 5.001 ms | 92.6% less time |

Baseline run estimates drifted by 1.5%; candidate miss estimates by 0.4% and
hit estimates by 2.1%. The miss overhead was consistent in both positions.
The cache therefore benefits repeated scans of the same input; opening only
once or alternating inputs incurs its small setup cost without a hit benefit.
Admission does not wait in production: it reparses until a stable identity has
been observed for two seconds, then promotes a fresh parse.

The isolated memory probe retained one extra descriptor. Cache-only process
RSS was 30.30 MiB versus 2.49 MiB during probation, remained unchanged over
100 reopens, and fell to 2.52 MiB after eviction. These are observed process
residency figures, not the estimated-allocation budget or an allocator release
guarantee. The probe's 100 ready reopens took 387 ms; its separate promotion
parse and explicit two-second wait were excluded from those reopen timings.

### Compressed layout reuse

The real chromosome-22 stage benchmark alternated fresh planning and reuse
over 31 repetitions. Both guarded paths acquired the source read-session
guard and prepared sample selection for every group, as production does.
The fixture used 26 chunks and 2,379 selected samples. The compared groups had
identical ordered pending chunks; no genotype data or statistical state was cached.

| Groups | Fresh guarded preparation median | Reused guarded preparation median | Time reduction |
| --- | ---: | ---: | ---: |
| 1 | 4.988 ms | 5.193 ms | No improvement |
| 8 | 37.885 ms | 21.025 ms | 44.5% |
| 32 | 156.401 ms | 89.342 ms | 42.9% |

Reuse won 28 of 31 paired repetitions with eight groups and 30 of 31 with
32 groups. Median paired savings were 14.817 ms and 58.854 ms, respectively;
these differ from subtraction of the independent medians above. Excluding
the guard, planning medians fell from 8.103 to 1.522 ms for eight groups and
from 29.281 to 1.417 ms for 32 groups. An already populated cache lookup
measured about 99 ns per call in 10,000-call bundles. The single-group case
has no reuse benefit. These stage timings exclude decoding, association
computation, and output, and do not imply comparable whole-run gains.
Guarded timings drifted substantially within the run; both execution-order
strata still favored reuse for eight and 32 groups. The pure-planning paired
medians saved 6.215 ms and 27.041 ms, with all 31 pairs favoring reuse.
The apparent single-group penalty is inconclusive because its distributions
overlap. Hot-loop lookup cost is not production group latency.

### Correctness qualification

The retained candidate passed 187 focused GPU tests on landau and all three
required-fixture upstream REGENIE parity cases: quantitative, binary score,
and approximate Firth. Production numerical kernels, convergence policies,
and the Parquet writer are unchanged from `f7acead8`.

Final integration review checked the earlier LOCO content snapshot against
resume semantics. The whole-file digest and indexed row digests describe the
same indexing read; deferred requested rows must match those digests, and
completed outputs match that indexed snapshot. Metadata checks cannot detect
every concurrent same-size edit with restored modification time and unchanged
coarse change time. This moves the snapshot point earlier without introducing
a numerical or provenance mismatch, and does not create an atomic filesystem
snapshot guarantee. Input immutability during execution remains required.

The standard `just check` lane passed native lint, Rust Clippy, Python lint
and typing, binding checks, and architecture/Justfile guardrails. `just test`
passed all 294 CPU tests. The Rust workspace suite passed during integration;
the final input/output rerun passed 117 tests, and all 48 genotype tests passed
after the final cache-ordering change. The real-data layout benchmark also
ran successfully. Documentation is checked with `just docs-build`.
Nix was unavailable, so all heavy checks used the documented server environment
on Slurm compute nodes rather than the login node.

### Full binary application

On landau, the unchanged hot-run harness executed baseline/candidate/candidate/
baseline, with 12 measured hot scans in each position and telemetry disabled.
The fixture contained 418,943 variants and 2,504 source samples, used 16,384-row
chunks, and ran the binary GPU pipeline with approximate Firth enabled at
`p_threshold=0.05`. Each position had a separate initially empty JAX cache,
one discarded compilation/warm-up scan, and separate complete output datasets.

| Position | Source | Median hot scan |
| --- | --- | ---: |
| 1 | Baseline | 0.635949 s |
| 2 | Candidate | 0.559730 s |
| 3 | Candidate | 0.561642 s |
| 4 | Baseline | 0.636496 s |

Pooling 24 hot scans per source gives **0.635954 s to 0.560595 s**, or **11.8%
less elapsed time**. Baseline round medians differed by 0.1% and candidate
round medians by 0.3%. These are complete same-process scans, including output,
not isolated kernel timings.

The separate fresh-process diagnostics did not improve: native execution was
20.158/20.257 s for baseline and 20.535/20.632 s for candidate; process wall time
was 22.582/22.749 s and 22.875/23.115 s, respectively. Each used its populated
persistent JAX cache but had no process-local BGEN index. With two diagnostics
per source, these establish no startup improvement; the hot result must not
be applied to independent CLI invocations. Discarded first scans took
45.8–53.7 s and are not a startup performance claim.

Recorded native SHA-256 values were identical within each source's two rounds:

- Baseline: `a1b20b101622d8ab141733b57ba81b12b372e9dc3f1ad07a5bb507d23bf9aada`.
- Candidate: `de428ebf65cd10546f3b0ca34e557899f6abd4a39d295f0d11a88f3e67eb0d2e`.

### Multiple sample masks

A supplementary quantitative GPU matrix used the same real genotypes with
synthetic traits and 1, 8, or 32 distinct shifted missingness masks. Each trait
selected 2,379 samples. Each configuration ran baseline then candidate, with
one discarded warm-up and three measured complete scans, telemetry off.

| Distinct groups | Baseline hot median | Candidate hot median | Observed reduction |
| --- | ---: | ---: | ---: |
| 1 | 0.528570 s | 0.431570 s | 18.4% |
| 8 | 2.395384 s | 2.310832 s | 3.5% |
| 32 | 8.873790 s | 8.804750 s | 0.8% |

These are exploratory timings with three repetitions and fixed source order,
not a balanced speed qualification. In particular, the small 32-group change
does not establish an application improvement. Every group still reads,
decodes, computes, and writes its own results, so modest planning savings do
not remove the dominant repeated work. The first timed candidate scan can
still promote the BGEN index; all configured timed scans are included.

The first hot outputs matched baseline bit for bit across all 41 traits,
17,176,663 rows, and 14 columns. Comparisons included null positions, exact
floating-point bits, row order, schemas, stable footer metadata, logical chunk
commits, and execution-plan metadata; differing physical part grouping was
allowed. Both binary benchmark pairs also passed the same exact comparison.

### Local evidence

Reports, native artifacts, source manifests, and raw measurements are under
the ignored `results/perf-20260923/` directory. `worktree-artifacts/` preserves
the standalone LOCO harness, layout-stage summarizer, multi-mask fixture
generator/driver/comparator, and rejected experiments. Generated data and
benchmark outputs are not committed. The maintained BGEN Criterion cases,
BGEN memory probe, and ignored real-data compressed-layout test remain in
the source tree for future measurements.
