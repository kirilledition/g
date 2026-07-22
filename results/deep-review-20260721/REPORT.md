# Deep application review

Date: 2026-07-21

Frozen revision: `29284ac6e87c74f3b2e5e9d3a1530ee934425cff` (`main == origin/main`)

Native extension SHA-256: `f39ae3cfde96ea79246280abb436ccd6337a85bf67473e3b41e5d85f5245f256`

Cargo lock SHA-256: `c4563d2ef46f7bb4646a9f145b6f723df6f4b10cb7577901290f7bf7559a47a8`
uv lock SHA-256: `bbe1d9f7ec0389dfc84ed5d9ca1b308c7657327b45807c9b49fc42f9eb214de2`

## Executive conclusion

The computational architecture is substantially healthier than its raw size suggests: the Cargo crate graph is acyclic, the Python import graph has no cycles, production lint suppressions are justified, the supported public surface is narrow, bounded pipeline queues and panic propagation are generally sound, current CI is green, and the full chromosome-22 path is very fast once resident.

The application is not ready for an external production release yet. Three areas should block release:

1. the safe BGEN reader exposes a file-backed mmap without an enforceable immutability contract, which violates `memmap2`'s safety requirements if another process modifies or truncates the file;
2. null-Firth score-increase tracking diverges from upstream REGENIE and can reject a converged iterate, trigger unnecessary fallback, or produce `FIRTH_FAILED`; and
3. output ownership/recovery is not transactional: preparation can poison a fresh path, two processes can own one run, finish can race an in-flight producer, and rename sequences are not crash-durable.

These do not imply a whole-application rewrite. The expected work is one small numerical correction, a focused internal redesign of output ownership/state transitions, and a performance-gated BGEN I/O safety decision. The broader crate/API boundaries can remain.

## Ranked findings

| Priority | Area | Finding | Why it matters | Recommended disposition |
| --- | --- | --- | --- | --- |
| Blocker | Rust soundness | Safe `BgenReaderCore::open` creates a file-backed mmap without guaranteeing the source stays immutable. | Concurrent mutation can cause undefined behavior; truncation can also SIGBUS. Metadata checks are only TOCTOU detection. | Define and enforce an immutable-file contract immediately; benchmark replacing the map with positional reads into reader-owned buffers as the durable safe design. |
| Blocker | Scientific correctness | Null-Firth stores the historical minimum score instead of the immediately previous score. | It disagrees with REGENIE's consecutive-increase rule and can reject an iterate REGENIE accepts. | Store the current score; add a deterministic state-transition test and an upstream-REGENIE numerical oracle using absolute tolerances. |
| Blocker | Output integrity | Output preparation, ownership, writer closure, and persistence are not one atomic state machine. | Normal validation failures can poison a path; concurrent processes can overwrite one another; close can miss a producer; a crash can leave manifest/data durability mismatched. | Add exclusive run ownership, transactional initialization/rollback, an atomic open/enqueue/close protocol, and file-plus-directory fsync ordering. |
| High | Release/supply chain | Mutable Actions tags and unrestricted Dependabot auto-merge coexist with a manual-only protected parity gate. | A JAX/CUDA/Arrow or build-action update can reach main without the scientific/GPU qualification used for releases. | Pin actions to audited SHAs and require recorded on-prem parity for runtime/native dependency updates. |
| High | CUDA correctness/provenance | No automated test actually executes the raw-CUDA Firth target; packed8 CUDA does not enforce its documented source/PTX hashes. | Independently maintained JAX/CUDA formulas and committed GPU artifacts can drift silently. | Add a serialized Landau differential test and give genotype CUDA the same build-time hash enforcement as compute CUDA. |
| Medium-high | Native ABI | Vendored OpenXLA FFI headers are tied to `jaxlib 0.11.0`, while Python accepts any `0.11.x`. | A supposedly compatible dependency update can change the native ABI assumptions. | Pin the exact supported JAX/JAXlib pair or add an explicit compatibility matrix and registration/execution gate. |
| Medium | Safe API invariants | Public metadata constructors validate only with `debug_assert!`. | Safe release callers can construct invalid ranges/codes/offsets and later panic. | Return `Result` from validated constructors or make a narrowly documented unchecked constructor private/internal. |
| Medium | Failure semantics | Late signals can replace the true backend/output error; telemetry failures can fail completed work or permanently strand JAX setup in `Configuring`. | Diagnostics can hide the root failure and an optional observer can disable subsequent work. | Preserve first-error precedence and isolate telemetry from computation/state-commit boundaries. |
| Medium | Observability | Raw-CUDA selection collapses capability/registration failures to a cached boolean; logging subscriber installation failure is also reduced to a success-like state. | The app can silently lose the measured CUDA gain or configured logging. | Record selected implementation and fallback reason once; surface subscriber non-installation explicitly. |
| Medium | Test/release gates | Coverage is advisory and full protected-data parity is manual rather than a required release check. | A green aggregate CI run does not prove coverage generation or chr22 scientific parity. | Keep GitHub CI lightweight, but require a signed/recorded on-prem parity result for release candidates. |
| Medium | Build economy | The full Arrow meta-crate defaults pull CSV/JSON/IPC; uv's default dev group makes isolated jobs install most of the GPU/dev stack. | Large compile, cache, and installation cost with no expected hot-path benefit. | Disable Arrow defaults/evaluate direct subcrates; split dependency groups and use `uv sync --only-group` in isolated jobs. |
| Medium | Internal architecture | `g-output` has the largest source-level dependency SCC, matching the ownership bugs; genotype contracts also point back into BGEN-specific types. | Responsibilities are harder to reason about even though the crate DAG is acyclic. | Move persistence/session contracts to neutral owner modules and split generic genotype pools/contracts from BGEN implementations. |
| Low-medium | Operations | Cancellation is checked only between chunks; worker drops synchronously join. | A hung CUDA/JAX call or BeeGFS write cannot honor graceful cancellation. | Document the limit; add bounded cancellation only if the backend/filesystem provides a safe mechanism. |
| Low-medium | Static policy | Ruff replaces default excludes and excludes all `scripts/`; Skylos is manual and product-only. | Local `.git` files can make Ruff nondeterministic while maintained scripts are not linted. | Use `extend-exclude`, choose an explicit scripts policy, and retain a no-upload advisory whole-repository scan. |

## Release-blocking evidence

### 1. BGEN mmap safety

[`BgenReaderCore::open`](../../crates/genotype/src/bgen/reader.rs) is a safe public function. At line 55 it calls `unsafe { MmapOptions::new().map(&source.file) }` without a safety argument that the application can enforce. The locked `memmap2 0.9.11` documentation states that all file-backed mapping constructors are unsafe because later modification in or out of process can produce undefined behavior.

The source fingerprint checks at reader lines 93-97 and around compatibility/delivery can detect some before/after changes. They cannot prevent a mutation while bytes are read, a truncation fault, or a mixed snapshot. The public input documentation currently overclaims that a changed source “fails” at [`documentation/public/input-files.md:55`](../../documentation/public/input-files.md).

The internal Rust lifetimes are otherwise correct: the reader owns both the file and map, sessions borrow the reader, and decoded/packed payloads use owned buffers. The defect is specifically the external mutation contract.

Short-term containment:

- document that the opened BGEN must be immutable for the complete reader lifetime;
- reject obviously unsafe permissions/ownership where practical and use a cooperative lock if the deployment controls writers;
- place a precise `SAFETY` comment before the block and enable the unsafe-documentation lint.

Those measures do not make an uncooperative external writer safe. The durable option is a benchmarked `pread`/positioned-read path into long-lived reader-owned buffers, or a truly immutable snapshot owned by the process.

### 2. Null-Firth consecutive-increase semantics

The production loop compares the new maximum score to `state.previous_score_maximum` at [`null.py:168`](../../src/g/compute/regenie2_binary/firth/null.py), then stores `minimum(current, previous)` at line 174. The field therefore becomes a historical minimum. Upstream REGENIE compares to the immediately preceding score and unconditionally assigns `score_max_old = score_max_new` at [`Step2_Models.cpp:1473`](../../reference/regenie-patched/src/Step2_Models.cpp).

The short trajectory `10, 1, 2, 1.5` gives counts `0, 0, 1, 2` in current g and `0, 0, 1, 0` in REGENIE. A 27-iteration alternating trajectory can make current g reach count 26 on a score below tolerance. g then masks convergence and marks failure, while REGENIE breaks for convergence before applying the increase heuristic.

This reaches production behavior. The first three null-model attempts enable the check; the final fallback disables it. The guaranteed consequence is unnecessary fallback and discarding a result upstream accepts. If the replacement path differs or fails, final statistics/status differ; a failed null likelihood becomes a `null_failed_mask` and candidates become `FIRTH_FAILED` with NaN statistics.

The full chr22 qualification did not exercise this trajectory, so its success does not disprove the defect. Current tests cover one regular convergence case and a zero-iteration failure, not non-monotone score tracking. The appropriate oracle is upstream REGENIE output, comparing numeric fields with `abs(value1 - value2) < tolerance`; old-g byte parity should remain diagnostic only.

### 3. Output lifecycle and durability

The output issues share one root cause: writer sessions and persistent run state do not have one exclusive transaction/state owner.

- **Poisoned fresh path.** [`OutputManager::open`](../../crates/output/src/manager.rs) prepares phenotype directories sequentially. [`prepare_output_run`](../../crates/output/src/manifest/run.rs) creates `parts/` before a manifest exists. Engine output opens before BGEN/input preparation. A normal later validation error therefore leaves a non-empty run directory that fresh mode rejects and resume mode rejects because no manifest exists. Multi-phenotype failures can strand only a prefix of runs.
- **No process ownership.** Manifest synchronization is a process-local `Mutex`; temp manifest and Parquet names are deterministic. There is no lock file, `O_EXCL`, or `flock`. Two jobs using the same output root can truncate temp files, overwrite parts, or lose commits.
- **Close/enqueue race.** [`writer_session.rs:112`](../../crates/output/src/session/writer_session.rs) removes a full batch from `pending_chunks`, unlocks, and only then increments the completion tracker in [`worker_pool.rs:65`](../../crates/output/src/session/worker_pool.rs). `finish` or `abort` can close and observe zero work in that gap, after which the producer enqueues an orphan part. This is reachable through the public delivery-state/free-write API even though the current engine serializes completion.
- **Incomplete crash durability.** A Parquet file is closed and renamed without syncing the file and `parts/` directory. The manifest temp file is synced, but its containing directory is not synced after rename. Power/node loss can leave a durable manifest referring to a lost part or lose the manifest rename. Strict resume then rejects rather than repairs the mismatch.
- **Silent missing manifest.** Lifecycle updates return success when `run_manifest.json` disappeared, so a completed call can report success after external deletion/concurrent cleanup.

The fix should be designed and tested as one protocol: acquire exclusive ownership, validate the complete run plan, create an initialization marker/manifest transactionally, make delivery and terminal transition mutually exclusive, durably publish each part before its manifest commit, durably publish terminal manifest state, and define recovery for every intermediate state.

## Compute and CUDA review

The raw-CUDA Firth integration is architecturally well placed in `g-compute-cuda`: it owns its driver adapter, FFI handler, PTX, and pure-JAX fallback. Shape/dtype/status validation and supplied-stream use were manually traced and no lifetime defect was found.

Remaining issues:

- [`tests/test_regenie2_binary_cuda_firth.py:53`](../../tests/test_regenie2_binary_cuda_firth.py) explicitly disables CUDA. Its current checks only evaluate abstract shapes. The independently maintained formulas in [`scalar_approx.py`](../../src/g/compute/regenie2_binary/firth/scalar_approx.py) and [`firth_components_kernel.cu`](../../crates/compute-cuda/native/firth_components_kernel.cu) need a real-device differential test covering prefix sizes, masks, clipping boundaries, invalid information, and registration/execution failures.
- [`src/binding/engine.rs:161`](../../src/binding/engine.rs) converts rich capability and registration errors into a process-lifetime cached `false`. Fallback is safe, but silent. Run diagnostics should record raw CUDA/JAX selection and reason.
- Firth's build script verifies maintained source/PTX hashes. [`crates/genotype-cuda/build.rs`](../../crates/genotype-cuda/build.rs) embeds packed8 PTX but does not enforce the hashes documented in its provenance README.
- Partial multi-trait resume selects active traits only at materialization. Already-completed traits are still scored and Firth-corrected, and capacity is sized for the full trait count. This is a low-priority optimization and needs a shape/cache benchmark before an API change.

## Architecture review

### Healthy boundaries

- Cargo workspace dependency graph: acyclic.
- Python graph: 35 modules, 114 static edges, zero cycles.
- Repository architecture checks pass for both languages.
- No duplicate canonical cross-crate contract type or convenience re-export was found.
- Engine/runner are high-coupling coordinators by responsibility, not accidental utility buckets.
- The public Python surface remains `g.cli` plus `_core.cli.run`.
- Bounded engine queues (1/2/2), backpressure, compute/materialization panic capture, first-error abort, and worker completion guards are directly tested and manually verified.

### Source-level strongly connected clusters

These are module ownership smells, not Cargo dependency cycles:

1. `g-output`: manifest/resume, manifest/writer, writer/timing, and writer-session/worker-pool pairs reference each other. This is the only cluster that aligns with high-severity runtime defects. A neutral persistence contract and one session-state owner would simplify it.
2. `g-genotype`: generic `common` contracts embed `bgen::CompressedPacked8Batch`, while BGEN consumes generic pools/contracts. `packed8_cache` also accepts the full reader only to obtain counts. Split neutral delivery contracts and pass the narrow counts.
3. Small low-risk pairs exist in interface partial/resolved/default handling, runtime telemetry writer/session, and input error/REGENIE handling.

Large hot files deserve caution, not automatic splitting: `genotype/src/bgen/simd.rs` is about 850 lines, the scheduler about 748, runner `run.rs` about 734, binding `engine.rs` about 636, and Python scalar Firth about 581. Skylos flags seven 100+ line JAX functions and the scalar-Firth module as a god-file candidate. Several nested functions deliberately define JAX control flow; purely metric-driven extraction can change StableHLO, cache size, and hot time. Refactor only behind the existing focused and whole-app gates.

## Static-analysis adjudication

The retained no-upload Skylos 4.29 artifact is [`skylos-static.json`](./skylos-static.json). It reports 21 functions, 128 imports, three classes, and two parameters as unused. Manual tracing finds no dead production item:

- Rust candidates have direct, trait-dispatch, facade, generated, cfg, or benchmark consumers; Clippy also accepts the imports.
- backend Python classes are constructed by exact-name `getattr` from Rust;
- the two `carry` parameters are required JAX callback positions;
- PyO3 registrations and XLA FFI entry points are dynamically consumed.

The raw Skylos grade (`F`, 55) must not be used as a release signal. Its security phase included `.tools`, `target`, and reference/worktree copies despite configured exclusions. Most of its 2,947 danger/security records are therefore third-party duplicates. The 38 owned-code path-taint reports model direct local CLI file paths as privileged untrusted sinks; canonicalization would change symlink/user-path semantics and does not create a security boundary. The manually found output ownership defects above are the actionable filesystem issues.

Valid Skylos/static signals were corroborated separately: mutable workflow tags, broad docs permissions, long hot functions, and a few tightly coupled stable modules. It found no secret. Its zero dependency-vulnerability count is not a CVE attestation because no external network vulnerability database scan was completed.

Suppressions were exhaustively reviewed:

- 26 Clippy `allow` sites are justified by ABI, JAX/CUDA, benchmark, generated-layout, or measured hot-boundary constraints.
- The benchmark-only `dead_code` allowance is justified.
- Four Python type suppressions and six Ruff policy ignores are justified.
- Supplemental unsafe linting emits 30 diagnostics. Most blocks already have a local invariant comment in a position Clippy does not recognize, or inherit a documented unsafe-function precondition. The BGEN mmap is the substantive exception. After fixing it, move comments to canonical `SAFETY` positions and enable `clippy::undocumented_unsafe_blocks` as a gate.
- No `TODO`, `FIXME`, `unimplemented!`, or verified removable wrapper survives the production call-graph review.

One Ruff configuration defect is real but low priority: a custom `exclude` replaces Ruff defaults, so a local `.git/logs/...py` file can make `ruff check .` fail, while all maintained `scripts/` are excluded. Use `extend-exclude` for repository exclusions and decide explicitly whether scripts are linted.

## Dependencies, CI, and build reproducibility

### High-value fixes

- Every workflow action uses a mutable tag, including checkout, setup, cache, artifact, Rust toolchain, and cargo-tool installers. Pin full commit SHAs and let Dependabot update the pins.
- Dependabot's generic auto-merge workflow has no ecosystem/semver risk split. PR CI deliberately excludes protected chr22 parity. Runtime/native dependency changes should require the on-prem GPU/parity result before merge.
- Vendored OpenXLA headers document exact `jaxlib 0.11.0`; `pyproject.toml` currently allows all `0.11.x`. Couple these explicitly.
- Docs grants Pages write and OIDC permissions at workflow scope. Limit them to deployment.
- Root Cargo package does not inherit the workspace's `publish = false`; add it before release work.
- Isolated `uv build` permits any Maturin 1.x and CI supplies no build constraint. Pin or hash-constrain the backend.

### Bloat and healthy controls

- Runtime Python dependencies are lean in count: JAX/CUDA, NumPy, and nvCOMP. Their binary payload is necessarily large.
- The monolithic default dev group is not lean. Offline dry runs found docs syncing 136 packages versus 25 with `--only-group docs`, and CUDA format syncing 119 versus one with `--only-group cuda-format`.
- Arrow defaults activate CSV, JSON, and IPC even though output uses arrays/buffers/schema plus Parquet. `default-features = false` should remove CSV/JSON; direct Arrow subcrates may reduce more. Parquet is already minimized to Arrow+Zstd.
- The exact extension is 50,492,496 bytes with debuginfo; `.text` is 6,119,689 bytes. Debug sections dominate the file. Embedded Firth and packed8 PTX are about 37.7 KiB and 43.5 KiB. This is a measurement baseline, not a deployment-size regression.
- Active duplicate Rust versions are only `getrandom` 0.3/0.4 and `syn` 2/3. They come from unrelated runtime ecosystems and proc-macro/build use; do not churn them.
- Cargo/uv locks validate offline, all registry artifacts are hashed, no Git dependency was found, Parquet defaults are already disabled, and vendored OpenXLA hashes match.

## Current performance characterization

Primary workload: one binary trait, approximate Firth, full 1KG chromosome 22 (418,943 variants), packed8 GPU delivery, 16,384-variant chunks, Firth batch size 512, capacity 1,024, eight writer threads, four direct Parquet parts, telemetry off. JAX/JAXlib are 0.11.0; the GPU is a V100 exposed as one addressable JAX device.

Exact-HEAD evidence is in [`current-headline/summary.json`](./current-headline/summary.json) and the preceding one-run [`current-hot-smoke/summary.json`](./current-hot-smoke/summary.json).

| State | Timing contract | Elapsed |
| --- | --- | ---: |
| First cold diagnostic | First observed exact-HEAD empty-JAX-cache lifecycle; likely also colder lower-level compiler/driver caches; one sample | 81.259 s |
| Empty application cache | Empty JAX persistent cache after lower-level caches had been exercised; discarded lifecycle | 32.164 s |
| Fresh process, populated cache | Native call / complete child process | 14.155 s / 16.577 s |
| Same-process hot production | Five telemetry-off lifecycles | median 0.607857 s; min 0.590128; max 0.611276 |
| Upstream REGENIE v4.1 CPU | Separate exclusive CPU node, 39 threads, stable fresh-process median | 10.393 s |

The state contract matters. Stable REGENIE fresh divided by resident g hot is about 17.1×, but populated-cache fresh-process g is about 1.36× slower inside the native call (1.60× including process startup). Cold g is much slower. Do not describe the app as “17× faster” without saying it is resident-hot GPU versus fresh CPU.

All current trials exited successfully, produced 418,943 rows, four parts, 10,767,971 bytes, and the same diagnostic Parquet hash. The nine-file, 1.14 MB JAX cache tree remained byte-identical during measurement. Output hashes prove run-to-run determinism for this build; they are not the primary scientific oracle.

The five-run sample is adequate for characterization, not a historical speed claim. The most recent position-balanced JAX-0.11 gate was statistically neutral at 0.615272 versus 0.615521 seconds. The accepted raw-CUDA Firth production gate previously improved whole-app hot time from 0.607211 to 0.597798 seconds, a paired geometric 2.641% with positive pair and block intervals. Current 0.608-second behavior should not be compared causally to that campaign without a new ABBA baseline/candidate gate.

### Current stage evidence

The current instrumented fresh-process diagnostic reports:

- JAX runtime configuration: 10.106 s;
- backend initialization: 0.246 s;
- native preparation: 0.175 s;
- native execution: 3.710 s;
- aggregate writer work across four parallel files: 0.361 s;
- terminal output finish: 0.0496 s;
- Arrow array bytes processed by writers: 31.94 MB;
- Parquet output: 10.77 MB.

These diagnostic totals are perturbed and overlapping; they are not additive hot wall percentages. The last complete device timeline predates current JAX/output changes but remains useful for the unchanged genotype delivery kernels: 216.309 ms total kernels, 142.258 ms nvCOMP inflate, 11.819 ms packed8 finalization, 13.777 ms H2D, and 5.209 ms D2H. It found no useful immediate auxiliary-stream overlap window. The current stage timing confirms output remains material, but terminal drain is lower than the older 126.7 ms profile.

The next optimization wave should therefore retain the existing order:

1. refresh a bounded current device timeline before changing compute;
2. benchmark BGEN safe I/O/index/packing changes necessitated by mmap safety;
3. optimize output only after correctness/transaction ownership is fixed, using both ready-all and GPU-paced finish; and
4. do not revisit larger chunks, an immediate auxiliary stream, or another DEFLATE decoder without new evidence overturning prior negative experiments.

## Tests and correctness gates

Current health:

- GitHub PR CI run `29828162301`: all 15 jobs passed at the frozen revision.
- Python: 179 passed; branch-aware total coverage 97% (1,444 statements, 31 missed, 90 branches, 10 partial).
- Rust tests: passed; aggregate LLVM coverage was 80.84% lines, 80.47% regions, and 75.35% functions.
- Full upstream-REGENIE chr22 qualification exists for quantitative, binary score, and approximate Firth. Binary approximate Firth checks all 418,943 rows, 17,938 corrections, exact keys/counts/classifications, and numeric absolute tolerances.

Important gaps:

- root PyO3 binding Rust files show 0% in cargo-llvm-cov because pytest loads a separately built extension;
- the CUDA Firth kernel is not executed by automated tests;
- no test covers non-monotone null-Firth score tracking;
- no multiprocess same-root, close/enqueue race, power-cut durability, or failed-preparation retry test exists;
- no late-signal-versus-primary-error or telemetry-isolation orchestration test exists;
- coverage jobs use `continue-on-error`, fail-under is zero, and aggregate CI does not require coverage;
- protected-data parity is a manual Slurm recipe rather than an in-repository required check.

The numerical policy should remain: upstream REGENIE is the primary oracle, and numeric equality means `abs(value1 - value2) < tolerance`. Comparisons to a previous g build and Parquet byte hashes are useful secondary regression diagnostics only.

## Other failure-path findings

- Runner's second interruption check can overwrite an already established backend/output failure, even though engine cleanup already chose abort versus interrupted flush from the original result.
- Enabled progress/telemetry errors propagate into delivery and can turn an already-completed output into CLI failure. During first JAX setup, a diagnostic telemetry error can leave global state permanently `Configuring`.
- `tracing_subscriber::try_init` failure is reduced to a boolean and the logger state still reports initialized, so embedding under an existing subscriber can silently omit requested sinks.
- Manifest lifecycle update helpers silently return success if the manifest disappeared.
- Output worker pool construction returns `Result` but uses panicking `std::thread::spawn`; pool drop ignores worker join panics.
- Graceful cancellation cannot interrupt an active backend/kernel/filesystem call; the documented second SIGTERM remains the hard escape.

These are real but do not call for a coordinator rewrite. Preserve primary-error precedence, make observer failure non-fatal at committed boundaries, use `thread::Builder::spawn`, and add focused state-machine tests.

## Recommended implementation order

1. Correct null-Firth previous-score assignment and add the deterministic transition test plus REGENIE-oracle coverage.
2. Decide and implement the BGEN safety contract. If changing mmap to positional reads, protect the current 0.608-second hot gate and focused reader throughput with alternating benchmarks.
3. Redesign output run ownership/initialization/closure/durability together; add failure injection and multiprocess tests before optimizing it further.
4. Add real-device CUDA differential coverage, packed8 PTX digest enforcement, and explicit CUDA fallback diagnostics.
5. Harden release provenance: pinned action SHAs, restricted dependency auto-merge, exact JAX/OpenXLA compatibility, required recorded on-prem parity.
6. Make coverage meaningful for bindings and failure paths; then raise thresholds gradually from the measured baseline.
7. Perform build-only cleanup (Arrow defaults, uv groups, Maturin constraint, root `publish=false`, Ruff excludes). Do not claim runtime speedups from these changes.
8. Only then start another profile-led optimization wave.

## Tools and limitations

Used: repository architecture checks, Ruff (including security/performance/argument families), ty, Clippy plus unsafe/complexity/unwrap supplements, cargo-machete, Cargo feature/duplicate trees, uv lock/tree checks, Vulture/manual call tracing, Skylos 4.29 in no-upload mode, `size`/`nm`, branch-aware Python coverage, Rust LLVM coverage, current release GPU lifecycle benchmarking, current stage timing, and retained JAX/Nsight/pprof/Memray/Scalene evidence.

Limitations:

- Linux `perf` is blocked by node policy and Nsight Compute counters are unavailable.
- The broad Skylos command wrote a valid result but the Slurm wrapper returned 110 because of a cluster PMIx transport error; the same cluster issue occurred after the coverage payload completed.
- No external network CVE audit was run because cargo-audit/cargo-deny/pip-audit are not installed; lock/hash and static SCA checks are not equivalent.
- No destructive power-loss test was performed.
- Current exact-HEAD hot characterization has five samples and is not an ABBA change comparison.
- Older deep profiles are explicitly marked where current JAX/output code makes their wall-time attribution stale.

No production source, test, workflow, or documentation file was modified by this review. Generated evidence is confined to ignored `results/deep-review-20260721/`.
