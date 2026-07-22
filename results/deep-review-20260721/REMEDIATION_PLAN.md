# Remediation and performance plan

Source review: [`REPORT.md`](./REPORT.md)

Planning baseline: `29284ac6e87c74f3b2e5e9d3a1530ee934425cff`
Prepared: 2026-07-22

## Objective

Reach an external-release-ready state without a broad application rewrite:

- scientific decisions agree with upstream REGENIE under the established numerical contract;
- safe Rust APIs cannot trigger undefined behavior through ordinary external file mutation;
- output has exclusive ownership, linearizable close, and explicit crash-durability ordering;
- CUDA artifacts and execution are independently qualified;
- release evidence is tied to the exact commit and cannot be bypassed by dependency automation;
- correctness changes preserve the current hot-performance envelope, and subsequent optimization starts from a fresh profile.

## Fixed contracts

1. Upstream REGENIE v4.1 is the primary scientific oracle. Previous-g and Parquet hashes are secondary diagnostics only.
2. Numeric parity uses exclusive absolute comparisons: `abs(g_value - regenie_value) < tolerance`; never byte equality.
3. The current upstream binary-Firth tolerances remain BETA `2e-3`, SE `1e-3`, CHISQ `3e-3`, and LOG10P `1e-3`, together with exact row keys, N, non-finite classes, correction/failure aggregates, and significance classifications.
4. A candidate intended to preserve current g numerics uses the tighter internal limits: beta `5e-7`, SE `2.5e-7`, chi-square `2e-6`, and log10p `5e-7`, with exact correction/fallback decisions and classifications.
5. Primary performance workload remains the full 418,943-row chromosome-22 binary approximate-Firth GPU run, 16,384 variants per chunk, Firth batch 512, capacity 1,024, eight writer threads, direct Parquet, telemetry off.
6. Primary performance metric remains already-compiled, same-process hot elapsed time. Cold/fresh-process timing is tracked separately and cannot justify a hot regression.
7. Correctness and durability are non-negotiable. If a first safe/durable implementation regresses, continue with another correct design or explicitly quantify the unavoidable cost; do not restore unsafe mmap behavior or omit durability syncs.
8. Contract versions remain `0` until release. Internal Rust and pre-release output APIs may change rather than carrying compatibility wrappers.
9. No protected data or benchmark/profile output is committed. GPU work is serialized on Landau; heavy CPU work uses an exclusive compute-node allocation.
10. Only tests/benchmarks required to prove a production fix move early. Broad tooling, scripts, dependency-group, and static-policy cleanup stays near the end.

## Delivery graph

```text
Freeze exact baseline + immediate CI containment
             |
             +--> Null-Firth correctness --------------------+
             +--> Metadata invariants -> safe BGEN reader ---+
             +--> Output transaction/state machine ----------+--> combined release gate
             |                         |
             |                         +--> runtime/error isolation
             +--> CUDA qualification/provenance -------------+
                                                               |
                          release/coverage/build hardening <----+
                                                               |
                          fresh whole-app profile -> optimization waves
```

## Phase 0 — Freeze evidence and contain the release boundary

### 0.1 Freeze at implementation start

Do not assume the review commit is still current when work begins. Record:

- main commit, dirty state, Cargo/uv lock hashes, native-library hash and size;
- Rust/Python/JAX/JAXlib/CUDA/nvCOMP/tool versions;
- CPU/GPU/NUMA/affinity and dataset/oracle hashes;
- exact production configuration and JAX-cache tree;
- current full upstream parity result;
- focused BGEN/output baselines and at least ten baseline-only GPU processes with five hot lifecycles each before interpreting candidate comparisons.

Store all evidence under ignored `results/remediation-<commit>/` and add the campaign to the performance ledger.

### 0.2 Small immediate CI containment branch

This is the only tooling work that precedes product blockers:

- pin every external GitHub Action and action-based tool installer to a reviewed full commit SHA, retaining the release tag in a comment;
- pin installed cargo tools explicitly;
- disable generic Dependabot auto-merge until exact-commit on-prem parity is enforced;
- scope Pages write/OIDC permissions to the deploy job;
- add code-owner coverage for workflows, dependency/lock files, CUDA/OpenXLA artifacts, build scripts, and parity metadata;
- add `publish = false` to the root Cargo package;
- add a CI invariant rejecting mutable external action refs.

Validate action syntax/security, PR/push/docs workflows, permissions, lock consistency, and root publish policy. Repository ruleset/CODEOWNERS enforcement is an external-admin step and must be recorded separately.

## Phase 1 — Correct null-Firth semantics

Branch/worktree: `fix/null-firth-score-history`.

### Implementation

- Replace the historical-minimum assignment with the immediately current maximum score.
- Mirror upstream control order: a converged iterate is accepted before the consecutive-increase failure heuristic matters.
- Extract only the smallest JAX-inlinable transition needed for deterministic testing; use a typed state rather than a bare multi-value tuple.
- Do not otherwise change solver, retry, step-halving, or correction-status policy.

### Tests

- Compare the production transition to an independent implementation of REGENIE's recurrence.
- Include `10, 1, 2, 1.5`, which must produce counts `0, 0, 1, 0`.
- Include `[0.5] + ([2.0, 1.5] * 12) + [2.0, 0.75]`; the final accepted iterate must not be rejected by accumulated non-consecutive increases.
- Cover the threshold at 25/26 consecutive increases, disabled checking, convergence on/after iteration two, non-finite inputs, and all fallback start policies.
- Retain regular null-model and injected failure coverage.

### Acceptance

- Full required chr22 upstream-REGENIE qualification passes the external tolerances and exact classification/status/count contract.
- Candidate versus frozen g remains inside the tighter internal tolerances for unaffected rows; byte equality is not required.
- StableHLO, executable/cache size, and synchronized focused Firth timing do not materially grow.
- A position-balanced whole-app comparison shows no stable hot regression.

Merge this first because it is small, independently provable, and changes the scientific baseline for all later work.

## Phase 2 — Make genotype contracts safe and remove unsafe file-backed mmap

Branch/worktree: `fix/bgen-owned-positioned-io`.

### 2.1 Make metadata invariants real in release builds

- Add `VariantMetadataInvariantError` with explicit variants for parallel-column lengths, offset count/start/end/order/bounds/UTF-8 boundaries, dictionary codes, and range order/bounds.
- Change `VariantMetadataStore::from_parts` and `VariantMetadataColumns::new` to return `Result`.
- Do not add a public unchecked convenience constructor. If profiling later proves repeated validation hot, keep any unchecked path private and narrowly justified.
- Map parser-produced invariant failures to `BgenError::InvalidFormat` with variant context.
- Replace debug-only panic tests with always-on error tests, including empty stores and multibyte UTF-8 boundaries.

This is one-time construction work. Measure BGEN open/index, but do not weaken the safe API to recover negligible startup cost.

### 2.2 Replace mmap with reader-owned positioned I/O

Target shape:

- `PositionedBgenSource` owns the opened `File`, stable identity/fingerprint, and length.
- A local `read_exact_at` helper loops around platform positioned reads and returns typed EOF/I/O errors.
- Index construction uses a bounded buffered positional cursor rather than per-field syscalls.
- Rayon worker scratch owns bounded coalesced source windows for 32-variant decode tiles.
- Raw-DEFLATE packing reads bounded chunk subwindows and copies validated members into the existing pooled slab; staging must be capped so UK Biobank-scale samples do not multiply peak memory.
- No borrowed slice may outlive reader-owned storage, and no file-backed mmap remains in the safe production path.
- Existing identity checks remain as mixed-snapshot/data-integrity detection. Truncation or mutation produces an error rather than UB/SIGBUS.

First benchmark the window size and coalescing policy. Do not add an alternate mmap feature flag, allocator replacement, or a new production I/O dependency.

### Tests and oracles

- Full/tail batches; sequential/random variant offsets; contiguous/nonmonotonic/invalid sample selection.
- Header changes, Adler corruption, truncation before/while reading, short reads, changed file identity, and concurrent overwrite/truncate using a temporary fixture.
- Existing packed8 probability/statistic/status/finalizer and dosage results remain exact against owned expected buffers.
- Synthetic UKB-style 64/256/1,024-variant tiles plus real chr22 16,384-variant chunks.
- Mutation tests may fail with any documented I/O/source-changed error, but may never crash, hang, or return a silently mixed result.

### Performance gate

- On an exclusive CPU node, alternate baseline/candidate processes with at least 30 samples for open/index, full/tail, fresh/pooled raw-DEFLATE, dosage, sequential/random access, byte throughput, allocation counts, and peak memory.
- Inspect syscall count and CPU attribution; avoid a design that turns every variant into one syscall.
- On Landau, run the established ten-block/20-pair ABBA gate with five hot lifecycles per process; extend to 40/60 pairs if ambiguous.
- Any stable hot regression causes another safe implementation iteration, not restoration of mmap. More than 1% memory growth requires a clearly positive whole-app result or explicit release approval.

Update public input documentation: mutation is detected/rejected as an I/O integrity condition; no unsafe “exact opened mmap” claim remains.

## Phase 3 — Rebuild output ownership as one transaction/state machine

Branch/worktree: `fix/output-run-transaction`.

This is the largest work package. Keep output, engine, and runner API changes in one owner lane; do not split overlapping state changes across agents.

### 3.1 Separate planning from mutation

- Make `OutputManager::open`/prepare read-only: resolve and deduplicate paths, parse resume state as a hint, and validate the complete multi-phenotype plan without creating directories.
- Move mutation after BGEN/input/header preparation.
- During initialization, acquire nonblocking exclusive leases for every normalized run identity in sorted order, then reread/revalidate under lease to close TOCTOU gaps.
- Prefer stable `std::fs::File::try_lock` sidecar leases. Ship this only after a cross-process and cross-node BeeGFS smoke proves enforcement; never substitute PID/stale-file guessing.
- Resolve symlink/path aliases for lock identity without rewriting user-visible path semantics.

### 3.2 Make initialization recoverable

- Replace `writer_sessions: Option<_> + terminal: bool` with private `ManagerState::{Planned, Active, Terminal}` or consuming prepared/active manager types.
- Active state owns every lease, durable store, session, and worker-pool owner.
- For fresh runs, construct complete initial manifests/config/parts in deterministic sibling staging directories carrying one transaction ID.
- Publish only valid initialized runs. Sync the parent after directory rename.
- A crash between multi-phenotype publishes may expose a prefix, but every exposed run must have a valid recoverable manifest state.
- Ordinary initialization failure rolls back only transaction-owned staging. Resume revalidates every run before mutating any.
- Because the contract is version 0, simplify the manifest state model rather than retaining compatibility wrappers for malformed pre-release runs.

### 3.3 Make close linearizable

- Replace `Mutex<Option<Vec<Job>>>` with `Mutex<SessionState::{Open { pending }, Closing(kind), Closed}>`.
- When detaching a full batch, reserve an RAII completion ticket while holding the same state mutex; only then unlock and perform the potentially blocking queue send.
- Finish/abort transitions under that mutex, reserves any tail before unlocking, rejects later writes, and waits every reserved ticket.
- Enqueue failure drops/rolls back its ticket. Finish cannot observe zero during detach-to-send; abort cannot return before already-reserved work terminates.
- After terminal return, no producer may publish a new part.

### 3.4 Give the pool one owner

- Split pool owner from lightweight session clients. Manager owns sender lifetime and worker handles.
- Spawn with `thread::Builder::spawn`; on the Nth failure close the channel and join already-started workers.
- Add explicit `shutdown_and_join()` that propagates worker panic. `Drop` remains best-effort only.
- Release run leases only after sessions and the pool are terminal.

### 3.5 Introduce neutral durable persistence ownership

Create private neutral modules such as `persistence/model.rs`, `store.rs`, and `durability.rs`, so manifest/resume/writer/session modules consume commit/status DTOs rather than importing one another's implementation.

Required ordering:

1. Arrow/Parquet close and flush;
2. temporary part `sync_all`;
3. rename to final part;
4. sync `parts/` directory;
5. expose the commit to manifest state;
6. write a unique create-new manifest temp, sync it, rename it, and sync the run directory;
7. combine final commits and `completed`/`interrupted` into one terminal manifest publication where possible.

Lifecycle updates require an existing manifest. Delete the silent missing-manifest success path and generic upsert except for explicit initial creation. Temp names are lease/transaction scoped; only current-owner or recovery code may remove them.

### Deterministic tests

Use a private generic `OutputIo`/fault-injection layer that monomorphizes to normal std I/O in production; add no production profiling/test dependency.

- Barrier exactly at detached-before-send for both finish and abort.
- Send failure ticket rollback, no hang, write-after-close, Nth thread-spawn failure, worker panic/join.
- Two-process owner/loser test: loser mutates nothing; kill owner and prove lease release.
- Symlink/path-alias collision and manifest TOCTOU.
- Failed multi-phenotype initialization followed by fresh retry.
- Inject failure at every create/write/sync/rename point, restart, and require either clean retry or strict resume—never partial JSON, missing referenced parts, or duplicate rows.
- Assert syscall ordering through fake I/O; add Linux child abrupt-exit tests and a serialized BeeGFS durability/lease smoke.
- Missing manifest during finish is an error.

### Correctness and performance acceptance

- Exactly one process owns a run path.
- Finish/abort is linearizable; completed status follows all tickets/tasks/commits and directory sync.
- Every published part has a valid footer and becomes durable before manifest reference.
- Every injected crash point restarts without missing/duplicate rows.
- Schema, metadata, row keys, correction/status, and numerical values match the upstream contract; output bytes are diagnostic only.
- Benchmark ready-all and GPU-paced output at 1/4/8 writers on BeeGFS, recording queue waits, enqueue, writer, sync/rename, finish, file bytes, allocations, and peak memory.
- Run engine replay at 8,192/16,384 variants and the full 20-pair/10-block hot ABBA gate.
- The session/pool refactor must not show a stable hot regression. If durability itself has a measurable unavoidable cost, report it as a new explicit baseline rather than silently weakening durability.

## Phase 4 — Preserve primary failures and isolate observers

Branch/worktree: `fix/runtime-outcome-isolation`, rebased after Phase 3.

### Outcome model

- Centralize resolution as `PrimaryOutcome::{Completed, Interrupted, Failed}` plus typed ancillary failures/warnings for timing, telemetry, logging, and late signals.
- Freeze the primary outcome when engine/output returns.
- Precedence: primary failure, then interruption, then ancillary failures. A late signal after durable completion is a warning, not a fabricated interrupted flush.
- Never discard completed artifact paths when an explicitly required profile/timing artifact fails.

### Observer policy

- Sink creation/configuration before work may be fatal when explicitly requested.
- Progress and normal telemetry emission after work starts are best-effort; record counters/warnings without aborting compute or rewriting committed success.
- Move JAX `complete_setup` before diagnostic emission. A telemetry failure leaves JAX configured; a true partial JAX mutation failure remains conservatively non-reusable.
- If a preinstalled tracing subscriber prevents requested g-owned sinks, return an explicit pre-work setup error or explicit external-subscriber status. Never record the requested topology as installed when it is absent.
- Replace silent missing-manifest lifecycle success with state-specific errors.

### Tests

- Exhaustive outcome matrix: backend/output error × signal × timing/telemetry/log close.
- Telemetry failure during progress/final artifact after output completion.
- JAX diagnostic failure leaves state `Configured`; actual setup failure remains guarded.
- Preinstalled subscriber behavior in a subprocess.
- Existing bounded queue/backpressure/panic tests remain green.

Cancellation remains bounded: make queue waits cancellation-aware and poll between stages, but do not pretend the first signal can preempt an in-flight CUDA/JAX/BeeGFS call. Keep synchronous cleanup/lease release and document the second-SIGTERM hard escape.

## Phase 5 — Complete CUDA qualification, provenance, and diagnostics

Branch/worktree: `fix/cuda-qualification`.

### Artifact and ABI ownership

- Add source and PTX SHA-256 verification to `genotype-cuda/build.rs`, matching `compute-cuda` and its provenance README.
- Verify all vendored OpenXLA header hashes during build.
- Pin the supported production JAX/JAXlib pair exactly to `0.11.0`, matching the headers used to build the FFI.
- Cache a typed selection result rather than `OnceLock<bool>`: raw CUDA or pure-JAX fallback with stable reason code and detail.
- Record selected implementation, JAX/JAXlib/FFI version, and fallback reason once after logging is active. Registration failure still falls back before compilation; never switch implementations in flight.

### Real-device differential suite

Add a dev/test-only private registration entry point; do not expand the production Python surface.

Cover:

- empty/small/full active prefixes and 400/900/1,024 candidates;
- sample boundaries around warp/block geometry;
- active/inactive masks and eta clipping at/around production limits;
- zero, near-threshold, invalid, and non-finite information;
- malformed shape/dtype/status, missing registration, unsupported capability, and cached fallback reason;
- exact valid/finite/NaN classes and correction/fallback decisions;
- finite component differences under the tight internal tolerances;
- final full chr22 output under upstream tolerances and exact classifications/counts.

Serialize this suite on Landau. CPU CI covers registration/version/fallback logic without pretending to execute the kernel.

### Performance gate

- Warm/synchronize focused 400/900/1,024 executable comparisons in balanced order; record CUDA-event and wall time, HLO, cache, temporary/device memory, PTX registers/spills/occupancy.
- For pure qualification/diagnostic changes, require non-regression. Any compute change additionally requires positive pair and block intervals.
- Run full upstream parity, then the established whole-app ABBA gate for any runtime-path change.
- Reject classification/status drift, tolerance breach, artifact mismatch, in-flight fallback, or more than 1% memory/code growth without a positive whole-app result.

## Phase 6 — Make release and coverage evidence enforceable

Branch/worktree sequence: `hardening/on-prem-release-gate`, then `hardening/coverage-gates`.

### Exact-commit on-prem qualification

- Reuse the existing qualification reports to build a sanitized bundle containing commit/clean state, lock/native hashes, environment versions, configuration and fixture/oracle hashes, all three workflows, row/count/classification checks, observed maxima, and exclusive tolerances.
- Reject wrong/stale commit, dirty build, missing workflow, changed hash, unsupported JAX, protected path leakage, or `difference >= tolerance`.
- Publish only sanitized metadata/digests, never protected data.
- A trusted post-job identity sets `on-prem-parity` on that exact SHA. Credentials must not be exposed to code under test.
- Runtime/native dependency changes require this status. Actions changes remain human reviewed because they can alter the gate itself.
- Serialize all Landau qualification under one concurrency key.

This requires a trusted GitHub App/token or self-hosted status publisher plus repository-admin rules. Until it exists, runtime/native auto-merge and external release remain disabled.

### Local authoritative coverage

- Split Python and Rust coverage into required generation jobs and add them to aggregate CI.
- Remove generation/job `continue-on-error`; keep Codecov upload best-effort.
- Start below the measured baseline with non-regression floors: Python branch-aware 95%; Rust lines 78%, regions 77%, functions 72%. Never lower them.
- Build the Maturin extension with LLVM coverage before pytest so supported PyO3 binding files are not hidden at 0%.
- Add an explicit nonzero binding-file assertion. CUDA correctness remains a real-device gate, not a percentage.

Validate threshold failures deliberately, parseability/nonempty reports, binding execution, and exact comparison to the frozen baseline.

## Phase 7 — Reproducibility, dependency bloat, and static policy

These are separate, small candidates; never combine them with hot-path code.

### Build/release reproducibility

- Pin Maturin's PEP 517 requirement and isolated build constraint exactly.
- Pin downloaded CI tools and required Rust doc toolchain; use a dated nightly only if stable cannot build docs.
- Declare current `target-cpu=native` wheels same-node artifacts. Design and test a portable CPU baseline before public wheel distribution.
- Add scheduled/advisory RustSec/license/source and lock-aware Python audits, then make them release-blocking after establishing an exception policy with owner/rationale/expiry.

Use direct `ctx7` during implementation to verify current uv build-constraint syntax; never `npx`.

### Rust dependency cleanup

1. Set Arrow `default-features = false`; verify CSV/JSON disappear while Parquet-required IPC remains.
2. Measure clean/incremental build time, cache/target footprint, extension sections, and output tests.
3. Only evaluate direct `arrow-array`/`arrow-buffer`/`arrow-schema` dependencies if the first step leaves material bloat.
4. Consider `sha2` default-feature cleanup only as a tiny independent change.

No runtime speed claim comes from dependency pruning. Run an output smoke and use schema/metadata/rows/values as the oracle.

### Python environment cleanup

- Preserve the current GPU-first product policy and remove/rename the misleading `cpu` dependency group. A genuinely CPU-only distribution is a separate packaging decision.
- Split the monolithic dev environment into composable format/lint, typecheck, test, coverage, build, data/tooling, profiling, static-analysis, docs, and CUDA-format groups.
- Keep an aggregate developer convenience group, but isolated CI/Just jobs use `--only-group`.
- Prove docs, Ruff, and CUDA-format environments install no JAX/CUDA wheels; record package/download counts before and after.

### Ruff and Skylos last

- Replace Ruff `exclude` with `extend-exclude` so `.git` and default caches remain excluded.
- Lint maintained scripts; exclude only generated/external paths with explicit rationale. Remove ineffective rule ignores.
- Define no-upload product scans for `src/` and `crates/`, plus a separate whole-repository advisory scan excluding `.tools`, target, reference, data, results, and generated environments.
- Never gate on raw Skylos grade or `--all`; manually adjudicate candidates and use dedicated CVE scanners.
- After BGEN safety cleanup, normalize `SAFETY` comment placement and enable `clippy::undocumented_unsafe_blocks`/unsafe-operation policy.

## Phase 8 — Fresh profile-led performance wave

Only start after Phases 1-5 are integrated and the combined correctness gate passes.

### 8.1 Re-establish the profile

- Fresh release build and one immutable populated JAX cache.
- Ten baseline-only processes × five hot lifecycles.
- Current chr22 deep profile: stage timing, JAX/device memory, cProfile/py-spy, Memray/Scalene, targeted Nsight Systems, isolated pprof-rs worktree; document unavailable perf/NCU counters.
- Re-rank by exposed wall milliseconds, active CPU, allocation/copy evidence, recoverable ceiling, correctness risk, HLO/cache/memory/code cost.

### 8.2 Expected lanes, subject to the new profile

1. **Safe BGEN I/O:** tune coalesced windows, index buffering, and direct-to-pooled-slab reads. Do not reintroduce mmap or global allocation changes.
2. **Output after durability:** reusable Arrow buffers, encoding copies, queue waits, and terminal sync batching. Require both ready-all and GPU-paced evidence; do not revive long-lived streaming actors without contrary evidence.
3. **JAX fresh-process latency:** split the 10.1-second runtime configuration and first-execution cost into import, device/context setup, cache lookup/deserialization, compilation, and first launch. Optimize/prewarm only proven components. A daemon/persistent service is a separate product decision, not an inferred refactor.
4. **CUDA/JAX compute:** use the refreshed trace and focused 400/900/1,024 gates. Prefer deleting materialization/control flow over adding kernels.
5. **Partial multi-trait resume:** consider active-trait compute pruning only if a real resume workload is measured and shape/cache consequences are favorable.

Do not retry larger chunks, immediate auxiliary-stream overlap, a second DEFLATE decoder, explicit AVX-512, broad fused reductions, full materialization, or streaming output actors unless new evidence invalidates the prior experiments.

### Candidate acceptance

- One hypothesis, predicted ceiling, focused benchmark, independent oracle, and complexity budget per experiment.
- CPU: exclusive node, fixed affinity/NUMA, alternating processes, at least 30 samples.
- GPU: discarded warmups, synchronized CUDA-event and wall timing, balanced order, serialized jobs.
- Candidate advances only with a positive paired/block 95% interval in its focused gate.
- Full app: ten ABBA blocks/20 pairs, five hot lifecycles; extend to 40/60 without discarding data if ambiguous.
- Stable hot regression is a veto for optional performance/refactor work. Kernel/dependency/public API/long-lived memory/substantial unsafe code requires an independently positive whole-app gate.
- Merge candidates individually, rebase/retest remaining lanes, then run a combined ABBA gate and fresh profile.
- Stop after two consecutive profile-led waves find no focused candidate that clears its gate and no untried measured hotspot remains.

## Parallel worktree and integration plan

After the small containment branch, development can run in parallel:

| Lane | Worktree branch | Ownership/conflicts |
| --- | --- | --- |
| A | `fix/null-firth-score-history` | Python compute/tests; independent; merge first. |
| B | `fix/bgen-owned-positioned-io` | genotype + genotype-contracts; owns metadata API and BGEN benchmarks. |
| C | `fix/output-run-transaction` | output + overlapping engine/runner lifecycle; one owner. |
| D | `fix/cuda-qualification` | compute-cuda/genotype-cuda/binding + GPU tests; Landau serialized. |
| E | `hardening/on-prem-release-gate` | prototype independently, integrate after product blockers. |

Integration order:

1. CI containment;
2. null-Firth;
3. metadata/BGEN safety;
4. output transaction;
5. runtime outcome isolation;
6. CUDA qualification/JAX pin;
7. exact-commit parity and coverage;
8. reproducibility/dependency/tooling cleanup;
9. combined release gate and performance wave.

Output and runtime integrate sequentially because they share terminal/error semantics. JAX pinning precedes uv-group cleanup because both modify `pyproject.toml`/`uv.lock`. Workflow-heavy containment, release, coverage, and environment changes integrate sequentially even if prototyped in parallel. Documentation may be drafted in parallel but receives one final consistency owner.

Every feature branch is pushed, merged only after its gate, rebased onto the new main, and its completed worktree is removed.

## Validation commands by layer

- Local/non-heavy: `just check-local`, focused Rust/Python tests, Ruff/ty, CUDA format/lint, architecture checks.
- CPU node: `just slurm-cpu-check`, `just slurm-cpu-test`, `just slurm-cpu-coverage`, targeted Criterion/engine replay.
- GPU node: `just slurm-gpu-test-parity-required`, focused CUDA/Firth tests, `just slurm-gpu-bench-binary-hot`.
- Profile: `just profile-chr22-binary-gpu-dry`, then `just profile-chr22-binary-gpu-full` with serialized GPU ownership.
- Documentation: `just docs-check` after every user/runtime/contract/deployment change.

Slurm PMIx wrapper exit 110 must be distinguished from payload status by checking produced summaries and native exit codes; it should also be reported to cluster operations rather than normalized as application success.

## Definition of done

- No safe production API contains file-backed mmap whose external-mutation safety is merely assumed.
- Null-Firth consecutive-increase behavior matches upstream and has a deterministic regression test.
- Output ownership, initialization, close, and durability invariants pass deterministic race/fault/restart and BeeGFS multiprocess tests.
- Primary backend/output failures cannot be masked by late signals or observers; telemetry cannot strand JAX configuration.
- Raw CUDA and pure JAX execute in a serialized real-device differential suite; source/PTX/OpenXLA provenance is build-enforced and fallback is observable.
- Exact-head full upstream parity passes all three workflows under exclusive absolute tolerances; the sanitized qualification is attached to the exact commit.
- Required local coverage generation passes its floors and supported PyO3 binding files are exercised.
- Workflow/action/dependency/build provenance is pinned and protected; runtime/native auto-merge cannot bypass on-prem parity.
- Combined 20-pair/10-block hot ABBA shows no stable unexplained regression, or any unavoidable durability cost is explicitly accepted and becomes the documented baseline.
- A fresh complete profile contains no untried measured hotspot from the campaign; accepted/rejected experiments and limitations are in the ledger.
- User-facing input/output/runtime/testing/release/performance documentation is consistent and `just docs-check` passes.
- Main and origin/main contain only individually gated commits; completed worktrees are removed; generated/protected evidence remains ignored.
