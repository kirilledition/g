# **Phase 2 Rust Arm: Native Bottleneck Replacement Plan**

## **1. Mission**

The Rust arm exists to remove proven host-side bottlenecks while preserving the current semantics of the Phase 1 engine.

The first target is not a full Rust rewrite of the project. The first target is a useful, measurable native acceleration path that plugs into the current Python/JAX architecture without regressing performance relative to the existing native libraries already in use.

The central lesson from the first native BED experiment is important: replacing an already-optimized Rust-backed reader with a slower handwritten Rust decoder is not progress. Phase 2 Rust work must now be driven by measurement, reuse of strong native components, and boundary minimization.

## **1.1 Current Status Snapshot**

The Rust arm is no longer hypothetical. We have already implemented and measured several additive prototypes, and those measurements now constrain the plan.

Completed so far:

* `g._core` is currently a minimal PyO3 placeholder module.
* A dedicated benchmark harness exists in `scripts/benchmark_plink_reader.py`.
* A dedicated hybrid logistic benchmark exists in `scripts/benchmark_logistic_fallback.py`.

Measured outcomes so far:

* The handwritten Rust BED reader and Rust preprocessing prototypes were removed after they were deprioritized.
* The incumbent `bed-reader` plus Python/JAX preprocessing path remains the maintained ingestion strategy.
* Host-side hybrid logistic cleanup did produce useful wins, so host orchestration remains a legitimate Rust-adjacent target even while GPU work becomes the primary Phase 2 track.

The Rust arm therefore remains active, but it is no longer the primary frontier for large expected speedups. The best remaining Rust work is narrow host-boundary reduction and selective orchestration cleanup, not a broad rewrite.

## **2. Current Rust-Relevant Code Map**

### **Build and Packaging Surface**

* `Cargo.toml` already builds a PyO3 extension module.
* `pyproject.toml` already uses maturin and the mixed `src/` layout.
* `src/lib.rs` currently exposes only a placeholder PyO3 module surface.
* `rustfmt.toml` and `clippy.toml` now define repository-local Rust formatting and lint policy.
* The current PyO3 build uses `abi3-py311` so the extension can consume newer buffer interfaces cleanly.

### **Python Host-Side Bottlenecks**

* `src/g/io/plink.py` owns BED chunk reads, missing-value handling, mean imputation, allele frequency calculation, and observation counting.
* `src/g/engine.py` depends on the chunk interface produced there.
* `src/g/compute/logistic.py` owns substantial host-side fallback orchestration for the hybrid logistic/Firth path.

### **Benchmark and Evaluation Surfaces Added During This Phase**

* `scripts/benchmark_plink_reader.py` compares the supported raw chunk reads and full iterator path.
* `scripts/benchmark_logistic_fallback.py` measures the hybrid logistic/Firth path on representative chunk sizes.
* `tests/test_phase1.py` now focuses on the maintained Python/JAX ingestion path.

### **Existing Native Components We Should Reuse First**

* `bed-reader` is already a multi-threaded Rust-backed BED engine and should be treated as the raw-read baseline to beat or wrap, not casually replaced with handwritten decode code.
* Polars already executes joins, scans, and output serialization in Rust. Python-side code around it should be optimized only when profiling shows wrapper overhead, conversion cost, or object churn dominates.

### **Type Contracts to Preserve**

* `src/g/models.py` defines the current handoff shapes that the Python engine expects.
* `src/g/io/tabular.py` should remain the source of truth for sample alignment unless there is a strong reason to move it later.

## **3. Why the Rust Arm Starts With I/O and Preprocessing**

The roadmap still points to I/O-first and host-preprocessing-first replacement for Phase 2, but with a tighter constraint: we should not rewrite raw BED decode unless we can clearly outperform or extend the existing `bed-reader` implementation.

Today the genotype path:

* reads BED data into host NumPy arrays through a Rust-backed library
* computes masks and simple statistics in Python/JAX
* transfers each chunk into the JAX execution flow

This means the best early native opportunity is not necessarily a new BED decoder. The better target is the host-side work surrounding decode: repeated preprocessing passes, Python boundary conversions, and per-chunk orchestration.

The implemented prototypes validated that framing. The raw BED decode rewrite underperformed, but the surrounding boundary and orchestration work produced actionable measurements and a clearer backlog.

## **4. Rust Arm Non-Goals**

The Rust arm should not:

* rewrite the whole engine in one pass
* replace well-optimized native libraries with inferior handwritten Rust just for ownership or aesthetic reasons
* duplicate all Python orchestration logic without measurement-based justification
* change the statistical definitions of missing-value handling, alignment, or association testing
* skip explicit documentation of allocation and copy behavior at the FFI boundary
* replace JAX mathematical kernels during the current phase unless profiling proves a host-side problem cannot be solved cleanly while keeping JAX in place
* chase micro-optimizations in Rust after benchmarks show the main remaining wins have shifted to JAX/GPU execution

## **5. Recommended Execution Order**

### **Step 0: Benchmark Before Replacing Anything**

Before writing substantial Rust code, benchmark the incumbent path and the candidate replacement at the right level of granularity.

At minimum, measure:

* raw chunk read time
* raw read plus `jax.device_put`
* full `iter_genotype_chunks()` wall time
* end-to-end linear and logistic wall time on representative subsets

The benchmark harness should separate raw decode wins from Python/Rust boundary losses.

Status:

* Done for chunk ingestion and preprocessing via `scripts/benchmark_plink_reader.py`.
* Partially done for hybrid logistic orchestration via `scripts/benchmark_logistic_fallback.py`.
* Still missing: end-to-end benchmark attribution across read, transfer, compute, and formatting in one report.

### **Step 1: Define the Native Replacement Contract**

Before changing a working path, document the exact Python behavior that the native path must preserve.

For BED chunk ingestion, that contract includes:

* sample order validation against `.fam`
* variant order matching `.bim`
* chunk slicing semantics
* missing genotype detection
* mean imputation behavior
* allele-one frequency calculation
* observation count calculation

This contract should be derived from the existing `src/g/io/plink.py` behavior and verified by tests.

### **Step 2: Reuse Existing Native Readers Before Writing Custom Decode**

The first serious optimization milestone should not be a handwritten decoder. It should reuse the best available raw-read implementation and fuse more surrounding work around it.

Preferred order:

* keep using Python `bed-reader` if it remains the fastest raw decode path
* if a Rust-side integration is needed, wrap or reuse the corresponding native reader logic rather than reimplementing PLINK decode naively
* only keep a custom decoder if it wins on both chunk-level and end-to-end benchmarks

Status:

* The handwritten decoder does not currently win and should remain a prototype only.
* The incumbent `bed-reader` path remains the raw-read baseline and the likely long-term ingestion source until a future GPU-oriented boundary design justifies something else.

Recommended design constraints:

* keep function names explicit and descriptive
* make ownership and copy behavior obvious
* return structured outputs rather than opaque tuples
* preserve the current Python-facing semantics even if the internal representation changes

The exact Rust API shape is flexible, but it should make it easy for Python to request contiguous variant chunks for a selected sample index list while avoiding repeated setup cost and avoidable copies.

### **Step 3: Parity-Check Native Reads Against the Existing Python Path**

Before swapping the engine to native ingestion, add a direct comparison harness that proves the Rust path matches the current `bed-reader` path on:

* raw genotype values
* missing masks
* variant metadata ordering
* observation counts
* allele frequency values

This should happen at the chunk level before end-to-end performance claims are made.

### **Step 4: Introduce an Optional Rust-Backed Chunk Iterator**

The safest first integration step is additive:

* keep the existing Python implementation available
* add a Rust-backed alternative behind a clear switch or internal selection path

That makes debugging much easier and allows chunk-level A/B comparison.

### **Step 5: Move Cheap Preprocessing Into Rust**

Once raw reads are correct and competitive, move the preprocessing now done in `src/g/io/plink.py` into Rust where useful:

* missing mask creation
* observation counting
* mean imputation
* allele-one frequency calculation

This is the most promising early Rust target because it removes repeated host-side passes over the same chunk without touching JAX association math.

Status:

* Implemented in `src/lib.rs` and `src/g/io/plink.py`.
* Correctness is covered by parity tests.
* Performance improved after reducing output-copy overhead, but the path still does not beat the incumbent Python/JAX preprocessing path on the current benchmarked setup.
* This means further work here should be opportunistic and benchmark-gated, not assumed to be the main Phase 2 win.

### **Step 6: Tighten the Python/Rust Memory Boundary**

After the native path exists, revisit how buffers are exposed to Python and then to JAX.

Key questions to answer explicitly:

* Which buffers are newly allocated in Rust?
* Which buffers are borrowed views?
* Which transitions force copies?
* Which transitions can later become zero-copy or DLPack-based?

This phase does not need to solve perfect zero-copy handoff, but it must eliminate obviously avoidable overhead such as reopening files per chunk, converting NumPy arrays to Python lists, or copying through temporary byte buffers.

Direct host-side reads into JAX should not be treated as an automatic win. For the current architecture, `bed-reader` naturally fills host-resident NumPy buffers, and JAX still needs an import step. On the current CPU-backed setup, `jax.device_put(...)` is already very cheap and benchmarked slightly better than `jnp.from_dlpack(...)` for representative chunk sizes. Unless a future DLPack-based path removes a real measured bottleneck or enables a lower-copy GPU handoff, the project should keep the NumPy-to-JAX boundary and focus on larger host-side costs.

Status:

* The Rust preprocessing path no longer returns large Python `bytes` payloads; it now uses buffer-protocol exports to reduce copy overhead.
* The handwritten BED reader still pays avoidable setup and transfer costs and should not be expanded further until it has a credible performance case.
* Direct host-to-JAX import was explicitly evaluated and deprioritized for now.

### **Step 7: Reassess Whether More Python Host Logic Should Move**

Only after ingestion and preprocessing are measured should the Rust arm consider moving more work, such as:

* chunk packaging helpers
* output-side formatting helpers
* simple orchestration that is demonstrably expensive
* host-side fallback orchestration in `src/g/compute/logistic.py` if JAX compute is not the dominant cost

Do not move work into Rust just because it seems elegant.

Status:

* Hybrid logistic fallback orchestration in `src/g/compute/logistic.py` has already shown measurable gains from reducing host-side Python overhead, even without moving JAX math out of Python.
* This remains the most plausible remaining Rust-adjacent host target if we decide to keep investing in the Rust arm during GPU-focused work.

## **5.1 Findings So Far**

### **What Worked**

* Benchmark-first iteration prevented bad architectural decisions from becoming defaults.
* Rust buffer-protocol exports materially improved the preprocessing prototype.
* Hybrid logistic cleanup benefited from reducing Python list handling, host patch loops, and unnecessary fallback-path work.

### **What Did Not Work**

* A handwritten Rust BED decode path did not beat `bed-reader`.
* A direct host-to-JAX import strategy did not outperform `jax.device_put(...)` on the current CPU-backed setup.
* Not every host/device cleanup in logistic produced a win; some changes regressed the benchmark and were intentionally reverted.

### **Strategic Conclusions**

* The Rust arm should stay narrow, additive, and benchmark-driven.
* The GPU/JAX arm is now the primary source of likely Phase 2 gains.
* Remaining Rust work should focus on selective orchestration/boundary cleanup, not broad native replacement.

## **6. Exact Python Behaviors the Rust Arm Must Preserve**

These behaviors are already embedded in the current Phase 1 engine and should be treated as part of the contract.

### **BED/FAM Alignment Guard**

`src/g/io/plink.py` validates that the BED sample order matches the aligned phenotype/covariate order. The native path must preserve this safety check or an equivalent one.

### **Chunk Semantics**

Chunk reads are contiguous by variant index and respect `chunk_size` and `variant_limit` semantics.

### **Performance Semantics**

No Rust replacement should be considered successful if it preserves correctness but loses to the current `bed-reader`-based chunk-ingestion baseline on warmed chunk-level benchmarks.

### **Missing-Value Semantics**

Missing genotype calls become part of a per-chunk boolean mask and are mean-imputed using the observed values for the same variant.

### **Metadata Semantics**

Variant metadata is loaded from `.bim` and must remain aligned with every chunk.

### **Numerical Semantics**

Observation counts and allele-one frequencies are not incidental metadata. They feed output frames and tests, so they must match current semantics.

## **7. Rust-Specific Engineering Guidance**

### **Use Structs, Not Bare Tuples**

The repository style guide explicitly forbids tuple-style complex returns. The Rust API should reflect that rule.

### **Document Allocation Behavior**

Every PyO3-exposed function that allocates or clones significant buffers should say so in rustdoc or adjacent documentation.

### **Keep Names Descriptive**

Avoid abbreviated genotype- and matrix-related names. Mirror the explicit naming style used in the Python codebase.

### **Prefer a Narrow First Surface**

A small, correct, benchmarkable native API is better than a wide but unstable one.

### **Prefer Fused Work Over Reimplemented Work**

If a strong native crate already handles one stage well, the Rust arm should focus on fusing adjacent host-side stages around it instead of rebuilding that stage from scratch.

## **8. Suggested Native Milestones**

### **Milestone 1: Benchmark-Validated Native Boundary**

The project has a benchmark harness that compares incumbent and candidate native paths at the raw-read, chunk-iterator, and end-to-end levels.

### **Milestone 2: Fused Native Preprocessing**

Rust returns missing masks, imputed genotypes, observation counts, and allele frequencies matching the current Python semantics while reusing or matching the incumbent raw-read performance.

Current status: correctness achieved, performance parity not yet achieved.

### **Milestone 3: Engine Integration**

`src/g/engine.py` can use the Rust-backed chunk path without changing user-visible behavior.

### **Milestone 4: Memory-Boundary Improvement**

The project has a documented plan or implementation for lower-copy transfer into downstream compute layers.

### **Milestone 5: Host-Orchestration Reduction**

If needed after ingestion work, Rust reduces expensive Python-side orchestration around hybrid logistic fallback or output formatting without replacing JAX math kernels.

Current status: partially underway in Python/JAX cleanup form; may still justify targeted Rust support later.

## **8.1 Backlog of Unfinished Rust Tasks**

These items remain plausible, but none should proceed without a clear benchmark case.

* Re-benchmark Rust preprocessing on GPU-backed JAX once the GPU lane is available; the CPU result may not be the final story.
* If preprocessing still matters, consider returning NumPy-owned arrays directly from Rust through tighter FFI rather than layering more Python reconstruction.
* If logistic host orchestration remains significant after JAX cleanup, consider moving Firth batch packaging/scatter helpers into Rust while leaving all numerical kernels in JAX.
* Investigate whether output-side formatting is materially expensive before attempting any Rust-side result serialization.
* Keep the handwritten BED reader as a reference/prototype only unless a future GPU-oriented boundary design gives it a clear reason to exist.

## **8.2 Explicitly Deferred or Rejected Ideas**

* A full Rust replacement for the current JAX math paths.
* A broad rewrite of sample alignment or Polars-driven tabular work.
* Direct host-side read into JAX arrays as a near-term optimization target.
* Continued optimization of the handwritten BED reader without evidence that it can beat `bed-reader` or unlock a new boundary strategy.

## **9. Suggested Agent Workflow**

Implementation agents on this arm should usually follow this loop:

1. Mirror one narrow Python behavior in Rust.
2. Build a direct chunk-level comparison against the old path.
3. Integrate it behind an additive switch.
4. Benchmark it in isolation.
5. Only then expand scope.

If a prototype loses to the incumbent native path, the agent should not keep polishing it blindly. Either redesign around a stronger native primitive or abandon that subpath.

If profiling shows that a different native target is more urgent than BED ingestion, the agent may adapt, but the replacement should still be justified with evidence.

## **10. Acceptance Criteria**

The Rust arm is successful when all of the following are true:

* `g._core` exposes a real native acceleration surface rather than a placeholder.
* At least one major host-side bottleneck has been replaced with Rust.
* The native path matches current Python semantics on chunk-level validation.
* The native path beats or matches the incumbent `bed-reader`-based baseline for the specific stage it replaces.
* The end-to-end engine can use the native path without breaking Phase 1 correctness gates.
* The remaining host-side bottlenecks are documented clearly enough to guide later zero-copy and kernel work.

For the current repository state, the Rust arm should be considered informative but incomplete: it has produced valuable infrastructure, measurements, and narrower targets, but it has not yet delivered a dominant end-to-end speedup over the incumbent host path.

## 11. Rust Arm Implementation Status (2026-03-23)

### Milestone Status

| Milestone | Status | Notes |
|---|---|---|
| 1: Benchmark-validated native boundary | ✅ Done | `benchmark_plink_reader.py` and `benchmark_logistic_fallback.py` exist. |
| 2: Fused native preprocessing | Removed | Prototype removed after deprioritization; revisit only if Rust work becomes active again. |
| 3: Engine integration | Deferred | `lib.rs` was reduced back to a placeholder surface while the engine stays on Python/bed-reader defaults. |
| 4: Memory-boundary improvement | Deferred | Buffer-protocol work was removed alongside the paused Rust ingestion prototypes. |
| 5: Host-orchestration reduction | ✅ Done | Achieved via Python/JAX cleanup rather than Rust: on-device merge, reduced `device_get` calls. |

### Completed Additive Rust Work

- Removed handwritten BED reader prototype from `lib.rs`.
- Removed Rust preprocessing prototype from `lib.rs`.
- Removed buffer-protocol exports tied to the paused ingestion prototype.
- Removed parity tests for native reader and native preprocessing paths.

### Decisions Made

- **Handwritten BED reader removed** — it did not outperform `bed-reader`, so keeping it in-tree added maintenance cost without product value.
- **Direct host-to-JAX import deprioritized** — `jax.device_put()` benchmarked slightly faster than `jnp.from_dlpack()` on CPU.
- **Host orchestration reduction achieved in Python/JAX** — the on-device merge refactor eliminated the need for Rust-side Firth batch packaging/scatter helpers.
- **Broad Rust rewrite explicitly rejected** — per §4, the Rust arm should stay narrow and benchmark-driven.

### §8.1 Backlog Disposition

| Backlog Item | Decision |
|---|---|
| Re-benchmark Rust preprocessing on GPU | Deferred — blocked on GPU bring-up |
| Return NumPy-owned arrays directly from Rust | Deferred — buffer-protocol exports already reduce copies |
| Move Firth batch packaging to Rust | Rejected — on-device merge makes this unnecessary |
| Investigate output-side formatting cost | Not needed — profiling showed formatting is negligible |
| Keep handwritten BED reader as reference | ✅ Kept as prototype, not production path |

### Remaining Work

1. Re-benchmark Rust preprocessing once GPU bring-up is available — CPU result may not be the final story.
2. If preprocessing becomes a bottleneck on GPU, consider tighter FFI (NumPy-owned arrays from Rust).
3. The Rust arm is now on hold pending GPU measurement results.
