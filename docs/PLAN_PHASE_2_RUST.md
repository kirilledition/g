# **Phase 2 Rust Arm: Native Bottleneck Replacement Plan**

## **1. Mission**

The Rust arm exists to remove proven host-side bottlenecks while preserving the current semantics of the Phase 1 engine.

The first target is not a full Rust rewrite of the project. The first target is a useful, measurable native acceleration path that plugs into the current Python/JAX architecture without regressing performance relative to the existing native libraries already in use.

The central lesson from the first native BED experiment is important: replacing an already-optimized Rust-backed reader with a slower handwritten Rust decoder is not progress. Phase 2 Rust work must now be driven by measurement, reuse of strong native components, and boundary minimization.

## **2. Current Rust-Relevant Code Map**

### **Build and Packaging Surface**

* `Cargo.toml` already builds a PyO3 extension module.
* `pyproject.toml` already uses maturin and the mixed `src/` layout.
* `src/lib.rs` now exposes an experimental native reader surface, but that surface is currently slower than the existing `bed-reader` path and should be treated as a prototype rather than the target architecture.

### **Python Host-Side Bottlenecks**

* `src/g/io/plink.py` owns BED chunk reads, missing-value handling, mean imputation, allele frequency calculation, and observation counting.
* `src/g/engine.py` depends on the chunk interface produced there.
* `src/g/compute/logistic.py` owns substantial host-side fallback orchestration for the hybrid logistic/Firth path.

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

## **4. Rust Arm Non-Goals**

The Rust arm should not:

* rewrite the whole engine in one pass
* replace well-optimized native libraries with inferior handwritten Rust just for ownership or aesthetic reasons
* duplicate all Python orchestration logic without measurement-based justification
* change the statistical definitions of missing-value handling, alignment, or association testing
* skip explicit documentation of allocation and copy behavior at the FFI boundary
* replace JAX mathematical kernels during the current phase unless profiling proves a host-side problem cannot be solved cleanly while keeping JAX in place

## **5. Recommended Execution Order**

### **Step 0: Benchmark Before Replacing Anything**

Before writing substantial Rust code, benchmark the incumbent path and the candidate replacement at the right level of granularity.

At minimum, measure:

* raw chunk read time
* raw read plus `jax.device_put`
* full `iter_genotype_chunks()` wall time
* end-to-end linear and logistic wall time on representative subsets

The benchmark harness should separate raw decode wins from Python/Rust boundary losses.

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

### **Step 6: Tighten the Python/Rust Memory Boundary**

After the native path exists, revisit how buffers are exposed to Python and then to JAX.

Key questions to answer explicitly:

* Which buffers are newly allocated in Rust?
* Which buffers are borrowed views?
* Which transitions force copies?
* Which transitions can later become zero-copy or DLPack-based?

This phase does not need to solve perfect zero-copy handoff, but it must eliminate obviously avoidable overhead such as reopening files per chunk, converting NumPy arrays to Python lists, or copying through temporary byte buffers.

### **Step 7: Reassess Whether More Python Host Logic Should Move**

Only after ingestion and preprocessing are measured should the Rust arm consider moving more work, such as:

* chunk packaging helpers
* output-side formatting helpers
* simple orchestration that is demonstrably expensive
* host-side fallback orchestration in `src/g/compute/logistic.py` if JAX compute is not the dominant cost

Do not move work into Rust just because it seems elegant.

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

### **Milestone 3: Engine Integration**

`src/g/engine.py` can use the Rust-backed chunk path without changing user-visible behavior.

### **Milestone 4: Memory-Boundary Improvement**

The project has a documented plan or implementation for lower-copy transfer into downstream compute layers.

### **Milestone 5: Host-Orchestration Reduction**

If needed after ingestion work, Rust reduces expensive Python-side orchestration around hybrid logistic fallback or output formatting without replacing JAX math kernels.

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
