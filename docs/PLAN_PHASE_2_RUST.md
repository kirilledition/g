# **Phase 2 Rust Arm: Native Bottleneck Replacement Plan**

## **1. Mission**

The Rust arm exists to replace the most expensive host-side components with native code while preserving the current semantics of the Phase 1 engine.

The first target is not a full Rust rewrite of the project. The first target is a useful, measurable native acceleration path that plugs into the current Python/JAX architecture.

## **2. Current Rust-Relevant Code Map**

### **Build and Packaging Surface**

* `Cargo.toml` already builds a PyO3 extension module.
* `pyproject.toml` already uses maturin and the mixed `src/` layout.
* `src/lib.rs` currently exposes only a placeholder module.

### **Python Host-Side Bottlenecks**

* `src/g/io/plink.py` owns BED chunk reads, missing-value handling, mean imputation, allele frequency calculation, and observation counting.
* `src/g/engine.py` depends on the chunk interface produced there.

### **Type Contracts to Preserve**

* `src/g/models.py` defines the current handoff shapes that the Python engine expects.
* `src/g/io/tabular.py` should remain the source of truth for sample alignment unless there is a strong reason to move it later.

## **3. Why the Rust Arm Starts With I/O and Preprocessing**

The roadmap already points to I/O-first replacement for Phase 2. That matches the current codebase shape.

Today the genotype path:

* reads BED data into host NumPy arrays
* computes masks and simple statistics in Python/JAX
* transfers each chunk into the JAX execution flow

This is exactly the kind of boundary where native code can help before custom kernels exist.

## **4. Rust Arm Non-Goals**

The Rust arm should not:

* rewrite the whole engine in one pass
* duplicate all Python orchestration logic without measurement-based justification
* change the statistical definitions of missing-value handling, alignment, or association testing
* skip explicit documentation of allocation and copy behavior at the FFI boundary

## **5. Recommended Execution Order**

### **Step 0: Define the Native Replacement Contract**

Before writing substantial Rust code, document the exact Python behavior that the native path must preserve.

For BED chunk ingestion, that contract includes:

* sample order validation against `.fam`
* variant order matching `.bim`
* chunk slicing semantics
* missing genotype detection
* mean imputation behavior
* allele-one frequency calculation
* observation count calculation

This contract should be derived from the existing `src/g/io/plink.py` behavior and verified by tests.

### **Step 1: Build a Real Rust BED Reader Surface**

The first native milestone should replace the placeholder `g._core` module with a real API for genotype chunk ingestion.

Recommended design constraints:

* keep function names explicit and descriptive
* make ownership and copy behavior obvious
* return structured outputs rather than opaque tuples
* preserve the current Python-facing semantics even if the internal representation changes

The exact Rust API shape is flexible, but it should make it easy for Python to request contiguous variant chunks for a selected sample index list.

### **Step 2: Parity-Check Native Reads Against the Existing Python Path**

Before swapping the engine to native ingestion, add a direct comparison harness that proves the Rust path matches the current `bed-reader` path on:

* raw genotype values
* missing masks
* variant metadata ordering
* observation counts
* allele frequency values

This should happen at the chunk level before end-to-end performance claims are made.

### **Step 3: Introduce an Optional Rust-Backed Chunk Iterator**

The safest first integration step is additive:

* keep the existing Python implementation available
* add a Rust-backed alternative behind a clear switch or internal selection path

That makes debugging much easier and allows chunk-level A/B comparison.

### **Step 4: Move Cheap Preprocessing Into Rust**

Once raw reads are correct, move the preprocessing now done in `src/g/io/plink.py` into Rust where useful:

* missing mask creation
* observation counting
* mean imputation
* allele-one frequency calculation

This work is especially valuable if it reduces Python overhead and avoids repeated host-side passes over the same chunk.

### **Step 5: Tighten the Python/Rust Memory Boundary**

After the native path exists, revisit how buffers are exposed to Python and then to JAX.

Key questions to answer explicitly:

* Which buffers are newly allocated in Rust?
* Which buffers are borrowed views?
* Which transitions force copies?
* Which transitions can later become zero-copy or DLPack-based?

This phase does not need to solve perfect zero-copy handoff, but it should make the path toward it obvious.

### **Step 6: Reassess Whether More Python Host Logic Should Move**

Only after ingestion and preprocessing are measured should the Rust arm consider moving more work, such as:

* chunk packaging helpers
* output-side formatting helpers
* simple orchestration that is demonstrably expensive

Do not move work into Rust just because it seems elegant.

## **6. Exact Python Behaviors the Rust Arm Must Preserve**

These behaviors are already embedded in the current Phase 1 engine and should be treated as part of the contract.

### **BED/FAM Alignment Guard**

`src/g/io/plink.py` validates that the BED sample order matches the aligned phenotype/covariate order. The native path must preserve this safety check or an equivalent one.

### **Chunk Semantics**

Chunk reads are contiguous by variant index and respect `chunk_size` and `variant_limit` semantics.

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

## **8. Suggested Native Milestones**

### **Milestone 1: Functional Native Reader**

Rust can read a BED chunk for a chosen sample subset and return values that match the current implementation.

### **Milestone 2: Native Preprocessing**

Rust also returns missing masks, imputed genotypes, observation counts, and allele frequencies matching the current Python semantics.

### **Milestone 3: Engine Integration**

`src/g/engine.py` can use the Rust-backed chunk path without changing user-visible behavior.

### **Milestone 4: Memory-Boundary Improvement**

The project has a documented plan or implementation for lower-copy transfer into downstream compute layers.

## **9. Suggested Agent Workflow**

Implementation agents on this arm should usually follow this loop:

1. Mirror one narrow Python behavior in Rust.
2. Build a direct chunk-level comparison against the old path.
3. Integrate it behind an additive switch.
4. Benchmark it in isolation.
5. Only then expand scope.

If profiling shows that a different native target is more urgent than BED ingestion, the agent may adapt, but the replacement should still be justified with evidence.

## **10. Acceptance Criteria**

The Rust arm is successful when all of the following are true:

* `g._core` exposes a real native acceleration surface rather than a placeholder.
* At least one major host-side bottleneck has been replaced with Rust.
* The native path matches current Python semantics on chunk-level validation.
* The end-to-end engine can use the native path without breaking Phase 1 correctness gates.
* The remaining host-side bottlenecks are documented clearly enough to guide later zero-copy and kernel work.
