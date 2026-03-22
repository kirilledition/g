# **Phase 1: Foundation and PLINK Parity**

## **1. Phase Objective**

The goal of Phase 1 is to build a Python MVP that matches the relevant PLINK2 association outputs for both continuous and binary traits on the CPU using the Phase 0 chr22 baseline dataset.

This phase optimizes for **correctness first**, then for a clean API and future GPU portability. Phase 1 is not a speed race. It is the stage where we lock down data alignment, statistical definitions, masking rules, and numerical behavior before we write custom kernels.

## **2. Non-Goals**

Phase 1 does not include:

* Custom Rust kernels.
* Custom CUDA kernels.
* BGEN support.
* Firth logistic regression.
* Whole-genome production-scale performance tuning.

## **3. Explicit Design Decisions**

### **Numerical Precision**

* **Primary Phase 1 precision:** float64.
* **Why:** Phase 1 is a CPU parity phase. Matching PLINK2 as closely as possible is more important than optimizing for consumer GPU throughput.
* **GPU note:** The local GPU is an NVIDIA GeForce RTX 4080 SUPER with compute capability 8.9. It supports float64, but consumer Ada cards have poor float64 throughput relative to float32, so float64 is a correctness choice for Phase 1, not a performance choice for Phase 2.
* **Implementation rule:** Enable JAX 64-bit mode from the start and only introduce float32 as an explicit later optimization path once parity is proven.

### **Python Version**

* **Recommended Phase 1 runtime:** Python 3.13.
* **Why:** Polars and JAX run on Python 3.14 in this environment, but `bed-reader` does not currently install cleanly on Python 3.14 here. It installs on Python 3.13.
* **Decision rule:** Keep Python 3.14 only if `bed-reader` is replaced. If we keep `bed-reader`, move the project runtime to Python 3.13 before implementation begins.

### **Parity Definition**

The plan should not promise vague "100% mathematical parity". Instead, Phase 1 parity means:

* Same sample inclusion and exclusion as the PLINK2 Phase 0 runs.
* Same variant order and additive genotype coding.
* Same missing-value handling for phenotypes, covariates, and genotypes.
* Same intercept and covariate treatment.
* Same output fields for the validated models.
* Numerical agreement with the saved Phase 0 baselines within strict tolerances.

## **4. Tech Stack**

### **Compute Engine: JAX**

* **Why:** JAX gives us compiled array code, `jit`, `vmap`, and `lax.while_loop`, which are all useful for writing vectorized CPU kernels now and reusing the same logic for GPU execution later.
* **Numerical rule:** Prefer `jax.numpy.linalg.solve` over explicit matrix inversion.
* **Execution rule:** Keep kernels purely functional and shape-stable so they are easy to `jit` and later move to GPU.

### **Tabular Data: Polars**

* **Why:** Polars is a strong fit for phenotype and covariate ingestion, joins, filtering, and deterministic column selection.
* **JAX integration:** Polars exposes `DataFrame.to_jax()`, but the API is documented as unstable. We can use it where it is clean and reliable, but the architecture must not depend on guaranteed zero-copy behavior.
* **Design rule:** Use Polars for table operations and schema enforcement first. Treat direct JAX export as an optimization, not as a core contract.

### **Genotype I/O: bed-reader**

* **Why:** It is a practical reader for PLINK BED data and supports indexed reads, which is enough for chunked variant streaming in Phase 1.
* **Scope rule:** Phase 1 supports `.bed/.bim/.fam` only.
* **Future rule:** Revisit BGEN in a later phase once BED parity is stable.

## **5. Core Architecture**

The Python package should be split into explicit modules that can later be replaced piecemeal by Rust without changing the public behavior.

* **`src/g/models.py`**
  * NamedTuple containers for aligned sample data, variant metadata, genotype chunks, and regression outputs.
* **`src/g/io/tabular.py`**
  * Read phenotype and covariate tables with Polars.
  * Enforce schema, missing-value rules, and deterministic row alignment against `.fam`.
* **`src/g/io/plink.py`**
  * Parse `.fam` and `.bim`.
  * Open `.bed` with `bed-reader` and stream genotype chunks.
  * Apply mean imputation for missing genotypes.
* **`src/g/compute/linear.py`**
  * JAX implementation of covariate-adjusted linear regression.
* **`src/g/compute/logistic.py`**
  * JAX implementation of covariate-adjusted logistic regression with IRLS.
* **`src/g/cli.py`**
  * Minimal CLI that mirrors the Phase 1 PLINK-like arguments.
* **`src/g/__init__.py`**
  * Lightweight Python entrypoint only. Do not keep the CLI coupled to the placeholder Rust extension.

## **6. Execution Plan**

### **Step 0: Environment and Entry Point Cleanup**

* Move the project runtime to Python 3.13 if `bed-reader` remains the BED backend.
* Decouple the Python CLI from the placeholder Rust extension so Phase 1 can run without `g._core`.
* Add the required Python dependencies for Phase 1 only after the version decision is locked.

### **Step 1: Define the Parity Contract from Phase 0**

* Inspect the Phase 0 PLINK baseline files and document the exact columns we will validate.
* Lock the continuous-trait parity targets.
* Lock the binary-trait parity targets.
* Document the exact tolerances and any acceptable non-convergence or filtered-variant behavior.

### **Step 2: Build the Sample Alignment Layer**

* Read `.fam`, phenotype, and covariate inputs.
* Join them deterministically by `FID` and `IID`.
* Drop rows with missing required values according to the chosen parity contract.
* Build a stable covariate design matrix with intercept handling defined in one place.

### **Step 3: Build BED Streaming**

* Wrap `bed-reader` so genotype data is read in variant chunks.
* Emit chunk metadata together with the genotype matrix.
* Implement PLINK-like mean imputation for missing genotype calls.
* Keep chunk interfaces JAX-friendly but do not force premature micro-optimizations.

### **Step 4: Implement Linear Regression First**

* Implement covariate-adjusted OLS in JAX.
* Use `jax.numpy.linalg.solve` rather than explicit inversion.
* Return typed outputs for effect size, standard error, test statistic, and p-value.
* Validate against the continuous PLINK baseline before moving on.

### **Step 5: Implement Logistic Regression Second**

* Implement IRLS with `jax.lax.while_loop`.
* Make convergence criteria, maximum iterations, and singular-matrix handling explicit.
* Return typed outputs plus status fields for convergence and numerical failure.
* Validate against the binary PLINK baseline after linear parity is already passing.

### **Step 6: Build the Validation Harness**

* Add unit tests for `.fam/.bim` parsing, sample alignment, missing-value filtering, design-matrix construction, and genotype imputation.
* Add golden tests that run the engine against the Phase 0 chr22 data.
* Compare generated outputs to `data/baselines/` using floating-point tolerant assertions.

### **Step 7: Add a Minimal Usable CLI**

* Support the smallest useful PLINK-like surface area:
  * `--bfile`
  * `--pheno`
  * `--covar`
  * `--glm linear`
  * `--glm logistic`
  * `--out`
* Keep the CLI thin and delegate all logic to typed library modules.

## **7. Exact Implementation Order**

This is the exact order of work to execute from the current repository state:

1. Change the package entrypoint so Python code does not depend on importing `g._core`.
2. Decide the Python runtime and, if we keep `bed-reader`, move the project from 3.14 to 3.13.
3. Add Phase 1 dependencies and lock them.
4. Create `src/g/models.py` with all Phase 1 NamedTuple result types.
5. Implement `src/g/io/tabular.py` for phenotype and covariate loading.
6. Implement `src/g/io/plink.py` for `.fam`, `.bim`, and chunked BED reads.
7. Add unit tests for alignment and I/O.
8. Implement `src/g/compute/linear.py` and make continuous parity pass first.
9. Implement `src/g/compute/logistic.py` and make binary parity pass second.
10. Add `src/g/cli.py` and wire the command-line interface.
11. Add `just` targets for Phase 1 runs and parity validation.
12. Run formatting, linting, type-checking, and tests until the full Phase 1 surface is green.

## **8. Acceptance Criteria**

Phase 1 is complete when all of the following are true:

* The engine runs on the Phase 0 chr22 dataset using the documented CLI.
* Continuous-trait results match the saved PLINK baseline within the documented tolerance.
* Binary-trait results match the saved PLINK baseline within the documented tolerance.
* The package passes `ruff`, `ty`, and `pytest`.
* The code follows the project style guide with full type coverage, explicit naming, NamedTuple returns, and Google-style docstrings.
