# **High-Performance GPU-Accelerated GWAS Engine**

## **Project Vision & Development Roadmap**

### **Executive Summary**

The objective is to build a next-generation, hardware-accelerated (GPU/TPU/Cerebras) Genome-Wide Association Study (GWAS) engine capable of handling biobank-scale datasets (e.g., UK Biobank) faster than current state-of-the-art (SOTA) tools like plink2 and regenie.

The final product will be a standalone Rust CLI tool bound to Python via FFI, allowing data scientists to write simple Python scripts while executing raw Rust/CUDA performance under the hood. To ensure rapid delivery and continuous iteration, the project will utilize the **Strangler Fig Pattern**: starting as a high-level Python MVP and systematically rewriting performance bottlenecks in Rust and custom hardware kernels.

### **Development Philosophy: The Strangler Fig Pattern**

Avoid the "two years in stealth" trap. The architecture will evolve through iterative substitution:

1. **Prototype Fast:** Pure Python delegating to JAX for auto-diff, XLA compilation, and rapid mathematical validation.
2. **Decouple I/O & Compute:** Establish strict interfaces between data ingestion and matrix operations.
3. **Strangulate Bottlenecks:** Profile the code. Substitute the slowest modules (starting with I/O, then specific math kernels) with highly optimized Rust/CUDA/Triton code.
4. **Zero-Copy Handoffs:** Use Apache Arrow, DLPack, and PyO3 to ensure zero-copy memory transfers between the Rust backend and Python frontend.

### **Phase 1: Foundation & plink Parity (Milestone 1)**

**Goal:** A Python package with a CLI interface that produces mathematically identical results to plink on the CPU for both binary and continuous traits.

* **Tech Stack:** Python, JAX, Polars, and bed-reader.
* **Focus Areas:**
  * Standard linear and logistic regression implementations.
  * Setting up the testing harness against plink outputs to guarantee 100% mathematical parity.
  * Defining the exact API footprint for the user-facing Python package within the unified src/ layout.
* **Deliverable:** A working, fully tested MVP that proves the math and API design, even if it is not yet the fastest tool on the market.

### **Phase 2: GPU Acceleration & Rust Integration (Milestone 2)**

**Goal:** Execute Phase 1 math on the GPU and outperform CPU-bound plink. Iteratively replace slow components with optimized Rust/CUDA until SOTA speed is achieved.

* **Tech Stack:** Rust, PyO3, maturin (build system), Apache Arrow, Triton/CUDA.
* **The PCIe Bottleneck:** Sending uncompressed float32 genotype matrices from RAM to VRAM will throttle the GPU.
* **The Optimization Strategy:**
  * **I/O First:** Rewrite the genomic file parsers (PLINK BED/BGEN/VCF) in Rust. Use Apache Arrow to pass pointers to Python instantly.
  * **Custom Kernels:** Write custom Triton or CUDA kernels that ingest compressed 2-bit PLINK BED formats directly into VRAM, decompressing and computing on the fly (leveraging INT4 Tensor Cores if applicable).
* **Deliverable:** A hybrid Python/Rust package that beats plink2 speed on a single GPU.

### **Phase 3: The regenie Boss Fight - Mixed Linear Models (Milestone 3)**

**Goal:** Implement Mixed Linear Models (MLMs) to handle massive biobanks (e.g., UK Biobank). Results must be identical to regenie.

* **The Challenge:** Calculating a massive $N \times N$ Genetic Relationship Matrix (GRM) will blow out VRAM. The engine must implement a two-step architecture to match regenie.
* **Step 1: Ridge Regression & LOCO**
  * Partition the genome into blocks.
  * Run ridge regression to estimate the polygenic effect.
  * Generate Leave-One-Chromosome-Out (LOCO) predictions.
* **Step 2: Association Testing & Firth Penalty**
  * Run single-variant tests using LOCO predictions as covariates.
  * **Crucial:** Implement Firth penalized logistic regression for binary traits to handle rare variants and unbalanced case-control ratios. The penalty term added to the log-likelihood is:

$$\tilde{l}(\theta) = l(\theta) + \frac{1}{2} \log |I(\theta)|$$

  where $I(\theta)$ is the Fisher information matrix.
* **The GPU Advantage:** Leverage the GPU's massive parallelism to batch multiple phenotype regressions into single, massive matrix multiplications, outperforming C++ CPU implementations.
* **Deliverable:** An MVP capable of replacing regenie for biobank-scale analysis.

### **Phase 4: Post-MVP & Quality of Life (QoL)**

**Goal:** Once SOTA performance is achieved, implement user-requested features and extreme precision improvements.

* **Log10 P-Values:** Calculate p-values directly in log10 space to prevent 64-bit float underflow (critical for highly significant hits in massive biobanks).
* **Optimized Storage:** Save phenotype data and summary statistics directly into Parquet/Arrow formats for rapid downstream analysis.
* **Ecosystem Integration:** Build native interfaces for tools like Hail.
* **Hardware Agnosticism:** Expand XLA/Rust bindings to support TPUs, AMD GPUs, and Cerebras waferscale engines.
