# **High-Performance GPU-Accelerated GWAS Engine**

## **Project Vision & Development Roadmap**

### **Executive Summary**

The objective is to build a next-generation, hardware-accelerated (GPU/TPU/Cerebras) Genome-Wide Association Study (GWAS) engine capable of handling biobank-scale datasets (e.g., UK Biobank) faster than current state-of-the-art (SOTA) tools like plink2 and regenie.

The final product will be a standalone Rust CLI tool bound to Python via FFI, allowing data scientists to write simple Python scripts while executing raw Rust/CUDA performance under the hood. To ensure rapid delivery and continuous iteration, the project will utilize the **Strangler Fig Pattern**: starting as a high-level Python MVP and systematically rewriting performance bottlenecks in Rust and custom hardware kernels.

### **Development Philosophy: The Strangler Fig Pattern**

Avoid the "two years in stealth" trap. The architecture will evolve through iterative substitution:

1. **Prototype Fast:** Pure Python delegating to JAX/PyTorch for auto-diff, XLA compilation, and rapid mathematical validation.  
2. **Decouple I/O & Compute:** Establish strict interfaces between data ingestion and matrix operations.  
3. **Strangulate Bottlenecks:** Profile the code. Substitute the slowest modules (starting with I/O, then specific math kernels) with highly optimized Rust/CUDA/Triton code.  
4. **Zero-Copy Handoffs:** Use Apache Arrow, DLPack, and PyO3 to ensure zero-copy memory transfers between the Rust backend and Python frontend.

### **Phase 1: Foundation & plink Parity (Milestone 1\)**

**Goal:** A Python package with a CLI interface that produces mathematically identical results to plink on the CPU for both binary and continuous traits.

* **Tech Stack:** Python, JAX/PyTorch, standard data science libraries (numpy, pandas/polars).  
* **Focus Areas:**  
  * Standard linear and logistic regression implementations.  
  * Setting up the testing harness against plink outputs to guarantee 100% mathematical parity.  
  * Defining the exact API footprint for the user-facing Python package.  
* **Deliverable:** A working, fully tested MVP that proves the math and API design, even if it is not yet the fastest tool on the market.

### **Phase 2: GPU Acceleration & Rust Integration (Milestone 2\)**

**Goal:** Execute Phase 1 math on the GPU and outperform CPU-bound plink. Iteratively replace slow components with optimized Rust/CUDA until SOTA speed is achieved.

* **Tech Stack:** Rust, PyO3, maturin (build system), Apache Arrow, Triton/CUDA.  
* **The PCIe Bottleneck:** Sending uncompressed float32 genotype matrices from RAM to VRAM will throttle the GPU.  
* **The Optimization Strategy:**  
  * **I/O First:** Rewrite the genomic file parsers (PLINK BED/BGEN/VCF) in Rust. Use Apache Arrow to pass pointers to Python instantly.  
  * **Custom Kernels:** Write custom Triton or CUDA kernels that ingest compressed 2-bit PLINK BED formats directly into VRAM, decompressing and computing on the fly (leveraging INT4 Tensor Cores if applicable).  
* **Deliverable:** A hybrid Python/Rust package that beats plink2 speed on a single GPU.

### **Phase 3: The regenie Boss Fight \- Mixed Linear Models (Milestone 3\)**

**Goal:** Implement Mixed Linear Models (MLMs) to handle massive biobanks (e.g., UK Biobank). Results must be identical to regenie.

* **The Challenge:** Calculating a massive ![][image1] Genetic Relationship Matrix (GRM) will blow out VRAM. The engine must implement a two-step architecture to match regenie.  
* **Step 1: Ridge Regression & LOCO**  
  * Partition the genome into blocks.  
  * Run ridge regression to estimate the polygenic effect.  
  * Generate Leave-One-Chromosome-Out (LOCO) predictions.  
* **Step 2: Association Testing & Firth Penalty**  
  * Run single-variant tests using LOCO predictions as covariates.  
  * **Crucial:** Implement Firth penalized logistic regression for binary traits to handle rare variants and unbalanced case-control ratios. The penalty term added to the log-likelihood is:  
    ![][image2]  
    where ![][image3] is the Fisher information matrix.  
* **The GPU Advantage:** Leverage the GPU's massive parallelism to batch multiple phenotype regressions into single, massive matrix multiplications, outperforming C++ CPU implementations.  
* **Deliverable:** An MVP capable of replacing regenie for biobank-scale analysis.

### **Phase 4: Post-MVP & Quality of Life (QoL)**

**Goal:** Once SOTA performance is achieved, implement user-requested features and extreme precision improvements.

* **Log10 P-Values:** Calculate p-values directly in log10 space to prevent 64-bit float underflow (critical for highly significant hits in massive biobanks).  
* **Optimized Storage:** Save phenotype data and summary statistics directly into Parquet/Arrow formats for rapid downstream analysis.  
* **Ecosystem Integration:** Build native interfaces for tools like Hail.  
* **Hardware Agnosticism:** Expand XLA/Rust bindings to support TPUs, AMD GPUs, and Cerebras waferscale engines.

### **Team Composition & Hiring Profile**

To execute this roadmap, the initial engineering hires must possess specific crossover skills:

1. **The Rust/Python FFI Specialist:** \* Deep understanding of PyO3, maturin, and the Python Global Interpreter Lock (GIL).  
   * Expertise in zero-copy memory management (Apache Arrow, DLPack).  
   * Tasked with building the seamless bridge between Python flexibility and Rust performance.  
2. **The ML Compiler/Hardware Engineer:** \* Expertise in JAX/PyTorch internals, OpenXLA, and custom kernel writing (Triton/CUDA).  
   * Understands memory bandwidth bottlenecks (PCIe transfers vs. VRAM compute).  
   * Tasked with making the math run blisteringly fast across different hardware accelerators.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAfCAYAAABAk1s1AAADC0lEQVR4Xu2XP4gTQRTGV7xGREQUovk3SRBDQLE4rLSwtVW7a6wU5CottBSsbcJVIgiCXHONncV1NoG0JwdWV1xjcY1wFsqh35fMO15eduLsXo4LMj8Ykn3zzWS/tzNvJ1mWSCQSif+SU865t7q1Wq2XVkTQ98Zq2TqdzjWrPSmazeZje3/1ev2M1ZGAnzWrE5ioK/iBh/j8Iw3xJSvED9bQd0/pthqNxnXOYbUnRa1Wu4hWN37Ws7CfD0r3lOOtbgKIVrCSXuFzm4P43WoE9G/iRpZtfJFQfkZJCPmpVCpn0b8f7Qfid1gdt9Tk21YjoG8XE1+w8UVC/EiiQn64+tA3jPYD8V423kJLMvmMvc2tubDg/nrihwmb5QcJeoG+FRsPAvG+fMfgz35yFraJ+uP39VDHYsBKvYtxB3yCts8C3S+8IM7beCwY3xc/TI4kyvux2p0iq6kH8Xu5brfbFTX5qtYWfgKT8AkfYH5nO4R/9ceAOfa0H+frLpvW+b6pWBAaxxN/YmKSqIEKc1tuoPVUrBD4nUcw8c3GCeZ9EOorAu9b+9FF3axU+uEWjQPiYbVavWRi6zJ55l+tMHEH17+1rgyYY5UJ0SsHsYN5JInbCHN9NX6YEPHDo4Jo6aevdLPxyZiAmZdEob32Oq6m+CcwAyaLyWGy+IlactVqyuBydgfRfjJfd72f+N0B8U8bIzIx2q6/3nElCnkA1qtRsuaxkgTM1+exwMaJ8tP21/GFHE/0JgZ8tHFi97b/XraQW44rUT+ywD8F5WcgfqwmiPMHMxv36L09QPvS7XbPWVEJRm8/SZDzRXwe288Fdofn8IwofqwgF9zcshufN6b+BwmmVt23/SXIPSIwWXnxIng/Gzau8Qkq5odFjwNs3KImji98AY7zeOD9PLNxjRtv93g/WOY3ZAAPmLZfA80adTZeFCYCbcvGNSWTdRpjPnk/ufVWwRU98m07phChaYfnixy4t7/bYBHwtC9jjuc2ngdM385mlAONU//lYv2w76h+EolEIpFIJBKJI/EXdKMq48M5UBkAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAfCAYAAAB+tjR7AAADLElEQVR4Xu2XP2hTURTGX2gFQUWkhtD8u0lQQkFwCAiCg4ODHXSwg4IuutjBTVB0cnFTEOnUpThIl+Ii3RweFER0VQQhkEKxk4iFCqU09TvvnZucHO99L2kCDuYHh+Sd77x7T86799yXIBgzZrRUKpXrsMva3yeTxpg7+MxoYeQ0Go1DmGwXNqe1Psng3vuwZ1rwks1mj+KGRY89h03re4J4on1KWAsKivtFsWzvdQB8b8vl8gPt9zFBCeFx3raD4vurYrF4ulAoTAWOx4SYJmLmtV/CP3a/Wq2eo2taLrhu4/OGI5bmrGi/F8TeFcke1roFVWhQjPZLSqXSVR4rtL58Pn8S119gGyI0Ar57sB3Yea05QRJLNlmtCeixLsBaWpBA/2riKnY2X71ePwbfGuy3jCXgq/Lcy1pzgsBtvqGlNYuJK0CV93YAVPUMx/Q8bhSjBv8madJv4bmdmibaMGQY9KEWLdB/cCLeZQL9J086qfzXkhKC/x3rM1rrwa5DWAsbq6B1S9JkRK1WO84xm1qDb5W1La0R8N9k/anWehCBK4GqiIWqyTG7WrOIH72qNfiarDW1RuDeS9DanIMfBHyigXDDBa1ZoE/zZOtaY+hEWuGYJHNWDm2yCG3D+MePsQMFnqoSomqh1gj007PQtiiGdr7UeHNFc/iWGe7Pmbj621qTdDaXFiRpycI/6xtH9F3S/jpkCD5Jw8Rk5ebSmiQtWeoirLv6KB0GpH3QmqWfZDPiMHisRQk/ZorzbRCb7JqSoieHDfqRj28nJt4T68ZxwkVwtaLDgBa41iWIPcHJOFsP9Cush9KPJJ+QP+2lR+QSai3CdFtWYqMn0loX/DMmPjRC5f+ORL5JnwvRul5rLYIGsslqzUVaLCa8BX2HLyfwAx/heq8nyAPiXvL4sx1nLpc7Yid12B4muNgdohcTb4DEJWO6xybZG637MPzekPaE+4ZfUujsX9DakNj26d5cB4ROqWWT0uYGxXRfEUddhChhWgreNjQoGO8z1vuLwHNgDAUGnzOefjsoOH5PmT434YHBBIvl+I/eMNWgtdpO68FD80/+io/5H/kDxtgz4TLjezcAAAAASUVORK5CYII=>