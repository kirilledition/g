# Project Vision & Guiding Principles

_To: Team Lead & Founding Engineers_  
_From: Project Founder_  
_Subject: What we are building, how we are building it, and why._

> This document captures intent and guiding principles. For detailed execution phases and milestones, see the [Roadmap](ROADMAP.md), [Phase 0 plan](PLAN_PHASE_0.md), and [Phase 1 plan](PLAN_PHASE_1.md).

## 1. The Core Objective (The "What")

We are building a next-generation, hardware-accelerated Genome-Wide Association Study (GWAS) engine.

The current State of the Art (SOTA) tools—like plink2 and regenie—are phenomenal, but they are heavily CPU-bound and written in legacy C++. We are building a tool designed for the modern accelerator era (Multi-GPU, TPUs, Cerebras) capable of chewing through biobank-scale datasets (like the UK Biobank) orders of magnitude faster than current tools.

## 2. The Ultimate End-State (The "Where we are headed")

When this project is "done," it will exist in two forms, powered by the exact same core engine:

- **For Data Scientists (The Python Package):** A highly polished Python library (`pip install g`) where users can write simple scripts, interact with their data in Polars/Arrow, and let the library handle the complex GPU memory management and math under the hood.
- **For Pipelines (The Rust CLI):** A standalone, compiled Rust binary that can be dropped into bash scripts or cloud pipelines without needing a Python environment at all.

## 3. The Strategy: The Strangler Fig Pattern (The "How")

We are not going to spend 2 years building a massive C++/Rust/CUDA engine in stealth, only to release it and find out our API is terrible or our math is slightly off.

We will build pragmatically:

- **Start High-Level:** We will build the MVP entirely in Python, using JAX (for XLA-compiled math) and Polars (for I/O). It will be correct, it will have a great API, but it might not be the fastest tool in the world yet.
- **Strangulate the Bottlenecks:** Once milestone-level parity is achieved (starting with plink2 and later regenie), we will profile it. We will ruthlessly rip out the slowest Python components module-by-module and replace them with zero-copy Rust FFI, custom Triton kernels, or raw CUDA.
- **Continuous Delivery:** At every step, we have a working, testable product that is constantly getting faster.

## 4. The "Boss Fights" (Major Technical Milestones)

Our roadmap is defined by dethroning current standards:

- **Boss 1 (plink parity):** Match standard linear/logistic regression math on the CPU.
- **Boss 2 (The PCIe Bottleneck):** Move the compute to the GPU and beat plink2 speed. This requires custom Rust/GPU kernels that can ingest compressed 2-bit genomic data directly into VRAM without uncompressing it in host RAM.
- **Boss 3 (regenie parity):** Implement Mixed Linear Models (MLMs) for massive biobanks. This requires Ridge Regression, Leave-One-Chromosome-Out (LOCO) predictions, and Firth penalized logistic regression for rare variants—all batched across massive GPU matrix multiplications.

## 5. The "Wishlist" (Quality of Life & Big Features)

Once we achieve SOTA speed, we will implement the features that modern computational biologists actually want, but legacy tools struggle to provide:

- **Log10 P-values:** Calculating p-values directly in log10 space to completely avoid 64-bit float underflow on highly significant biobank hits.
- **Modern I/O:** Outputting summary statistics natively to Apache Parquet or Arrow IPC, bypassing the need for massive CSV text files.
- **Hardware Agnosticism:** Seamlessly scaling from a local RTX 4080 to cloud TPU pods or Cerebras waferscale engines via OpenXLA.
- **Ecosystem Integration:** Native compatibility with modern genomics frameworks like Hail.
