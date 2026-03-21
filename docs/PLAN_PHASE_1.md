# **Phase 1: Foundation & plink Parity**

## **1\. Phase Objective**

The goal of Phase 1 is to build a pure Python Minimum Viable Product (MVP) that achieves 100% mathematical parity with plink2 for both continuous (linear) and binary (logistic) traits on the CPU.

This phase focuses strictly on **correctness, API design, and mathematical foundation**. We will rely on highly optimized Python libraries that are backed by Rust and C++ to ensure our baseline is already reasonably fast before we start writing custom hardware kernels.

## **2\. The Tech Stack**

### **Compute Engine: JAX**

* **Why:** JAX acts as a high-performance, drop-in replacement for NumPy. Because JAX compiles down to XLA (Accelerated Linear Algebra), the exact same Python code we write for the CPU in Phase 1 can be JIT-compiled to run on the GPU in Phase 2\. It also features automatic differentiation, which will be crucial for implementing Firth penalized logistic regression later.

### **Data Manipulation: Polars**

* **Why:** Polars is written in Rust, uses the Apache Arrow memory model, and heavily outperforms pandas.  
* **JAX Integration:** Polars natively supports zero-copy handoffs to JAX via df.to\_jax() and the DLPack protocol. This means we can process massive phenotype and covariate tables in Rust and hand the memory pointers directly to the JAX compute engine without Python serialization overhead.

### **Genomic I/O: bed-reader**

* **Why:** Maintained by the fastlmm team, bed-reader is a highly optimized, Rust-backed reader for PLINK .bed files. It will allow us to stream genotype chunks directly into JAX arrays.  
* **Future Extension (BGEN):** bed-reader is strictly for .bed files. For Phase 3 (UK Biobank/BGEN support), we will evaluate wrapping bgen-reader or writing a custom Rust BGEN parser that outputs directly to Apache Arrow.

## **3\. Core Architecture**

The Python package will be decoupled into strict modules to allow for future Rust strangulation.

* **io.py**:  
  * Parses .fam and .bim files using standard Python/Polars.  
  * Streams .bed variants using bed-reader.  
  * Reads phenotypes and covariates using Polars and strictly converts them to jax.Array.  
* **compute/linear.py**:  
  * JAX-compiled Ordinary Least Squares (OLS) solver for continuous traits.  
  * Fast matrix inversion and standard error calculation.  
* **compute/logistic.py**:  
  * JAX-compiled Newton-Raphson solver for binary traits.  
* **cli.py**:  
  * User interface mirroring plink arguments (e.g., \--bfile, \--pheno, \--covar).

## **4\. Execution Plan**

### **Step 1: The Polars-to-JAX Data Bridge**

* Implement the Polars reader for phenotype and covariate files.  
* Handle missing data (NaN imputation/filtering).  
* Build the to\_jax() zero-copy bridge to ensure the design matrix (![][image1]) and phenotype vector (![][image2]) are correctly formatted as float32/float64 JAX arrays.

### **Step 2: Genotype Streaming**

* Wrap bed-reader to yield chunks of genotypes (e.g., 10,000 variants at a time).  
* Implement on-the-fly mean imputation for missing genotypes (standard plink behavior).

### **Step 3: Math Kernels**

* **Linear Regression:** Implement vectorized OLS using jax.numpy.linalg. Calculate Betas, Standard Errors, T-statistics, and P-values.  
* **Logistic Regression:** Implement iterative Re-weighted Least Squares (IRLS) using JAX's jax.lax.while\_loop for fast, compiled convergence.

### **Step 4: The Validation Harness**

* Write pytest test suites that run the new engine against the data/1kg\_chr22\_full dataset generated in Phase 0\.  
* Assert that the generated Betas and P-values match the data/baselines/plink\_cont and plink\_bin outputs within a strict floating-point tolerance (e.g., 1e-6).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAfCAYAAADjuz3zAAABQklEQVR4XmNgGAXDC8jLy8/ChuXk5DoIqREXF+dGNgsFABVIAg1JA+JHQPZ/EFZQUHAAYgEkNYFAfBcq/x6odoaioqI8sjk4AVCDJsxgdDmgIWYgA0E0uhwxgAVmsJKSEj+yBNCFt0AGI4uRBICaT0ANnwLkMkLF9gANLoPxyQKysrI6MFcDcTQwnBuA9Cx0dWQBJINB+DS6PNkAaNh1mMFExzwxAOh9e5jBQLYHujxZAJTg0YLiLboasoA8IlfBgwNdDckAaEgwEO+Bsi2B+CfIYGBSc0FXSzSQh2TTR2hiOVBXP0EWJxpAc1UwujgQMMKCQ0ZGhhNdEi8AZV1oOGLNVUC5f1DDLdHlsAKgK7WBin9BNS1Cl4cCkIt3Q9U8A6ZpfXQFcIAlOYExML1mIKsDim1BVwPD6AXUKBgFo4AAAABWvHiTh9ObRAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAgCAYAAADwvkPPAAABIUlEQVR4Xu2UTUrDUBSF46CgILT+ZJK/l2QmCA4CpUtoxx12AV2CK+gOHAXBDbgGcSSCYzchiCOHkp6L95XkkJo+KBQkH9xB7jnvcPIC8byew2OMKbfMjL1Ci69M03RkxaptYFhSjofdMftkfN8/bRjrYkMgwjC8UN8Laxsg3tUCM9Yt0H7yPB/yvkEQBJcwvmvYI+sC9l9Zlo153wrMKw37ZA0hRlrxfitSX8OqKIpO7N78frWqKIpB3d8JDn3LwSRJbvV5jnlC+Dl7O8HBZ233qk3lngz7diKO42v7qjpz9jhhg/CqD6y5clRrtWDRCdzPjQa9odkZ604g5H4vrQRppGFXrDlj74v3O4PDHzaEB7+dKfv/hAMobML+np5/wRq44HNJ6s35hAAAAABJRU5ErkJggg==>