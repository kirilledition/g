# **Phase 0: Preparation & Baseline Benchmarking**

Before writing the core engine, we must establish a reproducible environment, standard datasets, and ground-truth baselines. This ensures development can happen on any server without pushing massive datasets to GitHub.

## **1\. Project Structure**

The repository uses Maturin's unified mixed-layout structure to support Rust (src/lib.rs) and Python (src/g/) side-by-side, coordinated by scripts/ and a Justfile.

├── data/                  \# .gitignore'd: Raw 1KG data and phenotypes  
├── scripts/               \# Standalone Python/Bash helper scripts  
│   ├── fetch\_1kg.py       \# Downloads 1000 Genomes data  
│   ├── simulate\_phenos.py \# Generates deterministic test phenotypes  
│   └── benchmark.py       \# Runs baselines & generates hardware reports  
├── src/                   \# Unified Source Directory  
│   ├── lib.rs             \# Rust PyO3 Extension Module  
│   └── g/                 \# Python Package Code  
├── tests/                 \# Pytest suite testing against baselines  
├── Justfile               \# Task runner (e.g., \`just setup-data\`)  
└── Cargo.toml             \# Rust dependencies

## **2\. Test Data Generation (just setup-data)**

* **The Data:** Use the 1000 Genomes Phase 3 dataset (in PLINK format).  
* **The Script:** scripts/fetch\_1kg.py will automatically download a specific chromosome (e.g., Chr 22\) from the Cog-Genomics public repository to save bandwidth.  
* **The Slice:** The script will use a local plink2 binary to create a "toy slice" (e.g., 5,000 variants and 1,000 samples) for instant unit testing, while keeping the full Chr 22 for performance benchmarking.  
* **The Phenotypes:** scripts/simulate\_phenos.py will read the .fam file and use numpy.random with a **fixed seed (e.g., 42\)** to generate:  
  * pheno\_cont.txt: A continuous trait (Standard Normal distribution).  
  * pheno\_bin.txt: A binary trait (0/1 with a 30% case prevalence).  
  * covariates.txt: Age and Sex (to test covariate regression).

## **3\. Ground Truth Baselines (just benchmark-baselines)**

scripts/benchmark.py will orchestrate the baseline runs, capture hardware specs (using psutil, cpuinfo, and GPUtil/torch.cuda), and output a benchmark\_report.json.

**PLINK2 Baseline Command (Linear/Logistic):**

\# Continuous  
plink2 \--bfile data/1kg\_chr22\_full \--pheno data/pheno\_cont.txt \--covar data/covariates.txt \--glm \--out data/baselines/plink\_cont

\# Binary (Firth fallback)  
plink2 \--bfile data/1kg\_chr22\_full \--pheno data/pheno\_bin.txt \--covar data/covariates.txt \--glm firth-fallback \--out data/baselines/plink\_bin

**Regenie Baseline Command (Mixed Linear Model):**

\# Step 1: Whole Genome Regression (Ridge/LOCO)  
regenie \--step 1 \--bed data/1kg\_chr22\_full \--phenoFile data/pheno\_bin.txt \--covarFile data/covariates.txt \--bt \--bsize 1000 \--out data/baselines/regenie\_step1

\# Step 2: Association Testing (Firth)  
regenie \--step 2 \--bgen data/1kg\_chr22\_full.bgen \--phenoFile data/pheno\_bin.txt \--covarFile data/covariates.txt \--bt \--firth \--approx \--pred data/baselines/regenie\_step1\_pred.list \--out data/baselines/regenie\_step2  
