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
├── pyproject.toml         \# Python packaging & Maturin config  
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

## **4. Phase 0 Realization Summary**

### **What has been done**
- Created the `scripts/` directory with three key scripts:
  - `fetch_1kg.py`: Downloads the 1000 Genomes Phase 3 data (Chr 22) in `.pgen`/`.pvar`/`.psam` format from the Cog-Genomics public repository, uncompresses it, and utilizes `plink2` to convert it into both standard PLINK BED/BIM/FAM (for `plink2` and `regenie` step 1 baselines) and BGEN format (for `regenie` step 2 baseline). It also creates a "toy slice" of 5,000 variants.
  - `simulate_phenos.py`: Uses `numpy` and `pandas` with a fixed seed to read the generated `.fam` file and output standardized `pheno_cont.txt`, `pheno_bin.txt`, and `covariates.txt` formatted for `plink2` and `regenie`.
  - `benchmark.py`: A wrapper script that checks for the availability of `plink2` and `regenie` in the PATH, detects the current system's CPU and GPU hardware specifications using `psutil`, `cpuinfo`, and `GPUtil`, executes the baseline commands outlined in Phase 0, and compiles a comprehensive JSON report with execution times and command statuses.
- Added `data/.gitignore` to make sure raw and processed data does not get checked into the git history.
- Set up the `Justfile` to orchestrate `just setup-data` and `just benchmark-baselines`.
- Added required Python packages (`numpy`, `pandas`, `psutil`, `py-cpuinfo`, `GPUtil`) to the `uv` dev-dependencies via `pyproject.toml`.

### **Problems met**
- Identifying exactly which PLINK-formatted files from the 1000 Genomes project were required. To optimize download speeds, I targeted compressed `.pgen.zst` files hosted by Cog-Genomics, then used `plink2` to process them locally into `bed` and `bgen` formats.
- The `benchmark.py` needed an efficient way to capture detailed hardware info, which required introducing several additional python dev dependencies (`psutil`, `py-cpuinfo`, `GPUtil`) to robustly gather CPU and CUDA device details.

### **What requires help to finish Phase 0 properly**
- **Test Baseline Validity:** We need human verification to execute `just benchmark-baselines` on a system with the full Nix environment (containing both `plink2` and `regenie`) to confirm that Regenie correctly reads the `.bgen` and `.bed` files generated by `fetch_1kg.py`.
- **Regenie Parameters Check:** The exact format for the `--pred` list required by `regenie` step 2 has historically been sensitive to file layouts. We must run the pipeline completely end-to-end to ensure the `.list` output from Step 1 perfectly meshes with the `regenie_step2` command generated in `benchmark.py`.
- **Establish the `tests/` Suite:** Now that the test data generators are available, we need to create the `pytest` baseline harness in `tests/` that checks output arrays against the PLINK and Regenie `.glm.linear`/`.glm.logistic` reports saved in `data/baselines/`.
