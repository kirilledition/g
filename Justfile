# Justfile for GWAS Engine (g)

set allow-duplicate-recipes := true

data_dir := env_var_or_default('GWAS_ENGINE_DATA_DIR', 'data')

# --- Data Preparation ---

setup-data:
    uv run python scripts/fetch_1kg.py
    uv run python scripts/simulate_phenos.py

# Run PLINK2/Regenie baselines and generate hardware report (excludes slow Hail benchmarks by default)
benchmark-baselines: setup-data
    uv run python scripts/benchmark.py

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU
benchmark-regenie-comparison-cpu: setup-data
    uv run python scripts/benchmark_regenie_comparison.py --cpu-only

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU+GPU
benchmark-regenie-comparison-gpu: setup-data
    uv run python scripts/benchmark_regenie_comparison.py --include-gpu

# Alias for comparison benchmark (CPU-only default)
benchmark-regenie-comparison: benchmark-regenie-comparison-cpu

# Run full baselines including Hail (slow - requires cached MatrixTable)
benchmark-baselines-full: setup-data
    HAIL_INCLUDE=1 uv run python scripts/benchmark.py

# --- Development ---

# Run REGENIE step 2 with local baseline predictions
regenie2-linear:
    uv run g regenie2-linear --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --pheno {{data_dir}}/pheno_cont.txt --pheno-name phenotype_continuous --covar {{data_dir}}/covariates.txt --covar-names age,sex --pred {{data_dir}}/baselines/regenie_step1_qt_pred.list --out {{data_dir}}/regenie2_linear

# Run CPU/GPU JAX runtime probe
probe-jax:
    uv run python scripts/probe_jax_runtime.py

# Benchmark PLINK reader and preprocessing paths
benchmark-plink-reader:
    uv run python scripts/benchmark_plink_reader.py

# Benchmark BGEN float32 read paths
benchmark-bgen-reader:
    uv run python scripts/benchmark_bgen_reader.py

# Profile full REGENIE step 2 execution
profile-regenie2-linear-detailed:
    mkdir -p {{data_dir}}/profiles/regenie2_linear_detailed
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 uv run python scripts/profile_regenie2_linear_detailed.py --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --pheno {{data_dir}}/pheno_cont.txt --pheno-name phenotype_continuous --covar {{data_dir}}/covariates.txt --covar-names age,sex --pred {{data_dir}}/baselines/regenie_step1_qt_pred.list --output-dir {{data_dir}}/profiles/regenie2_linear_detailed --report-name regenie2_linear_chr22_full --enable-jax-trace --enable-memory-profile --cprofile-sort cumulative

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU
profile-regenie-comparison-cpu: setup-data
    uv run python scripts/profile_regenie_comparison.py --cpu-only

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU+GPU
profile-regenie-comparison-gpu: setup-data
    uv run python scripts/profile_regenie_comparison.py --include-gpu

# Alias for unified profiling comparison (CPU-only default)
profile-regenie-comparison: profile-regenie-comparison-cpu

# Format code
format:
    uv run ruff format .
    cargo fmt

# Lint code
lint:
    uv run ruff check . --fix
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic

# Type check Python code
typecheck:
    uv run ty check src tests scripts

# Run all checks (format, lint, typecheck)
check: format lint typecheck

# Run CI lint checks without installing the project package
ci-lint:
    uv sync --group dev --frozen --no-install-project
    uv run --no-sync ruff check .

# Run CI type checks without installing the project package
ci-typecheck:
    uv sync --group dev --frozen --no-install-project
    uv run --no-sync ty check src tests scripts

# Run CI tests that exclude heavy data- and parity-dependent suites
ci-test:
    uv sync --group dev --frozen
    uv run --no-sync pytest tests/ -m "not phase0_data and not phase1_parity"

# Run tests
test:
    uv run pytest tests/

upgrade-deps:
    uv sync -U --group dev --group gpu
    nix flake update
