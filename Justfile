# Justfile for GWAS Engine (g)

set allow-duplicate-recipes := true

data_dir := env_var_or_default('GWAS_ENGINE_DATA_DIR', 'data')

# --- Data Preparation ---

# Download and prepare 1KG test data
setup-data:
    uv run python scripts/fetch_1kg.py
    uv run python scripts/simulate_phenos.py

# Run plink2/regenie baselines and generate hardware report
benchmark-baselines: setup-data
    uv run python scripts/benchmark.py

# --- Development ---

# Run the Phase 1 linear engine
phase1-linear:
    uv run g --bfile {{data_dir}}/1kg_chr22_full --pheno {{data_dir}}/pheno_cont.txt --pheno-name phenotype_continuous --covar {{data_dir}}/covariates.txt --covar-names age,sex --glm linear --out {{data_dir}}/phase1_linear

# Run the Phase 1 logistic engine
phase1-logistic:
    uv run g --bfile {{data_dir}}/1kg_chr22_full --pheno {{data_dir}}/pheno_bin.txt --pheno-name phenotype_binary --covar {{data_dir}}/covariates.txt --covar-names age,sex --glm logistic --out {{data_dir}}/phase1_logistic

# Evaluate Phase 1 parity and runtime versus PLINK
phase1-evaluate:
    uv run python scripts/evaluate_phase1.py

# Run CPU/GPU JAX runtime probe
probe-jax:
    uv run python scripts/probe_jax_runtime.py

# Benchmark JAX transfer/compute/format surfaces
benchmark-jax:
    uv run python scripts/benchmark_jax_execution.py

# Sweep chunk sizes for JAX compute kernels
benchmark-jax-chunks:
    uv run python scripts/benchmark_jax_chunk_sweep.py

# Sweep full logistic loop runtime across chunk sizes
benchmark-logistic-loop:
    uv run python scripts/benchmark_logistic_loop_sweep.py

# Benchmark logistic fallback-heavy chunks
benchmark-logistic-fallback:
    uv run python scripts/benchmark_logistic_fallback.py

# Benchmark PLINK reader and preprocessing paths
benchmark-plink-reader:
    uv run python scripts/benchmark_plink_reader.py

# Capture a full-chromosome logistic JAX trace and memory profile
profile-logistic-chr22:
    mkdir -p {{data_dir}}/profiles/jax_logistic_full_chr22
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 uv run python scripts/profile_full_chr22_jax.py --bfile {{data_dir}}/1kg_chr22_full --pheno {{data_dir}}/pheno_bin.txt --pheno-name phenotype_binary --covar {{data_dir}}/covariates.txt --covar-names age,sex --glm logistic --chunk-size 512 --trace-dir {{data_dir}}/profiles/jax_logistic_full_chr22 --memory-profile {{data_dir}}/profiles/jax_logistic_full_chr22_memory.prof

# Capture a full-chromosome linear JAX trace and memory profile
profile-linear-chr22:
    mkdir -p {{data_dir}}/profiles/jax_linear_full_chr22
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 uv run python scripts/profile_full_chr22_jax.py --bfile {{data_dir}}/1kg_chr22_full --pheno {{data_dir}}/pheno_cont.txt --pheno-name phenotype_continuous --covar {{data_dir}}/covariates.txt --covar-names age,sex --glm linear --chunk-size 512 --trace-dir {{data_dir}}/profiles/jax_linear_full_chr22 --memory-profile {{data_dir}}/profiles/jax_linear_full_chr22_memory.prof

# Capture detailed cProfile + JAX profiler report for full chr22 logistic
profile-logistic-detailed:
    mkdir -p {{data_dir}}/profiles/logistic_detailed
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 uv run python scripts/profile_full_chr22_detailed.py \
        --bfile {{data_dir}}/1kg_chr22_full \
        --pheno {{data_dir}}/pheno_bin.txt \
        --pheno-name phenotype_binary \
        --covar {{data_dir}}/covariates.txt \
        --covar-names age,sex \
        --glm logistic \
        --chunk-size 512 \
        --output-dir {{data_dir}}/profiles/logistic_detailed \
        --report-name logistic_chr22_full \
        --enable-jax-trace \
        --enable-memory-profile \
        --cprofile-sort cumulative

# Capture detailed cProfile + JAX profiler report for full chr22 linear
profile-linear-detailed:
    mkdir -p {{data_dir}}/profiles/linear_detailed
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 uv run python scripts/profile_full_chr22_detailed.py \
        --bfile {{data_dir}}/1kg_chr22_full \
        --pheno {{data_dir}}/pheno_cont.txt \
        --pheno-name phenotype_continuous \
        --covar {{data_dir}}/covariates.txt \
        --covar-names age,sex \
        --glm linear \
        --chunk-size 512 \
        --output-dir {{data_dir}}/profiles/linear_detailed \
        --report-name linear_chr22_full \
        --enable-jax-trace \
        --enable-memory-profile \
        --cprofile-sort cumulative

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
    uv run ty check .

# Run all checks (format, lint, typecheck)
check: format lint typecheck

# Run tests
test:
    uv run pytest tests/
