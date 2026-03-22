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
