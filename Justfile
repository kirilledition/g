# Justfile for GWAS Engine (g)

set allow-duplicate-recipes := true

# --- Data Preparation ---

# Download and prepare 1KG test data
setup-data:
    uv run scripts/fetch_1kg.py
    uv run scripts/simulate_phenos.py

# Run plink2/regenie baselines and generate hardware report
benchmark-baselines: setup-data
    uv run scripts/benchmark.py

# --- Development ---

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
    uv run pyright . || echo "Please ensure pyright/ty is installed"

# Run all checks (format, lint, typecheck)
check: format lint typecheck

# Run tests
test:
    uv run pytest tests/
