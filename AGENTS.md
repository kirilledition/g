# **AI Agent Instructions for GWAS Engine**

You are an AI coding assistant operating within a high-performance, mixed-language (Python \+ Rust) scientific computing repository building a GPU-accelerated GWAS engine. Read these instructions before proposing changes or running commands.

## **Environment & Tooling**

* **Package Management:** uv (Python) and cargo (Rust).  
* **Task Runner:** just. Always check the Justfile for available project-specific commands.  
* **Linting & Formatting:** Ensure code passes before committing:  
  * Python: uv run ruff format ., uv run ruff check . \--fix, and uv run ty (type-checking).  
  * Rust: cargo fmt and cargo clippy \--workspace \--all-targets \-- \-D warnings.

## **Repository Structure**

* pyproject.toml / Cargo.toml: Maturin mixed-layout configuration.  
* python/: Phase 1 pure Python MVP codebase.  
* src/: Phase 2+ Rust performance rewrites and FFI entry points.  
* tests/: Pytest suite for mathematical regressions and correctness.  
* scripts/: Dev-ops and preparation scripts (fetch\_1kg.py, benchmark.py).  
* data/: Local git-ignored directory for 1KG variants and simulated phenotypes. **Never commit files in this directory.**  
* docs/: Project documentation.

## **Coding Standards (Strictly Enforced)**

**You must strictly adhere to the rules defined in [styleguide](docs/STYLEGUIDE.md).** Do not write code without reading it. Key highlights include:

* 100% strict type coverage.  
* Full-word variable names only (no abbreviations or single-letter math variables).  
* No bare tuples for multiple return values (NamedTuple or struct required).  
* Docstrings must be in Google format without type duplication.

## **Testing Standards**

* Validate mathematical correctness by asserting floating-point parity against the Phase 0 baselines located in data/baselines/.  
* Use jax.numpy.allclose or numpy.testing.assert\_allclose (e.g., atol=1e-6) for comparing matrix outputs. Never use exact equality (==) for floating-point math.

## **Git & PR Workflow**

* **Conventional Commits:** Use standard prefixes (feat:, fix:, perf:, docs:, test:). Focus on *what* and *why*.  
* **Pre-commit:** Code must pass ruff, ty, and cargo clippy hooks before committing.

## **Boundaries & Constraints**

* **Ask First Before:**  
  * Adding new dependencies.  
  * Changing mathematical formulations of the regression solvers.  
* **Never Do This:**  
  * Commit credentials, API keys, or genomic data files.  
  * Bypass the type-checker (\# type: ignore) without explicit permission and a detailed explanatory comment. 