# **AI Agent Instructions for GWAS Engine**

You are an AI coding assistant working on high-performance GPU-accelerated GWAS engine with JAX

## **Environment & Tooling**

* **Package Management:** uv (Python). Run tooling via `nix develop --command ...`
* **Task Runner:** just. Always check the Justfile for available project-specific commands.  
* **Linting & Formatting:** Ensure code passes before committing (use nix develop):  
  * just check
  * just test
   
## **Repository Structure**

* src/: Unified source directory.  
* tests/: Pytest suite for mathematical regressions and correctness.  
* scripts/: Dev-ops and preparation scripts (fetch\_1kg.py, benchmark.py).  
* data/: Local git-ignored directory for 1KG variants and simulated phenotypes. **Never commit files in this directory.**  
* docs/: Project documentation.

## **Coding Standards (Strictly Enforced)**

**You must strictly adhere to the rules defined in [styleguide](docs/STYLEGUIDE.md).** Do not write code without reading it. Key highlights include:

* 100% strict type coverage.  
* Full-word variable names only.  
* No bare tuples for multiple return values (dataclass required).  
* Docstrings must be in Google format without type duplication.
* Default to module-qualified imports.