# **AI Agent Instructions for GWAS Engine**

You are an AI coding assistant working on high-performance GPU-accelerated GWAS engine with JAX

## **Environment & Tooling**

* **Package Management:** uv (Python).
* **Task Runner:** just. Always check the Justfile for available project-specific commands.  
* **Linting & Formatting:** Ensure code passes before committing (use nix develop):  
  * just check
  * just test
* New features should be implemented in separate git worktrees in ~/Projects/g-worktrees. History of that brunch should be on github. After feature is done, integrate to main and remove the worktree
* On gauss server use slurm srun on landau node to run code that requires nvidia gpu. Use cantor or other free node with all available cores to run cpu workloads. Do not run heavy computation, compilations, large test suits on head gauss node.
   
## **Repository Structure**

* src/: Unified source directory.  
* tests/: Pytest suite for mathematical regressions and correctness.  
* scripts/: Dev-ops and preparation scripts (fetch\_1kg.py, benchmark.py).  
* data/: Local git-ignored directory for 1KG variants and simulated phenotypes. **Never commit files in this directory.**  
* documentation/: Project documentation.

## **Documentation Rule**

When changing user-facing CLI behavior, configuration, input/output contracts,
runtime behavior, performance assumptions, or deployment workflow, update the
relevant page under documentation/ in the same branch. Run `just docs-build` before
finishing documentation changes.

## **Coding Standards (Strictly Enforced)**

**You must strictly adhere to the rules defined in [styleguide](documentation/development/STYLEGUIDE.md).** Do not write code without reading it. Key highlights include:

* 100% strict type coverage.  
* Full-word variable names only.  
* No bare tuples for multiple return values (dataclass required).  
* Docstrings must be in Google format without type duplication.
* Default to module-qualified imports.
