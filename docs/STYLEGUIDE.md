# **Development Style Guide & Project Rules**

## **1\. Core Philosophy**

Optimize for explicit, self-documenting code over terse keystroke-saving. Prioritize legibility and mathematical clarity. Engineers should never have to guess what variables represent.

## **2\. Python Guidelines**

### **Tooling & Environment**

* **Package Management:** Strictly managed by **uv**.  
* **Linting & Formatting:** Managed entirely by **ruff** (ruff format is the source of truth).

### **Imports**

* **Rule:** Default to module-qualified imports. Import the module, then access members through its namespace.  
  * Good: `import scipy.stats`; `scipy.stats.norm(...)`  
  * Good: `import sklearn.linear_model`; `sklearn.linear_model.LinearRegression(...)`  
  * Bad: `from scipy.stats import norm`  
  * Bad: `from sklearn.linear_model import LinearRegression`
* **Rule:** The only approved direct-import exceptions are `from pathlib import Path` and `from dataclasses import dataclass`.
* **Rule:** Import `typing` as a module and qualify all names. Never import from `typing`.  
  * Good: `import typing`; `typing.TYPE_CHECKING`; `typing.Any`  
  * Bad: `from typing import TYPE_CHECKING, Any`
* **Rule:** Import `enum` and `collections.abc` as modules and qualify all names.  
  * Good: `import enum`; `enum.StrEnum`  
  * Good: `import collections.abc`; `collections.abc.Iterator`  
  * Bad: `from enum import StrEnum`  
  * Bad: `from collections.abc import Iterator`
* **Rule:** Use conventional aliases only where they are already standard and improve readability.  
  * Approved examples: `import numpy as np`, `import numpy.typing as npt`, `import jax.numpy as jnp`, `import polars as pl`, `import pandas as pd`
* **Rule:** Keep imports out of functions, methods, and classes in production code under `src/g`.
  * Module-scope imports are the default.
  * `if typing.TYPE_CHECKING:` blocks are allowed for annotation-only imports.
  * Tests may use local imports when there is a concrete reason, such as optional dependencies or fixture isolation.
* **Rule:** Relative imports are not allowed.
* **Rule:** In production Python code under `src/g`, import first-party modules rather than first-party members.  
  * Good: `from g import api`; `api.ComputeConfig`  
  * Good: `from g.io import bgen`; `bgen.split_sample_file_line()`  
  * Bad: `from g.api import ComputeConfig`  
  * Bad: `from g.io.bgen import split_sample_file_line`

### **Naming Conventions**

* **Rule:** No abbreviations. No single-letter math variables. Use full, descriptive words.  
  * Bad: X, y, var, log\_var, maf  
  * Good: design\_matrix (or features), phenotypes (or targets), variance, log\_variance, minor\_allele\_frequency  
* **Rule:** No leading underscores for function or method names. We do not use pseudo-private indicators.  
  * Bad: def \_calculate\_variance():  
  * Good: def calculate\_variance():

### **Type Annotations**

* **Rule:** 100% Type Annotation Coverage.  
* Types must pass the **ty** type checker without implicit Any fallbacks. Use exact types (e.g., jax.Array, pl.DataFrame).
* **Rule:** Finite sets of string values must use `enum.StrEnum`, not `str` annotations, `Literal[...]`, or ad-hoc validation sets. This applies to configuration values, modes, formats, codecs, CLI choices, and any other closed choice domain.
* **Rule:** Define an enum in the narrowest valid scope. If it is used in exactly one production file, define it in that file. If it is used in more than one production file, define it in `src/g/types.py`.
* **Rule:** Test-only enums should not live in production modules. If a test-only enum is used in one test file, define it in that test file. If it is shared across multiple test files, define it in `tests/types.py`.

### **Return Types & Structured Data**

* **Rule:** Never return bare tuples for multiple values. Use `@dataclass(frozen=True)`.  
  **Bad:**  
  def compute\_regression(features: jax.Array, targets: jax.Array) \-\> tuple\[jax.Array, jax.Array\]:  
      return betas, standard\_errors

  **Good:**  
  from dataclasses import dataclass  
  import jax

  @dataclass(frozen=True)  
  class RegressionResult:  
      """Result of regression computation.

      Attributes:
          betas: Coefficient estimates.
          standard\_errors: Standard errors of estimates.

      """  
      betas: jax.Array  
      standard\_errors: jax.Array

  def compute\_regression(features: jax.Array, targets: jax.Array) \-\> RegressionResult:  
      return RegressionResult(betas=betas, standard\_errors=standard\_errors)

### **JAX Pytree Containers**

* **Rule:** For containers used inside JAX JIT boundaries (state containers, loop carriers), use `@dataclass(frozen=True)` with `@jax.tree_util.register_dataclass`.  

  **Good (for JIT-internal state):**  
  from dataclasses import dataclass  
  import jax

  @jax.tree\_util.register\_dataclass  
  @dataclass(frozen=True)  
  class LogisticState:  
      """State for batched logistic IRLS.

      Attributes:
          coefficients: Current coefficient estimates.
          converged\_mask: Boolean convergence mask.
          iteration\_count: Iteration counter.

      """  
      coefficients: jax.Array  
      converged\_mask: jax.Array  
      iteration\_count: jax.Array

* **Rule:** For containers with mixed types (strings, booleans, numpy arrays alongside JAX arrays) that are NOT passed through JIT boundaries, use `@dataclass(frozen=True)` without JAX registration.  

  **Good (for I/O and metadata containers):**  
  from dataclasses import dataclass  
  import jax  
  import numpy.typing as npt  
  import numpy as np

  @dataclass(frozen=True)  
  class GenotypeChunk:  
      """Genotype matrix with mixed metadata.

      Attributes:
          genotypes: JAX array of genotypes.
          has\_missing\_values: Python boolean flag.
          allele\_frequency: JAX array of frequencies.

      """  
      genotypes: jax.Array  
      has\_missing\_values: bool  
      allele\_frequency: jax.Array

* **Note on NamedTuple:** Avoid `typing.NamedTuple`. While it supports unpacking and indexing, `@dataclass(frozen=True)` provides better consistency, IDE support, and explicit immutability. The codebase standardizes on dataclasses for all structured return types

### **Documentation**

* Use the **Google Python Style Guide** format for docstrings.  
* **Rule: Omit types in docstrings.** Since types are strictly enforced in the function signature, duplicating them in the docstring creates maintenance overhead.  
* Detail Args:, Returns:, and Raises: with descriptions only.
