# **Development Style Guide & Project Rules**

## **1\. Core Philosophy**

Optimize for explicit, self-documenting code over terse keystroke-saving. Prioritize legibility and mathematical clarity. Engineers should never have to guess what variables represent.

## **2\. Python Guidelines**

### **Tooling & Environment**

* **Package Management:** Strictly managed by **uv**.  
* **Linting & Formatting:** Managed entirely by **ruff** (ruff format is the source of truth).

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

## **3\. Rust Guidelines**

Apply the same core philosophy, naming, and return-type rules as Python to ensure a seamless mental model.

### **Tooling**

* **Formatting:** Managed strictly by rustfmt (cargo fmt).  
* **Linting:** Compile with \#\!\[warn(clippy::pedantic)\] to enforce idiomatic Rust.

### **Naming Conventions**

* **Rule:** Full words with standard Rust casing (snake\_case for variables, CamelCase for structs).  
  * Bad: let g \= read\_bed();  
  * Good: let genotype\_matrix \= read\_bed();

### **Return Types & Structs**

* **Rule:** No bare tuples for complex returns. Define a struct.  
  **Bad:**  
  fn compute\_regression(features: \&Array2\<f32\>, targets: \&Array1\<f32\>) \-\> (Array1\<f32\>, Array1\<f32\>) { ... }

  **Good:**  
  pub struct RegressionResult {  
      pub betas: Array1\<f32\>,  
      pub standard\_errors: Array1\<f32\>,  
  }

  fn compute\_regression(features: \&Array2\<f32\>, targets: \&Array1\<f32\>) \-\> RegressionResult { ... }

### **Memory & FFI (The Boundary)**

* When crossing the FFI boundary (via PyO3), explicitly name memory ownership if copying (e.g., cloned\_matrix vs matrix\_view).  
* Use rustdoc (///) to explicitly document any memory allocations happening during the call.
