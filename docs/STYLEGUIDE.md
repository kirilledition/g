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

### **Return Types**

* **Rule:** Never return bare tuples for multiple values. Use a typing.NamedTuple.  
  **Bad:**  
  def compute\_regression(features: jax.Array, targets: jax.Array) \-\> tuple\[jax.Array, jax.Array\]:  
      return betas, standard\_errors

  **Good:**  
  from typing import NamedTuple  
  import jax

  class RegressionResult(NamedTuple):  
      betas: jax.Array  
      standard\_errors: jax.Array

  def compute\_regression(features: jax.Array, targets: jax.Array) \-\> RegressionResult:  
      return RegressionResult(betas=betas, standard\_errors=standard\_errors)

### **JAX Array Containers**

* **Rule:** For containers consisting exclusively of JAX arrays (all fields are `jax.Array`), use `@dataclass` with `jax.tree_util.register_dataclass`.  
  JAX provides optimized C++ pytree handling for dataclasses, resulting in ~25-35x faster tree operations (flatten/unflatten) compared to NamedTuple. This matters because JAX performs tree flattening/unflattening on every JIT compilation and transformation.

  **Bad (for pure JAX array containers):**  
  from typing import NamedTuple  
  import jax

  class LogisticState(NamedTuple):  
      coefficients: jax.Array  
      converged\_mask: jax.Array  
      iteration\_count: jax.Array

  **Good (for pure JAX array containers):**  
  from dataclasses import dataclass  
  import jax

  @jax.tree\_util.register\_dataclass  
  @dataclass  
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

* **Rule:** Continue using NamedTuple for containers with mixed types (strings, booleans, numpy arrays alongside JAX arrays). NamedTuple automatically handles non-array metadata without requiring explicit `meta_fields` registration.  

  **Good (for mixed types):**  
  from typing import NamedTuple  
  import jax  
  import numpy.typing as npt  
  import numpy as np

  class GenotypeChunk(NamedTuple):  
      """Genotype matrix with mixed metadata.

      Attributes:
          genotypes: JAX array of genotypes.
          has\_missing\_values: Python boolean flag.
          allele\_frequency: JAX array of frequencies.

      """  
      genotypes: jax.Array  
      has\_missing\_values: bool  
      allele\_frequency: jax.Array

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