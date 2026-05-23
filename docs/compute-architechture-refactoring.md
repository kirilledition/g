No. The compute module is **not optimal yet**. It is much better than the earlier architecture, but I would still rewrite it before adding SPA, exact Firth, or serious multi-phenotype batching.

My view:

```text
linear compute: acceptable, but should be simplified and deduplicated
binary score compute: workable, but duplicated
binary Firth compute: too monolithic and should be split now
overall compute package: transitional, not a clean long-term foundation
```

The main issue is not cosmetic. The current layout makes it too easy for sample-major and variant-major paths, single and multi paths, score and Firth paths, and config/default constants to drift from each other.

---

# What is good already

The current compute module has some good architectural ideas:

```text
prepare global state
prepare chromosome-specific state
compute chunk from chromosome state
```

That separation is correct for REGENIE Step 2 because LOCO predictions change by chromosome while genotypes stream by chunk.

These are good public concepts:

* `prepare_regenie2_linear_state`
* `prepare_regenie2_linear_chromosome_state`
* `compute_regenie2_linear_chunk_from_chromosome_state`
* `prepare_regenie2_binary_state`
* `prepare_regenie2_binary_chromosome_state`
* `compute_regenie2_binary_chunk_from_chromosome_state`

The JAX pytree dataclasses in:

```text
src/g/compute/regenie2_linear/state.py
src/g/compute/regenie2_linear/result.py
src/g/compute/regenie2_binary/state.py
src/g/compute/regenie2_binary/result.py
src/g/compute/regenie2_binary/firth/types.py
```

are also the right direction.

So I would **not** throw away the conceptual execution model. I would rewrite the file/module structure and reduce duplicated kernels.

---

# Main architectural problems

## 1. `regenie2_binary.py` is too large and has too many responsibilities

This is the biggest issue.

Current size:

```text
src/g/compute/regenie2_binary.py                  3042 lines
src/g/compute/regenie2_binary_variant_major.py     420 lines
src/g/compute/regenie2_linear.py                   544 lines
```

`regenie2_binary.py` currently contains all of this in one file:

```text
binary constants
default kernel config
binary state preparation
null logistic IRLS
score test
candidate planning integration
approximate Firth scalar solver
full-model Firth solver
null Firth solver
Firth line search
Firth result merging
sample-major chunk path
multi-trait wrappers
bottom-of-file circular import of variant-major code
```

That is too much. The file is now a correctness risk.

For example, `fit_covariate_only_firth_null_model()` is around `src/g/compute/regenie2_binary.py:2099-2213`, while chunk-level Firth correction is around `src/g/compute/regenie2_binary.py:2703-2965`, while binary score testing is around `src/g/compute/regenie2_binary.py:760-813`. These are different statistical layers and should not live in one module.

### Recommendation

Split binary compute by statistical responsibility, not by random helper grouping.

A better structure would be:

```text
src/g/compute/
  common/
    linalg.py
    pvalue.py
    genotype.py
    result.py
    policy.py

  regenie2_linear/
    state.py
    result.py
    score.py
    api.py

  regenie2_binary/
    config.py
    state.py
    result.py
    null_logistic.py
    score.py
    candidates.py
    variant_major_correction.py
    diagnostics.py

    firth/
      types.py
      null.py
      scalar_approx.py
      full_model.py
      line_search.py
      batch.py

    api.py
```

The engine should import only the public API layer, for example:

```python
from g.compute.regenie2_binary import api as binary_kernel
from g.compute.regenie2_linear import api as linear_kernel
```

The engine should not know about Firth internals, candidate batching internals, or variant-major implementation modules.

---

## 2. There is a circular import smell between binary and binary variant-major code

Earlier import shape:

```text
legacy flat binary module
    imported candidate planning, binary state/result types, and linear helpers

legacy variant-major module
    imported the flat binary module, candidate planning, binary state/result types, and linear helpers
```

That bottom import at `regenie2_binary.py:3042` is a red flag. It exists to break a dependency cycle.

The deeper issue is that `regenie2_binary_variant_major.py` needs helpers from `regenie2_binary.py`, while `regenie2_binary.py` also wants to expose variant-major functionality.

### Fix

Move shared helpers into independent modules.

For example:

```text
common/pvalue.py
  chi_squared_to_log10_p_value

common/linalg.py
  solve_positive_definite_system
  solve_from_positive_definite_matrix

common/genotype.py
  build_regenie_flipped_genotypes
  normalize_high_frequency_diploid_genotypes
  canonicalize_genotype_layout

regenie2_binary/score.py
  binary score-test math, layout-neutral or layout-specific wrappers

regenie2_binary/firth/*.py
  Firth-only helpers
```

Then both sample-major and variant-major code can import from common helpers without importing each other.

Also, binary compute should not import `regenie2_linear` just to get p-value and linear algebra helpers. Currently it does:

* `src/g/compute/regenie2_binary.py:14`
* `src/g/compute/regenie2_binary.py:576-582`
* `src/g/compute/regenie2_binary.py:791-794`
* `src/g/compute/regenie2_binary_variant_major.py:12`
* `src/g/compute/regenie2_binary_variant_major.py:54-57`

That should become `g.compute.common.linalg` and `g.compute.common.pvalue`.

---

## 3. Single-phenotype and multi-phenotype paths should not be separate kernels

The current linear module has separate single and multi functions:

```text
prepare_regenie2_linear_state
prepare_regenie2_multi_linear_state
prepare_regenie2_linear_chromosome_state
prepare_regenie2_multi_linear_chromosome_state
compute_regenie2_linear_chunk_from_chromosome_state
compute_regenie2_multi_linear_chunk_from_chromosome_state
compute_regenie2_linear_chunk_from_chromosome_state_variant_major
compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major
```

Relevant lines:

* `src/g/compute/regenie2_linear.py:53-100`
* `src/g/compute/regenie2_linear.py:103-146`
* `src/g/compute/regenie2_linear.py:190-219`
* `src/g/compute/regenie2_linear.py:223-249`
* `src/g/compute/regenie2_linear.py:253-315`
* `src/g/compute/regenie2_linear.py:319-372`
* `src/g/compute/regenie2_linear.py:376-439`
* `src/g/compute/regenie2_linear.py:443-505`

This works, but it duplicates formulas.

The cleaner design is:

```text
internal compute is always trait-major:
    phenotypes: traits x samples
    results:    traits x variants

single phenotype is just:
    traits = 1
```

Then the public single-trait API can squeeze the result back to 1D.

That would remove a lot of duplication and make multi-phenotype batching a first-class design instead of a parallel feature.

### Better model

Use one internal representation:

```python
LinearState:
    covariate_basis
    phenotype_residual_matrix  # traits x samples

LinearChromosomeState:
    adjusted_residual_matrix
    adjusted_residual_projection_matrix
    adjusted_residual_sum_squares
    degrees_of_freedom

LinearChunkResult:
    beta              # traits x variants
    standard_error    # traits x variants
    chi_squared       # traits x variants
    log10_p_value     # traits x variants
    valid_mask        # traits x variants
```

Then:

```python
compute_linear_chunk(...)
```

always returns `traits x variants`.

The single-trait wrapper becomes tiny:

```python
def compute_single_linear_chunk(...):
    result = compute_linear_chunk(...)
    return squeeze_trait_axis(result)
```

This would have made the QT SE/CHISQ/LOG10P bug easier to fix once instead of fixing several parallel paths.

---

## 4. Sample-major and variant-major formulas are duplicated

This is another important source of statistical drift.

Linear duplicates the same score-test formula across sample-major and variant-major paths:

* sample-major single: `src/g/compute/regenie2_linear.py:253-315`
* sample-major multi: `src/g/compute/regenie2_linear.py:319-372`
* variant-major single: `src/g/compute/regenie2_linear.py:376-439`
* variant-major multi: `src/g/compute/regenie2_linear.py:443-505`

Binary duplicates score-test logic too:

* sample-major score: `src/g/compute/regenie2_binary.py:760-813`
* variant-major score: `src/g/compute/regenie2_binary_variant_major.py:15-76`

This duplication already shows up in the invalid binary variant behavior: both paths set invalid variance cases to `CHISQ=0` and `LOG10P=0` instead of NaN.

### Better design

Pick one canonical compute layout internally, preferably:

```text
variant-major: variants x samples
```

That matches the strategic BGEN path better: BGEN decoding is variant-oriented, and your trusted BGEN path is already moving toward fused variant-major decode/stat delivery.

Then expose layout adapters:

```python
def as_variant_major(genotype_chunk, layout):
    if layout == VARIANT_MAJOR:
        return genotype_chunk
    if layout == SAMPLE_MAJOR:
        return genotype_chunk.T
```

After that, the core formulas should operate on `Gv`:

```text
Gv = variants x samples
R  = traits x samples
X  = samples x covariates
```

Linear score-test formula can then be centralized once.

Binary score-test formula can also be centralized once, with genotype flipping performed once per chunk.

---

## 5. Multi-binary currently vmaps the single-trait path and repeats genotype work

The multi-binary path currently does this:

* `src/g/compute/regenie2_binary.py:869-890`
* `src/g/compute/regenie2_binary.py:893-917`

It builds a single-trait chromosome state for each trait, then calls the single-trait chunk function under `jax.vmap`.

That is clean from a code-reuse perspective, but it is not optimal for the app’s main performance goal.

The genotype chunk is identical for all traits, but inside the single-trait score path you do genotype canonicalization/flipping again:

* `src/g/compute/regenie2_binary.py:767-769`
* `src/g/compute/regenie2_binary_variant_major.py:22-24`

For true multi-phenotype batching, this is the wrong direction. The genotype transformation should happen once:

```text
decode chunk once
transfer chunk once
flip/canonicalize genotype once
compute many traits
```

### Better design

Split binary score computation into two phases:

```python
canonical_genotype = prepare_binary_genotype_chunk(Gv)
score_result = compute_binary_score_for_trait_batch(chromosome_state, canonical_genotype)
```

Where:

```python
prepare_binary_genotype_chunk(...)
    - converts to variant-major
    - applies REGENIE allele flipping
    - returns genotype matrix and flip mask

compute_binary_score_for_trait_batch(...)
    - computes score/variance/beta/SE/p for all traits
    - does not redo genotype flipping
```

Firth correction can still operate per trait or per candidate batch, but score-only binary should be genuinely batched.

---

## 6. JAX runtime configuration still leaks into compute imports

Both main compute modules mutate JAX config at import time:

* `src/g/compute/regenie2_linear.py:14`
* `src/g/compute/regenie2_binary.py:16`

```python
jax.config.update("jax_enable_x64", val=True)
```

Linear compute also reads an environment variable at import time:

* `src/g/compute/regenie2_linear.py:16-21`

```python
LINEAR_COMPUTE_DTYPE_ENVIRONMENT_VARIABLE = "GWAS_ENGINE_LINEAR_COMPUTE_DTYPE"
LINEAR_COMPUTE_DTYPE = ...
```

This is not a clean compute architecture. Runtime policy should be owned by `g.jax_setup` / runner initialization, not by importing a kernel module.

### Better design

Use explicit static configs:

```python
@dataclass(frozen=True)
class LinearKernelConfig:
    compute_dtype: Literal["float32", "float64"]
    high_frequency_shift: bool
    pvalue_method: Literal["normal_logsf"]

@dataclass(frozen=True)
class BinaryKernelConfig:
    ...
```

Then make config part of the execution plan hash and JAX cache key.

The compute module should not read environment variables and should not mutate global JAX config at import.

---

## 7. Hidden constants are still spread through `regenie2_binary.py`

There are many module-level constants in `src/g/compute/regenie2_binary.py:18-40`:

```text
MINIMUM_PROBABILITY
MINIMUM_VARIANCE
RELATIVE_VARIANCE_TOLERANCE
DEFAULT_MAXIMUM_NULL_ITERATIONS
NULL_LOGISTIC_COEFFICIENT_TOLERANCE
FIRTH_NULL_MAXIMUM_ITERATIONS
FIRTH_NULL_GRADIENT_TOLERANCE
FIRTH_NULL_MAXIMUM_STEP_SIZE
FIRTH_MAXIMUM_ITERATIONS
FIRTH_PSEUDO_MAXIMUM_ITERATIONS
...
```

Some are true numerical constants. Others are compute-affecting policy. Those should not be invisible module state.

This is connected to the earlier issue I flagged: `fit_covariate_only_firth_null_model()` accepts `kernel_config` and then discards it around `src/g/compute/regenie2_binary.py:2099-2107`, while using module-level Firth-null constants later.

### Recommendation

Separate constants into two categories:

```text
Mathematical constants:
    allele count multiplier
    case threshold
    machine epsilon relationship

Kernel policy:
    iteration limits
    tolerances
    step sizes
    candidate capacity
    Firth batch size
    clipping thresholds
```

Kernel policy should be explicit in `BinaryKernelConfig` or in a nested config:

```python
@dataclass(frozen=True)
class LogisticNullConfig:
    maximum_iterations: int
    coefficient_tolerance: float
    minimum_variance: float

@dataclass(frozen=True)
class FirthNullConfig:
    maximum_iterations: int
    gradient_tolerance: float
    maximum_step_size: float
    fallback_iteration_multiplier: int
    fallback_step_divisor: float

@dataclass(frozen=True)
class ApproximateFirthConfig:
    maximum_iterations: int
    pseudo_maximum_iterations: int
    gradient_tolerance: float
    coefficient_tolerance: float
    likelihood_tolerance: float
    maximum_step_size: float
    line_search_maximum_attempts: int
```

Then:

```python
@dataclass(frozen=True)
class BinaryKernelConfig:
    logistic_null: LogisticNullConfig
    firth_null: FirthNullConfig
    approximate_firth: ApproximateFirthConfig
    candidate: FirthCandidateConfig
```

This would improve reproducibility and manifest correctness.

---

## 8. JIT boundaries are too scattered

Right now, many functions are individually jitted:

* `src/g/compute/regenie2_linear.py:189`
* `src/g/compute/regenie2_linear.py:222`
* `src/g/compute/regenie2_linear.py:252`
* `src/g/compute/regenie2_linear.py:318`
* `src/g/compute/regenie2_linear.py:375`
* `src/g/compute/regenie2_linear.py:442`
* `src/g/compute/regenie2_binary.py:591`
* `src/g/compute/regenie2_binary.py:639`
* `src/g/compute/regenie2_binary.py:718`
* `src/g/compute/regenie2_binary.py:760`
* `src/g/compute/regenie2_binary.py:869`
* `src/g/compute/regenie2_binary.py:893`
* `src/g/compute/regenie2_binary.py:2703`
* `src/g/compute/regenie2_binary.py:2995`
* `src/g/compute/regenie2_binary_variant_major.py:15`
* `src/g/compute/regenie2_binary_variant_major.py:85`
* `src/g/compute/regenie2_binary_variant_major.py:399`

That makes cache behavior harder to reason about.

I would prefer this rule:

```text
Internal math helpers are not jitted.
Only public executable kernels are jitted.
```

For example:

```python
@jax.jit
def prepare_chromosome_state(...):
    ...

@jax.jit
def compute_chunk(...):
    ...
```

But helper functions such as:

```python
compute_positive_variance_mask
build_regenie_flipped_genotypes
compute_information_components
compute_firth_convergence_mask
```

should usually be plain functions called inside jitted public kernels.

This makes compilation boundaries easier to test, name, profile, and warm.

---

## 9. Some static branches are implemented with `jnp.where`, causing unnecessary tracing/work

In Firth candidate correction, there are places like:

* `src/g/compute/regenie2_binary.py:2744-2750`
* `src/g/compute/regenie2_binary_variant_major.py:135-142`

The code uses:

```python
genotype_matrix_by_variant = jnp.where(
    kernel_config.use_block_firth_math,
    firth_raw_genotype_matrix_by_variant,
    residualize_and_scale_genotypes_for_approximate_firth(...),
)
```

But `kernel_config.use_block_firth_math` is static. This should be a Python `if`, not `jnp.where`.

With `jnp.where`, the expensive branch expression is still built/traced. For static algorithm selection, prefer:

```python
if kernel_config.use_block_firth_math:
    genotype_matrix_by_variant = firth_raw_genotype_matrix_by_variant
else:
    genotype_matrix_by_variant = residualize_and_scale_genotypes_for_approximate_firth(...)
```

This is both cleaner and more efficient.

---

## 10. Candidate-capacity fallback traces the large fallback branch too

In sample-major Firth correction:

* `src/g/compute/regenie2_binary.py:2957-2963`

and variant-major Firth correction:

* `src/g/compute/regenie2_binary_variant_major.py:361-367`

the code does:

```python
return jax.lax.cond(
    fallback_count <= bounded_candidate_capacity,
    lambda _: apply_candidate_corrections_with_capacity(bounded_candidate_capacity),
    lambda _: apply_candidate_corrections_with_capacity(variant_count),
    operand=None,
)
```

This solves a dynamic fallback problem, but JAX traces both branches. That means the “full variant count” candidate branch can still affect compile time and memory even when most chunks do not need it.

For production, I would avoid this design.

Cleaner options:

```text
Option A:
  Always use fixed candidate capacity.
  If overflow occurs, mark overflow and handle with a separate slower kernel.

Option B:
  Do a cheap pre-count outside the heavyweight Firth kernel.
  Dispatch to one of two separately compiled kernels.

Option C:
  Choose candidate capacity from bsize/pThresh upfront and make overflow a loud diagnostic.
```

Given the product goal, I prefer Option B:

```text
score-test kernel
  returns candidate count and mask

host/pipeline
  chooses normal candidate kernel or overflow kernel

Firth kernel
  has one fixed candidate capacity per compiled call
```

That gives clearer timing, clearer failure modes, and smaller JAX programs.

---

## 11. Linear variant-major receives genotype sums and discards them

The engine passes Rust-computed genotype sum-square stats into the linear variant-major path:

* `src/g/engine/callbacks.py:682`
* `src/g/engine/callbacks.py:909`

But the compute kernel deletes them:

* `src/g/compute/regenie2_linear.py:379-382`
* `src/g/compute/regenie2_linear.py:446-449`

Then it recomputes genotype sum squares inside JAX:

* `src/g/compute/regenie2_linear.py:386-390`
* `src/g/compute/regenie2_linear.py:453-457`

This is an interface smell.

It may be intentional because the linear kernel shifts high-frequency genotypes before computing sums:

* `src/g/compute/regenie2_linear.py:181-186`

If so, then the argument should not be passed at all. If Rust stats are meant to be used, the Python extension should expose enough statistics to compute the shifted sum of squares cheaply.

Rust has `dosage_sum` and `imputed_dosage_square_sum` in native `ChunkStats`:

* `src/genotype/common.rs:18-20`

but Python currently exposes only some stats in `_core.pyi`:

* `src/g/_core.pyi:10-21`

and I do not see `dosage_sum` exposed there.

### Recommendation

Either remove the unused `genotype_sum_squares` parameter from the compute API or expose the right Rust stats and use them.

For shifted genotypes:

```text
sum((g - offset)^2)
= raw_sum_squares - 2 * offset * raw_sum + n * offset^2
```

So Rust can provide:

```text
dosage_sum
imputed_dosage_square_sum
sample_count
```

and JAX can avoid rescanning `G` just to compute sum squares.

---

# What I would keep

I would keep these ideas:

```text
state → chromosome state → chunk result
JAX dataclass pytrees
trait-major result convention
separate diagnostic counters
device-resident Firth fallback
explicit BinaryKernelConfig
variant-major BGEN path
```

I would not rewrite everything blindly. The statistical algorithms are delicate. I would restructure around them with parity tests.

---

# Recommended target architecture

A cleaner long-term compute architecture would look like this:

```text
src/g/compute/
  __init__.py

  common/
    __init__.py
    linalg.py
    pvalue.py
    genotype.py
    result.py
    policy.py

  regenie2_linear/
    __init__.py
    state.py
    result.py
    score.py
    api.py

  regenie2_binary/
    __init__.py
    config.py
    state.py
    result.py
    null_logistic.py
    score.py
    candidates.py
    variant_major_correction.py
    diagnostics.py
    api.py

    firth/
      __init__.py
      types.py
      null.py
      scalar_approx.py
      full_model.py
      line_search.py
      batch.py
```

The engine should only call:

```python
linear_kernel.prepare_state(...)
linear_kernel.prepare_chromosome_state(...)
linear_kernel.compute_chunk(...)

binary_kernel.prepare_state(...)
binary_kernel.prepare_chromosome_state(...)
binary_kernel.compute_chunk(...)
```

Everything else should be private implementation detail.

---

# Preferred internal data model

I would standardize internal compute shapes:

```text
genotypes:
  variant-major: variants x samples

phenotypes/residuals:
  trait-major: traits x samples

linear output:
  traits x variants

binary output:
  traits x variants
```

Then single-trait mode is just:

```text
traits = 1
```

The public wrapper can squeeze the trait axis for existing engine code if needed.

This is the right foundation for future multi-phenotype batching.

---

# Suggested rewrite sequence

Do not start by rewriting the Firth math. Start by removing architectural risk without changing behavior.

## Progress notes

Completed cleanup so far:

* Extracted common helpers into `src/g/compute/common/`.
* Extracted linear state and score implementations out of the legacy linear facade.
* Extracted binary state, score, null-logistic, Firth, and candidate-correction implementations out of the legacy binary facade.
* Extracted sample-major and variant-major binary candidate correction modules.
* Removed trivial compute pass-through wrappers and constant aliases where callers can use the implementation module directly.
* Restructured flat compute files into `regenie2_linear/` and `regenie2_binary/` packages, with binary Firth kernels under `regenie2_binary/firth/`.
* Updated production imports to use the new package-local modules directly.
* Removed the `g.compute` compatibility exports and updated source, tests, and diagnostic scripts to import package modules directly.
* Extracted the shared REGENIE binary logistic clipping and deviance helpers out of the binary facade and Firth solvers into `regenie2_binary/logistic.py`.
* Centralized binary score-test result constructors in `regenie2_binary/result.py` so empty Firth diagnostics are built in one module.
* Split Firth step-halving and convergence checks from `firth/common.py` into `firth/line_search.py`.
* Renamed the Firth solver modules to match their statistical roles: `firth/scalar_approx.py` and `firth/full_model.py`.
* Removed duplicate Firth-local aliases for shared binary case-threshold, minimum-probability, and allele-count constants.
* Removed the duplicated sample-major Firth candidate-correction implementation; sample-major correction now transposes once and uses the variant-major kernel.
* Made Firth candidate-capacity planning explicit in `regenie2_binary/candidates.py`, separating bounded and overflow capacities before changing dispatch behavior.
* Extracted the fixed-capacity variant-major Firth correction body into a named kernel, leaving the existing entry point responsible only for capacity selection.
* Moved Firth candidate-count and capacity selection to host-side dispatch, so bounded and overflow fixed-capacity kernels are no longer selected through one heavyweight `lax.cond`; multi-trait Firth now loops over trait-specific fixed-capacity kernels while score-only multi-trait remains vectorized.
* Moved multi-trait binary result stacking constructors out of the binary facade into `regenie2_binary/result.py`.
* Moved the multi-to-single binary chromosome-state view helper out of the binary facade into `regenie2_binary/state.py`.
* Moved linear public chunk-compute entry points out of the package facade into `regenie2_linear/api.py`, and updated engine/tests/scripts to import that API module explicitly.
* Moved binary public chunk-compute entry points out of the package facade into `regenie2_binary/api.py`, matching the intended engine-facing API boundary.
* Folded the public binary variant-major chunk entry point into `regenie2_binary/api.py` and removed the now-empty `variant_major.py` boundary, so engine code no longer imports that implementation module directly.
* Unified linear state preparation around trait-major helpers: single-trait state and chromosome-state preparation now adapt through the same trait-major formulas used by multi-trait linear compute.
* Routed single-trait binary score testing through the trait-major score kernel, leaving one score-test formula and one genotype-flipping path for score-only binary statistics.
* Centralized the binary variance floor and relative variance tolerance in `regenie2_binary/config.py` so score, null logistic, state preparation, and Firth solvers share one numerical policy.
* Promoted Firth retry, line-search, pseudo-Firth, and null-Firth iteration limits into `BinaryKernelConfig`, so these compute-affecting policies are part of the execution plan instead of hidden module constants.
* Removed the sample-major binary correction adapter module; sample-major public calls now transpose once at the API boundary and call the canonical variant-major correction path directly.
* Routed sample-major multi-binary chunk execution through the variant-major API after one boundary transpose, avoiding repeated per-trait layout conversion in approximate-Firth paths.
* Centralized sample-major to variant-major dosage conversion in `compute/common/genotype.py` and routed linear and binary public chunk adapters through it.
* Moved the single-to-multi binary chromosome-state view helper from score math into `regenie2_binary/state.py`, keeping state container adapters together.
* Removed stale binary API callable aliases and cast indirection now that public chunk calls directly target the canonical function.
* Removed the redundant no-candidate branch from the fixed-capacity Firth correction kernel; the host capacity selector already returns before launching that kernel when no Firth candidates exist.
* Trimmed unused covariate transpose and Cholesky fields from linear chromosome states, and removed the single-trait stacked score matrix adapter in favor of storing the same whitened covariate matrix used by multi-trait linear compute.
* Removed the stale linear debug-script `genotype_sum_squares` call argument; the production linear compute API no longer receives Rust sum-square stats and intentionally recomputes shifted genotype sums after REGENIE-style high-frequency normalization.
* Collapsed the sample-major binary chunk API into a pure layout adapter that transposes once and calls the canonical variant-major binary chunk path, removing duplicated score-plus-correction orchestration.
* Trimmed unused binary chromosome-state fields for null Firth coefficients, fitted probabilities, and standardized residuals; kernels keep the consumed null offset, score residual, weights, and diagnostic counters.
* Removed stale binary result constructors left behind by trait-major score-test routing; score-only binary results are now built only through the multi-trait constructor and squeezed for single-trait callers.
* Split binary score-test result containers from Firth-corrected result containers, so score-only kernels no longer allocate empty Firth diagnostic arrays; the Firth correction boundary expands score results only when correction is requested.
* Removed the stale device-side Firth candidate overflow-mask helper; candidate overflow selection is now host-dispatched through explicit capacity plans.
* Routed native binary variant-major callbacks directly through variant-major compute APIs for both score-only and approximate-Firth paths, removing variant-major to sample-major to variant-major transpose churn on the hot callback path.
* Moved `BinaryKernelConfig` from binary result/state types into `regenie2_binary/config.py`, so binary kernel policy lives next to default policy constants instead of the pytree container module.
* Split linear pytree containers out of the old catch-all `regenie2_linear/types.py`: state containers now live in `regenie2_linear/state.py`, result containers live in `regenie2_linear/result.py`, and the obsolete types module was removed.
* Split binary pytree containers out of the old catch-all `regenie2_binary/types.py`: state containers now live in `regenie2_binary/state.py`, result containers live in `regenie2_binary/result.py`, and the obsolete types module was removed.
* Promoted the remaining Firth scalar policy values into `BinaryKernelConfig` and `GComputeConfig`: pseudo-response scale, sparse-carrier dosage threshold, full-model step-halving scale, and null-Firth step-halving scale. The binary Firth code now reads these through the execution plan instead of hidden module constants.
* Moved public linear and binary state-preparation entry points into `regenie2_linear/api.py` and `regenie2_binary/api.py`, so production engine callback and warm-cache code now uses the public compute API boundary for both preparation and chunk execution.
* Promoted binary probability/variance floors and relative score-variance tolerance into `BinaryKernelConfig` and `GComputeConfig`; score tests, null logistic state preparation, and Firth solvers now receive this numerical policy from the execution plan instead of reading module globals directly.
* Split `BinaryKernelConfig` into nested policy domains for score numerics, null logistic fitting, Firth candidate batching, approximate Firth, and null Firth. Call sites now read settings through the statistical subsystem that owns them instead of one flat bag of fields.
* Narrowed binary score-only versus Firth-corrected result types in tests, so the split binary result containers can be checked by `ty` without hiding score/Firth result-shape mismatches behind broad unions.
* Removed the stale covariate-only adjusted-weight helper from the full-model Firth module; covariate-only null Firth has its own implementation in `firth/null.py`, so the full-model module now exposes only helpers used by candidate Firth correction.
* Moved engine callback type annotations and binary diagnostic counting behind the public linear and binary compute API modules, so production engine code no longer imports compute result, state, or diagnostics implementation modules directly.
* Removed the remaining `firth/common.py` catch-all module: full-model penalized likelihood now lives in `firth/full_model.py`, and Firth reason-to-failure mapping lives next to the Firth reason enum in `firth/types.py`.

Intentional remaining adapters:

* Keep public chunk adapters that transpose sample-major input, unpack chromosome-state fields, or preserve the current JIT boundary.
* Keep dataclass builder functions where they construct result/state containers rather than only forwarding a call.
* Keep multi-trait approximate-Firth correction as per-trait fixed-capacity calls until a true trait-major Firth kernel is designed and parity-tested; score-only multi-trait is already vectorized.

## Phase 1: Extract common helpers

Move these out of linear/binary-specific files:

```text
solve_positive_definite_system
solve_from_positive_definite_matrix
chi_squared_to_log10_p_value
build_regenie_flipped_genotypes
genotype layout conversion
common result constructors
```

Create:

```text
compute/common/linalg.py
compute/common/pvalue.py
compute/common/genotype.py
compute/common/result.py
```

This breaks the binary/linear coupling and removes the circular import pressure.

## Phase 2: Make linear trait-major internally

Rewrite linear compute so there is one real implementation:

```python
compute_linear_chunk_trait_major(...)
```

Then single-trait functions become wrappers.

This should be low risk and high value.

## Phase 3: Make binary score trait-major internally

Before touching Firth, rewrite score-only binary as a real batched score kernel.

Important rule:

```text
genotype flipping happens once per chunk, not once per trait
```

Then verify:

```text
single binary score == trait 0 of multi binary score
sample-major score == variant-major score
```

## Phase 4: Split Firth modules

Move Firth internals into:

```text
firth/null.py
firth/scalar_approx.py
firth/full_model.py
firth/line_search.py
firth/batch.py
```

Keep the math identical at first. This is a file/module rewrite, not an algorithm rewrite.

## Phase 5: Fix candidate overflow architecture

Separate:

```text
score-test candidate discovery
candidate-count decision
Firth correction kernel
overflow fallback kernel
```

Do not compile normal and full-overflow candidate branches inside the same giant JAX function.

## Phase 6: Remove import-time runtime policy

Move all JAX config and dtype policy out of compute imports.

Compute should receive explicit kernel configs from the runner/execution plan.

Progress note:

* JAX x64 enablement is now represented in `GComputeConfig`, the runner's process-global runtime policy,
  and output run manifests. The default remains enabled because current parity-sensitive kernels still rely on it,
  but the setting is no longer hidden in `jax_setup.py`.

---

# Priority ranking

If I were reviewing this for pre-release readiness, I would rank compute-module architecture work like this:

```text
Release-blocking if adding more methods:
  1. Split binary Firth out of regenie2_binary.py
  2. Remove circular binary ↔ variant-major imports
  3. Centralize p-value/linalg helpers
  4. Remove import-time JAX config mutation

High priority:
  5. Make linear internally trait-major
  6. Make binary score-only internally trait-major
  7. Canonicalize genotype layout once per chunk
  8. Make all compute-affecting constants explicit config/policy

Medium priority:
  9. Use Rust-provided genotype stats or remove unused parameters
  10. Reduce score-only binary result memory by not carrying Firth diagnostic arrays unnecessarily (done)
  11. Clean up unused fields in state dataclasses
```

---

# Bottom line

The compute module is **not optimal**. It is a reasonable transitional implementation, but not the architecture I would want before external users or before adding SPA/exact Firth.

The best rewrite is not a total algorithm rewrite. It is a **module-boundary rewrite**:

```text
common math helpers
trait-major internal kernels
variant-major canonical genotype layout
real multi-trait binary score kernel
separate Firth package
explicit kernel policy/config
public API facade used by engine
```

I would do this now while the app is still pre-release. The longer you keep adding Firth/SPA/multi-phenotype features on top of the current `regenie2_binary.py`, the harder it will be to prove statistical equivalence and performance later.
