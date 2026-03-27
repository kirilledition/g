# 3D Batching Strategy for GWAS Regression Kernels

## 1. Mission

This document describes how to think about "3D batching" for the current Phase 1 regression kernels, what is already batched in the repository, what remains effectively unbatched, and where the most likely speedups are.

The goal is not to change the statistical model. The goal is to express more of the existing math as large, shape-stable, device-resident array programs so JAX and the backend linear algebra libraries can do more work per launch and less work per Python branch.

One important qualification: the kernels are more device-resident than the full pipeline, but chunk ingestion is still host-driven. `src/g/io/plink.py:362` reads BED chunks on host memory, the chunk is then transferred to device, and preprocessing still includes a host-visible missingness check in `src/g/io/plink.py:164`. In practice, this puts an upper bound on end-to-end gains from kernel-only batching, especially when chunk sizes are small or the device is fast relative to I/O.

This document is intended as an implementation brief for engineering work. Any change that alters the regression formulation rather than the execution strategy still requires explicit approval.

## 2. What "3D batching" means here

In this repository, there are three different batching ideas that can easily get conflated:

1. **Chunk batching across variants using a 2D genotype matrix**
   - Shape is typically `(sample_count, variant_count)`.
   - One matrix multiply computes statistics for many variants at once.
   - This is already heavily used by the linear kernel and by parts of the logistic kernel.

2. **True batched small-matrix algebra using a leading batch axis**
   - Shape is typically `(variant_count, parameter_count, parameter_count)` or `(variant_count, sample_count, parameter_count)`.
   - Each variant has its own Fisher information matrix, score vector, or masked design matrix.
   - This is the real "3D batching" pattern.

3. **Vectorized repeated solves or updates over many variants**
   - Often expressed through `jax.vmap`, `jnp.einsum`, batched `jnp.linalg.solve`, or explicit leading-batch dimensions.
   - This is the practical form of 3D batching in JAX.

For ordinary least squares, the first pattern often captures most of the available gain because the problem has a closed form and the covariate matrix is shared across variants.

For logistic regression, the second and third patterns matter much more because each IRLS iteration creates variant-specific weights, scores, and information matrices.

## 3. Why batching helps at all

The likely benefits do not come from tensors being inherently faster. They come from:

* removing Python loops over variants
* replacing many tiny launches with fewer larger kernels
* keeping intermediate arrays on device across multiple steps
* reusing chunk-invariant quantities such as the covariate matrix and covariate-only terms
* presenting regular shapes to XLA so the compiler can cache and optimize effectively

The likely non-benefits are also important:

* 3D batching does not eliminate the sequential dependence between IRLS iterations
* 3D batching does not make memory bandwidth disappear
* 3D batching can increase memory pressure if it materializes arrays that were previously implicit
* 3D batching is not automatically better than the current 2D formulation when the current formulation already maps well to GEMM

## 4. Current state: linear regression

### 4.1 What is already batched

The linear kernel is already strongly batched across variants.

Relevant code:

* `src/g/compute/linear.py:29`
* `src/g/compute/linear.py:31`
* `src/g/compute/linear.py:63`
* `src/g/compute/linear.py:72`

Current structure:

* `prepare_linear_association_state(...)` computes chunk-invariant covariate-only quantities once.
* `compute_linear_association_chunk(...)` accepts a full genotype chunk rather than one variant at a time.
* `covariate_matrix.T @ genotype_matrix` computes covariate/genotype crossproducts for all variants in the chunk in one operation.
* `genotype_matrix.T @ phenotype_residual` computes covariance with the residualized phenotype for all variants in one operation.
* The implementation avoids materializing the full residualized genotype matrix and instead works with sufficient statistics.

This means the linear path is already using the most important batching pattern: one shared design matrix against many variant columns.

### 4.2 What is not yet a 3D batched problem

The linear kernel does **not** currently create a per-variant batched design tensor. That is mostly correct.

For the implemented Phase 1 linear path:

* the covariate design is shared across variants
* genotype missingness is handled upstream by mean imputation in `src/g/io/plink.py:143`
* the genotype enters as one extra column per variant
* the needed outputs can be derived from crossproducts and residual sums of squares

Constructing a full tensor like `(variant_count, sample_count, parameter_count)` for linear regression would likely create more memory traffic than benefit.

Another guardrail: the current linear path is a sufficient-statistic computation rather than an explicit per-variant residualized-design materialization. It also preserves the present degrees-of-freedom calculation and clamps small negative residual-sum-of-squares terms caused by roundoff. Any batched reformulation must preserve these numerical conventions, particularly for nearly collinear or near-monomorphic variants.

### 4.3 Speedup expectation for linear

Expected upside from additional 3D reformulation is modest.

Most plausible wins are:

* preserving device residency through the whole chunk pipeline
* keeping shape stability across chunk sizes, including the final short chunk emitted from `src/g/io/plink.py:362`
* better chunk-size tuning for CPU versus GPU

Least plausible wins are:

* explicitly building per-variant design tensors
* replacing the current crossproduct-based formulation with a more literal batched normal-equation solve

Conclusion for linear:

* the main batching win is already present
* future speedups are more likely from orchestration and memory behavior than from deeper tensorization

## 5. Current state: standard logistic regression

### 5.1 Why logistic is a better fit for 3D batching

Unlike OLS, logistic regression is iterative.

Each IRLS iteration needs, per variant:

* probabilities over samples
* residuals over samples
* effective weights over samples
* a Fisher information matrix over parameters
* one or more linear solves for Newton or Schur-complement updates

That naturally produces batched arrays such as:

* probability matrix: `(variant_count, sample_count)`
* score matrix: `(variant_count, parameter_count)`
* Fisher information tensor: `(variant_count, parameter_count, parameter_count)`
* coefficient matrix: `(variant_count, parameter_count)`

This is the core reason logistic is a more natural target for 3D batching than linear.

### 5.2 What is already batched

Large parts of the standard logistic path already follow the 3D batching approach.

Relevant code:

* `src/g/compute/logistic.py:220`
* `src/g/compute/logistic.py:228`
* `src/g/compute/logistic.py:258`
* `src/g/compute/logistic.py:271`
* `src/g/compute/logistic.py:610`
* `src/g/compute/logistic.py:778`

Already-batched pieces:

* probability evaluation is vectorized across variants and samples
* score construction is vectorized across variants
* Fisher information assembly is vectorized across variants
* the no-missing path precomputes chunk-invariant constants once in `prepare_no_missing_logistic_constants(...)`
* the standard solver runs inside `jax.lax.while_loop`, which keeps the iterative core inside compiled JAX rather than in Python

There is also an important optimization already present in the masked standard logistic path:

* `src/g/compute/logistic.py:672` stacks two right-hand sides and solves them together
* `src/g/compute/logistic.py:678` uses `jax.vmap` with Cholesky solves over batched Fisher matrices

That is a direct example of the intended batching philosophy: if multiple linear systems share the same coefficient matrix, solve them together.

### 5.3 What remains inconsistent or only partially batched

The standard logistic implementation is not yet fully consistent across code paths.

Important gap:

* the no-missing standard path still uses two separate batched solves in the loop body at `src/g/compute/logistic.py:829` and `src/g/compute/logistic.py:831`
* the masked standard path already combines these solves using a stacked right-hand side and batched Cholesky solve

This means one immediate implementation target is obvious:

* unify the no-missing standard path with the same combined-solve strategy already used in the masked path

Another gap:

* final standard-error computation still performs a separate batched solve after the loop at `src/g/compute/logistic.py:883`

This may still be acceptable, but it should be revisited as part of a broader review of repeated solve patterns.

Another important practical constraint is memory footprint. Standard logistic already materializes several dense `(variant_count, sample_count)` intermediates during IRLS, including probabilities, residuals, weights, and weighted genotype matrices. Any additional tensorization should be judged against that existing working set rather than as a standalone abstraction improvement.

Critical semantic guardrail: in the masked logistic path, missing genotypes are not analytically mean-imputed. `src/g/engine.py:340` constructs `observation_mask`, and `src/g/compute/logistic.py` uses that mask to zero out residual and weight contributions from missing rows. The dense genotype matrix passed through the pipeline may contain imputed or sanitized values for storage convenience, but those values must never enter the masked logistic likelihood, score, information matrix, or fallback decision except through explicit masking. Any batching refactor that removes or weakens that masking changes the math.

### 5.4 Speedup expectation for standard logistic

Expected upside from further 3D batching is real but bounded.

Most plausible wins are:

* eliminating duplicated solve work in the no-missing path
* consolidating batched small-matrix solves wherever the same Fisher matrix is reused
* reducing host synchronization around chunk orchestration so standard logistic stays device-resident longer
* preserving shape stability so XLA compilation is reused across chunks

Less plausible wins are:

* trying to collapse the entire IRLS procedure into one giant matrix multiply
* materializing full per-variant design tensors when crossproducts are sufficient

Conclusion for standard logistic:

* the solver core is already substantially batched
* the remaining gains are in solve reuse, path unification, and device residency

## 6. Current state: Firth fallback and hybrid logistic orchestration

This is the part of the codebase where the "what remains to be batched" question matters most.

### 6.1 What is already batched

Relevant code:

* `src/g/compute/logistic.py:1374`
* `src/g/compute/logistic.py:1236`
* `src/g/compute/logistic.py:1279`

Already-batched pieces:

* single-variant Firth is vectorized into chunk-level computation through `jax.vmap`
* the no-missing Firth update path builds fixed-size batch plans on device
* the no-missing path merges Firth results back into the standard result on device
* fixed-size padding is used to reduce recompilation pressure from variable fallback batch sizes

This means the repository already contains the basic machinery needed for device-side batched Firth execution.

However, this path should be described as **partially** device-batched rather than fully device-side. It still performs a host-visible fallback decision and still loops over fallback batches in Python.

### 6.2 What remains effectively unbatched or host-driven

The masked logistic fallback path is still the clearest batching gap.

Relevant code:

* `src/g/compute/logistic.py:1483`
* `src/g/compute/logistic.py:1488`
* `src/g/compute/logistic.py:1494`
* `src/g/compute/logistic.py:1495`

Current issues:

* it transfers the fallback decision to host with `jax.device_get(...)`
* it transfers fallback masks to host
* it builds fallback index batches on host Python lists
* it loops over fallback batches in Python
* it performs host-dependent branching when choosing standard versus heuristic initial coefficients
* it merges active Firth results by repeatedly patching arrays batch by batch

This is the least batched and least GPU-friendly section of the current regression pipeline.

For accuracy, the no-missing fallback path is better than the masked path but still not fully solved. It retains a host-visible `jnp.any(...)` decision point and a Python loop over fixed-size fallback batches. The masked path simply has more of these problems at once.

### 6.3 Why this section is high value

This section combines several costs at once:

* host synchronization
* Python control flow
* irregular masking
* repeated gather/scatter operations
* iterative Firth work on a subset of variants

Improving only one of these costs may not be enough, but reducing several together can materially change end-to-end throughput on GPU.

### 6.4 Most plausible batching direction for Firth fallback

The existing no-missing implementation provides the best template.

Recommended direction:

1. keep the fallback mask on device
2. build padded fallback batch plans on device for the masked path as well
3. move heuristic-versus-standard initial coefficient selection onto device
4. run Firth batches through a fixed-shape compiled path
5. merge batch results on device using one reusable merge primitive

A refactor should not be considered complete unless the hot fallback path satisfies all of the following:

* no `jax.device_get(...)` on fallback masks inside chunk execution
* no Python list construction from fallback indices inside chunk execution
* no Python branch that selects heuristic-versus-standard initialization from host booleans
* no Python loop whose trip count depends on the number of fallback variants

The key idea is not "run Firth for every variant." The key idea is "run Firth only where needed, but keep the subset selection and subset execution in a batched device program."

## 7. High-probability speedups by area

This section ranks opportunities by expected payoff and implementation risk.

### 7.1 Highest-probability wins

#### A. Remove host synchronization from masked logistic fallback

Why it matters:

* this is the clearest remaining CPU-style orchestration in the hot path
* GPU performance is especially sensitive to host round-trips and Python control flow
* the no-missing path already shows the architectural direction

Expected win type:

* better GPU utilization
* lower end-to-end latency for chunks that trigger fallback
* fewer pipeline stalls

#### B. Unify repeated solves inside standard logistic

Why it matters:

* the masked path already demonstrated that stacked right-hand sides and Cholesky-based batched solves reduce work
* the no-missing path still leaves solve reuse on the table

Expected win type:

* lower IRLS iteration cost
* more consistent performance between masked and no-missing paths

#### C. Enforce shape-stable batched fallback execution

Why it matters:

* JAX performance depends heavily on reusing compiled programs
* varying fallback counts are a natural source of recompilation and control-flow noise
* the final genotype chunk can be smaller than the nominal `chunk_size`, which is a separate source of shape variation

Expected win type:

* fewer recompiles
* steadier runtime behavior across chunks

### 7.2 Medium-probability wins

#### D. Revisit final variance and standard-error solve patterns

Why it matters:

* repeated small solves after convergence can still be meaningful in aggregate
* the same factorization may be reusable within one variant update path

Expected win type:

* modest reduction in post-IRLS overhead

#### E. Fuse or simplify gather/scatter-heavy merge steps

Why it matters:

* repeated `.at[indices].set(...)` updates can become costly
* the merge logic currently touches many result arrays separately

Expected win type:

* lower result assembly overhead
* cleaner device-only fallback merge path

### 7.3 Lower-probability or lower-priority wins

#### F. Build explicit per-variant design tensors for standard logistic

Why it is lower probability:

* much of the needed structure is already expressible through crossproducts and `einsum`
* explicit design tensors may increase memory traffic and memory footprint
* this is more likely to help a custom-kernel path than the present optimized-JAX path

#### G. Move linear regression to explicit 3D tensors

Why it is lower probability:

* the linear kernel already captures the main variant batching benefit
* the current bottleneck is more likely orchestration and bandwidth than lack of tensor rank

## 8. Recommended implementation order

1. **Standard logistic solve cleanup**
   - port the masked path's stacked-right-hand-side solve strategy into the no-missing path
   - measure per-iteration time before and after

2. **Masked fallback device-plan refactor**
   - remove host-side fallback mask transfer
   - replace host batch construction with fixed-size device batch planning
   - keep merge behavior on device

3. **Common batching utilities**
   - factor repeated batch-plan and merge logic into shared helpers used by both masked and no-missing paths
   - avoid maintaining two batching strategies that drift apart

4. **Compilation-shape review**
   - confirm that chunk size, fallback batch size, and mask shapes do not trigger unnecessary recompilation

5. **Post-refactor measurement**
   - benchmark at minimum these regimes separately:
     - linear, no missing
     - logistic, no missing, zero fallback
     - logistic, masked, zero fallback
     - logistic, no missing, light fallback
     - logistic, masked, light fallback
     - logistic, masked, heavy fallback
   - for each regime, report:
     - first-call compile time
     - steady-state chunk time
     - total wall time including BED read and host/device transfer
     - peak device memory
     - fallback fraction and missingness fraction
     - sample count, covariate count, chunk size, and dtype
   - success criteria should be evaluated separately for CPU and GPU, because host I/O and transfer costs can dominate before kernel math does

## 9. Guardrails for implementation

These constraints are important because an apparently faster tensorized formulation can still be the wrong change.

### 9.1 Do not change the math without approval

Acceptable work:

* restructuring loops into batched array programs
* reusing factorizations or solving multiple right-hand sides together
* changing where masks, merges, and fallback plans are executed

Not acceptable without approval:

* changing the statistical objective
* changing the Firth penalty definition
* changing convergence rules to make batching easier

### 9.2 Watch memory pressure explicitly

Any batching change should record:

* peak device memory use
* whether new intermediate tensors are materialized
* whether the change increases memory traffic enough to erase compute gains

This matters especially for masked logistic, where naïvely constructing large masked design tensors can become expensive.

As a concrete budgeting heuristic, engineers should record the dominant dense working-set terms for each IRLS step as a function of `variant_count * sample_count * dtype_size`, because the current implementation already carries multiple arrays of that scale at once.

Memory budgeting must include both per-chunk working memory and run-level retained outputs. `src/g/engine.py` currently keeps chunk result arrays on device until final concatenation, so peak memory is not just the IRLS or Firth workspace for one chunk. Any batching change should report: (a) per-chunk transient workspace, (b) per-chunk retained output size, and (c) peak device memory for a full multi-chunk run.

### 9.3 Preserve parity-oriented edge-case behavior

The batching refactor must preserve behavior for:

* variants with complete separation
* variants with partial missingness
* variants that fail standard logistic and require Firth fallback
* variants with zero or near-zero information on the genotype term
* chunks with no fallback variants at all

### 9.3A Preserve per-variant IRLS state semantics

Batched execution must preserve the current per-variant solver semantics:

* convergence is decided per variant, not per chunk
* converged variants stop updating coefficients while unfinished variants continue
* iteration counts remain per variant
* the last stored information terms for each converged variant remain the source for final standard-error computation

Optimizations such as active-set compaction, variant reordering, or dropping converged rows from later iterations are acceptable only if they preserve these semantics exactly.

### 9.4 Measure end-to-end and microbenchmarks separately

A batching change can improve kernel time while hurting total runtime through larger transfers or memory pressure.

At minimum, measure:

* pure compute time for the standard logistic kernel
* pure compute time for fallback-heavy paths
* total chunk wall time
* compile time versus steady-state time

## 10. Blind spots to avoid during implementation

This section is intentionally explicit because these are the places most likely to fool an optimization effort.

### 10.1 Confusing "more batched" with "faster"

More tensor rank is not automatically better.

The correct comparison is not 2D versus 3D in the abstract. The correct comparison is:

* fewer launches?
* less Python?
* fewer transfers?
* acceptable memory cost?
* stable compilation shapes?

### 10.2 Ignoring the asymmetry between masked and no-missing paths

The masked path is not just a small variation of the no-missing path. Per-variant observation masks create real irregularity.

Any shared batching abstraction must account for that difference rather than assuming one path can be copied mechanically onto the other.

### 10.3 Over-optimizing the fallback subset and under-optimizing the common path

Standard logistic runs on every chunk. Firth fallback runs on a subset.

A large engineering effort on fallback batching is only justified if measurements show it dominates end-to-end runtime for the target workloads.

### 10.4 Ignoring compilation behavior

A faster kernel body can still lose overall if the new implementation causes recompilation due to shape variation.

This is one reason fixed-size padded batch plans are attractive.

It is also why the final short chunk deserves explicit attention during profiling. If it triggers separate compilations or noticeably worse steady-state behavior, padding or chunk normalization may be worth more than deeper tensorization.

### 10.5 Forgetting that result assembly is part of the pipeline

Even if compute is fully batched, repeated gathers, scatters, and host formatting can still dominate the total cost.

The correct optimization target is the full chunk pipeline, not just the Newton update equations.

## 11. Bottom line

The repository already uses the core batching ideas in the places where they matter most:

* linear regression is already well batched across variants
* standard logistic is already substantially expressed as batched array algebra
* no-missing Firth fallback already contains a partially device-batched execution pattern

The strongest remaining opportunity is not a wholesale rewrite into "3D tensors everywhere."

The strongest remaining opportunity is to extend the existing batching strategy consistently into the parts of the logistic pipeline that are still host-driven, especially masked Firth fallback and duplicated batched solves.

If implemented well, the most likely wins are:

* fewer host/device synchronizations
* lower Python orchestration overhead
* lower repeated solve cost inside logistic IRLS
* more stable compiled execution across chunks

If implemented poorly, the likely failure mode is:

* larger tensors
* more memory traffic
* more complicated code
* little or no wall-clock improvement

The engineering target should therefore be disciplined batching, not maximum tensor rank.
