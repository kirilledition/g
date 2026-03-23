# Source Code Review — `src/`

Comprehensive review of the GWAS engine source covering bugs, pitfalls, suboptimal patterns, and design concerns.

---

## 🔴 Potential Bugs

### 1. p-values are never computed in the linear path — silent placeholder

[linear.py:77-82](file:///home/kirill/Projects/g/src/g/compute/linear.py#L77-L82)

[compute_linear_association_chunk](file:///home/kirill/Projects/g/src/g/compute/linear.py#41-85) returns `placeholder_p_values` (all `NaN`) and labels the field `p_value`. The real p-value is only computed **later** inside [build_linear_output_frame](file:///home/kirill/Projects/g/src/g/engine.py#53-97) via `betainc`. This means:

- Anyone calling [compute_linear_association_chunk](file:///home/kirill/Projects/g/src/g/compute/linear.py#41-85) directly (e.g. a test or future API consumer) will silently get `NaN` p-values and *believe* the field is populated.
- The `LinearAssociationChunkResult.p_value` field is a lie — it's never the actual p-value at the source.

> [!CAUTION]
> If any downstream code trusts the `p_value` field from [LinearAssociationChunkResult](file:///home/kirill/Projects/g/src/g/models.py#58-66), it will use `NaN` values. The p-value computation should be moved into the kernel, or the placeholder field should be removed from the result type entirely.

### 2. Linear p-value recomputed on host from device-gotten values, then pushed back to device

[engine.py:73-80](file:///home/kirill/Projects/g/src/g/engine.py#L73-L80)

After `jax.device_get`, the absolute test statistic is put *back* onto a JAX array (`jnp.asarray`), `betainc` runs on device, and then the result is `device_get`'d again. This is a needless device→host→device→host round-trip. Either compute p-values inside the JIT kernel, or compute them entirely on host with `scipy.special.betainc`.

### 3. `firth_tolerance` is always clamped to `1e-12`

[logistic.py:906](file:///home/kirill/Projects/g/src/g/compute/logistic.py#L906)

```python
firth_tolerance = min(tolerance, FIRTH_TOLERANCE_FLOOR)  # FIRTH_TOLERANCE_FLOOR = 1e-12
```

Since the default tolerance is `1e-8`, `min(1e-8, 1e-12)` is always `1e-12`. This means **user-supplied `--tolerance` is completely ignored for Firth fallback**. It appears the intent was `max(tolerance, FIRTH_TOLERANCE_FLOOR)` (i.e. "at least as tight as the floor"). With `min`, the Firth solver always uses an extremely tight tolerance, which wastes iterations and may prevent convergence within `max_iterations`.

> [!WARNING]
> This is almost certainly a `min`/`max` swap bug. The effect is that Firth regressions converge much more slowly than intended.

### 4. [jax_setup.py](file:///home/kirill/Projects/g/src/g/jax_setup.py) has side effects at import time

[jax_setup.py:26-32](file:///home/kirill/Projects/g/src/g/jax_setup.py#L26-L32)

Module-level JAX config mutations (`jax.config.update(...)`) execute on import, which means:

- `from g import jax_setup  # noqa: F401` scattered across modules is fragile — import order can cause subtle issues.
- It is impossible to test with different JAX configs in the same process.
- The cache directory is created (`mkdir`) at import time, not at run time.

### 5. `covariate_only_coefficients` is computed but never used

[engine.py:253-256](file:///home/kirill/Projects/g/src/g/engine.py#L253-L256) produces `covariate_only_coefficients` (a zeros vector). It is passed through the call chain into [compute_logistic_association_chunk_with_mask](file:///home/kirill/Projects/g/src/g/compute/logistic.py#863-978), where it's immediately deleted:

```python
del covariate_only_coefficients  # logistic.py:874
```

This is dead code that adds parameter clutter and cognitive overhead.

### 6. Missing type annotations on [build_linear_output_frame](file:///home/kirill/Projects/g/src/g/engine.py#53-97) and [build_logistic_output_frame](file:///home/kirill/Projects/g/src/g/engine.py#99-141)

[engine.py:53-60](file:///home/kirill/Projects/g/src/g/engine.py#L53-L60), [engine.py:99-104](file:///home/kirill/Projects/g/src/g/engine.py#L99-L104)

Parameters `metadata`, `allele_one_frequency`, `observation_count`, and `linear_result`/`logistic_result` are all untyped `Any`, violating the project's "100% type annotation coverage" rule.

### 7. [convert_frame_to_float64_jax](file:///home/kirill/Projects/g/src/g/io/tabular.py#87-98) has no return type annotation

[tabular.py:87](file:///home/kirill/Projects/g/src/g/io/tabular.py#L87) — returns bare implicit `Any`.

### 8. [compute_hdiag_and_adjusted_weights](file:///home/kirill/Projects/g/src/g/compute/logistic.py#627-646) returns a bare tuple

[logistic.py:627-630](file:///home/kirill/Projects/g/src/g/compute/logistic.py#L627-L630) — returns `tuple[jax.Array, jax.Array, jax.Array]`, violating the style guide rule against bare tuples for multiple return values.

---

## 🟡 Design Concerns

### 9. `NamedTuple` models are immutable but used as mutable scratch pads

In [logistic.py:893-901](file:///home/kirill/Projects/g/src/g/compute/logistic.py#L893-L901), the host-transferred result fields are `.copy()`'d into mutable numpy arrays, then mutated in-place via fancy indexing. This works because `device_get` returns numpy arrays, but:

- It's semantically confusing — `NamedTuple` types suggest immutability.
- Future refactors (e.g. returning JAX arrays or dataclasses) could silently break the mutation.

### 10. Monolithic [logistic.py](file:///home/kirill/Projects/g/src/g/compute/logistic.py) — 971 lines, no sub-decomposition

This file is the single most complex module, containing:
- 7 `NamedTuple` state types
- Standard logistic IRLS
- Firth regression (single-variant + vmapped chunk)
- Firth batching and padding
- Heuristic pre-dispatch masking
- The main orchestration [compute_logistic_association_chunk_with_mask](file:///home/kirill/Projects/g/src/g/compute/logistic.py#863-978)

This makes it very hard to test individual components in isolation.

### 11. [run_linear_association](file:///home/kirill/Projects/g/src/g/engine.py#281-303) / [run_logistic_association](file:///home/kirill/Projects/g/src/g/engine.py#305-331) materialize everything in memory

[engine.py:291-302](file:///home/kirill/Projects/g/src/g/engine.py#L291-L302)

These functions `list(iter_*_output_frames(...))` and then `pl.concat(...)`. For a full genome run this could be hundreds of millions of variants worth of DataFrames held in memory simultaneously. The streaming [write_frame_iterator_to_tsv](file:///home/kirill/Projects/g/src/g/engine.py#333-342) path exists in the CLI but the `run_*` API functions defeat it.

### 12. `genotype_matrix.T` computed repeatedly

Throughout [logistic.py](file:///home/kirill/Projects/g/src/g/compute/logistic.py), `genotype_matrix.T` is computed and passed around as `genotype_matrix_by_variant`. The same transpose happens multiple times across function boundaries. Consider storing both shapes in [GenotypeChunk](file:///home/kirill/Projects/g/src/g/models.py#48-56) to avoid repeated transposes.

### 13. The Rust `_core` module is a dead placeholder

[lib.rs](file:///home/kirill/Projects/g/src/lib.rs) contains only [hello_from_bin()](file:///home/kirill/Projects/g/src/lib.rs#10-14). The entire Maturin build pipeline (including PyO3, `abi3-py39`, cdylib) is configured for a function that isn't used anywhere. This adds build complexity and compile time for zero value.

### 14. `open_bed` context manager holds the file open for the entire iteration

[plink.py:124-162](file:///home/kirill/Projects/g/src/g/io/plink.py#L124-L162)

The `with open_bed(...)` context wraps the entire generator. If the consumer pauses or abandons iteration, the BED file remains open. For lazy generators this is a resource-leak hazard.

### 15. No error handling or logging anywhere

There is zero usage of `logging`, no progress reporting, and no try/except guarding around numerical operations. If JAX hits a singular matrix in `jnp.linalg.inv` or `jnp.linalg.solve`, the failure mode is silent NaN propagation rather than a clear diagnostic.

### 16. The CLI has no `--version` flag and no `--verbose`/`--quiet`

Minor UX gap, but for a scientific tool, progress reporting and version stamping are important for reproducibility.

---

## 🟠 Performance Pitfalls

### 17. [prepare_linear_association_state](file:///home/kirill/Projects/g/src/g/compute/linear.py#14-39) is not JIT-compiled

[linear.py:14-38](file:///home/kirill/Projects/g/src/g/compute/linear.py#L14-L38)

While [compute_linear_association_chunk](file:///home/kirill/Projects/g/src/g/compute/linear.py#41-85) is `@jax.jit`, the state preparation (including `covariate_matrix.T @ covariate_matrix` and `jnp.linalg.inv`) runs eagerly. For large covariate matrices this misses JIT fusion opportunities.

### 18. `betainc` p-value path does an unnecessary device round-trip

Already noted above in bug #2—this is also a performance issue on GPU deployments.

### 19. Firth fallback runs a Python [for](file:///home/kirill/Projects/g/src/g/compute/logistic.py#81-88) loop over batches

[logistic.py:911-958](file:///home/kirill/Projects/g/src/g/compute/logistic.py#L911-L958)

Each Firth batch dispatches to [compute_firth_association_chunk_with_mask](file:///home/kirill/Projects/g/src/g/compute/logistic.py#796-827), calls `device_get`, and then does host-side numpy indexing. This serial Python loop prevents GPU pipeline overlap. If many variants need Firth fallback, this becomes the bottleneck.

### 20. Redundant full standard-logistic computation for heuristic-Firth variants

All variants go through [compute_standard_logistic_association_chunk_with_mask](file:///home/kirill/Projects/g/src/g/compute/logistic.py#435-596) first, including those that the heuristic already flagged for Firth. The standard IRLS work for these variants is entirely discarded. Pre-filtering before the standard pass (or using `jnp.where` to skip them inside the while-loop) would save computation.

### 21. `FIRTH_BATCH_SIZE = 64` is hardcoded

This constant isn't tunable via CLI or config, even though optimal batch size depends on GPU memory, covariate count, and sample count.

---

## 🔵 Style Guide Violations

| Issue | Location | Rule Violated |
|---|---|---|
| Missing type annotations on several public functions | `engine.py:53`, `engine.py:99`, `tabular.py:87` | 100% type coverage |
| Bare tuple return from [compute_hdiag_and_adjusted_weights](file:///home/kirill/Projects/g/src/g/compute/logistic.py#627-646) | `logistic.py:630` | No bare tuples |
| `HostStandardLogisticChunkEvaluation.coefficients` typed as `Any` | `logistic.py:68` | Exact types required |
| Leading underscore in `_core` module name | [lib.rs](file:///home/kirill/Projects/g/src/lib.rs), [_core.pyi](file:///home/kirill/Projects/g/src/g/_core.pyi) | No leading underscores† |

† The `_core` name is a PyO3/Maturin convention for native extension modules, so this is arguably exempt from the Python naming rule.

---

## Summary of Severity

| Category | Count |
|---|---|
| 🔴 Likely Bugs | 3 (#1, #3, #5 dead code) |
| 🟡 Design Debt | 8 |
| 🟠 Performance | 5 |
| 🔵 Style | 4 |

The most impactful issues to fix are:
1. **`min`/`max` bug in Firth tolerance** (#3) — changes numerical results
2. **Placeholder p-values in [LinearAssociationChunkResult](file:///home/kirill/Projects/g/src/g/models.py#58-66)** (#1) — API correctness trap
3. **Device round-trip for `betainc`** (#2, #18) — performance on GPU
4. **[logistic.py](file:///home/kirill/Projects/g/src/g/compute/logistic.py) decomposition** (#10) — maintainability

---

## Resolution Notes — 2026-03-23

This section records which review items were implemented immediately, which were explicitly cancelled, and which were deferred for later work.

### Fixed

- **#1** Linear p-values are now computed in `src/g/compute/linear.py` inside `compute_linear_association_chunk`, so `LinearAssociationChunkResult.p_value` is no longer a placeholder.
- **#2 / #18** The linear path no longer recomputes p-values in `src/g/engine.py`; `build_linear_output_frame` now forwards the kernel-produced values directly and avoids the host->device->host `betainc` round-trip.
- **#3** `src/g/compute/logistic.py` now uses `max(tolerance, FIRTH_TOLERANCE_FLOOR)` so user-provided logistic tolerance is no longer accidentally ignored for Firth fallback.
- **#5** Removed dead `covariate_only_coefficients` plumbing from `src/g/engine.py`, `src/g/compute/logistic.py`, and the affected tests.
- **#6** Added exact type annotations to `build_linear_output_frame`, `build_logistic_output_frame`, and `compute_logistic_association_with_missing_exclusion` in `src/g/engine.py`.
- **#7** Added an explicit return type to `convert_frame_to_float64_jax` in `src/g/io/tabular.py`.
- **#8** Replaced the bare tuple returned by `compute_hdiag_and_adjusted_weights` in `src/g/compute/logistic.py` with the named `AdjustedWeightComponents` result type.
- **#9** Tightened the host-side logistic transfer path in `src/g/compute/logistic.py` with a dedicated `HostLogisticAssociationChunkResult`, which makes the host mutation step more explicit.
- **#21** The hardcoded fixed Firth batch size concern is no longer current in the optimized code path; the implementation already uses `FIRTH_BATCH_BUCKETS` with bucket selection instead of a single fixed `FIRTH_BATCH_SIZE = 64` constant.

### Cancelled

- **#16** No CLI `--version` / `--verbose` / `--quiet` work was taken in this pass. The review item is valid as product UX debt, but it is not part of the current numerical-correctness and API-cleanup scope.
- **Style note under #13 / table row for `_core`** The leading-underscore naming complaint is cancelled as a style violation. `_core` is retained because it is the normal native-extension module convention for PyO3/Maturin and should not be treated as a project naming-rule breach.

### Deferred

- **#4** Import-time side effects in `src/g/jax_setup.py` are deferred. This needs a careful runtime-init design so JAX configuration order remains correct.
- **#10** Splitting `src/g/compute/logistic.py` into smaller modules is deferred as a larger maintainability refactor.
- **#11** The eager materialization in `run_linear_association` / `run_logistic_association` is deferred because changing it would alter the current DataFrame-returning API used by tests and evaluation scripts.
- **#12** Storing both genotype orientations in `GenotypeChunk` is deferred pending profiling evidence that repeated transposes remain material after the latest logistic changes.
- **#13** The broader concern that the Rust `_core` module is only a placeholder remains deferred as an architecture cleanup question.
- **#14** The `open_bed` generator-lifetime concern in `src/g/io/plink.py` is deferred.
- **#15** Logging / diagnostics are deferred as a separate observability pass.
- **#17** JIT-compiling `prepare_linear_association_state` is deferred until it is measured as a real bottleneck.
- **#19** The Python loop over Firth fallback batches is deferred; the current bucketed implementation reduced recompilation substantially, but further overlap/dispatch work is still open.
- **#20** Skipping standard-logistic work entirely for heuristic-Firth variants is deferred pending a careful parity and performance evaluation.
