## Overall verdict

Do **not** treat this as ready to replace REGENIE Step 2 yet.

The rewrite is moving in the right direction: the API is more config-centric, CLI/TOML/Python are closer to one normalization path, Polars is gone from the runtime path, and unsupported binary corrections are no longer silently pretended to work. But I found several **release blockers** in statistical correctness, multi-phenotype semantics, reproducibility, and safe automation.

The biggest concern is this: the product’s main performance strategy is multi-phenotype batching, but the current multi-phenotype alignment changes the analyzed sample set compared with separate single-phenotype runs. That means the optimization can change scientific results.

---

## What is directionally good

The recent rewrite materially improved the foundation.

`src/g/api.py` is now thin and delegates to the runner/config path rather than becoming its own orchestration layer.

CLI options are generated from a central option schema:

* `src/g/cli.py:112-155`
* `src/g/interface/options.py:72-226`
* `src/g/interface/options.py:307-624`

That is the right direction. It reduces drift between CLI flags and config normalization.

Unsupported methods are also better handled now. SPA is recognized but unsupported in `src/g/interface/options.py:295-304`, unsupported options are rejected in `src/g/interface/config.py:498-511`, and exact Firth / SPA are not silently executed in `src/g/compute/regenie2_binary.py:1659-1666`.

The effective config and manifest setup are also in a much better place:

* `src/g/runner.py:276-301`
* `src/g/runner.py:345-370`
* `src/g/execution_plan.py:173-207`
* `src/g/execution_plan.py:217-255`

So the architecture is no longer fundamentally wrong. The remaining problems are fixable, but several need to be fixed before this can be called statistically REGENIE-compatible.

---

# Release blockers

## 1. Quantitative-trait SE / CHISQ / LOG10P do not match the REGENIE fixture

The linear Step 2 kernel appears to produce the right **beta** for the REGENIE v4.1 fixture, but not the right standard error, chi-square, or log10 p-value.

The current implementation uses a full-model residual variance after fitting the genotype:

* `src/g/compute/regenie2_linear.py:238-259`

The test reference, however, documents the REGENIE-compatible default QT score-statistic formula using the null residual variance:

* `tests/test_regenie2_linear.py:183-204`

The current kernel computes degrees of freedom as:

* `src/g/compute/regenie2_linear.py:62`

and then uses the post-genotype residual sum of squares in the association statistic path.

I ran a targeted check against the existing REGENIE v4.1 fixture. Beta matches, but the other fields do not:

```text
beta
observed: [ 0.05026095  0.29938010 -0.06271657]
expected: [ 0.05026090  0.29938000 -0.06271650]

standard_error
observed: [0.45019916 0.41196188 0.48963654]
expected: [0.42739200 0.40212500 0.46493300]

chi_squared
observed: [0.01246385 0.52811890 0.01640653]
expected: [0.01382960 0.55427400 0.01819630]

log10_p_value
observed: [0.04043033 0.33031246 0.04668529]
expected: [0.04268720 0.34048600 0.04929640]
```

The existing test does not catch this because it only asserts the observed kernel beta:

* `tests/test_regenie2_linear.py:533-599`

Specifically, the reference calculation checks all fields, but the observed kernel assertion only verifies beta.

### Required fix

Decide explicitly whether the QT kernel is supposed to report REGENIE score-test output or full-model OLS output. For a REGENIE replacement, it should match the REGENIE score-test output.

The likely formula should be:

```text
beta = covariance / genotype_residual_sum_squares
SE = sqrt(null_mse / genotype_residual_sum_squares)
CHISQ = beta^2 / SE^2
null_mse = adjusted_residual_sum_squares / null_degrees_of_freedom
null_degrees_of_freedom = n - covariate_parameter_count
```

Then update all equivalent paths:

* scalar/sample-major linear path
* multi linear path
* variant-major linear path

Relevant implementation areas:

* `src/g/compute/regenie2_linear.py:238-259`
* `src/g/compute/regenie2_linear.py:274-338`
* `src/g/compute/regenie2_linear.py:443-467`

Add tests that assert **beta, SE, CHISQ, and LOG10P** against the REGENIE fixture.

---

## 2. Multi-phenotype batching currently changes the analyzed samples

This is the biggest architectural/statistical problem.

The current multi-phenotype alignment uses a shared complete-case intersection across all traits. That means:

```bash
g regenie --phenoColList A,B
```

does **not** necessarily produce the same result for phenotype `A` as:

```bash
g regenie --phenoCol A
```

If phenotype `B` has missing values in samples where `A` is present, the batched run drops those samples from `A`.

That violates the desired “decode once, compute many phenotypes” strategy because the optimization changes the estimand.

Relevant code:

* `src/sample.rs:251-294`
* `src/sample.rs:346-393`
* `src/g/engine/native_dispatch.py:47-60`
* `src/g/engine/native_dispatch.py:85-98`
* `src/g/engine/native_dispatch.py:131-151`
* `src/g/engine/native_dispatch.py:189-216`
* `src/g/engine/regenie2_pipeline.py:437-470`

The existing test confirms this is intentional current behavior:

* `tests/test_tabular.py:93-124`

### Required fix

Do not expose current multi-phenotype batching as equivalent to separate REGENIE runs.

The cleaner architecture is:

```text
decode genotype chunk once
transfer genotype chunk once
for each phenotype or phenotype-group:
    apply its own valid-sample mask/gather
    compute association
```

You can still batch phenotypes efficiently by grouping them into sample-alignment equivalence classes:

```text
group 1: phenotypes with same valid sample mask, covariates, LOCO predictions
group 2: another mask
...
```

For now, either:

1. implement per-trait masks/gathers, or
2. make complete-case batching an explicit non-default mode, for example:

```bash
--g-multi-phenotype-sample-mode complete-case
```

and document that it is **not** equivalent to separate REGENIE single-phenotype runs.

---

## 3. Multi-binary ignores user-specified binary kernel config

Single-phenotype binary runs pass the binary kernel config through the pipeline:

* `src/g/runner.py:236-241`
* `src/g/engine/regenie2_pipeline.py:162-169`
* `src/g/engine/callbacks.py:941-970`
* `src/g/engine/callbacks.py:1025-1037`
* `src/g/engine/callbacks.py:1061-1080`

But the multi-binary path drops it.

Problem locations:

* `src/g/runner.py:268-272`
* `src/g/engine/regenie2_pipeline.py:341-366`
* `src/g/engine/regenie2_pipeline.py:396-422`
* `src/g/engine/regenie2_pipeline.py:518-527`
* `src/g/engine/callbacks.py:1148-1157`
* `src/g/engine/callbacks.py:1298-1302`

The compute functions then fall back to `DEFAULT_BINARY_KERNEL_CONFIG`:

* `src/g/compute/regenie2_binary.py:450-456`
* `src/g/compute/regenie2_binary.py:569-576`
* `src/g/compute/regenie2_binary.py:593-600`

This affects reproducibility and correctness whenever users set options such as:

```text
--g-binary-null-maximum-iterations
--g-binary-null-tolerance
--g-firth-maximum-iterations
--g-firth-tolerance
--g-firth-maximum-step-size
--g-use-block-firth-math
--g-firth-batch-size
```

### Required fix

Thread `kernel_config` through the full multi-binary stack:

```text
runner
→ run_regenie2_multi_phenotype_binary_bgen_pipeline
→ run_regenie2_multi_phenotype_bgen_pipeline
→ dispatch_multi_phenotype_engine_pipeline
→ MultiBinaryRegenie2PipelineCallback
→ prepare_regenie2_multi_binary_chromosome_state
→ compute_regenie2_multi_binary_chunk
→ compute_regenie2_binary_variant_major_chunk
```

Then add a test where multi-binary with non-default kernel settings equals stacking the corresponding single-binary runs.

---

## 4. Phenotype names are unsafe path components

Phenotype names are currently used directly as output subdirectories:

* `src/g/execution_plan.py:180-187`
* `src/g/execution_plan.py:265-275`

Validation resolves phenotype names but does not reject duplicates or path-unsafe values:

* `src/g/interface/config.py:530-545`
* `src/g/interface/config.py:548-619`

This creates two problems.

First, duplicate phenotype names can produce multiple run plans writing to the same output directory.

Second, a phenotype name containing path separators or traversal can escape the intended output root:

```text
../bad
a/b
/tmp/outside
```

This is a safe automation issue, not just polish.

### Required fix

Reject duplicate phenotype names.

Also either reject unsafe names outright or map each phenotype to a safe deterministic directory name, for example:

```text
trait_0001_height
trait_0002_bmi
trait_0003_<hash>
```

The original phenotype name should remain in the manifest and output metadata, but it should not be trusted as a filesystem path component.

---

## 5. QT silently accepts binary-only flags

For quantitative traits, options such as these should not silently succeed:

```bash
--qt --firth --approx --pThresh 0.01
```

Currently they are parsed into `BinaryConfig`:

* `src/g/interface/config.py:238-244`

but ignored when the execution plan is quantitative:

* `src/g/execution_plan.py:199-203`

Validation rejects some invalid binary combinations, but it does not reject binary-only options under QT:

* `src/g/interface/config.py:548-619`

I confirmed this behavior with a direct config construction: a QT config with `firth=True`, `approx=True`, and `pThresh=0.01` succeeds.

### Required fix

Fail loudly when binary-only options are explicitly supplied for QT.

At minimum reject these under `--qt`:

```text
--firth
--approx
--firth-se
--spa
```

For `--pThresh`, because it has a default, you need to distinguish “defaulted” from “explicitly supplied.” The clean fix is to preserve option provenance through normalization, for example an `explicit_options` set or source map.

---

# High-priority issues

## 6. The Rust tabular reader is not yet the desired streaming parser architecture

The runtime is no longer using Polars, which is good. Phenotype and covariate readers are selected-column and buffered:

* `src/sample.rs:577-600`
* `src/sample.rs:684-729`
* `src/sample.rs:732-785`
* `src/sample.rs:818-877`

However, sample-file loading still reads the whole sample file into memory:

* `src/sample.rs:495-505`

Also, the parser is currently simple tab-splitting:

* `src/sample.rs:598-600`
* `src/sample.rs:646-668`

That may be acceptable for strict TSV-only input, but it is not yet the cleaner “Rust csv crate streaming TSV/CSV parser” direction you described.

### Recommendation

Use a single streaming tabular parser abstraction for sample, phenotype, and covariate files. The Rust `csv` crate is the right default choice.

This parser should:

```text
- stream rows
- select only required columns
- validate duplicate IDs
- validate required columns
- preserve deterministic row order
- handle CRLF
- clearly define missing-value tokens
- either support quoted CSV correctly or reject CSV mode explicitly
```

This is not as urgent as the statistical blockers, but it should be fixed before scaling this to large UKB-style files.

---

## 7. Output writing may become a performance bottleneck

The output path currently performs several materializations/copies.

Python callback materializes device arrays to host:

* `src/g/engine/callbacks.py:180-215`
* `src/g/engine/callbacks.py:218-261`

Rust then clones/copies chunk data:

* `src/output/session.rs:204-211`
* `src/output/session.rs:247-254`
* `src/output/writer.rs:83-148`

Finalization reads Arrow chunk files and writes final Parquet:

* `src/output/finalization.rs:42-62`

This is not necessarily wrong, but it needs measurement. For fast GPU kernels, output finalization and Python-side per-phenotype loops can dominate.

### Recommendation

Add stage timing benchmarks with:

```text
finalization on/off
Arrow chunk output only
Parquet final output
single phenotype
many phenotypes
large bsize
small bsize
```

Longer term, consider:

```text
- direct Parquet row-group writing
- fewer Python → Rust per-trait calls in multi-phenotype mode
- batched multi-trait writer API
- ownership transfer / zero-copy host buffers where safe
```

Do not optimize this blindly, but do benchmark it now.

---

## 8. JAX runtime bootstrapping remains fragile

The runner tries to configure JAX before importing the heavy pipeline:

* `src/g/runner.py:52-57`
* `src/g/runner.py:141-148`

That is good, but JAX-heavy modules still import JAX at module import time:

* `src/g/engine/native_dispatch.py:10-11`
* `src/g/engine/callbacks.py:12-24`
* `src/g/compute/*`

The risk is repeated Python use with different configs or early imports before `runner` configures the runtime. JAX backend/platform/cache settings are often process-global or sticky after initialization.

### Recommendation

Make JAX initialization policy explicit:

```text
- importing g.api should not initialize JAX
- first run configures JAX
- later incompatible runtime configs fail loudly
- compatible repeated runs are allowed
```

Add tests for:

```text
import g.api without importing JAX-heavy modules
run once with one device/cache config
run again with same config
run again with incompatible config and expect clear failure
```

This matters for reproducibility and safe automation.

---

## 9. Production binary variant-major path uses a module labeled experimental

The module is explicitly named and documented as experimental:

* `src/g/compute/regenie2_binary_variant_major_experimental.py:1`
* `src/g/compute/regenie2_binary_variant_major_experimental.py:21-25`
* `src/g/compute/regenie2_binary_variant_major_experimental.py:305-309`

But production callback code routes binary variant-major score-only computation through it:

* `src/g/engine/callbacks.py:1108-1116`

The multi-binary path also reaches variant-major computation through:

* `src/g/compute/regenie2_binary.py:603-614`
* `src/g/engine/callbacks.py:1259-1265`

### Recommendation

Either promote this to a production module and rename it, or stop using it in production.

Before promotion, add parity tests between sample-major and variant-major binary paths for:

```text
- score-only BT
- approximate Firth candidate selection
- covariates
- LOCO offsets
- missing/imputed genotypes
- monomorphic variants
- rare variants
- trusted and non-trusted BGEN paths
```

---

## 10. Manifest/resume does not yet cover enough compute-affecting state

The current manifest header includes useful fields such as input fingerprints, phenotype/covariate/pred info, sample count, variant count, chunk size, correction plan, trusted flag, and sample-key mode:

* `src/g/io/output.py:115-151`

But it does not appear to include all settings that can change output, including:

```text
binary kernel config / tolerances
Firth iteration and step-size knobs
JAX precision/backend policy
output schema version beyond manifest schema
decode tile size
multi-phenotype sample mode
output writer/finalization settings
```

Resume behavior also deserves sharper semantics. Commits are recorded on finish:

* `src/output/session.rs:142-159`
* `src/output/manifest.rs:19-55`

The Python fast resume path relies on manifest-committed chunks:

* `src/g/io/output.py:195-229`

Strict mode validates files, but the fast path does not appear to use the Rust chunk scanner in:

* `src/output/resume.rs:15-51`

### Recommendation

Add an execution-plan hash to the manifest covering all compute/output-affecting fields.

Then add resume-incompatibility tests for changes to:

```text
bsize
trait type
phenotype
covariates
predictions
binary correction plan
binary kernel config
sample-key mode
output schema
trusted BGEN mode
decode tile size
```

Also decide crash semantics explicitly:

```text
Option A: commit chunks as soon as they are durably written
Option B: only trust completed manifests and safely overwrite staged chunks
```

Either is acceptable, but it must be deterministic and tested.

---

# Testing gaps to close before calling this REGENIE-compatible

I would add these as release-blocking tests:

1. **QT REGENIE fixture full-field parity**
   Extend `tests/test_regenie2_linear.py:533-599` to assert observed kernel:

   ```text
   beta
   standard_error
   chi_squared
   log10_p_value
   ```

2. **Single vs multi phenotype equivalence**
   For missing phenotypes/covariates/predictions, each trait in a multi run must match its own single-trait run unless the user explicitly selected complete-case batching.

3. **Multi-binary kernel-config parity**
   Multi-binary with non-default kernel knobs must equal stacking single-binary runs.

4. **Unsafe phenotype names**
   Reject duplicates, `../`, path separators, absolute paths, empty names, and names that collide after sanitization.

5. **QT rejects binary-only flags**
   Especially `--firth`, `--approx`, `--firth-se`, `--spa`, and explicit `--pThresh`.

6. **Sample/covariate/phenotype parser tests**
   Large streaming files, duplicate IDs, missing values, CRLF, selected columns, and malformed rows.

7. **Resume incompatibility tests**
   Changed kernel config, changed correction plan, changed `bsize`, changed sample-key mode, corrupt/missing chunks.

8. **Repeated Python run tests**
   Same config should work. Incompatible JAX/runtime configs should fail loudly.

9. **Sample-major vs variant-major parity**
   QT and BT, trusted and non-trusted BGEN, missingness, monomorphic variants, rare variants.

10. **End-to-end REGENIE parity fixtures**
    Small deterministic BGEN fixtures against REGENIE for QT and BT, CPU mandatory, GPU optional/marked.

---

# Build/test notes

I could not run the Rust/native end-to-end path in this container because there is no Rust toolchain available, and the installed Python is `3.13.5` while the project declares Python `>=3.14,<3.15` with `pyo3` `abi3-py314`.

That means `_core` was not buildable/importable here, so I did not verify the native BGEN pipeline end to end.

I did run targeted Python/JAX checks where possible. The QT fixture mismatch above is from a direct targeted kernel check, not from speculation.

A broader pytest run that imported native-dependent modules failed collection because `_core` was unavailable. A smaller Python/JAX test run timed out before completion, so I am not claiming the suite passes.

---

# Recommended rewrite order

I would fix these in this order:

1. **Fix QT score-test statistics and strengthen the REGENIE fixture assertions.**
2. **Fix multi-phenotype sample semantics before investing further in batching.**
3. **Thread binary `kernel_config` through the multi-binary path.**
4. **Reject/sanitize unsafe and duplicate phenotype names.**
5. **Reject binary-only flags under QT.**
6. **Add an execution-plan hash to manifests and strengthen resume compatibility checks.**
7. **Replace the remaining ad hoc tabular parsing with a streaming Rust parser abstraction.**
8. **Benchmark output finalization and multi-phenotype writer overhead.**
9. **Promote or remove the “experimental” binary variant-major production path.**
10. **Add repeated-run JAX/runtime reproducibility tests.**

The core architectural direction is now much better than the old design, but the current code still has enough statistical and reproducibility issues that I would keep it firmly in pre-release until the blockers above are closed.
