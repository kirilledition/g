I re-reviewed the current snapshot. The biggest change is that the architecture is cleaner than before on the Rust side: PyO3 bindings are more isolated under `src/python/`, sample alignment has moved into `src/sample.rs`, output bindings moved to `src/python/output.rs`, and there is now a Rust architecture test enforcing that PyO3 does not leak across core modules.

The most important adjustment to my previous plan: **I would no longer treat the binary variant-major dispatch as a P0 bug.** The current docs indicate that direct variant-major JAX was tested and rejected for full-data Firth parity, so the current production choice—native variant-major decode followed by a device transpose into the existing sample-major JAX kernel—looks intentional and defensible.

Below is the revised prioritized plan for the current architecture.

---

# Revised prioritized plan

## P0 — Fix correctness and public-contract risks

### 1. Fix the binary correction contract

**Current issue**

The public contract still appears inconsistent:

* `README.md` says binary correction supports `firth_approximate` and `firth`.
* `src/g/types.py` exposes `firth_approximate` and `spa`.
* The SPA path in `src/g/compute/regenie2_binary.py` appears to return score-test results rather than applying an actual SPA correction.
* There is no clear public `"firth"` mode matching the README.

This is the highest-priority architecture issue because users will build pipelines around these options.

**Recommended direction**

Pick one of these two routes:

### Conservative route, best for now

Expose only the correction mode that is production-ready.

```text
public modes:
  firth_approximate
```

Then:

* remove or hide `spa` from the public enum;
* update README and CLI help text;
* mark SPA code as experimental/internal if you want to keep it;
* remove README references to `"firth"` unless there is a real exact-Firth implementation.

This is the safest option.

### Expansion route

Expose multiple modes only if each one has real behavior:

```text
none / score
firth_approximate
spa
firth
```

But then each mode needs a separate implementation, output labeling, and parity tests.

**Implementation guidance**

Create one source of truth for the public correction modes, probably `BinaryCorrection` in `src/g/types.py`. Then make API, CLI, README, tests, and output label generation derive from that contract.

Add tests like:

```text
tests/test_binary_correction_contract.py
```

Coverage should include:

* CLI accepts exactly the documented values.
* `g.regenie2(..., correction=...)` accepts exactly the documented values.
* README examples use only supported values.
* output `EXTRA` labels match the actual correction performed.
* SPA cannot silently label rows as SPA if score-test values were returned.

This should be done before performance work, because performance work is less useful if the public statistical meaning is ambiguous.

---

### 2. Harden sample identity alignment without silently changing behavior

**Current issue**

The new architecture moved sample alignment into Rust, which is good. But the current alignment contract still preserves the older semantics: phenotype and covariate rows are matched by `IID` only, while prediction alignment is more naturally tied to `FID/IID`.

That may be intentional for compatibility, but it is risky when `IID` is not globally unique.

**Do not silently switch the default to `(FID, IID)` yet.** That could break existing workflows.

**Recommended direction**

Keep the current default for compatibility, but make the identity contract explicit.

Add something like:

```python
sample_key_mode: Literal["iid", "fid_iid"] = "iid"
```

or, if you want less public API surface:

```python
strict_sample_identity: bool = True
```

**Implementation guidance**

Phase this in:

### Phase A: validation only

In the current IID-only mode, validate:

* duplicate `IID` in BGEN sample metadata;
* duplicate `IID` in phenotype table;
* duplicate `IID` in covariate table;
* duplicate or missing IDs in prediction sample keys;
* mismatch between aligned phenotype/covariate keys and prediction keys.

For the first release, duplicate-IID handling can be:

```text
error by default
allow only with explicit compatibility flag
```

Example config:

```python
allow_duplicate_iid_alignment: bool = False
```

### Phase B: add explicit `(FID, IID)` mode

Once validation is stable, add a proper `fid_iid` mode. In that mode:

* phenotype table must contain `FID` and `IID`;
* covariate table must contain `FID` and `IID`;
* sample file alignment uses both;
* LOCO prediction alignment uses the same key;
* all aligned arrays are sorted by BGEN sample index after joining.

### Phase C: document the contract

The README should say something like:

```text
By default, sample alignment uses IID and requires IID uniqueness.
For datasets where IID is not globally unique, use sample_key_mode="fid_iid".
```

This turns a hidden data-integrity risk into an explicit user-facing rule.

---

### 3. Keep the current parity-preserving variant-major binary path

**Current issue**

Previously I recommended changing the binary variant-major callback to call the direct variant-major JAX function. I would not recommend that now.

The current architecture appears to have learned from benchmarking: direct variant-major JAX had toy-test parity, but full-data Firth parity problems. Production now uses:

```text
Rust trusted variant-major BGEN decode
    ↓
device_put variant-major genotype matrix
    ↓
transpose on device
    ↓
existing sample-major binary JAX kernel
```

That is a reasonable production compromise.

**Recommended direction**

Keep this as the production path.

The direct variant-major binary JAX functions should be treated as experimental until Firth parity is solved.

**Implementation guidance**

Do three things:

1. Rename or mark the direct variant-major binary functions as internal/experimental.

For example:

```python
_compute_regenie2_binary_chunk_variant_major_experimental(...)
```

or add a docstring:

```python
"""
Experimental. Not used in production because full-data Firth parity has not
been established.
"""
```

2. Add a regression test that proves the trusted production path uses native variant-major decode plus transpose, not direct variant-major Firth math.

You already have tests around trusted/untrusted BGEN dispatch. Extend them so the expected behavior is explicit:

```text
trusted no-missing BGEN:
  uses run_bgen_variant_major_dosage_buffered_chunks
  computes through sample-major JAX kernel after transpose

untrusted BGEN:
  uses run_bgen_dosage_buffered_chunks
```

3. Add a parity-gate benchmark before ever promoting direct variant-major JAX.

Promotion criteria should be:

```text
same candidate count
same Firth convergence/failure count
same EXTRA labels
same beta/se/p values within agreed tolerance
same behavior across batch sizes
```

Until that is true on a representative full chromosome fixture, the current transpose path should remain.

---

## P1 — Make runtime behavior safer and easier to operate

### 4. Rename or rewire `prefetch_chunks`

**Current issue**

In the active buffered pipeline, `prefetch_chunks` no longer appears to mean native BGEN prefetch. It is effectively Python callback staging depth.

That is not necessarily bad, but the name is misleading.

**Recommended direction**

Rename the concept to what it actually does:

```python
staging_depth
```

Then either deprecate `prefetch_chunks` or keep it as a compatibility alias.

**Implementation guidance**

Best option:

```python
@dataclass
class ComputeConfig:
    staging_depth: int = 2
    prefetch_chunks: int | None = None  # deprecated alias
```

Validation:

```python
if prefetch_chunks is not None:
    warnings.warn(
        "prefetch_chunks is deprecated; use staging_depth. "
        "This controls Python callback staging depth, not native BGEN prefetch.",
        DeprecationWarning,
    )
    staging_depth = prefetch_chunks
```

CLI:

```text
--staging-depth
--prefetch-chunks  deprecated hidden alias or documented compatibility alias
```

Docs should say:

```text
staging_depth controls how many native chunks may be staged ahead of JAX compute.
It is not a guarantee of file-system read-ahead.
```

If you do want true native prefetch, add a separate option:

```python
native_prefetch_chunks: int = 1
```

But do not use the same option for both meanings.

---

### 5. Add a strict resume manifest

**Current issue**

The output writer is better now because it supports multi-chunk Arrow files and scans chunk identifiers. But resume integrity is still mostly inferred from chunk files and filenames. Single-chunk files can still be trusted by filename without opening the Arrow file.

That is fine for fast resume, but not enough for strict production recovery.

**Recommended direction**

Keep fast resume as the default if you want, but add a manifest-backed strict mode.

Suggested file:

```text
<output_dir>/run_manifest.json
```

Manifest contents:

```json
{
  "schema_version": 1,
  "association_mode": "regenie2_binary",
  "bgen_path": "...",
  "bgen_size_bytes": 123,
  "bgen_mtime_ns": 123,
  "sample_count": 1000,
  "variant_count": 500000,
  "chunk_size": 8192,
  "correction": "firth_approximate",
  "sample_key_mode": "iid",
  "trusted_no_missing_diploid": true,
  "chunks": [
    {
      "chunk_identifier": 0,
      "variant_start_index": 0,
      "variant_stop_index": 8192,
      "file": "chunks/chunk_000000000_000000003.arrow",
      "row_count": 8192,
      "committed": true
    }
  ],
  "finalized": false
}
```

**Implementation guidance**

Add:

```text
resume_mode = "fast" | "strict"
```

Fast mode:

* current behavior;
* trust existing chunk discovery;
* useful for normal interrupted runs.

Strict mode:

* open every Arrow file;
* validate schema;
* validate `chunk_identifier`;
* validate `variant_start_index` and `variant_stop_index`;
* validate row counts;
* validate run config compatibility;
* reject mixed output from different input/configs.

Also add a safety check before writing:

```text
if output directory contains manifest from incompatible run:
    error unless overwrite=True
```

This is especially important for biobank-scale jobs where silent partial-resume corruption is expensive.

---

### 6. Add preflight validation before JAX compile/warmup

**Current issue**

Several model/data problems may currently surface as numerical, JAX, Cholesky, or Firth failures rather than domain-specific errors.

**Recommended direction**

Add a dedicated preflight stage between sample alignment and JAX warmup.

Suggested module:

```text
src/g/engine/preflight.py
```

or, if you prefer to keep validation close to alignment:

```text
src/g/io/validation.py
```

**Implementation guidance**

Validate at least:

```text
sample identity:
  duplicate IID / duplicate FID-IID depending on mode
  missing phenotype rows
  missing covariate rows
  prediction coverage by chromosome

phenotype:
  non-null sample count
  binary phenotype contains valid coding
  enough cases and controls
  no all-missing traits

covariates:
  intercept behavior
  constant columns
  rank deficiency
  near-singular X'X
  sample_count > covariate_count + model_df

predictions:
  LOCO predictions exist for required chromosomes
  prediction sample order matches aligned sample order
  no unexpected NaN/Inf

BGEN/trusted path:
  trusted_no_missing_diploid only used after validation or explicit expert override
```

Return a structured object:

```python
@dataclass
class PreflightReport:
    n_samples: int
    n_variants: int
    n_cases: int | None
    n_controls: int | None
    covariate_rank: int
    warnings: list[str]
```

The CLI can print a short summary before compute begins. The API can expose it in logs or return metadata.

---

## P2 — Focus performance work where the current architecture says it matters

### 7. Prioritize Firth numerical invariance before more layout optimization

**Current issue**

The current optimization docs suggest that many obvious performance ideas were tested:

* direct variant-major JAX;
* larger batch-size changes;
* aggressive Firth refactors;
* alternate layout paths.

Some were faster or looked promising on toy fixtures, but failed full-data parity or gave marginal gains.

**Recommended direction**

The next binary-performance target should be Firth numerical invariance, not more genotype layout plumbing.

**Implementation guidance**

Create a parity harness that can run on:

```text
small toy fixture
medium representative fixture
full chromosome fixture, e.g. chr22
```

Track:

```text
score-test beta/se/p
candidate mask
Firth candidate count
Firth convergence count
Firth failure count
EXTRA labels
final beta/se/p for corrected variants
```

Then make every Firth optimization pass this harness.

Only after that should you revisit:

```text
direct variant-major JAX
larger candidate batches
blockwise Firth
mixed precision
candidate short-circuiting
```

This keeps performance work from breaking statistical reproducibility.

---

### 8. Replace environment-only trusted-path skipping with a validation cache

**Current issue**

There is an expert-style environment override for trusted no-missing diploid validation. That is useful during development, but fragile for production.

**Recommended direction**

Keep the override, but add a validation cache or manifest.

Suggested file:

```text
<cache_dir>/bgen_validation/<fingerprint>.json
```

Fingerprint should include:

```text
BGEN path
file size
mtime
maybe first/last block checksum
reader version
validation mode
sample count
variant count
```

**Implementation guidance**

Flow:

```text
if trusted_no_missing_diploid requested:
    check validation cache
    if valid cache hit:
        allow trusted path
    else:
        run validation
        write validation manifest
```

CLI options:

```text
--trusted-no-missing-diploid
--validate-trusted-bgen
--assume-trusted-bgen-validated
```

Use the environment variable only as a last-resort expert override.

This makes the fast path safer without forcing users to revalidate huge BGENs every run.

---

## P3 — Continue refactoring around the newer module boundaries

The Rust architecture is already moving in the right direction. The next refactor should focus mostly on the remaining large files.

### 9. Split the Python engine pipeline

**Current issue**

`src/g/engine/regenie2_pipeline.py` is still carrying too many responsibilities: orchestration, callbacks, timing, warmup, native dispatch, output finalization, and data movement.

**Recommended split**

```text
src/g/engine/
  regenie2_pipeline.py       # thin public orchestration entry point
  callbacks.py               # linear/binary pipeline callback classes
  native_dispatch.py          # calls into Regenie2RunEngine
  warm_cache.py               # JAX warmup/cache behavior
  timing.py                   # timing summaries
  preflight.py                # validation before compute
  output_session.py           # writer session lifecycle
```

**Guidance**

Do this after P0/P1 fixes. Refactors are safer once the public contract and resume behavior are stable.

---

### 10. Split binary compute into statistical submodules

**Current issue**

`src/g/compute/regenie2_binary.py` is still very large and mixes score tests, Firth correction, candidate labeling, batching, diagnostics, and variant-major experiments.

**Recommended split**

```text
src/g/compute/binary/
  __init__.py
  score.py
  firth.py
  correction.py
  candidate_plan.py
  diagnostics.py
  variant_major_experimental.py
  types.py
```

**Guidance**

Move pure functions first. Avoid changing math while moving code.

Good first extraction targets:

```text
score-test-only functions
candidate mask construction
EXTRA label generation
diagnostic structs
experimental variant-major functions
```

Leave Firth internals until you have the parity harness from P2.

---

### 11. Split Rust BGEN internals

**Current issue**

`src/genotype/bgen.rs` remains one of the largest architecture hotspots.

**Recommended split**

```text
src/genotype/bgen/
  mod.rs
  index.rs
  decode.rs
  metadata.rs
  sample_selection.rs
  trusted.rs
  profile.rs
```

**Guidance**

Keep the public Rust API stable:

```rust
pub use bgen::BgenReader;
```

Then move internals behind that facade.

Suggested order:

1. move profiling structs/functions;
2. move metadata/index parsing;
3. move sample-selection helpers;
4. move trusted no-missing path;
5. move decode internals last.

That order minimizes risk.

---

### 12. Split output writer by lifecycle

**Current issue**

`src/output/writer.rs` now does batching, Arrow schema, resume scanning, file writing, Parquet finalization, and output coordination.

**Recommended split**

```text
src/output/
  mod.rs
  schema.rs
  session.rs
  coordinator.rs
  arrow_file.rs
  parquet.rs
  resume.rs
  manifest.rs
```

**Guidance**

This pairs naturally with the strict manifest work.

Refactor sequence:

1. extract schema creation;
2. extract resume scanning;
3. add manifest support;
4. extract Parquet finalization;
5. leave session/coordinator cleanup for last.

---

### 13. Decide what `pipeline/backend.rs` is for

**Current issue**

The Rust `AssociationBackend` abstraction looks like a future architecture layer, but it does not appear central to the active Python/JAX production path yet.

That can become dead architecture if it remains unused.

**Recommended direction**

Choose one:

### Option A: keep and prove it

Add a tiny test backend and one engine path that consumes the trait. This proves the abstraction is real.

Example:

```rust
struct NoopAssociationBackend;
impl AssociationBackend for NoopAssociationBackend { ... }
```

Then test:

```text
engine can deliver AssociationChunk into backend
backend returns AssociationResults
```

### Option B: mark it explicitly dormant

If it is for a future Rust-native association engine, add a module-level comment:

```rust
//! Experimental backend abstraction for future Rust-native association kernels.
//! Not used by the current Python/JAX production path.
```

This prevents maintainers from assuming it is already part of the active runtime architecture.

---

# Items that changed from my previous plan

The previous plan had one recommendation I would now revise strongly:

```text
Old recommendation:
  Make the binary variant-major callback call the direct variant-major JAX function.

Current recommendation:
  Do not promote direct variant-major JAX yet.
  Keep the production transpose path because full-data Firth parity matters more than layout purity.
```

Also, the Rust/Python boundary has improved, so the plan should no longer say “move sample alignment into Rust” or “isolate PyO3” as primary tasks. Those are largely done. The next step is to **harden the contracts** around those new boundaries.

---

# Suggested execution order

## Sprint 1: public correctness

1. Fix binary correction contract.
2. Add sample identity duplicate validation.
3. Mark direct variant-major binary JAX as experimental/internal.
4. Add tests proving trusted binary production path uses variant-major native decode plus sample-major JAX compute.

## Sprint 2: operational safety

5. Rename `prefetch_chunks` to `staging_depth` or add `staging_depth` as the new canonical option.
6. Add preflight validation.
7. Add strict resume manifest.

## Sprint 3: performance foundation

8. Build the Firth parity harness.
9. Add trusted BGEN validation cache.
10. Revisit binary performance only after parity gates are in place.

## Sprint 4: maintainability

11. Split `regenie2_pipeline.py`.
12. Split `regenie2_binary.py`.
13. Split `bgen.rs`.
14. Split `writer.rs`.
15. Decide whether `pipeline/backend.rs` is active architecture or future-only scaffolding.

---

# Bottom line

The current architecture is in better shape than the prior snapshot. The Rust boundary is cleaner, sample alignment is more centralized, and the production binary trusted path appears intentionally designed around full-data parity.

The revised priority is therefore not “rewrite the pipeline.” It is:

```text
1. make the public statistical contract unambiguous;
2. make sample identity alignment explicit and safe;
3. preserve the current parity-first variant-major production path;
4. clarify runtime knobs like prefetch/staging depth;
5. add strict resume and preflight validation;
6. only then pursue deeper Firth/layout performance work.
```
