# P1 performance issues

## 7. The callback pipeline still serializes GPU work per chunk

In `src/g/engine/callbacks.py`, every chunk does:

```python
jax.device_put(...)
block_until_ready(genotype_device_array)
compute(...)
block_until_ready(result.log10_p_value)
enqueue result for materialization/write
```

Relevant lines:

* H2D blocking: `put_genotype_matrix_on_device`, lines `108-117`
* linear compute blocking: lines `546-554`
* binary compute blocking: lines `662-673`
* variant-major binary compute blocking: lines `687-706`

This gives good stage timings, but it limits overlap. The result materializer thread helps, but the compute worker still handles only one chunk at a time and forces synchronization after transfer and compute.

**Guidance**

Keep this behavior for profiling mode, but add a production async mode:

```text
decode chunk k+1 on CPU
H2D chunk k
compute chunk k-1 on GPU
D2H/write chunk k-2
```

A safer incremental version:

```python
device_array = jax.device_put(host_array)
# either keep host buffer alive until compute is launched,
# or use explicit safe-release discipline
result = compute(device_array)
result_queue.put(result)
```

The result worker can be the synchronization point:

```python
host_values = jax.device_get(result)
writer.write(...)
```

You will need an in-flight limit to avoid unbounded GPU memory usage.

---

## 8. Text finalization is likely slow

`src/output/finalization.rs` writes REGENIE text by iterating every row and every column and calling:

```rust
array_value_to_string(column.as_ref(), row_index)
```

Relevant lines: `107-119`.

That is convenient, but for millions of variants it will be slow. If `g-output-format=regenie` is the default, finalization can dominate wall time and make GPU improvements invisible.

**Guidance**

make Arrow/Parquet the only remove all switches and code related to REGENIE text output, we do not need compatibility

---

## 9. Phenotype/covariate parsing is Rust-native now, but still not optimal

Polars is gone, which is good. But `src/sample.rs` currently reads tabular files like this:

```rust
let table_content = std::fs::read_to_string(table_path)?;
let rows = lines.map(split_tabular_line).collect();
```

Relevant lines: `387-404`.

This materializes:

```text
entire file string
all rows
all columns
Vec<Vec<String>>
```

For large biobank phenotype/covariate files, this is not ideal.

**Guidance**

Replace this with a streaming selected-column TSV parser.

Target algorithm:

```text
read header
find FID/IID + selected phenotype/covariate columns
stream rows
ignore unneeded columns
fill hash maps or directly fill aligned arrays
```

This also pairs naturally with multi-phenotype execution: read the phenotype file once and fill a phenotype matrix.

---

## 10. Output writer still clones a lot of data

The Python callback materializes JAX arrays to host and calls the Rust writer with metadata and result arrays. The Rust writer then clones metadata/result data into jobs.

This is visible in `src/output/session.rs`, where chunk jobs are built from cloned vectors.

For very large runs, this adds CPU memory traffic and allocation overhead.

**Guidance**

Move toward Rust-owned chunk handles:

```rust
struct NativeChunkHandle {
    metadata: Arc<VariantMetadataColumns>,
    stats: Arc<ChunkStats>,
    chunk_identifier: i64,
}
```

Python should pass only:

```text
chunk handle
beta/se/chisq/log10p arrays
extra_code array
```

The writer should reuse metadata already owned by Rust instead of copying strings/vectors through Python and back into Rust.

---

## 11. Manifest updates may serialize writer workers

`src/output/manifest.rs` updates `run_manifest.json` with a global lock, reads the whole JSON, mutates it, pretty-prints it, syncs it, and renames it.

Relevant lines: `18-90`.

That is robust, but if writer batches are small, it can serialize writer workers and become visible.

**Guidance**

Use one of these:

```text
Option A:
  writer workers write Arrow files only
  coordinator records commits and updates manifest less frequently

Option B:
  append-only commits.jsonl
  compact to run_manifest.json at finish

Option C:
  increase chunks_per_arrow_file substantially by default
```

For performance, I would not update a pretty JSON manifest after every tiny batch.

---

## 12. Untrusted row-major preprocessing is still likely slow

The trusted variant-major path is much better now. It decodes and accumulates per-variant stats together, which is exactly the right direction.

But the untrusted/missing path still goes through row-major preprocessing, and the preprocessing path scans genotype buffers separately.

If the product is going to strongly encourage trusted BGEN, this is acceptable. If not, untrusted BGEN will remain much slower.

**Guidance**


```text
1. implement variant-major untrusted decode + stats + imputation path.
```

Do not invest in GPU micro-optimizations before knowing which BGEN path users will actually run.

---

# P1/P2 architecture issues

## 13. `api.py` is doing too much

`src/g/api.py` currently owns:

```text
public API
runtime configuration
binary correction normalization
multi-phenotype loop
output directory planning
engine dispatch
effective config writing
manifest extension
final REGENIE text finalization
```

This is better than before, but it is now the next architecture bottleneck.

**Guidance**

Create a true execution layer:

```text
src/g/runner.py
  regenie(config) -> RunArtifacts

src/g/execution_plan.py
  RegenieExecutionPlan
  PhenotypeRunPlan
  OutputPlan
  KernelConfig

src/g/api.py
  public Python wrappers only

src/g/cli.py
  CLI parsing only
```

The engine should receive an `ExecutionPlan`, not a public-facing `RegenieConfig`.

---

## 14. The CLI still duplicates the option table

You now have `src/g/interface/options.py`, which is good. But `src/g/cli.py` still hand-writes all Click options.

That means drift can reappear.

**Guidance**

Generate `g regenie` options from `OptionSpec`.

The `OptionSpec` should include at least:

```python
name
cli_flags
destination
section
type
default
support_level
help_text
multiple
is_flag
accepted_values
```

Then use it to drive:

```text
Click CLI
TOML schema
config template
config validation
docs/help
Python from_options()
```

This is worth doing before the option list grows further.

---

## 15. Effective config is written after the run

`src/g/api.py` writes `effective_config.toml` only after the engine completes:

```python
effective_config_path = artifacts.output_run_directory / "effective_config.toml"
interface_config.write_toml(regenie_config, effective_config_path)
```

Relevant lines: `204-214`.

If the run fails, the effective config may not be written, which is exactly when you most want it.

**Guidance**

Write the effective config before engine execution starts, after output directory preparation and before chunk processing.

Also write it into the manifest at the beginning, not only after success.

---

## 16. Runtime configuration order should be tightened

`configure_runtime()` imports binary compute modules before device configuration happens in `run_existing_engine()`.

Relevant lines:

* `configure_runtime`, `src/g/api.py:138-161`
* `configure_jax_device`, `src/g/api.py:240-243`
* `jax_setup.py` imports `jax` at module import time and sets `jax_enable_x64`, lines `9-10` and `77`

This may work if no backend is initialized early, but it is fragile. JAX platform selection should be decided before importing compute modules and before any code can call `jax.devices()` or trigger backend initialization.

**Guidance**

Make runtime bootstrap explicit:

```python
def configure_runtime_before_jax_import(plan):
    set env vars if needed
    import jax
    jax.config.update("jax_platforms", ...)
    jax.config.update("jax_compilation_cache_dir", ...)
    ...
```

Then import compute modules.

Also do not configure binary Firth runtime for quantitative runs.

---

## 17. `callbacks.py` has a possible shutdown/deadlock edge case

`stop_result_worker()` loops forever trying to put a sentinel into a bounded queue:

```python
while True:
    try:
        self.result_queue.put(None, timeout=0.1)
        return
    except queue.Full:
        continue
```

Relevant lines: `352-359`.

If the result worker has already failed and the result queue is full, this can spin forever. `raise_worker_error_if_present()` is not checked inside this loop.

**Guidance**

Change it to:

```python
def stop_result_worker(self):
    while True:
        if self.result_worker_error is not None:
            return
        if not self.result_worker_thread.is_alive():
            return
        try:
            self.result_queue.put(None, timeout=0.1)
            return
        except queue.Full:
            continue
```

Then join with a timeout and raise a structured error if the thread fails to stop.

---

## 18. `staging_depth=0` is accepted but silently coerced

`validate_config()` allows `--g-staging-depth 0`:

```python
if config.g_compute.staging_depth < 0:
    ...
```

But `NativeBgenCallbackRunner` does:

```python
self.dosage_queue_depth = max(1, staging_depth)
```

Relevant lines:

* config validation: `src/g/interface/config.py:576-578`
* callback coercion: `src/g/engine/callbacks.py:176-178`

This is a small contract bug.

**Guidance**

Either:

```text
require staging_depth >= 1
```

or make `0` mean a true synchronous/no-staging mode. I would require `>= 1`.

---

# Statistical/math concerns to verify

## 19. Linear LOCO/covariate handling should be parity-tested carefully

The linear state computes phenotype residuals against covariates and then subtracts LOCO predictions later. This may be correct depending on how Step 1 predictions are defined, but the subtle question is:

```text
residualize(y) - prediction
```

versus:

```text
residualize(y - prediction)
```

If LOCO predictions are not covariate-orthogonal, these can differ.

I am not saying this is wrong, but it is important enough to lock down with parity tests against REGENIE.

**Guidance**

Add a tiny fixture where covariates and LOCO predictions are correlated and compare both formulas against REGENIE output.
