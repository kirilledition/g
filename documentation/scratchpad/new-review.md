I reviewed the current `main` branch architecture. Static review only; I did not run the test suite or benchmarks.

## High-level verdict

The architecture is now in a good shape conceptually:

```text
CLI / TOML / Python API
        ↓
RegenieConfig
        ↓
ExecutionPlan
        ↓
runner
        ↓
native-driven BGEN pipeline
        ↓
Python/JAX compute callback
        ↓
Rust output writer
```

That is the right architecture family for the current hybrid Rust/Python/JAX design. The main strengths are:

```text
- public Python API is thin
- execution planning is explicit
- JAX runtime setup is isolated before heavy compute imports
- Rust owns BGEN/sample/output work
- output/resume/manifest design is serious
- telemetry/profiling is built into the run lifecycle
- packed8 / native stats / donated JAX inputs show real performance thinking
```

The main remaining architectural problems are:

```text
1. Python is still in the per-chunk hot loop.
2. Config/interface code on main is still too heavy, even though the Rust config branch is addressing this.
3. ExecutionPlan still leaks broad public config objects into lower layers.
4. `callbacks.py` is too large and owns too many responsibilities.
5. Output/result ownership still crosses Python/Rust more than ideal.
6. Multi-phenotype grouping is the right direction but needs to become the default semantic-preserving performance path.
7. The architecture needs a first-class backend planner if you want Rust-score / JAX / packed8 / future CUDA backends to coexist cleanly.
```

I would **not rewrite the whole app again**. I would continue the current direction, but sharpen boundaries.

---

# What is good

## 1. Public API is now appropriately thin

`src/g/api.py` is basically a small wrapper around `runner.regenie(...)` and `RegenieConfig.from_options(...)`. That is good; public API should not own orchestration, config semantics, or compute behavior. 

This is the right pattern:

```python
g.regenie(config)
g.regenie.from_options({...})
```

The API layer should remain boring.

---

## 2. Runner owns lifecycle and runtime policy

`runner.py` now owns process-global runtime setup, logging policy, JAX runtime configuration, execution dispatch, telemetry, run events, and final artifacts. This is a much cleaner center of gravity than earlier versions. It explicitly delays JAX-heavy pipeline imports until after runtime policy is applied. 

That lazy JAX boundary is important. You do not want compute modules imported before the app has chosen CPU/GPU, cache policy, transfer guard, precision policy, and related process-global JAX settings. The runner’s `configure_runtime_before_jax_import(...)` is the right idea. 

The logging policy is also now explicit and process-global, including queue size, lossy behavior, source location, span events, trace file, and trace event cap. 

---

## 3. ExecutionPlan is a useful boundary

`execution_plan.py` defines explicit immutable objects for:

```text
KernelConfig
OutputPlan
PhenotypeRunPlan
RegenieExecutionPlan
```

These are the right abstractions. They separate “what the user asked for” from “how the engine will run.”  

I especially like that binary correction flags become a `BinaryCorrectionPlan`, rather than letting `--firth`, `--approx`, `--spa`, and `--pThresh` leak everywhere. 

The kernel config construction is also clean: binary-specific and linear-specific numerical/kernel configs are built from resolved `GComputeConfig`, then passed downstream. 

---

## 4. Native pipeline context is explicit

`regenie2_pipeline.py` now has a `Regenie2PipelineContext` that gathers association mode, genotype source, phenotype/prediction/covariate paths, chunking, trusted BGEN policy, JAX device, dtype policy, genotype format, correction plan, writer settings, telemetry, and alignment config. 

That is a good lifecycle boundary. It is better than passing dozens of loose arguments through many functions.

The pipeline also builds manifest headers from the context and includes important execution-affecting settings such as chunk size, binary correction plan, trusted BGEN mode, BGEN validation mode, JAX device, genotype format, score/firth dtype, output format, writer settings, Arrow/Parquet compression, and multi-phenotype mode. 

That is the correct resume/reproducibility mindset.

---

## 5. Native sample alignment is in the right place

`crates/input/src/sample/` owns sample/phenotype/covariate alignment and has native structures for single phenotype, multi phenotype, phenotype groups, and grouped aligned sample data.

It uses a native streaming tabular reader structure rather than Python DataFrames. 

Single-phenotype and multi-phenotype alignment paths are explicit. The multi-phenotype path documents that complete-case intersection is not equivalent to running each phenotype separately, which is the right statistical honesty. 

---

## 6. Callback pipeline has matured

`callbacks.py` is now doing serious pipeline work:

```text
native delivery queue
JAX compute worker
result materialization/write queue
bounded in-flight slots
queue backpressure timing
optional exact JAX synchronization in profile mode
native stats passed into kernels
```

The transfer helpers block only when detailed timing is enabled, which is important for production throughput. 

The callback code also records queue pressure and per-chunk progress, which is exactly the kind of instrumentation you need to diagnose pipeline underutilization.  

---

## 7. Output writer architecture is much better than a Python writer

The Rust output writer has:

```text
NativeChunkHandle
Arc-backed metadata/stats
OnceLock-built Arrow writer arrays
worker pool
coordinator jobs
stage timing accumulator
manifest commits
finalization timing
```

`NativeChunkHandle` and cached `NativeChunkWriterArrays` are a good direction: metadata/stats are Rust-owned, and writer arrays are built lazily once per chunk. 

The output writer timing accumulator is detailed enough to separate metadata clone time, result buffer copy time, Arrow batch creation, file write, manifest commits, and finalization. 

That is good architecture for performance work.

---

## 8. Binary compute API now has serious JAX boundary work

The binary API has jitted entry points for chromosome state prep, score-only variant-major compute, donated-input compute, multi-binary score compute, packed8 decode/compute, and no-overflow Firth paths.   

That is a big improvement over earlier code. The existence of donated-input entry points and packed8 paths shows the right performance direction.

---

# Main architectural problems

## P0/P1 — Python is still in the per-chunk hot loop

The current hot loop is still roughly:

```text
Rust decodes chunk
    ↓
Python callback receives work item
    ↓
Python puts genotype/stats on device
    ↓
Python calls jitted JAX function
    ↓
Python enqueues JAX result
    ↓
Python result worker calls device_get
    ↓
Python calls Rust writer
```

You can see this in the callback queue consumer and result materialization paths.  

This is acceptable for the current JAX architecture, but it is still the main theoretical ceiling for single-trait and small-chunk workloads. The interpreter is not doing math, but it is orchestrating every chunk.

### Recommendation

Do not try to “compile Python.” Instead, reduce how often Python participates.

In order:

```text
1. Increase/default-tune bsize and queue depths based on profile data.
2. Keep one jitted call per chunk for score-only paths.
3. Group multiple chunks per Python dispatch where memory allows.
4. Push more result ownership through NativeChunkHandle.
5. Add a Rust score-only backend for low-arithmetic single-trait workloads.
```

The long-term architecture should be:

```text
Python:
  one call per run or coarse batch

Rust:
  streaming engine, BGEN, sample alignment, output

JAX:
  large batched kernels only
```

Not:

```text
Python:
  one orchestration step per chunk forever
```

---

## P1 — `callbacks.py` is too large and has too many responsibilities

`callbacks.py` currently owns:

```text
worker lifecycle
queues
queue backpressure timing
device transfer
chunk stats extraction
null logistic convergence policy
binary diagnostics
progress events
JAX compute dispatch
result materialization
writer calls
buffer/in-flight slot release
```

That is too much for one file/module. It is now a mini runtime.

The abstractions are good, but the module should be split before it becomes unmaintainable.

### Recommended split

```text
src/g/engine/callbacks/
  __init__.py
  base.py
    worker lifecycle
    queues
    shutdown/error handling

  transfer.py
    device_put helpers
    transfer metadata
    block_until_ready policy

  result_writer.py
    device_get
    public dtype narrowing
    writer_session bridge

  progress.py
    chromosome/progress telemetry

  diagnostics.py
    binary chunk diagnostics
    queue backpressure diagnostics

  linear.py
    linear callback implementations

  binary.py
    binary callback implementations

  multi.py
    multi-phenotype callbacks
```

Keep the public callback classes stable, but move implementation details out.

---

## P1 — ExecutionPlan still leaks too much public config downward

`KernelConfig` includes:

```python
alignment_config: config.GComputeConfig
```



That means a large public/user config object is passed into lower layers where only a subset is needed. In `runner.build_common_engine_arguments(...)`, `alignment_config` is passed through to the native pipeline. 

This blurs boundaries. `GComputeConfig` is a public resolved config section. The engine should receive narrower engine-specific config structs.

### Recommendation

Replace `alignment_config: config.GComputeConfig` with explicit narrow configs:

```python
@dataclass(frozen=True)
class AlignmentConfig:
    sample_key_mode: SampleKeyMode
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode
    null_logistic_nonconvergence_policy: NullLogisticNonconvergencePolicy

@dataclass(frozen=True)
class JaxKernelPolicy:
    device: Device
    matmul_precision: JaxMatmulPrecision | None
    score_dtype: FloatingPointDtype
    firth_dtype: FloatingPointDtype

@dataclass(frozen=True)
class NativeDecodeConfig:
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: TrustedBgenValidationMode
    bgen_decode_tile_variant_count: int
    gpu_genotype_format: GpuGenotypeFormat
```

Then `KernelConfig` contains those, not the full `GComputeConfig`.

This makes it obvious which settings affect alignment, JAX, native decode, binary kernel config, and output compatibility.

---

## P1 — Runner and pipeline duplicate lifecycle objects

You now have:

```text
ExecutionPlan
KernelConfig
OutputPlan
Regenie2PipelineContext
OutputWriterSettings
common_arguments dict
```

The conversion chain is somewhat repetitive:

```text
RegenieConfig
  -> ExecutionPlan
  -> common_arguments dict
  -> Regenie2PipelineContext
  -> native engine
```

`build_common_engine_arguments(...)` constructs a large untyped dict with many fields. 

That is a smell. You have strong dataclasses before and after, but an untyped dict in the middle.

### Recommendation

Replace the dict bridge with a typed object:

```python
@dataclass(frozen=True)
class EngineDispatchRequest:
    genotype_source_config: GenotypeSourceConfig
    phenotype_path: Path
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    phenotype_runs: tuple[PhenotypeRunPlan, ...]
    kernel_config: KernelConfig
    output_plan: OutputPlan
    telemetry_session: TelemetrySession | None
    stage_timing_recorder: StageTimingRecorder | None
```

Then:

```python
dispatch_one_phenotype_engine_pipeline(request, phenotype_run)
dispatch_multi_phenotype_engine_pipeline(request)
```

This would make mypy/ty catch a lot of mistakes that `**common_arguments` hides.

---

## P1 — Config architecture on `main` is improved but still transitional

`main` now uses `msgspec`, typed TOML schema/layers/defaults, and dataclasses without default constructors. That is a good improvement. `RegenieConfig` is a complete normalized config, and `from_options(...)` overlays typed layers over packaged defaults.  

But `config.py` is still very large and still manually maps typed TOML sections into the runtime dataclass field by field. 

Given you are already exploring a Rust CLI/config branch, I would not over-invest in cleaning this Python config layer further. The correct future is likely:

```text
Rust:
  clap
  serde/toml
  config layering
  validation
  PyO3 config objects

Python:
  thin consumer of resolved config
```

### Recommendation

For `main`, keep the current config layer stable enough to run. Do not add more complex Python config features. Put new architecture effort into the Rust config frontend branch.

When Rust config lands, delete as much Python config code as possible rather than keeping two engines.

---

## P1 — Output/result data still crosses Python/Rust more than necessary

The result path currently does:

```text
JAX arrays
  -> narrow to float32 on device
  -> jax.device_get into Python host values
  -> np.asarray(..., dtype=np.float32)
  -> writer_session.write_regenie2_native_chunk(...)
  -> Rust writer
```

The device-to-host materialization path is visible here. 

This is unavoidable if the writer is CPU-side and JAX is the compute backend. But the architecture should minimize Python object creation and copies around this boundary.

### Recommendation

Continue toward:

```text
Rust NativeChunkHandle passes through Python untouched
JAX returns only numeric result arrays
Python calls one PyO3 method with handle + arrays
Rust writer consumes arrays without extra metadata/stat cloning
```

You already have `NativeChunkHandle` and cached writer arrays in Rust.  The remaining goal is to make the Python path handle-only, not metadata/stats reconstruction.

---

## P1 — Multi-phenotype performance path still needs a clean planner

The current architecture clearly supports multi-phenotype dispatch: `runner` chooses `dispatch_multi_phenotype_engine_pipeline(...)` when there are multiple phenotype run plans. 

The pipeline has a `PreparedMultiPhenotypeGroupDelivery` abstraction for compatible phenotype groups fed by a union-sample native decode buffer. 

That is the right direction, but it needs to be made a first-class planning concept rather than an implementation detail.

### Recommendation

Add an explicit plan layer:

```python
@dataclass(frozen=True)
class PhenotypeComputeGroup:
    phenotype_indices: tuple[int, ...]
    phenotype_names: tuple[str, ...]
    sample_set_fingerprint: str
    covariate_design_fingerprint: str
    prediction_alignment_fingerprint: str
    sample_mode: MultiPhenotypeSampleMode
```

Then `ExecutionPlan` contains:

```python
phenotype_compute_groups: tuple[PhenotypeComputeGroup, ...]
```

This should distinguish:

```text
per-phenotype groups
complete-case group
identical-sample/covariate groups
future masked groups
```

Right now this logic seems split between sample alignment/native dispatch/pipeline preparation. It should become explicit in `ExecutionPlan`.

---

## P2 — Top-level runner imports are still heavier than ideal

`runner.py` imports `_core`, `execution_plan`, `jax_runtime`, `types`, run events, shutdown, telemetry, timing, config, and output at top level. 

This is not a big compute-path problem, but it means importing the API pulls in the native extension and a good amount of runtime machinery. Since this is a CLI/scientific engine, that may be acceptable.

But if you care about fast `g --help`, `g config validate`, and lightweight Python config inspection, keep pushing CLI/config into Rust and keep JAX/pipeline imports lazy.

The runner’s JAX-heavy pipeline import boundary is good.  I would not spend much time optimizing the rest until the Rust CLI/config branch settles.

---

# Suggested target architecture

## Near-term target

```text
Rust config frontend
  clap CLI
  serde/toml
  typed config validation
  PyO3 config objects

Python public API
  g.regenie(config)
  g.regenie.from_options(...)
  no config parsing logic beyond PyO3 shims

Python runner
  JAX runtime setup
  execution plan build
  dispatch engine requests

ExecutionPlan
  explicit phenotype compute groups
  explicit kernel/output/native/JAX configs

Rust native engine
  BGEN/index/sample/prediction/output
  chunk handles
  writer sessions
  resume/manifest repair

Python/JAX callbacks
  only compute dispatch + device materialization
  fewer per-chunk responsibilities
```

## Medium-term target

```text
backend planner:
  jax-score
  jax-packed8
  rust-score
  future cuda-score

Rust-score backend:
  single-trait score-only streaming CPU path
  removes JAX/Python overhead where GPU is not useful

JAX backend:
  multi-phenotype and Firth-heavy paths
```

## Long-term target

```text
Python no longer appears in the inner chunk loop for score-only paths.
JAX remains for accelerator-heavy batched/Firth paths.
Rust owns all I/O, chunk lifecycle, output, and maybe score-only compute.
```

---

# Recommended next refactor sequence

## 1. Finish Rust CLI/config branch

This is the highest architectural leverage.

Requirements before merge:

```text
- REGENIE CLI flags exactly supported.
- Non-REGENIE CLI flags use a clear g namespace or settled final naming.
- TOML shape is final and documented.
- Defaults live in TOML.
- Config construction does not duplicate defaults.
- CLI/config tests are restored.
- Python config is a thin PyO3 wrapper.
```

Do not keep both Python config and Rust config as competing engines.

---

## 2. Split `callbacks.py`

Make this a focused cleanup. Do not change compute math.

Suggested task:

```text
Split callback runtime into:
  base worker/queue runtime
  device transfer helpers
  result writer/materializer
  diagnostics/progress
  linear callbacks
  binary callbacks
  multi callbacks
```

Acceptance criteria:

```text
same tests pass
same telemetry fields
no changed output
module size reduced
```

---

## 3. Replace `common_arguments` dict with typed dispatch request

This will make engine boundaries much safer.

Current `build_common_engine_arguments(...)` returns a large dict that is spread into multiple pipeline calls. 

Replace it with dataclasses.

---

## 4. Make phenotype compute grouping explicit in ExecutionPlan

Move grouping semantics into planning, not hidden pipeline preparation.

This is important for performance and for explaining multi-phenotype semantics.

---

## 5. Continue NativeChunkHandle ownership cleanup

Use Rust handles end-to-end so Python is only the bridge for JAX numeric arrays.

---

## 6. Add a backend planner

Start small:

```text
BackendKind:
  JAX_DOSAGE
  JAX_PACKED8
  RUST_SCORE_EXPERIMENTAL
```

Then `ExecutionPlan` chooses the backend based on config and workload. This will make future native compute/CUDA experiments less invasive.

---

# Architecture scorecard

## Good

```text
Public API shape
ExecutionPlan boundary
JAX runtime boundary
Rust sample/BGEN/output ownership
Manifest/resume seriousness
Telemetry/profiling design
Packed8/JAX donated-input direction
Native stats passed to kernels
Multi-phenotype grouping direction
```

## Needs cleanup

```text
Callbacks module size and responsibility creep
Untyped dict boundary between runner and pipeline
Full GComputeConfig leaking into kernel/alignment layers
Python still in per-chunk hot loop
Config system still transitional on main
Output result handoff still Python-heavy
```

## Strategic missing piece

```text
A backend planner that can choose:
  Rust streaming score backend for low-arithmetic score-only jobs
  JAX GPU backend for multi-phenotype/Firth-heavy jobs
  packed8/device-decode backend when H2D is limiting
```

---

# Bottom line

The current architecture is good enough to keep evolving. I would not do a broad rewrite.

The best next architectural moves are:

```text
1. Finish the Rust CLI/config migration cleanly.
2. Split callbacks.py.
3. Replace loose dispatch dictionaries with typed dispatch requests.
4. Make phenotype compute groups explicit in ExecutionPlan.
5. Continue moving output/chunk ownership into Rust handles.
6. Add backend planning so Rust-score/JAX/packed8/future CUDA can coexist.
```

The biggest long-term performance ceiling is still not “JAX formulas” alone. It is the architecture where Python coordinates every chunk. The current queueing and donated-input work makes that tolerable, but the fastest final design will use Python at run/batch boundaries, not at every chunk boundary.
