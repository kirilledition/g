I re-reviewed the current snapshot as a performance review. I did not run a new benchmark, so the plan below is based on static code inspection plus the profiling numbers already present in your repo docs.

My read: **there probably are a few fixable mistakes, but there is also a real technology/algorithmic limit for single-trait REGENIE Step 2 on GPU.** The GPU only helps if enough arithmetic is done per decoded genotype byte. For one phenotype, especially quantitative score tests, most of the job is still CPU BGEN decode, host buffers, host-to-device transfer, output materialization, and writing. REGENIE’s C++ CPU implementation is already very good at that kind of streaming workload.

The biggest actionable points are:

1. **Your binary default is probably not apples-to-apples with REGENIE.** `g` defaults to `FIRTH_APPROXIMATE`, while REGENIE only uses Firth/SPA when flags such as `--firth`, `--approx`, `--spa`, and `--pThresh` are requested. REGENIE documents `--firth` as fallback for p-values below threshold, `--approx` as only meaningful with `--firth`, `--spa` as another fallback, and `--pThresh` defaulting to `0.05`. ([RGC GitHub][1])
2. **Your repo’s own profiles say the GPU is not the only bottleneck.** In `docs/regenie2-binary-gpu-optimization-plan.md`, the current full chr22 binary profile spends about `14.94s` in native delivery, `6.26s` in JAX compute, and `1.77s` in H2D transfer. In the older linear profile, JAX compute is only `0.071s` while host BGEN read is `9.216s`.
3. **You are only analyzing one phenotype at a time.** `src/g/api.py` takes `pheno_name: str`; REGENIE supports multiple phenotype columns/lists and writes output per phenotype. That matters because a GPU app wants to decode/transfer a genotype chunk once and multiply it against many phenotypes. REGENIE’s docs expose `--phenoCol`, `--phenoColList`, and multi-phenotype output behavior. ([RGC GitHub][1])
4. **There are concrete hot-path inefficiencies:** unconditional null-Firth work, fixed 50-iteration null logistic fitting, second-pass Rust genotype summarization, GPU rereads of genotype matrices, per-chunk device-to-host materialization, writer cloning, and not using direct variant-major score kernels in score-only mode.

Below is the optimization plan I would use.

---

## Diagnosis: why GPU is barely faster

### 1. For single-trait linear REGENIE, the GPU has too little work

Your linear kernel is already tiny:

```python
# src/g/compute/regenie2_linear.py:139-149
score_matrix = chromosome_state.stacked_score_matrix @ genotype_matrix
genotype_sum_squares = jnp.einsum("ij,ij->j", genotype_matrix, genotype_matrix)
projection_sum_squares = jnp.einsum("ij,ij->j", covariate_projection_coordinates, covariate_projection_coordinates)
```

That is basically:

```text
small K x N matrix  @  N x C genotype chunk
plus a few reductions
```

For one phenotype, this is memory bandwidth dominated. The GPU sees a chunk, does a small amount of math, and then the result has to come back to host. Your own linear doc says compute was only `0.39%` of wall in one profiled run, with host BGEN work dominating.

So for single-trait linear, a GPU may not beat REGENIE much unless you:

```text
decode once → transfer once → test many phenotypes
```

That should become a major product goal.

### 2. For binary, Firth is real GPU work, but your default may be doing too much

Current code:

```python
# src/g/api.py:47-51
@dataclasses.dataclass(frozen=True)
class Regenie2BinaryConfig:
    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE
```

and:

```python
# src/g/engine/regenie2_pipeline.py:1251-1252
trusted_no_missing_diploid: bool = False,
correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
```

That means a default binary run does approximate Firth-style fallback. REGENIE’s default binary Step 2 does **not** automatically mean approximate Firth; Firth/SPA are user flags. If you compare:

```bash
regenie --bt
```

against current:

```bash
g regenie2 --trait-type binary
```

then `g` may be doing much more statistical work.

Your repo profile says chr22 binary had `17,938` Firth candidates and JAX compute around `4.9–6.3s` depending on run. If REGENIE was run score-only, this is not a fair comparison.

### 3. Current pipeline still serializes around chunks more than ideal

JAX dispatch is asynchronous, but host inspection/materialization forces synchronization. JAX’s own docs note that Python can “run ahead” only until it needs to inspect/copy values back to host. ([docs.jax.dev][2])

Your callback does this per chunk:

```python
# src/g/engine/regenie2_pipeline.py:907-916
genotype_device_array = put_genotype_matrix_on_device(...)
result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(...)
block_until_ready(result.log10_p_value)
```

then:

```python
# src/g/engine/regenie2_pipeline.py:273-295
host_values = jax.device_get({...})
writer_session.write_regenie2_native_chunk(...)
```

That makes each chunk:

```text
H2D → JAX compute → wait → D2H → writer enqueue
```

You do have a decode/compute queue, but the GPU worker itself is not scheduling several chunks ahead and letting output materialization happen behind it.

### 4. Current score kernels reread genotype data

Linear:

```python
score_matrix = stacked_score_matrix @ genotype_matrix
genotype_sum_squares = einsum(genotype_matrix, genotype_matrix)
```

Binary:

```python
weighted_genotype_matrix = sqrtW[:, None] * genotype_matrix
projection_coordinates = P @ weighted_genotype_matrix
weighted_genotype_sum_squares = einsum(weighted_genotype_matrix, weighted_genotype_matrix)
score = genotype_matrix.T @ score_residual
```

These are mathematically clean, but they read/materialize the genotype matrix multiple times. For memory-bound kernels, that matters. Some of this can be replaced with native stats or fused variant-major kernels.

### 5. Rust decode does extra full scans

The trusted variant-major path decodes into `(variant, sample)` but then scans the whole matrix again:

```rust
// src/genotype/bgen.rs:618-620
let output_slice = unsafe { std::slice::from_raw_parts(...) };
preprocess::summarize_variant_major_dosage_matrix(...)
```

The row-major path also decodes and then preprocesses/scans:

```rust
// src/genotype/bgen.rs:548-558
self.read_dosage_f32_into_address_with_selection(...)?;
preprocess::preprocess_row_major_dosage_matrix(...)
```

And `preprocess_row_major_dosage_matrix` loops across all samples and variants, then loops again if missing values exist:

```rust
// src/genotype/preprocess.rs:39-60
for sample_index in 0..selected_sample_count {
    for variant_index in 0..selected_variant_count {
        ...
    }
}

// src/genotype/preprocess.rs:75-87
if has_missing_values {
    for sample_index in 0..selected_sample_count {
        for variant_index in 0..selected_variant_count {
            ...
        }
    }
}
```

For trusted no-missing BGEN, stats should be fused into decode. A second full scan is avoidable.

---

# Prioritized optimization plan

## P0 — Establish fair benchmarking before changing kernels

### Step 1: Compare equivalent statistical modes

Create four mandatory benchmark modes:

```text
A. REGENIE score-only binary
   regenie --bt

B. g score-only binary
   g regenie2 --trait-type binary   # after changing default to score-only

C. REGENIE approximate Firth
   regenie --bt --firth --approx --pThresh 0.01

D. g approximate Firth
   g regenie2 --trait-type binary --firth --approx --pThresh 0.01
```

Do not compare current default `g` binary against REGENIE score-only.

Implementation:

```python
@dataclass(frozen=True)
class Regenie2BinaryConfig:
    firth: bool = False
    approx: bool = False
    spa: bool = False
    p_threshold: float = 0.05
    firth_se: bool = False
```

Then normalize internally to:

```python
SCORE_ONLY
FIRTH_APPROXIMATE
FIRTH_EXACT
SPA
```

For now:

```text
--firth --approx  -> current approximate Firth implementation
--firth           -> error unless exact Firth is actually implemented
--spa             -> error unless real SPA is implemented
no flags          -> score-only
```

This is both a correctness fix and a performance fix.

### Step 2: Benchmark with matched output behavior

Your API default has:

```python
# src/g/api.py:35
finalize_parquet: bool = True
```

but CLI commands default `finalize_parquet=False`. Make sure benchmarks compare the same thing:

```text
no final parquet vs no final parquet
compressed output vs compressed output
same variant filters
same pThresh
same warm/cold cache policy
same trusted validation policy
same phenotype count
```

### Step 3: Add a standard benchmark report

Every benchmark should output:

```text
wall_time
variants_per_second
samples x variants processed per second
native_engine_delivery
bgen_decode_time
host_preprocess_time
host_to_device_transfer
jax_compute
device_to_host_materialization
output_write
writer_finish
firth_candidate_count
firth_failure_count
max_firth_candidates_per_chunk
GPU utilization
GPU memory bandwidth
CPU utilization
```

Your `StageTimingRecorder` is already close. Add derived metrics.

---

## P1 — Fix likely “dumb mistakes” first

### 1. Make binary score-only truly score-only

Right now `build_extra_code` always selects Firth/SPA candidates based on a hard-coded chi-square threshold:

```python
# src/g/compute/regenie2_binary.py:20
REGENIE_SCORE_CHISQ_THRESHOLD = 3.841458820694124

# src/g/compute/regenie2_binary.py:271-276
candidate_mask = chi_squared >= REGENIE_SCORE_CHISQ_THRESHOLD
correction_code = EXTRA_CODE_SPA if correction == SPA else EXTRA_CODE_FIRTH
```

Change this so score-only mode does not build correction labels or candidate masks:

```python
if plan.method == SCORE_ONLY:
    extra_code = zeros_or_null_failure_only(...)
else:
    candidate_mask = log10_p_value > -log10(plan.p_threshold)
```

Expected impact: potentially huge for binary comparisons, because default runs stop paying for Firth fallback.

### 2. Do not compute null Firth state unless Firth is enabled

Current binary chromosome-state preparation always does this:

```python
# src/g/compute/regenie2_binary.py:342-347
null_firth_penalized_log_likelihood = fit_covariate_only_firth_null_model(...)
```

That should not happen for:

```text
score-only
SPA
exact/non-Firth modes
chunks with zero Firth candidates, if you can delay it
```

Better structure:

```python
prepare_binary_score_state(...)
prepare_binary_firth_state(...)   # only if --firth
```

For approximate Firth, you can either prepare null Firth once per chromosome when Firth is enabled, or lazily compute it only after the first chunk on that chromosome has candidates. The latter saves work for chromosomes with no candidates but makes orchestration more complex.

### 3. Stop fitting null logistic with a fixed 50 iterations

Current null logistic fitting:

```python
# src/g/compute/regenie2_binary.py:280-307
return jax.lax.fori_loop(0, maximum_iterations, update_coefficients, initial_coefficients)
```

This always runs `DEFAULT_MAXIMUM_NULL_ITERATIONS = 50`.

REGENIE docs list `--niter` default as `30` for logistic regression. ([RGC GitHub][1]) More importantly, you should stop when converged.

Use a `while_loop` with:

```text
max_abs_delta < tolerance
or relative log-likelihood improvement < tolerance
or iteration == max_iterations
```

Add diagnostics:

```text
null_logistic_iterations_by_chromosome
null_logistic_converged
max_delta
```

Expected impact: moderate, especially for repeated chromosomes and warm-cache runs. It is also cleaner statistically.

### 4. Make `pThresh` real

Replace the fixed threshold:

```python
3.841458820694124
```

with the REGENIE flag:

```python
candidate_mask = log10_p_value > -math.log10(p_threshold)
```

This matters because common biobank commands use:

```bash
--pThresh 0.01
```

not always `0.05`. REGENIE documents `--pThresh` as the p-value threshold for Firth/SPA fallback with default `0.05`. ([RGC GitHub][1])

Lowering from `0.05` to `0.01` can cut Firth candidates substantially.

---

## P2 — Exploit variant-major score-only paths

You already have a direct variant-major binary score function:

```python
# src/g/compute/regenie2_binary.py:402-447
compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(...)
```

But production trusted binary currently does:

```python
# src/g/engine/regenie2_pipeline.py:946-953
genotype_device_array = put_genotype_matrix_on_device(genotype_matrix_by_variant, ...)
result = compute_regenie2_binary_chunk_from_chromosome_state(
    genotype_matrix=jnp.transpose(genotype_device_array),
    ...
)
```

That transpose path was reasonable for Firth parity. Your docs say direct variant-major Firth failed full chr22 parity, while variant-major native decode plus sample-major JAX preserved parity.

But for **score-only binary**, Firth parity is irrelevant. You should branch:

```python
if correction_plan.method == SCORE_ONLY and genotype_is_variant_major:
    result = compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major(...)
else:
    result = existing parity-preserving path
```

Expected impact: moderate. It avoids the transpose/sample-major fallback and uses a layout that is natural for variant-wise score testing.

Do the same for linear if trusted variant-major decode is available.

Add a linear variant-major kernel:

```python
@jax.jit
def compute_linear_variant_major(chromosome_state, genotype_matrix_by_variant, genotype_sum_squares):
    score_matrix_by_variant = genotype_matrix_by_variant @ chromosome_state.stacked_score_matrix.T
    covariate_projection_coordinates = score_matrix_by_variant[:, :-1]
    covariance_with_phenotype = score_matrix_by_variant[:, -1]
    projection_sum_squares = jnp.einsum("ij,ij->i", covariate_projection_coordinates, covariate_projection_coordinates)
    genotype_residual_sum_squares = genotype_sum_squares - projection_sum_squares
    ...
```

This should pair with native stats so the GPU does not have to compute `G^2` again.

---

## P3 — Fuse native decode and statistics

This is probably the highest ROI Rust optimization.

### Current situation

Trusted variant-major:

```rust
decode trusted variant-major → scan whole matrix to summarize
```

Row-major:

```rust
decode → preprocess scan → optional imputation scan
```

### Target

For trusted no-missing diploid BGEN:

```text
decode variant → write dosage row → accumulate stats for that variant
```

In the same loop, compute:

```text
dosage_sum
dosage_square_sum
observation_count
zero_count
nonzero_count
homozygous_ref_count
heterozygous_count
homozygous_alt_count
minor_allele_count
info_score inputs
sparse flags
```

Then `ChunkStats` can be built without scanning the output matrix.

Implementation target:

```rust
decode_trusted_unphased_eight_bit_variant_into_variant_major_matrix(...)
```

currently writes contiguous variant rows. Add local accumulators there and return per-variant summaries from the tile.

Expected impact: likely meaningful. Your binary docs already show native delivery is around `6.9s` even in hot trusted runs, and variant-major native decode halved profiled native output bytes from `8.39GB` to `4.20GB` while only slightly improving end-to-end time. Removing the second scan attacks remaining CPU memory traffic.

For untrusted/missing data, parallelize `preprocess_row_major_dosage_matrix` or add a variant-major safe path. The current preprocessing loops are single-threaded.

---

## P4 — Avoid GPU rereads using native stats

### Linear

This line is a full pass over genotype on GPU:

```python
# src/g/compute/regenie2_linear.py:143
genotype_sum_squares = jnp.einsum("ij,ij->j", genotype_matrix, genotype_matrix)
```

For trusted no-missing decode, Rust can provide `sum_g2` exactly. For imputed missing values, you need the post-imputation sum of squares, not merely the observed non-missing sum of squares.

Add to `ChunkStats`:

```rust
imputed_dosage_square_sum: Vec<f32>
```

Then linear becomes:

```python
genotype_sum_squares = jax.device_put(chunk_stats.imputed_dosage_square_sum)
```

This removes one full device-side genotype pass.

### Binary

Binary needs weighted sums:

```text
sum_i W_i g_i^2
```

Because `W_i` changes by chromosome/phenotype, Rust cannot fully precompute it unless you move binary score computation into Rust or pass W into native code.

But you can still optimize score-only variant-major with a fused custom kernel later:

```text
for variant row:
    compute score = dot(g, residual)
    compute weighted_g2 = dot(g*g, W)
    compute projection terms
    emit beta/se/chisq/log10p
```

JAX/XLA may already fuse some of this, so validate with profiler before writing custom kernels.

---

## P5 — Pipeline overlap: separate decode, GPU submit, materialization, writer

Current callback has one worker consuming chunks and doing:

```text
device_put
compute
block
device_get
writer enqueue
release host buffer
```

A better pipeline:

```text
Rust decode thread(s)
    ↓
host buffer queue
    ↓
GPU submit thread
    - device_put
    - launch JAX compute
    - release host buffer after H2D is safe
    - enqueue result future
    ↓
result materializer thread
    - copy_to_host_async / device_get
    - writer enqueue
    ↓
Rust writer workers
```

This aims to overlap:

```text
CPU BGEN decode of chunk k+1
GPU compute of chunk k
D2H/output write of chunk k-1
```

Caveat: do not reuse a NumPy host buffer until the H2D copy is definitely complete. Your current conservative release-after-write avoids data races but limits overlap. A safe intermediate version is:

```text
device_put
block_until_ready(genotype_device_array)   # H2D done
release host buffer
compute
```

That may still improve CPU decode overlap if buffer reuse is blocking native delivery.

Benchmark `staging_depth` separately:

```text
1, 2, 3, 4
```

Record:

```text
callback_queue_producer_blocking
native_engine_delivery
host_to_device_transfer
jax_compute
output_write
peak host memory
peak GPU memory
```

---

## P6 — Output writer tuning

Current writer clones every chunk into a job:

```rust
// src/output/writer.rs:197-214
chrom: metadata.chromosome.clone(),
genpos: metadata.position.clone(),
id: metadata.variant_identifier.clone(),
...
beta: beta.to_vec(),
se: standard_error.to_vec(),
chisq: chi_squared.to_vec(),
log10p: log10_p_value.to_vec(),
extra_code: extra_code.map(<[i32]>::to_vec),
```

Then it batches only 4 chunks per Arrow file:

```rust
// src/output/writer.rs:26-27
const REGENIE_STEP2_CHUNKS_PER_ARROW_FILE: usize = 4;
```

and uses ZSTD IPC compression:

```rust
// src/output/writer.rs:512-514
IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))
```

Optimize in this order:

1. Sweep chunks per Arrow file:

```text
4, 8, 16, 32
```

2. Benchmark Arrow compression modes:

```text
ZSTD
LZ4 if available
none for intermediate chunks
```

3. Avoid cloning metadata per chunk. Since the BGEN engine and writer are both Rust, make metadata `Arc<VariantMetadataColumns>` and move it through a Rust-owned chunk descriptor.

4. If final Parquet is the real output, consider writing Parquet row groups directly instead of Arrow IPC chunks plus final compaction.

Expected impact: small to moderate, but it reduces CPU pressure and makes overlap easier.

---

# Major structural optimization: multi-phenotype batching

This is the biggest strategic improvement if you want a GPU win.

Right now the public API is single phenotype:

```python
# src/g/api.py:127-129
pheno: Path | str,
pheno_name: str,
out: Path | str,
```

REGENIE supports selecting multiple phenotypes and writing results for multiple phenotypes. ([RGC GitHub][1])

For GPU, single phenotype is a poor workload. Multi-phenotype changes the arithmetic intensity.

## Linear multi-phenotype kernel

For `T` phenotypes:

```text
G: N x C genotype chunk
R: T x N adjusted residual matrix
P: K x N whitened covariate matrix
```

Compute:

```python
covariate_projection = P @ G       # K x C, shared across traits
covariance = R @ G                 # T x C
genotype_resid_ss = sum_g2 - sum(covariate_projection ** 2, axis=0)
beta = covariance / genotype_resid_ss[None, :]
```

The genotype chunk is decoded/transferred once, then reused across `T` traits.

This is where GPU can beat REGENIE more convincingly.

Implementation plan:

1. Add API:

```python
pheno_names: tuple[str, ...]
```

or REGENIE-like:

```text
--pheno-name
--pheno-name-list
```

2. Group phenotypes by identical aligned sample set and covariate design.

3. Load LOCO predictions into:

```python
loco_predictions: dict[chromosome, Float32[T, N]]
```

4. Compute per-chromosome residual matrix:

```python
adjusted_residuals: Float32[T, N]
rss: Float32[T]
```

5. Output either split by phenotype or add a phenotype column.

This should be a priority if the product goal is “faster than REGENIE,” not just “GPU version of one-trait REGENIE.”

## Binary multi-phenotype batching

Harder, but still possible.

For score-only binary:

```text
R: N x T residuals
W: N x T Bernoulli weights
Gv: C x N variant-major genotype
```

You can compute:

```python
score = Gv @ R              # C x T
weighted_g2 = (Gv * Gv) @ W # C x T
```

Projection terms require trait-specific weighted covariate projection matrices, so this becomes a batched GEMM problem.

Start with:

```text
multi-phenotype score-only binary
```

Then add approximate Firth batching later. Do not start with multi-trait Firth; it is the hardest part.

---

# Firth optimization plan

Your docs already show the core issue:

```text
batch size 64:  hot wall 7.471s, JAX compute 4.938s
batch size 512: hot wall 5.960s, JAX compute 3.431s
```

but larger batches changed thousands of outputs and were rejected. Similarly, block-math was faster but changed output classifications.

That means the next Firth optimization is **not** “try random bigger batches.” It is:

```text
make Firth numerics invariant across equivalent batch/layout choices
```

## Step 1: Build a Firth parity harness

For each candidate variant, record:

```text
variant id
score-test p
initial coefficients
heuristic-init flag
iteration count
converged flag
failure code
penalized log-likelihood
beta
se
chisq
log10p
EXTRA
```

Run the same candidate through:

```text
batch size 32
batch size 64
batch size 128
batch size 256
block math off
block math on
variant-major direct
sample-major fallback
```

This will tell you whether differences come from:

```text
candidate ordering
initialization
convergence threshold
step halving
float32 accumulation
Cholesky jitter
invalid-stat classification
batch padding/masking
```

## Step 2: Make the solver deterministic per variant

Rules:

1. Fixed maximum iterations.
2. Per-lane active mask, but identical math for active lanes regardless of batch size.
3. Same initialization path for the same variant.
4. Same step-halving policy.
5. Same convergence criterion.
6. Same invalid-stat fallback.
7. Prefer `float64` only for the tiny dense Firth solve if it fixes parity at acceptable cost.

This may allow you to accept the faster batch sizes and block formulation.

## Step 3: Only after parity, promote faster variants

Promote in this order:

```text
larger Firth batch size
block-math Firth
direct variant-major Firth
custom Pallas/Triton Firth kernel
```

The measured upside is real: your docs show ~1.5s saved by larger batch size and ~0.8s saved by block math on the profiled run. But they are not acceptable until parity is stable.

---

# JAX formula-level opportunities

## Linear

Current formula is reasonable. The two improvements are:

1. Use native `sum_g2` instead of GPU `einsum`.
2. Add variant-major kernel to avoid layout conversion and improve memory access for trusted BGEN.

Do not spend much time micro-optimizing linear JAX until decode/transfer/multi-phenotype are addressed.

## Binary score test

Current sample-major score path materializes:

```python
weighted_genotype_matrix = sqrtW[:, None] * genotype_matrix
```

Then uses it twice. In variant-major layout, a custom fused kernel could do all per-variant reductions in one pass:

```text
score
weighted g²
weighted covariate cross-products
candidate flag
```

But before custom kernels, inspect HLO/profiler output. XLA may already fuse some elementwise work, but the reductions/GEMMs are likely separate.

Action:

```python
jax.named_call(...)
```

around:

```text
weighted_genotype construction
projection_coordinates
weighted_genotype_sum_squares
score
p-value conversion
Firth correction
```

Then run JAX profiler/Nsight Systems.

## P-value calculation

Every variant computes:

```python
# src/g/compute/regenie2_linear.py:105-107
log_p_value = jnp.log(2.0) + jax.scipy.stats.norm.logsf(jnp.sqrt(safe_chi_squared))
```

This is accurate and stable, but special functions can be nontrivial. Check if it appears as a bottleneck. If yes:

1. Keep exact/stable path for final output.
2. Use cheaper thresholding for candidate selection where possible.
3. Consider computing full `LOG10P` on CPU only if D2H/output dominates less than GPU special-function cost. This is unlikely to be the first win, but worth profiling.

---

# Rust/native optimization plan

## 1. Fuse trusted variant-major stats into decode

Highest ROI native change.

Current:

```text
decode → second full matrix scan
```

Target:

```text
decode and summarize in the same variant loop
```

## 2. Add post-imputation square sum to `ChunkStats`

Needed for removing GPU `G²` pass in linear.

Current `ChunkStats` stores:

```rust
dosage_sum
dosage_variance_numerator
allele_count
...
```

Add:

```rust
dosage_square_sum
imputed_dosage_square_sum
```

Expose through `_core.pyi`.

## 3. Parallelize untrusted preprocessing

`preprocess_row_major_dosage_matrix` is single-threaded. Use Rayon by variant blocks, or switch untrusted decode to variant-major tile summaries.

## 4. Reduce Python/Rust round trips

Current native delivery does:

```text
Rust decode
→ Python callback object
→ JAX compute
→ Python calls Rust writer
```

Longer-term, create a Rust-owned `ChunkHandle`:

```rust
struct NativeChunkHandle {
    metadata: Arc<VariantMetadataColumns>,
    stats: Arc<ChunkStats>,
    buffer_id: usize,
}
```

Python receives the NumPy/JAX buffer and handle. After compute, Python passes only result arrays plus handle back to Rust. Avoid cloning metadata.

---

# Should you move to Triton, Pallas, CUDA, or Burn?

## My recommendation

Do **not** abandon JAX yet.

Use this order:

```text
1. Fix REGENIE-equivalent modes.
2. Add score-only fast paths.
3. Fuse Rust decode+stats.
4. Add multi-phenotype batching.
5. Make Firth numerics invariant.
6. Only then consider custom kernels.
```

## When JAX is enough

JAX is fine for:

```text
linear score scans
multi-phenotype GEMMs
binary score-only batched matrix operations
general orchestration of dense linear algebra
```

Especially once you batch many phenotypes, the workload becomes closer to large GEMMs, which JAX/XLA should handle well.

## When Pallas is appropriate

Pallas is a JAX kernel language for writing custom GPU/TPU kernels with finer control while staying in the JAX ecosystem. ([docs.jax.dev][3])

Use Pallas if profiling proves a JAX-generated binary score kernel is doing avoidable memory passes and you want to fuse:

```text
load genotype row
compute score
compute weighted g²
compute projection partials
emit candidate flag
```

Pallas is attractive because it keeps you inside JAX and avoids a full CUDA extension.

## When Triton is appropriate

Triton is a language and compiler for parallel programming aimed at high-throughput GPU kernels. ([Triton Language][4])

Use Triton if:

```text
Pallas cannot express the kernel well
you need more manual control over blocks/shared memory
you want a standalone custom GPU score/Firth kernel
```

Best Triton target:

```text
variant-major binary score-only fused kernel
```

Maybe later:

```text
batched Firth candidate kernel
```

Do not use Triton to fix BGEN decode/output; those are not GPU-kernel problems.

## When CUDA/C++ custom calls are appropriate

JAX has an FFI path for calling external native kernels. ([docs.jax.dev][5]) OpenXLA also documents custom calls/FFI as a way to register external operations with XLA. ([OpenXLA Project][6])

Use CUDA/FFI only if:

```text
you have a proven bottleneck kernel
Pallas/Triton cannot reach required performance
you are willing to maintain CUDA build/distribution complexity
```

CUDA is most justified for:

```text
fused binary score kernel
specialized Firth solver
possibly GPU decompression only if you radically redesign BGEN ingestion
```

## Burn / Rust tensor stack

Burn is a Rust deep learning framework/tensor library. ([Burn][7]) It may be interesting, but I would not move this app to Burn for the current problem. Your bottlenecks are BGEN streaming, statistical kernels, data layout, and output. Burn does not obviously solve those better than your current Rust + JAX split.

---

# Concrete implementation sequence

## Sprint 1: benchmark correctness and cheap wins

1. Add REGENIE-style binary flags.
2. Change binary default to score-only.
3. Implement `pThresh`.
4. Make `--spa` error until real SPA exists.
5. Make `--firth` without `--approx` error unless exact Firth exists.
6. Skip null-Firth state unless approximate Firth is enabled.
7. Replace fixed 50 null-logistic iterations with convergence-based loop.
8. Benchmark score-only REGENIE vs score-only `g`.

Success metric:

```text
g score-only binary should be much faster than current default binary if Firth candidates were the issue.
```

## Sprint 2: layout and native stats

1. Use direct variant-major binary score kernel for score-only trusted runs.
2. Add linear variant-major trusted kernel.
3. Add `imputed_dosage_square_sum` to `ChunkStats`.
4. Remove linear GPU `genotype_sum_squares` pass.
5. Fuse trusted variant-major decode stats into Rust decode.
6. Benchmark chunk size and staging depth.

Success metric:

```text
native_engine_delivery down
host_to_device_transfer same or lower
jax_compute lower for linear/score-only
```

## Sprint 3: overlap pipeline

1. Split callback into decode queue, GPU submit queue, materialization/write queue.
2. Release genotype host buffers after H2D copy is safe, not after output write.
3. Add in-flight result limit.
4. Add writer compression/batch-size sweeps.

Success metric:

```text
GPU idle gaps shrink in Nsight/JAX profiler
callback_queue_producer_blocking decreases
native decode overlaps JAX compute
```

## Sprint 4: multi-phenotype linear

1. Add `pheno_names`.
2. Group traits by compatible sample/covariate alignment.
3. Load LOCO predictions for multiple traits.
4. Implement `T x N @ N x C` residual GEMM.
5. Add output schema support for phenotype dimension.
6. Benchmark `T = 1, 2, 4, 8, 16, 32`.

Success metric:

```text
per-phenotype wall time drops as T increases
GPU utilization rises materially
BGEN decode cost amortizes across phenotypes
```

This is likely the biggest path to a convincing GPU advantage.

## Sprint 5: Firth parity and faster Firth

1. Build Firth parity harness.
2. Diagnose why batch size changes output.
3. Stabilize per-variant numerical path.
4. Promote larger batch size if parity holds.
5. Promote block math if parity holds.
6. Revisit direct variant-major Firth.
7. Only then consider Pallas/Triton/CUDA Firth kernels.

Success metric:

```text
batch size 128/256/512 gives same output as batch 64
JAX compute drops without EXTRA mismatches
```

---

# Expected impact ranking

| Opportunity                                    |  Expected impact |        Risk | Why                                                 |
| ---------------------------------------------- | ---------------: | ----------: | --------------------------------------------------- |
| REGENIE-equivalent binary default / score-only |        Very high |         Low | Current default likely does extra Firth work        |
| Multi-phenotype batching                       |        Very high | Medium/high | Best way to make GPU useful                         |
| Skip null-Firth unless needed                  |      Medium/high |         Low | Removes unconditional expensive binary work         |
| Fuse Rust decode + stats                       |      Medium/high |      Medium | Removes full host matrix scans                      |
| Score-only variant-major path                  |           Medium |  Low/medium | Avoids transpose/sample-major fallback              |
| Pipeline overlap                               |           Medium |      Medium | Reduces idle gaps and blocked stages                |
| Writer batching/compression tuning             |       Low/medium |         Low | Useful if output is visible in timings              |
| Larger Firth batches/block math                | High but blocked |        High | Already measured faster, currently not parity-safe  |
| Triton/Pallas/CUDA                             | Potentially high |        High | Only worth it after profiling identifies one kernel |

---

# Bottom line

The first thing I would fix is not Triton or CUDA. It is this:

```text
Make g run the same statistical workload as REGENIE.
```

Then I would optimize:

```text
score-only binary fast path
lazy Firth state
decode+stats fusion
multi-phenotype batching
pipeline overlap
Firth numerical invariance
```

Only after those are done would I move to Pallas/Triton/CUDA. For single-trait linear REGENIE Step 2, your current stack may already be near the practical speed limit because the GPU does not get enough arithmetic per decoded byte. For multi-phenotype scans and binary Firth-heavy workloads, the GPU can still win—but the architecture needs to reuse each decoded genotype chunk across much more work.

[1]: https://rgcgithub.github.io/regenie/options/ "Documentation - regenie"
[2]: https://docs.jax.dev/en/latest/async_dispatch.html?utm_source=chatgpt.com "Asynchronous dispatch"
[3]: https://docs.jax.dev/en/latest/pallas?utm_source=chatgpt.com "Pallas: a JAX kernel language"
[4]: https://triton-lang.org/?utm_source=chatgpt.com "Welcome to Triton's documentation! — Triton documentation"
[5]: https://docs.jax.dev/en/latest/ffi.html?utm_source=chatgpt.com "Foreign function interface (FFI)"
[6]: https://openxla.org/xla/custom_call?utm_source=chatgpt.com "XLA Custom Calls"
[7]: https://burn.dev/docs/burn/?utm_source=chatgpt.com "burn - Rust"
