What I found
- The code is currently built around float64 end to end: jax_enable_x64 is on in src/g/jax_setup.py:27, and PLINK chunks are decoded as float64 in src/g/io/plink.py:43 and src/g/io/plink.py:97.
- The main hot path is logistic, especially src/g/compute/logistic.py:1243 and src/g/compute/logistic.py:1379.
- You already have useful benchmarks/profilers:
  - just benchmark-jax, just benchmark-jax-chunks, just benchmark-logistic-loop, just benchmark-plink-reader
  - just profile-logistic-chr22, just profile-logistic-detailed
- Parity coverage against PLINK2 already exists in tests/test_phase1.py:263 and tests/test_phase1.py:317, with full-run comparison in scripts/evaluate_phase1.py.
Highest-value plan
- Baseline first
  - Run the existing JAX benchmarks and full logistic trace on the target GPU.
  - Use jax.profiler.trace(..., create_perfetto_trace=True) output from scripts/profile_full_chr22_jax.py:62; per JAX docs, view it with xprof --port 8791 <trace_dir>.
  - Capture a memory profile too; the repo already uses jax.profiler.save_device_memory_profile(...) in scripts/profile_full_chr22_jax.py:90.
- Fix obvious structural bottlenecks before touching precision
  - No-missing logistic currently computes Firth over the whole chunk if any variant falls back, then masks skipped lanes in-kernel: src/g/compute/logistic.py:1319. That is likely the single biggest waste on GPU.
  - Missing-data logistic still does host syncs and Python-side fallback batching via jax.device_get(...) and loops in src/g/compute/logistic.py:1427 to src/g/compute/logistic.py:1501.
  - PLINK ingestion likely matters too: CPU decode + per-chunk device_put, plus syncs like bool(jax.device_get(jnp.any(missing_mask))) in src/g/io/plink.py:163.
- Then tune kernel math with low-risk rewrites
  - Replace avoidable inverses like jnp.linalg.inv in src/g/compute/linear.py:28 with solve/factorization-based forms.
  - Audit repeated tiny solves/einsums in logistic IRLS (src/g/compute/logistic.py:667, src/g/compute/logistic.py:828, src/g/compute/logistic.py:1037) and prefer reused factorizations or dedicated linear-algebra paths where they preserve math.
  - Keep result semantics identical first; use benchmarks to confirm speedup before changing numerics.
- Treat mixed precision as a separate experiment, not the default path
  - JAX docs say default_matmul_precision / lax.Precision mainly affect 32-bit matmul/conv behavior on GPU; they do not change input/output dtypes by themselves.
  - Since this repo is explicitly x64 today, the safe order is:
    1. get profiling data,
    2. land structural speedups in current precision,
    3. only then try fp32/bfloat16 experiments behind a switch.
  - I would not start with float16; for this workload, float32 is the realistic first candidate, with bfloat16 only for carefully isolated operations if the profiler shows matmul-heavy kernels and parity survives.
Validation plan
- Fast checks after each change:
  - tests/test_phase1.py:263
  - tests/test_phase1.py:317
  - targeted compute tests in tests/test_compute_logistic.py
- End-to-end checks after each batch:
  - just benchmark-jax
  - just benchmark-logistic-loop
  - just benchmark-plink-reader
  - just phase1-evaluate
- Success criterion:
  - no regression in PLINK2 parity summary from scripts/evaluate_phase1.py
  - measurable runtime reduction on the target GPU, especially logistic full-loop throughput
Recommended execution order
- 1) Profile current logistic full run on GPU
- 2) Remove host sync / Python fallback orchestration
- 3) Restrict Firth work to true fallback variants only
- 4) Rework solver/factorization hotspots without changing dtype
- 5) Only then run controlled fp32/bfloat16 experiments behind flags
