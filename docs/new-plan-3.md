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