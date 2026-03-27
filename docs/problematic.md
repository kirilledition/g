# Problematic Areas

## Native BED wrapper dtype mismatch

- `src/g/io/plink.py:135` decodes `native_chunk.genotype_values_le` with `dtype=jax_setup.HOST_NUMPY_DTYPE`.
- `src/g/io/plink.py:184` decodes `native_result.imputed_genotype_values` with `dtype=jax_setup.HOST_NUMPY_DTYPE`.
- `src/lib.rs` exposes those native buffers as `f64` values.
- On the current `main`, `jax_setup.HOST_NUMPY_DTYPE` is `np.float32`, so the wrapper reinterprets `float64` bytes as `float32` and then fails to reshape native chunk buffers.

Observed impact:

- `tests/test_phase1.py::test_native_genotype_chunk_reader_matches_python_path` fails with `ValueError: cannot reshape array of size 80128 into shape (2504,16)`.
- `tests/test_phase1.py::test_native_genotype_chunk_preprocessing_matches_python_path` fails with the same reshape error.

Suggested follow-up:

- Decode Rust-native float buffers as `np.float64` explicitly before any cast to runtime or solver dtype.
- Re-run the native Phase 1 parity tests after the wrapper is corrected.
