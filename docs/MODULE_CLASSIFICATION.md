# Module Classification

This repository has no released backwards-compatibility contract. Keep production modules in one of the categories below and delete compatibility/deprecated modules instead of preserving alternate behavior paths.

## Public API

- `src/g/__init__.py`
- `src/g/api.py`
- `src/g/cli.py`
- `src/g/types.py`

## Active Internal Pipeline

- `src/g/_core.pyi`
- `src/g/compute/__init__.py`
- `src/g/compute/regenie2_binary.py`
- `src/g/compute/regenie2_binary_types.py`
- `src/g/compute/regenie2_linear.py`
- `src/g/compute/regenie2_linear_types.py`
- `src/g/engine/__init__.py`
- `src/g/engine/regenie2_pipeline.py`
- `src/g/io/__init__.py`
- `src/g/io/output.py`
- `src/g/io/source.py`
- `src/g/jax_setup.py`
- `src/python/mod.rs`
- `src/python/output.rs`
- `src/genotype/**`
- `src/output/**`
- `src/regenie/**`
- `src/lib.rs`

## Compatibility/Test-Only

- `tests/**`
- `scripts/**`
- `docs/**`

## Deprecated

No deprecated production modules are retained. The previous Python reader, Python BGEN wrapper, Python genotype preprocessing, Python sample alignment, and Python chunk compatibility modules were removed instead of classified as supported compatibility code.
