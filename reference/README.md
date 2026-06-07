# External References

This directory contains external reference implementations used for parity
debugging and algorithm inspection. These files are not active application code
for `g`, are not imported by the Python package, and are not linked by the Rust
native extension.

Contents:

- `regenie-patched/`: patched REGENIE C++ source used as the local original
  REGENIE reference for parity investigation.

The patched REGENIE source expects an external BGEN library through `BGEN_PATH`
when built directly. The previously tracked upstream BGEN release snapshot was
removed from active `main`; the full old archive remains recoverable from
branch `preserve-direct-association-g-code-20260607` and tag
`archive-direct-association-g-code-20260607`.
