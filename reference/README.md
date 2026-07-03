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

The project Justfile can also build this tree with the experimental Rust-backed
BGEN reader:

```bash
GWAS_ENGINE_REGENIE_BGEN_PATH=/path/to/bgen \
GWAS_ENGINE_REGENIE_USE_G_BGEN_READER=1 \
just data-build-patched-regenie
```

That mode links `crates/bgen-capi` into patched REGENIE through
`USE_G_BGEN_READER=1`. It is intended for Step 2 BGEN reader experiments and
benchmark evidence; it currently falls back or fails for unsupported Step 2
options such as `--minINFO`, set/mask, interaction, and correlation workflows.
