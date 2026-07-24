# Public API

## This crate owns

Frontend configuration normalization for canonical TOML plus the native CLI
dispatch frontend invoked by the installed Python console bootstrap. The CLI
accepts `--config` and the supported REGENIE step 2 flags; native tuning is
TOML-only. The optional `[input].bgen_content_sha256` selector is also
TOML-only and accepts exactly 64 lowercase hexadecimal characters. There is no
CLI digest flag. A CLI `--bgen` override replaces only the BGEN locator and
preserves a selector supplied by TOML.

## Public types

CLI dispatch payloads consumed by `g-runner`. The resolved BGEN locator and
optional canonical content selector are compiled into the immutable
`g-plan::InputPlan`.

## Public functions

Dispatch the native CLI frontend. TOML serialization, validation, and plan
compilation are implementation details.

## This crate must not expose

Runtime setup, BGEN opening, sample alignment, output writing, JAX backend choices after planning, or PyO3 lifecycle state.

## Performance constraints

Keep work at config/build time only. Do not perform data-file scans or
compute-side effects, and do not mirror fixed engine or reader policy as fake
configuration fields. Run validation still requires a BGEN locator string but
deliberately does not probe it with `Path::exists`; BGEN acquisition and
content selection belong to the engine and genotype reader. Sample, phenotype,
covariate, and prediction paths retain their frontend existence checks. This
separation permits a selected same-process snapshot-cache hit to use a missing
request locator, while an unselected open or selected cache miss must still
open the locator during engine preparation.

## Allowed downstream users

`g-runner` only.
