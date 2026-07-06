# Rust Crate Boundaries

Each Rust crate exposes one facade through crate-root `pub use` items. Implementation modules stay private unless explicitly documented in that crate's `PUBLIC_API.md`.

Allowed ownership:

- `g-plan`: stable run/config/prepared-plan DTOs.
- `g-interface`: user config and CLI/TOML/Python-option normalization.
- `g-cli`: native CLI dispatch shell.
- `g-genotype`: BGEN/genotype source, chunk specs, preprocessing stats.
- `g-input`: sample/phenotype/covariate/prediction alignment.
- `g-output`: output sessions, manifests, resume, finalization.
- `g-runtime`: logging, telemetry, timing, shutdown, runtime/JAX policy.
- `g-engine`: orchestration across genotype/input/output/runtime.
- root `g` PyO3 crate: Python facade only.

Rules:

- Do not add crate-root `pub mod` for implementation modules.
- Do not expose fake/test-only types from production facades; use `test_support`.
- Do not bind low-level Rust helper chains through Python when one Rust owner can call another directly.
- Keep hot paths batch-oriented and ownership-visible; avoid per-variant public calls and cross-crate JSON in compute paths.
- Update `PUBLIC_API.md` when adding or removing public facade items.
