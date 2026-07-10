# Rust Crate Boundaries

Each Rust crate exposes one facade through crate-root `pub use` items. Implementation modules stay private unless explicitly documented in that crate's `PUBLIC_API.md`.

Allowed ownership:

- `g-plan`: the canonical immutable `RunPlan`, numeric invariants, and host
  policy.
- `g-interface`: user config, CLI/TOML/Python-option normalization, and native
  CLI dispatch shell.
- `g-genotype`: BGEN/genotype source, chunk specs, preprocessing stats.
- `g-input`: sample/phenotype/covariate/prediction alignment.
- `g-output`: Parquet output sessions, manifests, and resume.
- `g-runtime`: logging, telemetry session/writer lifecycle, timing, shutdown,
  runtime/JAX policy.
- `g-engine`: coarse run orchestration across genotype/input/output/runtime,
  including completed artifacts and writer-completion events.
- root `g` PyO3 crate: Python facade only.

Rules:

- Do not add crate-root `pub mod` for implementation modules.
- Do not expose fake/test-only types from production facades. Delete obsolete
  test scaffolding when the corresponding tests are removed.
- Do not bind low-level Rust helper chains through Python when one Rust owner can call another directly.
- The binding invokes `g-engine::execute_coordinated_run` after creating the
  Python-backed `AssociationBackend`; it does not prepare or execute engine
  phases individually.
- Keep hot paths batch-oriented and ownership-visible; avoid per-variant public calls and cross-crate JSON in compute paths.
- Update `PUBLIC_API.md` when adding or removing public facade items.
- The root PyO3 crate depends directly on `g-interface`, `g-engine`,
  `g-runtime`, `g-plan`, PyO3, and NumPy only. `g-engine` re-exports the narrow
  error and output-completion types needed at the Python boundary; scheduler,
  input, writer, and buffer helpers remain private.
