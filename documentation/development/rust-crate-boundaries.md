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
- `g-runner`: CLI dispatch, process-global runtime sequencing, terminal
  rendering, and the single coordinated engine invocation.
- root `g` PyO3 crate: Python facade only.

Rules:

- Do not add crate-root `pub mod` for implementation modules.
- Do not expose fake/test-only types from production facades. Delete obsolete
  test scaffolding when the corresponding tests are removed.
- Do not bind low-level Rust helper chains through Python when one Rust owner can call another directly.
- `g-runner` invokes `g-engine::execute_coordinated_run` after its host creates
  the Python-backed `AssociationBackend`; the root never sequences domain
  crate calls.
- Keep hot paths batch-oriented and ownership-visible; avoid per-variant public calls and cross-crate JSON in compute paths.
- Update `PUBLIC_API.md` when adding or removing public facade items.
- The root PyO3 crate depends directly on `g-runner`, `g-engine`, PyO3, and
  NumPy only. It implements `g-runner::NativeRunHost` for Python attachment,
  JAX construction, NumPy exchange, signal classification, and `PyErr`
  adaptation. `g-interface`, `g-plan`, and `g-runtime` remain behind the
  runner boundary.
