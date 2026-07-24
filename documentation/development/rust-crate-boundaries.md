# Rust Crate Boundaries

Each Rust crate exposes one facade through crate-root `pub use` items. Implementation modules stay private unless explicitly documented in that crate's `PUBLIC_API.md`.

Allowed ownership:

- `g-plan`: the canonical immutable `RunPlan`, including its optional BGEN
  content selector, numeric invariants, and host policy.
- `g-interface`: user config, CLI/TOML normalization, and native CLI dispatch.
- `g-genotype-contracts`: canonical BGEN content identity, shared variant
  metadata, and output-facing genotype column storage.
- `g-genotype`: BGEN/genotype source, immutable read sessions, owned decoded
  batch/buffer types, chunk specs, and preprocessing statistics.
- `g-genotype-cuda`: optional capability-gated packed8 raw-DEFLATE delivery,
  nvCOMP integration, checked-in PTX artifacts, and private typed-XLA FFI
  handlers.
- `g-compute-cuda`: optional capability-gated CUDA association kernels,
  checked-in PTX artifacts, and private typed-XLA FFI handlers.
- `native/cuda-driver`: repository-private, header-only CUDA driver ABI,
  dynamic loading, device/context validation, and module/launch support shared
  by the two independent CUDA crates.
- `g-input`: sample/phenotype/covariate/prediction alignment.
- `g-output`: Parquet output sessions, manifests, resume, and the
  output-owned association-implementation compatibility DTO.
- `g-runtime`: logging, telemetry session/writer lifecycle, timing, shutdown,
  Rayon state, and generic structured-diagnostic transport.
- `g-engine`: coarse run orchestration across genotype/input/output/runtime,
  including packed8 negotiation, typed association implementation state,
  output-compatibility projection, completed artifacts, and writer-completion
  events.
- `g-runner`: CLI dispatch, process-global runtime sequencing, terminal
  rendering, JAX process policy, and the single coordinated engine invocation.
- root `g` PyO3 crate: Python facade and private owner-type-to-NumPy adaptation
  only.

Rules:

- Do not add crate-root `pub mod` for implementation modules.
- Do not expose fake/test-only types from production facades. Delete obsolete
  test scaffolding when the corresponding tests are removed.
- Producers and output consumers import shared output-facing metadata and
  statistic columns directly from `g-genotype-contracts`. Decode and compute
  payloads remain owned by `g-genotype`. Intermediary crates do not re-export
  or redefine owner-defined types.
- `g-plan` depends directly on `g-genotype-contracts` for
  `BgenContentSha256`, embeds that owner-defined type in `InputPlan`, and does
  not re-export or redefine it.
- `g-interface` requires the BGEN locator string but does not probe its
  existence. `g-engine` owns reconciliation of configured selector policy with
  persisted output agreement and calls `g-genotype::BgenReaderCore::open_request`.
  Only actual reader content evidence, never the request locator or selector,
  may become output authority.
- Do not bind low-level Rust helper chains through Python when one Rust owner can call another directly.
- `g-runner` invokes `g-engine::execute_coordinated_run` after its host creates
  the Python-backed `AssociationBackend`; the root never sequences domain
  crate calls.
- `g-engine` owns runtime implementation selection state and maps it once into
  `g-output::AssociationImplementationCompatibility` before activation.
  `g-output` owns the persisted schema, exact resume comparison, and
  prepublication mismatch rejection. Driver/device observations never enter
  that DTO.
- Keep hot paths batch-oriented and ownership-visible; avoid per-variant public calls and cross-crate JSON in compute paths.
- Keep compute CUDA independent from genotype delivery at the Rust crate and
  public API boundaries. Shared vendored OpenXLA FFI headers live once under
  repository-root `vendor/openxla`; private CUDA driver support lives once
  under `native/cuda-driver`. Both CUDA crates source-include that header-only
  support without depending on or re-exporting the other crate's contracts.
- Update `PUBLIC_API.md` when adding or removing public facade items.
- The root PyO3 crate depends directly on `g-runner`, `g-engine`, canonical
  `g-plan` contracts, PyO3, and NumPy. It also imports canonical backend
  payload types directly from `g-genotype`, `g-input`, and `g-output`; these
  are type-only boundary dependencies, not service calls. The root implements
  `g-runner::NativeRunHost` for Python attachment, JAX construction, NumPy
  exchange, signal classification, and `PyErr` adaptation. `g-interface` and
  `g-runtime` remain behind the runner boundary.
