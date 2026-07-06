# Architecture Notes

> Internal scratchpad. Not user docs. May stale.

## Core Rules

- Domain crates expose small facades; internals private or `pub(crate)`.
- Root `g` = PyO3 adapter, no policy.
- Python = public API + JAX kernels.
- Rust = host policy, I/O, runtime, manifests, scheduling, cleanup.
- No `g-common`, `g-utils`, dumping-ground crates.
- No permanent Python fallback after Rust owns contract.
- Move one boundary; delete old prod path.

## Crate Ownership

| Crate | Owns |
| --- | --- |
| `g-plan` | Request + prepared-plan contracts. |
| `g-interface` | CLI, TOML, defaults, overlay, validation, config-to-plan. |
| `g-cli` | Native CLI frontend, future process owner. |
| `g-genotype` | BGEN reader, chunk planning, preprocess, genotype stats. |
| `g-input` | Sample/phenotype/covariate/prediction alignment + grouping. |
| `g-output` | Output sessions, Arrow/Parquet/text writers, manifests, resume. |
| `g-runtime` | Runtime state, logging, telemetry, timing, shutdown, Rayon/JAX policy. |
| `g-engine` | Cross-domain coordinator, scheduling, preflight, backend trait, cleanup. |
| root `g` | PyO3 adapters, NumPy/JAX buffer bridges, module registration. |

## Facade Shape

Target crate root:

```rust
#![warn(clippy::pedantic)]

mod api;
mod error;
mod internal_module;

pub use api::*;
pub use error::CrateError;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

Avoid public implementation module trees:

```rust
pub mod parser;
pub mod scheduler;
pub mod writer;
```

Public module trees become accidental APIs.

## Dependency Direction

```text
g-plan
g-interface -> g-plan
g-cli -> g-interface
g-genotype
g-input
g-output -> g-plan
g-runtime
g-engine -> g-plan, g-genotype, g-input, g-output, g-runtime
root g -> internal crates
```

Only `g-engine` coordinates domains. Root `g` adapts to Python.

## Migration Pattern

1. Define/shrink Rust facade.
2. Move pure policy into the owning crate.
3. Add typed request/result structs.
4. Update PyO3 adapter to call the facade.
5. Remove Python fallback or wrapper.
6. Add guard if backslide easy.
7. Validate crate-local, stub, focused Python.

## Boundary Smells

- Root PyO3 exports tiny internal plan atoms.
- `Native*Policy` wraps stateless pure functions only.
- Typed Rust/Python values cross as `serde_json::Value` or `PyDict`.
- Python wrapper only renames one native call and has one caller.
- Tuple return crosses crate/Python boundary.
- Test fake/fixture public in prod API.
- Internal module rename touches another crate.

## Internal Cleanup Targets

| Area | Target |
| --- | --- |
| `g-plan` | Boring serializable contracts; split request/prepared subdomains only when growth forces it. |
| `g-interface` | Messy user config stops here; downstream sees clean `g-plan` values. |
| `g-cli` | Process frontend only: parse, signals, call interface, exit. |
| `g-genotype` | Typed batch facade; raw pointers, unsafe, SIMD quarantined. |
| `g-input` | Split sample/table/phenotype/covariate/alignment/prediction/grouping; typed errors. |
| `g-output` | One canonical chunk-handle write path; split chunk/session/manifest/resume/writer/finalization/timing. |
| `g-runtime` | Process side effects + typed observability; event strings private serialization. |
| `g-engine` | Workflow owner; split scheduler/delivery/effects/preflight/backend policies. |
| root `g` | PyO3 glue only; no run behavior, manifests, scheduling, or sample logic. |

Priority cleanup:

1. Split `g-engine::schedule`.
2. Split `g-output::session`.
3. Split `g-input::sample`.
4. Shrink `src/python/run_engine.rs`.
5. Split/type `g-runtime::run_events`.
6. Refine `g-genotype` facade.
7. Narrow `g-interface::GComputeConfigData`.
8. Keep `g-plan` from becoming universal types crate.

## Config Frontend

Rust owns the configuration frontend:

```text
CLI args / TOML / Python option dict
    -> typed layers
    -> packaged defaults
    -> overlay and provenance
    -> resolved RegenieConfigData
    -> run validation
    -> g-plan RunRequest
```

Owners: `options.rs` metadata; `cli/` parse+layer; `partial.rs` TOML shape;
`config.default.toml` defaults; `overlay.rs` merge+provenance;
`validation.rs`/`run_validation.rs` validation; `plan_request.rs` -> `g-plan`;
`src/g/interface/config.py` normalizes Python input + calls PyO3 only.

No second option table in Python/docs/tests/runner.

## Config Rules

- CLI = REGENIE-compat surface.
- TOML = native strict `g` config.
- Mutable defaults live only in `crates/interface/src/config.default.toml`.
- User intent from provenance, not value-vs-default compare.
- TOML/Python parse: no filesystem checks.
- File/env checks at run boundary.

## Dead-Code Checklist

Static dead-code tools = evidence, not proof. Check:

1. Production references with `rg`.
2. Python import boundaries.
3. PyO3 registration, `_core.pyi`, and production `_core` references.
4. Cargo dependencies with `cargo machete`.
5. Focused type/lint/stub checks.

High-confidence remove:

- Dependency unused by declaring package; real users declare own.
- Python helper has no prod refs, only wraps native.
- PyO3 class not registered, not in `_core.pyi`, not return type.
- Replaced prod module has architecture guard.

Low-confidence only:

- `vulture` in callback/compute/public API.
- PyO3 class lacks direct Python ref.
- Module imported through `importlib`.
- Enum member serialized in output/manifest/docs.

## Validation Lanes

```bash
cargo check -p <crate> --lib
cargo clippy -p <crate> --lib -- -D warnings -W clippy::pedantic
just check-core-stub
just check-rust-architecture
just check-python-architecture
uv run ty check src tests scripts tooling
uv run ruff check src tests tooling
```

Use SLURM for heavy CPU/GPU suites.
