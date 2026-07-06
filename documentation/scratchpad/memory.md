# Agent Memory

> Internal scratchpad. Not user docs. May stale.

## Cross-Cutting Rules

- No permanent Python fallback after Rust owns contract.
- Work from crate facade, not internals.
- Move one boundary; delete old prod path.
- Prod BGEN dispatch = variant-major or packed8 unless app benchmark says otherwise.
- Profile before custom kernels; stop if hot app runs no improve.

## Flat Scratchpad Files

Use flat files:

- `architecture.md` for crate, migration, config, and dead-code rules.
- `science.md` for REGENIE Step 2 parity rules.
- `performance.md` for output and build-profile performance notes.
- `tooling.md` for tooling and Justfile policy.
- `documentation.md` for docs architecture and docs guardrails.
- `memory.md` for this cross-cutting agent memory.

## Useful Validation Commands

```bash
uv run ty check src tests scripts tooling
uv run ruff check src tests tooling
just check-core-stub
just check-rust-architecture
just check-python-architecture
cargo test --lib --quiet
```

GPU via SLURM on `landau`; broad CPU via configured CPU SLURM.
