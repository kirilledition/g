# GWAS Engine (g)

A high-performance, GPU-accelerated Genome-Wide Association Study (GWAS) engine targeting biobank-scale datasets faster than current SOTA tools such as plink2 and regenie.

The project follows a hybrid **Strangler Fig** architecture: a flexible Python frontend (JAX + Polars) coupled to a high-throughput Rust/CUDA backend.

## Documentation

- [Roadmap & Architecture](docs/ROADMAP.md) — overarching vision, milestones, and delivery strategy.
- [Phase 0: Preparation & Baselines](docs/PLAN_PHASE_0.md) — reproducible data setup and benchmark baseline plan.
- [Phase 1: Foundation & plink Parity](docs/PLAN_PHASE_1.md) — pure Python/JAX correctness milestone and parity plan.
- [Development Style Guide](docs/STYLEGUIDE.md) — strict Python/Rust coding conventions.
- [AI Agent Instructions](AGENTS.md) — repository-specific operating rules for AI coding agents.

## Development Environment (Nix)

This repository includes a `flake.nix` dev shell with the core tooling for local development on Nix/NixOS, including `uv`, `just`, Rust toolchain, `plink2`, and `regenie`.
