# No-Nix Development

Some development environments for this repo have `uv` and `maturin` available
but do not expose `nix`, `cargo`, `rustfmt`, or `cargo-clippy` directly.

If `just` is installed, use the local no-Nix lane for day-to-day iteration:

```bash
just check-local
just test-local
```

Equivalent commands without `just`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests scripts
uv run pytest tests/test_core.py tests/test_io_output.py
uv run pytest tests/ -m "not phase0_data and not phase1_parity"
```

The pytest commands are the normal local smoke for the installed native
extension. When a Rust toolchain is available to maturin, they also exercise
extension rebuilds through the `uv` cache keys. For an explicit
native-extension compile smoke in that environment, run:

```bash
uv run maturin develop
```

Keep this lane local and focused. On the gauss login node, do not use the
reduced-toolchain lane as a substitute for full validation. Full Python suites,
Rust builds, Rust test builds, and mixed Python/Rust validation should use the
CPU SLURM lane:

```bash
just slurm-cpu-check
just slurm-cpu-test
just slurm-cpu-rust-build
just slurm-cpu-rust-test
```

The CPU SLURM recipes derive Cargo and pytest worker counts from the allocation.
Local `test-local` remains serial by default so focused failures are easy to
read and do not queue a SLURM job.

Full Rust formatting and clippy checks still require an environment with the
Rust toolchain available:

```bash
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
```

Rust coverage also requires `cargo-llvm-cov`:

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --all-targets --ignore-filename-regex '(^|/)(benches|tests)/' --fail-under-lines 90
```
