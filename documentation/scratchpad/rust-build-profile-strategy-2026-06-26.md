# Rust Build Profile Strategy

Yes. There are two separate problems here:

1. Your production/perf build can probably be tuned better.
2. Your everyday development build path is too expensive and needs a clearly separate fast profile.

Right now the repo has only one custom performance profile:

```toml
[profile.perf]
inherits = "release"
opt-level = 3
codegen-units = 1
debug = 1
lto = "fat"
strip = "none"

[profile.bench]
opt-level = 3
codegen-units = 1
debug = 1
lto = "fat"
strip = "none"
```

and the `install-perf-extension` recipe builds the PyO3 extension with that profile plus `RUSTFLAGS="-C target-cpu=native"`.

That is a maximum-optimization-ish profile, not a good default for iteration.

# My verdict

## Runtime performance

Your current `perf` profile is already aggressive:

```text
opt-level=3
codegen-units=1
fat LTO
target-cpu=native
```

Those are the usual expensive knobs. Cargo's docs describe higher `opt-level` as potentially faster but slower to compile, `codegen-units=1` as potentially faster generated code but slower compile, and LTO as whole-program optimization at longer link time.

So there may be extra performance available, but not from just "turning on LLVM." Rust is already using LLVM for these optimizations. The realistic next performance knobs are:

```text
PGO
ThinLTO vs FatLTO comparison
opt-level=2 vs opt-level=3 comparison
target-cpu policy
linker choice
hot-code-level algorithmic/layout changes
```

## Build speed

You are almost certainly wasting time by using `perf` too often. Fat LTO plus one codegen unit is one of the slowest reasonable combinations. It should be reserved for final benchmark/release builds, not normal development.

You need at least three named build modes:

```text
dev-fast        fast local PyO3 iteration
dev-opt         moderately optimized debug/testing extension
perf            production-ish benchmark build
perf-max        expensive final benchmark / release-candidate build
```

# Recommended Cargo profiles

I would change the profile strategy to this.

## 1. Fast local extension profile

Use for day-to-day Python tests, CLI smoke tests, and agent iteration.

```toml
[profile.dev-fast]
inherits = "dev"
opt-level = 0
debug = "line-tables-only"
split-debuginfo = "unpacked"
incremental = true
codegen-units = 256
lto = "off"
```

This gives fast builds and keeps line-level backtraces. Cargo's default dev profile already uses no optimization, incremental compilation, and many codegen units, which are compile-time friendly.

Add:

```bash
just install-dev-extension
```

```make
install-dev-extension:
    {{ server_env }} && uv run --no-sync maturin develop --profile dev-fast --uv
```

This should be the default for agents unless they are benchmarking.

## 2. Moderately optimized local profile

Use for tests that need native code to be not ridiculously slow, but where full LTO is wasteful.

```toml
[profile.dev-opt]
inherits = "dev"
opt-level = 1
debug = "line-tables-only"
split-debuginfo = "unpacked"
incremental = true
codegen-units = 256
lto = "off"
```

Optional dependency optimization experiment:

```toml
[profile.dev-opt.package."*"]
opt-level = 1
```

Be cautious with dependency overrides. Cargo's docs note that profile overrides can interact subtly with generics and monomorphization.

Add:

```bash
just install-dev-opt-extension
```

## 3. Routine benchmark profile

I would change your current `perf` profile to use ThinLTO, not FatLTO, unless benchmarks prove FatLTO wins enough to justify the build cost.

```toml
[profile.perf]
inherits = "release"
opt-level = 3
codegen-units = 8
debug = "line-tables-only"
lto = "thin"
strip = "none"
incremental = false
```

Cargo's docs describe ThinLTO as substantially faster than FatLTO while still offering similar gains in many cases.

Why `codegen-units = 8`? Because `1` is best for maximum cross-function optimization, but it is expensive. A routine benchmark profile should not be maximally painful. Test `1`, `4`, `8`, and `16`.

Keep `RUSTFLAGS="-C target-cpu=native"` for node-local performance builds.

## 4. Maximum performance profile

Keep a separate expensive profile for final numbers:

```toml
[profile.perf-max]
inherits = "release"
opt-level = 3
codegen-units = 1
debug = "line-tables-only"
lto = "fat"
strip = "none"
incremental = false
```

Use it only for release candidates, headline benchmarks, or regressions where the last few percent matter.

```bash
RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf-max --uv
```

## 5. Size/distribution profile

If you later publish wheels and care about size:

```toml
[profile.dist]
inherits = "release"
opt-level = 3
codegen-units = 16
debug = 0
lto = "thin"
strip = "symbols"
incremental = false
```

Do not use this for profiling. Stripping symbols makes profiler output less useful.

# Benchmark the build profiles directly

Add a tool or Justfile recipe:

```bash
just benchmark-rust-build-profiles
```

It should build the extension under:

```text
dev-fast
dev-opt
release
perf-thin-cgu8
perf-thin-cgu1
perf-fat-cgu1
perf-o2-thin-cgu8
perf-o3-thin-cgu8
```

and record:

```text
clean build time
incremental build time after touching src/python/mod.rs
incremental build time after touching genotype decode code
incremental build time after touching output writer code
wheel / .so size
import time
smoke run time
BGEN reader throughput
binary-hot GPU smoke time
```

Do not guess. This is exactly the kind of thing `tooling/` should standardize.

# Try `opt-level=2`

Do not assume `opt-level=3` wins. Cargo's docs explicitly recommend experimenting because level 3 can sometimes be slower than 2.

Add:

```toml
[profile.perf-o2]
inherits = "release"
opt-level = 2
codegen-units = 8
debug = "line-tables-only"
lto = "thin"
strip = "none"
```

Test it on:

```text
BGEN decode
preprocess
output writer
binary-hot GPU
linear CPU/GPU
```

For memory-bandwidth-heavy decode, `opt-level=2` may be competitive.

# Use PGO for serious release builds

The biggest build-flag performance opportunity is probably PGO, not more LTO. Rust supports profile-guided optimization through `-Cprofile-generate` and `-Cprofile-use`: build an instrumented binary, run representative workloads, merge `.profraw` files with `llvm-profdata`, then rebuild using the profile data.

For `g`, representative PGO workloads should include:

```text
BGEN decode full chunk path
packed8 decode path
sample/prediction alignment
output writer/finalization
binary score-only
binary approx-Firth
linear quantitative
```

I would add:

```bash
just pgo-generate-extension
just pgo-run-training
just pgo-merge
just pgo-install-extension
```

Sketch:

```bash
# 1. Build instrumented extension
RUSTFLAGS="-Cprofile-generate=/tmp/g-pgo-data -C target-cpu=native" \
  uv run --no-sync maturin develop --profile perf --uv

# 2. Run representative workloads
just benchmark-bgen-reader ...
just benchmark-regenie2-binary-hot-gpu-smoke ...
just regenie2-chr10-matrix ...

# 3. Merge
llvm-profdata merge -o /tmp/g-pgo-data/merged.profdata /tmp/g-pgo-data

# 4. Rebuild optimized
RUSTFLAGS="-Cprofile-use=/tmp/g-pgo-data/merged.profdata -Cllvm-args=-pgo-warn-missing-function -C target-cpu=native" \
  uv run --no-sync maturin develop --profile perf-max --uv
```

The rustc PGO docs recommend absolute profile paths and warn that using a Cargo `--target` helps avoid passing PGO flags to build scripts. For Maturin, verify the equivalent `--target` path before baking this into automation.

Do this only after the Rust host migration stabilizes more. PGO while the code is moving daily is annoying.

# Linker choice

Build time may improve with a faster linker. On Linux, Rust/rustc has stable support for LLD linker features on `x86_64-unknown-linux-gnu`, and the rustc docs describe `-Clinker-features=+lld` / `-lld` behavior.

Try explicitly:

```bash
RUSTFLAGS="-Clinker-features=+lld"
```

or, if you install `mold`:

```bash
RUSTFLAGS="-C link-arg=-fuse-ld=mold"
```

Measure:

```text
clean build
incremental link after touching Rust code
maturin develop
perf profile with LTO
```

Do not assume it changes runtime performance. This is mostly a build/link-time improvement.

# CPU targeting policy

Current `install-perf-extension` uses:

```bash
RUSTFLAGS="-C target-cpu=native"
```

That is correct for node-local benchmarking. `target-cpu=native` tells rustc to generate code for the host CPU.

But it is not safe as a universal distribution setting. If you build on a newer CPU and run on an older node, the extension may use unsupported instructions.

I recommend three modes:

```text
generic        portable wheel / CI
native         local benchmark on a known node
cluster-v3/v4  if you standardize on a cluster CPU baseline
```

Do not manually enable `target-feature=+avx2,+fma,...` unless you know the deployment baseline. Rust's docs warn that `target-feature` can produce undefined runtime behavior if used incorrectly. Prefer `target-cpu=native` for local perf and a named CPU baseline for portable cluster builds.

# Debug info

Your current `perf` uses:

```toml
debug = 1
strip = "none"
```

That is defensible because you use profilers. I would switch to:

```toml
debug = "line-tables-only"
```

for most perf/profiling profiles. It gives file/line symbolization with less debug payload than fuller debug info. Cargo documents `line-tables-only` as minimal debug info for filename/line number backtraces.

For deep profiling with native stacks, keep symbols available:

```text
debug = "line-tables-only"
strip = "none"
```

For distribution:

```text
debug = 0
strip = "symbols"
```

# Build parallelism problem

Your `server_env.sh` defines `gwas_engine_configure_cpu_parallelism()` and sets `CARGO_BUILD_JOBS` from SLURM allocation. But many recipes merely source `server_env.sh`; they do not necessarily call that function. `slurm-cpu-run` does call it, but `slurm-gpu-run` appears to source the environment and then runs the provided command without calling the CPU parallelism function.

That means some GPU-side builds may use too many cores or behave inconsistently.

I would update build-heavy recipes to explicitly call:

```bash
gwas_engine_configure_cpu_parallelism
```

before `cargo`, `maturin`, `clippy`, or `llvm-cov`.

Example:

```make
install-perf-extension:
    {{ server_env }} && gwas_engine_configure_cpu_parallelism && RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf --uv
```

Also add a log line:

```bash
echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-unset}"
```

This matters on SLURM.

# Use `sccache`

For agent-heavy Rust migration, I would strongly consider `sccache`.

Add optional support:

```bash
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"
```

only if `sccache` exists. Put cache under a fast local or shared location depending on the machine:

```bash
SCCACHE_DIR=/tmp/g-sccache
```

or a BeegFS path if sharing across worktrees is worth it. Measure; network filesystems can be slower.

This will likely help repeated agent builds more than another LTO tweak.

# Multi-crate will help build iteration

The planned multi-crate split is also a build-speed feature. Right now the root Rust crate is the PyO3 extension and includes PyO3, NumPy, Arrow, Parquet, BGEN, Clap, tracing, signal handling, hashing, and TOML all in one package.

Once you split into crates:

```text
g-genotype
g-output
g-interface
g-input
g-runtime
g-engine
root PyO3 adapter
```

Cargo can avoid recompiling genotype code when you only touch PyO3 adapter code, and it can run crate-specific tests/benches. You will still link the final `_core` extension, but incremental compilation should become more predictable.

That reinforces the migration plan: extract crates early.

# What I would change now

## Cargo.toml

Add profiles:

```toml
[profile.dev-fast]
inherits = "dev"
opt-level = 0
debug = "line-tables-only"
split-debuginfo = "unpacked"
incremental = true
codegen-units = 256
lto = "off"

[profile.dev-opt]
inherits = "dev"
opt-level = 1
debug = "line-tables-only"
split-debuginfo = "unpacked"
incremental = true
codegen-units = 256
lto = "off"

[profile.perf]
inherits = "release"
opt-level = 3
codegen-units = 8
debug = "line-tables-only"
lto = "thin"
strip = "none"
incremental = false

[profile.perf-max]
inherits = "release"
opt-level = 3
codegen-units = 1
debug = "line-tables-only"
lto = "fat"
strip = "none"
incremental = false

[profile.perf-o2]
inherits = "release"
opt-level = 2
codegen-units = 8
debug = "line-tables-only"
lto = "thin"
strip = "none"
incremental = false
```

I would also change `[profile.bench]` away from FatLTO by default:

```toml
[profile.bench]
opt-level = 3
codegen-units = 8
debug = "line-tables-only"
lto = "thin"
strip = "none"
```

Keep a separate `bench-max` workflow if you need FatLTO.

## Justfile

Add:

```make
install-dev-extension:
    {{ server_env }} && gwas_engine_configure_cpu_parallelism && uv run --no-sync maturin develop --profile dev-fast --uv

install-dev-opt-extension:
    {{ server_env }} && gwas_engine_configure_cpu_parallelism && uv run --no-sync maturin develop --profile dev-opt --uv

install-perf-extension:
    {{ server_env }} && gwas_engine_configure_cpu_parallelism && RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf --uv

install-perf-max-extension:
    {{ server_env }} && gwas_engine_configure_cpu_parallelism && RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf-max --uv
```

Add build profiling:

```make
benchmark-rust-build-profiles:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.rust_build_profiles
```

# What I would not do yet

I would not immediately:

```text
- make panic=abort for the PyO3 extension;
- manually enable target-feature flags;
- require nightly-only compiler features;
- adopt Cranelift for the main dev path;
- strip symbols in profiling builds;
- make FatLTO the default benchmark profile;
- do PGO before the Rust host migration stabilizes.
```

`panic=abort` can make failures harsher in an embedded Python process. Manual `target-feature` is risky across machines. Nightly/Cranelift adds another moving part during an already large migration.

# Expected wins

Roughly:

```text
Fast dev profile: likely large build-time win, slower native runtime
ThinLTO perf: likely large link-time win, maybe neutral runtime
perf-max FatLTO: keep best possible runtime, slow build
PGO: possible meaningful runtime win, operationally more complex
target-cpu=native: already good for local perf
sccache: likely major repeated-build win for agents
multi-crate split: likely major incremental-build and test-targeting win
```

# My recommendation

Make `perf-max` the expensive build and stop using it casually. Make `dev-fast` the default for agents and local correctness work. Make `perf` ThinLTO-based and use it for routine benchmarks. Add a build-profile benchmark harness so you can decide with data.

The most important immediate changes are:

```text
1. Add dev-fast/dev-opt/perf/perf-max profiles.
2. Change routine perf from fat LTO to thin LTO unless benchmarks prove otherwise.
3. Add explicit CARGO_BUILD_JOBS setup to build recipes.
4. Add sccache support.
5. Add tooling to compare build profiles.
6. Keep PGO as the next serious release-performance experiment.
```
