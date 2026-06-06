# Justfile for GWAS Engine (g)

set allow-duplicate-recipes := true
set shell := ["bash", "-cu"]

data_dir := env_var_or_default('GWAS_ENGINE_DATA_DIR', 'data')
python_version := env_var_or_default('GWAS_ENGINE_PYTHON_VERSION', '3.14')
tools_dir := env_var_or_default('GWAS_ENGINE_TOOLS_DIR', '.tools')
slurm_gpu_node := env_var_or_default('GWAS_ENGINE_GPU_NODE', 'landau')
slurm_partition := env_var_or_default('GWAS_ENGINE_SLURM_PARTITION', '')
slurm_account := env_var_or_default('GWAS_ENGINE_SLURM_ACCOUNT', '')
slurm_time_limit := env_var_or_default('GWAS_ENGINE_SLURM_TIME', '04:00:00')
slurm_cpus_per_task := env_var_or_default('GWAS_ENGINE_SLURM_CPUS_PER_TASK', '8')
slurm_memory := env_var_or_default('GWAS_ENGINE_SLURM_MEMORY', '64G')
slurm_gpu_count := env_var_or_default('GWAS_ENGINE_SLURM_GPUS_PER_TASK', '1')
slurm_extra_arguments := env_var_or_default('GWAS_ENGINE_SLURM_EXTRA_ARGS', '')
server_env := '. scripts/server_env.sh'

# --- Data Preparation ---

setup-data:
    {{server_env}} && uv run python scripts/fetch_1kg.py
    {{server_env}} && uv run python scripts/simulate_phenos.py

# Generate binary REGENIE step 1 predictions required by g binary step 2
setup-binary-baseline: setup-data
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    mkdir -p "{{data_dir}}/baselines"
    regenie \
      --step 1 \
      --bed "{{data_dir}}/1kg_chr22_full" \
      --phenoFile "{{data_dir}}/pheno_bin.txt" \
      --covarFile "{{data_dir}}/covariates.txt" \
      --bt \
      --cc12 \
      --force-step1 \
      --bsize 1000 \
      --out "{{data_dir}}/baselines/regenie_step1"
    test -s "{{data_dir}}/baselines/regenie_step1_pred.list"

# Prepare all local inputs required for binary REGENIE step 2 GPU execution
setup-regenie2-binary-gpu-inputs: setup-binary-baseline

# Verify local inputs required for binary REGENIE step 2 GPU execution
verify-regenie2-binary-gpu-inputs:
    #!/usr/bin/env bash
    set -euo pipefail
    test -s "{{data_dir}}/1kg_chr22_full.bgen"
    test -s "{{data_dir}}/1kg_chr22_full.sample"
    test -s "{{data_dir}}/pheno_bin.txt"
    test -s "{{data_dir}}/covariates.txt"
    test -s "{{data_dir}}/baselines/regenie_step1_pred.list"
    echo "Binary REGENIE step 2 GPU inputs are present."

# Run PLINK2/Regenie baselines and generate hardware report (excludes slow Hail benchmarks by default)
benchmark-baselines: setup-data
    {{server_env}} && uv run python scripts/benchmark.py

# Build and install the Rust extension using the opt-in native performance profile
install-perf-extension:
    {{server_env}} && RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf --uv

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU
benchmark-regenie-comparison-cpu: setup-data install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_regenie_comparison.py --cpu-only

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU+GPU
benchmark-regenie-comparison-gpu: setup-data install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_regenie_comparison.py --include-gpu

# Alias for comparison benchmark (CPU-only default)
benchmark-regenie-comparison: benchmark-regenie-comparison-cpu

# Run full baselines including Hail (slow - requires cached MatrixTable)
benchmark-baselines-full: setup-data
    {{server_env}} && HAIL_INCLUDE=1 uv run python scripts/benchmark.py

# --- Development ---

# Install repo-local command-line tools for the Ubuntu SLURM server
setup-server-tools:
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/g-uv-cache}" UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv run --no-project python scripts/bootstrap_server_tools.py

# Bootstrap a CPU-only development environment on the login node
bootstrap:
    {{server_env}} && uv python install {{python_version}}
    {{server_env}} && uv sync --python {{python_version}} --group dev

# Bootstrap a GPU-capable development environment for JAX CUDA work
bootstrap-gpu:
    {{server_env}} && uv python install {{python_version}}
    {{server_env}} && uv sync --python {{python_version}} --group dev --group gpu

# Install CUDA-capable Python dependencies into the current environment
install-gpu-dependencies:
    {{server_env}} && uv sync --python {{python_version}} --group dev --group gpu

# Check local toolchain prerequisites for development on the current host
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    required_commands=(uv cargo rustc)
    for command_name in "${required_commands[@]}"; do
      if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
      fi
    done
    resolved_python_version="$(
      uv run --python {{python_version}} python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"
    if [[ "${resolved_python_version}" != "{{python_version}}" ]]; then
      echo "Resolved Python ${resolved_python_version}, expected {{python_version}}." >&2
      exit 1
    fi
    echo "Core development toolchain looks usable on this host."

# Check server development prerequisites, local tools, and cache writability
doctor-server:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    required_commands=(just uv srun zstd cargo cargo-clippy cargo-fmt rustc rustfmt plink plink2 regenie)
    for command_name in "${required_commands[@]}"; do
      if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required server command: ${command_name}" >&2
        exit 1
      fi
    done
    mkdir -p "${UV_CACHE_DIR}"
    test -w "${UV_CACHE_DIR}"
    uv run --python {{python_version}} python -c 'import sys; print(f"python={sys.version_info.major}.{sys.version_info.minor}")'
    echo "hostname=$(hostname)"
    echo "tools_dir={{tools_dir}}"
    echo "uv_cache_dir=${UV_CACHE_DIR}"
    echo "Server development toolchain looks usable on this host."

# Check external baseline tools used by data prep and comparison benchmarks
doctor-baselines:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    required_commands=(plink plink2 regenie)
    for command_name in "${required_commands[@]}"; do
      if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required baseline command: ${command_name}" >&2
        exit 1
      fi
    done
    echo "Baseline benchmark tools are available on PATH."

# Probe JAX runtime on the current host
doctor-jax:
    {{server_env}} && uv run --python {{python_version}} python scripts/probe_jax_runtime.py

# Start an interactive SLURM shell on the configured GPU node
slurm-gpu-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    slurm_arguments=(
      "--nodelist={{slurm_gpu_node}}"
      "--gres=gpu:{{slurm_gpu_count}}"
      "--cpus-per-task={{slurm_cpus_per_task}}"
      "--mem={{slurm_memory}}"
      "--time={{slurm_time_limit}}"
    )
    if [[ -n "{{slurm_partition}}" ]]; then
      slurm_arguments+=("--partition={{slurm_partition}}")
    fi
    if [[ -n "{{slurm_account}}" ]]; then
      slurm_arguments+=("--account={{slurm_account}}")
    fi
    if [[ -n "{{slurm_extra_arguments}}" ]]; then
      read -r -a extra_arguments <<< "{{slurm_extra_arguments}}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    exec srun "${slurm_arguments[@]}" --pty bash -l

# Run an arbitrary command through SLURM on the configured GPU node
slurm-gpu-run +command_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    slurm_arguments=(
      "--nodelist={{slurm_gpu_node}}"
      "--gres=gpu:{{slurm_gpu_count}}"
      "--cpus-per-task={{slurm_cpus_per_task}}"
      "--mem={{slurm_memory}}"
      "--time={{slurm_time_limit}}"
    )
    if [[ -n "{{slurm_partition}}" ]]; then
      slurm_arguments+=("--partition={{slurm_partition}}")
    fi
    if [[ -n "{{slurm_account}}" ]]; then
      slurm_arguments+=("--account={{slurm_account}}")
    fi
    if [[ -n "{{slurm_extra_arguments}}" ]]; then
      read -r -a extra_arguments <<< "{{slurm_extra_arguments}}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    exec srun "${slurm_arguments[@]}" {{command_arguments}}

# Run another just recipe through SLURM on the configured GPU node
slurm-gpu-just +just_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    exec just slurm-gpu-run just {{just_arguments}}

# Run REGENIE step 2 with local baseline predictions
regenie-linear:
    {{server_env}} && uv run g regenie --step 2 --qt --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --phenoFile {{data_dir}}/pheno_cont.txt --phenoCol phenotype_continuous --covarFile {{data_dir}}/covariates.txt --covarColList age,sex --pred {{data_dir}}/baselines/regenie_step1_qt_pred.list --out {{data_dir}}/regenie_linear --g-output-format parquet

# Run binary REGENIE step 2 on chr22 with GPU JAX
regenie2-binary-gpu:
    {{server_env}} && uv run g regenie --step 2 --bt --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --phenoFile {{data_dir}}/pheno_bin.txt --phenoCol phenotype_binary --covarFile {{data_dir}}/covariates.txt --covarColList age,sex --pred {{data_dir}}/baselines/regenie_step1_pred.list --out {{data_dir}}/regenie2_binary_chr22_gpu --g-device gpu --firth --approx --g-output-format parquet

# Smoke test binary REGENIE step 2 on a small chr22 variant slice with GPU JAX
regenie2-binary-gpu-smoke:
    {{server_env}} && uv run g regenie --step 2 --bt --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --phenoFile {{data_dir}}/pheno_bin.txt --phenoCol phenotype_binary --covarFile {{data_dir}}/covariates.txt --covarColList age,sex --pred {{data_dir}}/baselines/regenie_step1_pred.list --out {{data_dir}}/regenie2_binary_chr22_gpu_smoke --g-device gpu --firth --approx --g-variant-limit 1000 --g-output-format parquet

# Run binary REGENIE step 2 through SLURM on the configured GPU node
slurm-regenie2-binary-gpu:
    {{server_env}} && just slurm-gpu-just regenie2-binary-gpu

# Smoke test binary REGENIE step 2 through SLURM on the configured GPU node
slurm-regenie2-binary-gpu-smoke:
    {{server_env}} && just slurm-gpu-just regenie2-binary-gpu-smoke

# Verify binary REGENIE step 2 GPU output artifacts
verify-regenie2-binary-gpu-output:
    #!/usr/bin/env bash
    set -euo pipefail
    run_directory="{{data_dir}}/regenie2_binary_chr22_gpu.regenie2_binary.run"
    test -d "${run_directory}/parts"
    find "${run_directory}/parts" -type f -name '*.parquet' | grep -q .
    echo "Binary REGENIE step 2 GPU output is present."

# Verify binary REGENIE step 2 GPU smoke output artifacts
verify-regenie2-binary-gpu-smoke-output:
    #!/usr/bin/env bash
    set -euo pipefail
    run_directory="{{data_dir}}/regenie2_binary_chr22_gpu_smoke.regenie2_binary.run"
    test -d "${run_directory}/parts"
    find "${run_directory}/parts" -type f -name '*.parquet' | grep -q .
    echo "Binary REGENIE step 2 GPU smoke output is present."

# Run CPU/GPU JAX runtime probe
probe-jax:
    {{server_env}} && uv run python scripts/probe_jax_runtime.py

# Benchmark BGEN float32 read paths
benchmark-bgen-reader: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_bgen_reader.py

# Benchmark REGENIE step 2 in fresh Python processes
benchmark-regenie2-linear-fresh-gpu: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu

# Benchmark REGENIE step 2 in fresh Python processes using Parquet dataset output plus finalization
benchmark-regenie2-linear-fresh-gpu-parquet: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu --finalize-parquet

# Benchmark binary REGENIE step 2 with cold, same-process hot, chunk-only, and finalized timings
benchmark-regenie2-binary-hot-gpu: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_regenie2_binary_hot.py --device gpu

# Benchmark output-stage timings across finalization, phenotype count, and bsize
benchmark-output-stages-gpu: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_output_stages.py --device gpu

# Smoke test binary REGENIE step 2 benchmark harness on a small variant slice
benchmark-regenie2-binary-hot-gpu-smoke: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/benchmark_regenie2_binary_hot.py --device gpu --variant-limit 1000 --no-include-cold-process --no-include-finalized-hot

# Submit binary hot benchmark to the configured GPU node
slurm-benchmark-regenie2-binary-hot-gpu:
    {{server_env}} && just slurm-gpu-just benchmark-regenie2-binary-hot-gpu

# Sequentially tune GPU REGENIE step 2 and active BGEN reader knobs
tune-regenie2-gpu: install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/tune_regenie2_gpu.py

# Run Rust Criterion benchmarks with native performance flags
benchmark-rust:
    {{server_env}} && RUSTFLAGS="-C target-cpu=native" cargo bench

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU
profile-regenie-comparison-cpu: setup-data install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/profile_regenie_comparison.py --cpu-only

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU+GPU
profile-regenie-comparison-gpu: setup-data install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/profile_regenie_comparison.py --include-gpu

# Alias for unified profiling comparison (CPU-only default)
profile-regenie-comparison: profile-regenie-comparison-cpu

# Run the deep REGENIE step 2 profiling harness on the current host
profile-regenie2-deep: setup-data install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/profile_regenie2_deep.py

# Smoke test the deep REGENIE step 2 profiling harness on the current host
profile-regenie2-deep-smoke: setup-data install-perf-extension
    {{server_env}} && uv run --no-sync python scripts/profile_regenie2_deep.py --smoke --skip-deep-profiles

# Submit one long landau SLURM job for the deep REGENIE step 2 profiling harness
profile-regenie2-deep-landau:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    slurm_arguments=(
      "--nodelist={{slurm_gpu_node}}"
      "--gres=gpu:1"
      "--cpus-per-task=8"
      "--mem=64G"
      "--time=12:00:00"
    )
    if [[ -n "{{slurm_partition}}" ]]; then
      slurm_arguments+=("--partition={{slurm_partition}}")
    fi
    if [[ -n "{{slurm_account}}" ]]; then
      slurm_arguments+=("--account={{slurm_account}}")
    fi
    if [[ -n "{{slurm_extra_arguments}}" ]]; then
      read -r -a extra_arguments <<< "{{slurm_extra_arguments}}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    exec srun "${slurm_arguments[@]}" bash -lc '. scripts/server_env.sh && just install-gpu-dependencies && just install-perf-extension && uv run --no-sync python scripts/profile_regenie2_deep.py'

# Format code
format:
    {{server_env}} && uv run ruff format .
    {{server_env}} && cargo fmt

# Lint code
lint:
    {{server_env}} && uv run ruff check . --fix
    {{server_env}} && cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic

# Type check Python code
typecheck:
    {{server_env}} && uv run ty check src tests scripts

# Run all checks (format, lint, typecheck)
check: format lint typecheck

# Check Python formatting without requiring Nix or direct Cargo access
format-local-check:
    uv run ruff format --check .

# Lint Python without applying fixes; useful in uv/maturin-only environments
lint-local:
    uv run ruff check .

# Type check Python in uv/maturin-only environments
typecheck-local:
    uv run ty check src tests scripts

# Focused no-Nix smoke tests that also rebuild the native extension through maturin
test-local-focused:
    uv run pytest tests/test_core.py tests/test_io_output.py

# Non-heavy no-Nix test suite
test-local:
    uv run pytest tests/ -m "not phase0_data and not phase1_parity"

# Local no-Nix verification lane; Rust fmt/clippy still require a full Cargo toolchain
check-local: format-local-check lint-local typecheck-local test-local-focused

# Run CI lint checks without installing the project package
ci-lint:
    {{server_env}} && uv sync --group dev --frozen --no-install-project
    {{server_env}} && uv run --no-sync ruff check .

# Run CI type checks without installing the project package
ci-typecheck:
    {{server_env}} && uv sync --group dev --frozen --no-install-project
    {{server_env}} && uv run --no-sync ty check src tests scripts

# Run CI tests that exclude heavy data- and parity-dependent suites
ci-test:
    {{server_env}} && uv sync --group dev --frozen
    {{server_env}} && uv run --no-sync pytest tests/ -m "not phase0_data and not phase1_parity"

# Run tests
test:
    {{server_env}} && uv run pytest tests/

# Run Python coverage gate
coverage-python:
    {{server_env}} && uv run pytest tests/ --cov=src/g --cov-report=term-missing --cov-fail-under=90

# Run Rust line coverage gate
coverage-rust:
    {{server_env}} && cargo llvm-cov --workspace --all-targets --fail-under-lines 90

# Run all coverage gates
coverage: coverage-python coverage-rust

# Generate docs/code-review.tasks.json from docs/code-review.md
codex-tasks-sync:
    {{server_env}} && uv run python scripts/codex_task_farm.py sync-manifest

# Check Codex task farm prerequisites
codex-tasks-doctor *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py doctor {{arguments}}

# List Codex task farm tasks
codex-tasks-list *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py list {{arguments}}

# Launch Codex task farm worker agents
codex-tasks-run *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py run {{arguments}}

# Show Codex task farm status
codex-tasks-status *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py status {{arguments}}

# Review one or more Codex task branches
codex-tasks-review +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py review {{arguments}}

# Integrate one or more reviewed Codex task branches into main
codex-tasks-integrate +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py integrate {{arguments}}

# Integrate all reviewed Codex task branches into the integration worktree in order
codex-tasks-integrate-ready *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py integrate-ready {{arguments}}

# Generate docs/code-review-2.tasks.json from docs/02.code-review-2-06-26.md
codex-review2-sync:
    {{server_env}} && uv run python scripts/codex_task_farm.py sync-manifest --source docs/02.code-review-2-06-26.md --manifest docs/code-review-2.tasks.json --plan docs/code-review-2-plan.md --state-dir .codex-task-worktrees/code-review-2 --branch-prefix codex/review2- --worktree-prefix ../g-worktrees/review2- --integration-branch integration/code-review-2 --integration-worktree ../g-worktrees/integration-code-review-2

# Check Review 2 task farm prerequisites
codex-review2-doctor *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json doctor {{arguments}}

# List Review 2 tasks
codex-review2-list *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json list {{arguments}}

# Claim Review 2 tasks without launching workers
codex-review2-claim *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json claim {{arguments}}

# Launch Review 2 worker agents
codex-review2-run *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json run {{arguments}}

# Show Review 2 status
codex-review2-status *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json status {{arguments}}

# Review one or more Review 2 task branches
codex-review2-review +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json review {{arguments}}

# Integrate one or more reviewed Review 2 task branches
codex-review2-integrate +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json integrate {{arguments}}

# Integrate all reviewed Review 2 task branches in order
codex-review2-integrate-ready *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json integrate-ready {{arguments}}

# Show Review 2 task branch diffs
codex-review2-diff +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json diff {{arguments}}

# Show Review 2 runtime logs
codex-review2-log +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json log {{arguments}}

# Mark Review 2 tasks blocked
codex-review2-block +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json block {{arguments}}

# Mark Review 2 tasks abandoned
codex-review2-abandon +arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json abandon {{arguments}}

# Reset stale Review 2 claims
codex-review2-reset-claim *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json reset-claim {{arguments}}

# Remove worktrees for integrated Review 2 tasks
codex-review2-clean-integrated *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json clean-integrated {{arguments}}

# Promote Review 2 integration branch to main
codex-review2-promote-to-main *arguments:
    {{server_env}} && uv run python scripts/codex_task_farm.py --manifest docs/code-review-2.tasks.json promote-to-main {{arguments}}

upgrade-python-deps:
    {{server_env}} && uv sync -U --group dev --group gpu

upgrade-nix-lock:
    nix flake update

upgrade-deps: upgrade-python-deps
