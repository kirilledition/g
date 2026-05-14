# Justfile for GWAS Engine (g)

set allow-duplicate-recipes := true

data_dir := env_var_or_default('GWAS_ENGINE_DATA_DIR', 'data')
python_version := env_var_or_default('GWAS_ENGINE_PYTHON_VERSION', '3.13')
slurm_gpu_node := env_var_or_default('GWAS_ENGINE_GPU_NODE', 'landau')
slurm_partition := env_var_or_default('GWAS_ENGINE_SLURM_PARTITION', '')
slurm_account := env_var_or_default('GWAS_ENGINE_SLURM_ACCOUNT', '')
slurm_time_limit := env_var_or_default('GWAS_ENGINE_SLURM_TIME', '04:00:00')
slurm_cpus_per_task := env_var_or_default('GWAS_ENGINE_SLURM_CPUS_PER_TASK', '8')
slurm_memory := env_var_or_default('GWAS_ENGINE_SLURM_MEMORY', '64G')
slurm_gpu_count := env_var_or_default('GWAS_ENGINE_SLURM_GPUS_PER_TASK', '1')
slurm_extra_arguments := env_var_or_default('GWAS_ENGINE_SLURM_EXTRA_ARGS', '')

# --- Data Preparation ---

setup-data:
    uv run python scripts/fetch_1kg.py
    uv run python scripts/simulate_phenos.py

# Run PLINK2/Regenie baselines and generate hardware report (excludes slow Hail benchmarks by default)
benchmark-baselines: setup-data
    uv run python scripts/benchmark.py

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU
benchmark-regenie-comparison-cpu: setup-data
    uv run python scripts/benchmark_regenie_comparison.py --cpu-only

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU+GPU
benchmark-regenie-comparison-gpu: setup-data
    uv run python scripts/benchmark_regenie_comparison.py --include-gpu

# Alias for comparison benchmark (CPU-only default)
benchmark-regenie-comparison: benchmark-regenie-comparison-cpu

# Run full baselines including Hail (slow - requires cached MatrixTable)
benchmark-baselines-full: setup-data
    HAIL_INCLUDE=1 uv run python scripts/benchmark.py

# --- Development ---

# Bootstrap a CPU-only development environment on the login node
bootstrap:
    uv python install {{python_version}}
    uv sync --python {{python_version}} --group dev

# Bootstrap a GPU-capable development environment for JAX CUDA work
bootstrap-gpu:
    uv python install {{python_version}}
    uv sync --python {{python_version}} --group dev --group gpu

# Check local toolchain prerequisites for development on the current host
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
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

# Check external baseline tools used by data prep and comparison benchmarks
doctor-baselines:
    #!/usr/bin/env bash
    set -euo pipefail
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
    uv run --python {{python_version}} python scripts/probe_jax_runtime.py

# Start an interactive SLURM shell on the configured GPU node
slurm-gpu-shell:
    #!/usr/bin/env bash
    set -euo pipefail
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
    if [[ "$#" -eq 0 ]]; then
      echo "Usage: just slurm-gpu-run <command...>" >&2
      exit 1
    fi
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
    exec srun "${slurm_arguments[@]}" "$@"

# Run another just recipe through SLURM on the configured GPU node
slurm-gpu-just +just_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$#" -eq 0 ]]; then
      echo "Usage: just slurm-gpu-just <recipe...>" >&2
      exit 1
    fi
    exec just slurm-gpu-run just "$@"

# Run REGENIE step 2 with local baseline predictions
regenie2-linear:
    uv run g regenie2-linear --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --pheno {{data_dir}}/pheno_cont.txt --pheno-name phenotype_continuous --covar {{data_dir}}/covariates.txt --covar-names age,sex --pred {{data_dir}}/baselines/regenie_step1_qt_pred.list --out {{data_dir}}/regenie2_linear

# Run CPU/GPU JAX runtime probe
probe-jax:
    uv run python scripts/probe_jax_runtime.py

# Benchmark PLINK reader and preprocessing paths
benchmark-plink-reader:
    uv run python scripts/benchmark_plink_reader.py

# Benchmark BGEN float32 read paths
benchmark-bgen-reader:
    uv run python scripts/benchmark_bgen_reader.py

# Benchmark REGENIE step 2 in fresh Python processes
benchmark-regenie2-linear-fresh-gpu:
    uv run python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu

# Benchmark REGENIE step 2 in fresh Python processes using Arrow chunks + Parquet finalization
benchmark-regenie2-linear-fresh-gpu-parquet:
    uv run python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu --finalize-parquet

# Sequentially tune GPU REGENIE step 2 and active BGEN reader knobs
tune-regenie2-gpu:
    uv run python scripts/tune_regenie2_gpu.py

# Profile full REGENIE step 2 execution
profile-regenie2-linear-detailed:
    mkdir -p {{data_dir}}/profiles/regenie2_linear_detailed
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 uv run python scripts/profile_regenie2_linear_detailed.py --bgen {{data_dir}}/1kg_chr22_full.bgen --sample {{data_dir}}/1kg_chr22_full.sample --pheno {{data_dir}}/pheno_cont.txt --pheno-name phenotype_continuous --covar {{data_dir}}/covariates.txt --covar-names age,sex --pred {{data_dir}}/baselines/regenie_step1_qt_pred.list --output-dir {{data_dir}}/profiles/regenie2_linear_detailed --report-name regenie2_linear_chr22_full --enable-jax-trace --enable-memory-profile --cprofile-sort cumulative

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU
profile-regenie-comparison-cpu: setup-data
    uv run python scripts/profile_regenie_comparison.py --cpu-only

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU+GPU
profile-regenie-comparison-gpu: setup-data
    uv run python scripts/profile_regenie_comparison.py --include-gpu

# Alias for unified profiling comparison (CPU-only default)
profile-regenie-comparison: profile-regenie-comparison-cpu

# Format code
format:
    uv run ruff format .
    cargo fmt

# Lint code
lint:
    uv run ruff check . --fix
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic

# Type check Python code
typecheck:
    uv run ty check src tests scripts

# Run all checks (format, lint, typecheck)
check: format lint typecheck

# Run CI lint checks without installing the project package
ci-lint:
    uv sync --group dev --frozen --no-install-project
    uv run --no-sync ruff check .

# Run CI type checks without installing the project package
ci-typecheck:
    uv sync --group dev --frozen --no-install-project
    uv run --no-sync ty check src tests scripts

# Run CI tests that exclude heavy data- and parity-dependent suites
ci-test:
    uv sync --group dev --frozen
    uv run --no-sync pytest tests/ -m "not phase0_data and not phase1_parity"

# Run tests
test:
    uv run pytest tests/

upgrade-python-deps:
    uv sync -U --group dev --group gpu

upgrade-nix-lock:
    nix flake update

upgrade-deps: upgrade-python-deps
