# Thin command entrypoints for GWAS Engine (g)

set shell := ["bash", "-cu"]

python_version := env_var_or_default('GWAS_ENGINE_PYTHON_VERSION', '3.14')
tools_dir := env_var_or_default('GWAS_ENGINE_TOOLS_DIR', '.tools')
cuda_repository_url := env_var_or_default('GWAS_ENGINE_CUDA_REPOSITORY_URL', 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64')
nsight_compute_cuda_version := env_var_or_default('GWAS_ENGINE_NSIGHT_COMPUTE_CUDA_VERSION', '12.2')
regenie_patched_source_dir := env_var_or_default('GWAS_ENGINE_REGENIE_PATCHED_SOURCE_DIR', 'reference/regenie-patched')
regenie_patched_output_dir := env_var_or_default('GWAS_ENGINE_REGENIE_PATCHED_OUTPUT_DIR', '.tools/regenie-patched/native')
slurm_gpu_node := env_var_or_default('GWAS_ENGINE_GPU_NODE', 'landau')
slurm_cpu_node := env_var_or_default('GWAS_ENGINE_CPU_NODE', 'cantor')
slurm_gpu_partition := env_var_or_default('GWAS_ENGINE_GPU_PARTITION', env_var_or_default('GWAS_ENGINE_SLURM_PARTITION', ''))
slurm_gpu_account := env_var_or_default('GWAS_ENGINE_GPU_ACCOUNT', env_var_or_default('GWAS_ENGINE_SLURM_ACCOUNT', ''))
slurm_gpu_time_limit := env_var_or_default('GWAS_ENGINE_GPU_TIME', env_var_or_default('GWAS_ENGINE_SLURM_TIME', '04:00:00'))
slurm_gpu_cpus_per_task := env_var_or_default('GWAS_ENGINE_GPU_CPUS_PER_TASK', env_var_or_default('GWAS_ENGINE_SLURM_CPUS_PER_TASK', '8'))
slurm_gpu_memory := env_var_or_default('GWAS_ENGINE_GPU_MEMORY', env_var_or_default('GWAS_ENGINE_SLURM_MEMORY', '64G'))
slurm_gpu_count := env_var_or_default('GWAS_ENGINE_GPU_GPUS_PER_TASK', env_var_or_default('GWAS_ENGINE_SLURM_GPUS_PER_TASK', '1'))
slurm_gpu_extra_arguments := env_var_or_default('GWAS_ENGINE_GPU_EXTRA_ARGS', env_var_or_default('GWAS_ENGINE_SLURM_EXTRA_ARGS', ''))
slurm_cpu_partition := env_var_or_default('GWAS_ENGINE_CPU_PARTITION', env_var_or_default('GWAS_ENGINE_SLURM_PARTITION', ''))
slurm_cpu_account := env_var_or_default('GWAS_ENGINE_CPU_ACCOUNT', env_var_or_default('GWAS_ENGINE_SLURM_ACCOUNT', ''))
slurm_cpu_time_limit := env_var_or_default('GWAS_ENGINE_CPU_TIME', env_var_or_default('GWAS_ENGINE_SLURM_TIME', '04:00:00'))
slurm_cpu_cpus_per_task := env_var_or_default('GWAS_ENGINE_CPU_CPUS_PER_TASK', env_var_or_default('GWAS_ENGINE_SLURM_CPUS_PER_TASK', '40'))
slurm_cpu_memory := env_var_or_default('GWAS_ENGINE_CPU_MEMORY', env_var_or_default('GWAS_ENGINE_SLURM_MEMORY', '128G'))
slurm_cpu_extra_arguments := env_var_or_default('GWAS_ENGINE_CPU_EXTRA_ARGS', env_var_or_default('GWAS_ENGINE_SLURM_EXTRA_ARGS', ''))
slurm_cpu_exclusive := env_var_or_default('GWAS_ENGINE_SLURM_EXCLUSIVE', '1')
server_env := '. tooling/server/server_env.sh'
symphony_elixir_dir := env_var_or_default('SYMPHONY_ELIXIR_DIR', '/mnt/beegfs/kirill/Projects/symphony/elixir')
symphony_port := env_var_or_default('SYMPHONY_PORT', '4000')
symphony_worktree_root := env_var_or_default('SYMPHONY_WORKTREE_ROOT', '/mnt/beegfs/kirill/Projects/g-worktrees/symphony')
cuda_native_sources := 'native/cuda-driver/cuda_driver.h crates/compute-cuda/native/firth_components_ffi.cc crates/compute-cuda/native/firth_components_kernel.cu crates/genotype-cuda/native/nvcomp_abi.h crates/genotype-cuda/native/packed8_deflate_ffi.cc crates/genotype-cuda/native/packed8_kernel.cu'

default: help

# Show available recipes and command-reference location
help:
    @printf 'GWAS Engine command reference: documentation/development/justfile.md\n\n'
    @just --list --unsorted

# --- data ---

# Download local 1KG fixture data
data-fetch:
    {{ server_env }} && uv run python -m tooling.cli.data --config-name data_fetch

# Simulate local phenotypes and covariates
data-simulate: data-fetch
    {{ server_env }} && uv run python -m tooling.cli.data --config-name data_simulate

# Fetch data and simulate phenotypes
data-prepare: data-simulate

# Generate binary REGENIE step 1 predictions
data-baseline-binary: data-prepare
    {{ server_env }} && uv run --no-sync python -m tooling.cli.data --config-name data_baseline_binary

# Generate quantitative REGENIE step 1 predictions
data-baseline-qt: data-prepare
    {{ server_env }} && uv run --no-sync python -m tooling.cli.data --config-name data_baseline_quantitative

# Verify binary GPU step 2 inputs
data-verify-binary-gpu-inputs:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.data --config-name data_verify_binary_gpu_inputs

# Build the patched REGENIE reference binary with native CPU performance flags
data-build-patched-regenie:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    source_directory="{{ regenie_patched_source_dir }}"
    output_directory="{{ regenie_patched_output_dir }}"
    bgen_path="${GWAS_ENGINE_REGENIE_BGEN_PATH:-${BGEN_PATH:-}}"
    if [[ -z "${bgen_path}" ]]; then
      echo "Set GWAS_ENGINE_REGENIE_BGEN_PATH or BGEN_PATH to the external BGEN library root." >&2
      exit 1
    fi
    if [[ ! -d "${source_directory}" ]]; then
      echo "Patched REGENIE source directory does not exist: ${source_directory}" >&2
      exit 1
    fi
    if [[ ! -d "${bgen_path}" ]]; then
      echo "BGEN library directory does not exist: ${bgen_path}" >&2
      exit 1
    fi
    source_path="$(cd "${source_directory}" && pwd -P)"
    bgen_path="$(cd "${bgen_path}" && pwd -P)"
    if [[ ! -f "${bgen_path}/build/libbgen.a" ]]; then
      echo "BGEN library archive not found: ${bgen_path}/build/libbgen.a" >&2
      exit 1
    fi
    case "${output_directory}" in
      /*) output_root="${output_directory}" ;;
      *) output_root="${PWD}/${output_directory}" ;;
    esac
    mkdir -p "${output_root}"
    output_path="${output_root}/regenie"
    job_count="${GWAS_ENGINE_REGENIE_BUILD_JOBS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$(nproc)}}}"
    job_count="${job_count%%(*}"
    if [[ ! "${job_count}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Resolved invalid REGENIE build job count: ${job_count}" >&2
      exit 1
    fi
    compiler="${GWAS_ENGINE_REGENIE_CXX:-${CXX:-g++}}"
    if ! command -v "${compiler}" >/dev/null 2>&1; then
      echo "C++ compiler not found on PATH: ${compiler}" >&2
      exit 1
    fi
    required_flags="-fopenmp"
    performance_flags="${GWAS_ENGINE_REGENIE_PERF_FLAGS:--march=native -mtune=native -flto -DNDEBUG}"
    extra_flags="${GWAS_ENGINE_REGENIE_EXTRA_CFLAGS:-}"
    combined_flags="${required_flags} ${performance_flags}${extra_flags:+ ${extra_flags}}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${job_count}}"
    if [[ "${GWAS_ENGINE_REGENIE_SKIP_CLEAN:-0}" != "1" ]]; then
      make -C "${source_path}" clean EFILE="${output_path}"
    fi
    make -C "${source_path}" --jobs="${job_count}" \
      BGEN_PATH="${bgen_path}" \
      EFILE="${output_path}" \
      CXX="${compiler}" \
      CFLAGS="${combined_flags}" \
      HAS_BOOST_IOSTREAM="${HAS_BOOST_IOSTREAM:-0}" \
      STATIC="${STATIC:-0}" \
      MKLROOT="${MKLROOT:-}" \
      OPENBLAS_ROOT="${OPENBLAS_ROOT:-}" \
      HTSLIB_PATH="${HTSLIB_PATH:-}"
    test -x "${output_path}"
    echo "Built patched REGENIE native binary at ${output_path}"

# --- development ---

# Install repo-local command-line tools for the Ubuntu SLURM server
server-setup-tools:
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/g-uv-cache}" UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv run --no-project --with 'hydra-core>=1.3.2' --with 'pooch>=1.8.2' python -m tooling.cli.server --config-name server_bootstrap_tools

# Bootstrap a CPU-only development environment on the login node
dev-bootstrap:
    {{ server_env }} && uv python install {{ python_version }}
    {{ server_env }} && uv sync --python {{ python_version }} --group dev

# Bootstrap a GPU-capable development environment for JAX CUDA work
dev-bootstrap-gpu:
    {{ server_env }} && uv python install {{ python_version }}
    {{ server_env }} && uv sync --python {{ python_version }} --group dev

# Install CUDA-capable Python dependencies into the current environment
dev-install-gpu-dependencies:
    {{ server_env }} && uv sync --python {{ python_version }} --group dev --frozen --no-install-project --inexact

# Install the native extension for development
dev-install:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && uv run --no-sync maturin develop --profile dev --uv

# Install the native extension using the maximum-performance release profile
dev-install-release:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && uv run --no-sync maturin develop --profile release --uv

# Install optional user-local profiler CLIs used by deep app profiling
dev-install-profiling-tools:
    {{ server_env }} && uv tool install py-spy
    {{ server_env }} && uv tool install scalene
    {{ server_env }} && uv tool install memray
    {{ server_env }} && uv tool install xprof
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && cargo install --locked samply flamegraph

# Install Nsight Systems and Nsight Compute into the repo-local tool directory
dev-install-nsight-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    if command -v nsys >/dev/null 2>&1 && command -v ncu >/dev/null 2>&1; then
      echo "Nsight Systems and Nsight Compute are already available on PATH."
      exit 0
    fi
    uv run --no-sync python -m tooling.cli.server --config-name server_nsight_tools "tool.repository_url='{{ cuda_repository_url }}'" "tool.nsight_compute_cuda_version='{{ nsight_compute_cuda_version }}'"

# Check local toolchain prerequisites for development on the current host
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    required_commands=(uv cargo rustc)
    if [[ "$(uname -s)" == "Linux" ]]; then
      required_commands+=(cc mold ld.mold)
    fi
    for command_name in "${required_commands[@]}"; do
      if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
      fi
    done
    gwas_engine_verify_mold_linker
    resolved_python_version="$(
      uv run --python {{ python_version }} python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"
    if [[ "${resolved_python_version}" != "{{ python_version }}" ]]; then
      echo "Resolved Python ${resolved_python_version}, expected {{ python_version }}." >&2
      exit 1
    fi
    echo "Core development toolchain looks usable on this host."

# Check server development prerequisites, local tools, and cache writability
doctor-server:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    required_commands=(git just uv srun zstd cargo cargo-clippy cargo-fmt rustc rustfmt cc mold ld.mold plink plink2 regenie)
    for command_name in "${required_commands[@]}"; do
      if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required server command: ${command_name}" >&2
        exit 1
      fi
    done
    gwas_engine_verify_mold_linker
    mkdir -p "${UV_CACHE_DIR}"
    test -w "${UV_CACHE_DIR}"
    uv run --python {{ python_version }} python -c 'import sys; print(f"python={sys.version_info.major}.{sys.version_info.minor}")'
    echo "hostname=$(hostname)"
    echo "tools_dir={{ tools_dir }}"
    echo "uv_cache_dir=${UV_CACHE_DIR}"
    echo "Server development toolchain looks usable on this host."

# Check external baseline tools used by data prep and comparison benchmarks
doctor-baselines:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
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
    {{ server_env }} && uv sync --group dev --frozen --no-install-project --inexact
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.performance --config-name performance_jax_runtime

# Check local Symphony prerequisites without starting the daemon
symphony-doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    symphony_env_file="${SYMPHONY_ENV_FILE:-$HOME/.config/g-symphony/env}"
    if [[ -f "${symphony_env_file}" ]]; then
      set -a
      . "${symphony_env_file}"
      set +a
    fi
    if ! command -v python3 >/dev/null 2>&1; then
      echo "FAIL python3 command: not found on PATH" >&2
      echo "  Remediation: Install Python 3 or load the server development environment." >&2
      exit 1
    fi
    python3 scripts/symphony_doctor.py \
      --repository-root "${PWD}" \
      --symphony-env-file "${symphony_env_file}" \
      --symphony-elixir-dir "{{symphony_elixir_dir}}" \
      --symphony-worktree-root "{{symphony_worktree_root}}"

# --- documentation ---

# Serve the Zensical documentation site locally
docs-serve:
    uv sync --group docs --frozen --no-install-project
    uv run --no-sync zensical serve

# Build the Zensical documentation site into documentation_rendered_website/
docs-build:
    uv sync --group docs --frozen --no-install-project
    uv run --no-sync zensical build --clean

# Build documentation and verify dynamic rendering guardrails
docs-check: docs-build
    uv run --no-sync python -m tooling.debug.check_docs_rendering

# --- Symphony ---

# Run Symphony against the repo workflow template
symphony-run:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    symphony_env_file="${SYMPHONY_ENV_FILE:-$HOME/.config/g-symphony/env}"
    if [[ -f "${symphony_env_file}" ]]; then
      set -a
      . "${symphony_env_file}"
      set +a
    fi
    if [[ -z "${LINEAR_API_KEY:-}" ]]; then
      echo "Missing LINEAR_API_KEY. Add it to ${symphony_env_file}." >&2
      exit 1
    fi
    if [[ -z "${LINEAR_PROJECT_SLUG:-}" ]]; then
      echo "Missing LINEAR_PROJECT_SLUG. Add it to ${symphony_env_file}." >&2
      exit 1
    fi
    if [[ ! "${LINEAR_PROJECT_SLUG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "LINEAR_PROJECT_SLUG contains unexpected characters." >&2
      exit 1
    fi
    mkdir -p "{{symphony_worktree_root}}"
    runtime_workflow="${SYMPHONY_RUNTIME_WORKFLOW:-/tmp/g-symphony-${USER:-user}.WORKFLOW.md}"
    escaped_project_slug="$(printf '%s' "${LINEAR_PROJECT_SLUG}" | sed 's/[#&\\]/\\&/g')"
    sed "s#__LINEAR_PROJECT_SLUG__#${escaped_project_slug}#g" WORKFLOW.md > "${runtime_workflow}"
    cd "{{symphony_elixir_dir}}"
    exec mise exec -- ./bin/symphony \
      --i-understand-that-this-will-be-running-without-the-usual-guardrails \
      --port "{{symphony_port}}" \
      "${runtime_workflow}"

# Safely fast-forward the local main checkout after a Symphony direct merge
symphony-sync-main *arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    repository_root="{{ justfile_directory() }}"
    cd "${repository_root}"
    . tooling/server/server_env.sh
    uv run --no-sync python -m tooling.cli.symphony_sync_main --repository "${repository_root}" {{ arguments }}

# Dry-run stale Symphony worktree and branch cleanup
symphony-cleanup *arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    symphony_env_file="${SYMPHONY_ENV_FILE:-$HOME/.config/g-symphony/env}"
    if [[ -f "${symphony_env_file}" ]]; then
      set -a
      . "${symphony_env_file}"
      set +a
    fi
    export SYMPHONY_WORKTREE_ROOT="{{symphony_worktree_root}}"
    uv run python -m tooling.cli.symphony_cleanup --repository "$PWD" --worktree-root "{{symphony_worktree_root}}" {{ arguments }}

# Apply stale Symphony worktree cleanup after reviewing symphony-cleanup
symphony-cleanup-apply *arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    symphony_env_file="${SYMPHONY_ENV_FILE:-$HOME/.config/g-symphony/env}"
    if [[ -f "${symphony_env_file}" ]]; then
      set -a
      . "${symphony_env_file}"
      set +a
    fi
    export SYMPHONY_WORKTREE_ROOT="{{symphony_worktree_root}}"
    uv run python -m tooling.cli.symphony_cleanup --repository "$PWD" --worktree-root "{{symphony_worktree_root}}" --apply {{ arguments }}

# --- SLURM substrates ---

# Start an interactive SLURM shell on the configured GPU node
slurm-gpu-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    repository_root="{{ justfile_directory() }}"
    slurm_arguments=(
      "--ntasks=1"
      "--nodelist={{ slurm_gpu_node }}"
      "--gres=gpu:{{ slurm_gpu_count }}"
      "--cpus-per-task={{ slurm_gpu_cpus_per_task }}"
      "--mem={{ slurm_gpu_memory }}"
      "--time={{ slurm_gpu_time_limit }}"
    )
    if [[ -n "{{ slurm_gpu_partition }}" ]]; then
      slurm_arguments+=("--partition={{ slurm_gpu_partition }}")
    fi
    if [[ -n "{{ slurm_gpu_account }}" ]]; then
      slurm_arguments+=("--account={{ slurm_gpu_account }}")
    fi
    if [[ -n "{{ slurm_gpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_gpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    printf -v job_command 'cd %q && . tooling/server/server_env.sh && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && exec bash -l' "${repository_root}"
    exec srun "${slurm_arguments[@]}" --pty bash -lc "${job_command}"

# Run a shell command through SLURM on the configured GPU node
slurm-gpu-run command:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    repository_root="{{ justfile_directory() }}"
    command='{{ command }}'
    slurm_arguments=(
      "--ntasks=1"
      "--nodelist={{ slurm_gpu_node }}"
      "--gres=gpu:{{ slurm_gpu_count }}"
      "--cpus-per-task={{ slurm_gpu_cpus_per_task }}"
      "--mem={{ slurm_gpu_memory }}"
      "--time={{ slurm_gpu_time_limit }}"
    )
    if [[ -n "{{ slurm_gpu_partition }}" ]]; then
      slurm_arguments+=("--partition={{ slurm_gpu_partition }}")
    fi
    if [[ -n "{{ slurm_gpu_account }}" ]]; then
      slurm_arguments+=("--account={{ slurm_gpu_account }}")
    fi
    if [[ -n "{{ slurm_gpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_gpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    printf -v job_command 'cd %q && . tooling/server/server_env.sh && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && %s' "${repository_root}" "${command}"
    exec srun "${slurm_arguments[@]}" bash -lc "${job_command}"

# Run another just recipe through SLURM on the configured GPU node
slurm-gpu-just +just_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    exec just slurm-gpu-run 'just {{ just_arguments }}'

# Start an interactive SLURM shell on the configured CPU node
slurm-cpu-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    repository_root="{{ justfile_directory() }}"
    slurm_arguments=(
      "--nodes=1"
      "--ntasks=1"
      "--cpus-per-task={{ slurm_cpu_cpus_per_task }}"
      "--mem={{ slurm_cpu_memory }}"
      "--time={{ slurm_cpu_time_limit }}"
    )
    if [[ -n "{{ slurm_cpu_node }}" ]]; then
      slurm_arguments+=("--nodelist={{ slurm_cpu_node }}")
    fi
    if [[ -n "{{ slurm_cpu_partition }}" ]]; then
      slurm_arguments+=("--partition={{ slurm_cpu_partition }}")
    fi
    if [[ -n "{{ slurm_cpu_account }}" ]]; then
      slurm_arguments+=("--account={{ slurm_cpu_account }}")
    fi
    case "{{ slurm_cpu_exclusive }}" in
      "" | 0 | false | False | no | No) ;;
      *) slurm_arguments+=("--exclusive") ;;
    esac
    if [[ -n "{{ slurm_cpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_cpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    printf -v job_command 'cd %q && . tooling/server/server_env.sh && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && echo "GWAS_ENGINE_ALLOCATED_CPU_COUNT=${GWAS_ENGINE_ALLOCATED_CPU_COUNT}" && echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}" && echo "GWAS_ENGINE_PYTEST_WORKERS=${GWAS_ENGINE_PYTEST_WORKERS}" && exec bash -l' "${repository_root}"
    exec srun "${slurm_arguments[@]}" --pty bash -lc "${job_command}"

# Run a shell command through SLURM on the configured CPU node
slurm-cpu-run command:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    repository_root="{{ justfile_directory() }}"
    command='{{ command }}'
    slurm_arguments=(
      "--nodes=1"
      "--ntasks=1"
      "--cpus-per-task={{ slurm_cpu_cpus_per_task }}"
      "--mem={{ slurm_cpu_memory }}"
      "--time={{ slurm_cpu_time_limit }}"
    )
    if [[ -n "{{ slurm_cpu_node }}" ]]; then
      slurm_arguments+=("--nodelist={{ slurm_cpu_node }}")
    fi
    if [[ -n "{{ slurm_cpu_partition }}" ]]; then
      slurm_arguments+=("--partition={{ slurm_cpu_partition }}")
    fi
    if [[ -n "{{ slurm_cpu_account }}" ]]; then
      slurm_arguments+=("--account={{ slurm_cpu_account }}")
    fi
    case "{{ slurm_cpu_exclusive }}" in
      "" | 0 | false | False | no | No) ;;
      *) slurm_arguments+=("--exclusive") ;;
    esac
    if [[ -n "{{ slurm_cpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_cpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    printf -v job_command 'cd %q && . tooling/server/server_env.sh && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && %s' "${repository_root}" "${command}"
    exec srun "${slurm_arguments[@]}" bash -lc "${job_command}"

# Run another just recipe through SLURM on the configured CPU node
slurm-cpu-just +just_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    exec just slurm-cpu-run 'just {{ just_arguments }}'

# Build patched REGENIE on all cores of a named SLURM node
slurm-cpu-build-patched-regenie node='':
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    build_node="{{ node }}"
    if [[ -z "${build_node}" ]]; then
      build_node="${GWAS_ENGINE_REGENIE_BUILD_NODE:-{{ slurm_cpu_node }}}"
    fi
    export GWAS_ENGINE_CPU_NODE="${build_node}"
    exec just slurm-cpu-run "GWAS_ENGINE_REGENIE_BUILD_NODE=${build_node} just data-build-patched-regenie"

# --- matrices ---

# Dry-run the standard chr10 binary/linear CPU/GPU/cache step 2 matrix
matrix-chr10-dry *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name matrix_chr10_dry {{ overrides }}

# Run the standard chr10 binary/linear CPU/GPU/cache step 2 matrix
matrix-chr10 *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name matrix_chr10 {{ overrides }}

# Run a small chr10 matrix smoke
matrix-chr10-smoke *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name matrix_chr10_smoke {{ overrides }}

# Submit the standard chr10 matrix through SLURM on the configured GPU node
slurm-gpu-matrix-chr10 *overrides:
    {{ server_env }} && just slurm-gpu-just matrix-chr10 {{ overrides }}

# Dry-run the standard chr22 binary/linear CPU/GPU/cache step 2 matrix
matrix-chr22-dry *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name matrix_chr22_dry {{ overrides }}

# Run the standard chr22 binary/linear CPU/GPU/cache step 2 matrix
matrix-chr22 *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name matrix_chr22 {{ overrides }}

# Run a small chr22 matrix smoke
matrix-chr22-smoke *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name matrix_chr22_smoke {{ overrides }}

# Submit the standard chr22 matrix through SLURM on the configured GPU node
slurm-gpu-matrix-chr22 *overrides:
    {{ server_env }} && just slurm-gpu-just matrix-chr22 {{ overrides }}

# --- benchmarks ---

# Run historical external baselines excluding the slow Hail suite
legacy-baselines: data-prepare
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name benchmark_baselines

# Run historical external baselines including the slow Hail suite
legacy-baselines-full: data-prepare
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name benchmark_baselines tool.include_hail=true

# Compare original regenie vs g quantitative step2 on CPU
legacy-regenie-comparison-cpu: data-prepare dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name benchmark_regenie_comparison

# Compare original regenie vs g quantitative step2 with GPU enabled
legacy-regenie-comparison-gpu: data-prepare dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name benchmark_regenie_comparison machine=landau_gpu tool.cpu_only=false tool.include_gpu=true

# Profile historical regenie comparison on CPU
legacy-profile-regenie-comparison-cpu: data-prepare dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name benchmark_profile_comparison

# Profile historical regenie comparison with GPU enabled
legacy-profile-regenie-comparison-gpu: data-prepare dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name benchmark_profile_comparison machine=landau_gpu tool.cpu_only=false tool.include_gpu=true

# Benchmark BGEN float32 read paths
bench-bgen-reader *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_bgen_reader --config-name bench_bgen_reader {{ overrides }}

# Benchmark Python callback overhead without BGEN decode work
bench-callback-overhead *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_callback_overhead --config-name bench_callback_overhead {{ overrides }}

# Benchmark Python callback overhead on the configured GPU machine profile
bench-callback-overhead-gpu *overrides: dev-install-gpu-dependencies
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_callback_overhead --config-name bench_callback_overhead_gpu {{ overrides }}

# Benchmark the already-compiled approximate-Firth executable
bench-firth-compute-gpu *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_firth_compute --config-name benchmark_firth_compute {{ overrides }}

# Benchmark REGENIE step 2 in fresh Python processes
bench-linear-startup-gpu: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name bench_linear_startup_gpu

# Benchmark REGENIE step 2 fresh process startup with Parquet finalization
bench-linear-startup-gpu-parquet: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark --config-name bench_linear_startup_gpu_parquet

# Benchmark binary REGENIE step 2 with cold, hot, chunk-only, and finalized timings
bench-binary-hot-gpu *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot --config-name bench_binary_hot_gpu {{ overrides }}

# Smoke test the binary REGENIE step 2 benchmark harness
bench-binary-hot-gpu-smoke *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot --config-name bench_binary_hot_gpu_smoke {{ overrides }}

# Benchmark output-stage timings across finalization, phenotype count, and bsize
bench-output-stages-gpu *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_output_stages --config-name bench_output_stages_gpu {{ overrides }}

# Benchmark single-trait chr22 linear GWAS against TorchGWAS
bench-torchgwas-chr22 *overrides: dev-install-gpu-dependencies dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_torchgwas_chr22 --config-name benchmark_torchgwas_chr22 {{ overrides }}

# Benchmark single-trait chr22 nominal dense association against tensorQTL
bench-tensorqtl-chr22 *overrides: dev-install-gpu-dependencies dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_tensorqtl_chr22 --config-name benchmark_tensorqtl_chr22 {{ overrides }}

# Submit binary hot benchmark to the configured GPU node
slurm-gpu-bench-binary-hot *overrides:
    {{ server_env }} && just slurm-gpu-just bench-binary-hot-gpu {{ overrides }}

# Submit the TorchGWAS chr22 benchmark to the configured GPU node
slurm-gpu-bench-torchgwas-chr22 *overrides:
    {{ server_env }} && just slurm-gpu-just bench-torchgwas-chr22 {{ overrides }}

# Submit the tensorQTL chr22 benchmark to the configured GPU node
slurm-gpu-bench-tensorqtl-chr22 *overrides:
    {{ server_env }} && just slurm-gpu-just bench-tensorqtl-chr22 {{ overrides }}

# Submit callback overhead microbenchmark to the configured CPU node
slurm-cpu-bench-callback-overhead *overrides:
    {{ server_env }} && just slurm-cpu-just bench-callback-overhead {{ overrides }}

# Submit callback overhead microbenchmark to the configured GPU node
slurm-gpu-bench-callback-overhead *overrides:
    {{ server_env }} && just slurm-gpu-just bench-callback-overhead-gpu {{ overrides }}

# Submit the focused approximate-Firth executable benchmark to the configured GPU node
slurm-gpu-bench-firth-compute *overrides:
    {{ server_env }} && just slurm-gpu-just bench-firth-compute-gpu {{ overrides }}

# --- performance ---

# Run the login-node-safe performance harness smoke benchmark
perf-smoke *arguments:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.performance --config-name performance_smoke {{ arguments }}

# Submit the standard CPU performance benchmark through the configured CPU SLURM node
perf-cpu *overrides:
    {{ server_env }} && just slurm-cpu-run '{{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_bgen_reader --config-name perf_cpu {{ overrides }}'

# Submit the standard GPU performance benchmark through the configured GPU SLURM node
perf-gpu *overrides:
    {{ server_env }} && just slurm-gpu-run '{{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot --config-name perf_gpu {{ overrides }}'

# Compare two benchmark JSON summaries
perf-compare baseline_json new_json:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.performance --config-name performance_compare tool.baseline_json='{{ baseline_json }}' tool.new_json='{{ new_json }}'

# Run CPU/GPU JAX runtime probe
perf-jax-runtime:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project --inexact
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.performance --config-name performance_jax_runtime

# Sequentially tune GPU REGENIE step 2 and active BGEN reader knobs
perf-tune-regenie2-gpu *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.tune_regenie2_gpu --config-name tune_regenie2_gpu {{ overrides }}

# Compare native extension build profiles and write timing reports
bench-rust-build-profiles *overrides:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && uv run --no-sync python -m tooling.cli.rust_build_profiles --config-name rust_build_profiles {{ overrides }}

# --- profiling ---

# Run native Criterion profiles on the configured CPU compute node
profile-rust-criterion:
    {{ server_env }} && GWAS_ENGINE_DATA_DIR="${GWAS_ENGINE_DATA_DIR:-{{ justfile_directory() }}/data}" just slurm-cpu-run 'cargo bench -p g-genotype --bench bgen_read'

# Run the deep REGENIE step 2 profiling harness on the current host
profile-deep *overrides: dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_regenie2_deep {{ overrides }}

# Write the deep REGENIE step 2 profiling plan without running workloads
profile-deep-dry *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_regenie2_deep tool.dry_run=true {{ overrides }}

# Smoke test the deep REGENIE step 2 profiling harness on the current host
profile-deep-smoke *overrides: dev-install-gpu-dependencies dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_regenie2_deep tool.smoke=true tool.skip_deep_profiles=true tool.enable_rust_criterion=false {{ overrides }}

# Write the full app profiling plan without running workloads
profile-app-full-dry *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_app_full_dry {{ overrides }}

# Smoke test the full app profiling bundle on the current host
profile-app-full-smoke *overrides: dev-install-gpu-dependencies dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_app_full_smoke {{ overrides }}

# Submit the full app profiling bundle through SLURM
profile-app-full *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    export GWAS_ENGINE_SLURM_TIME="${GWAS_ENGINE_SLURM_TIME:-12:00:00}"
    export GWAS_ENGINE_SLURM_CPUS_PER_TASK="${GWAS_ENGINE_SLURM_CPUS_PER_TASK:-8}"
    export GWAS_ENGINE_SLURM_MEMORY="${GWAS_ENGINE_SLURM_MEMORY:-64G}"
    export GWAS_ENGINE_SLURM_GPUS_PER_TASK="${GWAS_ENGINE_SLURM_GPUS_PER_TASK:-1}"
    exec just slurm-gpu-run '. tooling/server/server_env.sh && just dev-install-gpu-dependencies && just dev-install-release && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_app_full {{ overrides }}'

# Dry-run the chr10 GPU binary profiling campaign
profile-chr10-binary-gpu-dry *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_chr10_binary_gpu_dry {{ overrides }}

# Smoke test the chr10 GPU binary profiling campaign
profile-chr10-binary-gpu-smoke *overrides: dev-install-gpu-dependencies dev-install-release
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_chr10_binary_gpu_smoke {{ overrides }}

# Submit the chr10 GPU binary full profiling campaign through SLURM
profile-chr10-binary-gpu-full *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    export GWAS_ENGINE_SLURM_TIME="${GWAS_ENGINE_SLURM_TIME:-12:00:00}"
    export GWAS_ENGINE_SLURM_CPUS_PER_TASK="${GWAS_ENGINE_SLURM_CPUS_PER_TASK:-8}"
    export GWAS_ENGINE_SLURM_MEMORY="${GWAS_ENGINE_SLURM_MEMORY:-64G}"
    export GWAS_ENGINE_SLURM_GPUS_PER_TASK="${GWAS_ENGINE_SLURM_GPUS_PER_TASK:-1}"
    exec just slurm-gpu-run '. tooling/server/server_env.sh && just dev-install-gpu-dependencies && just dev-install-release && just dev-install-profiling-tools && just dev-install-nsight-tools && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_chr10_binary_gpu_full {{ overrides }}'

# Dry-run the focused chr22 GPU binary profiling campaign
profile-chr22-binary-gpu-dry *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_chr22_binary_gpu_dry {{ overrides }}

# Submit the focused chr22 GPU binary full profiling campaign through SLURM
profile-chr22-binary-gpu-full *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    export GWAS_ENGINE_SLURM_TIME="${GWAS_ENGINE_SLURM_TIME:-12:00:00}"
    export GWAS_ENGINE_SLURM_CPUS_PER_TASK="${GWAS_ENGINE_SLURM_CPUS_PER_TASK:-8}"
    export GWAS_ENGINE_SLURM_MEMORY="${GWAS_ENGINE_SLURM_MEMORY:-64G}"
    export GWAS_ENGINE_SLURM_GPUS_PER_TASK="${GWAS_ENGINE_SLURM_GPUS_PER_TASK:-1}"
    exec just slurm-gpu-run '. tooling/server/server_env.sh && just dev-install-gpu-dependencies && just dev-install-release && just dev-install-profiling-tools && just dev-install-nsight-tools && uv run --no-sync python -m tooling.cli.profile_regenie2_deep --config-name profile_chr22_binary_gpu_full {{ overrides }}'

# --- checks and tests ---

# Format maintained CUDA and native C++ sources
cuda-format:
    uv sync --group cuda-format --frozen --no-install-project
    uv run --no-sync clang-format --style=file -i {{ cuda_native_sources }}

# Check maintained CUDA and native C++ formatting without rewriting files
cuda-format-check:
    uv sync --group cuda-format --frozen --no-install-project
    uv run --no-sync clang-format --style=file --dry-run --Werror {{ cuda_native_sources }}

# Lint maintained CUDA and native C++ sources without requiring a GPU
cuda-lint:
    uv sync --group dev --group cuda-lint --frozen --no-install-project
    PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_cuda_native

# Format code
format: cuda-format
    {{ server_env }} && uv run ruff format .
    {{ server_env }} && cargo fmt

# Check Rust formatting without rewriting files
rust-format-check:
    {{ server_env }} && cargo fmt --all --check

# Lint code
lint: cuda-lint
    {{ server_env }} && uv run ruff check . --fix
    {{ server_env }} && cargo clippy --workspace --all-targets -- -W clippy::pedantic

# Check Rust lints without rewriting files
rust-lint-check:
    {{ server_env }} && cargo clippy --workspace --all-targets -- -W clippy::pedantic

# Type check Python code
typecheck:
    {{ server_env }} && uv run ty check src tests scripts tooling

# Verify Rust workspace dependency boundaries
check-rust-architecture:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_rust_architecture

# Verify Python package ownership boundaries
check-python-architecture:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_python_architecture

# Verify Python type stub exports are in sync with Rust `_core` registrations
check-core-stub:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_pyo3_stub

# Verify production Python code does not hide runtime policy in defaults
check-internal-defaults:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_internal_defaults

# Verify internal package initializers do not re-export aliases
check-internal-init-exports:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_internal_init_exports

# Verify the Justfile remains a thin config-backed command layer
check-justfile:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.debug --config-name debug_check_justfile

# Validate a Tooling Artifact Format artifact directory or JSON file
check-artifact-schema path:
    {{ server_env }} && PYTHONPATH=src:. uv run --no-sync python -m tooling.cli.schema_check --config-name schema_check tool.path='{{ path }}'

# Run all checks
check: format lint typecheck check-core-stub check-internal-defaults check-internal-init-exports check-rust-architecture check-python-architecture check-justfile

# Check formatting without requiring Nix or direct Cargo access
format-local-check: cuda-format-check
    uv run ruff format --check .

# Lint Python and native CUDA/C++ without applying fixes
lint-local: cuda-lint
    uv run ruff check .

# Type check Python in uv/maturin-only environments
typecheck-local:
    uv run ty check src tests scripts tooling

# Focused no-Nix smoke tests that also rebuild the native extension through maturin
test-local-focused:
    uv run pytest tests/test_core.py tests/test_io_output.py

# Non-heavy no-Nix test suite
test-local:
    uv run pytest tests/ -m "not phase0_data and not phase1_parity"

# Local no-Nix verification lane
check-local: format-local-check lint-local typecheck-local test-local-focused check-core-stub check-internal-defaults check-internal-init-exports check-rust-architecture check-python-architecture check-justfile

# Run CI lint checks without installing the project package
ci-lint:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && uv run --no-sync ruff check .

# Run CI type checks without installing the project package
ci-typecheck:
    {{ server_env }} && uv sync --group dev --frozen --no-install-project
    {{ server_env }} && uv run --no-sync ty check src tests scripts tooling

# Run CI tests that exclude heavy data- and parity-dependent suites
ci-test:
    {{ server_env }} && uv sync --group dev --frozen
    {{ server_env }} && uv run --no-sync pytest tests/ -m "not phase0_data and not phase1_parity"

# Run CPU-focused tests excluding data/parity workloads
test-cpu:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    pytest_arguments=()
    if gwas_engine_is_positive_integer "${GWAS_ENGINE_PYTEST_WORKERS:-}" && [[ "${GWAS_ENGINE_PYTEST_WORKERS}" -gt 1 ]]; then
      gwas_engine_configure_parallel_pytest_thread_limits
      pytest_arguments+=("-n" "${GWAS_ENGINE_PYTEST_WORKERS}")
    fi
    uv run pytest tests/ -m "not phase0_data and not phase1_parity" "${pytest_arguments[@]}"

# Run tests, using xdist when GWAS_ENGINE_PYTEST_WORKERS is configured
test:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    pytest_arguments=()
    if gwas_engine_is_positive_integer "${GWAS_ENGINE_PYTEST_WORKERS:-}" && [[ "${GWAS_ENGINE_PYTEST_WORKERS}" -gt 1 ]]; then
      gwas_engine_configure_parallel_pytest_thread_limits
      pytest_arguments+=("-n" "${GWAS_ENGINE_PYTEST_WORKERS}")
    fi
    uv run pytest tests/ "${pytest_arguments[@]}"

# Explicit full Python test-suite alias
test-full: test

# Run Python coverage gate
coverage-python:
    #!/usr/bin/env bash
    set -euo pipefail
    . tooling/server/server_env.sh
    pytest_arguments=()
    if gwas_engine_is_positive_integer "${GWAS_ENGINE_PYTEST_WORKERS:-}" && [[ "${GWAS_ENGINE_PYTEST_WORKERS}" -gt 1 ]]; then
      gwas_engine_configure_parallel_pytest_thread_limits
      pytest_arguments+=("-n" "${GWAS_ENGINE_PYTEST_WORKERS}")
    fi
    uv run pytest tests/ --cov=src/g --cov-report=term-missing --cov-fail-under=90 "${pytest_arguments[@]}"

# Run Rust line coverage gate
coverage-rust:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && cargo llvm-cov --workspace --all-targets --ignore-filename-regex '(^|/)(benches|tests)/' --fail-under-lines 90

# Run all coverage gates
coverage: coverage-python coverage-rust

# Build all Rust targets
rust-build:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && cargo build --workspace --all-targets

# Run the Rust test suite
rust-test:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && cargo test --workspace

# Run Rust Criterion benchmarks with native performance flags
rust-bench:
    {{ server_env }} && gwas_engine_configure_rust_build_environment && gwas_engine_log_rust_build_environment && cargo bench --workspace

# Run non-mutating Rust format, lint, build, tests, and architecture checks
rust-check: rust-format-check rust-lint-check rust-build rust-test check-rust-architecture

# Run the workspace-level validation lane for Rust migration phases
workspace-check: rust-check

# Run all checks on the configured CPU SLURM node
slurm-cpu-check:
    {{ server_env }} && just slurm-cpu-just check

# Run CPU-focused Python tests on the configured CPU SLURM node
slurm-cpu-test:
    {{ server_env }} && just slurm-cpu-just test-cpu

# Run the full Python test suite on the configured CPU SLURM node
slurm-cpu-test-full:
    {{ server_env }} && just slurm-cpu-just test-full

# Build all Rust targets on the configured CPU SLURM node
slurm-cpu-rust-build:
    {{ server_env }} && just slurm-cpu-just rust-build

# Run Rust tests on the configured CPU SLURM node
slurm-cpu-rust-test:
    {{ server_env }} && just slurm-cpu-just rust-test

# Run coverage on the configured CPU SLURM node
slurm-cpu-coverage:
    {{ server_env }} && just slurm-cpu-just coverage

# Upgrade Python lockfile dependencies, including the development group
upgrade-python-deps:
    {{ server_env }} && uv sync -U --group dev

# Upgrade the Nix flake lockfile
upgrade-nix-lock:
    nix flake update

# Upgrade Python and Nix dependency locks
upgrade-deps: upgrade-python-deps upgrade-nix-lock
