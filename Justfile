# Justfile for GWAS Engine (g)

set shell := ["bash", "-cu"]

data_dir := env_var_or_default('GWAS_ENGINE_DATA_DIR', 'data')
python_version := env_var_or_default('GWAS_ENGINE_PYTHON_VERSION', '3.14')
tools_dir := env_var_or_default('GWAS_ENGINE_TOOLS_DIR', '.tools')
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
perf_results_dir := env_var_or_default('GWAS_ENGINE_PERF_RESULTS_DIR', 'results/perf')
deep_profile_landau_budget_overrides := 'tool.chunk_sizes=[2048,4096] tool.staging_depths=[1,2] tool.output_writer_thread_counts=[1,4] tool.writer_queue_depth_multipliers=[1,2] tool.firth_batch_sizes=[32] tool.bgen_decode_tile_variant_counts=[64,128] tool.rayon_thread_counts=[4,8] tool.top_bgen_candidates=1 tool.top_finalists=2 tool.tuning_warmups=0 tool.tuning_trials=1 tool.finalist_warmups=0 tool.finalist_trials=2 tool.headline_warmups=0 tool.headline_trials=3 tool.max_subprocess_runs=1000 tool.max_major_profiler_runs=64'
server_env := '. scripts/server_env.sh'
symphony_elixir_dir := env_var_or_default('SYMPHONY_ELIXIR_DIR', '/mnt/beegfs/kirill/Projects/symphony/elixir')
symphony_port := env_var_or_default('SYMPHONY_PORT', '4000')
symphony_worktree_root := env_var_or_default('SYMPHONY_WORKTREE_ROOT', '/mnt/beegfs/kirill/Projects/g-worktrees/symphony')

# Show available recipes and point to the command reference
default: help

# Show available recipes and command-reference location
help:
    @printf 'GWAS Engine command reference: docs/development/justfile.md\n\n'
    @just --list --unsorted

# --- Data Preparation ---

# Download local 1KG fixture data and simulate phenotypes
setup-data:
    {{ server_env }} && uv run python scripts/fetch_1kg.py
    {{ server_env }} && uv run python scripts/simulate_phenos.py

# Generate binary REGENIE step 1 predictions required by g binary step 2
setup-binary-baseline: setup-data
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    mkdir -p "{{ data_dir }}/baselines"
    regenie \
      --step 1 \
      --bed "{{ data_dir }}/1kg_chr22_full" \
      --phenoFile "{{ data_dir }}/pheno_bin.txt" \
      --covarFile "{{ data_dir }}/covariates.txt" \
      --bt \
      --cc12 \
      --force-step1 \
      --bsize 1000 \
      --out "{{ data_dir }}/baselines/regenie_step1"
    test -s "{{ data_dir }}/baselines/regenie_step1_pred.list"

# Prepare all local inputs required for binary REGENIE step 2 GPU execution
setup-regenie2-binary-gpu-inputs: setup-binary-baseline

# Verify local inputs required for binary REGENIE step 2 GPU execution
verify-regenie2-binary-gpu-inputs:
    #!/usr/bin/env bash
    set -euo pipefail
    test -s "{{ data_dir }}/1kg_chr22_full.bgen"
    test -s "{{ data_dir }}/1kg_chr22_full.sample"
    test -s "{{ data_dir }}/pheno_bin.txt"
    test -s "{{ data_dir }}/covariates.txt"
    test -s "{{ data_dir }}/baselines/regenie_step1_pred.list"
    echo "Binary REGENIE step 2 GPU inputs are present."

# Run PLINK2/Regenie baselines and generate hardware report (excludes slow Hail benchmarks by default)
benchmark-baselines: setup-data
    {{ server_env }} && uv run python scripts/benchmark.py

# Build and install the Rust extension using the opt-in native performance profile
install-perf-extension:
    {{ server_env }} && RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf --uv

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU
benchmark-regenie-comparison-cpu: setup-data install-perf-extension
    {{ server_env }} && uv run --no-sync python scripts/benchmark_regenie_comparison.py --cpu-only

# Compare original regenie (all 4 programs) vs g quantitative step2 CPU+GPU
benchmark-regenie-comparison-gpu: setup-data install-perf-extension
    {{ server_env }} && uv run --no-sync python scripts/benchmark_regenie_comparison.py --include-gpu

# Alias for comparison benchmark (CPU-only default)
benchmark-regenie-comparison: benchmark-regenie-comparison-cpu

# Run full baselines including Hail (slow - requires cached MatrixTable)
benchmark-baselines-full: setup-data
    {{ server_env }} && HAIL_INCLUDE=1 uv run python scripts/benchmark.py

# --- Development ---

# Install repo-local command-line tools for the Ubuntu SLURM server
setup-server-tools:
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/g-uv-cache}" UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv run --no-project python scripts/bootstrap_server_tools.py

# Bootstrap a CPU-only development environment on the login node
bootstrap:
    {{ server_env }} && uv python install {{ python_version }}
    {{ server_env }} && uv sync --python {{ python_version }} --group dev

# Bootstrap a GPU-capable development environment for JAX CUDA work
bootstrap-gpu:
    {{ server_env }} && uv python install {{ python_version }}
    {{ server_env }} && uv sync --python {{ python_version }} --group dev --group gpu

# Install CUDA-capable Python dependencies into the current environment
install-gpu-dependencies:
    {{ server_env }} && uv sync --python {{ python_version }} --group dev --group gpu

# Install optional user-local profiler CLIs used by deep app profiling
install-profiling-tools:
    {{ server_env }} && uv tool install py-spy
    {{ server_env }} && uv tool install scalene
    {{ server_env }} && uv tool install memray
    {{ server_env }} && uv tool install xprof
    {{ server_env }} && cargo install --locked samply flamegraph

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
    . scripts/server_env.sh
    required_commands=(git just uv srun zstd cargo cargo-clippy cargo-fmt rustc rustfmt plink plink2 regenie)
    for command_name in "${required_commands[@]}"; do
      if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required server command: ${command_name}" >&2
        exit 1
      fi
    done
    mkdir -p "${UV_CACHE_DIR}"
    test -w "${UV_CACHE_DIR}"
    uv run --python {{ python_version }} python -c 'import sys; print(f"python={sys.version_info.major}.{sys.version_info.minor}")'
    echo "hostname=$(hostname)"
    echo "tools_dir={{ tools_dir }}"
    echo "uv_cache_dir=${UV_CACHE_DIR}"
    echo "Server development toolchain looks usable on this host."

# Check local Symphony prerequisites without starting the daemon
symphony-doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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

# Serve the Zensical documentation site locally
docs-serve:
    uv run --group docs zensical serve

# Build the Zensical documentation site into site/
docs-build:
    uv run --group docs zensical build --clean

# Run Symphony against the repo workflow template
symphony-run:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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
    . scripts/server_env.sh
    uv run --no-sync python -m tooling.cli.symphony_sync_main --repository "${repository_root}" {{ arguments }}

# Dry-run stale Symphony worktree and branch cleanup
symphony-cleanup *arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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
    . scripts/server_env.sh
    symphony_env_file="${SYMPHONY_ENV_FILE:-$HOME/.config/g-symphony/env}"
    if [[ -f "${symphony_env_file}" ]]; then
      set -a
      . "${symphony_env_file}"
      set +a
    fi
    export SYMPHONY_WORKTREE_ROOT="{{symphony_worktree_root}}"
    uv run python -m tooling.cli.symphony_cleanup --repository "$PWD" --worktree-root "{{symphony_worktree_root}}" --apply {{ arguments }}

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

# Build the patched REGENIE reference binary with native CPU performance flags
build-patched-regenie:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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

# Build patched REGENIE on all cores of a named SLURM node
slurm-build-patched-regenie node='':
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    build_node="{{ node }}"
    if [[ -z "${build_node}" ]]; then
      build_node="${GWAS_ENGINE_REGENIE_BUILD_NODE:-{{ slurm_cpu_node }}}"
    fi
    slurm_arguments=(
      "--nodes=1"
      "--ntasks=1"
      "--exclusive"
      "--mem={{ slurm_cpu_memory }}"
      "--time={{ slurm_cpu_time_limit }}"
    )
    if [[ -n "${build_node}" ]]; then
      slurm_arguments+=("--nodelist=${build_node}")
    fi
    if [[ -n "{{ slurm_cpu_partition }}" ]]; then
      slurm_arguments+=("--partition={{ slurm_cpu_partition }}")
    fi
    if [[ -n "{{ slurm_cpu_account }}" ]]; then
      slurm_arguments+=("--account={{ slurm_cpu_account }}")
    fi
    if [[ -n "{{ slurm_cpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_cpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    repository_root="$(pwd -P)"
    printf -v quoted_repository_root "%q" "${repository_root}"
    exec srun "${slurm_arguments[@]}" bash -lc "cd ${quoted_repository_root} && just build-patched-regenie"

# Probe JAX runtime on the current host
doctor-jax:
    {{ server_env }} && uv run --python {{ python_version }} python scripts/probe_jax_runtime.py

# Start an interactive SLURM shell on the configured GPU node
slurm-gpu-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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
    exec srun "${slurm_arguments[@]}" --pty bash -l

# Run a shell command through SLURM on the configured GPU node
slurm-gpu-run command:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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
    exec srun "${slurm_arguments[@]}" bash -lc "${command}"

# Run another just recipe through SLURM on the configured GPU node
slurm-gpu-just +just_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    exec just slurm-gpu-run 'just {{ just_arguments }}'

# Start an interactive SLURM shell on the configured CPU node
slurm-cpu-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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
      "" | 0 | false | False | no | No)
        ;;
      *)
        slurm_arguments+=("--exclusive")
        ;;
    esac
    if [[ -n "{{ slurm_cpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_cpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    printf -v job_command 'cd %q && . scripts/server_env.sh && gwas_engine_configure_cpu_parallelism && echo "GWAS_ENGINE_ALLOCATED_CPU_COUNT=${GWAS_ENGINE_ALLOCATED_CPU_COUNT}" && echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}" && echo "GWAS_ENGINE_PYTEST_WORKERS=${GWAS_ENGINE_PYTEST_WORKERS}" && exec bash -l' "${repository_root}"
    exec srun "${slurm_arguments[@]}" --pty bash -lc "${job_command}"

# Run a shell command through SLURM on the configured CPU node
slurm-cpu-run command:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
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
      "" | 0 | false | False | no | No)
        ;;
      *)
        slurm_arguments+=("--exclusive")
        ;;
    esac
    if [[ -n "{{ slurm_cpu_extra_arguments }}" ]]; then
      read -r -a extra_arguments <<< "{{ slurm_cpu_extra_arguments }}"
      slurm_arguments+=("${extra_arguments[@]}")
    fi
    printf -v job_command 'cd %q && . scripts/server_env.sh && gwas_engine_configure_cpu_parallelism && %s' "${repository_root}" "${command}"
    exec srun "${slurm_arguments[@]}" bash -lc "${job_command}"

# Run another just recipe through SLURM on the configured CPU node
slurm-cpu-just +just_arguments:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    exec just slurm-cpu-run 'just {{ just_arguments }}'

# Run REGENIE step 2 with local baseline predictions
regenie-linear:
    {{ server_env }} && uv run g regenie --step 2 --qt --bgen {{ data_dir }}/1kg_chr22_full.bgen --sample {{ data_dir }}/1kg_chr22_full.sample --phenoFile {{ data_dir }}/pheno_cont.txt --phenoCol phenotype_continuous --covarFile {{ data_dir }}/covariates.txt --covarColList age,sex --pred {{ data_dir }}/baselines/regenie_step1_qt_pred.list --out {{ data_dir }}/regenie_linear --g-output-format parquet

# Run binary REGENIE step 2 on chr22 with GPU JAX
regenie2-binary-gpu:
    {{ server_env }} && uv run g regenie --step 2 --bt --bgen {{ data_dir }}/1kg_chr22_full.bgen --sample {{ data_dir }}/1kg_chr22_full.sample --phenoFile {{ data_dir }}/pheno_bin.txt --phenoCol phenotype_binary --covarFile {{ data_dir }}/covariates.txt --covarColList age,sex --pred {{ data_dir }}/baselines/regenie_step1_pred.list --out {{ data_dir }}/regenie2_binary_chr22_gpu --g-device gpu --firth --approx --g-output-format parquet

# Smoke test binary REGENIE step 2 on a small chr22 variant slice with GPU JAX
regenie2-binary-gpu-smoke:
    {{ server_env }} && uv run g regenie --step 2 --bt --bgen {{ data_dir }}/1kg_chr22_full.bgen --sample {{ data_dir }}/1kg_chr22_full.sample --phenoFile {{ data_dir }}/pheno_bin.txt --phenoCol phenotype_binary --covarFile {{ data_dir }}/covariates.txt --covarColList age,sex --pred {{ data_dir }}/baselines/regenie_step1_pred.list --out {{ data_dir }}/regenie2_binary_chr22_gpu_smoke --g-device gpu --firth --approx --g-variant-limit 1000 --g-output-format parquet

# Run binary REGENIE step 2 through SLURM on the configured GPU node
slurm-regenie2-binary-gpu:
    {{ server_env }} && just slurm-gpu-just regenie2-binary-gpu

# Smoke test binary REGENIE step 2 through SLURM on the configured GPU node
slurm-regenie2-binary-gpu-smoke:
    {{ server_env }} && just slurm-gpu-just regenie2-binary-gpu-smoke

# Verify binary REGENIE step 2 GPU output artifacts
verify-regenie2-binary-gpu-output:
    #!/usr/bin/env bash
    set -euo pipefail
    run_directory="{{ data_dir }}/regenie2_binary_chr22_gpu.regenie2_binary.run"
    test -d "${run_directory}/parts"
    find "${run_directory}/parts" -type f -name '*.parquet' | grep -q .
    echo "Binary REGENIE step 2 GPU output is present."

# Verify binary REGENIE step 2 GPU smoke output artifacts
verify-regenie2-binary-gpu-smoke-output:
    #!/usr/bin/env bash
    set -euo pipefail
    run_directory="{{ data_dir }}/regenie2_binary_chr22_gpu_smoke.regenie2_binary.run"
    test -d "${run_directory}/parts"
    find "${run_directory}/parts" -type f -name '*.parquet' | grep -q .
    echo "Binary REGENIE step 2 GPU smoke output is present."

# Run CPU/GPU JAX runtime probe
probe-jax: doctor-jax

# Dry-run the standard chr10 binary/linear CPU/GPU/cache step 2 matrix
regenie2-chr10-matrix-dry-run *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix tool.dry_run=true {{ overrides }}

# Run the standard chr10 binary/linear CPU/GPU/cache step 2 matrix
regenie2-chr10-matrix *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix {{ overrides }}

# Submit the standard chr10 step 2 matrix through SLURM on the configured GPU node
slurm-regenie2-chr10-matrix *overrides:
    {{ server_env }} && just slurm-gpu-just regenie2-chr10-matrix {{ overrides }}

# Dry-run the standard chr22 binary/linear CPU/GPU/cache step 2 matrix
regenie2-chr22-matrix-dry-run *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name run_regenie2_chr22_matrix tool.dry_run=true {{ overrides }}

# Run the standard chr22 binary/linear CPU/GPU/cache step 2 matrix
regenie2-chr22-matrix *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.run_regenie2_matrix --config-name run_regenie2_chr22_matrix {{ overrides }}

# Submit the standard chr22 step 2 matrix through SLURM on the configured GPU node
slurm-regenie2-chr22-matrix *overrides:
    {{ server_env }} && just slurm-gpu-just regenie2-chr22-matrix {{ overrides }}

# Benchmark BGEN float32 read paths
benchmark-bgen-reader *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_bgen_reader {{ overrides }}

# Benchmark REGENIE step 2 in fresh Python processes
benchmark-regenie2-linear-fresh-gpu: install-perf-extension
    {{ server_env }} && uv run --no-sync python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu

# Benchmark REGENIE step 2 in fresh Python processes using Parquet dataset output plus finalization
benchmark-regenie2-linear-fresh-gpu-parquet: install-perf-extension
    {{ server_env }} && uv run --no-sync python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu --finalize-parquet

# Benchmark binary REGENIE step 2 with cold, same-process hot, chunk-only, and finalized timings
benchmark-regenie2-binary-hot-gpu *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot machine=landau_gpu {{ overrides }}

# Benchmark output-stage timings across finalization, phenotype count, and bsize
benchmark-output-stages-gpu *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_output_stages machine=landau_gpu {{ overrides }}

# Smoke test binary REGENIE step 2 benchmark harness on a small variant slice
benchmark-regenie2-binary-hot-gpu-smoke *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot machine=landau_gpu tool.variant_limit=1000 tool.include_cold_process=false tool.include_finalized_hot=false {{ overrides }}

# Submit binary hot benchmark to the configured GPU node
slurm-benchmark-regenie2-binary-hot-gpu *overrides:
    {{ server_env }} && just slurm-gpu-just benchmark-regenie2-binary-hot-gpu {{ overrides }}

# Run the login-node-safe performance harness smoke benchmark
perf-smoke *arguments:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.performance_smoke --output-root "{{ perf_results_dir }}/smoke" {{ arguments }}

# Submit the standard CPU performance benchmark through the configured CPU SLURM node
perf-cpu *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    run_directory="{{ perf_results_dir }}/cpu/bgen_reader_$(date -u +%Y%m%dT%H%M%SZ)"
    exec just slurm-cpu-just benchmark-bgen-reader "telemetry.json_summary_path=${run_directory}/bgen_reader_summary.json" "telemetry.markdown_summary_path=${run_directory}/bgen_reader_summary.md" {{ overrides }}

# Submit the standard GPU performance benchmark through the existing GPU SLURM recipe
perf-gpu *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    run_directory="{{ perf_results_dir }}/gpu/regenie2_binary_hot_$(date -u +%Y%m%dT%H%M%SZ)"
    exec just slurm-benchmark-regenie2-binary-hot-gpu "tool.output_dir=${run_directory}" "telemetry.json_summary_path=${run_directory}/regenie2_binary_hot_summary.json" {{ overrides }}

# Compare two benchmark JSON summaries
perf-compare baseline_json new_json:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.performance_compare '{{ baseline_json }}' '{{ new_json }}'

# Sequentially tune GPU REGENIE step 2 and active BGEN reader knobs
tune-regenie2-gpu *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.tune_regenie2_gpu machine=landau_gpu {{ overrides }}

# Run Rust Criterion benchmarks with native performance flags
benchmark-rust:
    {{ server_env }} && RUSTFLAGS="-C target-cpu=native" cargo bench

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU
profile-regenie-comparison-cpu: setup-data install-perf-extension
    {{ server_env }} && uv run --no-sync python scripts/profile_regenie_comparison.py --cpu-only

# Unified profiling comparison: original regenie (4 programs) + g quantitative step2 CPU+GPU
profile-regenie-comparison-gpu: setup-data install-perf-extension
    {{ server_env }} && uv run --no-sync python scripts/profile_regenie_comparison.py --include-gpu

# Alias for unified profiling comparison (CPU-only default)
profile-regenie-comparison: profile-regenie-comparison-cpu

# Run the deep REGENIE step 2 profiling harness on the current host
profile-regenie2-deep *overrides: install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu {{ overrides }}

# Write the deep REGENIE step 2 profiling plan without running workloads
profile-regenie2-deep-dry-run *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu tool.dry_run=true {{ overrides }}

# Smoke test the deep REGENIE step 2 profiling harness on the current host
profile-regenie2-deep-smoke *overrides: install-gpu-dependencies install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu tool.smoke=true tool.skip_deep_profiles=true tool.enable_rust_criterion=false {{ overrides }}

# Write the full app profiling plan without running workloads
profile-app-full-dry-run *overrides:
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu tool.include_regenie_baseline=false tool.dry_run=true {{ overrides }}

# Smoke test the full app profiling bundle on the current host
profile-app-full-smoke *overrides: install-gpu-dependencies install-perf-extension
    {{ server_env }} && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu tool.include_regenie_baseline=false tool.enable_rust_criterion=false tool.smoke=true {{ overrides }}

# Submit one long landau SLURM job for the deep REGENIE step 2 profiling harness
profile-regenie2-deep-landau *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    export GWAS_ENGINE_SLURM_TIME="${GWAS_ENGINE_SLURM_TIME:-12:00:00}"
    export GWAS_ENGINE_SLURM_CPUS_PER_TASK="${GWAS_ENGINE_SLURM_CPUS_PER_TASK:-8}"
    export GWAS_ENGINE_SLURM_MEMORY="${GWAS_ENGINE_SLURM_MEMORY:-64G}"
    export GWAS_ENGINE_SLURM_GPUS_PER_TASK="${GWAS_ENGINE_SLURM_GPUS_PER_TASK:-1}"
    exec just slurm-gpu-run '. scripts/server_env.sh && just install-gpu-dependencies && just install-perf-extension && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu tool.include_regenie_baseline=false {{ deep_profile_landau_budget_overrides }} {{ overrides }}'

# Submit one long landau SLURM job for the full app profiling bundle
profile-app-full-landau *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    . scripts/server_env.sh
    export GWAS_ENGINE_SLURM_TIME="${GWAS_ENGINE_SLURM_TIME:-12:00:00}"
    export GWAS_ENGINE_SLURM_CPUS_PER_TASK="${GWAS_ENGINE_SLURM_CPUS_PER_TASK:-8}"
    export GWAS_ENGINE_SLURM_MEMORY="${GWAS_ENGINE_SLURM_MEMORY:-64G}"
    export GWAS_ENGINE_SLURM_GPUS_PER_TASK="${GWAS_ENGINE_SLURM_GPUS_PER_TASK:-1}"
    exec just slurm-gpu-run '. scripts/server_env.sh && just install-gpu-dependencies && just install-perf-extension && uv run --no-sync python -m tooling.cli.profile_regenie2_deep machine=landau_gpu tool.include_regenie_baseline=false {{ deep_profile_landau_budget_overrides }} {{ overrides }}'

# Format code
format:
    {{ server_env }} && uv run ruff format .
    {{ server_env }} && cargo fmt

# Lint code
lint:
    {{ server_env }} && uv run ruff check . --fix
    {{ server_env }} && cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic

# Type check Python code
typecheck:
    {{ server_env }} && uv run ty check src tests scripts tooling

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
    uv run ty check src tests scripts tooling

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
    . scripts/server_env.sh
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
    . scripts/server_env.sh
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
    . scripts/server_env.sh
    pytest_arguments=()
    if gwas_engine_is_positive_integer "${GWAS_ENGINE_PYTEST_WORKERS:-}" && [[ "${GWAS_ENGINE_PYTEST_WORKERS}" -gt 1 ]]; then
      gwas_engine_configure_parallel_pytest_thread_limits
      pytest_arguments+=("-n" "${GWAS_ENGINE_PYTEST_WORKERS}")
    fi
    uv run pytest tests/ --cov=src/g --cov-report=term-missing --cov-fail-under=90 "${pytest_arguments[@]}"

# Run Rust line coverage gate
coverage-rust:
    {{ server_env }} && cargo llvm-cov --workspace --all-targets --ignore-filename-regex '(^|/)(benches|tests)/' --fail-under-lines 90

# Run all coverage gates
coverage: coverage-python coverage-rust

# Build all Rust targets
rust-build:
    {{ server_env }} && cargo build --workspace --all-targets

# Run the Rust test suite
rust-test:
    {{ server_env }} && cargo test --workspace

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

# Upgrade Python lockfile dependencies, including dev and GPU groups
upgrade-python-deps:
    {{ server_env }} && uv sync -U --group dev --group gpu

# Upgrade the Nix flake lockfile
upgrade-nix-lock:
    nix flake update

# Upgrade Python and Nix dependency locks
upgrade-deps: upgrade-python-deps upgrade-nix-lock
