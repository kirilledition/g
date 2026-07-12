#!/usr/bin/env bash

set -euo pipefail

if [ -n "${BASH_SOURCE:-}" ]; then
  repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
else
  repository_root="$(pwd)"
fi
tools_directory="${GWAS_ENGINE_TOOLS_DIR:-${repository_root}/.tools}"
repo_rust_directory="${tools_directory}/rust"
repo_cargo_home="${repo_rust_directory}/cargo"
repo_rustup_home="${repo_rust_directory}/rustup"

export PATH="${tools_directory}/bin:${HOME}/.local/bin:${repo_cargo_home}/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/g-uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [ -z "${CARGO_HOME:-}" ] && [ -x "${repo_cargo_home}/bin/cargo" ]; then
  export CARGO_HOME="${repo_cargo_home}"
fi
if [ -z "${RUSTUP_HOME:-}" ] && [ -x "${repo_cargo_home}/bin/rustup" ]; then
  export RUSTUP_HOME="${repo_rustup_home}"
fi
export PYO3_PYTHON="${PYO3_PYTHON:-python3.14}"

gwas_engine_is_positive_integer() {
  case "${1:-}" in
    "" | *[!0-9]*)
      return 1
      ;;
    0)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

gwas_engine_allocated_cpu_count() {
  if gwas_engine_is_positive_integer "${SLURM_CPUS_PER_TASK:-}"; then
    printf '%s\n' "${SLURM_CPUS_PER_TASK}"
    return
  fi
  if gwas_engine_is_positive_integer "${SLURM_CPUS_ON_NODE:-}"; then
    printf '%s\n' "${SLURM_CPUS_ON_NODE}"
    return
  fi
  nproc
}

gwas_engine_default_pytest_worker_count() {
  allocated_cpu_count="$1"
  default_worker_limit="${GWAS_ENGINE_CPU_PYTEST_WORKER_LIMIT:-8}"
  if ! gwas_engine_is_positive_integer "${default_worker_limit}"; then
    default_worker_limit="8"
  fi
  if [ "${allocated_cpu_count}" -lt "${default_worker_limit}" ]; then
    printf '%s\n' "${allocated_cpu_count}"
  else
    printf '%s\n' "${default_worker_limit}"
  fi
}

gwas_engine_configure_cpu_parallelism() {
  allocated_cpu_count="$(gwas_engine_allocated_cpu_count)"
  export GWAS_ENGINE_ALLOCATED_CPU_COUNT="${GWAS_ENGINE_ALLOCATED_CPU_COUNT:-${allocated_cpu_count}}"
  if [ -z "${CARGO_BUILD_JOBS:-}" ]; then
    if gwas_engine_is_positive_integer "${SLURM_CPUS_PER_TASK:-}"; then
      export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK}"
    elif gwas_engine_is_positive_integer "${SLURM_CPUS_ON_NODE:-}"; then
      export CARGO_BUILD_JOBS="${SLURM_CPUS_ON_NODE}"
    fi
  fi
  if [ -z "${GWAS_ENGINE_PYTEST_WORKERS:-}" ]; then
    if [ -n "${GWAS_ENGINE_CPU_PYTEST_WORKERS:-}" ]; then
      export GWAS_ENGINE_PYTEST_WORKERS="${GWAS_ENGINE_CPU_PYTEST_WORKERS}"
    else
      export GWAS_ENGINE_PYTEST_WORKERS="$(gwas_engine_default_pytest_worker_count "${allocated_cpu_count}")"
    fi
  fi
}

gwas_engine_configure_rust_build_environment() {
  gwas_engine_configure_cpu_parallelism
  if [ -z "${CARGO_TARGET_DIR:-}" ] && [ -n "${SLURMD_NODENAME:-}" ]; then
    export CARGO_TARGET_DIR="${repository_root}/target/slurm/${SLURMD_NODENAME}"
  fi
}

gwas_engine_verify_mold_linker() {
  if [ "$(uname -s)" != "Linux" ]; then
    return
  fi
  local probe_binary
  probe_binary="$(mktemp "${TMPDIR:-/tmp}/g-mold-probe.XXXXXX")"
  if ! cc -fuse-ld=mold -x c -o "${probe_binary}" - <<<'int main(void) { return 0; }'; then
    rm -f "${probe_binary}"
    echo "The cc compiler driver cannot link with -fuse-ld=mold." >&2
    return 1
  fi
  rm -f "${probe_binary}"
}

gwas_engine_log_rust_build_environment() {
  echo "GWAS_ENGINE_ALLOCATED_CPU_COUNT=${GWAS_ENGINE_ALLOCATED_CPU_COUNT:-unset}"
  echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-30 (Cargo config default)}"
  echo "CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target (Cargo default)}"
  echo "RUST_LINKER=cc with mold"
  echo "RUSTC_WRAPPER=${RUSTC_WRAPPER:-unset}"
  if [ -n "${SCCACHE_DIR:-}" ]; then
    echo "SCCACHE_DIR=${SCCACHE_DIR}"
  fi
}

gwas_engine_configure_parallel_pytest_thread_limits() {
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
}

if command -v "${PYO3_PYTHON}" >/dev/null 2>&1; then
  python_library_directory="$("${PYO3_PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
  if [ -n "${python_library_directory}" ]; then
    export LD_LIBRARY_PATH="${python_library_directory}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

if [ -z "${XDG_RUNTIME_DIR:-}" ] || [ ! -w "${XDG_RUNTIME_DIR}" ]; then
  user_identifier="$(id -u)"
  export XDG_RUNTIME_DIR="/tmp/g-runtime-${user_identifier}"
  mkdir -p "${XDG_RUNTIME_DIR}"
  chmod 700 "${XDG_RUNTIME_DIR}"
fi
