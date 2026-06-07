#!/usr/bin/env bash

set -euo pipefail

if [ -n "${BASH_SOURCE:-}" ]; then
  repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
