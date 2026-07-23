#!/usr/bin/env bash

set -euo pipefail
umask 077

readonly system_bash="/usr/bin/bash"
readonly system_ar="/usr/bin/ar"
readonly system_as="/usr/bin/as"
readonly system_cc="/usr/bin/cc"
readonly system_cxx="/usr/bin/c++"
readonly system_git="/usr/bin/git"
readonly system_scontrol="/usr/bin/scontrol"
readonly system_environment="/usr/bin/env"
readonly system_git_upload_pack="/usr/bin/git-upload-pack"
readonly system_ranlib="/usr/bin/ranlib"
readonly system_tar="/usr/bin/tar"
readonly bootstrap_relative_path="tooling/server/exact_parity_bootstrap.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: exact_parity_bootstrap.sh SOURCE_REPOSITORY EXPECTED_GIT_COMMIT" >&2
  exit 2
fi
for forbidden_environment_name in \
  BASH_ENV \
  ENV \
  LD_AUDIT \
  LD_PRELOAD \
  LD_LIBRARY_PATH \
  PYTHONHOME \
  PYTHONINSPECT \
  PYTHONPATH \
  PYTHONSTARTUP \
  PYTHONUSERBASE \
  PYO3_PYTHON \
  PYTEST_ADDOPTS \
  PYTEST_PLUGINS \
  UV_CONFIG_FILE \
  UV_ENV_FILE \
  UV_PROJECT \
  UV_PROJECT_ENVIRONMENT \
  UV_NO_CONFIG \
  UV_NO_MANAGED_PYTHON \
  UV_PYTHON \
  UV_PYTHON_DOWNLOADS \
  UV_WORKING_DIR \
  AR \
  AS \
  CC \
  CXX \
  RANLIB \
  CARGO \
  CARGO_BUILD_RUSTC_WRAPPER \
  CARGO_ENCODED_RUSTFLAGS \
  CARGO_HOME \
  CARGO_TARGET_DIR \
  GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_COMMON_DIR \
  GIT_CONFIG_COUNT \
  GIT_CONFIG_GLOBAL \
  GIT_CONFIG_NOSYSTEM \
  GIT_CONFIG_PARAMETERS \
  GIT_CONFIG_SYSTEM \
  GIT_DIR \
  GIT_EXEC_PATH \
  GIT_INDEX_FILE \
  GIT_OBJECT_DIRECTORY \
  GIT_REPLACE_REF_BASE \
  GIT_SHALLOW_FILE \
  GIT_TEMPLATE_DIR \
  GIT_WORK_TREE \
  RUSTC \
  RUSTC_WRAPPER \
  RUSTFLAGS \
  RUSTUP_HOME \
  RUSTUP_TOOLCHAIN; do
  if [[ -v "${forbidden_environment_name}" ]]; then
    echo "Trusted scheduler launch retained forbidden environment variable: ${forbidden_environment_name}" >&2
    exit 1
  fi
done
for inherited_git_config_name in "${!GIT_CONFIG_KEY_@}" "${!GIT_CONFIG_VALUE_@}"; do
  if [[ -n "${inherited_git_config_name}" ]]; then
    echo "Trusted scheduler launch retained forbidden environment variable: ${inherited_git_config_name}" >&2
    exit 1
  fi
done
for system_executable in \
  "${system_bash}" \
  "${system_ar}" \
  "${system_as}" \
  "${system_cc}" \
  "${system_cxx}" \
  "${system_git}" \
  "${system_scontrol}" \
  "${system_environment}" \
  "${system_git_upload_pack}" \
  "${system_ranlib}" \
  "/usr/bin/cp" \
  "/usr/bin/mktemp" \
  "/usr/bin/nvidia-smi" \
  "/usr/bin/rm" \
  "/usr/bin/stat" \
  "${system_tar}"; do
  if [[ ! -x "${system_executable}" ]]; then
    echo "Missing trusted host executable: ${system_executable}" >&2
    exit 1
  fi
done
compiler_program_environment=(
  "${system_environment}"
  -i
  "PATH=/usr/bin:/bin"
)
cc1_path="$(
  /usr/bin/realpath "$("${compiler_program_environment[@]}" "${system_cc}" -print-prog-name=cc1)"
)"
cc1plus_path="$(
  /usr/bin/realpath "$("${compiler_program_environment[@]}" "${system_cxx}" -print-prog-name=cc1plus)"
)"
collect2_path="$(
  /usr/bin/realpath "$("${compiler_program_environment[@]}" "${system_cc}" -print-prog-name=collect2)"
)"
for compiler_helper_path in "${cc1_path}" "${cc1plus_path}" "${collect2_path}"; do
  if [[ "${compiler_helper_path}" != /* || ! -x "${compiler_helper_path}" ]]; then
    echo "System compiler resolved a missing or non-absolute helper: ${compiler_helper_path}" >&2
    exit 1
  fi
done

source_repository="$(/usr/bin/realpath "$1")"
expected_git_commit="$2"
expected_bootstrap_sha256="${G_REGENIE_PARITY_BOOTSTRAP_SHA256:-}"
configured_uv_path="${G_REGENIE_PARITY_UV_PATH:-}"
configured_just_path="${G_REGENIE_PARITY_JUST_PATH:-}"
configured_cargo_path="${G_REGENIE_PARITY_CARGO_PATH:-}"
configured_cargo_cache_home="${G_REGENIE_PARITY_CARGO_CACHE_HOME:-}"
configured_mold_path="${G_REGENIE_PARITY_MOLD_PATH:-}"
configured_python_path="${G_REGENIE_PARITY_PYTHON_PATH:-}"
configured_rustc_path="${G_REGENIE_PARITY_RUSTC_PATH:-}"
configured_rustup_home="${G_REGENIE_PARITY_RUSTUP_HOME:-}"
if [[ ! "${expected_git_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected a scheduler-selected full lowercase Git SHA." >&2
  exit 1
fi
if [[ ! "${expected_bootstrap_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "The trusted scheduler must provide G_REGENIE_PARITY_BOOTSTRAP_SHA256." >&2
  exit 1
fi
executed_bootstrap_path="$(/usr/bin/realpath "$0")"
executed_bootstrap_sha256="$(/usr/bin/sha256sum "${executed_bootstrap_path}" | /usr/bin/cut -d ' ' -f 1)"
if [[ "${executed_bootstrap_sha256}" != "${expected_bootstrap_sha256}" ]]; then
  echo "Executed bootstrap file differs from its scheduler-provided digest." >&2
  exit 1
fi
for configured_tool_path in \
  "${configured_uv_path}" \
  "${configured_just_path}" \
  "${configured_cargo_path}" \
  "${configured_mold_path}" \
  "${configured_python_path}" \
  "${configured_rustc_path}"; do
  if [[ "${configured_tool_path}" != /* || ! -x "${configured_tool_path}" ]]; then
    echo "The trusted scheduler must provide absolute executable uv, just, cargo, mold, Python, and rustc paths." >&2
    exit 1
  fi
done
if [[ "${configured_rustup_home}" != /* || ! -d "${configured_rustup_home}" ]]; then
  echo "The trusted scheduler must provide an absolute existing Rustup home." >&2
  exit 1
fi
if [[ "${configured_cargo_cache_home}" != /* || ! -d "${configured_cargo_cache_home}" ]]; then
  echo "The trusted scheduler must provide an absolute existing Cargo cache home." >&2
  exit 1
fi
uv_path="$(/usr/bin/realpath "${configured_uv_path}")"
just_path="$(/usr/bin/realpath "${configured_just_path}")"
cargo_path="$(/usr/bin/realpath "${configured_cargo_path}")"
cargo_cache_home="$(/usr/bin/realpath "${configured_cargo_cache_home}")"
mold_path="$(/usr/bin/realpath "${configured_mold_path}")"
python_path="$(/usr/bin/realpath "${configured_python_path}")"
rustc_path="$(/usr/bin/realpath "${configured_rustc_path}")"
rustup_home="$(/usr/bin/realpath "${configured_rustup_home}")"
if [[ "${cargo_path}" == "${rustc_path}" || "${cargo_path##*/}" == "rustup" || "${rustc_path##*/}" == "rustup" ]]; then
  echo "The trusted scheduler must provide direct pinned cargo and rustc binaries, not Rustup proxies." >&2
  exit 1
fi

qualification_node="${SLURMD_NODENAME:-}"
qualification_hostname="$(/usr/bin/hostname -s)"
slurm_job_id="${SLURM_JOB_ID:-}"
slurm_step_id="${SLURM_STEP_ID:-}"
slurm_cpus_per_task="${SLURM_CPUS_PER_TASK:-}"
qualification_user="$(/usr/bin/id -un)"
qualification_user_identifier="$(/usr/bin/id -u)"
if [[ "${qualification_node}" != "landau" || "${qualification_hostname}" != "landau" ]]; then
  echo "Qualification requires a Slurm step on landau; observed node=${qualification_node:-unset}, host=${qualification_hostname}." >&2
  exit 1
fi
if [[ ! "${slurm_job_id}" =~ ^[0-9]+$ || ! "${slurm_step_id}" =~ ^[0-9]+$ ]]; then
  echo "Qualification requires numeric Slurm job and step IDs." >&2
  exit 1
fi
if [[ ! "${slurm_cpus_per_task}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Qualification requires a positive Slurm CPU allocation." >&2
  exit 1
fi
slurm_job_record="$("${system_scontrol}" show job "${slurm_job_id}" --oneliner)"
slurm_step_record="$("${system_scontrol}" show step "${slurm_job_id}.${slurm_step_id}" --oneliner)"
if [[ " ${slurm_job_record} " != *" JobState=RUNNING "* ]]; then
  echo "Slurm job ${slurm_job_id} is not running." >&2
  exit 1
fi
if [[ " ${slurm_job_record} " != *" NodeList=landau "* ]]; then
  echo "Slurm job ${slurm_job_id} is not allocated to landau." >&2
  exit 1
fi
if [[ " ${slurm_job_record} " != *" UserId=${qualification_user}(${qualification_user_identifier}) "* ]]; then
  echo "Slurm job ${slurm_job_id} belongs to a different user." >&2
  exit 1
fi
if [[ " ${slurm_step_record} " != *" StepId=${slurm_job_id}.${slurm_step_id} "* ]]; then
  echo "Slurm step identity does not match ${slurm_job_id}.${slurm_step_id}." >&2
  exit 1
fi
if [[ " ${slurm_step_record} " != *" State=RUNNING "* ]]; then
  echo "Slurm step ${slurm_job_id}.${slurm_step_id} is not running." >&2
  exit 1
fi
if [[ " ${slurm_step_record} " != *" NodeList=landau "* ]]; then
  echo "Slurm step ${slurm_job_id}.${slurm_step_id} is not running on landau." >&2
  exit 1
fi

export GIT_CONFIG_COUNT=0
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1
unset BASH_ENV ENV CDPATH LD_AUDIT LD_PRELOAD
unset GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE GIT_SHALLOW_FILE GIT_GRAFT_FILE
unset GIT_REPLACE_REF_BASE GIT_CEILING_DIRECTORIES GIT_DISCOVERY_ACROSS_FILESYSTEM
unset GIT_CONFIG_PARAMETERS GIT_TEMPLATE_DIR GIT_EXEC_PATH GIT_EXTERNAL_DIFF
unset GIT_SSH GIT_SSH_COMMAND GIT_ASKPASS SSH_ASKPASS
for git_config_variable in "${!GIT_CONFIG_KEY_@}" "${!GIT_CONFIG_VALUE_@}"; do
  unset "${git_config_variable}"
done

"${system_git}" -C "${source_repository}" --no-replace-objects cat-file -e "${expected_git_commit}^{commit}"
committed_bootstrap_sha256="$(
  "${system_git}" -C "${source_repository}" --no-replace-objects \
    cat-file blob "${expected_git_commit}:${bootstrap_relative_path}" |
    /usr/bin/sha256sum |
    /usr/bin/cut -d ' ' -f 1
)"
if [[ "${committed_bootstrap_sha256}" != "${expected_bootstrap_sha256}" ]]; then
  echo "Executed bootstrap digest does not match the scheduler-selected commit." >&2
  exit 1
fi

run_nonce="$(/usr/bin/python3 -I -c 'import secrets; print(secrets.token_hex(16))')"
run_started_at_utc="$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
qualification_root="$(
  /usr/bin/mktemp -d \
    "/tmp/g-parity-qualification-${qualification_user_identifier}-${slurm_job_id}-${run_nonce}.XXXXXX"
)"
qualification_root_identity="$(
  /usr/bin/stat --format='%u:%a' "${qualification_root}"
)"
if [[ -L "${qualification_root}" || "${qualification_root_identity}" != "${qualification_user_identifier}:700" ]]; then
  echo "Qualification root is not an owned mode-0700 directory: ${qualification_root}" >&2
  exit 1
fi
cleanup_qualification_root() {
  local command_status="$?"
  local expected_root_prefix
  local observed_root_identity
  trap - EXIT
  expected_root_prefix="/tmp/g-parity-qualification-${qualification_user_identifier}-${slurm_job_id}-${run_nonce}."
  set +e
  observed_root_identity="$(/usr/bin/stat --format='%u:%a' "${qualification_root}" 2>/dev/null)"
  if [[ "${qualification_root}" != "${expected_root_prefix}"* || -L "${qualification_root}" || "${observed_root_identity}" != "${qualification_user_identifier}:700" ]]; then
    echo "Refusing to clean unexpected qualification root: ${qualification_root}" >&2
    if [[ "${command_status}" -eq 0 ]]; then
      command_status=1
    fi
  elif ! /usr/bin/rm -rf -- "${qualification_root}"; then
    echo "Failed to clean qualification root: ${qualification_root}" >&2
    if [[ "${command_status}" -eq 0 ]]; then
      command_status=1
    fi
  fi
  exit "${command_status}"
}
trap cleanup_qualification_root EXIT
checkout_directory="${qualification_root}/checkout"
git_template_directory="${qualification_root}/empty-git-template"
cargo_home_directory="${qualification_root}/cargo-home"
trusted_bin_directory="${qualification_root}/trusted-bin"
runtime_directory="${qualification_root}/runtime"
temporary_directory="${qualification_root}/tmp"
pytest_basetemp_directory="${qualification_root}/pytest"
home_directory="${qualification_root}/home"
/usr/bin/mkdir "${git_template_directory}"
/usr/bin/mkdir "${cargo_home_directory}"
/usr/bin/mkdir "${trusted_bin_directory}"
/usr/bin/mkdir "${runtime_directory}"
/usr/bin/mkdir "${temporary_directory}"
/usr/bin/mkdir "${home_directory}"
for cargo_cache_entry in git registry; do
  if [[ -d "${cargo_cache_home}/${cargo_cache_entry}" ]]; then
    /usr/bin/cp -a "${cargo_cache_home}/${cargo_cache_entry}" "${cargo_home_directory}/${cargo_cache_entry}"
  fi
done
if [[ -f "${cargo_cache_home}/.global-cache" ]]; then
  /usr/bin/cp "${cargo_cache_home}/.global-cache" "${cargo_home_directory}/.global-cache"
fi
/usr/bin/ln -s "${cargo_path}" "${trusted_bin_directory}/cargo"
/usr/bin/ln -s "${system_ar}" "${trusted_bin_directory}/ar"
/usr/bin/ln -s "${system_as}" "${trusted_bin_directory}/as"
/usr/bin/ln -s "${system_cc}" "${trusted_bin_directory}/cc"
/usr/bin/ln -s "${system_cxx}" "${trusted_bin_directory}/c++"
/usr/bin/ln -s "${just_path}" "${trusted_bin_directory}/just"
/usr/bin/ln -s "${mold_path}" "${trusted_bin_directory}/mold"
/usr/bin/ln -s "${mold_path}" "${trusted_bin_directory}/ld.mold"
/usr/bin/ln -s "${python_path}" "${trusted_bin_directory}/python"
/usr/bin/ln -s "${system_ranlib}" "${trusted_bin_directory}/ranlib"
/usr/bin/ln -s "${rustc_path}" "${trusted_bin_directory}/rustc"
/usr/bin/ln -s "${uv_path}" "${trusted_bin_directory}/uv"

configured_data_directory="${GWAS_ENGINE_DATA_DIR:-${source_repository}/data}"
configured_report_base="${G_REGENIE_PARITY_REPORT_DIRECTORY:-${source_repository}/results/parity/qualification}"
/usr/bin/mkdir -p "${configured_report_base}"
data_directory="$(/usr/bin/realpath -m "${configured_data_directory}")"
report_base="$(/usr/bin/realpath "${configured_report_base}")"
report_directory="${report_base}/${slurm_job_id}/${slurm_step_id}/${run_nonce}"
/usr/bin/mkdir -p "${report_directory}"
expected_bundle_path="${report_directory}/qualification_bundle_${expected_git_commit}_${slurm_job_id}_${slurm_step_id}_${run_nonce}.json"
if [[ -e "${expected_bundle_path}" ]]; then
  echo "Qualification refuses an existing bundle path: ${expected_bundle_path}" >&2
  exit 1
fi

"${system_git}" \
  -c core.hooksPath=/dev/null \
  -c core.attributesFile=/dev/null \
  -c protocol.file.allow=always \
  --no-replace-objects \
  clone \
  --template="${git_template_directory}" \
  --upload-pack="${system_git_upload_pack}" \
  --no-local \
  --no-hardlinks \
  --no-checkout \
  "${source_repository}" \
  "${checkout_directory}"
"${system_git}" -C "${checkout_directory}" config core.hooksPath /dev/null
"${system_git}" -C "${checkout_directory}" config core.attributesFile /dev/null
"${system_git}" -C "${checkout_directory}" config core.fsmonitor false
"${system_git}" -C "${checkout_directory}" config core.untrackedCache false
"${system_git}" -C "${checkout_directory}" update-ref --no-deref HEAD "${expected_git_commit}"
"${system_git}" -C "${checkout_directory}" read-tree --reset "${expected_git_commit}"
"${system_git}" -C "${checkout_directory}" --no-replace-objects archive --format=tar "${expected_git_commit}" |
  "${system_tar}" -xf - -C "${checkout_directory}"
observed_git_commit="$("${system_git}" -C "${checkout_directory}" --no-replace-objects rev-parse HEAD)"
if [[ "${observed_git_commit}" != "${expected_git_commit}" ]]; then
  echo "Detached qualification checkout resolved the wrong commit." >&2
  exit 1
fi
checkout_bootstrap_sha256="$(
  "${system_git}" -C "${checkout_directory}" --no-replace-objects \
    cat-file blob "HEAD:${bootstrap_relative_path}" |
    /usr/bin/sha256sum |
    /usr/bin/cut -d ' ' -f 1
)"
if [[ "${checkout_bootstrap_sha256}" != "${expected_bootstrap_sha256}" ]]; then
  echo "Detached checkout contains the wrong qualification bootstrap." >&2
  exit 1
fi
cargo_config_search_directory="$(/usr/bin/dirname "${checkout_directory}")"
while true; do
  for cargo_config_name in .cargo/config .cargo/config.toml; do
    if [[ -e "${cargo_config_search_directory}/${cargo_config_name}" ]]; then
      echo "Qualification refuses an external Cargo config: ${cargo_config_search_directory}/${cargo_config_name}" >&2
      exit 1
    fi
  done
  if [[ "${cargo_config_search_directory}" == "/" ]]; then
    break
  fi
  cargo_config_search_directory="$(/usr/bin/dirname "${cargo_config_search_directory}")"
done

tool_sha256() {
  /usr/bin/sha256sum "$1" | /usr/bin/cut -d ' ' -f 1
}

trusted_path="${trusted_bin_directory}:/usr/bin:/bin"
tool_version_environment=(
  "${system_environment}"
  -i
  "HOME=${home_directory}"
  "PATH=${trusted_path}"
  "CARGO=${cargo_path}"
  "CARGO_BUILD_JOBS=${slurm_cpus_per_task}"
  "CARGO_HOME=${cargo_home_directory}"
  "RUSTC=${rustc_path}"
  "RUSTUP_HOME=${rustup_home}"
  "UV_NO_CONFIG=1"
)
if [[ -n "${G_REGENIE_PARITY_LD_LIBRARY_PATH:-}" ]]; then
  tool_version_environment+=("LD_LIBRARY_PATH=${G_REGENIE_PARITY_LD_LIBRARY_PATH}")
fi
toolchain_probe_directory="${qualification_root}/toolchain-probe"
/usr/bin/mkdir "${toolchain_probe_directory}"
toolchain_probe_environment=(
  "${system_environment}"
  -i
  "HOME=${home_directory}"
  "PATH=${trusted_path}"
  "AR=${system_ar}"
  "AS=${system_as}"
  "CC=${system_cc}"
  "CXX=${system_cxx}"
  "RANLIB=${system_ranlib}"
)
"${toolchain_probe_environment[@]}" "${system_cxx}" \
  -std=c++20 \
  -fuse-ld=mold \
  -x c++ \
  -c \
  -o "${toolchain_probe_directory}/probe.o" \
  - <<<'int g_parity_toolchain_probe() { return 0; }'
"${toolchain_probe_environment[@]}" "${system_ar}" \
  rcs "${toolchain_probe_directory}/libprobe.a" "${toolchain_probe_directory}/probe.o"
"${toolchain_probe_environment[@]}" "${system_ranlib}" "${toolchain_probe_directory}/libprobe.a"
"${toolchain_probe_environment[@]}" "${system_cxx}" \
  -fuse-ld=mold \
  -x c++ \
  -o "${toolchain_probe_directory}/probe" \
  - <<<'int main() { return 0; }'
bash_version="$("${system_bash}" --version | /usr/bin/head -n 1)"
ar_version="$("${system_ar}" --version | /usr/bin/head -n 1)"
as_version="$("${system_as}" --version | /usr/bin/head -n 1)"
cc_version="$("${system_cc}" --version | /usr/bin/head -n 1)"
cc1_version="${cc_version}; internal program cc1"
cc1plus_version="${cc_version}; internal program cc1plus"
cargo_version="$(
  cd "${checkout_directory}"
  "${tool_version_environment[@]}" "${cargo_path}" --version
)"
cxx_version="$("${system_cxx}" --version | /usr/bin/head -n 1)"
collect2_version="${cc_version}; internal program collect2"
environment_version="$("${system_environment}" --version | /usr/bin/head -n 1)"
git_version="$("${system_git}" --version)"
just_version="$("${just_path}" --version)"
mold_version="$("${mold_path}" --version | /usr/bin/head -n 1)"
python_version="$(
  cd "${checkout_directory}"
  "${tool_version_environment[@]}" "${python_path}" -I --version
)"
ranlib_version="$("${system_ranlib}" --version | /usr/bin/head -n 1)"
rustc_version="$(
  cd "${checkout_directory}"
  "${tool_version_environment[@]}" "${rustc_path}" --version
)"
scontrol_version="$("${system_scontrol}" --version)"
uv_version="$("${tool_version_environment[@]}" "${uv_path}" --version)"
uv_cache_directory="${qualification_root}/uv-cache"

inner_environment=(
  "${system_environment}"
  -i
  "HOME=${home_directory}"
  "USER=${qualification_user}"
  "LOGNAME=${qualification_user}"
  "PATH=${trusted_path}"
  "AR=${system_ar}"
  "AS=${system_as}"
  "CC=${system_cc}"
  "CARGO=${cargo_path}"
  "CARGO_BUILD_JOBS=${slurm_cpus_per_task}"
  "CARGO_HOME=${cargo_home_directory}"
  "CXX=${system_cxx}"
  "RANLIB=${system_ranlib}"
  "CARGO_TARGET_DIR=${qualification_root}/target"
  "RUSTC=${rustc_path}"
  "RUSTUP_HOME=${rustup_home}"
  "TMPDIR=${temporary_directory}"
  "XDG_RUNTIME_DIR=${runtime_directory}"
  "UV_CACHE_DIR=${uv_cache_directory}"
  "UV_LINK_MODE=copy"
  "UV_NO_MANAGED_PYTHON=1"
  "UV_NO_CONFIG=1"
  "UV_PYTHON=${python_path}"
  "UV_PYTHON_DOWNLOADS=never"
  "G_REGENIE_PARITY_JAX_CACHE_DIRECTORY=${qualification_root}/jax-cache"
  "G_REGENIE_PARITY_PYTEST_BASETEMP=${pytest_basetemp_directory}"
  "G_REGENIE_PARITY_REQUIRE_DATA=1"
  "GWAS_ENGINE_ALLOCATED_CPU_COUNT=${slurm_cpus_per_task}"
  "GWAS_ENGINE_DATA_DIR=${data_directory}"
  "G_REGENIE_PARITY_REPORT_DIRECTORY=${report_directory}"
  "G_REGENIE_PARITY_EXPECTED_GIT_COMMIT=${expected_git_commit}"
  "G_REGENIE_PARITY_SLURM_JOB_ID=${slurm_job_id}"
  "G_REGENIE_PARITY_SLURM_STEP_ID=${slurm_step_id}"
  "G_REGENIE_PARITY_RUN_NONCE=${run_nonce}"
  "G_REGENIE_PARITY_RUN_STARTED_AT_UTC=${run_started_at_utc}"
  "G_REGENIE_PARITY_BOOTSTRAP_RELATIVE_PATH=${bootstrap_relative_path}"
  "G_REGENIE_PARITY_BOOTSTRAP_SHA256=${expected_bootstrap_sha256}"
  "G_REGENIE_PARITY_EXPECTED_BUNDLE_PATH=${expected_bundle_path}"
  "G_REGENIE_PARITY_QUALIFICATION_CHECKOUT=${checkout_directory}"
  "G_REGENIE_PARITY_TOOL_BASH_PATH=${system_bash}"
  "G_REGENIE_PARITY_TOOL_BASH_SHA256=$(tool_sha256 "${system_bash}")"
  "G_REGENIE_PARITY_TOOL_BASH_VERSION=${bash_version}"
  "G_REGENIE_PARITY_TOOL_AR_PATH=${system_ar}"
  "G_REGENIE_PARITY_TOOL_AR_SHA256=$(tool_sha256 "${system_ar}")"
  "G_REGENIE_PARITY_TOOL_AR_VERSION=${ar_version}"
  "G_REGENIE_PARITY_TOOL_AS_PATH=${system_as}"
  "G_REGENIE_PARITY_TOOL_AS_SHA256=$(tool_sha256 "${system_as}")"
  "G_REGENIE_PARITY_TOOL_AS_VERSION=${as_version}"
  "G_REGENIE_PARITY_TOOL_CC_PATH=${system_cc}"
  "G_REGENIE_PARITY_TOOL_CC_SHA256=$(tool_sha256 "${system_cc}")"
  "G_REGENIE_PARITY_TOOL_CC_VERSION=${cc_version}"
  "G_REGENIE_PARITY_TOOL_CC1_PATH=${cc1_path}"
  "G_REGENIE_PARITY_TOOL_CC1_SHA256=$(tool_sha256 "${cc1_path}")"
  "G_REGENIE_PARITY_TOOL_CC1_VERSION=${cc1_version}"
  "G_REGENIE_PARITY_TOOL_CC1PLUS_PATH=${cc1plus_path}"
  "G_REGENIE_PARITY_TOOL_CC1PLUS_SHA256=$(tool_sha256 "${cc1plus_path}")"
  "G_REGENIE_PARITY_TOOL_CC1PLUS_VERSION=${cc1plus_version}"
  "G_REGENIE_PARITY_TOOL_CARGO_PATH=${cargo_path}"
  "G_REGENIE_PARITY_TOOL_CARGO_SHA256=$(tool_sha256 "${cargo_path}")"
  "G_REGENIE_PARITY_TOOL_CARGO_VERSION=${cargo_version}"
  "G_REGENIE_PARITY_TOOL_CXX_PATH=${system_cxx}"
  "G_REGENIE_PARITY_TOOL_CXX_SHA256=$(tool_sha256 "${system_cxx}")"
  "G_REGENIE_PARITY_TOOL_CXX_VERSION=${cxx_version}"
  "G_REGENIE_PARITY_TOOL_COLLECT2_PATH=${collect2_path}"
  "G_REGENIE_PARITY_TOOL_COLLECT2_SHA256=$(tool_sha256 "${collect2_path}")"
  "G_REGENIE_PARITY_TOOL_COLLECT2_VERSION=${collect2_version}"
  "G_REGENIE_PARITY_TOOL_ENV_PATH=${system_environment}"
  "G_REGENIE_PARITY_TOOL_ENV_SHA256=$(tool_sha256 "${system_environment}")"
  "G_REGENIE_PARITY_TOOL_ENV_VERSION=${environment_version}"
  "G_REGENIE_PARITY_TOOL_GIT_PATH=${system_git}"
  "G_REGENIE_PARITY_TOOL_GIT_SHA256=$(tool_sha256 "${system_git}")"
  "G_REGENIE_PARITY_TOOL_GIT_VERSION=${git_version}"
  "G_REGENIE_PARITY_TOOL_JUST_PATH=${just_path}"
  "G_REGENIE_PARITY_TOOL_JUST_SHA256=$(tool_sha256 "${just_path}")"
  "G_REGENIE_PARITY_TOOL_JUST_VERSION=${just_version}"
  "G_REGENIE_PARITY_TOOL_MOLD_PATH=${mold_path}"
  "G_REGENIE_PARITY_TOOL_MOLD_SHA256=$(tool_sha256 "${mold_path}")"
  "G_REGENIE_PARITY_TOOL_MOLD_VERSION=${mold_version}"
  "G_REGENIE_PARITY_TOOL_PYTHON_PATH=${python_path}"
  "G_REGENIE_PARITY_TOOL_PYTHON_SHA256=$(tool_sha256 "${python_path}")"
  "G_REGENIE_PARITY_TOOL_PYTHON_VERSION=${python_version}"
  "G_REGENIE_PARITY_TOOL_RANLIB_PATH=${system_ranlib}"
  "G_REGENIE_PARITY_TOOL_RANLIB_SHA256=$(tool_sha256 "${system_ranlib}")"
  "G_REGENIE_PARITY_TOOL_RANLIB_VERSION=${ranlib_version}"
  "G_REGENIE_PARITY_TOOL_RUSTC_PATH=${rustc_path}"
  "G_REGENIE_PARITY_TOOL_RUSTC_SHA256=$(tool_sha256 "${rustc_path}")"
  "G_REGENIE_PARITY_TOOL_RUSTC_VERSION=${rustc_version}"
  "G_REGENIE_PARITY_TOOL_SCONTROL_PATH=${system_scontrol}"
  "G_REGENIE_PARITY_TOOL_SCONTROL_SHA256=$(tool_sha256 "${system_scontrol}")"
  "G_REGENIE_PARITY_TOOL_SCONTROL_VERSION=${scontrol_version}"
  "G_REGENIE_PARITY_TOOL_UV_PATH=${uv_path}"
  "G_REGENIE_PARITY_TOOL_UV_SHA256=$(tool_sha256 "${uv_path}")"
  "G_REGENIE_PARITY_TOOL_UV_VERSION=${uv_version}"
  "PYTEST_ADDOPTS="
  "PYTEST_PLUGINS="
  "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  "PYTHONNOUSERSITE=1"
  "PYTHONSAFEPATH=1"
  "JAX_PLATFORMS=cuda"
  "GIT_CONFIG_COUNT=0"
  "GIT_CONFIG_GLOBAL=/dev/null"
  "GIT_CONFIG_NOSYSTEM=1"
  "GIT_NO_REPLACE_OBJECTS=1"
)
for allowed_environment_name in \
  SLURM_CPUS_ON_NODE \
  SLURM_CPUS_PER_TASK \
  SLURM_JOB_ID \
  SLURM_JOB_NODELIST \
  SLURM_JOB_UID \
  SLURM_JOB_USER \
  SLURM_LOCALID \
  SLURM_NODEID \
  SLURM_PROCID \
  SLURM_STEP_ID \
  SLURM_STEP_NODELIST \
  SLURMD_NODENAME \
  CUDA_HOME \
  CUDA_PATH \
  CUDA_VISIBLE_DEVICES \
  NVIDIA_VISIBLE_DEVICES; do
  if [[ -v "${allowed_environment_name}" ]]; then
    inner_environment+=("${allowed_environment_name}=${!allowed_environment_name}")
  fi
done
if [[ -n "${G_REGENIE_PARITY_LD_LIBRARY_PATH:-}" ]]; then
  inner_environment+=("LD_LIBRARY_PATH=${G_REGENIE_PARITY_LD_LIBRARY_PATH}")
fi

echo "qualification_node=${qualification_node}"
echo "slurm_job_id=${slurm_job_id}"
echo "slurm_step_id=${slurm_step_id}"
echo "run_nonce=${run_nonce}"
echo "bootstrap_sha256=${expected_bootstrap_sha256}"
echo "checkout_directory=${checkout_directory}"
echo "report_directory=${report_directory}"

"${inner_environment[@]}" "${system_bash}" --noprofile --norc -c \
  'cd "$1"; exec "$2" test-parity-required-exact-inner' \
  exact-parity-inner \
  "${checkout_directory}" \
  "${just_path}"
