#!/usr/bin/bash

set -euo pipefail
umask 077

readonly system_environment="/usr/bin/env"
readonly system_git="/usr/bin/git"
readonly system_git_upload_pack="/usr/bin/git-upload-pack"
readonly system_realpath="/usr/bin/realpath"
readonly system_tar="/usr/bin/tar"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: exact_parity_checkout.sh SOURCE_REPOSITORY EXPECTED_COMMIT TARGET_CHECKOUT SCRATCH_ROOT" >&2
  exit 2
fi

source_repository="$("${system_realpath}" "$1")"
expected_git_commit="$2"
target_checkout="$("${system_realpath}" -m "$3")"
scratch_root="$("${system_realpath}" "$4")"

if [[ ! "${expected_git_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected a full lowercase SHA-1 commit identifier." >&2
  exit 1
fi
if [[ ! -d "${source_repository}" || ! -d "${scratch_root}" || -L "${scratch_root}" ]]; then
  echo "Source repository and nonsymlink scratch root must exist." >&2
  exit 1
fi
if [[ "${target_checkout}" != "${scratch_root}/"* || -e "${target_checkout}" || -L "${target_checkout}" ]]; then
  echo "Target checkout must be a new direct descendant of the scratch root." >&2
  exit 1
fi
for required_executable in \
  "${system_environment}" \
  "${system_git}" \
  "${system_git_upload_pack}" \
  "${system_realpath}" \
  "${system_tar}" \
  "/usr/bin/mkdir"; do
  if [[ ! -x "${required_executable}" ]]; then
    echo "Missing exact-checkout host executable: ${required_executable}" >&2
    exit 1
  fi
done

empty_template_directory="${scratch_root}/empty-git-template"
shadow_repository="${scratch_root}/source-shadow.git"
if [[ -e "${empty_template_directory}" || -L "${empty_template_directory}" || -e "${shadow_repository}" || -L "${shadow_repository}" ]]; then
  echo "Exact-checkout scratch paths must not preexist." >&2
  exit 1
fi
/usr/bin/mkdir "${empty_template_directory}"

git_environment=(
  "${system_environment}"
  -i
  "HOME=${scratch_root}"
  "LC_ALL=C"
  "PATH=/usr/bin:/bin"
  "GIT_CONFIG_COUNT=1"
  "GIT_CONFIG_KEY_0=core.alternateRefsCommand"
  "GIT_CONFIG_VALUE_0=/usr/bin/false"
  "GIT_CONFIG_GLOBAL=/dev/null"
  "GIT_CONFIG_NOSYSTEM=1"
  "GIT_CONFIG_SYSTEM=/dev/null"
  "GIT_NO_LAZY_FETCH=1"
  "GIT_NO_REPLACE_OBJECTS=1"
)

source_object_format="$(
  "${git_environment[@]}" "${system_git}" -C "${source_repository}" \
    --no-replace-objects rev-parse --show-object-format
)"
if [[ "${source_object_format}" != "sha1" ]]; then
  echo "Exact qualification requires a SHA-1 source object database." >&2
  exit 1
fi
source_object_directory="$(
  "${git_environment[@]}" "${system_git}" -C "${source_repository}" \
    --no-replace-objects rev-parse --path-format=absolute --git-path objects
)"
source_object_directory="$("${system_realpath}" "${source_object_directory}")"
if [[ ! -d "${source_object_directory}" || -L "${source_object_directory}" ]]; then
  echo "Source Git object directory is missing or symbolic." >&2
  exit 1
fi
source_alternates_path="${source_object_directory}/info/alternates"
if [[ -e "${source_alternates_path}" || -L "${source_alternates_path}" ]]; then
  echo "Exact qualification requires a self-contained source object database." >&2
  exit 1
fi

"${git_environment[@]}" "${system_git}" \
  -c init.defaultBranch=qualification \
  init \
  --bare \
  --quiet \
  --template="${empty_template_directory}" \
  "${shadow_repository}"
/usr/bin/printf '%s\n' "${source_object_directory}" >"${shadow_repository}/objects/info/alternates"

selected_object_type="$(
  "${git_environment[@]}" "${system_git}" \
    --git-dir="${shadow_repository}" \
    --no-replace-objects \
    cat-file -t "${expected_git_commit}"
)"
if [[ "${selected_object_type}" != "commit" ]]; then
  echo "Scheduler-selected Git SHA must identify a commit object exactly." >&2
  exit 1
fi
"${git_environment[@]}" "${system_git}" \
  --git-dir="${shadow_repository}" \
  update-ref refs/heads/qualification "${expected_git_commit}"
"${git_environment[@]}" "${system_git}" \
  --git-dir="${shadow_repository}" \
  symbolic-ref HEAD refs/heads/qualification

"${git_environment[@]}" "${system_git}" \
  -c protocol.file.allow=always \
  clone \
  --branch=qualification \
  --no-checkout \
  --no-hardlinks \
  --no-local \
  --no-tags \
  --quiet \
  --single-branch \
  --template="${empty_template_directory}" \
  --upload-pack="${system_git_upload_pack}" \
  "${shadow_repository}" \
  "${target_checkout}"

target_alternates_path="${target_checkout}/.git/objects/info/alternates"
if [[ -e "${target_alternates_path}" || -L "${target_alternates_path}" ]]; then
  echo "Exact checkout unexpectedly retained an alternate object database." >&2
  exit 1
fi
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" remote remove origin
if "${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
  symbolic-ref --quiet refs/remotes/origin/HEAD >/dev/null; then
  "${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
    symbolic-ref --delete refs/remotes/origin/HEAD
fi
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" config core.hooksPath /dev/null
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" config core.attributesFile /dev/null
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" config core.fsmonitor false
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" config core.untrackedCache false

checkout_object_type="$(
  "${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
    --no-replace-objects cat-file -t "${expected_git_commit}"
)"
if [[ "${checkout_object_type}" != "commit" ]]; then
  echo "Exact checkout did not copy the selected commit object." >&2
  exit 1
fi
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
  update-ref --no-deref HEAD "${expected_git_commit}"
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
  update-ref -d refs/heads/qualification
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
  --no-replace-objects fsck --full --strict --no-reflogs "${expected_git_commit}"
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
  read-tree --reset "${expected_git_commit}"
"${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
  --no-replace-objects archive --format=tar "${expected_git_commit}" |
  "${system_tar}" -xf - -C "${target_checkout}"

observed_git_commit="$(
  "${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
    --no-replace-objects rev-parse HEAD
)"
observed_status="$(
  "${git_environment[@]}" "${system_git}" -C "${target_checkout}" \
    --no-replace-objects status --short
)"
if [[ "${observed_git_commit}" != "${expected_git_commit}" || -n "${observed_status}" ]]; then
  echo "Exact checkout materialization is not a clean detached selected commit." >&2
  exit 1
fi
