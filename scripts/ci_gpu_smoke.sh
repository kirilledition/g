#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/g-gpu-smoke-uv-cache-${USER}-${GITHUB_RUN_ID:-manual}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

. scripts/server_env.sh

export RUSTFLAGS="${RUSTFLAGS:---allow=non-local-definitions}"

artifact_directory="${G_GPU_SMOKE_ARTIFACT_DIR:-artifacts/gpu-smoke}"
mkdir -p \
  "${artifact_directory}/runtime" \
  "${artifact_directory}/linear/logs" \
  "${artifact_directory}/binary/logs"

uv sync --group dev --group gpu --frozen

uv run --no-sync python - <<'PY' | tee "${artifact_directory}/runtime/jax-devices.txt"
import jax

devices = jax.devices()
print(devices)
if not any(device.platform in {"cuda", "gpu"} for device in devices):
    raise SystemExit("No JAX GPU device is visible.")
PY

printf 'trait %s\n' "${PWD}/tests/data/cli_smoke/trait.loco" > "${artifact_directory}/linear/pred.list"
uv run --no-sync g regenie \
  --step 2 \
  --qt \
  --bgen tests/data/bgen/haplotypes.bgen \
  --phenoFile tests/data/cli_smoke/phenotypes.tsv \
  --phenoCol trait \
  --covarFile tests/data/cli_smoke/covariates.tsv \
  --covarColList age \
  --pred "${artifact_directory}/linear/pred.list" \
  --out "${artifact_directory}/linear/smoke" \
  --g-output-run-directory "${artifact_directory}/linear/output-runs" \
  --bsize 2 \
  --g-device gpu \
  --g-variant-limit 4 \
  --g-staging-depth 1 \
  --g-output-format parquet \
  --g-output-chunks-per-arrow-file 1 \
  --g-output-arrow-compression none \
  --g-telemetry profile \
  --g-log-dir "${artifact_directory}/linear/logs" \
  --g-log-file "${artifact_directory}/linear/logs/events-and-tracing.jsonl" \
  --g-stage-timings-json "${artifact_directory}/linear/logs/stage-timings.json" \
  --g-profile-summary-json "${artifact_directory}/linear/logs/profile.summary.json" \
  --no-g-log-stderr

printf 'case %s\n' "${PWD}/tests/data/cli_smoke/trait.loco" > "${artifact_directory}/binary/pred.list"
uv run --no-sync g regenie \
  --step 2 \
  --bt \
  --bgen tests/data/bgen/haplotypes.bgen \
  --phenoFile tests/data/cli_smoke/binary_phenotypes.tsv \
  --phenoCol case \
  --pred "${artifact_directory}/binary/pred.list" \
  --out "${artifact_directory}/binary/smoke" \
  --g-output-run-directory "${artifact_directory}/binary/output-runs" \
  --bsize 2 \
  --g-device gpu \
  --g-variant-limit 4 \
  --g-staging-depth 1 \
  --g-output-format parquet \
  --g-output-chunks-per-arrow-file 1 \
  --g-output-arrow-compression none \
  --g-null-logistic-nonconvergence warn \
  --g-telemetry profile \
  --g-log-dir "${artifact_directory}/binary/logs" \
  --g-log-file "${artifact_directory}/binary/logs/events-and-tracing.jsonl" \
  --g-stage-timings-json "${artifact_directory}/binary/logs/stage-timings.json" \
  --g-profile-summary-json "${artifact_directory}/binary/logs/profile.summary.json" \
  --no-g-log-stderr
