# Verily Workbench deployment

The Workbench runner packages a **REGENIE Step 2** analysis with pinned inputs,
an explicit data release, conservative memory planning and success-only result
publication. It can be developed and checked using public or synthetic data.
All of Us data access, release-specific preparation, reference parity and
full-cohort performance must be qualified inside your authorized workspace.

Use [the All of Us preparation guide](all-of-us.md) for scientific preparation
and the pilot checklist. Genome-wide REGENIE Step 1 remains external: supply
one matching LOCO prediction file per selected phenotype. This package neither
downloads a REGENIE binary nor substitutes a chromosome-only Step 1 fit.

## Choose an execution route

| Route | Intended use | Remaining workspace check |
| --- | --- | --- |
| Existing GPU Workbench app, isolated Python environment | First GPU pilot and profiling | Python installation policy, GPU visibility, disk and data permissions |
| Pinned command-line container | Reproducible task runtime | Approved registry, image access and NVIDIA container runtime |
| Portable WDL chromosome scatter | CPU orchestration and input/output localization | Workflow backend, disk sizing, service-account access and call caching |

The shipped image is a command-line task, not a web server. Do not point the
Custom App devcontainer selector at it and assume it provides Jupyter. Workbench
also documents a **custom base image** route that installs JupyterLab and its
workspace dependencies during app creation; qualify that integration inside
the actual workspace before relying on it. A bespoke devcontainer must satisfy
Workbench's app-network and exposed-port requirements. See [custom base images](https://support.workbench.verily.com/docs/guides/cloud_apps/advanced_app_usage/create_container_images/)
and [custom apps](https://support.workbench.verily.com/docs/guides/cloud_apps/cloud_app_types/custom/).

## Install in an existing Workbench app

Transfer the reviewed source revision and its qualified **portable** Linux
x86-64 wheel into the workspace using approved software-transfer procedures.
Keep participant data and all run artifacts in the workspace. The local
developer extension built with `target-cpu=native` is not a portable wheel.
Generic CPU flags do not make a wheel compatible with every Linux release:
its wheel platform tag and minimum glibc version must also match the Workbench
app. The container provides a matching pinned operating system. If an existing
app cannot load the qualified wheel, build against that app or an appropriate
manylinux build environment before running the pilot.

Create a dedicated environment from the project root; do not replace the
notebook's existing environment:

```bash
uv python install 3.14.7
UV_PROJECT_ENVIRONMENT=.venv-workbench uv sync \
  --python 3.14.7 --locked --no-default-groups --group workbench --no-install-project
uv pip install --python .venv-workbench/bin/python --no-deps /approved/path/g-0.1.0-PORTABLE.whl
export PYTHONPATH="$PWD"
export PATH="$PWD/.venv-workbench/bin:$PATH"
.venv-workbench/bin/g --help
.venv-workbench/bin/python -c 'import jax; print(jax.devices("gpu"))'
```

Replace the illustrative wheel path with the actual artifact. This installs
the lockfile's CUDA 12 dependencies, JAX 0.11.0 and nvCOMP 5.3.0.16, plus
Hydra, OmegaConf and PyArrow. The Workbench group excludes the profiling,
testing, linting and documentation dependency groups. GPU drivers are provided
by the host, not installed by this environment. Use a GPU supported by the
locked JAX libraries and the embedded compute_70 PTX; actual GPU detection and
decoder tests are required before accepting a new machine type.

The runner supports local paths and GCS. GCS mode uses the Workbench-provided
`gcloud storage` command and its workspace identity; it can accept an explicit
requester-pays billing project. Do not copy credentials into a software image.

## Build the portable container

Build on an x86-64 machine allowed to run container builds, from the repository
root. On Gauss, builds belong on a compute node, never the head node's Docker
daemon:

```bash
docker build --platform linux/amd64 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --file deploy/workbench/Dockerfile --tag g-workbench:reviewed .
docker run --rm g-workbench:reviewed --help
docker run --rm --entrypoint g g-workbench:reviewed --help
```

The multi-stage Dockerfile pins Python 3.14.7, Rust 1.97.1 and uv 0.11.14 by
verified amd64 digest, uses a dated Debian package snapshot, builds with the
Cargo and uv lockfiles and records their hashes in `/opt/g/build-locks.sha256`.
Rust and C++ compilation explicitly target generic x86-64. There is no mold
requirement or host CUDA toolkit dependency for the native extension: its
checked-in verified PTX is embedded at build time. JAX still compiles
shape-specific programs and uses its locked CUDA runtime components.
The runtime runs as UID/GID 1000 and contains no Rust compiler,
maturin, tests or full development environment. Tooling source is included to
support the Hydra runner, but only the Workbench dependencies are installed.

The build context uses an allowlist and excludes local environments, Git
metadata, results, genomic files, credentials and generated native binaries.
The Workbench image CI builds and checks the Dockerfile on a hosted CPU runner,
then runs real native quantitative and binary jobs using the participant-free
demo. It requires committed outputs for all 32 variants and checks exact
missing-call sample counts (118 versus 128). It does not publish an image or
run GPU qualification.

For a GPU pilot with **already localized** inputs and a local output directory:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/approved/workspace/pilot,dst=/work \
  g-workbench:reviewed --config-name workbench \
  tool.action=run tool.manifest=/work/job.json \
  tool.work_directory=/work/attempts tool.dry_run=false
```

All manifest paths must exist inside the container, and the mounted work
directory must be writable by the chosen user. The core image intentionally
does not include Google Cloud CLI. Localize GCS objects inside the workspace
first, use WDL file localization, or run the environment route with Workbench's
existing `gcloud`. Shell command substitutions above contain only local user
and revision metadata.

Before publishing the image later, choose an approved Artifact Registry
repository and confirm that the app/workflow identity can pull it. Record and
use the published image **digest**, rather than a mutable tag, for research
attempts. Registry creation, image publication, GPU quota allocation and paid
cloud resources are not performed by these assets. Workbench's documented
[image creation workflow](https://support.workbench.verily.com/docs/guides/cloud_apps/advanced_app_usage/create_container_images/)
explains workspace registry authentication and access.

## Pilot and result publication

Create a version-one job manifest following the preparation guide. It records
each input's SHA-256 and byte size, the release/reference build, selected
phenotypes and covariates, LOCO predictions, explicit resource limits and the
output location. GCS inputs can also pin an immutable generation.

```bash
.venv-workbench/bin/python -m tooling.cli.workbench --config-name workbench \
  tool.action=preflight tool.manifest=/approved/workspace/job.json \
  tool.work_directory=/approved/workspace/attempts
.venv-workbench/bin/python -m tooling.cli.workbench --config-name workbench \
  tool.action=run tool.manifest=/approved/workspace/job.json \
  tool.work_directory=/approved/workspace/attempts tool.dry_run=false
```

Each attempt gets its own localized inputs, native configuration, cache,
logs and status. An output bundle is accepted only after checking completed
Parquet files and native completion metadata; `COMMITTED.json` is published
last. A failed attempt does not publish a successful bundle. Retain failed
attempts and logs in the workspace for diagnosis.

Size persistent disk for the original/localized inputs, the runner's verified
copies, temporary files, outputs and JAX cache. This first implementation can
hold two copies of an already localized BGEN. The explicit memory estimate
chooses a smaller chunk for large cohorts; it is a planning bound, not proof
that every JAX/native allocation fits. Observe peak memory during the pilot
before increasing sample counts or concurrent traits.

## Portable WDL workflow

`deploy/workbench/all_of_us_step2.wdl` scatters independent chromosome jobs.
Every BGEN, sample, phenotype, optional covariate and prediction is declared as
a WDL `File`; the executor localizes them. Predictions must appear in the same
order as the manifest's `predictions` array. The helper changes only the URI
to the localized path, removes cloud-only generation fields and retains the
original content hashes and byte sizes. The runner verifies those bytes before
execution. Both source and localized manifests are returned for provenance.

Validate the workflow without executing it:

```bash
uv tool run --python 3.12 --from miniwdl==1.15.0 miniwdl check \
  deploy/workbench/all_of_us_step2.wdl
```

Workflow inputs are `all_of_us_step2.chromosomes` (an array of `ChromosomeJob`
objects), `all_of_us_step2.image` (a registry image digest),
`all_of_us_step2.cpu_threads` and `all_of_us_step2.memory_gib`. Each job has
`manifest`, `bgen`, `sample`, `phenotype`, nullable `covariates`, and an ordered
`predictions` file array. An executor can receive workspace GCS URLs for the
`File` fields. Each call returns a `result.tar` containing one committed
bundle, together with source and localized manifests. These outputs remain
within the approved workflow storage destination.

This portable template reserves **CPU only**, and rejects GPU manifests. It
declares only container, CPU and memory runtime settings. Actual disk sizing,
GPU resource keys, retries, service-account permissions and task-cache policy
are backend-specific and must be configured and tested inside the workspace.
Use the direct GPU app route for the first GPU pilot. WDL syntax validation
does not qualify Verily execution or All of Us data access.

## Recovery and qualification limits

The native engine's resume identity still depends on local filesystem identity.
Freshly downloaded copies on another VM do not preserve it. Keep the original
attempt and localized files for same-VM recovery; rerun an interrupted
chromosome as a new attempt after a VM change. The Workbench publication layer
does not relax native input-identity checks or promise cross-VM resume.

Before research use, run reference parity against REGENIE, missing-call and
allele-orientation checks, low-count and binary-imbalance cases, interrupted
attempt tests, and full-cohort memory/disk/cost measurements. The existing
2,504-sample public-data measurements are not All of Us capacity evidence.
WGS missingness currently sends BGEN decoding through the CPU fallback while
association can still use the GPU; record the actual selected execution path.

Keep individual-level data, prediction files, manifests, logs and profiling
artifacts inside the controlled workspace. Summary-statistic export requires
the applicable All of Us dissemination review. See the [current dissemination policy](https://support.researchallofus.org/hc/en-us/articles/22346276580372-Data-and-Statistics-Dissemination-Policy).
