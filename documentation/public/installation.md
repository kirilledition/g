# Installation

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 source checkout installs | Public user docs |

`g` is pre-release and is not published on PyPI. Install it from a Git checkout and run it through
the checkout's `uv` environment. This keeps `g` in a repository-local `.venv/` and does not install
packages into your system Python, Conda base environment, or other projects.

Use this page when you want to run your own GWAS. If you want to change the code, skip to
[Development Installation](#development-installation).

## External Tools

Install or load these tools before syncing the Python environment:

| Tool | Needed for | Install page |
| --- | --- | --- |
| `git` | Clone the repository | [Git install guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) |
| `uv` | Create the isolated Python environment and install Python `3.14` | [uv installation](https://docs.astral.sh/uv/getting-started/installation/) |
| Rust/Cargo | Build the native extension from source | [rustup installation](https://rustup.rs/) |
| C/C++ build tools | Compile native dependencies used by Rust crates | [GCC binaries](https://gcc.gnu.org/install/binaries.html) or your cluster module guide |
| NVIDIA driver and a CUDA-capable node | GPU runs only | [JAX installation](https://docs.jax.dev/en/latest/installation.html) |
| upstream `regenie` | Produce Step 1 prediction lists for `--pred` | [REGENIE installation](https://rgcgithub.github.io/regenie/install/) |
| SLURM | Cluster job submission | [SLURM `sbatch`](https://slurm.schedmd.com/sbatch.html) |

`just` is not required to run `g`; it is a development task runner for this repository.

## Supported Platforms

| Platform | Status |
| --- | --- |
| Linux source checkout | Primary supported install path. |
| Linux GPU node with compatible NVIDIA driver | Supported through the GPU dependency group and JAX CUDA wheels. |
| Shared Linux cluster without root | Supported when the checkout, `.venv/`, and caches are user-writable. |
| macOS | Untested for production scans; CPU-only development may work if the native extension builds locally. |
| Windows | Unsupported and untested. |

Known unsupported distribution modes:

- PyPI package: not published.
- Conda package: not published.
- System Python or shared Conda base install: not recommended.

## Verify The Install

After any install flow, verify the command surface from the same checkout and
environment that will run the scan:

```bash
uv run g --help
uv run g regenie --help
uv run python -c "import g; print(g.__name__)"
```

## CPU Install From Source

Clone the repository, let `uv` install the required Python version, and sync only runtime
dependencies:

```bash
git clone https://github.com/kirilledition/g.git
cd g
uv python install 3.14
uv sync --python 3.14 --no-dev
uv run g --help
uv run g regenie --help
```

`uv sync` creates `.venv/` inside the checkout and installs the local project there. Re-run
`uv sync --python 3.14 --no-dev` after pulling a new commit.

Use `--frozen` when you want `uv` to enforce the checked-in lockfile without changing it:

```bash
uv sync --python 3.14 --no-dev --frozen
```

First checks after install:

```bash
uv run g --help
uv run g regenie --help
```

## GPU Install From Source

First make sure the cluster node or workstation has a compatible NVIDIA driver. Then install the
repository GPU dependency group:

```bash
git clone https://github.com/kirilledition/g.git
cd g
uv python install 3.14
uv sync --python 3.14 --no-dev --group gpu
uv run python -c "import jax; print(jax.devices())"
```

The `gpu` group installs the CUDA-enabled JAX extra declared by this checkout. If JAX does not list
the expected GPU, compare the installed extra with the current [JAX installation
matrix](https://docs.jax.dev/en/latest/installation.html), then adjust the environment before
measuring performance.

Run `g` with:

```bash
--device gpu
```

Use `--device cpu` for CPU execution. CPU mode is installed by the base runtime dependencies.

## Run Your GWAS

`g` implements REGENIE Step 2 over BGEN input. It does not implement Step 1, so create the Step 1
prediction list with upstream `regenie` first and pass that file with `--pred`.

Minimal CPU shape:

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_name \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_regenie2 \
  --device cpu \
  --format parquet
```

For a GPU run, use the same command with `--device gpu` and submit it on a GPU node rather than a
login node.

See [Quickstart](quickstart.md) for quantitative and binary command examples, [CLI](cli.md) for
supported flags, [Input Files](input-files.md) for input contracts, and
[Output Files](output-files.md) for artifact expectations.

## Linux Cluster Without Root

On shared clusters, keep the checkout and caches in a user-writable filesystem such as `$SCRATCH`
when available:

```bash
mkdir -p "${SCRATCH:-$HOME}/gwas-engine" "${SCRATCH:-$HOME}/.cache/uv"
export UV_CACHE_DIR="${SCRATCH:-$HOME}/.cache/uv"
export UV_LINK_MODE=copy

cd "${SCRATCH:-$HOME}/gwas-engine"
git clone https://github.com/kirilledition/g.git
cd g
uv python install 3.14
uv sync --python 3.14 --no-dev
```

If your site provides environment modules, load `git`, Rust/Cargo, and GPU-driver-compatible CUDA
modules before running `uv sync`. If it does not, use the user-local installers linked in
[External Tools](#external-tools). Avoid `sudo` and avoid installing into a shared system Python.

Generic GPU SLURM job shape:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=g-regenie2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd /path/to/g
export UV_CACHE_DIR="${SCRATCH:-$HOME}/.cache/uv"
export UV_LINK_MODE=copy

uv run --no-sync g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_name \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_regenie2 \
  --device gpu
```

Adjust scheduler options for your cluster. Do not run GPU scans or large CPU scans on a login node.

## Development Installation

Development workflows use `just`, development dependency groups, local checks, and sometimes the
repository's SLURM wrappers. Start with the development documentation instead of the consumer flow:

- [Development](../development/index.md)
- [No-Nix Development](../development/no-nix-development.md)
- [Server Gauss SLURM](../development/server-gauss-slurm.md)

Common development setup:

```bash
just dev-bootstrap
just doctor
just check-local
```

GPU-capable development setup:

```bash
just dev-bootstrap-gpu
just doctor-jax
```

Documentation dependencies also belong to the development flow:

```bash
uv sync --group docs
just docs-build
```
