# Troubleshooting

This page lists common failures and the first checks to run.

## Check First

Run these before debugging a full GWAS command:

```bash
uv run g --help
uv run g regenie --help
uv run g config init
uv run g config explain bgen
```

For GPU runs, check the target node:

```bash
hostname
uv run python -c "import jax; print(jax.devices())"
```

For input-dependent failures, confirm the required files exist and are non-empty:

```bash
test -s /path/to/genotypes.bgen
test -s /path/to/phenotypes.tsv
test -s /path/to/regenie_step1_pred.list
```

## `g regenie` Rejects An Option

Some REGENIE-style options are recognized but not implemented. BED/PGEN input,
SPA, categorical covariate flags, sample/variant filters, and exact Firth
without `--approx` are current examples.

Check the option registry:

```bash
uv run g config explain <option-name>
```

See [Compatibility](compatibility.md) for the supported and unsupported surface.

## Missing Step 1 Predictions

`g` does not implement REGENIE Step 1. Produce prediction lists with upstream
`regenie` and pass the list through `--pred`.

For repository fixture-data examples, use the development recipes listed in
[Quickstart](quickstart.md#repository-fixture-data).

## Sample Alignment Fails

Check:

- phenotype and covariate tables contain `IID`;
- tables also contain `FID` when `--g-sample-key-mode fid_iid` is used;
- `iid` mode has globally unique non-empty IIDs;
- `fid_iid` mode has unique `(FID, IID)` pairs;
- binary phenotypes use REGENIE-style `1 = control`, `2 = case` coding;
- selected phenotype and covariate columns are present and spelled exactly.

See [Input Files](input-files.md#sample-identity).

## BGEN Trusted Fast Path Issues

Start with the default validation mode:

```bash
--g-trusted-bgen-validation-mode cache_on_miss
```

Use `force_validate` when validating a file or cache state. Use
`assume_validated` only for expert workflows where the exact input has already
been checked.

If trusted mode fails, rerun without:

```bash
--g-trusted-no-missing-diploid
```

and compare whether the failure is specific to the optimized path.

## GPU Is Not Used

Probe JAX on the same kind of node where the scan runs:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Common causes:

- command ran on a login node without NVIDIA devices;
- GPU dependency group was not installed;
- NVIDIA driver and installed JAX CUDA extra are incompatible;
- scheduler job did not request or receive a GPU;
- command passed `--g-device cpu` through CLI or config.

If the accelerator is visible but performance does not improve, check whether
BGEN decode, transfer, or output dominate. See [GPU and Clusters](gpu-and-clusters.md)
and [Performance Guide](performance-guide.md).

## Resume Does Not Reuse Existing Output

Every resumable run writes `run_manifest.json` and `effective_config.toml`.
Resume only when the manifest and execution-plan-affecting inputs still match.

Use strict validation when in doubt:

```bash
--g-resume --g-resume-mode strict
```

Common causes:

- no `run_manifest.json` exists;
- the output run directory exists but was not produced by the same analysis;
- a source file changed size or modification time;
- a trait, covariate, binary correction, output, dtype, or sample-key option
  changed.

See [Resume and Manifest](resume-and-manifest.md).

## Output Looks Missing

`--out` is a prefix. The default run root is:

```text
<out>.g/
```

Look for per-phenotype directories such as:

```text
trait_0001_phenotype.regenie2_linear.run/
trait_0001_phenotype.regenie2_binary.run/
```

Parquet output uses `parts/`; Arrow output uses `chunks/`; REGENIE text output
uses `regenie/` plus `final.regenie`. See [Output Files](output-files.md).

## Documentation Build Fails

Documentation builds are part of the development workflow:

```bash
uv sync --group docs
just docs-build
```

Most failures are stale Markdown links or pages missing from `zensical.toml`.
The generated `documentation_rendered_website/` directory is ignored by git and
can be removed when you need a clean local rebuild.
