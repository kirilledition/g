# Troubleshooting

This page lists common issues and the first checks to run.

## `g regenie` Rejects an Option

Some REGENIE-style options are recognized but not implemented. BED/PGEN input, SPA, categorical covariate flags, and exact Firth without `--approx` are current examples.

Check the option registry:

```bash
uv run g config explain <option-name>
```

## Missing Step 1 Predictions

`g` does not implement REGENIE Step 1. Produce prediction lists with upstream `regenie` and pass the list through `--pred`.

For repository fixture-data examples, use the development recipes listed in
[Quickstart](quickstart.md#repository-fixture-data).

## Sample Alignment Fails

Check that phenotype and covariate tables contain `IID`. If using `--g-sample-key-mode fid_iid`, also check `FID`.

Rules:

- `iid` mode requires globally unique non-empty IIDs.
- `fid_iid` mode requires unique `(FID, IID)` pairs.
- Binary phenotypes should use REGENIE-style `1` and `2` coding.

## BGEN Trusted Fast Path Issues

Start with the default validation mode:

```bash
--g-trusted-bgen-validation-mode cache_on_miss
```

Use `force_validate` when validating a file or cache state. Use `assume_validated` only for expert workflows where the input has already been checked.

## GPU Is Not Used

Probe JAX first:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Run the probe on a GPU node, not only on a login node. If the accelerator is visible but performance
does not improve, check whether the run is dominated by BGEN decode, transfer, or output. See
[GPU and SLURM](gpu-and-slurm.md) for batch-job examples.

## Resume Does Not Reuse Existing Output

Every resumable run writes `run_manifest.json` and `effective_config.toml`. Resume only when the manifest and execution-plan-affecting inputs still match.

Use:

```bash
--g-resume --g-resume-mode strict
```

when you need stronger validation of existing chunks.

## Documentation Build Fails

Documentation builds are part of the development workflow:

```bash
uv sync --group docs
just docs-build
```

The generated `documentation_rendered_website/` directory is ignored by git and can be removed when you need a clean local rebuild.
