# Troubleshooting

This page lists common issues and the first checks to run.

## `g regenie` Rejects an Option

Check `uv run g regenie --help` for the currently supported CLI surface on this
experimental Rust frontend branch. REGENIE flags that are absent from help are
not accepted yet.

Common absent flags include `--bed`, `--pgen`, `--keep`, `--remove`,
`--extract`, `--exclude`, `--catCovarList`, `--test`, `--t2e`, and `--spa`.

## Missing Step 1 Predictions

`g` does not implement REGENIE Step 1. Produce prediction lists with upstream `regenie` and pass the list through `--pred`.

For local binary examples:

```bash
just setup-binary-baseline
```

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
just doctor-jax
```

On the server, use SLURM for GPU checks:

```bash
just slurm-gpu-run 'uv run --no-sync python scripts/probe_jax_runtime.py'
```

If the accelerator is visible but performance does not improve, check whether the run is dominated by BGEN decode, transfer, or output.

## Resume Does Not Reuse Existing Output

Every resumable run writes `run_manifest.json` and `effective_config.toml`. Resume only when the manifest and execution-plan-affecting inputs still match.

Use:

```bash
--g-resume --g-resume-mode strict
```

when you need stronger validation of existing chunks.

## Documentation Build Fails

Install and build through the project recipes:

```bash
uv sync --group docs
just docs-build
```

The generated `site/` directory is ignored by git and can be removed when you need a clean local rebuild.
