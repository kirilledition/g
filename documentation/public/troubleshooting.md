# Troubleshooting

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 common user failures | Public user docs |

This page lists common failures and the first checks to run.

## Check First

Run these before debugging a full GWAS command:

```bash
uv run g --help
uv run g regenie --help
```

This experimental Rust CLI/config branch does not expose the previous
`g config init`, `g config validate`, or `g config explain` helper commands.

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

Check `uv run g regenie --help` for the currently supported CLI surface on this
experimental Rust frontend branch. Familiar REGENIE flags that are absent from
help are not accepted yet.

Common absent flags include `--bed`, `--pgen`, `--keep`, `--remove`,
`--extract`, `--exclude`, `--catCovarList`, `--test`, `--t2e`, and `--spa`.

See [Compatibility](compatibility.md) for the supported and unsupported surface.

## TOML Config Fails To Parse Or Validate

Check that the file is valid TOML and that option names use the current
sectioned config surface. The current Rust frontend validates config as part of
constructing the `g regenie` run; it does not provide a separate config-only
validation command.

`g` rejects unknown sections, unknown keys, empty selected column names, and
binary-only options in quantitative mode. For CLI-to-TOML spelling, see
[Configuration](configuration.md#cli-to-toml-mapping).

## `g regenie` Prints `Error: ...`

Runtime failures are reported as a concise stderr line and exit code `1`, not a
Python traceback. Re-run with telemetry or logging enabled when you need the
structured failure event and detailed diagnostics:

```toml
[diagnostics]
telemetry = "profile"
log_dir = "/path/to/logs"
```

## Missing Step 1 Predictions

`g` does not implement REGENIE Step 1. Produce prediction lists with upstream
`regenie` and pass the list through `--pred`.

For repository fixture-data examples, use the development recipes listed in
[Quickstart](quickstart.md#repository-fixture-data).

## Binary Phenotype Coding Fails

Binary traits must use REGENIE-style coding in the phenotype file:

| Value | Meaning |
| --- | --- |
| `1` | Control. |
| `2` | Case. |

Other non-missing values are rejected rather than silently recoded. Confirm the
selected phenotype column, missing-value tokens, and delimiter before debugging
the binary kernel.

## Sample Alignment Fails

Check:

- phenotype and covariate tables contain `IID`;
- tables also contain `FID` when `--sample_key_mode fid_iid` is used;
- `iid` mode has globally unique non-empty IIDs;
- `fid_iid` mode has unique `(FID, IID)` pairs;
- binary phenotypes use REGENIE-style `1 = control`, `2 = case` coding;
- selected phenotype and covariate columns are present and spelled exactly.

See [Input Files](input-files.md#sample-identity).

## BGEN Trusted Fast Path Issues

Start with the default validation mode:

```toml
[compute]
trusted_bgen_validation_mode = "cache_on_miss"
```

Use `force_validate` when validating a file or cache state. Use
`assume_validated` only for expert workflows where the exact input has already
been checked.

If trusted mode fails, rerun with:

```toml
[compute]
trusted_no_missing_diploid = false
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
- `[compute].device` resolved to `"cpu"` in the effective config.

If the accelerator is visible but performance does not improve, check whether
BGEN decode, transfer, or output dominate. See [GPU and Clusters](gpu-and-clusters.md)
and [Performance Guide](performance-guide.md).

## Out Of Memory

Reduce the largest shape-driving knobs first:

```toml
[trait]
bsize = 4096

[compute]
firth_batch_size = 256

[output]
writer_queue_depth = 1
```

For GPU runs, also check whether the command is repeatedly recompiling with
different shapes or keeping too many results in flight. Use
`[diagnostics].telemetry = "profile"` on a representative bounded run before
changing production settings.

## Resume Does Not Reuse Existing Output

Every resumable run writes `run_manifest.json` and `effective_config.toml`.
Resume only when the manifest and execution-plan-affecting inputs still match.

Use strict validation when in doubt:

```toml
[output]
resume = true
resume_mode = "strict"
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

## Finalization Fails But Chunks Exist

Chunked output is the resumable authority. If `final.parquet` or
`final.regenie` is missing after an interruption or storage failure, rerun the
same command with:

```toml
[output]
resume = true
resume_mode = "strict"
```

If finalization continues to fail, inspect free space and permissions for the
run directory and destination filesystem.

## Approximate Firth Reports `TEST_FAIL`

`TEST_FAIL` on approximate-Firth rows means score testing completed but the
fallback correction did not produce a valid corrected statistic for that
variant. First checks:

- confirm `--bt --firth --approx` was intended;
- compare candidate density by changing `--pThresh` on a small subset;
- inspect profile logs for Firth solver iteration or line-search failures;
- compare against upstream REGENIE only with equivalent Firth settings.

## Documentation Build Fails

Documentation builds are part of the development workflow:

```bash
uv sync --group docs
just docs-build
```

Most failures are stale Markdown links or pages missing from `zensical.toml`.
The generated `documentation_rendered_website/` directory is ignored by git and
can be removed when you need a clean local rebuild.
