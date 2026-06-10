# Python API

The Python API is a small execution wrapper around the same normalized
configuration path used by the CLI.

## Public Entry Points

```python
from pathlib import Path

from g import api
from g.interface import config

regenie_config = config.RegenieConfig.from_options(
    {
        "step": 2,
        "qt": True,
        "bgen": Path("/path/to/genotypes.bgen"),
        "sample": Path("/path/to/genotypes.sample"),
        "phenoFile": Path("/path/to/phenotypes.tsv"),
        "phenoCol": "phenotype_continuous",
        "covarFile": Path("/path/to/covariates.tsv"),
        "covarColList": "age,sex",
        "pred": Path("/path/to/regenie_step1_qt_pred.list"),
        "out": Path("/path/to/output/g_quantitative_regenie2"),
    }
)

artifacts = api.regenie(regenie_config)
print(artifacts.output_run_directory)
```

Direct option-dictionary form:

```python
from pathlib import Path

from g import api

artifacts = api.regenie.from_options(
    {
        "step": 2,
        "bt": True,
        "bgen": Path("/path/to/genotypes.bgen"),
        "sample": Path("/path/to/genotypes.sample"),
        "phenoFile": Path("/path/to/phenotypes.tsv"),
        "phenoCol": "phenotype_binary",
        "pred": Path("/path/to/regenie_step1_pred.list"),
        "firth": True,
        "approx": True,
        "pThresh": 0.01,
        "out": Path("/path/to/output/g_binary_firth_regenie2"),
    }
)
```

## Option Names

`from_options()` accepts canonical REGENIE names such as `phenoFile` and
`pThresh`, plus native configuration sections such as
`{"compute": {"device": "gpu"}}`.

The same validation applies as the CLI:

- unknown options fail;
- recognized unsupported options fail when active;
- `--qt` and `--bt` semantics are preserved;
- `trait_type` is a Python-only convenience alias for selecting quantitative or
  binary mode.

For the complete option surface, see [CLI](cli.md) and use:

```bash
uv run g config explain
```

## Return Value

`api.regenie(...)` returns `g.runner.RunArtifacts`, exposed as
`g.api.RunArtifacts`. It contains paths such as:

| Attribute | Meaning |
| --- | --- |
| `output_run_directory` | Per-phenotype run directory for single-phenotype runs. |
| `final_dataset` | Parquet dataset directory when part output is available. |
| `final_parquet` | Finalized Parquet path when finalization wrote one. |
| `final_regenie` | Final REGENIE text path when text output is selected. |
| `effective_config` | Written effective config path. |
| `phenotype_artifacts` | Per-phenotype artifacts for multi-phenotype runs. |
| `run_id` | Telemetry run identifier when telemetry created one. |

## Runtime Boundaries

The API configures process-global logging, Rayon thread count, and JAX runtime
policy before execution. Repeated calls in the same process should reuse
compatible runtime settings. Start a new process if you need incompatible
logging, JAX, or thread-pool settings across runs.

For batching-related performance guidance, see [Performance Guide](performance-guide.md).
