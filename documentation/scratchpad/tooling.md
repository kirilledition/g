# Tooling Notes

> Internal scratchpad. Not user docs. May stale.

## Durable Direction

`tooling/` = dev-only. Standardize benchmark, profiling, data-prep, server setup, guardrails. Not second prod implementation.

```text
Production behavior: src/g and crates/*
Development workflows: tooling/
Workflow names: Justfile
Workflow truth: tooling/configs/
```

Justfile policy:

```text
Environment variables configure the machine.
Hydra configs configure the workflow.
Justfile recipes select the workflow.
```

## Keep

- Hydra saved configs for repeatable workflows.
- Shared command, path, cache, report, and logging helpers.
- Versioned machine-readable benchmark/profile reports.
- Markdown summary = reading layer, not truth.
- Justfile = thin menu.

## Avoid

- Large script-shaped modules.
- Duplicate hand-built `g regenie` argv.
- Inline Python child scripts drifting from public API.
- JSON without schema name/version.
- Long Hydra override lists in Justfile.
- Node-specific common recipe names.

## Future Framework Shape

Useful shared pieces:

```text
ToolSpec
typed tool config dataclasses
RegenieRunSpec
render_g_regenie_cli()
render_regenie_config()
expected_output_run_directory()
versioned report models
artifact manifest helpers
```

## Durable Artifact Format

Durable outputs:

```text
artifact_manifest.json
report.json
events.jsonl
metrics.jsonl
summary.md
logs/
config/
commands/
```

Machine-readable files need schema name/version, producer, git head+dirty, run ID/status, primary metrics, artifact path/size/hash when practical.

`summary.md` summarizes, not replaces, `report.json`.
