I reviewed the current `main` documentation as a documentation system. The foundation is good: you now have a real site tree under `documentation/`, explicit Zensical navigation, and a clear audience split into public, development, and scratchpad material. The docs are still draft-like, but the direction is much better than the old loose `docs/` pile.

My main recommendation: **turn the current docs into a strict information architecture with one owner per topic, and make README only a portal.** Right now several topics are repeated across README, public pages, and development pages. That will drift quickly while the CLI/config design is changing.

---

# Current state: what is good

## 1. The directory split is right

The root documentation page already defines the intended split:

```text
documentation/public/      user-facing guidance
documentation/development/ maintainer/contributor guidance
documentation/scratchpad/  internal work products and historical notes
```

That is exactly the right segmentation. The root page says public docs are for installing/configuring/running, development docs are for maintainers, and scratchpad is for internal notes and historical artifacts. 

The development documentation page repeats the same model and adds operational rules for building and publishing docs. 

Keep this.

## 2. Zensical navigation is explicit and useful

`zensical.toml` has a clean navigation tree:

```text
Home
User Guide
Development
Scratchpad
```

with ordered pages for installation, quickstart, algorithm, CLI, configuration, input/output, GPU/SLURM, troubleshooting, architecture, style, tooling, logging, performance, SIMD, and scratchpad notes. 

This is good. Explicit nav is better than file-system auto-nav for a scientific tool.

## 3. Public docs already have the right core pages

The public guide has:

```text
index
getting-started
installation
quickstart
algorithm
cli
configuration
input-output
gpu-and-slurm
troubleshooting
```

That is a strong user-facing set. The public index accurately states the project scope: pre-release BGEN-backed REGENIE Step 2, no Step 1, Rust file handling, JAX kernels, BGEN 1.2, Arrow/Parquet output, GPU through JAX. 

## 4. Algorithm page is unusually valuable

The algorithm page is one of the best docs in the set. It explains the actual model, LOCO usage, binary score test, approximate Firth surface, parameter effects, and result fields. It also explicitly states that changing phenotype/covariate/prediction/sample settings changes statistics, not just runtime. 

This page should remain a first-class public page, not a scratchpad.

---

# Biggest problems to fix

## 1. README is too much of a second documentation site

The README currently includes:

```text
product description
architecture
documentation map
current status
setup
quickstart
CLI model
config files
Python API
input conventions
multi-phenotype behavior
output layout
telemetry
performance notes
architecture
development workflow
known limitations
```

For example, README has full quickstart commands, config examples, Python API examples, output layout, telemetry tables, performance knobs, and development commands.   

That duplicates `documentation/public/quickstart.md`, `configuration.md`, `input-output.md`, `gpu-and-slurm.md`, `development/index.md`, and `architecture.md`.

**Recommendation:** make README a short portal:

```text
1. What is g?
2. Current scope / limitations
3. Install in 3 commands
4. One minimal Step 2 command
5. Link to full docs
6. Link to development docs
```

Target README length: around 150–250 lines, not a full manual.

Why: when the config branch lands, the README will drift first. The canonical details should live in the site pages.

---

## 2. Some docs already contradict each other

### `--bsize` default mismatch

README says:

```text
--bsize ... default 8192
```



But `src/g/config.default.toml` on `main` says:

```toml
[trait]
bsize = 16384
```



The public configuration example also uses `bsize = 8192`, which is fine as an example, but should not be described as the default. 

**Fix:** never handwrite defaults in prose unless generated or verified. Prefer:

```text
Default: see `g config init` or `src/g/config.default.toml`.
```

For stable public docs, include defaults only in one generated “Option reference” page.

---

### CLI/config commands may drift with Rust CLI branch

Public CLI docs list:

```text
g config init
g config validate
g config explain
```



The active Rust CLI/config branch may remove or redesign those commands. When that branch settles, this page must be updated immediately.

**Recommendation:** add a docs rule: any branch that changes CLI/config must update:

```text
documentation/public/cli.md
documentation/public/configuration.md
documentation/development/configuration_cli_architecture.md
README.md only if the top-level examples change
```

---

## 3. Public configuration page is too thin for a central feature

`documentation/public/configuration.md` says config files use the same option names as CLI, grouped by section.  But then the example uses:

```toml
[g.compute]
device = "cpu"
staging-depth = 1
```

while the CLI uses:

```bash
--g-device
--g-staging-depth
```

The development config architecture page explains this better: `[g.compute]` supplies the namespace, so TOML keys do not repeat `g-`. 

**Fix:** move that explanation into the public configuration page too. Users need it more than developers.

Also add:

```text
- CLI override semantics
- boolean negative flags
- required values vs defaults
- effective_config.toml
- config examples for:
  - server tuning config
  - run-specific config
  - mixed TOML + CLI phenotype override
```

---

## 4. Development docs contain too many operational one-offs

`documentation/development/index.md` includes a very specific worktree path:

```text
/mnt/beegfs/kirill/Projects/g-worktrees/symphony
```



That is useful for your current server, but it should not be in the main development landing page. Put site-specific paths in:

```text
development/server-gauss.md
development/symphony.md
```

The development index should be generic and durable.

---

## 5. Scratchpad is published in nav

The Zensical nav includes scratchpad pages directly.  The docs operations page says all three directories can be published, but scratchpad-only updates do not rebuild/deploy automatically. 

This is okay for a private/pre-release project, but for final docs I would not publish scratchpad in the main public site nav. It can confuse users.

**Recommendation for final version:**

```text
Public site:
  public/
  maybe selected development pages

Internal site or repo-only:
  scratchpad/
```

If you keep scratchpad published, add a clear banner to every scratchpad page:

```text
Internal development note. Not user-facing documentation. May be stale.
```

---

# Documentation principles for the final version

Use these as the project documentation contract.

## 1. One page owns one topic

No major topic should have multiple canonical versions.

Recommended ownership:

| Topic                             | Canonical owner                                  |
| --------------------------------- | ------------------------------------------------ |
| Product summary                   | `README.md`, `documentation/public/index.md`     |
| Install                           | `documentation/public/installation.md`           |
| First run                         | `documentation/public/quickstart.md`             |
| CLI flags                         | `documentation/public/cli.md`                    |
| TOML config                       | `documentation/public/configuration.md`          |
| Input/output file contracts       | `documentation/public/input-output.md`           |
| Mathematical/statistical behavior | `documentation/public/algorithm.md`              |
| GPU/SLURM user usage              | `documentation/public/gpu-and-slurm.md`          |
| Errors and fixes                  | `documentation/public/troubleshooting.md`        |
| Internal architecture             | `documentation/development/architecture.md`      |
| Config/CLI developer contract     | `documentation/development/configuration-cli.md` |
| Telemetry/logging design          | `documentation/development/telemetry.md`         |
| Performance methodology           | `documentation/development/performance.md`       |
| Historical experiments            | `documentation/scratchpad/`                      |

README should link, not duplicate.

## 2. Public docs describe behavior, development docs describe implementation

Public:

```text
What should I run?
What files do I need?
What output do I get?
What does this option mean statistically?
How do I troubleshoot?
```

Development:

```text
Where is this implemented?
How do I add an option?
How do I add a backend?
How do I test parity?
How do I profile?
What invariants must not be broken?
```

Scratchpad:

```text
What did we try?
What failed?
What historical benchmark/review should we preserve?
```

## 3. Every page must declare freshness

At top or bottom of long pages, include:

```md
Status: pre-release draft
Applies to: main branch as of <commit/date>
Owner: public/user docs | development/architecture | scratchpad
```

For pages like `algorithm.md`, `configuration_cli_architecture.md`, and `simd-optimization-reference.md`, this matters a lot.

## 4. Defaults should not be copied manually

Do not write:

```text
default bsize is 8192
```

in many places.

Use one of:

```text
See `g config init`
See `src/g/config.default.toml`
Generated option reference
```

For final docs, generate an option reference page from the Rust/Python config schema.

## 5. Examples should be executable shapes, not exact local paths

Examples should use placeholders:

```bash
/path/to/genotypes.bgen
/path/to/phenotypes.tsv
```

Repository fixture examples should be clearly marked as developer/evaluator examples, not normal user examples. Quickstart currently does this reasonably. 

## 6. Public docs should not depend on `just`

Public install/run docs should use:

```bash
uv run g ...
```

Development docs can use:

```bash
just ...
```

The installation page already says `just` is not required to run `g`; it is a development task runner.  Good rule.

---

# Recommended final file structure

I would restructure to this:

```text
README.md

documentation/
  index.md

  public/
    index.md
    installation.md
    quickstart.md
    cli.md
    configuration.md
    input-files.md
    output-files.md
    algorithm.md
    gpu-and-clusters.md
    troubleshooting.md
    api-python.md
    compatibility.md
    performance-guide.md

  development/
    index.md
    architecture.md
    configuration-frontend.md
    execution-pipeline.md
    compute-kernels.md
    native-io.md
    output-writer.md
    telemetry.md
    testing-and-parity.md
    benchmarking.md
    style-guide.md
    tooling.md
    server-gauss.md
    automation.md
    roadmap.md
    documentation.md

  scratchpad/
    index.md
    decisions/
      simd-bgen.md
      output-performance.md
      firth-parity.md
    reviews/
      YYYY-MM-DD-code-review.md
    experiments/
      YYYY-MM-DD-benchmark-name.md
```

Notes:

* Rename uppercase docs in development for consistency:

  * `ROADMAP.md` → `roadmap.md`
  * `STYLEGUIDE.md` → `style-guide.md`
  * `NO_NIX_DEVELOPMENT.md` → `no-nix-development.md`
  * `UBUNTU_SLURM_DEVELOPMENT.md` → `server-gauss-slurm.md` or `ubuntu-slurm-development.md`
* Use lowercase kebab-case for all doc filenames.
* Keep old uppercase filenames only if you need backward compatibility with repo links; otherwise pre-release means rename now.

---

# Suggested final public docs

## `public/index.md`

Purpose: orientation.

Content:

```text
- What g is
- What g is not
- Current support matrix
- Where to start
- Warning: pre-release
```

Current page is close. It already states BGEN-backed Step 2, no Step 1, and has a support table. 

Add:

```text
- “Choose your path”:
  - I have REGENIE Step 1 predictions → Quickstart
  - I need to install on GPU cluster → Installation + GPU/Clusters
  - I need exact output schema → Output files
  - I need to understand statistics → Algorithm
```

---

## `public/installation.md`

Current page is strong. It distinguishes consumer install from development install, lists external tools, and has CPU/GPU/source/SLURM install flows. 

Final improvements:

```text
- Add “supported platforms” table.
- Add “known unsupported install modes”:
  - PyPI not published
  - Conda package not published
  - Windows unsupported/untested if true
- Add “verify install” section:
  uv run g --help
  uv run g regenie --help
  uv run python -c "import g; ..."
```

---

## `public/quickstart.md`

Current page is good and focused. It includes quantitative, binary score, approximate Firth, GPU, REGENIE text output, and fixture data sections. 

Final improvements:

```text
- Add “minimum required files” table.
- Add “what command prints on success”.
- Add “where to find final output”.
- Add “expected run directory” link to output page.
```

---

## `public/cli.md`

Current page is too shallow for final docs. It lists commands and main options, but it should become the authoritative CLI behavior page. 

Final structure:

```md
# CLI

## Entry points
g
g regenie
g-regenie

## Compatibility goal
Existing REGENIE Step 2 commands should mostly work by replacing `regenie` with `g regenie`.

## Command grammar
g regenie [input] [trait] [binary] [output] [g-runtime]

## Supported REGENIE flags
Table:
  option
  type
  default/source
  allowed in qt/bt
  notes

## g-specific flags
Table grouped by:
  compute
  output
  diagnostics

## Boolean overrides
--flag / --no-flag

## Unsupported flags
How unsupported REGENIE flags fail

## Exit codes
0 success
1 runtime/config error
2 CLI usage error

## Examples
quantitative, binary, approximate Firth, config override
```

Generate the option tables if possible.

---

## `public/configuration.md`

This should become one of the central pages.

Current page is only 112 lines and covers merge order, examples, trace caps, sections, and a couple tuning overrides. 

Final structure:

```md
# Configuration

## Overview
TOML config = reproducible equivalent of CLI.

## Merge order
packaged defaults < user TOML < explicit CLI

## Required values
bgen, phenoFile, phenoCol, pred, out

## Sections
[input], [trait], [binary], [output], [g.compute], [g.output], [g.diagnostics]

## CLI-to-TOML mapping
--g-device -> [g.compute] device
--pThresh -> [binary] pThresh

## Boolean override rules
absence vs false, --no-* flags

## Server tuning config pattern
server.toml with technical settings + CLI phenotype/out

## Effective config
where it is written, what it contains

## Validation
what `g config validate` checks and what only run-time checks

## Examples
minimal quantitative
binary Firth
GPU server config
telemetry profile config
```

---

## Split `public/input-output.md`

Current page combines inputs, output layout, schema, text output, resume, and reproducibility. It is good but may grow too large. 

For final docs, split:

```text
input-files.md
output-files.md
resume-and-manifest.md
```

If you keep one page, add a clearer table of file formats:

```text
Input:
  BGEN
  .sample
  phenotype TSV
  covariate TSV
  pred list

Output:
  run directory
  logs
  Arrow chunks
  Parquet parts
  final.regenie
  effective_config.toml
  run_manifest.json
```

---

## `public/algorithm.md`

Keep as a long reference page. It is good.

Final improvements:

```text
- Add “trust and parity status” section.
- Add “implementation notes vs mathematical contract” boxes.
- Add “what affects statistics” table.
- Add “what affects performance only” table.
- Add “known differences from REGENIE” if any.
```

It already states the implemented statistical surface and the execution flow. 

---

## `public/gpu-and-slurm.md`

Current page is good: JAX device probe, GPU/CPU SLURM examples, cluster notes, runtime knobs, fair comparison warning. 

Final improvements:

```text
- Rename to `gpu-and-clusters.md`.
- Add cold vs warm JAX compilation note.
- Add JAX persistent cache section.
- Add “when GPU may not help” section.
- Add “benchmark protocol” link to performance guide.
```

---

## `public/troubleshooting.md`

Current page is a good seed but too short. 

Final sections to add:

```text
- Config parsing errors
- Unknown flag
- Unsupported REGENIE flag
- Missing Step 1 predictions
- Sample/covariate alignment
- Binary phenotype coding
- BGEN validation/trusted mode
- GPU not visible
- JAX compilation cache issues
- Out of memory
- Resume rejects manifest
- Output finalization failed
- Approximate Firth non-convergence or TEST_FAIL
- Performance slower than expected
```

For each issue, use:

```md
## Symptom

## Likely cause

## First check

## Fix
```

---

## Add `public/api-python.md`

Current README includes Python API usage.  Move that to a page.

Sections:

```text
- g.regenie(config)
- RegenieConfig.from_toml
- g.regenie.from_options
- RunArtifacts
- Multiple phenotypes
- Python process-global JAX runtime caveats
- Example notebooks/scripts
```

---

## Add `public/compatibility.md`

This project’s user story is “replace `regenie` with `g regenie` for Step 2 BGEN scans.” Make that explicit.

Sections:

```text
- Supported drop-in cases
- Required differences
  - output defaults
  - Step 1 not implemented
  - BGEN only
- Unsupported REGENIE flags
- Binary Firth behavior
- Output compatibility
- How to compare results fairly
```

This page will reduce confusion.

---

## Add `public/performance-guide.md`

Do not bury performance guidance in README.

Sections:

```text
- Cold start vs warm cache vs steady-state
- CPU vs GPU expectations
- Single phenotype vs multi-phenotype
- BGEN decode bottlenecks
- JAX compilation cache
- Telemetry modes
- Benchmark protocol vs REGENIE
- Recommended flags for profiling
```

The GPU page already says GPU acceleration is workload-dependent and single-trait runs may be limited by BGEN decode, transfer, or output.  Expand that into a performance page.

---

# Suggested final development docs

## `development/index.md`

Keep short. Current page has setup, checks, docs, coding rules, and task/worktree notes. 

Final structure:

```text
- Quick setup
- Common checks
- Where to read next
- Contribution rules
```

Move site-specific Symphony worktree paths out.

---

## `development/architecture.md`

Current page is a useful overview but will become stale with Rust config migration: it still says `cli.py` is a Click CLI generated from `OptionSpec` and `interface/options.py` is the option registry. 

Final architecture page should be high-level and stable:

```text
- System diagram
- Python/Rust/JAX responsibilities
- Config frontend
- Execution plan
- Engine pipeline
- Native BGEN/sample/prediction/output
- JAX compute modules
- Telemetry
- Manifest/resume
- Key invariants
```

Avoid too many exact file paths unless they are stable.

---

## `development/configuration-frontend.md`

Rename from `configuration_cli_architecture.md` when final.

This should be the developer contract for adding options. Current page is already close and detailed. It states goals, flow, user behavior, TOML behavior, Python API behavior, developer architecture, defaults, constants, manifest policy, and tests.   

When the Rust config branch settles, update this page to match the final design.

Final sections:

```text
- Goals
- Hard contracts
  - CLI REGENIE compatibility
  - no hidden defaults
  - config/default/manifest invariants
- Rust frontend modules
- Adding a REGENIE-compatible CLI flag
- Adding a g-specific runtime option
- Adding a TOML-only/internal option
- Defaults policy
- Validation policy
- Provenance and explicit options
- Manifest policy
- Required tests
```

---

## `development/telemetry.md`

Rename `logging-setup.md` to `telemetry.md`.

Content:

```text
- Modes: off/progress/profile/trace
- Event streams
- What can be logged in production
- What may synchronize JAX
- Rust tracing ownership
- Python telemetry boundaries
- Profile summary schema
- Trace caps
```

This is important because telemetry directly affects performance.

---

## `development/testing-and-parity.md`

Add this. It is missing.

Sections:

```text
- Unit tests
- Rust tests
- Python tests
- CLI/config tests
- REGENIE parity tests
- GPU tests
- Large data tests
- Markers
- What must be parity-tested before claiming support
```

This app needs explicit parity policy.

---

## `development/benchmarking.md`

Add or expand from `performance-discovery.md`.

Sections:

```text
- Benchmark taxonomy
  - cold start
  - warm cache
  - steady state
- REGENIE comparison rules
- Required environment capture
- Profile mode caveats
- Stage timing interpretation
- How to store benchmark artifacts
```

---

## `development/native-io.md`

Add when stable.

Content:

```text
- BGEN reader
- sample alignment
- prediction loading
- output writer
- manifest/resume
- trusted BGEN path
```

---

## `development/compute-kernels.md`

Add when stable.

Content:

```text
- linear kernels
- binary score kernels
- approximate Firth kernels
- dtype policy
- JAX static args
- shape/cache policy
- host-device transfer rules
```

---

# File naming conventions

Use these rules:

```text
1. All documentation filenames lowercase kebab-case.
2. No spaces, no underscores, no uppercase names except README.md.
3. User docs go in documentation/public/.
4. Maintainer docs go in documentation/development/.
5. Historical notes go in documentation/scratchpad/.
6. Scratchpad filenames should include date or topic:
   YYYY-MM-DD-short-topic.md
7. Do not put active tasks in public docs.
8. Do not put server-private paths in public docs.
9. If a page is draft or historical, say so at the top.
```

Recommended renames:

```text
documentation/development/ROADMAP.md
  -> documentation/development/roadmap.md

documentation/development/STYLEGUIDE.md
  -> documentation/development/style-guide.md

documentation/development/NO_NIX_DEVELOPMENT.md
  -> documentation/development/no-nix-development.md

documentation/development/UBUNTU_SLURM_DEVELOPMENT.md
  -> documentation/development/ubuntu-slurm-development.md
  or server-gauss-slurm.md if it is site-specific

documentation/development/configuration_cli_architecture.md
  -> documentation/development/configuration-frontend.md

documentation/development/logging-setup.md
  -> documentation/development/telemetry.md
```

Update `zensical.toml` after renames.

---

# Rules for connecting pages

Use a simple documentation graph:

```text
README
  -> public/index
  -> public/quickstart
  -> development/index

public/index
  -> installation
  -> quickstart
  -> cli
  -> configuration
  -> input/output
  -> algorithm
  -> troubleshooting

quickstart
  -> installation
  -> input/output
  -> algorithm

cli
  -> configuration
  -> algorithm
  -> compatibility

configuration
  -> cli
  -> input/output
  -> development/configuration-frontend

input/output
  -> algorithm
  -> troubleshooting

gpu-and-clusters
  -> installation
  -> performance-guide
  -> troubleshooting

development/index
  -> architecture
  -> style-guide
  -> testing-and-parity
  -> benchmarking
  -> documentation

architecture
  -> configuration-frontend
  -> native-io
  -> compute-kernels
  -> telemetry
```

Each page should have:

```md
## Next steps

- ...
```

or a “See also” section.

---

# Content I would remove or demote

## From README

Move most of this out:

```text
full quickstart commands
full config example
Python API details
input conventions
multi-phenotype details
output layout details
telemetry details
performance knobs
full architecture tree
development workflow details
```

Keep one minimal command and links.

## From public pages

Remove development-only fixture details unless clearly marked. Quickstart currently has repository fixture data; that is okay but should be under a “Developer/evaluator fixture” box. 

## From development index

Move server-specific path and Symphony-specific worktree note out. 

---

# Specific corrections to make now

1. Fix `--bsize` default mismatch. README says default `8192`, but config default is `16384`.  

2. Update architecture docs when the Rust config branch lands. Current architecture says Click/OptionSpec/Python config owns CLI/config. 

3. Decide whether `g config init|validate|explain` remain public. CLI and configuration docs currently rely on them.  

4. Clarify public configuration naming: CLI `--g-device` maps to TOML `[g.compute] device`, not exactly identical key spelling. Public config page currently says TOML uses same option names as CLI. 

5. Decide whether scratchpad is truly published. Zensical nav includes it. 

---

# Final documentation version I would aim for

A strong final docs set would look like this:

```text
README.md
  short portal, not full manual

documentation/public/
  index.md
  installation.md
  quickstart.md
  cli.md
  configuration.md
  compatibility.md
  input-files.md
  output-files.md
  algorithm.md
  gpu-and-clusters.md
  performance-guide.md
  api-python.md
  troubleshooting.md

documentation/development/
  index.md
  architecture.md
  configuration-frontend.md
  execution-pipeline.md
  native-io.md
  compute-kernels.md
  telemetry.md
  testing-and-parity.md
  benchmarking.md
  style-guide.md
  tooling.md
  server-gauss-slurm.md
  automation.md
  documentation.md
  roadmap.md

documentation/scratchpad/
  index.md
  decisions/
  reviews/
  experiments/
```

This gives you:

```text
public docs:
  stable user contract

development docs:
  stable implementation contract

scratchpad:
  useful but explicitly non-authoritative history
```

---

# Bottom line

The new documentation is much better organized than before. The public/development/scratchpad split is the right foundation, and Zensical navigation is already set up well. The biggest changes I would make before finalizing:

```text
1. Make README a portal, not a duplicate manual.
2. Fix drift around defaults and CLI/config commands.
3. Use lowercase kebab-case filenames.
4. Keep public docs behavior-focused and development docs implementation-focused.
5. Add missing final pages: compatibility, Python API, performance guide, testing/parity, compute kernels, native I/O.
6. Treat scratchpad as non-authoritative, and consider not publishing it in final public nav.
```

Once the Rust config/CLI branch settles, update the documentation around that new architecture immediately; configuration and CLI are currently the highest-drift pages.
