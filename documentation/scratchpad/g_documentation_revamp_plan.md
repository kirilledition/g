# Documentation Revamp Plan for `g`

**Goal:** Turn the current documentation into a best-in-class user and developer documentation system for a pre-release, scientific GWAS engine. The primary user is a professional bioinformatician porting or building REGENIE Step 2 workflows. The secondary user is a student or new analyst who understands GWAS basics imperfectly and needs enough context to install, run, inspect output, and avoid statistical mistakes.

**Repository baseline:** current `main` around `6ff24455`.

---

## 1. Documentation vision

The documentation should answer four questions with minimal friction:

```text
Can I install it?
Can I run a real or tutorial analysis?
Can I trust and interpret the results?
Can I debug, tune, and resume it safely?
```

The final system should feel like:

```text
README.md
    short product portal

documentation/public/
    user guide, tutorials, how-to pages, references, troubleshooting

documentation/development/
    maintainer contracts, architecture, testing, benchmarking, tooling, migration

documentation/scratchpad/
    historical/internal notes, explicitly non-authoritative
```

The user documentation should be organized as a learning journey, not only as a reference manual.

Use this model:

```text
Learn
    What is g? What is REGENIE Step 2? What files do I need?

Run
    Install, run tutorial data, port a REGENIE command, run real data.

Understand
    Statistical model, output fields, manifests, multi-phenotype semantics.

Reference
    CLI, TOML, Python API, exact input/output contracts, troubleshooting.
```

---

## 2. Non-negotiable documentation principles

### 2.1 One authoritative owner per topic

Do not duplicate detailed contracts across README, Quickstart, CLI, Configuration, and troubleshooting.

Canonical ownership:

| Topic | Canonical page |
| --- | --- |
| Product summary | `README.md` and `documentation/public/index.md` |
| Install | `documentation/public/installation.md` |
| First runnable tutorial | `documentation/public/tutorial-first-run.md` |
| Existing REGENIE command migration | `documentation/public/port-regenie-command.md` |
| REGENIE Step 1 to `g` Step 2 | `documentation/public/regenie-step1-to-g-step2.md` |
| Command gallery | `documentation/public/quickstart.md` |
| CLI grammar and behavior | `documentation/public/cli.md` |
| TOML and config merging | `documentation/public/configuration.md` |
| Input contracts | `documentation/public/input-files.md` |
| Output contracts | `documentation/public/output-files.md` |
| Resume and manifests | `documentation/public/resume-and-manifest.md` |
| Statistical meaning | `documentation/public/algorithm.md` |
| GPU/cluster execution | `documentation/public/gpu-and-clusters.md` |
| Performance tuning | `documentation/public/performance-guide.md` |
| Python usage | `documentation/public/api-python.md` |
| Errors | `documentation/public/troubleshooting.md` |
| Terms | `documentation/public/glossary.md` |
| Common questions | `documentation/public/faq.md` |
| Internal architecture | `documentation/development/architecture.md` |
| Multicrate/Rust migration | `documentation/development/rust-migration.md` |
| Configuration frontend implementation | `documentation/development/configuration-frontend.md` |
| Native I/O implementation | `documentation/development/native-io.md` |
| JAX/statistical kernels | `documentation/development/compute-kernels.md` |
| Testing/parity policy | `documentation/development/testing-and-parity.md` |
| Benchmarks | `documentation/development/benchmarking.md` |
| Tooling | `documentation/development/tooling.md` |
| Justfile | `documentation/development/justfile.md` |

### 2.2 Public docs explain behavior; development docs explain implementation

Public docs should answer:

```text
What do I run?
What files do I need?
What output do I get?
Which options change statistics?
How do I compare to REGENIE?
How do I recover from failure?
```

Development docs should answer:

```text
Where is this implemented?
What invariants must hold?
How do I add an option?
How do I add a backend?
How do I test correctness and parity?
How do I benchmark without fooling myself?
```

### 2.3 Every page needs an audience box

At the top of every major page, add a short box:

```md
| Audience | Use this page when | Related pages |
| --- | --- | --- |
| Bioinformaticians running REGENIE Step 2 workflows | You want to port an existing `regenie --step 2` command to `g regenie` | Compatibility, Input Files, Output Files |
```

For long development pages:

```md
| Status | Applies to | Owner |
| --- | --- | --- |
| Current migration contract | `main` and active Rust migration branches | Architecture / runtime maintainers |
```

### 2.4 Every public page needs “Next steps”

At the bottom of every public page:

```md
## Next steps

- If your command ran successfully, read [Output Files](output-files.md).
- If you need to understand the statistics, read [Algorithm](algorithm.md).
- If the command failed, read [Troubleshooting](troubleshooting.md).
```

### 2.5 Examples must be tagged by purpose

Every command example should say:

```md
**Use this when:** binary score-only analysis on CPU.

**Expected output:** `<out>.g/trait_0001_<phenotype>.regenie2_binary.run/`.

**Next check:** inspect `run_manifest.json` and Parquet part files.
```

### 2.6 Separate statistical options from runtime options

Every option table should classify whether the option:

```text
changes statistics
changes output representation
changes performance only
controls diagnostics only
is experimental
```

This is essential for scientific trust.

### 2.7 Do not hand-maintain defaults in multiple places

Defaults should come from:

```text
src/interface/config.default.toml
Rust option metadata
generated option reference
effective_config.toml from a real run
```

Docs may show example values, but must not claim defaults unless they are generated or tested.

### 2.8 Scratchpad is internal and non-authoritative

Scratchpad pages may be published by direct path, but should not be in primary public navigation. Every scratchpad page should have a banner:

```md
> Internal scratchpad note. Not user-facing documentation. May be stale.
```

---

## 3. Target public documentation structure

Recommended final `documentation/public/`:

```text
documentation/public/
  index.md
  getting-started.md
  installation.md

  tutorial-first-run.md
  regenie-step1-to-g-step2.md
  port-regenie-command.md
  quickstart.md

  quantitative.md                # optional, if command examples grow too large
  binary.md                      # optional
  approximate-firth.md           # optional
  multi-phenotype.md             # optional

  input-files.md
  output-files.md
  output-analysis.md
  resume-and-manifest.md

  algorithm.md
  algorithm-details.md           # optional appendix if algorithm.md is still long
  compatibility.md

  cli.md
  configuration.md
  generated-options.md           # generated or generated section, optional filename
  api-python.md

  gpu-and-clusters.md
  performance-guide.md
  cache-warming.md               # add when `g cache warm regenie` exists

  troubleshooting.md
  faq.md
  glossary.md
```

Do not add all optional pages immediately. Start with missing high-value pages.

Required new pages:

```text
tutorial-first-run.md
regenie-step1-to-g-step2.md
port-regenie-command.md
output-analysis.md
faq.md
glossary.md
```

Optional split pages later:

```text
quantitative.md
binary.md
approximate-firth.md
multi-phenotype.md
algorithm-details.md
cache-warming.md
```

---

## 4. Target development documentation structure

Recommended final `documentation/development/`:

```text
documentation/development/
  index.md
  documentation.md
  architecture.md
  rust-migration.md
  roadmap.md

  style-guide.md
  no-nix-development.md
  server-gauss-slurm.md
  tooling.md
  justfile.md
  dev-tooling-architecture.md

  configuration-frontend.md
  execution-pipeline.md          # add
  native-io.md
  output-writer.md               # optional if native-io grows too large
  compute-kernels.md
  telemetry.md
  runtime-and-shutdown.md        # optional if telemetry grows too large

  testing-and-parity.md
  regenie-parity-suite.md
  benchmarking.md
  artifact-format.md             # add if tooling reports continue growing
  automation.md                  # add or fold into tooling/justfile
  symphony.md
```

Development docs should stay contract-oriented. Move historical performance discovery, old reviews, and exploratory research into scratchpad unless they are maintained as current policy.

---

## 5. Navigation plan

Update `zensical.toml` to make the public guide flow from beginner to reference:

```toml
{ "User Guide" = [
  "public/index.md",
  "public/getting-started.md",
  "public/installation.md",

  "public/tutorial-first-run.md",
  "public/regenie-step1-to-g-step2.md",
  "public/port-regenie-command.md",
  "public/quickstart.md",

  "public/input-files.md",
  "public/output-files.md",
  "public/output-analysis.md",
  "public/resume-and-manifest.md",

  "public/algorithm.md",
  "public/compatibility.md",

  "public/cli.md",
  "public/configuration.md",
  "public/api-python.md",

  "public/gpu-and-clusters.md",
  "public/performance-guide.md",
  "public/troubleshooting.md",
  "public/faq.md",
  "public/glossary.md",
] }
```

Keep development nav explicit and ordered:

```toml
{ "Development" = [
  "development/index.md",
  "development/documentation.md",
  "development/architecture.md",
  "development/rust-migration.md",
  "development/roadmap.md",
  "development/style-guide.md",
  "development/no-nix-development.md",
  "development/server-gauss-slurm.md",
  "development/tooling.md",
  "development/justfile.md",
  "development/dev-tooling-architecture.md",
  "development/configuration-frontend.md",
  "development/execution-pipeline.md",
  "development/native-io.md",
  "development/compute-kernels.md",
  "development/telemetry.md",
  "development/testing-and-parity.md",
  "development/regenie-parity-suite.md",
  "development/benchmarking.md",
  "development/artifact-format.md",
  "development/symphony.md",
] }
```

Do not add scratchpad pages to primary nav unless they are promoted and rewritten.

---

# 6. Phase-by-phase implementation plan

## Phase 0 — Documentation inventory and drift audit

### Objective

Establish the current docs baseline and identify stale, duplicated, missing, or contradictory content.

### Actions

1. Generate page inventory:

```bash
find documentation -name '*.md' | sort
```

2. Generate nav inventory from `zensical.toml`.

3. Classify every page:

```text
public-current
public-needs-rewrite
development-current
development-needs-rewrite
scratchpad-keep
scratchpad-promote
scratchpad-archive
delete
```

4. Build a topic ownership map.

5. Search for stale terms:

```text
g config init
g config validate
g config explain
--g-
src/g/config.default.toml
click
OptionSpec
old BGEN/output paths
old default values
old branch names
```

6. Search for duplicate command examples.

7. Search for hard-coded defaults.

8. Search for local/server-private paths in public docs.

9. Search for development-only `just` commands in public user pages.

10. Run docs build and record current broken links or warnings:

```bash
just docs-build
```

### Deliverables

```text
documentation/scratchpad/docs-revamp-audit.md
```

with tables:

```text
page
current status
target status
canonical topic
issues
planned action
```

### Acceptance criteria

- Every Markdown page is classified.
- Every public topic has one canonical owner.
- Known stale strings are listed.
- Existing docs build result is recorded.

---

## Phase 1 — Information architecture and navigation

### Objective

Make the docs structure obvious before rewriting content.

### Actions

1. Update `documentation/public/index.md` into a “Choose your path” landing page.

2. Keep `documentation/index.md` short and audience-focused.

3. Update `zensical.toml` with final or transitional nav order.

4. Add placeholder pages for missing high-value public docs:

```text
tutorial-first-run.md
regenie-step1-to-g-step2.md
port-regenie-command.md
output-analysis.md
faq.md
glossary.md
```

Placeholders should include:

```md
# Page Title

> This page is being rewritten as part of the documentation revamp.

## What this page will cover

...
```

5. Add or update page metadata boxes on all public pages.

6. Add “Next steps” sections to public pages.

### Acceptance criteria

- Public nav follows beginner → tutorial → contracts → references.
- New placeholders build successfully.
- No scratchpad page is in primary nav.
- Every public page has a clear audience/use-case statement.

---

## Phase 2 — README as portal only

### Objective

Prevent README from becoming a duplicate manual.

### Actions

Rewrite `README.md` to include only:

```text
1. One-paragraph product description.
2. Current scope table.
3. Installation in 3–5 commands.
4. One minimal quantitative Step 2 command.
5. Links to user docs.
6. Links to development docs.
7. Pre-release warning.
```

Remove detailed duplicated content:

```text
full command gallery
long configuration examples
input-file contracts
output schema details
resume details
Python API details
performance tuning tables
developer workflow details
architecture tree
```

Keep README short enough to scan from GitHub.

### Acceptance criteria

- README is a portal.
- README links to canonical pages for details.
- README examples are smoke-tested or parser-tested.
- README does not include mutable defaults except through links.

---

## Phase 3 — First-run tutorial

### Objective

Make a student or evaluator able to complete a real tiny run.

### New page

```text
documentation/public/tutorial-first-run.md
```

### Required structure

```md
# Tutorial: Your First `g regenie` Run

## Audience

## What you will do

## Prerequisites

## Get tutorial data

## Inspect the files

## Run quantitative Step 2

## Inspect the output directory

## Read the first result rows

## Run binary score test

## Run approximate Firth

## Interrupt and resume safely

## Clean up

## What to read next
```

### Requirements

- Use repository fixture data or a tooling-generated tutorial dataset.
- If fixture generation requires development dependencies, clearly label it as “developer/evaluator tutorial”.
- If no user-friendly dataset exists, create one or add a documented data-fetch command.
- Show exact expected output tree.
- Show how to read Parquet with Python.
- Optionally show R/Arrow.

### Example output inspection

```python
import polars as pl

results = pl.scan_parquet(
    "data/tutorial/results/example.g/trait_0001_height.regenie2_linear.run/parts/*.parquet"
)
print(results.select(["CHROM", "GENPOS", "ID", "BETA", "SE", "CHISQ", "LOG10P"]).head().collect())
```

### Acceptance criteria

- A fresh checkout with documented setup can run the tutorial.
- Commands do not rely on hidden local paths.
- The tutorial states expected output files.
- The tutorial links to Output Files, Algorithm, and Troubleshooting.
- Commands are covered by docs tests or a manual smoke recipe.

---

## Phase 4 — REGENIE Step 1 to `g` Step 2 guide

### Objective

Stop scattering “`g` does not run Step 1” across many pages and give users a full transition path.

### New page

```text
documentation/public/regenie-step1-to-g-step2.md
```

### Required structure

```md
# From REGENIE Step 1 to `g` Step 2

## What `g` needs from upstream REGENIE

## Minimal upstream REGENIE Step 1 shape

## Prediction-list files

## LOCO files

## How `g` resolves prediction-list paths

## Matching phenotypes between Step 1 and Step 2

## Sample identity and order

## Quantitative example

## Binary example

## Common mistakes

## Next steps
```

### Key concepts to explain

```text
- g does not fit Step 1 models.
- g consumes Step 1 prediction lists through --pred.
- Prediction lists map phenotype names to LOCO prediction files.
- Relative LOCO paths are resolved from the prediction-list directory.
- Step 2 statistics depend on Step 1 predictions, trait mode, covariates, chromosome, and aligned sample set.
```

### Common mistakes to include

```text
- Passing a LOCO file instead of a prediction list.
- Prediction phenotype name does not match --phenoCol.
- Missing chromosome prediction.
- Binary/quantitative Step 1 mismatch.
- Different sample-key mode between files.
- Changed LOCO file causing resume rejection.
```

### Acceptance criteria

- Quickstart, Installation, Input Files, and Troubleshooting link here instead of repeating long Step 1 explanations.
- The page contains at least one concrete upstream REGENIE Step 1 command shape.
- The page contains one `g regenie` Step 2 command that consumes the Step 1 output.

---

## Phase 5 — Port an existing REGENIE command

### Objective

Serve professional bioinformaticians who already have REGENIE Step 2 scripts.

### New page

```text
documentation/public/port-regenie-command.md
```

### Required structure

```md
# Port a REGENIE Step 2 Command to `g`

## Compatibility goal

## Fast migration checklist

## Supported drop-in BGEN Step 2 shape

## Example: quantitative

## Example: binary score

## Example: approximate Firth

## Unsupported flags and what to do

## Output differences

## Fair comparison checklist

## Troubleshooting migration failures
```

### Include a mapping table

| Upstream REGENIE pattern | `g` behavior |
| --- | --- |
| `regenie --step 2 --bgen ...` | Use `g regenie --step 2 --bgen ...`. |
| `--pred` | Required; points to Step 1 prediction list. |
| `--qt` / `--bt` | Supported. |
| `--firth --approx` | Implemented but experimental/parity-sensitive. |
| `--bed`, `--pgen` | Not supported. |
| `--keep`, `--remove`, `--extract`, `--exclude` | Not supported. |
| `--spa` | Not supported. |
| `--catCovarList` | Not supported. |

### Acceptance criteria

- Existing compatibility page links here.
- Unsupported flags are consistent with live CLI behavior.
- Examples are parser-tested.
- The page explicitly warns against comparing score-only output to approximate-Firth REGENIE output.

---

## Phase 6 — Quickstart cleanup

### Objective

Make Quickstart a command gallery, not the only tutorial.

### Actions

1. Keep concise examples:

```text
quantitative CPU
binary score CPU
binary approximate Firth CPU
multi-phenotype per-phenotype
multi-phenotype complete-case
GPU
REGENIE text output
resume
config file
```

2. Remove long explanatory sections that belong in new pages.

3. Add “Use this when” and “Expected output” under each example.

4. Add a “Before running” checklist:

```text
- BGEN exists
- sample file or embedded samples available
- phenotype column exists
- binary phenotypes are 1/2-coded
- Step 1 prediction list exists
- output directory is new or --resume is intentional
```

### Acceptance criteria

- Quickstart is easy to skim.
- Every example links to a deeper page.
- Quickstart does not duplicate detailed input/output contracts.

---

## Phase 7 — Algorithm rewrite

### Objective

Make the statistical documentation readable while retaining traceability.

### Actions

1. Replace current dense pseudo-symbol style with:

```text
Words for data:
  Trait, Covariates, Genotype, LOCO, Samples

Symbols for scalar statistics:
  β, SE, χ², p, LOG10P
```

2. Keep output fields uppercase only when referring to output columns.

3. Structure:

```md
# Algorithm

## What g tests

## Notation

## Execution flow

## Sample and input alignment

## Quantitative Step 2

## Binary score test

## Binary approximate Firth fallback

## Genotype handling

## Multi-phenotype behavior

## Parameters that can change results

## Reading output rows

## Operational expectations

## Appendix: residualization

## References
```

4. Add links/references for every major idea:

```text
REGENIE Step 2
LOCO predictions
linear residualization
logistic score test
Firth bias reduction
Wilks likelihood-ratio statistic
BGEN probability layout
implementation files
```

5. Make references compact but useful:

```md
[^binary]: REGENIE documentation, “Binary traits”.
    Implementation: `src/g/compute/regenie2_binary/state.py`,
    `score.py`, and `api.py`.
```

6. Consider splitting heavy derivations to `algorithm-details.md` if the main page remains too long.

### Acceptance criteria

- A non-statistician student can understand the main text.
- A statistician can trace formulas and sources.
- Every result field in Output Files has a corresponding explanation.
- No invented obscure pseudo-symbols like `ChiAssoc`.
- References include external sources and implementation pointers.

---

## Phase 8 — Input/output/output-analysis improvements

### Objective

Make data contracts and result usage practical.

### 8.1 `input-files.md`

Add:

```text
- Small examples of phenotype, covariate, sample, and prediction-list files.
- Exact binary phenotype examples.
- Explicit delimiter expectations.
- Example of structurally short row vs explicit missing field.
- “How to inspect your input files” commands.
- Diagram: BGEN/sample/pheno/covar/pred alignment.
```

Example table snippets:

```text
FID IID height disease age sex
0   s1  170.2  1       44  F
0   s2  NA     2       58  M
```

### 8.2 `output-files.md`

Add:

```text
- “What command prints on success.”
- “How to find my results.”
- “How to read Parquet in Python.”
- “How to read Parquet in R.”
- “How to interpret correction method/status.”
- “Why final.parquet may be absent.”
```

### 8.3 New `output-analysis.md`

Add practical downstream examples:

```text
- list output files
- load Parquet with Polars
- load with R arrow
- filter genome-wide significant hits
- count correction statuses
- export selected columns to TSV
- join with variant annotation
```

Example:

```python
import polars as pl

results = pl.scan_parquet("results/run.g/trait_0001_trait.regenie2_binary.run/parts/*.parquet")
summary = (
    results
    .group_by(["CORRECTION_METHOD", "CORRECTION_STATUS"])
    .len()
    .collect()
)
print(summary)
```

R example:

```r
library(arrow)
library(dplyr)

ds <- open_dataset("results/run.g/trait_0001_trait.regenie2_binary.run/parts")
hits <- ds |> filter(LOG10P > 7.3) |> collect()
print(hits)
```

### Acceptance criteria

- A user can inspect both inputs and outputs without reading source code.
- Output-analysis page gives Python and R examples.
- Output schema table remains canonical and versioned.
- Output examples do not contradict schema types.

---

## Phase 9 — Resume and manifest tutorialization

### Objective

Keep the strong resume reference but add practical recovery recipes.

### Actions

In `resume-and-manifest.md`, add:

```text
- quick “When should I use resume?” box
- interruption exercise using tutorial data
- fast vs strict decision tree
- examples of common mismatch messages
- non-mutating incompatible resume guarantee
- “how to safely restart a SLURM job” section
```

Decision tree:

```text
Did the previous run exit cleanly after first SIGINT?
    yes -> --resume --resume_mode fast is usually OK
    no / storage unreliable / files moved -> --resume --resume_mode strict
Did you change input/config/statistical options?
    yes -> choose a new --out
```

### Acceptance criteria

- Users know when to choose fast or strict.
- Resume examples are linked from Troubleshooting and Quickstart.
- Manifest fields remain described but not overduplicated.

---

## Phase 10 — GPU, cluster, performance, and cache docs

### Objective

Make performance docs practical without promising universal speedups.

### 10.1 Split responsibility

`gpu-and-clusters.md` should own:

```text
- environment verification
- CPU/GPU SLURM scripts
- login-node warnings
- cache directory placement
- scheduler resource examples
- JAX device visibility
```

`performance-guide.md` should own:

```text
- bottleneck diagnosis
- runtime tuning
- cold/warm/hot benchmarks
- JAX compilation cache
- output writer tuning
- GPU genotype format
- fair comparison rules
```

### 10.2 Add JAX cache section

Add to Performance Guide:

```md
## JAX compilation cache

First runs may include compilation. Repeated runs can reuse persistent cache
entries when JAX, device, flags, shapes, and static arguments match.

Recommended:
- choose a cache directory on fast trusted storage;
- enable `--jax_persistent_cache`;
- set `--jax_cache_dir`;
- keep cache permissions restricted.
```

When product cache warming exists, add:

```text
documentation/public/cache-warming.md
```

### 10.3 Add “when GPU may not help”

Explain:

```text
- small runs
- single phenotype
- BGEN decode bottleneck
- host-device transfer bottleneck
- output finalization bottleneck
- first-run compilation
```

### Acceptance criteria

- GPU page is not a tuning encyclopedia.
- Performance guide clearly distinguishes startup, compilation, steady state, and output finalization.
- Warnings about statistical equivalence are present.
- No benchmark numbers are claimed as portable guarantees.

---

## Phase 11 — Troubleshooting expansion

### Objective

Make troubleshooting symptom-first and useful for students.

### Structure

Use a consistent template:

```md
## Symptom or error text

### What it means

### First checks

### Fix

### Related pages
```

### Add sections

```text
g regenie rejects an option
unsupported REGENIE flag
missing command / wrong entrypoint
missing Step 1 prediction list
prediction phenotype not found
missing LOCO chromosome
sample alignment fails
duplicate IID or FID/IID
binary phenotype coding is wrong
selected column missing
structurally short row
BGEN sample mismatch
BGEN trusted fast path rejected
GPU is not visible
GPU is visible but run is slow
JAX compilation cache not reused
out of memory
output directory already exists
resume rejected
final.parquet missing
approximate Firth TEST_FAIL rows
configuration parse error
TOML unknown key
Python API runtime-policy conflict
```

### Acceptance criteria

- Every common failure links to exact fix commands.
- Troubleshooting links to Input Files, Compatibility, Resume, GPU, and Performance Guide.
- Error examples are derived from actual current behavior where possible.

---

## Phase 12 — FAQ and glossary

### Objective

Make the docs approachable for students and reduce repeated explanations.

### `faq.md`

Add short questions:

```text
Does g run REGENIE Step 1?
Can I use BED/PGEN?
Can I use categorical covariates?
What does --pred point to?
What is LOCO?
Why is BETA sometimes from a score test?
What is LOG10P?
What is approximate Firth?
Why did approximate Firth fail for some variants?
What is the difference between per-phenotype and complete-case?
Why did GPU not help?
Why is final.parquet missing?
Can I resume after changing pThresh?
Can I compare output directly to REGENIE?
Can I share a JAX cache?
```

### `glossary.md`

Include:

```text
A1FREQ
ALLELE0 / ALLELE1
approximate Firth
BETA
BGEN
binary score test
CHISQ / χ²
complete-case
covariate
dosage
effective_config.toml
FID / IID
genotype
INFO
JAX
LOCO
LOG10P
manifest
Parquet parts
per-phenotype
phenotype
prediction list
REGENIE Step 1
REGENIE Step 2
resume
sample key
score test
SE
```

Each glossary entry should be 2–4 sentences and link to the canonical page.

### Acceptance criteria

- Glossary explains enough for a student to read Quickstart and Algorithm.
- FAQ entries link to detailed pages.
- Terms are consistent across all public docs.

---

## Phase 13 — CLI/config generated references

### Objective

Prevent CLI/config docs from drifting.

### Actions

1. Add tooling to export option metadata from Rust.

Desired output model:

```json
{
  "flag": "--bgen",
  "toml_path": "input.bgen",
  "python_option": "bgen",
  "type": "path",
  "required": true,
  "default_source": null,
  "allowed_values": null,
  "trait_modes": ["quantitative", "binary"],
  "statistical_effect": "changes_results",
  "category": "input"
}
```

2. Generate:

```text
documentation/public/generated-options.md
```

or generated tables included in:

```text
cli.md
configuration.md
```

3. Add generated docs check:

```bash
just docs-check-generated
```

4. CI fails if generated docs are stale.

### Categories

```text
input
trait
binary
compute
output
diagnostics
metadata
unsupported-recognized
```

### Acceptance criteria

- Docs do not hand-maintain comprehensive option lists.
- CLI, TOML, Python API, defaults, and docs share one metadata source.
- Option docs state whether changing option changes statistics.

---

## Phase 14 — Documentation tests and automation

### Objective

Make documentation reliable enough to trust during a fast-moving pre-release.

### Add docs checks

```text
tooling.cli.docs_check
```

or equivalent scripts.

Checks:

```text
- all nav pages exist
- all public pages in nav
- no scratchpad page in primary nav
- no broken Markdown links
- no stale forbidden strings
- no public docs references to old flags
- all documented `g regenie` flags exist in live help
- all documented command examples parse
- generated option docs are up to date
- output schema docs match code or generated schema
- every public page has audience box
- every public page has Next steps
```

### Command-example tests

Add a convention:

````md
```bash test=parse
uv run g regenie --step 2 --qt --bgen /tmp/example.bgen ...
```
````

The docs checker can replace paths with sentinel temp paths and run config parsing/dry-run if available.

If dry-run does not exist, at minimum verify:

```text
flags are recognized
mutually exclusive flags are not accidentally combined
required option names are current
```

### Acceptance criteria

- `just docs-build` remains required.
- `just docs-check` or `tooling.cli.docs_check` becomes part of CI.
- Stale option names are caught before merge.

---

## Phase 15 — Development docs cleanup

### Objective

Keep development docs useful for maintainers without overwhelming user docs.

### Actions

1. Update `architecture.md` to reflect the multicrate reality:

```text
Rust workspace crates
root PyO3 adapter
g-plan
g-interface
g-genotype
g-input
g-output
g-runtime
g-engine
Python/JAX compute backend
```

2. Add `execution-pipeline.md`:

```text
RunRequest
PreparedRunPlan
input preparation
output preparation
scheduler
backend callbacks
writer lifecycle
shutdown
```

3. Update `rust-migration.md` with current phase status.

4. Split or clarify `tooling.md` as it grows:

```text
tooling.md = operational reference
dev-tooling-architecture.md = architecture
artifact-format.md = report/artifact schema
```

5. Review `justfile.md` after Justfile cleanup.

6. Move research notes out of development nav if they are not current policy.

### Acceptance criteria

- Development docs reflect current multicrate architecture.
- No old Click/OptionSpec/Python-config ownership claims remain.
- Tooling/reporting docs match current Tooling Artifact Format.
- Scratchpad is clearly separated from maintainer contracts.

---

## Phase 16 — Visual and UX improvements

### Objective

Make the documentation easier to read and navigate.

### Diagrams to add

```text
1. Product scope:
   REGENIE Step 1 -> prediction list -> g Step 2

2. Input alignment:
   BGEN samples + sample file + phenotype + covariate + LOCO -> aligned samples

3. Execution:
   config -> inputs -> null state -> BGEN chunks -> JAX kernels -> output writer

4. Multi-phenotype modes:
   per-phenotype independent masks vs complete-case intersection

5. Output layout:
   <out>.g -> trait run directories -> parts/chunks/regenie -> manifest/config/logs

6. Resume lifecycle:
   fresh run -> committed chunks -> interrupt -> resume validation -> finalization

7. Runtime/performance:
   decode -> host/device transfer -> JAX compute -> result materialization -> writer
```

### Style rules

```text
- Prefer short paragraphs.
- Use tables for contracts.
- Use diagrams for process.
- Use callouts for warnings.
- Use code blocks for exact commands.
- Avoid giant pages with unrelated topics.
- Put math details in appendices.
- Include links after every major concept.
```

### Acceptance criteria

- Public pages are skimmable.
- New users can follow a path without opening five pages at once.
- Statistical caveats are visible, not buried.

---

# 7. Page-by-page implementation checklist

## README.md

- [ ] Reduce to portal.
- [ ] Keep one minimal command.
- [ ] Link to user guide and development guide.
- [ ] Remove detailed duplicate manual content.
- [ ] State pre-release scope.

## documentation/public/index.md

- [ ] Add choose-your-path table.
- [ ] Add support matrix or link to compatibility.
- [ ] Add “What g is / What g is not.”
- [ ] Link to first-run tutorial and REGENIE migration page.

## getting-started.md

- [ ] Rewrite as path router, or merge into index.
- [ ] Include decision tree.
- [ ] Link to tutorial-first-run.

## installation.md

- [ ] Add supported/untested platforms table.
- [ ] Add verify-install section.
- [ ] Clarify CPU vs GPU dependency groups.
- [ ] Keep development install separate.
- [ ] Avoid heavy run examples; link to Quickstart.

## tutorial-first-run.md

- [ ] Create.
- [ ] Use real small data.
- [ ] Show expected output tree.
- [ ] Show output inspection.
- [ ] Include resume exercise.

## regenie-step1-to-g-step2.md

- [ ] Create.
- [ ] Explain Step 1 prediction list.
- [ ] Include upstream REGENIE Step 1 example.
- [ ] Include `g` Step 2 example.
- [ ] Explain LOCO path resolution.

## port-regenie-command.md

- [ ] Create.
- [ ] Include migration checklist.
- [ ] Include supported/unsupported flag mapping.
- [ ] Include fair comparison checklist.

## quickstart.md

- [ ] Keep command gallery.
- [ ] Add use-case labels.
- [ ] Add expected output per command.
- [ ] Link to deeper pages.
- [ ] Remove developer-only fixture detail or put in callout.

## input-files.md

- [ ] Add sample file examples.
- [ ] Add phenotype/covariate TSV examples.
- [ ] Add prediction list example.
- [ ] Add alignment diagram.
- [ ] Add shell checks.

## output-files.md

- [ ] Add “what command prints.”
- [ ] Add “how to find output.”
- [ ] Add reason `final.parquet` may be absent.
- [ ] Keep schema table canonical.
- [ ] Link to output-analysis.

## output-analysis.md

- [ ] Create.
- [ ] Add Python Polars examples.
- [ ] Add R Arrow examples.
- [ ] Add correction-status counts.
- [ ] Add filtering hits.
- [ ] Add TSV export example.

## resume-and-manifest.md

- [ ] Add decision tree.
- [ ] Add interruption/restart recipe.
- [ ] Add mismatch examples.
- [ ] Keep manifest field list.

## algorithm.md

- [ ] Rewrite with readable notation.
- [ ] Move heavy derivations to appendix.
- [ ] Add references per section.
- [ ] Add implementation pointers.
- [ ] Clearly classify score vs Firth rows.

## compatibility.md

- [ ] Link to port-regenie-command.
- [ ] Keep support matrix.
- [ ] Keep experimental status of approximate Firth.
- [ ] Ensure unsupported flags match help.

## cli.md

- [ ] Replace hand-maintained comprehensive tables with generated sections when possible.
- [ ] Keep grammar and semantics.
- [ ] Add option effect classification.
- [ ] Add examples but not too many.

## configuration.md

- [ ] Add config file patterns:
  - minimal run config
  - reusable cluster runtime config
  - phenotype/out override from CLI
  - telemetry/profile config
- [ ] Keep merge order.
- [ ] Add unknown-key behavior.
- [ ] Add generated option/TOML mapping.

## api-python.md

- [ ] Add multi-phenotype example.
- [ ] Add process-global runtime warning.
- [ ] Add config-from-TOML example.
- [ ] Link to CLI/config docs.

## gpu-and-clusters.md

- [ ] Keep SLURM templates.
- [ ] Add cache placement advice.
- [ ] Move tuning details to Performance Guide.
- [ ] Add login-node warning near every heavy command.

## performance-guide.md

- [ ] Add cold/warm/hot definitions.
- [ ] Add JAX persistent cache section.
- [ ] Add `g cache warm regenie` section when implemented.
- [ ] Add bottleneck diagnosis table.
- [ ] Add fair comparison checklist.

## troubleshooting.md

- [ ] Expand symptom-first.
- [ ] Add actual error examples.
- [ ] Add fix commands.
- [ ] Link each symptom to canonical page.

## faq.md

- [ ] Create.
- [ ] Keep answers short.
- [ ] Link to detailed pages.

## glossary.md

- [ ] Create.
- [ ] Link terms to canonical pages.
- [ ] Keep definitions beginner-friendly.

---

# 8. Acceptance test matrix

## Docs build

```bash
just docs-build
```

Must pass.

## Link/nav check

```bash
uv run python -m tooling.cli.docs_check tool.name=links
```

Must verify:

```text
nav page exists
all public pages in nav
no scratchpad in nav
no broken local links
```

## Stale string check

Check forbidden strings in public docs:

```text
--g-
g config init
g config validate
g config explain
Click CLI
OptionSpec
src/g/config.default.toml
```

Allow exceptions only in migration/history pages.

## Command parser check

Extract documented `g regenie` commands and verify:

```text
all flags are recognized
trait mode is valid
binary-only flags not used in quantitative examples
unsupported flags not shown as supported
```

## Generated option check

```bash
just docs-check-generated
```

Must fail if Rust option metadata changes without docs regeneration.

## Output schema check

Verify `output-files.md` schema table matches code-generated output schema.

## Tutorial smoke

At least one CI or manual release-gate job should run:

```bash
tutorial-first-run quantitative
tutorial-first-run binary score
tutorial-first-run resume exercise
```

---

# 9. Suggested issue breakdown for agent orchestration

Create one epic:

```text
Epic: Documentation revamp for best-in-class user experience
```

Then issues:

1. **Docs IA audit and topic ownership map**
2. **Rework Zensical navigation and public index**
3. **Rewrite README as short portal**
4. **Add first-run tutorial with fixture data**
5. **Add REGENIE Step 1 to g Step 2 guide**
6. **Add port-existing-REGENIE-command guide**
7. **Rewrite Quickstart as command gallery**
8. **Rewrite Algorithm page**
9. **Expand Input Files with concrete examples**
10. **Expand Output Files and add Output Analysis page**
11. **Improve Resume and Manifest with recovery recipes**
12. **Expand GPU/Clusters and Performance Guide split**
13. **Expand Troubleshooting symptom-first**
14. **Add FAQ**
15. **Add Glossary**
16. **Generate CLI/config option reference**
17. **Add docs-check tooling**
18. **Update development architecture for multicrate state**
19. **Update tooling/artifact-format docs**
20. **Final docs polish, diagrams, and link checks**

Each issue should include:

```text
scope
files to edit
non-goals
acceptance criteria
tests/checks
```

---

# 10. Agent instruction block

Use this as the implementation instruction:

> Revamp `g` documentation into a best-in-class user and developer documentation system. Preserve the existing public/development/scratchpad split, but reorganize the public docs as a learning journey: install, tutorial, REGENIE migration, command gallery, input/output contracts, statistics, compatibility, CLI/config reference, GPU/performance, troubleshooting, FAQ, and glossary. README must become a short portal, not a duplicate manual. Every public page should state audience/use-case, link to next steps, and avoid stale hand-maintained defaults. Add missing tutorial and migration pages. Rewrite the algorithm page using words for data and conventional symbols for scalar statistics. Add practical output-analysis examples in Python and R. Expand troubleshooting by symptom. Add automation so documented commands, option references, nav links, and output schema do not drift from code. Keep development docs focused on implementation contracts and move historical/exploratory notes to scratchpad.

---

# 11. Definition of done

The documentation revamp is complete when:

```text
- README is a short portal.
- Public docs guide a new user from install to first successful run.
- A professional can port a REGENIE Step 2 BGEN workflow safely.
- A student can understand required files and output columns.
- Algorithm docs are readable and referenced.
- Input/output/resume contracts are precise and linked.
- Troubleshooting covers common failure symptoms with fixes.
- FAQ and glossary exist.
- CLI/config option docs are generated or verified from code.
- Documented commands parse against the current CLI.
- Output schema docs are verified against writer schema.
- Development docs match the multicrate architecture.
- Scratchpad is clearly non-authoritative.
- `just docs-build` and docs checks pass in CI.
```
