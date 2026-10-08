# All of Us on Verily Workbench

| Status | Applies to | Owner |
| --- | --- | --- |
| Experimental; local validation does not qualify controlled data | Workbench tooling introduced 2026-10-08 | Development maintainers |

This workflow prepares the existing BGEN-backed REGENIE Step 2 engine for use
inside an authorized All of Us Researcher Workbench workspace. Participant
data remains in that workspace. The engine still requires external REGENIE
Step 1 predictions; it does not train them.

## Start with the synthetic smoke test

Install the repository with the minimal `workbench` dependency group in your
compute environment. Source installation builds the native extension; on the
Gauss cluster run installation and execution on a compute node.

```bash
uv sync --locked --no-default-groups --group workbench
just workspace-demo tool.output_directory=/path/to/new/synthetic-bundle
just workspace-preflight tool.manifest=/path/to/new/synthetic-bundle/quantitative.job.json tool.work_directory=/path/to/private/attempts
just workspace-run tool.manifest=/path/to/new/synthetic-bundle/quantitative.job.json tool.work_directory=/path/to/private/attempts
just workspace-run tool.manifest=/path/to/new/synthetic-bundle/binary.job.json tool.work_directory=/path/to/private/attempts
```

The bundle contains 128 invented samples and 32 diploid chromosome 22 variants,
including missing calls. Both manifests use CPU execution initially. The
synthetic zero LOCO values test file alignment and execution only. They are
not trained predictions and do not establish scientific validity or All of Us
performance. Manifests contain absolute paths: regenerate the bundle after
moving to another filesystem.

The binary fixture deliberately uses a correction threshold near one to exercise
approximate Firth candidates. This is an integration-test setting; choose the
correction mode and threshold for the actual research analysis separately.

For a GPU smoke test, create a separate copy of a generated job manifest,
change `resources.device` to `gpu`, and declare `resources.gpu_memory_gib`
using the actual available GPU memory. A new native subprocess establishes
GPU policy for each attempt. Keep the original CPU results for comparison.

## Local validation and workspace handoff

Local validation on 2026-10-08 covers real PLINK PGEN/BED/BGEN preparation
round trips, preserved missing calls and allele order, malformed export
rejection, bounded input inspection, immutable-input checks, interrupted
attempts and complete-output publication. CPU and V100 GPU synthetic Step 2
results pass the existing strict reference comparison against REGENIE v4.1.
The binary test selects 32 approximate Firth corrections with zero failures
on each device. Reference tolerances were not widened.

These are small integration checks on invented samples with zero LOCO
predictions. They do not establish research calibration, All of Us file
compatibility, full-cohort memory capacity or performance on a Workbench GPU.

Move the reviewed source revision, lockfiles and qualified software artifact
into the workspace through approved software-transfer procedures. Follow
[deployment](workbench-deployment.md) to choose the container or an isolated
environment; check the wheel's Python, architecture and minimum glibc tags
against the actual app. Regenerate the synthetic bundle there and rerun both
traits before preparing any controlled inputs. Then resolve the release's
actual files and proceed through the qualification stages below.

## Prepare real inputs inside the workspace

Use the selected data collection's current Data Dictionary to resolve file
locations. Do not copy old Terra bucket names or assume its environment
variables exist in the new Workbench. Pin the release, source object
generations, hashes and auxiliary QC-file versions.

All of Us documents chromosome-sharded BGEN and PGEN for the ACAF, exome and
ClinVar smaller WGS callsets. The complete joint WGS callset uses Hail VDS.
The existing BGEN route is the first target; exact BGEN encodings must be
inspected in the workspace. See the official
[genomic data organization](https://support.researchallofus.org/hc/en-us/articles/49999549117588-How-the-All-of-Us-Genomic-and-multi-omics-data-are-organized).

Use [Cohort preparation](all-of-us-preparation.md) to select local PGEN, BED or
BGEN inputs and export an explicitly chosen cohort. Keep phenotype coding,
sample ID mapping, variant selection, missingness, allele orientation and
research-specific QC decisions in the preparation audit. Relatedness handling
and ancestry covariates are methodological decisions; the tool does not
silently select an unrelated or ancestry-specific cohort.

For CDRv9, the August 2026 notice identifies seven affected srWGS/array
participant IDs to exclude using the supplied list. Pass that list through
the preparation workflow; do not place IDs in source code. See the
[release-specific QC notice](https://support.researchallofus.org/hc/en-us/articles/52422066751252-Incremental-Data-Change-for-Curated-Data-Repository-Version-9-Genomic-Data).

Run external REGENIE Step 1 genome-wide on the intended analysis cohort and
training variants, preserving the selected phenotype/covariate conventions.
A chromosome 22 Step 2 pilot still needs correctly trained chromosome-specific
LOCO predictions. Record those files as pinned inputs in the
[Workbench job manifest](workbench-runner.md).

## Execute with explicit resource and storage budgets

Use [Input profiling](workbench-input-profile.md) to inspect a bounded BGEN
prefix. This is an inspection report, not a replacement for the native
reader's full validation. Valid missing genotypes remain usable through host
dosage decoding and GPU association. They currently disable the input-wide
packed GPU decoder; do not fill missing source calls to obtain a faster path.

The runner estimates a conservative chunk working set from the declared host
or GPU memory and selects a variant width no larger than the requested cap.
It preserves the sample cohort. The estimate does not model every allocator,
compiler, output or solver allocation and is not an out-of-memory guarantee.
Measure the actual workload before raising limits or running many traits.

The [deployment package](workbench-deployment.md) supplies a portable container
and a CPU workflow template. GPU execution and the selected workflow backend's
GPU configuration require qualification in the actual workspace. The engine
uses local filesystem inputs and atomic output commits; the runner localizes
cloud inputs and publishes completed attempts. Cloud retries use fresh attempt
directories. Native cross-VM resume is not enabled by bypassing file identity.

## Qualification before research use

Progress from a small autosomal cohort to full-cohort chromosome 22, then the
intended autosomal variant set. At each stage:

- Compare quantitative and binary results with REGENIE using identical samples,
  covariates, allele orientation, missingness, predictions and correction modes.
- Check observed sample counts, frequencies, null/status values, low allele
  counts and imbalanced binary traits. Preserve default GPU compilation's known
  numerical variation rather than assuming bitwise reproducibility.
- Measure peak host/GPU memory, local disk, complete runtime and cloud cost,
  including preparation, transfer, compilation and output persistence.
- Interrupt a task, verify that incomplete attempts are not committed, and
  rerun into a fresh attempt. Check durable output hashes and provenance.

Initial scope is autosomal biallelic diploid single-variant Step 2 association.
Sex-chromosome ploidy, full VDS ingestion, native PGEN ingestion, structural
variants and gene-level burden/SKAT analyses require separate implementation
or qualification.

## Results remain controlled

Local manifests, effective configs, prediction files, logs and result bundles
may contain controlled information. Publication to a workspace bucket is
internal persistence, not permission to export. The tools do not decide that
a GWAS result is safe to disseminate and do not automatically upload anything
to external support services or source repositories.

Apply the current All of Us
[Data User Code of Conduct](https://support.researchallofus.org/hc/en-us/articles/22346176432532-Data-User-Code-of-Conduct)
and [Data and Statistics Dissemination Policy](https://support.researchallofus.org/hc/en-us/articles/22346276580372-Data-and-Statistics-Dissemination-Policy)
before exporting results. Clarify rare-variant and derivable-small-count cases
with the program rather than treating an allele-count threshold as automatic
export approval.
