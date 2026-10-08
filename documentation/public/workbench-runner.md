# Workbench job preparation and execution

The Workbench runner prepares a local, isolated REGENIE Step 2 attempt from a
versioned manifest. It verifies input bytes, checks structural sample alignment,
chooses a smaller variant chunk when declared memory requires it, launches the
installed `g regenie` executable, and publishes only completed, validated outputs.
It imports neither JAX nor the native extension during preflight.

This package can be tested with synthetic and public inputs on this machine.
It has **not been qualified against All of Us data**. Run the first real-data
qualification inside the authorized workspace and retain every artifact there.
The runner does not select a scientific cohort, perform genotype QC, generate
ancestry covariates, or establish that a study design is statistically valid.

## Inputs and scientific scope

Prepare autosomal, biallelic diploid BGEN Layout 2 inputs and an Oxford `.sample`
file. Probability encoding, phasing, ploidy and missingness still undergo the
native reader's full validation when the engine starts. The bounded BGEN profiler
provides an encoding inventory, not a whole-file eligibility certificate.

Phenotype and covariate tables have `FID IID` identifiers and tab-separated
numeric columns. Missing values use empty fields, `NA`, `NaN`, `nan`, or `-9`. Binary phenotypes
use **1 for controls and 2 for cases**. Every selected phenotype must have enough
aligned, nonmissing samples; binary traits must retain both classes. Sample order
in these tables need not match the BGEN order. Duplicate sample pairs and
ambiguous `FID_IID` serialization are rejected.

Supply a genuine, genome-wide external REGENIE Step 1 LOCO prediction file for
each selected phenotype. The runner generates a prediction list pointing to the
verified local copies. A LOCO header begins `FID_IID` followed by unique serialized
sample keys; autosomal chromosome rows use `1` through `22` or `chr1` through `chr22`.
Additional genome-wide rows such as `X`, `chrX` or `23` are accepted and
normalized under the native LOCO grammar; this does not qualify sex-chromosome analysis.
Selected samples must have finite predictions. Structural checks do not prove
that predictions came from the intended study or a correctly executed Step 1.

The default sample policy is `per-phenotype`: each trait retains its own available
samples after phenotype and covariate alignment. The original native strict
resume identity is unchanged. A retry creates a new attempt rather than resuming
outputs copied from another VM.

## Manifest version 1

The following illustrates the exact schema. Replace the example paths,
checksums and byte sizes with those of your prepared files. The all-zero digests
are placeholders and will fail verification unless they match actual content.
No real participant identifiers or All of Us bucket paths are included.

```json
{
  "schema_version": 1,
  "dataset": {
    "name": "synthetic-pilot",
    "release": "fixture-v1",
    "reference_build": "GRCh38"
  },
  "inputs": {
    "bgen": {
      "uri": "/work/inputs/chr22.bgen",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size_bytes": 123456
    },
    "sample": {
      "uri": "/work/inputs/chr22.sample",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size_bytes": 1234
    },
    "pheno_file": {
      "uri": "/work/inputs/phenotypes.tsv",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size_bytes": 1234
    },
    "covar_file": {
      "uri": "/work/inputs/covariates.tsv",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size_bytes": 1234
    }
  },
  "predictions": [
    {
      "phenotype": "trait",
      "uri": "/work/inputs/trait.loco",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size_bytes": 1234
    }
  ],
  "analysis": {
    "trait_type": "quantitative",
    "phenotype_columns": ["trait"],
    "covariate_columns": ["age", "sex"]
  },
  "resources": {
    "device": "gpu",
    "cpu_threads": 8,
    "writer_threads": 2,
    "gpu_memory_gib": 16,
    "memory_fraction": 0.5,
    "max_chunk_variants": 16384
  },
  "output_uri": "/work/published"
}
```

Unknown or duplicate JSON fields are errors. Columns must begin with an ASCII
letter and contain at most 80 letters, digits, dots, underscores or hyphens;
`FID` and `IID` are reserved. Predictions must cover selected phenotypes exactly
once. Omit `covar_file` when `covariate_columns` is empty. The source dataset name,
release and reference build are explicit provenance labels, not automatic release
or genome-build validation.

For a CPU job use `"device": "cpu"` and declare `"memory_gib": 32` in resources.
For a binary job use `"trait_type": "binary"` and add this analysis setting:

```json
"binary": {
  "fallback_method": "firth_approximate",
  "p_threshold": 0.05
}
```

The threshold must remain strictly between zero and one after native float32
rounding; the rounded value is written explicitly to the generated TOML. The other correction
choice is `score_only`. Optional
`resources.firth_batch_size` is a positive integer for binary jobs. When
approximate Firth is selected, the default uses a power-of-two batch no greater than 256 or
`max_chunk_variants`, reduced further when necessary to fit the declared
full-cohort preparation budget. An explicit batch that cannot fit is rejected.

## Preflight, profiling and execution

Use an explicit private work directory on sufficiently large local disk. It must
be owned by the current user and not writable by other users or groups. All
commands create a new owner-only attempt directory and print its path. They
leave existing attempts untouched. The source manifest and input files remain
unchanged.

```bash
python -m tooling.cli.workbench --config-name workbench_preflight \
  tool.manifest=/work/job.json tool.work_directory=/work/attempts

python -m tooling.cli.workbench --config-name workbench_profile \
  tool.manifest=/work/job.json tool.work_directory=/work/attempts \
  tool.profile_max_variants=128

python -m tooling.cli.workbench --config-name workbench_run \
  tool.manifest=/work/job.json tool.work_directory=/work/attempts
```

The repository also provides `just workspace-preflight`, `just workspace-profile`
and `just workspace-run` with the same overrides. In an installed Workbench
container, the `python -m` entrypoints are sufficient.

`preflight` hashes local sources, inspects the BGEN header, validates local sample,
phenotype, covariate and LOCO alignment when all inputs are local, and checks free
disk. For remote sources it checks cloud metadata, but reports content verification
as pending until localization. It never downloads GCS inputs. A remote BGEN means
sample count and output disk requirements cannot yet be determined; those checks
run again after localization.

`profile` localizes and verifies every input before bounded encoding inspection.
`run` performs localization, structural alignment, capacity planning and one
fresh native execution. Change `tool.runner_prefix` only when the executable
needs an explicit command prefix; it is a shell-free argument list, normally
`[g,regenie]`.

For a plan without cloud requests, input localization or engine execution:

```bash
python -m tooling.cli.workbench --config-name workbench_run \
  tool.manifest=/work/job.json tool.work_directory=/work/attempts \
  tool.dry_run=true
```

Dry-run still writes local manifest, preflight and status evidence and verifies
already-local source bytes. It does not download inputs, invoke `gcloud`, import
JAX, execute the engine, or publish outputs. A dry-run with remote inputs reports
those checks as pending rather than claiming success at validation.

## Memory and disk planning

The chunk estimator preserves the full cohort and reduces variant width. It
reserves one GiB for backend/compiler state, float64 design and per-trait arrays,
eight float32 chunk matrices plus two per selected trait, and additional Firth
workspace. For Firth, it models all grouped candidate lanes up to the native
per-trait capacity of 1024 (also explicitly set in generated TOML), padded to
the solver batch width, including
covariate-dependent gathered projection arrays and separate solver batch state. The native engine pads final or chromosome-tail chunks to this chosen width,
so the estimate uses the full width even for a short tail. Trait-by-chunk indices
also remain within the engine's signed 32-bit index domain. It caps the estimate
at the declared memory multiplied by
`memory_fraction`, which must be positive and at most 0.8. A configuration unable
to accommodate even one estimated variant is rejected.

This is a **conservative working-set heuristic, not an out-of-memory guarantee**.
Compiler temporaries, genotype encodings, allocator fragmentation, simultaneous
phenotype groups, driver allocations and other processes can change actual use.
The declaration is not a hardware probe. Qualify real cohort sizes and monitor
both host and GPU memory before increasing widths or trait concurrency. Do not
reduce the cohort implicitly to make a job fit.

GPU subprocesses request CUDA explicitly, enable float64, disable JAX's initial
GPU preallocation, and use the declared memory fraction for its allocator setting.
CPU/BLAS thread counts and the local trusted compilation cache are explicit. The
runner does not change the engine's statistical or GPU compiler precision policy.
Never share the compilation cache with untrusted writers.

The disk preflight reserves all localized input bytes, three estimated output
copies at 512 bytes per variant per trait, and two GiB of compilation headroom.
Actual files can be larger, especially when identifiers are long. The launch
checks free local disk again after a remote BGEN has been localized. Keep enough
space on a separate local publication filesystem as well; the final copy fails
without replacing a previous published attempt if that filesystem fills.

## GCS localization and publication

Each input URI may instead be a literal `gs://bucket/object` with the same
SHA-256 and byte size. An optional `generation` is a positive decimal **string**:

```json
{
  "uri": "gs://example-workspace/prepared/chr22.bgen",
  "generation": "1234567890123456",
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "size_bytes": 123456
}
```

Queries, wildcard patterns, traversal components and embedded generation suffixes
are rejected. GCS mode requires the external authenticated Google Cloud CLI.
Optional `tool.billing_project=PROJECT` supplies its requester billing project.
Authentication and workspace permissions remain platform responsibilities.

The runner describes the pinned generation, or resolves the current generation
once, and downloads `gs://bucket/object#GENERATION` into a `.partial` file. It
disables automatic content-encoding decompression and verifies SHA-256 and byte
size before making that input available to the engine.
The resolved generation is recorded in provenance. It never substitutes a newer
generation after a failed download. Partial inputs remain diagnostic artifacts.
[Cloud CLI copy contract](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp).

Set `output_uri` to an authorized workspace `gs://bucket/prefix` to publish there.
Every attempt uses a unique child prefix. Each upload uses a create-only
precondition and must pass server-observed size and MD5 verification; a missing
MD5 fails closed. The upload subprocess disables parallel composite uploads
through a process-local Cloud CLI environment setting, leaving the user's saved
CLI configuration unchanged. All data and provenance files are verified
before the `COMMITTED.json` marker is sent last.
[Generation preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions).

Cloud publication is not an atomic directory transaction. A failed transfer can
leave partial objects without a commit marker. A lost acknowledgement during the
last upload can leave an unconfirmed commit marker; failed attempt status records
`publication_committed: null` after publication has started. Treat an uncertain
attempt as requiring inspection, and start a new attempt for a retry. No cloud
success is returned unless every verification finishes.

Local publication copies and verifies the complete bundle into a hidden sibling
directory and renames it on the same filesystem. The destination appears with the
commit marker and all files together. Existing destinations are never reused.

## Evidence and remaining qualification

Each attempt retains the input manifest, verified local inputs, generated TOML,
relative LOCO list, alignment and capacity reports, provenance, engine stdout and
stderr, native output directories and a terminal `status.json`. Cancellation
forwards an interrupt to the isolated engine process group, including launcher
descendants, and allows up to 30 seconds for native chunk commits to be preserved
before killing any remaining group members. A second interrupt skips the
remaining grace period and forces group shutdown immediately. Failed attempts
remain intact.

A successful exit code alone is insufficient. Publication additionally requires
exactly the expected completed phenotype manifests, contiguous chunk coverage of
all input variants, referenced Parquet parts with matching actual footer row
counts and chunk commit metadata, safe paths, and SHA-256 identities. This check
does not compare association values against REGENIE or inspect every data page.
The successful bundle contains outputs, logs and provenance, excludes localized
source inputs and cache, and carries a hashed file inventory in `COMMITTED.json`.

Treat **all generated bundles, logs, paths, identifiers, predictions and summary
statistics as restricted workspace artifacts**. The commit marker explicitly does
not certify external export safety. Apply the actual platform dissemination rules
before any export. No automated blanket export decision is made.

On All of Us, pin the actual release, apply release-specific exclusions and QC,
prepare the intended ancestry/relatedness design, execute external Step 1, and
compare quantitative and binary Step 2 results with REGENIE. Then qualify full
cohort memory, missing-call decode throughput, interrupted-job behavior, input
localization, cloud permissions and cost. These tasks need authorized workspace
access and remain separate from local synthetic validation.
