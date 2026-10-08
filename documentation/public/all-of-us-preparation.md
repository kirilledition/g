# All of Us cohort preparation

The local preparation tool builds an explicit, auditable handoff from accessible
genotypes and research tables to `g` Step 2. It runs without All of Us credentials
and does not download participant data. Run it inside the approved workspace when
using controlled inputs; its tables, keep lists, logs, and manifests contain
participant information and remain workspace artifacts.

This first version accepts local PGEN/PVAR/PSAM, BED/BIM/FAM, or BGEN plus an
Oxford sample file. It selects human autosomes, exactly two alleles, and
upstream-confirmed diploid calls. Hail VDS filtering, allele normalization,
genotype QC, and densification are upstream workspace steps. A VDS is not a
directly supported preparation input. The tool applies no automatic ancestry,
relatedness, or sex exclusions.

## Explicit identities and research inputs

Create a tab-delimited identity map with these four columns:

```text
source_FID	source_IID	FID	IID
0	synthetic-001	family-a	participant-a
0	synthetic-002	family-b	participant-b
```

The source pairs must cover the complete genotype sample file exactly once;
they are not inferred from a phenotype ID or a truncated identifier. The target
pairs are the canonical identities used by phenotypes, covariates, exclusions,
Step 1 predictions, and Step 2. If a PSAM omits FID, its source FID is explicitly
`0`, following PLINK's convention. Nonzero SID fields require upstream
normalization. Identifiers retain leading zeroes and are never parsed as numbers.

Duplicate source or target pairs, blanks, whitespace or double quotes in identities, and
ambiguous LOCO tokens fail. For example, `(a_b, c)` and `(a, b_c)` cannot coexist
because both serialize as `a_b_c`. Ordinary underscores are allowed.

Phenotype and covariate TSVs must have `FID`, `IID`, and the selected numeric
columns. Every selected participant needs a physical table row; use an explicit
missing value when an observation is unavailable. Accepted missing tokens are
empty fields, `NA`, `NaN`, `nan`, and `-9`. Prepared tables normalize these tokens
to `NA`, preserving missingness without filling any value. Binary phenotypes
use **1 = control and 2 = case**. Nonfinite values, float32 overflow, malformed
rows, and duplicate table identities fail.

Keep, remove, and release-exclusion files have tab-delimited `FID` and `IID`
headers. `#FID` is also accepted for reuse of a generated PLINK keep list.
An empty keep list fails; a header-only exclusion list is valid. Unknown keep
pairs fail instead of silently shrinking the cohort. Release-exclusion lists can
include participants outside the current source: matching and absent counts are
reported. Convert any provider-supplied one-column participant list into explicit
canonical pairs through the approved workspace identity mapping first.

Supply the current release's flagged-sample list using
`tool.release_exclusion_files=[...]`, and record `tool.release_label`. The tool
does not embed release-specific IDs or assume that a historical exclusion list
is current. Do not check these inputs into Git.

## Plan and table-only preparation

From a repository installation with the tooling dependencies available:

```bash
python -m tooling.cli.data --config-name data_all_of_us \
  tool.mode=plan \
  tool.source_format=tables \
  tool.identity_map=/workspace/inputs/identity-map.tsv \
  tool.phenotype_file=/workspace/inputs/phenotypes.tsv \
  'tool.phenotype_columns=[trait]' \
  tool.covariate_file=/workspace/inputs/covariates.tsv \
  'tool.covariate_columns=[age,sex,PC1,PC2]' \
  tool.keep_file=/workspace/inputs/cohort.tsv \
  'tool.release_exclusion_files=[/workspace/inputs/release-exclusions.tsv]' \
  tool.release_label=chosen-release \
  tool.output_directory=/workspace/work/prepared-tables
```

`plan` reads and validates inputs and prints JSON. It does not create the output
directory, execute PLINK, or create Hydra log/config directories. `tables` uses
the same arguments with `tool.mode=tables` to materialize aligned analysis tables
and keep lists without PLINK. In table-only source mode, identity-map row order
defines cohort order. With a genotype source, physical genotype sample order
defines it.

Plan and tables modes also work with `source_format=pgen`, `bed`, or `bgen` to
validate accessible sample identities and render the future conversion commands.
Genotype conversion requires `execute` explicitly.

## Genotype preparation

For an already QC-selected PGEN source:

```bash
python -m tooling.cli.data --config-name data_all_of_us \
  tool.mode=execute \
  tool.source_format=pgen \
  tool.input_prefix=/workspace/inputs/chr22-qc \
  tool.confirm_diploid=true \
  tool.identity_map=/workspace/inputs/identity-map.tsv \
  tool.phenotype_file=/workspace/inputs/phenotypes.tsv \
  'tool.phenotype_columns=[trait]' \
  tool.covariate_file=/workspace/inputs/covariates.tsv \
  'tool.covariate_columns=[age,sex,PC1,PC2]' \
  tool.keep_file=/workspace/inputs/cohort.tsv \
  'tool.release_exclusion_files=[/workspace/inputs/release-exclusions.tsv]' \
  tool.variant_extract_file=/workspace/inputs/chr22-variants.txt \
  tool.minimum_allele_count=5 \
  tool.minimum_variant_call_rate=0.98 \
  tool.release_label=chosen-release \
  tool.threads=8 \
  tool.memory_megabytes=16384 \
  tool.output_directory=/workspace/work/prepared-chr22
```

The filter values above illustrate configuration; choose thresholds for the
actual research analysis. Both filters are optional. Allele counts include
nonfounders and are computed in the selected preparation cohort. Call rate
counts available dosages, rather than treating an uncertain but present dosage
as absent. These filters are not recomputed separately for every phenotype's
missingness mask. Prepare a separate, explicitly defined cohort when the study
requires filters calculated on a particular trait's complete cases.

For compressed PVAR input, set `tool.pvar_zstd=true`. BED input uses
`tool.source_format=bed` and its fileset prefix. BGEN input requires
`tool.source_format=bgen`, `tool.input_bgen`, `tool.input_sample`, and an explicit
`tool.bgen_reference=ref-first`, `ref-last`, or `ref-unknown` assertion. Do not
guess the reference convention from the file extension.

For PLINK conversion, the Oxford sample file must place `ID_1` and `ID_2` in its
first two columns. The engine's direct-reader contract permits other positions,
but the PLINK importer does not. Normalize the sample header upstream when
necessary, preserving each sample row's identity and order.

`confirm_diploid=true` records an upstream input assertion; it is not a scan that
qualifies every source genotype's ploidy. Check ploidy and upstream QC in the
workspace before making that assertion.

The audited argument vectors execute directly without a shell:

1. Select the source FID/IID pairs and variants into `selected.pgen/.pvar/.psam`.
2. Apply the explicit identity update and write canonical
   `genotypes.pgen/.pvar/.psam` plus Layout-2 BGEN 1.2 with `ref-first`, `bits=8`,
   and a two-ID Oxford sample file.

The intermediate and canonical PGEN files remain in the attempt for inspection
and genome-wide Step 1 reuse. Budget local disk for both PGEN filesets, the BGEN,
temporary PLINK conversion files, and outputs. BGEN preparation can be a
substantial CPU and storage workload.

No fill-missing, genotype-imputation, or mean-imputation option is used. Hard-call
roundtrip tests cover 0/0, 0/1, 1/1, missing calls, and allele order. Arbitrary
BGEN genotype posteriors are **not losslessly preserved**: PLINK converts them
to dosages, then reconstructs probabilities for eight-bit export. Qualify dosage
precision and INFO effects for probabilistic inputs. BED A2 and other
provisional-REF sources are not proof of biological reference alleles; normalize
and verify the genome/reference convention upstream. See the official
[PLINK input](https://www.cog-genomics.org/plink/2.0/input#bgen),
[export](https://www.cog-genomics.org/plink/2.0/data#export), and
[filter documentation](https://www.cog-genomics.org/plink/2.0/filter#geno).

## Audit and failures

`preparation.json` records the release/build labels, input paths and fingerprints,
identity-list hashes, selected/excluded counts, columns and coding, missingness
counts, complete-case counts, filters, exact PLINK arguments, and output hashes.
The installed PLINK version and each command's stdout/stderr are retained.
Before success, physical sample identities and order must match at each stage;
retained PVAR IDs must be unique and nonmissing; BGEN dimensions must agree with
the canonical sample and variant counts. Every exported BGEN record boundary is
checked without decompressing genotype blocks, including empty compressed
payloads, truncation, and unexpected trailing bytes. This also rejects malformed
tiny-cohort exports observed with the local pinned PLINK build despite its exit
status being zero.

Large genotype inputs default to path, size, modification/change times, inode,
and device identity to avoid an additional full input read during a plan. Set
`tool.hash_genotype_inputs=true` for full source content hashing. Small identity,
table, and list inputs are always hashed; successful output files are always
hashed. Inputs are revalidated before and after execution. These local
fingerprints are preparation provenance, not a cross-VM `g` resume contract.

An existing output directory is always refused. A failed attempt retains its
partial outputs and a `failed` manifest; correct the cause and choose a new
directory. `tables_prepared` records table-only completion and is not a completed
genotype export. A `complete` preparation does not replace scientific validation
of association results or a full BGEN probability-validation scan.

## External genome-wide REGENIE Step 1

`g` implements Step 2 and consumes external REGENIE LOCO predictions. Prepare
the same canonical cohort on **genome-wide** QC-selected common markers, using
the same identity map, exclusion policy, phenotypes, and covariates. The
chromosome-22 pilot is a Step 2 pilot; it is not an adequate Step 1 training
dataset. Marker QC, LD pruning, population covariates, and relatedness strategy
must be chosen for the study.

The preparation command also writes `genotypes.pgen/.pvar/.psam` with canonical
IDs. For a **genome-wide** prepared dataset and a reviewed marker extraction
list, an external quantitative-trait handoff is:

```bash
regenie --step 1 \
  --pgen /workspace/work/prepared-genome-wide/genotypes \
  --keep /workspace/work/prepared-genome-wide/regenie.keep \
  --extract /workspace/inputs/step1-markers.txt \
  --phenoFile /workspace/work/prepared-genome-wide/phenotypes.tsv \
  --phenoColList trait \
  --covarFile /workspace/work/prepared-genome-wide/covariates.tsv \
  --covarColList age,sex,PC1,PC2 \
  --bsize 1000 --threads 8 \
  --out /workspace/work/step1/quantitative
```

For binary phenotypes add **`--bt --cc12`**, because the prepared table uses
1/2 coding. `regenie.keep` is deliberately headerless. The companion
`cohort.keep.tsv` has a PLINK-style header for preparation reuse. Prepared missing
observations are `NA`, as required by REGENIE. See
[REGENIE's input and Step 1 options](https://rgcgithub.github.io/regenie/options/).

Create the Step 1 output directory before starting REGENIE. Pin its version and
record all Step 1 inputs and settings. Pass its resulting phenotype-matched
`*_pred.list` to `g --pred`, with the same selected covariates and 1/2 phenotype
coding. Keep participant-level predictions inside the workspace. Validate a
quantitative and binary Step 2 pilot against REGENIE before scaling out.
