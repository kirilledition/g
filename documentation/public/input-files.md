# Input Files

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 input file contracts | Public user docs |

This page is the canonical user-facing input contract for `g regenie`.

For the statistical use of each input, see [Algorithm](algorithm.md). For the
CLI names and TOML mapping, see [CLI](cli.md) and [Configuration](configuration.md).

## Required Files

| Input | CLI | TOML | Required |
| --- | --- | --- | --- |
| BGEN genotype file | `--bgen` | `[input].bgen` | Yes. |
| Sample file | `--sample` | `[input].sample` | Yes. |
| Phenotype table | `--phenoFile` | `[input].pheno_file` | Yes. |
| Phenotype columns | Repeated `--phenoCol` | `[input].pheno_columns` | Yes. |
| Covariate table | `--covarFile` | `[input].covar_file` | Required when covariates are selected. |
| Covariate columns | Repeated `--covarCol` | `[input].covar_columns` | Required when covariates are selected. |
| Step 1 prediction list | `--pred` | `[input].pred` | Yes. |

## Genotypes

`g` currently supports Layout 2 BGEN for Step 2 scans. This includes the
uncompressed and zlib encodings defined by BGEN v1.2 and the Zstandard
compression extension added by
[BGEN v1.3](https://www.chg.ox.ac.uk/~gav/bgen_format/spec/latest.html).

Supported:

- Layout 2 genotype input with uncompressed, zlib, or Zstandard variant
  blocks.
- Oxford `.sample` files through `--sample`.

`--sample` is required. Sample identities are loaded from that Oxford `.sample`
file, not from identifiers embedded in the BGEN. Adjacent `.sample` files are
not discovered implicitly.

TOML may additionally provide an optional BGEN content selector:

```toml
[input]
bgen = "/path/to/genotypes.bgen"
bgen_content_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
```

The selector must be exactly 64 lowercase hexadecimal characters. It has no
CLI flag, and `--bgen` overrides only the locator while preserving a selector
from TOML. The locator remains required and must exist during frontend
validation; the digest is not a locator-free input mode.

In the current integration stage, the frontend validates this selector and
carries it in the immutable input plan, but the engine does not yet use it when
opening the BGEN. A public run therefore remains an unselected locator-based
open: configuring the selector does not yet activate content-addressed reuse
or pin the run to those bytes.

The BGEN header sample count must equal the aligned Oxford `.sample` metadata
count. Header offsets and reserved flags, any embedded sample-block framing,
variant counts, and probability ranges are validated before genotype delivery.
Variant identifiers, RSIDs, chromosome labels, and alleles must be valid UTF-8;
malformed or truncated input fails instead of being decoded lossily. An
individual allele, encoded probability payload, or uncompressed probability
block may contain at most 8 MiB so delivery and decompression remain
memory-bounded.

Files no larger than 256 MiB are captured into an immutable owned snapshot
before indexing. Unselected opens always capture and parse their supplied
locator independently; they never consult, publish, or replace the process
snapshot registry. Separately, the genotype library's content-selected internal
request may reuse one fully parsed canonical payload for the latest
authenticated digest and byte count, including its index and metadata, until a
different selected source passes open/index validation and replaces it or the
process exits. This internal request path is not yet selected by the planned
TOML field described above. A rejected candidate does not evict the valid
payload. Probability corruption discovered later still fails safely during
validation or decode, but that selected content may already have replaced the
earlier registry entry. Later mutation or replacement of the configured file
cannot alter a run already backed by its owned snapshot. The registry can
retain more than 256 MiB in total because parsed index and metadata allocations
are additional to the source bytes. Capturing and parsing a replacement
temporarily overlaps the retained payload with the candidate. Concurrent
selected cold opens each add their own candidate allocation, and readers can
extend old-payload overlap after publication. Larger files use bounded
positioned reads and are rechecked during delivery; changing one during a run
fails with a source error.

Unsupported options, which are absent from the CLI and rejected as unknown:

- `--bed`
- `--pgen`
- variant filtering through `--extract` and `--exclude`
- sample filtering through `--keep` and `--remove`

Eligible GPU runs select the packed8 fast path automatically, including
multi-phenotype groups and groups with different aligned sample selections.
Before using it, the engine validates that every BGEN variant is compatible
with the no-missing diploid 8-bit representation. The scan also rejects invalid
probability lengths, normalization sums, padding, reserved flags, and nonzero
probabilities for missing samples before choosing a delivery format. For
authoritative owned content, both typed outcomes—packed8 compatible and dosage
required—are cached under a revision-0 key containing the exact BGEN digest,
byte count, sample count, and variant count. Snapshot-backed validation reads
only the verified immutable payload. Positioned unattested input never reads
or writes a compatibility marker; it is checked after indexing, before and
after compatibility validation, and around genotype delivery. Cache-directory
lookup or persistence failures trigger an uncached scan and do not disable
packed8.
Otherwise-supported
biallelic diploid Layout-2 variants with missing values, phased probabilities,
or a bit depth other than 8 fall back to dosage delivery. Multiallelic,
non-diploid or otherwise unsupported input fails instead
of falling back. Compatibility validation and packed8 selection are internal
policies rather than configuration keys.

When that validated representation is stored in zlib-compressed Layout-2
blocks, the GPU path transfers the raw DEFLATE members and decodes them with
nvCOMP on the active device. Uncompressed and Zstandard blocks remain fully
supported through host packed8 decoding; this difference is an internal
delivery policy and does not change accepted BGEN input or output semantics.

## Sample Identity

Sample alignment always uses the `(FID, IID)` pair. Both values must be
non-empty, and each pair must be unique. There is no public IID-only matching
mode. The same identity rule applies across the Oxford `.sample` file,
phenotype rows, covariate rows, and prediction rows.

The Oxford `.sample` header must contain `ID_1` and `ID_2`, and its following
type row must mark both columns with type `0`. `ID_1` supplies `FID` and `ID_2`
supplies `IID` for the other inputs. The columns may appear anywhere in the
sample header; their names and types are required. Fields are separated by
ASCII whitespace, as defined by the Oxford format; Unicode whitespace is not a
delimiter.

## Phenotypes And Covariates

Phenotype and covariate tables are parsed by the native Rust path. Tables are
expected to include both `FID` and `IID`.

Column selection rules:

- Repeat `--phenoCol` for each phenotype.
- Repeat `--covarCol` for each covariate.
- Multiple phenotypes write one output run per phenotype.

Binary phenotypes use REGENIE-style coding:

| Input value | Internal value |
| --- | --- |
| `1` | Control, recoded to `0`. |
| `2` | Case, recoded to `1`. |

Missing tokens include empty string, `NA`, `NaN`, `nan`, and `-9`.
Rows must still physically contain each selected field: `FID`, `IID`, selected
phenotype columns, and selected covariate columns. A structurally short row that
ends before one of those columns fails instead of being treated as missing. Use
an explicit empty field with the delimiter present, such as a trailing tab for
the final selected column, when the intended value is missing.
Empty `FID` or `IID` fields always fail, including on rows that do not match the
sample file or would otherwise be excluded for missing phenotype or covariate
values.

`--catCovarList` is not accepted and fails as an unknown option.

## Step 1 Predictions

`--pred` must point to a prediction list produced by upstream REGENIE Step 1.
`g` does not produce Step 1 predictions.

Each prediction-list row maps one phenotype to one LOCO prediction file. Relative
LOCO paths are resolved from the prediction-list directory. Run manifests record
the prediction-list content hash and the content hash of each selected LOCO file
for the phenotype or compatible compute group, so changing a referenced LOCO file
prevents resume even when the prediction-list file itself is unchanged. Prediction
lists and LOCO files use ASCII whitespace delimiters; Unicode whitespace is not a
delimiter.

Each LOCO header starts with `FID_IID`; every following sample token must encode
non-empty FID and IID values separated by an underscore, and each resulting
pair must be unique.

Step 2 statistics depend on the prediction file, trait mode, covariates,
chromosome, and aligned sample set. Changing the prediction list can change
results even when the tested BGEN file is unchanged.

## Multi-Phenotype Sample Semantics

`[compute] multi_phenotype_sample_mode` controls how `g` aligns rows for multiple
requested phenotypes:

- `per-phenotype` (default): each phenotype uses its own complete-case sample set.
  This is the statistical equivalent to running each phenotype in a separate
  single-phenotype CLI run with identical options.
- `complete-case`: all requested phenotypes share one intersection of complete
  phenotype and covariate rows.

Every phenotype run manifest records the selected mode, the aligned sample count,
the sample-set fingerprint, the covariate-design fingerprint, and the prediction
alignment fingerprint used for that phenotype. Resume treats these fields as
result-affecting, so a per-phenotype output cannot be resumed as complete-case
output or vice versa.

This is a statistical choice, not only an execution strategy:

- Use `per-phenotype` when you want each trait to be analyzed on its own largest
  non-missing sample.
- Use `complete-case` when you want all traits analyzed on the same cohort
  (for strict per-trait comparability) or when missingness is nearly identical.

`complete-case` can change test statistics when phenotype missingness differs across
traits because it changes `sample_count`, covariate projections, and LOCO alignment
for every phenotype in the command. In that situation, this mode can bias or lose
power for phenotypes with trait-specific missingness patterns.

Performance implications:

- `per-phenotype` may still group traits that share compatible complete-case
  samples internally and therefore can reuse some startup and decode work.
- `complete-case` usually increases sample-mask and projection reuse because one
  shared sample intersection is computed once, but it can lower effective sample
  size versus per-phenotype analysis.

For full definitions and implementation details, see
[Algorithm > Multi-Phenotype Behavior](algorithm.md#multi-phenotype-behavior) and
[Configuration](configuration.md#cli-to-toml-mapping) for the exact
TOML setting name.
