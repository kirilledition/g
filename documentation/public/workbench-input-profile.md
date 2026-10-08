# Inspecting Workbench BGEN inputs

Before starting an association pilot, inspect a small prefix of a localized
BGEN file and validate its Oxford sample file. The Workbench input profiler
uses the Python standard library and imports neither JAX nor the native
extension. It reads declarations and encoding flags, without logging participant
identifiers, variant identifiers, positions, alleles, or genotype values.

This inspection is useful on a CPU Workbench environment before provisioning a
GPU. It does **not** replace the native engine's input validation, cohort QC,
phenotype/covariate alignment, or scientific qualification against REGENIE.

## Running an inspection

The reusable interface is:

```python
import dataclasses
import json
from pathlib import Path

from tooling.workbench import profile

result = profile.profile_bgen(
    path=Path("/work/inputs/chr22.bgen"),
    max_variants=100,
    sample_path=Path("/work/inputs/chr22.sample"),
)
print(json.dumps(dataclasses.asdict(result), indent=2))
```

Use files inside the approved Workbench environment. A `gs://` URI must first be
localized; this profiler accepts local filesystem paths. `read_header(path)`
provides the same header parser for cheap sample/variant counts without scanning
variant records or reading embedded sample identifiers.

The preparation workflow additionally calls `validate_record_boundaries(path)`
after export. This traverses every declared variant's metadata and stored block
boundaries by seeking over genotype payloads, without decompression. It rejects
truncated exports, empty compressed members, inconsistent minimum expanded
lengths, and trailing bytes, even if the upstream export command reported
success. This completion check is distinct from the bounded encoding inventory:
it neither inspects genotype flags nor validates compressed data or probabilities.

The result includes the declared sample and variant counts, compression and
layout, inspected record count, observed phasing and probability precisions,
ploidy bounds, missing-call count, and the number of inspected records outside
the engine's biallelic diploid contract. Missing calls are counted across
sample-record observations, rather than interpreted as distinct participants.

## Interpreting the result

`inspection_scope="prefix_only"` means only the first `max_variants` records
were inspected. These records are a sequential prefix, **not** a random or
representative sample. Later records may have different encodings, missingness,
unsupported ploidy, or corruption.

| Observed encoding | `decode_route` | Meaning |
| --- | --- | --- |
| Zlib, biallelic diploid, unphased, 8-bit, no missing calls in inspected records | `packed8_candidate_requires_native_preflight` | Candidate for the fast compressed decoder, pending native validation of the complete source. |
| Biallelic diploid with missing calls, phasing, another valid precision, or uncompressed/Zstandard storage | `generic_dosage_then_gpu_association` | Supported encoding family; generic native dosage decoding can feed GPU association. It does not establish availability or throughput on a particular GPU. |
| Non-biallelic or non-diploid inspected records | `unsupported_records_observed` | Prepare supported autosomal biallelic diploid inputs upstream without silently changing the biological interpretation. |
| No declared records | `no_records_inspected` | There is no encoding evidence and this is not a usable association input. |

Even when `inspection_scope="all_declared_records"`, this tool validates
structural sizes and flags only. It does not decode or validate stored
probability values, certify hard calls, compare embedded identifiers with an
Oxford file, establish reference/effect allele semantics, or inspect the
analysis's phenotype and prediction tables.

Consequently, every successful report retains:

```json
{
  "probability_values_validated": false,
  "native_preflight_required": true,
  "whole_file_fastpath_qualified": false
}
```

`sampled_packed8_compatible=true` describes observed structural encodings only.
The current engine chooses its packed route using compatibility of the complete
BGEN source. A later missing call can therefore change the route for the entire
file; a passing prefix never promises GPU decompression of that file.

## Bounds and malformed inputs

The profiler supports BGEN Layout 2 with uncompressed or zlib blocks. Zstandard
inspection uses Python 3.14's optional standard-library `compression.zstd`
support; no additional package is silently installed. A Python build without
that support reports an actionable inspection error. The native engine's
independent Zstandard support is unaffected.

The default limits are 64 MiB per stored genotype block and 256 MiB per expanded
block. Decompression stops one byte beyond the declared expanded length, and
Zstandard's decoder window is bounded. These limits apply per record; the
profiler does not allocate a genotype matrix or load the entire BGEN file.
Explicitly increase `max_compressed_bytes` or `max_decompressed_bytes` only when
the intended source requires it. A record-count limit bounds the sequential
scan, while allele metadata is skipped without retaining its contents.

Invalid magic, overlapping or out-of-file offsets, reserved flags, inconsistent
counts, malformed/truncated blocks, invalid phase/precision/ploidy declarations,
and length mismatches raise `BgenProfileError`. A corrupted uninspected suffix
can still pass a prefix inspection. An entire declared-record inspection also
rejects trailing file bytes. Changes to the opened BGEN's size or modification
metadata during profiling are rejected.

If an Oxford file is supplied, it is streamed completely to check its header,
type row, exact row widths, BGEN sample count, and unique `(FID, IID)` pairs.
Repeated family identifiers are valid, as are repeated individual identifiers
in different families. The uniqueness set uses memory proportional to sample
count. Identifiers are omitted from both the returned report and validation
errors. An explicit 65,536-character line limit prevents unbounded malformed
sample rows.
