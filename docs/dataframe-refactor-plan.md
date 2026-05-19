My performance recommendation is:

```text
Required runtime:
  Rust native parsers + NumPy/JAX arrays + Rust Arrow/Parquet writer

Optional only:
  PyArrow for reading Arrow/Parquet output from Python
  Polars only for dev scripts / exploratory analysis, or remove it entirely
```

So: **do not replace Polars with PyArrow in the core. Replace Polars with no DataFrame library in the core.**

That is the fastest option for your app.

---

# Why Polars is not helping your current app

Polars is fast for DataFrame-style workloads, but your hot path is not really a DataFrame workload. Your hot path is:

```text
BGEN decode
native preprocessing
host → device transfer
JAX compute
device → host transfer
Rust Arrow/Parquet output
```

In the current snapshot, Polars is used mainly in these places:

```text
src/g/io/samples.py          phenotype/covariate DataFrame joins
src/g/io/bgen/sample.py      sample table construction
src/g/io/bgen/metadata.py    variant metadata DataFrame construction
src/g/io/reader.py           legacy variant_table protocol
src/g/io/output.py           read/scan chunk helpers
scripts/                     benchmark/output inspection
```

The active engine path already relies on Rust for BGEN chunk delivery and Rust for output writing. Resume scanning is also now native via `_core.scan_committed_chunk_identifiers`. So Polars mostly adds:

```text
Rust/Python boundary crossing
DataFrame construction
DataFrame → NumPy conversion
NumPy → JAX conversion
extra imports and binary dependency weight
```

One especially wasteful path is embedded-sample handling:

```text
Rust engine sample identifiers
    ↓
Python ndarray/list
    ↓
Polars DataFrame
    ↓
to_numpy / to_list
    ↓
Rust align_sample_data(...)
```

That should be:

```text
Rust engine sample identifiers
    ↓
Rust align_sample_data(...)
```

No DataFrame. No Python table object.

---

# Best replacement: Rust selected-column parsing and direct arrays

For phenotype, covariate, and sample alignment, the fastest architecture is a Rust-native streaming parser that reads only the columns you need and fills typed arrays directly.

The target should be:

```text
BGEN sample IDs / .sample file
    ↓
Rust sample index map
    ↓
Rust TSV phenotype/covariate parser
    ↓
NativeAlignedSampleData
    ↓
NumPy/JAX arrays
```

Not:

```text
TSV → Polars DataFrame → join → DataFrame → NumPy → JAX
```

And not:

```text
TSV → PyArrow Table → join → NumPy → JAX
```

A proper Rust parser is the right tool here. The Rust `csv` crate is designed for fast, flexible CSV/TSV reading and supports configurable delimiters such as tabs. ([Docs.rs][1])

---

# What I would implement

## 1. Remove Polars from required dependencies

Change:

```toml
dependencies = [
    "jax[cpu]>=0.10.0",
    "numpy>=2.4.5",
    "polars>=1.40.1",
    "typer>=0.25.1",
]
```

to something like:

```toml
dependencies = [
    "jax[cpu]>=0.10.0",
    "numpy>=2.4.5",
    "click>=8.0",
]
```

Then, only if you want optional output inspection:

```toml
[project.optional-dependencies]
arrow = [
    "pyarrow>=24.0.0",
]

polars = [
    "polars>=1.40.1",
]

dev = [
    "pyarrow>=24.0.0",
    "polars>=1.40.1",
]
```

But for maximum performance purity, the production CLI should not require either PyArrow or Polars.

---

## 2. Delete the Polars alignment path

The current `src/g/io/samples.py` path should not be part of the production engine.

Replace:

```python
load_phenotype_or_covariate_table(...)
load_aligned_sample_data_from_individual_identifier_table(...)
build_aligned_sample_data(...)
convert_frame_to_float32_jax(...)
```

with a Rust-backed call only.

Current active Python wrapper:

```python
_core.align_sample_data(
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_name,
    covariate_path,
    covariate_names,
    is_binary_trait,
)
```

Better: avoid even passing sample identifiers through Python when possible.

Add a Rust method on `Regenie2RunEngine`:

```python
native_aligned_sample_data = engine.align_sample_data(
    sample_path=str(sample_path) if sample_path else None,
    phenotype_path=str(phenotype_path),
    phenotype_name=phenotype_name,
    covariate_path=str(covariate_path) if covariate_path else None,
    covariate_names=list(covariate_names) if covariate_names else None,
    is_binary_trait=is_binary_trait,
)
```

Inside Rust:

```text
if external .sample exists:
    parse .sample in Rust
elif BGEN has embedded samples:
    use engine-owned sample identifiers directly
else:
    error
```

That removes the embedded-sample Polars round trip entirely.

---

## 3. Parse phenotype/covariate tables directly into aligned arrays

Do not build a table and then join.

Use this algorithm:

```text
1. Build HashMap<IID, sample_position> from the selected BGEN samples.
2. Allocate phenotype_vector = vec![NaN; sample_count].
3. Read phenotype TSV once.
4. For each row:
     find IID
     if IID in sample map:
         parse selected phenotype value
         store into phenotype_vector[sample_position]
5. If covariates exist:
     allocate covariate_matrix = sample_count x (1 + covariate_count)
     fill intercept column with 1.0
     read covariate TSV once
     fill selected covariate values for matched samples
6. Compact to samples with non-missing phenotype and covariates.
7. Return NativeAlignedSampleData.
```

This is faster than a DataFrame join because it avoids:

```text
materializing unused columns
materializing unused rows
allocating intermediate DataFrames
string-heavy join tables
DataFrame → NumPy conversion
```

It also gives you a natural place to validate:

```text
duplicate IID
missing phenotype
missing covariates
binary phenotype coding
case/control counts
FID/IID mode later
```

For multi-phenotype support, extend the same parser to fill:

```text
phenotype_matrix: sample_count x phenotype_count
```

instead of reading the phenotype file repeatedly.

---

## 4. Replace `build_sample_identifier_table` with arrays or remove it

Current code:

```python
def build_sample_identifier_table(sample_identifiers: np.ndarray) -> pl.DataFrame:
    ...
```

This should disappear from the active path.

Use either:

```python
@dataclass(frozen=True)
class SampleIdentifierArrays:
    sample_indices: np.ndarray
    family_identifiers: list[str]
    individual_identifiers: list[str]
```

or, better, keep it fully inside Rust.

For external `.sample`, parse in Rust.

For embedded BGEN samples, use Rust engine sample identifiers.

For generated samples, error or generate in Rust.

---

## 5. Remove Polars variant metadata tables

Current code has:

```python
build_bgen_variant_table(...)
build_variant_table_from_arrays(...)
build_variant_table_from_core_metadata(...)
```

returning `pl.DataFrame`.

The engine does not need this. It needs arrays:

```python
@dataclass(frozen=True)
class VariantTableArrays:
    chromosome_values: np.ndarray
    variant_identifier_values: np.ndarray
    position_values: np.ndarray
    allele_one_values: np.ndarray
    allele_two_values: np.ndarray
```

You already have `VariantTableArrays`. Make that the only internal representation.

If a user wants a table later, provide optional helpers:

```python
artifacts.variant_metadata_arrow()
artifacts.variant_metadata_polars()
```

but do not use those in the engine.

---

## 6. Keep output writing in Rust

Your output writer is already the right direction:

```text
Rust writer
  → Arrow IPC chunks
  → optional Parquet finalization
```

Do not bring PyArrow into the production write path.

PyArrow is useful only if Python users want to inspect output. Arrow is designed as a cross-language columnar format, and PyArrow is the Python binding over the C++ Arrow implementation. ([Apache Arrow][2]) Arrow IPC is also designed for efficient reading/writing with low memory use and memory-mapped access. ([Apache Arrow][3])

So the optional inspection API can be:

```python
def read_output_arrow(path):
    import pyarrow.parquet as pq
    return pq.read_table(path)
```

But the engine should not depend on it.

---

# Should you keep Polars anywhere?

Only optionally.

Polars is still excellent for developer scripts and exploratory output analysis. Its lazy Parquet scanning can push down projections and predicates, which is useful for queries like “show only variants with LOG10P > 8.” ([Polars User Guide][4])

So this is fine in scripts:

```python
import polars as pl

hits = (
    pl.scan_parquet("results/final.parquet")
    .filter(pl.col("LOG10P") > 8)
    .select("CHROM", "GENPOS", "ID", "BETA", "LOG10P")
    .collect()
)
```

But that should be a dev/optional dependency, not a production dependency.

---

# Should you use PyArrow instead of Polars?

For the core: **no**.

For optional output reading: **yes, PyArrow is the better default optional bridge** because your output is already Arrow/Parquet.

The split I would use:

| Task                               | Fastest choice                      |
| ---------------------------------- | ----------------------------------- |
| Phenotype/covariate/sample parsing | Rust `csv` / custom Rust TSV parser |
| Sample alignment                   | Rust arrays + hash maps             |
| Variant metadata in engine         | Rust/native arrays                  |
| Genotype chunk processing          | Rust + NumPy/JAX                    |
| Output writing                     | Rust Arrow/Parquet writer           |
| Output inspection from Python      | optional PyArrow                    |
| Developer exploratory analysis     | optional Polars                     |
| Core runtime DataFrame library     | none                                |

If forced to choose one optional Python table library, I would choose **PyArrow** because it matches your output format. But for the production engine, choose **neither**.

---

# What performance improvement to expect

Dropping Polars probably will **not** magically make the GPU genome scan much faster, because Polars is not the dominant part of the BGEN/JAX hot loop.

But replacing Polars with Rust-native parsing will improve:

```text
CLI import/startup time
sample/phenotype/covariate alignment time
peak memory during alignment
embedded-sample path overhead
dependency size
architecture clarity
future multi-phenotype parsing efficiency
```

The biggest runtime benefit will show up when:

```text
phenotype/covariate files are large
many phenotypes are selected
sample counts are large
you currently rebuild DataFrames repeatedly
```

For single-phenotype runs, the major bottlenecks will still be BGEN decode, transfer, JAX compute, and output writing.

---

# Concrete migration plan

## Phase 1: Remove Polars from imports

Delete top-level `import polars as pl` from production modules.

Affected files:

```text
src/g/io/samples.py
src/g/io/bgen/sample.py
src/g/io/bgen/metadata.py
src/g/io/reader.py
src/g/io/output.py
```

For output helper functions, either delete them or make them optional:

```python
def read_chunk_file_arrow(path):
    try:
        import pyarrow.ipc as ipc
    except ImportError as error:
        raise ImportError("Install g[arrow] to read Arrow chunks from Python.") from error
```

## Phase 2: Make Rust alignment the only production path

Delete or quarantine:

```text
src/g/io/samples.py
src/g/io/source.py load_aligned_sample_data_from_source
```

Then route all active runs through:

```python
load_native_bgen_run_input(...)
```

and make that call Rust directly.

## Phase 3: Move embedded-sample alignment fully into Rust

Replace this:

```python
sample_table = bgen.build_sample_identifier_table(np.asarray(engine.sample_identifiers(), dtype=np.str_))
native_aligned_sample_data = load_native_aligned_sample_data_from_individual_identifier_table(
    sample_table=sample_table,
    ...
)
```

with this:

```python
native_aligned_sample_data = engine.align_sample_data_from_embedded_samples(
    phenotype_path=str(phenotype_path),
    phenotype_name=phenotype_name,
    covariate_path=str(covariate_path) if covariate_path else None,
    covariate_names=list(covariate_names) if covariate_names else None,
    is_binary_trait=is_binary_trait,
)
```

Even better, expose one Rust method:

```python
native_aligned_sample_data = engine.align_sample_data(
    sample_path=str(resolved_sample_path) if resolved_sample_path else None,
    phenotype_path=str(phenotype_path),
    phenotype_name=phenotype_name,
    covariate_path=str(covariate_path) if covariate_path else None,
    covariate_names=list(covariate_names) if covariate_names else None,
    is_binary_trait=is_binary_trait,
)
```

## Phase 4: Replace DataFrame metadata with arrays

Make `VariantTableArrays` the canonical Python-side type.

Remove `variant_table: pl.DataFrame` from the protocol unless it is explicitly optional/dev-only.

## Phase 5: Update scripts

Benchmark scripts currently use Polars mainly to count/read output rows. Replace with either:

```python
# If you keep PyArrow optional/dev
import pyarrow.dataset as ds
row_count = ds.dataset(path).count_rows()
```

or better, use run metadata from the Rust writer so scripts do not scan output just to count rows.

PyArrow’s dataset API is designed for multi-file and potentially larger-than-memory tabular datasets. ([Apache Arrow][5])

## Phase 6: Add anti-regression tests

Add tests like:

```python
def test_import_g_does_not_import_polars():
    import sys
    import g

    assert "polars" not in sys.modules
```

and:

```python
def test_production_cli_help_does_not_require_polars():
    ...
```

Also add:

```text
test_sample_alignment_does_not_construct_dataframe
test_embedded_sample_alignment_stays_in_rust
test_output_resume_uses_native_scan
```

---

# Final answer

For maximum performance, the best option is:

```text
Drop Polars from the production app.
Do not replace it with PyArrow in the core.
Implement phenotype/covariate/sample parsing and alignment in Rust.
Keep output writing in Rust Arrow/Parquet.
Use PyArrow only as an optional output-inspection bridge.
Keep Polars only as an optional dev/exploratory tool, or remove it completely.
```

The performance-efficient architecture is **not “Polars vs PyArrow.”** It is:

```text
No Python DataFrame library in the hot path.
```

[1]: https://docs.rs/csv/latest/csv/tutorial/index.html?utm_source=chatgpt.com "csv::tutorial - Rust"
[2]: https://arrow.apache.org/docs/python/index.html?utm_source=chatgpt.com "Python — Apache Arrow v24.0.0"
[3]: https://arrow.apache.org/docs/python/ipc.html?utm_source=chatgpt.com "Streaming, Serialization, and IPC — Apache Arrow v24.0.0"
[4]: https://docs.pola.rs/api/python/dev/reference/api/polars.scan_parquet.html?utm_source=chatgpt.com "polars.scan_parquet — Polars documentation"
[5]: https://arrow.apache.org/docs/python/dataset.html?utm_source=chatgpt.com "Tabular Datasets — Apache Arrow v24.0.0"
