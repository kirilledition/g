# REGENIE Step 2 Linear Association Implementation Plan

## Overview

This document describes the implementation plan for `g regenie2-linear`, a JAX-accelerated REGENIE step 2 linear association testing command. The feature accepts REGENIE step 1 output (LOCO predictions), reads test genotypes from BGEN files, and produces association statistics matching REGENIE's output format.

## Background

### REGENIE Step 2 Algorithm

REGENIE step 2 for quantitative (continuous) traits follows this mathematical formulation:

1. **Residualize phenotype against covariates**: Remove covariate effects from the phenotype
2. **Subtract LOCO predictions**: Remove leave-one-chromosome-out (LOCO) genetic predictions from the residual
3. **Score test**: For each variant on the left-out chromosome, compute association statistics

The core formula:
```
y_residual = phenotype - covariates_effect - LOCO_prediction
stat = (y_residual' * G) / sqrt(G' * G)
chi_squared = stat^2
log10_p = -log10(chi2_to_p(chi_squared, df=1))
```

### Key Differences from Existing `g linear` Command

| Aspect | `g linear` | `g regenie2-linear` |
|--------|------------|---------------------|
| Input | Genotypes + phenotype + covariates | Genotypes + phenotype + covariates + **LOCO predictions** |
| Covariate handling | Full regression with Cholesky factorization | Residualization only (LOCO captures genetic background) |
| Test statistic | t-statistic | Chi-squared |
| P-value format | Raw p-value | LOG10P (negative log10 p-value) |
| Per-chromosome state | None | LOCO predictions change per chromosome |

### REGENIE File Formats

**`_pred.list` format** (space-delimited):
```
phenotype_name /path/to/phenotype.loco
```

**`.loco` file format** (space-delimited matrix):
```
FID_IID sample1_fid_iid sample2_fid_iid ...
1 pred_chr1_sample1 pred_chr1_sample2 ...
2 pred_chr2_sample1 pred_chr2_sample2 ...
...
22 pred_chr22_sample1 pred_chr22_sample2 ...
```
- Header row contains sample identifiers formatted as `FID_IID`
- Row 1 contains chromosome 1 LOCO predictions (i.e., predictions when chr1 is left out)
- Sample IDs use underscore separator (e.g., `0_HG00096`)

---

## Implementation Phases

### Phase 0: Generate Continuous Phenotype REGENIE Baselines

**Goal**: Generate REGENIE baselines for continuous phenotype to enable parity testing.

**Files to modify**:
- `scripts/benchmark.py`

**Changes**:

1. Add `regenie_step1_qt_prediction_list_path` to `BaselinePaths`:
   ```python
   regenie_step1_qt_prediction_list_path: Path  # data/baselines/regenie_step1_qt_pred.list
   ```

2. Add `build_regenie_step1_continuous_command()`:
   ```python
   def build_regenie_step1_continuous_command(regenie_executable: str, baseline_paths: BaselinePaths) -> list[str]:
       """Build the Regenie step 1 continuous trait command."""
       return [
           regenie_executable,
           "--step", "1",
           "--bed", str(baseline_paths.bed_prefix),
           "--phenoFile", str(baseline_paths.continuous_phenotype_path),
           "--covarFile", str(baseline_paths.covariate_path),
           "--qt",  # quantitative trait (instead of --bt)
           "--force-step1",
           "--bsize", "1000",
           "--out", str(baseline_paths.baseline_directory / "regenie_step1_qt"),
       ]
   ```

3. Add `build_regenie_step2_continuous_command()`:
   ```python
   def build_regenie_step2_continuous_command(regenie_executable: str, baseline_paths: BaselinePaths) -> list[str]:
       """Build the Regenie step 2 continuous trait command."""
       return [
           regenie_executable,
           "--step", "2",
           "--bgen", str(baseline_paths.bgen_path),
           "--sample", str(baseline_paths.sample_path),
           "--ref-first",
           "--phenoFile", str(baseline_paths.continuous_phenotype_path),
           "--covarFile", str(baseline_paths.covariate_path),
           "--qt",  # quantitative trait
           "--bsize", "400",
           "--pred", str(baseline_paths.regenie_step1_qt_prediction_list_path),
           "--out", str(baseline_paths.baseline_directory / "regenie_step2_qt"),
       ]
   ```

4. Update `main()` to run both continuous REGENIE steps:
   - Add `regenie_step1_qt` command execution
   - Add `regenie_step2_qt` command execution (conditional on step 1 success)

**Expected outputs**:
- `data/baselines/regenie_step1_qt_pred.list`
- `data/baselines/regenie_step1_qt_1.loco`
- `data/baselines/regenie_step2_phenotype_continuous.regenie`

**Tests to add** (in `tests/test_phase0.py`):
- `test_regenie_step_two_continuous_command_uses_qt_flag()`

---

### Phase 1: LOCO File Parsers (`src/g/io/regenie.py`)

**Goal**: Create I/O module for reading REGENIE step 1 output files.

**New file**: `src/g/io/regenie.py`

**Data structures**:

```python
@dataclass(frozen=True)
class LocoSampleIndex:
    """Sample alignment index for LOCO predictions.
    
    Attributes:
        family_identifiers: Family identifiers from LOCO header.
        individual_identifiers: Individual identifiers from LOCO header.
        loco_sample_count: Number of samples in the LOCO file.
    
    """
    family_identifiers: npt.NDArray[np.str_]
    individual_identifiers: npt.NDArray[np.str_]
    loco_sample_count: int


@dataclass(frozen=True)
class LocoPredictions:
    """LOCO predictions for all chromosomes.
    
    Attributes:
        sample_index: Sample alignment information.
        chromosome_predictions: Dictionary mapping chromosome string to prediction array.
    
    """
    sample_index: LocoSampleIndex
    chromosome_predictions: dict[str, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class PredictionListEntry:
    """Single entry from a _pred.list file.
    
    Attributes:
        phenotype_name: Name of the phenotype.
        loco_file_path: Path to the .loco file.
    
    """
    phenotype_name: str
    loco_file_path: Path
```

**Protocol** (for future extensibility):

```python
class Step1PredictionSource(typing.Protocol):
    """Protocol for step 1 prediction sources.
    
    Allows future implementations (e.g., native step 1) to share the same interface.
    """
    
    def get_chromosome_predictions(
        self,
        chromosome: str,
        sample_family_identifiers: npt.NDArray[np.str_],
        sample_individual_identifiers: npt.NDArray[np.str_],
    ) -> jax.Array:
        """Return LOCO predictions for a chromosome aligned to the sample order."""
        ...
```

**Functions**:

```python
def parse_prediction_list_file(prediction_list_path: Path) -> list[PredictionListEntry]:
    """Parse a REGENIE _pred.list file."""

def parse_loco_file(loco_file_path: Path) -> LocoPredictions:
    """Parse a REGENIE .loco file into sample index and predictions."""

def parse_loco_sample_identifiers(header_line: str) -> LocoSampleIndex:
    """Parse the FID_IID header line from a .loco file."""

def build_sample_alignment_indices(
    loco_sample_index: LocoSampleIndex,
    target_family_identifiers: npt.NDArray[np.str_],
    target_individual_identifiers: npt.NDArray[np.str_],
) -> npt.NDArray[np.int64]:
    """Build indices to align LOCO samples to target sample order."""

def load_aligned_chromosome_predictions(
    loco_predictions: LocoPredictions,
    chromosome: str,
    target_family_identifiers: npt.NDArray[np.str_],
    target_individual_identifiers: npt.NDArray[np.str_],
) -> jax.Array:
    """Load and align LOCO predictions for a specific chromosome."""
```

**Error handling**:
- `ValueError` for malformed files (missing header, wrong column count, invalid chromosome)
- `KeyError` for missing chromosomes in LOCO file
- Raise clear errors when samples don't align

---

### Phase 2: Compute Kernel (`src/g/compute/regenie2_linear.py`)

**Goal**: JAX kernel implementing REGENIE step 2 linear regression with LOCO adjustment.

**New file**: `src/g/compute/regenie2_linear.py`

**Data structures** (add to `src/g/models.py`):

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearState:
    """Precomputed state for REGENIE step 2 linear association.
    
    Attributes:
        covariate_matrix: Covariate design matrix (including intercept).
        covariate_projection: Projection matrix for residualization.
        phenotype_residual: Phenotype residualized against covariates (before LOCO).
    
    """
    covariate_matrix: jax.Array
    covariate_projection: jax.Array
    phenotype_residual: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2LinearChunkResult:
    """Association outputs for a REGENIE step 2 linear chunk.
    
    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        valid_mask: Boolean mask for valid statistics.
    
    """
    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array
```

**Functions**:

```python
def prepare_regenie2_linear_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> Regenie2LinearState:
    """Prepare covariate projection and phenotype residual.
    
    Residualizes the phenotype against covariates but does NOT subtract
    LOCO predictions (that happens per-chromosome in the chunk function).
    """

def compute_regenie2_linear_chunk(
    state: Regenie2LinearState,
    genotype_matrix: jax.Array,
    loco_predictions: jax.Array,
) -> Regenie2LinearChunkResult:
    """Compute REGENIE step 2 linear association for a genotype chunk.
    
    Args:
        state: Precomputed covariate state.
        genotype_matrix: Genotype dosage matrix (samples x variants).
        loco_predictions: LOCO predictions for this chromosome (samples,).
    
    Returns:
        Association statistics for the chunk.
    
    Mathematical formulation:
        1. adjusted_residual = phenotype_residual - loco_predictions
        2. For each variant g:
           - g_residual = g - X @ (X'X)^-1 @ X' @ g  (residualize genotype)
           - beta = (g_residual' @ adjusted_residual) / (g_residual' @ g_residual)
           - var = residual_variance / (g_residual' @ g_residual)
           - chi_squared = beta^2 / var
           - log10_p = -log10(chi2_to_p(chi_squared, df=1))
    """

def chi_squared_to_log10_p_value(chi_squared: jax.Array) -> jax.Array:
    """Convert chi-squared statistics to negative log10 p-values."""
```

**Key implementation notes**:
- Use `jax.scipy.stats.chi2.sf()` for p-value calculation (survival function)
- Handle numerical precision for very small p-values (large -log10 p)
- Vectorize over variants using `jax.vmap` or batch matrix operations

---

### Phase 3: API and Orchestration (`src/g/regenie2.py`)

**Goal**: High-level API function and configuration for REGENIE step 2.

**New file**: `src/g/regenie2.py`

**Configuration**:

```python
@dataclasses.dataclass(frozen=True)
class Regenie2LinearConfig:
    """Configuration for REGENIE step 2 linear association.
    
    Attributes:
        prediction_list_path: Path to the _pred.list file from REGENIE step 1.
        phenotype_name: Name of phenotype (must match entry in prediction list).
    
    """
    prediction_list_path: Path
    phenotype_name: str
```

**API function**:

```python
def regenie2_linear(
    *,
    bgen: Path | str,
    sample: Path | str | None = None,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    pred: Path | str,
    compute: api.ComputeConfig | None = None,
    solver: Regenie2LinearConfig | None = None,
) -> api.RunArtifacts:
    """Run REGENIE step 2 linear association and write results to disk.
    
    Args:
        bgen: Path to BGEN file containing test variants.
        sample: Optional sample file path.
        pheno: Path to phenotype file.
        pheno_name: Phenotype column name.
        out: Output path or prefix.
        covar: Optional covariate file path.
        covar_names: Covariate column names.
        pred: Path to REGENIE step 1 _pred.list file.
        compute: Compute configuration.
        solver: REGENIE step 2 solver configuration.
    
    Returns:
        Artifacts pointing to output files.
    
    """
```

**Engine integration**:
- Create `iter_regenie2_linear_output_frames()` generator in `src/g/engine.py`
- Handle per-chromosome LOCO prediction switching
- Track current chromosome and reload predictions when chromosome changes

---

### Phase 4: CLI Command (`src/g/cli.py`)

**Goal**: Add `g regenie2-linear` subcommand.

**Modifications to `src/g/cli.py`**:

```python
@app.command("regenie2-linear", no_args_is_help=True)
def run_regenie2_linear_command(
    bgen: Path = typer.Option(..., help="BGEN file path."),
    sample: Path | None = typer.Option(None, help="Optional BGEN sample-file path."),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name."),
    out: Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate names."),
    pred: Path = typer.Option(..., help="REGENIE step 1 _pred.list file path."),
    chunk_size: int | None = typer.Option(None, help="Variants per chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_mode: types.OutputMode = typer.Option(types.OutputMode.TSV, help="Output format."),
) -> None:
    """Run REGENIE step 2 linear association scan."""
```

**Notes**:
- No `--bfile` option (REGENIE step 2 uses BGEN for test variants)
- Requires `--pred` option for step 1 predictions
- Default chunk size should match existing linear default (2048)

---

### Phase 5: Output Schema (`src/g/io/output.py`)

**Goal**: Add output schema for REGENIE step 2 linear results.

**New schema**:

```python
REGENIE2_LINEAR_OUTPUT_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
    "chunk_identifier": pl.Int64(),
    "variant_start_index": pl.Int64(),
    "variant_stop_index": pl.Int64(),
    "chromosome": pl.String(),
    "position": pl.Int64(),
    "variant_identifier": pl.String(),
    "allele_one": pl.String(),
    "allele_two": pl.String(),
    "allele_one_frequency": pl.Float32(),
    "observation_count": pl.Int32(),
    "beta": pl.Float32(),
    "standard_error": pl.Float32(),
    "chi_squared": pl.Float32(),
    "log10_p_value": pl.Float32(),
    "is_valid": pl.Boolean(),
}
```

**Modifications**:
- Add `REGENIE2_LINEAR` to `types.AssociationMode` enum
- Update `get_output_schema()` to handle new mode
- Add `Regenie2LinearChunkPayload` dataclass to `src/g/engine.py`

---

### Phase 6: Tests

**Goal**: Comprehensive test coverage for all new functionality.

#### Unit Tests: `tests/test_io_regenie.py`

```python
def test_parse_prediction_list_file_extracts_entries() -> None:
    """Ensure _pred.list parsing extracts phenotype name and path."""

def test_parse_loco_file_extracts_all_chromosomes() -> None:
    """Ensure .loco file parsing extracts all chromosome predictions."""

def test_parse_loco_sample_identifiers_splits_fid_iid() -> None:
    """Ensure FID_IID format is correctly parsed."""

def test_build_sample_alignment_indices_handles_reordering() -> None:
    """Ensure sample alignment works when orders differ."""

def test_build_sample_alignment_indices_raises_on_missing_samples() -> None:
    """Ensure clear error when target samples missing from LOCO."""

def test_load_aligned_chromosome_predictions_returns_jax_array() -> None:
    """Ensure aligned predictions are returned as JAX array."""
```

#### Compute Tests: `tests/test_regenie2_linear.py`

```python
def test_prepare_regenie2_linear_state_creates_projection() -> None:
    """Ensure state preparation creates valid projection matrix."""

def test_compute_regenie2_linear_chunk_matches_manual_calculation() -> None:
    """Validate chunk computation against manual numpy calculation."""

def test_chi_squared_to_log10_p_value_handles_edge_cases() -> None:
    """Ensure log10 p-value conversion handles small and large chi-squared."""

def test_compute_regenie2_linear_chunk_handles_zero_variance_genotypes() -> None:
    """Ensure monomorphic variants are marked invalid."""
```

#### Parity Tests: `tests/test_regenie2_parity.py`

```python
@pytest.mark.phase0_data
def test_regenie2_linear_matches_regenie_baseline_beta() -> None:
    """Validate beta estimates match REGENIE within tolerance."""

@pytest.mark.phase0_data
def test_regenie2_linear_matches_regenie_baseline_log10p() -> None:
    """Validate -log10(p) values match REGENIE within tolerance."""

@pytest.mark.phase0_data
def test_regenie2_linear_api_produces_valid_output() -> None:
    """End-to-end test of regenie2_linear API function."""
```

**Tolerance guidelines** (from existing test patterns):
- Beta: `atol=1e-3` (absolute tolerance)
- Log10 p-value: log10 error of `1e-2` (i.e., `|log10(our_p) - log10(regenie_p)| < 1e-2`)
- Standard error: `rtol=1e-2` (relative tolerance)

---

## File Summary

### New Files (7)

| File | Purpose |
|------|---------|
| `src/g/io/regenie.py` | LOCO file parsers and Step1PredictionSource protocol |
| `src/g/compute/regenie2_linear.py` | JAX compute kernel |
| `src/g/regenie2.py` | API function and Regenie2LinearConfig |
| `tests/test_io_regenie.py` | LOCO parser unit tests |
| `tests/test_regenie2_linear.py` | Compute kernel unit tests |
| `tests/test_regenie2_parity.py` | Parity tests against REGENIE baselines |

### Modified Files (7)

| File | Changes |
|------|---------|
| `scripts/benchmark.py` | Add continuous phenotype REGENIE commands |
| `src/g/cli.py` | Add `regenie2-linear` command |
| `src/g/io/output.py` | Add `REGENIE2_LINEAR_OUTPUT_SCHEMA` |
| `src/g/models.py` | Add `Regenie2LinearState`, `Regenie2LinearChunkResult` |
| `src/g/__init__.py` | Export `regenie2_linear` |
| `src/g/types.py` | Add `REGENIE2_LINEAR` to `AssociationMode` enum |
| `src/g/engine.py` | Add chunk payload and iterator for REGENIE2 |
| `tests/test_phase0.py` | Add test for continuous REGENIE command |

---

## Implementation Order

1. **Phase 0** (Baseline generation) - Required first for parity testing
2. **Phase 1** (I/O parsers) - Foundation for all other phases
3. **Phase 2** (Compute kernel) - Core algorithm
4. **Phase 5** (Output schema) - Required before Phase 3
5. **Phase 3** (API/orchestration) - Ties everything together
6. **Phase 4** (CLI) - User-facing command
7. **Phase 6** (Tests) - Can be developed incrementally with each phase

---

## Future Extensions

This architecture supports:

1. **`g regenie2-logistic`**: Binary trait extension using same LOCO infrastructure
2. **Native step 1 implementation**: `Step1PredictionSource` protocol allows swapping REGENIE output with our own predictions
3. **Multi-phenotype analysis**: Prediction list format supports multiple phenotypes
4. **Distributed execution**: Per-chromosome LOCO switching enables chromosome-parallel processing

---

## Open Questions

1. **Chromosome naming**: Should we normalize chromosome strings (e.g., "chr22" vs "22")? REGENIE uses numeric-only.
2. **Memory management**: Should we cache all LOCO predictions at startup or load per-chromosome on demand?
3. **Missing samples**: How should we handle samples present in phenotype but missing from LOCO file?

---

## References

- REGENIE source: `/home/kirill/Projects/regenie/src/Step2_Models.cpp`
- REGENIE documentation: `/home/kirill/Projects/regenie/docs/docs/overview.md`
- Existing linear kernel: `src/g/compute/linear.py`
- BGEN reader: `src/g/io/bgen.py`
