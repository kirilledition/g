#![warn(clippy::pedantic)]

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

const BED_MAGIC_HEADER: [u8; 3] = [0x6C, 0x1B, 0x01];

/// Structured output for a contiguous PLINK BED chunk read.
///
/// Allocation behavior:
/// - Allocates one variant decode buffer sized to the number of packed bytes per variant.
/// - Allocates one contiguous host matrix buffer for `sample_count * variant_count` `f64` values.
/// - Allocates one Python `bytes` object containing a little-endian copy of the decoded matrix.
#[pyclass(get_all)]
struct NativeBedChunkReadResult {
    sample_count: usize,
    variant_count: usize,
    genotype_values_le: Py<PyBytes>,
}

/// Structured output for host-side genotype preprocessing.
///
/// Allocation behavior:
/// - Allocates one contiguous host matrix buffer for the mean-imputed genotype matrix.
/// - Allocates one contiguous host mask buffer with one byte per genotype entry.
/// - Allocates one allele-frequency vector and one observation-count vector.
/// - Allocates four Python `bytes` objects to expose those buffers back to Python.
#[pyclass(get_all)]
struct NativePreprocessedGenotypeChunkResult {
    sample_count: usize,
    variant_count: usize,
    imputed_genotype_values_le: Py<PyBytes>,
    missing_mask_values: Py<PyBytes>,
    allele_one_frequency_le: Py<PyBytes>,
    observation_count_le: Py<PyBytes>,
}

struct BedDimensions {
    sample_count: usize,
    variant_count: usize,
    packed_bytes_per_variant: usize,
}

/// Read a contiguous block of variants from a PLINK BED file.
///
/// The returned values match the current Python `bed-reader` path with `count_A1=True`
/// and are laid out in row-major order for a `(sample_count, variant_count)` matrix.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn read_bed_chunk_f64(
    py: Python<'_>,
    bed_path: PathBuf,
    sample_indices: Vec<usize>,
    variant_start: usize,
    variant_stop: usize,
) -> PyResult<Py<NativeBedChunkReadResult>> {
    if variant_stop < variant_start {
        return Err(PyValueError::new_err("variant_stop must be greater than or equal to variant_start."));
    }

    let bed_dimensions = infer_bed_dimensions(&bed_path)?;
    validate_requested_indices(
        &sample_indices,
        bed_dimensions.sample_count,
        variant_start,
        variant_stop,
        bed_dimensions.variant_count,
    )?;

    let variant_count = variant_stop - variant_start;
    let mut bed_file = File::open(&bed_path).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut bed_header = [0_u8; 3];
    bed_file.read_exact(&mut bed_header).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if bed_header != BED_MAGIC_HEADER {
        return Err(PyValueError::new_err("BED file does not use the expected SNP-major PLINK header."));
    }

    let mut packed_variant_buffer = vec![0_u8; bed_dimensions.packed_bytes_per_variant];
    let mut genotype_values = vec![0.0_f64; sample_indices.len() * variant_count];
    for variant_offset in 0..variant_count {
        let file_variant_index = variant_start + variant_offset;
        let variant_file_offset =
            compute_variant_file_offset(file_variant_index, bed_dimensions.packed_bytes_per_variant)?;
        bed_file
            .seek(SeekFrom::Start(variant_file_offset))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        bed_file.read_exact(&mut packed_variant_buffer).map_err(|error| PyValueError::new_err(error.to_string()))?;

        for (output_sample_index, sample_index) in sample_indices.iter().copied().enumerate() {
            let genotype_value = decode_bed_value(&packed_variant_buffer, sample_index);
            let output_index = output_sample_index * variant_count + variant_offset;
            genotype_values[output_index] = genotype_value;
        }
    }

    let genotype_value_bytes = encode_f64_bytes(&genotype_values);

    Py::new(
        py,
        NativeBedChunkReadResult {
            sample_count: sample_indices.len(),
            variant_count,
            genotype_values_le: PyBytes::new(py, &genotype_value_bytes).unbind(),
        },
    )
}

/// Preprocess a row-major genotype matrix with Phase 1 missing-value semantics.
///
/// The input buffer must expose a C-contiguous `float64` matrix with shape
/// `(sample_count, variant_count)`. Missing values are represented by `NaN`.
#[pyfunction]
fn preprocess_genotype_matrix_f64(
    py: Python<'_>,
    genotype_matrix: &Bound<'_, PyAny>,
) -> PyResult<Py<NativePreprocessedGenotypeChunkResult>> {
    let genotype_buffer = PyBuffer::<f64>::get(genotype_matrix)?;
    if genotype_buffer.dimensions() != 2 {
        return Err(PyValueError::new_err("Genotype matrix buffer must be two-dimensional."));
    }

    let genotype_values = genotype_buffer
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("Genotype matrix buffer must be C-contiguous float64 data."))?;
    let shape = genotype_buffer.shape();
    let sample_count = shape[0];
    let variant_count = shape[1];
    let element_count = sample_count
        .checked_mul(variant_count)
        .ok_or_else(|| PyValueError::new_err("Genotype matrix shape exceeds supported range."))?;
    if genotype_values.len() != element_count {
        return Err(PyValueError::new_err("Genotype matrix buffer length does not match its shape."));
    }

    let mut imputed_genotype_values = vec![0.0_f64; element_count];
    let mut missing_mask_values = vec![0_u8; element_count];
    let mut observed_genotype_totals = vec![0.0_f64; variant_count];
    let mut observation_counts = vec![0_i64; variant_count];

    for sample_index in 0..sample_count {
        let row_offset = sample_index * variant_count;
        for variant_index in 0..variant_count {
            let value = genotype_values[row_offset + variant_index].get();
            if value.is_nan() {
                missing_mask_values[row_offset + variant_index] = 1;
            } else {
                imputed_genotype_values[row_offset + variant_index] = value;
                observed_genotype_totals[variant_index] += value;
                observation_counts[variant_index] += 1;
            }
        }
    }

    let mut allele_one_frequency_values = Vec::with_capacity(variant_count);
    let mut column_means = Vec::with_capacity(variant_count);
    for variant_index in 0..variant_count {
        let observation_count = observation_counts[variant_index];
        let column_mean = if observation_count > 0 {
            let observation_count_f64 = f64::from(
                u32::try_from(observation_count)
                    .map_err(|_| PyValueError::new_err("Observation count exceeds supported range."))?,
            );
            observed_genotype_totals[variant_index] / observation_count_f64
        } else {
            0.0
        };
        column_means.push(column_mean);
        allele_one_frequency_values.push(column_mean / 2.0);
    }

    for (variant_index, column_mean) in column_means.iter().enumerate() {
        for sample_index in 0..sample_count {
            let output_index = sample_index * variant_count + variant_index;
            if missing_mask_values[output_index] != 0 {
                imputed_genotype_values[output_index] = *column_mean;
            }
        }
    }

    let imputed_genotype_bytes = encode_f64_bytes(&imputed_genotype_values);
    let allele_one_frequency_bytes = encode_f64_bytes(&allele_one_frequency_values);
    let observation_count_bytes = encode_i64_bytes(&observation_counts);

    Py::new(
        py,
        NativePreprocessedGenotypeChunkResult {
            sample_count,
            variant_count,
            imputed_genotype_values_le: PyBytes::new(py, &imputed_genotype_bytes).unbind(),
            missing_mask_values: PyBytes::new(py, &missing_mask_values).unbind(),
            allele_one_frequency_le: PyBytes::new(py, &allele_one_frequency_bytes).unbind(),
            observation_count_le: PyBytes::new(py, &observation_count_bytes).unbind(),
        },
    )
}

fn infer_bed_dimensions(bed_path: &Path) -> PyResult<BedDimensions> {
    let family_path = bed_path.with_extension("fam");
    let family_file = File::open(&family_path).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let family_reader = BufReader::new(family_file);
    let sample_count = family_reader.lines().try_fold(0_usize, |line_count, line_result| {
        line_result.map(|_| line_count + 1).map_err(|error| PyValueError::new_err(error.to_string()))
    })?;
    if sample_count == 0 {
        return Err(PyValueError::new_err("FAM file does not contain any samples."));
    }

    let packed_bytes_per_variant = sample_count.div_ceil(4);
    let bed_file_size = std::fs::metadata(bed_path).map_err(|error| PyValueError::new_err(error.to_string()))?.len();
    let genotype_bytes = bed_file_size
        .checked_sub(3)
        .ok_or_else(|| PyValueError::new_err("BED file is too small to contain a PLINK header."))?;
    let packed_bytes_per_variant_u64 = u64::try_from(packed_bytes_per_variant)
        .map_err(|_| PyValueError::new_err("Packed variant byte count exceeds supported range."))?;
    if genotype_bytes % packed_bytes_per_variant_u64 != 0 {
        return Err(PyValueError::new_err("BED file size is not consistent with the inferred sample count."));
    }

    let variant_count = usize::try_from(genotype_bytes / packed_bytes_per_variant_u64)
        .map_err(|_| PyValueError::new_err("Variant count exceeds supported range."))?;
    Ok(BedDimensions { sample_count, variant_count, packed_bytes_per_variant })
}

fn validate_requested_indices(
    sample_indices: &[usize],
    sample_count: usize,
    variant_start: usize,
    variant_stop: usize,
    variant_count: usize,
) -> PyResult<()> {
    for sample_index in sample_indices {
        if *sample_index >= sample_count {
            return Err(PyValueError::new_err("Requested sample index is out of range for the BED/FAM file."));
        }
    }
    if variant_stop > variant_count {
        return Err(PyValueError::new_err("Requested variant range exceeds the number of variants in the BED file."));
    }
    if variant_start == variant_stop {
        return Err(PyValueError::new_err("Requested BED chunk must contain at least one variant."));
    }
    Ok(())
}

fn compute_variant_file_offset(variant_index: usize, packed_bytes_per_variant: usize) -> PyResult<u64> {
    let variant_byte_offset = variant_index
        .checked_mul(packed_bytes_per_variant)
        .ok_or_else(|| PyValueError::new_err("Variant byte offset exceeds supported range."))?;
    let file_offset = 3_usize
        .checked_add(variant_byte_offset)
        .ok_or_else(|| PyValueError::new_err("BED file offset exceeds supported range."))?;
    u64::try_from(file_offset).map_err(|_| PyValueError::new_err("BED file offset exceeds supported range."))
}

fn decode_bed_value(packed_variant_buffer: &[u8], sample_index: usize) -> f64 {
    let packed_byte = packed_variant_buffer[sample_index / 4];
    let packed_code = (packed_byte >> ((sample_index % 4) * 2)) & 0b11;
    match packed_code {
        0b00 => 2.0,
        0b01 => f64::NAN,
        0b10 => 1.0,
        0b11 => 0.0,
        _ => unreachable!("Only two-bit BED genotype encodings are possible."),
    }
}

fn encode_f64_bytes(values: &[f64]) -> Vec<u8> {
    let mut output_bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        output_bytes.extend_from_slice(&value.to_le_bytes());
    }
    output_bytes
}

fn encode_i64_bytes(values: &[i64]) -> Vec<u8> {
    let mut output_bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        output_bytes.extend_from_slice(&value.to_le_bytes());
    }
    output_bytes
}

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from g!".to_string()
}

/// A Python module implemented in Rust.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBedChunkReadResult>()?;
    module.add_class::<NativePreprocessedGenotypeChunkResult>()?;
    module.add_function(wrap_pyfunction!(hello_from_bin, module)?)?;
    module.add_function(wrap_pyfunction!(preprocess_genotype_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(read_bed_chunk_f64, module)?)?;
    Ok(())
}
