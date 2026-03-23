#![warn(clippy::pedantic)]

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
        return Err(PyValueError::new_err(
            "variant_stop must be greater than or equal to variant_start.",
        ));
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
    let mut bed_file =
        File::open(&bed_path).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut bed_header = [0_u8; 3];
    bed_file
        .read_exact(&mut bed_header)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if bed_header != BED_MAGIC_HEADER {
        return Err(PyValueError::new_err(
            "BED file does not use the expected SNP-major PLINK header.",
        ));
    }

    let mut packed_variant_buffer = vec![0_u8; bed_dimensions.packed_bytes_per_variant];
    let mut genotype_values = vec![0.0_f64; sample_indices.len() * variant_count];
    for variant_offset in 0..variant_count {
        let file_variant_index = variant_start + variant_offset;
        let variant_file_offset = compute_variant_file_offset(
            file_variant_index,
            bed_dimensions.packed_bytes_per_variant,
        )?;
        bed_file
            .seek(SeekFrom::Start(variant_file_offset))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        bed_file
            .read_exact(&mut packed_variant_buffer)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        for (output_sample_index, sample_index) in sample_indices.iter().copied().enumerate() {
            let genotype_value = decode_bed_value(&packed_variant_buffer, sample_index);
            let output_index = output_sample_index * variant_count + variant_offset;
            genotype_values[output_index] = genotype_value;
        }
    }

    let mut genotype_value_bytes =
        Vec::with_capacity(genotype_values.len() * std::mem::size_of::<f64>());
    for genotype_value in genotype_values {
        genotype_value_bytes.extend_from_slice(&genotype_value.to_le_bytes());
    }

    Py::new(
        py,
        NativeBedChunkReadResult {
            sample_count: sample_indices.len(),
            variant_count,
            genotype_values_le: PyBytes::new(py, &genotype_value_bytes).unbind(),
        },
    )
}

fn infer_bed_dimensions(bed_path: &Path) -> PyResult<BedDimensions> {
    let family_path = bed_path.with_extension("fam");
    let family_file =
        File::open(&family_path).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let family_reader = BufReader::new(family_file);
    let sample_count = family_reader
        .lines()
        .try_fold(0_usize, |line_count, line_result| {
            line_result
                .map(|_| line_count + 1)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })?;
    if sample_count == 0 {
        return Err(PyValueError::new_err(
            "FAM file does not contain any samples.",
        ));
    }

    let packed_bytes_per_variant = sample_count.div_ceil(4);
    let bed_file_size = std::fs::metadata(bed_path)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .len();
    let genotype_bytes = bed_file_size
        .checked_sub(3)
        .ok_or_else(|| PyValueError::new_err("BED file is too small to contain a PLINK header."))?;
    let packed_bytes_per_variant_u64 = u64::try_from(packed_bytes_per_variant)
        .map_err(|_| PyValueError::new_err("Packed variant byte count exceeds supported range."))?;
    if genotype_bytes % packed_bytes_per_variant_u64 != 0 {
        return Err(PyValueError::new_err(
            "BED file size is not consistent with the inferred sample count.",
        ));
    }

    let variant_count = usize::try_from(genotype_bytes / packed_bytes_per_variant_u64)
        .map_err(|_| PyValueError::new_err("Variant count exceeds supported range."))?;
    Ok(BedDimensions {
        sample_count,
        variant_count,
        packed_bytes_per_variant,
    })
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
            return Err(PyValueError::new_err(
                "Requested sample index is out of range for the BED/FAM file.",
            ));
        }
    }
    if variant_stop > variant_count {
        return Err(PyValueError::new_err(
            "Requested variant range exceeds the number of variants in the BED file.",
        ));
    }
    if variant_start == variant_stop {
        return Err(PyValueError::new_err(
            "Requested BED chunk must contain at least one variant.",
        ));
    }
    Ok(())
}

fn compute_variant_file_offset(
    variant_index: usize,
    packed_bytes_per_variant: usize,
) -> PyResult<u64> {
    let variant_byte_offset = variant_index
        .checked_mul(packed_bytes_per_variant)
        .ok_or_else(|| PyValueError::new_err("Variant byte offset exceeds supported range."))?;
    let file_offset = 3_usize
        .checked_add(variant_byte_offset)
        .ok_or_else(|| PyValueError::new_err("BED file offset exceeds supported range."))?;
    u64::try_from(file_offset)
        .map_err(|_| PyValueError::new_err("BED file offset exceeds supported range."))
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

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from g!".to_string()
}

/// A Python module implemented in Rust.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBedChunkReadResult>()?;
    module.add_function(wrap_pyfunction!(hello_from_bin, module)?)?;
    module.add_function(wrap_pyfunction!(read_bed_chunk_f64, module)?)?;
    Ok(())
}
