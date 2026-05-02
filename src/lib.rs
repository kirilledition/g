#![warn(clippy::pedantic)]

pub mod bgen;
pub mod output;

use std::collections::HashMap;

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::bgen::{BgenError, BgenReaderCore, ReaderProfileSnapshot, VariantMetadataLists};
use crate::output::{PyOutputWriterSession, finalize_output_run_chunks};

#[pyclass]
struct PyBgenReader {
    reader: BgenReaderCore,
}

#[pymethods]
impl PyBgenReader {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (bgen_path, trusted_no_missing_diploid=false))]
    fn new(bgen_path: String, trusted_no_missing_diploid: bool) -> PyResult<Self> {
        let reader = BgenReaderCore::open(std::path::Path::new(&bgen_path), trusted_no_missing_diploid)
            .map_err(convert_bgen_error)?;
        Ok(Self { reader })
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.reader.sample_count()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.reader.variant_count()
    }

    #[getter]
    fn contains_embedded_samples(&self) -> bool {
        self.reader.contains_embedded_samples()
    }

    #[getter]
    fn bgen_path(&self) -> String {
        self.reader.bgen_path().display().to_string()
    }

    fn sample_identifiers(&self) -> Vec<String> {
        self.reader.sample_identifiers()
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.reader.chromosome_boundary_indices()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepare_sample_selection(&self, sample_indices: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        self.reader.prepare_sample_selection(sample_indices.as_slice()?).map_err(convert_bgen_error)
    }

    fn clear_prepared_sample_selection(&self) -> PyResult<()> {
        self.reader.clear_prepared_sample_selection().map_err(convert_bgen_error)
    }

    fn reset_profile(&self) {
        self.reader.reset_profile();
    }

    fn profile_snapshot(&self) -> HashMap<String, u64> {
        build_profile_snapshot_dict(&self.reader.profile_snapshot())
    }

    fn validate_trusted_no_missing_diploid(&self) -> PyResult<()> {
        self.reader.validate_trusted_no_missing_diploid().map_err(convert_bgen_error)
    }

    fn variant_metadata_slice(&self, variant_start: usize, variant_stop: usize) -> PyResult<VariantMetadataLists> {
        self.reader.variant_metadata_slice(variant_start, variant_stop).map_err(convert_bgen_error)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn read_dosage_f32<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let sample_index_values = sample_indices.as_slice()?;
        let selected_sample_count = sample_index_values.len();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let dosage_values = py
            .detach(|| self.reader.read_dosage_f32(sample_index_values, variant_start, variant_stop))
            .map_err(convert_bgen_error)?;
        let dosage_matrix = Array2::from_shape_vec((selected_sample_count, selected_variant_count), dosage_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(dosage_matrix.into_pyarray(py))
    }

    fn read_dosage_f32_prepared<'py>(
        &self,
        py: Python<'py>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let dosage_values = py
            .detach(|| self.reader.read_dosage_f32_prepared(variant_start, variant_stop))
            .map_err(convert_bgen_error)?;
        let selected_sample_count =
            if selected_variant_count == 0 { 0 } else { dosage_values.len() / selected_variant_count };
        let dosage_matrix = Array2::from_shape_vec((selected_sample_count, selected_variant_count), dosage_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(dosage_matrix.into_pyarray(py))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn read_dosage_f32_into<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        variant_start: usize,
        variant_stop: usize,
        mut output_array: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let sample_index_values = sample_indices.as_slice()?;
        let selected_sample_count = sample_index_values.len();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let output_shape = output_array.shape();
        if output_shape != [selected_sample_count, selected_variant_count] {
            return Err(PyValueError::new_err(format!(
                "Output array shape mismatch: expected ({selected_sample_count}, {selected_variant_count}), observed ({}, {}).",
                output_shape[0], output_shape[1],
            )));
        }
        if !output_array.is_c_contiguous() {
            return Err(PyValueError::new_err("Output array for BGEN dosage reads must be C-contiguous float32."));
        }

        let output_slice = output_array.as_slice_mut().map_err(|_| {
            PyValueError::new_err("Output array for BGEN dosage reads must expose a contiguous mutable slice.")
        })?;
        let output_pointer_address = output_slice.as_mut_ptr() as usize;
        let output_value_count = output_slice.len();

        py.detach(|| {
            self.reader.read_dosage_f32_into_address(
                sample_index_values,
                variant_start,
                variant_stop,
                output_pointer_address,
                output_value_count,
            )
        })
        .map_err(convert_bgen_error)
    }

    fn read_dosage_f32_into_prepared<'py>(
        &self,
        py: Python<'py>,
        variant_start: usize,
        variant_stop: usize,
        mut output_array: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let output_shape = output_array.shape();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        if output_shape[1] != selected_variant_count {
            return Err(PyValueError::new_err(format!(
                "Output array shape mismatch: expected variant width {selected_variant_count}, observed {}.",
                output_shape[1],
            )));
        }
        if !output_array.is_c_contiguous() {
            return Err(PyValueError::new_err("Output array for BGEN dosage reads must be C-contiguous float32."));
        }

        let output_slice = output_array.as_slice_mut().map_err(|_| {
            PyValueError::new_err("Output array for BGEN dosage reads must expose a contiguous mutable slice.")
        })?;
        let output_pointer_address = output_slice.as_mut_ptr() as usize;
        let output_value_count = output_slice.len();

        py.detach(|| {
            self.reader.read_dosage_f32_into_address_prepared(
                variant_start,
                variant_stop,
                output_pointer_address,
                output_value_count,
            )
        })
        .map_err(convert_bgen_error)
    }

    #[allow(clippy::unused_self)]
    fn close(&self) {}
}

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from g!".to_string()
}

fn validate_supported_layout(combination_count: usize, is_phased: bool) -> PyResult<()> {
    if matches!((combination_count, is_phased), (3, false) | (4, true)) {
        return Ok(());
    }
    Err(PyValueError::new_err(
        "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported.",
    ))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn convert_probability_tensor_to_dosage_f32<'py>(
    py: Python<'py>,
    probability_tensor: PyReadonlyArray3<'py, f32>,
    combination_count: usize,
    is_phased: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    validate_supported_layout(combination_count, is_phased)?;

    let probability_array = probability_tensor.as_array();
    let shape = probability_array.shape();
    let sample_count = shape[0];
    let variant_count = shape[1];
    let observed_combination_count = shape[2];

    if observed_combination_count != combination_count {
        return Err(PyValueError::new_err(format!(
            "Probability tensor combination count mismatch: expected {combination_count}, observed {observed_combination_count}.",
        )));
    }

    let dosage_values = py.detach(move || {
        let mut values = vec![0.0_f32; sample_count * variant_count];
        match (combination_count, is_phased) {
            (3, false) => {
                for sample_index in 0..sample_count {
                    for variant_index in 0..variant_count {
                        let dosage_value = probability_array[[sample_index, variant_index, 1]]
                            + (2.0 * probability_array[[sample_index, variant_index, 2]]);
                        values[(sample_index * variant_count) + variant_index] = dosage_value;
                    }
                }
            }
            (4, true) => {
                for sample_index in 0..sample_count {
                    for variant_index in 0..variant_count {
                        let dosage_value = probability_array[[sample_index, variant_index, 1]]
                            + probability_array[[sample_index, variant_index, 3]];
                        values[(sample_index * variant_count) + variant_index] = dosage_value;
                    }
                }
            }
            _ => unreachable!(),
        }
        values
    });

    let dosage_matrix = Array2::from_shape_vec((sample_count, variant_count), dosage_values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(dosage_matrix.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn convert_probability_matrix_to_dosage_f32<'py>(
    py: Python<'py>,
    probability_matrix: PyReadonlyArray2<'py, f32>,
    combination_count: usize,
    is_phased: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    validate_supported_layout(combination_count, is_phased)?;

    let probability_array = probability_matrix.as_array();
    let shape = probability_array.shape();
    let sample_count = shape[0];
    let observed_combination_count = shape[1];

    if observed_combination_count != combination_count {
        return Err(PyValueError::new_err(format!(
            "Probability matrix combination count mismatch: expected {combination_count}, observed {observed_combination_count}.",
        )));
    }

    let dosage_values = py.detach(move || {
        let mut values = vec![0.0_f32; sample_count];
        match (combination_count, is_phased) {
            (3, false) => {
                for sample_index in 0..sample_count {
                    values[sample_index] =
                        probability_array[[sample_index, 1]] + (2.0 * probability_array[[sample_index, 2]]);
                }
            }
            (4, true) => {
                for sample_index in 0..sample_count {
                    values[sample_index] = probability_array[[sample_index, 1]] + probability_array[[sample_index, 3]];
                }
            }
            _ => unreachable!(),
        }
        values
    });

    Ok(Array1::from_vec(dosage_values).into_pyarray(py))
}

fn convert_bgen_error(error: BgenError) -> PyErr {
    match error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            PyValueError::new_err(message)
        }
        BgenError::Io(io_error) => PyRuntimeError::new_err(io_error.to_string()),
    }
}

fn build_profile_snapshot_dict(profile_snapshot: &ReaderProfileSnapshot) -> HashMap<String, u64> {
    HashMap::from([
        ("sample_selection_prepare_ns".to_string(), profile_snapshot.sample_selection_prepare_ns),
        ("sample_selection_prepare_count".to_string(), profile_snapshot.sample_selection_prepare_count),
        ("compressed_block_fetch_ns".to_string(), profile_snapshot.compressed_block_fetch_ns),
        ("compressed_block_fetch_count".to_string(), profile_snapshot.compressed_block_fetch_count),
        ("compressed_byte_count".to_string(), profile_snapshot.compressed_byte_count),
        ("decompression_ns".to_string(), profile_snapshot.decompression_ns),
        ("decompression_count".to_string(), profile_snapshot.decompression_count),
        ("uncompressed_byte_count".to_string(), profile_snapshot.uncompressed_byte_count),
        ("zlib_stream_count".to_string(), profile_snapshot.zlib_stream_count),
        ("probability_decode_ns".to_string(), profile_snapshot.probability_decode_ns),
        ("probability_decode_count".to_string(), profile_snapshot.probability_decode_count),
        ("variant_decode_count".to_string(), profile_snapshot.variant_decode_count),
        ("output_write_ns".to_string(), profile_snapshot.output_write_ns),
        ("output_write_count".to_string(), profile_snapshot.output_write_count),
        ("output_byte_count".to_string(), profile_snapshot.output_byte_count),
        ("decode_tile_count".to_string(), profile_snapshot.decode_tile_count),
        ("selected_sample_count".to_string(), profile_snapshot.selected_sample_count),
        ("metadata_slice_ns".to_string(), profile_snapshot.metadata_slice_ns),
        ("metadata_slice_count".to_string(), profile_snapshot.metadata_slice_count),
    ])
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBgenReader>()?;
    module.add_class::<PyOutputWriterSession>()?;
    module.add_function(wrap_pyfunction!(finalize_output_run_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(hello_from_bin, module)?)?;
    module.add_function(wrap_pyfunction!(convert_probability_tensor_to_dosage_f32, module)?)?;
    module.add_function(wrap_pyfunction!(convert_probability_matrix_to_dosage_f32, module)?)?;
    Ok(())
}
