#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use std::collections::HashMap;
use std::path::Path;

use g_engine::Regenie2RunEngineCore;
use g_genotype::common::ChunkSpec as NativeChunkSpec;
use g_input::sample::{self, AlignmentInputs, MultiAlignmentInputs};
use g_runtime::trusted_validation as native_trusted_validation;
use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyReadwriteArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::errors::{convert_bgen_error, convert_genotype_error};
use super::genotype::{
    ChunkStats, VariantMetadata, VariantMetadataTuple, build_committed_identifier_set,
    convert_variant_metadata_columns_to_tuple,
};
use super::profile::build_profile_snapshot_dict;
use super::sample_alignment::{
    NativeAlignedSampleData, NativeGroupedAlignedSampleData, NativeMultiAlignedSampleData, parse_sample_key_mode,
};

#[pyclass]
struct Regenie2RunEngine {
    engine: Regenie2RunEngineCore,
}

#[pymethods]
impl Regenie2RunEngine {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (bgen_path, chunk_size, variant_limit=None, trusted_no_missing_diploid=false))]
    fn new(
        py: Python<'_>,
        bgen_path: String,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
    ) -> PyResult<Self> {
        let engine = py
            .detach(|| {
                Regenie2RunEngineCore::open_bgen(
                    Path::new(&bgen_path),
                    chunk_size,
                    variant_limit,
                    trusted_no_missing_diploid,
                )
            })
            .map_err(|error| convert_bgen_error("open_bgen", error))?;
        Ok(Self { engine })
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.engine.reader().sample_count()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.engine.reader().variant_count()
    }

    #[getter]
    fn contains_embedded_samples(&self) -> bool {
        self.engine.reader().contains_embedded_samples()
    }

    fn sample_identifiers(&self) -> Vec<String> {
        self.engine.reader().sample_identifiers()
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_name,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_name: String,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    sample::align_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_name,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = AlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || sample::align_sample_data(inputs))
            .map(NativeAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_names,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_multi_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_names: Vec<String>,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeMultiAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    sample::align_multi_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_names,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeMultiAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = MultiAlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || sample::align_multi_sample_data(inputs))
            .map(NativeMultiAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_names,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_grouped_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_names: Vec<String>,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeGroupedAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    sample::align_grouped_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_names,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeGroupedAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = MultiAlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || sample::align_grouped_sample_data(&inputs))
            .map(NativeGroupedAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.engine.reader().chromosome_boundary_indices()
    }

    fn variant_metadata_slice(
        &self,
        py: Python<'_>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<VariantMetadataTuple> {
        py.detach(|| self.engine.reader().variant_metadata_slice(variant_start, variant_stop))
            .map(convert_variant_metadata_columns_to_tuple)
            .map_err(|error| convert_bgen_error("read_variant_metadata_slice", error))
    }

    #[pyo3(signature = (variant_limit=None))]
    fn required_chromosomes(&self, variant_limit: Option<usize>) -> PyResult<Vec<String>> {
        self.engine.required_chromosomes(variant_limit).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn reset_profile(&self) {
        self.engine.reader().reset_profile();
    }

    fn profile_snapshot(&self) -> HashMap<String, u64> {
        build_profile_snapshot_dict(&self.engine.reader().profile_snapshot())
    }

    fn validate_trusted_no_missing_diploid(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.engine.reader().validate_trusted_no_missing_diploid())
            .map_err(|error| convert_bgen_error("validate_trusted_no_missing_diploid", error))
    }

    fn mark_trusted_no_missing_diploid_validated(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.engine.reader().mark_trusted_no_missing_diploid_validated())
            .map_err(|error| convert_bgen_error("mark_trusted_no_missing_diploid_validated", error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn validate_trusted_no_missing_diploid_with_default_cache(
        &self,
        py: Python<'_>,
        bgen_path: String,
        validation_mode: String,
    ) -> PyResult<()> {
        let cache_directory = py
            .detach(native_trusted_validation::default_trusted_bgen_validation_cache_directory)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        self.validate_trusted_no_missing_diploid_with_cache_directory(
            py,
            &bgen_path,
            &validation_mode,
            cache_directory.as_path(),
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None, callback_batch_size=1))]
    fn run_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None, callback_batch_size=1))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None, callback_batch_size=1))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeMultiAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None))]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeMultiAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }
}

fn flush_variant_major_dosage_batch<'py>(
    compute_dosage_chunk_batch_method: &Bound<'py, PyAny>,
    metadata_batch: &mut Vec<Py<VariantMetadata>>,
    output_array_batch: &mut Vec<Py<PyAny>>,
    stats_batch: &mut Vec<Py<ChunkStats>>,
) -> PyResult<()> {
    if metadata_batch.is_empty() {
        return Ok(());
    }
    let metadata_values = std::mem::take(metadata_batch);
    let output_array_values = std::mem::take(output_array_batch);
    let stats_values = std::mem::take(stats_batch);
    compute_dosage_chunk_batch_method.call1((metadata_values, output_array_values, stats_values))?;
    Ok(())
}

impl Regenie2RunEngine {
    fn validate_trusted_no_missing_diploid_with_cache_directory(
        &self,
        py: Python<'_>,
        bgen_path: &str,
        validation_mode: &str,
        cache_directory: &Path,
    ) -> PyResult<()> {
        native_trusted_validation::require_cache_backed_trusted_bgen_validation_mode(validation_mode)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let sample_count = i64::try_from(self.engine.reader().sample_count())
            .map_err(|_| PyValueError::new_err("BGEN sample count exceeds the native validation cache range."))?;
        let variant_count = i64::try_from(self.engine.reader().variant_count())
            .map_err(|_| PyValueError::new_err("BGEN variant count exceeds the native validation cache range."))?;
        let fingerprint = py
            .detach(|| {
                native_trusted_validation::build_trusted_bgen_validation_fingerprint(
                    &native_trusted_validation::TrustedBgenValidationFingerprintInput {
                        bgen_path: bgen_path.into(),
                        sample_count,
                        variant_count,
                        trusted_no_missing_diploid: true,
                    },
                )
            })
            .map_err(PyOSError::new_err)?;
        let cache_path =
            native_trusted_validation::build_trusted_bgen_validation_cache_path(cache_directory, &fingerprint);
        let cache_lookup_plan = py
            .detach(|| {
                native_trusted_validation::plan_trusted_bgen_validation_cache_lookup(validation_mode, &cache_path)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if cache_lookup_plan.should_mark_validated {
            py.detach(|| self.engine.reader().mark_trusted_no_missing_diploid_validated())
                .map_err(|error| convert_bgen_error("mark_trusted_no_missing_diploid_validated", error))?;
        }
        if !cache_lookup_plan.should_validate {
            return Ok(());
        }
        py.detach(|| self.engine.reader().validate_trusted_no_missing_diploid())
            .map_err(|error| convert_bgen_error("validate_trusted_no_missing_diploid", error))?;
        if cache_lookup_plan.should_write_cache {
            py.detach(|| {
                native_trusted_validation::write_trusted_bgen_validation_cache_payload(
                    &cache_path,
                    fingerprint,
                    Path::new(&bgen_path),
                    sample_count,
                    variant_count,
                )
            })
            .map_err(PyOSError::new_err)?;
        }
        Ok(())
    }

    fn run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        if callback_batch_size == 0 {
            return Err(PyValueError::new_err("callback_batch_size must be positive."));
        }
        py.detach(|| self.engine.reader().prepare_sample_selection(sample_index_values))
            .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

        let run_result = self.run_prepared_bgen_variant_major_dosage_buffered_chunks(
            py,
            sample_index_values.len(),
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        );
        let clear_result = py
            .detach(|| self.engine.reader().clear_prepared_sample_selection())
            .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        py.detach(|| self.engine.reader().prepare_sample_selection(sample_index_values))
            .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

        let run_result = self.run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks(
            py,
            sample_index_values.len(),
            callback,
            committed_chunk_identifiers,
        );
        let clear_result = py
            .detach(|| self.engine.reader().clear_prepared_sample_selection())
            .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    fn run_prepared_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self
            .engine
            .plan_chunks(&committed_identifier_set)
            .map_err(|error| convert_genotype_error("plan_chunks", error))?;
        let acquire_dosage_buffer_method = callback.getattr("acquire_variant_major_dosage_buffer")?;
        if callback_batch_size > 1 {
            return self.run_prepared_bgen_variant_major_dosage_buffered_chunk_batches(
                py,
                selected_sample_count,
                callback,
                &chunk_specs,
                &acquire_dosage_buffer_method,
                callback_batch_size,
            );
        }
        let chunk_batch_plan = g_engine::plan_chunk_batches(&chunk_specs, callback_batch_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let processed_chunk_count = chunk_batch_plan.chunk_count();
        let compute_dosage_chunk_method = callback.getattr("compute_preprocessed_variant_major_dosage_chunk")?;
        for chunk_batch in chunk_batch_plan.into_chunk_batches() {
            for chunk_spec in &chunk_batch {
                py.check_signals()?;
                let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                let output_array_object =
                    acquire_dosage_buffer_method.call1((selected_variant_count, selected_sample_count))?;
                let stats = {
                    let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                    let output_shape = output_array.shape();
                    if output_shape != [selected_variant_count, selected_sample_count] {
                        return Err(PyValueError::new_err(format!(
                            "Reusable variant-major BGEN dosage buffer shape mismatch: expected ({selected_variant_count}, {}), observed ({}, {}).",
                            selected_sample_count, output_shape[0], output_shape[1],
                        )));
                    }
                    if !output_array.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must be C-contiguous float32.",
                        ));
                    }
                    let output_slice = output_array.as_slice_mut().map_err(|_| {
                        PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must expose a contiguous mutable slice.",
                        )
                    })?;
                    let output_pointer_address = output_slice.as_mut_ptr() as usize;
                    let output_value_count = output_slice.len();
                    let chunk_stats = py
                        .detach(|| {
                            self.engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                                chunk_spec.variant_start_index,
                                chunk_spec.variant_stop_index,
                                output_pointer_address,
                                output_value_count,
                            )
                        })
                        .map_err(|error| {
                            convert_bgen_error(
                                "read_preprocessed_variant_major_dosage_f32_into_address_prepared",
                                error,
                            )
                        })?;
                    Py::new(py, ChunkStats::new(chunk_stats))?
                };
                let variant_start_index = chunk_spec.variant_start_index;
                let variant_stop_index = chunk_spec.variant_stop_index;
                let metadata_columns = py
                    .detach(|| self.engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                    .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
                let metadata =
                    Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
                compute_dosage_chunk_method.call1((metadata, output_array_object, stats))?;
            }
        }
        Ok(processed_chunk_count)
    }

    fn run_prepared_bgen_variant_major_dosage_buffered_chunk_batches<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        chunk_specs: &[NativeChunkSpec],
        acquire_dosage_buffer_method: &Bound<'py, PyAny>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        let compute_dosage_chunk_batch_method =
            callback.getattr("compute_preprocessed_variant_major_dosage_chunk_batch")?;
        let chunk_batch_plan = g_engine::plan_chunk_batches(chunk_specs, callback_batch_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let processed_chunk_count = chunk_batch_plan.chunk_count();
        let mut metadata_batch: Vec<Py<VariantMetadata>> = Vec::with_capacity(callback_batch_size);
        let mut output_array_batch: Vec<Py<PyAny>> = Vec::with_capacity(callback_batch_size);
        let mut stats_batch: Vec<Py<ChunkStats>> = Vec::with_capacity(callback_batch_size);
        for chunk_batch in chunk_batch_plan.into_chunk_batches() {
            for chunk_spec in &chunk_batch {
                py.check_signals()?;
                let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                let output_array_object =
                    acquire_dosage_buffer_method.call1((selected_variant_count, selected_sample_count))?;
                let stats = {
                    let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                    let output_shape = output_array.shape();
                    if output_shape != [selected_variant_count, selected_sample_count] {
                        return Err(PyValueError::new_err(format!(
                            "Reusable variant-major BGEN dosage buffer shape mismatch: expected ({selected_variant_count}, {}), observed ({}, {}).",
                            selected_sample_count, output_shape[0], output_shape[1],
                        )));
                    }
                    if !output_array.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must be C-contiguous float32.",
                        ));
                    }
                    let output_slice = output_array.as_slice_mut().map_err(|_| {
                        PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must expose a contiguous mutable slice.",
                        )
                    })?;
                    let output_pointer_address = output_slice.as_mut_ptr() as usize;
                    let output_value_count = output_slice.len();
                    let chunk_stats = py
                        .detach(|| {
                            self.engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                                chunk_spec.variant_start_index,
                                chunk_spec.variant_stop_index,
                                output_pointer_address,
                                output_value_count,
                            )
                        })
                        .map_err(|error| {
                            convert_bgen_error(
                                "read_preprocessed_variant_major_dosage_f32_into_address_prepared",
                                error,
                            )
                        })?;
                    Py::new(py, ChunkStats::new(chunk_stats))?
                };
                let variant_start_index = chunk_spec.variant_start_index;
                let variant_stop_index = chunk_spec.variant_stop_index;
                let metadata_columns = py
                    .detach(|| self.engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                    .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
                let metadata =
                    Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
                metadata_batch.push(metadata);
                output_array_batch.push(output_array_object.unbind());
                stats_batch.push(stats);
            }
            flush_variant_major_dosage_batch(
                &compute_dosage_chunk_batch_method,
                &mut metadata_batch,
                &mut output_array_batch,
                &mut stats_batch,
            )?;
        }
        Ok(processed_chunk_count)
    }

    fn run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self
            .engine
            .plan_chunks(&committed_identifier_set)
            .map_err(|error| convert_genotype_error("plan_chunks", error))?;
        let chunk_batch_plan =
            g_engine::plan_chunk_batches(&chunk_specs, 1).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let processed_chunk_count = chunk_batch_plan.chunk_count();
        let acquire_packed_buffer_method = callback.getattr("acquire_variant_major_packed8_probability_pair_buffer")?;
        let compute_packed_chunk_method =
            callback.getattr("compute_preprocessed_variant_major_packed8_probability_pair_chunk")?;
        for chunk_batch in chunk_batch_plan.into_chunk_batches() {
            for chunk_spec in &chunk_batch {
                py.check_signals()?;
                let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                let output_array_object =
                    acquire_packed_buffer_method.call1((selected_variant_count, selected_sample_count))?;
                let stats = {
                    let mut output_array = output_array_object.extract::<PyReadwriteArray3<'_, u8>>()?;
                    let output_shape = output_array.shape();
                    if output_shape != [selected_variant_count, selected_sample_count, 2] {
                        return Err(PyValueError::new_err(format!(
                            "Reusable variant-major BGEN packed8 probability-pair buffer shape mismatch: expected ({selected_variant_count}, {}, 2), observed ({}, {}, {}).",
                            selected_sample_count, output_shape[0], output_shape[1], output_shape[2],
                        )));
                    }
                    if !output_array.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "Reusable variant-major BGEN packed8 probability-pair buffer must be C-contiguous uint8.",
                        ));
                    }
                    let output_slice = output_array.as_slice_mut().map_err(|_| {
                        PyValueError::new_err(
                            "Reusable variant-major BGEN packed8 probability-pair buffer must expose a contiguous mutable slice.",
                        )
                    })?;
                    let output_pointer_address = output_slice.as_mut_ptr() as usize;
                    let output_value_count = output_slice.len();
                    let chunk_stats = py
                        .detach(|| {
                            self.engine
                                .reader()
                                .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                                    chunk_spec.variant_start_index,
                                    chunk_spec.variant_stop_index,
                                    output_pointer_address,
                                    output_value_count,
                                )
                        })
                        .map_err(|error| {
                            convert_bgen_error(
                                "read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared",
                                error,
                            )
                        })?;
                    Py::new(py, ChunkStats::new(chunk_stats))?
                };
                let variant_start_index = chunk_spec.variant_start_index;
                let variant_stop_index = chunk_spec.variant_stop_index;
                let metadata_columns = py
                    .detach(|| self.engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                    .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
                let metadata =
                    Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
                compute_packed_chunk_method.call1((metadata, output_array_object, stats))?;
            }
        }
        Ok(processed_chunk_count)
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Regenie2RunEngine>()?;
    Ok(())
}
