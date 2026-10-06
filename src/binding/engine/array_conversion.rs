//! Checked NumPy conversion and immutable native-backed array ownership.

use std::sync::Arc;

use numpy::ndarray::{Array2, ArrayView, ArrayView1, ArrayView2, ArrayView3, Ix1, Ix2};
use numpy::{
    Element, IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use g_engine as native_engine;
use g_genotype as native_genotype;
use g_output as native_output;

use super::PyJaxBackendError;

#[pyclass(frozen)]
struct Packed8ArrayOwner {
    values: native_genotype::PooledPacked8Buffer,
}

#[pyclass(frozen)]
struct CompressedPacked8BatchOwner {
    batch: native_genotype::CompressedPacked8Batch,
}

pub(super) struct CompressedBatchArrays<'py> {
    pub(super) slab: Bound<'py, PyArray1<u8>>,
    pub(super) metadata: Bound<'py, PyArray2<u32>>,
}

#[pyclass(frozen)]
struct SampleSelectionArrayOwner {
    file_indices: Arc<[u32]>,
}

pub(super) fn into_python_sample_selection(
    py: Python<'_>,
    file_indices: Arc<[u32]>,
) -> PyResult<Bound<'_, PyArray1<u32>>> {
    let owner = Bound::new(py, SampleSelectionArrayOwner { file_indices })?;
    let selection_view = ArrayView1::from(&owner.get().file_indices[..]);
    let selected_sample_indices = unsafe {
        // The frozen private owner retains immutable Arc storage as the
        // ndarray base until Python finishes its one group-level upload.
        PyArray1::borrow_from_array(&selection_view, owner.clone().into_any())
    };
    selected_sample_indices.readwrite().make_nonwriteable();
    Ok(selected_sample_indices)
}

pub(super) fn compressed_batch_arrays(
    py: Python<'_>,
    batch: native_genotype::CompressedPacked8Batch,
    logical_variant_count: usize,
) -> Result<CompressedBatchArrays<'_>, PyJaxBackendError> {
    let owner = Bound::new(py, CompressedPacked8BatchOwner { batch }).map_err(PyJaxBackendError::Python)?;
    let slab_view = ArrayView1::from(owner.get().batch.raw_deflate_slab());
    let slab = unsafe {
        // Both immutable arrays retain the frozen owner, preventing pooled
        // storage reuse before asynchronous device transfer consumes it.
        PyArray1::borrow_from_array(&slab_view, owner.clone().into_any())
    };
    slab.readwrite().make_nonwriteable();
    let metadata_view = ArrayView2::from_shape((logical_variant_count, 3), owner.get().batch.member_metadata())
        .map_err(|error| {
            PyJaxBackendError::InvalidInput(format!("invalid compressed packed8 metadata shape: {error}"))
        })?;
    let metadata = unsafe {
        // The same owner retains the metadata allocation until its final view
        // is released. Neither view permits Python-side mutation.
        PyArray2::borrow_from_array(&metadata_view, owner.clone().into_any())
    };
    metadata.readwrite().make_nonwriteable();
    Ok(CompressedBatchArrays { slab, metadata })
}

pub(super) fn into_python_genotype_batch(
    py: Python<'_>,
    genotypes: native_genotype::OwnedGenotypeBuffer,
    variant_count: usize,
    sample_count: usize,
) -> PyResult<Py<PyAny>> {
    match genotypes {
        native_genotype::OwnedGenotypeBuffer::Dosage(values) => {
            Ok(Array2::from_shape_vec((variant_count, sample_count), values)
                .expect("engine-validated dosage matrix shape")
                .into_pyarray(py)
                .into_any()
                .unbind())
        }
        native_genotype::OwnedGenotypeBuffer::Packed8(values) => {
            let owner = Bound::new(py, Packed8ArrayOwner { values })?;
            let array_view = ArrayView3::from_shape((variant_count, sample_count, 2), &owner.get().values[..])
                .map_err(|error| PyValueError::new_err(format!("Invalid packed8 genotype shape: {error}")))?;
            let values = unsafe {
                // The frozen private owner never mutates or reallocates its buffer. The ndarray
                // receives an owned reference to that owner as its base, so the pooled allocation
                // cannot be returned until the final ndarray reference is dropped.
                PyArray3::borrow_from_array(&array_view, owner.clone().into_any())
            };
            values.readwrite().make_nonwriteable();
            Ok(values.into_any().unbind())
        }
    }
}

pub(super) fn parse_host_materialized_batch(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    output_statistics: Option<g_genotype_contracts::ChunkOutputStatistics>,
    logical_variant_count: usize,
) -> PyResult<native_engine::MaterializedAssociationBatch> {
    let association_payload = payload.getattr("association")?;
    let association = parse_host_association_batch(py, &association_payload, logical_variant_count)?;
    let raw_statistics_payload = payload.getattr("raw_packed8_statistics")?;
    let genotype_statistics = match (output_statistics, raw_statistics_payload.is_none()) {
        (Some(statistics), true) => native_engine::MaterializedGenotypeStatistics::Ready(statistics),
        (Some(_), false) => {
            return Err(PyValueError::new_err(
                "Host-decoded association output unexpectedly included packed8 raw statistics.",
            ));
        }
        (None, true) => {
            return Err(PyValueError::new_err(
                "Compressed packed8 association output omitted its raw genotype statistics.",
            ));
        }
        (None, false) => native_engine::MaterializedGenotypeStatistics::Packed8Raw(parse_packed8_raw_statistics(
            py,
            &raw_statistics_payload,
            logical_variant_count,
        )?),
    };
    Ok(native_engine::MaterializedAssociationBatch { association, genotype_statistics })
}

fn parse_packed8_raw_statistics(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    logical_variant_count: usize,
) -> PyResult<native_genotype::Packed8RawStatistics> {
    let dosage_sums =
        parse_host_vector::<u64>(py, &payload.getattr("dosage_sums")?, "dosage_sums", logical_variant_count)?;
    let dosage_square_sums = parse_host_vector::<u64>(
        py,
        &payload.getattr("dosage_square_sums")?,
        "dosage_square_sums",
        logical_variant_count,
    )?;
    let statuses = parse_host_vector::<u32>(py, &payload.getattr("statuses")?, "statuses", logical_variant_count)?;
    let selected_sample_count = payload.getattr("selected_sample_count")?.extract::<usize>()?;
    Ok(native_genotype::Packed8RawStatistics { dosage_sums, dosage_square_sums, statuses, selected_sample_count })
}

fn parse_host_vector<ElementType: Element + Copy>(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    label: &str,
    expected_value_count: usize,
) -> PyResult<Vec<ElementType>> {
    let values = payload.cast::<PyUntypedArray>()?;
    if !values.dtype().is_equiv_to(&dtype::<ElementType>(py)) {
        return Err(PyValueError::new_err(format!("{label} must use {} dtype.", dtype::<ElementType>(py))));
    }
    let values = values.cast::<PyArray<ElementType, Ix1>>()?.readonly();
    if values.shape() != [expected_value_count] {
        return Err(PyValueError::new_err(format!(
            "{label} shape {:?} does not match logical variant count {expected_value_count}.",
            values.shape()
        )));
    }
    Ok(copy_array_values(&values.as_array()))
}

fn parse_host_association_batch(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    logical_variant_count: usize,
) -> PyResult<native_output::Regenie2StatisticBatch> {
    let beta_object = payload.getattr("beta")?;
    let standard_error_object = payload.getattr("standard_error")?;
    let chi_squared_object = payload.getattr("chi_squared")?;
    let log10_p_value_object = payload.getattr("log10_p_value")?;
    let beta = beta_object.cast::<PyUntypedArray>()?;
    let standard_error = standard_error_object.cast::<PyUntypedArray>()?;
    let chi_squared = chi_squared_object.cast::<PyUntypedArray>()?;
    let log10_p_value = log10_p_value_object.cast::<PyUntypedArray>()?;
    let observed_dtype = beta.dtype();
    for (label, values) in
        [("standard_error", standard_error), ("chi_squared", chi_squared), ("log10_p_value", log10_p_value)]
    {
        if !values.dtype().is_equiv_to(&observed_dtype) {
            return Err(PyValueError::new_err(format!("{label} dtype must match beta dtype.")));
        }
    }
    if !observed_dtype.is_equiv_to(&dtype::<f32>(py)) {
        return Err(PyValueError::new_err("Host association statistics must use float32 dtype."));
    }
    let beta = beta.cast::<PyArray<f32, Ix2>>()?.readonly();
    let standard_error = standard_error.cast::<PyArray<f32, Ix2>>()?.readonly();
    let chi_squared = chi_squared.cast::<PyArray<f32, Ix2>>()?.readonly();
    let log10_p_value = log10_p_value.cast::<PyArray<f32, Ix2>>()?.readonly();
    let expected_shape = beta.shape();
    for (label, observed_shape) in [
        ("standard_error", standard_error.shape()),
        ("chi_squared", chi_squared.shape()),
        ("log10_p_value", log10_p_value.shape()),
    ] {
        if observed_shape != expected_shape {
            return Err(PyValueError::new_err(format!(
                "{label} shape {observed_shape:?} does not match beta shape {expected_shape:?}."
            )));
        }
    }
    let (trait_count, materialized_variant_count) = (expected_shape[0], expected_shape[1]);
    if logical_variant_count != materialized_variant_count {
        return Err(PyValueError::new_err(format!(
            "materialized variant count {materialized_variant_count} does not match logical variant count {logical_variant_count}."
        )));
    }
    let correction_code_object = payload.getattr("correction_code")?;
    let correction_code = if correction_code_object.is_none() {
        None
    } else {
        Some(parse_correction_codes(
            py,
            correction_code_object.cast::<PyUntypedArray>()?,
            trait_count,
            logical_variant_count,
        )?)
    };
    Ok(native_output::Regenie2StatisticBatch {
        trait_count,
        variant_count: logical_variant_count,
        beta: copy_array_values(&beta.as_array()),
        standard_error: copy_array_values(&standard_error.as_array()),
        chi_squared: copy_array_values(&chi_squared.as_array()),
        log10_p_value: copy_array_values(&log10_p_value.as_array()),
        correction_code,
    })
}

fn copy_array_values<ElementType: Element + Copy, Dimension: numpy::ndarray::Dimension>(
    values: &ArrayView<'_, ElementType, Dimension>,
) -> Vec<ElementType> {
    // JAX may return a read-only NumPy view of device-owned CPU storage. The
    // asynchronous writer needs an independent allocation, not merely a live
    // Python reference or a NumPy writeability flag. Arrow subsequently takes
    // ownership of this Vec without another statistic-value copy.
    match values.as_slice() {
        Some(contiguous_values) => contiguous_values.to_vec(),
        None => values.iter().copied().collect(),
    }
}

fn parse_correction_codes(
    py: Python<'_>,
    values: &Bound<'_, PyUntypedArray>,
    trait_count: usize,
    logical_variant_count: usize,
) -> PyResult<Vec<u8>> {
    if !values.dtype().is_equiv_to(&dtype::<u8>(py)) {
        return Err(PyValueError::new_err("correction_code must use uint8 dtype."));
    }
    let values = values.cast::<PyArray<u8, Ix2>>()?.readonly();
    if values.shape() != [trait_count, logical_variant_count] {
        return Err(PyValueError::new_err(format!(
            "correction_code shape {:?} does not match statistic shape ({trait_count}, {logical_variant_count}).",
            values.shape()
        )));
    }
    Ok(copy_array_values(&values.as_array()))
}

#[cfg(test)]
mod tests {
    use numpy::ndarray::{Array2, ShapeBuilder, s};

    use super::copy_array_values;

    #[test]
    fn contiguous_statistics_keep_trait_major_order() {
        let statistics = Array2::from_shape_vec((2, 3), vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("statistic dimensions match storage");
        assert_eq!(copy_array_values(&statistics.view()), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn fortran_contiguous_statistics_keep_trait_major_order() {
        let statistics = Array2::from_shape_vec((2, 3).f(), vec![0.0_f32, 3.0, 1.0, 4.0, 2.0, 5.0])
            .expect("column-major statistic dimensions match storage");
        assert!(statistics.as_slice_memory_order().is_some());
        assert!(statistics.as_slice().is_none());
        assert_eq!(copy_array_values(&statistics.view()), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn strided_statistics_keep_trait_major_order() {
        let statistics =
            Array2::from_shape_vec((2, 6), vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
                .expect("statistic dimensions match storage");
        assert_eq!(copy_array_values(&statistics.slice(s![.., ..;2])), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn materialized_statistics_own_storage_after_source_mutation() {
        let mut statistics = Array2::from_shape_vec((2, 3), vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("statistic dimensions match storage");
        let owned_statistics = copy_array_values(&statistics.view());
        statistics.fill(99.0);
        drop(statistics);
        assert_eq!(owned_statistics, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn strided_correction_codes_keep_trait_major_order() {
        let codes = Array2::from_shape_vec((2, 4), vec![0_u8, 1, 2, 3, 4, 5, 6, 7])
            .expect("correction dimensions match storage");
        assert_eq!(copy_array_values(&codes.slice(s![.., ..;2])), vec![0, 2, 4, 6]);
    }
}
