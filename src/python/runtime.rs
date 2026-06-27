use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_genotype::bgen::set_bgen_decode_tile_variant_count;
use g_runtime::{RayonRuntimeError, configure_global_rayon_thread_pool};

use super::errors;

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
pub(super) fn configure_bgen_decode_tile_variant_count(tile_variant_count: usize) -> PyResult<()> {
    set_bgen_decode_tile_variant_count(tile_variant_count)
        .map_err(|error| errors::convert_bgen_error("configure_bgen_decode_tile_variant_count", error))
}

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
pub(super) fn configure_rayon_global_thread_pool(thread_count: usize) -> PyResult<()> {
    configure_global_rayon_thread_pool(thread_count).map_err(|error| rayon_runtime_error_to_py(&error))
}

fn rayon_runtime_error_to_py(error: &RayonRuntimeError) -> PyErr {
    match error {
        RayonRuntimeError::InvalidThreadCount => PyValueError::new_err(error.to_string()),
        RayonRuntimeError::GlobalThreadPool { .. } => PyRuntimeError::new_err(error.to_string()),
    }
}
