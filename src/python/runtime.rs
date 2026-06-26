use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_genotype::bgen::set_bgen_decode_tile_variant_count;

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
    if thread_count == 0 {
        return Err(PyValueError::new_err("Rayon thread count must be positive."));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
