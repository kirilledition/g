//! PyO3 registration boundary for `_core.cli`.

mod driver;

use pyo3::prelude::*;

pub(crate) use driver::NativeCliRunResult;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunResult>()?;
    module.add_function(wrap_pyfunction!(driver::run, module)?)?;
    Ok(())
}
