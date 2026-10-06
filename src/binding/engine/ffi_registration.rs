//! Runtime compatibility and process-lifetime XLA FFI registration.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
#[cfg(target_os = "linux")]
use pyo3::types::PyCapsule;
use pyo3::types::{PyDict, PyModule};

use crate::binding::cli::python_interruption_signal_name;

static NVCOMP_FFI_REGISTRATION: PyOnceLock<()> = PyOnceLock::new();
static FIRTH_COMPONENTS_FFI_REGISTRATION: PyOnceLock<bool> = PyOnceLock::new();
const SUPPORTED_JAX_VERSION: &str = "0.11.0";
const SUPPORTED_JAXLIB_VERSION: &str = "0.11.0";

pub(super) fn validate_jax_runtime_versions(py: Python<'_>) -> PyResult<()> {
    let jax_version = PyModule::import(py, "jax")?.getattr("__version__")?.extract::<String>()?;
    let jaxlib_version = PyModule::import(py, "jaxlib")?.getattr("__version__")?.extract::<String>()?;
    if let Some(message) = jax_runtime_version_error(&jax_version, &jaxlib_version) {
        return Err(PyRuntimeError::new_err(message));
    }
    Ok(())
}

fn jax_runtime_version_error(jax_version: &str, jaxlib_version: &str) -> Option<String> {
    if jax_version == SUPPORTED_JAX_VERSION && jaxlib_version == SUPPORTED_JAXLIB_VERSION {
        return None;
    }
    Some(format!(
        "Unsupported JAX runtime: g requires jax=={SUPPORTED_JAX_VERSION} and jaxlib=={SUPPORTED_JAXLIB_VERSION} because its native XLA FFI handlers are built against headers from that jaxlib release; observed jax=={jax_version} and jaxlib=={jaxlib_version}. Recreate the environment with `uv sync --frozen` before running g."
    ))
}

pub(super) fn register_nvcomp_ffi_target(py: Python<'_>) -> PyResult<()> {
    NVCOMP_FFI_REGISTRATION.get_or_try_init(py, || register_nvcomp_ffi_target_once(py)).copied()
}

fn contextual_backend_error(py: Python<'_>, error: PyErr, context: &str) -> PyErr {
    if python_interruption_signal_name(py, &error).is_some() {
        error
    } else {
        PyRuntimeError::new_err(format!("{context}: {error}"))
    }
}

#[cfg(target_os = "linux")]
fn register_nvcomp_ffi_target_once(py: Python<'_>) -> PyResult<()> {
    let nvcomp_module = PyModule::import(py, "nvidia.libnvcomp").map_err(|error| {
        contextual_backend_error(py, error, "GPU packed8 delivery requires the official nvidia-libnvcomp-cu12 package")
    })?;
    let loaded_library = nvcomp_module.call_method0("load_library").map_err(|error| {
        contextual_backend_error(py, error, "The official nvidia.libnvcomp loader failed to load libnvcomp.so.5")
    })?;
    if loaded_library.is_none() {
        return Err(PyRuntimeError::new_err("The official nvidia.libnvcomp loader could not find libnvcomp.so.5."));
    }

    let capability = g_genotype_cuda::initialize_nvcomp_runtime(0)
        .map_err(|error| PyRuntimeError::new_err(format!("nvCOMP runtime initialization failed: {error}")))?;
    let keyword_arguments = PyDict::new(py);
    keyword_arguments.set_item("platform", "CUDA")?;
    keyword_arguments.set_item("api_version", 1)?;
    let jax_ffi = PyModule::import(py, "jax")?.getattr("ffi")?;
    for (target, handler) in [
        (g_genotype_cuda::PACKED8_DEFLATE_FFI_TARGET, g_genotype_cuda::packed8_deflate_ffi_handler(&capability)),
        (
            g_genotype_cuda::PACKED8_SOURCE_DEFLATE_FFI_TARGET,
            g_genotype_cuda::packed8_source_deflate_ffi_handler(&capability),
        ),
    ] {
        // SAFETY: `handler` is the process-lifetime address of a linked typed-XLA
        // FFI handler, and the capsule has no destructor or borrowed storage.
        let capsule = unsafe { PyCapsule::new_with_pointer(py, handler, c"xla._CUSTOM_CALL_TARGET")? };
        jax_ffi
            .call_method("register_ffi_target", (target, capsule), Some(&keyword_arguments))
            .map_err(|error| contextual_backend_error(py, error, "JAX nvCOMP FFI target registration failed"))?;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn register_nvcomp_ffi_target_once(_py: Python<'_>) -> PyResult<()> {
    Err(PyRuntimeError::new_err("GPU packed8 delivery through nvCOMP is supported only on Linux."))
}

pub(super) fn register_firth_components_ffi_target(py: Python<'_>) -> PyResult<bool> {
    FIRTH_COMPONENTS_FFI_REGISTRATION
        .get_or_try_init(py, || optional_ffi_registration_result(py, register_firth_components_ffi_target_once(py)))
        .copied()
}

fn optional_ffi_registration_result(py: Python<'_>, result: PyResult<bool>) -> PyResult<bool> {
    result.or_else(|error| if python_interruption_signal_name(py, &error).is_some() { Err(error) } else { Ok(false) })
}

#[cfg(target_os = "linux")]
fn register_firth_components_ffi_target_once(py: Python<'_>) -> PyResult<bool> {
    let Ok(capability) = g_compute_cuda::initialize_firth_components_runtime(0) else {
        return Ok(false);
    };
    let handler = g_compute_cuda::firth_components_ffi_handler(&capability);
    // SAFETY: The linked typed-XLA FFI handler has process lifetime, and the
    // capsule has no destructor or borrowed storage.
    let capsule = unsafe { PyCapsule::new_with_pointer(py, handler, c"xla._CUSTOM_CALL_TARGET")? };
    let keyword_arguments = PyDict::new(py);
    keyword_arguments.set_item("platform", "CUDA")?;
    keyword_arguments.set_item("api_version", 1)?;
    PyModule::import(py, "jax")?.getattr("ffi")?.call_method(
        "register_ffi_target",
        (g_compute_cuda::FIRTH_COMPONENTS_FFI_TARGET, capsule),
        Some(&keyword_arguments),
    )?;
    Ok(true)
}

#[cfg(not(target_os = "linux"))]
fn register_firth_components_ffi_target_once(_py: Python<'_>) -> PyResult<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError};
    use pyo3::prelude::*;
    use pyo3::sync::PyOnceLock;

    use super::{
        SUPPORTED_JAX_VERSION, SUPPORTED_JAXLIB_VERSION, contextual_backend_error, jax_runtime_version_error,
        optional_ffi_registration_result,
    };
    use crate::binding::cli::NativeSigtermRequested;

    #[test]
    fn backend_setup_context_preserves_interruptions() {
        Python::initialize();
        Python::attach(|py| {
            for error in [PyKeyboardInterrupt::new_err("stop"), NativeSigtermRequested::new_err("terminate")] {
                let original_exception = error.value(py).as_ptr();
                let contextual_error = contextual_backend_error(py, error, "loading GPU runtime failed");
                assert_eq!(contextual_error.value(py).as_ptr(), original_exception);
            }
            let failure = contextual_backend_error(py, PyRuntimeError::new_err("KeyboardInterrupt"), "GPU setup");
            assert!(failure.is_instance_of::<PyRuntimeError>(py));
            assert!(failure.to_string().contains("GPU setup"));
        });
    }

    #[test]
    fn optional_backend_registration_retries_after_interruption() {
        Python::initialize();
        Python::attach(|py| {
            let registration = PyOnceLock::new();
            let error = registration
                .get_or_try_init(py, || optional_ffi_registration_result(py, Err(PyKeyboardInterrupt::new_err("stop"))))
                .expect_err("optional setup must preserve interruption");
            assert!(error.is_instance_of::<PyKeyboardInterrupt>(py));
            assert!(registration.get(py).is_none());
            assert!(
                *registration
                    .get_or_try_init(py, || optional_ffi_registration_result(py, Ok(true)))
                    .expect("interrupted setup can be retried")
            );
            assert!(
                !optional_ffi_registration_result(py, Err(PyRuntimeError::new_err("optional GPU support unavailable")))
                    .expect("ordinary optional-support failures retain fallback")
            );
        });
    }

    #[test]
    fn exact_supported_jax_pair_is_accepted() {
        assert_eq!(jax_runtime_version_error(SUPPORTED_JAX_VERSION, SUPPORTED_JAXLIB_VERSION), None);
    }

    #[test]
    fn wrong_jax_version_is_rejected_independently() {
        let error = jax_runtime_version_error("0.11.1", SUPPORTED_JAXLIB_VERSION)
            .expect("a mismatched JAX version should be rejected");

        assert!(error.contains("jax==0.11.0 and jaxlib==0.11.0"));
        assert!(error.contains("observed jax==0.11.1 and jaxlib==0.11.0"));
        assert!(error.contains("uv sync --frozen"));
    }

    #[test]
    fn wrong_jaxlib_version_is_rejected_independently() {
        let error = jax_runtime_version_error(SUPPORTED_JAX_VERSION, "0.11.1")
            .expect("a mismatched jaxlib version should be rejected");

        assert!(error.contains("observed jax==0.11.0 and jaxlib==0.11.1"));
    }

    #[test]
    fn local_and_prerelease_suffixes_are_rejected() {
        for (jax_version, jaxlib_version) in [
            ("0.11.0+local", SUPPORTED_JAXLIB_VERSION),
            (SUPPORTED_JAX_VERSION, "0.11.0+local"),
            ("0.11.0rc1", SUPPORTED_JAXLIB_VERSION),
            (SUPPORTED_JAX_VERSION, "0.11.0rc1"),
        ] {
            assert!(
                jax_runtime_version_error(jax_version, jaxlib_version).is_some(),
                "version suffix should be rejected for jax={jax_version}, jaxlib={jaxlib_version}"
            );
        }
    }
}
