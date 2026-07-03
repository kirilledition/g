//! PyO3 adapters for run metadata and artifact payload construction.

use std::path::Path;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use super::config::RegenieConfig;
use super::run_events;
use g_interface as interface;
use g_output::OutputWriterError;
use g_runtime::run_metadata as native_run_metadata;

#[pyclass]
pub(crate) struct NativeRunMetadataBuilder;

#[pymethods]
impl NativeRunMetadataBuilder {
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_execution_run_artifacts(
        &self,
        association_mode: String,
        phenotype_count: i64,
        output_format: String,
        output_run_directories: Vec<String>,
        chunks_directories: Vec<String>,
        effective_configs: Vec<String>,
        phenotype_names: Vec<String>,
        final_output_paths: Vec<Option<String>>,
    ) -> PyResult<run_events::NativeRunArtifacts> {
        native_run_metadata::build_execution_run_artifacts_from_sequences(
            native_run_metadata::ExecutionRunArtifactsSequenceInput {
                association_mode,
                phenotype_count,
                output_format,
                output_run_directories,
                chunks_directories,
                effective_configs,
                phenotype_names,
                final_output_paths,
            },
        )
        .map(run_events::NativeRunArtifacts::from_payload)
        .map_err(|error| run_metadata_error_to_py(&error))
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_execution_run_artifacts_payload<'py>(
        &self,
        py: Python<'py>,
        association_mode: String,
        phenotype_count: i64,
        output_format: String,
        output_run_directories: Vec<String>,
        chunks_directories: Vec<String>,
        effective_configs: Vec<String>,
        phenotype_names: Vec<String>,
        final_output_paths: Vec<Option<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let artifacts = native_run_metadata::build_execution_run_artifacts_from_sequences(
            native_run_metadata::ExecutionRunArtifactsSequenceInput {
                association_mode,
                phenotype_count,
                output_format,
                output_run_directories,
                chunks_directories,
                effective_configs,
                phenotype_names,
                final_output_paths,
            },
        )
        .map_err(|error| run_metadata_error_to_py(&error))?;
        run_artifacts_payload_to_dict(py, &artifacts)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn extend_run_manifest_metadata(
        &self,
        py: Python<'_>,
        run_directory: String,
        phenotype_name: String,
        effective_config: String,
        output_format: String,
        device: String,
        staging_depth: i64,
        native_callback_batch_size: i64,
        threads: Option<i64>,
        writer_threads: i64,
        writer_queue_depth: i64,
        chunks_per_arrow_file: i64,
        arrow_compression: String,
        parquet_compression: String,
        output_statistic_dtype: String,
        bgen_decode_tile_variant_count: i64,
        trusted_no_missing_diploid: bool,
        trusted_bgen_validation_mode: String,
    ) -> PyResult<()> {
        extend_manifest_metadata(
            py,
            run_directory,
            phenotype_name,
            effective_config,
            output_format,
            device,
            staging_depth,
            native_callback_batch_size,
            threads,
            writer_threads,
            writer_queue_depth,
            chunks_per_arrow_file,
            arrow_compression,
            parquet_compression,
            output_statistic_dtype,
            bgen_decode_tile_variant_count,
            trusted_no_missing_diploid,
            trusted_bgen_validation_mode,
        )
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn write_run_start_metadata(
        &self,
        py: Python<'_>,
        config: &RegenieConfig,
        run_directory: String,
        phenotype_name: String,
        effective_config: String,
        output_format: String,
        device: String,
        staging_depth: i64,
        native_callback_batch_size: i64,
        threads: Option<i64>,
        writer_threads: i64,
        writer_queue_depth: i64,
        chunks_per_arrow_file: i64,
        arrow_compression: String,
        parquet_compression: String,
        output_statistic_dtype: String,
        bgen_decode_tile_variant_count: i64,
        trusted_no_missing_diploid: bool,
        trusted_bgen_validation_mode: String,
    ) -> PyResult<()> {
        interface::write_toml(config.data(), Path::new(&effective_config))
            .map_err(|error| config_error_to_py("write_run_start_metadata", &error))?;
        extend_manifest_metadata(
            py,
            run_directory,
            phenotype_name,
            effective_config,
            output_format,
            device,
            staging_depth,
            native_callback_batch_size,
            threads,
            writer_threads,
            writer_queue_depth,
            chunks_per_arrow_file,
            arrow_compression,
            parquet_compression,
            output_statistic_dtype,
            bgen_decode_tile_variant_count,
            trusted_no_missing_diploid,
            trusted_bgen_validation_mode,
        )
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRunMetadataBuilder>()?;
    Ok(())
}

fn run_artifacts_payload_to_dict<'py>(
    py: Python<'py>,
    artifacts: &native_run_metadata::RunArtifactsPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    set_optional_string(py, &payload, "output_run_directory", artifacts.output_run_directory.as_deref())?;
    set_optional_string(py, &payload, "final_dataset", artifacts.final_dataset.as_deref())?;
    set_optional_string(py, &payload, "final_parquet", artifacts.final_parquet.as_deref())?;
    set_optional_string(py, &payload, "final_regenie", artifacts.final_regenie.as_deref())?;
    set_optional_string(py, &payload, "effective_config", artifacts.effective_config.as_deref())?;
    let phenotype_artifacts = artifacts
        .phenotype_artifacts
        .iter()
        .map(|phenotype_artifact| run_artifacts_payload_to_dict(py, phenotype_artifact))
        .collect::<PyResult<Vec<_>>>()?;
    payload.set_item("phenotype_artifacts", PyTuple::new(py, &phenotype_artifacts)?)?;
    set_optional_string(py, &payload, "phenotype_name", artifacts.phenotype_name.as_deref())?;
    set_optional_string(py, &payload, "association_mode", artifacts.association_mode.as_deref())?;
    set_optional_i64(py, &payload, "phenotype_count", artifacts.phenotype_count)?;
    set_optional_string(py, &payload, "run_id", artifacts.run_id.as_deref())?;
    Ok(payload)
}

fn set_optional_string(py: Python<'_>, payload: &Bound<'_, PyDict>, key: &str, value: Option<&str>) -> PyResult<()> {
    match value {
        Some(text) => payload.set_item(key, text),
        None => payload.set_item(key, py.None()),
    }
}

fn set_optional_i64(py: Python<'_>, payload: &Bound<'_, PyDict>, key: &str, value: Option<i64>) -> PyResult<()> {
    match value {
        Some(integer) => payload.set_item(key, integer),
        None => payload.set_item(key, py.None()),
    }
}

fn run_metadata_error_to_py(error: &native_run_metadata::RunMetadataError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[expect(clippy::too_many_arguments, reason = "PyO3 adapter mirrors run manifest extension fields.")]
#[expect(clippy::needless_pass_by_value, reason = "PyO3 extracts Python strings into owned values.")]
fn extend_manifest_metadata(
    py: Python<'_>,
    run_directory: String,
    phenotype_name: String,
    effective_config: String,
    output_format: String,
    device: String,
    staging_depth: i64,
    native_callback_batch_size: i64,
    threads: Option<i64>,
    writer_threads: i64,
    writer_queue_depth: i64,
    chunks_per_arrow_file: i64,
    arrow_compression: String,
    parquet_compression: String,
    output_statistic_dtype: String,
    bgen_decode_tile_variant_count: i64,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: String,
) -> PyResult<()> {
    let extension = native_run_metadata::build_run_manifest_extension(native_run_metadata::RunManifestExtensionInput {
        phenotype_name,
        effective_config,
        output_format,
        device,
        staging_depth,
        native_callback_batch_size,
        threads,
        writer_threads,
        writer_queue_depth,
        chunks_per_arrow_file,
        arrow_compression,
        parquet_compression,
        output_statistic_dtype,
        bgen_decode_tile_variant_count,
        trusted_no_missing_diploid,
        trusted_bgen_validation_mode,
    });
    let command = serde_json::to_value(&extension.command).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let runtime = serde_json::to_value(&extension.runtime).map_err(|error| PyValueError::new_err(error.to_string()))?;
    py.detach(|| g_output::extend_run_manifest_metadata(Path::new(&run_directory), command, runtime))
        .map_err(output_writer_error_to_py)
}

fn config_error_to_py(operation: &str, error: &interface::ConfigError) -> PyErr {
    let error_message = error.to_string();
    tracing::warn!(
        target: "g.python.run_metadata",
        g_event = "native_config_error",
        operation = operation,
        error_message = %error_message,
        "Converting Rust config error to Python."
    );
    PyValueError::new_err(error_message)
}

fn output_writer_error_to_py(error: OutputWriterError) -> PyErr {
    match error {
        OutputWriterError::InvalidInput(message) => PyValueError::new_err(message),
        OutputWriterError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}
