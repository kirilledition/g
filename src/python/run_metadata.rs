//! PyO3 adapters for run metadata and artifact payload construction.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_runtime::run_metadata as native_run_metadata;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_phenotype_run_artifacts_payload<'py>(
    py: Python<'py>,
    output_run_directory: String,
    chunks_directory: String,
    effective_config: String,
    phenotype_name: String,
    association_mode: String,
    phenotype_count: i64,
    output_format: String,
    final_output_path: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let artifacts =
        native_run_metadata::build_phenotype_run_artifacts(native_run_metadata::PhenotypeRunArtifactsInput {
            output_run_directory,
            chunks_directory,
            effective_config,
            phenotype_name,
            association_mode,
            phenotype_count,
            output_format,
            final_output_path,
        });
    run_artifacts_payload_to_dict(py, &artifacts)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_multi_run_artifacts_payload<'py>(
    py: Python<'py>,
    association_mode: String,
    phenotype_count: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let artifacts = native_run_metadata::build_multi_run_artifacts(&association_mode, phenotype_count);
    run_artifacts_payload_to_dict(py, &artifacts)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_run_manifest_extension_payload<'py>(
    py: Python<'py>,
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
) -> PyResult<Bound<'py, PyDict>> {
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
    let payload = PyDict::new(py);
    payload.set_item("command", run_manifest_command_to_dict(py, &extension.command)?)?;
    payload.set_item("runtime", run_manifest_runtime_to_dict(py, &extension.runtime)?)?;
    Ok(payload)
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
    payload.set_item("phenotype_artifacts", PyTuple::new(py, Vec::<String>::new())?)?;
    set_optional_string(py, &payload, "phenotype_name", artifacts.phenotype_name.as_deref())?;
    set_optional_string(py, &payload, "association_mode", artifacts.association_mode.as_deref())?;
    set_optional_i64(py, &payload, "phenotype_count", artifacts.phenotype_count)?;
    set_optional_string(py, &payload, "run_id", artifacts.run_id.as_deref())?;
    Ok(payload)
}

fn run_manifest_command_to_dict<'py>(
    py: Python<'py>,
    command: &native_run_metadata::RunManifestCommandPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("interface", command.interface)?;
    payload.set_item("phenotype", &command.phenotype)?;
    payload.set_item("effective_config", &command.effective_config)?;
    payload.set_item("output_format", &command.output_format)?;
    Ok(payload)
}

fn run_manifest_runtime_to_dict<'py>(
    py: Python<'py>,
    runtime: &native_run_metadata::RunManifestRuntimePayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("device", &runtime.device)?;
    payload.set_item("staging_depth", runtime.staging_depth)?;
    payload.set_item("native_callback_batch_size", runtime.native_callback_batch_size)?;
    set_optional_i64(py, &payload, "threads", runtime.threads)?;
    payload.set_item("writer_threads", runtime.writer_threads)?;
    payload.set_item("writer_queue_depth", runtime.writer_queue_depth)?;
    payload.set_item("chunks_per_arrow_file", runtime.chunks_per_arrow_file)?;
    payload.set_item("arrow_compression", &runtime.arrow_compression)?;
    payload.set_item("parquet_compression", &runtime.parquet_compression)?;
    payload.set_item("output_statistic_dtype", &runtime.output_statistic_dtype)?;
    payload.set_item("bgen_decode_tile_variant_count", runtime.bgen_decode_tile_variant_count)?;
    payload.set_item("trusted_no_missing_diploid", runtime.trusted_no_missing_diploid)?;
    payload.set_item("trusted_bgen_validation_mode", &runtime.trusted_bgen_validation_mode)?;
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
