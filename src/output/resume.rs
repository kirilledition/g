#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::Schema;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use serde_json::Value;

use crate::output::writer::OutputWriterError;

pub fn scan_committed_chunk_identifiers(chunks_directory: &Path) -> Result<Vec<i64>, OutputWriterError> {
    if !chunks_directory.exists() {
        return Ok(Vec::new());
    }
    let mut committed_identifiers = BTreeSet::new();
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputWriterError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    for chunk_file_path in chunk_file_paths {
        if let Some((first_chunk_identifier, None)) = parse_chunk_file_name(&chunk_file_path) {
            committed_identifiers.insert(first_chunk_identifier);
            continue;
        }
        let input_file = File::open(&chunk_file_path).map_err(OutputWriterError::runtime)?;
        let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
        for maybe_batch in file_reader {
            let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
            let chunk_identifier_array = batch
                .column_by_name("chunk_identifier")
                .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| {
                    OutputWriterError::Runtime(
                        "Rust output writer could not read chunk identifiers from Arrow chunk.".to_string(),
                    )
                })?;
            for row_index in 0..chunk_identifier_array.len() {
                if !chunk_identifier_array.is_null(row_index) {
                    committed_identifiers.insert(chunk_identifier_array.value(row_index));
                }
            }
        }
    }
    Ok(committed_identifiers.into_iter().collect())
}

pub fn validate_strict_manifest_chunks(
    chunks_directory: &Path,
    manifest_json: &str,
) -> Result<Vec<i64>, OutputWriterError> {
    let manifest = serde_json::from_str::<Value>(manifest_json).map_err(OutputWriterError::runtime)?;
    let committed_chunks = manifest.get("committed_chunks").and_then(Value::as_array).ok_or_else(|| {
        OutputWriterError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string())
    })?;
    let mut committed_identifiers = BTreeSet::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for committed_chunk in committed_chunks {
        let chunk_identifier = read_manifest_integer(committed_chunk, "chunk_identifier")?;
        let variant_start_index = read_manifest_integer(committed_chunk, "variant_start_index")?;
        let variant_stop_index = read_manifest_integer(committed_chunk, "variant_stop_index")?;
        let row_count = read_manifest_integer(committed_chunk, "row_count")?;
        let chunk_file_name = committed_chunk.get("chunk_file_name").and_then(Value::as_str).ok_or_else(|| {
            OutputWriterError::InvalidInput(
                "Run manifest committed chunk entry is missing chunk_file_name.".to_string(),
            )
        })?;
        let chunk_file_path = chunks_directory.join(chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        let chunk_observation = inspect_manifest_chunk_file(&chunk_file_path, chunk_identifier)?;
        match expected_schema.as_ref() {
            Some(schema) if schema.as_ref() != chunk_observation.schema.as_ref() => {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume found incompatible Arrow schema in {}.",
                    chunk_file_path.display()
                )));
            }
            None => expected_schema = Some(Arc::clone(&chunk_observation.schema)),
            Some(_) => {}
        }
        if chunk_observation.row_count != row_count {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume row count mismatch for chunk {chunk_identifier}."
            )));
        }
        if chunk_observation.variant_start_index != Some(variant_start_index)
            || chunk_observation.variant_stop_index != Some(variant_stop_index)
        {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume variant range mismatch for chunk {chunk_identifier}."
            )));
        }
        committed_identifiers.insert(chunk_identifier);
    }
    Ok(committed_identifiers.into_iter().collect())
}

struct ManifestChunkObservation {
    schema: Arc<Schema>,
    row_count: i64,
    variant_start_index: Option<i64>,
    variant_stop_index: Option<i64>,
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> Result<i64, OutputWriterError> {
    committed_chunk.get(field_name).and_then(Value::as_i64).ok_or_else(|| {
        OutputWriterError::InvalidInput(format!("Run manifest committed chunk entry is missing {field_name}."))
    })
}

fn inspect_manifest_chunk_file(
    chunk_file_path: &Path,
    chunk_identifier: i64,
) -> Result<ManifestChunkObservation, OutputWriterError> {
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
    let schema = file_reader.schema();
    let mut row_count = 0_i64;
    let mut observed_start: Option<i64> = None;
    let mut observed_stop: Option<i64> = None;
    for maybe_batch in file_reader {
        let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
        let chunk_identifier_array = read_int64_column(&batch, "chunk_identifier")?;
        let variant_start_array = read_int64_column(&batch, "variant_start_index")?;
        let variant_stop_array = read_int64_column(&batch, "variant_stop_index")?;
        for row_index in 0..chunk_identifier_array.len() {
            if chunk_identifier_array.is_null(row_index) || chunk_identifier_array.value(row_index) != chunk_identifier
            {
                continue;
            }
            row_count += 1;
            if !variant_start_array.is_null(row_index) {
                let value = variant_start_array.value(row_index);
                observed_start = Some(observed_start.map_or(value, |current| current.min(value)));
            }
            if !variant_stop_array.is_null(row_index) {
                let value = variant_stop_array.value(row_index);
                observed_stop = Some(observed_stop.map_or(value, |current| current.max(value)));
            }
        }
    }
    Ok(ManifestChunkObservation {
        schema,
        row_count,
        variant_start_index: observed_start,
        variant_stop_index: observed_stop,
    })
}

fn read_int64_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    column_name: &str,
) -> Result<&'a Int64Array, OutputWriterError> {
    batch.column_by_name(column_name).and_then(|column| column.as_any().downcast_ref::<Int64Array>()).ok_or_else(|| {
        OutputWriterError::Runtime(format!("Rust output writer could not read {column_name} from Arrow chunk."))
    })
}

fn parse_chunk_file_name(chunk_file_path: &Path) -> Option<(i64, Option<i64>)> {
    let file_name = chunk_file_path.file_name()?.to_str()?;
    let chunk_name = file_name.strip_prefix("chunk_")?.strip_suffix(".arrow")?;
    let chunk_parts = chunk_name.split('_').collect::<Vec<_>>();
    match chunk_parts.as_slice() {
        [first_chunk_identifier] => first_chunk_identifier.parse::<i64>().ok().map(|identifier| (identifier, None)),
        [first_chunk_identifier, last_chunk_identifier] => {
            first_chunk_identifier.parse::<i64>().ok().zip(last_chunk_identifier.parse::<i64>().ok().map(Some))
        }
        _ => None,
    }
}
