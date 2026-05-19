use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Int64Array};
use arrow::ipc::reader::FileReader as ArrowFileReader;

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
