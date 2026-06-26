#![allow(clippy::missing_errors_doc)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};
use serde_json::Value;

use crate::manifest;
use crate::schema;
use crate::writer::{self, OutputWriterError};

pub fn scan_committed_chunk_identifiers(chunks_directory: &Path) -> Result<Vec<i64>, OutputWriterError> {
    Ok(scan_committed_chunk_commits(chunks_directory)?
        .into_iter()
        .map(|chunk_commit| chunk_commit.chunk_identifier)
        .collect())
}

pub fn validate_strict_manifest_chunks(
    chunks_directory: &Path,
    manifest_json: &str,
) -> Result<Vec<i64>, OutputWriterError> {
    let manifest_commits = read_manifest_chunk_commits(manifest_json)?;
    let mut committed_identifiers = BTreeSet::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for (chunk_file_name, chunk_commits) in group_manifest_commits_by_file(manifest_commits) {
        let chunk_file_path = chunks_directory.join(&chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        validate_manifest_chunk_file_commits(
            &chunk_file_path,
            &chunk_commits,
            &mut expected_schema,
            &mut committed_identifiers,
        )?;
    }
    Ok(committed_identifiers.into_iter().collect())
}

pub fn repair_strict_manifest_chunk_commits(
    chunks_directory: &Path,
    manifest_json: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    let mut repaired_commits = read_manifest_chunk_commits(manifest_json)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    let scanned_commits = scan_committed_chunk_commits(chunks_directory)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    for existing_commit in repaired_commits.values() {
        let chunk_file_path = chunks_directory.join(&existing_commit.chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        match scanned_commits.get(&existing_commit.chunk_identifier) {
            Some(scanned_commit) if scanned_commit == existing_commit => {}
            Some(_) => {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {}.",
                    existing_commit.chunk_identifier
                )));
            }
            None => {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume manifest references unobserved commit metadata for chunk {}.",
                    existing_commit.chunk_identifier
                )));
            }
        }
    }
    for (chunk_identifier, chunk_commit) in scanned_commits {
        if let Some(existing_commit) = repaired_commits.get(&chunk_identifier) {
            if existing_commit != &chunk_commit {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {chunk_identifier}."
                )));
            }
        } else {
            repaired_commits.insert(chunk_identifier, chunk_commit);
        }
    }
    Ok(repaired_commits.into_values().collect())
}

#[derive(Clone)]
struct ChunkCommitObservation {
    chunk_identifier: i64,
    output_format: String,
    compression: String,
    variant_start_index: i64,
    variant_stop_index: i64,
    row_count: i64,
}

struct ChunkFileCommitObservation {
    schema: Arc<Schema>,
    chunk_commits: Vec<manifest::RunManifestChunkCommit>,
}

fn read_manifest_chunk_commits(
    manifest_json: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    manifest::read_run_manifest_chunk_commits_from_text(manifest_json).map_err(OutputWriterError::InvalidInput)
}

fn read_optional_manifest_string(committed_chunk: &Value, field_name: &str) -> Option<String> {
    committed_chunk.get(field_name).and_then(Value::as_str).map(str::to_string)
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> Result<i64, OutputWriterError> {
    committed_chunk.get(field_name).and_then(Value::as_i64).ok_or_else(|| {
        OutputWriterError::InvalidInput(format!("Run manifest committed chunk entry is missing {field_name}."))
    })
}

fn scan_committed_chunk_commits(
    chunks_directory: &Path,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    if !chunks_directory.exists() {
        return Ok(Vec::new());
    }
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputWriterError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| {
            chunk_file_path
                .extension()
                .is_some_and(|extension| extension == "arrow" || extension == "parquet" || extension == "regenie")
        })
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    let mut chunk_commits = BTreeMap::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for chunk_file_path in chunk_file_paths {
        let chunk_file_observation = inspect_chunk_file_commits(&chunk_file_path)?;
        match expected_schema.as_ref() {
            Some(expected_schema) if expected_schema.fields() != chunk_file_observation.schema.fields() => {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume found incompatible Arrow schema in {}.",
                    chunk_file_path.display()
                )));
            }
            None => expected_schema = Some(Arc::clone(&chunk_file_observation.schema)),
            Some(_) => {}
        }
        for chunk_commit in chunk_file_observation.chunk_commits {
            if chunk_commits.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
                return Err(OutputWriterError::InvalidInput(
                    "Strict resume found duplicate Arrow commit metadata for a chunk.".to_string(),
                ));
            }
        }
    }
    Ok(chunk_commits.into_values().collect())
}

fn group_manifest_commits_by_file(
    manifest_commits: Vec<manifest::RunManifestChunkCommit>,
) -> BTreeMap<String, Vec<manifest::RunManifestChunkCommit>> {
    let mut chunk_commits_by_file = BTreeMap::<String, Vec<manifest::RunManifestChunkCommit>>::new();
    for chunk_commit in manifest_commits {
        chunk_commits_by_file.entry(chunk_commit.chunk_file_name.clone()).or_default().push(chunk_commit);
    }
    chunk_commits_by_file
}

fn validate_manifest_chunk_file_commits(
    chunk_file_path: &Path,
    expected_commits: &[manifest::RunManifestChunkCommit],
    expected_schema: &mut Option<Arc<Schema>>,
    committed_identifiers: &mut BTreeSet<i64>,
) -> Result<(), OutputWriterError> {
    let chunk_file_observation = inspect_chunk_file_commits(chunk_file_path)?;
    match expected_schema.as_ref() {
        Some(expected_schema) if expected_schema.fields() != chunk_file_observation.schema.fields() => {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume found incompatible Arrow schema in {}.",
                chunk_file_path.display()
            )));
        }
        None => *expected_schema = Some(Arc::clone(&chunk_file_observation.schema)),
        Some(_) => {}
    }
    let observed_commits = collect_chunk_commits_by_identifier(chunk_file_observation.chunk_commits)?;
    let expected_commit_identifiers =
        expected_commits.iter().map(|chunk_commit| chunk_commit.chunk_identifier).collect::<BTreeSet<_>>();
    let observed_commit_identifiers = observed_commits.keys().copied().collect::<BTreeSet<_>>();
    for expected_commit in expected_commits {
        let Some(observed_commit) = observed_commits.get(&expected_commit.chunk_identifier) else {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume manifest commit set does not match chunk file {}.",
                chunk_file_path.display()
            )));
        };
        validate_manifest_chunk_commit(expected_commit, observed_commit)?;
        committed_identifiers.insert(expected_commit.chunk_identifier);
    }
    if observed_commit_identifiers != expected_commit_identifiers {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume manifest commit set does not match chunk file {}.",
            chunk_file_path.display()
        )));
    }
    Ok(())
}

fn validate_manifest_chunk_commit(
    expected_commit: &manifest::RunManifestChunkCommit,
    observed_commit: &manifest::RunManifestChunkCommit,
) -> Result<(), OutputWriterError> {
    if observed_commit.variant_start_index != expected_commit.variant_start_index
        || observed_commit.variant_stop_index != expected_commit.variant_stop_index
    {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume variant range mismatch for chunk {}.",
            expected_commit.chunk_identifier
        )));
    }
    if observed_commit.row_count != expected_commit.row_count {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume row count mismatch for chunk {}.",
            expected_commit.chunk_identifier
        )));
    }
    if observed_commit != expected_commit {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume found conflicting commit metadata for chunk {}.",
            expected_commit.chunk_identifier
        )));
    }
    Ok(())
}

fn collect_chunk_commits_by_identifier(
    chunk_commits: Vec<manifest::RunManifestChunkCommit>,
) -> Result<BTreeMap<i64, manifest::RunManifestChunkCommit>, OutputWriterError> {
    let mut chunk_commits_by_identifier = BTreeMap::new();
    for chunk_commit in chunk_commits {
        if chunk_commits_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
            return Err(OutputWriterError::InvalidInput(
                "Strict resume found duplicate Arrow commit metadata for a chunk.".to_string(),
            ));
        }
    }
    Ok(chunk_commits_by_identifier)
}

fn inspect_chunk_file_commits(chunk_file_path: &Path) -> Result<ChunkFileCommitObservation, OutputWriterError> {
    if chunk_file_path.extension().is_some_and(|extension| extension == "parquet") {
        return inspect_parquet_chunk_file_commits(chunk_file_path);
    }
    if chunk_file_path.extension().is_some_and(|extension| extension == "regenie") {
        return inspect_regenie_text_chunk_file_commits(chunk_file_path);
    }
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
    let schema = file_reader.schema();
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputWriterError::Runtime("Rust output writer chunk file name is not UTF-8.".to_string()))?
        .to_string();
    let Some(chunk_commits) = read_schema_chunk_commits(schema.as_ref())? else {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume Arrow chunk is missing chunk commit metadata: {}",
            chunk_file_path.display()
        )));
    };
    let chunk_commits = inspect_metadata_chunk_file_commits(file_reader, chunk_commits, &chunk_file_name)?;
    Ok(ChunkFileCommitObservation { schema, chunk_commits })
}

fn inspect_metadata_chunk_file_commits(
    file_reader: ArrowFileReader<File>,
    chunk_commits: Vec<ChunkCommitObservation>,
    chunk_file_name: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    let mut batch_row_counts = Vec::with_capacity(chunk_commits.len());
    for maybe_batch in file_reader {
        let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
        batch_row_counts.push(i64::try_from(batch.num_rows()).map_err(OutputWriterError::runtime)?);
    }
    if batch_row_counts.len() != chunk_commits.len() {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume batch count mismatch for chunk file {chunk_file_name}."
        )));
    }
    let mut manifest_commits = Vec::with_capacity(chunk_commits.len());
    for (observed_row_count, chunk_commit) in batch_row_counts.iter().zip(chunk_commits) {
        if *observed_row_count != chunk_commit.row_count {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume row count mismatch for chunk {}.",
                chunk_commit.chunk_identifier
            )));
        }
        manifest_commits.push(manifest::RunManifestChunkCommit {
            chunk_identifier: chunk_commit.chunk_identifier,
            output_format: chunk_commit.output_format,
            compression: chunk_commit.compression,
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: usize::try_from(chunk_commit.row_count).map_err(OutputWriterError::runtime)?,
            chunk_file_name: chunk_file_name.to_string(),
        });
    }
    Ok(manifest_commits)
}

fn inspect_parquet_chunk_file_commits(chunk_file_path: &Path) -> Result<ChunkFileCommitObservation, OutputWriterError> {
    let schema = read_parquet_arrow_schema(chunk_file_path)?;
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputWriterError::Runtime("Rust output writer part file name is not UTF-8.".to_string()))?
        .to_string();
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let parquet_reader = SerializedFileReader::new(input_file).map_err(OutputWriterError::runtime)?;
    let file_metadata = parquet_reader.metadata().file_metadata();
    let observed_row_count = file_metadata.num_rows();
    let chunk_commit_text = file_metadata
        .key_value_metadata()
        .and_then(|metadata| metadata.iter().find(|entry| entry.key == schema::CHUNK_COMMITS_METADATA_KEY))
        .and_then(|entry| entry.value.as_deref())
        .ok_or_else(|| {
            OutputWriterError::InvalidInput(format!(
                "Strict resume Parquet part is missing chunk commit metadata: {}",
                chunk_file_path.display()
            ))
        })?;
    let chunk_commits = read_chunk_commit_observations_text(chunk_commit_text)?;
    let summed_row_count = chunk_commits
        .iter()
        .try_fold(0_i64, |total, chunk_commit| total.checked_add(chunk_commit.row_count).ok_or(()))
        .map_err(|()| OutputWriterError::Runtime("Rust output writer Parquet row count overflowed.".to_string()))?;
    if summed_row_count != observed_row_count {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume Parquet row count mismatch for part {chunk_file_name}."
        )));
    }
    let mut manifest_commits = Vec::with_capacity(chunk_commits.len());
    for chunk_commit in chunk_commits {
        if chunk_commit.output_format != "parquet" {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume Parquet part has non-Parquet commit metadata for chunk {}.",
                chunk_commit.chunk_identifier
            )));
        }
        manifest_commits.push(manifest::RunManifestChunkCommit {
            chunk_identifier: chunk_commit.chunk_identifier,
            output_format: chunk_commit.output_format,
            compression: chunk_commit.compression,
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: usize::try_from(chunk_commit.row_count).map_err(OutputWriterError::runtime)?,
            chunk_file_name: chunk_file_name.clone(),
        });
    }
    Ok(ChunkFileCommitObservation { schema, chunk_commits: manifest_commits })
}

fn inspect_regenie_text_chunk_file_commits(
    chunk_file_path: &Path,
) -> Result<ChunkFileCommitObservation, OutputWriterError> {
    let schema = Arc::clone(schema::get_regenie_step2_final_schema(schema::OutputStatisticDtype::Float32));
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputWriterError::Runtime("Rust output writer text part file name is not UTF-8.".to_string()))?
        .to_string();
    let sidecar_path = writer::build_regenie_text_metadata_sidecar_path(chunk_file_path);
    let chunk_commit_text = std::fs::read_to_string(&sidecar_path).map_err(|error| {
        OutputWriterError::InvalidInput(format!(
            "Strict resume REGENIE text part is missing chunk commit metadata: {} ({error})",
            sidecar_path.display()
        ))
    })?;
    let chunk_commits = read_chunk_commit_observations_text(&chunk_commit_text)?;
    let observed_row_count = count_regenie_text_rows(chunk_file_path)?;
    let summed_row_count = chunk_commits
        .iter()
        .try_fold(0_i64, |total, chunk_commit| total.checked_add(chunk_commit.row_count).ok_or(()))
        .map_err(|()| {
            OutputWriterError::Runtime("Rust output writer REGENIE text row count overflowed.".to_string())
        })?;
    if summed_row_count != observed_row_count {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume REGENIE text row count mismatch for part {chunk_file_name}."
        )));
    }
    let mut manifest_commits = Vec::with_capacity(chunk_commits.len());
    for chunk_commit in chunk_commits {
        if chunk_commit.output_format != "regenie" {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume REGENIE text part has non-REGENIE commit metadata for chunk {}.",
                chunk_commit.chunk_identifier
            )));
        }
        manifest_commits.push(manifest::RunManifestChunkCommit {
            chunk_identifier: chunk_commit.chunk_identifier,
            output_format: chunk_commit.output_format,
            compression: chunk_commit.compression,
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: usize::try_from(chunk_commit.row_count).map_err(OutputWriterError::runtime)?,
            chunk_file_name: chunk_file_name.clone(),
        });
    }
    Ok(ChunkFileCommitObservation { schema, chunk_commits: manifest_commits })
}

fn count_regenie_text_rows(chunk_file_path: &Path) -> Result<i64, OutputWriterError> {
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let mut input_reader = BufReader::new(input_file);
    let mut header_line = String::new();
    input_reader.read_line(&mut header_line).map_err(OutputWriterError::runtime)?;
    let observed_header = header_line.trim_end_matches(['\r', '\n']);
    let expected_header = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n');
    if observed_header != expected_header {
        return Err(OutputWriterError::InvalidInput(format!(
            "Strict resume REGENIE text part has an unexpected header: {}",
            chunk_file_path.display()
        )));
    }
    let mut row_count = 0_i64;
    let mut row_line = String::new();
    let expected_column_count = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n').split('\t').count();
    loop {
        row_line.clear();
        let read_byte_count = input_reader.read_line(&mut row_line).map_err(OutputWriterError::runtime)?;
        if read_byte_count == 0 {
            break;
        }
        let row = row_line.trim_end_matches(['\r', '\n']);
        if row.split('\t').count() != expected_column_count {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume REGENIE text part has a row with an unexpected column count: {}",
                chunk_file_path.display()
            )));
        }
        row_count = row_count.checked_add(1).ok_or_else(|| {
            OutputWriterError::Runtime("Rust output writer REGENIE text row count overflowed.".to_string())
        })?;
    }
    Ok(row_count)
}

fn read_parquet_arrow_schema(chunk_file_path: &Path) -> Result<Arc<Schema>, OutputWriterError> {
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(input_file).map_err(OutputWriterError::runtime)?;
    Ok(parquet_reader.schema().clone())
}

fn read_schema_chunk_commits(chunk_schema: &Schema) -> Result<Option<Vec<ChunkCommitObservation>>, OutputWriterError> {
    let Some(chunk_commits_text) = chunk_schema.metadata().get(schema::CHUNK_COMMITS_METADATA_KEY) else {
        return Ok(None);
    };
    Ok(Some(read_chunk_commit_observations_text(chunk_commits_text)?))
}

fn read_chunk_commit_observations_text(
    chunk_commits_text: &str,
) -> Result<Vec<ChunkCommitObservation>, OutputWriterError> {
    let chunk_commit_values = serde_json::from_str::<Value>(chunk_commits_text).map_err(OutputWriterError::runtime)?;
    let chunk_commit_array = chunk_commit_values.as_array().ok_or_else(|| {
        OutputWriterError::Runtime("Rust output writer chunk commit metadata must be a list.".to_string())
    })?;
    let mut chunk_commits = Vec::with_capacity(chunk_commit_array.len());
    for chunk_commit_value in chunk_commit_array {
        chunk_commits.push(ChunkCommitObservation {
            chunk_identifier: read_manifest_integer(chunk_commit_value, "chunk_identifier")?,
            output_format: read_optional_manifest_string(chunk_commit_value, "output_format")
                .unwrap_or_else(|| "arrow".to_string()),
            compression: read_optional_manifest_string(chunk_commit_value, "compression")
                .unwrap_or_else(|| "none".to_string()),
            variant_start_index: read_manifest_integer(chunk_commit_value, "variant_start_index")?,
            variant_stop_index: read_manifest_integer(chunk_commit_value, "variant_stop_index")?,
            row_count: read_manifest_integer(chunk_commit_value, "row_count")?,
        });
    }
    Ok(chunk_commits)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::{ArrayRef, Float32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::KeyValue;

    use crate::schema as output_schema;
    use crate::writer as output_writer;

    use super::{
        repair_strict_manifest_chunk_commits, scan_committed_chunk_identifiers, validate_strict_manifest_chunks,
    };

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-resume-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("resume test directory should be created");
        directory_path
    }

    fn write_arrow_file(path: &Path, schema: &Arc<Schema>, columns: Vec<ArrayRef>) {
        let batch = RecordBatch::try_new(Arc::clone(schema), columns).expect("record batch should build");
        let file = std::fs::File::create(path).expect("Arrow file should be created");
        let mut writer = FileWriter::try_new(file, schema.as_ref()).expect("Arrow writer should be created");
        writer.write(&batch).expect("Arrow batch should be written");
        writer.finish().expect("Arrow writer should finish");
    }

    fn write_arrow_batches(path: &Path, schema: &Arc<Schema>, batches: &[RecordBatch]) {
        let file = std::fs::File::create(path).expect("Arrow file should be created");
        let mut writer = FileWriter::try_new(file, schema.as_ref()).expect("Arrow writer should be created");
        for batch in batches {
            writer.write(batch).expect("Arrow batch should be written");
        }
        writer.finish().expect("Arrow writer should finish");
    }

    fn write_parquet_batches(path: &Path, schema: &Arc<Schema>, batches: &[RecordBatch], chunk_commits_json: &str) {
        let file = std::fs::File::create(path).expect("Parquet file should be created");
        let mut writer =
            ArrowWriter::try_new(file, Arc::clone(schema), None).expect("Parquet writer should be created");
        writer.append_key_value_metadata(KeyValue {
            key: output_schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
            value: Some(chunk_commits_json.to_string()),
        });
        for batch in batches {
            writer.write(batch).expect("Parquet batch should be written");
        }
        writer.close().expect("Parquet writer should close");
    }

    fn required_resume_schema_with_commits(extra_field: Option<Field>, chunk_commits_json: &str) -> Arc<Schema> {
        let mut fields = vec![
            Field::new("chunk_identifier", DataType::Int64, false),
            Field::new("variant_start_index", DataType::Int64, false),
            Field::new("variant_stop_index", DataType::Int64, false),
        ];
        if let Some(field) = extra_field {
            fields.push(field);
        }
        Arc::new(
            Schema::new(fields).with_metadata(
                [(output_schema::CHUNK_COMMITS_METADATA_KEY.to_string(), chunk_commits_json.to_string())]
                    .into_iter()
                    .collect(),
            ),
        )
    }

    fn schema_metadata_with_commits(chunk_commits_json: &str) -> Arc<Schema> {
        Arc::new(
            Schema::new(vec![Field::new("value", DataType::Int64, false)]).with_metadata(
                [(output_schema::CHUNK_COMMITS_METADATA_KEY.to_string(), chunk_commits_json.to_string())]
                    .into_iter()
                    .collect(),
            ),
        )
    }

    fn parquet_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]))
    }

    #[test]
    fn scan_reads_grouped_chunk_identifiers_from_schema_metadata_without_bookkeeping_columns() {
        let directory_path = create_test_directory();
        let schema = Arc::new(Schema::new(vec![Field::new("not_chunk_identifier", DataType::Int64, false)]).with_metadata(
            [(
                output_schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
                r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1},{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":3,"row_count":1}]"#.to_string(),
            )]
            .into_iter()
            .collect(),
        ));
        let first_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef])
                .expect("first batch should build");
        let second_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![2])) as ArrayRef])
                .expect("second batch should build");
        write_arrow_batches(&directory_path.join("chunk_0_2.arrow"), &schema, &[first_batch, second_batch]);

        let committed_identifiers =
            scan_committed_chunk_identifiers(&directory_path).expect("metadata-backed chunks should scan");

        assert_eq!(committed_identifiers, vec![0, 2]);

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn scan_reads_single_chunk_identifier_from_arrow_metadata() {
        let directory_path = create_test_directory();
        let schema = required_resume_schema_with_commits(
            None,
            r#"[{"chunk_identifier":3,"variant_start_index":3,"variant_stop_index":4,"row_count":1}]"#,
        );
        write_arrow_file(
            &directory_path.join("chunk_000000007.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![4])) as ArrayRef,
            ],
        );

        let committed_identifiers =
            scan_committed_chunk_identifiers(&directory_path).expect("metadata-backed Arrow chunk should scan");

        assert_eq!(committed_identifiers, vec![3]);

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn scan_and_strict_resume_read_parquet_part_footer_commits_and_ignore_tmp_files() {
        let directory_path = create_test_directory();
        let schema = parquet_schema();
        let first_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef])
                .expect("first batch should build");
        let second_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![2, 3])) as ArrayRef])
                .expect("second batch should build");
        let chunk_commits_json = r#"[{"chunk_identifier":0,"output_format":"parquet","compression":"none","variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"part_000000000_000000002.parquet"},{"chunk_identifier":2,"output_format":"parquet","compression":"none","variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"part_000000000_000000002.parquet"}]"#;
        write_parquet_batches(
            &directory_path.join("part_000000000_000000002.parquet"),
            &schema,
            &[first_batch, second_batch],
            chunk_commits_json,
        );
        std::fs::write(directory_path.join("part_000000004.parquet.tmp"), b"incomplete")
            .expect("tmp file should be written");

        let committed_identifiers =
            scan_committed_chunk_identifiers(&directory_path).expect("Parquet part should scan");

        assert_eq!(committed_identifiers, vec![0, 2]);
        let manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"output_format":"parquet","compression":"none","variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"part_000000000_000000002.parquet"},{"chunk_identifier":2,"output_format":"parquet","compression":"none","variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"part_000000000_000000002.parquet"}]}"#;
        assert_eq!(
            validate_strict_manifest_chunks(&directory_path, manifest).expect("Parquet manifest should validate"),
            vec![0, 2],
        );

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn scan_and_strict_resume_read_regenie_text_sidecar_commits() {
        let directory_path = create_test_directory();
        let part_file_path = directory_path.join("part_000000000_000000002.regenie");
        let chunk_commits_json = r#"[{"chunk_identifier":0,"output_format":"regenie","compression":"none","variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"part_000000000_000000002.regenie"},{"chunk_identifier":2,"output_format":"regenie","compression":"none","variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"part_000000000_000000002.regenie"}]"#;
        std::fs::write(
            &part_file_path,
            format!(
                "{}22\t100\tvariant0\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tNA\tscore\tsuccess\n22\t102\tvariant2\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tTEST_FAIL\tfirth_approximate\tfailed\n22\t103\tvariant3\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tNA\tscore\tsuccess\n",
                output_writer::REGENIE_STEP2_TEXT_HEADER
            ),
        )
        .expect("REGENIE text part should be written");
        std::fs::write(output_writer::build_regenie_text_metadata_sidecar_path(&part_file_path), chunk_commits_json)
            .expect("REGENIE text sidecar should be written");

        let committed_identifiers =
            scan_committed_chunk_identifiers(&directory_path).expect("REGENIE text part should scan");

        assert_eq!(committed_identifiers, vec![0, 2]);
        let manifest = format!(r#"{{"committed_chunks":{chunk_commits_json}}}"#);
        assert_eq!(
            validate_strict_manifest_chunks(&directory_path, &manifest).expect("REGENIE text manifest should validate"),
            vec![0, 2],
        );

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn strict_resume_rejects_parquet_part_row_count_mismatch() {
        let directory_path = create_test_directory();
        let schema = parquet_schema();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef])
            .expect("batch should build");
        write_parquet_batches(
            &directory_path.join("part_000000000.parquet"),
            &schema,
            &[batch],
            r#"[{"chunk_identifier":0,"output_format":"parquet","compression":"none","variant_start_index":0,"variant_stop_index":2,"row_count":2,"chunk_file_name":"part_000000000.parquet"}]"#,
        );

        let error = scan_committed_chunk_identifiers(&directory_path)
            .expect_err("Parquet row count mismatch should fail")
            .to_string();

        assert!(error.contains("row count mismatch"));

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn scan_rejects_schema_metadata_row_count_mismatch() {
        let directory_path = create_test_directory();
        let schema = Arc::new(
            Schema::new(vec![Field::new("value", DataType::Int64, false)]).with_metadata(
                [(
                    output_schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
                    r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":2}]"#
                        .to_string(),
                )]
                .into_iter()
                .collect(),
            ),
        );
        write_arrow_file(
            &directory_path.join("chunk_0.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );

        let error = scan_committed_chunk_identifiers(&directory_path)
            .expect_err("metadata row count mismatch should fail")
            .to_string();

        assert!(error.contains("row count mismatch"));

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn scan_rejects_arrow_metadata_without_chunk_identifier() {
        let directory_path = create_test_directory();
        let schema =
            schema_metadata_with_commits(r#"[{"variant_start_index":0,"variant_stop_index":1,"row_count":1}]"#);
        write_arrow_file(
            &directory_path.join("chunk_0_1.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );

        let error = scan_committed_chunk_identifiers(&directory_path)
            .expect_err("Arrow metadata without chunk_identifier should fail")
            .to_string();
        assert!(error.contains("chunk_identifier"));

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn strict_manifest_validates_schema_metadata_backed_grouped_chunks() {
        let directory_path = create_test_directory();
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]).with_metadata(
            [(
                output_schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
                r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1},{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":4,"row_count":2}]"#.to_string(),
            )]
            .into_iter()
            .collect(),
        ));
        let first_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef])
                .expect("first batch should build");
        let second_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![2, 3])) as ArrayRef])
                .expect("second batch should build");
        write_arrow_batches(&directory_path.join("chunk_0_2.arrow"), &schema, &[first_batch, second_batch]);
        let manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"chunk_0_2.arrow"},{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"chunk_0_2.arrow"}]}"#;

        let committed_identifiers = validate_strict_manifest_chunks(&directory_path, manifest)
            .expect("metadata-backed strict validation should pass");

        assert_eq!(committed_identifiers, vec![0, 2]);

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn strict_manifest_rejects_missing_arrow_metadata_and_schema_mismatches() {
        let directory_path = create_test_directory();
        let missing_column_schema = Arc::new(Schema::new(vec![Field::new("chunk_identifier", DataType::Int64, false)]));
        write_arrow_file(
            &directory_path.join("chunk_0_0.arrow"),
            &missing_column_schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        let missing_column_manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"chunk_0_0.arrow"}]}"#;
        let missing_column_error = validate_strict_manifest_chunks(&directory_path, missing_column_manifest)
            .expect_err("missing Arrow commit metadata should fail")
            .to_string();
        assert!(missing_column_error.contains("missing chunk commit metadata"));

        let schema = required_resume_schema_with_commits(
            None,
            r#"[{"chunk_identifier":1,"variant_start_index":1,"variant_stop_index":2,"row_count":1}]"#,
        );
        write_arrow_file(
            &directory_path.join("chunk_1_1.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
            ],
        );
        let extra_schema = required_resume_schema_with_commits(
            Some(Field::new("extra", DataType::Float32, false)),
            r#"[{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":3,"row_count":1}]"#,
        );
        write_arrow_file(
            &directory_path.join("chunk_2_2.arrow"),
            &extra_schema,
            vec![
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3])) as ArrayRef,
                Arc::new(Float32Array::from(vec![0.5])) as ArrayRef,
            ],
        );
        let schema_mismatch_manifest = r#"{"committed_chunks":[{"chunk_identifier":1,"variant_start_index":1,"variant_stop_index":2,"row_count":1,"chunk_file_name":"chunk_1_1.arrow"},{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":3,"row_count":1,"chunk_file_name":"chunk_2_2.arrow"}]}"#;
        let schema_mismatch_error = validate_strict_manifest_chunks(&directory_path, schema_mismatch_manifest)
            .expect_err("strict resume should reject incompatible schemas")
            .to_string();
        assert!(schema_mismatch_error.contains("incompatible Arrow schema"));

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn repair_strict_manifest_commits_discovers_extra_arrow_chunks_and_missing_directories() {
        let directory_path = create_test_directory();
        let schema = schema_metadata_with_commits(
            r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1},{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":4,"row_count":2}]"#,
        );
        let first_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef])
                .expect("first batch should build");
        let second_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![2, 3])) as ArrayRef])
                .expect("second batch should build");
        write_arrow_batches(&directory_path.join("chunk_0_2.arrow"), &schema, &[first_batch, second_batch]);
        let manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"chunk_0_2.arrow"}]}"#;

        let repaired_commits =
            repair_strict_manifest_chunk_commits(&directory_path, manifest).expect("repair should add missing chunk");

        assert_eq!(
            repaired_commits.iter().map(|chunk_commit| chunk_commit.chunk_identifier).collect::<Vec<_>>(),
            vec![0, 2],
        );

        let missing_directory_path = directory_path.join("missing");
        let empty_repaired_commits =
            repair_strict_manifest_chunk_commits(&missing_directory_path, r#"{"committed_chunks":[]}"#)
                .expect("missing chunks directory should repair as empty");
        assert!(empty_repaired_commits.is_empty());

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn repair_strict_manifest_rejects_negative_rows_duplicates_and_schema_metadata_errors() {
        let directory_path = create_test_directory();
        let schema = schema_metadata_with_commits(
            r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1}]"#,
        );
        write_arrow_file(
            &directory_path.join("chunk_0.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );

        let negative_rows_manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":-1,"chunk_file_name":"chunk_0.arrow"}]}"#;
        assert!(
            repair_strict_manifest_chunk_commits(&directory_path, negative_rows_manifest)
                .expect_err("negative row_count should fail")
                .to_string()
                .contains("non-negative")
        );

        write_arrow_file(
            &directory_path.join("chunk_0_duplicate.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        assert!(
            repair_strict_manifest_chunk_commits(&directory_path, r#"{"committed_chunks":[]}"#)
                .expect_err("duplicate chunk commits should fail")
                .to_string()
                .contains("duplicate")
        );

        let invalid_metadata_directory_path = create_test_directory();
        let invalid_metadata_schema = schema_metadata_with_commits(r#"{"chunk_identifier":0}"#);
        write_arrow_file(
            &invalid_metadata_directory_path.join("chunk_invalid.arrow"),
            &invalid_metadata_schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        assert!(
            scan_committed_chunk_identifiers(&invalid_metadata_directory_path)
                .expect_err("non-list metadata should fail")
                .to_string()
                .contains("must be a list")
        );

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
        std::fs::remove_dir_all(invalid_metadata_directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn repair_strict_manifest_rejects_scanned_schema_mismatches_and_metadata_batch_mismatches() {
        let schema_mismatch_directory_path = create_test_directory();
        let schema = required_resume_schema_with_commits(
            None,
            r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1}]"#,
        );
        write_arrow_file(
            &schema_mismatch_directory_path.join("chunk_0.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            ],
        );
        let extra_schema = required_resume_schema_with_commits(
            Some(Field::new("extra", DataType::Float32, false)),
            r#"[{"chunk_identifier":1,"variant_start_index":1,"variant_stop_index":2,"row_count":1}]"#,
        );
        write_arrow_file(
            &schema_mismatch_directory_path.join("chunk_1.arrow"),
            &extra_schema,
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
                Arc::new(Float32Array::from(vec![0.5])) as ArrayRef,
            ],
        );
        assert!(
            repair_strict_manifest_chunk_commits(&schema_mismatch_directory_path, r#"{"committed_chunks":[]}"#)
                .expect_err("scanned schema mismatch should fail")
                .to_string()
                .contains("incompatible Arrow schema")
        );

        let batch_count_directory_path = create_test_directory();
        let batch_count_schema = schema_metadata_with_commits(
            r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1},{"chunk_identifier":1,"variant_start_index":1,"variant_stop_index":2,"row_count":1}]"#,
        );
        write_arrow_file(
            &batch_count_directory_path.join("chunk_0_1.arrow"),
            &batch_count_schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        assert!(
            scan_committed_chunk_identifiers(&batch_count_directory_path)
                .expect_err("metadata batch count mismatch should fail")
                .to_string()
                .contains("batch count")
        );
        assert!(
            repair_strict_manifest_chunk_commits(&batch_count_directory_path, r#"{"committed_chunks":[]}"#)
                .expect_err("repair metadata batch count mismatch should fail")
                .to_string()
                .contains("batch count")
        );

        let row_count_directory_path = create_test_directory();
        let row_count_schema = schema_metadata_with_commits(
            r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":2,"row_count":2}]"#,
        );
        write_arrow_file(
            &row_count_directory_path.join("chunk_0.arrow"),
            &row_count_schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        assert!(
            repair_strict_manifest_chunk_commits(&row_count_directory_path, r#"{"committed_chunks":[]}"#)
                .expect_err("repair metadata row count mismatch should fail")
                .to_string()
                .contains("row count")
        );

        std::fs::remove_dir_all(schema_mismatch_directory_path).expect("resume test directory should be removed");
        std::fs::remove_dir_all(batch_count_directory_path).expect("resume test directory should be removed");
        std::fs::remove_dir_all(row_count_directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn strict_resume_rejects_arrow_chunks_without_commit_metadata() {
        let directory_path = create_test_directory();
        let schema = Arc::new(Schema::new(vec![Field::new("chunk_identifier", DataType::Int64, false)]));
        write_arrow_file(
            &directory_path.join("chunk_without_metadata.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );

        assert!(
            scan_committed_chunk_identifiers(&directory_path)
                .expect_err("metadata-free Arrow chunks should fail")
                .to_string()
                .contains("missing chunk commit metadata")
        );
        assert!(
            repair_strict_manifest_chunk_commits(&directory_path, r#"{"committed_chunks":[]}"#)
                .expect_err("metadata-free Arrow chunks should fail")
                .to_string()
                .contains("missing chunk commit metadata")
        );
        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn strict_manifest_reports_missing_metadata_commit_as_commit_set_mismatch() {
        let directory_path = create_test_directory();
        let schema = schema_metadata_with_commits(
            r#"[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1}]"#,
        );
        write_arrow_file(
            &directory_path.join("chunk_0.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        let manifest = r#"{"committed_chunks":[{"chunk_identifier":7,"variant_start_index":7,"variant_stop_index":8,"row_count":1,"chunk_file_name":"chunk_0.arrow"}]}"#;

        let error = validate_strict_manifest_chunks(&directory_path, manifest)
            .expect_err("missing metadata commit should fail")
            .to_string();

        assert!(error.contains("commit set"));

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }
}
