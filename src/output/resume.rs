#![allow(clippy::missing_errors_doc)]

#[cfg(any(test, feature = "python"))]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::Schema;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use serde_json::Value;

use crate::output::manifest;
use crate::output::schema;
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
        let input_file = File::open(&chunk_file_path).map_err(OutputWriterError::runtime)?;
        let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
        if let Some(chunk_commits) = read_schema_chunk_commits(file_reader.schema().as_ref())? {
            validate_schema_chunk_commit_batches(file_reader, &chunk_commits)?;
            committed_identifiers.extend(chunk_commits.into_iter().map(|chunk_commit| chunk_commit.chunk_identifier));
            continue;
        }
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
    let manifest_commits = read_manifest_chunk_commits(manifest_json)?;
    let mut committed_identifiers = BTreeSet::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for committed_chunk in manifest_commits {
        let chunk_file_path = chunks_directory.join(&committed_chunk.chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        let chunk_observation = inspect_manifest_chunk_file(&chunk_file_path, committed_chunk.chunk_identifier)?;
        match expected_schema.as_ref() {
            Some(expected_schema) if expected_schema.fields() != chunk_observation.schema.fields() => {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume found incompatible Arrow schema in {}.",
                    chunk_file_path.display()
                )));
            }
            None => expected_schema = Some(Arc::clone(&chunk_observation.schema)),
            Some(_) => {}
        }
        let row_count = i64::try_from(committed_chunk.row_count).map_err(OutputWriterError::runtime)?;
        if chunk_observation.row_count != row_count {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume row count mismatch for chunk {}.",
                committed_chunk.chunk_identifier
            )));
        }
        if chunk_observation.variant_start_index != Some(committed_chunk.variant_start_index)
            || chunk_observation.variant_stop_index != Some(committed_chunk.variant_stop_index)
        {
            return Err(OutputWriterError::InvalidInput(format!(
                "Strict resume variant range mismatch for chunk {}.",
                committed_chunk.chunk_identifier
            )));
        }
        committed_identifiers.insert(committed_chunk.chunk_identifier);
    }
    Ok(committed_identifiers.into_iter().collect())
}

#[cfg(any(test, feature = "python"))]
pub(crate) fn repair_strict_manifest_chunk_commits(
    chunks_directory: &Path,
    manifest_json: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    validate_strict_manifest_chunks(chunks_directory, manifest_json)?;
    let mut repaired_commits = read_manifest_chunk_commits(manifest_json)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    for chunk_commit in scan_committed_chunk_commits(chunks_directory)? {
        match repaired_commits.get(&chunk_commit.chunk_identifier) {
            Some(existing_commit) if existing_commit != &chunk_commit => {
                return Err(OutputWriterError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {}.",
                    chunk_commit.chunk_identifier
                )));
            }
            Some(_) => {}
            None => {
                repaired_commits.insert(chunk_commit.chunk_identifier, chunk_commit);
            }
        }
    }
    Ok(repaired_commits.into_values().collect())
}

struct ManifestChunkObservation {
    schema: Arc<Schema>,
    row_count: i64,
    variant_start_index: Option<i64>,
    variant_stop_index: Option<i64>,
}

#[derive(Clone)]
struct ChunkCommitObservation {
    chunk_identifier: i64,
    variant_start_index: i64,
    variant_stop_index: i64,
    row_count: i64,
}

#[cfg(any(test, feature = "python"))]
struct ChunkFileCommitObservation {
    schema: Arc<Schema>,
    chunk_commits: Vec<manifest::RunManifestChunkCommit>,
}

fn read_manifest_chunk_commits(
    manifest_json: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    let manifest = serde_json::from_str::<Value>(manifest_json).map_err(OutputWriterError::runtime)?;
    let committed_chunks = manifest.get("committed_chunks").and_then(Value::as_array).ok_or_else(|| {
        OutputWriterError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string())
    })?;
    committed_chunks
        .iter()
        .map(|committed_chunk| {
            let chunk_file_name = committed_chunk.get("chunk_file_name").and_then(Value::as_str).ok_or_else(|| {
                OutputWriterError::InvalidInput(
                    "Run manifest committed chunk entry is missing chunk_file_name.".to_string(),
                )
            })?;
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: read_manifest_integer(committed_chunk, "chunk_identifier")?,
                variant_start_index: read_manifest_integer(committed_chunk, "variant_start_index")?,
                variant_stop_index: read_manifest_integer(committed_chunk, "variant_stop_index")?,
                row_count: read_manifest_usize(committed_chunk, "row_count")?,
                chunk_file_name: chunk_file_name.to_string(),
            })
        })
        .collect()
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> Result<i64, OutputWriterError> {
    committed_chunk.get(field_name).and_then(Value::as_i64).ok_or_else(|| {
        OutputWriterError::InvalidInput(format!("Run manifest committed chunk entry is missing {field_name}."))
    })
}

fn read_manifest_usize(committed_chunk: &Value, field_name: &str) -> Result<usize, OutputWriterError> {
    let value = read_manifest_integer(committed_chunk, field_name)?;
    usize::try_from(value).map_err(|_| {
        OutputWriterError::InvalidInput(format!(
            "Run manifest committed chunk entry {field_name} must be non-negative."
        ))
    })
}

#[cfg(any(test, feature = "python"))]
fn scan_committed_chunk_commits(
    chunks_directory: &Path,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    if !chunks_directory.exists() {
        return Ok(Vec::new());
    }
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputWriterError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
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

#[cfg(any(test, feature = "python"))]
fn inspect_chunk_file_commits(chunk_file_path: &Path) -> Result<ChunkFileCommitObservation, OutputWriterError> {
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
    let schema = file_reader.schema();
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputWriterError::Runtime("Rust output writer chunk file name is not UTF-8.".to_string()))?
        .to_string();
    let chunk_commits = if let Some(chunk_commits) = read_schema_chunk_commits(schema.as_ref())? {
        inspect_metadata_chunk_file_commits(file_reader, chunk_commits, &chunk_file_name)?
    } else {
        inspect_legacy_chunk_file_commits(file_reader, &chunk_file_name)?
    };
    Ok(ChunkFileCommitObservation { schema, chunk_commits })
}

#[cfg(any(test, feature = "python"))]
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
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: usize::try_from(chunk_commit.row_count).map_err(OutputWriterError::runtime)?,
            chunk_file_name: chunk_file_name.to_string(),
        });
    }
    Ok(manifest_commits)
}

#[cfg(any(test, feature = "python"))]
fn inspect_legacy_chunk_file_commits(
    file_reader: ArrowFileReader<File>,
    chunk_file_name: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
    let mut observations = BTreeMap::<i64, ChunkCommitObservation>::new();
    for maybe_batch in file_reader {
        let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
        let chunk_identifier_array = read_int64_column(&batch, "chunk_identifier")?;
        let variant_start_array = read_int64_column(&batch, "variant_start_index")?;
        let variant_stop_array = read_int64_column(&batch, "variant_stop_index")?;
        for row_index in 0..chunk_identifier_array.len() {
            if chunk_identifier_array.is_null(row_index) {
                continue;
            }
            let chunk_identifier = chunk_identifier_array.value(row_index);
            let variant_start_index = read_required_int64_value(variant_start_array, row_index, "variant_start_index")?;
            let variant_stop_index = read_required_int64_value(variant_stop_array, row_index, "variant_stop_index")?;
            observations
                .entry(chunk_identifier)
                .and_modify(|observation| {
                    observation.variant_start_index = observation.variant_start_index.min(variant_start_index);
                    observation.variant_stop_index = observation.variant_stop_index.max(variant_stop_index);
                    observation.row_count += 1;
                })
                .or_insert(ChunkCommitObservation {
                    chunk_identifier,
                    variant_start_index,
                    variant_stop_index,
                    row_count: 1,
                });
        }
    }
    observations
        .into_values()
        .map(|chunk_commit| {
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: chunk_commit.chunk_identifier,
                variant_start_index: chunk_commit.variant_start_index,
                variant_stop_index: chunk_commit.variant_stop_index,
                row_count: usize::try_from(chunk_commit.row_count).map_err(OutputWriterError::runtime)?,
                chunk_file_name: chunk_file_name.to_string(),
            })
        })
        .collect()
}

#[cfg(any(test, feature = "python"))]
fn read_required_int64_value(
    column: &Int64Array,
    row_index: usize,
    column_name: &str,
) -> Result<i64, OutputWriterError> {
    if column.is_null(row_index) {
        return Err(OutputWriterError::Runtime(format!("Rust output writer found null {column_name} in Arrow chunk.")));
    }
    Ok(column.value(row_index))
}

fn read_schema_chunk_commits(chunk_schema: &Schema) -> Result<Option<Vec<ChunkCommitObservation>>, OutputWriterError> {
    let Some(chunk_commits_text) = chunk_schema.metadata().get(schema::CHUNK_COMMITS_METADATA_KEY) else {
        return Ok(None);
    };
    let chunk_commit_values = serde_json::from_str::<Value>(chunk_commits_text).map_err(OutputWriterError::runtime)?;
    let chunk_commit_array = chunk_commit_values.as_array().ok_or_else(|| {
        OutputWriterError::Runtime("Rust output writer chunk commit metadata must be a list.".to_string())
    })?;
    let mut chunk_commits = Vec::with_capacity(chunk_commit_array.len());
    for chunk_commit_value in chunk_commit_array {
        chunk_commits.push(ChunkCommitObservation {
            chunk_identifier: read_manifest_integer(chunk_commit_value, "chunk_identifier")?,
            variant_start_index: read_manifest_integer(chunk_commit_value, "variant_start_index")?,
            variant_stop_index: read_manifest_integer(chunk_commit_value, "variant_stop_index")?,
            row_count: read_manifest_integer(chunk_commit_value, "row_count")?,
        });
    }
    Ok(Some(chunk_commits))
}

fn validate_schema_chunk_commit_batches(
    file_reader: ArrowFileReader<File>,
    chunk_commits: &[ChunkCommitObservation],
) -> Result<(), OutputWriterError> {
    let mut batch_row_counts = Vec::with_capacity(chunk_commits.len());
    for maybe_batch in file_reader {
        let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
        batch_row_counts.push(i64::try_from(batch.num_rows()).map_err(OutputWriterError::runtime)?);
    }
    if batch_row_counts.len() != chunk_commits.len() {
        return Err(OutputWriterError::InvalidInput(
            "Arrow chunk commit metadata batch count does not match the file batches.".to_string(),
        ));
    }
    for (observed_row_count, observed_chunk_commit) in batch_row_counts.iter().zip(chunk_commits.iter()) {
        if *observed_row_count != observed_chunk_commit.row_count {
            return Err(OutputWriterError::InvalidInput(format!(
                "Arrow chunk commit metadata row count mismatch for chunk {}.",
                observed_chunk_commit.chunk_identifier
            )));
        }
    }
    Ok(())
}

fn inspect_manifest_chunk_file(
    chunk_file_path: &Path,
    chunk_identifier: i64,
) -> Result<ManifestChunkObservation, OutputWriterError> {
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
    let schema = file_reader.schema();
    if let Some(chunk_commits) = read_schema_chunk_commits(schema.as_ref())? {
        return inspect_metadata_manifest_chunk_file(file_reader, schema, &chunk_commits, chunk_identifier);
    }
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

fn inspect_metadata_manifest_chunk_file(
    file_reader: ArrowFileReader<File>,
    schema: Arc<Schema>,
    chunk_commits: &[ChunkCommitObservation],
    chunk_identifier: i64,
) -> Result<ManifestChunkObservation, OutputWriterError> {
    let Some(chunk_commit) = chunk_commits.iter().find(|commit| commit.chunk_identifier == chunk_identifier).cloned()
    else {
        return Ok(ManifestChunkObservation {
            schema,
            row_count: 0,
            variant_start_index: None,
            variant_stop_index: None,
        });
    };
    validate_schema_chunk_commit_batches(file_reader, chunk_commits)?;
    Ok(ManifestChunkObservation {
        schema,
        row_count: chunk_commit.row_count,
        variant_start_index: Some(chunk_commit.variant_start_index),
        variant_stop_index: Some(chunk_commit.variant_stop_index),
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::{ArrayRef, Float32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;

    use crate::output::schema as output_schema;

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

    fn required_resume_schema(extra_field: Option<Field>) -> Arc<Schema> {
        let mut fields = vec![
            Field::new("chunk_identifier", DataType::Int64, false),
            Field::new("variant_start_index", DataType::Int64, false),
            Field::new("variant_stop_index", DataType::Int64, false),
        ];
        if let Some(field) = extra_field {
            fields.push(field);
        }
        Arc::new(Schema::new(fields))
    }

    fn nullable_resume_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("chunk_identifier", DataType::Int64, true),
            Field::new("variant_start_index", DataType::Int64, true),
            Field::new("variant_stop_index", DataType::Int64, true),
        ]))
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
    fn scan_reads_single_chunk_identifier_from_arrow_contents() {
        let directory_path = create_test_directory();
        let schema = required_resume_schema(None);
        write_arrow_file(
            &directory_path.join("chunk_000000007.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![4])) as ArrayRef,
            ],
        );

        let committed_identifiers = scan_committed_chunk_identifiers(&directory_path).expect("Arrow chunk should scan");

        assert_eq!(committed_identifiers, vec![3]);

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
    fn scan_rejects_range_chunk_arrow_without_chunk_identifier_column() {
        let directory_path = create_test_directory();
        let schema = Arc::new(Schema::new(vec![Field::new("not_chunk_identifier", DataType::Int64, false)]));
        write_arrow_file(
            &directory_path.join("chunk_0_1.arrow"),
            &schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );

        let error = scan_committed_chunk_identifiers(&directory_path)
            .expect_err("range chunk without chunk_identifier column should fail")
            .to_string();
        assert!(error.contains("chunk identifiers"));

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
    fn strict_manifest_rejects_missing_columns_and_schema_mismatches() {
        let directory_path = create_test_directory();
        let missing_column_schema = Arc::new(Schema::new(vec![Field::new("chunk_identifier", DataType::Int64, false)]));
        write_arrow_file(
            &directory_path.join("chunk_0_0.arrow"),
            &missing_column_schema,
            vec![Arc::new(Int64Array::from(vec![0])) as ArrayRef],
        );
        let missing_column_manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"chunk_0_0.arrow"}]}"#;
        let missing_column_error = validate_strict_manifest_chunks(&directory_path, missing_column_manifest)
            .expect_err("missing variant_start_index should fail")
            .to_string();
        assert!(missing_column_error.contains("variant_start_index"));

        let schema = required_resume_schema(None);
        write_arrow_file(
            &directory_path.join("chunk_1_1.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
            ],
        );
        let extra_schema = required_resume_schema(Some(Field::new("extra", DataType::Float32, false)));
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
        let schema = required_resume_schema(None);
        write_arrow_file(
            &schema_mismatch_directory_path.join("chunk_0.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            ],
        );
        let extra_schema = required_resume_schema(Some(Field::new("extra", DataType::Float32, false)));
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
    fn repair_strict_manifest_reads_legacy_arrow_commits_and_rejects_null_required_values() {
        let directory_path = create_test_directory();
        let schema = nullable_resume_schema();
        write_arrow_file(
            &directory_path.join("chunk_legacy.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(0), Some(0), None, Some(2)])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(0), Some(1), Some(9), Some(2)])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(10), Some(4)])) as ArrayRef,
            ],
        );

        let repaired_commits = repair_strict_manifest_chunk_commits(&directory_path, r#"{"committed_chunks":[]}"#)
            .expect("legacy chunks should repair from contents");

        assert_eq!(
            repaired_commits
                .iter()
                .map(|chunk_commit| {
                    (
                        chunk_commit.chunk_identifier,
                        chunk_commit.variant_start_index,
                        chunk_commit.variant_stop_index,
                        chunk_commit.row_count,
                    )
                })
                .collect::<Vec<_>>(),
            vec![(0, 0, 2, 2), (2, 2, 4, 1)],
        );
        let manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":2,"row_count":2,"chunk_file_name":"chunk_legacy.arrow"}]}"#;
        assert_eq!(
            validate_strict_manifest_chunks(&directory_path, manifest)
                .expect("legacy manifest should validate selected rows"),
            vec![0],
        );

        let null_required_directory_path = create_test_directory();
        write_arrow_file(
            &null_required_directory_path.join("chunk_null.arrow"),
            &schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(0)])) as ArrayRef,
                Arc::new(Int64Array::from(vec![None])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1)])) as ArrayRef,
            ],
        );
        assert!(
            repair_strict_manifest_chunk_commits(&null_required_directory_path, r#"{"committed_chunks":[]}"#)
                .expect_err("null variant_start_index should fail")
                .to_string()
                .contains("null variant_start_index")
        );

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
        std::fs::remove_dir_all(null_required_directory_path).expect("resume test directory should be removed");
    }

    #[test]
    fn strict_manifest_reports_missing_metadata_commit_as_row_count_mismatch() {
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

        assert!(error.contains("row count mismatch"));

        std::fs::remove_dir_all(directory_path).expect("resume test directory should be removed");
    }
}
