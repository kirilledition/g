use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::OutputError;
use crate::persistence::model::OutputChunkCommit;
use crate::{manifest, schema};

pub(crate) fn repair_strict_manifest_chunk_commits(
    parts_directory: &Path,
    manifest_json: &str,
    planned_chunk_ranges: &[Range<usize>],
) -> Result<Vec<OutputChunkCommit>, OutputError> {
    let planned_chunk_stops_by_start = build_planned_chunk_stops_by_start(planned_chunk_ranges)?;
    let mut repaired_commits = BTreeMap::new();
    for chunk_commit in manifest::read_run_manifest_chunk_commits_from_text(manifest_json)? {
        validate_chunk_commit_geometry(&chunk_commit, &planned_chunk_stops_by_start)?;
        repaired_commits.insert(chunk_commit.chunk_identifier, chunk_commit);
    }
    let scanned_commits = scan_committed_chunk_commits(parts_directory, &planned_chunk_stops_by_start)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    for existing_commit in repaired_commits.values() {
        let chunk_file_path = parts_directory.join(&existing_commit.chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        match scanned_commits.get(&existing_commit.chunk_identifier) {
            Some(scanned_commit) if scanned_commit == existing_commit => {}
            Some(_) => {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {}.",
                    existing_commit.chunk_identifier
                )));
            }
            None => {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume manifest references unobserved commit metadata for chunk {}.",
                    existing_commit.chunk_identifier
                )));
            }
        }
    }
    for (chunk_identifier, chunk_commit) in scanned_commits {
        if let Some(existing_commit) = repaired_commits.get(&chunk_identifier) {
            if existing_commit != &chunk_commit {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {chunk_identifier}."
                )));
            }
        } else {
            repaired_commits.insert(chunk_identifier, chunk_commit);
        }
    }
    Ok(repaired_commits.into_values().collect())
}

struct PartCommitObservation {
    schema: Arc<Schema>,
    chunk_commits: Vec<OutputChunkCommit>,
}

fn scan_committed_chunk_commits(
    parts_directory: &Path,
    planned_chunk_stops_by_start: &BTreeMap<usize, usize>,
) -> Result<Vec<OutputChunkCommit>, OutputError> {
    if !parts_directory.exists() {
        return Ok(Vec::new());
    }
    let mut part_paths = Vec::new();
    for directory_entry in std::fs::read_dir(parts_directory).map_err(OutputError::runtime)? {
        let part_path = directory_entry.map_err(OutputError::runtime)?.path();
        if part_path.extension().is_some_and(|extension| extension == "parquet") {
            part_paths.push(part_path);
        }
    }
    part_paths.sort();

    let mut chunk_commits = BTreeMap::new();
    for part_path in part_paths {
        let observation = inspect_parquet_part(&part_path)?;
        // Parquet exposes file key-value metadata through the reconstructed
        // Arrow schema. Chunk-commit metadata is validated separately below;
        // logical compatibility is the ordered field contract.
        if observation.schema.fields() != schema::REGENIE_STEP2_CHUNK_SCHEMA.fields() {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume found an incompatible schema in {}.",
                part_path.display()
            )));
        }
        for chunk_commit in observation.chunk_commits {
            validate_chunk_commit_geometry(&chunk_commit, planned_chunk_stops_by_start)?;
            if chunk_commits.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
                return Err(OutputError::InvalidInput(
                    "Strict resume found duplicate commit metadata for a chunk.".to_string(),
                ));
            }
        }
    }
    Ok(chunk_commits.into_values().collect())
}

fn build_planned_chunk_stops_by_start(
    planned_chunk_ranges: &[Range<usize>],
) -> Result<BTreeMap<usize, usize>, OutputError> {
    let mut planned_chunk_stops_by_start = BTreeMap::new();
    for chunk_range in planned_chunk_ranges {
        if chunk_range.start >= chunk_range.end {
            return Err(OutputError::InvalidInput(format!(
                "Planned output chunk range {}..{} is empty or reversed.",
                chunk_range.start, chunk_range.end
            )));
        }
        if planned_chunk_stops_by_start.insert(chunk_range.start, chunk_range.end).is_some() {
            return Err(OutputError::InvalidInput(format!(
                "Planned output chunk geometry has duplicate start index {}.",
                chunk_range.start
            )));
        }
    }
    Ok(planned_chunk_stops_by_start)
}

fn validate_chunk_commit_geometry(
    chunk_commit: &OutputChunkCommit,
    planned_chunk_stops_by_start: &BTreeMap<usize, usize>,
) -> Result<(), OutputError> {
    if chunk_commit.chunk_identifier != chunk_commit.variant_start_index {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume chunk {} does not identify its variant start index {}.",
            chunk_commit.chunk_identifier, chunk_commit.variant_start_index
        )));
    }
    let chunk_start = usize::try_from(chunk_commit.variant_start_index).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Strict resume chunk {} has an out-of-bounds start index.",
            chunk_commit.chunk_identifier
        ))
    })?;
    let chunk_stop = usize::try_from(chunk_commit.variant_stop_index).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Strict resume chunk {} has an out-of-bounds stop index.",
            chunk_commit.chunk_identifier
        ))
    })?;
    let row_count = usize::try_from(chunk_commit.row_count).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Strict resume chunk {} has an out-of-bounds row count.",
            chunk_commit.chunk_identifier
        ))
    })?;
    let Some(expected_chunk_stop) = planned_chunk_stops_by_start.get(&chunk_start).copied() else {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume chunk {} is not present in the current BGEN chunk plan.",
            chunk_commit.chunk_identifier
        )));
    };
    let expected_row_count = expected_chunk_stop.checked_sub(chunk_start).ok_or_else(|| {
        OutputError::InvalidInput(format!(
            "Planned output chunk range {chunk_start}..{expected_chunk_stop} is reversed."
        ))
    })?;
    if chunk_stop != expected_chunk_stop || row_count != expected_row_count {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume chunk {} geometry does not match the current BGEN chunk plan.",
            chunk_commit.chunk_identifier
        )));
    }
    Ok(())
}

fn inspect_parquet_part(part_path: &Path) -> Result<PartCommitObservation, OutputError> {
    let input_file = File::open(part_path).map_err(OutputError::runtime)?;
    let parquet_arrow_reader = ParquetRecordBatchReaderBuilder::try_new(input_file).map_err(OutputError::runtime)?;
    let part_schema = parquet_arrow_reader.schema().clone();
    let file_metadata = parquet_arrow_reader.metadata().file_metadata();

    let part_file_name = part_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputError::Runtime("Parquet part file name is not UTF-8.".to_string()))?
        .to_string();
    let chunk_commit_text = file_metadata
        .key_value_metadata()
        .and_then(|metadata| metadata.iter().find(|entry| entry.key == schema::CHUNK_COMMITS_METADATA_KEY))
        .and_then(|entry| entry.value.as_deref())
        .ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Strict resume Parquet part is missing chunk commit metadata: {}",
                part_path.display()
            ))
        })?;
    let chunk_commits = manifest::read_chunk_commits_from_text(chunk_commit_text, &part_file_name)?;
    let committed_row_count = chunk_commits
        .iter()
        .try_fold(0_i64, |total, chunk_commit| total.checked_add(chunk_commit.row_count))
        .ok_or_else(|| OutputError::Runtime("Parquet committed row count overflowed.".to_string()))?;
    if committed_row_count != file_metadata.num_rows() {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume row count mismatch for Parquet part {part_file_name}."
        )));
    }
    Ok(PartCommitObservation { schema: part_schema, chunk_commits })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::KeyValue;

    use super::{
        build_planned_chunk_stops_by_start, repair_strict_manifest_chunk_commits, scan_committed_chunk_commits,
        validate_chunk_commit_geometry,
    };
    use crate::persistence::model::OutputChunkCommit;

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);
            let sequence = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp =
                SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
            let path = std::env::temp_dir()
                .join(format!("g-output-resume-schema-{}-{timestamp}-{sequence}", std::process::id()));
            std::fs::create_dir_all(&path).expect("test directory is created");
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_empty_parquet_part(parts_directory: &Path, file_name: &str, schema: Schema) {
        let output_file = File::create(parts_directory.join(file_name)).expect("part file creates");
        let mut writer = ArrowWriter::try_new(output_file, Arc::new(schema), None).expect("Arrow writer initializes");
        writer.append_key_value_metadata(KeyValue {
            key: crate::schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
            value: Some("[]".to_string()),
        });
        writer.close().expect("empty part closes");
    }

    fn chunk_commit(
        chunk_identifier: i64,
        variant_start_index: i64,
        variant_stop_index: i64,
        row_count: i64,
    ) -> OutputChunkCommit {
        OutputChunkCommit {
            chunk_identifier,
            variant_start_index,
            variant_stop_index,
            row_count,
            chunk_file_name: "part.parquet".to_string(),
        }
    }

    #[test]
    fn planned_chunk_geometry_rejects_empty_reversed_and_duplicate_starts() {
        let valid = build_planned_chunk_stops_by_start(&[0..3, 3..5]).expect("valid plan builds");
        assert_eq!(valid, BTreeMap::from([(0, 3), (3, 5)]));

        let reversed_start = 3;
        let reversed_stop = 2;
        for range in [2..2, reversed_start..reversed_stop] {
            let error = build_planned_chunk_stops_by_start(std::slice::from_ref(&range))
                .expect_err("invalid range is rejected");
            assert!(error.to_string().contains("empty or reversed"));
        }
        let error = build_planned_chunk_stops_by_start(&[0..2, 0..3]).expect_err("duplicate start is rejected");
        assert!(error.to_string().contains("duplicate start index 0"));
    }

    #[test]
    fn strict_geometry_accepts_exact_plan_and_rejects_identity_or_shape_changes() {
        let planned_stops = BTreeMap::from([(4, 7)]);
        validate_chunk_commit_geometry(&chunk_commit(4, 4, 7, 3), &planned_stops).expect("exact geometry is valid");

        let cases = [
            (chunk_commit(5, 4, 7, 3), "does not identify its variant start"),
            (chunk_commit(-1, -1, 7, 3), "out-of-bounds start"),
            (chunk_commit(4, 4, -1, 3), "out-of-bounds stop"),
            (chunk_commit(4, 4, 7, -1), "out-of-bounds row count"),
            (chunk_commit(8, 8, 10, 2), "not present in the current BGEN chunk plan"),
            (chunk_commit(4, 4, 8, 4), "geometry does not match"),
            (chunk_commit(4, 4, 7, 2), "geometry does not match"),
        ];
        for (commit, expected_message) in cases {
            let error = validate_chunk_commit_geometry(&commit, &planned_stops).expect_err("bad geometry is rejected");
            assert!(error.to_string().contains(expected_message), "unexpected error: {error}");
        }
    }

    #[test]
    fn strict_repair_accepts_empty_manifest_when_parts_directory_is_absent() {
        let directory = TestDirectory::new();
        let missing_parts = directory.path.join("missing-parts");
        let planned_range = 0..3;
        let repaired = repair_strict_manifest_chunk_commits(
            &missing_parts,
            r#"{"committed_chunks": []}"#,
            std::slice::from_ref(&planned_range),
        )
        .expect("empty missing parts repair is valid");
        assert!(repaired.is_empty());
    }

    #[test]
    fn strict_repair_rejects_duplicate_manifest_identifiers_before_scanning_parts() {
        let manifest = r#"{
            "committed_chunks": [
                {"chunk_identifier": 0, "variant_start_index": 0, "variant_stop_index": 3, "row_count": 3, "chunk_file_name": "first.parquet"},
                {"chunk_identifier": 0, "variant_start_index": 0, "variant_stop_index": 3, "row_count": 3, "chunk_file_name": "second.parquet"}
            ]
        }"#;
        let planned_range = 0..3;
        let error =
            repair_strict_manifest_chunk_commits(Path::new("unused"), manifest, std::slice::from_ref(&planned_range))
                .expect_err("duplicate identifier is rejected");
        assert!(error.to_string().contains("duplicate chunk identifiers"));
    }

    #[test]
    fn strict_scan_rejects_parquet_field_type_order_and_nullability_changes() {
        let canonical_fields = crate::schema::REGENIE_STEP2_CHUNK_SCHEMA.fields().iter().cloned().collect::<Vec<_>>();
        let mut type_changed = canonical_fields.clone();
        type_changed[1] = Arc::new(Field::new("GENPOS", DataType::Int32, false));
        let mut order_changed = canonical_fields.clone();
        order_changed.swap(0, 1);
        let mut nullability_changed = canonical_fields;
        nullability_changed[6] = Arc::new(Field::new("INFO", DataType::Float32, false));

        for (case_name, fields) in
            [("type", type_changed), ("order", order_changed), ("nullability", nullability_changed)]
        {
            let directory = TestDirectory::new();
            write_empty_parquet_part(&directory.path, &format!("{case_name}.parquet"), Schema::new(fields));
            let error = scan_committed_chunk_commits(&directory.path, &BTreeMap::new())
                .expect_err("incompatible field contract is rejected");
            assert!(error.to_string().contains("incompatible schema"), "unexpected error: {error}");
        }
    }

    #[test]
    fn strict_scan_accepts_canonical_fields_with_footer_metadata() {
        let directory = TestDirectory::new();
        write_empty_parquet_part(
            &directory.path,
            "canonical.parquet",
            crate::schema::REGENIE_STEP2_CHUNK_SCHEMA.as_ref().clone(),
        );

        let commits =
            scan_committed_chunk_commits(&directory.path, &BTreeMap::new()).expect("canonical empty part is accepted");
        assert!(commits.is_empty());
    }
}
