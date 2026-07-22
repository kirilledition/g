use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::error::{OutputError, OutputResult};
use crate::persistence::model::OutputChunkCommit;

pub(super) fn insert_or_validate_chunk_commit(
    committed_chunks_by_identifier: &mut BTreeMap<i64, OutputChunkCommit>,
    chunk_commit: OutputChunkCommit,
) -> OutputResult<()> {
    match committed_chunks_by_identifier.get(&chunk_commit.chunk_identifier) {
        Some(existing_commit) if existing_commit != &chunk_commit => Err(OutputError::InvalidInput(format!(
            "Run manifest has conflicting commit metadata for chunk {}.",
            chunk_commit.chunk_identifier
        ))),
        Some(_) => Ok(()),
        None => {
            committed_chunks_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit);
            Ok(())
        }
    }
}

pub(super) fn chunk_commit_to_value(chunk_commit: &OutputChunkCommit) -> Value {
    json!({
        "chunk_identifier": chunk_commit.chunk_identifier,
        "variant_start_index": chunk_commit.variant_start_index,
        "variant_stop_index": chunk_commit.variant_stop_index,
        "row_count": chunk_commit.row_count,
        "chunk_file_name": chunk_commit.chunk_file_name,
    })
}

pub(super) fn read_run_manifest_chunk_commit(committed_chunk: &Value) -> OutputResult<OutputChunkCommit> {
    let chunk_file_name = committed_chunk.get("chunk_file_name").and_then(Value::as_str).ok_or_else(|| {
        OutputError::InvalidInput("Run manifest committed chunk entry is missing chunk_file_name.".to_string())
    })?;
    read_chunk_commit(committed_chunk, chunk_file_name)
}

pub(crate) fn read_chunk_commits_from_text(
    chunk_commits_text: &str,
    chunk_file_name: &str,
) -> OutputResult<Vec<OutputChunkCommit>> {
    let chunk_commit_values = serde_json::from_str::<Value>(chunk_commits_text).map_err(OutputError::runtime)?;
    let chunk_commit_array = chunk_commit_values
        .as_array()
        .ok_or_else(|| OutputError::Runtime("Rust output writer chunk commit metadata must be a list.".to_string()))?;
    chunk_commit_array.iter().map(|chunk_commit| read_chunk_commit(chunk_commit, chunk_file_name)).collect()
}

fn read_chunk_commit(committed_chunk: &Value, chunk_file_name: &str) -> OutputResult<OutputChunkCommit> {
    Ok(OutputChunkCommit {
        chunk_identifier: read_manifest_integer(committed_chunk, "chunk_identifier")?,
        variant_start_index: read_manifest_integer(committed_chunk, "variant_start_index")?,
        variant_stop_index: read_manifest_integer(committed_chunk, "variant_stop_index")?,
        row_count: read_manifest_non_negative_integer(committed_chunk, "row_count")?,
        chunk_file_name: chunk_file_name.to_string(),
    })
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> OutputResult<i64> {
    committed_chunk.get(field_name).and_then(Value::as_i64).ok_or_else(|| {
        OutputError::InvalidInput(format!("Run manifest committed chunk entry is missing {field_name}."))
    })
}

fn read_manifest_non_negative_integer(committed_chunk: &Value, field_name: &str) -> OutputResult<i64> {
    let value = read_manifest_integer(committed_chunk, field_name)?;
    if value < 0 {
        return Err(OutputError::InvalidInput(format!(
            "Run manifest committed chunk entry {field_name} must be non-negative."
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{
        chunk_commit_to_value, insert_or_validate_chunk_commit, read_chunk_commits_from_text,
        read_run_manifest_chunk_commit,
    };
    use crate::persistence::model::OutputChunkCommit;

    fn chunk_commit(chunk_identifier: i64, chunk_file_name: &str) -> OutputChunkCommit {
        OutputChunkCommit {
            chunk_identifier,
            variant_start_index: chunk_identifier,
            variant_stop_index: chunk_identifier + 3,
            row_count: 3,
            chunk_file_name: chunk_file_name.to_string(),
        }
    }

    #[test]
    fn chunk_commit_round_trip_preserves_public_geometry() {
        let expected_commit = chunk_commit(12, "part_000000012.parquet");
        let value = chunk_commit_to_value(&expected_commit);

        assert_eq!(read_run_manifest_chunk_commit(&value).expect("commit reads"), expected_commit);
    }

    #[test]
    fn insertion_accepts_identical_replay_and_rejects_conflicting_replay() {
        let mut commits = BTreeMap::new();
        let original_commit = chunk_commit(12, "part_000000012.parquet");
        insert_or_validate_chunk_commit(&mut commits, chunk_commit(12, "part_000000012.parquet"))
            .expect("first commit inserts");
        insert_or_validate_chunk_commit(&mut commits, chunk_commit(12, "part_000000012.parquet"))
            .expect("identical replay is idempotent");
        assert_eq!(commits.get(&12), Some(&original_commit));

        let error = insert_or_validate_chunk_commit(&mut commits, chunk_commit(12, "different.parquet"))
            .expect_err("conflicting replay is rejected");
        assert!(error.to_string().contains("conflicting commit metadata"));
    }

    #[test]
    fn parquet_footer_commit_parser_uses_observed_part_name() {
        let footer = json!([{
            "chunk_identifier": 4,
            "variant_start_index": 4,
            "variant_stop_index": 7,
            "row_count": 3,
            "chunk_file_name": "untrusted-name.parquet",
        }]);

        let commits = read_chunk_commits_from_text(&footer.to_string(), "observed.parquet").expect("footer reads");
        assert_eq!(commits, [chunk_commit(4, "observed.parquet")]);
    }

    #[test]
    fn malformed_chunk_commits_report_each_required_contract() {
        let malformed_cases = [
            (json!({}), "chunk_file_name"),
            (json!({"chunk_file_name": "part.parquet"}), "chunk_identifier"),
            (
                json!({
                    "chunk_file_name": "part.parquet",
                    "chunk_identifier": 0,
                    "variant_start_index": 0,
                }),
                "variant_stop_index",
            ),
            (
                json!({
                    "chunk_file_name": "part.parquet",
                    "chunk_identifier": 0,
                    "variant_start_index": 0,
                    "variant_stop_index": 1,
                }),
                "row_count",
            ),
            (
                json!({
                    "chunk_file_name": "part.parquet",
                    "chunk_identifier": 0,
                    "variant_start_index": 0,
                    "variant_stop_index": 1,
                    "row_count": -1,
                }),
                "non-negative",
            ),
        ];

        for (value, expected_message) in malformed_cases {
            let error = read_run_manifest_chunk_commit(&value).expect_err("malformed commit is rejected");
            assert!(error.to_string().contains(expected_message), "unexpected error: {error}");
        }

        let error = read_chunk_commits_from_text("{}", "part.parquet").expect_err("footer must be a list");
        assert!(error.to_string().contains("must be a list"));
        assert!(read_chunk_commits_from_text("not-json", "part.parquet").is_err());
    }
}
