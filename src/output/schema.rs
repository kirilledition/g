use std::sync::{Arc, OnceLock};

use arrow::array::{ArrayRef, StringArray, new_null_array};
use arrow::datatypes::{DataType, Field, Schema};

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";

pub(crate) fn get_regenie_step2_chunk_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_CHUNK_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_CHUNK_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_chunk_schema()))
}

pub(crate) fn build_extra_string_array(extra_code: Option<Vec<i32>>, row_count: usize) -> Result<ArrayRef, String> {
    let Some(extra_code_values) = extra_code else {
        return Ok(new_null_array(&DataType::Utf8, row_count));
    };
    let mut extra_strings: Vec<Option<&str>> = Vec::with_capacity(extra_code_values.len());
    for extra_code_value in extra_code_values {
        match extra_code_value {
            0..=2 => extra_strings.push(None),
            3 => extra_strings.push(Some("TEST_FAIL")),
            unsupported_extra_code_value => {
                return Err(format!("Unsupported REGENIE step 2 extra code: {unsupported_extra_code_value}"));
            }
        }
    }
    Ok(Arc::new(StringArray::from(extra_strings)))
}

pub(crate) fn get_regenie_step2_final_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_FINAL_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_FINAL_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_final_schema()))
}

fn build_regenie_step2_chunk_schema() -> Schema {
    Schema::new(vec![
        Field::new("CHROM", DataType::Utf8, true),
        Field::new("GENPOS", DataType::Int64, true),
        Field::new("ID", DataType::Utf8, true),
        Field::new("ALLELE0", DataType::Utf8, true),
        Field::new("ALLELE1", DataType::Utf8, true),
        Field::new("A1FREQ", DataType::Float32, true),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, true),
        Field::new("TEST", DataType::Utf8, true),
        Field::new("BETA", DataType::Float32, true),
        Field::new("SE", DataType::Float32, true),
        Field::new("CHISQ", DataType::Float32, true),
        Field::new("LOG10P", DataType::Float32, true),
        Field::new("EXTRA", DataType::Utf8, true),
    ])
}

fn build_regenie_step2_final_schema() -> Schema {
    Schema::new(vec![
        Field::new("CHROM", DataType::Utf8, true),
        Field::new("GENPOS", DataType::Int64, true),
        Field::new("ID", DataType::Utf8, true),
        Field::new("ALLELE0", DataType::Utf8, true),
        Field::new("ALLELE1", DataType::Utf8, true),
        Field::new("A1FREQ", DataType::Float32, true),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, true),
        Field::new("TEST", DataType::Utf8, true),
        Field::new("BETA", DataType::Float32, true),
        Field::new("SE", DataType::Float32, true),
        Field::new("CHISQ", DataType::Float32, true),
        Field::new("LOG10P", DataType::Float32, true),
        Field::new("EXTRA", DataType::Utf8, true),
    ])
}
