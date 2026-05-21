use std::sync::{Arc, OnceLock};

use arrow::array::{ArrayRef, DictionaryArray, Int8Array, StringArray, StringDictionaryBuilder, new_null_array};
use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema};

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";

pub(crate) fn get_regenie_step2_chunk_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_CHUNK_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_CHUNK_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_chunk_schema()))
}

pub(crate) fn build_dictionary_string_array(
    values: &[String],
) -> Result<arrow::array::DictionaryArray<Int32Type>, String> {
    let mut builder = StringDictionaryBuilder::<Int32Type>::new();
    for value in values {
        builder.append(value).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

pub(crate) fn build_constant_dictionary_string_array(
    row_count: usize,
    value: &str,
) -> Result<arrow::array::DictionaryArray<Int8Type>, String> {
    DictionaryArray::<Int8Type>::try_new(
        Int8Array::from(vec![0_i8; row_count]),
        Arc::new(StringArray::from(vec![value])),
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn build_extra_string_array(extra_code: Option<Vec<i32>>, row_count: usize) -> Result<ArrayRef, String> {
    let Some(extra_code_values) = extra_code else {
        return Ok(new_null_array(&DataType::Utf8, row_count));
    };
    let mut extra_strings: Vec<Option<&str>> = Vec::with_capacity(extra_code_values.len());
    for extra_code_value in extra_code_values {
        match extra_code_value {
            0 => extra_strings.push(None),
            1 => extra_strings.push(Some("FIRTH")),
            2 => extra_strings.push(Some("SPA")),
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
    let large_dictionary_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let small_dictionary_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    Schema::new(vec![
        Field::new("CHROM", large_dictionary_type.clone(), true),
        Field::new("GENPOS", DataType::Int64, true),
        Field::new("ID", DataType::Utf8, true),
        Field::new("ALLELE0", large_dictionary_type.clone(), true),
        Field::new("ALLELE1", large_dictionary_type.clone(), true),
        Field::new("A1FREQ", DataType::Float32, true),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, true),
        Field::new("TEST", small_dictionary_type.clone(), true),
        Field::new("BETA", DataType::Float32, true),
        Field::new("SE", DataType::Float32, true),
        Field::new("CHISQ", DataType::Float32, true),
        Field::new("LOG10P", DataType::Float32, true),
        Field::new("EXTRA", DataType::Utf8, true),
    ])
}

fn build_regenie_step2_final_schema() -> Schema {
    let large_dictionary_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let small_dictionary_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    Schema::new(vec![
        Field::new("CHROM", large_dictionary_type.clone(), true),
        Field::new("GENPOS", DataType::Int64, true),
        Field::new("ID", DataType::Utf8, true),
        Field::new("ALLELE0", large_dictionary_type.clone(), true),
        Field::new("ALLELE1", large_dictionary_type.clone(), true),
        Field::new("A1FREQ", DataType::Float32, true),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, true),
        Field::new("TEST", small_dictionary_type.clone(), true),
        Field::new("BETA", DataType::Float32, true),
        Field::new("SE", DataType::Float32, true),
        Field::new("CHISQ", DataType::Float32, true),
        Field::new("LOG10P", DataType::Float32, true),
        Field::new("EXTRA", DataType::Utf8, true),
    ])
}
