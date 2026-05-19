use std::sync::{Arc, OnceLock};

use arrow::array::{StringArray, StringDictionaryBuilder};
use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema};

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
    let mut builder = StringDictionaryBuilder::<Int8Type>::new();
    for _ in 0..row_count {
        builder.append(value).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

pub(crate) fn build_extra_string_array(extra_code: Vec<Option<i32>>) -> Result<StringArray, String> {
    let mut values: Vec<Option<&str>> = Vec::with_capacity(extra_code.len());
    for maybe_extra_code_value in extra_code {
        match maybe_extra_code_value {
            None | Some(0) => values.push(None),
            Some(1) => values.push(Some("FIRTH")),
            Some(2) => values.push(Some("SPA")),
            Some(3) => values.push(Some("TEST_FAIL")),
            Some(extra_code_value) => return Err(format!("Unsupported REGENIE step 2 extra code: {extra_code_value}")),
        }
    }
    Ok(StringArray::from(values))
}

pub(crate) fn get_regenie_step2_final_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_FINAL_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_FINAL_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_final_schema()))
}

fn build_regenie_step2_chunk_schema() -> Schema {
    let large_dictionary_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let small_dictionary_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    Schema::new(vec![
        Field::new("chunk_identifier", DataType::Int64, true),
        Field::new("variant_start_index", DataType::Int64, true),
        Field::new("variant_stop_index", DataType::Int64, true),
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
