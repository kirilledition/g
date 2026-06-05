use std::sync::{Arc, OnceLock};

use arrow::array::{Array, ArrayRef, Int32Array, StringArray, new_null_array};
use arrow::datatypes::{DataType, Field, Schema};

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";

pub(crate) fn get_regenie_step2_chunk_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_CHUNK_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_CHUNK_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_chunk_schema()))
}

pub(crate) fn build_extra_string_array(extra_code: Option<ArrayRef>, row_count: usize) -> Result<ArrayRef, String> {
    let Some(extra_code_array) = extra_code else {
        return Ok(new_null_array(&DataType::Utf8, row_count));
    };
    let extra_code_values = extra_code_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| "REGENIE step 2 extra code must be an int32 array.".to_string())?;
    if extra_code_values.len() != row_count {
        return Err("REGENIE step 2 extra code row count does not match metadata row count.".to_string());
    }
    let mut extra_strings: Vec<Option<&str>> = Vec::with_capacity(extra_code_values.len());
    for row_index in 0..extra_code_values.len() {
        if extra_code_values.is_null(row_index) {
            extra_strings.push(None);
            continue;
        }
        let extra_code_value = extra_code_values.value(row_index);
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, ArrayRef, Float32Array, Int32Array};

    use super::{build_extra_string_array, get_regenie_step2_chunk_schema, get_regenie_step2_final_schema};

    #[test]
    fn extra_string_array_maps_nulls_supported_codes_and_errors() {
        let null_extra = build_extra_string_array(None, 2).expect("missing extra code should create null strings");
        assert_eq!(null_extra.len(), 2);

        let supported_extra =
            build_extra_string_array(Some(Arc::new(Int32Array::from(vec![None, Some(3)])) as ArrayRef), 2)
                .expect("supported extra code should map");
        let supported_extra_values = supported_extra
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .expect("extra should be a string array");
        assert!(supported_extra_values.is_null(0));
        assert_eq!(supported_extra_values.value(1), "TEST_FAIL");

        assert!(
            build_extra_string_array(Some(Arc::new(Int32Array::from(vec![4])) as ArrayRef), 1)
                .expect_err("unsupported extra code should fail")
                .contains("Unsupported")
        );
        assert!(
            build_extra_string_array(Some(Arc::new(Int32Array::from(vec![1])) as ArrayRef), 2)
                .expect_err("extra code length mismatch should fail")
                .contains("row count")
        );
        assert!(
            build_extra_string_array(Some(Arc::new(Float32Array::from(vec![1.0])) as ArrayRef), 1)
                .expect_err("wrong extra code type should fail")
                .contains("int32")
        );
    }

    #[test]
    fn schema_singletons_have_expected_final_columns() {
        assert_eq!(get_regenie_step2_chunk_schema().fields().len(), 14);
        assert_eq!(get_regenie_step2_final_schema().fields().len(), 14);
    }
}
