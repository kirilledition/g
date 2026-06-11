use std::sync::{Arc, OnceLock};

use arrow::array::{Array, ArrayRef, Int32Array, StringArray, StringBuilder, new_null_array};
use arrow::datatypes::{DataType, Field, Schema};

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";

pub(crate) fn get_regenie_step2_chunk_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_CHUNK_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_CHUNK_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_chunk_schema()))
}

pub(crate) fn build_extra_string_array(extra_code: Option<ArrayRef>, row_count: usize) -> Result<ArrayRef, String> {
    let Some(extra_code_array) = extra_code else {
        return Ok(build_null_extra_string_array(row_count));
    };
    let extra_code_values = extra_code_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| "REGENIE step 2 extra code must be an int32 array.".to_string())?;
    if extra_code_values.len() != row_count {
        return Err("REGENIE step 2 extra code row count does not match metadata row count.".to_string());
    }
    if extra_code_values.null_count() == row_count {
        return Ok(build_null_extra_string_array(row_count));
    }
    let mut has_test_fail = false;
    for row_index in 0..extra_code_values.len() {
        if extra_code_values.is_null(row_index) {
            continue;
        }
        let extra_code_value = extra_code_values.value(row_index);
        match extra_code_value {
            0..=2 => {}
            3 => has_test_fail = true,
            unsupported_extra_code_value => {
                return Err(format!("Unsupported REGENIE step 2 extra code: {unsupported_extra_code_value}"));
            }
        }
    }
    if !has_test_fail {
        return Ok(build_null_extra_string_array(row_count));
    }

    let mut extra_string_builder = StringBuilder::with_capacity(row_count, row_count * "TEST_FAIL".len());
    for row_index in 0..extra_code_values.len() {
        if !extra_code_values.is_null(row_index) && extra_code_values.value(row_index) == 3 {
            extra_string_builder.append_value("TEST_FAIL");
        } else {
            extra_string_builder.append_null();
        }
    }
    Ok(Arc::new(extra_string_builder.finish()))
}

pub(crate) fn build_null_extra_string_array(row_count: usize) -> ArrayRef {
    new_null_array(&DataType::Utf8, row_count)
}

pub(crate) fn build_correction_method_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
) -> Result<ArrayRef, String> {
    build_correction_label_array(
        extra_code,
        row_count,
        "correction method",
        |extra_code_value| match extra_code_value {
            0 => Some("score"),
            1 => Some("firth_approximate"),
            2 => Some("spa"),
            3 => Some("firth_approximate"),
            _ => None,
        },
        "score",
    )
}

pub(crate) fn build_correction_status_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
) -> Result<ArrayRef, String> {
    build_correction_label_array(
        extra_code,
        row_count,
        "correction status",
        |extra_code_value| match extra_code_value {
            0..=2 => Some("success"),
            3 => Some("failed"),
            _ => None,
        },
        "success",
    )
}

fn build_correction_label_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
    label_kind: &str,
    label_for_code: impl Fn(i32) -> Option<&'static str>,
    default_label: &'static str,
) -> Result<ArrayRef, String> {
    let Some(extra_code_array) = extra_code else {
        return Ok(Arc::new(StringArray::from(vec![default_label; row_count])));
    };
    let extra_code_values = extra_code_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| format!("REGENIE step 2 {label_kind} code must be an int32 array."))?;
    if extra_code_values.len() != row_count {
        return Err(format!("REGENIE step 2 {label_kind} row count does not match metadata row count."));
    }
    let mut label_builder = StringBuilder::with_capacity(row_count, row_count * default_label.len());
    for row_index in 0..extra_code_values.len() {
        if extra_code_values.is_null(row_index) {
            label_builder.append_value(default_label);
            continue;
        }
        let extra_code_value = extra_code_values.value(row_index);
        let Some(label) = label_for_code(extra_code_value) else {
            return Err(format!("Unsupported REGENIE step 2 extra code: {extra_code_value}"));
        };
        label_builder.append_value(label);
    }
    Ok(Arc::new(label_builder.finish()))
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
        Field::new("CORRECTION_METHOD", DataType::Utf8, true),
        Field::new("CORRECTION_STATUS", DataType::Utf8, true),
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
        Field::new("CORRECTION_METHOD", DataType::Utf8, true),
        Field::new("CORRECTION_STATUS", DataType::Utf8, true),
    ])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, ArrayRef, Float32Array, Int32Array, StringArray};

    use super::{
        build_correction_method_array, build_correction_status_array, build_extra_string_array,
        get_regenie_step2_chunk_schema, get_regenie_step2_final_schema,
    };

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
    fn extra_string_array_fast_paths_null_and_success_codes() {
        let all_null_extra =
            build_extra_string_array(Some(Arc::new(Int32Array::from(vec![None, None])) as ArrayRef), 2)
                .expect("all-null extra code should map");
        assert_eq!(all_null_extra.len(), 2);
        assert_eq!(all_null_extra.null_count(), 2);

        let all_success_extra = build_extra_string_array(
            Some(Arc::new(Int32Array::from(vec![Some(0), Some(1), Some(2), None])) as ArrayRef),
            4,
        )
        .expect("all-success extra code should map");
        assert_eq!(all_success_extra.len(), 4);
        assert_eq!(all_success_extra.null_count(), 4);
    }

    #[test]
    fn extra_string_array_only_allocates_labels_when_test_fail_is_present() {
        let mixed_extra = build_extra_string_array(
            Some(Arc::new(Int32Array::from(vec![Some(0), Some(3), None, Some(2)])) as ArrayRef),
            4,
        )
        .expect("mixed extra code should map");
        let mixed_extra_values =
            mixed_extra.as_any().downcast_ref::<StringArray>().expect("extra should be a string array");

        assert!(mixed_extra_values.is_null(0));
        assert_eq!(mixed_extra_values.value(1), "TEST_FAIL");
        assert!(mixed_extra_values.is_null(2));
        assert!(mixed_extra_values.is_null(3));
    }

    #[test]
    fn correction_arrays_map_supported_extra_codes_and_defaults() {
        let extra_code_array =
            Some(Arc::new(Int32Array::from(vec![Some(0), Some(1), Some(2), Some(3), None])) as ArrayRef);

        let correction_method_array =
            build_correction_method_array(extra_code_array.as_ref().map(Arc::clone), 5).expect("methods should map");
        let correction_method_values =
            correction_method_array.as_any().downcast_ref::<StringArray>().expect("methods should be strings");
        assert_eq!(
            (0..correction_method_values.len()).map(|index| correction_method_values.value(index)).collect::<Vec<_>>(),
            vec!["score", "firth_approximate", "spa", "firth_approximate", "score"]
        );

        let correction_status_array = build_correction_status_array(extra_code_array, 5).expect("statuses should map");
        let correction_status_values =
            correction_status_array.as_any().downcast_ref::<StringArray>().expect("statuses should be strings");
        assert_eq!(
            (0..correction_status_values.len()).map(|index| correction_status_values.value(index)).collect::<Vec<_>>(),
            vec!["success", "success", "success", "failed", "success"]
        );

        let default_method_array =
            build_correction_method_array(None, 2).expect("missing extra code should default to score");
        let default_method_values =
            default_method_array.as_any().downcast_ref::<StringArray>().expect("default methods should be strings");
        assert_eq!(default_method_values.value(0), "score");
        assert_eq!(default_method_values.value(1), "score");
    }

    #[test]
    fn schema_singletons_have_expected_final_columns() {
        assert_eq!(get_regenie_step2_chunk_schema().fields().len(), 16);
        assert_eq!(get_regenie_step2_final_schema().fields().len(), 16);
    }
}
