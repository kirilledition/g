use std::sync::{Arc, OnceLock};

use arrow::array::{Array, ArrayRef, Int32Array, StringArray, StringBuilder, new_null_array};
use arrow::datatypes::{DataType, Field, Schema};

use crate::error::OutputError;

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";
pub(crate) const OUTPUT_SCHEMA_VERSION: &str = "2";

type OutputSchemaResult<T> = Result<T, OutputError>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum OutputStatisticDtype {
    #[default]
    Float32,
    Float64,
}

impl OutputStatisticDtype {
    pub(crate) fn parse(value: &str) -> OutputSchemaResult<Self> {
        match value {
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            unsupported_value => Err(OutputError::InvalidInput(format!(
                "Output statistic dtype must be 'float32' or 'float64', observed '{unsupported_value}'."
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }

    fn arrow_data_type(self) -> DataType {
        match self {
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
        }
    }
}

pub(crate) fn get_regenie_step2_chunk_schema(output_statistic_dtype: OutputStatisticDtype) -> &'static Arc<Schema> {
    static REGENIE_STEP2_CHUNK_FLOAT32_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    static REGENIE_STEP2_CHUNK_FLOAT64_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    match output_statistic_dtype {
        OutputStatisticDtype::Float32 => REGENIE_STEP2_CHUNK_FLOAT32_SCHEMA
            .get_or_init(|| Arc::new(build_regenie_step2_chunk_schema(output_statistic_dtype))),
        OutputStatisticDtype::Float64 => REGENIE_STEP2_CHUNK_FLOAT64_SCHEMA
            .get_or_init(|| Arc::new(build_regenie_step2_chunk_schema(output_statistic_dtype))),
    }
}

pub(crate) fn build_extra_string_array(extra_code: Option<ArrayRef>, row_count: usize) -> OutputSchemaResult<ArrayRef> {
    let Some(extra_code_array) = extra_code else {
        return Ok(build_null_extra_string_array(row_count));
    };
    let extra_code_values = extra_code_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| OutputError::InvalidInput("REGENIE step 2 extra code must be an int32 array.".to_string()))?;
    if extra_code_values.len() != row_count {
        return Err(OutputError::InvalidInput(
            "REGENIE step 2 extra code row count does not match metadata row count.".to_string(),
        ));
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
                return Err(OutputError::InvalidInput(format!(
                    "Unsupported REGENIE step 2 extra code: {unsupported_extra_code_value}"
                )));
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
) -> OutputSchemaResult<ArrayRef> {
    build_correction_label_array(
        extra_code,
        row_count,
        "correction method",
        |extra_code_value| match extra_code_value {
            0 => Some("score"),
            1 | 3 => Some("firth_approximate"),
            2 => Some("spa"),
            _ => None,
        },
        "score",
    )
}

pub(crate) fn build_correction_status_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
) -> OutputSchemaResult<ArrayRef> {
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
) -> OutputSchemaResult<ArrayRef> {
    let Some(extra_code_array) = extra_code else {
        return Ok(Arc::new(StringArray::from(vec![default_label; row_count])));
    };
    let extra_code_values = extra_code_array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
        OutputError::InvalidInput(format!("REGENIE step 2 {label_kind} code must be an int32 array."))
    })?;
    if extra_code_values.len() != row_count {
        return Err(OutputError::InvalidInput(format!(
            "REGENIE step 2 {label_kind} row count does not match metadata row count."
        )));
    }
    let mut label_builder = StringBuilder::with_capacity(row_count, row_count * default_label.len());
    for row_index in 0..extra_code_values.len() {
        if extra_code_values.is_null(row_index) {
            label_builder.append_value(default_label);
            continue;
        }
        let extra_code_value = extra_code_values.value(row_index);
        let Some(label) = label_for_code(extra_code_value) else {
            return Err(OutputError::InvalidInput(format!(
                "Unsupported REGENIE step 2 extra code: {extra_code_value}"
            )));
        };
        label_builder.append_value(label);
    }
    Ok(Arc::new(label_builder.finish()))
}

pub(crate) fn get_regenie_step2_final_schema(output_statistic_dtype: OutputStatisticDtype) -> &'static Arc<Schema> {
    static REGENIE_STEP2_FINAL_FLOAT32_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    static REGENIE_STEP2_FINAL_FLOAT64_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    match output_statistic_dtype {
        OutputStatisticDtype::Float32 => REGENIE_STEP2_FINAL_FLOAT32_SCHEMA
            .get_or_init(|| Arc::new(build_regenie_step2_final_schema(output_statistic_dtype))),
        OutputStatisticDtype::Float64 => REGENIE_STEP2_FINAL_FLOAT64_SCHEMA
            .get_or_init(|| Arc::new(build_regenie_step2_final_schema(output_statistic_dtype))),
    }
}

pub(crate) fn output_statistic_dtype_from_schema(schema: &Schema) -> OutputSchemaResult<OutputStatisticDtype> {
    let statistic_column_names = ["BETA", "SE", "CHISQ", "LOG10P"];
    let mut observed_dtype: Option<OutputStatisticDtype> = None;
    for column_name in statistic_column_names {
        let field = schema.field_with_name(column_name).map_err(OutputError::runtime)?;
        let column_dtype = match field.data_type() {
            DataType::Float32 => OutputStatisticDtype::Float32,
            DataType::Float64 => OutputStatisticDtype::Float64,
            data_type => {
                return Err(OutputError::InvalidInput(format!(
                    "REGENIE step 2 public statistic column {column_name} must be Float32 or Float64, observed {data_type}.",
                )));
            }
        };
        if let Some(previous_dtype) = observed_dtype {
            if previous_dtype != column_dtype {
                return Err(OutputError::InvalidInput(
                    "REGENIE step 2 public statistic columns must use one common dtype.".to_string(),
                ));
            }
        } else {
            observed_dtype = Some(column_dtype);
        }
    }
    observed_dtype
        .ok_or_else(|| OutputError::InvalidInput("REGENIE step 2 public statistic columns are missing.".to_string()))
}

fn build_regenie_step2_chunk_schema(output_statistic_dtype: OutputStatisticDtype) -> Schema {
    let statistic_data_type = output_statistic_dtype.arrow_data_type();
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
        Field::new("BETA", statistic_data_type.clone(), true),
        Field::new("SE", statistic_data_type.clone(), true),
        Field::new("CHISQ", statistic_data_type.clone(), true),
        Field::new("LOG10P", statistic_data_type, true),
        Field::new("EXTRA", DataType::Utf8, true),
        Field::new("CORRECTION_METHOD", DataType::Utf8, true),
        Field::new("CORRECTION_STATUS", DataType::Utf8, true),
    ])
}
fn build_regenie_step2_final_schema(output_statistic_dtype: OutputStatisticDtype) -> Schema {
    let statistic_data_type = output_statistic_dtype.arrow_data_type();
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
        Field::new("BETA", statistic_data_type.clone(), true),
        Field::new("SE", statistic_data_type.clone(), true),
        Field::new("CHISQ", statistic_data_type.clone(), true),
        Field::new("LOG10P", statistic_data_type, true),
        Field::new("EXTRA", DataType::Utf8, true),
        Field::new("CORRECTION_METHOD", DataType::Utf8, true),
        Field::new("CORRECTION_STATUS", DataType::Utf8, true),
    ])
}
