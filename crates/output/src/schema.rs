use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, Schema};

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";

pub(crate) static REGENIE_STEP2_CHUNK_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_regenie_step2_chunk_schema()));
pub(crate) static REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_regenie_step2_parquet_record_batch_schema(&REGENIE_STEP2_CHUNK_SCHEMA)));

fn build_regenie_step2_chunk_schema() -> Schema {
    let statistic_data_type = DataType::Float32;
    Schema::new(vec![
        Field::new("CHROM", DataType::Utf8, false),
        Field::new("GENPOS", DataType::Int64, false),
        Field::new("ID", DataType::Utf8, false),
        Field::new("ALLELE0", DataType::Utf8, false),
        Field::new("ALLELE1", DataType::Utf8, false),
        Field::new("A1FREQ", DataType::Float32, false),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, false),
        Field::new("BETA", statistic_data_type.clone(), false),
        Field::new("SE", statistic_data_type.clone(), false),
        Field::new("CHISQ", statistic_data_type.clone(), false),
        Field::new("LOG10P", statistic_data_type, false),
        Field::new("CORRECTION_METHOD", DataType::Utf8, false),
        Field::new("CORRECTION_STATUS", DataType::Utf8, false),
    ])
}

fn build_regenie_step2_parquet_record_batch_schema(chunk_schema: &Schema) -> Schema {
    let fields = chunk_schema
        .fields()
        .iter()
        .map(|field| {
            let data_type = match field.name().as_str() {
                "CORRECTION_METHOD" | "CORRECTION_STATUS" => {
                    DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
                }
                _ => field.data_type().clone(),
            };
            Field::new(field.name().clone(), data_type, field.is_nullable()).with_metadata(field.metadata().clone())
        })
        .collect::<Vec<_>>();
    Schema::new_with_metadata(fields, chunk_schema.metadata().clone())
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::DataType;

    use super::{REGENIE_STEP2_CHUNK_SCHEMA, REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA};

    #[test]
    fn result_schema_has_stable_columns_and_nullable_info() {
        let expected_columns = [
            ("CHROM", DataType::Utf8, false),
            ("GENPOS", DataType::Int64, false),
            ("ID", DataType::Utf8, false),
            ("ALLELE0", DataType::Utf8, false),
            ("ALLELE1", DataType::Utf8, false),
            ("A1FREQ", DataType::Float32, false),
            ("INFO", DataType::Float32, true),
            ("N", DataType::Int32, false),
            ("BETA", DataType::Float32, false),
            ("SE", DataType::Float32, false),
            ("CHISQ", DataType::Float32, false),
            ("LOG10P", DataType::Float32, false),
            ("CORRECTION_METHOD", DataType::Utf8, false),
            ("CORRECTION_STATUS", DataType::Utf8, false),
        ];

        assert_eq!(REGENIE_STEP2_CHUNK_SCHEMA.fields().len(), expected_columns.len());
        for (field, (name, data_type, nullable)) in REGENIE_STEP2_CHUNK_SCHEMA.fields().iter().zip(expected_columns) {
            assert_eq!(field.name(), name);
            assert_eq!(field.data_type(), &data_type);
            assert_eq!(field.is_nullable(), nullable);
        }
    }

    #[test]
    fn parquet_staging_uses_uint8_dictionaries_only_for_correction_labels() {
        for (public_field, staging_field) in
            REGENIE_STEP2_CHUNK_SCHEMA.fields().iter().zip(REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA.fields())
        {
            let expected_data_type = match public_field.name().as_str() {
                "CORRECTION_METHOD" | "CORRECTION_STATUS" => {
                    DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
                }
                _ => public_field.data_type().clone(),
            };
            assert_eq!(staging_field.name(), public_field.name());
            assert_eq!(staging_field.data_type(), &expected_data_type);
            assert_eq!(staging_field.is_nullable(), public_field.is_nullable());
        }
    }
}
