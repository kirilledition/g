use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, Schema};

pub(crate) const CHUNK_COMMITS_METADATA_KEY: &str = "g.output.chunk_commits";

static REGENIE_STEP2_CHUNK_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_regenie_step2_chunk_schema()));
static REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_regenie_step2_parquet_record_batch_schema(&REGENIE_STEP2_CHUNK_SCHEMA)));

pub(crate) fn get_regenie_step2_chunk_schema() -> &'static Arc<Schema> {
    &REGENIE_STEP2_CHUNK_SCHEMA
}

pub(crate) fn get_regenie_step2_parquet_record_batch_schema() -> &'static Arc<Schema> {
    &REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA
}

fn build_regenie_step2_chunk_schema() -> Schema {
    let statistic_data_type = DataType::Float32;
    Schema::new(vec![
        Field::new("CHROM", DataType::Utf8, false),
        Field::new("GENPOS", DataType::Int64, false),
        Field::new("ID", DataType::Utf8, false),
        Field::new("ALLELE0", DataType::Utf8, false),
        Field::new("ALLELE1", DataType::Utf8, false),
        Field::new("A1FREQ", DataType::Float32, false),
        Field::new("INFO", DataType::Float32, false),
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
