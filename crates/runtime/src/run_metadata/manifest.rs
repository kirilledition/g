use super::types::{
    RunManifestCommandPayload, RunManifestExtensionInput, RunManifestExtensionPayload, RunManifestRuntimePayload,
};

pub(super) const COMMAND_INTERFACE: &str = "g regenie";

#[must_use]
pub fn build_run_manifest_extension(input: RunManifestExtensionInput) -> RunManifestExtensionPayload {
    let RunManifestExtensionInput {
        phenotype_name,
        effective_config,
        output_format,
        device,
        staging_depth,
        threads,
        writer_threads,
        writer_queue_depth,
        chunks_per_arrow_file,
        arrow_compression,
        parquet_compression,
        output_statistic_dtype,
        bgen_decode_tile_variant_count,
        trusted_no_missing_diploid,
        trusted_bgen_validation_mode,
    } = input;
    RunManifestExtensionPayload {
        command: RunManifestCommandPayload {
            interface: COMMAND_INTERFACE,
            phenotype: phenotype_name,
            effective_config,
            output_format,
        },
        runtime: RunManifestRuntimePayload {
            device,
            staging_depth,
            threads,
            writer_threads,
            writer_queue_depth,
            chunks_per_arrow_file,
            arrow_compression,
            parquet_compression,
            output_statistic_dtype,
            bgen_decode_tile_variant_count,
            trusted_no_missing_diploid,
            trusted_bgen_validation_mode,
        },
    }
}
