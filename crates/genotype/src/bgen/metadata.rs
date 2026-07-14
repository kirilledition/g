#[derive(Debug)]
pub(super) struct VariantRecord {
    pub(super) probability_payload_offset: u64,
    pub(super) probability_payload_length: u32,
    pub(super) declared_uncompressed_block_length: u32,
    #[cfg(test)]
    pub(super) resolved_variant_identifier: String,
}
