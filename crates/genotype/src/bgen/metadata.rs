#[derive(Debug)]
pub(super) struct VariantRecord {
    pub(super) probability_payload_offset: usize,
    pub(super) probability_payload_length: usize,
    pub(super) declared_uncompressed_block_length: usize,
    #[cfg(test)]
    pub(super) resolved_variant_identifier: String,
}
