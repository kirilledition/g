use crate::genotype::common::VariantMetadataColumns;

#[derive(Debug)]
pub(super) struct VariantRecord {
    pub(super) probability_payload_offset: usize,
    pub(super) probability_payload_length: usize,
    pub(super) declared_uncompressed_block_length: usize,
    pub(super) chromosome: String,
    pub(super) resolved_variant_identifier: String,
    pub(super) position: i64,
    pub(super) counted_allele: String,
    pub(super) reference_allele: String,
}

pub(super) fn build_variant_metadata_columns(selected_variant_records: &[VariantRecord]) -> VariantMetadataColumns {
    let chromosome_values =
        selected_variant_records.iter().map(|variant_record| variant_record.chromosome.clone()).collect();
    let variant_identifier_values = selected_variant_records
        .iter()
        .map(|variant_record| variant_record.resolved_variant_identifier.clone())
        .collect();
    let position_values = selected_variant_records.iter().map(|variant_record| variant_record.position).collect();
    let allele_one_values =
        selected_variant_records.iter().map(|variant_record| variant_record.counted_allele.clone()).collect();
    let allele_two_values =
        selected_variant_records.iter().map(|variant_record| variant_record.reference_allele.clone()).collect();
    VariantMetadataColumns {
        chromosome: chromosome_values,
        variant_identifier: variant_identifier_values,
        position: position_values,
        allele_one: allele_one_values,
        allele_two: allele_two_values,
    }
}

pub(super) fn build_chromosome_boundary_indices(variant_records: &[VariantRecord]) -> Vec<usize> {
    let mut chromosome_boundary_indices = Vec::with_capacity(variant_records.len().min(256) + 1);
    chromosome_boundary_indices.push(0);
    for variant_index in 1..variant_records.len() {
        if variant_records[variant_index].chromosome != variant_records[variant_index - 1].chromosome {
            chromosome_boundary_indices.push(variant_index);
        }
    }
    chromosome_boundary_indices.push(variant_records.len());
    chromosome_boundary_indices
}
