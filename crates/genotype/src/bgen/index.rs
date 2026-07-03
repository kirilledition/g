use super::BgenError;
use super::decode::{ThreadScratch, read_exact_bytes, read_probability_block, read_u16_at, read_u32_at, u32_to_usize};
use super::format::{ALLELE_LENGTH_SIZE_IN_BYTES, CompressionType, VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES};
use super::metadata::VariantRecord;
use super::profile::ThreadLocalProfileSnapshot;

pub(super) fn parse_sample_identifier_block(
    mmap: &[u8],
    sample_block_offset: usize,
    first_variant_offset: usize,
    expected_sample_count: usize,
) -> Result<Vec<String>, BgenError> {
    let block_length = u32_to_usize(read_u32_at(mmap, sample_block_offset)?)?;
    let sample_block_stop = sample_block_offset + block_length;
    if sample_block_stop > first_variant_offset {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block overlaps the first variant block.".to_string(),
        ));
    }

    let observed_sample_count = u32_to_usize(read_u32_at(mmap, sample_block_offset + 4)?)?;
    if observed_sample_count != expected_sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Embedded BGEN sample block reports {observed_sample_count} samples, but the header reports {expected_sample_count}.",
        )));
    }

    let mut cursor = sample_block_offset + 8;
    let mut sample_identifiers = Vec::with_capacity(expected_sample_count);
    for _sample_index in 0..expected_sample_count {
        let identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let identifier_bytes = read_exact_bytes(mmap, cursor, identifier_length)?;
        sample_identifiers.push(String::from_utf8_lossy(identifier_bytes).into_owned());
        cursor += identifier_length;
    }
    if cursor != sample_block_stop {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block length does not match the encoded sample identifiers.".to_string(),
        ));
    }

    Ok(sample_identifiers)
}

pub(super) fn parse_variant_records(
    mmap: &[u8],
    first_variant_offset: usize,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
) -> Result<Vec<VariantRecord>, BgenError> {
    let mut cursor = first_variant_offset;
    let mut variant_records = Vec::with_capacity(variant_count);

    for variant_index in 0..variant_count {
        let variant_offset = cursor;
        let variant_identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let variant_identifier =
            String::from_utf8_lossy(read_exact_bytes(mmap, cursor, variant_identifier_length)?).into_owned();
        cursor += variant_identifier_length;

        let rsid_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let rsid = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, rsid_length)?).into_owned();
        cursor += rsid_length;

        let chromosome_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let chromosome = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, chromosome_length)?).into_owned();
        cursor += chromosome_length;

        let position = i64::from(read_u32_at(mmap, cursor)?);
        cursor += 4;

        let allele_count = read_u16_at(mmap, cursor)?;
        cursor += 2;
        if allele_count != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only diploid biallelic BGEN variants are supported. Variant index {variant_index} reports {allele_count} alleles.",
            )));
        }

        let mut allele_values = Vec::with_capacity(usize::from(allele_count));
        for _allele_index in 0..usize::from(allele_count) {
            let allele_length = u32_to_usize(read_u32_at(mmap, cursor)?)?;
            cursor += ALLELE_LENGTH_SIZE_IN_BYTES;
            let allele_value = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, allele_length)?).into_owned();
            cursor += allele_length;
            allele_values.push(allele_value);
        }

        let genotype_block_offset = cursor;
        let total_block_length = u32_to_usize(read_u32_at(mmap, genotype_block_offset)?)?;
        let block_payload_offset = genotype_block_offset + 4;
        let (probability_payload_offset, probability_payload_length, declared_uncompressed_block_length) =
            match compression_type {
                CompressionType::None => (block_payload_offset, total_block_length, total_block_length),
                CompressionType::Zlib => {
                    let declared_uncompressed_block_length = u32_to_usize(read_u32_at(mmap, block_payload_offset)?)?;
                    let probability_payload_length = total_block_length.checked_sub(4).ok_or_else(|| {
                        BgenError::InvalidFormat(
                            "Compressed BGEN blocks must include a four-byte uncompressed length prefix.".to_string(),
                        )
                    })?;
                    (block_payload_offset + 4, probability_payload_length, declared_uncompressed_block_length)
                }
            };
        cursor += 4 + total_block_length;
        if cursor > mmap.len() {
            return Err(BgenError::InvalidFormat(format!(
                "Variant index {variant_index} points beyond the end of the BGEN file.",
            )));
        }

        if variant_index == 0 {
            validate_variant_probability_block(
                mmap,
                compression_type,
                &VariantRecord {
                    variant_offset,
                    probability_payload_offset,
                    probability_payload_length,
                    declared_uncompressed_block_length,
                    chromosome: chromosome.clone(),
                    resolved_variant_identifier: if rsid.is_empty() {
                        variant_identifier.clone()
                    } else {
                        rsid.clone()
                    },
                    position,
                    counted_allele: allele_values[1].clone(),
                    reference_allele: allele_values[0].clone(),
                },
                sample_count,
                "first variant",
            )?;
        }

        let reference_allele = allele_values[0].clone();
        let counted_allele = allele_values[1].clone();
        let resolved_variant_identifier = if rsid.is_empty() { variant_identifier } else { rsid.clone() };

        variant_records.push(VariantRecord {
            variant_offset,
            probability_payload_offset,
            probability_payload_length,
            declared_uncompressed_block_length,
            chromosome,
            resolved_variant_identifier,
            position,
            counted_allele,
            reference_allele,
        });
    }

    Ok(variant_records)
}

fn validate_variant_probability_block(
    mmap: &[u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    variant_label: &str,
) -> Result<(), BgenError> {
    let mut thread_scratch = ThreadScratch::default();
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        &mut thread_scratch,
        &mut thread_local_profile_snapshot,
        false,
    )?;
    let observed_sample_count = u32_to_usize(read_u32_at(probability_block, 0)?)?;
    if observed_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "The {variant_label} stores {observed_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }
    Ok(())
}
