use std::collections::HashMap;
use std::sync::Arc;

use g_genotype_contracts::VariantMetadataStore;

use super::BgenError;
use super::decode::{ThreadScratch, read_exact_bytes, read_probability_block, read_u16_at, read_u32_at, u32_to_usize};
use super::format::{ALLELE_LENGTH_SIZE_IN_BYTES, CompressionType, VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES};
use super::metadata::VariantRecord;

const LAYOUT_TWO_FIXED_PROBABILITY_BLOCK_LENGTH: usize = 10;
const MAXIMUM_SUPPORTED_PROBABILITY_BLOCK_BYTES_PER_SAMPLE: usize = 9;

pub(super) struct ParsedVariantIndex {
    pub(super) variant_records: Vec<VariantRecord>,
    pub(super) variant_metadata: Arc<VariantMetadataStore>,
    pub(super) chromosome_boundary_indices: Vec<usize>,
}

#[derive(Default)]
struct StringDictionaryBuilder {
    values: Vec<Arc<str>>,
    codes_by_value: HashMap<Arc<str>, u32>,
}

impl StringDictionaryBuilder {
    fn intern(&mut self, bytes: &[u8]) -> Result<u32, BgenError> {
        let text = String::from_utf8_lossy(bytes);
        if let Some(code) = self.codes_by_value.get(text.as_ref()) {
            return Ok(*code);
        }
        let code = u32::try_from(self.values.len())
            .map_err(|_| BgenError::Range("BGEN metadata dictionary exceeds the uint32 index domain.".to_string()))?;
        let value = Arc::<str>::from(text.into_owned());
        self.codes_by_value.insert(Arc::clone(&value), code);
        self.values.push(value);
        Ok(code)
    }
}

pub(super) fn validate_sample_identifier_block(
    mmap: &[u8],
    sample_block_offset: usize,
    first_variant_offset: usize,
    expected_sample_count: usize,
) -> Result<(), BgenError> {
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
    for _sample_index in 0..expected_sample_count {
        let identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let _identifier_bytes = read_exact_bytes(mmap, cursor, identifier_length)?;
        cursor += identifier_length;
    }
    if cursor != sample_block_stop {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block length does not match the encoded sample identifiers.".to_string(),
        ));
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(super) fn parse_variant_index(
    mmap: &[u8],
    first_variant_offset: usize,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
) -> Result<ParsedVariantIndex, BgenError> {
    let maximum_block_length = maximum_supported_probability_block_length(sample_count)?;
    let minimum_block_length = sample_count + LAYOUT_TWO_FIXED_PROBABILITY_BLOCK_LENGTH;
    let mut cursor = first_variant_offset;
    let mut variant_records = Vec::with_capacity(variant_count);
    let mut chromosome_codes = Vec::with_capacity(variant_count);
    let mut variant_identifier_text = String::new();
    let mut variant_identifier_offsets = Vec::with_capacity(variant_count.saturating_add(1));
    variant_identifier_offsets.push(0_u32);
    let mut position = Vec::with_capacity(variant_count);
    let mut allele_one_codes = Vec::with_capacity(variant_count);
    let mut allele_two_codes = Vec::with_capacity(variant_count);
    let mut chromosome_boundary_indices = Vec::with_capacity(variant_count.min(256) + 1);
    chromosome_boundary_indices.push(0);
    let mut metadata_text_dictionary = StringDictionaryBuilder::default();
    let mut previous_chromosome_bytes: &[u8] = &[];
    let mut previous_chromosome_code = 0_u32;

    for variant_index in 0..variant_count {
        let variant_identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let variant_identifier_bytes = read_exact_bytes(mmap, cursor, variant_identifier_length)?;
        cursor += variant_identifier_length;

        let rsid_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let rsid_bytes = read_exact_bytes(mmap, cursor, rsid_length)?;
        cursor += rsid_length;

        let chromosome_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let chromosome_bytes = read_exact_bytes(mmap, cursor, chromosome_length)?;
        let chromosome_code = if variant_index > 0 && chromosome_bytes == previous_chromosome_bytes {
            previous_chromosome_code
        } else {
            metadata_text_dictionary.intern(chromosome_bytes)?
        };
        previous_chromosome_bytes = chromosome_bytes;
        previous_chromosome_code = chromosome_code;
        cursor += chromosome_length;

        let variant_position = i64::from(read_u32_at(mmap, cursor)?);
        cursor += 4;

        let allele_count = read_u16_at(mmap, cursor)?;
        cursor += 2;
        if allele_count != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only diploid biallelic BGEN variants are supported. Variant index {variant_index} reports {allele_count} alleles.",
            )));
        }

        let reference_allele_code = read_allele(mmap, &mut cursor, &mut metadata_text_dictionary)?;
        let counted_allele_code = read_allele(mmap, &mut cursor, &mut metadata_text_dictionary)?;

        let total_block_length = u32_to_usize(read_u32_at(mmap, cursor)?)?;
        let (probability_payload_offset, probability_payload_length, uncompressed_block_length) = match compression_type
        {
            CompressionType::None => (cursor + 4, total_block_length, total_block_length),
            CompressionType::Zlib | CompressionType::Zstandard => {
                let uncompressed_block_length = u32_to_usize(read_u32_at(mmap, cursor + 4)?)?;
                let probability_payload_length = total_block_length
                    .checked_sub(4)
                    .filter(|payload_length| *payload_length != 0)
                    .ok_or_else(|| {
                        BgenError::InvalidFormat(
                            "Compressed BGEN blocks must include a four-byte uncompressed length prefix and a non-empty payload."
                                .to_string(),
                        )
                    })?;
                (cursor + 8, probability_payload_length, uncompressed_block_length)
            }
        };
        validate_block_length(uncompressed_block_length, minimum_block_length, maximum_block_length, variant_index)?;
        cursor += 4 + total_block_length;
        if cursor > mmap.len() {
            return Err(BgenError::InvalidFormat(format!(
                "Variant index {variant_index} points beyond the end of the BGEN file.",
            )));
        }

        let resolved_identifier_bytes = if rsid_bytes.is_empty() { variant_identifier_bytes } else { rsid_bytes };
        variant_identifier_text.push_str(&String::from_utf8_lossy(resolved_identifier_bytes));
        let variant_identifier_stop = u32::try_from(variant_identifier_text.len()).map_err(|_| {
            BgenError::Range("BGEN variant identifiers exceed the four-gibibyte metadata arena limit.".to_string())
        })?;
        variant_identifier_offsets.push(variant_identifier_stop);
        let variant_record = VariantRecord {
            probability_payload_offset,
            probability_payload_length,
            declared_uncompressed_block_length: uncompressed_block_length,
        };
        if variant_index == 0 {
            validate_variant_probability_block(mmap, compression_type, &variant_record, sample_count, "first variant")?;
        }
        if chromosome_codes.last().is_some_and(|previous_code| *previous_code != chromosome_code) {
            chromosome_boundary_indices.push(variant_index);
        }
        chromosome_codes.push(chromosome_code);
        position.push(variant_position);
        allele_one_codes.push(counted_allele_code);
        allele_two_codes.push(reference_allele_code);
        variant_records.push(variant_record);
    }

    chromosome_boundary_indices.push(variant_count);
    Ok(ParsedVariantIndex {
        variant_records,
        variant_metadata: Arc::new(VariantMetadataStore::from_parts(
            metadata_text_dictionary.values.into_boxed_slice(),
            chromosome_codes.into_boxed_slice(),
            variant_identifier_text.into_boxed_str(),
            variant_identifier_offsets.into_boxed_slice(),
            position.into_boxed_slice(),
            allele_one_codes.into_boxed_slice(),
            allele_two_codes.into_boxed_slice(),
        )),
        chromosome_boundary_indices,
    })
}

fn maximum_supported_probability_block_length(sample_count: usize) -> Result<usize, BgenError> {
    sample_count
        .checked_mul(MAXIMUM_SUPPORTED_PROBABILITY_BLOCK_BYTES_PER_SAMPLE)
        .and_then(|sample_bytes| sample_bytes.checked_add(LAYOUT_TWO_FIXED_PROBABILITY_BLOCK_LENGTH))
        .ok_or_else(|| {
            BgenError::Range(
                "BGEN sample count overflows the maximum supported Layout 2 probability block length.".to_string(),
            )
        })
}

fn validate_block_length(
    declared_length: usize,
    minimum_supported_length: usize,
    maximum_supported_length: usize,
    variant_index: usize,
) -> Result<(), BgenError> {
    if declared_length < minimum_supported_length {
        return Err(BgenError::InvalidFormat(format!(
            "Variant index {variant_index} declares an uncompressed probability block of {declared_length} bytes, but a Layout 2 block requires at least {minimum_supported_length} bytes for this sample count.",
        )));
    }
    if declared_length > maximum_supported_length {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant index {variant_index} declares an uncompressed probability block of {declared_length} bytes, but a supported biallelic diploid Layout 2 block contains at most {maximum_supported_length} bytes for this sample count.",
        )));
    }
    Ok(())
}

fn read_allele(
    mmap: &[u8],
    cursor: &mut usize,
    metadata_text_dictionary: &mut StringDictionaryBuilder,
) -> Result<u32, BgenError> {
    let allele_length = u32_to_usize(read_u32_at(mmap, *cursor)?)?;
    *cursor += ALLELE_LENGTH_SIZE_IN_BYTES;
    let allele_code = metadata_text_dictionary.intern(read_exact_bytes(mmap, *cursor, allele_length)?)?;
    *cursor += allele_length;
    Ok(allele_code)
}

fn validate_variant_probability_block(
    mmap: &[u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    variant_label: &str,
) -> Result<(), BgenError> {
    let mut thread_scratch = ThreadScratch::default();
    let probability_block = read_probability_block(mmap, compression_type, variant_record, &mut thread_scratch)?;
    let observed_sample_count = u32_to_usize(read_u32_at(probability_block, 0)?)?;
    if observed_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "The {variant_label} stores {observed_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }
    Ok(())
}
