use std::collections::HashMap;
use std::sync::Arc;

use g_genotype_contracts::VariantMetadataStore;

use super::BgenError;
use super::decode::{ThreadScratch, read_probability_block, read_u32_at, u32_to_usize};
use super::error::contextualize_variant_metadata_invariant;
use super::format::CompressionType;
use super::metadata::VariantRecord;
use super::source::{BgenSnapshotCursor, BgenSource, BgenSourceCursor, MAXIMUM_SOURCE_WINDOW_BYTE_COUNT};

const LAYOUT_TWO_FIXED_PROBABILITY_BLOCK_LENGTH: usize = 10;
const MAXIMUM_SUPPORTED_PROBABILITY_BLOCK_BYTES_PER_SAMPLE: usize = 9;
const MINIMUM_VARIANT_INDEX_BYTES: u64 = 24;

pub(super) struct ParsedVariantIndex {
    pub(super) variant_records: Vec<VariantRecord>,
    pub(super) variant_metadata: Arc<VariantMetadataStore>,
    pub(super) chromosome_boundary_indices: Vec<usize>,
}

struct StringDictionaryBuilder {
    values: Vec<Arc<str>>,
    codes_by_value: HashMap<Arc<str>, u32>,
    ascii_codes_by_byte: [Option<u32>; 128],
}

impl Default for StringDictionaryBuilder {
    fn default() -> Self {
        Self { values: Vec::new(), codes_by_value: HashMap::new(), ascii_codes_by_byte: [None; 128] }
    }
}

impl StringDictionaryBuilder {
    fn intern(&mut self, text: &str) -> Result<u32, BgenError> {
        let ascii_byte_index = match text.as_bytes() {
            [byte] if byte.is_ascii() => Some(usize::from(*byte)),
            _ => None,
        };
        if let Some(ascii_byte_index) = ascii_byte_index
            && let Some(code) = self.ascii_codes_by_byte[ascii_byte_index]
        {
            return Ok(code);
        }

        let code = if let Some(code) = self.codes_by_value.get(text) {
            *code
        } else {
            let code = u32::try_from(self.values.len()).map_err(|_| {
                BgenError::Range("BGEN metadata dictionary exceeds the uint32 index domain.".to_string())
            })?;
            self.values.try_reserve(1).map_err(|source| {
                BgenError::Range(format!("Could not reserve a BGEN metadata dictionary value: {source}."))
            })?;
            self.codes_by_value.try_reserve(1).map_err(|source| {
                BgenError::Range(format!("Could not reserve a BGEN metadata dictionary code: {source}."))
            })?;
            let value = Arc::<str>::from(text);
            self.codes_by_value.insert(Arc::clone(&value), code);
            self.values.push(value);
            code
        };
        if let Some(ascii_byte_index) = ascii_byte_index {
            self.ascii_codes_by_byte[ascii_byte_index] = Some(code);
        }
        Ok(code)
    }
}

pub(super) fn validate_sample_identifier_block(
    source: &BgenSource,
    sample_block_offset: u64,
    first_variant_offset: u64,
    expected_sample_count: usize,
) -> Result<(), BgenError> {
    let mut cursor = BgenSourceCursor::new_bounded_sequential(source, sample_block_offset, first_variant_offset)?;
    let block_length = u64::from(cursor.read_u32()?);
    if block_length < 8 {
        return Err(BgenError::InvalidFormat(format!(
            "Embedded BGEN sample block length must be at least 8 bytes. Observed {block_length}.",
        )));
    }
    let sample_block_stop = sample_block_offset
        .checked_add(block_length)
        .ok_or_else(|| BgenError::Range("Embedded BGEN sample block range overflowed uint64.".to_string()))?;
    if sample_block_stop > first_variant_offset {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block overlaps the first variant block.".to_string(),
        ));
    }

    let observed_sample_count = u32_to_usize(cursor.read_u32()?)?;
    if observed_sample_count != expected_sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Embedded BGEN sample block reports {observed_sample_count} samples, but the header reports {expected_sample_count}.",
        )));
    }

    for sample_index in 0..expected_sample_count {
        validate_cursor_read_within_block(&cursor, sample_block_stop, 2, sample_index)?;
        let identifier_length = usize::from(cursor.read_u16()?);
        validate_cursor_read_within_block(&cursor, sample_block_stop, identifier_length, sample_index)?;
        cursor.skip_exact(identifier_length)?;
    }
    if cursor.position() != sample_block_stop {
        return Err(BgenError::InvalidFormat(
            "Embedded BGEN sample block length does not match the encoded sample identifiers.".to_string(),
        ));
    }

    Ok(())
}

fn validate_cursor_read_within_block(
    cursor: &BgenSourceCursor<'_>,
    sample_block_stop: u64,
    byte_count: usize,
    sample_index: usize,
) -> Result<(), BgenError> {
    let read_stop =
        cursor
            .position()
            .checked_add(u64::try_from(byte_count).map_err(|_| {
                BgenError::Range("Embedded BGEN sample identifier length does not fit uint64.".to_string())
            })?)
            .ok_or_else(|| BgenError::Range("Embedded BGEN sample identifier range overflowed uint64.".to_string()))?;
    if read_stop > sample_block_stop {
        return Err(BgenError::InvalidFormat(format!(
            "Embedded BGEN sample identifier {sample_index} extends beyond its declared block length.",
        )));
    }
    Ok(())
}

pub(super) fn parse_variant_index(
    source: &BgenSource,
    first_variant_offset: u64,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
) -> Result<ParsedVariantIndex, BgenError> {
    validate_variant_count_against_source(source, first_variant_offset, variant_count)?;
    let maximum_block_length = maximum_supported_probability_block_length(sample_count)?;
    let minimum_block_length =
        sample_count.checked_add(LAYOUT_TWO_FIXED_PROBABILITY_BLOCK_LENGTH).ok_or_else(|| {
            BgenError::Range("BGEN minimum Layout 2 probability block length overflowed usize.".to_string())
        })?;
    if let Some(snapshot) = source.snapshot_bytes() {
        let mut cursor = BgenSnapshotCursor::new(snapshot, first_variant_offset)?;
        return parse_snapshot_variant_index(
            source,
            &mut cursor,
            variant_count,
            sample_count,
            compression_type,
            minimum_block_length,
            maximum_block_length,
        );
    }

    let mut cursor = if variant_count == 0 {
        BgenSourceCursor::new(source, first_variant_offset)?
    } else {
        BgenSourceCursor::new_bounded_sequential(source, first_variant_offset, source.length())?
    };
    parse_variant_index_with_cursor(
        source,
        &mut cursor,
        variant_count,
        sample_count,
        compression_type,
        minimum_block_length,
        maximum_block_length,
    )
}

// The owned-snapshot path deliberately keeps a concrete direct-slice cursor.
// The positioned parser needs buffer-polymorphic bytes, whose generic wrapper
// measurably slows the index loop over hundreds of thousands of variants.
#[allow(clippy::too_many_lines)]
fn parse_snapshot_variant_index(
    source: &BgenSource,
    cursor: &mut BgenSnapshotCursor<'_>,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
    minimum_block_length: usize,
    maximum_block_length: usize,
) -> Result<ParsedVariantIndex, BgenError> {
    let mut variant_records = reserved_variant_column(variant_count, "variant records")?;
    let mut chromosome_codes = reserved_variant_column(variant_count, "chromosome codes")?;
    let mut variant_identifier_text = String::new();
    let offset_count = variant_count
        .checked_add(1)
        .ok_or_else(|| BgenError::Range("BGEN variant identifier offset count overflowed usize.".to_string()))?;
    let mut variant_identifier_offsets = reserved_variant_column(offset_count, "variant identifier offsets")?;
    variant_identifier_offsets.push(0_u32);
    let mut position = reserved_variant_column(variant_count, "variant positions")?;
    let mut allele_one_codes = reserved_variant_column(variant_count, "allele-one codes")?;
    let mut allele_two_codes = reserved_variant_column(variant_count, "allele-two codes")?;
    let boundary_capacity = variant_count
        .min(256)
        .checked_add(1)
        .ok_or_else(|| BgenError::Range("BGEN chromosome-boundary capacity overflowed usize.".to_string()))?;
    let mut chromosome_boundary_indices = reserved_variant_column(boundary_capacity, "chromosome boundary indices")?;
    chromosome_boundary_indices.push(0);
    let mut metadata_text_dictionary = StringDictionaryBuilder::default();
    let mut previous_chromosome_bytes = &[][..];
    let mut previous_chromosome_code = 0_u32;

    for variant_index in 0..variant_count {
        let variant_identifier_length = usize::from(cursor.read_u16()?);
        let variant_identifier_bytes = cursor.read_bytes(variant_identifier_length)?;
        let variant_identifier = parse_metadata_text(variant_identifier_bytes, variant_index, "variant identifier")?;

        let rsid_length = usize::from(cursor.read_u16()?);
        let rsid_bytes = if rsid_length == 0 { None } else { Some(cursor.read_bytes(rsid_length)?) };
        let resolved_identifier_text = match rsid_bytes {
            None => variant_identifier,
            Some(bytes) => parse_metadata_text(bytes, variant_index, "rsid")?,
        };

        let chromosome_length = usize::from(cursor.read_u16()?);
        let chromosome_bytes = cursor.read_bytes(chromosome_length)?;
        let chromosome_text = parse_metadata_text(chromosome_bytes, variant_index, "chromosome")?;
        let chromosome_code = if variant_index > 0 && chromosome_bytes == previous_chromosome_bytes {
            previous_chromosome_code
        } else {
            metadata_text_dictionary.intern(chromosome_text)?
        };
        previous_chromosome_bytes = chromosome_bytes;
        previous_chromosome_code = chromosome_code;

        let variant_position = i64::from(cursor.read_u32()?);

        let allele_count = cursor.read_u16()?;
        if allele_count != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only diploid biallelic BGEN variants are supported. Variant index {variant_index} reports {allele_count} alleles.",
            )));
        }

        let reference_allele_code =
            read_snapshot_allele(cursor, &mut metadata_text_dictionary, variant_index, "reference allele")?;
        let counted_allele_code =
            read_snapshot_allele(cursor, &mut metadata_text_dictionary, variant_index, "counted allele")?;

        let total_block_length = cursor.read_u32()?;
        let (probability_payload_offset, probability_payload_length, uncompressed_block_length) = match compression_type
        {
            CompressionType::None => (cursor.position(), total_block_length, total_block_length),
            CompressionType::Zlib | CompressionType::Zstandard => {
                let uncompressed_block_length = cursor.read_u32()?;
                let probability_payload_length = total_block_length
                    .checked_sub(4)
                    .filter(|payload_length| *payload_length != 0)
                    .ok_or_else(|| {
                        BgenError::InvalidFormat(
                            "Compressed BGEN blocks must include a four-byte uncompressed length prefix and a non-empty payload."
                                .to_string(),
                        )
                    })?;
                (cursor.position(), probability_payload_length, uncompressed_block_length)
            }
        };
        let probability_payload_byte_count = u32_to_usize(probability_payload_length)?;
        validate_source_window_field_length(probability_payload_byte_count, variant_index, "probability payload")?;
        let uncompressed_block_byte_count = u32_to_usize(uncompressed_block_length)?;
        validate_source_window_field_length(
            uncompressed_block_byte_count,
            variant_index,
            "uncompressed probability block",
        )?;
        validate_block_length(
            uncompressed_block_byte_count,
            minimum_block_length,
            maximum_block_length,
            variant_index,
        )?;
        cursor.skip_payload_exact(probability_payload_byte_count)?;

        let variant_identifier_stop = variant_identifier_text
            .len()
            .checked_add(resolved_identifier_text.len())
            .and_then(|identifier_stop| u32::try_from(identifier_stop).ok())
            .ok_or_else(|| {
                BgenError::Range("BGEN variant identifiers exceed the four-gibibyte metadata arena limit.".to_string())
            })?;
        variant_identifier_text
            .try_reserve(resolved_identifier_text.len())
            .map_err(|source| BgenError::Range(format!("Could not reserve BGEN variant identifier text: {source}.")))?;
        variant_identifier_text.push_str(resolved_identifier_text);
        variant_identifier_offsets.push(variant_identifier_stop);
        let variant_record = VariantRecord {
            probability_payload_offset,
            probability_payload_length,
            declared_uncompressed_block_length: uncompressed_block_length,
        };
        if variant_index == 0 {
            validate_variant_probability_block(
                source,
                compression_type,
                &variant_record,
                sample_count,
                "first variant",
            )?;
        }
        if chromosome_codes.last().is_some_and(|previous_code| *previous_code != chromosome_code) {
            chromosome_boundary_indices.try_reserve(1).map_err(|source| {
                BgenError::Range(format!("Could not reserve a BGEN chromosome boundary: {source}."))
            })?;
            chromosome_boundary_indices.push(variant_index);
        }
        chromosome_codes.push(chromosome_code);
        position.push(variant_position);
        allele_one_codes.push(counted_allele_code);
        allele_two_codes.push(reference_allele_code);
        variant_records.push(variant_record);
    }

    if cursor.position() != source.length() {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN header reports {variant_count} variants ending at byte {}, but the source contains {} bytes.",
            cursor.position(),
            source.length(),
        )));
    }

    chromosome_boundary_indices.try_reserve(1).map_err(|source| {
        BgenError::Range(format!("Could not reserve the final BGEN chromosome boundary: {source}."))
    })?;
    chromosome_boundary_indices.push(variant_count);
    Ok(ParsedVariantIndex {
        variant_records,
        variant_metadata: Arc::new(
            VariantMetadataStore::from_parts(
                metadata_text_dictionary.values.into_boxed_slice(),
                chromosome_codes.into_boxed_slice(),
                variant_identifier_text.into_boxed_str(),
                variant_identifier_offsets.into_boxed_slice(),
                position.into_boxed_slice(),
                allele_one_codes.into_boxed_slice(),
                allele_two_codes.into_boxed_slice(),
            )
            .map_err(|error| {
                contextualize_variant_metadata_invariant("Parsed BGEN variant metadata violates its invariants", error)
            })?,
        ),
        chromosome_boundary_indices,
    })
}

// Parsing remains sequential so every bounds check advances the same audited
// BGEN cursor; splitting it would obscure the byte-order and offset invariant.
#[allow(clippy::too_many_lines)]
fn parse_variant_index_with_cursor(
    source: &BgenSource,
    cursor: &mut BgenSourceCursor<'_>,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
    minimum_block_length: usize,
    maximum_block_length: usize,
) -> Result<ParsedVariantIndex, BgenError> {
    let mut variant_records = reserved_variant_column(variant_count, "variant records")?;
    let mut chromosome_codes = reserved_variant_column(variant_count, "chromosome codes")?;
    let mut variant_identifier_text = String::new();
    let offset_count = variant_count
        .checked_add(1)
        .ok_or_else(|| BgenError::Range("BGEN variant identifier offset count overflowed usize.".to_string()))?;
    let mut variant_identifier_offsets = reserved_variant_column(offset_count, "variant identifier offsets")?;
    variant_identifier_offsets.push(0_u32);
    let mut position = reserved_variant_column(variant_count, "variant positions")?;
    let mut allele_one_codes = reserved_variant_column(variant_count, "allele-one codes")?;
    let mut allele_two_codes = reserved_variant_column(variant_count, "allele-two codes")?;
    let boundary_capacity = variant_count
        .min(256)
        .checked_add(1)
        .ok_or_else(|| BgenError::Range("BGEN chromosome-boundary capacity overflowed usize.".to_string()))?;
    let mut chromosome_boundary_indices = reserved_variant_column(boundary_capacity, "chromosome boundary indices")?;
    chromosome_boundary_indices.push(0);
    let mut metadata_text_dictionary = StringDictionaryBuilder::default();
    let mut previous_chromosome_bytes = Vec::new();
    let mut previous_chromosome_code = 0_u32;
    let mut variant_identifier_buffer = Vec::new();
    let mut rsid_buffer = Vec::new();
    let mut metadata_field_buffer = Vec::new();

    for variant_index in 0..variant_count {
        let variant_identifier_length = usize::from(cursor.read_u16()?);
        let variant_identifier_bytes = cursor.read_bytes(variant_identifier_length, &mut variant_identifier_buffer)?;
        let variant_identifier =
            parse_metadata_text(variant_identifier_bytes.as_ref(), variant_index, "variant identifier")?;

        let rsid_length = usize::from(cursor.read_u16()?);
        let rsid_bytes = if rsid_length == 0 { None } else { Some(cursor.read_bytes(rsid_length, &mut rsid_buffer)?) };
        let resolved_identifier_text = match &rsid_bytes {
            None => variant_identifier,
            Some(bytes) => parse_metadata_text(bytes.as_ref(), variant_index, "rsid")?,
        };

        let chromosome_length = usize::from(cursor.read_u16()?);
        let chromosome_bytes = cursor.read_bytes(chromosome_length, &mut metadata_field_buffer)?;
        let chromosome_text = parse_metadata_text(chromosome_bytes.as_ref(), variant_index, "chromosome")?;
        let chromosome_code = if variant_index > 0 && chromosome_bytes.as_ref() == previous_chromosome_bytes {
            previous_chromosome_code
        } else {
            let chromosome_code = metadata_text_dictionary.intern(chromosome_text)?;
            previous_chromosome_bytes.clear();
            previous_chromosome_bytes.try_reserve(chromosome_bytes.as_ref().len()).map_err(|source| {
                BgenError::Range(format!("Could not reserve BGEN chromosome comparison storage: {source}."))
            })?;
            previous_chromosome_bytes.extend_from_slice(chromosome_bytes.as_ref());
            chromosome_code
        };
        previous_chromosome_code = chromosome_code;

        let variant_position = i64::from(cursor.read_u32()?);

        let allele_count = cursor.read_u16()?;
        if allele_count != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only diploid biallelic BGEN variants are supported. Variant index {variant_index} reports {allele_count} alleles.",
            )));
        }

        let reference_allele_code = read_allele(
            cursor,
            &mut metadata_field_buffer,
            &mut metadata_text_dictionary,
            variant_index,
            "reference allele",
        )?;
        let counted_allele_code = read_allele(
            cursor,
            &mut metadata_field_buffer,
            &mut metadata_text_dictionary,
            variant_index,
            "counted allele",
        )?;

        let total_block_length = cursor.read_u32()?;
        let (probability_payload_offset, probability_payload_length, uncompressed_block_length) = match compression_type
        {
            CompressionType::None => (cursor.position(), total_block_length, total_block_length),
            CompressionType::Zlib | CompressionType::Zstandard => {
                let uncompressed_block_length = cursor.read_u32()?;
                let probability_payload_length = total_block_length
                    .checked_sub(4)
                    .filter(|payload_length| *payload_length != 0)
                    .ok_or_else(|| {
                        BgenError::InvalidFormat(
                            "Compressed BGEN blocks must include a four-byte uncompressed length prefix and a non-empty payload."
                                .to_string(),
                        )
                    })?;
                (cursor.position(), probability_payload_length, uncompressed_block_length)
            }
        };
        let probability_payload_byte_count = u32_to_usize(probability_payload_length)?;
        validate_source_window_field_length(probability_payload_byte_count, variant_index, "probability payload")?;
        let uncompressed_block_byte_count = u32_to_usize(uncompressed_block_length)?;
        validate_source_window_field_length(
            uncompressed_block_byte_count,
            variant_index,
            "uncompressed probability block",
        )?;
        validate_block_length(
            uncompressed_block_byte_count,
            minimum_block_length,
            maximum_block_length,
            variant_index,
        )?;
        cursor.skip_payload_exact(probability_payload_byte_count)?;

        let variant_identifier_stop = variant_identifier_text
            .len()
            .checked_add(resolved_identifier_text.len())
            .and_then(|identifier_stop| u32::try_from(identifier_stop).ok())
            .ok_or_else(|| {
                BgenError::Range("BGEN variant identifiers exceed the four-gibibyte metadata arena limit.".to_string())
            })?;
        variant_identifier_text
            .try_reserve(resolved_identifier_text.len())
            .map_err(|source| BgenError::Range(format!("Could not reserve BGEN variant identifier text: {source}.")))?;
        variant_identifier_text.push_str(resolved_identifier_text);
        variant_identifier_offsets.push(variant_identifier_stop);
        let variant_record = VariantRecord {
            probability_payload_offset,
            probability_payload_length,
            declared_uncompressed_block_length: uncompressed_block_length,
        };
        if variant_index == 0 {
            validate_variant_probability_block(
                source,
                compression_type,
                &variant_record,
                sample_count,
                "first variant",
            )?;
        }
        if chromosome_codes.last().is_some_and(|previous_code| *previous_code != chromosome_code) {
            chromosome_boundary_indices.try_reserve(1).map_err(|source| {
                BgenError::Range(format!("Could not reserve a BGEN chromosome boundary: {source}."))
            })?;
            chromosome_boundary_indices.push(variant_index);
        }
        chromosome_codes.push(chromosome_code);
        position.push(variant_position);
        allele_one_codes.push(counted_allele_code);
        allele_two_codes.push(reference_allele_code);
        variant_records.push(variant_record);
    }

    if cursor.position() != source.length() {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN header reports {variant_count} variants ending at byte {}, but the source contains {} bytes.",
            cursor.position(),
            source.length(),
        )));
    }

    chromosome_boundary_indices.try_reserve(1).map_err(|source| {
        BgenError::Range(format!("Could not reserve the final BGEN chromosome boundary: {source}."))
    })?;
    chromosome_boundary_indices.push(variant_count);
    Ok(ParsedVariantIndex {
        variant_records,
        variant_metadata: Arc::new(
            VariantMetadataStore::from_parts(
                metadata_text_dictionary.values.into_boxed_slice(),
                chromosome_codes.into_boxed_slice(),
                variant_identifier_text.into_boxed_str(),
                variant_identifier_offsets.into_boxed_slice(),
                position.into_boxed_slice(),
                allele_one_codes.into_boxed_slice(),
                allele_two_codes.into_boxed_slice(),
            )
            .map_err(|error| {
                contextualize_variant_metadata_invariant("Parsed BGEN variant metadata violates its invariants", error)
            })?,
        ),
        chromosome_boundary_indices,
    })
}

fn reserved_variant_column<Value>(capacity: usize, label: &str) -> Result<Vec<Value>, BgenError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|source| BgenError::Range(format!("Could not reserve {capacity} BGEN {label}: {source}.")))?;
    Ok(values)
}

fn validate_variant_count_against_source(
    source: &BgenSource,
    first_variant_offset: u64,
    variant_count: usize,
) -> Result<(), BgenError> {
    let remaining_length = source
        .length()
        .checked_sub(first_variant_offset)
        .ok_or_else(|| BgenError::InvalidFormat("BGEN first-variant offset exceeds the source length.".to_string()))?;
    let maximum_variant_count = remaining_length / MINIMUM_VARIANT_INDEX_BYTES;
    let variant_count_u64 = u64::try_from(variant_count)
        .map_err(|_| BgenError::Range("BGEN variant count does not fit uint64.".to_string()))?;
    if variant_count_u64 > maximum_variant_count {
        return Err(BgenError::InvalidFormat(format!(
            "BGEN header reports {variant_count} variants, but only {remaining_length} bytes remain after the first-variant offset.",
        )));
    }
    Ok(())
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

#[inline]
fn read_snapshot_allele(
    cursor: &mut BgenSnapshotCursor<'_>,
    metadata_text_dictionary: &mut StringDictionaryBuilder,
    variant_index: usize,
    field_label: &str,
) -> Result<u32, BgenError> {
    let allele_length = u32_to_usize(cursor.read_u32()?)?;
    validate_source_window_field_length(allele_length, variant_index, field_label)?;
    let allele_bytes = cursor.read_bytes(allele_length)?;
    let allele_text = parse_metadata_text(allele_bytes, variant_index, field_label)?;
    metadata_text_dictionary.intern(allele_text)
}

#[inline]
fn read_allele(
    cursor: &mut BgenSourceCursor<'_>,
    allele_buffer: &mut Vec<u8>,
    metadata_text_dictionary: &mut StringDictionaryBuilder,
    variant_index: usize,
    field_label: &str,
) -> Result<u32, BgenError> {
    let allele_length = u32_to_usize(cursor.read_u32()?)?;
    validate_source_window_field_length(allele_length, variant_index, field_label)?;
    let allele_bytes = cursor.read_bytes(allele_length, allele_buffer)?;
    let allele_text = parse_metadata_text(allele_bytes.as_ref(), variant_index, field_label)?;
    metadata_text_dictionary.intern(allele_text)
}

#[inline]
fn parse_metadata_text<'bytes>(
    bytes: &'bytes [u8],
    variant_index: usize,
    field_label: &str,
) -> Result<&'bytes str, BgenError> {
    std::str::from_utf8(bytes).map_err(|source| {
        BgenError::InvalidFormat(format!(
            "Variant index {variant_index} {field_label} contains invalid UTF-8: {source}.",
        ))
    })
}

#[inline]
fn validate_source_window_field_length(
    field_length: usize,
    variant_index: usize,
    field_label: &str,
) -> Result<(), BgenError> {
    if field_length > MAXIMUM_SOURCE_WINDOW_BYTE_COUNT {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant index {variant_index} {field_label} contains {field_length} bytes, but this reader supports at most {MAXIMUM_SOURCE_WINDOW_BYTE_COUNT} bytes in one Layout 2 source field.",
        )));
    }
    Ok(())
}

fn validate_variant_probability_block(
    source: &BgenSource,
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    variant_label: &str,
) -> Result<(), BgenError> {
    let mut source_buffer = Vec::new();
    let source_window = source.read_variant_window(std::slice::from_ref(variant_record), &mut source_buffer)?;
    let mut thread_scratch = ThreadScratch::default();
    let probability_block =
        read_probability_block(source_window, compression_type, variant_record, &mut thread_scratch)?;
    let observed_sample_count = u32_to_usize(read_u32_at(probability_block, 0)?)?;
    if observed_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "The {variant_label} stores {observed_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }
    Ok(())
}
