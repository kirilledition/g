use super::super::metadata::VariantRecord;
use super::super::simd;
use super::super::{BgenError, CompressionType};
use super::matrix::ThreadScratch;

pub(in crate::bgen) const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
pub(in crate::bgen) const PLOIDY_MASK: u8 = 0x3F;
const UNUSED_PLOIDY_FLAG_MASK: u8 = 0x40;

pub(in crate::bgen) struct ParsedLayoutTwoProbabilityBlock<'block> {
    pub(in crate::bgen) minimum_ploidy: u8,
    pub(in crate::bgen) maximum_ploidy: u8,
    pub(in crate::bgen) sample_ploidy_and_missingness: &'block [u8],
    pub(in crate::bgen) phased_flag: u8,
    pub(in crate::bgen) probability_bit_count: u8,
    pub(in crate::bgen) probability_bytes: &'block [u8],
}

pub(in crate::bgen) fn parse_layout_two_probability_block(
    probability_block: &[u8],
    sample_count: usize,
) -> Result<ParsedLayoutTwoProbabilityBlock<'_>, BgenError> {
    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(probability_block, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }

    let allele_count = read_u16_at(probability_block, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat("is not biallelic.".to_string()));
    }

    let minimum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let sample_ploidy_and_missingness = read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;
    let phased_flag = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(probability_block, cursor)?;
    cursor += 1;

    Ok(ParsedLayoutTwoProbabilityBlock {
        minimum_ploidy,
        maximum_ploidy,
        sample_ploidy_and_missingness,
        phased_flag,
        probability_bit_count,
        probability_bytes: &probability_block[cursor..],
    })
}

pub(in crate::bgen) fn validate_layout_two_probability_values(
    probability_block: &ParsedLayoutTwoProbabilityBlock<'_>,
    sample_count: usize,
) -> Result<bool, BgenError> {
    if probability_block.minimum_ploidy != 2 || probability_block.maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "uses ploidy bounds [{}, {}], but dosage reads require diploid variants.",
            probability_block.minimum_ploidy, probability_block.maximum_ploidy,
        )));
    }
    if probability_block.phased_flag > 1 {
        return Err(BgenError::InvalidFormat(format!(
            "uses phased flag {}, but BGEN Layout 2 requires 0 or 1.",
            probability_block.phased_flag,
        )));
    }
    if !(1..=32).contains(&probability_block.probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "uses {} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
            probability_block.probability_bit_count,
        )));
    }

    let expected_probability_byte_count =
        layout_two_probability_byte_count(sample_count, probability_block.probability_bit_count)?;
    if probability_block.probability_bytes.len() != expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "contains {} probability bytes, but its encoding requires exactly {expected_probability_byte_count}.",
            probability_block.probability_bytes.len(),
        )));
    }

    let all_samples_present =
        simd::all_samples_present_diploid_simd_or_scalar(probability_block.sample_ploidy_and_missingness);
    if all_samples_present && probability_block.phased_flag == 0 && probability_block.probability_bit_count == 8 {
        if simd::all_unphased_eight_bit_probability_pairs_valid_simd_or_scalar(probability_block.probability_bytes) {
            return Ok(false);
        }
        return Err(BgenError::InvalidFormat(
            "contains an 8-bit probability pair whose values sum above 255.".to_string(),
        ));
    }

    let maximum_probability_value = if probability_block.probability_bit_count == 32 {
        u64::from(u32::MAX)
    } else {
        (1_u64 << probability_block.probability_bit_count) - 1
    };
    let mut probability_reader = PackedProbabilityReader::new(probability_block.probability_bytes);
    let mut has_missing_samples = false;
    for (file_sample_index, ploidy_and_missingness) in
        probability_block.sample_ploidy_and_missingness.iter().copied().enumerate()
    {
        let is_missing = validate_diploid_sample_flags(ploidy_and_missingness, file_sample_index)?;
        has_missing_samples |= is_missing;
        let first_probability = probability_reader.read_probability(probability_block.probability_bit_count)?;
        let second_probability = probability_reader.read_probability(probability_block.probability_bit_count)?;
        validate_stored_probability_pair(
            first_probability,
            second_probability,
            maximum_probability_value,
            probability_block.phased_flag,
            is_missing,
            file_sample_index,
        )?;
    }
    if !probability_reader.has_only_zero_padding() {
        return Err(BgenError::InvalidFormat(
            "contains nonzero padding bits after its stored probabilities.".to_string(),
        ));
    }
    Ok(has_missing_samples)
}

pub(in crate::bgen) fn layout_two_probability_byte_count(
    sample_count: usize,
    probability_bit_count: u8,
) -> Result<usize, BgenError> {
    let stored_probability_count = sample_count.checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while sizing BGEN probability values.".to_string())
    })?;
    stored_probability_count
        .checked_mul(usize::from(probability_bit_count))
        .map(|bit_count| bit_count.div_ceil(8))
        .ok_or_else(|| BgenError::InvalidFormat("Integer overflow while sizing BGEN probability bits.".to_string()))
}

pub(in crate::bgen) fn validate_diploid_sample_flags(
    ploidy_and_missingness: u8,
    file_sample_index: usize,
) -> Result<bool, BgenError> {
    if (ploidy_and_missingness & UNUSED_PLOIDY_FLAG_MASK) != 0 {
        return Err(BgenError::InvalidFormat(format!(
            "contains reserved ploidy flag bits at file sample index {file_sample_index}.",
        )));
    }
    let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
    if observed_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
        )));
    }
    Ok((ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0)
}

#[inline]
pub(in crate::bgen) fn validate_stored_probability_pair(
    first_probability: u32,
    second_probability: u32,
    maximum_probability_value: u64,
    phased_flag: u8,
    is_missing: bool,
    file_sample_index: usize,
) -> Result<(), BgenError> {
    if is_missing && (first_probability != 0 || second_probability != 0) {
        return Err(BgenError::InvalidFormat(format!(
            "stores nonzero probabilities for missing file sample index {file_sample_index}.",
        )));
    }
    if phased_flag == 0 && u64::from(first_probability) + u64::from(second_probability) > maximum_probability_value {
        return Err(BgenError::InvalidFormat(format!(
            "stores probabilities above the normalization range at file sample index {file_sample_index}.",
        )));
    }
    Ok(())
}

pub(in crate::bgen) fn read_probability_block<'a>(
    mmap: &'a [u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    thread_scratch: &'a mut ThreadScratch,
) -> Result<&'a [u8], BgenError> {
    match compression_type {
        CompressionType::None => {
            read_exact_bytes(mmap, variant_record.probability_payload_offset, variant_record.probability_payload_length)
        }
        CompressionType::Zlib => {
            let compressed_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            decompress_zlib_block_into_scratch(
                compressed_payload,
                variant_record.declared_uncompressed_block_length,
                thread_scratch,
            )?;
            Ok(&thread_scratch.decompressed_probability_block[..variant_record.declared_uncompressed_block_length])
        }
        CompressionType::Zstandard => {
            let compressed_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            decompress_zstandard_block_into_scratch(
                compressed_payload,
                variant_record.declared_uncompressed_block_length,
                thread_scratch,
            )?;
            Ok(&thread_scratch.decompressed_probability_block[..variant_record.declared_uncompressed_block_length])
        }
    }
}

fn decompress_zlib_block_into_scratch(
    compressed_payload: &[u8],
    expected_length: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), BgenError> {
    ensure_decompression_buffer_length(&mut thread_scratch.decompressed_probability_block, expected_length)?;
    let output_buffer = &mut thread_scratch.decompressed_probability_block[..expected_length];
    let mut consumed_input_length = 0;
    // SAFETY: the per-thread decompressor is live and uniquely borrowed. The
    // index rejects empty compressed payloads, and both slices remain valid for
    // the exact lengths passed to C.
    let result = unsafe {
        libdeflate_sys::libdeflate_zlib_decompress_ex(
            thread_scratch.zlib_decompressor.as_ptr(),
            compressed_payload.as_ptr().cast(),
            compressed_payload.len(),
            output_buffer.as_mut_ptr().cast(),
            output_buffer.len(),
            &raw mut consumed_input_length,
            std::ptr::null_mut(),
        )
    };
    match result {
        libdeflate_sys::libdeflate_result_LIBDEFLATE_SUCCESS => {}
        libdeflate_sys::libdeflate_result_LIBDEFLATE_BAD_DATA => {
            return Err(BgenError::InvalidFormat("Zlib-compressed BGEN block contains invalid zlib data.".to_string()));
        }
        libdeflate_sys::libdeflate_result_LIBDEFLATE_SHORT_OUTPUT => {
            return Err(BgenError::InvalidFormat(
                "Zlib-compressed BGEN block produced an incomplete output block.".to_string(),
            ));
        }
        libdeflate_sys::libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
            return Err(BgenError::InvalidFormat(
                "Zlib-compressed BGEN block exceeds its declared uncompressed length.".to_string(),
            ));
        }
        unexpected_result => {
            return Err(BgenError::InvalidFormat(format!(
                "Zlib decompression returned unsupported result code {unexpected_result}.",
            )));
        }
    }
    if consumed_input_length != compressed_payload.len() {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib-compressed BGEN block consumed {consumed_input_length} of {} payload bytes.",
            compressed_payload.len(),
        )));
    }
    Ok(())
}

fn decompress_zstandard_block_into_scratch(
    compressed_payload: &[u8],
    expected_length: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), BgenError> {
    ensure_decompression_buffer_length(&mut thread_scratch.decompressed_probability_block, expected_length)?;
    let zstandard_decompressor =
        thread_scratch.zstandard_decompressor.get_or_insert_with(zstd::bulk::Decompressor::default);
    let decompressed_length = zstandard_decompressor
        .decompress_to_buffer(compressed_payload, &mut thread_scratch.decompressed_probability_block[..expected_length])
        .map_err(|error| {
            BgenError::InvalidFormat(format!("Zstandard-compressed BGEN block contains invalid data: {error}"))
        })?;
    if decompressed_length != expected_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zstandard-compressed BGEN block expanded to {decompressed_length} bytes, but the header declared {expected_length} bytes.",
        )));
    }
    Ok(())
}

fn ensure_decompression_buffer_length(buffer: &mut Vec<u8>, expected_length: usize) -> Result<(), BgenError> {
    if buffer.len() >= expected_length {
        return Ok(());
    }
    buffer.try_reserve_exact(expected_length - buffer.len()).map_err(|source| {
        BgenError::Range(format!(
            "Could not reserve {expected_length} bytes for BGEN probability decompression: {source}.",
        ))
    })?;
    buffer.resize(expected_length, 0);
    Ok(())
}

pub(super) struct PackedProbabilityReader<'a> {
    packed_probability_bytes: &'a [u8],
    byte_offset: usize,
    bit_buffer: u64,
    buffered_bit_count: u8,
}

impl<'a> PackedProbabilityReader<'a> {
    pub(super) fn new(packed_probability_bytes: &'a [u8]) -> Self {
        Self { packed_probability_bytes, byte_offset: 0, bit_buffer: 0, buffered_bit_count: 0 }
    }

    pub(super) fn read_probability(&mut self, bit_count: u8) -> Result<u32, BgenError> {
        while self.buffered_bit_count < bit_count {
            let next_probability_byte = self.packed_probability_bytes.get(self.byte_offset).ok_or_else(|| {
                BgenError::InvalidFormat(
                    "Packed BGEN probability stream ended before all probabilities were decoded.".to_string(),
                )
            })?;
            self.bit_buffer |= u64::from(*next_probability_byte) << self.buffered_bit_count;
            self.buffered_bit_count += 8;
            self.byte_offset += 1;
        }

        let mask = if bit_count == 32 { u64::from(u32::MAX) } else { (1_u64 << bit_count) - 1 };
        let probability_value =
            u32::try_from(self.bit_buffer & mask).expect("masked BGEN probability value should fit u32");
        self.bit_buffer >>= bit_count;
        self.buffered_bit_count -= bit_count;
        Ok(probability_value)
    }

    pub(super) fn has_only_zero_padding(&self) -> bool {
        self.byte_offset == self.packed_probability_bytes.len() && self.bit_buffer == 0
    }
}

fn read_u8_at(buffer: &[u8], offset: usize) -> Result<u8, BgenError> {
    Ok(*read_exact_bytes(buffer, offset, 1)?
        .first()
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected empty byte slice.".to_string()))?)
}

pub(in crate::bgen) fn read_u16_at(buffer: &[u8], offset: usize) -> Result<u16, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 2)?;
    let byte_array: [u8; 2] = bytes
        .try_into()
        .map_err(|_| BgenError::InvalidFormat("Failed to decode a two-byte integer from the BGEN file.".to_string()))?;
    Ok(u16::from_le_bytes(byte_array))
}

pub(in crate::bgen) fn read_u32_at(buffer: &[u8], offset: usize) -> Result<u32, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 4)?;
    let byte_array: [u8; 4] = bytes.try_into().map_err(|_| {
        BgenError::InvalidFormat("Failed to decode a four-byte integer from the BGEN file.".to_string())
    })?;
    Ok(u32::from_le_bytes(byte_array))
}

pub(in crate::bgen) fn read_exact_bytes(buffer: &[u8], offset: usize, length: usize) -> Result<&[u8], BgenError> {
    let stop = offset
        .checked_add(length)
        .ok_or_else(|| BgenError::InvalidFormat("Integer overflow while slicing BGEN file bytes.".to_string()))?;
    buffer
        .get(offset..stop)
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected end of file while reading BGEN bytes.".to_string()))
}

pub(in crate::bgen) fn u32_to_usize(value: u32) -> Result<usize, BgenError> {
    usize::try_from(value).map_err(|_| {
        BgenError::InvalidFormat(format!(
            "BGEN integer value {value} does not fit into the native platform usize type.",
        ))
    })
}
