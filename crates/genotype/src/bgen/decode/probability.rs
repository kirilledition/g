use std::mem::MaybeUninit;
use std::time::Instant;

use flate2::{FlushDecompress, Status};

use super::super::metadata::VariantRecord;
use super::super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::super::{BgenError, CompressionType};
use super::matrix::ThreadScratch;

pub(in crate::bgen) fn read_probability_block<'a>(
    mmap: &'a [u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    thread_scratch: &'a mut ThreadScratch,
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    profiling_enabled: bool,
) -> Result<&'a [u8], BgenError> {
    let compressed_block_fetch_start_time = profiling_enabled.then(Instant::now);
    match compression_type {
        CompressionType::None => {
            let block_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            if let Some(compressed_block_fetch_start_time) = compressed_block_fetch_start_time {
                thread_local_profile_snapshot.compressed_block_fetch_ns +=
                    elapsed_nanoseconds(compressed_block_fetch_start_time);
                thread_local_profile_snapshot.compressed_block_fetch_count += 1;
                thread_local_profile_snapshot.compressed_byte_count +=
                    u64::try_from(variant_record.probability_payload_length).unwrap_or(u64::MAX);
            }
            if profiling_enabled {
                thread_local_profile_snapshot.uncompressed_byte_count +=
                    u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
            }
            Ok(block_payload)
        }
        CompressionType::Zlib => {
            let compressed_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            if let Some(compressed_block_fetch_start_time) = compressed_block_fetch_start_time {
                thread_local_profile_snapshot.compressed_block_fetch_ns +=
                    elapsed_nanoseconds(compressed_block_fetch_start_time);
                thread_local_profile_snapshot.compressed_block_fetch_count += 1;
                thread_local_profile_snapshot.compressed_byte_count +=
                    u64::try_from(variant_record.probability_payload_length).unwrap_or(u64::MAX);
            }

            let decompression_start_time = profiling_enabled.then(Instant::now);
            decompress_zlib_block_into_scratch(
                compressed_payload,
                variant_record.declared_uncompressed_block_length,
                thread_scratch,
            )?;
            if let Some(decompression_start_time) = decompression_start_time {
                thread_local_profile_snapshot.decompression_ns += elapsed_nanoseconds(decompression_start_time);
                thread_local_profile_snapshot.decompression_count += 1;
            }
            if profiling_enabled {
                thread_local_profile_snapshot.uncompressed_byte_count +=
                    u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
                thread_local_profile_snapshot.zlib_stream_count += 1;
            }
            Ok(thread_scratch.decompressed_probability_block.as_slice())
        }
    }
}

fn decompress_zlib_block_into_scratch(
    compressed_payload: &[u8],
    expected_length: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), BgenError> {
    thread_scratch.decompressed_probability_block.clear();
    if thread_scratch.decompressed_probability_block.capacity() < expected_length {
        thread_scratch
            .decompressed_probability_block
            .reserve(expected_length - thread_scratch.decompressed_probability_block.capacity());
    }
    thread_scratch.zlib_decompressor.reset(true);
    let total_output_before = thread_scratch.zlib_decompressor.total_out();
    let output_buffer: &mut [MaybeUninit<u8>] =
        &mut thread_scratch.decompressed_probability_block.spare_capacity_mut()[..expected_length];
    let status = thread_scratch
        .zlib_decompressor
        .decompress_uninit(compressed_payload, output_buffer, FlushDecompress::Finish)
        .map_err(std::io::Error::from)?;
    if status != Status::StreamEnd {
        return Err(BgenError::InvalidFormat(
            "Zlib-compressed BGEN block did not terminate at stream end.".to_string(),
        ));
    }
    let decompressed_length = usize::try_from(thread_scratch.zlib_decompressor.total_out() - total_output_before)
        .map_err(|_| BgenError::InvalidFormat("Decoded zlib block length does not fit into usize.".to_string()))?;
    if decompressed_length != expected_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib-compressed BGEN block expanded to {decompressed_length} bytes, but the header declared {expected_length} bytes.",
        )));
    }
    unsafe {
        thread_scratch.decompressed_probability_block.set_len(decompressed_length);
    }
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
}

pub(in crate::bgen) fn read_u8_at(buffer: &[u8], offset: usize) -> Result<u8, BgenError> {
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
