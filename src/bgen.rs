use std::borrow::Cow;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use thiserror::Error;

const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
const PLOIDY_MASK: u8 = 0x3F;
const VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES: usize = 2;
const ALLELE_LENGTH_SIZE_IN_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionType {
    None,
    Zlib,
    Zstd,
}

impl TryFrom<u32> for CompressionType {
    type Error = BgenError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Zlib),
            2 => Ok(Self::Zstd),
            unsupported_value => Err(BgenError::UnsupportedFormat(format!(
                "Unsupported BGEN compression flag {unsupported_value}. Only uncompressed, zlib, and zstd blocks are supported.",
            ))),
        }
    }
}

#[derive(Debug)]
pub struct BgenReaderCore {
    bgen_path: PathBuf,
    mmap: Mmap,
    sample_count: usize,
    variant_count: usize,
    contains_embedded_samples: bool,
    sample_identifiers: Vec<String>,
    compression_type: CompressionType,
    variant_records: Vec<VariantRecord>,
    chromosome_boundary_indices: Vec<usize>,
}

#[derive(Debug)]
struct VariantRecord {
    genotype_block_offset: usize,
    chromosome: String,
    resolved_variant_identifier: String,
    position: i64,
    counted_allele: String,
    reference_allele: String,
}

#[derive(Error, Debug)]
pub enum BgenError {
    #[error("{0}")]
    InvalidFormat(String),
    #[error("{0}")]
    UnsupportedFormat(String),
    #[error("{0}")]
    Range(String),
    #[error("I/O error while reading BGEN file: {0}")]
    Io(#[from] std::io::Error),
}

pub type VariantMetadataLists = (Vec<String>, Vec<String>, Vec<i64>, Vec<String>, Vec<String>);

impl BgenReaderCore {
    pub fn open(bgen_path: &Path) -> Result<Self, BgenError> {
        let file = File::open(bgen_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let first_variant_offset = 4 + u32_to_usize(read_u32_at(&mmap, 0)?)?;
        let header_block_length = u32_to_usize(read_u32_at(&mmap, 4)?)?;
        if header_block_length < 20 {
            return Err(BgenError::InvalidFormat(format!(
                "BGEN header block length must be at least 20 bytes. Observed {header_block_length}.",
            )));
        }
        let variant_count = u32_to_usize(read_u32_at(&mmap, 8)?)?;
        let sample_count = u32_to_usize(read_u32_at(&mmap, 12)?)?;

        let magic_offset = 16;
        let magic_number = read_exact_bytes(&mmap, magic_offset, 4)?;
        if magic_number != b"bgen" && magic_number != [0_u8, 0, 0, 0] {
            return Err(BgenError::InvalidFormat(
                "BGEN header magic number must be `bgen` or four zero bytes.".to_string(),
            ));
        }

        let header_flags_offset = 4 + header_block_length - 4;
        let header_flags = read_u32_at(&mmap, header_flags_offset)?;
        let compression_type = CompressionType::try_from(header_flags & 0b11)?;
        let layout_identifier = (header_flags >> 2) & 0b1111;
        if layout_identifier != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only BGEN Layout 2 is supported by the native Rust reader. Observed layout {layout_identifier}.",
            )));
        }
        let contains_embedded_samples = ((header_flags >> 31) & 1) == 1;

        let sample_block_offset = 4 + header_block_length;
        let sample_identifiers = if contains_embedded_samples {
            parse_sample_identifier_block(&mmap, sample_block_offset, first_variant_offset, sample_count)?
        } else {
            Vec::new()
        };

        let variant_records =
            parse_variant_records(&mmap, first_variant_offset, variant_count, sample_count, compression_type)?;
        let chromosome_boundary_indices = build_chromosome_boundary_indices(&variant_records);

        Ok(Self {
            bgen_path: bgen_path.to_path_buf(),
            mmap,
            sample_count,
            variant_count,
            contains_embedded_samples,
            sample_identifiers,
            compression_type,
            variant_records,
            chromosome_boundary_indices,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn variant_count(&self) -> usize {
        self.variant_count
    }

    pub fn contains_embedded_samples(&self) -> bool {
        self.contains_embedded_samples
    }

    pub fn sample_identifiers(&self) -> Vec<String> {
        self.sample_identifiers.clone()
    }

    pub fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.chromosome_boundary_indices.clone()
    }

    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataLists, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;

        let selected_variant_records = &self.variant_records[variant_start..variant_stop];
        let chromosome_values = selected_variant_records
            .iter()
            .map(|variant_record| variant_record.chromosome.clone())
            .collect();
        let variant_identifier_values = selected_variant_records
            .iter()
            .map(|variant_record| variant_record.resolved_variant_identifier.clone())
            .collect();
        let position_values = selected_variant_records
            .iter()
            .map(|variant_record| variant_record.position)
            .collect();
        let allele_one_values = selected_variant_records
            .iter()
            .map(|variant_record| variant_record.counted_allele.clone())
            .collect();
        let allele_two_values = selected_variant_records
            .iter()
            .map(|variant_record| variant_record.reference_allele.clone())
            .collect();

        Ok((
            chromosome_values,
            variant_identifier_values,
            position_values,
            allele_one_values,
            allele_two_values,
        ))
    }

    pub fn read_dosage_f32(
        &self,
        sample_indices: &[i64],
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<Vec<f32>, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection = build_sample_selection(self.sample_count, sample_indices)?;
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let selected_variant_records = &self.variant_records[variant_start..variant_stop];

        let mut row_major_dosage_values = vec![0.0_f32; selected_sample_count * selected_variant_count];
        if selected_sample_count == 0 || selected_variant_count == 0 {
            return Ok(row_major_dosage_values);
        }

        let output_pointer_address = row_major_dosage_values.as_mut_ptr() as usize;
        selected_variant_records
            .par_iter()
            .enumerate()
            .try_for_each(|(variant_index, variant_record)| {
                let output_pointer = output_pointer_address as *mut f32;
                decode_variant_dosages_into_row_major_matrix(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    &sample_selection.file_to_selected_index,
                    variant_record,
                    output_pointer,
                    variant_index,
                    selected_variant_count,
                )
            })?;

        Ok(row_major_dosage_values)
    }

    pub fn bgen_path(&self) -> &Path {
        &self.bgen_path
    }
}

#[derive(Debug)]
struct SampleSelection {
    selected_sample_count: usize,
    file_to_selected_index: Vec<usize>,
}

fn validate_variant_bounds(variant_start: usize, variant_stop: usize, variant_count: usize) -> Result<(), BgenError> {
    if variant_start > variant_stop || variant_stop > variant_count {
        return Err(BgenError::Range(format!(
            "Variant bounds must satisfy 0 <= start <= stop <= {variant_count}. Received start={variant_start}, stop={variant_stop}.",
        )));
    }
    Ok(())
}

fn build_sample_selection(sample_count: usize, sample_indices: &[i64]) -> Result<SampleSelection, BgenError> {
    let mut file_to_selected_index = vec![usize::MAX; sample_count];
    for (selected_index, raw_sample_index) in sample_indices.iter().enumerate() {
        let sample_index = usize::try_from(*raw_sample_index).map_err(|_| {
            BgenError::Range(format!(
                "Sample indices must be non-negative. Observed sample index {raw_sample_index}.",
            ))
        })?;
        if sample_index >= sample_count {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} is out of bounds for a BGEN file with {sample_count} samples.",
            )));
        }
        if file_to_selected_index[sample_index] != usize::MAX {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} was requested more than once in the same read.",
            )));
        }
        file_to_selected_index[sample_index] = selected_index;
    }
    Ok(SampleSelection {
        selected_sample_count: sample_indices.len(),
        file_to_selected_index,
    })
}

fn parse_sample_identifier_block(
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

fn parse_variant_records(
    mmap: &[u8],
    first_variant_offset: usize,
    variant_count: usize,
    sample_count: usize,
    compression_type: CompressionType,
) -> Result<Vec<VariantRecord>, BgenError> {
    let mut cursor = first_variant_offset;
    let mut variant_records = Vec::with_capacity(variant_count);

    for variant_index in 0..variant_count {
        let variant_identifier_length = usize::from(read_u16_at(mmap, cursor)?);
        cursor += VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES;
        let variant_identifier = String::from_utf8_lossy(read_exact_bytes(mmap, cursor, variant_identifier_length)?)
            .into_owned();
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
                genotype_block_offset,
                sample_count,
                "first variant",
            )?;
        }

        let reference_allele = allele_values[0].clone();
        let counted_allele = allele_values[1].clone();
        let resolved_variant_identifier = if rsid.is_empty() {
            variant_identifier
        } else {
            rsid.clone()
        };

        variant_records.push(VariantRecord {
            genotype_block_offset,
            chromosome,
            resolved_variant_identifier,
            position,
            counted_allele,
            reference_allele,
        });
    }

    Ok(variant_records)
}

fn build_chromosome_boundary_indices(variant_records: &[VariantRecord]) -> Vec<usize> {
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

fn validate_variant_probability_block(
    mmap: &[u8],
    compression_type: CompressionType,
    genotype_block_offset: usize,
    sample_count: usize,
    variant_label: &str,
) -> Result<(), BgenError> {
    let probability_block = read_probability_block(mmap, compression_type, genotype_block_offset)?;
    let observed_sample_count = u32_to_usize(read_u32_at(probability_block.as_ref(), 0)?)?;
    if observed_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "The {variant_label} stores {observed_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
fn decode_variant_dosages_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    file_to_selected_index: &[usize],
    variant_record: &VariantRecord,
    output_pointer: *mut f32,
    variant_index: usize,
    variant_count: usize,
) -> Result<(), BgenError> {
    let probability_block = read_probability_block(mmap, compression_type, variant_record.genotype_block_offset)?;
    let block_bytes = probability_block.as_ref();

    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(block_bytes, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let allele_count = read_u16_at(block_bytes, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not biallelic.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let minimum_ploidy = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but the native Rust reader currently supports diploid BGEN variants only.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(block_bytes, cursor, sample_count)?;
    cursor += sample_count;

    let phased_flag = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    if !(1..=32).contains(&probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' uses {probability_bit_count} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
            variant_record.resolved_variant_identifier,
        )));
    }

    if phased_flag == 0 && probability_bit_count == 8 {
        return decode_unphased_eight_bit_dosages_into_row_major_matrix(
            sample_ploidy_and_missingness,
            &block_bytes[cursor..],
            file_to_selected_index,
            variant_record,
            output_pointer,
            variant_index,
            variant_count,
        );
    }

    let probability_scale_denominator = if probability_bit_count == 32 {
        f64::from(u32::MAX)
    } else {
        f64::from((1_u32 << probability_bit_count) - 1)
    };
    let mut bit_reader = PackedProbabilityReader::new(&block_bytes[cursor..]);

    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }
        let is_missing = (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;

        let dosage_value = match phased_flag {
            0 => {
                let homozygous_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let heterozygous_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                if is_missing {
                    f32::NAN
                } else {
                    let dosage_value = 2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability);
                    dosage_value as f32
                }
            }
            1 => {
                let first_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let second_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                if is_missing {
                    f32::NAN
                } else {
                    let dosage_value =
                        2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability);
                    dosage_value as f32
                }
            }
            unsupported_flag => {
                return Err(BgenError::InvalidFormat(format!(
                    "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                    variant_record.resolved_variant_identifier,
                )))
            }
        };

        let selected_index = file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            let output_offset = (selected_index * variant_count) + variant_index;
            unsafe {
                // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
    }

    Ok(())
}

fn decode_unphased_eight_bit_dosages_into_row_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    file_to_selected_index: &[usize],
    variant_record: &VariantRecord,
    output_pointer: *mut f32,
    variant_index: usize,
    variant_count: usize,
) -> Result<(), BgenError> {
    let expected_probability_byte_count = sample_ploidy_and_missingness.len().checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if packed_probability_bytes.len() < expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' ended before all 8-bit probabilities were decoded.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let reciprocal_scale = 1.0_f32 / 255.0_f32;
    let mut probability_offset = 0;
    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let homozygous_reference_probability =
            f32::from(packed_probability_bytes[probability_offset]) * reciprocal_scale;
        let heterozygous_probability = f32::from(packed_probability_bytes[probability_offset + 1]) * reciprocal_scale;
        probability_offset += 2;

        let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
            f32::NAN
        } else {
            2.0_f32 - ((2.0_f32 * homozygous_reference_probability) + heterozygous_probability)
        };

        let selected_index = file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            let output_offset = (selected_index * variant_count) + variant_index;
            unsafe {
                // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                output_pointer.add(output_offset).write(dosage_value);
            }
        }
    }

    Ok(())
}

fn read_probability_block(
    mmap: &[u8],
    compression_type: CompressionType,
    genotype_block_offset: usize,
) -> Result<Cow<'_, [u8]>, BgenError> {
    let total_block_length = u32_to_usize(read_u32_at(mmap, genotype_block_offset)?)?;
    let block_payload_offset = genotype_block_offset + 4;
    match compression_type {
        CompressionType::None => {
            let block_payload = read_exact_bytes(mmap, block_payload_offset, total_block_length)?;
            Ok(Cow::Borrowed(block_payload))
        }
        CompressionType::Zlib | CompressionType::Zstd => {
            let uncompressed_block_length = u32_to_usize(read_u32_at(mmap, block_payload_offset)?)?;
            let compressed_payload_offset = block_payload_offset + 4;
            let compressed_payload_length = total_block_length.checked_sub(4).ok_or_else(|| {
                BgenError::InvalidFormat(
                    "Compressed BGEN blocks must include a four-byte uncompressed length prefix.".to_string(),
                )
            })?;
            let compressed_payload = read_exact_bytes(mmap, compressed_payload_offset, compressed_payload_length)?;
            let decompressed_block = match compression_type {
                CompressionType::Zlib => decompress_zlib_block(compressed_payload, uncompressed_block_length)?,
                CompressionType::Zstd => decompress_zstd_block(compressed_payload, uncompressed_block_length)?,
                CompressionType::None => unreachable!(),
            };
            Ok(Cow::Owned(decompressed_block))
        }
    }
}

fn decompress_zlib_block(compressed_payload: &[u8], expected_length: usize) -> Result<Vec<u8>, BgenError> {
    let mut decoder = ZlibDecoder::new(compressed_payload);
    let mut decompressed_block = Vec::with_capacity(expected_length);
    decoder.read_to_end(&mut decompressed_block)?;
    if decompressed_block.len() != expected_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib-compressed BGEN block expanded to {} bytes, but the header declared {expected_length} bytes.",
            decompressed_block.len(),
        )));
    }
    Ok(decompressed_block)
}

fn decompress_zstd_block(compressed_payload: &[u8], expected_length: usize) -> Result<Vec<u8>, BgenError> {
    let decompressed_block = zstd::bulk::decompress(compressed_payload, expected_length).map_err(|error| {
        BgenError::InvalidFormat(format!(
            "Failed to decompress a zstd-compressed BGEN block: {error}",
        ))
    })?;
    if decompressed_block.len() != expected_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zstd-compressed BGEN block expanded to {} bytes, but the header declared {expected_length} bytes.",
            decompressed_block.len(),
        )));
    }
    Ok(decompressed_block)
}

struct PackedProbabilityReader<'a> {
    packed_probability_bytes: &'a [u8],
    bit_offset: usize,
}

impl<'a> PackedProbabilityReader<'a> {
    fn new(packed_probability_bytes: &'a [u8]) -> Self {
        Self {
            packed_probability_bytes,
            bit_offset: 0,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_probability(&mut self, bit_count: u8) -> Result<u32, BgenError> {
        let bit_count_usize = usize::from(bit_count);
        let byte_offset = self.bit_offset / 8;
        let bit_index_in_byte = self.bit_offset % 8;
        let last_required_bit = self.bit_offset + bit_count_usize;
        let last_required_byte = last_required_bit.div_ceil(8);
        if last_required_byte > self.packed_probability_bytes.len() {
            return Err(BgenError::InvalidFormat(
                "Packed BGEN probability stream ended before all probabilities were decoded.".to_string(),
            ));
        }

        let mut window = 0_u64;
        let bytes_to_copy = (self.packed_probability_bytes.len() - byte_offset).min(8);
        for copied_byte_index in 0..bytes_to_copy {
            window |= u64::from(self.packed_probability_bytes[byte_offset + copied_byte_index])
                << (copied_byte_index * 8);
        }

        let mask = if bit_count == 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << bit_count) - 1
        };
        let probability_value = ((window >> bit_index_in_byte) & mask) as u32;
        self.bit_offset += bit_count_usize;
        Ok(probability_value)
    }
}

fn read_u8_at(buffer: &[u8], offset: usize) -> Result<u8, BgenError> {
    Ok(*read_exact_bytes(buffer, offset, 1)?
        .first()
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected empty byte slice.".to_string()))?)
}

fn read_u16_at(buffer: &[u8], offset: usize) -> Result<u16, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 2)?;
    let byte_array: [u8; 2] = bytes.try_into().map_err(|_| {
        BgenError::InvalidFormat("Failed to decode a two-byte integer from the BGEN file.".to_string())
    })?;
    Ok(u16::from_le_bytes(byte_array))
}

fn read_u32_at(buffer: &[u8], offset: usize) -> Result<u32, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 4)?;
    let byte_array: [u8; 4] = bytes.try_into().map_err(|_| {
        BgenError::InvalidFormat("Failed to decode a four-byte integer from the BGEN file.".to_string())
    })?;
    Ok(u32::from_le_bytes(byte_array))
}

fn read_exact_bytes(buffer: &[u8], offset: usize, length: usize) -> Result<&[u8], BgenError> {
    let stop = offset.checked_add(length).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while slicing BGEN file bytes.".to_string())
    })?;
    buffer.get(offset..stop).ok_or_else(|| {
        BgenError::InvalidFormat("Unexpected end of file while reading BGEN bytes.".to_string())
    })
}

fn u32_to_usize(value: u32) -> Result<usize, BgenError> {
    usize::try_from(value).map_err(|_| {
        BgenError::InvalidFormat(format!(
            "BGEN integer value {value} does not fit into the native platform usize type.",
        ))
    })
}
