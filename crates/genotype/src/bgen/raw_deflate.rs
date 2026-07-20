use std::fmt;
use std::sync::Arc;

use crate::common::{ChunkSpec, SessionBufferPool};

use super::BgenError;
use super::decode::read_exact_bytes;
use super::format::CompressionType;
use super::reader::{BgenReadSession, validate_variant_bounds};
use super::sample_selection::SampleSelection;

const MEMBER_METADATA_VALUE_COUNT: usize = 3;
const ZLIB_HEADER_LENGTH: usize = 2;
const ZLIB_ADLER32_LENGTH: usize = 4;
const ZLIB_COMPRESSION_METHOD_DEFLATE: u8 = 8;
const ZLIB_MAXIMUM_WINDOW_SIZE_CODE: u8 = 7;
const ZLIB_PRESET_DICTIONARY_FLAG: u8 = 0x20;

const _: () = assert!(g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT == size_of::<u32>());

#[derive(Debug, Default)]
pub(super) struct CompressedPacked8Storage {
    raw_deflate_words: Vec<u32>,
    member_metadata: Vec<u32>,
}

pub(super) type CompressedPacked8BufferPool = SessionBufferPool<CompressedPacked8Storage>;

#[derive(Debug)]
pub(super) struct CompressedPacked8SessionState {
    pool: Arc<CompressedPacked8BufferPool>,
    transfer: CompressedPacked8Transfer,
}

/// Sample selection applied after GPU decompression of packed8 BGEN rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompressedPacked8SampleSelection {
    /// Preserve one contiguous range, including identity selection at index zero.
    Contiguous { file_index_start: u32 },
    /// Gather file samples in the requested order using one session-owned conversion.
    Indexed { file_indices: Arc<[u32]> },
}

/// Immutable sample-transfer geometry shared by every compressed session batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompressedPacked8Transfer {
    /// Sample selection applied after decompression.
    pub sample_selection: CompressedPacked8SampleSelection,
    /// Number of samples stored in every decompressed BGEN row.
    pub file_sample_count: usize,
    /// Number of samples retained after GPU selection.
    pub selected_sample_count: usize,
}

/// Pooled raw-DEFLATE members and GPU upload metadata for one packed8 BGEN batch.
pub struct CompressedPacked8Batch {
    storage: CompressedPacked8Storage,
    pool: Arc<CompressedPacked8BufferPool>,
}

/// Fixed raw-DEFLATE slab shape derived from an actual run chunk plan.
pub struct CompressedPacked8BatchLayout {
    slab_byte_count: usize,
}

impl CompressedPacked8Batch {
    /// Return the four-byte-aligned raw-DEFLATE slab.
    #[must_use]
    pub fn raw_deflate_slab(&self) -> &[u8] {
        let slab_byte_count = self.storage.raw_deflate_words.len() * g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT;
        // SAFETY: u32 storage provides four-byte alignment, every byte in the
        // logical word range is initialized by the packer, and u8 permits every
        // possible bit pattern.
        unsafe { std::slice::from_raw_parts(self.storage.raw_deflate_words.as_ptr().cast(), slab_byte_count) }
    }

    /// Return array-of-struct member metadata with rows `[offset, size, expected_adler32]`.
    #[must_use]
    pub fn member_metadata(&self) -> &[u32] {
        &self.storage.member_metadata
    }
}

impl fmt::Debug for CompressedPacked8Batch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompressedPacked8Batch")
            .field("raw_deflate_byte_count", &self.raw_deflate_slab().len())
            .field("member_count", &(self.storage.member_metadata.len() / MEMBER_METADATA_VALUE_COUNT))
            .finish_non_exhaustive()
    }
}

impl Drop for CompressedPacked8Batch {
    fn drop(&mut self) {
        let mut storage = std::mem::take(&mut self.storage);
        storage.member_metadata.clear();
        self.pool.release(storage);
    }
}

struct ParsedZlibMember<'member> {
    raw_deflate_bytes: &'member [u8],
    expected_adler32: u32,
    zlib_header: u16,
}

fn build_compressed_sample_selection(sample_selection: &SampleSelection) -> CompressedPacked8SampleSelection {
    match sample_selection {
        SampleSelection::Identity { .. } => CompressedPacked8SampleSelection::Contiguous { file_index_start: 0 },
        SampleSelection::Contiguous { file_index_start, .. } => CompressedPacked8SampleSelection::Contiguous {
            file_index_start: u32::try_from(*file_index_start)
                .expect("BGEN sample indices from the uint32 file domain must fit uint32"),
        },
        SampleSelection::Indexed { selected_file_indices, .. } => CompressedPacked8SampleSelection::Indexed {
            file_indices: selected_file_indices
                .iter()
                .map(|file_index| {
                    u32::try_from(*file_index).expect("BGEN sample indices from the uint32 file domain must fit uint32")
                })
                .collect(),
        },
    }
}

impl BgenReadSession<'_> {
    fn compressed_packed8_session_state(&self) -> &CompressedPacked8SessionState {
        self.compressed_packed8_state.get_or_init(|| CompressedPacked8SessionState {
            pool: Arc::new(CompressedPacked8BufferPool::default()),
            transfer: CompressedPacked8Transfer {
                sample_selection: build_compressed_sample_selection(&self.sample_selection),
                file_sample_count: self.reader.sample_count(),
                selected_sample_count: self.sample_selection.selected_sample_count(),
            },
        })
    }

    /// Return immutable sample-transfer geometry shared by compressed batches.
    #[must_use]
    pub fn compressed_packed8_transfer(&self) -> &CompressedPacked8Transfer {
        &self.compressed_packed8_session_state().transfer
    }

    /// Pack real zlib members for GPU raw-DEFLATE decompression.
    ///
    /// The method strips RFC 1950 framing without decompressing, aligns each
    /// RFC 1951 member to four bytes, and retains the expected Adler-32 value in
    /// one interleaved metadata upload. Compute-only tail rows are represented
    /// by scalar geometry and are never fabricated in the compressed stream.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bounds, non-zlib sources, incomplete packed8
    /// validation, malformed zlib framing, or a batch exceeding uint32 metadata.
    pub fn pack_compressed_packed8_batch(
        &self,
        layout: &CompressedPacked8BatchLayout,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<CompressedPacked8Batch, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.reader.variant_count())?;
        let logical_variant_count = variant_stop - variant_start;
        if self.reader.compression_type != CompressionType::Zlib {
            return Err(BgenError::UnsupportedFormat(
                "Compressed packed8 delivery requires a zlib-compressed BGEN source.".to_string(),
            ));
        }
        self.reader.validate_packed8_probability_pair_preconditions()?;
        let compressed_state = self.compressed_packed8_session_state();

        let variant_records = &self.reader.variant_records[variant_start..variant_stop];
        let metadata_value_count = logical_variant_count
            .checked_mul(MEMBER_METADATA_VALUE_COUNT)
            .ok_or_else(|| BgenError::Range("Raw-DEFLATE member metadata length overflowed usize.".to_string()))?;
        let layout_slab_word_count = layout.slab_byte_count / g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT;
        let mut storage = compressed_state
            .pool
            .take_matching(|candidate| {
                candidate.raw_deflate_words.len() == layout_slab_word_count
                    && candidate.member_metadata.capacity() >= metadata_value_count
            })
            .unwrap_or_default();
        let slab_is_initialized = storage.raw_deflate_words.len() == layout_slab_word_count;
        prepare_storage(&mut storage, layout_slab_word_count, metadata_value_count)?;

        let slab_pointer = storage.raw_deflate_words.as_mut_ptr().cast::<u8>();
        let metadata_pointer = storage.member_metadata.spare_capacity_mut().as_mut_ptr().cast::<u32>();
        let mut member_offset = 0_usize;
        // Zero cannot be a valid DEFLATE zlib header, so it marks the cache as
        // uninitialized without adding an Option or borrowed cache to the hot
        // loop. The explicit zero check still sends a malformed zero header
        // through full validation.
        let mut last_validated_zlib_header = 0_u16;
        for (relative_variant_index, variant_record) in variant_records.iter().enumerate() {
            let member = self.reader.zlib_member(variant_record).map_err(|error| {
                self.reader.contextualize_variant_error(variant_start + relative_variant_index, error)
            })?;
            if member.zlib_header != last_validated_zlib_header || last_validated_zlib_header == 0 {
                validate_zlib_header(member.zlib_header.to_ne_bytes()).map_err(|error| {
                    self.reader.contextualize_variant_error(variant_start + relative_variant_index, error)
                })?;
                last_validated_zlib_header = member.zlib_header;
            }
            let member_length = member.raw_deflate_bytes.len();
            let aligned_member_length = align_member_length(member_length)?;
            let next_member_offset = member_offset
                .checked_add(aligned_member_length)
                .ok_or_else(|| BgenError::Range("Raw-DEFLATE batch byte count overflowed usize.".to_string()))?;
            if next_member_offset > layout.slab_byte_count {
                return Err(BgenError::Range(
                    "Raw-DEFLATE batch exceeds the slab shape derived from the run chunk plan.".to_string(),
                ));
            }
            let member_offset_u32 = u32::try_from(member_offset)
                .map_err(|_| BgenError::Range("Raw-DEFLATE member offset exceeds uint32.".to_string()))?;
            let member_length_u32 = u32::try_from(member_length)
                .map_err(|_| BgenError::Range("Raw-DEFLATE member length exceeds uint32.".to_string()))?;
            let metadata_offset = relative_variant_index * MEMBER_METADATA_VALUE_COUNT;
            // SAFETY: prepare_storage reserves the fixed slab and three
            // metadata values per member. The checked next offset proves this
            // member and its alignment gap fit before the disjoint writes. A
            // fresh slab initializes its gap explicitly; a pooled slab retains
            // initialized bytes from its previous fixed-shape packing. Three
            // scalar writes initialize every published metadata value.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    member.raw_deflate_bytes.as_ptr(),
                    slab_pointer.add(member_offset),
                    member_length,
                );
                if !slab_is_initialized {
                    std::ptr::write_bytes(
                        slab_pointer.add(member_offset + member_length),
                        0,
                        aligned_member_length - member_length,
                    );
                }
                std::ptr::write(metadata_pointer.add(metadata_offset), member_offset_u32);
                std::ptr::write(metadata_pointer.add(metadata_offset + 1), member_length_u32);
                std::ptr::write(metadata_pointer.add(metadata_offset + 2), member.expected_adler32);
            }
            member_offset = next_member_offset;
        }
        // SAFETY: member copies and alignment-gap writes initialize the real
        // prefix of a fresh slab. The checked layout bound proves the remaining
        // tail fits its allocation. Publishing the fixed word length happens
        // only after every byte is initialized. A pooled slab already has that
        // initialized length. Every metadata value was initialized by the
        // three writes per member above.
        unsafe {
            if !slab_is_initialized {
                std::ptr::write_bytes(slab_pointer.add(member_offset), 0, layout.slab_byte_count - member_offset);
                storage.raw_deflate_words.set_len(layout_slab_word_count);
            }
            storage.member_metadata.set_len(metadata_value_count);
        }

        Ok(CompressedPacked8Batch { storage, pool: Arc::clone(&compressed_state.pool) })
    }
}

impl super::reader::BgenReaderCore {
    /// Derive one fixed compressed slab shape from the actual run chunk plan.
    ///
    /// # Errors
    ///
    /// Returns an error for incomplete packed8 validation, invalid chunk
    /// bounds, malformed compressed lengths, or uint32 overflow. Non-zlib
    /// sources return `None` so callers can retain host delivery.
    pub fn plan_compressed_packed8_batch_layout(
        &self,
        chunk_specs: &[ChunkSpec],
    ) -> Result<Option<CompressedPacked8BatchLayout>, BgenError> {
        if self.compression_type != CompressionType::Zlib {
            return Ok(None);
        }
        self.validate_packed8_probability_pair_preconditions()?;

        let mut maximum_slab_byte_count = 0_usize;
        for chunk_spec in chunk_specs {
            validate_variant_bounds(
                chunk_spec.variant_start_index,
                chunk_spec.variant_stop_index,
                self.variant_count(),
            )?;
            let slab_byte_count = aligned_slab_byte_count(
                &self.variant_records[chunk_spec.variant_start_index..chunk_spec.variant_stop_index],
            )?;
            maximum_slab_byte_count = maximum_slab_byte_count.max(slab_byte_count);
        }
        let maximum_metadata_offset = usize::try_from(u32::MAX)
            .map_err(|_| BgenError::Range("The current usize domain cannot represent uint32 offsets.".to_string()))?;
        if maximum_slab_byte_count > maximum_metadata_offset {
            return Err(BgenError::Range(
                "Raw-DEFLATE batch layout exceeds the four-gibibyte uint32 offset domain.".to_string(),
            ));
        }
        Ok(Some(CompressedPacked8BatchLayout { slab_byte_count: maximum_slab_byte_count }))
    }

    fn zlib_member<'reader>(
        &'reader self,
        variant_record: &super::metadata::VariantRecord,
    ) -> Result<ParsedZlibMember<'reader>, BgenError> {
        let payload_offset = usize::try_from(variant_record.probability_payload_offset)
            .expect("uint64 BGEN offsets must fit the supported 64-bit usize domain");
        let payload_length = usize::try_from(variant_record.probability_payload_length)
            .map_err(|_| BgenError::Range("BGEN zlib payload length does not fit usize.".to_string()))?;
        let payload = read_exact_bytes(&self.mmap, payload_offset, payload_length)?;
        parse_zlib_member(payload)
    }
}

fn prepare_storage(
    storage: &mut CompressedPacked8Storage,
    layout_slab_word_count: usize,
    metadata_value_count: usize,
) -> Result<(), BgenError> {
    storage.member_metadata.clear();
    if storage.raw_deflate_words.len() != layout_slab_word_count {
        storage.raw_deflate_words.clear();
        if storage.raw_deflate_words.capacity() < layout_slab_word_count {
            storage.raw_deflate_words.try_reserve_exact(layout_slab_word_count).map_err(|source| {
                BgenError::Range(format!("Could not reserve {layout_slab_word_count} raw-DEFLATE words: {source}."))
            })?;
        }
    }
    if storage.member_metadata.capacity() < metadata_value_count {
        storage.member_metadata.try_reserve_exact(metadata_value_count).map_err(|source| {
            BgenError::Range(format!("Could not reserve {metadata_value_count} raw-DEFLATE metadata values: {source}."))
        })?;
    }
    Ok(())
}

fn aligned_slab_byte_count(variant_records: &[super::metadata::VariantRecord]) -> Result<usize, BgenError> {
    let mut slab_byte_count = 0_usize;
    for variant_record in variant_records {
        let raw_deflate_length = usize::try_from(variant_record.probability_payload_length)
            .map_err(|_| BgenError::Range("BGEN zlib payload length does not fit usize.".to_string()))?
            .checked_sub(ZLIB_HEADER_LENGTH + ZLIB_ADLER32_LENGTH)
            .filter(|member_length| *member_length != 0)
            .ok_or_else(|| {
                BgenError::InvalidFormat("Zlib framing must contain a non-empty raw-DEFLATE member.".to_string())
            })?;
        slab_byte_count = slab_byte_count
            .checked_add(align_member_length(raw_deflate_length)?)
            .ok_or_else(|| BgenError::Range("Raw-DEFLATE batch byte count overflowed usize.".to_string()))?;
    }
    Ok(slab_byte_count)
}

fn align_member_length(member_length: usize) -> Result<usize, BgenError> {
    member_length
        .checked_add(g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT - 1)
        .map(|padded_length| padded_length & !(g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT - 1))
        .ok_or_else(|| BgenError::Range("Raw-DEFLATE member alignment overflowed usize.".to_string()))
}

fn parse_zlib_member(payload: &[u8]) -> Result<ParsedZlibMember<'_>, BgenError> {
    let minimum_payload_length = ZLIB_HEADER_LENGTH + ZLIB_ADLER32_LENGTH + 1;
    if payload.len() < minimum_payload_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib payload contains {} bytes, but framing and a raw-DEFLATE member require at least {minimum_payload_length}.",
            payload.len(),
        )));
    }
    let zlib_header = u16::from_ne_bytes([payload[0], payload[1]]);
    let raw_deflate_stop = payload.len() - ZLIB_ADLER32_LENGTH;
    let expected_adler32 = u32::from_be_bytes(
        payload[raw_deflate_stop..].try_into().expect("validated zlib Adler-32 trailer must contain four bytes"),
    );
    Ok(ParsedZlibMember {
        raw_deflate_bytes: &payload[ZLIB_HEADER_LENGTH..raw_deflate_stop],
        expected_adler32,
        zlib_header,
    })
}

fn validate_zlib_header(zlib_header: [u8; ZLIB_HEADER_LENGTH]) -> Result<(), BgenError> {
    let [compression_method_and_window, flags] = zlib_header;
    if compression_method_and_window & 0x0F != ZLIB_COMPRESSION_METHOD_DEFLATE {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib payload uses compression method {}, but BGEN zlib delivery requires DEFLATE.",
            compression_method_and_window & 0x0F,
        )));
    }
    if compression_method_and_window >> 4 > ZLIB_MAXIMUM_WINDOW_SIZE_CODE {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib payload uses invalid window size code {}.",
            compression_method_and_window >> 4,
        )));
    }
    let header = u16::from(compression_method_and_window) << 8 | u16::from(flags);
    if header % 31 != 0 {
        return Err(BgenError::InvalidFormat(
            "Zlib CMF/FLG header does not satisfy the RFC 1950 FCHECK value.".to_string(),
        ));
    }
    if flags & ZLIB_PRESET_DICTIONARY_FLAG != 0 {
        return Err(BgenError::UnsupportedFormat(
            "Zlib preset dictionaries are not supported for raw-DEFLATE GPU delivery.".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::bgen::reader::BgenReaderCore;
    use crate::common::Packed8Compatibility;

    const PROBABILITY_BLOCK_LENGTH: u32 = 19;
    const RAW_DEFLATE_FIRST: [u8; 17] = [99, 102, 96, 96, 96, 98, 96, 2, 1, 6, 14, 6, 134, 255, 64, 4, 0];
    const RAW_DEFLATE_SECOND: [u8; 17] = [99, 102, 96, 96, 96, 98, 96, 2, 1, 6, 142, 255, 12, 12, 64, 4, 0];
    const RAW_DEFLATE_THIRD: [u8; 18] = [99, 102, 96, 96, 96, 98, 96, 2, 1, 6, 142, 6, 6, 134, 255, 255, 25, 0];
    const ZLIB_FIRST: [u8; 23] =
        [120, 1, 99, 102, 96, 96, 96, 98, 96, 2, 1, 6, 14, 6, 134, 255, 64, 4, 0, 6, 11, 2, 22];
    const ZLIB_SECOND: [u8; 23] =
        [120, 218, 99, 102, 96, 96, 96, 98, 96, 2, 1, 6, 142, 255, 12, 12, 64, 4, 0, 10, 7, 2, 22];
    const ZLIB_THIRD: [u8; 24] =
        [120, 156, 99, 102, 96, 96, 96, 98, 96, 2, 1, 6, 142, 6, 6, 134, 255, 255, 25, 0, 9, 11, 2, 150];

    struct TemporaryBgenFile {
        path: PathBuf,
    }

    impl TemporaryBgenFile {
        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TemporaryBgenFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn temporary_bgen_path() -> PathBuf {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after unix epoch").as_nanos();
        std::env::temp_dir().join(format!("g-raw-deflate-{}-{timestamp}.bgen", std::process::id()))
    }

    fn append_bgen_string(bytes: &mut Vec<u8>, value: &str) {
        let value_length = u16::try_from(value.len()).expect("BGEN string length should fit u16");
        bytes.extend_from_slice(&value_length.to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn append_zlib_variant(bytes: &mut Vec<u8>, variant_index: usize, zlib_payload: &[u8]) {
        append_bgen_string(bytes, &format!("var-{variant_index}"));
        append_bgen_string(bytes, &format!("rs-{variant_index}"));
        append_bgen_string(bytes, "22");
        bytes.extend_from_slice(
            &u32::try_from(variant_index + 1).expect("test variant position should fit u32").to_le_bytes(),
        );
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"A");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"G");
        let total_block_length =
            u32::try_from(zlib_payload.len() + size_of::<u32>()).expect("compressed test block length should fit u32");
        bytes.extend_from_slice(&total_block_length.to_le_bytes());
        bytes.extend_from_slice(&PROBABILITY_BLOCK_LENGTH.to_le_bytes());
        bytes.extend_from_slice(zlib_payload);
    }

    fn write_independent_zlib_bgen() -> TemporaryBgenFile {
        let path = temporary_bgen_path();
        let mut bytes = vec![0_u8; 24];
        bytes[0..4].copy_from_slice(&20_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&20_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&3_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&3_u32.to_le_bytes());
        bytes[16..20].copy_from_slice(b"bgen");
        bytes[20..24].copy_from_slice(&(1_u32 | (2_u32 << 2)).to_le_bytes());
        append_zlib_variant(&mut bytes, 0, &ZLIB_FIRST);
        append_zlib_variant(&mut bytes, 1, &ZLIB_SECOND);
        append_zlib_variant(&mut bytes, 2, &ZLIB_THIRD);
        fs::write(&path, bytes).expect("independent zlib BGEN fixture should be written");
        TemporaryBgenFile { path }
    }

    #[test]
    fn zlib_member_parser_validates_independent_headers_framing_and_alignment() {
        for (zlib_payload, expected_raw_deflate, expected_adler32) in [
            (ZLIB_FIRST.as_slice(), RAW_DEFLATE_FIRST.as_slice(), 0x060B_0216),
            (ZLIB_SECOND.as_slice(), RAW_DEFLATE_SECOND.as_slice(), 0x0A07_0216),
            (ZLIB_THIRD.as_slice(), RAW_DEFLATE_THIRD.as_slice(), 0x090B_0296),
        ] {
            let parsed = parse_zlib_member(zlib_payload).expect("independent zlib fixture should parse");
            assert_eq!(parsed.raw_deflate_bytes, expected_raw_deflate);
            assert_eq!(parsed.expected_adler32, expected_adler32);
            validate_zlib_header(parsed.zlib_header.to_ne_bytes()).expect("independent zlib header should validate");
        }

        assert!(
            validate_zlib_header([0x79, 0x01])
                .expect_err("non-DEFLATE method should fail")
                .to_string()
                .contains("method")
        );
        assert!(
            validate_zlib_header([0x88, 0x1C])
                .expect_err("oversized window should fail")
                .to_string()
                .contains("window")
        );
        assert!(
            validate_zlib_header([0x78, 0x00]).expect_err("invalid FCHECK should fail").to_string().contains("FCHECK")
        );
        assert!(
            validate_zlib_header([0x78, 0x20])
                .expect_err("preset dictionary should fail")
                .to_string()
                .contains("dictionaries")
        );
        let Err(truncation_error) = parse_zlib_member(&[0_u8; 6]) else {
            panic!("truncated zlib framing should fail");
        };
        assert!(truncation_error.to_string().contains("at least"));
        assert_eq!(align_member_length(1).expect("one byte should align"), 4);
        assert_eq!(align_member_length(4).expect("four bytes should remain aligned"), 4);
        assert_eq!(align_member_length(5).expect("five bytes should align"), 8);
        assert!(align_member_length(usize::MAX).is_err());
    }

    #[test]
    fn compressed_packed8_batch_packs_independent_full_tail_and_pooled_members() {
        let fixture = write_independent_zlib_bgen();
        let reader = BgenReaderCore::open(fixture.path()).expect("independent zlib BGEN should open");
        assert_eq!(
            reader.packed8_compatibility_with_cache().expect("independent zlib BGEN compatibility should validate"),
            Packed8Compatibility::Compatible
        );
        let chunk_specs = [
            ChunkSpec { variant_start_index: 0, variant_stop_index: 2 },
            ChunkSpec { variant_start_index: 2, variant_stop_index: 3 },
        ];
        let layout = reader
            .plan_compressed_packed8_batch_layout(&chunk_specs)
            .expect("compressed packed8 layout should plan")
            .expect("zlib input should produce a compressed packed8 layout");
        assert_eq!(layout.slab_byte_count, 40);
        let session = reader.read_session(&[0, 1, 2]).expect("identity read session should build");

        let full_batch =
            session.pack_compressed_packed8_batch(&layout, 0, 2).expect("full compressed packed8 batch should pack");
        assert_eq!(full_batch.member_metadata(), &[0, 17, 0x060B_0216, 20, 17, 0x0A07_0216]);
        assert_eq!(&full_batch.raw_deflate_slab()[0..17], &RAW_DEFLATE_FIRST);
        assert_eq!(&full_batch.raw_deflate_slab()[17..20], &[0_u8; 3]);
        assert_eq!(&full_batch.raw_deflate_slab()[20..37], &RAW_DEFLATE_SECOND);
        assert_eq!(full_batch.raw_deflate_slab().len(), 40);
        let pooled_storage_pointer = full_batch.raw_deflate_slab().as_ptr();
        drop(full_batch);

        let tail_batch =
            session.pack_compressed_packed8_batch(&layout, 2, 3).expect("tail compressed packed8 batch should pack");
        assert_eq!(tail_batch.raw_deflate_slab().as_ptr(), pooled_storage_pointer);
        assert_eq!(tail_batch.member_metadata(), &[0, 18, 0x090B_0296]);
        assert_eq!(&tail_batch.raw_deflate_slab()[0..18], &RAW_DEFLATE_THIRD);
        assert_eq!(tail_batch.raw_deflate_slab().len(), 40);
        drop(tail_batch);

        let pooled_full_batch = session
            .pack_compressed_packed8_batch(&layout, 0, 2)
            .expect("pooled full compressed packed8 batch should pack");
        assert_eq!(pooled_full_batch.raw_deflate_slab().as_ptr(), pooled_storage_pointer);
        assert_eq!(pooled_full_batch.member_metadata(), &[0, 17, 0x060B_0216, 20, 17, 0x0A07_0216]);
        assert_eq!(&pooled_full_batch.raw_deflate_slab()[0..17], &RAW_DEFLATE_FIRST);
        assert_eq!(&pooled_full_batch.raw_deflate_slab()[20..37], &RAW_DEFLATE_SECOND);
        drop(pooled_full_batch);

        session.finish().expect("compressed packed8 session source should remain stable");
    }
}
