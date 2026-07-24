use std::fmt;
use std::fs::File;
use std::io::{ErrorKind, Read, Result as IoResult, Seek, SeekFrom};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use g_genotype_contracts::{BgenSourceIdentity, VariantMetadataStore};

use super::BgenError;
use super::format::CompressionType;
use super::metadata::VariantRecord;

const INDEX_METADATA_BUFFER_BYTE_COUNT: usize = 64;
const SMALL_PAYLOAD_MAXIMUM_BYTE_COUNT: usize = 4 * 1024;
pub(super) const MAXIMUM_SOURCE_WINDOW_BYTE_COUNT: usize = 8 * 1024 * 1024;
pub(crate) const MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT: u64 = 256 * 1024 * 1024;

pub(super) struct BgenSource {
    file: File,
    identity: BgenSourceIdentity,
    length: u64,
    snapshot: BgenSnapshotState,
    snapshot_cache_hit: bool,
    snapshot_cache: &'static BgenSnapshotCache,
}

enum BgenSnapshotState {
    None,
    Pending(Vec<u8>),
    Parsed(Arc<BgenSnapshotPayload>),
}

pub(super) type BgenSnapshotCache = Mutex<Option<Arc<BgenSnapshotPayload>>>;

#[derive(Debug)]
pub(super) struct BgenSnapshotPayload {
    pub(super) identity: BgenSourceIdentity,
    pub(super) bytes: Vec<u8>,
    pub(super) sample_count: usize,
    pub(super) compression_type: CompressionType,
    pub(super) variant_records: Vec<VariantRecord>,
    pub(super) variant_metadata: Arc<VariantMetadataStore>,
    pub(super) chromosome_boundary_indices: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(super) struct BgenByteWindow<'bytes> {
    absolute_start: u64,
    bytes: &'bytes [u8],
}

pub(super) struct BgenSourceCursor<'source> {
    source: &'source BgenSource,
    snapshot: Option<&'source [u8]>,
    position: u64,
    buffer_start: u64,
    buffer_fill_stop: u64,
    buffer_valid_length: usize,
    buffer: Vec<u8>,
    next_buffer_byte_count: usize,
    retain_buffer_across_skips: bool,
}

pub(super) struct BgenSnapshotCursor<'snapshot> {
    snapshot: &'snapshot [u8],
    position: usize,
}

pub(super) enum BgenCursorBytes<'snapshot, 'buffer> {
    Snapshot(&'snapshot [u8]),
    Buffered(&'buffer [u8]),
}

impl fmt::Debug for BgenSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BgenSource")
            .field("identity", &self.identity)
            .field("length", &self.length)
            .field("snapshot_byte_count", &self.snapshot_bytes().map(<[u8]>::len))
            .field("snapshot_is_parsed", &matches!(&self.snapshot, BgenSnapshotState::Parsed(_)))
            .field("snapshot_cache_hit", &self.snapshot_cache_hit)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BgenByteWindow<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BgenByteWindow")
            .field("absolute_start", &self.absolute_start)
            .field("byte_count", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl BgenSource {
    pub(super) fn open(bgen_path: &Path) -> Result<Self, BgenError> {
        Self::open_with_snapshot_limit(bgen_path, MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT)
    }

    pub(super) fn open_with_snapshot_limit(
        bgen_path: &Path,
        maximum_snapshot_byte_count: u64,
    ) -> Result<Self, BgenError> {
        Self::open_with_snapshot_limit_and_cache(bgen_path, maximum_snapshot_byte_count, bgen_snapshot_cache())
    }

    fn open_with_snapshot_limit_and_cache(
        bgen_path: &Path,
        maximum_snapshot_byte_count: u64,
        snapshot_cache: &'static BgenSnapshotCache,
    ) -> Result<Self, BgenError> {
        let configured_path =
            if bgen_path.is_absolute() { bgen_path.to_path_buf() } else { std::env::current_dir()?.join(bgen_path) };
        let file = File::open(&configured_path)?;
        let identity = capture_bgen_source_identity(&configured_path, &file)?;
        let length = identity.file_size;
        let mut source = Self {
            file,
            identity,
            length,
            snapshot: BgenSnapshotState::None,
            snapshot_cache_hit: false,
            snapshot_cache,
        };
        let snapshot = source.capture_or_reuse_snapshot(maximum_snapshot_byte_count)?;
        source.snapshot_cache_hit = matches!(&snapshot, BgenSnapshotState::Parsed(_));
        source.snapshot = snapshot;
        Ok(source)
    }

    #[cfg(test)]
    pub(super) fn open_with_test_snapshot_cache(
        bgen_path: &Path,
        snapshot_cache: &'static BgenSnapshotCache,
    ) -> Result<Self, BgenError> {
        Self::open_with_snapshot_limit_and_cache(bgen_path, MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT, snapshot_cache)
    }

    pub(super) fn identity(&self) -> &BgenSourceIdentity {
        &self.identity
    }

    pub(super) fn length(&self) -> u64 {
        self.length
    }

    pub(super) fn snapshot_bytes(&self) -> Option<&[u8]> {
        match &self.snapshot {
            BgenSnapshotState::None => None,
            BgenSnapshotState::Pending(bytes) => Some(bytes),
            BgenSnapshotState::Parsed(payload) => Some(&payload.bytes),
        }
    }

    pub(super) fn snapshot_payload(&self) -> Option<&BgenSnapshotPayload> {
        match &self.snapshot {
            BgenSnapshotState::Parsed(payload) => Some(payload),
            BgenSnapshotState::None | BgenSnapshotState::Pending(_) => None,
        }
    }

    #[cfg(any(test, feature = "benchmark-internals"))]
    pub(super) fn snapshot_cache_hit(&self) -> bool {
        self.snapshot_cache_hit
    }

    #[cfg(test)]
    pub(super) fn snapshot_payload_arc(&self) -> Option<Arc<BgenSnapshotPayload>> {
        match &self.snapshot {
            BgenSnapshotState::Parsed(payload) => Some(Arc::clone(payload)),
            BgenSnapshotState::None | BgenSnapshotState::Pending(_) => None,
        }
    }

    pub(super) fn publish_snapshot_payload(
        &mut self,
        sample_count: usize,
        compression_type: CompressionType,
        variant_records: Vec<VariantRecord>,
        variant_metadata: Arc<VariantMetadataStore>,
        chromosome_boundary_indices: Vec<usize>,
    ) -> Result<(), BgenError> {
        let snapshot = std::mem::replace(&mut self.snapshot, BgenSnapshotState::None);
        let BgenSnapshotState::Pending(bytes) = snapshot else {
            self.snapshot = snapshot;
            return Err(BgenError::InvalidFormat(
                "A parsed BGEN snapshot can only be published from newly captured immutable bytes.".to_string(),
            ));
        };
        let parsed_payload = Arc::new(BgenSnapshotPayload {
            identity: self.identity.clone(),
            bytes,
            sample_count,
            compression_type,
            variant_records,
            variant_metadata,
            chromosome_boundary_indices,
        });
        let canonical_payload;
        let displaced_payload;
        {
            let mut cache = lock_snapshot_cache(self.snapshot_cache);
            if let Some(payload) = cache.as_ref().filter(|payload| payload.identity == self.identity) {
                canonical_payload = Arc::clone(payload);
                displaced_payload = None;
            } else {
                displaced_payload = cache.replace(Arc::clone(&parsed_payload));
                canonical_payload = parsed_payload;
            }
        }
        drop(displaced_payload);
        self.snapshot = BgenSnapshotState::Parsed(canonical_payload);
        Ok(())
    }

    pub(super) fn full_snapshot_window(&self) -> Option<BgenByteWindow<'_>> {
        self.snapshot_bytes().map(|bytes| BgenByteWindow { absolute_start: 0, bytes })
    }

    pub(super) fn is_unchanged(&self) -> IoResult<bool> {
        if !self.is_open_file_unchanged()? {
            return Ok(false);
        }
        let configured_metadata = match std::fs::metadata(&self.identity.configured_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        };
        bgen_source_metadata_matches(&self.identity, &configured_metadata)
    }

    pub(super) fn is_open_file_unchanged(&self) -> IoResult<bool> {
        bgen_source_metadata_matches(&self.identity, &self.file.metadata()?)
    }

    pub(super) fn read_u32_at(&self, offset: u64) -> Result<u32, BgenError> {
        let mut bytes = [0_u8; size_of::<u32>()];
        self.read_exact_at(offset, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub(super) fn read_exact_at(&self, offset: u64, output: &mut [u8]) -> Result<(), BgenError> {
        validate_source_window_byte_count(output.len())?;
        let requested_length = output.len();
        let requested_length_u64 = u64::try_from(requested_length)
            .map_err(|_| BgenError::Range("BGEN positioned-read length does not fit uint64.".to_string()))?;
        let requested_stop = offset
            .checked_add(requested_length_u64)
            .ok_or_else(|| BgenError::Range("BGEN positioned-read range overflowed uint64.".to_string()))?;
        if requested_stop > self.length {
            return Err(unexpected_end_of_file(
                offset,
                requested_length,
                self.length.saturating_sub(offset).min(requested_length_u64),
            ));
        }

        if let Some(snapshot) = self.snapshot_bytes() {
            let snapshot_start = usize::try_from(offset)
                .map_err(|_| BgenError::Range("BGEN snapshot offset does not fit usize.".to_string()))?;
            let snapshot_stop = snapshot_start
                .checked_add(requested_length)
                .ok_or_else(|| BgenError::Range("BGEN snapshot range overflowed usize.".to_string()))?;
            let snapshot_bytes = snapshot.get(snapshot_start..snapshot_stop).ok_or_else(|| {
                BgenError::InvalidFormat("BGEN snapshot does not contain the requested source range.".to_string())
            })?;
            output.copy_from_slice(snapshot_bytes);
            return Ok(());
        }

        read_file_exact_at(&self.file, offset, output)
    }

    pub(super) fn read_window<'window>(
        &'window self,
        absolute_start: u64,
        byte_count: usize,
        buffer: &'window mut Vec<u8>,
    ) -> Result<BgenByteWindow<'window>, BgenError> {
        validate_source_window_byte_count(byte_count)?;
        if let Some(snapshot) = self.snapshot_bytes() {
            let relative_start = usize::try_from(absolute_start)
                .map_err(|_| BgenError::Range("BGEN snapshot offset does not fit usize.".to_string()))?;
            let relative_stop = relative_start
                .checked_add(byte_count)
                .ok_or_else(|| BgenError::Range("BGEN snapshot window overflowed usize.".to_string()))?;
            let bytes = snapshot.get(relative_start..relative_stop).ok_or_else(|| {
                BgenError::InvalidFormat("BGEN snapshot does not contain the requested source window.".to_string())
            })?;
            return Ok(BgenByteWindow { absolute_start, bytes });
        }

        ensure_initialized_buffer_length(buffer, byte_count)?;
        self.read_exact_at(absolute_start, &mut buffer[..byte_count])?;
        Ok(BgenByteWindow { absolute_start, bytes: &buffer[..byte_count] })
    }

    pub(super) fn read_variant_window<'window>(
        &'window self,
        variant_records: &[VariantRecord],
        buffer: &'window mut Vec<u8>,
    ) -> Result<BgenByteWindow<'window>, BgenError> {
        let first_record = variant_records.first().ok_or_else(|| {
            BgenError::Range("A BGEN source window requires at least one variant record.".to_string())
        })?;
        let last_record =
            variant_records.last().expect("non-empty BGEN variant-record windows must contain a final record");
        let absolute_start = first_record.probability_payload_offset;
        let absolute_stop = variant_payload_stop(last_record)?;
        let byte_count =
            usize::try_from(absolute_stop.checked_sub(absolute_start).ok_or_else(|| {
                BgenError::InvalidFormat("BGEN variant payload offsets are not ordered.".to_string())
            })?)
            .map_err(|_| BgenError::Range("BGEN source-window length does not fit usize.".to_string()))?;
        self.read_window(absolute_start, byte_count, buffer)
    }

    fn capture_or_reuse_snapshot(&self, maximum_snapshot_byte_count: u64) -> Result<BgenSnapshotState, BgenError> {
        if maximum_snapshot_byte_count == 0 || self.length > maximum_snapshot_byte_count {
            return Ok(BgenSnapshotState::None);
        }
        if let Some(payload) = lock_snapshot_cache(self.snapshot_cache).as_ref()
            && payload.identity == self.identity
        {
            return Ok(BgenSnapshotState::Parsed(Arc::clone(payload)));
        }

        let snapshot_length = usize::try_from(self.length)
            .map_err(|_| BgenError::Range("BGEN snapshot length does not fit usize.".to_string()))?;
        let mut snapshot = Vec::new();
        snapshot.try_reserve_exact(snapshot_length).map_err(|source| {
            BgenError::Range(format!("Could not reserve {snapshot_length} bytes for a BGEN snapshot: {source}."))
        })?;
        let mut snapshot_file = self.file.try_clone()?;
        snapshot_file.seek(SeekFrom::Start(0))?;
        snapshot_file.take(self.length).read_to_end(&mut snapshot)?;
        if snapshot.len() != snapshot_length {
            return Err(BgenError::Io(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "BGEN source ended while its owned snapshot was captured: expected {snapshot_length} bytes, observed {}.",
                    snapshot.len(),
                ),
            )));
        }
        self.ensure_unchanged_after_snapshot_capture()?;
        Ok(BgenSnapshotState::Pending(snapshot))
    }

    fn ensure_unchanged_after_snapshot_capture(&self) -> Result<(), BgenError> {
        if self.is_unchanged()? {
            return Ok(());
        }
        Err(BgenError::InvalidFormat(
            "BGEN source changed while its immutable snapshot was being captured.".to_string(),
        ))
    }
}

fn bgen_snapshot_cache() -> &'static BgenSnapshotCache {
    static CACHE: OnceLock<BgenSnapshotCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
pub(super) fn new_test_snapshot_cache() -> &'static BgenSnapshotCache {
    Box::leak(Box::new(Mutex::new(None)))
}

fn lock_snapshot_cache(
    snapshot_cache: &BgenSnapshotCache,
) -> std::sync::MutexGuard<'_, Option<Arc<BgenSnapshotPayload>>> {
    snapshot_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<'bytes> BgenByteWindow<'bytes> {
    #[cfg(test)]
    pub(super) fn from_bytes(bytes: &'bytes [u8]) -> Self {
        Self { absolute_start: 0, bytes }
    }

    pub(super) fn variant_payload(self, variant_record: &VariantRecord) -> Result<&'bytes [u8], BgenError> {
        let relative_start = variant_record
            .probability_payload_offset
            .checked_sub(self.absolute_start)
            .ok_or_else(|| BgenError::InvalidFormat("BGEN payload precedes its loaded source window.".to_string()))?;
        let relative_start = usize::try_from(relative_start)
            .map_err(|_| BgenError::Range("BGEN source-window offset does not fit usize.".to_string()))?;
        let payload_length = usize::try_from(variant_record.probability_payload_length)
            .map_err(|_| BgenError::Range("BGEN probability payload length does not fit usize.".to_string()))?;
        let relative_stop = relative_start
            .checked_add(payload_length)
            .ok_or_else(|| BgenError::Range("BGEN source-window payload range overflowed usize.".to_string()))?;
        self.bytes.get(relative_start..relative_stop).ok_or_else(|| {
            BgenError::InvalidFormat("BGEN payload extends beyond its loaded source window.".to_string())
        })
    }
}

impl<'snapshot> BgenSnapshotCursor<'snapshot> {
    pub(super) fn new(snapshot: &'snapshot [u8], position: u64) -> Result<Self, BgenError> {
        let position = usize::try_from(position)
            .map_err(|_| BgenError::Range("BGEN snapshot cursor position does not fit usize.".to_string()))?;
        if position > snapshot.len() {
            return Err(unexpected_end_of_file(
                u64::try_from(position).expect("owned BGEN snapshot offsets must fit uint64"),
                0,
                0,
            ));
        }
        Ok(Self { snapshot, position })
    }

    #[inline]
    pub(super) fn position(&self) -> u64 {
        u64::try_from(self.position).expect("owned BGEN snapshot offsets must fit uint64")
    }

    #[inline]
    pub(super) fn read_u16(&mut self) -> Result<u16, BgenError> {
        Ok(u16::from_le_bytes(
            self.read_bytes(size_of::<u16>())?.try_into().expect("snapshot reads requested exactly one uint16"),
        ))
    }

    #[inline]
    pub(super) fn read_u32(&mut self) -> Result<u32, BgenError> {
        Ok(u32::from_le_bytes(
            self.read_bytes(size_of::<u32>())?.try_into().expect("snapshot reads requested exactly one uint32"),
        ))
    }

    #[inline]
    pub(super) fn read_bytes(&mut self, byte_count: usize) -> Result<&'snapshot [u8], BgenError> {
        validate_source_window_byte_count(byte_count)?;
        let read_start = self.position;
        let read_stop = read_start
            .checked_add(byte_count)
            .ok_or_else(|| BgenError::Range("BGEN snapshot cursor range overflowed usize.".to_string()))?;
        let bytes = self.snapshot.get(read_start..read_stop).ok_or_else(|| {
            unexpected_end_of_file(
                u64::try_from(read_start).expect("owned BGEN snapshot offsets must fit uint64"),
                byte_count,
                u64::try_from(self.snapshot.len().saturating_sub(read_start))
                    .expect("owned BGEN snapshot lengths must fit uint64"),
            )
        })?;
        self.position = read_stop;
        Ok(bytes)
    }

    #[inline]
    pub(super) fn skip_payload_exact(&mut self, byte_count: usize) -> Result<(), BgenError> {
        self.read_bytes(byte_count).map(|_| ())
    }
}

impl AsRef<[u8]> for BgenCursorBytes<'_, '_> {
    // The separate arms preserve the enum's independent snapshot and buffer
    // lifetimes; one or-pattern incorrectly requires them to be unified.
    #[allow(clippy::match_same_arms)]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Snapshot(bytes) => bytes,
            Self::Buffered(bytes) => bytes,
        }
    }
}

impl<'source> BgenSourceCursor<'source> {
    pub(super) fn new(source: &'source BgenSource, position: u64) -> Result<Self, BgenError> {
        Self::new_with_buffer_byte_count(source, position, source.length(), INDEX_METADATA_BUFFER_BYTE_COUNT, false)
    }

    pub(super) fn new_bounded_sequential(
        source: &'source BgenSource,
        position: u64,
        sequential_stop: u64,
    ) -> Result<Self, BgenError> {
        let sequential_byte_count = sequential_stop
            .checked_sub(position)
            .ok_or_else(|| BgenError::InvalidFormat("BGEN sequential cursor bounds are reversed.".to_string()))?
            .min(u64::try_from(MAXIMUM_SOURCE_WINDOW_BYTE_COUNT).expect("source-window ceiling must fit uint64"));
        if sequential_byte_count == 0 {
            return Err(BgenError::InvalidFormat("BGEN sequential cursor range is empty.".to_string()));
        }
        if sequential_stop > source.length() {
            return Err(unexpected_end_of_file(
                position,
                usize::try_from(sequential_byte_count).expect("bounded BGEN sequential cursor lengths must fit usize"),
                source.length().saturating_sub(position),
            ));
        }
        let next_buffer_byte_count =
            usize::try_from(sequential_byte_count).expect("bounded BGEN sequential cursor lengths must fit usize");
        Self::new_with_buffer_byte_count(source, position, sequential_stop, next_buffer_byte_count, true)
    }

    fn new_with_buffer_byte_count(
        source: &'source BgenSource,
        position: u64,
        buffer_fill_stop: u64,
        next_buffer_byte_count: usize,
        retain_buffer_across_skips: bool,
    ) -> Result<Self, BgenError> {
        if position > source.length() {
            return Err(unexpected_end_of_file(position, 0, 0));
        }
        Ok(Self {
            source,
            snapshot: source.snapshot_bytes(),
            position,
            buffer_start: position,
            buffer_fill_stop,
            buffer_valid_length: 0,
            buffer: Vec::new(),
            next_buffer_byte_count,
            retain_buffer_across_skips,
        })
    }

    pub(super) fn position(&self) -> u64 {
        self.position
    }

    pub(super) fn read_u16(&mut self) -> Result<u16, BgenError> {
        if let Some(bytes) = self.read_snapshot_bytes(size_of::<u16>())? {
            return Ok(u16::from_le_bytes(bytes.try_into().expect("snapshot reads requested exactly one uint16")));
        }
        let mut bytes = [0_u8; size_of::<u16>()];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    pub(super) fn read_u32(&mut self) -> Result<u32, BgenError> {
        if let Some(bytes) = self.read_snapshot_bytes(size_of::<u32>())? {
            return Ok(u32::from_le_bytes(bytes.try_into().expect("snapshot reads requested exactly one uint32")));
        }
        let mut bytes = [0_u8; size_of::<u32>()];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub(super) fn read_bytes<'buffer>(
        &mut self,
        byte_count: usize,
        output: &'buffer mut Vec<u8>,
    ) -> Result<BgenCursorBytes<'source, 'buffer>, BgenError> {
        if let Some(bytes) = self.read_snapshot_bytes(byte_count)? {
            return Ok(BgenCursorBytes::Snapshot(bytes));
        }
        ensure_initialized_buffer_length(output, byte_count)?;
        self.read_exact(&mut output[..byte_count])?;
        Ok(BgenCursorBytes::Buffered(&output[..byte_count]))
    }

    pub(super) fn skip_exact(&mut self, byte_count: usize) -> Result<(), BgenError> {
        let byte_count_u64 = u64::try_from(byte_count)
            .map_err(|_| BgenError::Range("BGEN cursor skip length does not fit uint64.".to_string()))?;
        let next_position = self
            .position
            .checked_add(byte_count_u64)
            .ok_or_else(|| BgenError::Range("BGEN cursor position overflowed uint64.".to_string()))?;
        if next_position > self.source.length() {
            return Err(unexpected_end_of_file(
                self.position,
                byte_count,
                self.source.length().saturating_sub(self.position),
            ));
        }
        self.position = next_position;
        Ok(())
    }

    pub(super) fn skip_payload_exact(&mut self, byte_count: usize) -> Result<(), BgenError> {
        self.skip_exact(byte_count)?;
        if self.retain_buffer_across_skips {
            return Ok(());
        }
        if byte_count <= SMALL_PAYLOAD_MAXIMUM_BYTE_COUNT {
            self.next_buffer_byte_count = MAXIMUM_SOURCE_WINDOW_BYTE_COUNT;
        } else {
            self.next_buffer_byte_count = INDEX_METADATA_BUFFER_BYTE_COUNT;
            self.buffer_valid_length = 0;
        }
        Ok(())
    }

    fn read_exact(&mut self, mut output: &mut [u8]) -> Result<(), BgenError> {
        if let Some(bytes) = self.read_snapshot_bytes(output.len())? {
            output.copy_from_slice(bytes);
            return Ok(());
        }

        if output.len() > INDEX_METADATA_BUFFER_BYTE_COUNT {
            if self.buffer_contains_position() {
                let relative_position = usize::try_from(self.position - self.buffer_start)
                    .expect("bounded BGEN cursor buffer offsets must fit usize");
                let available_length = self.buffer_valid_length - relative_position;
                let copy_length = available_length.min(output.len());
                output[..copy_length].copy_from_slice(&self.buffer[relative_position..relative_position + copy_length]);
                self.position += u64::try_from(copy_length).expect("bounded BGEN cursor copies must fit uint64");
                output = &mut output[copy_length..];
            }
            if !output.is_empty() {
                self.source.read_exact_at(self.position, output)?;
                self.position += u64::try_from(output.len()).expect("bounded direct BGEN cursor reads must fit uint64");
                self.buffer_start = self.position;
                self.buffer_valid_length = 0;
            }
            return Ok(());
        }

        while !output.is_empty() {
            if !self.buffer_contains_position() {
                self.fill_buffer()?;
            }
            let relative_position = usize::try_from(self.position - self.buffer_start)
                .expect("bounded BGEN cursor buffer offsets must fit usize");
            let available_length = self.buffer_valid_length - relative_position;
            let copy_length = available_length.min(output.len());
            output[..copy_length].copy_from_slice(&self.buffer[relative_position..relative_position + copy_length]);
            self.position += u64::try_from(copy_length).expect("bounded BGEN cursor copies must fit uint64");
            output = &mut output[copy_length..];
        }
        Ok(())
    }

    fn read_snapshot_bytes(&mut self, byte_count: usize) -> Result<Option<&'source [u8]>, BgenError> {
        let Some(snapshot) = self.snapshot else {
            return Ok(None);
        };
        validate_source_window_byte_count(byte_count)?;
        let relative_start = usize::try_from(self.position)
            .map_err(|_| BgenError::Range("BGEN snapshot cursor position does not fit usize.".to_string()))?;
        let relative_stop = relative_start
            .checked_add(byte_count)
            .ok_or_else(|| BgenError::Range("BGEN snapshot cursor range overflowed usize.".to_string()))?;
        let bytes = snapshot.get(relative_start..relative_stop).ok_or_else(|| {
            unexpected_end_of_file(self.position, byte_count, self.source.length().saturating_sub(self.position))
        })?;
        self.position = u64::try_from(relative_stop)
            .map_err(|_| BgenError::Range("BGEN snapshot cursor position does not fit uint64.".to_string()))?;
        Ok(Some(bytes))
    }

    fn buffer_contains_position(&self) -> bool {
        let buffer_stop = self.buffer_start
            + u64::try_from(self.buffer_valid_length).expect("bounded BGEN cursor buffers must fit uint64");
        self.position >= self.buffer_start && self.position < buffer_stop
    }

    fn fill_buffer(&mut self) -> Result<(), BgenError> {
        let next_buffer_start = self.position;
        let remaining_length = self.buffer_fill_stop.saturating_sub(next_buffer_start);
        if remaining_length == 0 {
            return Err(unexpected_end_of_file(self.position, 1, 0));
        }
        let buffer_length = usize::try_from(
            remaining_length
                .min(u64::try_from(self.next_buffer_byte_count).expect("index cursor buffer size must fit uint64")),
        )
        .expect("bounded BGEN cursor buffer size must fit usize");
        ensure_initialized_buffer_length(&mut self.buffer, buffer_length)?;
        self.buffer_valid_length = 0;
        self.source.read_exact_at(next_buffer_start, &mut self.buffer[..buffer_length])?;
        self.buffer_start = next_buffer_start;
        self.buffer_valid_length = buffer_length;
        Ok(())
    }
}

pub(super) fn coalesced_variant_window_stop(
    variant_records: &[VariantRecord],
    variant_start: usize,
) -> Result<usize, BgenError> {
    let first_record = variant_records.get(variant_start).ok_or_else(|| {
        BgenError::Range("BGEN coalesced-window start is outside the variant-record range.".to_string())
    })?;
    let absolute_start = first_record.probability_payload_offset;
    let first_window_length = variant_payload_stop(first_record)?
        .checked_sub(absolute_start)
        .ok_or_else(|| BgenError::InvalidFormat("BGEN variant payload offsets are not ordered.".to_string()))?;
    validate_source_window_byte_count(
        usize::try_from(first_window_length)
            .map_err(|_| BgenError::Range("BGEN source-window length does not fit usize.".to_string()))?,
    )?;
    let mut variant_stop = variant_start + 1;
    while let Some(variant_record) = variant_records.get(variant_stop) {
        let absolute_stop = variant_payload_stop(variant_record)?;
        let window_length = absolute_stop
            .checked_sub(absolute_start)
            .ok_or_else(|| BgenError::InvalidFormat("BGEN variant payload offsets are not ordered.".to_string()))?;
        if window_length
            > u64::try_from(MAXIMUM_SOURCE_WINDOW_BYTE_COUNT).expect("source-window ceiling must fit uint64")
        {
            break;
        }
        variant_stop += 1;
    }
    Ok(variant_stop)
}

fn variant_payload_stop(variant_record: &VariantRecord) -> Result<u64, BgenError> {
    variant_record
        .probability_payload_offset
        .checked_add(u64::from(variant_record.probability_payload_length))
        .ok_or_else(|| BgenError::Range("BGEN probability payload range overflowed uint64.".to_string()))
}

fn ensure_initialized_buffer_length(buffer: &mut Vec<u8>, byte_count: usize) -> Result<(), BgenError> {
    validate_source_window_byte_count(byte_count)?;
    if buffer.len() < byte_count {
        buffer.try_reserve(byte_count - buffer.len()).map_err(|source| {
            BgenError::Range(format!("Could not reserve {byte_count} bytes for a BGEN source buffer: {source}."))
        })?;
        buffer.resize(byte_count, 0_u8);
    }
    Ok(())
}

fn validate_source_window_byte_count(byte_count: usize) -> Result<(), BgenError> {
    if byte_count > MAXIMUM_SOURCE_WINDOW_BYTE_COUNT {
        return Err(BgenError::Range(format!(
            "BGEN source windows cannot exceed {MAXIMUM_SOURCE_WINDOW_BYTE_COUNT} bytes. Requested {byte_count} bytes.",
        )));
    }
    Ok(())
}

fn read_file_exact_at(file: &File, mut offset: u64, mut output: &mut [u8]) -> Result<(), BgenError> {
    let requested_offset = offset;
    let requested_length = output.len();
    let mut completed_length = 0_usize;
    while !output.is_empty() {
        match file.read_at(output, offset) {
            Ok(0) => {
                return Err(unexpected_end_of_file(
                    requested_offset,
                    requested_length,
                    u64::try_from(completed_length)
                        .expect("completed usize read lengths from the supported 64-bit domain must fit uint64"),
                ));
            }
            Ok(read_length) => {
                completed_length += read_length;
                offset = offset
                    .checked_add(u64::try_from(read_length).expect("positioned read lengths must fit uint64"))
                    .ok_or_else(|| BgenError::Range("BGEN positioned-read offset overflowed uint64.".to_string()))?;
                output = &mut output[read_length..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(BgenError::Io(error)),
        }
    }
    Ok(())
}

fn unexpected_end_of_file(offset: u64, requested_length: usize, available_length: u64) -> BgenError {
    BgenError::Io(std::io::Error::new(
        ErrorKind::UnexpectedEof,
        format!(
            "Unexpected end of file while reading BGEN bytes: positioned read at offset {offset} requested {requested_length} bytes, observed {available_length}."
        ),
    ))
}

fn capture_bgen_source_identity(configured_path: &Path, file: &File) -> IoResult<BgenSourceIdentity> {
    let metadata = file.metadata()?;
    Ok(BgenSourceIdentity {
        configured_path: configured_path.to_path_buf(),
        canonical_path: configured_path.canonicalize().ok(),
        device_identifier: metadata.dev(),
        inode_identifier: metadata.ino(),
        change_time_nanoseconds: checked_timestamp_nanoseconds(
            metadata.ctime(),
            metadata.ctime_nsec(),
            "BGEN change timestamp does not fit signed nanoseconds.",
        )?,
        modification_time_nanoseconds: checked_timestamp_nanoseconds(
            metadata.mtime(),
            metadata.mtime_nsec(),
            "BGEN modification timestamp does not fit signed nanoseconds.",
        )?,
        file_size: metadata.size(),
    })
}

fn bgen_source_metadata_matches(
    expected_identity: &BgenSourceIdentity,
    metadata: &std::fs::Metadata,
) -> IoResult<bool> {
    Ok(expected_identity.device_identifier == metadata.dev()
        && expected_identity.inode_identifier == metadata.ino()
        && expected_identity.change_time_nanoseconds
            == checked_timestamp_nanoseconds(
                metadata.ctime(),
                metadata.ctime_nsec(),
                "BGEN change timestamp does not fit signed nanoseconds.",
            )?
        && expected_identity.modification_time_nanoseconds
            == checked_timestamp_nanoseconds(
                metadata.mtime(),
                metadata.mtime_nsec(),
                "BGEN modification timestamp does not fit signed nanoseconds.",
            )?
        && expected_identity.file_size == metadata.size())
}

fn checked_timestamp_nanoseconds(seconds: i64, nanoseconds: i64, error_message: &'static str) -> IoResult<i64> {
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|whole_seconds| whole_seconds.checked_add(nanoseconds))
        .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, error_message))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temporary_source_path(label: &str) -> PathBuf {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after unix epoch").as_nanos();
        std::env::temp_dir().join(format!("g-bgen-source-{label}-{}-{timestamp}.bgen", std::process::id()))
    }

    #[test]
    fn snapshot_variant_windows_borrow_the_owned_bytes_directly() {
        let path = temporary_source_path("snapshot");
        fs::write(&path, [10_u8, 20, 30, 40]).expect("snapshot source should be written");
        let source = BgenSource::open(&path).expect("snapshot source should open");
        let snapshot = source.snapshot_bytes().expect("small source should own a snapshot");
        let variant_record = VariantRecord {
            probability_payload_offset: 1,
            probability_payload_length: 2,
            declared_uncompressed_block_length: 2,
        };
        let mut unused_buffer = Vec::new();
        let source_window = source
            .read_variant_window(std::slice::from_ref(&variant_record), &mut unused_buffer)
            .expect("snapshot source window should build");
        let payload = source_window.variant_payload(&variant_record).expect("snapshot payload should resolve");

        assert_eq!(payload, [20, 30]);
        assert_eq!(payload.as_ptr(), snapshot[1..3].as_ptr());
        assert!(unused_buffer.is_empty(), "the direct snapshot path should not allocate a positioned window");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn positioned_cursor_reads_large_fields_directly_and_preserves_following_values() {
        let path = temporary_source_path("positioned-large-field");
        let large_field = vec![b'A'; usize::from(u16::MAX)];
        let following_value = 0xA1B2_C3D4_u32;
        let mut bytes = 7_u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&large_field);
        bytes.extend_from_slice(&following_value.to_le_bytes());
        fs::write(&path, bytes).expect("positioned cursor source should be written");
        OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("positioned cursor source should reopen")
            .set_len(MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT + 1)
            .expect("positioned cursor source should become sparse");
        let source = BgenSource::open(&path).expect("positioned cursor source should open");
        assert!(source.snapshot_bytes().is_none());
        let mut cursor = BgenSourceCursor::new(&source, 0).expect("positioned cursor should build");

        assert_eq!(cursor.read_u32().expect("prefix should decode"), 7);
        let mut output = Vec::new();
        let decoded_field =
            cursor.read_bytes(large_field.len(), &mut output).expect("large field should read directly");
        assert_eq!(decoded_field.as_ref(), large_field);
        assert_eq!(cursor.read_u32().expect("following value should remain aligned"), following_value);

        let _ = fs::remove_file(path);
    }
}
