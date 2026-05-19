use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct ReaderProfileSnapshot {
    pub sample_selection_prepare_ns: u64,
    pub sample_selection_prepare_count: u64,
    pub compressed_block_fetch_ns: u64,
    pub compressed_block_fetch_count: u64,
    pub compressed_byte_count: u64,
    pub decompression_ns: u64,
    pub decompression_count: u64,
    pub uncompressed_byte_count: u64,
    pub zlib_stream_count: u64,
    pub probability_decode_ns: u64,
    pub probability_decode_count: u64,
    pub variant_decode_count: u64,
    pub output_write_ns: u64,
    pub output_write_count: u64,
    pub output_byte_count: u64,
    pub decode_tile_count: u64,
    pub selected_sample_count: u64,
    pub metadata_slice_ns: u64,
    pub metadata_slice_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ThreadLocalProfileSnapshot {
    pub(crate) compressed_block_fetch_ns: u64,
    pub(crate) compressed_block_fetch_count: u64,
    pub(crate) compressed_byte_count: u64,
    pub(crate) decompression_ns: u64,
    pub(crate) decompression_count: u64,
    pub(crate) uncompressed_byte_count: u64,
    pub(crate) zlib_stream_count: u64,
    pub(crate) probability_decode_ns: u64,
    pub(crate) probability_decode_count: u64,
    pub(crate) variant_decode_count: u64,
    pub(crate) output_write_ns: u64,
    pub(crate) output_write_count: u64,
    pub(crate) output_byte_count: u64,
    pub(crate) decode_tile_count: u64,
    pub(crate) selected_sample_count: u64,
}

#[derive(Debug, Default)]
pub(crate) struct ReaderProfiling {
    enabled: AtomicBool,
    sample_selection_prepare_ns: AtomicU64,
    sample_selection_prepare_count: AtomicU64,
    compressed_block_fetch_ns: AtomicU64,
    compressed_block_fetch_count: AtomicU64,
    compressed_byte_count: AtomicU64,
    decompression_ns: AtomicU64,
    decompression_count: AtomicU64,
    uncompressed_byte_count: AtomicU64,
    zlib_stream_count: AtomicU64,
    probability_decode_ns: AtomicU64,
    probability_decode_count: AtomicU64,
    variant_decode_count: AtomicU64,
    output_write_ns: AtomicU64,
    output_write_count: AtomicU64,
    output_byte_count: AtomicU64,
    decode_tile_count: AtomicU64,
    selected_sample_count: AtomicU64,
    metadata_slice_ns: AtomicU64,
    metadata_slice_count: AtomicU64,
}

impl ReaderProfiling {
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn reset(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        self.sample_selection_prepare_ns.store(0, Ordering::Relaxed);
        self.sample_selection_prepare_count.store(0, Ordering::Relaxed);
        self.compressed_block_fetch_ns.store(0, Ordering::Relaxed);
        self.compressed_block_fetch_count.store(0, Ordering::Relaxed);
        self.compressed_byte_count.store(0, Ordering::Relaxed);
        self.decompression_ns.store(0, Ordering::Relaxed);
        self.decompression_count.store(0, Ordering::Relaxed);
        self.uncompressed_byte_count.store(0, Ordering::Relaxed);
        self.zlib_stream_count.store(0, Ordering::Relaxed);
        self.probability_decode_ns.store(0, Ordering::Relaxed);
        self.probability_decode_count.store(0, Ordering::Relaxed);
        self.variant_decode_count.store(0, Ordering::Relaxed);
        self.output_write_ns.store(0, Ordering::Relaxed);
        self.output_write_count.store(0, Ordering::Relaxed);
        self.output_byte_count.store(0, Ordering::Relaxed);
        self.decode_tile_count.store(0, Ordering::Relaxed);
        self.selected_sample_count.store(0, Ordering::Relaxed);
        self.metadata_slice_ns.store(0, Ordering::Relaxed);
        self.metadata_slice_count.store(0, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> ReaderProfileSnapshot {
        ReaderProfileSnapshot {
            sample_selection_prepare_ns: self.sample_selection_prepare_ns.load(Ordering::Relaxed),
            sample_selection_prepare_count: self.sample_selection_prepare_count.load(Ordering::Relaxed),
            compressed_block_fetch_ns: self.compressed_block_fetch_ns.load(Ordering::Relaxed),
            compressed_block_fetch_count: self.compressed_block_fetch_count.load(Ordering::Relaxed),
            compressed_byte_count: self.compressed_byte_count.load(Ordering::Relaxed),
            decompression_ns: self.decompression_ns.load(Ordering::Relaxed),
            decompression_count: self.decompression_count.load(Ordering::Relaxed),
            uncompressed_byte_count: self.uncompressed_byte_count.load(Ordering::Relaxed),
            zlib_stream_count: self.zlib_stream_count.load(Ordering::Relaxed),
            probability_decode_ns: self.probability_decode_ns.load(Ordering::Relaxed),
            probability_decode_count: self.probability_decode_count.load(Ordering::Relaxed),
            variant_decode_count: self.variant_decode_count.load(Ordering::Relaxed),
            output_write_ns: self.output_write_ns.load(Ordering::Relaxed),
            output_write_count: self.output_write_count.load(Ordering::Relaxed),
            output_byte_count: self.output_byte_count.load(Ordering::Relaxed),
            decode_tile_count: self.decode_tile_count.load(Ordering::Relaxed),
            selected_sample_count: self.selected_sample_count.load(Ordering::Relaxed),
            metadata_slice_ns: self.metadata_slice_ns.load(Ordering::Relaxed),
            metadata_slice_count: self.metadata_slice_count.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn merge_thread_local_snapshot(&self, thread_local_snapshot: &ThreadLocalProfileSnapshot) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.compressed_block_fetch_ns.fetch_add(thread_local_snapshot.compressed_block_fetch_ns, Ordering::Relaxed);
        self.compressed_block_fetch_count
            .fetch_add(thread_local_snapshot.compressed_block_fetch_count, Ordering::Relaxed);
        self.compressed_byte_count.fetch_add(thread_local_snapshot.compressed_byte_count, Ordering::Relaxed);
        self.decompression_ns.fetch_add(thread_local_snapshot.decompression_ns, Ordering::Relaxed);
        self.decompression_count.fetch_add(thread_local_snapshot.decompression_count, Ordering::Relaxed);
        self.uncompressed_byte_count.fetch_add(thread_local_snapshot.uncompressed_byte_count, Ordering::Relaxed);
        self.zlib_stream_count.fetch_add(thread_local_snapshot.zlib_stream_count, Ordering::Relaxed);
        self.probability_decode_ns.fetch_add(thread_local_snapshot.probability_decode_ns, Ordering::Relaxed);
        self.probability_decode_count.fetch_add(thread_local_snapshot.probability_decode_count, Ordering::Relaxed);
        self.variant_decode_count.fetch_add(thread_local_snapshot.variant_decode_count, Ordering::Relaxed);
        self.output_write_ns.fetch_add(thread_local_snapshot.output_write_ns, Ordering::Relaxed);
        self.output_write_count.fetch_add(thread_local_snapshot.output_write_count, Ordering::Relaxed);
        self.output_byte_count.fetch_add(thread_local_snapshot.output_byte_count, Ordering::Relaxed);
        self.decode_tile_count.fetch_add(thread_local_snapshot.decode_tile_count, Ordering::Relaxed);
        self.selected_sample_count.fetch_add(thread_local_snapshot.selected_sample_count, Ordering::Relaxed);
    }

    pub(crate) fn record_sample_selection_prepare(&self, duration_nanoseconds: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.sample_selection_prepare_ns.fetch_add(duration_nanoseconds, Ordering::Relaxed);
        self.sample_selection_prepare_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_metadata_slice(&self, duration_nanoseconds: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.metadata_slice_ns.fetch_add(duration_nanoseconds, Ordering::Relaxed);
        self.metadata_slice_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_selected_sample_count(&self, selected_sample_count: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.selected_sample_count
            .fetch_add(u64::try_from(selected_sample_count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
}

pub(crate) fn elapsed_nanoseconds(start_time: Instant) -> u64 {
    u64::try_from(start_time.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
