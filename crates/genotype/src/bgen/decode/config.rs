use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::super::BgenError;
use super::super::sample_selection::SampleSelection;

const DEFAULT_DECODE_TILE_VARIANT_COUNT: usize = 64;
static DECODE_TILE_VARIANT_COUNT: AtomicUsize = AtomicUsize::new(DEFAULT_DECODE_TILE_VARIANT_COUNT);
static ROW_MAJOR_DIRECT_WRITE_ENABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn decode_tile_variant_count() -> usize {
    DECODE_TILE_VARIANT_COUNT.load(Ordering::Relaxed)
}

#[allow(clippy::missing_errors_doc)]
pub fn set_decode_tile_variant_count(tile_variant_count: usize) -> Result<(), BgenError> {
    if tile_variant_count == 0 {
        return Err(BgenError::Range("BGEN decode tile variant count must be positive.".to_string()));
    }
    DECODE_TILE_VARIANT_COUNT.store(tile_variant_count, Ordering::Relaxed);
    Ok(())
}

pub fn set_row_major_direct_write_enabled(enabled: bool) {
    ROW_MAJOR_DIRECT_WRITE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub(super) fn row_major_direct_write_enabled(profiling_enabled: bool, sample_selection: &SampleSelection) -> bool {
    !profiling_enabled
        && ROW_MAJOR_DIRECT_WRITE_ENABLED.load(Ordering::Relaxed)
        && (sample_selection.is_identity || sample_selection.contiguous_file_index_start.is_some())
}
