use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::BgenError;

const DEFAULT_DECODE_TILE_VARIANT_COUNT: usize = 64;
static DECODE_TILE_VARIANT_COUNT: AtomicUsize = AtomicUsize::new(DEFAULT_DECODE_TILE_VARIANT_COUNT);

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
