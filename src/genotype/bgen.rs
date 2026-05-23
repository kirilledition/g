mod decode;
mod error;
mod format;
mod index;
mod metadata;
mod profile;
mod reader;
mod sample_selection;
mod simd;
mod trusted;

pub use decode::set_decode_tile_variant_count as set_bgen_decode_tile_variant_count;
pub use error::BgenError;
pub use format::CompressionType;
pub use profile::ReaderProfileSnapshot;
pub use reader::BgenReaderCore;

#[doc(hidden)]
pub fn benchmark_decode_trusted_identity_mode(
    mode_name: &str,
    packed_probability_bytes: &[u8],
    output_values: &mut [f32],
) -> u64 {
    simd::benchmark_decode_trusted_unphased_eight_bit_identity_mode(
        mode_name,
        packed_probability_bytes,
        decode::unphased_eight_bit_dosage_lookup(),
        output_values,
    )
}
