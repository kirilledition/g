use std::ptr::NonNull;
use std::sync::OnceLock;

use flate2::Decompress;

use super::super::BgenError;
#[cfg(test)]
use super::super::profile::ThreadLocalProfileSnapshot;
use super::probability::read_exact_bytes;
use crate::buffer::OutputBufferAddress;

pub(super) const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
pub(super) const PLOIDY_MASK: u8 = 0x3F;

pub(in crate::bgen) struct VariantMajorOutputMatrix<Value> {
    pointer: NonNull<Value>,
    row_value_count: usize,
    row_context: &'static str,
}

impl<Value> VariantMajorOutputMatrix<Value> {
    /// Builds a typed view over a caller-owned variant-major output matrix.
    ///
    /// # Safety
    ///
    /// `output_pointer_address` must point to writable memory with enough initialized
    /// storage for every row requested through this helper. Concurrent workers must
    /// request disjoint variant rows for the same allocation.
    pub(in crate::bgen) unsafe fn from_pointer_address(
        output_pointer_address: OutputBufferAddress,
        row_value_count: usize,
        row_context: &'static str,
    ) -> Result<Self, BgenError> {
        if row_value_count == 0 {
            return Err(BgenError::Range(format!("{row_context} output row length must be positive.")));
        }
        let value_alignment = std::mem::align_of::<Value>();
        let output_pointer_address = output_pointer_address.get();
        if !output_pointer_address.is_multiple_of(value_alignment) {
            return Err(BgenError::Range(format!(
                "{row_context} output pointer is not aligned to {value_alignment} bytes.",
            )));
        }
        let pointer = NonNull::new(std::ptr::with_exposed_provenance_mut::<Value>(output_pointer_address))
            .ok_or_else(|| BgenError::Range(format!("{row_context} output pointer is null.")))?;
        Ok(Self { pointer, row_value_count, row_context })
    }

    pub(in crate::bgen) fn row_mut(&mut self, variant_index: usize) -> Result<&mut [Value], BgenError> {
        let row_offset = variant_index.checked_mul(self.row_value_count).ok_or_else(|| {
            BgenError::Range(format!("Integer overflow while locating {} output row.", self.row_context))
        })?;
        let row_pointer = unsafe {
            // Constructor callers guarantee that the backing allocation spans the requested rows.
            self.pointer.as_ptr().add(row_offset)
        };
        Ok(unsafe { std::slice::from_raw_parts_mut(row_pointer, self.row_value_count) })
    }
}

#[cfg(test)]
pub(super) struct RowMajorOutputMatrix<Value> {
    pointer: NonNull<Value>,
    row_value_count: usize,
    row_context: &'static str,
}

#[cfg(test)]
impl<Value> RowMajorOutputMatrix<Value> {
    /// Builds a typed view over a caller-owned row-major output matrix.
    ///
    /// # Safety
    ///
    /// `output_pointer_address` must point to writable memory with enough initialized
    /// storage for every row requested through this helper. Concurrent workers must
    /// request disjoint variant columns or row ranges for the same allocation.
    pub(super) unsafe fn from_pointer_address(
        output_pointer_address: OutputBufferAddress,
        row_value_count: usize,
        row_context: &'static str,
    ) -> Result<Self, BgenError> {
        if row_value_count == 0 {
            return Err(BgenError::Range(format!("{row_context} output row length must be positive.")));
        }
        let value_alignment = std::mem::align_of::<Value>();
        let output_pointer_address = output_pointer_address.get();
        if !output_pointer_address.is_multiple_of(value_alignment) {
            return Err(BgenError::Range(format!(
                "{row_context} output pointer is not aligned to {value_alignment} bytes.",
            )));
        }
        let pointer = NonNull::new(output_pointer_address as *mut Value)
            .ok_or_else(|| BgenError::Range(format!("{row_context} output pointer is null.")))?;
        Ok(Self { pointer, row_value_count, row_context })
    }

    pub(super) fn row_mut(&mut self, row_index: usize) -> Result<&mut [Value], BgenError> {
        let row_offset = row_index.checked_mul(self.row_value_count).ok_or_else(|| {
            BgenError::Range(format!("Integer overflow while locating {} output row.", self.row_context))
        })?;
        let row_pointer = unsafe {
            // Constructor callers guarantee that the backing allocation spans the requested rows.
            self.pointer.as_ptr().add(row_offset)
        };
        Ok(unsafe { std::slice::from_raw_parts_mut(row_pointer, self.row_value_count) })
    }

    pub(super) fn row_range_mut(
        &mut self,
        row_index: usize,
        column_start: usize,
        value_count: usize,
    ) -> Result<&mut [Value], BgenError> {
        let row_context = self.row_context;
        let column_stop = column_start.checked_add(value_count).ok_or_else(|| {
            BgenError::Range(format!("Integer overflow while locating {row_context} output row range."))
        })?;
        let row_values = self.row_mut(row_index)?;
        row_values
            .get_mut(column_start..column_stop)
            .ok_or_else(|| BgenError::Range(format!("{row_context} output row range exceeds the row length.")))
    }

    pub(super) fn column_mut(&mut self, column_index: usize) -> Result<RowMajorOutputColumnMut<'_, Value>, BgenError> {
        if column_index >= self.row_value_count {
            return Err(BgenError::Range(format!(
                "{} output column {column_index} exceeds the row length {}.",
                self.row_context, self.row_value_count,
            )));
        }
        Ok(RowMajorOutputColumnMut { matrix: self, column_index })
    }
}

#[cfg(test)]
pub(super) struct RowMajorOutputColumnMut<'a, Value> {
    matrix: &'a mut RowMajorOutputMatrix<Value>,
    column_index: usize,
}

#[cfg(test)]
impl<Value> RowMajorOutputColumnMut<'_, Value> {
    /// Writes one value in the validated column.
    ///
    /// # Safety
    ///
    /// `row_index` must be within the caller-owned matrix row count covered by the
    /// constructor safety contract. Parallel callers must own disjoint columns or
    /// row spans.
    pub(super) unsafe fn write_unchecked(&mut self, row_index: usize, value: Value) {
        let value_offset = (row_index * self.matrix.row_value_count) + self.column_index;
        let value_pointer = unsafe {
            // The matrix safety contract covers all rows written through this column view.
            self.matrix.pointer.as_ptr().add(value_offset)
        };
        unsafe {
            value_pointer.write(value);
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(in crate::bgen) struct DosageTileDecodeResult {
    pub(in crate::bgen) profile_snapshot: ThreadLocalProfileSnapshot,
    pub(in crate::bgen) selected_dosage_totals: Vec<f32>,
}

pub(in crate::bgen) struct VariantMajorTileStatsMut<'a> {
    pub(in crate::bgen) dosage_sum: &'a mut [f32],
    pub(in crate::bgen) dosage_square_sum: &'a mut [f32],
    pub(in crate::bgen) observation_count: &'a mut [i32],
    pub(in crate::bgen) sparse_candidate_counts: Option<VariantMajorSparseCandidateCountsMut<'a>>,
}

pub(in crate::bgen) struct VariantMajorSparseCandidateCountsMut<'a> {
    pub(in crate::bgen) zero_count: &'a mut [i32],
    pub(in crate::bgen) homozygous_alternate_count: &'a mut [i32],
}

#[cfg(test)]
pub(super) fn build_variant_decode_result(
    profile_snapshot: ThreadLocalProfileSnapshot,
    selected_dosage_total: f32,
) -> VariantDecodeResult {
    VariantDecodeResult {
        profile_snapshot,
        selected_dosage_total,
        selected_dosage_square_total: 0.0,
        selected_observation_count: 0,
        has_missing_values: false,
        zero_count: 0,
        homozygous_alternate_count: 0,
    }
}

pub(in crate::bgen) fn selected_sample_count_to_i32(selected_sample_count: usize) -> Result<i32, BgenError> {
    i32::try_from(selected_sample_count).map_err(|_| {
        BgenError::Range(format!(
            "Selected sample count {selected_sample_count} exceeds the supported i32 statistics range.",
        ))
    })
}

pub(in crate::bgen) fn unphased_eight_bit_dosage_lookup() -> &'static [f32] {
    static UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP: OnceLock<Vec<f32>> = OnceLock::new();
    UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP.get_or_init(|| {
        let reciprocal_scale = 1.0_f32 / 255.0_f32;
        let mut dosage_lookup = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for packed_probability_index in 0..=u16::MAX {
            let homozygous_reference_probability = f32::from(
                u8::try_from(packed_probability_index & 0x00FF).expect("low packed probability byte should fit u8"),
            ) * reciprocal_scale;
            let heterozygous_probability = f32::from(
                u8::try_from((packed_probability_index & 0xFF00) >> 8)
                    .expect("high packed probability byte should fit u8"),
            ) * reciprocal_scale;
            dosage_lookup.push(2.0_f32 - ((2.0_f32 * homozygous_reference_probability) + heterozygous_probability));
        }
        dosage_lookup
    })
}

pub(super) fn exact_eight_bit_probability_pairs(packed_probability_bytes: &[u8]) -> &[[u8; 2]] {
    let (probability_pairs, []) = packed_probability_bytes.as_chunks::<2>() else {
        unreachable!("8-bit BGEN probability byte slices are built from two bytes per sample");
    };
    probability_pairs
}

pub(in crate::bgen) fn packed_eight_bit_probability_index(
    [homozygous_reference_probability_byte, heterozygous_probability_byte]: [u8; 2],
) -> usize {
    usize::from(homozygous_reference_probability_byte) | (usize::from(heterozygous_probability_byte) << 8)
}

pub(in crate::bgen) fn read_eight_bit_probability_pair(buffer: &[u8], offset: usize) -> Result<[u8; 2], BgenError> {
    let probability_bytes = read_exact_bytes(buffer, offset, 2)?;
    let ([probability_pair], []) = probability_bytes.as_chunks::<2>() else {
        unreachable!("selected 8-bit BGEN probability reads request exactly two bytes");
    };
    Ok(*probability_pair)
}

pub(in crate::bgen) struct ThreadScratch {
    pub(super) zlib_decompressor: Decompress,
    pub(super) decompressed_probability_block: Vec<u8>,
    #[cfg(test)]
    pub(super) dosage_tile: Vec<f32>,
}

impl Default for ThreadScratch {
    fn default() -> Self {
        Self {
            zlib_decompressor: Decompress::new(true),
            decompressed_probability_block: Vec::new(),
            #[cfg(test)]
            dosage_tile: Vec::new(),
        }
    }
}

#[cfg(test)]
pub(super) fn record_variant_decode_if_enabled(
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    profiling_enabled: bool,
) {
    if profiling_enabled {
        thread_local_profile_snapshot.variant_decode_count += 1;
    }
}
