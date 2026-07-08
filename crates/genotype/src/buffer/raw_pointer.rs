use std::ptr::NonNull;

use crate::bgen::BgenError;

pub(crate) struct RowMajorDosageBuffer {
    pointer: Option<NonNull<f32>>,
    value_count: usize,
}

impl RowMajorDosageBuffer {
    /// Builds a typed view over a caller-owned row-major dosage buffer.
    ///
    /// # Safety
    ///
    /// `output_pointer_address` must point to writable storage for `value_count`
    /// f32 values. Values must be initialized before any borrowed slice is read.
    /// The caller must guarantee exclusive access for the lifetime of slices
    /// borrowed from this helper.
    pub(crate) unsafe fn from_pointer_address(
        output_pointer_address: usize,
        value_count: usize,
        buffer_context: &'static str,
    ) -> Result<Self, BgenError> {
        if value_count == 0 {
            return Ok(Self { pointer: None, value_count });
        }

        let value_alignment = std::mem::align_of::<f32>();
        if !output_pointer_address.is_multiple_of(value_alignment) {
            return Err(BgenError::Range(format!(
                "{buffer_context} output pointer is not aligned to {value_alignment} bytes.",
            )));
        }
        let pointer = NonNull::new(output_pointer_address as *mut f32)
            .ok_or_else(|| BgenError::Range(format!("{buffer_context} output pointer is null.")))?;
        Ok(Self { pointer: Some(pointer), value_count })
    }

    pub(crate) fn pointer_address(&self) -> usize {
        self.pointer.map_or_else(|| NonNull::<f32>::dangling().as_ptr() as usize, |pointer| pointer.as_ptr() as usize)
    }

    pub(crate) fn values_mut(&mut self) -> &mut [f32] {
        let Some(pointer) = self.pointer else {
            return &mut [];
        };
        unsafe {
            // The constructor safety contract ties this non-null, aligned pointer to
            // writable storage spanning `value_count` f32 values, with exclusive access
            // while the returned mutable slice is alive.
            std::slice::from_raw_parts_mut(pointer.as_ptr(), self.value_count)
        }
    }
}
