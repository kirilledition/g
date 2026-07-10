//! Internal caller-owned buffer adapters.

#[cfg(test)]
use std::ptr::NonNull;

#[cfg(test)]
use crate::bgen::BgenError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputBufferAddress(usize);

impl OutputBufferAddress {
    #[must_use]
    pub fn from_mut_ptr<Value>(pointer: *mut Value) -> Self {
        Self(pointer.expose_provenance())
    }

    pub(crate) const fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputValueCount(usize);

impl OutputValueCount {
    #[must_use]
    pub const fn new(value_count: usize) -> Self {
        Self(value_count)
    }

    pub(crate) const fn get(self) -> usize {
        self.0
    }
}

#[cfg(test)]
pub(crate) struct RowMajorDosageBuffer {
    pointer: Option<NonNull<f32>>,
    value_count: usize,
}

#[cfg(test)]
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
        output_pointer_address: OutputBufferAddress,
        value_count: OutputValueCount,
        buffer_context: &'static str,
    ) -> Result<Self, BgenError> {
        let value_count = value_count.get();
        if value_count == 0 {
            return Ok(Self { pointer: None, value_count });
        }

        let value_alignment = std::mem::align_of::<f32>();
        let output_pointer_address = output_pointer_address.get();
        if !output_pointer_address.is_multiple_of(value_alignment) {
            return Err(BgenError::Range(format!(
                "{buffer_context} output pointer is not aligned to {value_alignment} bytes.",
            )));
        }
        let pointer = NonNull::new(output_pointer_address as *mut f32)
            .ok_or_else(|| BgenError::Range(format!("{buffer_context} output pointer is null.")))?;
        Ok(Self { pointer: Some(pointer), value_count })
    }

    pub(crate) fn pointer_address(&self) -> OutputBufferAddress {
        self.pointer.map_or_else(
            || OutputBufferAddress::from_mut_ptr(NonNull::<f32>::dangling().as_ptr()),
            |pointer| OutputBufferAddress::from_mut_ptr(pointer.as_ptr()),
        )
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
