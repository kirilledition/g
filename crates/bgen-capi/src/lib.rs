#![allow(clippy::missing_errors_doc)]

use std::cell::UnsafeCell;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use g_genotype::bgen::BgenReaderCore;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GBgenStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidArgument = 2,
    ReaderError = 3,
    Panic = 4,
}

#[repr(C)]
pub struct GBgenReader {
    reader: UnsafeCell<BgenReaderCore>,
    last_error: Mutex<String>,
}

unsafe impl Send for GBgenReader {}
unsafe impl Sync for GBgenReader {}

impl GBgenReader {
    fn new(reader: BgenReaderCore) -> Self {
        Self { reader: UnsafeCell::new(reader), last_error: Mutex::new(String::new()) }
    }

    fn reader(&self) -> &BgenReaderCore {
        unsafe {
            // The native reader uses interior synchronization for prepared sample
            // selection and profiling. Shared FFI methods only borrow it immutably.
            &*self.reader.get()
        }
    }

    fn set_last_error(&self, message: String) -> GBgenStatus {
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = message;
        }
        GBgenStatus::ReaderError
    }

    fn clear_last_error(&self) {
        if let Ok(mut last_error) = self.last_error.lock() {
            last_error.clear();
        }
    }

    fn copy_last_error(&self, message_buffer: *mut c_char, message_buffer_length: usize) -> usize {
        let message = self
            .last_error
            .lock()
            .map_or_else(|_| "BGEN reader error mutex was poisoned.".to_string(), |value| value.clone());
        copy_message_to_buffer(&message, message_buffer, message_buffer_length)
    }
}

#[unsafe(no_mangle)]
/// Open a BGEN reader.
///
/// # Safety
///
/// `bgen_path` must point to a valid NUL-terminated string, and `reader_out`
/// must point to writable storage for one reader handle.
pub unsafe extern "C" fn g_bgen_reader_open(
    bgen_path: *const c_char,
    trusted_no_missing_diploid: bool,
    reader_out: *mut *mut GBgenReader,
) -> GBgenStatus {
    ffi_status_without_reader(|| {
        if bgen_path.is_null() || reader_out.is_null() {
            return Err(GBgenStatus::NullPointer);
        }
        let path_string = unsafe { CStr::from_ptr(bgen_path) }.to_str().map_err(|_| GBgenStatus::InvalidArgument)?;
        match BgenReaderCore::open(Path::new(path_string), trusted_no_missing_diploid) {
            Ok(reader) => {
                unsafe {
                    *reader_out = Box::into_raw(Box::new(GBgenReader::new(reader)));
                }
                Ok(())
            }
            Err(_) => Err(GBgenStatus::ReaderError),
        }
    })
}

#[unsafe(no_mangle)]
/// Close a BGEN reader handle.
///
/// # Safety
///
/// `reader` must be null or a handle returned by `g_bgen_reader_open` that has
/// not already been closed.
pub unsafe extern "C" fn g_bgen_reader_close(reader: *mut GBgenReader) {
    if reader.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(reader));
    }
}

#[unsafe(no_mangle)]
/// Copy the last reader error into a caller-owned buffer.
///
/// # Safety
///
/// `reader` must be null or a valid reader handle. `message_buffer` must be null
/// or point to `message_buffer_length` writable bytes.
pub unsafe extern "C" fn g_bgen_reader_last_error(
    reader: *const GBgenReader,
    message_buffer: *mut c_char,
    message_buffer_length: usize,
) -> usize {
    if reader.is_null() {
        return copy_message_to_buffer("BGEN reader handle is null.", message_buffer, message_buffer_length);
    }
    unsafe { &*reader }.copy_last_error(message_buffer, message_buffer_length)
}

#[unsafe(no_mangle)]
/// Return sample count.
///
/// # Safety
///
/// `reader` must be a valid reader handle and `sample_count_out` must point to
/// writable storage for one `usize`.
pub unsafe extern "C" fn g_bgen_reader_sample_count(
    reader: *const GBgenReader,
    sample_count_out: *mut usize,
) -> GBgenStatus {
    ffi_status(reader, || {
        if sample_count_out.is_null() {
            return Err(GBgenStatus::NullPointer);
        }
        let reader = unsafe { &*reader };
        unsafe {
            *sample_count_out = reader.reader().sample_count();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
/// Return variant count.
///
/// # Safety
///
/// `reader` must be a valid reader handle and `variant_count_out` must point to
/// writable storage for one `usize`.
pub unsafe extern "C" fn g_bgen_reader_variant_count(
    reader: *const GBgenReader,
    variant_count_out: *mut usize,
) -> GBgenStatus {
    ffi_status(reader, || {
        if variant_count_out.is_null() {
            return Err(GBgenStatus::NullPointer);
        }
        let reader = unsafe { &*reader };
        unsafe {
            *variant_count_out = reader.reader().variant_count();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
/// Return byte offset for one indexed variant.
///
/// # Safety
///
/// `reader` must be a valid reader handle and `variant_offset_out` must point to
/// writable storage for one `u64`.
pub unsafe extern "C" fn g_bgen_reader_variant_offset(
    reader: *const GBgenReader,
    variant_index: usize,
    variant_offset_out: *mut u64,
) -> GBgenStatus {
    ffi_status(reader, || {
        if variant_offset_out.is_null() {
            return Err(GBgenStatus::NullPointer);
        }
        let reader = unsafe { &*reader };
        match reader.reader().variant_offset(variant_index) {
            Ok(variant_offset) => {
                let variant_offset = u64::try_from(variant_offset).map_err(|_| GBgenStatus::InvalidArgument)?;
                unsafe {
                    *variant_offset_out = variant_offset;
                }
                Ok(())
            }
            Err(error) => Err(reader.set_last_error(error.to_string())),
        }
    })
}

#[unsafe(no_mangle)]
/// Prepare sample selection for later reads.
///
/// # Safety
///
/// `reader` must be a valid reader handle. `sample_indices` must be null only
/// when `sample_count` is zero; otherwise it must point to `sample_count` values.
pub unsafe extern "C" fn g_bgen_reader_prepare_samples(
    reader: *mut GBgenReader,
    sample_indices: *const i64,
    sample_count: usize,
) -> GBgenStatus {
    ffi_status(reader, || {
        let sample_indices = raw_slice(sample_indices, sample_count)?;
        let reader = unsafe { &*reader };
        match reader.reader().prepare_sample_selection(sample_indices) {
            Ok(()) => Ok(()),
            Err(error) => Err(reader.set_last_error(error.to_string())),
        }
    })
}

#[unsafe(no_mangle)]
/// Read variant-major dosages by indexed variant positions.
///
/// # Safety
///
/// `reader` must be a valid reader handle. `variant_indices` must be null only
/// when `variant_count` is zero. `output_values` must point to
/// `output_value_count` writable `f32` values when output count is nonzero.
pub unsafe extern "C" fn g_bgen_reader_read_variant_major_dosage_by_indices(
    reader: *mut GBgenReader,
    variant_indices: *const usize,
    variant_count: usize,
    output_values: *mut f32,
    output_value_count: usize,
) -> GBgenStatus {
    ffi_status(reader, || {
        let variant_indices = raw_slice(variant_indices, variant_count)?;
        validate_output_buffer(output_values, output_value_count)?;
        let reader = unsafe { &*reader };
        match reader.reader().read_variant_major_dosage_f32_into_address_by_indices_prepared(
            variant_indices,
            output_values as usize,
            output_value_count,
        ) {
            Ok(()) => Ok(()),
            Err(error) => Err(reader.set_last_error(error.to_string())),
        }
    })
}

#[unsafe(no_mangle)]
/// Read variant-major dosages by BGEN byte offsets.
///
/// # Safety
///
/// `reader` must be a valid reader handle. `variant_offsets` must be null only
/// when `variant_count` is zero. `output_values` must point to
/// `output_value_count` writable `f32` values when output count is nonzero.
pub unsafe extern "C" fn g_bgen_reader_read_variant_major_dosage_by_offsets(
    reader: *mut GBgenReader,
    variant_offsets: *const u64,
    variant_count: usize,
    output_values: *mut f32,
    output_value_count: usize,
) -> GBgenStatus {
    ffi_status(reader, || {
        let variant_offsets = raw_slice(variant_offsets, variant_count)?;
        validate_output_buffer(output_values, output_value_count)?;
        let reader = unsafe { &*reader };
        match reader.reader().read_variant_major_dosage_f32_into_address_by_offsets_prepared(
            variant_offsets,
            output_values as usize,
            output_value_count,
        ) {
            Ok(()) => Ok(()),
            Err(error) => Err(reader.set_last_error(error.to_string())),
        }
    })
}

fn ffi_status(reader: *const GBgenReader, callback: impl FnOnce() -> Result<(), GBgenStatus>) -> GBgenStatus {
    if reader.is_null() {
        return GBgenStatus::NullPointer;
    }
    let reader_reference = unsafe { &*reader };
    reader_reference.clear_last_error();
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(Ok(())) => GBgenStatus::Ok,
        Ok(Err(status)) => status,
        Err(_) => reader_reference.set_last_error("Rust BGEN reader panicked across the C ABI boundary.".to_string()),
    }
}

fn ffi_status_without_reader(callback: impl FnOnce() -> Result<(), GBgenStatus>) -> GBgenStatus {
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(Ok(())) => GBgenStatus::Ok,
        Ok(Err(status)) => status,
        Err(_) => GBgenStatus::Panic,
    }
}

fn raw_slice<'a, T>(pointer: *const T, value_count: usize) -> Result<&'a [T], GBgenStatus> {
    if value_count == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err(GBgenStatus::NullPointer);
    }
    Ok(unsafe { slice::from_raw_parts(pointer, value_count) })
}

fn validate_output_buffer(output_values: *mut f32, output_value_count: usize) -> Result<(), GBgenStatus> {
    if output_value_count > 0 && output_values.is_null() {
        return Err(GBgenStatus::NullPointer);
    }
    Ok(())
}

fn copy_message_to_buffer(message: &str, message_buffer: *mut c_char, message_buffer_length: usize) -> usize {
    let message_bytes = message.as_bytes();
    let required_length = message_bytes.len() + 1;
    if message_buffer.is_null() || message_buffer_length == 0 {
        return required_length;
    }
    let copy_length = message_bytes.len().min(message_buffer_length - 1);
    unsafe {
        ptr::copy_nonoverlapping(message_bytes.as_ptr(), message_buffer.cast::<u8>(), copy_length);
        *message_buffer.add(copy_length) = 0;
    }
    required_length
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::ffi::CString {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/regenie-patched/example/example.bgen");
        std::ffi::CString::new(path.to_string_lossy().as_bytes()).expect("fixture path should not contain nul bytes")
    }

    #[test]
    fn c_api_opens_fixture_and_reads_variant_major_dosage() {
        let mut reader = ptr::null_mut();
        let status = unsafe { g_bgen_reader_open(fixture_path().as_ptr(), false, &raw mut reader) };
        assert_eq!(status, GBgenStatus::Ok);
        assert!(!reader.is_null());

        let mut sample_count = 0_usize;
        assert_eq!(unsafe { g_bgen_reader_sample_count(reader, &raw mut sample_count) }, GBgenStatus::Ok,);
        assert!(sample_count > 0);

        let sample_indices = (0..i64::try_from(sample_count).expect("sample count should fit i64")).collect::<Vec<_>>();
        assert_eq!(
            unsafe { g_bgen_reader_prepare_samples(reader, sample_indices.as_ptr(), sample_indices.len()) },
            GBgenStatus::Ok,
        );

        let variant_indices = [0_usize, 1_usize];
        let mut output = vec![f32::NAN; sample_count * variant_indices.len()];
        assert_eq!(
            unsafe {
                g_bgen_reader_read_variant_major_dosage_by_indices(
                    reader,
                    variant_indices.as_ptr(),
                    variant_indices.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            },
            GBgenStatus::Ok,
        );
        assert!(output.iter().any(|value| value.is_finite()));

        let mut first_offset = 0_u64;
        assert_eq!(unsafe { g_bgen_reader_variant_offset(reader, 0, &raw mut first_offset) }, GBgenStatus::Ok,);
        assert!(first_offset > 0);

        unsafe {
            g_bgen_reader_close(reader);
        }
    }

    #[test]
    fn c_api_rejects_null_handles() {
        let mut sample_count = 0_usize;
        assert_eq!(unsafe { g_bgen_reader_sample_count(ptr::null(), &raw mut sample_count) }, GBgenStatus::NullPointer,);
    }
}
