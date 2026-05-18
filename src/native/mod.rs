//! Pure Rust REGENIE step 2 execution support.

pub mod alignment;
#[cfg(feature = "cuda-kernel")]
pub mod cuda_linear;
pub mod linear;
