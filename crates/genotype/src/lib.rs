//! Genotype reader contracts and format-specific implementations.

mod api;
mod bgen;
mod buffer;
mod common;
pub mod debug;
mod error;
pub mod internal;
mod planner;
mod preprocess;
mod source;

pub use api::*;
