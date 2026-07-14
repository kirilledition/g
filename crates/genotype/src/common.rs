//! Genotype reader and preprocessing contracts.

#![allow(clippy::missing_errors_doc)]

use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use g_genotype_contracts::ChunkOutputStatistics;

// One decoded batch, one active compute batch, and one pending submission can
// retain host input concurrently. Excess outstanding buffers drop instead of
// blocking the bounded scheduler.
const PACKED8_BUFFER_POOL_CAPACITY: usize = 3;

#[derive(Default)]
pub(crate) struct Packed8BufferPool {
    available_buffers: Mutex<Vec<Vec<u8>>>,
}

impl Packed8BufferPool {
    pub(crate) fn acquire(self: &Arc<Self>, required_capacity: usize) -> PooledPacked8Buffer {
        let mut available_buffers = self.available_buffers.lock().unwrap_or_else(|error| error.into_inner());
        let matching_buffer_index = available_buffers.iter().position(|buffer| buffer.capacity() == required_capacity);
        let reused_values = matching_buffer_index.map(|buffer_index| available_buffers.swap_remove(buffer_index));
        drop(available_buffers);
        let mut values = reused_values.unwrap_or_else(|| Vec::with_capacity(required_capacity));
        values.clear();
        PooledPacked8Buffer { values, pool: Some(Arc::clone(self)) }
    }

    fn release(&self, mut values: Vec<u8>) {
        values.clear();
        let mut available_buffers = self.available_buffers.lock().unwrap_or_else(|error| error.into_inner());
        if available_buffers.len() < PACKED8_BUFFER_POOL_CAPACITY {
            available_buffers.push(values);
        }
    }
}

impl fmt::Debug for Packed8BufferPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let available_buffers = self.available_buffers.lock().unwrap_or_else(|error| error.into_inner());
        formatter
            .debug_struct("Packed8BufferPool")
            .field("available_buffer_count", &available_buffers.len())
            .finish_non_exhaustive()
    }
}

/// Immutable packed probability bytes that return session-managed allocations to their pool on drop.
pub struct PooledPacked8Buffer {
    pub(crate) values: Vec<u8>,
    pool: Option<Arc<Packed8BufferPool>>,
}

impl fmt::Debug for PooledPacked8Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PooledPacked8Buffer")
            .field("value_count", &self.values.len())
            .field("capacity", &self.values.capacity())
            .finish()
    }
}

impl PartialEq for PooledPacked8Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl Eq for PooledPacked8Buffer {}

impl Deref for PooledPacked8Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl From<Vec<u8>> for PooledPacked8Buffer {
    fn from(values: Vec<u8>) -> Self {
        Self { values, pool: None }
    }
}

impl Drop for PooledPacked8Buffer {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.as_ref() {
            pool.release(std::mem::take(&mut self.values));
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DosageSummary {
    pub(crate) dosage_sum: f32,
    pub(crate) dosage_square_sum: f32,
    pub(crate) observation_count: i32,
    pub(crate) zero_count: i32,
    pub(crate) homozygous_alternate_count: i32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ChunkSpec {
    pub variant_start_index: usize,
    pub variant_stop_index: usize,
}

#[derive(Debug, PartialEq)]
pub struct ChunkStats {
    pub output: ChunkOutputStatistics,
    pub compute: ChunkComputeStatistics,
}

#[derive(Debug, PartialEq)]
pub struct ChunkComputeStatistics {
    pub genotype_mean: Vec<f32>,
    pub imputed_dosage_square_sum: Option<Vec<f32>>,
    pub sparse_candidate_mask: Option<Vec<bool>>,
}

/// Owned variant-major genotype values transferred into association compute.
#[derive(Debug, PartialEq)]
pub enum OwnedGenotypeBuffer {
    /// Dosage values with shape `variants x samples`.
    Dosage(Vec<f32>),
    /// BGEN probability pairs with shape `variants x samples x 2`.
    Packed8(PooledPacked8Buffer),
}

/// Fully decoded host batch ready for association scheduling.
#[derive(Debug, PartialEq)]
pub struct DecodedGenotypeBatch {
    pub variant_start_index: usize,
    pub logical_variant_count: usize,
    pub compute_variant_count: usize,
    pub sample_count: usize,
    pub genotypes: OwnedGenotypeBuffer,
    pub statistics: ChunkStats,
}

/// Per-run policy for statistics retained after genotype decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkStatisticsPolicy {
    pub retain_imputed_dosage_square_sum: bool,
    pub collect_sparse_candidate_mask: bool,
}

/// Result of validating whether a BGEN can use packed8 GPU delivery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Packed8Compatibility {
    /// Probability pairs can be transferred and decoded as packed bytes.
    Compatible,
    /// The BGEN is valid for dosage delivery but not packed8 delivery.
    RequiresDosage,
}
