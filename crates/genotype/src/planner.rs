//! Chunk planning for native genotype pipelines.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeSet;

use crate::common::ChunkSpec;
use crate::error::{GenotypeError, GenotypeResult};

pub(crate) fn plan_chromosome_homogeneous_chunks(
    variant_count: usize,
    chunk_size: usize,
    chromosome_boundary_indices: &[usize],
    committed_chunk_identifiers: &BTreeSet<usize>,
) -> GenotypeResult<Vec<ChunkSpec>> {
    if chunk_size == 0 {
        return Err(GenotypeError::InvalidInput("Chunk size must be positive.".to_string()));
    }
    if variant_count == 0 {
        return Ok(Vec::new());
    }

    let normalized_boundaries = normalize_chromosome_boundaries(chromosome_boundary_indices, variant_count)?;
    let mut chunk_specs = Vec::new();
    for chromosome_bounds in normalized_boundaries.windows(2) {
        let chromosome_stop = chromosome_bounds[1];
        let mut variant_start = chromosome_bounds[0];
        while variant_start < chromosome_stop {
            let variant_stop = chromosome_stop.min(variant_start.saturating_add(chunk_size));
            if !committed_chunk_identifiers.contains(&variant_start) {
                chunk_specs.push(ChunkSpec { variant_start_index: variant_start, variant_stop_index: variant_stop });
            }
            variant_start = variant_stop;
        }
    }
    Ok(chunk_specs)
}

fn normalize_chromosome_boundaries(
    chromosome_boundary_indices: &[usize],
    total_variant_count: usize,
) -> GenotypeResult<Vec<usize>> {
    let mut boundaries = Vec::with_capacity(chromosome_boundary_indices.len().max(2) + 2);
    boundaries.push(0);
    for boundary_index in chromosome_boundary_indices {
        if *boundary_index > total_variant_count {
            continue;
        }
        boundaries.push(*boundary_index);
    }
    boundaries.push(total_variant_count);
    boundaries.sort_unstable();
    boundaries.dedup();
    if boundaries.first() != Some(&0) || boundaries.last() != Some(&total_variant_count) {
        return Err(GenotypeError::InvalidInput(
            "Chromosome boundaries could not be normalized for the requested variant range.".to_string(),
        ));
    }
    Ok(boundaries)
}
