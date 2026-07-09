//! Chunk planning for native genotype pipelines.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeSet;

use crate::common::ChunkSpec;
use crate::error::{GenotypeError, GenotypeResult};

pub(crate) fn plan_chromosome_homogeneous_chunks(
    variant_count: usize,
    chunk_size: usize,
    variant_limit: Option<usize>,
    chromosome_boundary_indices: &[usize],
    committed_chunk_identifiers: &BTreeSet<usize>,
) -> GenotypeResult<Vec<ChunkSpec>> {
    if chunk_size == 0 {
        return Err(GenotypeError::InvalidInput("Chunk size must be positive.".to_string()));
    }
    let total_variant_count = variant_limit.map_or(variant_count, |limit| limit.min(variant_count));
    if total_variant_count == 0 {
        return Ok(Vec::new());
    }

    let normalized_boundaries = normalize_chromosome_boundaries(chromosome_boundary_indices, total_variant_count)?;
    let mut chunk_specs = Vec::new();
    let mut variant_start = 0;
    while variant_start < total_variant_count {
        let variant_stop = total_variant_count.min(variant_start + chunk_size);
        append_chromosome_homogeneous_subchunks(
            &mut chunk_specs,
            variant_start,
            variant_stop,
            &normalized_boundaries,
            committed_chunk_identifiers,
        );
        variant_start = variant_stop;
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

fn append_chromosome_homogeneous_subchunks(
    chunk_specs: &mut Vec<ChunkSpec>,
    variant_start: usize,
    variant_stop: usize,
    chromosome_boundary_indices: &[usize],
    committed_chunk_identifiers: &BTreeSet<usize>,
) {
    let mut current_start = variant_start;
    for boundary_index in chromosome_boundary_indices {
        if *boundary_index <= current_start {
            continue;
        }
        if *boundary_index >= variant_stop {
            break;
        }
        if !committed_chunk_identifiers.contains(&current_start) {
            chunk_specs.push(ChunkSpec { variant_start_index: current_start, variant_stop_index: *boundary_index });
        }
        current_start = *boundary_index;
    }
    if current_start != variant_stop && !committed_chunk_identifiers.contains(&current_start) {
        chunk_specs.push(ChunkSpec { variant_start_index: current_start, variant_stop_index: variant_stop });
    }
}
