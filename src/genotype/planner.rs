//! Chunk planning for native genotype pipelines.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeSet;

use crate::genotype::common::{ChunkSpec, GenotypeError};

#[must_use]
pub fn resolve_total_variant_count(variant_count: usize, variant_limit: Option<usize>) -> usize {
    variant_limit.map_or(variant_count, |limit| limit.min(variant_count))
}

pub fn plan_chromosome_homogeneous_chunks(
    variant_count: usize,
    chunk_size: usize,
    variant_limit: Option<usize>,
    chromosome_boundary_indices: &[usize],
    committed_chunk_identifiers: &BTreeSet<usize>,
) -> Result<Vec<ChunkSpec>, GenotypeError> {
    if chunk_size == 0 {
        return Err(GenotypeError::InvalidInput("Chunk size must be positive.".to_string()));
    }
    let total_variant_count = resolve_total_variant_count(variant_count, variant_limit);
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
) -> Result<Vec<usize>, GenotypeError> {
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
        append_chunk_spec(chunk_specs, current_start, *boundary_index, committed_chunk_identifiers);
        current_start = *boundary_index;
    }
    append_chunk_spec(chunk_specs, current_start, variant_stop, committed_chunk_identifiers);
}

fn append_chunk_spec(
    chunk_specs: &mut Vec<ChunkSpec>,
    variant_start: usize,
    variant_stop: usize,
    committed_chunk_identifiers: &BTreeSet<usize>,
) {
    if variant_start == variant_stop || committed_chunk_identifiers.contains(&variant_start) {
        return;
    }
    chunk_specs.push(ChunkSpec { variant_start_index: variant_start, variant_stop_index: variant_stop });
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::plan_chromosome_homogeneous_chunks;
    use crate::genotype::common::ChunkSpec;

    #[test]
    fn planner_splits_chunks_at_chromosome_boundaries() {
        let chunks = plan_chromosome_homogeneous_chunks(12, 5, None, &[0, 3, 9, 12], &BTreeSet::new())
            .expect("planning should succeed");

        assert_eq!(
            chunks,
            vec![
                ChunkSpec { variant_start_index: 0, variant_stop_index: 3 },
                ChunkSpec { variant_start_index: 3, variant_stop_index: 5 },
                ChunkSpec { variant_start_index: 5, variant_stop_index: 9 },
                ChunkSpec { variant_start_index: 9, variant_stop_index: 10 },
                ChunkSpec { variant_start_index: 10, variant_stop_index: 12 },
            ],
        );
    }

    #[test]
    fn planner_applies_variant_limit_and_resume_skips() {
        let committed_chunk_identifiers = BTreeSet::from([4_usize]);
        let chunks = plan_chromosome_homogeneous_chunks(20, 4, Some(10), &[0, 6, 20], &committed_chunk_identifiers)
            .expect("planning should succeed");

        assert_eq!(
            chunks,
            vec![
                ChunkSpec { variant_start_index: 0, variant_stop_index: 4 },
                ChunkSpec { variant_start_index: 6, variant_stop_index: 8 },
                ChunkSpec { variant_start_index: 8, variant_stop_index: 10 },
            ],
        );
    }
}
