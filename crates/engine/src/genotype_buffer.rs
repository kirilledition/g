//! Genotype chunk metadata validation.

use std::sync::Arc;

use g_genotype::{GenotypeError, GenotypeResult};
use g_genotype_contracts::VariantMetadataColumns;

/// Resolve the chromosome represented by one planner-produced chunk.
///
/// # Errors
///
/// Returns an error for empty or inconsistent metadata. The chunk planner owns
/// the chromosome-homogeneity invariant.
pub(crate) fn homogeneous_chunk_chromosome(
    metadata: &VariantMetadataColumns,
    variant_count: usize,
) -> GenotypeResult<Arc<str>> {
    if variant_count == 0 {
        return Err(GenotypeError::InvalidInput("Association delivery received an empty BGEN chunk.".to_string()));
    }
    if metadata.len() != variant_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Chromosome metadata contains {} values for a {variant_count}-variant chunk.",
            metadata.len()
        )));
    }
    let chromosome = metadata
        .shared_chromosome(0)
        .ok_or_else(|| GenotypeError::InvalidInput("Association delivery chunk has no chromosome.".to_string()))?;
    debug_assert!(metadata.chromosomes().all(|value| value == chromosome.as_ref()));
    Ok(chromosome)
}
