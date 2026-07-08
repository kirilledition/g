//! Internal cross-crate genotype helpers.

pub use crate::planner::{plan_chromosome_homogeneous_chunks, resolve_total_variant_count};
pub use crate::preprocess::{
    build_chunk_stats_from_summaries, build_empty_chunk_stats, increment_dosage_summary_counts,
    preprocess_row_major_dosage_matrix, summarize_variant_major_dosage_matrix,
};
