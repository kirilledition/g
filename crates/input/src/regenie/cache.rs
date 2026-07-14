use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::PredictionError;
use super::alignment::{LocoSampleAlignment, build_sample_alignment, validate_loco_sample_keys};
use super::loco::{LocoFileIndex, LocoSampleIndex, index_loco_file, parse_loco_sample_identifiers};

#[derive(Debug, Default)]
pub(super) struct LocoFileIndexCache {
    indexes_by_path: HashMap<PathBuf, Arc<LocoFileIndex>>,
    headers_by_digest: HashMap<[u8; 32], Arc<LocoSampleIndex>>,
}

pub(super) struct LocoFileIndexReference {
    pub(super) file_index: Arc<LocoFileIndex>,
    pub(super) sample_index: Arc<LocoSampleIndex>,
}

#[derive(Debug, Default)]
pub(super) struct LocoAlignmentCache {
    alignments_by_header: HashMap<[u8; 32], Arc<LocoSampleAlignment>>,
}

impl LocoFileIndexCache {
    pub(super) fn index(&mut self, loco_file_path: &Path) -> Result<LocoFileIndexReference, PredictionError> {
        let cache_key = loco_file_path.canonicalize().unwrap_or_else(|_| loco_file_path.to_path_buf());
        let file_index = if let Some(file_index) = self.indexes_by_path.get(&cache_key) {
            Arc::clone(file_index)
        } else {
            let indexed_file = index_loco_file(&cache_key)?;
            match self.headers_by_digest.entry(indexed_file.file_index.header_digest) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    debug_assert_eq!(entry.get().identifiers().len(), indexed_file.file_index.sample_count);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let sample_index =
                        parse_loco_sample_identifiers(indexed_file.header_line, indexed_file.file_index.sample_count);
                    validate_loco_sample_keys(&sample_index)?;
                    entry.insert(Arc::new(sample_index));
                }
            }
            let file_index = Arc::new(indexed_file.file_index);
            self.indexes_by_path.insert(cache_key, Arc::clone(&file_index));
            file_index
        };
        let sample_index = self
            .headers_by_digest
            .get(&file_index.header_digest)
            .expect("each cached LOCO file index registers its parsed header");
        Ok(LocoFileIndexReference { file_index, sample_index: Arc::clone(sample_index) })
    }
}

impl LocoAlignmentCache {
    pub(super) fn alignment(
        &mut self,
        loco_file_index: &LocoFileIndex,
        loco_sample_index: &LocoSampleIndex,
        target_family_identifiers: &[String],
        target_individual_identifiers: &[String],
        target_sample_indices: &[usize],
    ) -> Result<Arc<LocoSampleAlignment>, PredictionError> {
        let cache_key = loco_file_index.header_digest;
        if let Some(alignment) = self.alignments_by_header.get(&cache_key) {
            return Ok(Arc::clone(alignment));
        }
        let alignment = Arc::new(build_sample_alignment(
            loco_sample_index,
            target_family_identifiers,
            target_individual_identifiers,
            target_sample_indices,
        )?);
        self.alignments_by_header.insert(cache_key, Arc::clone(&alignment));
        Ok(alignment)
    }
}
