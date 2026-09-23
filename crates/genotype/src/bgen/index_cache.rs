//! Bounded process-local reuse of an immutable index from a verified source.

use std::fs::File;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use g_genotype_contracts::BgenSourceIdentity;

use super::index::ParsedVariantIndex;
use super::packed8_cache::ValidationCacheSource;

const MAXIMUM_INDEX_STORAGE_BYTES: usize = 256 * 1024 * 1024;
// BeeGFS may expose one-second ctime resolution. A same-size rewrite with
// restored mtime can alias an earlier snapshot inside that second. Observe a
// pinned identity for longer than this window, then admit only a fresh parse.
const SOURCE_PROBATION: Duration = Duration::from_secs(2);

enum CachedIndexState {
    Observing { observed_at: Instant },
    Ready { index: ParsedVariantIndex },
}

struct CachedIndex {
    identity: BgenSourceIdentity,
    // Retaining one descriptor prevents the cached inode from being recycled.
    // No memory mapping or genotype payload is retained by the cache.
    file: File,
    state: CachedIndexState,
}

#[derive(Default)]
pub(super) struct IndexCacheLookup {
    pub(super) index: Option<ParsedVariantIndex>,
    // Issued before parsing begins, never after a parse overlapping probation.
    pub(super) admission_observed_at: Option<Instant>,
}

pub(super) struct IndexCache {
    entry: Option<CachedIndex>,
    maximum_storage_bytes: usize,
}

impl IndexCache {
    pub(super) fn new(maximum_storage_bytes: usize) -> Self {
        Self { entry: None, maximum_storage_bytes }
    }

    pub(super) fn lookup(&mut self, source: &ValidationCacheSource) -> IndexCacheLookup {
        let observed_at = Instant::now();
        if let Some(entry) = self.entry.as_ref()
            && same_source_identity(&entry.identity, &source.identity)
            && source.matches_opened_file(&entry.file).ok() == Some(true)
        {
            return match &entry.state {
                CachedIndexState::Ready { index } => {
                    IndexCacheLookup { index: Some(index.clone()), admission_observed_at: None }
                }
                CachedIndexState::Observing { observed_at: first_observed_at } => IndexCacheLookup {
                    index: None,
                    admission_observed_at: (observed_at.duration_since(*first_observed_at) >= SOURCE_PROBATION)
                        .then_some(*first_observed_at),
                },
            };
        }
        self.entry = source.file.try_clone().ok().map(|file| CachedIndex {
            identity: source.identity.clone(),
            file,
            state: CachedIndexState::Observing { observed_at },
        });
        IndexCacheLookup::default()
    }

    pub(super) fn insert(
        &mut self,
        source: &ValidationCacheSource,
        index: &ParsedVariantIndex,
        admission_observed_at: Instant,
    ) {
        if index.retained_storage_bytes > self.maximum_storage_bytes {
            return;
        }
        let Some(entry) = self.entry.as_mut() else {
            return;
        };
        if !same_source_identity(&entry.identity, &source.identity)
            || source.matches_opened_file(&entry.file).ok() != Some(true)
        {
            return;
        }
        if let CachedIndexState::Observing { observed_at } = entry.state
            && observed_at == admission_observed_at
        {
            entry.state = CachedIndexState::Ready { index: index.clone() };
        }
    }

    #[cfg(test)]
    pub(super) fn age_observation_for_test(&mut self, elapsed: Duration) {
        if let Some(entry) = self.entry.as_mut()
            && let CachedIndexState::Observing { observed_at } = &mut entry.state
        {
            *observed_at = Instant::now().checked_sub(elapsed).expect("test observation should fit monotonic time");
        }
    }
}

pub(super) fn process_index_cache() -> &'static Mutex<IndexCache> {
    static INDEX_CACHE: OnceLock<Mutex<IndexCache>> = OnceLock::new();
    INDEX_CACHE.get_or_init(|| Mutex::new(IndexCache::new(MAXIMUM_INDEX_STORAGE_BYTES)))
}

fn same_source_identity(left: &BgenSourceIdentity, right: &BgenSourceIdentity) -> bool {
    left.device_identifier == right.device_identifier
        && left.inode_identifier == right.inode_identifier
        && left.change_time_nanoseconds == right.change_time_nanoseconds
        && left.modification_time_nanoseconds == right.modification_time_nanoseconds
        && left.file_size == right.file_size
}
