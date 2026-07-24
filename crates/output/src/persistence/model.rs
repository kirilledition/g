use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{OutputError, OutputResult};
use crate::manifest::build_manifest_value_sha256;
use crate::persistence::identifier::AttemptIdentifier;

pub(crate) type OutputTransactionIdentifier = AttemptIdentifier;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputChunkCommit {
    pub(crate) chunk_identifier: i64,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: i64,
    pub(crate) chunk_file_name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputChunkGeometry {
    pub(crate) chunk_identifier: i64,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: i64,
}

impl OutputChunkGeometry {
    pub(crate) fn from_range(range: &Range<usize>) -> OutputResult<Self> {
        let chunk_identifier = i64::try_from(range.start).map_err(|_| {
            OutputError::InvalidInput("Output chunk start exceeds the signed manifest index range.".to_string())
        })?;
        let variant_stop_index = i64::try_from(range.end).map_err(|_| {
            OutputError::InvalidInput("Output chunk stop exceeds the signed manifest index range.".to_string())
        })?;
        let row_count = i64::try_from(range.len()).map_err(|_| {
            OutputError::InvalidInput("Output chunk length exceeds the signed manifest count range.".to_string())
        })?;
        Ok(Self { chunk_identifier, variant_start_index: chunk_identifier, variant_stop_index, row_count })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalChunkPlan {
    chunks: Arc<[OutputChunkGeometry]>,
    chunks_by_identifier: Arc<BTreeMap<i64, OutputChunkGeometry>>,
    sha256: Arc<str>,
}

impl CanonicalChunkPlan {
    pub(crate) fn try_new(planned_chunk_ranges: &[Range<usize>]) -> OutputResult<Self> {
        if planned_chunk_ranges.is_empty() {
            return Err(OutputError::InvalidInput("Canonical output chunk plan must not be empty.".to_string()));
        }
        let mut chunks = Vec::with_capacity(planned_chunk_ranges.len());
        let mut expected_start = 0_usize;
        for range in planned_chunk_ranges {
            if range.start != expected_start || range.start >= range.end {
                return Err(OutputError::InvalidInput(format!(
                    "Canonical output chunk plan must cover one contiguous range from zero; expected start {expected_start}, observed {}..{}.",
                    range.start, range.end
                )));
            }
            chunks.push(OutputChunkGeometry::from_range(range)?);
            expected_start = range.end;
        }
        let chunks_by_identifier =
            chunks.iter().cloned().map(|chunk| (chunk.chunk_identifier, chunk)).collect::<BTreeMap<_, _>>();
        if chunks_by_identifier.len() != chunks.len() {
            return Err(OutputError::InvalidInput(
                "Canonical output chunk plan contains duplicate chunk identifiers.".to_string(),
            ));
        }
        let hash_value = json!({
            "algorithm": "sha256",
            "chunks": &chunks,
        });
        let sha256 = build_manifest_value_sha256(&hash_value)?;
        Ok(Self { chunks: chunks.into(), chunks_by_identifier: Arc::new(chunks_by_identifier), sha256: sha256.into() })
    }

    #[cfg(test)]
    pub(crate) fn chunks(&self) -> &[OutputChunkGeometry] {
        &self.chunks
    }

    pub(crate) fn chunk(&self, chunk_identifier: i64) -> Option<&OutputChunkGeometry> {
        self.chunks_by_identifier.get(&chunk_identifier)
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn chunk_identifiers(&self) -> BTreeSet<i64> {
        self.chunks.iter().map(|chunk| chunk.chunk_identifier).collect()
    }

    pub(crate) fn validate_commit(&self, chunk_commit: &OutputChunkCommit) -> OutputResult<()> {
        let Some(expected) = self.chunk(chunk_commit.chunk_identifier) else {
            return Err(OutputError::InvalidInput(format!(
                "Output chunk {} is not present in the canonical chunk plan.",
                chunk_commit.chunk_identifier
            )));
        };
        if expected.variant_start_index != chunk_commit.variant_start_index
            || expected.variant_stop_index != chunk_commit.variant_stop_index
            || expected.row_count != chunk_commit.row_count
        {
            return Err(OutputError::InvalidInput(format!(
                "Output chunk {} geometry does not match the canonical chunk plan.",
                chunk_commit.chunk_identifier
            )));
        }
        Ok(())
    }

    pub(crate) fn validate_exact_coverage<'commit>(
        &self,
        commits: impl IntoIterator<Item = &'commit OutputChunkCommit>,
    ) -> OutputResult<()> {
        let mut observed_identifiers = BTreeSet::new();
        for commit in commits {
            self.validate_commit(commit)?;
            if !observed_identifiers.insert(commit.chunk_identifier) {
                return Err(OutputError::InvalidInput(format!(
                    "Output coverage contains duplicate chunk identifier {}.",
                    commit.chunk_identifier
                )));
            }
        }
        let expected_identifiers = self.chunk_identifiers();
        if observed_identifiers != expected_identifiers {
            let missing_identifiers =
                expected_identifiers.difference(&observed_identifiers).copied().collect::<Vec<_>>();
            let unexpected_identifiers =
                observed_identifiers.difference(&expected_identifiers).copied().collect::<Vec<_>>();
            return Err(OutputError::InvalidInput(format!(
                "Output chunk coverage is not exact; missing {missing_identifiers:?}, unexpected {unexpected_identifiers:?}."
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputPartBinding {
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) phenotype_name: String,
    pub(crate) execution_plan_sha256: String,
    pub(crate) chunk_plan_sha256: String,
}

#[cfg(test)]
mod tests {
    use super::{CanonicalChunkPlan, OutputChunkCommit};

    #[test]
    fn canonical_chunk_plan_requires_contiguous_ordered_coverage() {
        let plan = CanonicalChunkPlan::try_new(&[0..3, 3..5]).expect("canonical plan builds");
        assert_eq!(plan.chunks().len(), 2);
        assert_eq!(plan.chunk_identifiers().into_iter().collect::<Vec<_>>(), [0, 3]);
        assert_eq!(plan.sha256().len(), 64);
        assert_eq!(CanonicalChunkPlan::try_new(&[0..3, 3..5]).expect("same plan builds").sha256(), plan.sha256());

        for malformed_ranges in [
            Vec::new(),
            std::iter::once(1..3).collect(),
            std::iter::once(0..0).collect(),
            vec![0..3, 4..5],
            vec![0..3, 2..5],
        ] {
            assert!(CanonicalChunkPlan::try_new(&malformed_ranges).is_err());
        }
    }

    #[test]
    fn canonical_chunk_plan_validates_geometry_and_exact_coverage() {
        let plan = CanonicalChunkPlan::try_new(&[0..3, 3..5]).expect("canonical plan builds");
        let commits = [
            OutputChunkCommit {
                chunk_identifier: 0,
                variant_start_index: 0,
                variant_stop_index: 3,
                row_count: 3,
                chunk_file_name: "part-0.parquet".to_string(),
            },
            OutputChunkCommit {
                chunk_identifier: 3,
                variant_start_index: 3,
                variant_stop_index: 5,
                row_count: 2,
                chunk_file_name: "part-3.parquet".to_string(),
            },
        ];
        plan.validate_exact_coverage(&commits).expect("exact coverage is accepted");
        assert!(plan.validate_exact_coverage(&commits[..1]).is_err());
        assert!(plan.validate_exact_coverage([&commits[0], &commits[0], &commits[1]]).is_err());

        let conflicting_commit = OutputChunkCommit { variant_stop_index: 4, row_count: 4, ..commits[0].clone() };
        assert!(plan.validate_commit(&conflicting_commit).is_err());
    }
}
