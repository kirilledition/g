use std::assert_matches;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::sample::SampleKeyMode;

use super::{
    LocoPredictionCache, MultiPredictionSource, PredictionError, PredictionListEntry, PredictionSource,
    normalize_chromosome, resolve_prediction_loco_paths,
};

static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

struct FixtureDirectory {
    path: PathBuf,
}

impl FixtureDirectory {
    fn new() -> Self {
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("g-regenie-tests-{}-{fixture_id}", std::process::id()));
        fs::create_dir_all(&path).expect("regenie test fixture directory should be created");
        Self { path }
    }

    fn write_file(&self, file_name: &str, contents: &str) -> PathBuf {
        let path = self.path.join(file_name);
        fs::write(&path, contents).expect("regenie test fixture should be written");
        path
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn normalizes_chromosome_labels() {
    assert_eq!(normalize_chromosome("chr22"), "22");
    assert_eq!(normalize_chromosome("CHR01"), "1");
    assert_eq!(normalize_chromosome("chrX"), "x");
}

#[test]
fn prediction_source_aligns_loco_samples_and_normalizes_chromosomes() {
    let fixture = FixtureDirectory::new();
    let loco_path = fixture.write_file("trait.loco", "FID_IID F2_I2 F1_I1\nchr01 0.2 0.1\nX 0.4 0.3\n");
    let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

    let source = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1", "F2"]),
        &strings(&["I1", "I2"]),
        SampleKeyMode::FidIid,
    )
    .expect("prediction source should load");

    assert_eq!(source.chromosome_predictions("1").expect("chr1 predictions"), &[0.1, 0.2]);
    assert_eq!(source.chromosome_predictions("chrX").expect("chrX predictions"), &[0.3, 0.4]);
}

#[test]
fn prediction_source_resolves_relative_loco_paths_from_prediction_list_directory() {
    let fixture = FixtureDirectory::new();
    fixture.write_file("trait.loco", "FID_IID F1_I1\n22 0.7\n");
    let prediction_list_path = fixture.write_file("pred.list", "trait trait.loco\n");

    let source = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect("relative LOCO path should resolve from prediction-list directory");

    assert_eq!(source.chromosome_predictions("22").expect("chr22 predictions"), &[0.7]);
}

#[test]
fn resolves_prediction_loco_paths_in_requested_phenotype_order() {
    let fixture = FixtureDirectory::new();
    fixture.write_file("first.loco", "FID_IID F1_I1\n22 0.1\n");
    let second_loco_path = fixture.write_file("second.loco", "FID_IID F1_I1\n22 0.2\n");
    let prediction_list_path =
        fixture.write_file("pred.list", &format!("first first.loco\nsecond {}\n", second_loco_path.display()));

    let resolved_loco_paths = resolve_prediction_loco_paths(&prediction_list_path, &strings(&["second", "first"]))
        .expect("prediction LOCO paths should resolve");

    assert_eq!(resolved_loco_paths[0].phenotype_name, "second");
    assert_eq!(resolved_loco_paths[0].loco_file_path, second_loco_path);
    assert_eq!(resolved_loco_paths[1].phenotype_name, "first");
    assert_eq!(resolved_loco_paths[1].loco_file_path, fixture.path.join("first.loco"));
}

#[test]
fn resolve_prediction_loco_paths_reports_missing_requested_phenotype() {
    let fixture = FixtureDirectory::new();
    fixture.write_file("trait.loco", "FID_IID F1_I1\n22 0.1\n");
    let prediction_list_path = fixture.write_file("pred.list", "trait trait.loco\n");

    let error = resolve_prediction_loco_paths(&prediction_list_path, &strings(&["missing"]))
        .expect_err("missing phenotype should be rejected");

    assert_matches!(error, PredictionError::MissingPhenotype { .. });
}

#[test]
fn identity_loco_alignment_reuses_prediction_buffer() {
    let prediction_values: Arc<[f32]> = vec![1.0, 2.0, 3.0].into();

    let identity_aligned_values = super::align_prediction_values(&prediction_values, &[0, 1, 2]);
    let reordered_values = super::align_prediction_values(&prediction_values, &[2, 0]);

    assert!(Arc::ptr_eq(&prediction_values, &identity_aligned_values));
    assert!(!Arc::ptr_eq(&prediction_values, &reordered_values));
    assert_eq!(reordered_values.as_ref(), &[3.0, 1.0]);
}

#[test]
fn multi_prediction_source_builds_trait_major_prediction_matrix() {
    let fixture = FixtureDirectory::new();
    let first_loco_path = fixture.write_file("first.loco", "FID_IID F1_I1 F2_I2\n22 1.0 2.0\n");
    let second_loco_path = fixture.write_file("second.loco", "FID_IID F1_I1 F2_I2\n22 3.0 4.0\n");
    let prediction_list_path = fixture.write_file(
        "pred.list",
        &format!("first {}\nsecond {}\n", first_loco_path.display(), second_loco_path.display()),
    );

    let source = MultiPredictionSource::load(
        &prediction_list_path,
        &strings(&["first", "second"]),
        &strings(&["F1", "F2"]),
        &strings(&["I1", "I2"]),
        SampleKeyMode::FidIid,
    )
    .expect("multi prediction source should load");

    let (trait_count, sample_count, prediction_values) =
        source.chromosome_prediction_matrix("chr22").expect("chr22 prediction matrix should be available");
    assert_eq!(trait_count, 2);
    assert_eq!(sample_count, 2);
    assert_eq!(prediction_values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(source.cached_chromosome_prediction_matrix_count(), 1);

    let (_, _, cached_prediction_values) =
        source.chromosome_prediction_matrix("22").expect("cached chr22 prediction matrix should be available");
    assert_eq!(cached_prediction_values, prediction_values);
    assert_eq!(source.cached_chromosome_prediction_matrix_count(), 1);
}

#[test]
fn multi_prediction_source_reuses_cached_loco_file_for_repeated_paths() {
    let fixture = FixtureDirectory::new();
    let loco_path = fixture.write_file("shared.loco", "FID_IID F1_I1 F2_I2\n22 1.0 2.0\n");
    let entries = vec![
        PredictionListEntry { phenotype_name: "first".to_string(), loco_file_path: loco_path.clone() },
        PredictionListEntry { phenotype_name: "second".to_string(), loco_file_path: loco_path },
    ];
    let mut loco_prediction_cache = LocoPredictionCache::default();

    let first_source = MultiPredictionSource::load_from_entries_with_cache(
        &entries,
        &strings(&["first", "second"]),
        &strings(&["F1", "F2"]),
        &strings(&["I1", "I2"]),
        SampleKeyMode::FidIid,
        &mut loco_prediction_cache,
    )
    .expect("multi prediction source should load repeated LOCO path");

    let (_, _, prediction_values) =
        first_source.chromosome_prediction_matrix("22").expect("shared LOCO predictions should align");
    assert_eq!(prediction_values, vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(loco_prediction_cache.cached_file_count(), 1);

    let second_source = MultiPredictionSource::load_from_entries_with_cache(
        &entries,
        &strings(&["second"]),
        &strings(&["F2"]),
        &strings(&["I2"]),
        SampleKeyMode::FidIid,
        &mut loco_prediction_cache,
    )
    .expect("second grouped-style load should reuse cached LOCO path");

    let (_, _, grouped_prediction_values) =
        second_source.chromosome_prediction_matrix("chr22").expect("subset predictions should align");
    assert_eq!(grouped_prediction_values, vec![2.0]);
    assert_eq!(loco_prediction_cache.cached_file_count(), 1);
}

#[test]
fn multi_prediction_source_reports_iid_and_matrix_consistency_errors() {
    let fixture = FixtureDirectory::new();
    let loco_path = fixture.write_file("duplicate-iid.loco", "FID_IID F1_I1 F2_I1\n22 1.0 2.0\n");
    let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

    let duplicate_target_error = MultiPredictionSource::load(
        &prediction_list_path,
        &strings(&["trait"]),
        &strings(&["F1", "F2"]),
        &strings(&["I1", "I1"]),
        SampleKeyMode::Iid,
    )
    .expect_err("duplicate target IIDs should fail in IID mode");
    assert_matches!(duplicate_target_error, PredictionError::DuplicateTargetIid { .. });

    let duplicate_loco_error = MultiPredictionSource::load(
        &prediction_list_path,
        &strings(&["trait"]),
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::Iid,
    )
    .expect_err("duplicate LOCO IIDs should fail in IID mode");
    assert_matches!(duplicate_loco_error, PredictionError::DuplicateLocoIid { .. });

    let empty_target_error = MultiPredictionSource::load(
        &prediction_list_path,
        &strings(&["trait"]),
        &strings(&["F1"]),
        &strings(&[""]),
        SampleKeyMode::Iid,
    )
    .expect_err("empty target IID should fail in IID mode");
    assert_matches!(empty_target_error, PredictionError::EmptyTargetIid);

    let empty_loco_path = fixture.write_file("empty-iid.loco", "FID_IID F1_\n22 1.0\n");
    let empty_loco_list_path = fixture.write_file("empty-iid.list", &format!("trait {}\n", empty_loco_path.display()));
    let empty_loco_error = MultiPredictionSource::load(
        &empty_loco_list_path,
        &strings(&["trait"]),
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::Iid,
    )
    .expect_err("empty LOCO IID should fail in IID mode");
    assert_matches!(empty_loco_error, PredictionError::EmptyLocoIid);

    let source = MultiPredictionSource {
        phenotype_names: strings(&["first", "second"]),
        chromosome_predictions_by_trait: vec![
            HashMap::from([("22".to_string(), Arc::<[f32]>::from(vec![1.0, 2.0]))]),
            HashMap::from([("22".to_string(), Arc::<[f32]>::from(vec![3.0]))]),
        ],
        chromosome_prediction_matrix_cache: Mutex::new(HashMap::new()),
    };
    let matrix_error =
        source.chromosome_prediction_matrix("chr22").expect_err("inconsistent trait sample counts should fail");
    assert_matches!(
        matrix_error,
        PredictionError::LocoPredictionCountMismatch { expected_count: 2, observed_count: 1, .. }
    );
}

#[test]
fn prediction_source_reports_missing_phenotype_and_chromosome() {
    let fixture = FixtureDirectory::new();
    let loco_path = fixture.write_file("trait.loco", "FID_IID F1_I1\n22 1.0\n");
    let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

    let missing_phenotype_error = PredictionSource::load(
        &prediction_list_path,
        "missing",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("missing phenotype should be rejected");
    assert_matches!(missing_phenotype_error, PredictionError::MissingPhenotype { .. });

    let source = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect("prediction source should load");
    let missing_chromosome_error =
        source.chromosome_predictions("chr1").expect_err("missing chromosome should be rejected");
    assert_matches!(missing_chromosome_error, PredictionError::MissingChromosome { .. });
}

#[test]
fn prediction_source_rejects_malformed_prediction_list_and_loco_files() {
    let fixture = FixtureDirectory::new();
    let malformed_list_path = fixture.write_file("bad.list", "trait only extra\n");
    let malformed_list_error = PredictionSource::load(
        &malformed_list_path,
        "trait",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("malformed prediction list should be rejected");
    assert_matches!(
        malformed_list_error,
        PredictionError::InvalidPredictionListLine { line_number: 1, field_count: 3 }
    );

    let duplicate_chromosome_loco_path = fixture.write_file("duplicate.loco", "FID_IID F1_I1\n22 1.0\nchr22 2.0\n");
    let duplicate_list_path =
        fixture.write_file("duplicate.list", &format!("trait {}\n", duplicate_chromosome_loco_path.display()));
    let duplicate_error = PredictionSource::load(
        &duplicate_list_path,
        "trait",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("duplicate chromosome should be rejected");
    assert_matches!(duplicate_error, PredictionError::DuplicateChromosome { .. });

    let invalid_value_loco_path = fixture.write_file("invalid.loco", "FID_IID F1_I1\n22 nope\n");
    let invalid_value_list_path =
        fixture.write_file("invalid.list", &format!("trait {}\n", invalid_value_loco_path.display()));
    let invalid_value_error = PredictionSource::load(
        &invalid_value_list_path,
        "trait",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("invalid prediction value should be rejected");
    assert_matches!(invalid_value_error, PredictionError::InvalidPredictionValue { .. });
}

#[test]
fn prediction_source_validates_target_and_loco_sample_keys() {
    let fixture = FixtureDirectory::new();
    let loco_path = fixture.write_file("trait.loco", "FID_IID F1_I1 F1_I1\n22 1.0 2.0\n");
    let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

    let duplicate_target_error = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1", "F1"]),
        &strings(&["I1", "I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("duplicate target sample key should be rejected");
    assert_matches!(duplicate_target_error, PredictionError::DuplicateTargetSampleKey { .. });

    let duplicate_loco_error = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("duplicate LOCO sample key should be rejected");
    assert_matches!(duplicate_loco_error, PredictionError::DuplicateLocoSampleKey { .. });
}

#[test]
fn prediction_source_reports_missing_target_samples() {
    let fixture = FixtureDirectory::new();
    let loco_path = fixture.write_file("trait.loco", "FID_IID F1_I1\n22 1.0\n");
    let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

    let error = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F2"]),
        &strings(&["I2"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("missing target sample should be rejected");

    assert_matches!(error, PredictionError::MissingTargetSamples(_));
}
