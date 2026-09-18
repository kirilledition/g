use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::regenie::{PredictionError, PredictionLocoPath};
use crate::sample::{PhenotypeGroupLoadRequest, SampleIdentifierData};
use crate::test_support::TemporaryDirectory;
use crate::{InputError, load_aligned_phenotype_groups, load_sample_identifier_data_from_sample_file};

const TRAIT_NAMES: [&str; 2] = ["trait-a", "trait-b"];

struct InputFixture {
    directory: TemporaryDirectory,
    sample_path: PathBuf,
    phenotype_path: PathBuf,
    covariate_path: PathBuf,
    prediction_list_path: PathBuf,
    first_loco_path: PathBuf,
}

impl InputFixture {
    fn new(test_name: &str) -> Self {
        let directory = TemporaryDirectory::new(test_name);
        let sample_path = directory.write(
            "samples.sample",
            "ID_1 ID_2\n0 0\nfamily-1 individual-1\nfamily-2 individual-2\nfamily-3 individual-3\nfamily-4 individual-4\n",
        );
        let phenotype_path = directory.write(
            "phenotypes.tsv",
            "FID\tIID\ttrait-a\ttrait-b\nfamily-3\tindividual-3\t3\t30\nfamily-1\tindividual-1\t1\tNA\nfamily-4\tindividual-4\t-9\t40\nfamily-2\tindividual-2\t2\t20\n",
        );
        let covariate_path = directory.write(
            "covariates.tsv",
            "FID\tIID\tage\nfamily-4\tindividual-4\t40\nfamily-2\tindividual-2\t20\nfamily-1\tindividual-1\t10\nfamily-3\tindividual-3\t30\n",
        );
        let first_loco_path = directory.write(
            "trait-a.loco",
            "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n1 0.1 0.2 0.3 0.4\nchr2 1.1 1.2 1.3 1.4\n",
        );
        directory.write(
            "trait-b.loco",
            "FID_IID family-4_individual-4 family-2_individual-2 family-1_individual-1 family-3_individual-3\n1 4 2 1 3\nchr2 14 12 11 13\n",
        );
        let prediction_list_path = directory.write("pred.list", "trait-b trait-b.loco\ntrait-a trait-a.loco\n");
        Self { directory, sample_path, phenotype_path, covariate_path, prediction_list_path, first_loco_path }
    }

    fn sample_identifiers(&self) -> SampleIdentifierData {
        load_sample_identifier_data_from_sample_file(&self.sample_path, 4)
            .expect("valid Oxford sample fixture should load")
    }

    fn phenotype_names() -> Vec<String> {
        TRAIT_NAMES.iter().map(ToString::to_string).collect()
    }

    fn prediction_loco_paths(&self, phenotype_names: &[String]) -> Vec<PredictionLocoPath> {
        crate::resolve_prediction_loco_paths(&self.prediction_list_path, phenotype_names)
            .expect("valid prediction list fixture should resolve")
    }

    fn request<'input>(
        &'input self,
        sample_identifiers: &'input SampleIdentifierData,
        prediction_loco_paths: &'input [PredictionLocoPath],
        phenotype_names: &'input [String],
        sample_mode: g_plan::MultiPhenotypeSampleMode,
    ) -> PhenotypeGroupLoadRequest<'input> {
        debug_assert!(self.directory.path().is_dir());
        PhenotypeGroupLoadRequest {
            sample_identifiers,
            phenotype_path: path_text(&self.phenotype_path),
            prediction_loco_paths,
            phenotype_names,
            covariate_path: Some(path_text(&self.covariate_path)),
            covariate_names: None,
            is_binary_trait: false,
            sample_mode,
        }
    }

    fn replace_first_loco_file(&self) {
        let replacement_path = self.directory.write(
            "replacement.loco",
            "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n1 9.1 9.2 9.3 9.4\nchr2 1.1 1.2 1.3 1.4\n",
        );
        std::fs::rename(replacement_path, &self.first_loco_path)
            .expect("replacing an indexed LOCO fixture should succeed");
    }
}

fn path_text(path: &Path) -> &str {
    path.to_str().expect("temporary paths should be UTF-8")
}

fn assert_f32_values(actual_values: &[f32], expected_values: &[f32]) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        assert!((actual_value - expected_value).abs() < 1.0e-6, "expected {expected_value}, observed {actual_value}");
    }
}

#[test]
fn oxford_sample_identity_and_prediction_list_preserve_requested_order() {
    let fixture = InputFixture::new("identity");
    let sample_identifiers = fixture.sample_identifiers();
    assert_eq!(sample_identifiers.family_identifiers, ["family-1", "family-2", "family-3", "family-4"]);
    assert_eq!(
        sample_identifiers.individual_identifiers,
        ["individual-1", "individual-2", "individual-3", "individual-4"]
    );
    assert!(matches!(
        load_sample_identifier_data_from_sample_file(&fixture.sample_path, 3),
        Err(InputError::SampleAlignment(message)) if message.contains("contains 4 rows")
    ));

    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    assert_eq!(prediction_loco_paths[0].phenotype_name.as_ref(), "trait-a");
    assert_eq!(prediction_loco_paths[1].phenotype_name.as_ref(), "trait-b");
    assert_eq!(prediction_loco_paths[0].loco_file_path, fixture.directory.path().join("trait-a.loco"));
    assert!(matches!(
        crate::resolve_prediction_loco_paths(&fixture.prediction_list_path, &["missing-trait".to_string()]),
        Err(InputError::Prediction(PredictionError::MissingPhenotype { phenotype_name, .. }))
            if phenotype_name == "missing-trait"
    ));
}

#[test]
fn per_phenotype_and_complete_case_modes_build_expected_groups() {
    let fixture = InputFixture::new("grouping");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);

    let per_phenotype_groups = load_aligned_phenotype_groups(&fixture.request(
        &sample_identifiers,
        &prediction_loco_paths,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::PerPhenotype,
    ))
    .expect("per-phenotype fixture should align");
    assert_eq!(per_phenotype_groups.len(), 2);
    assert_eq!(per_phenotype_groups[0].phenotype_group.phenotype_names, ["trait-a"]);
    assert_eq!(per_phenotype_groups[0].sample_indices, [0, 1, 2]);
    assert_f32_values(&per_phenotype_groups[0].phenotype_values, &[1.0, 2.0, 3.0]);
    assert_eq!(per_phenotype_groups[1].phenotype_group.phenotype_names, ["trait-b"]);
    assert_eq!(per_phenotype_groups[1].sample_indices, [1, 2, 3]);
    assert_f32_values(&per_phenotype_groups[1].phenotype_values, &[20.0, 30.0, 40.0]);

    let complete_case_groups = load_aligned_phenotype_groups(&fixture.request(
        &sample_identifiers,
        &prediction_loco_paths,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::CompleteCase,
    ))
    .expect("complete-case fixture should align");
    assert_eq!(complete_case_groups.len(), 1);
    let complete_case_group = &complete_case_groups[0];
    assert_eq!(complete_case_group.phenotype_group.phenotype_names, TRAIT_NAMES);
    assert_eq!(complete_case_group.sample_indices, [1, 2]);
    assert_f32_values(&complete_case_group.phenotype_values, &[2.0, 3.0, 20.0, 30.0]);
    assert_eq!(complete_case_group.covariate_names, ["intercept", "age"]);
    assert_f32_values(&complete_case_group.covariate_values, &[1.0, 20.0, 1.0, 30.0]);
    assert_eq!(
        complete_case_group.phenotype_group.sample_set_fingerprint,
        "8ae279492f58b7edacc94e71cba156e754fe4efcd14e52be7ee8353cb6aa4181"
    );
    assert_eq!(
        complete_case_group.phenotype_group.covariate_design_fingerprint,
        "66181d307c281b198a49378fdcf6cfb8af54dfb98ad501c7671eadaa04cd87d0"
    );
    assert_eq!(
        complete_case_group.phenotype_group.phenotype_design_fingerprint,
        "89c4cc005378904feb8b5a6f10c24fd7b5d22f89cc576601121a9098335e57fd"
    );
    assert_eq!(
        complete_case_group.phenotype_group.prediction_alignment_fingerprint,
        "9510e3414e68f6f1f8aa3fb8fb755f96f2bcc16d7d123704876ed36c365eaa48"
    );
}

#[test]
fn duplicate_table_rows_are_rejected_for_phenotypes_and_covariates() {
    let fixture = InputFixture::new("duplicate-table-rows");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let duplicate_phenotype_path = fixture.directory.write(
        "duplicate-phenotypes.tsv",
        "FID\tIID\ttrait-a\ttrait-b\nfamily-1\tindividual-1\t1\t10\nfamily-1\tindividual-1\t2\t20\n",
    );
    let phenotype_request = PhenotypeGroupLoadRequest {
        phenotype_path: path_text(&duplicate_phenotype_path),
        ..fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::CompleteCase,
        )
    };
    assert!(matches!(
        load_aligned_phenotype_groups(&phenotype_request),
        Err(InputError::SampleAlignment(message)) if message.contains("Duplicate sample key")
    ));

    let duplicate_covariate_path = fixture
        .directory
        .write("duplicate-covariates.tsv", "FID\tIID\tage\nfamily-1\tindividual-1\t10\nfamily-1\tindividual-1\t20\n");
    let covariate_request = PhenotypeGroupLoadRequest {
        covariate_path: Some(path_text(&duplicate_covariate_path)),
        ..fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::CompleteCase,
        )
    };
    assert!(matches!(
        load_aligned_phenotype_groups(&covariate_request),
        Err(InputError::SampleAlignment(message)) if message.contains("Duplicate sample key")
    ));
}

#[test]
fn nonfinite_phenotypes_and_covariates_return_typed_errors() {
    let fixture = InputFixture::new("nonfinite-values");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let nonfinite_phenotype_path = fixture
        .directory
        .write("nonfinite-phenotypes.tsv", "FID\tIID\ttrait-a\ttrait-b\nfamily-1\tindividual-1\tInfinity\t10\n");
    let phenotype_request = PhenotypeGroupLoadRequest {
        phenotype_path: path_text(&nonfinite_phenotype_path),
        ..fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::CompleteCase,
        )
    };
    assert!(matches!(
        load_aligned_phenotype_groups(&phenotype_request),
        Err(InputError::NonFinitePhenotypeValue { phenotype_name, value })
            if phenotype_name == "trait-a" && value == "Infinity"
    ));

    let nonfinite_covariate_path = fixture.directory.write(
        "nonfinite-covariates.tsv",
        "FID\tIID\tage\nfamily-1\tindividual-1\tinf\nfamily-2\tindividual-2\t20\nfamily-3\tindividual-3\t30\nfamily-4\tindividual-4\t40\n",
    );
    let covariate_request = PhenotypeGroupLoadRequest {
        covariate_path: Some(path_text(&nonfinite_covariate_path)),
        ..fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::CompleteCase,
        )
    };
    assert!(matches!(
        load_aligned_phenotype_groups(&covariate_request),
        Err(InputError::NonFiniteCovariateValue { covariate_name, value })
            if covariate_name == "age" && value == "inf"
    ));
}

#[test]
fn prediction_catalog_shape_and_order_are_rejected_without_indexing() {
    let fixture = InputFixture::new("prediction-catalog");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let valid_catalog = fixture.prediction_loco_paths(&phenotype_names);
    let short_catalog = &valid_catalog[..1];
    let short_request = fixture.request(
        &sample_identifiers,
        short_catalog,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::CompleteCase,
    );
    assert!(matches!(
        load_aligned_phenotype_groups(&short_request),
        Err(InputError::Prediction(PredictionError::PredictionCatalogLengthMismatch {
            expected_count: 2,
            observed_count: 1,
        }))
    ));

    let reordered_catalog = vec![
        PredictionLocoPath {
            phenotype_name: Arc::from("trait-b"),
            loco_file_path: fixture.directory.path().join("does-not-exist-b.loco"),
        },
        PredictionLocoPath {
            phenotype_name: Arc::from("trait-a"),
            loco_file_path: fixture.directory.path().join("does-not-exist-a.loco"),
        },
    ];
    let reordered_request = fixture.request(
        &sample_identifiers,
        &reordered_catalog,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::CompleteCase,
    );
    assert!(matches!(
        load_aligned_phenotype_groups(&reordered_request),
        Err(InputError::Prediction(PredictionError::PredictionCatalogPhenotypeMismatch {
            phenotype_index: 0,
            expected_name,
            observed_name,
        })) if expected_name == "trait-a" && observed_name == "trait-b"
    ));
}

#[test]
fn duplicate_prediction_list_phenotypes_are_rejected() {
    let directory = TemporaryDirectory::new("duplicate-prediction-list");
    let prediction_list_path = directory.write("pred.list", "trait-a first.loco\ntrait-a second.loco\n");
    let result = crate::resolve_prediction_loco_paths(&prediction_list_path, &["trait-a".to_string()]);
    assert!(matches!(
        result,
        Err(InputError::Prediction(PredictionError::DuplicatePredictionListPhenotype {
            phenotype_name,
            first_line_number: 1,
            duplicate_line_number: 2,
        })) if phenotype_name == "trait-a"
    ));
}

#[test]
fn deferred_prediction_lifecycle_aligns_and_counts_repeated_chromosomes() {
    let fixture = InputFixture::new("deferred-lifecycle");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let mut groups = load_aligned_phenotype_groups(&fixture.request(
        &sample_identifiers,
        &prediction_loco_paths,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::PerPhenotype,
    ))
    .expect("prediction lifecycle fixture should align");
    let group = &mut groups[1];
    assert!(matches!(group.take_chromosome_prediction_matrix("1"), Err(PredictionError::MissingChromosome { .. })));
    assert!(matches!(group.plan_prediction_uses(&[Arc::from("3")]), Err(PredictionError::MissingChromosome { .. })));
    group
        .plan_prediction_uses(&[Arc::from("chr1"), Arc::from("2"), Arc::from("1")])
        .expect("available repeated chromosomes should plan");
    let first = group.take_chromosome_prediction_matrix("1").expect("first chromosome use should materialize");
    assert_eq!((first.trait_count, first.sample_count), (1, 3));
    assert_f32_values(&first.prediction_values, &[2.0, 3.0, 4.0]);
    let second = group.take_chromosome_prediction_matrix("chr2").expect("second chromosome should materialize");
    assert_f32_values(&second.prediction_values, &[12.0, 13.0, 14.0]);
    let final_first = group.take_chromosome_prediction_matrix("chr1").expect("final repeated use should transfer");
    assert_f32_values(&final_first.prediction_values, &[2.0, 3.0, 4.0]);
    assert!(matches!(group.take_chromosome_prediction_matrix("1"), Err(PredictionError::MissingChromosome { .. })));
}

#[test]
fn deferred_prediction_source_rejects_replaced_loco_file() {
    let fixture = InputFixture::new("deferred-replacement");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let mut groups = load_aligned_phenotype_groups(&fixture.request(
        &sample_identifiers,
        &prediction_loco_paths,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::CompleteCase,
    ))
    .expect("deferred replacement fixture should index");
    groups[0].plan_prediction_uses(&[Arc::from("1")]).expect("indexed chromosome should plan");
    fixture.replace_first_loco_file();
    assert!(matches!(
        groups[0].take_chromosome_prediction_matrix("1"),
        Err(PredictionError::IndexedLocoFileChanged { .. })
    ));
}

#[test]
fn predictions_preserve_identity_and_reordered_full_sample_alignments() {
    let fixture = InputFixture::new("full-prediction-alignments");
    fixture.directory.write(
        "phenotypes.tsv",
        "FID\tIID\ttrait-a\ttrait-b\nfamily-1\tindividual-1\t1\t10\nfamily-2\tindividual-2\t2\t20\nfamily-3\tindividual-3\t3\t30\nfamily-4\tindividual-4\t4\t40\n",
    );
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = InputFixture::phenotype_names();
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let mut groups = load_aligned_phenotype_groups(&fixture.request(
        &sample_identifiers,
        &prediction_loco_paths,
        &phenotype_names,
        g_plan::MultiPhenotypeSampleMode::CompleteCase,
    ))
    .expect("complete sample fixture should align");
    assert_eq!(groups.len(), 1);
    groups[0].plan_prediction_uses(&[Arc::from("1")]).expect("chromosome should plan");
    let predictions = groups[0].take_chromosome_prediction_matrix("1").expect("all samples should materialize");
    assert_eq!(predictions.trait_count, 2);
    assert_eq!(predictions.sample_count, 4);
    assert_f32_values(&predictions.prediction_values, &[0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn excluded_prediction_values_follow_each_phenotype_sample_mask() {
    for excluded_value in ["NA", "NaN", "inf", "1e100", "invalid"] {
        let fixture = InputFixture::new("excluded-predictions");
        fixture.directory.write(
            "trait-a.loco",
            &format!(
                "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n1 0.1 0.2 0.3 {excluded_value}\n"
            ),
        );
        fixture.directory.write(
            "trait-b.loco",
            &format!(
                "FID_IID family-4_individual-4 family-2_individual-2 family-1_individual-1 family-3_individual-3\n1 4 2 {excluded_value} 3\n"
            ),
        );
        let sample_identifiers = fixture.sample_identifiers();
        let phenotype_names = InputFixture::phenotype_names();
        let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
        let mut per_phenotype_groups = load_aligned_phenotype_groups(&fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::PerPhenotype,
        ))
        .expect("excluded predictions should not prevent phenotype alignment");
        assert_eq!(per_phenotype_groups.len(), 2);
        for (group_index, group) in per_phenotype_groups.iter_mut().enumerate() {
            group.plan_prediction_uses(&[Arc::from("1")]).expect("chromosome should plan");
            let predictions = group
                .take_chromosome_prediction_matrix("1")
                .expect("only the phenotype's selected predictions should be parsed");
            assert_eq!(predictions.trait_count, 1);
            assert_eq!(predictions.sample_count, 3);
            let expected_values = if group_index == 0 { [0.1, 0.2, 0.3] } else { [2.0, 3.0, 4.0] };
            assert_f32_values(&predictions.prediction_values, &expected_values);
        }

        let mut complete_case_groups = load_aligned_phenotype_groups(&fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::CompleteCase,
        ))
        .expect("excluded predictions should not prevent complete-case alignment");
        assert_eq!(complete_case_groups.len(), 1);
        complete_case_groups[0].plan_prediction_uses(&[Arc::from("1")]).expect("chromosome should plan");
        let predictions = complete_case_groups[0]
            .take_chromosome_prediction_matrix("1")
            .expect("only complete-case predictions should be parsed");
        assert_eq!(predictions.trait_count, 2);
        assert_eq!(predictions.sample_count, 2);
        assert_f32_values(&predictions.prediction_values, &[0.2, 0.3, 2.0, 3.0]);
    }
}

#[test]
fn selected_invalid_prediction_values_return_typed_errors() {
    for selected_value in ["NA", "invalid", "NaN", "inf", "1e100"] {
        let fixture = InputFixture::new("selected-invalid-predictions");
        fixture.directory.write(
            "trait-a.loco",
            &format!(
                "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n1 0.1 {selected_value} 0.3 NA\n"
            ),
        );
        let sample_identifiers = fixture.sample_identifiers();
        let phenotype_names = vec!["trait-a".to_string()];
        let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
        let mut groups = load_aligned_phenotype_groups(&fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::PerPhenotype,
        ))
        .expect("prediction values should be validated lazily");
        groups[0].plan_prediction_uses(&[Arc::from("1")]).expect("chromosome should plan");
        let result = groups[0].take_chromosome_prediction_matrix("1");
        if matches!(selected_value, "NA" | "invalid") {
            assert!(matches!(
                result,
                Err(PredictionError::InvalidPredictionValue { line_number: 2, value, .. })
                    if value == selected_value
            ));
        } else {
            assert!(matches!(
                result,
                Err(PredictionError::NonFinitePredictionValue { line_number: 2, value })
                    if value == selected_value
            ));
        }
    }
}

#[test]
fn excluded_samples_do_not_relax_prediction_row_shape_validation() {
    for prediction_row in ["1 0.1 0.2 0.3", "1 0.1 0.2 0.3 NA NA"] {
        let fixture = InputFixture::new("excluded-prediction-row-shape");
        fixture.directory.write(
            "trait-a.loco",
            &format!(
                "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n{prediction_row}\n"
            ),
        );
        let sample_identifiers = fixture.sample_identifiers();
        let phenotype_names = vec!["trait-a".to_string()];
        let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
        let result = load_aligned_phenotype_groups(&fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::PerPhenotype,
        ));
        assert!(matches!(
            result,
            Err(InputError::Prediction(PredictionError::LocoPredictionCountMismatch {
                line_number: 2,
                expected_count: 4,
                observed_count,
            })) if observed_count == prediction_row.split_ascii_whitespace().count() - 1
        ));
    }
}

#[test]
fn prediction_fingerprints_include_excluded_sample_values() {
    let fixture = InputFixture::new("excluded-prediction-fingerprint");
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = vec!["trait-a".to_string()];
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let mut fingerprints = Vec::new();
    for excluded_value in ["NA", "invalid"] {
        fixture.directory.write(
            "trait-a.loco",
            &format!(
                "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n1 0.1 0.2 0.3 {excluded_value}\n"
            ),
        );
        let mut groups = load_aligned_phenotype_groups(&fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::PerPhenotype,
        ))
        .expect("excluded prediction values should index");
        groups[0].plan_prediction_uses(&[Arc::from("1")]).expect("chromosome should plan");
        let predictions = groups[0].take_chromosome_prediction_matrix("1").expect("selected values should load");
        assert_f32_values(&predictions.prediction_values, &[0.1, 0.2, 0.3]);
        fingerprints.push(groups[0].phenotype_group.prediction_alignment_fingerprint.clone());
    }
    assert_ne!(fingerprints[0], fingerprints[1]);
}

#[test]
fn underscore_sample_identifiers_align_by_the_complete_serialized_key() {
    let fixture = InputFixture::new("underscore-prediction-identifiers");
    let sample_identifiers = SampleIdentifierData {
        family_identifiers: ["family_one", "_family", "family", "excluded_family"].map(str::to_string).to_vec(),
        individual_identifiers: ["individual_one", "person", "person_", "individual"].map(str::to_string).to_vec(),
    };
    fixture.directory.write(
        "phenotypes.tsv",
        "FID\tIID\ttrait-a\nfamily\tperson_\t3\nfamily_one\tindividual_one\t1\n_family\tperson\t2\nexcluded_family\tindividual\tNA\n",
    );
    fixture.directory.write(
        "trait-a.loco",
        "FID_IID family_person_ excluded_family_individual _family_person family_one_individual_one\n1 0.3 NA 0.2 0.1\n",
    );
    let phenotype_names = vec!["trait-a".to_string()];
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    let request = PhenotypeGroupLoadRequest {
        covariate_path: None,
        ..fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::PerPhenotype,
        )
    };
    let mut groups = load_aligned_phenotype_groups(&request).expect("underscores in either identifier should align");
    assert_eq!(groups[0].sample_indices, [0, 1, 2]);
    groups[0].plan_prediction_uses(&[Arc::from("1")]).expect("chromosome should plan");
    let predictions = groups[0].take_chromosome_prediction_matrix("1").expect("underscore identifiers should load");
    assert_f32_values(&predictions.prediction_values, &[0.1, 0.2, 0.3]);
}

#[test]
fn duplicate_serialized_loco_header_keys_are_rejected() {
    let fixture = InputFixture::new("duplicate-loco-header-keys");
    fixture.directory.write(
        "trait-a.loco",
        "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4 family-4_individual-4\n1 0.1 0.2 0.3 NA NA\n",
    );
    let sample_identifiers = fixture.sample_identifiers();
    let phenotype_names = vec!["trait-a".to_string()];
    let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
    assert!(matches!(
        load_aligned_phenotype_groups(&fixture.request(
            &sample_identifiers,
            &prediction_loco_paths,
            &phenotype_names,
            g_plan::MultiPhenotypeSampleMode::PerPhenotype,
        )),
        Err(InputError::Prediction(PredictionError::DuplicateLocoSampleKey { sample_key }))
            if sample_key == "family-4_individual-4"
    ));
}

#[test]
fn ambiguous_serialized_target_keys_are_rejected_even_when_one_sample_is_excluded() {
    for second_phenotype_value in ["2", "NA"] {
        let fixture = InputFixture::new("ambiguous-target-keys");
        let sample_identifiers = SampleIdentifierData {
            family_identifiers: ["family_one", "family", "other"].map(str::to_string).to_vec(),
            individual_identifiers: ["individual", "one_individual", "person"].map(str::to_string).to_vec(),
        };
        fixture.directory.write(
            "phenotypes.tsv",
            &format!(
                "FID\tIID\ttrait-a\nfamily_one\tindividual\t1\nfamily\tone_individual\t{second_phenotype_value}\nother\tperson\t3\n"
            ),
        );
        fixture.directory.write("trait-a.loco", "FID_IID family_one_individual other_person\n1 0.1 0.3\n");
        let phenotype_names = vec!["trait-a".to_string()];
        let prediction_loco_paths = fixture.prediction_loco_paths(&phenotype_names);
        let request = PhenotypeGroupLoadRequest {
            covariate_path: None,
            ..fixture.request(
                &sample_identifiers,
                &prediction_loco_paths,
                &phenotype_names,
                g_plan::MultiPhenotypeSampleMode::PerPhenotype,
            )
        };
        assert!(matches!(
            load_aligned_phenotype_groups(&request),
            Err(InputError::Prediction(PredictionError::AmbiguousTargetSampleKey { sample_key }))
                if sample_key == "family_one_individual"
        ));
    }
}
