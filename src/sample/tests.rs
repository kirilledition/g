use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    AlignmentInputs, MultiAlignmentInputs, SampleKeyMode, align_grouped_sample_data, align_multi_sample_data,
    align_multi_sample_data_from_sample_file, align_sample_data, align_sample_data_from_sample_file,
    validate_sample_file_header,
};

static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

struct FixtureDirectory {
    path: PathBuf,
}

impl FixtureDirectory {
    fn new() -> Self {
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("g-sample-tests-{}-{fixture_id}", std::process::id()));
        fs::create_dir_all(&path).expect("sample test fixture directory should be created");
        Self { path }
    }

    fn write_file(&self, file_name: &str, contents: &str) -> String {
        let path = self.path.join(file_name);
        fs::write(&path, contents).expect("sample test fixture should be written");
        path.to_string_lossy().into_owned()
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

fn base_alignment_inputs(phenotype_path: String, phenotype_name: &str) -> AlignmentInputs {
    AlignmentInputs {
        sample_indices: vec![2, 0, 1],
        family_identifiers: strings(&["F3", "F1", "F2"]),
        individual_identifiers: strings(&["I3", "I1", "I2"]),
        phenotype_path,
        phenotype_name: phenotype_name.to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    }
}

#[test]
fn aligns_quantitative_samples_with_covariate_complete_cases_by_fid_iid() {
    let fixture = FixtureDirectory::new();
    let phenotype_path =
        fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1.5\nF2\tI2\tNA\nF3\tI3\t2.5\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\tbatch\nF1\tI1\t41\t0\nF3\tI3\t63\t1\n");
    let mut inputs = base_alignment_inputs(phenotype_path, "trait");
    inputs.covariate_path = Some(covariate_path);

    let aligned = align_sample_data(inputs).expect("quantitative sample alignment should succeed");

    assert_eq!(aligned.sample_indices, vec![0, 2]);
    assert_eq!(aligned.family_identifiers, strings(&["F1", "F3"]));
    assert_eq!(aligned.individual_identifiers, strings(&["I1", "I3"]));
    assert_eq!(aligned.phenotype_name, "trait");
    assert_eq!(aligned.phenotype_vector, vec![1.5, 2.5]);
    assert_eq!(aligned.covariate_names, strings(&["intercept", "age", "batch"]));
    assert_eq!(aligned.covariate_row_count, 2);
    assert_eq!(aligned.covariate_column_count, 3);
    assert_eq!(aligned.covariate_matrix_values, vec![1.0, 41.0, 0.0, 1.0, 63.0, 1.0]);
    assert!(!aligned.is_binary_trait);
}

#[test]
fn ignores_invalid_covariate_values_for_phenotype_missing_samples() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1.5\nF2\tI2\tNA\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t41\nF2\tI2\tbad\n");
    let mut inputs = base_alignment_inputs(phenotype_path, "trait");
    inputs.sample_indices = vec![0, 1];
    inputs.family_identifiers = strings(&["F1", "F2"]);
    inputs.individual_identifiers = strings(&["I1", "I2"]);
    inputs.covariate_path = Some(covariate_path);

    let aligned = align_sample_data(inputs).expect("unused invalid covariate should not fail alignment");

    assert_eq!(aligned.sample_indices, vec![0]);
    assert_eq!(aligned.phenotype_vector, vec![1.5]);
    assert_eq!(aligned.covariate_matrix_values, vec![1.0, 41.0]);
}

#[test]
fn aligns_binary_samples_and_recodes_regenie_case_control_values() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "IID\tcase\nI1\t1\nI2\t2\nI3\t1\n");
    let inputs = AlignmentInputs {
        sample_indices: vec![0, 1, 2],
        family_identifiers: strings(&["F1", "F2", "F3"]),
        individual_identifiers: strings(&["I1", "I2", "I3"]),
        phenotype_path,
        phenotype_name: "case".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: true,
        sample_key_mode: SampleKeyMode::Iid,
    };

    let aligned = align_sample_data(inputs).expect("binary sample alignment should succeed");

    assert_eq!(aligned.phenotype_vector, vec![0.0, 1.0, 0.0]);
    assert_eq!(aligned.covariate_names, strings(&["intercept"]));
    assert!(aligned.is_binary_trait);
}

#[test]
fn rejects_invalid_binary_phenotype_value() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "IID\tcase\nI1\t3\n");
    let inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path,
        phenotype_name: "case".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: true,
        sample_key_mode: SampleKeyMode::Iid,
    };

    let error = align_sample_data(inputs).expect_err("invalid binary phenotype should be rejected");

    assert!(error.contains("Binary phenotype must contain only values 1 and 2"));
}

#[test]
fn aligns_multi_phenotype_complete_cases_to_shared_sample_set() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture
        .write_file("phenotypes.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t10\t20\nF2\tI2\t11\tNA\nF3\tI3\tNA\t22\n");
    let inputs = MultiAlignmentInputs {
        sample_indices: vec![0, 1, 2],
        family_identifiers: strings(&["F1", "F2", "F3"]),
        individual_identifiers: strings(&["I1", "I2", "I3"]),
        phenotype_path,
        phenotype_names: strings(&["trait_a", "trait_b"]),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let aligned = align_multi_sample_data(inputs).expect("multi-phenotype alignment should succeed");

    assert_eq!(aligned.sample_indices, vec![0]);
    assert_eq!(aligned.phenotype_names, strings(&["trait_a", "trait_b"]));
    assert_eq!(aligned.phenotype_row_count, 2);
    assert_eq!(aligned.phenotype_column_count, 1);
    assert_eq!(aligned.phenotype_matrix_values, vec![10.0, 20.0]);
    assert_eq!(aligned.covariate_matrix_values, vec![1.0]);
}

#[test]
fn rejects_short_single_phenotype_selected_row() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("short-phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\n");
    let inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: phenotype_path.clone(),
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let error = align_sample_data(inputs).expect_err("short selected phenotype row should fail");

    assert!(error.contains(&format!(
            "Phenotype table '{phenotype_path}' line 2 is missing selected column 'trait' at column index 2; row has 2 fields."
        )));
}

#[test]
fn rejects_short_selected_covariate_row() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
    let covariate_path = fixture.write_file("short-covariates.tsv", "FID\tIID\tage\tsex\nF1\tI1\t40\n");
    let inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: Some(covariate_path.clone()),
        covariate_names: Some(strings(&["sex"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let error = align_sample_data(inputs).expect_err("short selected covariate row should fail");

    assert!(error.contains(&format!(
            "Covariate table '{covariate_path}' line 2 is missing selected column 'sex' at column index 3; row has 3 fields."
        )));
}

#[test]
fn rejects_short_multi_phenotype_selected_row() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("short-multi.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t10\n");
    let inputs = MultiAlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: phenotype_path.clone(),
        phenotype_names: strings(&["trait_a", "trait_b"]),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let error = align_multi_sample_data(inputs).expect_err("short selected multi-phenotype row should fail");

    assert!(error.contains(&format!(
            "Phenotype table '{phenotype_path}' line 2 is missing selected column 'trait_b' at column index 3; row has 3 fields."
        )));
}

#[test]
fn accepts_explicit_empty_selected_fields_as_missing_values() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t\nF2\tI2\t2\nF3\tI3\t3\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF2\tI2\t\nF3\tI3\t50\n");
    let inputs = AlignmentInputs {
        sample_indices: vec![0, 1, 2],
        family_identifiers: strings(&["F1", "F2", "F3"]),
        individual_identifiers: strings(&["I1", "I2", "I3"]),
        phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: Some(covariate_path),
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let aligned = align_sample_data(inputs).expect("explicit empty selected fields should remain missing values");

    assert_eq!(aligned.sample_indices, vec![2]);
    assert_eq!(aligned.phenotype_vector, vec![3.0]);
    assert_eq!(aligned.covariate_matrix_values, vec![1.0, 50.0]);
}

#[test]
fn groups_per_phenotype_alignments_by_identical_sample_sets() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file(
        "phenotypes.tsv",
        "FID\tIID\ttrait_a\ttrait_b\ttrait_c\nF1\tI1\t10\t20\t30\nF2\tI2\t11\t21\tNA\nF3\tI3\tNA\tNA\t32\n",
    );
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\nF2\tI2\t50\nF3\tI3\t60\n");
    let inputs = MultiAlignmentInputs {
        sample_indices: vec![2, 0, 1],
        family_identifiers: strings(&["F3", "F1", "F2"]),
        individual_identifiers: strings(&["I3", "I1", "I2"]),
        phenotype_path,
        phenotype_names: strings(&["trait_a", "trait_b", "trait_c"]),
        covariate_path: Some(covariate_path),
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let grouped = align_grouped_sample_data(&inputs).expect("grouped alignment should succeed");

    assert_eq!(grouped.groups.len(), 2);
    assert_eq!(grouped.groups[0].phenotype_indices, vec![0, 1]);
    assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_names, strings(&["trait_a", "trait_b"]));
    assert_eq!(grouped.groups[0].aligned_sample_data.sample_indices, vec![0, 1]);
    assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_row_count, 2);
    assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_column_count, 2);
    assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_matrix_values, vec![10.0, 11.0, 20.0, 21.0]);
    assert_eq!(grouped.groups[0].aligned_sample_data.covariate_matrix_values, vec![1.0, 40.0, 1.0, 50.0]);
    assert_eq!(grouped.groups[1].phenotype_indices, vec![2]);
    assert_eq!(grouped.groups[1].aligned_sample_data.phenotype_names, strings(&["trait_c"]));
    assert_eq!(grouped.groups[1].aligned_sample_data.sample_indices, vec![0, 2]);
    assert_eq!(grouped.groups[1].aligned_sample_data.phenotype_matrix_values, vec![30.0, 32.0]);
    assert_eq!(grouped.groups[1].aligned_sample_data.covariate_matrix_values, vec![1.0, 40.0, 1.0, 60.0]);
}

#[test]
fn rejects_empty_multi_phenotype_request() {
    let inputs = MultiAlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: "unused.tsv".to_string(),
        phenotype_names: Vec::new(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::Iid,
    };

    let error = align_multi_sample_data(inputs).expect_err("empty phenotype list should be rejected");

    assert!(error.contains("At least one phenotype is required"));
}

#[test]
fn rejects_duplicate_iids_when_iid_mode_is_requested() {
    let inputs = AlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["same", "same"]),
        phenotype_path: "unused.tsv".to_string(),
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::Iid,
    };

    let error = align_sample_data(inputs).expect_err("duplicate IID should be rejected before file IO");

    assert!(error.contains("Duplicate IID 'same'"));
}

#[test]
fn rejects_covariate_names_without_covariate_table() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
    let mut inputs = base_alignment_inputs(phenotype_path, "trait");
    inputs.covariate_names = Some(strings(&["age"]));

    let error = align_sample_data(inputs).expect_err("covariate names require a covariate table");

    assert!(error.contains("Covariate names cannot be provided without a covariate table"));
}

#[test]
fn rejects_missing_requested_covariate_column() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\n");
    let mut inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: Some(covariate_path),
        covariate_names: Some(strings(&["missing"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let error = align_sample_data(inputs.clone()).expect_err("missing covariate should be rejected");
    inputs.covariate_names = Some(strings(&["age"]));
    assert!(align_sample_data(inputs).is_ok());
    assert!(error.contains("Covariate columns are missing"));
}

#[test]
fn covers_sample_file_header_and_count_errors() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
    let sample_path = fixture.write_file("study.sample", "ID_1 ID_2 missing\n0 0 0\nF1 I1 0\nF2 I2 0\n");

    assert!(
        align_sample_data_from_sample_file(
            Path::new(&sample_path),
            1,
            phenotype_path.clone(),
            "trait".to_string(),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample count mismatch should fail")
        .contains("BGEN contains 1 samples")
    );
    assert!(
        align_multi_sample_data_from_sample_file(
            Path::new(&sample_path),
            1,
            phenotype_path,
            strings(&["trait"]),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("multi sample count mismatch should fail")
        .contains("BGEN contains 1 samples")
    );

    assert!(
        validate_sample_file_header(Path::new("empty.sample"), &[], &[])
            .expect_err("empty sample header should fail")
            .contains("does not contain any columns")
    );
    assert!(
        validate_sample_file_header(Path::new("bad-first-type.sample"), &strings(&["ID_1"]), &strings(&["D"]))
            .expect_err("first identifier type should be zero")
            .contains("first identifier column")
    );
    assert!(
        validate_sample_file_header(
            Path::new("bad-id2-type.sample"),
            &strings(&["ID_1", "ID_2"]),
            &strings(&["0", "D"]),
        )
        .expect_err("ID_2 type should be zero")
        .contains("'ID_2'")
    );
}

#[test]
fn covers_table_identifier_duplicate_and_missing_value_edges() {
    let fixture = FixtureDirectory::new();

    let empty_identifier_phenotype_path = fixture.write_file("empty-iid.tsv", "IID\ttrait\n\t1\n");
    let empty_identifier_inputs = AlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["", ""]),
        phenotype_path: empty_identifier_phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::Iid,
    };
    assert!(
        align_sample_data(empty_identifier_inputs)
            .expect_err("empty IIDs should not align")
            .contains("No aligned samples")
    );

    let missing_fid_phenotype_path = fixture.write_file("missing-fid.tsv", "IID\ttrait\nI1\t1\n");
    let missing_fid_inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: missing_fid_phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    assert!(
        align_sample_data(missing_fid_inputs)
            .expect_err("FID is required in fid_iid mode")
            .contains("Identifier column 'FID'")
    );

    let nonnumeric_phenotype_path = fixture.write_file("nonnumeric.tsv", "IID\ttrait\nI1\tbad\n");
    let nonnumeric_inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: nonnumeric_phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::Iid,
    };
    assert!(
        align_sample_data(nonnumeric_inputs)
            .expect_err("nonnumeric phenotype should fail")
            .contains("Failed to parse phenotype")
    );

    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\nF2\tI2\t2\n");
    let duplicate_covariate_path =
        fixture.write_file("duplicate-covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\nF1\tI1\t41\n");
    let duplicate_covariate_inputs = AlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["I1", "I2"]),
        phenotype_path: phenotype_path.clone(),
        phenotype_name: "trait".to_string(),
        covariate_path: Some(duplicate_covariate_path),
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    assert!(
        align_sample_data(duplicate_covariate_inputs)
            .expect_err("duplicate covariate sample should fail")
            .contains("covariate table")
    );

    let missing_covariate_path =
        fixture.write_file("missing-covariates.tsv", "FID\tIID\tage\nF1\tI1\tNA\nF2\tI2\t50\n");
    let missing_covariate_inputs = AlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["I1", "I2"]),
        phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: Some(missing_covariate_path),
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    let aligned = align_sample_data(missing_covariate_inputs).expect("missing covariate should drop one sample");
    assert_eq!(aligned.sample_indices, vec![1]);
    assert_eq!(aligned.covariate_matrix_values, vec![1.0, 50.0]);
}

#[test]
fn covers_multi_alignment_covariate_and_duplicate_edges() {
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t1\t2\n");
    let covariate_names_without_table = MultiAlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: phenotype_path.clone(),
        phenotype_names: strings(&["trait_a", "trait_b"]),
        covariate_path: None,
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    assert!(
        align_multi_sample_data(covariate_names_without_table)
            .expect_err("multi covariate names without table should fail")
            .contains("Covariate names cannot be provided")
    );

    let duplicate_multi_phenotype_path =
        fixture.write_file("duplicate-multi.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t1\t2\nF1\tI1\t3\t4\n");
    let duplicate_multi_inputs = MultiAlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: duplicate_multi_phenotype_path,
        phenotype_names: strings(&["trait_a", "trait_b"]),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    assert!(
        align_multi_sample_data(duplicate_multi_inputs)
            .expect_err("duplicate multi phenotype keys should fail")
            .contains("phenotype table")
    );

    let missing_fid_covariate_path = fixture.write_file("missing-fid-covariates.tsv", "IID\tage\nI1\t40\n");
    let missing_fid_covariate_inputs = MultiAlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&["F1"]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path,
        phenotype_names: strings(&["trait_a", "trait_b"]),
        covariate_path: Some(missing_fid_covariate_path),
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    assert!(
        align_multi_sample_data(missing_fid_covariate_inputs)
            .expect_err("FID is required in covariate table")
            .contains("Identifier column 'FID'")
    );
}
