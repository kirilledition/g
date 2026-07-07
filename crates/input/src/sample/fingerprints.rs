use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::types::{AlignedSampleData, MultiAlignedSampleData, ResolvedPhenotypeComputeGroup, SampleKeyMode};

const GROUP_MODE_COMPLETE_CASE: &str = "complete-case";
const GROUP_MODE_PER_PHENOTYPE_COMPATIBLE: &str = "per-phenotype-compatible";
const GROUP_MODE_SINGLE_PHENOTYPE: &str = "single-phenotype";
const SAMPLE_MODE_COMPLETE_CASE: &str = "complete-case";
const SAMPLE_MODE_PER_PHENOTYPE: &str = "per-phenotype";

#[must_use]
pub fn resolve_single_phenotype_compute_group(
    aligned_sample_data: &AlignedSampleData,
    phenotype_name: String,
    prediction_list_path: Option<&str>,
    sample_key_mode: SampleKeyMode,
) -> ResolvedPhenotypeComputeGroup {
    let phenotype_names = vec![phenotype_name];
    build_resolved_compute_group(
        GROUP_MODE_SINGLE_PHENOTYPE,
        vec![0],
        phenotype_names,
        SAMPLE_MODE_PER_PHENOTYPE,
        aligned_sample_data,
        prediction_list_path,
        sample_key_mode,
    )
}

#[must_use]
pub fn resolve_per_phenotype_compute_group(
    aligned_sample_data: &MultiAlignedSampleData,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    prediction_list_path: Option<&str>,
    sample_key_mode: SampleKeyMode,
) -> ResolvedPhenotypeComputeGroup {
    build_resolved_compute_group(
        GROUP_MODE_PER_PHENOTYPE_COMPATIBLE,
        phenotype_indices,
        phenotype_names,
        SAMPLE_MODE_PER_PHENOTYPE,
        aligned_sample_data,
        prediction_list_path,
        sample_key_mode,
    )
}

#[must_use]
pub fn resolve_complete_case_compute_group(
    aligned_sample_data: &MultiAlignedSampleData,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    prediction_list_path: Option<&str>,
    sample_key_mode: SampleKeyMode,
) -> ResolvedPhenotypeComputeGroup {
    build_resolved_compute_group(
        GROUP_MODE_COMPLETE_CASE,
        phenotype_indices,
        phenotype_names,
        SAMPLE_MODE_COMPLETE_CASE,
        aligned_sample_data,
        prediction_list_path,
        sample_key_mode,
    )
}

fn build_resolved_compute_group(
    group_mode: &str,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    sample_mode: &str,
    aligned_sample_data: &impl FingerprintAlignedSampleData,
    prediction_list_path: Option<&str>,
    sample_key_mode: SampleKeyMode,
) -> ResolvedPhenotypeComputeGroup {
    let sample_set_fingerprint = fingerprint_sample_set(aligned_sample_data);
    let prediction_alignment_fingerprint = prediction_list_path
        .map(|path| fingerprint_prediction_alignment(path, sample_key_mode, &sample_set_fingerprint, &phenotype_names));
    ResolvedPhenotypeComputeGroup {
        group_mode: group_mode.to_string(),
        phenotype_indices,
        phenotype_names,
        sample_mode: sample_mode.to_string(),
        sample_set_fingerprint,
        covariate_design_fingerprint: fingerprint_covariate_design(aligned_sample_data),
        prediction_alignment_fingerprint,
    }
}

trait FingerprintAlignedSampleData {
    fn sample_indices(&self) -> &[i64];

    fn family_identifiers(&self) -> &[String];

    fn individual_identifiers(&self) -> &[String];

    fn covariate_names(&self) -> &[String];

    fn covariate_matrix_values(&self) -> &[f32];

    fn covariate_row_count(&self) -> usize;

    fn covariate_column_count(&self) -> usize;
}

impl FingerprintAlignedSampleData for AlignedSampleData {
    fn sample_indices(&self) -> &[i64] {
        &self.sample_indices
    }

    fn family_identifiers(&self) -> &[String] {
        &self.family_identifiers
    }

    fn individual_identifiers(&self) -> &[String] {
        &self.individual_identifiers
    }

    fn covariate_names(&self) -> &[String] {
        &self.covariate_names
    }

    fn covariate_matrix_values(&self) -> &[f32] {
        &self.covariate_matrix_values
    }

    fn covariate_row_count(&self) -> usize {
        self.covariate_row_count
    }

    fn covariate_column_count(&self) -> usize {
        self.covariate_column_count
    }
}

impl FingerprintAlignedSampleData for MultiAlignedSampleData {
    fn sample_indices(&self) -> &[i64] {
        &self.sample_indices
    }

    fn family_identifiers(&self) -> &[String] {
        &self.family_identifiers
    }

    fn individual_identifiers(&self) -> &[String] {
        &self.individual_identifiers
    }

    fn covariate_names(&self) -> &[String] {
        &self.covariate_names
    }

    fn covariate_matrix_values(&self) -> &[f32] {
        &self.covariate_matrix_values
    }

    fn covariate_row_count(&self) -> usize {
        self.covariate_row_count
    }

    fn covariate_column_count(&self) -> usize {
        self.covariate_column_count
    }
}

fn fingerprint_sample_set(aligned_sample_data: &impl FingerprintAlignedSampleData) -> String {
    let mut fingerprint_hash = Sha256::new();
    update_fingerprint(&mut fingerprint_hash, "sample-set-v1");
    update_i64_array_fingerprint(
        &mut fingerprint_hash,
        "int64",
        &[aligned_sample_data.sample_indices().len()],
        aligned_sample_data.sample_indices(),
    );
    update_string_sequence_fingerprint(&mut fingerprint_hash, aligned_sample_data.family_identifiers());
    update_string_sequence_fingerprint(&mut fingerprint_hash, aligned_sample_data.individual_identifiers());
    finalize_sha256_hex(fingerprint_hash)
}

fn fingerprint_covariate_design(aligned_sample_data: &impl FingerprintAlignedSampleData) -> String {
    let mut fingerprint_hash = Sha256::new();
    update_fingerprint(&mut fingerprint_hash, "covariate-design-v1");
    update_string_sequence_fingerprint(&mut fingerprint_hash, aligned_sample_data.covariate_names());
    update_f32_array_fingerprint(
        &mut fingerprint_hash,
        "float32",
        &[aligned_sample_data.covariate_row_count(), aligned_sample_data.covariate_column_count()],
        aligned_sample_data.covariate_matrix_values(),
    );
    finalize_sha256_hex(fingerprint_hash)
}

fn fingerprint_prediction_alignment(
    prediction_list_path: &str,
    sample_key_mode: SampleKeyMode,
    sample_set_fingerprint: &str,
    phenotype_names: &[String],
) -> String {
    let mut fingerprint_hash = Sha256::new();
    update_fingerprint(&mut fingerprint_hash, "prediction-alignment-v1");
    update_fingerprint(&mut fingerprint_hash, prediction_list_path);
    update_fingerprint(&mut fingerprint_hash, sample_key_mode.as_str());
    update_fingerprint(&mut fingerprint_hash, sample_set_fingerprint);
    update_string_sequence_fingerprint(&mut fingerprint_hash, phenotype_names);
    finalize_sha256_hex(fingerprint_hash)
}

fn finalize_sha256_hex(fingerprint_hash: Sha256) -> String {
    let digest_bytes = fingerprint_hash.finalize();
    let mut digest_text = String::with_capacity(digest_bytes.len() * 2);
    for digest_byte in digest_bytes {
        write!(&mut digest_text, "{digest_byte:02x}").expect("writing to String must succeed");
    }
    digest_text
}

fn update_i64_array_fingerprint(fingerprint_hash: &mut Sha256, dtype_name: &str, shape: &[usize], values: &[i64]) {
    update_fingerprint(fingerprint_hash, dtype_name);
    update_fingerprint(fingerprint_hash, &python_shape_repr(shape));
    for value in values {
        fingerprint_hash.update(value.to_ne_bytes());
    }
}

fn update_f32_array_fingerprint(fingerprint_hash: &mut Sha256, dtype_name: &str, shape: &[usize], values: &[f32]) {
    update_fingerprint(fingerprint_hash, dtype_name);
    update_fingerprint(fingerprint_hash, &python_shape_repr(shape));
    for value in values {
        fingerprint_hash.update(value.to_ne_bytes());
    }
}

fn update_string_sequence_fingerprint(fingerprint_hash: &mut Sha256, values: &[String]) {
    update_fingerprint(fingerprint_hash, &values.len().to_string());
    for value in values {
        update_fingerprint(fingerprint_hash, value);
    }
}

fn update_fingerprint(fingerprint_hash: &mut Sha256, value: &str) {
    let encoded_value = value.as_bytes();
    fingerprint_hash.update(encoded_value.len().to_string().as_bytes());
    fingerprint_hash.update(b":");
    fingerprint_hash.update(encoded_value);
}

fn python_shape_repr(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [axis_length] => format!("({axis_length},)"),
        _ => format!("({})", shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    }
}
