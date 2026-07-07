use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleAlignmentError {
    message: String,
}

impl SampleAlignmentError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SampleAlignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SampleAlignmentError {}

impl From<String> for SampleAlignmentError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleKeyMode {
    Iid,
    FidIid,
}

impl SampleKeyMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Iid => "iid",
            Self::FidIid => "fid_iid",
        }
    }

    #[must_use]
    pub fn from_str_value(value: &str) -> Option<Self> {
        match value {
            "iid" => Some(Self::Iid),
            "fid_iid" => Some(Self::FidIid),
            _ => None,
        }
    }

    #[must_use]
    pub fn accepted_values() -> &'static [&'static str] {
        &["iid", "fid_iid"]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignedSampleData {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_name: String,
    pub phenotype_vector: Vec<f32>,
    pub covariate_names: Vec<String>,
    pub covariate_matrix_values: Vec<f32>,
    pub covariate_row_count: usize,
    pub covariate_column_count: usize,
    pub is_binary_trait: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAlignedSampleData {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_names: Vec<String>,
    pub phenotype_matrix_values: Vec<f32>,
    pub phenotype_row_count: usize,
    pub phenotype_column_count: usize,
    pub covariate_names: Vec<String>,
    pub covariate_matrix_values: Vec<f32>,
    pub covariate_row_count: usize,
    pub covariate_column_count: usize,
    pub is_binary_trait: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignedPhenotypeGroup {
    pub phenotype_indices: Vec<usize>,
    pub aligned_sample_data: MultiAlignedSampleData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupedAlignedSampleData {
    pub groups: Vec<AlignedPhenotypeGroup>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPhenotypeComputeGroup {
    pub group_mode: String,
    pub phenotype_indices: Vec<usize>,
    pub phenotype_names: Vec<String>,
    pub sample_mode: String,
    pub sample_set_fingerprint: String,
    pub covariate_design_fingerprint: String,
    pub prediction_alignment_fingerprint: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SampleIdentifierData {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct AlignmentInputs {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_path: String,
    pub phenotype_name: String,
    pub covariate_path: Option<String>,
    pub covariate_names: Option<Vec<String>>,
    pub is_binary_trait: bool,
    pub sample_key_mode: SampleKeyMode,
}

#[derive(Clone, Debug)]
pub struct MultiAlignmentInputs {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_path: String,
    pub phenotype_names: Vec<String>,
    pub covariate_path: Option<String>,
    pub covariate_names: Option<Vec<String>>,
    pub is_binary_trait: bool,
    pub sample_key_mode: SampleKeyMode,
}
