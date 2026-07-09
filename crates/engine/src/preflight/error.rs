#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreflightError {
    #[error("BGEN input contains no variants.")]
    EmptyBgenInput,
    #[error("BGEN scan contains no variants.")]
    EmptyBgenScan,
    #[error("Could not read chromosome boundary metadata: {message}")]
    ChromosomeMetadata { message: String },
    #[error("{label} cannot be negative: {count}")]
    NegativeCount { label: &'static str, count: i64 },
    #[error("{label} exceeds native int64 capacity.")]
    CountOverflow { label: &'static str },
    #[error("Phenotype matrix must be two-dimensional.")]
    PhenotypeMatrixDimension,
    #[error("Phenotype matrix must contain at least one trait.")]
    EmptyPhenotypeTraitSet,
    #[error("Phenotype matrix must contain at least one sample.")]
    EmptyPhenotypeSampleSet,
    #[error("Phenotype matrix value count does not match its shape.")]
    PhenotypeMatrixValueCountMismatch,
    #[error("Covariate matrix must be two-dimensional.")]
    CovariateMatrixDimension,
    #[error("Covariate matrix shape exceeds supported size.")]
    CovariateMatrixShapeOverflow,
    #[error("Covariate matrix value count does not match its shape.")]
    CovariateMatrixValueCountMismatch,
    #[error("Covariate matrix sample count does not match phenotype sample count.")]
    CovariateSampleCountMismatch,
    #[error("Sample count must exceed the number of covariate degrees of freedom.")]
    NonPositiveResidualDegreesOfFreedom,
    #[error("{label} contains non-finite values.")]
    NonFiniteArray { label: String },
    #[error("Covariate matrix is rank deficient.")]
    CovariateMatrixRankDeficient,
    #[error("Binary phenotype must be coded as 0/1 after alignment.")]
    BinaryPhenotypeCoding,
    #[error("Binary phenotype must contain at least one case and one control.")]
    BinaryPhenotypeMissingClass,
    #[error(
        "Prediction sample count for chromosome {chromosome} is {actual_sample_count}, expected {expected_sample_count}."
    )]
    PredictionSampleCountMismatch { chromosome: String, actual_sample_count: i64, expected_sample_count: i64 },
    #[error("Prediction matrix shape for chromosome {chromosome} is {actual_shape}, expected {expected_shape}.")]
    PredictionMatrixShapeMismatch { chromosome: String, actual_shape: String, expected_shape: String },
}
